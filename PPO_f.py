import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

import wandb
from Agent import Agent
from Utils import make_env


def policy_rollout(agent, envs, num_steps, num_paralel_envs, writer, global_step, env_name):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    obs = torch.zeros((num_steps, num_paralel_envs, np.prod(envs.single_observation_space.shape))).to(device)
    actions = torch.zeros((num_steps, num_paralel_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((num_steps, num_paralel_envs)).to(device)
    rewards = torch.zeros((num_steps, num_paralel_envs)).to(device)
    dones = torch.zeros((num_steps, num_paralel_envs)).to(device)
    values = torch.zeros((num_steps, num_paralel_envs)).to(device)

    next_obs, _ = envs.reset()
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(num_paralel_envs).to(device)

    for step in range(num_steps):
        global_step += num_paralel_envs
        obs[step] = next_obs
        dones[step] = next_done

        with torch.no_grad():
            action, logprob, _, value,start, end = agent.get_action_and_value(next_obs)
        values[step] = value.flatten()
        actions[step] = action
        logprobs[step] = logprob

        next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
        done = np.logical_or(terminations, truncations)
        rewards[step] = torch.tensor(reward).flatten().to(device)
        next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

        if "final_info" in infos:
            for info in infos["final_info"]:
                # Skip the envs that are not done
                if info:
                    # print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar(f"charts/episodic_return_{env_name}", info["episode"]["r"], global_step)
                    writer.add_scalar(f"charts/episodic_length_{env_name}", info["episode"]["l"], global_step)
    return obs, next_obs, actions, logprobs, rewards, dones, values, next_obs, next_done, global_step


def calculate_returns(agent, envs, gamma, gae_lambda, num_steps, num_paralel_envs, writer, global_step, env_name):
    obs, next_obs, actions, logprobs, rewards, dones, values, last_obs, last_done, global_step = policy_rollout(agent,
                                                                                                                envs,
                                                                                                                num_steps,
                                                                                                                num_paralel_envs,
                                                                                                                writer,
                                                                                                                global_step, env_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with torch.no_grad():
        next_value = agent.get_value(next_obs).reshape(1, -1)
        advantages = torch.zeros_like(rewards).to(device)
        lastgaelam = 0
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                next_non_terminal = 1.0 - last_done
                nextvalues = next_value
            else:
                next_non_terminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            delta = rewards[t] + gamma * nextvalues * next_non_terminal - values[t]
            advantages[t] = lastgaelam = delta + gamma * gae_lambda * next_non_terminal * lastgaelam
        returns = advantages + values

    b_obs = obs.reshape((-1, np.prod(envs.single_observation_space.shape)))
    b_logprobs = logprobs.reshape(-1)
    b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = values.reshape(-1)

    return b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values, global_step

def calculate_loss(agent, b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values, mb_inds, clip_coef, ent_coef, vf_coef):
    _, newlogprob, entropy, newvalue, start, end = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
    logratio = newlogprob - b_logprobs[mb_inds]
    ratio = torch.exp(logratio)
    mb_advantages = b_advantages[mb_inds]
    pg_loss1 = -mb_advantages * ratio
    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

    newvalue = newvalue.view(-1)
    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
    v_clipped = b_values[mb_inds] + torch.clamp(newvalue - b_values[mb_inds], -clip_coef, clip_coef)
    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
    v_loss = 0.5 * v_loss_max.mean()

    entropy_loss = entropy.mean()
    loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef
    return loss, v_loss, pg_loss, entropy_loss, start, end