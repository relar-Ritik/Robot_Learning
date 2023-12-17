import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

import wandb
from Agent import Agent, SharedBackBone
from Utils import make_env
from PPO_f import policy_rollout, calculate_loss, calculate_returns

exp_name = f"Ant8_using6"
xml = "ant8.xml"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
wandb_project_name = "PPO"
load = True
env_id = "Ant-v5"
total_timesteps = 10000000
learning_rate = 5e-4
num_paralel_envs = 16
num_steps = 2048  # num of steps per each env for policy rollout.
gamma = 0.99
gae_lambda = 0.95
num_minibatches = 128
update_epochs = 10  # number of times to update policy before new rollout
clip_coef = 0.2
ent_coef = 0.0
vf_coef = 0.5
max_grad_norm = 0.5

batch_size = int(num_paralel_envs * num_steps)
minibatch_size = int(batch_size // num_minibatches)

run_name = f"{exp_name}__{int(time.time())}"

wandb.init(
    project="Joint_train",
    sync_tensorboard=True,
    name=run_name,
    save_code=True,
)

writer = SummaryWriter(f"runs/{run_name}")

envs = gym.vector.SyncVectorEnv([make_env(env_id, i, run_name, gamma, xml) for i in range(num_paralel_envs)])
assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

shared_back_bone = SharedBackBone(req_backbone_grad=False)

# shared_back_bone.load_state_dict(torch.load("share_backbone_ant6.pt"))
agent = Agent(envs, shared_back_bone).to(device)

# agent.load_state_dict(torch.load("total_agent.pt"))

optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=5e-5)
# optimizer = optim.Adam([
#     {'params': agent.critic_start.parameters()},
#     {'params': agent.critic_end.parameters()},
#     {'params': agent.actor_start.parameters()},
#     {'params': agent.actor_mean.parameters()},
#     {'params': agent.actor_logstd.parameters()},
#     {'params': agent.backbone.parameters(), 'lr': 1e-7},
# ], lr=learning_rate, eps=5e-5)

global_step = 0
start_time = time.time()
num_updates = total_timesteps // batch_size

for update in trange(1, num_updates + 1):
    b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values, global_step = calculate_returns(agent,
                                                                                                            envs,
                                                                                                            gamma,
                                                                                                            gae_lambda,
                                                                                                            num_steps,
                                                                                                            num_paralel_envs,
                                                                                                            writer,
                                                                                                            global_step,
                                                                                                            "ant8")


    b_inds = np.arange(batch_size)
    clipfracs = []
    for epoch in range(update_epochs):
        np.random.shuffle(b_inds)
        for start in range(0, batch_size, minibatch_size):
            mb_inds = b_inds[start: start + minibatch_size]

            _, newlogprob, entropy, newvalue,_,_ = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
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
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
            optimizer.step()

    y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

    # TRY NOT TO MODIFY: record rewards for plotting purposes
    writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
    writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
    writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
    writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
    writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
    writer.add_scalar("losses/explained_variance", explained_var, global_step)
    # print("SPS:", int(global_step / (time.time() - start_time)))
    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

# torch.save(shared_back_bone.state_dict(), "share_backbone_ant8.pt")
# torch.save(agent.state_dict(), "total_agent_ant8.pt")
envs.close()
writer.close()
wandb.finish()
