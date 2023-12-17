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
from PPO_f import calculate_returns, calculate_loss

exp_name = f"SharedTraining_diff_loss__{time.time()}"
xml = "ant8.xml"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
wandb_project_name = "PPO"
load = False
env_id = "Ant-v5"
total_timesteps = 5000000
learning_rate = 5e-4
num_paralel_envs = 20
num_steps = 1024  # num of steps per each env for policy rollout.
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

run_name = f"{'Joint_train'}{load}__{exp_name}__{int(time.time())}"

wandb.init(
    project="Joint_train",
    sync_tensorboard=True,
    name=run_name,
    save_code=True,
)

writer = SummaryWriter(f"runs/{run_name}")

envs4 = gym.vector.SyncVectorEnv(
    [make_env(env_id, i, run_name, gamma, xml="antCC.xml") for i in range(num_paralel_envs)])
envs6 = gym.vector.SyncVectorEnv(
    [make_env(env_id, i, run_name, gamma, xml="ant6.xml") for i in range(num_paralel_envs)])
assert isinstance(envs4.single_action_space, gym.spaces.Box), "only continuous action space is supported"
assert isinstance(envs6.single_action_space, gym.spaces.Box), "only continuous action space is supported"

shared_back_bone4 = SharedBackBone(req_backbone_grad=True)
shared_back_bone6 = SharedBackBone(req_backbone_grad=True)
agent4 = Agent(envs4, shared_back_bone4).to(device)
agent6 = Agent(envs6, shared_back_bone6).to(device)

# agent.load_state_dict(torch.load("total_agent.pt"))

optimizer_4 = optim.Adam(agent4.parameters(), lr=learning_rate, eps=5e-5)
optimizer_6 = optim.Adam(agent6.parameters(), lr=learning_rate, eps=5e-5)

# optimizer = optim.Adam([
#     {'params': agent.critic_start.parameters()},
#     {'params': agent.critic_end.parameters()},
#     {'params': agent.actor_start.parameters()},
#     {'params': agent.actor_mean.parameters()},
#     {'params': agent.actor_logstd.parameters()},
#     {'params': agent.backbone.parameters(), 'lr': 1e-8},
# ], lr=learning_rate, eps=5e-5)

global_step = 0
start_time = time.time()
num_updates = total_timesteps // batch_size

for update in trange(1, num_updates + 1):
    b_obs4, b_logprobs4, b_actions4, b_advantages4, b_returns4, b_values4, global_step1 = calculate_returns(agent4,
                                                                                                            envs4,
                                                                                                            gamma,
                                                                                                            gae_lambda,
                                                                                                            num_steps,
                                                                                                            num_paralel_envs,
                                                                                                            writer,
                                                                                                            global_step,
                                                                                                            "ant4")
    b_obs6, b_logprobs6, b_actions6, b_advantages6, b_returns6, b_values6, global_step2 = calculate_returns(agent6,
                                                                                                            envs6,
                                                                                                            gamma,
                                                                                                            gae_lambda,
                                                                                                            num_steps,
                                                                                                            num_paralel_envs,
                                                                                                            writer,
                                                                                                            global_step,
                                                                                                            "ant6")
    global_step = max(global_step1, global_step2)
    b_inds = np.arange(batch_size)
    clipfracs = []
    for epoch in range(update_epochs):
        np.random.shuffle(b_inds)
        for start in range(0, batch_size, minibatch_size):
            mb_inds = b_inds[start: start + minibatch_size]
            loss4, v_loss4, pg_loss4, entropy_loss4, start4, end4 = calculate_loss(agent4, b_obs4, b_logprobs4,
                                                                                   b_actions4, b_advantages4,
                                                                                   b_returns4, b_values4, mb_inds,
                                                                                   clip_coef, ent_coef, vf_coef)
            loss6, v_loss6, pg_loss6, entropy_loss6, start6, end6 = calculate_loss(agent6, b_obs6, b_logprobs6,
                                                                                   b_actions6,
                                                                                   b_advantages6, b_returns6, b_values6,
                                                                                   mb_inds,
                                                                                   clip_coef, ent_coef, vf_coef)

            loss = loss4 + loss6 + torch.nn.functional.mse_loss(start4, start6) + torch.nn.functional.mse_loss(end4,
                                                                                                               end4)

            optimizer_4.zero_grad()
            optimizer_6.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm(agent4.parameters(), max_grad_norm)
            optimizer_4.step()
            nn.utils.clip_grad_norm(agent6.parameters(), max_grad_norm)
            optimizer_6.step()

    y_pred, y_true = b_values4.cpu().numpy(), b_returns4.cpu().numpy()
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

    # TRY NOT TO MODIFY: record rewards for plotting purposes
    writer.add_scalar("charts/learning_rate", optimizer_6.param_groups[0]["lr"], global_step)
    writer.add_scalar("losses/value_loss", v_loss4.item(), global_step)
    writer.add_scalar("losses/policy_loss", pg_loss4.item(), global_step)
    writer.add_scalar("losses/entropy", entropy_loss4.item(), global_step)
    writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
    writer.add_scalar("losses/explained_variance", explained_var, global_step)
    # print("SPS:", int(global_step / (time.time() - start_time)))
    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

torch.save(shared_back_bone4.state_dict(), "share_backbone_ant4.pt")
torch.save(shared_back_bone6.state_dict(), "share_backbone_ant6.pt")
torch.save(agent4.state_dict(), "total_agent_ant4tw_diff_loss.pt")
torch.save(agent6.state_dict(), "total_agent_ant6tw_diff_loss.pt")
envs4.close()
envs6.close()
writer.close()
wandb.finish()
