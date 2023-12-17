import numpy as np

import torch
import torch.nn as nn
from torch.distributions.normal import Normal


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class SharedBackBone(nn.Module):
    def __init__(self, req_backbone_grad=False):
        super().__init__()
        self.critic_backbone = nn.Sequential(
            layer_init(nn.Linear(256, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 64)),
            nn.ReLU()
        )
        self.actor_backbone = nn.Sequential(
            layer_init(nn.Linear(512, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 64)),
            nn.Tanh()
        )
        for params in self.critic_backbone.parameters():
            params.requires_grad=req_backbone_grad
        for params in self.actor_backbone.parameters():
            params.requires_grad=req_backbone_grad


class Agent(nn.Module):
    def __init__(self, envs, backbone: SharedBackBone):
        super().__init__()
        self.input_size = np.array(envs.single_observation_space.shape).prod()
        self.backbone = backbone
        self.critic_start = nn.Sequential(
            layer_init(nn.Linear(self.input_size, 256)),
            nn.ReLU())
        self.critic_end = layer_init(nn.Linear(64, 1), std=1.0)

        self.actor_start = nn.Sequential(
            layer_init(nn.Linear(self.input_size, 512)),
            nn.ReLU())

        self.actor_mean = layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01)
        self.actor_logstd = layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01)

    def get_value(self, x):
        x = self.critic_start(x)
        x = self.backbone.critic_backbone(x)
        return self.critic_end(x)

    def get_action_and_value(self, x, action=None):
        x = x.view(-1, self.input_size)
        ac_z_start = self.actor_start(x)
        ac_z = self.backbone.actor_backbone(ac_z_start)
        action_mean = self.actor_mean(ac_z)
        action_logstd = self.actor_logstd(ac_z)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.get_value(x), ac_z_start, ac_z
