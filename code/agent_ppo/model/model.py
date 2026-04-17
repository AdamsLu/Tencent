#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Neural network model for Gorge Chase PPO.
峡谷追猎 PPO 神经网络模型。

架构：
    Single Encoder: [vector_dim]D → FC → FC → hidden
    PPO Heads:      hidden → FC → Actor/Critic
"""

import torch
import torch.nn as nn

from agent_ppo.conf.conf import Config


def make_fc_layer(in_features, out_features):
    """Create a linear layer with orthogonal initialization.

    创建正交初始化的线性层。
    """
    fc = nn.Linear(in_features, out_features)
    nn.init.orthogonal_(fc.weight.data)
    nn.init.zeros_(fc.bias.data)
    return fc


class Model(nn.Module):
    """Single-encoder MLP backbone + Actor/Critic dual heads.

    单encoder架构：MLP向量骨干 + Actor/Critic双头。
    """

    def __init__(self, device=None):
        super().__init__()
        self.model_name = "gorge_chase_single_encoder"
        self.device = device

        vector_dim = Config.DIM_OF_OBSERVATION
        hidden_dim = 256
        mid_dim = 128
        head_dim = 128
        action_num = Config.ACTION_NUM
        value_num = Config.VALUE_NUM

        # Single encoder backbone / 单encoder特征骨干
        self.encoder = nn.Sequential(
            make_fc_layer(vector_dim, hidden_dim),
            nn.ReLU(),
            make_fc_layer(hidden_dim, mid_dim),
            nn.ReLU(),
        )

        # Actor head / 策略头（加深一层）
        self.actor_head = nn.Sequential(
            make_fc_layer(mid_dim, head_dim),
            nn.ReLU(),
            make_fc_layer(head_dim, action_num),
        )

        # Critic head / 价值头（加深一层）
        self.critic_head = nn.Sequential(
            make_fc_layer(mid_dim, head_dim),
            nn.ReLU(),
            make_fc_layer(head_dim, value_num),
        )

    def forward(self, obs, inference=False):
        """Forward pass with single vector input.

        单输入前向传播。
        obs 期望为 [B, vector_dim]，若误传 tuple/list 则仅使用第一个向量输入。

        Args:
            obs: 观测张量或元组
            inference: 是否推理模式

        Returns:
            logits, value
        """
        if isinstance(obs, (tuple, list)):
            obs = obs[0]

        hidden = self.encoder(obs)

        logits = self.actor_head(hidden)
        value = self.critic_head(hidden)
        return logits, value

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()
