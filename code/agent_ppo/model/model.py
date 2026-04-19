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
    Raw obs split → Actor group encoders + Actor fusion + Actor head
                  → Critic group encoders + Critic fusion + Critic head
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


def make_mlp(in_features, hidden_dims):
    """Create a small MLP with ReLU between linear layers."""
    layers = []
    current_dim = in_features
    for out_dim in hidden_dims:
        layers.append(make_fc_layer(current_dim, out_dim))
        layers.append(nn.ReLU())
        current_dim = out_dim
    return nn.Sequential(*layers)


class Model(nn.Module):
    """Fully separated Actor/Critic networks.

    Actor 与 Critic 从分组编码器到输出头完全分离，不共享参数。
    """

    def __init__(self, device=None):
        super().__init__()
        self.model_name = "gorge_chase_actor_critic_separated"
        self.device = device

        self.feature_split_shape = list(Config.FEATURE_SPLIT_SHAPE)
        self.hero_dim = self.feature_split_shape[0]
        self.monster_dim = self.feature_split_shape[1]
        self.treasure_dim = self.feature_split_shape[3]
        self.buff_dim = self.feature_split_shape[13]
        self.map_dim = self.feature_split_shape[15]
        self.progress_dim = self.feature_split_shape[16]

        self.actor_feature_encoders = nn.ModuleDict(
            {
                "hero": make_mlp(self.hero_dim, [Config.HERO_ENCODER_DIM]),
                "monster": make_mlp(self.monster_dim, [Config.MONSTER_ENCODER_DIM]),
                "treasure": make_mlp(self.treasure_dim, [Config.TREASURE_ENCODER_DIM]),
                "buff": make_mlp(self.buff_dim, [Config.SPEED_BUFF_ENCODER_DIM]),
            }
        )

        self.critic_feature_encoders = nn.ModuleDict(
            {
                "hero": make_mlp(self.hero_dim, [Config.HERO_ENCODER_DIM]),
                "monster": make_mlp(self.monster_dim, [Config.MONSTER_ENCODER_DIM]),
                "treasure": make_mlp(self.treasure_dim, [Config.TREASURE_ENCODER_DIM]),
                "buff": make_mlp(self.buff_dim, [Config.SPEED_BUFF_ENCODER_DIM]),
            }
        )

        fusion_input_dim = (
            Config.HERO_ENCODER_DIM
            + Config.MONSTER_ENCODER_DIM
            + Config.TREASURE_ENCODER_DIM
            + Config.SPEED_BUFF_ENCODER_DIM
            + self.map_dim
            + self.progress_dim
        )
        hidden_dim = 256
        mid_dim = Config.FUSION_HIDDEN_DIM
        head_dim = 128
        action_num = Config.ACTION_NUM
        value_num = Config.VALUE_NUM

        # Actor fusion backbone / Actor 融合骨干
        self.actor_encoder = nn.Sequential(
            make_fc_layer(fusion_input_dim, hidden_dim),
            nn.ReLU(),
            make_fc_layer(hidden_dim, mid_dim),
            nn.ReLU(),
        )

        # Critic fusion backbone / Critic 融合骨干
        self.critic_encoder = nn.Sequential(
            make_fc_layer(fusion_input_dim, hidden_dim),
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

    def _encode_group_slots(self, group_tensor, encoder):
        """Encode fixed-count object slots with a shared encoder and mean pool valid slots."""
        batch_size, slot_num, feat_dim = group_tensor.shape
        flat_group = group_tensor.reshape(batch_size * slot_num, feat_dim)
        encoded = encoder(flat_group).reshape(batch_size, slot_num, -1)
        valid_mask = group_tensor[..., 0:1].clamp(min=0.0, max=1.0)
        encoded = encoded * valid_mask
        valid_count = valid_mask.sum(dim=1).clamp(min=1.0)
        return encoded.sum(dim=1) / valid_count

    def _split_obs_groups(self, obs):
        split_obs = torch.split(obs, self.feature_split_shape, dim=-1)
        hero_feat = split_obs[0]
        monster_feats = torch.stack(split_obs[1:3], dim=1)
        treasure_feats = torch.stack(split_obs[3:13], dim=1)
        buff_feats = torch.stack(split_obs[13:15], dim=1)
        map_feat = split_obs[15]
        progress_feat = split_obs[16]
        return hero_feat, monster_feats, treasure_feats, buff_feats, map_feat, progress_feat

    def _encode_branch(self, hero_feat, monster_feats, treasure_feats, buff_feats, map_feat, progress_feat, encoders):
        hero_emb = encoders["hero"](hero_feat)
        monster_emb = self._encode_group_slots(monster_feats, encoders["monster"])
        treasure_emb = self._encode_group_slots(treasure_feats, encoders["treasure"])
        buff_emb = self._encode_group_slots(buff_feats, encoders["buff"])
        return torch.cat(
            [hero_emb, monster_emb, treasure_emb, buff_emb, map_feat, progress_feat],
            dim=-1,
        )

    def _ensure_2d(self, obs):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        return obs

    def forward(self, obs, inference=False):
        """Forward pass with grouped feature encoders.

        obs 期望为 [B, vector_dim] 或 [vector_dim]。

        Args:
            obs: 观测张量或元组
            inference: 是否推理模式

        Returns:
            logits, value
        """
        if isinstance(obs, (tuple, list)):
            obs = obs[0]

        obs = self._ensure_2d(obs)
        hero_feat, monster_feats, treasure_feats, buff_feats, map_feat, progress_feat = self._split_obs_groups(obs)

        actor_fused = self._encode_branch(
            hero_feat,
            monster_feats,
            treasure_feats,
            buff_feats,
            map_feat,
            progress_feat,
            self.actor_feature_encoders,
        )
        critic_fused = self._encode_branch(
            hero_feat,
            monster_feats,
            treasure_feats,
            buff_feats,
            map_feat,
            progress_feat,
            self.critic_feature_encoders,
        )

        actor_hidden = self.actor_encoder(actor_fused)
        critic_hidden = self.critic_encoder(critic_fused)

        logits = self.actor_head(actor_hidden)
        value = self.critic_head(critic_hidden)
        return logits, value

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()
