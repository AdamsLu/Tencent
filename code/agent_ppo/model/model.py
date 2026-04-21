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


def make_map_encoder(out_dim):
    """Create a compact CNN encoder for 1x128x128 global map input."""
    return nn.Sequential(
        nn.Conv2d(Config.GLOBAL_MAP_CHANNELS, 16, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        make_fc_layer(64, out_dim),
        nn.ReLU(),
    )


def masked_mean_pool(embeddings, valid_mask):
    """Mean-pool embeddings with slot-level valid mask.

    Args:
        embeddings: [batch, num_slots, embed_dim]
        valid_mask: [batch, num_slots, 1]
    """
    masked = embeddings * valid_mask
    count = valid_mask.sum(dim=1).clamp(min=1.0)
    return masked.sum(dim=1) / count


class MonsterEncoder(nn.Module):
    """Dual-path encoder for monster slots.

    precise_dim: precise_dist(1) + precise_dir_onehot(9) = 10
    bin_dim: dist_bin_onehot(6) + dir_bin_onehot(9) + speed(1) + hist_vel(10) = 26
    """

    def __init__(self, precise_dim, bin_dim, embed_dim):
        super().__init__()
        self.precise_path = make_mlp(precise_dim, [embed_dim])
        self.bin_path = make_mlp(bin_dim, [embed_dim])
        self.fusion = make_fc_layer(embed_dim * 2, embed_dim)
        self.relu = nn.ReLU()

    def forward(self, precise_feat, bin_feat, in_view_mask):
        precise_emb = self.precise_path(precise_feat)
        precise_emb = precise_emb * in_view_mask
        bin_emb = self.bin_path(bin_feat)
        combined = torch.cat([precise_emb, bin_emb], dim=-1)
        return self.relu(self.fusion(combined))


class OrganEncoder(nn.Module):
    """Single-path encoder for static organ slots (no bin routing)."""

    def __init__(self, precise_dim, embed_dim):
        super().__init__()
        self.precise_path = make_mlp(precise_dim, [embed_dim])
        self.proj = make_fc_layer(embed_dim, embed_dim)
        self.relu = nn.ReLU()

    def forward(self, precise_feat):
        precise_emb = self.precise_path(precise_feat)
        return self.relu(self.proj(precise_emb))


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
        self.local_map_dim = self.feature_split_shape[15]
        self.global_map_dim = self.feature_split_shape[16]
        self.progress_dim = self.feature_split_shape[17]

        # Slot layout (40D):
        # [slot_valid(1), is_in_view(1), in_view_mask(1), out_view_mask(1),
        #  precise_dist(1), precise_dir(9), dist_bin(6), dir_bin(9), speed(1), hist_vel(10)]
        self.slot_valid_slice = slice(0, 1)
        self.in_view_mask_slice = slice(2, 3)
        self.precise_dist_slice = slice(4, 5)
        self.precise_dir_slice = slice(5, 14)
        self.dist_bin_slice = slice(14, 20)
        self.dir_bin_slice = slice(20, 29)
        self.speed_slice = slice(29, 30)
        self.hist_vel_slice = slice(30, 40)

        self.precise_dim = 10
        self.bin_dim = 26

        self.actor_hero_encoder = make_mlp(self.hero_dim, [Config.HERO_ENCODER_DIM])
        self.critic_hero_encoder = make_mlp(self.hero_dim, [Config.HERO_ENCODER_DIM])

        self.actor_monster_encoder = MonsterEncoder(self.precise_dim, self.bin_dim, Config.MONSTER_ENCODER_DIM)
        self.critic_monster_encoder = MonsterEncoder(self.precise_dim, self.bin_dim, Config.MONSTER_ENCODER_DIM)

        self.actor_treasure_encoder = OrganEncoder(self.precise_dim, Config.TREASURE_ENCODER_DIM)
        self.critic_treasure_encoder = OrganEncoder(self.precise_dim, Config.TREASURE_ENCODER_DIM)

        self.actor_buff_encoder = OrganEncoder(self.precise_dim, Config.SPEED_BUFF_ENCODER_DIM)
        self.critic_buff_encoder = OrganEncoder(self.precise_dim, Config.SPEED_BUFF_ENCODER_DIM)

        self.actor_local_map_encoder = make_mlp(
            self.local_map_dim,
            [Config.LOCAL_MAP_ENCODER_HIDDEN_DIM, Config.LOCAL_MAP_ENCODER_DIM],
        )
        self.critic_local_map_encoder = make_mlp(
            self.local_map_dim,
            [Config.LOCAL_MAP_ENCODER_HIDDEN_DIM, Config.LOCAL_MAP_ENCODER_DIM],
        )

        self.actor_map_encoder = make_map_encoder(Config.MAP_ENCODER_DIM)
        self.critic_map_encoder = make_map_encoder(Config.MAP_ENCODER_DIM)

        fusion_input_dim = (
            Config.HERO_ENCODER_DIM
            + Config.MONSTER_ENCODER_DIM
            + Config.TREASURE_ENCODER_DIM
            + Config.SPEED_BUFF_ENCODER_DIM
            + Config.LOCAL_MAP_ENCODER_DIM
            + Config.MAP_ENCODER_DIM
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

    def actor_parameters(self):
        return (
            list(self.actor_hero_encoder.parameters())
            + list(self.actor_monster_encoder.parameters())
            + list(self.actor_treasure_encoder.parameters())
            + list(self.actor_buff_encoder.parameters())
            + list(self.actor_local_map_encoder.parameters())
            + list(self.actor_map_encoder.parameters())
            + list(self.actor_encoder.parameters())
            + list(self.actor_head.parameters())
        )

    def critic_parameters(self):
        return (
            list(self.critic_hero_encoder.parameters())
            + list(self.critic_monster_encoder.parameters())
            + list(self.critic_treasure_encoder.parameters())
            + list(self.critic_buff_encoder.parameters())
            + list(self.critic_local_map_encoder.parameters())
            + list(self.critic_map_encoder.parameters())
            + list(self.critic_encoder.parameters())
            + list(self.critic_head.parameters())
        )

    def _split_entity_slot_features(self, group_tensor):
        """Split 40D slot feature into dual-path inputs and pooling masks."""
        valid_mask = group_tensor[..., self.slot_valid_slice].clamp(min=0.0, max=1.0)
        in_view_mask = group_tensor[..., self.in_view_mask_slice].clamp(min=0.0, max=1.0)

        precise_feat = torch.cat(
            [
                group_tensor[..., self.precise_dist_slice],
                group_tensor[..., self.precise_dir_slice],
            ],
            dim=-1,
        )
        bin_feat = torch.cat(
            [
                group_tensor[..., self.dist_bin_slice],
                group_tensor[..., self.dir_bin_slice],
                group_tensor[..., self.speed_slice],
                group_tensor[..., self.hist_vel_slice],
            ],
            dim=-1,
        )
        return precise_feat, bin_feat, in_view_mask, valid_mask

    def _encode_monster_slots(self, group_tensor, encoder):
        """Encode monster slots with dual-path routing and slot_valid masked pooling."""
        batch_size, slot_num, _ = group_tensor.shape
        precise_feat, bin_feat, in_view_mask, valid_mask = self._split_entity_slot_features(group_tensor)

        flat_precise = precise_feat.reshape(batch_size * slot_num, -1)
        flat_bin = bin_feat.reshape(batch_size * slot_num, -1)
        flat_in_view_mask = in_view_mask.reshape(batch_size * slot_num, 1)

        encoded = encoder(flat_precise, flat_bin, flat_in_view_mask).reshape(batch_size, slot_num, -1)
        return masked_mean_pool(encoded, valid_mask)

    def _encode_organ_slots(self, group_tensor, encoder):
        """Encode static organ slots from precise branch only and slot_valid mask."""
        batch_size, slot_num, _ = group_tensor.shape
        precise_feat, _, _, valid_mask = self._split_entity_slot_features(group_tensor)

        flat_precise = precise_feat.reshape(batch_size * slot_num, -1)
        encoded = encoder(flat_precise).reshape(batch_size, slot_num, -1)
        return masked_mean_pool(encoded, valid_mask)

    def _split_obs_groups(self, obs):
        split_obs = torch.split(obs, self.feature_split_shape, dim=-1)
        hero_feat = split_obs[0]
        monster_feats = torch.stack(split_obs[1:3], dim=1)
        treasure_feats = torch.stack(split_obs[3:13], dim=1)
        buff_feats = torch.stack(split_obs[13:15], dim=1)
        local_map_feat = split_obs[15]
        global_map_feat = split_obs[16]
        progress_feat = split_obs[17]
        return hero_feat, monster_feats, treasure_feats, buff_feats, local_map_feat, global_map_feat, progress_feat

    def _reshape_global_map(self, global_map_feat):
        batch_size = global_map_feat.shape[0]
        return global_map_feat.reshape(
            batch_size,
            Config.GLOBAL_MAP_CHANNELS,
            Config.GLOBAL_MAP_SIZE,
            Config.GLOBAL_MAP_SIZE,
        )

    def _encode_branch(
        self,
        hero_feat,
        monster_feats,
        treasure_feats,
        buff_feats,
        local_map_feat,
        global_map_feat,
        progress_feat,
        hero_encoder,
        monster_encoder,
        treasure_encoder,
        buff_encoder,
        local_map_encoder,
        map_encoder,
    ):
        hero_emb = hero_encoder(hero_feat)
        monster_emb = self._encode_monster_slots(monster_feats, monster_encoder)
        treasure_emb = self._encode_organ_slots(treasure_feats, treasure_encoder)
        buff_emb = self._encode_organ_slots(buff_feats, buff_encoder)
        local_map_emb = local_map_encoder(local_map_feat)
        # progress_feat[:, 1] 用作 global map 分支门控：<enable_step 时为0，达到阈值后为1。
        map_gate = progress_feat[:, 1:2].clamp(min=0.0, max=1.0)
        map_emb = map_encoder(self._reshape_global_map(global_map_feat)) * map_gate
        return torch.cat(
            [hero_emb, monster_emb, treasure_emb, buff_emb, local_map_emb, map_emb, progress_feat],
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
        (
            hero_feat,
            monster_feats,
            treasure_feats,
            buff_feats,
            local_map_feat,
            global_map_feat,
            progress_feat,
        ) = self._split_obs_groups(obs)

        actor_fused = self._encode_branch(
            hero_feat,
            monster_feats,
            treasure_feats,
            buff_feats,
            local_map_feat,
            global_map_feat,
            progress_feat,
            self.actor_hero_encoder,
            self.actor_monster_encoder,
            self.actor_treasure_encoder,
            self.actor_buff_encoder,
            self.actor_local_map_encoder,
            self.actor_map_encoder,
        )
        critic_fused = self._encode_branch(
            hero_feat,
            monster_feats,
            treasure_feats,
            buff_feats,
            local_map_feat,
            global_map_feat,
            progress_feat,
            self.critic_hero_encoder,
            self.critic_monster_encoder,
            self.critic_treasure_encoder,
            self.critic_buff_encoder,
            self.critic_local_map_encoder,
            self.critic_map_encoder,
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
