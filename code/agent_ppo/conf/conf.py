#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Configuration for Gorge Chase PPO.
峡谷追猎 PPO 配置。
"""


class Config:

    # Raw feature split / 原始特征切分（供分组编码器使用）
    #    hero:14(基础4 + 历史5步速度向量10)
    #    + monster0:40 + monster1:40
    #    + treasure[10]*40
    #    + speed_buff[2]*40
    #    + local_map:441(21×21，先过MLP编码)
    #    + global_map:16384(1×128×128，单张全局图，走CNN)
    #    + progress:2
    FEATURES = [
        14,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        441,
        16384,
        2,
    ]
    FEATURE_SPLIT_SHAPE = FEATURES
    FEATURE_LEN = sum(FEATURE_SPLIT_SHAPE)
    DIM_OF_OBSERVATION = FEATURE_LEN

    # Group encoder output dims / 分组编码器输出维度
    HERO_ENCODER_DIM = 32
    MONSTER_ENCODER_DIM = 32
    TREASURE_ENCODER_DIM = 32
    SPEED_BUFF_ENCODER_DIM = 32
    LOCAL_MAP_ENCODER_HIDDEN_DIM = 128
    LOCAL_MAP_ENCODER_DIM = 64
    MAP_ENCODER_DIM = 64
    GLOBAL_MAP_SIZE = 128
    GLOBAL_MAP_CHANNELS = 1
    # 在该步数之前，global map CNN 不接入 Actor/Critic（槽位补0）。
    GLOBAL_MAP_ENABLE_STEP = 1000
    # local_map/global_map/progress 由编码后拼接到融合层输入。
    FUSION_HIDDEN_DIM = 128

    # Action space / 动作空间：8个移动 + 8个闪现方向
    ACTION_NUM = 16

    # Movement legality / 移动合法性
    # False: 斜向移动必须两侧都可通行；True: 允许切角，但双侧同时封死时仍禁止
    ALLOW_CORNER_CUT_MOVE = True

    # Value head / 价值头：单头生存奖励
    VALUE_NUM = 1

    # PPO hyperparameters / PPO 超参数
    GAMMA = 0.99
    LAMDA = 0.95
    # 分离学习率：仅下调actor，critic保持原值不变
    ACTOR_LEARNING_RATE_START = 0.0002
    CRITIC_LEARNING_RATE_START = 0.0003
    BETA_START = 0.001
    CLIP_PARAM = 0.2
    VF_COEF = 1.0
    GRAD_CLIP_RANGE = 0.5
    REWARD_SCALE = 0.01
    PRIORITY_REPLAY_ENABLE = False
    PRIORITY_REPLAY_TEMPERATURE = 1.0
    PRIORITY_REPLAY_UNIFORM_RATIO = 0.1

