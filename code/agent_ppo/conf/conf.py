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
    # Raw feature split / 原始特征切分（共647维，供分组编码器使用）
    #
    # hero:4
    # + monster0:14 + monster1:14
    # + treasure[10]*7
    # + speed_buff[2]*7
    # + map:441(21×21)
    # + progress:2
    # + candidate_action:56
    # + action_risk_benefit:32
    FEATURES = [
        4,   # hero
        14,  # monster0
        14,  # monster1

        # treasure[10] * 7
        7, 7, 7, 7, 7, 7, 7, 7, 7, 7,

        # speed_buff[2] * 7
        7, 7,

        441, # local map 21x21
        2,   # progress
        56,  # candidate action features
        32,  # action risk / benefit features
    ]
    FEATURE_SPLIT_SHAPE = FEATURES
    FEATURE_LEN = sum(FEATURE_SPLIT_SHAPE)
    DIM_OF_OBSERVATION = FEATURE_LEN

    # Group encoder output dims / 分组编码器输出维度
    HERO_ENCODER_DIM = 32
    MONSTER_ENCODER_DIM = 32
    TREASURE_ENCODER_DIM = 32
    SPEED_BUFF_ENCODER_DIM = 32

    # map/progress 直接拼接到融合层输入，不经过 embedding。
    FUSION_HIDDEN_DIM = 128

    # Action space / 动作空间：8个移动 + 8个闪现方向
    ACTION_NUM = 16

    # Movement legality / 移动合法性
    # False: 斜向切角非法；True: 允许斜向切角
    ALLOW_CORNER_CUT_MOVE = True

    # Value head / 价值头：单头生存奖励
    VALUE_NUM = 1

    # PPO hyperparameters / PPO 超参数
    GAMMA = 0.99
    LAMDA = 0.95

    # 分离学习率：仅下调 actor，critic 保持原值不变
    ACTOR_LEARNING_RATE_START = 0.0002
    CRITIC_LEARNING_RATE_START = 0.0003
    BETA_START = 0.001
    CLIP_PARAM = 0.2
    VF_COEF = 1.0
    GRAD_CLIP_RANGE = 0.5
    REWARD_SCALE = 0.02

