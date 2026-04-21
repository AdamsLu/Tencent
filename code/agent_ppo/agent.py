#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Agent class for Gorge Chase PPO.
峡谷追猎 PPO Agent 主类。
"""

import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import numpy as np
from kaiwudrl.interface.agent import BaseAgent

from agent_ppo.algorithm.algorithm import Algorithm
from agent_ppo.conf.conf import Config
from agent_ppo.feature.definition import ActData, ObsData
from agent_ppo.feature.preprocessor import Preprocessor
from agent_ppo.model.model import Model


class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        torch.manual_seed(0)
        self.device = device
        self.model = Model(device).to(self.device)

        # Actor/Critic 全分离参数组：各自独立学习率。
        actor_params = list(self.model.actor_parameters())
        critic_params = list(self.model.critic_parameters())
        self.optimizer = torch.optim.Adam(
            params=[
                {"params": actor_params, "lr": Config.ACTOR_LEARNING_RATE_START},
                {"params": critic_params, "lr": Config.CRITIC_LEARNING_RATE_START},
            ],
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        self.algorithm = Algorithm(self.model, self.optimizer, self.device, logger, monitor)
        self.preprocessor = Preprocessor(logger=logger, monitor=monitor)
        self.last_action = -1
        self.logger = logger
        self.monitor = monitor
        super().__init__(agent_type, device, logger, monitor)

    def reset(self, env_obs=None):
        """Reset per-episode state.

        每局开始时重置状态。
        """
        self.preprocessor.reset()
        self.last_action = -1

    def set_episode_mode(self, mode):
        """Set current episode mode for preprocessor diagnostics routing."""
        if hasattr(self.preprocessor, "set_episode_mode"):
            self.preprocessor.set_episode_mode(mode)

    def set_approach_gravity_stage(self, stage):
        """Set approach gravity stage computed by workflow."""
        if hasattr(self.preprocessor, "set_approach_gravity_stage"):
            self.preprocessor.set_approach_gravity_stage(stage)

    def observation_process(self, env_obs):
        """Convert raw env_obs to ObsData and remain_info.

        将原始观测转换为 ObsData 和 remain_info。
        同时缓存地图记忆图像供后续模型推理使用。
        """
        feature, legal_action, reward = self.preprocessor.feature_process(
            env_obs, self.last_action
        )
        obs_data = ObsData(
            feature=list(feature),
            legal_action=legal_action,
        )
        reward_info = self.preprocessor.last_reward_info if hasattr(self.preprocessor, 'last_reward_info') else {}
        remain_info = {"reward": reward, "reward_info": reward_info}
        return obs_data, remain_info

    def predict(self, list_obs_data, list_state=None):
        """Stochastic inference for training (exploration).

        训练时随机采样动作（探索）。
        """
        list_act_data = []
        for obs_data in list_obs_data:
            feature = obs_data.feature
            legal_action = obs_data.legal_action

            _, value, prob = self._run_model(feature, legal_action)

            action = self._legal_sample_with_mask(prob, legal_action, use_max=False)
            d_action = self._legal_sample_with_mask(prob, legal_action, use_max=True)

            list_act_data.append(
                ActData(
                    action=[action],
                    d_action=[d_action],
                    prob=list(prob),
                    value=value,
                )
            )

        return list_act_data

    def exploit(self, env_obs):
        """Greedy inference for evaluation.

        评估时贪心选择动作（利用）。
        """
        obs_data, _ = self.observation_process(env_obs)
        act_data = self.predict([obs_data])
        return self.action_process(act_data[0], is_stochastic=False)

    def learn(self, list_sample_data):
        """Train the model.

        训练模型。
        """
        return self.algorithm.learn(list_sample_data)

    def save_model(self, path=None, id="1"):
        """Save model checkpoint.

        保存模型检查点。
        """
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        state_dict_cpu = {k: v.clone().cpu() for k, v in self.model.state_dict().items()}
        torch.save(state_dict_cpu, model_file_path)
        self.logger.info(f"save model {model_file_path} successfully")

    def load_model(self, path=None, id="1"):
        """Load model checkpoint.

        加载模型检查点。
        """
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        ckpt_state = torch.load(model_file_path, map_location=self.device)
        try:
            self.model.load_state_dict(ckpt_state)
            self.logger.info(f"load model {model_file_path} successfully")
            return
        except RuntimeError:
            # 兼容旧版共享网络权重：feature_encoders/encoder -> actor_* 与 critic_*。
            remapped_state = dict(ckpt_state)
            for k, v in ckpt_state.items():
                if k.startswith("feature_encoders."):
                    suffix = k[len("feature_encoders."):]
                    remapped_state[f"actor_feature_encoders.{suffix}"] = v
                    remapped_state[f"critic_feature_encoders.{suffix}"] = v
                elif k.startswith("encoder."):
                    suffix = k[len("encoder."):]
                    remapped_state[f"actor_encoder.{suffix}"] = v
                    remapped_state[f"critic_encoder.{suffix}"] = v

            missing_keys, unexpected_keys = self.model.load_state_dict(remapped_state, strict=False)
            self.logger.info(
                f"load model {model_file_path} with compatibility remap successfully; "
                f"missing_keys={len(missing_keys)}, unexpected_keys={len(unexpected_keys)}"
            )

    def action_process(self, act_data, is_stochastic=True):
        """Unpack ActData to int action and update last_action.

        解包 ActData 为 int 动作并记录 last_action。
        """
        action = act_data.action if is_stochastic else act_data.d_action
        self.last_action = int(action[0])
        return int(action[0])

    def _run_model(self, feature, legal_action):
        """Run model inference with vector input, return logits, value, prob.

        执行模型推理（单输入向量模式），返回 logits、value 和动作概率。
        """
        self.model.set_eval_mode()
        vec_tensor = torch.tensor(np.array([feature]), dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits, value = self.model(vec_tensor, inference=True)

        logits_np = logits.cpu().numpy()[0]
        value_np = value.cpu().numpy()[0]

        legal_action_np = np.array(legal_action, dtype=np.float32)
        prob = self._legal_soft_max(logits_np, legal_action_np)

        return logits_np, value_np, prob

    def _legal_soft_max(self, input_hidden, legal_action):
        """Softmax with legal action masking (numpy).

        合法动作掩码下的 softmax（numpy 版）。
        """
        _w, _e = 1e20, 1e-5
        tmp = input_hidden - _w * (1.0 - legal_action)
        tmp_max = np.max(tmp, keepdims=True)
        tmp = np.clip(tmp - tmp_max, -_w, 1)
        tmp = (np.exp(tmp) + _e) * legal_action
        prob_sum = float(np.sum(tmp))
        if prob_sum <= 1e-12:
            legal_sum = float(np.sum(legal_action))
            if legal_sum <= 1e-12:
                return np.ones_like(legal_action, dtype=np.float32) / float(len(legal_action))
            return legal_action / (legal_sum + 1e-6)
        return tmp / (prob_sum * 1.00001)

    def _legal_sample(self, probs, use_max=False):
        """Sample action from probability distribution.

        按概率分布采样动作。
        """
        if use_max:
            return int(np.argmax(probs))
        return int(np.argmax(np.random.multinomial(1, probs, size=1)))

    def _legal_sample_with_mask(self, probs, legal_action, use_max=False):
        """Sample action strictly inside legal_action support.

        在 legal_action 掩码约束下采样/贪心选动作，保证动作索引一定合法。
        """
        probs_np = np.asarray(probs, dtype=np.float64)
        mask_np = np.asarray(legal_action, dtype=np.float64)
        if probs_np.ndim != 1:
            probs_np = probs_np.reshape(-1)
        if mask_np.ndim != 1:
            mask_np = mask_np.reshape(-1)

        n = int(min(len(probs_np), len(mask_np)))
        if n <= 0:
            return 0

        probs_np = probs_np[:n]
        mask_np = mask_np[:n]
        legal_idx = np.flatnonzero(mask_np > 0.5)

        # 防御性兜底：若掩码异常全0，退化为全动作集合，避免崩溃。
        if len(legal_idx) == 0:
            legal_idx = np.arange(n, dtype=np.int64)

        legal_probs = np.clip(probs_np[legal_idx], 0.0, None)

        if use_max:
            if float(np.sum(legal_probs)) <= 1e-12:
                return int(legal_idx[0])
            return int(legal_idx[int(np.argmax(legal_probs))])

        prob_sum = float(np.sum(legal_probs))
        if prob_sum <= 1e-12:
            return int(np.random.choice(legal_idx))

        legal_probs = legal_probs / prob_sum
        sampled = np.random.multinomial(1, legal_probs, size=1)
        return int(legal_idx[int(np.argmax(sampled))])
