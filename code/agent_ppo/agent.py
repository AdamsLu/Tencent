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

import os
import json
import zipfile
import tempfile
import glob

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

        # 分组学习率：actor单独下调；共享encoder与critic使用原学习率。
        actor_params = list(self.model.actor_head.parameters())
        critic_params = list(self.model.critic_head.parameters())
        encoder_params = list(self.model.encoder.parameters())
        self.optimizer = torch.optim.Adam(
            params=[
                {"params": actor_params, "lr": Config.ACTOR_LEARNING_RATE_START},
                {"params": critic_params, "lr": Config.CRITIC_LEARNING_RATE_START},
                {"params": encoder_params, "lr": Config.CRITIC_LEARNING_RATE_START},
            ],
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        self.algorithm = Algorithm(self.model, self.optimizer, self.device, logger, monitor)
        self.preprocessor = Preprocessor()
        self.last_action = -1
        self.logger = logger
        self.monitor = monitor
        super().__init__(agent_type, device, logger, monitor)
        # ===== resume from packaged zip =====
        default_resume_zip = os.path.join(
            os.path.dirname(__file__),
            "init_models",
            "resume_model.zip",
        )
        self._resume_zip_path = os.environ.get(
            "KAIWU_RESUME_ZIP", default_resume_zip
        ).strip()

        # 在本次训练第一次成功 save_model 之前，始终优先用 zip 初始化，
        # 避免平台外层反复 load_model_by_source 时把模型又覆盖回“从头开始”的状态。
        self._resume_zip_hold_until_save = os.path.isfile(self._resume_zip_path)
        self._resume_zip_loaded = False
        self._resume_extract_dir = None

    def reset(self, env_obs=None):
        """Reset per-episode state.

        每局开始时重置状态。
        """
        self.preprocessor.reset()
        self.last_action = -1

    def _resolve_resume_model_from_zip(self):
        if not self._resume_zip_path or not os.path.isfile(self._resume_zip_path):
            return None

        if self._resume_extract_dir is None:
            self._resume_extract_dir = tempfile.mkdtemp(prefix="kaiwu_resume_")
            with zipfile.ZipFile(self._resume_zip_path, "r") as zf:
                zf.extractall(self._resume_extract_dir)

        meta_path = os.path.join(self._resume_extract_dir, "ckpt", "kaiwu.json")
        if os.path.isfile(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            model_paths = meta.get("model_file_path") or []
            if model_paths:
                candidate = os.path.join(self._resume_extract_dir, model_paths[0])
                if os.path.isfile(candidate):
                    return candidate

        # fallback: 自动找 ckpt 目录里的 pkl
        matches = sorted(
            glob.glob(os.path.join(self._resume_extract_dir, "ckpt", "model.ckpt-*.pkl"))
        )
        if matches:
            return matches[-1]

        return None

    def _try_load_resume_zip(self):
        if not self._resume_zip_hold_until_save:
            return False

        model_file_path = self._resolve_resume_model_from_zip()
        if not model_file_path:
            if self.logger:
                self.logger.warning(
                    f"resume zip enabled but no valid checkpoint found in {self._resume_zip_path}"
                )
            return False

        self.model.load_state_dict(torch.load(model_file_path, map_location=self.device))
        self._resume_zip_loaded = True

        if self.logger:
            self.logger.info(
                f"resume training from packaged zip model: {model_file_path}"
            )
        return True

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

            action = self._legal_sample(prob, use_max=False)
            d_action = self._legal_sample(prob, use_max=True)

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
        """Save model checkpoint."""
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        state_dict_cpu = {k: v.clone().cpu() for k, v in self.model.state_dict().items()}
        torch.save(state_dict_cpu, model_file_path)
        self.logger.info(f"save model {model_file_path} successfully")

        # 一旦本次训练已经成功保存出新模型，就恢复平台默认的 checkpoint 流程
        self._resume_zip_hold_until_save = False

    def load_model(self, path=None, id="1"):
        """Load model checkpoint.
        优先支持从指定 zip 初始化训练；在第一次 save_model 之前持续生效。
        """
        # 先尝试用用户指定的 zip 初始化
        if self._try_load_resume_zip():
            return

        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        self.model.load_state_dict(torch.load(model_file_path, map_location=self.device))
        self.logger.info(f"load model {model_file_path} successfully")

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
