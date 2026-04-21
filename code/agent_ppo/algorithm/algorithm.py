#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

PPO algorithm implementation for Gorge Chase PPO.
峡谷追猎 PPO 算法实现。

损失组成：
    total_loss = vf_coef * value_loss + policy_loss - beta * entropy_loss

  - value_loss  : Clipped value function loss（裁剪价值函数损失）
  - policy_loss : PPO Clipped surrogate objective（PPO 裁剪替代目标）
  - entropy_loss: Action entropy regularization（动作熵正则化，鼓励探索）
"""

import os
import time

import numpy as np
import torch
from agent_ppo.conf.conf import Config


class Algorithm:
    def __init__(self, model, optimizer, device=None, logger=None, monitor=None):
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.parameters = [p for pg in self.optimizer.param_groups for p in pg["params"]]
        self.logger = logger
        self.monitor = monitor

        self.label_size = Config.ACTION_NUM
        self.value_num = Config.VALUE_NUM
        self.var_beta = Config.BETA_START
        self.vf_coef = Config.VF_COEF
        self.clip_param = Config.CLIP_PARAM

        self.last_report_monitor_time = 0
        self.train_step = 0

    def learn(self, list_sample_data):
        """Training entry: PPO update on a batch of SampleData.

        训练入口：对一批 SampleData 执行 PPO 更新。
        """
        self._refresh_targets_with_current_critic(list_sample_data)
        list_sample_data = self._priority_resample_batch(list_sample_data)

        obs = torch.stack([f.obs for f in list_sample_data]).to(self.device)
        legal_action = torch.stack([f.legal_action for f in list_sample_data]).to(self.device)
        act = torch.stack([f.act for f in list_sample_data]).to(self.device).view(-1, 1)
        old_prob = torch.stack([f.prob for f in list_sample_data]).to(self.device)
        reward = torch.stack([f.reward for f in list_sample_data]).to(self.device)
        advantage = torch.stack([f.advantage for f in list_sample_data]).to(self.device)
        old_value = torch.stack([f.value for f in list_sample_data]).to(self.device)
        reward_sum = torch.stack([f.reward_sum for f in list_sample_data]).to(self.device)

        self.model.set_train_mode()
        self.optimizer.zero_grad()

        logits, value_pred = self.model(obs)

        total_loss, info_list = self._compute_loss(
            logits=logits,
            value_pred=value_pred,
            legal_action=legal_action,
            old_action=act,
            old_prob=old_prob,
            advantage=advantage,
            old_value=old_value,
            reward_sum=reward_sum,
            reward=reward,
        )

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters, Config.GRAD_CLIP_RANGE)
        self.optimizer.step()
        self.train_step += 1

        adv_raw = advantage.view(-1)
        adv_mean = float(adv_raw.mean().item())
        adv_std = float(adv_raw.std(unbiased=False).item())
        adv_abs_mean = float(torch.abs(adv_raw).mean().item())

        now = time.time()
        if now - self.last_report_monitor_time >= 60:
            results = {
                "total_loss": round(total_loss.item(), 4),
                "value_loss": round(info_list[0].item(), 4),
                "policy_loss": round(info_list[1].item(), 4),
                "entropy_loss": round(info_list[2].item(), 4),
                "explained_variance": round(info_list[3].item(), 6),
                "clip_count": round(info_list[4].item(), 4),
                "clip_rate": round(info_list[5].item(), 6),
                "clip_abs_overflow": round(info_list[6].item(), 6),
                "reward": round(reward.mean().item(), 4),
                "adv": round(adv_abs_mean, 6),
                "adv_mean": round(adv_mean, 6),
                "adv_std": round(adv_std, 6),
            }
            self.logger.info(
                f"[train] total_loss:{results['total_loss']} "
                f"policy_loss:{results['policy_loss']} "
                f"value_loss:{results['value_loss']} "
                f"entropy:{results['entropy_loss']} "
                f"explained_variance:{results['explained_variance']} "
                f"clip_count:{results['clip_count']} "
                f"clip_rate:{results['clip_rate']} "
                f"clip_abs_overflow:{results['clip_abs_overflow']} "
                f"adv:{results['adv']} "
                f"adv_mean:{results['adv_mean']} "
                f"adv_std:{results['adv_std']}"
            )
            if self.monitor:
                self.monitor.put_data({os.getpid(): results})
            self.last_report_monitor_time = now

    def _refresh_targets_with_current_critic(self, list_sample_data):
        """Refresh reward_sum/advantage with current critic to reduce stale targets.

        使用当前 Critic 基于 (obs, next_obs, reward, done) 重新计算目标，
        避免长期复用旧 value 估计带来的 target 过时问题。
        """
        if not list_sample_data:
            return

        def _column_tensor(values):
            t = torch.stack(values).to(self.device)
            return t.reshape(len(values), -1)[:, :1]

        obs = torch.stack([f.obs for f in list_sample_data]).to(self.device)
        obs = obs.reshape(len(list_sample_data), -1)

        next_obs_items = []
        for sample in list_sample_data:
            cur_obs = sample.obs
            candidate_next_obs = getattr(sample, "next_obs", None)
            if candidate_next_obs is None:
                next_obs_items.append(cur_obs)
                continue

            cur_obs_numel = int(torch.as_tensor(cur_obs).numel())
            next_obs_numel = int(torch.as_tensor(candidate_next_obs).numel())
            if next_obs_numel != cur_obs_numel:
                next_obs_items.append(cur_obs)
            else:
                next_obs_items.append(candidate_next_obs)

        next_obs = torch.stack(next_obs_items).to(self.device)
        next_obs = next_obs.reshape(len(list_sample_data), -1)

        reward = _column_tensor([f.reward for f in list_sample_data])
        done = _column_tensor([f.done for f in list_sample_data])

        self.model.set_eval_mode()
        with torch.no_grad():
            _, value_current = self.model(obs)
            _, next_value_current = self.model(next_obs)

        value_current = value_current.reshape(len(list_sample_data), -1)[:, :1]
        next_value_current = next_value_current.reshape(len(list_sample_data), -1)[:, :1]

        not_done = 1.0 - done
        # Transition-level replay does not guarantee full trajectory order;
        # use one-step bootstrap targets with current critic values.
        delta = reward + Config.GAMMA * not_done * next_value_current - value_current
        advantage = delta
        reward_sum = advantage + value_current

        advantage_cpu = advantage.detach().cpu()
        reward_sum_cpu = reward_sum.detach().cpu()
        for i, sample in enumerate(list_sample_data):
            sample.advantage = advantage_cpu[i].reshape(-1)[:1]
            sample.reward_sum = reward_sum_cpu[i].reshape(-1)[:1]

    def _priority_resample_batch(self, list_sample_data):
        """Resample a batch according to reward_sum + advantage priority.

        高优先级样本允许重复，低优先级样本允许丢弃，然后重排到固定 batch size。
        """
        if not bool(getattr(Config, "PRIORITY_REPLAY_ENABLE", True)):
            return list_sample_data

        batch_size = len(list_sample_data)
        if batch_size <= 1:
            return list_sample_data

        priorities = []
        for sample in list_sample_data:
            reward_sum = float(np.asarray(sample.reward_sum, dtype=np.float32).reshape(-1)[0])
            advantage = float(np.asarray(sample.advantage, dtype=np.float32).reshape(-1)[0])
            priority = reward_sum + advantage
            if not np.isfinite(priority):
                priority = 0.0
            priorities.append(priority)

        priority_arr = np.asarray(priorities, dtype=np.float64)
        priority_mean = float(np.mean(priority_arr))
        priority_std = float(np.std(priority_arr))
        normalized = (priority_arr - priority_mean) / max(priority_std, 1e-6)

        temperature = max(float(getattr(Config, "PRIORITY_REPLAY_TEMPERATURE", 1.0)), 1e-6)
        logits = normalized / temperature
        logits = logits - float(np.max(logits))
        weights = np.exp(logits)
        weight_sum = float(np.sum(weights))
        if not np.isfinite(weight_sum) or weight_sum <= 1e-12:
            probs = np.ones(batch_size, dtype=np.float64) / float(batch_size)
        else:
            probs = weights / weight_sum

        uniform_ratio = float(getattr(Config, "PRIORITY_REPLAY_UNIFORM_RATIO", 0.1))
        uniform_ratio = min(max(uniform_ratio, 0.0), 1.0)
        probs = (1.0 - uniform_ratio) * probs + uniform_ratio * (np.ones(batch_size, dtype=np.float64) / float(batch_size))
        probs = probs / float(np.sum(probs))

        sample_indices = np.random.choice(batch_size, size=batch_size, replace=True, p=probs)
        np.random.shuffle(sample_indices)
        return [list_sample_data[int(idx)] for idx in sample_indices]

    def _compute_loss(
        self,
        logits,
        value_pred,
        legal_action,
        old_action,
        old_prob,
        advantage,
        old_value,
        reward_sum,
        reward,
    ):
        """Compute standard PPO loss (policy + value + entropy).

        计算标准 PPO 损失（策略损失 + 价值损失 + 熵正则化）。
        """
        # Masked softmax / 合法动作掩码 softmax
        prob_dist = self._masked_softmax(logits, legal_action)

        # Policy loss (PPO Clip) / 策略损失
        one_hot = torch.nn.functional.one_hot(old_action[:, 0].long(), self.label_size).float()
        new_prob = (one_hot * prob_dist).sum(1, keepdim=True)
        old_action_prob = (one_hot * old_prob).sum(1, keepdim=True).clamp(1e-9)
        ratio = new_prob / old_action_prob
        clip_low = 1.0 - self.clip_param
        clip_high = 1.0 + self.clip_param
        adv = advantage.view(-1, 1)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        policy_loss1 = -ratio * adv
        clipped_ratio = ratio.clamp(clip_low, clip_high)
        policy_loss2 = -clipped_ratio * adv
        policy_loss = torch.maximum(policy_loss1, policy_loss2).mean()

        # Clip monitor stats / 裁剪监控统计
        clipped_mask = (ratio < clip_low) | (ratio > clip_high)
        clip_count = clipped_mask.float().sum()
        clip_rate = clipped_mask.float().mean()
        clip_abs_overflow = torch.abs(ratio - clipped_ratio).mean()

        # Value loss (Clipped) / 价值损失
        vp = value_pred
        ov = old_value
        tdret = reward_sum
        value_clip = ov + (vp - ov).clamp(-self.clip_param, self.clip_param)
        value_loss = (
            0.5
            * torch.maximum(
                torch.square(tdret - vp),
                torch.square(tdret - value_clip),
            ).mean()
        )

        # Entropy loss / 熵损失
        entropy_loss = (-prob_dist * torch.log(prob_dist.clamp(1e-9, 1))).sum(1).mean()

        explained_variance = self._compute_explained_variance(value_pred, reward_sum)

        # Total loss / 总损失
        total_loss = self.vf_coef * value_loss + policy_loss - self.var_beta * entropy_loss

        return total_loss, [
            value_loss,
            policy_loss,
            entropy_loss,
            explained_variance,
            clip_count,
            clip_rate,
            clip_abs_overflow,
        ]

    def _compute_explained_variance(self, value_pred, target):
        """Compute explained variance between predicted value and target return."""
        y = target.detach().view(-1)
        y_pred = value_pred.detach().view(-1)
        y_var = torch.var(y, unbiased=False)
        if float(y_var.item()) <= 1e-12:
            return torch.zeros(1, device=y.device, dtype=y.dtype).squeeze(0)
        ev = 1.0 - torch.var(y - y_pred, unbiased=False) / y_var
        return torch.clamp(ev, -1.0, 1.0)

    def _masked_softmax(self, logits, legal_action):
        """Softmax with legal action masking (suppress illegal actions).

        合法动作掩码下的 softmax（将非法动作概率压为极小值）。
        """
        label_max, _ = torch.max(logits * legal_action, dim=1, keepdim=True)
        label = logits - label_max
        label = label * legal_action
        label = label + 1e5 * (legal_action - 1)
        return torch.nn.functional.softmax(label, dim=1)
