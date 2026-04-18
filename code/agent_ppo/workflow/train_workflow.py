#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Training workflow for Gorge Chase PPO.
峡谷追猎 PPO 训练工作流。

训练流程：
1. 正式PPO训练阶段
   - 训练/评估方案（AGENTS规则）：
     - 10张地图：7张训练 [1-7]，3张评估 [8-10]
     - 每10局训练加入1局评估
     - 训练局：从train_maps随机抽取
     - 评估局：从eval_maps随机抽取
"""

import copy
import os
import time
import random

import numpy as np
from agent_ppo.feature.definition import SampleData, sample_process
from tools.metrics_utils import get_training_metrics
from tools.train_env_conf_validate import read_usr_conf
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery
from collections import deque

def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    last_save_model_time = time.time()
    env = envs[0]
    agent = agents[0]

    usr_conf = read_usr_conf("agent_ppo/conf/train_env_conf.toml", logger)
    if usr_conf is None:
        logger.error("usr_conf is None, please check agent_ppo/conf/train_env_conf.toml")
        return

    # ========== Formal PPO Training / 正式PPO训练阶段 ==========
    episode_runner = EpisodeRunner(
        env=env,
        agent=agent,
        usr_conf=usr_conf,
        logger=logger,
        monitor=monitor,
    )

    while True:
        for g_data in episode_runner.run_episodes():
            agent.send_sample_data(g_data)
            g_data.clear()

            now = time.time()
            if now - last_save_model_time >= 1800:
                agent.save_model()
                last_save_model_time = now


class EpisodeRunner:
    """Episode runner for formal PPO training with train/eval alternation.
    """

    def __init__(self, env, agent, usr_conf, logger, monitor):
        self.env = env
        self.agent = agent
        self.usr_conf = usr_conf
        self.logger = logger
        self.monitor = monitor
        self.episode_cnt = 0
        self.train_episode_cnt = 0
        self.eval_episode_cnt = 0
        self.eval_win_count = 0
        self.last_report_monitor_time = 0
        self.last_get_training_metrics_time = 0
        self.eval_pending = False

        self.train_maps = usr_conf.get("env_conf", {}).get("train_maps", [1, 2, 3, 4, 5, 6, 7])
        self.eval_maps = usr_conf.get("env_conf", {}).get("eval_maps", [8, 9, 10])
        self.eval_interval = usr_conf.get("env_conf", {}).get("eval_interval", 10)
        self.map_random = usr_conf.get("env_conf", {}).get("map_random", True)
        self.max_step = int(usr_conf.get("env_conf", {}).get("max_step", 1000))

        # ================= curriculum config =================
        self.curriculum_cfg = usr_conf.get("curriculum", {})
        self.curriculum_stage = int(self.curriculum_cfg.get("initial_stage", 1))
        self.curriculum_stage = max(1, min(4, self.curriculum_stage))

        self.curriculum_window_size = int(self.curriculum_cfg.get("metric_window_size", 30))
        self.curriculum_min_train_episodes = int(
            self.curriculum_cfg.get("min_train_episodes_per_stage", 30)
        )

        self.stage_metric_window = deque(maxlen=self.curriculum_window_size)
        self.stage_train_episode_cnt = 0
        self.stage_transition_cnt = 0
        self._reset_reward_accumulators()

    def _reset_reward_accumulators(self):
        self.reward_components = {
            "survive_reward": 0.0,
            "treasure_reward": 0.0,
            "speed_buff_reward": 0.0,
            "speed_buff_approach_reward": 0.0,
            "treasure_approach_reward": 0.0,
            "monster_dist_shaping": 0.0,
            "late_survive_reward": 0.0,
            "danger_penalty": 0.0,
            "wall_collision_penalty": 0.0,
            "flash_fail_penalty": 0.0,
            "flash_escape_reward": 0.0,
            "flash_survival_reward": 0.0,
            "speed_buff_escape_reward": 0.0,
            "safe_zone_reward": 0.0,
            "flash_abuse_penalty": 0.0,
            "flash_abuse_penalty_caught": 0.0,
            "exploration_reward": 0.0,
            "centroid_away_reward": 0.0,
            "idle_wander_penalty": 0.0,
            "dead_end_penalty": 0.0,
        }

    def _update_reward_accumulators(self, reward_info):
        if reward_info:
            for key in self.reward_components:
                if key in reward_info:
                    self.reward_components[key] += reward_info[key]

    def _get_reward_monitor_data(self):
        return {k: round(v, 4) for k, v in self.reward_components.items()}

    def _get_curriculum_stage(self):
        preprocessor = getattr(self.agent, "preprocessor", None)
        if preprocessor is None:
            return 1
        return int(getattr(preprocessor, "curriculum_stage", 1))

    def _get_curriculum_stage_name(self, stage=None):
        if stage is None:
            stage = self._get_curriculum_stage()
        stage_name_map = {
            1: "survival_base",
            2: "explore_and_stabilize",
            3: "safe_resource_acquisition",
            4: "full_game_and_skill_refine",
        }
        return stage_name_map.get(int(stage), "unknown")

    def _apply_curriculum_stage_to_agent(self):
        if hasattr(self.agent, "preprocessor") and self.agent.preprocessor is not None:
            self.agent.preprocessor.set_curriculum_stage(self.curriculum_stage)

    def _get_stage_name(self):
        if hasattr(self.agent, "preprocessor") and self.agent.preprocessor is not None:
            return self.agent.preprocessor.get_curriculum_stage_name()
        stage_name_map = {
            1: "survival_base",
            2: "explore_and_stabilize",
            3: "safe_resource_acquisition",
            4: "full_game_and_skill_refine",
        }
        return stage_name_map.get(self.curriculum_stage, "unknown")

    def _safe_div(self, num, den, default=0.0):
        den = float(den)
        if abs(den) <= 1e-8:
            return float(default)
        return float(num) / den

    def _estimate_event_count(self, reward_value, reward_name, default_coef=1.0):
        coef = default_coef
        if hasattr(self.agent, "preprocessor") and self.agent.preprocessor is not None:
            coef = abs(self.agent.preprocessor.get_reward_term_coef(reward_name, "coef", default_coef))
        coef = float(abs(coef))
        if coef <= 1e-8:
            return 0.0
        return abs(float(reward_value)) / coef

    def _build_episode_metrics(self, step, sim_score):
        rc = self.reward_components

        survival_rate = self._safe_div(step, self.max_step, 0.0)

        wall_collision_rate = self._safe_div(
            self._estimate_event_count(rc.get("wall_collision_penalty", 0.0), "wall_collision_penalty", 0.1),
            step,
            0.0,
        )

        danger_penalty_per_step = self._safe_div(
            abs(rc.get("danger_penalty", 0.0)),
            step,
            0.0,
        )

        idle_penalty_per_step = self._safe_div(
            abs(rc.get("idle_wander_penalty", 0.0)),
            step,
            0.0,
        )

        dead_end_penalty_per_step = self._safe_div(
            abs(rc.get("dead_end_penalty", 0.0)),
            step,
            0.0,
        )

        exploration_score = float(
            rc.get("exploration_reward", 0.0) + rc.get("centroid_away_reward", 0.0)
        )

        treasure_count = self._estimate_event_count(
            rc.get("treasure_reward", 0.0), "treasure_reward", 1.0
        )

        buff_count = self._estimate_event_count(
            rc.get("speed_buff_reward", 0.0), "speed_buff_reward", 1.0
        )

        treasure_approach_reward = float(rc.get("treasure_approach_reward", 0.0))

        return {
            "survival_rate": float(survival_rate),
            "wall_collision_rate": float(wall_collision_rate),
            "danger_penalty_per_step": float(danger_penalty_per_step),
            "idle_penalty_per_step": float(idle_penalty_per_step),
            "dead_end_penalty_per_step": float(dead_end_penalty_per_step),
            "exploration_score": float(exploration_score),
            "treasure_count": float(treasure_count),
            "buff_count": float(buff_count),
            "treasure_approach_reward": float(treasure_approach_reward),
            "sim_score": float(sim_score),
        }

    def _mean_metric(self, name):
        if not self.stage_metric_window:
            return 0.0
        return float(np.mean([m.get(name, 0.0) for m in self.stage_metric_window]))

    def _check_stage_transition_ready(self):
        if self.curriculum_stage >= 4:
            return False, {}
        if self.stage_train_episode_cnt < self.curriculum_min_train_episodes:
            return False, {}
        if len(self.stage_metric_window) < self.curriculum_window_size:
            return False, {}

        avg_metrics = {
            "survival_rate": self._mean_metric("survival_rate"),
            "wall_collision_rate": self._mean_metric("wall_collision_rate"),
            "danger_penalty_per_step": self._mean_metric("danger_penalty_per_step"),
            "idle_penalty_per_step": self._mean_metric("idle_penalty_per_step"),
            "dead_end_penalty_per_step": self._mean_metric("dead_end_penalty_per_step"),
            "exploration_score": self._mean_metric("exploration_score"),
            "treasure_count": self._mean_metric("treasure_count"),
            "buff_count": self._mean_metric("buff_count"),
            "treasure_approach_reward": self._mean_metric("treasure_approach_reward"),
            "sim_score": self._mean_metric("sim_score"),
        }

        if self.curriculum_stage == 1:
            cond = (
                avg_metrics["survival_rate"] >= float(
                    self.curriculum_cfg.get("stage1_to_stage2", {}).get("avg_survival_rate", 0.35)
                )
                and avg_metrics["wall_collision_rate"] <= float(
                    self.curriculum_cfg.get("stage1_to_stage2", {}).get("max_wall_collision_rate", 0.025)
                )
                and avg_metrics["danger_penalty_per_step"] <= float(
                    self.curriculum_cfg.get("stage1_to_stage2", {}).get("max_danger_penalty_per_step", 0.015)
                )
            )
            return cond, avg_metrics

        if self.curriculum_stage == 2:
            cond = (
                avg_metrics["survival_rate"] >= float(
                    self.curriculum_cfg.get("stage2_to_stage3", {}).get("avg_survival_rate", 0.55)
                )
                and avg_metrics["exploration_score"] >= float(
                    self.curriculum_cfg.get("stage2_to_stage3", {}).get("min_exploration_score", 0.08)
                )
                and avg_metrics["idle_penalty_per_step"] <= float(
                    self.curriculum_cfg.get("stage2_to_stage3", {}).get("max_idle_penalty_per_step", 0.010)
                )
                and avg_metrics["dead_end_penalty_per_step"] <= float(
                    self.curriculum_cfg.get("stage2_to_stage3", {}).get("max_dead_end_penalty_per_step", 0.004)
                )
            )
            return cond, avg_metrics

        if self.curriculum_stage == 3:
            cond = (
                avg_metrics["survival_rate"] >= float(
                    self.curriculum_cfg.get("stage3_to_stage4", {}).get("avg_survival_rate", 0.70)
                )
                and avg_metrics["treasure_count"] >= float(
                    self.curriculum_cfg.get("stage3_to_stage4", {}).get("min_treasure_count", 0.80)
                )
                and avg_metrics["buff_count"] >= float(
                    self.curriculum_cfg.get("stage3_to_stage4", {}).get("min_buff_count", 0.20)
                )
                and avg_metrics["treasure_approach_reward"] >= float(
                    self.curriculum_cfg.get("stage3_to_stage4", {}).get("min_treasure_approach_reward", 0.03)
                )
            )
            return cond, avg_metrics

        return False, avg_metrics

    def _maybe_advance_curriculum_stage(self):
        ready, avg_metrics = self._check_stage_transition_ready()
        if not ready:
            return

        old_stage = self.curriculum_stage
        self.curriculum_stage = min(4, self.curriculum_stage + 1)
        self.stage_transition_cnt += 1
        self.stage_metric_window.clear()
        self.stage_train_episode_cnt = 0
        self._apply_curriculum_stage_to_agent()

        self.logger.info(
            "[curriculum] stage advance: %d -> %d (%s) | "
            "survival_rate=%.4f wall_collision_rate=%.4f danger_per_step=%.4f "
            "exploration_score=%.4f idle_per_step=%.4f dead_end_per_step=%.4f "
            "treasure_count=%.4f buff_count=%.4f treasure_approach=%.4f sim_score=%.4f"
            % (
                old_stage,
                self.curriculum_stage,
                self._get_stage_name(),
                avg_metrics.get("survival_rate", 0.0),
                avg_metrics.get("wall_collision_rate", 0.0),
                avg_metrics.get("danger_penalty_per_step", 0.0),
                avg_metrics.get("exploration_score", 0.0),
                avg_metrics.get("idle_penalty_per_step", 0.0),
                avg_metrics.get("dead_end_penalty_per_step", 0.0),
                avg_metrics.get("treasure_count", 0.0),
                avg_metrics.get("buff_count", 0.0),
                avg_metrics.get("treasure_approach_reward", 0.0),
                avg_metrics.get("sim_score", 0.0),
            )
        )

    def _is_eval_episode(self):
        if self.eval_interval <= 0 or not self.eval_maps:
            return False

        # 仅在“刚达到训练间隔”时置位一次，消费后立即清零，
        # 避免达到阈值后连续多局都被判为eval。
        if (not self.eval_pending
                and self.train_episode_cnt > 0
                and self.train_episode_cnt % self.eval_interval == 0):
            self.eval_pending = True

        if self.eval_pending:
            self.eval_pending = False
            return True

        return False

    def run_episodes(self):
        """Run episodes with train/eval alternation.
        """
        while True:
            now = time.time()
            if now - self.last_get_training_metrics_time >= 60:
                training_metrics = get_training_metrics()
                self.last_get_training_metrics_time = now
                if training_metrics is not None:
                    self.logger.info(f"training_metrics is {training_metrics}")

            is_eval = self._is_eval_episode()
            if is_eval:
                selected_map = random.choice(self.eval_maps)
                mode = "eval"
                self.eval_episode_cnt += 1
            else:
                selected_map = random.choice(self.train_maps)
                mode = "train"
                self.train_episode_cnt += 1

            episode_conf = copy.deepcopy(self.usr_conf)
            episode_conf["env_conf"]["map"] = [selected_map]
            episode_conf["mode"] = mode

            env_obs = self.env.reset(episode_conf)
            if handle_disaster_recovery(env_obs, self.logger):
                continue

            self.agent.reset(env_obs)
            self._apply_curriculum_stage_to_agent()
            self.agent.load_model(id="latest")

            obs_data, remain_info = self.agent.observation_process(env_obs)
            collector = []

            self.episode_cnt += 1
            done = False
            step = 0
            total_reward = 0.0

            # 记录最近一次闪现动作所在的样本帧索引，用于延迟奖励/惩罚离线回填
            last_flash_frame_idx = None

            self._reset_reward_accumulators()

            curr_stage = self._get_curriculum_stage()
            curr_stage_name = self._get_curriculum_stage_name(curr_stage)

            self.logger.info(
                f"Episode {self.episode_cnt} start | "
                f"Mode: {mode} | Map: {selected_map} | "
                f"Stage: {curr_stage} ({curr_stage_name})"
            )

            while not done:
                act_data = self.agent.predict(list_obs_data=[obs_data])[0]
                act = self.agent.action_process(act_data)

                env_reward, env_obs = self.env.step(act)

                if handle_disaster_recovery(env_obs, self.logger):
                    break

                terminated = env_obs["terminated"]
                truncated = env_obs["truncated"]
                step += 1
                done = terminated or truncated

                _obs_data, _remain_info = self.agent.observation_process(env_obs)

                reward = np.array(_remain_info.get("reward", [0.0]), dtype=np.float32)
                total_reward += float(reward[0])

                reward_info = _remain_info.get("reward_info", {})
                self._update_reward_accumulators(reward_info)

                # 将延迟闪现奖励/惩罚A回填到闪现动作帧：
                # 1) 危险闪现成功奖励 flash_escape_reward
                # 2) 10%CD后触发的闪现衰减存活奖励 flash_survival_reward
                # 3) 闪现后10步内被抓惩罚 flash_abuse_penalty_caught
                flash_delayed_credit = (
                    float(reward_info.get("flash_escape_reward", 0.0))
                    + float(reward_info.get("flash_survival_reward", 0.0))
                    + float(reward_info.get("flash_abuse_penalty_caught", 0.0))
                )

                # 当前步reward先扣除延迟项，避免重复记到非闪现动作
                reward_for_current = reward.copy()
                if abs(flash_delayed_credit) > 1e-12:
                    reward_for_current = reward_for_current - np.array(
                        [flash_delayed_credit], dtype=np.float32
                    )

                    # 回填到对应闪现动作帧（若存在）
                    if (
                        last_flash_frame_idx is not None
                        and 0 <= last_flash_frame_idx < len(collector)
                    ):
                        collector[last_flash_frame_idx].reward = (
                            collector[last_flash_frame_idx].reward
                            + np.array([flash_delayed_credit], dtype=np.float32)
                        )

                final_reward = np.zeros(1, dtype=np.float32)
                sim_score = 0
                if done:
                    env_info = env_obs["observation"]["env_info"]
                    sim_score = env_info.get("total_score", 0)

                    if terminated:
                        final_reward[0] = -50.0
                        result_str = "FAIL"
                    else:
                        final_reward[0] = 50.0
                        result_str = "WIN"
                        if mode == "eval":
                            self.eval_win_count += 1

                    eval_win_rate = 0.0
                    if self.eval_episode_cnt > 0:
                        eval_win_rate = self.eval_win_count / self.eval_episode_cnt

                    curr_stage = self._get_curriculum_stage()
                    curr_stage_name = self._get_curriculum_stage_name(curr_stage)

                    self.logger.info(
                        f"[GAMEOVER] episode:{self.episode_cnt} mode:{mode} map:{selected_map} "
                        f"stage:{curr_stage}({curr_stage_name}) "
                        f"steps:{step} result:{result_str} sim_score:{sim_score:.1f} "
                        f"total_reward:{total_reward:.3f} eval_win_rate:{eval_win_rate:.2%}"
                    )

                    if mode == "train":
                        episode_metrics = self._build_episode_metrics(step=step, sim_score=sim_score)
                        self.stage_metric_window.append(episode_metrics)
                        self.stage_train_episode_cnt += 1

                        self.logger.info(
                            "[curriculum] stage=%d (%s) | "
                            "stage_train_eps=%d | window=%d/%d | "
                            "survival_rate=%.4f wall_collision_rate=%.4f danger_per_step=%.4f "
                            "exploration_score=%.4f idle_per_step=%.4f dead_end_per_step=%.4f "
                            "treasure_count=%.4f buff_count=%.4f treasure_approach=%.4f sim_score=%.4f"
                            % (
                                self.curriculum_stage,
                                self._get_stage_name(),
                                self.stage_train_episode_cnt,
                                len(self.stage_metric_window),
                                self.curriculum_window_size,
                                episode_metrics["survival_rate"],
                                episode_metrics["wall_collision_rate"],
                                episode_metrics["danger_penalty_per_step"],
                                episode_metrics["exploration_score"],
                                episode_metrics["idle_penalty_per_step"],
                                episode_metrics["dead_end_penalty_per_step"],
                                episode_metrics["treasure_count"],
                                episode_metrics["buff_count"],
                                episode_metrics["treasure_approach_reward"],
                                episode_metrics["sim_score"],
                            )
                        )

                        self._maybe_advance_curriculum_stage()

                # 终局奖励并入训练用 reward，确保 GAE / PPO 直接学习到胜负信号。
                shaped_reward = reward_for_current + final_reward

                frame = SampleData(
                    obs=np.array(obs_data.feature, dtype=np.float32),
                    legal_action=np.array(obs_data.legal_action, dtype=np.float32),
                    act=np.array([act_data.action[0]], dtype=np.float32),
                    reward=shaped_reward,
                    done=np.array([float(done)], dtype=np.float32),
                    reward_sum=np.zeros(1, dtype=np.float32),
                    value=np.array(act_data.value, dtype=np.float32).flatten()[:1],
                    next_value=np.zeros(1, dtype=np.float32),
                    advantage=np.zeros(1, dtype=np.float32),
                    prob=np.array(act_data.prob, dtype=np.float32),
                )
                collector.append(frame)

                # 当前动作为闪现时，记录其样本帧索引，供后续延迟项回填
                if 8 <= act <= 15:
                    last_flash_frame_idx = len(collector) - 1

                if done:
                    now = time.time()
                    if now - self.last_report_monitor_time >= 60 and self.monitor:
                        env_info = env_obs["observation"].get("env_info", {})
                        train_pool_size = int(len(self.train_maps))
                        eval_pool_size = int(len(self.eval_maps))
                        configured_total_map = int(train_pool_size + eval_pool_size)
                        curr_stage = self._get_curriculum_stage()

                        monitor_data = {
                            "reward": round(total_reward + float(final_reward[0]), 4),
                            "episode_steps": step,
                            "episode_cnt": self.episode_cnt,
                            "sim_score": sim_score,
                            "mode": 1 if mode == "eval" else 0,
                            "map_id": selected_map,

                            # total_map展示为训练+评估地图总数（固定配置总量）
                            "total_map": configured_total_map,
                            # 保留环境原始total_map便于排查（通常为单局传入map列表长度）
                            "env_total_map": int(env_info.get("total_map", 0) or 0),
                            "train_map_pool_size": train_pool_size,
                            "eval_map_pool_size": eval_pool_size,
                            "configured_total_map": configured_total_map,

                            # ===== 新增 =====
                            "curriculum_stage": curr_stage,
                            "curriculum_stage_transition_cnt": int(getattr(self, "stage_transition_cnt", 0)),
                        }

                        if mode == "train":
                            monitor_data["train_map_id"] = int(selected_map)
                        monitor_data.update(self._get_reward_monitor_data())
                        if self.eval_episode_cnt > 0:
                            monitor_data["eval_win_rate"] = round(
                                self.eval_win_count / self.eval_episode_cnt, 4
                            )

                        self.monitor.put_data({os.getpid(): monitor_data})
                        self.last_report_monitor_time = now

                    if collector:
                        collector = sample_process(collector)
                        yield collector
                    break

                obs_data = _obs_data
                remain_info = _remain_info
