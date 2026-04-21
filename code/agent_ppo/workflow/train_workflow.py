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
from agent_ppo.conf.conf import Config
from agent_ppo.feature.definition import SampleData, sample_process
from tools.metrics_utils import get_training_metrics
from tools.train_env_conf_validate import read_usr_conf
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    if not envs:
        if logger is not None:
            logger.error("workflow received empty envs, skip current run")
        return
    if not agents:
        if logger is not None:
            logger.error("workflow received empty agents, skip current run")
        return

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
            "no_movement_case": 0.0,
            "move_mask_consistency_hit": 0.0,
            "move_mask_consistency_total": 0.0,
            "exploration_reward": 0.0,
            "visit_tracking_reward": 0.0,
            "centroid_away_reward": 0.0,
            "idle_wander_penalty": 0.0,
            "dead_end_penalty": 0.0,
        }
        self.treasure_circle_enter_total = 0.0
        self.treasure_circle_hit_total = 0.0
        self.treasure_circle_hit_rate = 0.0

    def _update_reward_accumulators(self, reward_info):
        if reward_info:
            for key in self.reward_components:
                if key in reward_info:
                    self.reward_components[key] += reward_info[key]
            if "treasure_circle_enter_total" in reward_info:
                self.treasure_circle_enter_total = float(reward_info["treasure_circle_enter_total"])
            if "treasure_circle_hit_total" in reward_info:
                self.treasure_circle_hit_total = float(reward_info["treasure_circle_hit_total"])
            if "treasure_circle_hit_rate" in reward_info:
                self.treasure_circle_hit_rate = float(reward_info["treasure_circle_hit_rate"])

    def _get_reward_monitor_data(self):
        data = {k: round(v, 4) for k, v in self.reward_components.items()}
        data["treasure_circle_enter_total"] = round(self.treasure_circle_enter_total, 4)
        data["treasure_circle_hit_total"] = round(self.treasure_circle_hit_total, 4)
        data["treasure_circle_hit_rate"] = round(self.treasure_circle_hit_rate, 4)
        return data

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

    def _is_monster_active(self, monster):
        if not isinstance(monster, dict):
            return False
        pos = monster.get("pos", None)
        if not isinstance(pos, dict):
            return False
        try:
            mx = int(pos.get("x", 0))
            mz = int(pos.get("z", 0))
        except Exception:
            return False
        return not (mx == 0 and mz == 0)

    def _resolve_approach_gravity_stage(self, env_obs):
        """Resolve stage in workflow, keeping preprocessor reward logic lightweight."""
        try:
            observation = env_obs.get("observation", {}) if isinstance(env_obs, dict) else {}
            frame_state = observation.get("frame_state", {}) if isinstance(observation, dict) else {}
            env_info = observation.get("env_info", {}) if isinstance(observation, dict) else {}

            monsters = frame_state.get("monsters", []) if isinstance(frame_state, dict) else []
            m0 = monsters[0] if len(monsters) > 0 else None
            m1 = monsters[1] if len(monsters) > 1 else None

            m0_active = self._is_monster_active(m0)
            m1_active = self._is_monster_active(m1)

            base_speed = float(env_info.get("monster_speed", 1.0))
            speed_eps = 0.5
            try:
                pre = getattr(self.agent, "preprocessor", None)
                if pre is not None and hasattr(pre, "_global_cfg"):
                    speed_eps = float(pre._global_cfg("monster_speedup_detect_eps", 0.5))
            except Exception:
                speed_eps = 0.5

            def speedup(monster, active):
                if (not active) or (not isinstance(monster, dict)):
                    return False
                cur_speed = float(monster.get("speed", base_speed))
                return cur_speed >= (base_speed + speed_eps)

            m0_speedup = speedup(m0, m0_active)
            m1_speedup = speedup(m1, m1_active)

            monster_interval = int(env_info.get("monster_interval", -1))
            step_no = int(observation.get("step_no", env_info.get("step_no", 0)))
            m1_should_spawn_by_step = (monster_interval > 0) and (step_no >= monster_interval)
            second_monster_spawned = m1_active or m1_should_spawn_by_step

            if m1_speedup:
                return "second_monster_speedup"
            if m0_speedup:
                return "first_monster_speedup"
            if second_monster_spawned:
                return "second_monster_spawn"
            return "base"
        except Exception:
            return "base"

    def _apply_approach_gravity_stage(self, env_obs):
        stage = self._resolve_approach_gravity_stage(env_obs)
        if hasattr(self.agent, "set_approach_gravity_stage"):
            self.agent.set_approach_gravity_stage(stage)
        return stage

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
            if hasattr(self.agent, "set_episode_mode"):
                self.agent.set_episode_mode(mode)
            self._apply_approach_gravity_stage(env_obs)
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

            self.logger.info(
                f"Episode {self.episode_cnt} start | Mode: {mode} | Map: {selected_map}"
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

                self._apply_approach_gravity_stage(env_obs)
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

                    self.logger.info(
                        f"[GAMEOVER] episode:{self.episode_cnt} mode:{mode} map:{selected_map} "
                        f"steps:{step} result:{result_str} sim_score:{sim_score:.1f} "
                        f"total_reward:{total_reward:.3f} eval_win_rate:{eval_win_rate:.2%}"
                    )

                # 终局奖励并入训练用 reward，确保 GAE / PPO 直接学习到胜负信号。
                shaped_reward = (reward_for_current + final_reward) * Config.REWARD_SCALE

                frame = SampleData(
                    obs=np.array(obs_data.feature, dtype=np.float32),
                    next_obs=np.array(_obs_data.feature, dtype=np.float32),
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
                        }
                        if mode == "train":
                            monitor_data["train_map_id"] = int(selected_map)
                        monitor_data.update(self._get_reward_monitor_data())
                        consistency_total = float(
                            self.reward_components.get("move_mask_consistency_total", 0.0)
                        )
                        consistency_hit = float(
                            self.reward_components.get("move_mask_consistency_hit", 0.0)
                        )
                        monitor_data["move_mask_consistency_rate"] = round(
                            (consistency_hit / consistency_total) if consistency_total > 1e-6 else 0.0,
                            4,
                        )
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
