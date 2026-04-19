#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Feature preprocessor and reward design for Gorge Chase PPO.
峡谷追猎 PPO 特征预处理与奖励设计。

实现的奖励/惩罚项：
1. 生存奖励（稠密）*
2. 宝箱分奖励（稀疏）
3. 加速buff获取奖励（稀疏）
4. 加速buff靠近奖励（稠密）
5. 宝箱接近奖励（平方反比引力，稠密）
6. 怪物距离shaping（稠密）*
7. 后期生存奖励（稠密）*
8. 危险惩罚（稠密）*
9. 撞墙/无效移动惩罚（稀疏）
10. 闪现操作失败惩罚（稀疏）
11. 危险闪现成功奖励（延迟到10%CD窗口结算）
12. 加速期间逃离额外奖励（衰减稠密）
13. 安全区驻留奖励（稠密）
14. 开图奖励（稀疏，出生保护期后生效，系数已下调）
15. 轨迹质心远离奖励（稠密）
16. 原地不动/小范围徘徊惩罚（稠密）
17. 闪现滥用惩罚（10步内被抓/安全区无收益闪现）
18. 死角/死路惩罚（进入后每步惩罚，远离后重置）
19. 坐标访问奖励（首次+0.5，第二次+0.25，之后不再奖励）
"""

import os
import copy
import json
from collections import deque
import numpy as np
from agent_ppo.conf.conf import Config

try:
    import tomllib
except ModuleNotFoundError:
    tomllib = None

# Map size / 地图尺寸（128×128）
MAP_SIZE = 128.0
# Visit tracking map size / 访问计数矩阵尺寸（128×138）
VISIT_TRACK_ROWS = 128
VISIT_TRACK_COLS = 138
# Max monster speed / 最大怪物速度
MAX_MONSTER_SPEED = 5.0
# Max distance bucket / 距离桶最大值
MAX_DIST_BUCKET = 5.0
# Max flash cooldown / 最大闪现冷却步数
MAX_FLASH_CD = 2000.0
# Max buff duration / buff最大持续时间
MAX_BUFF_DURATION = 50.0
# Max buff refresh cooldown / buff刷新冷却归一化上限
MAX_BUFF_REFRESH = 200.0
# Local view size / 局部视野尺寸（21×21）
LOCAL_MAP_VIEW_SIZE = 21
REWARD_CONF_FILE = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "conf", "reward_conf.toml")
)


def _norm(v, v_max, v_min=0.0):
    """Normalize value to [0, 1].

    将值归一化到 [0, 1]。
    """
    v = float(np.clip(v, v_min, v_max))
    return (v - v_min) / (v_max - v_min) if (v_max - v_min) > 1e-6 else 0.0


class Preprocessor:
    def __init__(self, logger=None, monitor=None):
        self.logger = logger
        self.monitor = monitor
        self._debug_dump_obs_fields = os.environ.get("KAIWU_DUMP_OBS_FIELDS", "0") == "1"
        self._obs_field_dumped = False
        self._jsonl_max_lines_default = max(
            1, int(os.environ.get("KAIWU_JSONL_MAX_LINES", "5000"))
        )
        self._jsonl_roll_check_interval = max(
            1, int(os.environ.get("KAIWU_JSONL_ROLL_CHECK_INTERVAL", "1"))
        )
        self._jsonl_write_counter = {}
        self._wall_collision_log_path = os.environ.get(
            "KAIWU_WALL_COLLISION_LOG",
            os.path.normpath(
                os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "..",
                    "log",
                    "wall_collision_cases.jsonl",
                )
            ),
        )
        self._wall_collision_log_max_lines = max(
            1,
            int(
                os.environ.get(
                    "KAIWU_WALL_COLLISION_LOG_MAX_LINES",
                    str(self._jsonl_max_lines_default),
                )
            ),
        )
        self._eval_wall_collision_log_path = os.environ.get(
            "KAIWU_EVAL_WALL_COLLISION_LOG",
            os.path.normpath(
                os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "..",
                    "log",
                    "eval_wall_collision_cases.jsonl",
                )
            ),
        )
        self._eval_wall_collision_log_max_lines = max(
            1,
            int(
                os.environ.get(
                    "KAIWU_EVAL_WALL_COLLISION_LOG_MAX_LINES",
                    str(self._jsonl_max_lines_default),
                )
            ),
        )
        self._action_mask_eval_log_enable = os.environ.get(
            "KAIWU_ACTION_MASK_EVAL_LOG_ENABLE", "1"
        ) == "1"
        self._action_mask_eval_log_in_eval = os.environ.get(
            "KAIWU_ACTION_MASK_EVAL_LOG_IN_EVAL", "0"
        ) == "1"
        self._action_mask_eval_log_anomaly_only = os.environ.get(
            "KAIWU_ACTION_MASK_EVAL_LOG_ANOMALY_ONLY", "0"
        ) == "1"
        self._action_mask_eval_log_path = os.environ.get(
            "KAIWU_ACTION_MASK_EVAL_LOG",
            os.path.normpath(
                os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "..",
                    "log",
                    "action_mask_eval_cases.jsonl",
                )
            ),
        )
        self._action_mask_eval_log_max_lines = max(
            1,
            int(
                os.environ.get(
                    "KAIWU_ACTION_MASK_EVAL_LOG_MAX_LINES",
                    str(self._jsonl_max_lines_default),
                )
            ),
        )
        self._no_movement_log_enable = os.environ.get(
            "KAIWU_NO_MOVEMENT_LOG_ENABLE", "1"
        ) == "1"
        self._no_movement_log_path = os.environ.get(
            "KAIWU_NO_MOVEMENT_LOG",
            os.path.normpath(
                os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "..",
                    "log",
                    "no_movement_cases.jsonl",
                )
            ),
        )
        self._no_movement_log_max_lines = max(
            1,
            int(
                os.environ.get(
                    "KAIWU_NO_MOVEMENT_LOG_MAX_LINES",
                    str(self._jsonl_max_lines_default),
                )
            ),
        )
        self._trim_jsonl_file(
            self._wall_collision_log_path,
            self._wall_collision_log_max_lines,
            force=True,
        )
        self._trim_jsonl_file(
            self._eval_wall_collision_log_path,
            self._eval_wall_collision_log_max_lines,
            force=True,
        )
        self._trim_jsonl_file(
            self._action_mask_eval_log_path,
            self._action_mask_eval_log_max_lines,
            force=True,
        )
        self._trim_jsonl_file(
            self._no_movement_log_path,
            self._no_movement_log_max_lines,
            force=True,
        )
        self.reward_cfg = self._default_reward_config()
        self.reset()

    def _extract_center_patch(self, map_info, patch_size=5):
        """Extract a centered local patch from map_info, out-of-bound filled with -1."""
        if map_info is None:
            return [[-1 for _ in range(patch_size)] for _ in range(patch_size)]

        rows = len(map_info)
        cols = len(map_info[0]) if rows > 0 else 0
        if rows == 0 or cols == 0:
            return [[-1 for _ in range(patch_size)] for _ in range(patch_size)]

        center_r = rows // 2
        center_c = cols // 2
        half = patch_size // 2
        patch = []
        for r in range(center_r - half, center_r + half + 1):
            row_vals = []
            for c in range(center_c - half, center_c + half + 1):
                if 0 <= r < rows and 0 <= c < cols:
                    row_vals.append(int(map_info[r][c]))
                else:
                    row_vals.append(-1)
            patch.append(row_vals)
        return patch

    def _trim_jsonl_file(self, file_path, max_lines, force=False):
        """Trim JSONL file to latest max_lines records."""
        if max_lines is None or int(max_lines) <= 0:
            return

        file_path = os.path.normpath(file_path)
        write_count = int(self._jsonl_write_counter.get(file_path, 0))
        if (not force) and (write_count % self._jsonl_roll_check_interval != 0):
            return

        try:
            if not os.path.isfile(file_path):
                return
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            if len(lines) <= int(max_lines):
                return
            lines = lines[-int(max_lines):]
            with open(file_path, "w", encoding="utf-8") as f:
                f.writelines(lines)
        except Exception:
            # 滚动裁剪失败不影响主流程。
            pass

    def _append_jsonl_record(self, file_path, record, max_lines):
        """Append one JSONL record and keep only latest max_lines entries."""
        file_path = os.path.normpath(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        self._jsonl_write_counter[file_path] = int(self._jsonl_write_counter.get(file_path, 0)) + 1
        self._trim_jsonl_file(file_path, max_lines=max_lines, force=False)

    def _log_wall_collision_case(
        self,
        hero_pos,
        last_action,
        legal_action,
        map_info,
        step_displacement,
        is_speed_phase,
    ):
        """Append one wall-collision diagnostic sample when the penalty branch fires."""
        try:
            record = {
                "step_no": int(self.step_no),
                "hero_pos": {"x": int(hero_pos["x"]), "z": int(hero_pos["z"])},
                "last_action": int(last_action),
                "legal_action": [int(x) for x in (legal_action or [])],
                "move_legal_action_0_7": [int(x) for x in (legal_action or [])[:8]],
                "step_displacement": float(step_displacement) if step_displacement is not None else None,
                "is_speed_phase": bool(is_speed_phase),
                "map_center_5x5": self._extract_center_patch(map_info, patch_size=5),
            }
            self._append_jsonl_record(
                self._wall_collision_log_path,
                record,
                self._wall_collision_log_max_lines,
            )
        except Exception:
            # 日志写入失败不影响训练流程
            pass

    def _log_eval_wall_collision_case(
        self,
        hero_pos,
        last_action,
        legal_action,
        map_info,
        step_displacement,
        is_speed_phase,
    ):
        """Append one eval-stage wall-collision diagnostic sample to JSONL file."""
        try:
            record = {
                "step_no": int(self.step_no),
                "episode_mode": "eval",
                "hero_pos": {"x": int(hero_pos["x"]), "z": int(hero_pos["z"] )},
                "last_action": int(last_action),
                "legal_action": [int(x) for x in (legal_action or [])],
                "move_legal_action_0_7": [int(x) for x in (legal_action or [])[:8]],
                "step_displacement": float(step_displacement) if step_displacement is not None else None,
                "is_speed_phase": bool(is_speed_phase),
                "map_center_5x5": self._extract_center_patch(map_info, patch_size=5),
            }
            self._append_jsonl_record(
                self._eval_wall_collision_log_path,
                record,
                self._eval_wall_collision_log_max_lines,
            )
        except Exception:
            # 日志写入失败不影响训练流程
            pass

    def set_episode_mode(self, mode):
        """Set current episode mode for diagnostics routing."""
        mode_str = str(mode).strip().lower() if mode is not None else "train"
        self.episode_mode = "eval" if mode_str == "eval" else "train"

    def _log_action_mask_eval_case(self, record, anomaly=False):
        """Append one action-mask effectiveness record to JSONL file."""
        if (not self._action_mask_eval_log_enable) or (
            self._action_mask_eval_log_anomaly_only and (not anomaly)
        ):
            return
        if (self.episode_mode == "eval") and (not self._action_mask_eval_log_in_eval):
            return

        try:
            self._append_jsonl_record(
                self._action_mask_eval_log_path,
                record,
                self._action_mask_eval_log_max_lines,
            )
        except Exception:
            # 日志写入失败不影响训练流程
            pass

    def _log_no_movement_case(self, record):
        """Append one no-movement case record to dedicated JSONL file."""
        if not self._no_movement_log_enable:
            return

        try:
            self._append_jsonl_record(
                self._no_movement_log_path,
                record,
                self._no_movement_log_max_lines,
            )
        except Exception:
            # 日志写入失败不影响训练流程
            pass

        if self.logger is not None:
            try:
                self.logger.info(f"[NO_MOVE] {json.dumps(record, ensure_ascii=False)}")
            except Exception:
                pass

    def _log_action_mask_effect_case(
        self,
        last_action,
        step_displacement,
        wall_collision_triggered,
        flash_fail_triggered,
        is_speed_phase_exec,
        map_info,
    ):
        """Evaluate previous-step decision mask against execution outcome and record it."""
        if self.last_decision_legal_action is None:
            return {
                "no_movement_case": 0.0,
                "move_mask_consistency_hit": 0.0,
                "move_mask_consistency_total": 0.0,
            }

        action_idx = int(last_action)
        decision_legal = None
        if 0 <= action_idx < len(self.last_decision_legal_action):
            decision_legal = int(self.last_decision_legal_action[action_idx])

        action_type = "other"
        move_dir_idx = None
        if 0 <= action_idx <= 7:
            action_type = "move"
            move_dir_idx = action_idx
        elif 8 <= action_idx <= 15:
            action_type = "flash"
            move_dir_idx = action_idx - 8

        move_dir_legal = None
        if (
            move_dir_idx is not None
            and self.last_decision_move_mask is not None
            and 0 <= move_dir_idx < len(self.last_decision_move_mask)
        ):
            move_dir_legal = int(self.last_decision_move_mask[move_dir_idx])

        move_dir_legal_corrected = None
        if (
            move_dir_idx is not None
            and self.last_decision_move_mask_corrected is not None
            and 0 <= move_dir_idx < len(self.last_decision_move_mask_corrected)
        ):
            move_dir_legal_corrected = int(self.last_decision_move_mask_corrected[move_dir_idx])

        move_mask_consistent = None
        move_mask_consistency_hit = 0.0
        move_mask_consistency_total = 0.0
        if action_type == "move":
            move_mask_consistency_total = 1.0
            move_mask_consistent = bool(move_dir_legal_corrected == 1)
            move_mask_consistency_hit = 1.0 if move_mask_consistent else 0.0

        wall_thr = float(self._global_cfg("wall_displacement_threshold", 0.5))
        is_effective_move = None
        if action_type == "move" and step_displacement is not None:
            is_effective_move = bool(step_displacement >= wall_thr)

        anomaly = False
        if decision_legal == 0:
            anomaly = True
        if action_type == "move" and decision_legal == 1 and wall_collision_triggered:
            anomaly = True
        if action_type == "flash" and decision_legal == 1 and flash_fail_triggered:
            anomaly = True

        record = {
            "exec_step_no": int(self.step_no),
            "decision_step_no": int(self.last_decision_step_no)
            if self.last_decision_step_no is not None
            else None,
            "last_action": action_idx,
            "action_type": action_type,
            "step_displacement": float(step_displacement) if step_displacement is not None else None,
            "wall_displacement_threshold": wall_thr,
            "is_effective_move": is_effective_move,
            "decision_action_legal": decision_legal,
            "decision_move_dir_legal": move_dir_legal,
            "decision_move_dir_legal_corrected": move_dir_legal_corrected,
            "move_mask_consistent": move_mask_consistent,
            "wall_collision_penalty_applied": bool(wall_collision_triggered),
            "flash_fail_penalty_applied": bool(flash_fail_triggered),
            "is_speed_phase_exec": bool(is_speed_phase_exec),
            "is_speed_phase_decision": bool(self.last_decision_is_speed_phase)
            if self.last_decision_is_speed_phase is not None
            else None,
            "decision_legal_action": [
                int(x) for x in (self.last_decision_legal_action or [])
            ],
            "decision_move_mask_0_7": [
                int(x) for x in (self.last_decision_move_mask or [])
            ],
            "decision_move_mask_corrected_0_7": [
                int(x) for x in (self.last_decision_move_mask_corrected or [])
            ],
            "decision_map_center_5x5": self.last_decision_map_center_5x5,
            "current_map_center_5x5": self._extract_center_patch(map_info, patch_size=5),
            "anomaly": bool(anomaly),
        }
        self._log_action_mask_eval_case(record, anomaly=anomaly)

        if action_type == "move" and is_effective_move is False:
            no_move_record = dict(record)
            no_move_record["episode_mode"] = str(self.episode_mode)
            self._log_no_movement_case(no_move_record)
            return {
                "no_movement_case": 1.0,
                "move_mask_consistency_hit": move_mask_consistency_hit,
                "move_mask_consistency_total": move_mask_consistency_total,
            }

        return {
            "no_movement_case": 0.0,
            "move_mask_consistency_hit": move_mask_consistency_hit,
            "move_mask_consistency_total": move_mask_consistency_total,
        }

    def _default_reward_config(self):
        """Default reward config used when external config is missing.

        默认奖励配置：当外部配置文件不存在或解析失败时使用。
        """
        return {
            "global": {
                "birth_protection_steps": 10,
                "trajectory_window": 20,
                "safe_treasure_monster_dist": 0.2,
                "danger_threshold": 0.15,
                "late_danger_threshold": 0.25,
                "wall_displacement_threshold": 0.5,
                "normal_move_step": 1,
                "buff_move_step": 2,
                "idle_displacement_threshold": 0.25,
                "wander_radius_threshold": 2.5,
                "wander_min_points": 8,
                "idle_wander_reset_distance": 6.0,
                "dead_end_check_radius": 8,
                "dead_end_opening_clusters_threshold": 1,
                "dead_end_max_reachable_cells": 90,
                "dead_end_max_reachable_ratio": 0.45,
                "dead_end_reset_distance": 8.0,
                "flash_fail_ratio": 0.3,
                "flash_expected_dist_orthogonal": 10.0,
                "flash_expected_dist_diagonal": 8.0,
                "centroid_norm_max": MAP_SIZE * 0.5,
            },
            "survive_reward": {"enable": True, "coef": 0.1},
            "treasure_reward": {"enable": True, "coef": 1.0},
            "speed_buff_reward": {"enable": True, "coef": 0.2},
            "speed_buff_approach_reward": {"enable": True, "coef": 0.2},
            "treasure_approach_reward": {
                "enable": True,
                "gravity_coef": 0.0015,
                "min_reward": 0.001,
                "max_reward": 0.1,
                "min_dist_norm": 0.05,
            },
            "monster_dist_shaping": {"enable": True, "coef": 1.0},
            "late_survive_reward": {"enable": True, "base": 0.02, "shaping_coef": 0.1},
            "danger_penalty": {"enable": True, "coef": -0.1, "power": 2.0},
            "wall_collision_penalty": {"enable": True, "coef": -0.1},
            "flash_fail_penalty": {"enable": True, "coef": -0.15},
            "flash_escape_reward": {"enable": True, "base": 1.0, "dist_gain_coef": 1.0},
            "flash_survival_reward": {"enable": True, "init": 0.5, "decay": 0.95},
            "speed_buff_escape_reward": {
                "enable": True,
                "init": 0.05,
                "decay": 0.97,
                "base": 1.0,
                "dist_delta_coef": 0.1,
            },
            "flash_abuse_penalty": {
                "enable": True,
                "caught_within_steps": 10,
                "caught_coef": -1.0,
                "safe_zone_no_treasure_coef": -1.0,
            },
            "safe_zone_reward": {"enable": True, "coef": 0.01},
            "exploration_reward": {"enable": True, "coef_per_cell": 0.0002},
            "visit_tracking_reward": {
                "enable": True,
                "first_visit": 0.5,
                "second_visit": 0.25,
            },
            "centroid_away_reward": {"enable": True, "coef": 0.005},
            "idle_wander_penalty": {
                "enable": True,
                "idle_coef": -0.15,
                "wander_coef": -0.25,
                "idle_growth": 0.12,
                "wander_growth": 0.08,
            },
            "dead_end_penalty": {
                "enable": True,
                "coef": -0.5,
            },
        }

    def _deep_update(self, base, updates):
        """Recursively update nested dict values."""
        for key, val in updates.items():
            if isinstance(val, dict) and isinstance(base.get(key), dict):
                self._deep_update(base[key], val)
            else:
                base[key] = val

    def _load_reward_config(self):
        """Load reward config from TOML file and merge with defaults."""
        cfg = copy.deepcopy(self._default_reward_config())
        if tomllib is None:
            self.reward_cfg = cfg
            return

        if not os.path.isfile(REWARD_CONF_FILE):
            self.reward_cfg = cfg
            return

        try:
            with open(REWARD_CONF_FILE, "rb") as f:
                parsed = tomllib.load(f)
            reward_section = parsed.get("reward", parsed)
            if isinstance(reward_section, dict):
                self._deep_update(cfg, reward_section)
        except Exception:
            # 配置异常时回退默认值，确保训练不中断。
            pass

        self.reward_cfg = cfg

    def _cfg(self, section, key, default):
        """Get config value by section/key with fallback."""
        sec = self.reward_cfg.get(section, {})
        if not isinstance(sec, dict):
            return default
        return sec.get(key, default)

    def _global_cfg(self, key, default):
        g = self.reward_cfg.get("global", {})
        if not isinstance(g, dict):
            return default
        return g.get(key, default)

    def reset(self):
        """Reset per-episode state for reward computation.

        每局开始时重置奖励计算所需的状态变量。
        """
        self._load_reward_config()

        self.step_no = 0
        self.max_step = 200

        # ========== 历史状态记录（用于帧间比较）==========
        # 上一帧最近怪物归一化距离
        self.last_min_monster_dist_norm = 0.5
        # 上一帧宝箱得分
        self.last_treasure_score = 0.0
        # 上一帧已收集宝箱数量
        self.last_treasure_collected_count = 0
        # 上一帧已收集buff数量
        self.last_collected_buff = 0
        # 上一帧到最近宝箱的归一化距离
        self.last_min_treasure_dist_norm = 1.0
        # 上一帧到最近加速buff的归一化距离
        self.last_min_speed_buff_dist_norm = 1.0
        # 宝箱/加速buff势能基线（用于势能差分 shaping）
        self.last_treasure_potential = 0.0
        self.last_speed_buff_potential = 0.0
        self.last_treasure_potential_active = False
        self.last_speed_buff_potential_active = False
        # 上一帧英雄位置（用于检测撞墙/无效移动）
        self.last_hero_pos = None
        # 上一帧最近怪物归一化距离（用于闪现判断）
        self.last_min_monster_dist_before_flash = 0.5
        # 上一帧是否使用了闪现
        self.last_used_flash = False

        # ========== 衰减奖励状态变量 ==========
        # 闪现成功评估状态（危险时闪现后，在10%CD窗口结算）
        self.flash_escape_active = False
        self.flash_escape_steps = 0
        self.flash_escape_window_steps = 0
        self.flash_escape_pre_dist = 0.0
        self.flash_success_blocked = False
        self.flash_survival_decay = 0.0
        # 从最近一次闪现起累计的步数（每次闪现重置，用于限制10%-30%CD发奖窗口）
        self.flash_since_use_steps = None
        self.flash_survival_window_end_steps = 0

        # 闪现滥用检测状态（用于10步内被抓惩罚）
        self.flash_recent_steps = None

        # 加速期间逃离奖励当前值（衰减稠密，获取buff时重置为初始值）
        self.speed_buff_escape_decay = 0.0
        # 上一帧是否持有加速buff（用于检测buff获取时刻）
        self.last_had_speed_buff = False

        # ========== 闪现失败检测状态 ==========
        # 上一帧使用的闪现动作ID（用于计算期望闪现距离）
        self.last_flash_action = -1

        # ========== 最近怪物追踪状态 ==========
        # 上一帧最近怪物的序号（0或1），-1 表示无怪物
        self.last_nearest_monster_idx = -1

        # ========== 合法动作掩码评估状态 ==========
        # 上一决策帧的合法动作信息（用于评估掩码是否与执行结果一致）
        self.last_decision_step_no = None
        self.last_decision_legal_action = None
        self.last_decision_move_mask = None
        self.last_decision_move_mask_corrected = None
        self.last_decision_is_speed_phase = None
        self.last_decision_map_center_5x5 = None
        # 当前局模式（train/eval），用于将评估期撞墙写入独立日志。
        self.episode_mode = "train"

        # ========== 地图记忆 / 开图奖励状态 ==========
        # 全局探索记忆图（128x128），记录英雄局部视野扫过的区域
        self.explored_map = np.zeros((int(MAP_SIZE), int(MAP_SIZE)), dtype=np.float32)
        # 上一帧已探索格子总数（用于计算新增开图数）
        self.last_explored_count = 0
        # 英雄坐标访问计数矩阵（128x138），用于首访/二访奖励。
        self.visit_count_map = np.zeros((VISIT_TRACK_ROWS, VISIT_TRACK_COLS), dtype=np.uint8)

        # ========== 出生保护 / 开图奖励 ==========
        # 出生后前N步不计入开图探索奖励（避免出生位置视野带来的虚假奖励）
        self.BIRTH_PROTECTION_STEPS = int(self._global_cfg("birth_protection_steps", 10))
        # 当前已过保护期步数计数（达到BIRTH_PROTECTION_STEPS后才开始计奖）
        self.birth_step_counter = 0

        # ========== 轨迹质心 / 远离奖励 ==========
        # 记录英雄最近N步的坐标轨迹，用于计算质心
        self.TRAJECTORY_WINDOW = int(self._global_cfg("trajectory_window", 20))  # 滑动窗口大小
        self.trajectory_buffer = []  # [(x, z), ...] 坐标列表

        # ========== 原地/徘徊惩罚状态 ==========
        # 连续原地步数（步数越长惩罚越重）
        self.idle_streak_steps = 0
        # 连续徘徊步数（持续越久惩罚越重）
        self.wander_streak_steps = 0
        # 原地/徘徊区域锚点：离开该区域超过阈值后重置惩罚计数
        self.idle_wander_anchor_pos = None

        # ========== 死角/死路惩罚状态 ==========
        # 进入死角后持续惩罚，直到远离锚点阈值
        self.dead_end_active = False
        self.dead_end_anchor_pos = None

        # 存储最新的reward_info供agent获取
        self.last_reward_info = {}

    def _is_half_surrounded_dead_end(self, map_info):
        """Detect dead-end by ray probing + boundary tracing + compactness.

        死角判定流程：
        1) 以中心向16方向发射网格射线，抵达最大半径且落点可通行则记为开口候选；
        2) 对候选点沿障碍/越界交界线做双侧爬行，若短程内双侧交汇或闭环则判为封闭；
        3) 对真实开口端点做空间聚类，统计独立开口簇数；
        4) 结合局部BFS可达区域紧凑度，开口极少且可达受限时判定死胡同。
        """
        if map_info is None:
            return False

        rows = len(map_info)
        cols = len(map_info[0]) if rows > 0 else 0
        if rows == 0 or cols == 0:
            return False

        cr = rows // 2
        cc = cols // 2
        if map_info[cr][cc] == 0:
            return False

        max_radius = int(self._global_cfg("dead_end_check_radius", 8))
        max_radius = max(1, min(max_radius, cr, cc, rows - 1 - cr, cols - 1 - cc))

        # 约为圆周的1/8，默认 ~ pi*R/4
        trace_steps = int(self._global_cfg("dead_end_boundary_trace_steps", int(round(np.pi * max_radius / 4.0))))
        trace_steps = max(4, trace_steps)

        opening_clusters_threshold = int(self._global_cfg("dead_end_opening_clusters_threshold", 1))
        max_reachable_cells = int(self._global_cfg("dead_end_max_reachable_cells", 90))
        max_reachable_ratio = float(self._global_cfg("dead_end_max_reachable_ratio", 0.45))

        def in_bounds(r, c):
            return 0 <= r < rows and 0 <= c < cols

        def passable(r, c):
            return in_bounds(r, c) and map_info[r][c] != 0

        def is_interface_cell(r, c):
            if not passable(r, c):
                return False
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (not in_bounds(nr, nc)) or map_info[nr][nc] == 0:
                    return True
            return False

        def interface_neighbors(r, c):
            out = []
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if is_interface_cell(nr, nc):
                        out.append((nr, nc))
            return out

        def nearest_interface_seed(seed_r, seed_c):
            if is_interface_cell(seed_r, seed_c):
                return (seed_r, seed_c)
            for dist in range(1, 3):
                for dr in range(-dist, dist + 1):
                    for dc in range(-dist, dist + 1):
                        rr, cc2 = seed_r + dr, seed_c + dc
                        if is_interface_cell(rr, cc2):
                            return (rr, cc2)
            return None

        def crawl_one_side(start, tangent, side_sign):
            if start is None:
                return [None]
            path = [start]
            visited_local = {start}
            for _ in range(trace_steps):
                r, c = path[-1]
                cands = []
                for nr, nc in interface_neighbors(r, c):
                    if (nr, nc) in visited_local:
                        continue
                    # 沿切向优先，限制在探测半径附近
                    vr, vc = nr - r, nc - c
                    tangential = side_sign * (vr * tangent[0] + vc * tangent[1])
                    radial = abs(max(abs(nr - cr), abs(nc - cc)) - max_radius)
                    if tangential >= -1e-6:
                        cands.append((tangential, -radial, nr, nc))
                if not cands:
                    break
                cands.sort(reverse=True)
                _, _, best_r, best_c = cands[0]
                nxt = (best_r, best_c)
                path.append(nxt)
                visited_local.add(nxt)
                if len(path) > 3 and max(abs(nxt[0] - start[0]), abs(nxt[1] - start[1])) <= 1:
                    break
            return path

        def opening_is_closed(endpoint, ray_dr, ray_dc):
            seed = nearest_interface_seed(endpoint[0], endpoint[1])
            if seed is None:
                return False

            # 切向方向（与射线垂直）
            tangent = np.array([-ray_dc, ray_dr], dtype=np.float32)
            norm = float(np.linalg.norm(tangent))
            if norm < 1e-6:
                tangent = np.array([0.0, 1.0], dtype=np.float32)
            else:
                tangent /= norm

            left_path = crawl_one_side(seed, tangent, side_sign=1.0)
            right_path = crawl_one_side(seed, tangent, side_sign=-1.0)

            left_end = left_path[-1] if left_path and left_path[-1] is not None else None
            right_end = right_path[-1] if right_path and right_path[-1] is not None else None
            if left_end is None or right_end is None:
                return False

            # 双侧交汇（或首尾闭合）视为封闭。
            if max(abs(left_end[0] - right_end[0]), abs(left_end[1] - right_end[1])) <= 1:
                return True
            if max(abs(left_end[0] - seed[0]), abs(left_end[1] - seed[1])) <= 1 and len(left_path) > 3:
                return True
            if max(abs(right_end[0] - seed[0]), abs(right_end[1] - seed[1])) <= 1 and len(right_path) > 3:
                return True
            return False

        # ===== 1) 射线探测候选开口 =====
        ray_dirs = [
            (-1, 0), (-1, 1), (0, 1), (1, 1),
            (1, 0), (1, -1), (0, -1), (-1, -1),
            (-2, 1), (-1, 2), (1, 2), (2, 1),
            (2, -1), (1, -2), (-1, -2), (-2, -1),
        ]
        opening_endpoints = []
        for dr, dc in ray_dirs:
            scale = float(max(abs(dr), abs(dc)))
            reached = True
            end_cell = None
            for step in range(1, max_radius + 1):
                rr = cr + int(round(dr * step / scale))
                cc2 = cc + int(round(dc * step / scale))
                if not passable(rr, cc2):
                    reached = False
                    break
                end_cell = (rr, cc2)
            if reached and end_cell is not None:
                # ===== 2) 边界追踪剔除伪开口 =====
                if not opening_is_closed(end_cell, dr, dc):
                    opening_endpoints.append(end_cell)

        # ===== 3) 开口端点聚类 =====
        opening_clusters = 0
        remaining = set(opening_endpoints)
        cluster_dist = max(1, int(round(max_radius * 0.4)))
        while remaining:
            opening_clusters += 1
            seed = remaining.pop()
            stack = [seed]
            while stack:
                r, c = stack.pop()
                to_remove = []
                for rr, cc2 in remaining:
                    if max(abs(rr - r), abs(cc2 - c)) <= cluster_dist:
                        to_remove.append((rr, cc2))
                for item in to_remove:
                    remaining.remove(item)
                    stack.append(item)

        # ===== 4) 局部BFS紧凑度 =====
        dirs4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        visited = set()
        q = deque()
        q.append((cr, cc))
        visited.add((cr, cc))

        while q:
            r, c = q.popleft()
            for dr, dc in dirs4:
                nr, nc = r + dr, c + dc
                if not passable(nr, nc):
                    continue

                bubble_dist = max(abs(nr - cr), abs(nc - cc))
                if bubble_dist <= max_radius:
                    if (nr, nc) not in visited:
                        visited.add((nr, nc))
                        q.append((nr, nc))

        reachable_cells = len(visited)
        bubble_area = float((2 * max_radius + 1) * (2 * max_radius + 1))
        reachable_ratio = reachable_cells / bubble_area if bubble_area > 1e-6 else 1.0

        # 角落通常可达区域较大，不应判为死胡同；窄死路通常“开口少+可达小”。
        is_compact = (reachable_cells <= max_reachable_cells) or (reachable_ratio <= max_reachable_ratio)
        return opening_clusters <= opening_clusters_threshold and is_compact

    def _compute_nearest_organ_distance(self, organs, hero_pos, target_sub_type):
        """Compute nearest target organ distance and position."""
        min_dist = float(MAP_SIZE * 1.41)
        nearest_pos = None

        for organ in organs:
            if organ.get("sub_type") == target_sub_type and organ.get("status") == 1:
                o_pos = organ["pos"]
                dist = np.sqrt((hero_pos["x"] - o_pos["x"]) ** 2 + (hero_pos["z"] - o_pos["z"]) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    nearest_pos = o_pos

        if nearest_pos is None:
            return 1.0, None, False

        return _norm(min_dist, MAP_SIZE * 1.41), nearest_pos, True

    def _filter_visible_organs(self, organs, hero_pos, target_sub_type=None):
        """Filter visible organs from frame_state.organs by type.

        协议约定 frame_state.organs 已是局部视野可观测到的宝箱/buff对象，
        这里不再做二次可见性判定。
        """
        visible = []
        for organ in organs:
            if target_sub_type is not None and organ.get("sub_type") != target_sub_type:
                continue
            if organ.get("pos", None) is None:
                continue
            visible.append(organ)
        return visible

    def _compute_potential_from_distance(self, dist_norm, gravity_coef, min_dist_norm, min_reward, max_reward):
        """Compute bounded potential from normalized distance.

        将距离映射为有界势能，距离越小势能越高。
        """
        dist_for_force = max(float(min_dist_norm), float(dist_norm))
        potential = float(gravity_coef) / (dist_for_force * dist_for_force)
        return float(np.clip(potential, float(min_reward), float(max_reward)))

    def _compute_nearest_treasure_feature(self, organs, hero_pos):
        """Get nearest treasure feature as [x_norm, z_norm, dist_norm]."""
        visible_treasures = self._filter_visible_organs(organs, hero_pos, target_sub_type=1)
        dist_norm, nearest_pos, found = self._compute_nearest_organ_distance(
            visible_treasures, hero_pos, 1
        )
        if not found:
            return np.array([0.0, 0.0, 1.0], dtype=np.float32), 1.0
        return np.array(
            [_norm(nearest_pos["x"], MAP_SIZE), _norm(nearest_pos["z"], MAP_SIZE), dist_norm],
            dtype=np.float32,
        ), dist_norm

    def _get_treasure_collected_count(self, env_info, hero):
        """Read treasure collected count from multiple possible protocol keys."""
        keys = [
            env_info.get("treasures_collected", None),
            env_info.get("treasure_collected_count", None),
            env_info.get("treasure_count", None),
            hero.get("treasure_collected_count", None),
            hero.get("treasures_collected", None),
            hero.get("treasure_count", None),
        ]
        for v in keys:
            if v is None:
                continue
            try:
                return int(v)
            except Exception:
                continue
        return 0

    def _find_target_touch_steps(self, hero_pos, organs, move_step, target_sub_type):
        """Find earliest touch step for each move action [0..7], 0 means not reachable in one action."""
        target_cells = set()
        for organ in organs:
            if organ.get("sub_type") != target_sub_type or int(organ.get("status", 0)) != 1:
                continue
            pos = organ.get("pos", {})
            try:
                px = int(pos.get("x", -1))
                pz = int(pos.get("z", -1))
            except Exception:
                continue
            target_cells.add((px, pz))

        if not target_cells:
            return [0] * 8

        hx = int(hero_pos["x"])
        hz = int(hero_pos["z"])
        # 0..7: E, NE, N, NW, W, SW, S, SE
        action_delta = [
            (1, 0),
            (1, -1),
            (0, -1),
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
        ]

        touch_steps = [0] * 8
        max_step = max(1, int(move_step))
        for act, (dx, dz) in enumerate(action_delta):
            for step in range(1, max_step + 1):
                if (hx + dx * step, hz + dz * step) in target_cells:
                    touch_steps[act] = step
                    break

        return touch_steps

    def _preprocess_move_action_mask(self, map_info, has_speed_buff):
        """Preprocess legality for movement actions [0..7] using map neighborhood.

        普通速度按 1 格检测；加速状态按 2 格检测（逐格路径检查）。
        """
        if map_info is None:
            return [1] * 8

        rows = len(map_info)
        cols = len(map_info[0]) if rows > 0 else 0
        if rows == 0 or cols == 0:
            return [1] * 8

        center_r = rows // 2
        center_c = cols // 2

        move_step = int(self._global_cfg("buff_move_step", 2)) if has_speed_buff else int(
            self._global_cfg("normal_move_step", 1)
        )
        move_step = max(1, move_step)

        # 0..7: E, NE, N, NW, W, SW, S, SE
        dirs8 = [
            (0, 1),
            (-1, 1),
            (-1, 0),
            (-1, -1),
            (0, -1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

        def is_passable(r, c):
            return 0 <= r < rows and 0 <= c < cols and map_info[r][c] != 0

        mask = [1] * 8
        for i, (dr, dc) in enumerate(dirs8):
            legal = True
            for step in range(1, move_step + 1):
                rr = center_r + dr * step
                cc = center_c + dc * step
                if not is_passable(rr, cc):
                    legal = False
                    break
                # 切角移动不合法（可配置）：斜向移动时，必须同时满足横纵两个邻格都可通行。
                if (not bool(getattr(Config, "ALLOW_CORNER_CUT_MOVE", False))) and dr != 0 and dc != 0:
                    prev_r = center_r + dr * (step - 1)
                    prev_c = center_c + dc * (step - 1)
                    if not (is_passable(rr, prev_c) and is_passable(prev_r, cc)):
                        legal = False
                        break
            mask[i] = 1 if legal else 0

        return mask

    def _compute_nearest_speed_buff_feature(self, organs, hero_pos, env_info):
        """Get nearest speed buff feature as [cd_norm, x_norm, z_norm, dist_norm]."""
        visible_buffs = self._filter_visible_organs(organs, hero_pos, target_sub_type=2)
        dist_norm, nearest_pos, found = self._compute_nearest_organ_distance(
            visible_buffs, hero_pos, 2
        )

        # 若场上存在可拾取buff，则冷却视为0；否则用配置冷却值归一化。
        if found:
            cd_norm = 0.0
            return np.array(
                [cd_norm, _norm(nearest_pos["x"], MAP_SIZE), _norm(nearest_pos["z"], MAP_SIZE), dist_norm],
                dtype=np.float32,
            )

        cd_norm = _norm(float(env_info.get("buff_refresh_time", 0.0)), MAX_BUFF_REFRESH)
        return np.array([cd_norm, 0.0, 0.0, 1.0], dtype=np.float32)

    def _compute_direction_one_hot_from_pos(self, target_pos, hero_pos):
        """Encode relative direction from hero to a target position as 9D one-hot."""
        dx = float(target_pos["x"]) - float(hero_pos["x"])
        dz = float(target_pos["z"]) - float(hero_pos["z"])
        if abs(dx) < 1e-6 and abs(dz) < 1e-6:
            direction = 0
        else:
            horizontal = 1 if dx > 0 else (-1 if dx < 0 else 0)
            vertical = 1 if dz > 0 else (-1 if dz < 0 else 0)
            direction_map = {
                (1, 0): 1,
                (1, 1): 2,
                (0, 1): 3,
                (-1, 1): 4,
                (-1, 0): 5,
                (-1, -1): 6,
                (0, -1): 7,
                (1, -1): 8,
            }
            direction = direction_map.get((horizontal, vertical), 0)

        one_hot = np.zeros(9, dtype=np.float32)
        one_hot[int(direction)] = 1.0
        return one_hot

    def _collect_organ_slots(self, organs, hero_pos, target_sub_type, slot_num):
        """Collect fixed-count organ slots sorted by distance, pad missing slots with 0.

        Slot layout: available_flag, x, z, dist, dir_onehot(9).
        """
        slots = []
        for organ in organs:
            if organ.get("sub_type") != target_sub_type:
                continue
            pos = organ.get("pos", None)
            if pos is None:
                continue
            dist = np.sqrt((hero_pos["x"] - pos["x"]) ** 2 + (hero_pos["z"] - pos["z"]) ** 2)
            dist_norm = _norm(dist, MAP_SIZE * 1.41)
            dir_feat = self._compute_direction_one_hot_from_pos(pos, hero_pos)
            available_flag = 1.0 if int(organ.get("status", 0)) == 1 else 0.0
            feat = np.concatenate(
                [
                    np.array([available_flag], dtype=np.float32),
                    np.array([
                        _norm(pos["x"], MAP_SIZE),
                        _norm(pos["z"], MAP_SIZE),
                        dist_norm,
                    ], dtype=np.float32),
                    dir_feat,
                ]
            )

            slots.append((dist, feat))

        slots.sort(key=lambda x: x[0])
        selected = [f for _, f in slots[:slot_num]]

        while len(selected) < slot_num:
            selected.append(np.zeros(13, dtype=np.float32))

        return selected

    def _compute_safe_zone_quadrant(self, monsters):
        """Compute the safest quadrant by the farthest corner point.

        根据当前怪物位置，选出离怪物最远的地图角点，并返回该角点所在象限。

        Returns:
            tuple[int, tuple[float, float]] | tuple[None, None]:
                safe_quadrant_id 和 safest_corner；无活跃怪物时返回 (None, None)
        """
        active_monsters = []
        for m in monsters:
            pos = m.get("pos", {"x": 0, "z": 0})
            if pos["x"] == 0 and pos["z"] == 0:
                continue
            active_monsters.append((float(pos["x"]), float(pos["z"])))

        if not active_monsters:
            return None, None

        ms = float(MAP_SIZE)
        half = ms / 2.0
        corners = [
            (0.0, 0.0),
            (ms, 0.0),
            (0.0, ms),
            (ms, ms),
        ]

        best_corner = None
        best_score = -1.0
        for cx, cz in corners:
            # 使用“到最近怪物的距离”作为安全度，确保选到真正远离怪物的角点。
            score = min(np.sqrt((mx - cx) ** 2 + (mz - cz) ** 2) for mx, mz in active_monsters)
            if score > best_score:
                best_score = score
                best_corner = (cx, cz)

        corner_x, corner_z = best_corner
        if corner_x < half and corner_z < half:
            return 0, best_corner
        if corner_x >= half and corner_z < half:
            return 1, best_corner
        if corner_x < half and corner_z >= half:
            return 2, best_corner
        return 3, best_corner

    def _is_in_safe_quadrant(self, hero_pos, safe_quadrant_id):
        """Check whether hero is inside the chosen safe quadrant.

        判断英雄是否位于选中的安全象限内。
        """
        if safe_quadrant_id is None:
            return False

        half = MAP_SIZE / 2.0
        x = float(hero_pos["x"])
        z = float(hero_pos["z"])

        if safe_quadrant_id == 0:
            return x < half and z < half
        if safe_quadrant_id == 1:
            return x >= half and z < half
        if safe_quadrant_id == 2:
            return x < half and z >= half
        return x >= half and z >= half

    def _compute_nearest_monster_idx(self, monsters, hero_pos=None):
        """Find index of nearest active monster to hero.

        找到距离英雄最近的活跃怪物序号。
        未出生的怪物（位置为原点或不存在）不计入，避免训练震荡。

        Args:
            monsters: 怪物状态列表
            hero_pos: 英雄位置 {"x": int, "z": int}，用于精确距离计算

        Returns:
            int: 最近怪物序号(0或1)，无活跃怪物返回-1
        """
        min_dist = float("inf")
        nearest_idx = -1
        for i, m in enumerate(monsters):
            pos = m.get("pos", {"x": 0, "z": 0})
            # 排除未出生怪物：位置为(0,0)视为未生成
            if pos["x"] == 0 and pos["z"] == 0:
                continue
            if hero_pos is not None:
                # 基于英雄位置计算欧式距离（更准确）
                dist = np.sqrt((hero_pos["x"] - pos["x"]) ** 2 + (hero_pos["z"] - pos["z"]) ** 2)
            else:
                # 无hero_pos时降级使用距原点近似
                dist = np.sqrt(pos["x"] ** 2 + pos["z"] ** 2)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
        return nearest_idx

    def _compute_relative_direction_one_hot(self, target, hero_pos):
        """Encode relative direction as a 9D one-hot vector.

        将相对方位编码为 9 维 one-hot：0=重叠，1=东，2=东北，3=北，4=西北，5=西，6=西南，7=南，8=东南。
        优先使用协议里的 hero_relative_direction；若缺失则基于坐标差计算。
        """
        direction = target.get("hero_relative_direction", None)
        if direction is None:
            pos = target.get("pos", None)
            if pos is None:
                direction = 0
            else:
                dx = float(pos["x"]) - float(hero_pos["x"])
                dz = float(pos["z"]) - float(hero_pos["z"])
                if abs(dx) < 1e-6 and abs(dz) < 1e-6:
                    direction = 0
                else:
                    horizontal = 1 if dx > 0 else (-1 if dx < 0 else 0)
                    vertical = 1 if dz > 0 else (-1 if dz < 0 else 0)
                    direction_map = {
                        (1, 0): 1,
                        (1, 1): 2,
                        (0, 1): 3,
                        (-1, 1): 4,
                        (-1, 0): 5,
                        (-1, -1): 6,
                        (0, -1): 7,
                        (1, -1): 8,
                    }
                    direction = direction_map.get((horizontal, vertical), 0)

        direction = int(direction)
        direction = 0 if direction < 0 or direction > 8 else direction
        one_hot = np.zeros(9, dtype=np.float32)
        one_hot[direction] = 1.0
        return one_hot

    def _collect_key_paths(self, obj, prefix=""):
        """Collect all nested key paths from dict/list structure."""
        paths = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                child = f"{prefix}.{k}" if prefix else str(k)
                paths.append(child)
                paths.extend(self._collect_key_paths(v, child))
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                child = f"{prefix}[{i}]"
                paths.append(child)
                paths.extend(self._collect_key_paths(v, child))
        return paths

    def _maybe_dump_obs_fields(self, env_obs):
        """Dump raw observation field paths once for debugging schema mismatches."""
        if (not self._debug_dump_obs_fields) or self._obs_field_dumped:
            return

        try:
            all_paths = sorted(set(self._collect_key_paths(env_obs)))
            keyword_paths = [
                p for p in all_paths if ("flash" in p.lower() or "cooldown" in p.lower())
            ]
            print("[OBS_FIELD_DUMP] begin")
            print(f"[OBS_FIELD_DUMP] total_paths={len(all_paths)}")
            for p in all_paths:
                print(f"[OBS_FIELD] {p}")
            print("[OBS_FIELD_DUMP] flash_or_cooldown_paths")
            for p in keyword_paths:
                print(f"[OBS_FIELD_FLASH] {p}")
            print("[OBS_FIELD_DUMP] end")
            self._obs_field_dumped = True
        except Exception as e:
            print(f"[OBS_FIELD_DUMP_ERROR] {e}")

    def _compute_rewards(
        self,
        frame_state,
        env_info,
        hero_pos,
        monster_feats,
        last_action,
        legal_action=None,
        map_info=None,
        new_explored_cells=0,
        terminated=False,
        truncated=False,
    ):
        """Compute all reward components based on current and historical state.

        根据当前帧和历史状态计算所有奖励分量。
        返回总奖励值和各分量的详细信息字典。

        Args:
            frame_state: 当前帧状态数据
            env_info: 环境信息
            hero_pos: 英雄当前位置 {"x": int, "z": int}
            monster_feats: 怪物特征列表（已处理好的numpy数组列表）
            last_action: 上一帧执行的动作ID
            map_info: 局部地图信息（用于安全区计算）
            new_explored_cells: 本帧新探索的格子数（用于开图奖励）
            terminated: 当前帧是否终局-失败（被抓）
            truncated: 当前帧是否因步数上限终局

        Returns:
            total_reward (float): 总即时奖励
            reward_info (dict): 各奖励分量的详细信息（用于调试/监控）
        """

        # ========== 提取当前帧关键信息 ==========
        hero = frame_state["heroes"]
        cur_treasure_score = float(hero.get("treasure_score", 0.0))
        cur_treasures_collected = self._get_treasure_collected_count(env_info, hero)
        cur_collected_buff = int(env_info.get("collected_buff", 0))

        # 计算当前帧最近怪物的归一化距离（不受视野限制，
        #    视野外怪物同样提供有效坐标和距离信息）
        cur_min_monster_dist_norm = 1.0
        for m_feat in monster_feats:
            # m_feat[4] 始终为有效归一化距离（已移除视野门控）
            if m_feat[4] > 0:  # 仅排除未生成的空怪物（全零向量）
                cur_min_monster_dist_norm = min(cur_min_monster_dist_norm, m_feat[4])

        # 计算当前帧到最近宝箱/加速buff的距离（仅最近目标）
        organs = frame_state.get("organs", [])
        visible_treasure_organs = self._filter_visible_organs(organs, hero_pos, target_sub_type=1)
        visible_speed_buff_organs = self._filter_visible_organs(organs, hero_pos, target_sub_type=2)
        cur_min_treasure_dist_norm, _, cur_treasure_found = self._compute_nearest_organ_distance(
            visible_treasure_organs, hero_pos, 1
        )
        cur_min_speed_buff_dist_norm, _, cur_speed_buff_found = self._compute_nearest_organ_distance(
            visible_speed_buff_organs, hero_pos, 2
        )

        # 判断当前是否使用了闪现动作（动作8-15为闪现）
        cur_used_flash = 8 <= last_action <= 15

        # 获取怪物速度信息（用于判断是否进入加速阶段）
        monster_speed = env_info.get("monster_speed", 1)
        monster_speedup = env_info.get("monster_speedup", 500)

        # 计算本步位移（供多项惩罚复用）
        step_displacement = None
        if self.last_hero_pos is not None:
            step_displacement = np.sqrt(
                (hero_pos["x"] - self.last_hero_pos["x"]) ** 2
                + (hero_pos["z"] - self.last_hero_pos["z"]) ** 2
            )

        # ========== 开始计算各奖励分量 ==========

        # 1. 【稠密】生存奖励 —— 每一步固定给一个基础reward，
        #    保证模型在没有拿到显式收益时也能收到"先活着"的信号
        survive_reward = 0.0
        if self._cfg("survive_reward", "enable", True):
            survive_reward = float(self._cfg("survive_reward", "coef", 0.1))

        # 2. 【稀疏】宝箱分奖励 —— 比较当前帧和上一帧的宝箱得分，
        #    只有真的吃到宝箱、宝箱分上涨时才触发
        treasure_reward = 0.0
        treasure_delta = cur_treasure_score - self.last_treasure_score
        treasure_collected_delta = cur_treasures_collected - self.last_treasure_collected_count
        if self._cfg("treasure_reward", "enable", True):
            # 优先按“宝箱数量增量”计奖：拿到1个宝箱给1次奖励。
            if treasure_collected_delta > 0:
                treasure_reward = float(self._cfg("treasure_reward", "coef", 1.0)) * treasure_collected_delta
            # 兼容缺字段场景：回退到 treasure_score 增量。
            elif treasure_collected_delta == 0 and treasure_delta > 0 and cur_treasures_collected == 0:
                treasure_reward = float(self._cfg("treasure_reward", "coef", 1.0)) * treasure_delta

        # 3. 【稀疏】加速buff获取奖励 —— 比较当前帧和上一帧拿到的buff数量，
        #    拿到新buff时触发固定奖励
        speed_buff_reward = 0.0
        buff_delta = cur_collected_buff - self.last_collected_buff
        if self._cfg("speed_buff_reward", "enable", True) and buff_delta > 0:
            # 每拾取一个加速buff给予固定奖励
            speed_buff_reward = float(self._cfg("speed_buff_reward", "coef", 0.2)) * buff_delta

        # 4. 【稠密】加速buff靠近奖励 —— 仅计算最近加速buff，接近时给正向奖励
        speed_buff_approach_reward = 0.0
        speed_buff_potential_active = cur_speed_buff_found and (buff_delta <= 0)
        if self._cfg("speed_buff_approach_reward", "enable", True):
            if speed_buff_potential_active:
                speed_buff_potential = self._compute_potential_from_distance(
                    cur_min_speed_buff_dist_norm,
                    self._cfg("speed_buff_approach_reward", "gravity_coef", 0.0015),
                    self._cfg("speed_buff_approach_reward", "min_dist_norm", 0.05),
                    self._cfg("speed_buff_approach_reward", "min_reward", 0.001),
                    self._cfg("speed_buff_approach_reward", "max_reward", 0.1),
                )
                if self.last_speed_buff_potential_active:
                    speed_buff_approach_reward = speed_buff_potential - self.last_speed_buff_potential
                else:
                    speed_buff_approach_reward = speed_buff_potential
                self.last_speed_buff_potential = speed_buff_potential
                self.last_speed_buff_potential_active = True
            else:
                self.last_speed_buff_potential = 0.0
                self.last_speed_buff_potential_active = False

        # 5. 【稠密】宝箱接近奖励（平方反比引力）—— 只对最近宝箱计算，接近时触发
        treasure_approach_reward = 0.0
        treasure_potential_active = cur_treasure_found and (treasure_collected_delta <= 0)
        if self._cfg("treasure_approach_reward", "enable", True):
            if treasure_potential_active:
                treasure_potential = self._compute_potential_from_distance(
                    cur_min_treasure_dist_norm,
                    self._cfg("treasure_approach_reward", "gravity_coef", 0.0015),
                    self._cfg("treasure_approach_reward", "min_dist_norm", 0.05),
                    self._cfg("treasure_approach_reward", "min_reward", 0.001),
                    self._cfg("treasure_approach_reward", "max_reward", 0.1),
                )
                if self.last_treasure_potential_active:
                    treasure_approach_reward = treasure_potential - self.last_treasure_potential
                else:
                    treasure_approach_reward = treasure_potential
                self.last_treasure_potential = treasure_potential
                self.last_treasure_potential_active = True
            else:
                self.last_treasure_potential = 0.0
                self.last_treasure_potential_active = False

        # 6. 【稠密】怪物距离 shaping —— 比较当前帧和上一帧到最近怪物的距离，
        #    离怪物更远就加分，离怪物更近就减弱甚至变成负反馈
        monster_dist_delta_raw = cur_min_monster_dist_norm - self.last_min_monster_dist_norm
        monster_dist_shaping = 0.0
        if self._cfg("monster_dist_shaping", "enable", True):
            monster_dist_shaping = float(self._cfg("monster_dist_shaping", "coef", 1.0)) * monster_dist_delta_raw

        # 7. 【稠密】后期生存奖励 —— 怪物进入加速阶段后，
        #    明显提高"和怪物拉距离""维持安全空间"这类项的重要性
        late_survive_reward = 0.0
        is_late_game = self.step_no > monster_speedup
        if self._cfg("late_survive_reward", "enable", True) and is_late_game:
            # 后期额外增加基础生存奖励权重
            late_survive_reward = float(self._cfg("late_survive_reward", "base", 0.02))
            # 后期对怪物距离shaping进行加成
            if monster_dist_delta_raw > 0:  # 只在拉远距离时加成
                late_survive_reward += float(
                    self._cfg("late_survive_reward", "shaping_coef", 0.1)
                ) * monster_dist_delta_raw

        # 8. 【稠密】危险惩罚 —— 直接看最近怪物距离是否低于阈值，
        #    怪物越近惩罚越重；加速后同样的距离会更危险
        danger_penalty = 0.0
        danger_threshold = float(self._global_cfg("danger_threshold", 0.15))  # 基础危险距离阈值
        # 加速阶段降低阈值（同样距离更危险）
        if is_late_game:
            danger_threshold = float(self._global_cfg("late_danger_threshold", 0.25))
        if self._cfg("danger_penalty", "enable", True) and cur_min_monster_dist_norm < danger_threshold:
            # 距离越近惩罚越重，使用指数衰减
            danger_penalty = float(self._cfg("danger_penalty", "coef", -0.1)) * (
                1 + danger_threshold - cur_min_monster_dist_norm
            ) ** float(self._cfg("danger_penalty", "power", 2.0))

        # 9. 【稀疏】撞墙 / 无效移动惩罚 —— 只在惩罚分支里直接记录，不额外实现碰撞检测。
        #    每次触发都记录一条，不做去重。
        wall_collision_penalty = 0.0
        is_move_action = 0 <= int(last_action) <= 7
        if step_displacement is not None:
            # 位移小于0.5格视为无效移动（正常移动至少1格）
            if (
                self._cfg("wall_collision_penalty", "enable", True)
                and step_displacement < float(self._global_cfg("wall_displacement_threshold", 0.5))
                and is_move_action
            ):
                wall_collision_penalty = float(self._cfg("wall_collision_penalty", "coef", -0.1))
                log_kwargs = {
                    "hero_pos": hero_pos,
                    "last_action": last_action,
                    "legal_action": legal_action,
                    "map_info": map_info,
                    "step_displacement": step_displacement,
                    "is_speed_phase": (float(hero.get("buff_remaining_time", 0)) > 0),
                }
                if self.episode_mode == "eval":
                    self._log_eval_wall_collision_case(**log_kwargs)
                else:
                    self._log_wall_collision_case(**log_kwargs)

        # 10. 【稀疏】闪现操作失败惩罚 —— 使用闪现动作后位移远小于期望距离，
        #     说明闪现目标位置被障碍物阻挡，闪现未能到达预期目标
        flash_fail_penalty = 0.0
        flash_fail_triggered = False
        if (
            self._cfg("flash_fail_penalty", "enable", True)
            and cur_used_flash
            and self.last_flash_action >= 8
            and self.last_hero_pos is not None
        ):
            flash_displacement = np.sqrt(
                (hero_pos["x"] - self.last_hero_pos["x"]) ** 2 +
                (hero_pos["z"] - self.last_hero_pos["z"]) ** 2
            )
            # 根据闪现方向确定期望距离：
            #   动作8,10,12,14（正交方向）：期望10格
            #   动作9,11,13,15（斜向方向）：期望8格
            last_flash_dir = self.last_flash_action - 8  # 0-7 对应8个方向
            is_diagonal = last_flash_dir in {1, 3, 5, 7}
            expected_dist = float(self._global_cfg("flash_expected_dist_diagonal", 8.0)) if is_diagonal else float(
                self._global_cfg("flash_expected_dist_orthogonal", 10.0)
            )
            # 实际位移小于期望的30%则判定为失败（被墙阻挡）
            if flash_displacement < expected_dist * float(self._global_cfg("flash_fail_ratio", 0.3)):
                flash_fail_penalty = float(self._cfg("flash_fail_penalty", "coef", -0.15))
                flash_fail_triggered = True

        # 记录上一决策动作与执行结果，用于分析合法动作掩码是否有效。
        no_movement_case = 0.0
        move_mask_consistency_hit = 0.0
        move_mask_consistency_total = 0.0
        if step_displacement is not None and int(last_action) >= 0:
            mask_effect_info = self._log_action_mask_effect_case(
                last_action=last_action,
                step_displacement=step_displacement,
                wall_collision_triggered=(wall_collision_penalty != 0.0),
                flash_fail_triggered=flash_fail_triggered,
                is_speed_phase_exec=(float(hero.get("buff_remaining_time", 0)) > 0),
                map_info=map_info,
            )
            no_movement_case = float(mask_effect_info.get("no_movement_case", 0.0))
            move_mask_consistency_hit = float(mask_effect_info.get("move_mask_consistency_hit", 0.0))
            move_mask_consistency_total = float(mask_effect_info.get("move_mask_consistency_total", 0.0))

        # 11.【延迟结算】危险闪现成功奖励
        #     条件：闪现前最近怪物在危险阈值内；若10%CD窗口内未被抓，
        #     在窗口结束时一次性给 base + max(0, 距离增量)*coef
        flash_escape_reward = 0.0
        flash_survival_reward = 0.0
        if self._cfg("flash_escape_reward", "enable", True):
            # 启动危险闪现评估
            if cur_used_flash:
                # 优先使用环境配置冷却值（docs: env_info.flash_cooldown），英雄冷却仅用于回退。
                flash_cd_total = float(env_info.get("flash_cooldown", hero.get("flash_cooldown", MAX_FLASH_CD)))
                flash_cd_total = max(1.0, flash_cd_total)
                self.flash_escape_window_steps = max(1, int(round(flash_cd_total * 0.1)))
                self.flash_survival_window_end_steps = max(
                    self.flash_escape_window_steps,
                    int(round(flash_cd_total * 0.3)),
                )
                self.flash_escape_steps = 0
                # 每次闪现独立计数：闪现后步数重置为0，避免跨闪现越界发奖。
                self.flash_since_use_steps = 0
                self.flash_escape_pre_dist = float(self.last_min_monster_dist_norm)
                self.flash_escape_active = self.flash_escape_pre_dist < danger_threshold
                self.flash_success_blocked = False
                self.flash_survival_decay = 0.0

                # 启动闪现滥用检测计数（用于10步内被抓惩罚）
                self.flash_recent_steps = 0

            if self.flash_since_use_steps is not None:
                self.flash_since_use_steps += 1

            # 更新评估步数
            if self.flash_escape_active:
                self.flash_escape_steps += 1
                caught_now = bool(terminated and (not truncated))

                if caught_now:
                    self.flash_escape_active = False
                    self.flash_success_blocked = True
                    self.flash_survival_decay = 0.0
                elif self.flash_escape_steps >= self.flash_escape_window_steps:
                    if not self.flash_success_blocked:
                        dist_gain = max(0.0, cur_min_monster_dist_norm - self.flash_escape_pre_dist)
                        flash_escape_reward = float(self._cfg("flash_escape_reward", "base", 1.0)) + float(
                            self._cfg("flash_escape_reward", "dist_gain_coef", 1.0)
                        ) * dist_gain
                        if self._cfg("flash_survival_reward", "enable", True):
                            self.flash_survival_decay = float(
                                self._cfg("flash_survival_reward", "init", 0.5)
                            )
                    self.flash_escape_active = False

        # 闪现衰减存活奖励：仅在10%CD触发后开始发放
        if self._cfg("flash_survival_reward", "enable", True) and self.flash_survival_decay > 0:
            if self.flash_success_blocked:
                self.flash_survival_decay = 0.0
            elif self.flash_since_use_steps is None:
                self.flash_survival_decay = 0.0
            elif self.flash_since_use_steps > self.flash_survival_window_end_steps:
                # 超过30%CD立即取消衰减奖励，等待下一次闪现重置。
                self.flash_survival_decay = 0.0
                self.flash_since_use_steps = None
            else:
                flash_survival_reward = self.flash_survival_decay
                self.flash_survival_decay *= float(self._cfg("flash_survival_reward", "decay", 0.95))

        # 判断当前是否持有加速buff（buff_remaining_time > 0 表示生效中）
        cur_had_speed_buff = float(hero.get("buff_remaining_time", 0)) > 0

        # 12.【衰减稠密】加速期间逃离额外奖励 —— 获取加速buff时重置为初始值0.05，
        #     持有期间每步根据怪物距离变化给奖：远离加分，靠近扣分，同时整体衰减
        speed_buff_escape_reward = 0.0
        # 检测到刚获取buff的时刻（上一帧没有，当前帧有），重置衰减
        if (
            self._cfg("speed_buff_escape_reward", "enable", True)
            and cur_had_speed_buff
            and not self.last_had_speed_buff
        ):
            self.speed_buff_escape_decay = float(self._cfg("speed_buff_escape_reward", "init", 0.05))
        if (
            self._cfg("speed_buff_escape_reward", "enable", True)
            and self.speed_buff_escape_decay > 0
            and cur_had_speed_buff
        ):
            # 基于距离变化计算逃离方向奖励：远离加分，靠近扣分
            dist_delta = cur_min_monster_dist_norm - self.last_min_monster_dist_norm
            escape_component = float(self._cfg("speed_buff_escape_reward", "dist_delta_coef", 0.1)) * dist_delta
            # 乘以当前衰减系数得到最终奖励
            speed_buff_escape_reward = self.speed_buff_escape_decay * (
                float(self._cfg("speed_buff_escape_reward", "base", 1.0)) + escape_component
            )
            # 每步指数衰减（衰减率0.97，比闪现衰减慢一些，buff持续时间较长）
            self.speed_buff_escape_decay *= float(self._cfg("speed_buff_escape_reward", "decay", 0.97))

        # 13.【稠密】安全区驻留奖励 —— 当前位于“离怪物最远点”所在象限时，
        #     每步给予固定奖励，鼓励稳定停留在安全象限中
        safe_zone_reward = 0.0
        monsters_raw = frame_state.get("monsters", [])
        safe_quadrant_id, _ = self._compute_safe_zone_quadrant(monsters_raw)
        is_in_safe_zone = self._is_in_safe_quadrant(hero_pos, safe_quadrant_id)
        if self._cfg("safe_zone_reward", "enable", True) and is_in_safe_zone:
            safe_zone_reward = float(self._cfg("safe_zone_reward", "coef", 0.01))

        # 14.【稀疏】开图奖励 —— 当英雄局部视野覆盖了未探索过的区域时，
        #     根据新记录的格子数给予较小奖励。出生后前N步不计入（出生保护期）。
        exploration_reward = 0.0
        if self._cfg("exploration_reward", "enable", True) and new_explored_cells > 0:
            # 出生保护：前BIRTH_PROTECTION_STEPS步内不计入开图奖励
            if self.birth_step_counter >= self.BIRTH_PROTECTION_STEPS:
                # 下调开图奖励，避免覆盖怪物距离 shaping
                exploration_reward = float(
                    self._cfg("exploration_reward", "coef_per_cell", 0.0002)
                ) * new_explored_cells

        # 19.【稀疏】坐标访问奖励 —— 用128x138矩阵统计坐标访问次数：
        #     第1次访问 +0.5，第2次访问 +0.25，第3次及以后 0。
        visit_tracking_reward, visit_tracking_count = self._compute_visit_tracking_reward(hero_pos)

        # 15.【稠密】轨迹质心远离奖励 —— 记录英雄最近N步坐标，计算质心，
        #     鼓励英雄远离历史轨迹质心（避免原地打转/重复路径）
        centroid_away_reward = 0.0
        idle_wander_penalty = 0.0
        # 将当前坐标加入轨迹缓冲区
        self.trajectory_buffer.append((float(hero_pos["x"]), float(hero_pos["z"])))
        if len(self.trajectory_buffer) > self.TRAJECTORY_WINDOW:
            self.trajectory_buffer.pop(0)  # 保持滑动窗口大小

        # 16.【稠密】原地不动/小范围徘徊惩罚
        #     - 原地不动：本步位移低于阈值，给固定负奖励
        #     - 小范围徘徊：最近轨迹的最大离心半径低于阈值，按比例追加负奖励
        #     - 连续停留/徘徊时惩罚会随持续时间递增；离开区域超过阈值后重置
        if self._cfg("idle_wander_penalty", "enable", True):
            current_pos = np.array([float(hero_pos["x"]), float(hero_pos["z"])], dtype=np.float32)
            if self.idle_wander_anchor_pos is None:
                self.idle_wander_anchor_pos = current_pos

            reset_dist = float(self._global_cfg("idle_wander_reset_distance", 6.0))
            if reset_dist > 1e-6 and self.idle_wander_anchor_pos is not None:
                away_dist = float(np.linalg.norm(current_pos - self.idle_wander_anchor_pos))
                if away_dist >= reset_dist:
                    self.idle_streak_steps = 0
                    self.wander_streak_steps = 0
                    self.idle_wander_anchor_pos = current_pos

            if step_displacement is not None and step_displacement < float(
                self._global_cfg("idle_displacement_threshold", 0.25)
            ):
                self.idle_streak_steps += 1
                idle_scale = 1.0 + float(self._cfg("idle_wander_penalty", "idle_growth", 0.08)) * (
                    self.idle_streak_steps - 1
                )
                idle_wander_penalty += float(self._cfg("idle_wander_penalty", "idle_coef", -0.03)) * idle_scale
            else:
                self.idle_streak_steps = 0

            wander_min_points = int(self._global_cfg("wander_min_points", 8))
            wander_radius_threshold = float(self._global_cfg("wander_radius_threshold", 2.5))
            if len(self.trajectory_buffer) >= wander_min_points and wander_radius_threshold > 1e-6:
                coords = np.array(self.trajectory_buffer)
                centroid = np.mean(coords, axis=0)
                max_radius = float(np.sqrt(np.max(np.sum((coords - centroid) ** 2, axis=1))))
                if max_radius < wander_radius_threshold:
                    self.wander_streak_steps += 1
                    ratio = 1.0 - (max_radius / wander_radius_threshold)
                    wander_scale = 1.0 + float(
                        self._cfg("idle_wander_penalty", "wander_growth", 0.05)
                    ) * (self.wander_streak_steps - 1)
                    idle_wander_penalty += float(
                        self._cfg("idle_wander_penalty", "wander_coef", -0.05)
                    ) * ratio * wander_scale
                else:
                    self.wander_streak_steps = 0
            else:
                self.wander_streak_steps = 0

        # 17.【稀疏】闪现滥用惩罚
        #     1) 闪现后10步内被抓 -1
        #     2) 在安全区闪现且未拿到宝箱 -1
        flash_abuse_penalty = 0.0
        flash_abuse_penalty_caught = 0.0
        flash_abuse_penalty_safe_zone = 0.0
        if self._cfg("flash_abuse_penalty", "enable", True):
            # 安全区无收益闪现惩罚（当步）
            if cur_used_flash and is_in_safe_zone and treasure_delta <= 0:
                flash_abuse_penalty_safe_zone += float(
                    self._cfg("flash_abuse_penalty", "safe_zone_no_treasure_coef", -1.0)
                )
                self.flash_success_blocked = True
                self.flash_escape_active = False
                self.flash_survival_decay = 0.0

            # 闪现后N步内被抓惩罚
            if self.flash_recent_steps is not None:
                caught_window = int(self._cfg("flash_abuse_penalty", "caught_within_steps", 10))
                if bool(terminated and (not truncated)) and self.flash_recent_steps <= caught_window:
                    decay_enable = bool(self._cfg("flash_abuse_penalty", "caught_decay_enable", True))
                    if decay_enable:
                        decay_min_step = int(self._cfg("flash_abuse_penalty", "caught_decay_min_step", 1))
                        decay_max_step = int(self._cfg("flash_abuse_penalty", "caught_decay_max_step", 15))
                        coef_start = float(self._cfg("flash_abuse_penalty", "caught_coef_start", -10.0))
                        coef_end = float(self._cfg("flash_abuse_penalty", "caught_coef_end", -1.0))

                        s = max(1, int(self.flash_recent_steps))
                        if s <= decay_min_step:
                            caught_coef = coef_start
                        elif s >= decay_max_step:
                            caught_coef = coef_end
                        else:
                            t = float(s - decay_min_step) / float(max(1, decay_max_step - decay_min_step))
                            caught_coef = coef_start + (coef_end - coef_start) * t
                        flash_abuse_penalty_caught += caught_coef
                    else:
                        flash_abuse_penalty_caught += float(self._cfg("flash_abuse_penalty", "caught_coef", -1.0))
                    self.flash_success_blocked = True
                    self.flash_escape_active = False
                    self.flash_survival_decay = 0.0
                    self.flash_recent_steps = None
                else:
                    self.flash_recent_steps += 1
                    if self.flash_recent_steps > caught_window:
                        self.flash_recent_steps = None

        flash_abuse_penalty = flash_abuse_penalty_caught + flash_abuse_penalty_safe_zone

        # 18.【稠密】死角/死路惩罚（进入后每步惩罚，远离后停止）
        dead_end_penalty = 0.0
        if self._cfg("dead_end_penalty", "enable", True):
            curr = np.array([float(hero_pos["x"]), float(hero_pos["z"])], dtype=np.float32)
            dead_end_now = self._is_half_surrounded_dead_end(map_info)

            if dead_end_now and not self.dead_end_active:
                self.dead_end_active = True
                self.dead_end_anchor_pos = curr

            if self.dead_end_active and self.dead_end_anchor_pos is not None:
                reset_dist = float(self._global_cfg("dead_end_reset_distance", 8.0))
                move_away = float(np.linalg.norm(curr - self.dead_end_anchor_pos))
                if move_away >= max(1e-6, reset_dist):
                    self.dead_end_active = False
                    self.dead_end_anchor_pos = None
                else:
                    dead_end_penalty = float(self._cfg("dead_end_penalty", "coef", -0.5))

        # 当积累了足够多的轨迹点后计算质心
        if len(self.trajectory_buffer) >= 3:  # 至少需要3个点才有意义
            coords = np.array(self.trajectory_buffer)
            centroid_x = np.mean(coords[:, 0])
            centroid_z = np.mean(coords[:, 1])
            # 计算当前点到质心的距离（归一化）
            dist_to_centroid = np.sqrt(
                (hero_pos["x"] - centroid_x) ** 2 +
                (hero_pos["z"] - centroid_z) ** 2
            )
            dist_norm = _norm(
                dist_to_centroid,
                float(self._global_cfg("centroid_norm_max", MAP_SIZE * 0.5)),
            )
            # 距离质心越远奖励越高，鼓励离开历史活动区域
            if self._cfg("centroid_away_reward", "enable", True):
                centroid_away_reward = float(self._cfg("centroid_away_reward", "coef", 0.005)) * dist_norm

        # ========== 汇总所有奖励分量 ==========
        total_reward = (
            survive_reward +
            treasure_reward +
            speed_buff_reward +
            speed_buff_approach_reward +
            treasure_approach_reward +
            monster_dist_shaping +
            late_survive_reward +
            danger_penalty +
            wall_collision_penalty +
            flash_fail_penalty +
            flash_escape_reward +
            flash_survival_reward +
            speed_buff_escape_reward +
            safe_zone_reward +
            flash_abuse_penalty +
            dead_end_penalty +
            exploration_reward +
            visit_tracking_reward +
            centroid_away_reward +
            idle_wander_penalty
        )

        # ========== 更新历史状态（供下一帧使用）==========
        self.last_min_monster_dist_norm = cur_min_monster_dist_norm
        self.last_treasure_score = cur_treasure_score
        self.last_treasure_collected_count = cur_treasures_collected
        self.last_collected_buff = cur_collected_buff
        self.last_min_treasure_dist_norm = cur_min_treasure_dist_norm
        self.last_min_speed_buff_dist_norm = cur_min_speed_buff_dist_norm
        self.last_hero_pos = {"x": hero_pos["x"], "z": hero_pos["z"]}
        # 更新闪现前的怪物距离（用于下次判断闪现效果）
        if cur_used_flash:
            self.last_min_monster_dist_before_flash = cur_min_monster_dist_norm
        self.last_used_flash = cur_used_flash
        # 更新buff持有状态（用于检测buff获取时刻）
        self.last_had_speed_buff = cur_had_speed_buff

        # 记录当前闪现动作ID（供下一帧检测闪现失败用）
        if cur_used_flash:
            self.last_flash_action = last_action
        # 更新最近怪物序号追踪
        self.last_nearest_monster_idx = self._compute_nearest_monster_idx(monsters_raw, hero_pos)

        # 构建奖励详情字典（用于调试/日志）
        reward_info = {
            "survive_reward": survive_reward,
            "treasure_reward": treasure_reward,
            "treasure_collected_delta": float(treasure_collected_delta),
            "speed_buff_reward": speed_buff_reward,
            "speed_buff_approach_reward": speed_buff_approach_reward,
            "treasure_approach_reward": treasure_approach_reward,
            "monster_dist_shaping": monster_dist_shaping,
            "late_survive_reward": late_survive_reward,
            "danger_penalty": danger_penalty,
            "wall_collision_penalty": wall_collision_penalty,
            "flash_fail_penalty": flash_fail_penalty,
            "flash_escape_reward": flash_escape_reward,
            "flash_survival_reward": flash_survival_reward,
            "flash_escape_active": float(self.flash_escape_active),
            "flash_escape_steps": float(self.flash_escape_steps),
            "flash_escape_window_steps": float(self.flash_escape_window_steps),
            "flash_since_use_steps": float(self.flash_since_use_steps)
            if self.flash_since_use_steps is not None
            else -1.0,
            "flash_survival_window_end_steps": float(self.flash_survival_window_end_steps),
            "flash_success_blocked": float(self.flash_success_blocked),
            "flash_survival_decay": self.flash_survival_decay,
            "speed_buff_escape_reward": speed_buff_escape_reward,
            "speed_escape_decay_value": self.speed_buff_escape_decay,
            "safe_zone_reward": safe_zone_reward,
            "flash_abuse_penalty": flash_abuse_penalty,
            "flash_abuse_penalty_caught": flash_abuse_penalty_caught,
            "flash_abuse_penalty_safe_zone": flash_abuse_penalty_safe_zone,
            "dead_end_penalty": dead_end_penalty,
            "dead_end_active": float(self.dead_end_active),
            "no_movement_case": no_movement_case,
            "move_mask_consistency_hit": move_mask_consistency_hit,
            "move_mask_consistency_total": move_mask_consistency_total,
            "exploration_reward": exploration_reward,
            "visit_tracking_reward": visit_tracking_reward,
            "visit_tracking_count": visit_tracking_count,
            "centroid_away_reward": centroid_away_reward,
            "idle_wander_penalty": idle_wander_penalty,
            "idle_streak_steps": float(self.idle_streak_steps),
            "wander_streak_steps": float(self.wander_streak_steps),
            "total_reward": total_reward,
        }

        # 更新已探索格子计数
        self.last_explored_count = int(np.sum(self.explored_map > 0))

        # 存储reward_info供agent获取
        self.last_reward_info = reward_info

        return total_reward, reward_info

    def _compute_min_treasure_distance(self, organs, hero_pos):
        """Compute normalized distance to the nearest treasure chest.

        计算到最近宝箱的归一化距离。

        Args:
            organs: 物件状态列表（包含宝箱和buff）
            hero_pos: 英雄当前位置

        Returns:
            float: 归一化后的最近宝箱距离 [0, 1]
        """
        visible_treasures = self._filter_visible_organs(organs, hero_pos, target_sub_type=1)
        dist_norm, _, _ = self._compute_nearest_organ_distance(visible_treasures, hero_pos, 1)
        return dist_norm

    def _compute_visit_tracking_reward(self, hero_pos):
        """Track 128x138 coordinate visits and return sparse visit reward.

        规则：首次访问 +0.5，第二次访问 +0.25，第三次及以后不再奖励。
        """
        hx = int(hero_pos.get("x", -1))
        hz = int(hero_pos.get("z", -1))

        if not (0 <= hz < VISIT_TRACK_ROWS and 0 <= hx < VISIT_TRACK_COLS):
            return 0.0, 0.0

        prev_count = int(self.visit_count_map[hz, hx])
        cur_count = min(prev_count + 1, 255)
        self.visit_count_map[hz, hx] = cur_count

        if not self._cfg("visit_tracking_reward", "enable", True):
            return 0.0, float(cur_count)

        if cur_count == 1:
            return float(self._cfg("visit_tracking_reward", "first_visit", 0.5)), float(cur_count)
        if cur_count == 2:
            return float(self._cfg("visit_tracking_reward", "second_visit", 0.25)), float(cur_count)
        return 0.0, float(cur_count)

    def _update_explored_map(self, hero_pos, map_info):
        """Update global explored map with current local FOV.

        将当前局部视野中扫过的区域记录到全局探索记忆图中。

        规则：
        - 局部视野中可通行的格子（map_info值!=0）标记为已探索
        - 障碍物格子（map_info==0）不标记为"英雄走过"但可计入新发现奖励
          （因为首次看到障碍物本身也是有价值的信息）

        Args:
            hero_pos: 英雄当前位置 {"x": int, "z": int}
            map_info: 局部地图信息（21×21栅格），1=可通行，0=障碍物

        Returns:
            new_cells_count (int): 本帧新增的已探索格子数
        """
        if map_info is None:
            return 0

        map_rows = len(map_info)
        map_cols = len(map_info[0]) if map_rows > 0 else 0
        if map_rows < 21 or map_cols < 21:
            return 0

        center = map_rows // 2  # 通常=10
        hx, hz = int(hero_pos["x"]), int(hero_pos["z"])
        new_count = 0

        for r in range(map_rows):
            for c in range(map_cols):
                # 局部坐标 → 全局坐标
                global_z = hz + (r - center)
                global_x = hx + (c - center)
                # 边界检查
                if 0 <= global_x < int(MAP_SIZE) and 0 <= global_z < int(MAP_SIZE):
                    # 只标记可通行区域为"已探索"
                    if map_info[r][c] != 0 and self.explored_map[global_z, global_x] == 0:
                        self.explored_map[global_z, global_x] = 1.0
                        new_count += 1
                    # 障碍物格子：即使不可通行，也标记（用于统计首次发现）
                    elif map_info[r][c] == 0 and self.explored_map[global_z, global_x] == 0:
                        # 障碍物用特殊值-1标记"已看见"，不计入探索面积
                        self.explored_map[global_z, global_x] = -1.0
                        new_count += 1  # 首次看到的障碍物也算开图

        return new_count

    def feature_process(self, env_obs, last_action):
        """Process env_obs into feature vector, legal_action mask, and reward.

        将 env_obs 转换为特征向量、合法动作掩码和即时奖励。
        """
        observation = env_obs["observation"]
        self._maybe_dump_obs_fields(env_obs)
        frame_state = observation["frame_state"]
        env_info = observation["env_info"]
        map_info = observation["map_info"]
        legal_act_raw = observation["legal_action"]
        terminated = bool(env_obs.get("terminated", False))
        truncated = bool(env_obs.get("truncated", False))

        self.step_no = observation["step_no"]
        self.max_step = env_info.get("max_step", 200)

        # 出生保护步数计数器递增
        if self.birth_step_counter < self.BIRTH_PROTECTION_STEPS:
            self.birth_step_counter += 1

        # Hero self features (4D) / 英雄自身特征
        hero = frame_state["heroes"]
        hero_pos = hero["pos"]
        hero_x_norm = _norm(hero_pos["x"], MAP_SIZE)
        hero_z_norm = _norm(hero_pos["z"], MAP_SIZE)
        flash_cd_norm = _norm(hero["flash_cooldown"], MAX_FLASH_CD)
        buff_remain_norm = _norm(hero["buff_remaining_time"], MAX_BUFF_DURATION)

        hero_feat = np.array([hero_x_norm, hero_z_norm, flash_cd_norm, buff_remain_norm], dtype=np.float32)
        organs = frame_state.get("organs", [])
        visible_treasure_organs = self._filter_visible_organs(organs, hero_pos, target_sub_type=1)
        visible_speed_buff_organs = self._filter_visible_organs(organs, hero_pos, target_sub_type=2)

        # Treasure slots (10 x 13) / 宝箱槽位（10个对象，不足补-1）
        treasure_slots = self._collect_organ_slots(
            organs=visible_treasure_organs,
            hero_pos=hero_pos,
            target_sub_type=1,
            slot_num=10,
        )

        # Speed-buff slots (2 x 13) / 加速buff槽位（2个对象，不足补-1）
        speed_buff_slots = self._collect_organ_slots(
            organs=visible_speed_buff_organs,
            hero_pos=hero_pos,
            target_sub_type=2,
            slot_num=2,
        )

        # Monster features (14D x 2) / 怪物特征（每个怪物独立槽位，不足补-1）
        #    结构：is_in_view + x + z + speed + dist + 9维相对方向one-hot
        monsters = frame_state.get("monsters", [])
        monster_feats = []
        for i in range(2):
            if i < len(monsters):
                m = monsters[i]
                is_in_view = float(m.get("is_in_view", 0))
                m_pos = m["pos"]
                # 无论是否在视野内，都使用原始坐标计算归一化位置、速度和距离
                m_x_norm = _norm(m_pos["x"], MAP_SIZE)
                m_z_norm = _norm(m_pos["z"], MAP_SIZE)
                m_speed_norm = _norm(m.get("speed", 1), MAX_MONSTER_SPEED)

                # Euclidean distance / 欧式距离（基于绝对坐标，不受视野限制）
                raw_dist = np.sqrt((hero_pos["x"] - m_pos["x"]) ** 2 + (hero_pos["z"] - m_pos["z"]) ** 2)
                dist_norm = _norm(raw_dist, MAP_SIZE * 1.41)
                dir_feat = self._compute_relative_direction_one_hot(m, hero_pos)
                monster_feat = np.concatenate(
                    [np.array([is_in_view, m_x_norm, m_z_norm, m_speed_norm, dist_norm], dtype=np.float32), dir_feat]
                )
                monster_feats.append(monster_feat)
            else:
                monster_feats.append(np.full(14, -1.0, dtype=np.float32))

        # Local map features (441D = 21×21) / 局部地图特征（最大视野给actor）
        #    使用完整 21×21 局部视野作为决策输入。
        MAP_VIEW_SIZE = LOCAL_MAP_VIEW_SIZE
        map_feat = np.zeros(MAP_VIEW_SIZE * MAP_VIEW_SIZE, dtype=np.float32)
        if map_info is not None:
            map_rows = len(map_info)
            map_cols = len(map_info[0]) if map_rows > 0 else 0
            center = map_rows // 2  # 通常=10
            flat_idx = 0
            for row in range(MAP_VIEW_SIZE):
                for col in range(MAP_VIEW_SIZE):
                    r = center - (MAP_VIEW_SIZE // 2) + row
                    c = center - (MAP_VIEW_SIZE // 2) + col
                    if 0 <= r < map_rows and 0 <= c < map_cols:
                        map_feat[flat_idx] = float(map_info[r][c] != 0)
                    flat_idx += 1

        # Legal action mask (16D) / 合法动作掩码
        legal_action = [1] * 16
        if isinstance(legal_act_raw, list) and legal_act_raw:
            if isinstance(legal_act_raw[0], bool):
                for j in range(min(16, len(legal_act_raw))):
                    legal_action[j] = int(legal_act_raw[j])
            else:
                valid_set = {int(a) for a in legal_act_raw if int(a) < 16}
                legal_action = [1 if j in valid_set else 0 for j in range(16)]

        raw_legal_action = legal_action.copy()
        if sum(legal_action) == 0:
            legal_action = [1] * 16
            raw_legal_action = legal_action.copy()

        # 预处理0-7移动动作合法性：普通/加速邻域检测
        has_speed_buff = float(hero.get("buff_remaining_time", 0)) > 0
        move_mask = self._preprocess_move_action_mask(map_info, has_speed_buff)
        for i in range(8):
            legal_action[i] = 1 if (legal_action[i] and move_mask[i]) else 0

        # 一致性保护：若环境原始合法动作中存在可移动方向，
        # 但预处理把0-7全部清零，则回退到环境的移动合法性，避免误判导致贴墙抖动。
        raw_move_legal_cnt = sum(raw_legal_action[:8])
        masked_move_legal_cnt = sum(legal_action[:8])
        if raw_move_legal_cnt > 0 and masked_move_legal_cnt == 0:
            for i in range(8):
                legal_action[i] = int(raw_legal_action[i])

        legal_after_move_mask = legal_action.copy()

        # 在已有合法mask上做宝箱检测：
        # 1) 若动作在一步内可摸到宝箱，则其视为“可执行吃宝箱动作”。
        # 2) 加速状态下，若第1步就能摸到宝箱，则该方向强制合法（即使第2步撞墙）。
        # 3) 若存在“可执行吃宝箱动作”，则0-7只保留这些动作。
        move_step = int(self._global_cfg("buff_move_step", 2)) if has_speed_buff else int(
            self._global_cfg("normal_move_step", 1)
        )
        treasure_touch_steps = self._find_target_touch_steps(
            hero_pos, visible_treasure_organs, move_step, target_sub_type=1
        )
        speed_buff_touch_steps = self._find_target_touch_steps(
            hero_pos, visible_speed_buff_organs, move_step, target_sub_type=2
        )
        enable_speed_buff_mask = not has_speed_buff

        if has_speed_buff:
            for i in range(8):
                if treasure_touch_steps[i] == 1:
                    legal_action[i] = 1

        direct_touch_legal = [
            i
            for i in range(8)
            if (
                treasure_touch_steps[i] > 0
                or (enable_speed_buff_mask and speed_buff_touch_steps[i] > 0)
            )
            and legal_action[i] == 1
        ]
        if direct_touch_legal:
            for i in range(8):
                legal_action[i] = 1 if i in direct_touch_legal else 0

        if sum(legal_action) == 0:
            # 方向候选筛选过严时，回退到筛选前的合法掩码，避免放开全部非法动作。
            if sum(legal_after_move_mask) > 0:
                legal_action = legal_after_move_mask
            elif sum(raw_legal_action) > 0:
                legal_action = raw_legal_action
            else:
                legal_action = [1] * 8 + [0] * 8

        # Progress features (2D) / 进度特征
        step_norm = _norm(self.step_no, self.max_step)
        survival_ratio = step_norm
        progress_feat = np.array([step_norm, survival_ratio], dtype=np.float32)

        # Concatenate features / 拼接特征
        feature = np.concatenate(
            [
                hero_feat,
                monster_feats[0],
                monster_feats[1],
                treasure_slots[0],
                treasure_slots[1],
                treasure_slots[2],
                treasure_slots[3],
                treasure_slots[4],
                treasure_slots[5],
                treasure_slots[6],
                treasure_slots[7],
                treasure_slots[8],
                treasure_slots[9],
                speed_buff_slots[0],
                speed_buff_slots[1],
                map_feat,
                progress_feat,
            ]
        )

        # ====== 地图记忆：更新全局探索图并计算开图数据 ======
        new_explored_cells = self._update_explored_map(hero_pos, map_info)

        # ====== 计算完整奖励（调用新的多分量奖励函数）======
        reward, reward_info = self._compute_rewards(
            frame_state, env_info, hero_pos, monster_feats, last_action,
            legal_action=legal_action,
            map_info=map_info,
            new_explored_cells=new_explored_cells,
            terminated=terminated,
            truncated=truncated,
        )

        # 保存当前决策帧的合法动作信息，供下一帧评估掩码有效性。
        self.last_decision_step_no = int(self.step_no)
        self.last_decision_legal_action = [int(x) for x in legal_action]
        self.last_decision_move_mask = [int(x) for x in move_mask]
        self.last_decision_move_mask_corrected = [int(x) for x in legal_action[:8]]
        self.last_decision_is_speed_phase = bool(has_speed_buff)
        self.last_decision_map_center_5x5 = self._extract_center_patch(map_info, patch_size=5)

        return feature, legal_action, [reward]
