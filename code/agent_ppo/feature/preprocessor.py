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
"""

# ============================= 代码结构说明 =============================
# 1. 基础配置与状态管理
#    - _default_reward_config / _load_reward_config / reset
# 2. 地图与几何辅助
#    - _is_half_surrounded_dead_end / _update_explored_map / _count_free_neighbors
# 3. 目标与方向特征
#    - _compute_nearest_organ_distance / _collect_organ_slots / 方向one-hot编码
# 4. 动作后果建模（候选动作特征）
#    - _simulate_next_position / _build_candidate_action_features
#    - _build_action_risk_benefit_features
# 5. 探索事件建模
#    - _region_id_from_pos / _get_local_connected_region_anchor / _compute_exploration_events
# 6. 动作掩码分层
#    - _preprocess_move_action_mask 负责硬合法性
#    - action_risk_benefit_feat 负责软风险/收益指数
# 7. 课程训练奖励
#    - _is_stage_enabled 控制当前阶段启用的奖励项
#    - _compute_rewards 统一结算即时奖励
# 8. 主入口
#    - feature_process: 生成 feature / legal_action / reward
# =====================================================================

import os
import copy
from collections import deque
import numpy as np

try:
    import tomllib
except ModuleNotFoundError:
    tomllib = None

# Map size / 地图尺寸（128×128）
MAP_SIZE = 128.0
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
    def __init__(self):
        self._debug_dump_obs_fields = os.environ.get("KAIWU_DUMP_OBS_FIELDS", "0") == "1"
        self._obs_field_dumped = False
        self.reward_cfg = self._default_reward_config()
        self.reset()

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

    def set_curriculum_stage(self, stage):
        """Set current curriculum stage to [1, 4]."""
        try:
            stage = int(stage)
        except Exception:
            stage = 1
        self.curriculum_stage = max(1, min(4, stage))

    def get_curriculum_stage(self):
        return int(getattr(self, "curriculum_stage", 1))

    def get_curriculum_stage_name(self):
        stage = self.get_curriculum_stage()
        stage_name_map = {
            1: "survival_base",
            2: "explore_and_stabilize",
            3: "safe_resource_acquisition",
            4: "full_game_and_skill_refine",
        }
        return stage_name_map.get(stage, "unknown")

    def get_reward_term_coef(self, section, key="coef", default=1.0):
        """Public helper for workflow-side metric normalization."""
        return float(self._cfg(section, key, default))

    def _is_stage_enabled(self, name):
        """Check whether a reward term is enabled in current curriculum stage.

        Stage 1: 生存底座与基础移动
        Stage 2: 探索建图与脱困稳定化
        Stage 3: 安全前提下的资源获取
        Stage 4: 技能时机与综合博弈精修
        """
        stage = self.get_curriculum_stage()

        stage_reward_map = {
            1: {
                "survive_reward",
                "monster_dist_shaping",
                "danger_penalty",
                "wall_collision_penalty",
            },
            2: {
                "survive_reward",
                "monster_dist_shaping",
                "danger_penalty",
                "wall_collision_penalty",
                "exploration_reward",
                "centroid_away_reward",
                "idle_wander_penalty",
                "dead_end_penalty",
                "safe_zone_reward",
            },
            3: {
                "survive_reward",
                "monster_dist_shaping",
                "danger_penalty",
                "wall_collision_penalty",
                "exploration_reward",
                "centroid_away_reward",
                "idle_wander_penalty",
                "dead_end_penalty",
                "safe_zone_reward",
                "treasure_reward",
                "treasure_approach_reward",
                "speed_buff_reward",
                "speed_buff_approach_reward",
                "speed_buff_escape_reward",
            },
            4: {
                "survive_reward",
                "monster_dist_shaping",
                "danger_penalty",
                "wall_collision_penalty",
                "exploration_reward",
                "centroid_away_reward",
                "idle_wander_penalty",
                "dead_end_penalty",
                "safe_zone_reward",
                "treasure_reward",
                "treasure_approach_reward",
                "speed_buff_reward",
                "speed_buff_approach_reward",
                "speed_buff_escape_reward",
                "flash_fail_penalty",
                "flash_escape_reward",
                "flash_survival_reward",
                "flash_abuse_penalty",
                "late_survive_reward",
            },
        }

        enabled_names = stage_reward_map.get(stage, stage_reward_map[1])
        return name in enabled_names

    def _is_survival_only_stage(self):
        """Return whether current curriculum should suppress resource guidance.

        Stage 1 / 2:
        - 关闭宝箱 / buff 导向偏置
        - 关闭资源靠近型动作收益特征
        - 让模型先把“生存 + 探索 + 脱困”学稳

        Stage 3 / 4:
        - 开启资源相关观测与动作引导
        """
        return self.get_curriculum_stage() <= 2

    def reset(self):
        """Reset per-episode state for reward computation.
        每局开始时只重置“局内状态”，不重置“训练阶段”。
        """

        # ===== 保留跨局课程阶段 =====
        prev_stage = int(getattr(self, "curriculum_stage", 1))

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

        # ========== 地图记忆 / 开图奖励状态 ==========
        # 全局探索记忆图（128x128），记录英雄局部视野扫过的区域
        self.explored_map = np.zeros((int(MAP_SIZE), int(MAP_SIZE)), dtype=np.float32)
        # 上一帧已探索格子总数（用于计算新增开图数）
        self.last_explored_count = 0

        # ========== 出生保护 / 开图奖励 ==========
        # 出生后前N步不计入开图探索奖励（避免出生位置视野带来的虚假奖励）
        self.BIRTH_PROTECTION_STEPS = int(self._global_cfg("birth_protection_steps", 10))
        # 当前已过保护期步数计数（达到BIRTH_PROTECTION_STEPS后才开始计奖）
        self.birth_step_counter = 0

        # ========== 轨迹质心 / 远离奖励 ==========
        # 记录英雄最近N步的坐标轨迹，用于计算质心
        self.TRAJECTORY_WINDOW = int(self._global_cfg("trajectory_window", 20))
        # 滑动窗口大小
        self.trajectory_buffer = []  # [(x, z), ...]

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

        # ========== 恢复跨局课程阶段 ==========
        self.curriculum_stage = max(1, min(4, prev_stage))

        # ========== 高层探索事件记忆 ==========
        # 首次发现宝箱区域、首次进入新连通区域、首次进入远离历史轨迹区域
        self.discovered_treasure_regions = set()
        self.visited_region_anchors = set()
        self.visited_far_trajectory_regions = set()

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

    def _compute_nearest_treasure_feature(self, organs, hero_pos):
        """Get nearest treasure feature as [x_norm, z_norm, dist_norm]."""
        dist_norm, nearest_pos, found = self._compute_nearest_organ_distance(organs, hero_pos, 1)
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
                # 切角移动不合法：斜向移动时，必须同时满足横纵两个邻格都可通行。
                if dr != 0 and dc != 0:
                    prev_r = center_r + dr * (step - 1)
                    prev_c = center_c + dc * (step - 1)
                    if not (is_passable(rr, prev_c) and is_passable(prev_r, cc)):
                        legal = False
                        break
            mask[i] = 1 if legal else 0

        return mask

    # do1:增加候选动作特征
    def _count_free_neighbors(self, map_info, center_r, center_c, radius=1):
        """Count local passable ratio around a center cell.

        统计某个局部中心周围的可通行比例，作为开阔度近似。
        返回值范围 [0, 1]，越大表示越开阔。
        """
        if map_info is None:
            return 0.0

        rows = len(map_info)
        cols = len(map_info[0]) if rows > 0 else 0
        if rows == 0 or cols == 0:
            return 0.0

        total = 0
        free = 0
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr = center_r + dr
                cc = center_c + dc
                if 0 <= rr < rows and 0 <= cc < cols:
                    total += 1
                    if map_info[rr][cc] != 0:
                        free += 1

        if total == 0:
            return 0.0
        return float(free) / float(total)

    def _extract_local_map_centered(self, map_info, center_r, center_c, view_size=21):
        """Extract a local binary map centered at (center_r, center_c).

        以任意局部位置为中心，裁剪一个固定大小的局部地图。
        超出边界部分按障碍处理（填0）。
        """
        out = np.zeros((view_size, view_size), dtype=np.int32)
        if map_info is None:
            return out

        rows = len(map_info)
        cols = len(map_info[0]) if rows > 0 else 0
        if rows == 0 or cols == 0:
            return out

        half = view_size // 2
        for r in range(view_size):
            for c in range(view_size):
                src_r = center_r - half + r
                src_c = center_c - half + c
                if 0 <= src_r < rows and 0 <= src_c < cols:
                    out[r, c] = 1 if map_info[src_r][src_c] != 0 else 0
        return out

    def _simulate_next_position(self, hero_pos, map_info, action_idx, has_speed_buff):
        """Simulate next position for movement action [0..7].

        按当前移动规则模拟执行动作后的落点：
        - 普通移动按1格
        - 加速状态按配置步数
        - 若路径中途非法，则停在最后一个合法格
        - 若起步就非法，则停在原地
        """
        hx = int(hero_pos["x"])
        hz = int(hero_pos["z"])

        if map_info is None:
            return {"x": hx, "z": hz}, None, None

        rows = len(map_info)
        cols = len(map_info[0]) if rows > 0 else 0
        if rows == 0 or cols == 0:
            return {"x": hx, "z": hz}, None, None

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

        dr, dc = dirs8[action_idx]

        def is_passable(r, c):
            return 0 <= r < rows and 0 <= c < cols and map_info[r][c] != 0

        last_valid_r = center_r
        last_valid_c = center_c

        for step in range(1, move_step + 1):
            rr = center_r + dr * step
            cc = center_c + dc * step

            if not is_passable(rr, cc):
                break

            # 斜向移动仍保持和当前mask一致的切角约束
            if dr != 0 and dc != 0:
                prev_r = center_r + dr * (step - 1)
                prev_c = center_c + dc * (step - 1)
                if not (is_passable(rr, prev_c) and is_passable(prev_r, cc)):
                    break

            last_valid_r = rr
            last_valid_c = cc

        delta_r = last_valid_r - center_r
        delta_c = last_valid_c - center_c

        next_pos = {
            "x": int(np.clip(hx + delta_c, 0, int(MAP_SIZE) - 1)),
            "z": int(np.clip(hz + delta_r, 0, int(MAP_SIZE) - 1)),
        }
        return next_pos, last_valid_r, last_valid_c

    def _build_candidate_action_features(self, hero_pos, monsters, organs, map_info, has_speed_buff):
        """Build action-conditioned features for movement actions [0..7].

        每个动作输出7维：
        [next_x_norm, next_z_norm, next_min_monster_dist_norm,
         next_in_safe_quadrant, next_closer_to_treasure,
         next_openness, next_dead_end_flag]
        """
        feat_per_action = 7
        out = np.zeros(8 * feat_per_action, dtype=np.float32)

        # 当前参考量：最近宝箱距离、安全象限
        cur_treasure_dist_norm, _, _ = self._compute_nearest_organ_distance(organs, hero_pos, 1)
        safe_quadrant_id, _ = self._compute_safe_zone_quadrant(monsters)

        for a in range(8):
            next_pos, next_r, next_c = self._simulate_next_position(
                hero_pos=hero_pos,
                map_info=map_info,
                action_idx=a,
                has_speed_buff=has_speed_buff,
            )

            # 1-2) 下一位置
            next_x_norm = _norm(next_pos["x"], MAP_SIZE)
            next_z_norm = _norm(next_pos["z"], MAP_SIZE)

            # 3) 下一位置到最近怪距离
            next_min_monster_dist_norm = 1.0
            for m in monsters:
                pos = m.get("pos", {"x": 0, "z": 0})
                if pos["x"] == 0 and pos["z"] == 0:
                    continue
                dist = np.sqrt(
                    (float(next_pos["x"]) - float(pos["x"])) ** 2 +
                    (float(next_pos["z"]) - float(pos["z"])) ** 2
                )
                next_min_monster_dist_norm = min(
                    next_min_monster_dist_norm,
                    _norm(dist, MAP_SIZE * 1.41),
                )

            # 4) 下一位置是否位于安全象限
            next_in_safe_quadrant = 1.0 if self._is_in_safe_quadrant(next_pos, safe_quadrant_id) else 0.0

            # 5) 下一位置是否更接近宝箱
            next_treasure_dist_norm, _, found_treasure = self._compute_nearest_organ_distance(organs, next_pos, 1)
            next_closer_to_treasure = 0.0
            if (not self._is_survival_only_stage()) and found_treasure and next_treasure_dist_norm < cur_treasure_dist_norm:
                next_closer_to_treasure = 1.0

            # 6) 下一位置周围开阔度
            if next_r is None or next_c is None:
                next_openness = 0.0
                next_dead_end_flag = 0.0
            else:
                next_openness = self._count_free_neighbors(map_info, next_r, next_c, radius=1)

                # 7) 下一位置是否可能进入 dead-end
                local_map = self._extract_local_map_centered(map_info, next_r, next_c, view_size=21)
                next_dead_end_flag = 1.0 if self._is_half_surrounded_dead_end(local_map) else 0.0

            base = a * feat_per_action
            out[base: base + feat_per_action] = np.array(
                [
                    next_x_norm,
                    next_z_norm,
                    next_min_monster_dist_norm,
                    next_in_safe_quadrant,
                    next_closer_to_treasure,
                    next_openness,
                    next_dead_end_flag,
                ],
                dtype=np.float32,
            )

        return out

    def _build_action_risk_benefit_features(self, hero_pos, monsters, organs, map_info, has_speed_buff, move_mask):
        """Build soft risk/benefit indices for each move action [0..7].

        输出结构：8 × 4 = 32维
        [risk_monster, risk_dead_end, benefit_treasure, benefit_openness]

        作用：
        - move_mask 仍然负责“硬禁用非法动作”
        - 这个方法负责为剩余合法动作附加软风险/收益指数
        """
        feat_per_action = 4
        out = np.zeros(8 * feat_per_action, dtype=np.float32)

        cur_treasure_dist_norm, _, _ = self._compute_nearest_organ_distance(organs, hero_pos, 1)

        for a in range(8):
            base = a * feat_per_action

            if move_mask[a] == 0:
                out[base: base + feat_per_action] = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
                continue

            next_pos, next_r, next_c = self._simulate_next_position(
                hero_pos=hero_pos,
                map_info=map_info,
                action_idx=a,
                has_speed_buff=has_speed_buff,
            )

            next_min_monster_dist_norm = 1.0
            for m in monsters:
                pos = m.get("pos", {"x": 0, "z": 0})
                if pos["x"] == 0 and pos["z"] == 0:
                    continue
                dist = np.sqrt(
                    (float(next_pos["x"]) - float(pos["x"])) ** 2 +
                    (float(next_pos["z"]) - float(pos["z"])) ** 2
                )
                next_min_monster_dist_norm = min(next_min_monster_dist_norm, _norm(dist, MAP_SIZE * 1.41))
            risk_monster = 1.0 - next_min_monster_dist_norm

            risk_dead_end = 0.0
            benefit_openness = 0.0
            if next_r is not None and next_c is not None:
                local_map = self._extract_local_map_centered(map_info, next_r, next_c, view_size=21)
                risk_dead_end = 1.0 if self._is_half_surrounded_dead_end(local_map) else 0.0
                benefit_openness = self._count_free_neighbors(map_info, next_r, next_c, radius=1)

            next_treasure_dist_norm, _, found_treasure = self._compute_nearest_organ_distance(organs, next_pos, 1)
            benefit_treasure = 0.0
            if (not self._is_survival_only_stage()) and found_treasure:
                benefit_treasure = max(0.0, cur_treasure_dist_norm - next_treasure_dist_norm)

            out[base: base + feat_per_action] = np.array(
                [risk_monster, risk_dead_end, benefit_treasure, benefit_openness],
                dtype=np.float32,
            )

        return out

    # do1

    def _compute_nearest_speed_buff_feature(self, organs, hero_pos, env_info):
        """Get nearest speed buff feature as [cd_norm, x_norm, z_norm, dist_norm]."""
        dist_norm, nearest_pos, found = self._compute_nearest_organ_distance(organs, hero_pos, 2)

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
        """Collect fixed-count organ slots sorted by distance, pad missing slots with -1.

        Slot layout: x, z, dist, dir_onehot(9), available_flag.
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
                    np.array([
                        _norm(pos["x"], MAP_SIZE),
                        _norm(pos["z"], MAP_SIZE),
                        dist_norm,
                    ], dtype=np.float32),
                    dir_feat,
                    np.array([available_flag], dtype=np.float32),
                ]
            )

            slots.append((dist, feat))

        slots.sort(key=lambda x: x[0])
        selected = [f for _, f in slots[:slot_num]]

        while len(selected) < slot_num:
            selected.append(np.full(13, -1.0, dtype=np.float32))

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
        map_info=None,
        exploration_events=None,
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
            exploration_events: 三类探索事件字典（首次发现宝箱区域/新连通区域/远离历史轨迹区域）
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
        cur_min_treasure_dist_norm, _, _ = self._compute_nearest_organ_distance(organs, hero_pos, 1)
        cur_min_speed_buff_dist_norm, _, _ = self._compute_nearest_organ_distance(organs, hero_pos, 2)

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
        speed_buff_dist_delta = self.last_min_speed_buff_dist_norm - cur_min_speed_buff_dist_norm
        if (
            self._cfg("speed_buff_approach_reward", "enable", True)
            and speed_buff_dist_delta > 0
            and cur_min_speed_buff_dist_norm < 1.0
        ):
            speed_buff_approach_reward = float(
                self._cfg("speed_buff_approach_reward", "coef", 0.2)
            ) * speed_buff_dist_delta

        # 5. 【稠密】宝箱接近奖励（平方反比引力）—— 只对最近宝箱计算，接近时触发
        treasure_approach_reward = 0.0
        treasure_dist_delta = self.last_min_treasure_dist_norm - cur_min_treasure_dist_norm
        if (
            self._cfg("treasure_approach_reward", "enable", True)
            and treasure_dist_delta > 0
            and cur_min_treasure_dist_norm < 1.0
        ):
            min_dist_norm = float(self._cfg("treasure_approach_reward", "min_dist_norm", 0.05))
            dist_for_force = max(min_dist_norm, cur_min_treasure_dist_norm)
            force_reward = float(self._cfg("treasure_approach_reward", "gravity_coef", 0.0015)) / (
                dist_for_force * dist_for_force
            )
            treasure_approach_reward = float(
                np.clip(
                    force_reward,
                    float(self._cfg("treasure_approach_reward", "min_reward", 0.001)),
                    float(self._cfg("treasure_approach_reward", "max_reward", 0.1)),
                )
            )

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

        # 9. 【稀疏】撞墙 / 无效移动惩罚 —— 如果这一动作之后几乎没有产生有效位移，
        #    说明走到了墙上或者做了无效动作，就触发惩罚
        wall_collision_penalty = 0.0
        if step_displacement is not None:
            # 位移小于0.5格视为无效移动（正常移动至少1格）
            if (
                self._cfg("wall_collision_penalty", "enable", True)
                and step_displacement < float(self._global_cfg("wall_displacement_threshold", 0.5))
                and last_action >= 0
            ):
                wall_collision_penalty = float(self._cfg("wall_collision_penalty", "coef", -0.1))

        # 10. 【稀疏】闪现操作失败惩罚 —— 使用闪现动作后位移远小于期望距离，
        #     说明闪现目标位置被障碍物阻挡，闪现未能到达预期目标
        flash_fail_penalty = 0.0
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
                self.flash_escape_steps = 0
                self.flash_escape_pre_dist = float(self.last_min_monster_dist_norm)
                self.flash_escape_active = self.flash_escape_pre_dist < danger_threshold
                self.flash_success_blocked = False
                self.flash_survival_decay = 0.0

                # 启动闪现滥用检测计数（用于10步内被抓惩罚）
                self.flash_recent_steps = 0

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

        # 14.【稀疏】探索事件奖励 —— 将“开图”切换为三类首次事件：
        #     1) 首次发现宝箱区域
        #     2) 首次到达新连通区域
        #     3) 首次进入远离历史轨迹区域
        exploration_reward = 0.0
        exploration_treasure_region_reward = 0.0
        exploration_connected_region_reward = 0.0
        exploration_far_traj_reward = 0.0

        if exploration_events is None:
            exploration_events = {
                "first_discover_treasure_region": 0.0,
                "first_arrive_new_connected_region": 0.0,
                "first_enter_far_trajectory_region": 0.0,
            }

        if self._cfg("exploration_reward", "enable", True) and self.birth_step_counter >= self.BIRTH_PROTECTION_STEPS:
            exploration_treasure_region_reward = 0.20 * float(
                exploration_events.get("first_discover_treasure_region", 0.0)
            )
            exploration_connected_region_reward = 0.12 * float(
                exploration_events.get("first_arrive_new_connected_region", 0.0)
            )
            exploration_far_traj_reward = 0.08 * float(
                exploration_events.get("first_enter_far_trajectory_region", 0.0)
            )
            exploration_reward = (
                exploration_treasure_region_reward
                + exploration_connected_region_reward
                + exploration_far_traj_reward
            )

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

        # ========== 课程训练阶段控制（当前仅实现Stage 1）==========
        # Stage 1：只保留生存、危险规避、怪物距离 shaping、撞墙惩罚
        if not self._is_stage_enabled("treasure_reward"):
            treasure_reward = 0.0
        if not self._is_stage_enabled("speed_buff_reward"):
            speed_buff_reward = 0.0
        if not self._is_stage_enabled("speed_buff_approach_reward"):
            speed_buff_approach_reward = 0.0
        if not self._is_stage_enabled("treasure_approach_reward"):
            treasure_approach_reward = 0.0
        if not self._is_stage_enabled("late_survive_reward"):
            late_survive_reward = 0.0
        if not self._is_stage_enabled("flash_fail_penalty"):
            flash_fail_penalty = 0.0
        if not self._is_stage_enabled("flash_escape_reward"):
            flash_escape_reward = 0.0
            flash_survival_reward = 0.0
        if not self._is_stage_enabled("speed_buff_escape_reward"):
            speed_buff_escape_reward = 0.0
        if not self._is_stage_enabled("safe_zone_reward"):
            safe_zone_reward = 0.0
        if not self._is_stage_enabled("exploration_reward"):
            exploration_reward = 0.0
            exploration_treasure_region_reward = 0.0
            exploration_connected_region_reward = 0.0
            exploration_far_traj_reward = 0.0
        if not self._is_stage_enabled("centroid_away_reward"):
            centroid_away_reward = 0.0
        if not self._is_stage_enabled("idle_wander_penalty"):
            idle_wander_penalty = 0.0
        if not self._is_stage_enabled("flash_abuse_penalty"):
            flash_abuse_penalty = 0.0
        if not self._is_stage_enabled("dead_end_penalty"):
            dead_end_penalty = 0.0

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
            "exploration_reward": exploration_reward,
            "exploration_treasure_region_reward": exploration_treasure_region_reward,
            "exploration_connected_region_reward": exploration_connected_region_reward,
            "exploration_far_traj_reward": exploration_far_traj_reward,
            "curriculum_stage": float(self.curriculum_stage),
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
        dist_norm, _, _ = self._compute_nearest_organ_distance(organs, hero_pos, 1)
        return dist_norm

    def _region_id_from_pos(self, pos, cell_size=8):
        """Map a global position to a coarse region id.

        用粗粒度网格对全局空间分桶，便于做“首次到达/首次发现”类稀疏奖励。
        """
        gx = int(pos["x"]) // int(cell_size)
        gz = int(pos["z"]) // int(cell_size)
        return (gx, gz)

    def _get_local_connected_region_anchor(self, map_info):
        """Get a coarse anchor for the current local connected component.

        在局部地图上从英雄当前位置做4邻域BFS，
        返回当前连通区域的粗粒度锚点，用于“首次进入新连通区域”事件。
        """
        if map_info is None:
            return None

        rows = len(map_info)
        cols = len(map_info[0]) if rows > 0 else 0
        if rows == 0 or cols == 0:
            return None

        cr = rows // 2
        cc = cols // 2
        if map_info[cr][cc] == 0:
            return None

        visited = set()
        q = deque()
        q.append((cr, cc))
        visited.add((cr, cc))
        dirs4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while q:
            r, c = q.popleft()
            for dr, dc in dirs4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and map_info[nr][nc] != 0:
                    if (nr, nc) not in visited:
                        visited.add((nr, nc))
                        q.append((nr, nc))

        if not visited:
            return None

        rs = [p[0] for p in visited]
        cs = [p[1] for p in visited]
        anchor_r = (min(rs) + max(rs)) // 2
        anchor_c = (min(cs) + max(cs)) // 2
        return (int(anchor_r // 2), int(anchor_c // 2))

    def _compute_exploration_events(self, hero_pos, organs, map_info):
        """Compute sparse exploration events.

        事件定义：
        1) 首次发现宝箱区域
        2) 首次进入新的局部连通区域
        3) 首次进入远离历史轨迹质心的区域
        """
        events = {
            "first_discover_treasure_region": 0.0,
            "first_arrive_new_connected_region": 0.0,
            "first_enter_far_trajectory_region": 0.0,
        }

        # 1) 首次发现宝箱区域：只对当前附近/视野内的宝箱做首次区域奖励
        for organ in organs:
            if organ.get("sub_type") != 1 or int(organ.get("status", 0)) != 1:
                continue
            pos = organ.get("pos", None)
            if pos is None:
                continue
            dist = np.sqrt(
                (float(hero_pos["x"]) - float(pos["x"])) ** 2 +
                (float(hero_pos["z"]) - float(pos["z"])) ** 2
            )
            if dist <= 12.0:
                rid = self._region_id_from_pos(pos, cell_size=8)
                if rid not in self.discovered_treasure_regions:
                    self.discovered_treasure_regions.add(rid)
                    events["first_discover_treasure_region"] = 1.0
                    break

        # 2) 首次进入新的局部连通区域
        region_anchor = self._get_local_connected_region_anchor(map_info)
        if region_anchor is not None and region_anchor not in self.visited_region_anchors:
            self.visited_region_anchors.add(region_anchor)
            events["first_arrive_new_connected_region"] = 1.0

        # 3) 首次进入远离历史轨迹区域
        if len(self.trajectory_buffer) >= 5:
            coords = np.array(self.trajectory_buffer, dtype=np.float32)
            centroid = np.mean(coords, axis=0)
            dist_to_centroid = float(np.sqrt(
                (float(hero_pos["x"]) - centroid[0]) ** 2 +
                (float(hero_pos["z"]) - centroid[1]) ** 2
            ))
            far_threshold = 8.0
            if dist_to_centroid >= far_threshold:
                rid = self._region_id_from_pos(hero_pos, cell_size=8)
                if rid not in self.visited_far_trajectory_regions:
                    self.visited_far_trajectory_regions.add(rid)
                    events["first_enter_far_trajectory_region"] = 1.0

        return events

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
            new_cells_count (int): 本帧新增的已观测格子数（当前仅用于地图记忆，不再直接用于奖励）
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

        # Treasure slots (10 x 13) / 宝箱槽位（10个对象，不足补-1）
        treasure_slots = self._collect_organ_slots(
            organs=organs,
            hero_pos=hero_pos,
            target_sub_type=1,
            slot_num=10,
        )

        # Speed-buff slots (2 x 13) / 加速buff槽位（2个对象，不足补-1）
        speed_buff_slots = self._collect_organ_slots(
            organs=organs,
            hero_pos=hero_pos,
            target_sub_type=2,
            slot_num=2,
        )

        # Stage 1 仅训练生存 / 避险 / 避撞墙，
        # 因此这里显式屏蔽 treasure / buff 相关槽位，避免策略网络被资源导向信息牵引。
        if self._is_survival_only_stage():
            treasure_slots = [np.full(13, -1.0, dtype=np.float32) for _ in range(10)]
            speed_buff_slots = [np.full(13, -1.0, dtype=np.float32) for _ in range(2)]

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
        MAP_VIEW_SIZE = 21
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

        # 保存应用 move_mask 之后的合法动作，供后续兜底使用。
        legal_after_move_mask = legal_action.copy()

        # 资源直达优先（宝箱 / buff）只在后续阶段启用。
        # Stage 1 明确关闭该逻辑，避免“只学活下来”阶段被资源动作偏置干扰。
        if not self._is_survival_only_stage():
            # 在已有合法mask上做宝箱检测：
            # 1) 若动作在一步内可摸到宝箱，则其视为“可执行吃宝箱动作”。
            # 2) 加速状态下，若第1步就能摸到宝箱，则该方向强制合法（即使第2步撞墙）。
            # 3) 若存在“可执行吃宝箱动作”，则0-7只保留这些动作。
            move_step = int(self._global_cfg("buff_move_step", 2)) if has_speed_buff else int(
                self._global_cfg("normal_move_step", 1)
            )
            treasure_touch_steps = self._find_target_touch_steps(hero_pos, organs, move_step, target_sub_type=1)
            speed_buff_touch_steps = self._find_target_touch_steps(hero_pos, organs, move_step, target_sub_type=2)
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
            # 分层mask回退策略（严格模式）：
            # 1) 优先保留经过 move_mask 过滤后的合法移动；
            # 2) 若 0-7 全部不可走，则只放开环境原始合法的非移动动作（如闪现）；
            # 3) 只有在所有动作都被清空时，才退回环境原始合法动作作为最终兜底。
            if sum(legal_after_move_mask[:8]) > 0:
                legal_action = legal_after_move_mask
            elif sum(raw_legal_action[8:]) > 0:
                legal_action = [0] * 16
                for i in range(8, 16):
                    legal_action[i] = int(raw_legal_action[i])
            elif sum(raw_legal_action) > 0:
                legal_action = [int(x) for x in raw_legal_action]
            else:
                legal_action = [0] * 16

        # Progress features (2D) / 进度特征
        step_norm = _norm(self.step_no, self.max_step)
        survival_ratio = step_norm
        progress_feat = np.array([step_norm, survival_ratio], dtype=np.float32)

        # do1
        # Candidate action features (8 x 7 = 56D) / 候选移动动作特征
        candidate_action_feat = self._build_candidate_action_features(
            hero_pos=hero_pos,
            monsters=monsters,
            organs=organs,
            map_info=map_info,
            has_speed_buff=has_speed_buff,
        )

        # Layered mask features (8 x 4 = 32D) / 分层mask软风险-收益特征
        action_risk_benefit_feat = self._build_action_risk_benefit_features(
            hero_pos=hero_pos,
            monsters=monsters,
            organs=organs,
            map_info=map_info,
            has_speed_buff=has_speed_buff,
            move_mask=move_mask,
        )

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
                candidate_action_feat,
                action_risk_benefit_feat,
            ]
        )

        # ====== 地图记忆：更新全局探索图 ======
        _ = self._update_explored_map(hero_pos, map_info)

        # ====== 高层探索事件 ======
        exploration_events = self._compute_exploration_events(
            hero_pos=hero_pos,
            organs=organs,
            map_info=map_info,
        )

        # ====== 计算完整奖励（调用新的多分量奖励函数）======
        reward, reward_info = self._compute_rewards(
            frame_state, env_info, hero_pos, monster_feats, last_action,
            map_info=map_info,
            exploration_events=exploration_events,
            terminated=terminated,
            truncated=truncated,
        )

        return feature, legal_action, [reward]
