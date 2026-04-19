
#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: OpenAI (based on user's requested refactor target)

Feature preprocessor and reward design for Gorge Chase PPO.

本版本的核心改动：
1. 维护 episode 内已探索地图：
   - 1 表示可通行
   - 0 表示障碍物
   - -1 表示未知
2. 提供统一的沿路距离（BFS）接口：
   - 视野内：在当前 21x21 局部地图上计算
   - 已探索：在 episode 内的已探索地图上计算
   - 未知：不可计算，返回 None
3. 宝箱 / buff 特征改成“前7维有效”：
   [sub_type, status, rel_x, rel_z, euclid_bucket_id, rel_direction, visible]
   为兼容当前仓库 Config.FEATURES 仍为 12 个 13 维槽位的情况，
   若 Config 中槽位维度仍是 13，则后 6 维自动补 0。
4. 所有与“距离 shaping”有关的奖励，都改成基于“沿路距离变化量”：
   - 远离怪物：正奖励
   - 靠近怪物：负奖励
   - 接近宝箱 / buff：正奖励
   - 远离宝箱 / buff：负奖励
   并严格区分 visible / explored / unknown 三种情况。

说明：
- 这是一个“可直接覆盖”的 preprocessor.py。
- 若你后续同步把 conf.py 里 12 个对象槽位从 13 改成 7，
  本文件会自动输出严格的 7 维对象槽；否则保持兼容输出 13 维槽位。
"""

import os
import copy
from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from agent_ppo.conf.conf import Config

try:
    import tomllib
except ModuleNotFoundError:
    tomllib = None


MAP_SIZE = 128
LOCAL_MAP_VIEW_SIZE = 21
LOCAL_HALF = LOCAL_MAP_VIEW_SIZE // 2
UNKNOWN_CELL = -1
BLOCK_CELL = 0
PASSABLE_CELL = 1

MAX_DIST_BUCKET = 5.0
MAX_FLASH_CD = 2000.0
MAX_BUFF_DURATION = 50.0
MAX_BUFF_REFRESH = 200.0

REWARD_CONF_FILE = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "conf", "reward_conf.toml")
)


def _norm(v: float, v_max: float, v_min: float = 0.0) -> float:
    """Normalize value into [0, 1]."""
    try:
        v = float(v)
    except Exception:
        v = 0.0
    v = float(np.clip(v, v_min, v_max))
    if (v_max - v_min) <= 1e-6:
        return 0.0
    return (v - v_min) / (v_max - v_min)


def _clip_delta(v: float, lo: float = -1.0, hi: float = 1.0) -> float:
    """Clip a delta reward to a stable range."""
    return float(np.clip(float(v), lo, hi))


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return int(default)


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _euclid(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    ax, az = _safe_float(a.get("x", 0.0)), _safe_float(a.get("z", 0.0))
    bx, bz = _safe_float(b.get("x", 0.0)), _safe_float(b.get("z", 0.0))
    return float(np.sqrt((ax - bx) ** 2 + (az - bz) ** 2))


def _bucketize_distance(dist: float) -> int:
    """
    将欧氏距离映射到 0~5 的桶编号。
    这里使用较保守的局部距离分桶；只用于你要求的对象特征，
    不用于 reward / action feature 的沿路距离计算。
    """
    if dist <= 1.0:
        return 0
    if dist <= 3.0:
        return 1
    if dist <= 5.0:
        return 2
    if dist <= 7.0:
        return 3
    if dist <= 10.0:
        return 4
    return 5


def _relative_direction_id(hero_pos: Dict[str, Any], target_pos: Dict[str, Any]) -> int:
    """
    物体相对英雄的方位编号：
    0=重叠，1=东，2=东北，3=北，4=西北，5=西，6=西南，7=南，8=东南
    """
    dx = _safe_float(target_pos.get("x", 0.0)) - _safe_float(hero_pos.get("x", 0.0))
    dz = _safe_float(target_pos.get("z", 0.0)) - _safe_float(hero_pos.get("z", 0.0))
    if abs(dx) < 1e-6 and abs(dz) < 1e-6:
        return 0

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
    return int(direction_map.get((horizontal, vertical), 0))


class Preprocessor:
    def __init__(self):
        self.reward_cfg = self._default_reward_config()
        self._load_reward_config()

        # 当前仓库 conf.py 里仍是 12 个 13 维槽位；若你后续把 conf 改成 7，则这里会自动切换。
        self.organ_slot_dim = 13
        try:
            feature_list = list(getattr(Config, "FEATURES", []))
            organ_dims = feature_list[3:15]  # 10 treasure + 2 buff
            if organ_dims and all(int(d) == 7 for d in organ_dims):
                self.organ_slot_dim = 7
        except Exception:
            self.organ_slot_dim = 13

        self.reset()

    # ---------------------------------------------------------------------
    # 基础配置与状态管理
    # ---------------------------------------------------------------------
    def _default_reward_config(self) -> Dict[str, Dict[str, Any]]:
        return {
            "global": {
                "normal_move_step": 1,
                "buff_move_step": 2,
                "danger_threshold_steps": 6.0,
                "flash_step": 4,
            },
            "survive_reward": {"enable": True, "coef": 0.02},
            "treasure_reward": {"enable": True, "coef": 1.0},
            "speed_buff_reward": {"enable": True, "coef": 0.25},
            "treasure_approach_reward": {"enable": True, "coef": 0.08},
            "speed_buff_approach_reward": {"enable": True, "coef": 0.05},
            "monster_dist_shaping": {"enable": True, "coef": 0.12},
            "wall_collision_penalty": {"enable": True, "coef": -0.05},
            "exploration_reward": {"enable": True, "coef_per_cell": 0.0005},
            "idle_wander_penalty": {"enable": True, "coef": -0.02},
        }

    def _deep_update(self, base: Dict[str, Any], updates: Dict[str, Any]) -> None:
        for key, val in updates.items():
            if isinstance(val, dict) and isinstance(base.get(key), dict):
                self._deep_update(base[key], val)
            else:
                base[key] = val

    def _load_reward_config(self) -> None:
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
            pass
        self.reward_cfg = cfg

    def _cfg(self, section: str, key: str, default: Any) -> Any:
        sec = self.reward_cfg.get(section, {})
        if not isinstance(sec, dict):
            return default
        return sec.get(key, default)

    def _global_cfg(self, key: str, default: Any) -> Any:
        sec = self.reward_cfg.get("global", {})
        if not isinstance(sec, dict):
            return default
        return sec.get(key, default)

    def set_curriculum_stage(self, stage: int) -> None:
        try:
            stage = int(stage)
        except Exception:
            stage = 1
        self.curriculum_stage = max(1, min(4, stage))

    def get_curriculum_stage(self) -> int:
        return int(getattr(self, "curriculum_stage", 1))

    def get_curriculum_stage_name(self) -> str:
        mapping = {
            1: "survival_base",
            2: "explore_and_stabilize",
            3: "safe_resource_acquisition",
            4: "full_game_and_skill_refine",
        }
        return mapping.get(self.get_curriculum_stage(), "unknown")

    def get_reward_term_coef(self, section: str, key: str = "coef", default: float = 1.0) -> float:
        return float(self._cfg(section, key, default))

    def _is_stage_enabled(self, name: str) -> bool:
        stage = self.get_curriculum_stage()
        stage_reward_map = {
            1: {"survive_reward", "monster_dist_shaping", "wall_collision_penalty"},
            2: {
                "survive_reward",
                "monster_dist_shaping",
                "wall_collision_penalty",
                "exploration_reward",
                "idle_wander_penalty",
            },
            3: {
                "survive_reward",
                "monster_dist_shaping",
                "wall_collision_penalty",
                "exploration_reward",
                "idle_wander_penalty",
                "treasure_reward",
                "speed_buff_reward",
                "treasure_approach_reward",
                "speed_buff_approach_reward",
            },
            4: {
                "survive_reward",
                "monster_dist_shaping",
                "wall_collision_penalty",
                "exploration_reward",
                "idle_wander_penalty",
                "treasure_reward",
                "speed_buff_reward",
                "treasure_approach_reward",
                "speed_buff_approach_reward",
            },
        }
        return name in stage_reward_map.get(stage, stage_reward_map[1])

    def _is_survival_only_stage(self) -> bool:
        return self.get_curriculum_stage() <= 2

    def reset(self) -> None:
        self.curriculum_stage = getattr(self, "curriculum_stage", 1)

        # 已探索地图：
        # -1 未知，0 障碍，1 可通行
        self.explored_map = np.full((MAP_SIZE, MAP_SIZE), UNKNOWN_CELL, dtype=np.int8)

        # 记录最近一次看见的器官 / 怪物位置，用于“已探索图上的沿路距离”
        self.organ_memory: Dict[Any, Dict[str, Any]] = {}
        self.monster_memory: Dict[Any, Dict[str, Any]] = {}

        self.prev_hero_pos: Optional[Dict[str, int]] = None
        self.prev_move_succeeded = True

        self.prev_nearest_monster_path_dist: Optional[int] = None
        self.prev_nearest_treasure_path_dist: Optional[int] = None
        self.prev_nearest_buff_path_dist: Optional[int] = None

        self.prev_treasure_count = 0
        self.prev_has_speed_buff = False

        self.step_count = 0
        self.last_reward_info: Dict[str, float] = {}

    # ---------------------------------------------------------------------
    # 协议解析辅助：尽量兼容不同 env_obs 结构
    # ---------------------------------------------------------------------
    def _get_from_paths(self, obj: Any, paths: Sequence[Sequence[str]], default: Any = None) -> Any:
        for path in paths:
            cur = obj
            ok = True
            for key in path:
                if isinstance(cur, dict) and key in cur:
                    cur = cur[key]
                else:
                    ok = False
                    break
            if ok:
                return cur
        return default

    def _extract_hero(self, env_obs: Dict[str, Any]) -> Dict[str, Any]:
        hero = self._get_from_paths(
            env_obs,
            [
                ("hero",),
                ("player",),
                ("player_state",),
                ("frame_state", "hero"),
                ("frame_state", "player"),
                ("frame_state", "player_state"),
                ("observation", "hero"),
                ("obs", "hero"),
            ],
            default=None,
        )
        if hero is None:
            # 兜底：找到第一个带 pos 的 dict。
            stack = [env_obs]
            while stack:
                cur = stack.pop()
                if isinstance(cur, dict):
                    if isinstance(cur.get("pos"), dict) and "x" in cur["pos"] and "z" in cur["pos"]:
                        hero = cur
                        break
                    stack.extend(cur.values())
                elif isinstance(cur, list):
                    stack.extend(cur)
        if not isinstance(hero, dict):
            hero = {}
        hero.setdefault("pos", {"x": 0, "z": 0})
        return hero

    def _extract_monsters(self, env_obs: Dict[str, Any]) -> List[Dict[str, Any]]:
        monsters = self._get_from_paths(
            env_obs,
            [
                ("monsters",),
                ("monster",),
                ("monster_states",),
                ("frame_state", "monsters"),
                ("frame_state", "monster"),
                ("frame_state", "monster_states"),
                ("observation", "monsters"),
                ("obs", "monsters"),
            ],
            default=None,
        )
        if isinstance(monsters, dict):
            monsters = [monsters]
        if not isinstance(monsters, list):
            monsters = []
        return [m for m in monsters if isinstance(m, dict)]

    def _extract_organs(self, env_obs: Dict[str, Any]) -> List[Dict[str, Any]]:
        organs = self._get_from_paths(
            env_obs,
            [
                ("organs",),
                ("organ",),
                ("organ_states",),
                ("frame_state", "organs"),
                ("frame_state", "organ"),
                ("frame_state", "organ_states"),
                ("observation", "organs"),
                ("obs", "organs"),
            ],
            default=None,
        )
        if isinstance(organs, dict):
            organs = [organs]
        if not isinstance(organs, list):
            organs = []
        return [o for o in organs if isinstance(o, dict)]

    def _extract_map_info(self, env_obs: Dict[str, Any]) -> Optional[List[List[int]]]:
        map_info = self._get_from_paths(
            env_obs,
            [
                ("map_info",),
                ("map",),
                ("local_map",),
                ("frame_state", "map_info"),
                ("frame_state", "map"),
                ("observation", "map_info"),
                ("obs", "map_info"),
            ],
            default=None,
        )
        if map_info is None:
            return None
        try:
            arr = np.array(map_info, dtype=np.int32)
            if arr.ndim != 2:
                return None
            return arr.tolist()
        except Exception:
            return None

    def _extract_env_info(self, env_obs: Dict[str, Any]) -> Dict[str, Any]:
        env_info = self._get_from_paths(
            env_obs,
            [
                ("env_info",),
                ("game_info",),
                ("frame_state", "env_info"),
                ("frame_state", "game_info"),
            ],
            default=None,
        )
        if isinstance(env_info, dict):
            return env_info
        return env_obs if isinstance(env_obs, dict) else {}

    def _extract_done_flags(self, env_obs: Dict[str, Any]) -> Tuple[bool, bool]:
        terminated = self._get_from_paths(
            env_obs,
            [("terminated",), ("done",), ("frame_state", "terminated"), ("frame_state", "done")],
            default=False,
        )
        truncated = self._get_from_paths(
            env_obs,
            [("truncated",), ("frame_state", "truncated")],
            default=False,
        )
        return bool(terminated), bool(truncated)

    def _hero_pos(self, hero: Dict[str, Any]) -> Dict[str, int]:
        pos = hero.get("pos", {}) if isinstance(hero, dict) else {}
        return {"x": _safe_int(pos.get("x", 0)), "z": _safe_int(pos.get("z", 0))}

    def _has_speed_buff(self, hero: Dict[str, Any], env_info: Dict[str, Any]) -> bool:
        keys = [
            hero.get("has_speed_buff", None),
            hero.get("speed_buff", None),
            hero.get("speed_up", None),
            hero.get("is_speed_up", None),
            hero.get("speed_duration", None),
            hero.get("buff_duration", None),
            env_info.get("has_speed_buff", None),
            env_info.get("speed_up", None),
        ]
        for v in keys:
            if v is None:
                continue
            if isinstance(v, bool):
                return bool(v)
            if _safe_float(v, 0.0) > 0:
                return True
        return False

    def _flash_cd_norm(self, hero: Dict[str, Any], env_info: Dict[str, Any]) -> float:
        keys = [
            hero.get("flash_cd", None),
            hero.get("flash_cooldown", None),
            hero.get("flash_skill_cd", None),
            env_info.get("flash_cd", None),
            env_info.get("flash_cooldown", None),
            env_info.get("flash_skill_cd", None),
        ]
        for v in keys:
            if v is None:
                continue
            return _norm(_safe_float(v, 0.0), MAX_FLASH_CD)
        return 0.0

    def _get_treasure_collected_count(self, env_info: Dict[str, Any], hero: Dict[str, Any]) -> int:
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

    # ---------------------------------------------------------------------
    # 地图、视野、记忆与 BFS
    # ---------------------------------------------------------------------
    def _is_valid_global(self, x: int, z: int) -> bool:
        return 0 <= x < MAP_SIZE and 0 <= z < MAP_SIZE

    def _is_visible_in_current_view(self, hero_pos: Dict[str, int], target_pos: Dict[str, int], map_info: Optional[List[List[int]]]) -> bool:
        if map_info is None:
            return False
        dx = int(target_pos["x"]) - int(hero_pos["x"])
        dz = int(target_pos["z"]) - int(hero_pos["z"])
        if abs(dx) > LOCAL_HALF or abs(dz) > LOCAL_HALF:
            return False
        r = LOCAL_HALF + dz
        c = LOCAL_HALF + dx
        if not (0 <= r < len(map_info) and 0 <= c < len(map_info[0])):
            return False
        return True

    def _visible_local_rc(self, hero_pos: Dict[str, int], target_pos: Dict[str, int]) -> Tuple[int, int]:
        dx = int(target_pos["x"]) - int(hero_pos["x"])
        dz = int(target_pos["z"]) - int(hero_pos["z"])
        return LOCAL_HALF + dz, LOCAL_HALF + dx

    def _update_explored_map(self, hero_pos: Dict[str, int], map_info: Optional[List[List[int]]]) -> int:
        """
        用当前 21x21 视野更新 episode 级已探索地图。
        规则：
        - 1: 可通行
        - 0: 障碍
        - -1: 未知
        返回本帧首次被观测到的格子数。
        """
        if map_info is None:
            return 0
        arr = np.array(map_info, dtype=np.int32)
        if arr.ndim != 2:
            return 0

        rows, cols = arr.shape
        center_r = rows // 2
        center_c = cols // 2
        hx, hz = int(hero_pos["x"]), int(hero_pos["z"])
        new_count = 0

        for r in range(rows):
            for c in range(cols):
                gx = hx + (c - center_c)
                gz = hz + (r - center_r)
                if not self._is_valid_global(gx, gz):
                    continue
                cell_value = PASSABLE_CELL if int(arr[r, c]) != 0 else BLOCK_CELL
                if int(self.explored_map[gz, gx]) == UNKNOWN_CELL:
                    new_count += 1
                self.explored_map[gz, gx] = cell_value
        return int(new_count)

    def _update_object_memory(
        self,
        hero_pos: Dict[str, int],
        organs: List[Dict[str, Any]],
        monsters: List[Dict[str, Any]],
        map_info: Optional[List[List[int]]],
    ) -> None:
        for idx, organ in enumerate(organs):
            pos = organ.get("pos", {})
            ox, oz = _safe_int(pos.get("x", 0)), _safe_int(pos.get("z", 0))
            if not self._is_valid_global(ox, oz):
                continue
            key = organ.get("config_id", None)
            if key is None:
                key = (int(organ.get("sub_type", 0)), idx)
            self.organ_memory[key] = {
                "sub_type": _safe_int(organ.get("sub_type", 0)),
                "status": _safe_int(organ.get("status", 0)),
                "pos": {"x": ox, "z": oz},
                "last_seen_step": self.step_count,
                "visible": self._is_visible_in_current_view(hero_pos, {"x": ox, "z": oz}, map_info),
            }

        for idx, monster in enumerate(monsters):
            pos = monster.get("pos", {})
            mx, mz = _safe_int(pos.get("x", 0)), _safe_int(pos.get("z", 0))
            if not self._is_valid_global(mx, mz):
                continue
            if mx == 0 and mz == 0:
                continue
            key = monster.get("config_id", None)
            if key is None:
                key = idx
            self.monster_memory[key] = {
                "pos": {"x": mx, "z": mz},
                "speed": _safe_float(
                    monster.get("speed", monster.get("move_speed", monster.get("velocity", 0.0))),
                    0.0,
                ),
                "last_seen_step": self.step_count,
                "visible": self._is_visible_in_current_view(hero_pos, {"x": mx, "z": mz}, map_info),
            }

    def _bfs_local_distance(
        self,
        map_info: Optional[List[List[int]]],
        start_rc: Tuple[int, int],
        goal_rc: Tuple[int, int],
    ) -> Optional[int]:
        """
        在局部地图上计算 8 邻接最短路距离。

        说明：
        - 1 表示可通行，0 表示障碍
        - 返回值为最短步数；不可达则返回 None
        - 该版本使用 8 邻接：上下左右 + 4 个对角方向
        """
        if map_info is None:
            return None

        arr = np.array(map_info, dtype=np.int32)
        if arr.ndim != 2 or arr.size == 0:
            return None

        rows, cols = arr.shape
        sr, sc = start_rc
        gr, gc = goal_rc

        if not (0 <= sr < rows and 0 <= sc < cols and 0 <= gr < rows and 0 <= gc < cols):
            return None

        # 起点或终点不可通行，直接返回不可达
        if int(arr[sr, sc]) == 0 or int(arr[gr, gc]) == 0:
            return None

        q = deque([(sr, sc)])
        dist = np.full((rows, cols), -1, dtype=np.int32)
        dist[sr, sc] = 0

        # 8邻接方向
        dirs8 = [
            (-1, 0), (1, 0), (0, -1), (0, 1),   # 上下左右
            (-1, -1), (-1, 1), (1, -1), (1, 1)  # 四个对角
        ]

        while q:
            r, c = q.popleft()
            if r == gr and c == gc:
                return int(dist[r, c])

            for dr, dc in dirs8:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if dist[nr, nc] < 0 and int(arr[nr, nc]) != 0:
                        dist[nr, nc] = dist[r, c] + 1
                        q.append((nr, nc))

        return None

    def _bfs_explored_distance(
        self,
        start_pos: Dict[str, int],
        goal_pos: Dict[str, int],
    ) -> Optional[int]:
        """
        在已探索地图上计算 8 邻接最短路距离。

        说明：
        - PASSABLE_CELL 表示可通行
        - 返回值为最短步数；不可达则返回 None
        - 使用 8 邻接：上下左右 + 4 个对角方向
        """
        sx, sz = int(start_pos["x"]), int(start_pos["z"])
        gx, gz = int(goal_pos["x"]), int(goal_pos["z"])

        if not (self._is_valid_global(sx, sz) and self._is_valid_global(gx, gz)):
            return None

        if int(self.explored_map[sz, sx]) != PASSABLE_CELL:
            return None
        if int(self.explored_map[gz, gx]) != PASSABLE_CELL:
            return None

        q = deque([(sx, sz)])
        dist = np.full((MAP_SIZE, MAP_SIZE), -1, dtype=np.int32)
        dist[sz, sx] = 0

        # 8邻接方向
        dirs8 = [
            (-1, 0), (1, 0), (0, -1), (0, 1),   # 上下左右
            (-1, -1), (-1, 1), (1, -1), (1, 1)  # 四个对角
        ]

        while q:
            x, z = q.popleft()
            if x == gx and z == gz:
                return int(dist[z, x])

            base = int(dist[z, x])
            for dx, dz in dirs8:
                nx, nz = x + dx, z + dz

                if not self._is_valid_global(nx, nz):
                    continue
                if dist[nz, nx] >= 0:
                    continue
                if int(self.explored_map[nz, nx]) != PASSABLE_CELL:
                    continue

                dist[nz, nx] = base + 1
                q.append((nx, nz))

        return None

    def compute_path_distance(
        self,
        start_pos: Dict[str, int],
        target_pos: Dict[str, int],
        map_info: Optional[List[List[int]]] = None,
    ) -> Dict[str, Any]:
        """
        统一沿路距离接口，供其他模块调用。

        返回：
        {
            "distance": Optional[int],   # 可计算时返回步数，否则为 None
            "mode": "visible" | "explored" | "unknown",
        }

        规则：
        1) 若 target 在当前 21x21 视野内，则优先在局部视野图上做 BFS；
        2) 若局部不可算，则尝试在已探索地图上做 BFS；
        3) 未知区域不可计算，返回 None / unknown。
        """
        if map_info is not None and self._is_visible_in_current_view(start_pos, target_pos, map_info):
            start_rc = (LOCAL_HALF, LOCAL_HALF)
            goal_rc = self._visible_local_rc(start_pos, target_pos)
            dist_local = self._bfs_local_distance(map_info, start_rc, goal_rc)
            if dist_local is not None:
                return {"distance": int(dist_local), "mode": "visible"}

        dist_explored = self._bfs_explored_distance(start_pos, target_pos)
        if dist_explored is not None:
            return {"distance": int(dist_explored), "mode": "explored"}

        return {"distance": None, "mode": "unknown"}

    def _nearest_known_target_distance(
        self,
        hero_pos: Dict[str, int],
        targets: List[Dict[str, Any]],
        map_info: Optional[List[List[int]]],
        target_sub_type: Optional[int] = None,
    ) -> Tuple[Optional[int], Optional[Dict[str, Any]], str]:
        """
        返回最近目标的沿路距离、目标对象、以及 distance mode。
        优先使用当前视野里的目标；若视野外但记忆点位在已探索图内，则用 explored BFS。
        """
        best_dist: Optional[int] = None
        best_obj: Optional[Dict[str, Any]] = None
        best_mode = "unknown"

        for obj in targets:
            if target_sub_type is not None and _safe_int(obj.get("sub_type", 0)) != int(target_sub_type):
                continue
            if int(obj.get("status", 1)) != 1:
                continue
            pos = obj.get("pos", {})
            tx, tz = _safe_int(pos.get("x", 0)), _safe_int(pos.get("z", 0))
            if not self._is_valid_global(tx, tz):
                continue
            dist_info = self.compute_path_distance(hero_pos, {"x": tx, "z": tz}, map_info)
            d = dist_info["distance"]
            if d is None:
                continue
            if best_dist is None or d < best_dist:
                best_dist = d
                best_obj = obj
                best_mode = str(dist_info["mode"])

        return best_dist, best_obj, best_mode

    def _chebyshev_distance(
        self,
        a: Dict[str, int],
        b: Dict[str, int],
    ) -> int:
       """
       8邻接网格下的几何步数距离（无障碍时的最短步数）。
       当怪物在当前视野内，但局部/已探索 BFS 都不可计算时，
       用它作为怪物距离的兜底值，避免 monster_dist_shaping 长期失效。
       """
       ax, az = int(a["x"]), int(a["z"])
       bx, bz = int(b["x"]), int(b["z"])
       return int(max(abs(ax - bx), abs(az - bz)))

    def compute_monster_distance(
            self,
            start_pos: Dict[str, int],
            target_pos: Dict[str, int],
            map_info: Optional[List[List[int]]] = None,
        ) -> Dict[str, Any]:
        """
        怪物专用距离接口。

        返回：
        {
            "distance": Optional[int],
            "mode": "visible_path" | "explored_path" | "visible_fallback" | "unknown",
        }

        规则：
        1) 先用相对坐标判断怪物是否在当前21x21视野内；
        2) 若在视野内，优先在局部视野图上做 BFS；
        3) 若局部 BFS 不可达，则尝试在已探索地图上做 BFS；
        4) 若两种 BFS 都失败，但怪物仍在当前视野内，则使用 8邻接几何距离 作为兜底；
        5) 否则返回 None / unknown。
        """
        is_visible = self._is_in_local_window(start_pos, target_pos, LOCAL_HALF)

        dx = int(target_pos["x"]) - int(start_pos["x"])
        dz = int(target_pos["z"]) - int(start_pos["z"])
        print(
            "[monster_distance_debug]",
            "start=", start_pos,
            "target=", target_pos,
            "dx=", dx,
            "dz=", dz,
            "is_visible=", self._is_in_local_window(start_pos, target_pos, LOCAL_HALF),
        )

        # 1) 当前视野内局部 BFS
        if is_visible and map_info is not None:
            start_rc = (LOCAL_HALF, LOCAL_HALF)
            goal_rc = self._visible_local_rc(start_pos, target_pos)

            # goal_rc 必须落在当前局部图范围内
            if goal_rc is not None:
                dist_local = self._bfs_local_distance(map_info, start_rc, goal_rc)
                if dist_local is not None:
                    return {"distance": int(dist_local), "mode": "visible_path"}

        # 2) 已探索地图 BFS
        dist_explored = self._bfs_explored_distance(start_pos, target_pos)
        if dist_explored is not None:
            return {"distance": int(dist_explored), "mode": "explored_path"}

        # 3) 视野内兜底：8邻接几何距离
        if is_visible:
            fallback_dist = self._chebyshev_distance(start_pos, target_pos)
            return {"distance": int(fallback_dist), "mode": "visible_fallback"}

        # 4) 不可计算
        return {"distance": None, "mode": "unknown"}

    def _nearest_known_monster_distance(
        self,
        hero_pos: Dict[str, int],
        monsters: List[Dict[str, Any]],
        map_info: Optional[List[List[int]]],
    ) -> Tuple[Optional[int], Optional[Dict[str, Any]], str]:
        """
        返回最近怪物的距离、对应怪物对象、以及距离模式。

        距离优先级：
        1) 当前视野内局部 BFS 距离（visible_path）
        2) 已探索地图 BFS 距离（explored_path）
        3) 当前视野内 8邻接几何兜底距离（visible_fallback）
        4) 都不可计算则 unknown
        """
        best_dist: Optional[int] = None
        best_obj: Optional[Dict[str, Any]] = None
        best_mode = "unknown"

        for m in monsters:
            pos = m.get("pos", {})
            tx, tz = _safe_int(pos.get("x", 0)), _safe_int(pos.get("z", 0))
            if not self._is_valid_global(tx, tz):
                continue

            dist_info = self.compute_monster_distance(
                hero_pos,
                {"x": tx, "z": tz},
                map_info,
            )
            d = dist_info["distance"]
            if d is None:
                continue

            if best_dist is None or d < best_dist:
                best_dist = d
                best_obj = m
                best_mode = str(dist_info["mode"])

        return best_dist, best_obj, best_mode

    def _is_in_local_window(
            self,
            center_pos: Dict[str, int],
            target_pos: Dict[str, int],
            half_size: int = LOCAL_HALF,
        ) -> bool:
        """
        判断 target 是否落在以 center_pos 为中心的局部窗口内。
        默认窗口大小为 21x21，因此 half_size=10。
        """
        cx, cz = int(center_pos["x"]), int(center_pos["z"])
        tx, tz = int(target_pos["x"]), int(target_pos["z"])
        dx = tx - cx
        dz = tz - cz
        return abs(dx) <= half_size and abs(dz) <= half_size

    # ---------------------------------------------------------------------
    # 对象筛选与特征构建
    # ---------------------------------------------------------------------
    def _filter_visible_organs(
        self,
        organs: List[Dict[str, Any]],
        hero_pos: Dict[str, int],
        map_info: Optional[List[List[int]]],
        target_sub_type: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        out = []
        for organ in organs:
            if target_sub_type is not None and _safe_int(organ.get("sub_type", 0)) != int(target_sub_type):
                continue
            pos = organ.get("pos", {})
            ox, oz = _safe_int(pos.get("x", 0)), _safe_int(pos.get("z", 0))
            if not self._is_valid_global(ox, oz):
                continue
            if self._is_visible_in_current_view(hero_pos, {"x": ox, "z": oz}, map_info):
                out.append(organ)
        return out

    def _collect_memory_targets(self, target_sub_type: int) -> List[Dict[str, Any]]:
        out = []
        for item in self.organ_memory.values():
            if int(item.get("sub_type", 0)) != int(target_sub_type):
                continue
            out.append(
                {
                    "sub_type": int(item.get("sub_type", 0)),
                    "status": int(item.get("status", 0)),
                    "pos": {
                        "x": int(item.get("pos", {}).get("x", 0)),
                        "z": int(item.get("pos", {}).get("z", 0)),
                    },
                }
            )
        return out

    def _encode_object_slot_7d(
        self,
        hero_pos: Dict[str, int],
        obj: Optional[Dict[str, Any]],
        visible: bool,
    ) -> np.ndarray:
        """
        用户要求的 7 维对象特征：
        [物件类型, 状态, 相对x, 相对z, 欧氏距离桶编号, 相对方位, 是否可见]
        """
        feat = np.zeros(7, dtype=np.float32)
        if obj is None or not visible:
            return feat

        pos = obj.get("pos", {})
        ox, oz = _safe_int(pos.get("x", 0)), _safe_int(pos.get("z", 0))
        hx, hz = int(hero_pos["x"]), int(hero_pos["z"])
        dx = ox - hx
        dz = oz - hz
        euclid_dist = _euclid({"x": hx, "z": hz}, {"x": ox, "z": oz})
        bucket_id = _bucketize_distance(euclid_dist)
        direction_id = _relative_direction_id(hero_pos, {"x": ox, "z": oz})

        feat[0] = float(_safe_int(obj.get("sub_type", 0)))
        feat[1] = float(_safe_int(obj.get("status", 0)))
        feat[2] = float(dx) / float(LOCAL_HALF)   # 相对位置归一化到 [-1, 1] 附近
        feat[3] = float(dz) / float(LOCAL_HALF)
        feat[4] = float(bucket_id)
        feat[5] = float(direction_id)
        feat[6] = 1.0
        return feat

    def _pad_object_slot(self, feat7: np.ndarray) -> np.ndarray:
        if self.organ_slot_dim <= 7:
            return feat7.astype(np.float32)
        out = np.zeros(self.organ_slot_dim, dtype=np.float32)
        out[:7] = feat7[:7]
        return out

    def _collect_organ_slots(
        self,
        organs: List[Dict[str, Any]],
        hero_pos: Dict[str, int],
        map_info: Optional[List[List[int]]],
        target_sub_type: int,
        slot_num: int,
    ) -> np.ndarray:
        """
        目标：宝箱在前、buff 在后；各自按与 agent 的沿路距离从近到远排序。
        特征只对“当前可见”物体生效；若不可见，则该槽位全 0（visible=0）。
        """
        visible_objs = self._filter_visible_organs(organs, hero_pos, map_info, target_sub_type=target_sub_type)

        sortable: List[Tuple[int, Dict[str, Any]]] = []
        for obj in visible_objs:
            pos = obj.get("pos", {})
            dist_info = self.compute_path_distance(
                hero_pos,
                {"x": _safe_int(pos.get("x", 0)), "z": _safe_int(pos.get("z", 0))},
                map_info,
            )
            if dist_info["distance"] is None:
                continue
            sortable.append((int(dist_info["distance"]), obj))

        sortable.sort(key=lambda x: x[0])
        slots: List[np.ndarray] = []

        for _, obj in sortable[:slot_num]:
            feat7 = self._encode_object_slot_7d(hero_pos, obj, visible=True)
            slots.append(self._pad_object_slot(feat7))

        while len(slots) < slot_num:
            slots.append(np.zeros(self.organ_slot_dim, dtype=np.float32))

        return np.concatenate(slots, axis=0).astype(np.float32)

    def _build_hero_feature(self, hero: Dict[str, Any], env_info: Dict[str, Any]) -> np.ndarray:
        hero_pos = self._hero_pos(hero)
        has_speed_buff = 1.0 if self._has_speed_buff(hero, env_info) else 0.0
        flash_cd_norm = self._flash_cd_norm(hero, env_info)
        return np.array(
            [
                _norm(hero_pos["x"], MAP_SIZE - 1),
                _norm(hero_pos["z"], MAP_SIZE - 1),
                has_speed_buff,
                flash_cd_norm,
            ],
            dtype=np.float32,
        )

    def _build_monster_feature(
        self,
        hero_pos: Dict[str, int],
        monster: Optional[Dict[str, Any]],
        map_info: Optional[List[List[int]]],
    ) -> np.ndarray:
        """
        保持 14 维以兼容现有 Config。
        含义：
        [visible, active, rel_x, rel_z, path_dist_norm, path_bucket_norm, dir_id_norm,
         speed_norm, mode_visible, mode_explored, mode_known, abs_x_norm, abs_z_norm, reserved]
        """
        feat = np.zeros(14, dtype=np.float32)
        if not isinstance(monster, dict):
            return feat

        pos = monster.get("pos", {})
        mx, mz = _safe_int(pos.get("x", 0)), _safe_int(pos.get("z", 0))
        if not self._is_valid_global(mx, mz):
            return feat
        if mx == 0 and mz == 0:
            return feat

        dist_info = self.compute_path_distance(hero_pos, {"x": mx, "z": mz}, map_info)
        path_dist = dist_info["distance"]
        mode = str(dist_info["mode"])
        visible = 1.0 if mode == "visible" else 0.0
        known = 1.0 if path_dist is not None else 0.0

        dx = mx - int(hero_pos["x"])
        dz = mz - int(hero_pos["z"])
        direction_id = _relative_direction_id(hero_pos, {"x": mx, "z": mz})
        speed = _safe_float(monster.get("speed", monster.get("move_speed", 0.0)), 0.0)

        feat[0] = visible
        feat[1] = 1.0
        feat[2] = float(dx) / float(MAP_SIZE)
        feat[3] = float(dz) / float(MAP_SIZE)
        feat[4] = 1.0 if path_dist is None else _norm(path_dist, MAP_SIZE * 2.0)
        feat[5] = 1.0 if path_dist is None else _norm(min(_bucketize_distance(path_dist), 5), 5.0)
        feat[6] = float(direction_id) / 8.0
        feat[7] = _norm(speed, 5.0)
        feat[8] = 1.0 if mode == "visible" else 0.0
        feat[9] = 1.0 if mode == "explored" else 0.0
        feat[10] = known
        feat[11] = _norm(mx, MAP_SIZE - 1)
        feat[12] = _norm(mz, MAP_SIZE - 1)
        feat[13] = 0.0
        return feat

    # ---------------------------------------------------------------------
    # 动作合法性、一步后果特征与软风险收益
    # ---------------------------------------------------------------------
    def _preprocess_move_action_mask(self, map_info: Optional[List[List[int]]], has_speed_buff: bool) -> List[int]:
        if map_info is None:
            return [1] * 8
        arr = np.array(map_info, dtype=np.int32)
        if arr.ndim != 2 or arr.size == 0:
            return [1] * 8

        rows, cols = arr.shape
        center_r = rows // 2
        center_c = cols // 2
        move_step = int(self._global_cfg("buff_move_step", 2)) if has_speed_buff else int(
            self._global_cfg("normal_move_step", 1)
        )
        move_step = max(1, move_step)

        dirs8 = [
            (0, 1),   # E
            (-1, 1),  # NE
            (-1, 0),  # N
            (-1, -1), # NW
            (0, -1),  # W
            (1, -1),  # SW
            (1, 0),   # S
            (1, 1),   # SE
        ]

        def is_passable(r: int, c: int) -> bool:
            return 0 <= r < rows and 0 <= c < cols and int(arr[r, c]) != 0

        mask = [1] * 8
        for i, (dr, dc) in enumerate(dirs8):
            legal = True
            for step in range(1, move_step + 1):
                rr = center_r + dr * step
                cc = center_c + dc * step
                if not is_passable(rr, cc):
                    legal = False
                    break
                # 斜向不允许切角
                if dr != 0 and dc != 0:
                    prev_r = center_r + dr * (step - 1)
                    prev_c = center_c + dc * (step - 1)
                    if not (is_passable(rr, prev_c) and is_passable(prev_r, cc)):
                        legal = False
                        break
            mask[i] = 1 if legal else 0
        return mask

    def _preprocess_flash_action_mask(self, map_info: Optional[List[List[int]]], flash_cd_norm: float) -> List[int]:
        if flash_cd_norm > 1e-6:
            return [0] * 8
        if map_info is None:
            return [1] * 8

        arr = np.array(map_info, dtype=np.int32)
        if arr.ndim != 2 or arr.size == 0:
            return [1] * 8

        rows, cols = arr.shape
        center_r = rows // 2
        center_c = cols // 2
        flash_step = int(self._global_cfg("flash_step", 4))
        dirs8 = [
            (0, 1), (-1, 1), (-1, 0), (-1, -1),
            (0, -1), (1, -1), (1, 0), (1, 1),
        ]

        def is_passable(r: int, c: int) -> bool:
            return 0 <= r < rows and 0 <= c < cols and int(arr[r, c]) != 0

        mask = [1] * 8
        for i, (dr, dc) in enumerate(dirs8):
            rr = center_r + dr * flash_step
            cc = center_c + dc * flash_step
            mask[i] = 1 if is_passable(rr, cc) else 0
        return mask

    def _simulate_next_position(
        self,
        hero_pos: Dict[str, int],
        map_info: Optional[List[List[int]]],
        action_idx: int,
        has_speed_buff: bool,
    ) -> Tuple[Dict[str, int], Optional[int], Optional[int]]:
        """
        模拟 8 个移动动作之一执行后的落点。
        若路径中途非法，则停在最后一个合法格；
        若起步就非法，则停在原地。
        """
        hx = int(hero_pos["x"])
        hz = int(hero_pos["z"])
        if map_info is None:
            return {"x": hx, "z": hz}, None, None

        arr = np.array(map_info, dtype=np.int32)
        if arr.ndim != 2 or arr.size == 0:
            return {"x": hx, "z": hz}, None, None

        rows, cols = arr.shape
        center_r = rows // 2
        center_c = cols // 2

        move_step = int(self._global_cfg("buff_move_step", 2)) if has_speed_buff else int(
            self._global_cfg("normal_move_step", 1)
        )
        move_step = max(1, move_step)

        action_delta = [
            (1, 0),   # E
            (1, -1),  # NE
            (0, -1),  # N
            (-1, -1), # NW
            (-1, 0),  # W
            (-1, 1),  # SW
            (0, 1),   # S
            (1, 1),   # SE
        ]
        dx, dz = action_delta[int(action_idx) % 8]

        def is_passable(r: int, c: int) -> bool:
            return 0 <= r < rows and 0 <= c < cols and int(arr[r, c]) != 0

        last_valid_x = hx
        last_valid_z = hz
        last_valid_r = center_r
        last_valid_c = center_c

        for step in range(1, move_step + 1):
            rr = center_r + dz * step
            cc = center_c + dx * step
            if not is_passable(rr, cc):
                break
            if dz != 0 and dx != 0:
                prev_r = center_r + dz * (step - 1)
                prev_c = center_c + dx * (step - 1)
                if not (is_passable(rr, prev_c) and is_passable(prev_r, cc)):
                    break
            last_valid_x = hx + dx * step
            last_valid_z = hz + dz * step
            last_valid_r = rr
            last_valid_c = cc

        return {"x": last_valid_x, "z": last_valid_z}, last_valid_r, last_valid_c

    def _count_free_neighbors(self, map_info: Optional[List[List[int]]], center_r: int, center_c: int, radius: int = 1) -> float:
        if map_info is None:
            return 0.0
        arr = np.array(map_info, dtype=np.int32)
        if arr.ndim != 2:
            return 0.0

        rows, cols = arr.shape
        total = 0
        free = 0
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr = center_r + dr
                cc = center_c + dc
                if 0 <= rr < rows and 0 <= cc < cols:
                    total += 1
                    if int(arr[rr, cc]) != 0:
                        free += 1
        return 0.0 if total == 0 else float(free) / float(total)

    def _extract_local_map_centered(
        self,
        map_info: Optional[List[List[int]]],
        center_r: int,
        center_c: int,
        view_size: int = 21,
    ) -> np.ndarray:
        out = np.zeros((view_size, view_size), dtype=np.int32)
        if map_info is None:
            return out
        arr = np.array(map_info, dtype=np.int32)
        if arr.ndim != 2 or arr.size == 0:
            return out

        rows, cols = arr.shape
        half = view_size // 2
        for r in range(view_size):
            for c in range(view_size):
                src_r = center_r - half + r
                src_c = center_c - half + c
                if 0 <= src_r < rows and 0 <= src_c < cols:
                    out[r, c] = 1 if int(arr[src_r, src_c]) != 0 else 0
        return out

    def _safe_quadrant_id(self, hero_pos: Dict[str, int], monsters: List[Dict[str, Any]]) -> int:
        """
        以怪物重心的反方向作为粗略安全象限：
        0: 右下, 1: 左下, 2: 左上, 3: 右上
        """
        valid = []
        for m in monsters:
            pos = m.get("pos", {})
            mx, mz = _safe_int(pos.get("x", 0)), _safe_int(pos.get("z", 0))
            if not self._is_valid_global(mx, mz):
                continue
            if mx == 0 and mz == 0:
                continue
            valid.append((mx, mz))
        if not valid:
            return 0
        mean_x = float(sum(v[0] for v in valid)) / float(len(valid))
        mean_z = float(sum(v[1] for v in valid)) / float(len(valid))
        hx, hz = float(hero_pos["x"]), float(hero_pos["z"])
        # 怪物在英雄的哪个象限 -> 安全象限取相反方向
        dx = mean_x - hx
        dz = mean_z - hz
        if dx >= 0 and dz >= 0:
            return 2  # 怪物右下 -> 安全左上
        if dx < 0 and dz >= 0:
            return 3  # 怪物左下 -> 安全右上
        if dx < 0 and dz < 0:
            return 0  # 怪物左上 -> 安全右下
        return 1      # 怪物右上 -> 安全左下

    def _is_in_safe_quadrant(self, hero_pos: Dict[str, int], pos: Dict[str, int], safe_quadrant_id: int) -> bool:
        dx = int(pos["x"]) - int(hero_pos["x"])
        dz = int(pos["z"]) - int(hero_pos["z"])
        if safe_quadrant_id == 0:
            return dx >= 0 and dz >= 0
        if safe_quadrant_id == 1:
            return dx < 0 and dz >= 0
        if safe_quadrant_id == 2:
            return dx < 0 and dz < 0
        return dx >= 0 and dz < 0

    def _is_local_dead_end(self, local_map: np.ndarray) -> bool:
        """
        简化版死路判断：
        以局部图中中心点出发，若可达区域太小，且边界开口很少，则判为死路。
        """
        if local_map.size == 0:
            return False
        rows, cols = local_map.shape
        cr, cc = rows // 2, cols // 2
        if int(local_map[cr, cc]) == 0:
            return True

        q = deque([(cr, cc)])
        dist = np.full((rows, cols), -1, dtype=np.int32)
        dist[cr, cc] = 0
        dirs4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        reachable = 0
        boundary_openings = 0

        while q:
            r, c = q.popleft()
            reachable += 1
            if r == 0 or c == 0 or r == rows - 1 or c == cols - 1:
                boundary_openings += 1
            for dr, dc in dirs4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and dist[nr, nc] < 0 and int(local_map[nr, nc]) != 0:
                    dist[nr, nc] = dist[r, c] + 1
                    q.append((nr, nc))

        return (reachable <= 30) and (boundary_openings <= 6)

    def _build_candidate_action_features(
        self,
        hero_pos: Dict[str, int],
        monsters: List[Dict[str, Any]],
        treasure_targets: List[Dict[str, Any]],
        map_info: Optional[List[List[int]]],
        has_speed_buff: bool,
    ) -> np.ndarray:
        """
        8 个移动动作 × 7 维：
        [next_x_norm, next_z_norm, next_min_monster_path_dist_norm,
         next_in_safe_quadrant, next_closer_to_treasure,
         next_openness, next_dead_end_flag]
        """
        feat_per_action = 7
        out = np.zeros(8 * feat_per_action, dtype=np.float32)

        cur_treasure_dist, _, _ = self._nearest_known_target_distance(
            hero_pos, treasure_targets, map_info, target_sub_type=1
        )
        safe_quad = self._safe_quadrant_id(hero_pos, monsters)

        for a in range(8):
            next_pos, next_r, next_c = self._simulate_next_position(hero_pos, map_info, a, has_speed_buff)
            base = a * feat_per_action

            # 1-2) 下一位置
            out[base + 0] = _norm(next_pos["x"], MAP_SIZE - 1)
            out[base + 1] = _norm(next_pos["z"], MAP_SIZE - 1)

            # 3) 下一位置到最近怪物的沿路距离
            next_monster_dist, _, _ = self._nearest_known_target_distance(
                next_pos,
                [{"sub_type": -1, "status": 1, "pos": m.get("pos", {})} for m in monsters],
                map_info,
                target_sub_type=None,
            )
            out[base + 2] = 1.0 if next_monster_dist is None else _norm(next_monster_dist, MAP_SIZE * 2.0)

            # 4) 是否位于粗略安全象限
            out[base + 3] = 1.0 if self._is_in_safe_quadrant(hero_pos, next_pos, safe_quad) else 0.0

            # 5) 是否更接近宝箱（使用沿路距离）
            next_treasure_dist, _, _ = self._nearest_known_target_distance(
                next_pos, treasure_targets, map_info, target_sub_type=1
            )
            closer = 0.0
            if cur_treasure_dist is not None and next_treasure_dist is not None:
                closer = 1.0 if next_treasure_dist < cur_treasure_dist else 0.0
            out[base + 4] = closer

            # 6) 开阔度
            openness = 0.0
            if next_r is not None and next_c is not None:
                openness = self._count_free_neighbors(map_info, next_r, next_c, radius=1)
            out[base + 5] = float(openness)

            # 7) 死路标记
            dead_end = 0.0
            if next_r is not None and next_c is not None:
                local_map = self._extract_local_map_centered(map_info, next_r, next_c, view_size=21)
                dead_end = 1.0 if self._is_local_dead_end(local_map) else 0.0
            out[base + 6] = dead_end

        return out.astype(np.float32)

    def _build_action_risk_benefit_features(
        self,
        hero_pos: Dict[str, int],
        monsters: List[Dict[str, Any]],
        treasure_targets: List[Dict[str, Any]],
        map_info: Optional[List[List[int]]],
        has_speed_buff: bool,
        move_mask: Sequence[int],
    ) -> np.ndarray:
        """
        8 × 4 = 32 维：
        [risk_monster, risk_dead_end, benefit_treasure, benefit_openness]
        这里也改成沿路距离驱动。
        """
        feat_per_action = 4
        out = np.zeros(8 * feat_per_action, dtype=np.float32)

        cur_treasure_dist, _, _ = self._nearest_known_target_distance(
            hero_pos, treasure_targets, map_info, target_sub_type=1
        )
        cur_monster_dist, _, monster_dist_mode = self._nearest_known_monster_distance(
            hero_pos,
            monsters,
            map_info,
        )

        for a in range(8):
            base = a * feat_per_action
            if int(move_mask[a]) == 0:
                out[base:base + feat_per_action] = 0.0
                continue

            next_pos, next_r, next_c = self._simulate_next_position(hero_pos, map_info, a, has_speed_buff)
            next_monster_dist, _, _ = self._nearest_known_target_distance(
                next_pos,
                [{"sub_type": -1, "status": 1, "pos": m.get("pos", {})} for m in monsters],
                map_info,
                target_sub_type=None,
            )
            next_treasure_dist, _, _ = self._nearest_known_target_distance(
                next_pos, treasure_targets, map_info, target_sub_type=1
            )

            # risk_monster：越靠近怪物风险越大
            if next_monster_dist is None:
                risk_monster = 0.0
            else:
                if cur_monster_dist is None:
                    risk_monster = 1.0 / (1.0 + float(next_monster_dist))
                else:
                    # 靠近怪物时风险增大
                    risk_monster = max(
                        0.0, float(cur_monster_dist - next_monster_dist)
                    )
                    risk_monster = float(np.clip(risk_monster / 3.0, 0.0, 1.0))

            # risk_dead_end
            risk_dead_end = 0.0
            benefit_openness = 0.0
            if next_r is not None and next_c is not None:
                local_map = self._extract_local_map_centered(map_info, next_r, next_c, view_size=21)
                risk_dead_end = 1.0 if self._is_local_dead_end(local_map) else 0.0
                benefit_openness = self._count_free_neighbors(map_info, next_r, next_c, radius=1)

            # benefit_treasure：更接近宝箱则更高
            benefit_treasure = 0.0
            if cur_treasure_dist is not None and next_treasure_dist is not None:
                delta = float(cur_treasure_dist - next_treasure_dist)
                benefit_treasure = float(np.clip(delta / 3.0, 0.0, 1.0))

            out[base + 0] = risk_monster
            out[base + 1] = risk_dead_end
            out[base + 2] = benefit_treasure
            out[base + 3] = benefit_openness

        return out.astype(np.float32)

    # ---------------------------------------------------------------------
    # 奖励设计：所有距离 shaping 都使用“沿路距离变化量”
    # ---------------------------------------------------------------------
    def _compute_rewards(
        self,
        hero: Dict[str, Any],
        env_info: Dict[str, Any],
        hero_pos: Dict[str, int],
        map_info: Optional[List[List[int]]],
        treasure_targets: List[Dict[str, Any]],
        buff_targets: List[Dict[str, Any]],
        monsters: List[Dict[str, Any]],
        last_action: Optional[int],
        legal_action: Sequence[int],
        new_explored_cells: int,
        terminated: bool,
        truncated: bool,
    ) -> np.ndarray:
        reward_terms: Dict[str, float] = {
            "survive_reward": 0.0,
            "treasure_reward": 0.0,
            "speed_buff_reward": 0.0,
            "treasure_approach_reward": 0.0,
            "speed_buff_approach_reward": 0.0,
            "monster_dist_shaping": 0.0,
            "wall_collision_penalty": 0.0,
            "exploration_reward": 0.0,
            "idle_wander_penalty": 0.0,
        }

        # 1) 生存奖励
        if self._is_stage_enabled("survive_reward") and not terminated and not truncated:
            reward_terms["survive_reward"] = float(self._cfg("survive_reward", "coef", 0.02))

        # 2) 稀疏宝箱奖励
        treasure_count = self._get_treasure_collected_count(env_info, hero)
        treasure_delta = int(treasure_count - self.prev_treasure_count)
        if self._is_stage_enabled("treasure_reward") and treasure_delta > 0:
            reward_terms["treasure_reward"] = float(treasure_delta) * float(
                self._cfg("treasure_reward", "coef", 1.0)
            )
        self.prev_treasure_count = treasure_count

        # 3) 稀疏 buff 奖励：buff 状态从无到有视为获取
        has_speed_buff = self._has_speed_buff(hero, env_info)
        if (
            self._is_stage_enabled("speed_buff_reward")
            and has_speed_buff
            and (not self.prev_has_speed_buff)
        ):
            reward_terms["speed_buff_reward"] = float(self._cfg("speed_buff_reward", "coef", 0.25))
        self.prev_has_speed_buff = has_speed_buff

        # 4) 沿路距离 shaping：宝箱 / buff / 怪物
        cur_treasure_dist, _, _ = self._nearest_known_target_distance(
            hero_pos, treasure_targets, map_info, target_sub_type=1
        )
        cur_buff_dist, _, _ = self._nearest_known_target_distance(
            hero_pos, buff_targets, map_info, target_sub_type=2
        )
        cur_monster_dist, _, monster_dist_mode = self._nearest_known_monster_distance(
            hero_pos,
            monsters,
            map_info,
        )

        if self._is_stage_enabled("treasure_approach_reward"):
            if self.prev_nearest_treasure_path_dist is not None and cur_treasure_dist is not None:
                # 更接近宝箱为正，远离为负
                delta = float(self.prev_nearest_treasure_path_dist - cur_treasure_dist)
                reward_terms["treasure_approach_reward"] = float(
                    self._cfg("treasure_approach_reward", "coef", 0.08)
                ) * _clip_delta(delta)
        self.prev_nearest_treasure_path_dist = cur_treasure_dist

        if self._is_stage_enabled("speed_buff_approach_reward"):
            if self.prev_nearest_buff_path_dist is not None and cur_buff_dist is not None:
                delta = float(self.prev_nearest_buff_path_dist - cur_buff_dist)
                reward_terms["speed_buff_approach_reward"] = float(
                    self._cfg("speed_buff_approach_reward", "coef", 0.05)
                ) * _clip_delta(delta)
        self.prev_nearest_buff_path_dist = cur_buff_dist

        if self._is_stage_enabled("monster_dist_shaping"):
            if self.prev_nearest_monster_path_dist is not None and cur_monster_dist is not None:
                # 远离怪物：正；靠近怪物：负
                delta = float(cur_monster_dist - self.prev_nearest_monster_path_dist)
                reward_terms["monster_dist_shaping"] = float(
                    self._cfg("monster_dist_shaping", "coef", 0.12)
                ) * _clip_delta(delta)

        if cur_monster_dist is not None:
            self.prev_nearest_monster_path_dist = cur_monster_dist

        enabled = self._is_stage_enabled("monster_dist_shaping")
        coef = float(self._cfg("monster_dist_shaping", "coef", 0.12))



        if enabled:
            if self.prev_nearest_monster_path_dist is not None and cur_monster_dist is not None:
                delta = float(cur_monster_dist - self.prev_nearest_monster_path_dist)
                clipped = _clip_delta(delta)
                print(
                    "[monster_dist_shaping detail]",
                    "delta=", delta,
                    "clipped=", clipped,
                    "reward=", coef * clipped,
                )

        # 5) 撞墙 / 无效移动惩罚
        if self._is_stage_enabled("wall_collision_penalty") and last_action is not None and 0 <= int(last_action) < 8:
            if legal_action and int(legal_action[int(last_action)]) == 0:
                reward_terms["wall_collision_penalty"] = float(
                    self._cfg("wall_collision_penalty", "coef", -0.05)
                )
            elif self.prev_hero_pos is not None:
                moved = (
                    int(hero_pos["x"]) != int(self.prev_hero_pos["x"]) or
                    int(hero_pos["z"]) != int(self.prev_hero_pos["z"])
                )
                if not moved:
                    reward_terms["wall_collision_penalty"] = float(
                        self._cfg("wall_collision_penalty", "coef", -0.05)
                    )
                self.prev_move_succeeded = moved

        # 6) 开图奖励
        if self._is_stage_enabled("exploration_reward") and new_explored_cells > 0:
            reward_terms["exploration_reward"] = float(new_explored_cells) * float(
                self._cfg("exploration_reward", "coef_per_cell", 0.0005)
            )

        # 7) 原地不动 / 小范围徘徊惩罚
        if self._is_stage_enabled("idle_wander_penalty") and self.prev_hero_pos is not None:
            if (
                int(hero_pos["x"]) == int(self.prev_hero_pos["x"])
                and int(hero_pos["z"]) == int(self.prev_hero_pos["z"])
            ):
                reward_terms["idle_wander_penalty"] = float(
                    self._cfg("idle_wander_penalty", "coef", -0.02)
                )

        self.prev_hero_pos = {"x": int(hero_pos["x"]), "z": int(hero_pos["z"])}

        total_reward = float(sum(reward_terms.values()))
        self.last_reward_info = reward_terms
        return np.array([total_reward], dtype=np.float32)

    # ---------------------------------------------------------------------
    # 主入口
    # ---------------------------------------------------------------------
    def feature_process(self, env_obs: Dict[str, Any], last_action: Optional[int]):
        """
        主入口：
        输入原始 env_obs，输出：
        - feature: 观测向量
        - legal_action: 16 维合法动作 mask
        - reward: shape=(1,) 的奖励
        """
        hero = self._extract_hero(env_obs)
        env_info = self._extract_env_info(env_obs)
        monsters = self._extract_monsters(env_obs)
        organs = self._extract_organs(env_obs)
        map_info = self._extract_map_info(env_obs)
        terminated, truncated = self._extract_done_flags(env_obs)

        hero_pos = self._hero_pos(hero)

        # 1) 更新 episode 地图记忆
        new_explored_cells = self._update_explored_map(hero_pos, map_info)

        # 2) 更新对象记忆
        self._update_object_memory(hero_pos, organs, monsters, map_info)

        # 3) 组装“当前目标列表”
        #    当前可见器官直接来自 organs；视野外但已见过的器官来自 organ_memory。
        visible_treasures = self._filter_visible_organs(organs, hero_pos, map_info, target_sub_type=1)
        visible_buffs = self._filter_visible_organs(organs, hero_pos, map_info, target_sub_type=2)

        memory_treasures = self._collect_memory_targets(1)
        memory_buffs = self._collect_memory_targets(2)

        # 可用于路径计算的目标集：
        treasure_targets = visible_treasures if visible_treasures else memory_treasures
        buff_targets = visible_buffs if visible_buffs else memory_buffs

        # 4) hero / monster 特征
        hero_feat = self._build_hero_feature(hero, env_info)

        # 怪物排序：优先按沿路距离从近到远，保留 2 个
        monster_candidates: List[Tuple[int, Dict[str, Any]]] = []
        for m in monsters:
            pos = m.get("pos", {})
            mx, mz = _safe_int(pos.get("x", 0)), _safe_int(pos.get("z", 0))
            if not self._is_valid_global(mx, mz):
                continue
            if mx == 0 and mz == 0:
                continue
            dist_info = self.compute_path_distance(hero_pos, {"x": mx, "z": mz}, map_info)
            if dist_info["distance"] is None:
                continue
            monster_candidates.append((int(dist_info["distance"]), m))
        monster_candidates.sort(key=lambda x: x[0])
        top_monsters = [m for _, m in monster_candidates[:2]]
        while len(top_monsters) < 2:
            top_monsters.append(None)

        monster0_feat = self._build_monster_feature(hero_pos, top_monsters[0], map_info)
        monster1_feat = self._build_monster_feature(hero_pos, top_monsters[1], map_info)

        # 5) 宝箱 / buff 槽位
        treasure_slots = self._collect_organ_slots(organs, hero_pos, map_info, target_sub_type=1, slot_num=10)
        buff_slots = self._collect_organ_slots(organs, hero_pos, map_info, target_sub_type=2, slot_num=2)

        # 6) 地图特征
        if map_info is None:
            local_map_feat = np.zeros((LOCAL_MAP_VIEW_SIZE, LOCAL_MAP_VIEW_SIZE), dtype=np.float32)
        else:
            local_map_feat = (np.array(map_info, dtype=np.float32) != 0).astype(np.float32)
            if local_map_feat.shape != (LOCAL_MAP_VIEW_SIZE, LOCAL_MAP_VIEW_SIZE):
                fixed_map = np.zeros((LOCAL_MAP_VIEW_SIZE, LOCAL_MAP_VIEW_SIZE), dtype=np.float32)
                rows = min(LOCAL_MAP_VIEW_SIZE, local_map_feat.shape[0])
                cols = min(LOCAL_MAP_VIEW_SIZE, local_map_feat.shape[1])
                fixed_map[:rows, :cols] = local_map_feat[:rows, :cols]
                local_map_feat = fixed_map
        local_map_feat = local_map_feat.reshape(-1)

        # 7) progress 特征：步数 & 已探索比例
        known_cells = int(np.sum(self.explored_map != UNKNOWN_CELL))
        progress_feat = np.array(
            [
                _norm(self.step_count, 2000.0),
                _norm(known_cells, float(MAP_SIZE * MAP_SIZE)),
            ],
            dtype=np.float32,
        )

        # 8) 动作 mask
        has_speed_buff = self._has_speed_buff(hero, env_info)
        move_mask = self._preprocess_move_action_mask(map_info, has_speed_buff)
        flash_mask = self._preprocess_flash_action_mask(map_info, self._flash_cd_norm(hero, env_info))
        legal_action = list(move_mask) + list(flash_mask)

        # 9) 候选动作特征与软风险收益
        candidate_action_feat = self._build_candidate_action_features(
            hero_pos, monsters, treasure_targets, map_info, has_speed_buff
        )
        action_risk_benefit_feat = self._build_action_risk_benefit_features(
            hero_pos, monsters, treasure_targets, map_info, has_speed_buff, move_mask
        )

        # 10) 拼接总特征
        feature = np.concatenate(
            [
                hero_feat,
                monster0_feat,
                monster1_feat,
                treasure_slots,
                buff_slots,
                local_map_feat.astype(np.float32),
                progress_feat,
                candidate_action_feat,
                action_risk_benefit_feat,
            ],
            axis=0,
        ).astype(np.float32)

        # 与当前仓库 Config 维度做兜底对齐：
        expected_dim = int(getattr(Config, "DIM_OF_OBSERVATION", len(feature)))
        if len(feature) < expected_dim:
            padded = np.zeros(expected_dim, dtype=np.float32)
            padded[: len(feature)] = feature
            feature = padded
        elif len(feature) > expected_dim:
            feature = feature[:expected_dim]

        reward = self._compute_rewards(
            hero=hero,
            env_info=env_info,
            hero_pos=hero_pos,
            map_info=map_info,
            treasure_targets=treasure_targets,
            buff_targets=buff_targets,
            monsters=monsters,
            last_action=last_action,
            legal_action=legal_action,
            new_explored_cells=new_explored_cells,
            terminated=terminated,
            truncated=truncated,
        )

        self.step_count += 1
        return feature, legal_action, reward
