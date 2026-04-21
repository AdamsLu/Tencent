
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
                "flash_step_cardinal": 10,
                "flash_step_diagonal": 8,
            },
            "training": {
                # True: 按阶段训练
                # False: 直接训练（忽略阶段奖励表，只看各 reward.enable）
                "use_curriculum": True,
            },
            "survive_reward": {"enable": True, "coef": 0.02},
            "treasure_reward": {"enable": True, "coef": 1.0},
            "speed_buff_reward": {"enable": True, "coef": 0.25},
            "treasure_approach_reward": {"enable": True, "coef": 0.08},
            "speed_buff_approach_reward": {"enable": True, "coef": 0.05},
            "monster_dist_shaping": {"enable": True, "coef": 0.12},
            "wall_collision_penalty": {"enable": False, "coef": -0.05},
            "danger_penalty": {"enable": False, "coef": -0.06, "power": 2.0},
            "exploration_reward": {"enable": True, "coef_per_cell": 0.0005},
            "idle_wander_penalty": {"enable": False, "idle_coef": -0.01, "wander_coef": -0.02, "idle_growth": 0.02, "wander_growth": 0.02},
            "dead_end_penalty": {"enable": False, "coef": -0.05},
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
        if not self.use_curriculum_training():
            return "direct_train"

        mapping = {
            1: "survival_base",
            2: "explore_and_stabilize",
            3: "safe_resource_acquisition",
            4: "full_game_and_skill_refine",
        }
        return mapping.get(self.get_curriculum_stage(), "unknown")

    def get_reward_term_coef(self, section: str, key: str = "coef", default: float = 1.0) -> float:
        return float(self._cfg(section, key, default))

    def use_curriculum_training(self) -> bool:
        return bool(self._cfg("training", "use_curriculum", True))

    def _reward_cfg_enabled(self, name: str) -> bool:
        return bool(self._cfg(name, "enable", True))

    def _is_stage_enabled(self, name: str) -> bool:
        # 先看 reward_conf.toml 里的总开关
        cfg_on = self._reward_cfg_enabled(name)
        if not cfg_on:
            return False

        # 如果不按阶段训练，则只看 enable，不看阶段表
        if not self.use_curriculum_training():
            return True

        # 按阶段训练时：阶段表 AND enable
        stage = self.get_curriculum_stage()
        stage_reward_map = {
            1: {"survive_reward", "monster_dist_shaping", "danger_penalty", "wall_collision_penalty", "dead_end_penalty"},
            2: {
                "survive_reward",
                "monster_dist_shaping",
                "exploration_reward",
                "idle_wander_penalty",
                "danger_penalty",
                "wall_collision_penalty",
                "dead_end_penalty",
            },
            3: {
                "survive_reward",
                "monster_dist_shaping",
                "exploration_reward",
                "idle_wander_penalty",
                "treasure_reward",
                "speed_buff_reward",
                "treasure_approach_reward",
                "speed_buff_approach_reward",
                "danger_penalty",
                "wall_collision_penalty",
                "dead_end_penalty",
            },
            4: {
                "survive_reward",
                "monster_dist_shaping",
                "exploration_reward",
                "idle_wander_penalty",
                "treasure_reward",
                "speed_buff_reward",
                "treasure_approach_reward",
                "speed_buff_approach_reward",
                "danger_penalty",
                "wall_collision_penalty",
                "dead_end_penalty",
            },
        }
        return name in stage_reward_map.get(stage, stage_reward_map[1])

    def _is_survival_only_stage(self) -> bool:
        if not self.use_curriculum_training():
            return False
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

        # State for idle/wander penalty tracking
        self._idle_steps: int = 0
        self._wander_points: deque = deque(maxlen=int(self._global_cfg("trajectory_window", 20)))

        # State for dead-end penalty tracking
        self._in_dead_end: bool = False
        self._dead_end_anchor: Optional[Tuple[int, int]] = None

        self.step_count = 0
        self.last_reward_info: Dict[str, float] = {}
        self.monster_oov_bucket_counter = {}
        self.prev_move_mask_for_last_action = None
        self.prev_flash_mask_for_last_action = None
        self.prev_legal_action_for_last_action = None
        self.prev_simulated_next_positions = None
        self.prev_move_mask_debug = None
        self.prev_map_info_debug = None
        self.prev_raw_hero_debug = None
        self.prev_raw_env_info_debug = None

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
        """
        更稳地提取 hero。
        关键修复：
        1) 优先支持 heroes / frame_state.heroes / observation.frame_state.heroes
        2) 若命中的是 list，则选第一个带合法 pos 的对象
        3) 尽量不要再用“第一个带 pos 的 dict”这种高风险兜底
        """
        hero = self._get_from_paths(
            env_obs,
            [
                ("heroes",),
                ("hero",),
                ("player",),
                ("player_state",),

                ("frame_state", "heroes"),
                ("frame_state", "hero"),
                ("frame_state", "player"),
                ("frame_state", "player_state"),

                ("observation", "heroes"),
                ("observation", "hero"),
                ("observation", "player"),
                ("observation", "player_state"),

                ("observation", "frame_state", "heroes"),
                ("observation", "frame_state", "hero"),
                ("observation", "frame_state", "player"),
                ("observation", "frame_state", "player_state"),

                ("obs", "heroes"),
                ("obs", "hero"),
                ("obs", "player"),
                ("obs", "player_state"),
            ],
            default=None,
        )

        # heroes 可能是 list
        if isinstance(hero, list):
            hero_candidates = []
            for item in hero:
                if not isinstance(item, dict):
                    continue
                pos = item.get("pos", {})
                if isinstance(pos, dict) and "x" in pos and "z" in pos:
                    hero_candidates.append(item)

            hero = hero_candidates[0] if hero_candidates else None

        # 若是 dict，但没有 pos，则视为无效
        if isinstance(hero, dict):
            pos = hero.get("pos", {})
            if not (isinstance(pos, dict) and "x" in pos and "z" in pos):
                hero = None

        # 不再盲目使用“第一个带 pos 的 dict”兜底
        if not isinstance(hero, dict):
            print("[hero_extract_warning] failed to locate real hero, fallback to zero pos")
            hero = {"pos": {"x": 0, "z": 0}}

        print(
            "[hero_extract_debug]",
            "hero_keys=", list(hero.keys()) if isinstance(hero, dict) else None,
            "hero=", hero,
        )
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
                ("observation", "monster"),
                ("observation", "monster_states"),

                # 关键补充：你的日志里 env_obs 是 observation.frame_state.*
                ("observation", "frame_state", "monsters"),
                ("observation", "frame_state", "monster"),
                ("observation", "frame_state", "monster_states"),

                ("obs", "monsters"),
                ("obs", "monster"),
                ("obs", "monster_states"),
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
                ("observation", "organ"),
                ("observation", "organ_states"),

                # 关键补充
                ("observation", "frame_state", "organs"),
                ("observation", "frame_state", "organ"),
                ("observation", "frame_state", "organ_states"),

                ("obs", "organs"),
                ("obs", "organ"),
                ("obs", "organ_states"),
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
                ("frame_state", "local_map"),

                ("observation", "map_info"),
                ("observation", "map"),
                ("observation", "local_map"),

                # 关键补充
                ("observation", "frame_state", "map_info"),
                ("observation", "frame_state", "map"),
                ("observation", "frame_state", "local_map"),

                ("obs", "map_info"),
                ("obs", "map"),
                ("obs", "local_map"),
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

                ("observation", "env_info"),
                ("observation", "game_info"),

                # 关键补充
                ("observation", "frame_state", "env_info"),
                ("observation", "frame_state", "game_info"),
            ],
            default=None,
        )
        if isinstance(env_info, dict):
            return env_info
        return env_obs if isinstance(env_obs, dict) else {}

    def _extract_done_flags(self, env_obs: Dict[str, Any]) -> Tuple[bool, bool]:
        terminated = self._get_from_paths(
            env_obs,
            [
                ("terminated",),
                ("done",),
                ("frame_state", "terminated"),
                ("frame_state", "done"),
                ("observation", "terminated"),
                ("observation", "done"),
                ("observation", "frame_state", "terminated"),
                ("observation", "frame_state", "done"),
            ],
            default=False,
        )

        truncated = self._get_from_paths(
            env_obs,
            [
                ("truncated",),
                ("frame_state", "truncated"),
                ("observation", "truncated"),
                ("observation", "frame_state", "truncated"),
            ],
            default=False,
        )
        return bool(terminated), bool(truncated)

    def _extract_env_legal_action(self, env_obs: Dict[str, Any]) -> Optional[List[int]]:
        """
        读取环境原生 legal_action。

        目标输出长度固定为 16：
        - 前 8 个：移动
        - 后 8 个：闪现

        若找不到或格式异常，则返回 None。
        """
        raw_legal_action = self._get_from_paths(
            env_obs,
            [
                ("legal_action",),
                ("frame_state", "legal_action"),
                ("observation", "legal_action"),
                ("observation", "frame_state", "legal_action"),
                ("obs", "legal_action"),
            ],
            default=None,
        )

        if raw_legal_action is None:
            return None

        try:
            arr = np.array(raw_legal_action, dtype=np.int32).reshape(-1)
        except Exception:
            return None

        if arr.size < 16:
            return None

        arr = arr[:16]
        arr = (arr > 0).astype(np.int32)
        return arr.tolist()

    def _debug_compare_legal_action(
        self,
        hero_pos: Dict[str, int],
        self_legal_action: Sequence[int],
        env_legal_action: Optional[Sequence[int]],
        final_legal_action: Sequence[int],
        every_n_steps: int = 1,
    ) -> None:
        """
        打印：
        - 自己算的 mask
        - 环境原生 mask
        - 最终取交集后的 mask
        - 两者不一致的位置
        """
        if every_n_steps <= 0:
            every_n_steps = 1
        if (self.step_count % every_n_steps) != 0:
            return

        self_arr = np.array(self_legal_action, dtype=np.int32).reshape(-1)
        final_arr = np.array(final_legal_action, dtype=np.int32).reshape(-1)

        env_arr = None
        if env_legal_action is not None:
            env_arr = np.array(env_legal_action, dtype=np.int32).reshape(-1)

        mismatch_idx = []
        if env_arr is not None and env_arr.shape == self_arr.shape:
            mismatch_idx = np.where(self_arr != env_arr)[0].tolist()

        print(
            "[legal_action_compare]",
            "step=", self.step_count,
            "hero_pos=", hero_pos,
            "self_legal_action=", self_arr.tolist(),
            "env_legal_action=", None if env_arr is None else env_arr.tolist(),
            "final_legal_action=", final_arr.tolist(),
            "mismatch_idx=", mismatch_idx,
        )

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

    def _get_monster_field_from_paths(
        self,
        monster: Dict[str, Any],
        paths: Sequence[Sequence[str]],
        default: Any = None,
    ) -> Any:
        for path in paths:
            cur = monster
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

    def _parse_direction_id(self, raw_dir: Any) -> Optional[int]:
        """
        统一解析怪物相对方向，输出与 _relative_direction_id 一致的 1~8 编码：
        1=东, 2=东北, 3=北, 4=西北, 5=西, 6=西南, 7=南, 8=东南
        """
        if raw_dir is None:
            return None

        # 数字直接解析
        try:
            d = int(raw_dir)
            if 1 <= d <= 8:
                return d
        except Exception:
            pass

        # 字符串兜底
        s = str(raw_dir).strip().lower()
        mapping = {
            "e": 1, "east": 1, "东": 1,
            "ne": 2, "northeast": 2, "north_east": 2, "东北": 2,
            "n": 3, "north": 3, "北": 3,
            "nw": 4, "northwest": 4, "north_west": 4, "西北": 4,
            "w": 5, "west": 5, "西": 5,
            "sw": 6, "southwest": 6, "south_west": 6, "西南": 6,
            "s": 7, "south": 7, "南": 7,
            "se": 8, "southeast": 8, "south_east": 8, "东南": 8,
        }
        return mapping.get(s, None)

    def _extract_monster_out_of_view_hint(
        self,
        monster: Dict[str, Any],
    ) -> Tuple[Optional[int], Optional[int]]:
        """
        提取视野外怪物的方向 + 距离桶提示。

        当前从日志确认到的字段：
        - hero_relative_direction: 相对方向
        - hero_l2_distance: 相对距离桶 / 距离等级
        - is_in_view: 是否在视野内（0 表示不在视野内）
        """
        if not isinstance(monster, dict):
            return None, None

        # 若怪物明确在视野内，就不走这个 hint 分支
        is_in_view = monster.get("is_in_view", None)
        try:
            if is_in_view is not None and int(is_in_view) == 1:
                return None, None
        except Exception:
            pass

        # 先读你环境里已经确认存在的字段
        raw_dir = monster.get("hero_relative_direction", None)
        raw_bucket = monster.get("hero_l2_distance", None)

        # 再保留一层兜底，避免后续环境字段轻微变化
        if raw_dir is None:
            raw_dir = self._get_monster_field_from_paths(
                monster,
                [
                    ("rel_direction",),
                    ("relative_direction",),
                    ("direction",),
                    ("dir",),
                    ("feature", "rel_direction"),
                    ("feature", "relative_direction"),
                    ("feature", "direction"),
                    ("relative", "direction"),
                ],
                default=None,
            )

        if raw_bucket is None:
            raw_bucket = self._get_monster_field_from_paths(
                monster,
                [
                    ("distance_bucket",),
                    ("dist_bucket",),
                    ("relative_distance_bucket",),
                    ("distance_level",),
                    ("distance_bin",),
                    ("euclid_bucket_id",),
                    ("feature", "distance_bucket"),
                    ("feature", "dist_bucket"),
                    ("relative", "distance_bucket"),
                ],
                default=None,
            )

        direction_id = self._parse_direction_id(raw_dir)

        bucket_id = None
        try:
            bucket_id = int(raw_bucket)
        except Exception:
            bucket_id = None

        if bucket_id is not None:
            bucket_id = int(np.clip(bucket_id, 0, 5))

        print(
            "[monster_hint_parse]",
            "raw_dir=", raw_dir,
            "raw_bucket=", raw_bucket,
            "direction_id=", direction_id,
            "bucket_id=", bucket_id,
            "is_in_view=", is_in_view,
        )

        return direction_id, bucket_id

    def _debug_print_out_of_view_monsters(
        self,
        hero_pos: Dict[str, int],
        monsters: List[Dict[str, Any]],
        map_info: Optional[List[List[int]]],
        every_n_steps: int = 20,
    ) -> None:
        """
        调试打印：
        1) 视野外 monster 的原始字段
        2) 方向 / 距离桶解析结果
        3) 若能解析，打印近似目标位置
        为避免刷屏，默认每 20 step 打一次。
        """
        if every_n_steps <= 0:
            every_n_steps = 1
        if (self.step_count % every_n_steps) != 0:
            return

        print("\n[monster_oov_debug] step=", self.step_count, "hero_pos=", hero_pos)

        for idx, m in enumerate(monsters):
            pos = m.get("pos", {}) if isinstance(m, dict) else {}
            mx, mz = _safe_int(pos.get("x", 0)), _safe_int(pos.get("z", 0))

            target_pos = None
            if self._is_valid_global(mx, mz) and not (mx == 0 and mz == 0):
                target_pos = {"x": mx, "z": mz}

            # 判断是否在当前局部视野内
            is_visible = False
            if target_pos is not None:
                is_visible = self._is_in_local_window(hero_pos, target_pos, LOCAL_HALF)

            # 只打印“视野外”怪物
            if is_visible:
                continue

            direction_id, dist_bucket = self._extract_monster_out_of_view_hint(m)
            if dist_bucket is not None:
                self.monster_oov_bucket_counter[dist_bucket] = self.monster_oov_bucket_counter.get(dist_bucket, 0) + 1

            print("    oov_bucket_counter =", self.monster_oov_bucket_counter)

            approx_target = None
            if direction_id is not None and dist_bucket is not None:
                steps = self._monster_bucket_to_steps(dist_bucket)
                dx, dz = self._direction_id_to_delta(direction_id)
                approx_target = {
                    "x": int(np.clip(int(hero_pos["x"]) + dx * steps, 0, MAP_SIZE - 1)),
                    "z": int(np.clip(int(hero_pos["z"]) + dz * steps, 0, MAP_SIZE - 1)),
                }

            print("\n  [monster_oov_debug:item]", idx)
            print("    raw_monster =", m)
            print("    raw_keys    =", list(m.keys()) if isinstance(m, dict) else None)
            print("    pos         =", pos)
            print("    target_pos  =", target_pos)
            print("    is_visible  =", is_visible)
            print("    direction_id=", direction_id)
            print("    dist_bucket =", dist_bucket)
            print("    approx_target =", approx_target)

    def _direction_id_to_delta(self, direction_id: int) -> Tuple[int, int]:
        """
        与 _relative_direction_id 的编码保持一致。
        """
        mapping = {
            1: (1, 0),    # 东
            2: (1, 1),    # 东北
            3: (0, 1),    # 北
            4: (-1, 1),   # 西北
            5: (-1, 0),   # 西
            6: (-1, -1),  # 西南
            7: (0, -1),   # 南
            8: (1, -1),   # 东南
        }
        return mapping.get(int(direction_id), (0, 0))

    def _monster_bucket_to_steps(self, bucket_id: int) -> int:
        """
        把距离桶映射成一个单调递增的近似步数。
        这里只求“相对合理 + 单调”，用于 shaping，不追求精确测距。
        """
        bucket_to_steps = {
            0: 1,
            1: 3,
            2: 5,
            3: 7,
            4: 10,
            5: 14,
        }
        return int(bucket_to_steps.get(int(bucket_id), 14))

    def _compute_out_of_view_escape_bonus(
        self,
        prev_hero_pos: Optional[Dict[str, int]],
        hero_pos: Dict[str, int],
        monster: Optional[Dict[str, Any]],
    ) -> float:
        """
        视野外怪物的额外方向奖励：
        - 朝“远离怪物”的方向移动 -> 正
        - 朝“接近怪物”的方向移动 -> 负
        - 与怪物方向正交或没动 -> 接近 0
        返回值裁剪到 [-1, 1]
        """
        if prev_hero_pos is None or not isinstance(monster, dict):
            return 0.0

        direction_id, _ = self._extract_monster_out_of_view_hint(monster)
        if direction_id is None:
            return 0.0

        move_dx = int(hero_pos["x"]) - int(prev_hero_pos["x"])
        move_dz = int(hero_pos["z"]) - int(prev_hero_pos["z"])
        if move_dx == 0 and move_dz == 0:
            return 0.0

        monster_dx, monster_dz = self._direction_id_to_delta(direction_id)
        # away vector = 怪物方向的反方向
        away_dx, away_dz = -monster_dx, -monster_dz

        dot = float(move_dx * away_dx + move_dz * away_dz)
        move_norm = float(np.sqrt(move_dx * move_dx + move_dz * move_dz))
        away_norm = float(np.sqrt(away_dx * away_dx + away_dz * away_dz))
        if move_norm < 1e-6 or away_norm < 1e-6:
            return 0.0

        score = dot / (move_norm * away_norm)
        return float(np.clip(score, -1.0, 1.0))

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
        target_pos: Optional[Dict[str, int]] = None,
        map_info: Optional[List[List[int]]] = None,
        monster: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        怪物专用距离接口。

        返回 mode：
        - visible_path
        - explored_path
        - visible_fallback
        - out_of_view_direction_only
        - out_of_view_hint
        - unknown
        """
        has_valid_target = False
        is_visible = False

        if isinstance(target_pos, dict):
            tx = _safe_int(target_pos.get("x", 0))
            tz = _safe_int(target_pos.get("z", 0))
            if self._is_valid_global(tx, tz) and not (tx == 0 and tz == 0):
                has_valid_target = True
                target_pos = {"x": tx, "z": tz}
                is_visible = self._is_in_local_window(start_pos, target_pos, LOCAL_HALF)

        # 1) 当前视野内局部 BFS
        if has_valid_target and is_visible and map_info is not None:
            start_rc = (LOCAL_HALF, LOCAL_HALF)
            goal_rc = self._visible_local_rc(start_pos, target_pos)
            dist_local = self._bfs_local_distance(map_info, start_rc, goal_rc)
            if dist_local is not None:
                return {
                    "distance": int(dist_local),
                    "mode": "visible_path",
                }

        # 2) 已探索地图 BFS
        if has_valid_target:
            dist_explored = self._bfs_explored_distance(start_pos, target_pos)
            if dist_explored is not None:
                return {
                    "distance": int(dist_explored),
                    "mode": "explored_path",
                }

        # 3) 视野内兜底
        if has_valid_target and is_visible:
            fallback_dist = self._chebyshev_distance(start_pos, target_pos)
            return {
                "distance": int(fallback_dist),
                "mode": "visible_fallback",
            }

        # 4) 视野外 hint
        if isinstance(monster, dict):
            direction_id, dist_bucket = self._extract_monster_out_of_view_hint(monster)

            # 4.1 只有方向，没有可靠距离：只返回方向模式
            if direction_id is not None and (dist_bucket is None or dist_bucket == 0):
                return {
                    "distance": None,
                    "mode": "out_of_view_direction_only",
                    "direction_id": int(direction_id),
                    "distance_bucket": dist_bucket,
                }

            # 4.2 方向 + 距离都可靠时，再启用近似距离
            if direction_id is not None and dist_bucket is not None:
                approx_dist = self._monster_bucket_to_steps(dist_bucket)
                dir_dx, dir_dz = self._direction_id_to_delta(direction_id)
                approx_target_pos = {
                    "x": int(np.clip(int(start_pos["x"]) + dir_dx * approx_dist, 0, MAP_SIZE - 1)),
                    "z": int(np.clip(int(start_pos["z"]) + dir_dz * approx_dist, 0, MAP_SIZE - 1)),
                }
                return {
                    "distance": int(approx_dist),
                    "mode": "out_of_view_hint",
                    "direction_id": int(direction_id),
                    "distance_bucket": int(dist_bucket),
                    "approx_target_pos": approx_target_pos,
                }

        return {"distance": None, "mode": "unknown"}

    def _nearest_known_monster_distance(
        self,
        hero_pos: Dict[str, int],
        monsters: List[Dict[str, Any]],
        map_info: Optional[List[List[int]]],
    ) -> Tuple[Optional[int], Optional[Dict[str, Any]], str]:
        """
        返回最近怪物的距离、对应怪物对象、以及距离模式。
        """
        best_dist: Optional[int] = None
        best_obj: Optional[Dict[str, Any]] = None
        best_mode = "unknown"

        # 先存“有距离”的候选
        distance_candidates: List[Tuple[int, Dict[str, Any], str]] = []
        # 再存“只有方向”的候选
        direction_only_candidates: List[Tuple[Dict[str, Any], str]] = []

        for m in monsters:
            pos = m.get("pos", {}) if isinstance(m, dict) else {}
            tx, tz = _safe_int(pos.get("x", 0)), _safe_int(pos.get("z", 0))

            target_pos = None
            if self._is_valid_global(tx, tz) and not (tx == 0 and tz == 0):
                target_pos = {"x": tx, "z": tz}

            dist_info = self.compute_monster_distance(
                start_pos=hero_pos,
                target_pos=target_pos,
                map_info=map_info,
                monster=m,
            )
            d = dist_info.get("distance", None)
            mode = str(dist_info.get("mode", "unknown"))

            if d is not None:
                distance_candidates.append((int(d), m, mode))
            elif mode == "out_of_view_direction_only":
                direction_only_candidates.append((m, mode))

        if distance_candidates:
            distance_candidates.sort(key=lambda x: x[0])
            best_dist, best_obj, best_mode = distance_candidates[0]
            return best_dist, best_obj, best_mode

        if direction_only_candidates:
            best_obj, best_mode = direction_only_candidates[0]
            return None, best_obj, best_mode

        return None, None, "unknown"

    def _resolve_monster_reference_position(
        self,
        hero_pos: Dict[str, int],
        monster: Dict[str, Any],
    ) -> Tuple[Optional[Dict[str, int]], str]:
        """
        基于“当前 hero 位置”解析怪物参考绝对位置：
        1) 若 env 已给出可靠绝对坐标，则直接用；
        2) 否则若给了 视野外方向 + 距离桶，则构造一个近似绝对位置；
        3) 否则返回 None。
        """
        if not isinstance(monster, dict):
            return None, "unknown"

        pos = monster.get("pos", {})
        mx, mz = _safe_int(pos.get("x", 0)), _safe_int(pos.get("z", 0))
        if self._is_valid_global(mx, mz) and not (mx == 0 and mz == 0):
            return {"x": mx, "z": mz}, "absolute"

        direction_id, dist_bucket = self._extract_monster_out_of_view_hint(monster)
        if direction_id is None or dist_bucket is None or dist_bucket == 0:
            return None, "unknown"

        approx_steps = self._monster_bucket_to_steps(dist_bucket)
        dx, dz = self._direction_id_to_delta(direction_id)

        ax = int(np.clip(int(hero_pos["x"]) + dx * approx_steps, 0, MAP_SIZE - 1))
        az = int(np.clip(int(hero_pos["z"]) + dz * approx_steps, 0, MAP_SIZE - 1))
        
        print(
            "[resolve_monster_ref_debug]",
            "step=", self.step_count,
            "hero_pos=", hero_pos,
            "raw_monster=", monster,
            "direction_id=", direction_id,
            "dist_bucket=", dist_bucket,
        )
        return {"x": ax, "z": az}, "hint"

    def _build_planning_monsters(
        self,
        hero_pos: Dict[str, int],
        monsters: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        为候选动作评估构造“固定参考怪物位置”列表。

        关键点：
        - 对视野外怪物，方向/距离桶是“相对当前 hero”的观测；
        - 在评估 next_pos 时，不能把这个 hint 重新相对 next_pos 解释，
          否则等于把怪物也跟着 hero 一起平移了；
        - 所以要先基于当前 hero 固定出一个参考绝对位置，再用它评估 next_pos。
        """
        out: List[Dict[str, Any]] = []
        for m in monsters:
            ref_pos, source = self._resolve_monster_reference_position(hero_pos, m)
            if ref_pos is None:
                continue

            mm = dict(m)
            mm["pos"] = {"x": int(ref_pos["x"]), "z": int(ref_pos["z"])}
            mm["_planning_source"] = source
            out.append(mm)
        return out

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
        [visible, active, rel_x, rel_z, dist_norm, dist_bucket_norm, dir_id_norm,
         speed_norm, mode_visible, mode_explored, mode_known, abs_x_norm, abs_z_norm, mode_hint]
        """
        feat = np.zeros(14, dtype=np.float32)
        if not isinstance(monster, dict):
            return feat

        pos = monster.get("pos", {})
        mx, mz = _safe_int(pos.get("x", 0)), _safe_int(pos.get("z", 0))
        target_pos = None
        if self._is_valid_global(mx, mz) and not (mx == 0 and mz == 0):
            target_pos = {"x": mx, "z": mz}

        dist_info = self.compute_monster_distance(
            start_pos=hero_pos,
            target_pos=target_pos,
            map_info=map_info,
            monster=monster,
        )

        dist_val = dist_info.get("distance", None)
        mode = str(dist_info.get("mode", "unknown"))
        if dist_val is None:
            return feat

        # 用于构建相对/绝对位置特征的“参考位置”
        ref_pos = None
        direction_id = None
        dist_bucket = None

        if mode == "out_of_view_hint":
            ref_pos = dist_info.get("approx_target_pos", None)
            direction_id = dist_info.get("direction_id", None)
            dist_bucket = dist_info.get("distance_bucket", None)
        else:
            ref_pos = target_pos
            if ref_pos is not None:
                direction_id = _relative_direction_id(hero_pos, ref_pos)
            if dist_val is not None:
                dist_bucket = _bucketize_distance(float(dist_val))

        if ref_pos is None:
            return feat

        rx, rz = int(ref_pos["x"]), int(ref_pos["z"])
        dx = rx - int(hero_pos["x"])
        dz = rz - int(hero_pos["z"])

        visible_flag = 1.0 if mode in {"visible_path", "visible_fallback"} else 0.0
        explored_flag = 1.0 if mode == "explored_path" else 0.0
        known_flag = 1.0 if dist_val is not None else 0.0
        hint_flag = 1.0 if mode == "out_of_view_hint" else 0.0
        speed = _safe_float(monster.get("speed", monster.get("move_speed", 0.0)), 0.0)

        feat[0] = visible_flag
        feat[1] = 1.0
        feat[2] = float(dx) / float(MAP_SIZE)
        feat[3] = float(dz) / float(MAP_SIZE)
        feat[4] = _norm(float(dist_val), MAP_SIZE * 2.0)
        feat[5] = 0.0 if dist_bucket is None else _norm(min(int(dist_bucket), 5), 5.0)
        feat[6] = 0.0 if direction_id is None else float(direction_id) / 8.0
        feat[7] = _norm(speed, 5.0)
        feat[8] = visible_flag
        feat[9] = explored_flag
        feat[10] = known_flag
        feat[11] = _norm(rx, MAP_SIZE - 1)
        feat[12] = _norm(rz, MAP_SIZE - 1)
        feat[13] = hint_flag
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
        """
        闪现动作合法性判断。

        与环境动作表保持一致：
        - 8:  右  -> 10 格
        - 9:  右上 -> 8 格
        - 10: 上  -> 10 格
        - 11: 左上 -> 8 格
        - 12: 左  -> 10 格
        - 13: 左下 -> 8 格
        - 14: 下  -> 10 格
        - 15: 右下 -> 8 格

        说明：
        这里仍保持“只检查落点格是否可通行”的逻辑；
        这和你当前代码风格一致，只是把步长改对。
        若后续确认环境要求闪现路径中间也不能穿障碍，再进一步改成路径检查版本。
        """
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

        # 方向顺序与环境表一致：
        # 右, 右上, 上, 左上, 左, 左下, 下, 右下
        dirs8 = [
            (0, 1),    # E
            (-1, 1),   # NE
            (-1, 0),   # N
            (-1, -1),  # NW
            (0, -1),   # W
            (1, -1),   # SW
            (1, 0),    # S
            (1, 1),    # SE
        ]

        # 与环境动作表一致：直线10、对角8
        flash_steps = [
            int(self._global_cfg("flash_step_cardinal", 10)),   # E
            int(self._global_cfg("flash_step_diagonal", 8)),    # NE
            int(self._global_cfg("flash_step_cardinal", 10)),   # N
            int(self._global_cfg("flash_step_diagonal", 8)),    # NW
            int(self._global_cfg("flash_step_cardinal", 10)),   # W
            int(self._global_cfg("flash_step_diagonal", 8)),    # SW
            int(self._global_cfg("flash_step_cardinal", 10)),   # S
            int(self._global_cfg("flash_step_diagonal", 8)),    # SE
        ]

        def is_passable(r: int, c: int) -> bool:
            return 0 <= r < rows and 0 <= c < cols and int(arr[r, c]) != 0

        mask = [1] * 8
        for i, (dr, dc) in enumerate(dirs8):
            step = max(1, int(flash_steps[i]))
            rr = center_r + dr * step
            cc = center_c + dc * step
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

        # 固定“当前帧下怪物的参考绝对位置”，供 next_pos 评估使用
        planning_monsters = self._build_planning_monsters(hero_pos, monsters)

        for a in range(8):
            next_pos, next_r, next_c = self._simulate_next_position(hero_pos, map_info, a, has_speed_buff)
            base = a * feat_per_action

            # 1-2) 下一位置
            out[base + 0] = _norm(next_pos["x"], MAP_SIZE - 1)
            out[base + 1] = _norm(next_pos["z"], MAP_SIZE - 1)

            # 3) 下一位置到最近怪物的距离
            next_monster_dist, _, _ = self._nearest_known_monster_distance(
                next_pos,
                planning_monsters,
                map_info,
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

            print(
                "[candidate_action_debug]",
                "step=", self.step_count,
                "action=", a,
                "hero_pos=", hero_pos,
                "next_pos=", next_pos,
                "move_mask[a]=", None if a >= len(self._preprocess_move_action_mask(map_info, has_speed_buff)) else self._preprocess_move_action_mask(map_info, has_speed_buff)[a],
                "next_r=", next_r,
                "next_c=", next_c,
            )

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
        """
        feat_per_action = 4
        out = np.zeros(8 * feat_per_action, dtype=np.float32)

        cur_treasure_dist, _, _ = self._nearest_known_target_distance(
            hero_pos, treasure_targets, map_info, target_sub_type=1
        )

        planning_monsters = self._build_planning_monsters(hero_pos, monsters)
        cur_monster_dist, _, _ = self._nearest_known_monster_distance(
            hero_pos,
            planning_monsters,
            map_info,
        )

        for a in range(8):
            base = a * feat_per_action
            if int(move_mask[a]) == 0:
                out[base:base + feat_per_action] = 0.0
                continue

            next_pos, next_r, next_c = self._simulate_next_position(hero_pos, map_info, a, has_speed_buff)

            next_monster_dist, _, _ = self._nearest_known_monster_distance(
                next_pos,
                planning_monsters,
                map_info,
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
                    risk_monster = max(0.0, float(cur_monster_dist - next_monster_dist))
                    risk_monster = float(np.clip(risk_monster / 3.0, 0.0, 1.0))

            # risk_dead_end / benefit_openness
            risk_dead_end = 0.0
            benefit_openness = 0.0
            if next_r is not None and next_c is not None:
                local_map = self._extract_local_map_centered(map_info, next_r, next_c, view_size=21)
                risk_dead_end = 1.0 if self._is_local_dead_end(local_map) else 0.0
                benefit_openness = self._count_free_neighbors(map_info, next_r, next_c, radius=1)

            # benefit_treasure
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
            "danger_penalty": 0.0,
            "wall_collision_penalty": 0.0,
            "is_illegal_action": 0.0,
            "is_blocked_after_legal": 0.0,
            "is_wall_collision": 0.0,
            "exploration_reward": 0.0,
            "idle_wander_penalty": 0.0,
            "dead_end_penalty": 0.0,
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
            reward_terms["speed_buff_reward"] = float(self._cfg("speed_buff_reward", "coef", 1))
        self.prev_has_speed_buff = has_speed_buff

        # 4) 沿路距离 shaping：宝箱 / buff / 怪物
        cur_treasure_dist, _, _ = self._nearest_known_target_distance(
            hero_pos, treasure_targets, map_info, target_sub_type=1
        )
        cur_buff_dist, _, _ = self._nearest_known_target_distance(
            hero_pos, buff_targets, map_info, target_sub_type=2
        )
        cur_monster_dist, nearest_monster_obj, monster_dist_mode = self._nearest_known_monster_distance(
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
                    self._cfg("speed_buff_approach_reward", "coef", 0.1)
                ) * _clip_delta(delta)
        self.prev_nearest_buff_path_dist = cur_buff_dist

        if self._is_stage_enabled("monster_dist_shaping"):
            coef = float(self._cfg("monster_dist_shaping", "coef", 0.12))

            # A. 有可靠距离时，不再裁切，直接使用距离变化量
            if self.prev_nearest_monster_path_dist is not None and cur_monster_dist is not None:
                delta = float(cur_monster_dist - self.prev_nearest_monster_path_dist)
                reward_terms["monster_dist_shaping"] = coef * delta

            # B. 视野外只有方向时，只给“背离怪物方向”的奖励
            elif monster_dist_mode == "out_of_view_direction_only":
                escape_bonus = self._compute_out_of_view_escape_bonus(
                    self.prev_hero_pos,
                    hero_pos,
                    nearest_monster_obj,
                )
                reward_terms["monster_dist_shaping"] = 0.5 * coef * float(escape_bonus)

            # C. 视野外方向+距离都可靠时：距离 shaping + 方向 bonus
            elif monster_dist_mode == "out_of_view_hint":
                escape_bonus = self._compute_out_of_view_escape_bonus(
                    self.prev_hero_pos,
                    hero_pos,
                    nearest_monster_obj,
                )
                reward_terms["monster_dist_shaping"] += 0.5 * coef * float(escape_bonus)

        # 只有在当前距离真的可用时，才更新 prev
        if cur_monster_dist is not None:
            self.prev_nearest_monster_path_dist = cur_monster_dist

        enabled = self._is_stage_enabled("monster_dist_shaping")
        coef = float(self._cfg("monster_dist_shaping", "coef", 0.12))

        if enabled:
            if self.prev_nearest_monster_path_dist is not None and cur_monster_dist is not None:
                delta = float(cur_monster_dist - self.prev_nearest_monster_path_dist)
                print(
                    "[monster_dist_shaping detail]",
                    "delta=", delta,
                    "reward=", coef * delta,
                )

        # 4b) 危险惩罚：最近怪物距离过近时按幂函数惩罚
        if self._is_stage_enabled("danger_penalty") and cur_monster_dist is not None:
            danger_thr = float(self._global_cfg("danger_threshold", 0.15)) * MAP_SIZE
            if self.step_count >= 300:
                danger_thr = float(self._global_cfg("late_danger_threshold", 0.25)) * MAP_SIZE
            d_coef = float(self._cfg("danger_penalty", "coef", -0.06))
            d_power = float(self._cfg("danger_penalty", "power", 2.0))
            d = float(cur_monster_dist)
            if d < danger_thr:
                x = float(np.clip((danger_thr - d) / max(danger_thr, 1e-6), 0.0, 1.0))
                reward_terms["danger_penalty"] = float(d_coef * (x ** d_power))

        # 5) 撞墙 / 无效移动惩罚
        if last_action is not None and 0 <= int(last_action) < 8:
            prev_mask_val = None
            if (
                self.prev_legal_action_for_last_action is not None
                and int(last_action) < len(self.prev_legal_action_for_last_action)
            ):
                prev_mask_val = int(self.prev_legal_action_for_last_action[int(last_action)])

            prev_sim_target = None
            if (
                self.prev_simulated_next_positions is not None
                and 0 <= int(last_action) < len(self.prev_simulated_next_positions)
            ):
                prev_sim_target = self.prev_simulated_next_positions[int(last_action)]

            moved = None
            if self.prev_hero_pos is not None:
                moved = (
                    int(hero_pos["x"]) != int(self.prev_hero_pos["x"]) or
                    int(hero_pos["z"]) != int(self.prev_hero_pos["z"])
                )

            is_illegal_action = 0.0
            is_blocked_after_legal = 0.0

            # A. 上一步动作在“上一步 mask”里就非法
            if prev_mask_val == 0:
                is_illegal_action = 1.0

            # B. 上一步动作在 mask 中合法，但执行后没动
            elif prev_mask_val == 1 and self.prev_hero_pos is not None and moved is False:
                is_blocked_after_legal = 1.0

            reward_terms["is_illegal_action"] = float(is_illegal_action)
            reward_terms["is_blocked_after_legal"] = float(is_blocked_after_legal)
            reward_terms["is_wall_collision"] = float(is_illegal_action + is_blocked_after_legal)

            print(
                "[wall_collision_split_debug]",
                "step=", self.step_count,
                "last_action=", last_action,
                "prev_mask_val=", prev_mask_val,
                "prev_sim_target=", prev_sim_target,
                "prev_hero_pos=", self.prev_hero_pos,
                "hero_pos=", hero_pos,
                "moved=", moved,
                "is_illegal_action=", reward_terms["is_illegal_action"],
                "is_blocked_after_legal=", reward_terms["is_blocked_after_legal"],
                "is_wall_collision=", reward_terms["is_wall_collision"],
            )

            if prev_mask_val == 1 and moved is False:
                self._debug_stuck_move_case(
                    prev_hero_pos=self.prev_hero_pos,
                    last_action=int(last_action),
                    prev_sim_target=prev_sim_target,
                    prev_map_info=self.prev_map_info_debug,
                )
                print(
                    "[hero_state_when_blocked]",
                    "step=", self.step_count,
                    "last_action=", last_action,
                    "prev_raw_hero=", self.prev_raw_hero_debug,
                    "prev_raw_env_info=", self.prev_raw_env_info_debug,
                    "curr_raw_hero=", hero,
                    "curr_raw_env_info=", env_info,
                )

            # 惩罚项是否启用，与统计解耦
            if self._is_stage_enabled("wall_collision_penalty") and reward_terms["is_wall_collision"] > 0.5:
                reward_terms["wall_collision_penalty"] = float(
                    self._cfg("wall_collision_penalty", "coef", -0.05)
                )

            if moved is not None:
                self.prev_move_succeeded = bool(moved)

        # 6) 开图奖励
        if self._is_stage_enabled("exploration_reward") and new_explored_cells > 0:
            reward_terms["exploration_reward"] = float(new_explored_cells) * float(
                self._cfg("exploration_reward", "coef_per_cell", 0.0005)
            )

        # 7) 原地不动 / 小范围徘徊惩罚
        if self._is_stage_enabled("idle_wander_penalty"):
            cur_x, cur_z = int(hero_pos["x"]), int(hero_pos["z"])
            self._wander_points.append((cur_x, cur_z))

            idle_coef = float(self._cfg("idle_wander_penalty", "idle_coef", -0.01))
            idle_growth = float(self._cfg("idle_wander_penalty", "idle_growth", 0.02))
            if (
                self.prev_hero_pos is not None
                and cur_x == int(self.prev_hero_pos["x"])
                and cur_z == int(self.prev_hero_pos["z"])
            ):
                self._idle_steps += 1
                reward_terms["idle_wander_penalty"] += float(
                    idle_coef * (1.0 + idle_growth * self._idle_steps)
                )
            else:
                self._idle_steps = 0

            wander_coef = float(self._cfg("idle_wander_penalty", "wander_coef", -0.02))
            wander_growth = float(self._cfg("idle_wander_penalty", "wander_growth", 0.02))
            wander_min_pts = int(self._global_cfg("wander_min_points", 8))
            wander_radius_thr = float(self._global_cfg("wander_radius_threshold", 2.5))
            if len(self._wander_points) >= wander_min_pts and self._idle_steps == 0:
                sum_x = sum_z = 0
                for px, pz in self._wander_points:
                    sum_x += px
                    sum_z += pz
                n = len(self._wander_points)
                cx = sum_x / n
                cz = sum_z / n
                wander_radius_thr_sq = wander_radius_thr ** 2
                max_r_sq = max(
                    (px - cx) ** 2 + (pz - cz) ** 2
                    for px, pz in self._wander_points
                )
                if max_r_sq < wander_radius_thr_sq:
                    reward_terms["idle_wander_penalty"] += float(
                        wander_coef * (1.0 + wander_growth * n)
                    )

        # 8) 死角惩罚：agent 停留在局部死角时持续惩罚，离开后重置
        if self._is_stage_enabled("dead_end_penalty") and map_info is not None:
            cur_x, cur_z = int(hero_pos["x"]), int(hero_pos["z"])
            map_arr = np.array(map_info, dtype=np.int32)
            in_dead_end_now = self._is_local_dead_end(map_arr)
            if in_dead_end_now:
                self._in_dead_end = True
                self._dead_end_anchor = (cur_x, cur_z)
            elif self._in_dead_end and self._dead_end_anchor is not None:
                reset_dist = float(self._global_cfg("dead_end_reset_distance", 8.0))
                ax, az = self._dead_end_anchor
                dist = float(((cur_x - ax) ** 2 + (cur_z - az) ** 2) ** 0.5)
                if dist >= reset_dist:
                    self._in_dead_end = False
                    self._dead_end_anchor = None
            if self._in_dead_end:
                reward_terms["dead_end_penalty"] = float(
                    self._cfg("dead_end_penalty", "coef", -0.05)
                )

        self.prev_hero_pos = {"x": int(hero_pos["x"]), "z": int(hero_pos["z"])}

        total_reward = float(sum(reward_terms.values()))
        self.last_reward_info = reward_terms
        return np.array([total_reward], dtype=np.float32)

    def _debug_stuck_move_case(
        self,
        prev_hero_pos: Dict[str, int],
        last_action: int,
        prev_sim_target: Optional[Dict[str, Any]],
        prev_map_info: Optional[List[List[int]]],
    ) -> None:
        """
        当出现：
        - 上一帧动作在 mask 中合法
        - 但 env 执行后 hero 没动
        时，打印上一帧局部地图的关键信息，验证本地 mask 逻辑是否自洽。
        """
        if prev_map_info is None or prev_sim_target is None:
            print("[stuck_case_debug] prev_map_info or prev_sim_target is None")
            return

        arr = np.array(prev_map_info, dtype=np.int32)
        if arr.ndim != 2 or arr.size == 0:
            print("[stuck_case_debug] invalid prev_map_info")
            return

        rows, cols = arr.shape
        cr, cc = rows // 2, cols // 2   # 21x21 的中心点，一般就是 (10,10)

        # 与 _simulate_next_position 完全一致的方向定义
        action_delta = [
            (1, 0),    # 0 E
            (1, -1),   # 1 NE
            (0, -1),   # 2 N
            (-1, -1),  # 3 NW
            (-1, 0),   # 4 W
            (-1, 1),   # 5 SW
            (0, 1),    # 6 S
            (1, 1),    # 7 SE
        ]
        dx, dz = action_delta[int(last_action)]

        # 目标格在局部图中的坐标
        tr, tc = cr + dz, cc + dx

        info = {
            "center_rc": (cr, cc),
            "target_rc": (tr, tc),
            "center_cell": None,
            "target_cell": None,
            "side1_rc": None,
            "side1_cell": None,
            "side2_rc": None,
            "side2_cell": None,
            "patch_5x5": None,
        }

        def get_cell(r, c):
            if 0 <= r < rows and 0 <= c < cols:
                return int(arr[r, c])
            return "OOB"

        info["center_cell"] = get_cell(cr, cc)
        info["target_cell"] = get_cell(tr, tc)

        # 若是对角动作，再打印两侧切角格
        if dx != 0 and dz != 0:
            side1 = (cr + dz, cc)   # 纵向相邻格
            side2 = (cr, cc + dx)   # 横向相邻格
            info["side1_rc"] = side1
            info["side1_cell"] = get_cell(*side1)
            info["side2_rc"] = side2
            info["side2_cell"] = get_cell(*side2)

        # 打印以中心点为核心的 5x5 patch
        patch = []
        for r in range(cr - 2, cr + 3):
            row = []
            for c in range(cc - 2, cc + 3):
                row.append(get_cell(r, c))
            patch.append(row)
        info["patch_5x5"] = patch

        print(
            "[stuck_case_debug]",
            "prev_hero_pos=", prev_hero_pos,
            "last_action=", last_action,
            "prev_sim_target=", prev_sim_target,
            "local_info=", info,
        )

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

        print(
            "[debug parse]",
            "hero_pos=", hero_pos,
            "monster_num=", len(monsters),
            "has_map=", map_info is not None,
            "map_shape=", None if map_info is None else (len(map_info), len(map_info[0]) if len(map_info) > 0 else 0),
        )

        self._debug_print_out_of_view_monsters(hero_pos, monsters, map_info, every_n_steps=20)

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

            target_pos = None
            if self._is_valid_global(mx, mz) and not (mx == 0 and mz == 0):
                target_pos = {"x": mx, "z": mz}

            dist_info = self.compute_monster_distance(
                start_pos=hero_pos,
                target_pos=target_pos,
                map_info=map_info,
                monster=m,
            )
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

        # 自己基于局部地图/规则计算的 mask
        self_legal_action = list(move_mask) + list(flash_mask)

        # 环境原生 legal_action
        env_legal_action = self._extract_env_legal_action(env_obs)

        # 最终 mask：优先取交集
        if env_legal_action is not None and len(env_legal_action) >= 16:
            legal_action = [
                int(self_legal_action[i]) & int(env_legal_action[i])
                for i in range(16)
            ]
        else:
            legal_action = list(self_legal_action)

        self._debug_compare_legal_action(
            hero_pos=hero_pos,
            self_legal_action=self_legal_action,
            env_legal_action=env_legal_action,
            final_legal_action=legal_action,
            every_n_steps=1,
        )

        simulated_next_positions = []
        for a in range(8):
            sim_next_pos, sim_r, sim_c = self._simulate_next_position(hero_pos, map_info, a, has_speed_buff)
            simulated_next_positions.append(
                {
                    "action": a,
                    "mask": int(move_mask[a]),
                    "next_pos": dict(sim_next_pos),
                    "next_r": sim_r,
                    "next_c": sim_c,
                }
            )

        print(
            "[mask_debug_current]",
            "step=", self.step_count,
            "hero_pos=", hero_pos,
            "last_action=", last_action,
            "move_mask=", move_mask,
            "flash_mask=", flash_mask,
            "self_legal_action=", self_legal_action,
            "env_legal_action=", env_legal_action,
            "final_legal_action=", legal_action,
            "simulated_next_positions=", simulated_next_positions,
        )

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
        self.prev_move_mask_for_last_action = list(move_mask)
        self.prev_flash_mask_for_last_action = list(flash_mask)
        self.prev_legal_action_for_last_action = list(legal_action)
        self.prev_simulated_next_positions = copy.deepcopy(simulated_next_positions)
        self.prev_move_mask_debug = list(move_mask)
        self.prev_map_info_debug = copy.deepcopy(map_info)
        self.prev_raw_hero_debug = copy.deepcopy(hero)
        self.prev_raw_env_info_debug = copy.deepcopy(env_info)
        self.step_count += 1
        return feature, legal_action, reward
