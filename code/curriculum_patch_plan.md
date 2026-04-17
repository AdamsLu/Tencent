# 腾讯开悟 D01 四阶段课程训练改造方案（贴当前仓库）

说明：我这边无法直接写回你的 GitHub 仓库，所以我把**可直接粘贴的代码修改块**整理成了这份补丁说明。  
按下面的“替换 / 新增”去改，你就能得到：

- 四阶段课程训练
- 基于指标而不是纯步数的阶段切换
- 日志里显示当前阶段
- monitor 数据里增加当前阶段
- 保持 actor 网络为 MLP（当前代码本来就是 MLP，无需从线性重构）

---

## 0. actor 网络是否已经是 MLP

是的，**你现在的 actor 已经是 MLP**，不是线性策略头。当前 `model.py` 结构是：

- `encoder`: `vector_dim -> 256 -> 128`
- `actor_head`: `128 -> 128 -> action_num`
- `critic_head`: `128 -> 128 -> value_num`

所以这里**不建议为了“改成 MLP”再动网络结构**，否则会引入额外变量，影响你先把课程训练跑通。

---

## 1. 修改 `code/agent_ppo/feature/preprocessor.py`

### 1.1 在 `class Preprocessor` 里，新增下面这组方法
建议放在 `_global_cfg` 之后、`_is_stage_enabled` 之前。

```python
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
```

### 1.2 用下面这版替换 `_is_stage_enabled`

```python
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
```

### 1.3 用下面这版替换 `_is_survival_only_stage`

这一步非常关键。  
你当前代码里很多“资源槽位 / 靠近宝箱特征 / 资源引导偏置”都是用这个方法控制的。  
如果你想让 **stage 2 继续只学“活 + 探索”，而不是提前学资源贪心**，那这里必须让 stage 1 和 stage 2 都继续算 survival-only。

```python
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
```

### 1.4 修改 `reset()` 中课程阶段初始化方式

你当前 `reset()` 里会把 `self.curriculum_stage = 1` 重新写死。  
这样每开新局都会掉回 stage 1。  
把这一段：

```python
        # ========== 课程训练阶段控制 ==========
        # 当前先只启用 stage 1：生存、危险规避、撞墙规避
        self.curriculum_stage = 1
```

替换为：

```python
        # ========== 课程训练阶段控制 ==========
        # 保留外部 workflow 设置的 stage，不在每局 reset 时回退到 1
        prev_stage = int(getattr(self, "curriculum_stage", 1))
        self.curriculum_stage = max(1, min(4, prev_stage))
```

---

## 2. 修改 `code/agent_ppo/workflow/train_workflow.py`

### 2.1 在 import 区域补一个 `deque`

在文件顶部 import 区域增加：

```python
from collections import deque
```

### 2.2 在 `EpisodeRunner.__init__` 里加入课程训练配置与状态

在 `self.map_random = ...` 后面、`self._reset_reward_accumulators()` 前面，插入下面这段：

```python
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
```

### 2.3 在 `EpisodeRunner` 里新增下面这组辅助方法

建议放在 `_get_reward_monitor_data()` 后、`_is_eval_episode()` 前面。

```python
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
```

### 2.4 在 episode 开始时，把当前阶段同步到 preprocessor，并打日志

在 `run_episodes()` 里，找到这段：

```python
        self.agent.reset(env_obs)
        self.agent.load_model(id="latest")
```

替换成：

```python
        self.agent.reset(env_obs)
        self._apply_curriculum_stage_to_agent()
        self.agent.load_model(id="latest")
```

然后把 episode 开始日志：

```python
        self.logger.info(
            f"Episode {self.episode_cnt} start | Mode: {mode} | Map: {selected_map}"
        )
```

替换成：

```python
        self.logger.info(
            f"Episode {self.episode_cnt} start | "
            f"Mode: {mode} | Map: {selected_map} | "
            f"Stage: {self.curriculum_stage} ({self._get_stage_name()})"
        )
```

### 2.5 在 episode 结束时，记录阶段指标并尝试切阶段

在 `if done:` 里、`self.logger.info([GAMEOVER]...)` 之后，插入下面这段：

```python
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
```

### 2.6 给 `[GAMEOVER]` 日志补当前阶段

把原来的：

```python
                self.logger.info(
                    f"[GAMEOVER] episode:{self.episode_cnt} mode:{mode} map:{selected_map} "
                    f"steps:{step} result:{result_str} sim_score:{sim_score:.1f} "
                    f"total_reward:{total_reward:.3f} eval_win_rate:{eval_win_rate:.2%}"
                )
```

替换成：

```python
                self.logger.info(
                    f"[GAMEOVER] episode:{self.episode_cnt} mode:{mode} map:{selected_map} "
                    f"stage:{self.curriculum_stage}({self._get_stage_name()}) "
                    f"steps:{step} result:{result_str} sim_score:{sim_score:.1f} "
                    f"total_reward:{total_reward:.3f} eval_win_rate:{eval_win_rate:.2%}"
                )
```

### 2.7 给 monitor 数据补当前阶段字段

在 `monitor_data = { ... }` 里增加：

```python
                    "curriculum_stage": int(self.curriculum_stage),
                    "curriculum_stage_transition_cnt": int(self.stage_transition_cnt),
```

建议放在 `"configured_total_map": configured_total_map,` 后面。

---

## 3. 修改 `code/agent_ppo/conf/train_env_conf.toml`

在文件末尾追加下面这一段：

```toml
[curriculum]
initial_stage = 1
metric_window_size = 30
min_train_episodes_per_stage = 30

[curriculum.stage1_to_stage2]
avg_survival_rate = 0.35
max_wall_collision_rate = 0.025
max_danger_penalty_per_step = 0.015

[curriculum.stage2_to_stage3]
avg_survival_rate = 0.55
min_exploration_score = 0.08
max_idle_penalty_per_step = 0.010
max_dead_end_penalty_per_step = 0.004

[curriculum.stage3_to_stage4]
avg_survival_rate = 0.70
min_treasure_count = 0.80
min_buff_count = 0.20
min_treasure_approach_reward = 0.03
```

---

## 4. 这些指标为什么这样定

### Stage 1 -> Stage 2
目标：先确认“会活、少撞、少贴怪”。

所以看：
- `avg_survival_rate`
- `wall_collision_rate`
- `danger_penalty_per_step`

这一步不看宝箱，不看 buff，不看闪现。

### Stage 2 -> Stage 3
目标：确认“活下来”的同时，已经具备较稳定的探索与脱困质量。

所以看：
- `avg_survival_rate`
- `exploration_score = exploration_reward + centroid_away_reward`
- `idle_penalty_per_step`
- `dead_end_penalty_per_step`

这样能避免“活下来了，但一直在局部绕圈”的假进步。

### Stage 3 -> Stage 4
目标：确认“拿资源”已经融入主线，而不是看到资源就送命。

所以看：
- `avg_survival_rate`
- `treasure_count`
- `buff_count`
- `treasure_approach_reward`

这里的 `treasure_count` / `buff_count` 是用 reward 除以对应 reward coef 反推的估计次数。

---

## 5. 四阶段训练流程总结

### 阶段1：生存底座与基础移动
只开：
- survive_reward
- monster_dist_shaping
- danger_penalty
- wall_collision_penalty

目标：
- 学会合法移动
- 少撞墙
- 不轻易贴怪
- 先把“活下来”学稳

### 阶段2：探索建图与脱困稳定化
在阶段1基础上再开：
- exploration_reward
- centroid_away_reward
- idle_wander_penalty
- dead_end_penalty
- safe_zone_reward

目标：
- 少绕圈
- 少进死角
- 主动扩图
- 生存 + 探索联合优化

同时仍然让 `_is_survival_only_stage()` 为真，避免太早学资源贪心。

### 阶段3：安全前提下的资源获取
在阶段2基础上再开：
- treasure_reward
- treasure_approach_reward
- speed_buff_reward
- speed_buff_approach_reward
- speed_buff_escape_reward

目标：
- 学会安全拿宝箱
- 学会拿 buff 并把它用在脱险上
- 形成“先评估风险，再拿资源”的策略

### 阶段4：技能时机与综合博弈精修
在阶段3基础上再开：
- flash_fail_penalty
- flash_escape_reward
- flash_survival_reward
- flash_abuse_penalty
- late_survive_reward

目标：
- 闪现只在该用时用
- 后期怪物加速阶段也能稳
- 统一平衡 生存 / 探索 / 宝箱 / buff / 技能时机

---

## 6. 一点实现建议

优先顺序建议你按这个来：

1. 先改 `preprocessor.py`
2. 再改 `train_workflow.py`
3. 再补 `train_env_conf.toml`
4. 先跑 stage1/stage2 是否正常切换
5. 再观察 stage2 是否真的保持“探索而不贪资源”
