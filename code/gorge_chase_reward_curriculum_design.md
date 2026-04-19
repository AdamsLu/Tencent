# 峡谷追猎 PPO：奖励设计、课程阶段与阈值整理

## 1. 说明与范围

本文档分成两部分：

1. **当前代码实际生效设计**：按仓库 `main` 分支当前实现整理。
2. **建议版课程奖励/阈值设计**：这是基于你当前训练现象与阶段目标给出的改进方案，不是仓库现成配置。

> 重要说明：当前代码里，**阶段之间的奖励控制主要是“启用/关闭”**，而不是“每个阶段一套独立权重表”。也就是说，`reward_conf.toml` 里只有一套全局参数，阶段切换由 `preprocessor.py` 里的 `stage_reward_map` 决定哪些奖励项在当前阶段生效。若你要采用本文后半部分的“每阶段不同权重设计”，需要把 `reward_conf` 扩展成 stage-specific 配置，或在 `_compute_rewards` 中增加 stage multiplier。

---

## 2. 当前代码中的奖励相关项总表

### 2.1 监控面板中的奖励项

当前训练工作流会累计并上报如下奖励项：

- survive_reward
- treasure_reward
- speed_buff_reward
- speed_buff_approach_reward
- treasure_approach_reward
- monster_dist_shaping
- late_survive_reward
- danger_penalty
- wall_collision_penalty
- flash_fail_penalty
- flash_escape_reward
- flash_survival_reward
- speed_buff_escape_reward
- safe_zone_reward
- flash_abuse_penalty
- flash_abuse_penalty_caught
- exploration_reward
- centroid_away_reward
- idle_wander_penalty
- dead_end_penalty

此外，`train_workflow.py` 还会在终局额外并入：

- **失败终局奖励**：`-50`
- **成功终局奖励**：`+50`

---

## 3. 奖励设计项与对应公式

下面分为两类：

- **A 类：代码配置与注释可直接确认的公式/结构**
- **B 类：根据代码注释、变量命名与监控逻辑重构的设计公式**

> 说明：GitHub 网页对大文件的逐行抽取不稳定，`_compute_rewards` 的每一行没法完整可靠地网页级复原；因此个别项我按代码注释、配置字段和训练工作流里的统计方式做了重构。用于调参和课程设计是足够的；若你后续要做“逐行复现”或写论文附录，建议你本地再直接 grep `_compute_rewards` 二次核对。

### 3.1 生存与安全类

| 奖励项 | 设计公式 | 说明 |
|---|---|---|
| survive_reward | `r_survive = c_survive` | 每步固定稠密生存奖励。 |
| monster_dist_shaping | `r_monster = c_monster * f(d_monster)` | 根据与最近怪物距离做稠密 shaping；从监控曲线看主要提供正向安全引导。可理解为“距离越安全，奖励越高”或“距离增益越大，奖励越高”。 |
| danger_penalty | `r_danger = c_danger * max(0, danger_threshold - d_monster)^power` | 代码里明确有 `danger_threshold` 与 `power`。这是典型的阈值内幂次惩罚。 |
| late_survive_reward | `r_late = base + shaping_coef * g(d_monster, t)` | 后期生存奖励，设计上用于中后段继续强调稳态存活。 |
| safe_zone_reward | `r_safe_zone = c_safe_zone * 1[safe_zone]` | 安全区驻留稠密奖励。 |
| wall_collision_penalty | `r_wall = c_wall * 1[位移不足/无效移动]` | 撞墙或无效移动惩罚。代码里有 `wall_displacement_threshold`。 |
| dead_end_penalty | `r_dead_end = c_dead_end * 1[dead_end_active]` | 进入死角/死路后持续惩罚，远离锚点后重置。 |

### 3.2 资源获取类

| 奖励项 | 设计公式 | 说明 |
|---|---|---|
| treasure_reward | `r_treasure = c_treasure * ΔT` | 获取宝箱的稀疏奖励，`ΔT` 为本步新增宝箱数。 |
| speed_buff_reward | `r_buff = c_buff * ΔB` | 获取加速 buff 的稀疏奖励，`ΔB` 为本步新增 buff 数。 |
| speed_buff_approach_reward | `r_buff_appr = c_buff_appr * max(0, d_buff(t-1) - d_buff(t))` | 靠近 buff 的稠密奖励。 |
| treasure_approach_reward | `r_treasure_appr = clip(g / max(d_treasure^2, d_min^2), r_min, r_max)` | 代码注释明确写的是“平方反比引力”；当前配置由 `gravity_coef / min_reward / max_reward / min_dist_norm` 控制。 |
| speed_buff_escape_reward | `r_buff_escape(t) = decay_state * (base + dist_delta_coef * Δd_monster)` | 加速期间逃离额外奖励；有 `init`、`decay`、`base`、`dist_delta_coef`。 |

### 3.3 探索与反发呆类

| 奖励项 | 设计公式 | 说明 |
|---|---|---|
| exploration_reward | `r_explore = coef_per_cell * new_explored_cells` | 出生保护期后，按新增开图格数给奖励。 |
| centroid_away_reward | `r_centroid = c_centroid * dist(hero, centroid(trajectory)) / centroid_norm_max` | 轨迹质心远离奖励，鼓励远离最近轨迹中心，减少局部绕圈。 |
| idle_wander_penalty | `r_idle = idle_coef * (1 + idle_growth * idle_streak) + wander_coef * (1 + wander_growth * wander_streak)` | 原地不动/小范围徘徊惩罚，持续越久惩罚越大。 |

### 3.4 闪现技能类

| 奖励项 | 设计公式 | 说明 |
|---|---|---|
| flash_fail_penalty | `r_flash_fail = c_flash_fail * 1[actual_flash_dist < flash_fail_ratio * expected_flash_dist]` | 闪现距离明显不足时给惩罚。代码里有 `flash_fail_ratio`、正交/对角期望闪现距离。 |
| flash_escape_reward | `r_flash_escape = base + dist_gain_coef * (d_after - d_before)` | 危险时成功闪现后的延迟奖励，在 10% 冷却窗口后结算。 |
| flash_survival_reward | `r_flash_survival(t) = init * decay^(t-1)` | 闪现脱险后的衰减存活奖励。 |
| flash_abuse_penalty | `r_flash_abuse = c_caught 或 c_safe_zone_no_treasure` | 闪现滥用惩罚：如短时间内被抓，或在安全区无收益乱闪。 |
| flash_abuse_penalty_caught | `r_flash_abuse_caught(k) = linear_interp(c_start, c_end, k)` | 当前配置中有 caught decay，说明被抓惩罚会随“闪后到被抓的步数”做线性衰减。 |

### 3.5 终局奖励

| 奖励项 | 设计公式 | 说明 |
|---|---|---|
| final_reward_fail | `-50` | 终局失败直接并入训练 reward。 |
| final_reward_win | `+50` | 终局成功直接并入训练 reward。 |

---

## 4. 当前代码里所有奖励参数（实际生效值）

### 4.1 全局 shaping 常量

| 参数 | 当前值 |
|---|---:|
| birth_protection_steps | 10 |
| trajectory_window | 20 |
| safe_treasure_monster_dist | 0.2 |
| danger_threshold | 0.15 |
| late_danger_threshold | 0.25 |
| wall_displacement_threshold | 0.5 |
| normal_move_step | 1 |
| buff_move_step | 2 |
| idle_displacement_threshold | 0.25 |
| wander_radius_threshold | 2.5 |
| wander_min_points | 8 |
| idle_wander_reset_distance | 6.0 |
| dead_end_check_radius | 8 |
| dead_end_boundary_trace_steps | 6 |
| dead_end_opening_clusters_threshold | 1 |
| dead_end_max_reachable_cells | 90 |
| dead_end_max_reachable_ratio | 0.45 |
| dead_end_reset_distance | 8.0 |
| flash_fail_ratio | 0.3 |
| flash_expected_dist_orthogonal | 10.0 |
| flash_expected_dist_diagonal | 8.0 |
| centroid_norm_max | 64.0 |

### 4.2 奖励参数表

| 奖励项 | 当前参数 |
|---|---|
| survive_reward | `coef = 0.5` |
| treasure_reward | `coef = 2.0` |
| speed_buff_reward | `coef = 2.0` |
| speed_buff_approach_reward | `coef = 0.2` |
| treasure_approach_reward | `gravity_coef = 0.00005`, `min_reward = 0.0001`, `max_reward = 0.05`, `min_dist_norm = 0.005` |
| monster_dist_shaping | `coef = 20.0` |
| late_survive_reward | `base = 0.02`, `shaping_coef = 0.1` |
| danger_penalty | `coef = -0.1`, `power = 2.0` |
| wall_collision_penalty | `coef = -0.1` |
| flash_fail_penalty | `coef = -0.15` |
| flash_escape_reward | `base = 1.0`, `dist_gain_coef = 1.0` |
| flash_survival_reward | `init = 0.5`, `decay = 0.90` |
| flash_abuse_penalty | `caught_within_steps = 15`, `caught_coef = -1.0`, `safe_zone_no_treasure_coef = -1.0`, `caught_decay_enable = true`, `caught_decay_min_step = 1`, `caught_decay_max_step = 15`, `caught_coef_start = -10.0`, `caught_coef_end = -1.0` |
| speed_buff_escape_reward | `init = 0.05`, `decay = 0.97`, `base = 1.0`, `dist_delta_coef = 0.1` |
| safe_zone_reward | `coef = 0.01` |
| exploration_reward | `coef_per_cell = 0.0002` |
| centroid_away_reward | `coef = 0.05` |
| idle_wander_penalty | `idle_coef = -0.03`, `wander_coef = -0.05`, `idle_growth = 0.02`, `wander_growth = 0.02` |
| dead_end_penalty | `enable = false`, `coef = -0.1` |

---

## 5. 当前课程阶段与激活奖励

### 阶段命名

| 阶段 | 名称 |
|---|---|
| 1 | survival_base |
| 2 | explore_and_stabilize |
| 3 | safe_resource_acquisition |
| 4 | full_game_and_skill_refine |

### 5.1 阶段 1：survival_base

激活奖励：

- survive_reward
- monster_dist_shaping
- danger_penalty
- wall_collision_penalty

目标：

- 先学会活下来
- 会躲怪
- 会基础移动
- 少撞墙

### 5.2 阶段 2：explore_and_stabilize

在阶段 1 基础上新增：

- exploration_reward
- centroid_away_reward
- idle_wander_penalty
- dead_end_penalty
- safe_zone_reward

目标：

- 学会开图
- 减少原地抖动
- 减少局部绕圈
- 脱困更稳定

### 5.3 阶段 3：safe_resource_acquisition

在阶段 2 基础上新增：

- treasure_reward
- treasure_approach_reward
- speed_buff_reward
- speed_buff_approach_reward
- speed_buff_escape_reward

目标：

- 在安全前提下，开始主动拿资源
- 学会接近宝箱与 buff
- 把“探索”转化为“资源收割”

### 5.4 阶段 4：full_game_and_skill_refine

在阶段 3 基础上新增：

- flash_fail_penalty
- flash_escape_reward
- flash_survival_reward
- flash_abuse_penalty
- late_survive_reward

目标：

- 学会技能释放时机
- 学会高风险场景下的综合博弈
- 学会中后期收官

### 5.5 当前代码机制上的一个关键事实

当前代码在阶段 1/2 会压制资源导向特征，阶段 3/4 才打开资源相关观测与动作收益特征。也就是说，**前两阶段不仅是奖励没开，连观测/动作偏置也有意做了资源抑制**。

---

## 6. 当前代码中的阶段切换指标与阈值（实际生效）

### 6.1 窗口与最小训练局数

| 项目 | 当前值 |
|---|---:|
| initial_stage | 1 |
| metric_window_size | 20 |
| min_train_episodes_per_stage | 20 |

### 6.2 阶段切换时实际统计的指标

每个训练局结束后，工作流会构造并加入滚动窗口的指标：

- `survival_rate = step / max_step`
- `wall_collision_rate = estimated_wall_collisions / step`
- `danger_penalty_per_step = |danger_penalty| / step`
- `idle_penalty_per_step = |idle_wander_penalty| / step`
- `dead_end_penalty_per_step = |dead_end_penalty| / step`
- `exploration_score = exploration_reward + centroid_away_reward`
- `treasure_count = |treasure_reward| / coef_treasure_reward`
- `buff_count = |speed_buff_reward| / coef_speed_buff_reward`
- `treasure_approach_reward = 累积 treasure_approach_reward`
- `sim_score = 环境总分`

### 6.3 当前生效阈值

#### 阶段 1 -> 阶段 2

| 指标 | 当前阈值 |
|---|---:|
| avg_survival_rate | `>= 0.22` |
| max_wall_collision_rate | `<= 0.08` |
| max_danger_penalty_per_step | `<= 0.035` |

#### 阶段 2 -> 阶段 3

| 指标 | 当前阈值 |
|---|---:|
| avg_survival_rate | `>= 0.38` |
| min_exploration_score | `>= 0.03` |
| max_idle_penalty_per_step | `<= 0.03` |
| max_dead_end_penalty_per_step | `<= 0.02` |

#### 阶段 3 -> 阶段 4

| 指标 | 当前阈值 |
|---|---:|
| avg_survival_rate | `>= 0.52` |
| min_treasure_count | `>= 0.30` |
| min_buff_count | `>= 0.08` |
| min_treasure_approach_reward | `>= 0.01` |

### 6.4 当前阈值体系的一个问题

你当前代码里 `dead_end_penalty` 是 `enable = false`，但阶段 2 -> 3 又使用 `dead_end_penalty_per_step` 做切换判断。这样一来这项指标几乎没有约束力，等于门槛里有一项实际上接近失效。建议要么启用 dead-end 惩罚，要么删掉该切换指标。

---

## 7. 结论：当前代码的真实课程机制

### 当前代码真实策略

- **阶段之间主要靠“开关奖励项”推进，不靠阶段权重重分配。**
- **所有奖励参数当前只有一套全局值。**
- **阶段切换按 20 局滚动均值判断。**
- **切阶段后窗口清空，阶段训练局数重置。**

这套设计的优点是简单、稳定。

缺点是：

1. 阶段 3 开始时，资源奖励虽然打开了，但前期学到的“保命惯性”很强；如果不降低一部分安全/探索项相对权重，模型容易停留在“安全探索 + 顺手拿资源”，而不是“主动规划拿资源”。
2. 阶段 4 想学闪现，仅靠打开技能奖励常常不够，因为正样本太稀疏，反而更容易先学会“别乱闪”。

---

## 8. 建议版：每阶段奖励权重设计

## 8.1 设计原则

我建议你不要再采用“全阶段共用一套参数，只做启停”的方式，而改成：

- **阶段 1：安全底盘最强**
- **阶段 2：探索最强，安全仍为硬约束**
- **阶段 3：资源获取成为主导，安全改为约束项，不再是主导项**
- **阶段 4：技能收益与技能误用惩罚一起打开，但正向技能奖励必须足够可学**

下面我给的是一套**建议的 stage-specific 参数表**。你可以直接照着改成 4 份配置，或在代码里增加 stage multiplier。

---

## 8.2 建议参数总表（按阶段）

### A. 生存与安全类

| 奖励项 | 当前值 | 建议 S1 | 建议 S2 | 建议 S3 | 建议 S4 | 设计意图 |
|---|---:|---:|---:|---:|---:|---|
| survive_reward.coef | 0.5 | 0.60 | 0.45 | 0.35 | 0.30 | 前期活下来最重要，后期让位于资源/技能。 |
| monster_dist_shaping.coef | 20.0 | 20.0 | 14.0 | 8.0 | 6.0 | 逐阶段降权，避免阶段 3/4 过度保守。 |
| danger_penalty.coef | -0.1 | -0.12 | -0.12 | -0.12 | -0.14 | 一直保强，作为硬风险约束。 |
| danger_penalty.power | 2.0 | 2.0 | 2.0 | 2.0 | 2.2 | 阶段 4 对贴怪更敏感。 |
| wall_collision_penalty.coef | -0.1 | -0.12 | -0.10 | -0.08 | -0.08 | 前期矫正移动，后期不让它干扰资源路线微调。 |
| safe_zone_reward.coef | 0.01 | 0.00 | 0.015 | 0.008 | 0.005 | 只在阶段 2 稍强，阶段 3/4 降低，防止赖在安全区。 |
| late_survive_reward.base | 0.02 | 0.00 | 0.00 | 0.01 | 0.02 | 只在后期逐渐启用。 |
| late_survive_reward.shaping_coef | 0.1 | 0.00 | 0.00 | 0.05 | 0.08 | 阶段 4 更强调中后盘收官。 |
| dead_end_penalty.coef | -0.1（未启用） | 0.00 | -0.04 | -0.06 | -0.08 | 建议从阶段 2 起启用，轻量约束死路。 |

### B. 探索类

| 奖励项 | 当前值 | 建议 S1 | 建议 S2 | 建议 S3 | 建议 S4 | 设计意图 |
|---|---:|---:|---:|---:|---:|---|
| exploration_reward.coef_per_cell | 0.0002 | 0.00 | 0.00025 | 0.00012 | 0.00008 | 阶段 2 主导开图；阶段 3/4 降权，避免为探索而探索。 |
| centroid_away_reward.coef | 0.05 | 0.00 | 0.06 | 0.03 | 0.02 | 阶段 2 提高离开旧轨迹倾向；阶段 3/4 降低。 |
| idle_wander_penalty.idle_coef | -0.03 | 0.00 | -0.03 | -0.03 | -0.025 | 阶段 2/3 保留，压制原地抖动。 |
| idle_wander_penalty.wander_coef | -0.05 | 0.00 | -0.05 | -0.05 | -0.04 | 防止模型一直在安全区外缘绕圈。 |
| idle_wander_penalty.idle_growth | 0.02 | 0.00 | 0.02 | 0.02 | 0.015 | 后期稍微放松增长，避免过拟合追求“永远在动”。 |
| idle_wander_penalty.wander_growth | 0.02 | 0.00 | 0.02 | 0.02 | 0.015 | 同上。 |

### C. 资源类

| 奖励项 | 当前值 | 建议 S1 | 建议 S2 | 建议 S3 | 建议 S4 | 设计意图 |
|---|---:|---:|---:|---:|---:|---|
| treasure_reward.coef | 2.0 | 0.0 | 0.0 | 3.0 | 3.5 | 阶段 3 开始把拿箱子做成主目标。 |
| treasure_approach_reward.gravity_coef | 0.00005 | 0.0 | 0.0 | 0.00008 | 0.00010 | 阶段 3/4 增强主动接近宝箱。 |
| treasure_approach_reward.min_reward | 0.0001 | 0.0 | 0.0 | 0.0002 | 0.00025 | 保证远距离也有弱引导。 |
| treasure_approach_reward.max_reward | 0.05 | 0.0 | 0.0 | 0.08 | 0.10 | 近距离收束时让引导更明显。 |
| treasure_approach_reward.min_dist_norm | 0.005 | 0.005 | 0.005 | 0.005 | 0.005 | 这个一般不动。 |
| speed_buff_reward.coef | 2.0 | 0.0 | 0.0 | 1.0 | 1.2 | buff 要有价值，但不要压过宝箱主线。 |
| speed_buff_approach_reward.coef | 0.2 | 0.0 | 0.0 | 0.08 | 0.10 | 给一点路线引导即可，不宜过强。 |
| speed_buff_escape_reward.init | 0.05 | 0.0 | 0.0 | 0.08 | 0.10 | 有 buff 后逃生更有收益。 |
| speed_buff_escape_reward.decay | 0.97 | 0.97 | 0.97 | 0.97 | 0.96 | 阶段 4 更鼓励“拿到 buff 立刻创造优势”。 |
| speed_buff_escape_reward.base | 1.0 | 0.0 | 0.0 | 0.8 | 1.0 | 阶段 3 适中，阶段 4 恢复强度。 |
| speed_buff_escape_reward.dist_delta_coef | 0.1 | 0.0 | 0.0 | 0.08 | 0.10 | 让“拿 buff 后拉开怪物距离”成为可学信号。 |

### D. 闪现技能类

| 奖励项 | 当前值 | 建议 S1 | 建议 S2 | 建议 S3 | 建议 S4 | 设计意图 |
|---|---:|---:|---:|---:|---:|---|
| flash_fail_penalty.coef | -0.15 | 0.0 | 0.0 | 0.0 | -0.20 | 只在阶段 4 启用。 |
| flash_escape_reward.base | 1.0 | 0.0 | 0.0 | 0.0 | 1.2 | 阶段 4 正向奖励要够强，不然只会学会别闪。 |
| flash_escape_reward.dist_gain_coef | 1.0 | 0.0 | 0.0 | 0.0 | 1.0 | 保持即可。 |
| flash_survival_reward.init | 0.5 | 0.0 | 0.0 | 0.0 | 0.35 | 不建议太高，否则会鼓励“闪后苟住”而不是“闪后转收益”。 |
| flash_survival_reward.decay | 0.90 | 0.90 | 0.90 | 0.90 | 0.92 | 稍慢衰减。 |
| flash_abuse_penalty.caught_coef | -1.0 | 0.0 | 0.0 | 0.0 | -1.0 | 阶段 4 开启即可。 |
| flash_abuse_penalty.safe_zone_no_treasure_coef | -1.0 | 0.0 | 0.0 | 0.0 | -0.6 | 安全区乱闪要罚，但别罚过头。 |
| flash_abuse_penalty.caught_coef_start | -10.0 | 0.0 | 0.0 | 0.0 | -6.0 | 当前起始惩罚偏重，建议减小。 |
| flash_abuse_penalty.caught_coef_end | -1.0 | 0.0 | 0.0 | 0.0 | -1.0 | 保持尾部惩罚。 |
| flash_abuse_penalty.caught_within_steps | 15 | 15 | 15 | 15 | 12 | 缩短窗口，提高“近期乱闪被抓”的识别精度。 |

---

## 9. 建议版阶段切换阈值设计

当前阈值整体偏“尽快切阶段”。如果你想让每个阶段学得更扎实，我建议采用下面这套：

### 9.1 建议的窗口设置

| 项目 | 当前 | 建议 |
|---|---:|---:|
| metric_window_size | 20 | 25 |
| min_train_episodes_per_stage | 20 | 25 |

理由：

- 20 局窗口对单局波动仍然比较敏感。
- 25 局对这种地图+探索+避怪任务更稳。

### 9.2 建议阈值

#### 阶段 1 -> 阶段 2

| 指标 | 当前 | 建议 |
|---|---:|---:|
| avg_survival_rate | 0.22 | **0.25** |
| max_wall_collision_rate | 0.08 | **0.06** |
| max_danger_penalty_per_step | 0.035 | **0.030** |

解释：

- 阶段 2 会打开探索项；如果阶段 1 的保命底盘不够稳，阶段 2 很容易学歪成乱跑。
- 所以 S1->S2 建议略收紧。

#### 阶段 2 -> 阶段 3

| 指标 | 当前 | 建议 |
|---|---:|---:|
| avg_survival_rate | 0.38 | **0.40** |
| min_exploration_score | 0.03 | **0.05** |
| max_idle_penalty_per_step | 0.03 | **0.020** |
| max_dead_end_penalty_per_step | 0.02 | **0.010** |

解释：

- 阶段 3 一旦打开资源项，如果阶段 2 的探索稳定性不够，会变成“乱探 + 偶遇资源”。
- 所以必须要求 agent 已经学会“有方向地移动”，而不是单纯增加位移。

#### 阶段 3 -> 阶段 4

| 指标 | 当前 | 建议 |
|---|---:|---:|
| avg_survival_rate | 0.52 | **0.50** |
| min_treasure_count | 0.30 | **0.55** |
| min_buff_count | 0.08 | **0.15** |
| min_treasure_approach_reward | 0.01 | **0.015** |

解释：

- 这里我反而把 survival_rate 稍微放松一点，但把资源指标明显抬高。
- 原因是阶段 4 的核心不该只是“活得更久”，而是“会在复杂场景下做技能+资源+生存的综合决策”。
- 如果 S3 没有把宝箱和 buff 获取学扎实，S4 往往会变成“保守生存 + 不会技能”。

---

## 10. 我对你的最终建议

### 10.1 如果你现在只能做最小改动

那我建议你先只做三件事：

1. **阶段 3 开始，把 `monster_dist_shaping`、`exploration_reward`、`centroid_away_reward` 降权。**
2. **阶段 3 开始，把 `treasure_reward` 和 `treasure_approach_reward` 提权。**
3. **把 `dead_end_penalty` 真正启用，不然阶段 2 -> 3 的 dead-end 门槛没有意义。**

### 10.2 如果你愿意做一次完整升级

建议你把 `reward_conf.toml` 扩展成：

```toml
[reward.stage1.survive_reward]
coef = 0.60

[reward.stage2.survive_reward]
coef = 0.45

[reward.stage3.survive_reward]
coef = 0.35

[reward.stage4.survive_reward]
coef = 0.30
```

然后 `_cfg()` 先查 `reward.stage{curr_stage}`，查不到再回退到全局 `reward.xxx`。这样你的课程式训练就会从“开关式课程”升级成“开关 + 重加权课程”。

### 10.3 从策略塑形角度的一句话总结

- **阶段 1：让 agent 会活。**
- **阶段 2：让 agent 会动、会探、不会瞎晃。**
- **阶段 3：让 agent 会为了拿资源而规划路线。**
- **阶段 4：让 agent 学会在风险下用技能换收益。**

如果阶段 3 还保留太强的安全/探索塑形，模型就会停在“保守巡航”；
如果阶段 4 只加技能惩罚、不加足够强的技能成功正反馈，模型就会停在“别乱用技能”。

---

## 11. 可直接抄到配置里的建议版阈值摘要

```toml
[curriculum]
initial_stage = 1
metric_window_size = 25
min_train_episodes_per_stage = 25

[curriculum.stage1_to_stage2]
avg_survival_rate = 0.25
max_wall_collision_rate = 0.06
max_danger_penalty_per_step = 0.030

[curriculum.stage2_to_stage3]
avg_survival_rate = 0.40
min_exploration_score = 0.05
max_idle_penalty_per_step = 0.020
max_dead_end_penalty_per_step = 0.010

[curriculum.stage3_to_stage4]
avg_survival_rate = 0.50
min_treasure_count = 0.55
min_buff_count = 0.15
min_treasure_approach_reward = 0.015
```

---

## 12. 可直接执行的落地顺序

1. 先把 **dead_end_penalty 启用**。
2. 再做 **阶段 3 的重加权**：降安全/探索，升 treasure 主线。
3. 然后再做 **阶段 4 技能学习**：提高正向闪现收益，适度保留误用惩罚。
4. 最后再收紧阈值，避免过早切阶段。

---

## 13. 来源文件（当前代码）

- `code/agent_ppo/feature/preprocessor.py`
- `code/agent_ppo/conf/reward_conf.toml`
- `code/agent_ppo/conf/train_env_conf.toml`
- `code/agent_ppo/workflow/train_workflow.py`
- `code/agent_ppo/agent.py`

