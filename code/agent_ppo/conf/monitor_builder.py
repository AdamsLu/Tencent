#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Monitor panel configuration builder for Gorge Chase.
峡谷追猎监控面板配置构建器。
"""


from kaiwudrl.common.monitor.monitor_config_builder import MonitorConfigBuilder


def build_monitor():
    """
    # This function is used to create monitoring panel configurations for custom indicators.
    # 该函数用于创建自定义指标的监控面板配置。
    """
    monitor = MonitorConfigBuilder()
    monitor.title("峡谷追猎")

    monitor.add_group(group_name="算法指标", group_name_en="algorithm")
    algorithm_panels = [
        ("累积回报", "reward"),
        ("总损失", "total_loss"),
        ("价值损失", "value_loss"),
        ("策略损失", "policy_loss"),
        ("熵损失", "entropy_loss"),
        ("解释方差", "explained_variance"),
        ("Clip次数", "clip_count"),
        ("Clip率", "clip_rate"),
        ("Clip越界绝对值", "clip_abs_overflow"),
        ("优势绝对均值", "adv"),
        ("优势均值", "adv_mean"),
        ("优势标准差", "adv_std"),
    ]
    for panel_name, metric_name in algorithm_panels:
        monitor.add_panel(name=panel_name, name_en=metric_name, type="line")
        monitor.add_metric(metrics_name=metric_name, expr=f"avg({metric_name}{{}})")
        monitor.end_panel()
    monitor.end_group()

    monitor.add_group(group_name="奖励分项", group_name_en="reward_components")
    reward_panels = [
        ("生存奖励", "survive_reward"),
        ("宝箱分奖励", "treasure_reward"),
        ("Buff奖励", "speed_buff_reward"),
        ("Buff靠近奖励", "speed_buff_approach_reward"),
        ("宝箱接近奖励", "treasure_approach_reward"),
        ("怪物距离Shaping", "monster_dist_shaping"),
        ("后期生存奖励", "late_survive_reward"),
        ("危险惩罚", "danger_penalty"),
        ("撞墙惩罚", "wall_collision_penalty"),
        ("闪现失败惩罚", "flash_fail_penalty"),
        ("危险闪现成功奖励", "flash_escape_reward"),
        ("闪现衰减存活奖励", "flash_survival_reward"),
        ("加速逃离奖励", "speed_buff_escape_reward"),
        ("安全区奖励", "safe_zone_reward"),
        ("闪现滥用惩罚", "flash_abuse_penalty"),
        ("闪现被抓惩罚", "flash_abuse_penalty_caught"),
        ("开图奖励", "exploration_reward"),
        ("坐标访问奖励", "visit_tracking_reward"),
        ("质心远离奖励", "centroid_away_reward"),
        ("原地徘徊惩罚", "idle_wander_penalty"),
        ("死角死路惩罚", "dead_end_penalty"),
        ("无移动次数", "no_movement_case"),
        ("MoveMask一致率", "move_mask_consistency_rate"),
    ]
    for panel_name, metric_name in reward_panels:
        monitor.add_panel(name=panel_name, name_en=metric_name, type="line")
        monitor.add_metric(metrics_name=metric_name, expr=f"avg({metric_name}{{}})")
        monitor.end_panel()
    monitor.end_group()

    monitor.add_group(group_name="评估指标", group_name_en="evaluation_metrics")
    eval_panels = [
        ("游戏分数", "sim_score"),
        ("评估局胜率", "eval_win_rate"),
        ("训练-评估模式", "mode"),
        ("地图ID", "map_id"),
        ("训练地图ID", "train_map_id"),
        ("地图总数训评", "total_map"),
        ("训练可用地图数", "train_map_pool_size"),
        ("评估可用地图数", "eval_map_pool_size"),
    ]
    for panel_name, metric_name in eval_panels:
        monitor.add_panel(name=panel_name, name_en=metric_name, type="line")
        monitor.add_metric(metrics_name=metric_name, expr=f"avg({metric_name}{{}})")
        monitor.end_panel()
    monitor.end_group()

    config_dict = monitor.build()
    return config_dict
