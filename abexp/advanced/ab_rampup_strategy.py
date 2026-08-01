#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AB 实验开量策略 (Ramp-up Strategy)
=================================
从验证后的"获胜版本"逐步放量到全量的工程实践。

阶段:
  1. AB 验证（5% 流量 / 1 万用户）
     确认显著提升 + 无副作用
  2. 灰度发布 1% → 10% → 25% → 50% → 100%
     每阶段监控核心指标
  3. 长期跟踪（7/30/90 天）
     检测新奇效应衰减

工程实现:
  - Feature Flag 系统
  - 自动告警
  - 一键回滚
"""

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats


# ============================ 阶段定义 ============================

RAMP_UP_STAGES = [
    {"stage": "AB 验证",     "pct": 5,   "duration_days": 7,  "exit_criterion": "p < 0.05 且提升 ≥ 业务阈值"},
    {"stage": "灰度 1%",     "pct": 1,   "duration_days": 2,  "exit_criterion": "监控无异常"},
    {"stage": "灰度 10%",    "pct": 10,  "duration_days": 3,  "exit_criterion": "监控无异常"},
    {"stage": "灰度 25%",    "pct": 25,  "duration_days": 3,  "exit_criterion": "核心指标稳定"},
    {"stage": "灰度 50%",    "pct": 50,  "duration_days": 3,  "exit_criterion": "核心指标稳定"},
    {"stage": "灰度 100%",   "pct": 100, "duration_days": 1,  "exit_criterion": "无故障则上线"},
    {"stage": "长期跟踪 7d",  "pct": 100, "duration_days": 7,  "exit_criterion": "无新奇效应衰减"},
    {"stage": "长期跟踪 30d", "pct": 100, "duration_days": 30, "exit_criterion": "效果稳定"},
    {"stage": "长期跟踪 90d", "pct": 100, "duration_days": 90, "exit_criterion": "效果稳定 = 全量正式上线"},
]


# ============================ 模拟开量过程 ============================

def simulate_rampup(
    baseline_conversion: float = 0.10,
    treatment_lift: float = 0.02,
    novelty_decay: float = 0.005,  # 新奇效应衰减
    days: int = 30,
    daily_traffic: int = 10000,
) -> pd.DataFrame:
    """
    模拟开量过程中的转化率变化

    Args:
        baseline_conversion: 对照组基线转化率
        treatment_lift: 实验组相对提升（绝对值）
        novelty_decay: 新奇效应衰减（每日）
        days: 模拟天数
        daily_traffic: 每日流量

    Returns:
        DataFrame with day, treatment_conversion, control_conversion
    """
    records = []
    rng = np.random.default_rng(20260728)

    for day in range(days):
        # 新奇效应：开量初期效应更大，逐渐衰减到真实效应
        novelty = novelty_decay * max(0, 7 - day)
        true_lift = treatment_lift + novelty

        # 模拟每个用户的转化
        treat_conv = rng.binomial(1, baseline_conversion + true_lift, size=daily_traffic).mean()
        ctrl_conv = rng.binomial(1, baseline_conversion, size=daily_traffic).mean()

        records.append({
            "day": day + 1,
            "treatment_conversion": float(treat_conv),
            "control_conversion": float(ctrl_conv),
            "lift": float(treat_conv - ctrl_conv),
            "novelty": float(novelty),
            "true_lift": float(true_lift),
        })

    return pd.DataFrame(records)


def detect_anomalies(
    daily_data: pd.DataFrame,
    metric_col: str = "treatment_conversion",
    window_days: int = 3,
    sigma_threshold: float = 2.0,
) -> pd.DataFrame:
    """
    检测开量过程中的异常

    Args:
        daily_data: 模拟开量数据
        metric_col: 检测的指标列
        window_days: 滑动窗口
        sigma_threshold: 几 sigma 算异常

    Returns:
        DataFrame with anomaly markers
    """
    df = daily_data.copy()
    rolling_mean = df[metric_col].rolling(window=window_days).mean()
    rolling_std = df[metric_col].rolling(window=window_days).std()

    # 偏离滚动均值 超过 sigma 倍
    df["expected"] = rolling_mean.shift(1)  # 昨天的滚动均值作为预期
    df["z_score"] = (df[metric_col] - df["expected"]) / rolling_std.shift(1).replace(0, np.nan)
    df["is_anomaly"] = df["z_score"].abs() > sigma_threshold

    return df


# ============================ 开量决策引擎 ============================

def auto_rampup_decision(
    current_pct: float,
    daily_data: pd.DataFrame,
    target_pct: float = 100,
    safety_threshold_p: float = 0.01,  # 如果出问题（p > 0.01）就暂停
) -> Dict:
    """
    自动开量决策
    - 检查最近几天是否有异常
    - 决定下一阶段: continue / pause / rollback

    Returns:
        dict with action (continue/pause/rollback), reason, next_pct
    """
    # 检查最近 3 天转化率是否有异常下降
    recent = daily_data.tail(3)
    if len(recent) < 3:
        return {"action": "continue", "reason": "数据不足", "next_pct": current_pct}

    # 简单 SRM 检查：实验组 / 对照组转化率差距
    lift_recent = recent["lift"].mean()
    lift_baseline = daily_data.head(7)["lift"].mean()

    # 异常检测：近期效果下降 > 50%
    if lift_recent < lift_baseline * 0.5 and current_pct > 5:
        return {
            "action": "pause",
            "reason": f"近期转化提升下降：{lift_recent*100:.2f}% vs 基线 {lift_baseline*100:.2f}%",
            "next_pct": current_pct,
            "alert_level": "medium"
        }

    # 如果转化率出现极端下降
    if recent["treatment_conversion"].mean() < recent["control_conversion"].mean() - 0.02:
        return {
            "action": "rollback",
            "reason": "实验组转化低于对照组，可能有副作用",
            "next_pct": 0,
            "alert_level": "critical"
        }

    # 正常：继续放
    next_pct = min(current_pct * 2, target_pct)
    return {
        "action": "continue",
        "reason": f"效果稳定，下一阶段放量到 {next_pct}%",
        "next_pct": next_pct,
        "alert_level": "normal"
    }


# ============================ 演示 ============================

def demo():
    print("=" * 78)
    print(" AB 实验开量策略演示".center(40))
    print("=" * 78)

    print("\n【阶段 1】AB 验证（5% 流量）")
    print("-" * 78)
    for stage in RAMP_UP_STAGES[:1]:
        print(f"  阶段: {stage['stage']}")
        print(f"  流量: {stage['pct']}%")
        print(f"  周期: {stage['duration_days']} 天")
        print(f"  通过标准: {stage['exit_criterion']}")

    print("\n【阶段 2】灰度发布 1% → 100%")
    print("-" * 78)
    for stage in RAMP_UP_STAGES[1:6]:
        print(f"  ✓ {stage['stage']:<15} ({stage['pct']:>3}%)  →  {stage['exit_criterion']}")

    print("\n【阶段 3】长期跟踪")
    print("-" * 78)
    for stage in RAMP_UP_STAGES[6:]:
        print(f"  ✓ {stage['stage']:<15} ({stage['duration_days']:>2}d)  →  {stage['exit_criterion']}")

    print("\n【模拟】开量过程数据")
    print("-" * 78)
    df = simulate_rampup(
        baseline_conversion=0.10,
        treatment_lift=0.02,
        novelty_decay=0.005,
        days=14,
    )
    print(f"  对照组转化率: {df['control_conversion'].mean()*100:.2f}%")
    print(f"  实验组转化率: {df['treatment_conversion'].mean()*100:.2f}%")
    print(f"  平均提升: {df['lift'].mean()*100:.2f}% (真实: {df['true_lift'].iloc[7]*100:.2f}%)")
    print(f"  新奇效应: 第 1 天额外提升 {df['novelty'].iloc[0]*100:.2f}% (模拟)")

    print("\n【自动开量决策】")
    print("-" * 78)
    # 模拟 7 天后的决策
    decision = auto_rampup_decision(current_pct=10, daily_data=df.head(7))
    for k, v in decision.items():
        print(f"  {k}: {v}")

    print("\n【关键建议】")
    print(" 1. 不要直接 100% 开量，会放大任何潜在问题")
    print(" 2. 每阶段都要确保监控到位，1% 阶段故障只影响 1% 用户")
    print(" 3. 长期跟踪关注新奇效应：开量初期效果好，30 天后归零很常见")
    print(" 4. 90 天后效果稳定 = 全量正式上线")


if __name__ == "__main__":
    demo()
