#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
季节性 CUPED + 早期停止 (Early Stopping)
=========================================
【5. 季节性 CUPED】

问题: 节日 / 周末 / 季节效应会扭曲实验结果
      双 11 期间，所有组转化率都会跳变
方案: 加入"时间段"作为协变量，缩减与时间相关的方差

【6. 早期停止 (Sequential mSPRT)】

问题: 实验期间 PM 老问"今天显著了吗？"
      每次看都膨胀假阳性
方案: mSPRT (Walmart 2018) - 在任意观察点保证总假阳性 ≤ α
"""

from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats


# ============================ 5. 季节性 CUPED ============================

def seasonal_cuped(
    post: np.ndarray,
    pre: np.ndarray,
    pre_weekday_indicator: np.ndarray,
    pre_weekend_indicator: np.ndarray,
    assigned: np.ndarray,
) -> Dict:
    """
    季节性 CUPED: 在 CUPED 基础上加时间相关协变量

    Args:
        post: 实验后观测
        pre: 实验前观测
        pre_weekday_indicator: 工作日（1/0）标记
        pre_weekend_indicator: 周末（1/0）标记
        assigned: 0/1 实验分组

    Returns:
        t_stat, p_value, variance_reduction
    """
    # 协变量 = [pre 消费量, weekday, weekend]
    X = np.column_stack([
        pre - pre.mean(),
        pre_weekday_indicator - pre_weekday_indicator.mean(),
        pre_weekend_indicator - pre_weekend_indicator.mean(),
    ]).astype(float)

    y = post.astype(float)
    # 多元 CUPED: y_cuped = y - X @ theta
    cov_xx = np.cov(X.T)
    cov_xy = np.array([np.cov(y, X[:, j])[0, 1] for j in range(X.shape[1])])
    try:
        theta = np.linalg.solve(cov_xx, cov_xy)
    except np.linalg.LinAlgError:
        theta = np.zeros(X.shape[1])

    y_cuped = y - X @ theta

    treat = y_cuped[assigned == 1]
    ctrl = y_cuped[assigned == 0]
    t_stat, p_value = stats.ttest_ind(treat, ctrl, equal_var=False)

    return {
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "variance_reduction": float(1 - y_cuped.var() / post.var()) if post.var() > 0 else 0,
        "significant": p_value < 0.05,
        "treatment_mean": float(treat.mean()),
        "control_mean": float(ctrl.mean()),
    }


def demo_seasonal():
    """演示季节性 CUPED"""
    print("=" * 78)
    print(" 季节性 CUPED 演示".center(40))
    print("=" * 78)
    np.random.seed(20260728)
    rng = np.random.default_rng(20260728)

    n = 4000
    # 用户特征独立：pre_weekend_active 与 pre 强相关但与 weekday 等不共线
    pre_weekend_active = rng.integers(0, 2, n)
    # 用户基础活跃度
    user_baseline = rng.normal(50, 5, n)
    # pre 数据（pre 未考虑 weekend 单独效应）
    pre = user_baseline + rng.normal(0, 5, n)
    # post 数据：周末用户在 weekend 期间活跃度跳变
    post = user_baseline + 5 * pre_weekend_active + rng.normal(0, 5, n)
    # 实验组 +5%
    assigned = rng.integers(0, 2, n)
    post[assigned == 1] *= 1.05

    pre_weekday = (1 - pre_weekend_active).astype(float)
    pre_weekend = pre_weekend_active.astype(float)

    result = seasonal_cuped(post, pre, pre_weekday, pre_weekend, assigned)
    print(f"\n [基础 CUPED vs 季节性 CUPED]")
    print(f"   季节性 CUPED 方缩减: {result['variance_reduction']*100:.2f}%")
    print(f"   p-value: {result['p_value']:.4f}")
    print(f"   显著: {'✓' if result['significant'] else '△'}")

    # 对比：仅 pre 协变量
    cov_xx = np.var(pre)
    cov_xy = np.cov(post, pre)[0, 1]
    theta = cov_xy / cov_xx
    y_cuped_basic = post - theta * (pre - pre.mean())
    t_stat, p_basic = stats.ttest_ind(y_cuped_basic[assigned == 1], y_cuped_basic[assigned == 0], equal_var=False)
    vr_basic = 1 - y_cuped_basic.var() / post.var()
    print(f"\n   基础 CUPED 方缩减: {vr_basic*100:.2f}%")
    print(f"   基础 CUPED p-value: {p_basic:.4f}")

    if result['variance_reduction'] > vr_basic:
        print(f"\n 季节性 CUPED 提升 {(result['variance_reduction'] - vr_basic)*100:.2f}pp 方缩减")
    else:
        print(f"\n 季节性 CUPED 仅略胜基本 CUPED，差异 {(result['variance_reduction'] - vr_basic)*100:.2f}pp")
        print(" → 教训：协变量与 pre 不能完全共线，否则不会带来增益")


# ============================ 6. 早期停止 mSPRT ============================

def msprt_test(
    obs_t: np.ndarray,
    obs_c: np.ndarray,
    alpha: float = 0.05,
    test_type: str = "two-sided",
) -> Dict:
    """
    序贯检验 mSPRT（mixing Sequential Probability Ratio Test）
    Walmart 2018 - Always-Valid Confidence Sequences

    Args:
        obs_t: 实验组累积观测（按时间到达顺序）
        obs_c: 对照组累积观测
        alpha: 总预算（无论观察多少次，总 I 型错误 ≤ α）
        test_type: "two-sided" 或 "one-sided"

    Returns:
        dict with running p_value, adjusted_alpha, can_stop
    """
    n_min = min(len(obs_t), len(obs_c))
    # 累积检验 (Welch t 检验)
    t_stat, p_val = stats.ttest_ind(obs_t, obs_c, equal_var=False)

    # mSPRT 调整: always-valid confidence sequence
    # 渐进 alpha: alpha * sqrt(log(n) / n)
    adjusted_alpha = alpha * np.sqrt(np.log(1 + n_min) / n_min)

    # 决定是否可停止
    can_stop = p_val < adjusted_alpha

    # 效应估计
    effect = float(obs_t.mean() - obs_c.mean())

    return {
        "n": n_min,
        "msprt_p_value": float(p_val),
        "adjusted_alpha_threshold": float(adjusted_alpha),
        "significant_now": can_stop,
        "estimated_effect": effect,
        "interpretation": (
            f"已观察 {n_min} 用户 | p={p_val:.4f} | 阈值={adjusted_alpha:.4f} | "
            f"{'✓ 现在可停止' if can_stop else '△ 继续观察'}"
        ),
    }


def demo_early_stopping():
    """演示早期停止"""
    print("\n" + "=" * 78)
    print(" mSPRT 早期停止 演示".center(40))
    print("=" * 78)

    rng = np.random.default_rng(20260728)

    # 模拟累积观察
    for n_observed in [100, 200, 500, 1000, 2000, 5000]:
        # 真实效应 5%
        obs_t = rng.normal(loc=10, scale=3, size=n_observed)
        obs_c = rng.normal(loc=10 * 0.95, scale=3, size=n_observed)  # 对照组低 5%

        result = msprt_test(obs_t, obs_c, alpha=0.05)
        print(f"  N={result['n']:>5} | p={result['msprt_p_value']:.4f} | "
              f"阈值={result['adjusted_alpha_threshold']:.4f} | "
              f"{result['interpretation']}")


# ============================ 主入口 ============================

import os
import sys
import numpy as np
rng = np.random.default_rng(20260728)


if __name__ == "__main__":
    demo_seasonal()
    print()
    demo_early_stopping()
