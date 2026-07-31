#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A/A 测试模块 (基线噪声验证)
========================
问题: 实验开始前应该做 A/A 测试（对照组 vs 对照组）
      验证指标本身的统计性质（基线 P-value 分布、CI 宽度）

方案: 跑 1000+ 次 A/A 测试，验证:
  - Type I Error 接近 α=5%
  - 95% CI 宽度合理（基线波动范围）
  - 比率差是均值为 0 的正态分布（系统无偏差）
"""

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats


def run_aa_test(
    base_metric_func,
    n_users: int = 1000,
    n_trials: int = 1000,
    alpha: float = 0.05,
    metric_type: str = "binary",  # "binary" or "continuous"
) -> Dict:
    """
    运行 A/A 测试

    Args:
        base_metric_func: 无参函数，返回每用户 baseline metric 值
        n_users: 每组用户数
        n_trials: A/A 测试次数（应 ≥ 1000 以获得稳定估计）
        alpha: 显著性阈值

    Returns:
        dict with type I error, power, CI width, bias distribution
    """
    p_values = []
    effect_estimates = []
    ci_widths = []

    for trial in range(n_trials):
        # 两个"对照组"用同一基线分布
        baseline_a = base_metric_func(n_users)
        baseline_b = base_metric_func(n_users)

        if metric_type == "binary":
            # Z 检验
            p_a = baseline_a.mean()
            p_b = baseline_b.mean()
            p_pool = (baseline_a.sum() + baseline_b.sum()) / (2 * n_users)
            se = np.sqrt(p_pool * (1 - p_pool) * (2 / n_users))
            z = (p_a - p_b) / se if se > 0 else 0
            p_val = 2 * (1 - stats.norm.cdf(abs(z)))
            effect = (p_a - p_b) / p_pool if p_pool > 0 else 0
            # 95% CI 宽度
            se_diff = np.sqrt(p_a*(1-p_a)/n_users + p_b*(1-p_b)/n_users)
            ci_width = 2 * 1.96 * se_diff
        else:
            t_stat, p_val = stats.ttest_ind(baseline_a, baseline_b, equal_var=False)
            effect = baseline_a.mean() - baseline_b.mean()
            se_diff = np.sqrt(baseline_a.var()/n_users + baseline_b.var()/n_users)
            ci_width = 2 * 1.96 * se_diff

        p_values.append(p_val)
        effect_estimates.append(effect)
        ci_widths.append(ci_width)

    p_values = np.array(p_values)
    effect_estimates = np.array(effect_estimates)
    ci_widths = np.array(ci_widths)

    # Type I Error: p < alpha 的比例
    type_i_error = (p_values < alpha).mean()

    # P-value 分布（应接近均匀）
    p_uniform_pass = stats.kstest(p_values, "uniform").pvalue

    return {
        "n_trials": n_trials,
        "type_i_error": float(type_i_error),  # 应接近 alpha
        "alpha_target": alpha,
        "type_i_error_pass": abs(type_i_error - alpha) < 0.015,  # 在 1.5% 范围内
        "p_value_median": float(np.median(p_values)),  # 应接近 0.5
        "p_value_mean": float(np.mean(p_values)),  # 应接近 0.5
        "p_value_dist_uniform": float(p_uniform_pass),  # KS 检验
        "effect_mean": float(np.mean(effect_estimates)),  # 应接近 0
        "effect_std": float(np.std(effect_estimates)),
        "effect_95_ci": (
            float(np.percentile(effect_estimates, 2.5)),
            float(np.percentile(effect_estimates, 97.5)),
        ),
        "avg_ci_width": float(np.mean(ci_widths)),  # 平均 CI 宽度
        "ci_width_std": float(np.std(ci_widths)),
        "p_values": p_values,
        "effect_estimates": effect_estimates,
        "ci_widths": ci_widths,
    }


def demo_with_real_data():
    """用 Kaggle 真实数据演示 A/A 测试"""
    print("=" * 78)
    print(" A/A 测试：基线噪声验证（Kaggle 真实数据）".center(40))
    print("=" * 78)

    path = "/Users/mac/.cache/kagglehub/datasets/computingvictor/transactions-fraud-datasets/versions/1"
    if not os.path.exists(path):
        print(" Kaggle 数据未下载，使用 mock 数据演示")
        # 用 lognormal 模拟
        rng = np.random.default_rng(42)
        base_func = lambda n: rng.lognormal(mean=4.0, sigma=1.2, size=n)
        result = run_aa_test(base_func, n_users=500, n_trials=500, metric_type="continuous")
    else:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from did_cuped_consumption import build_user_consumption, load_transactions
        trans = load_transactions()
        user_data = build_user_consumption(trans)
        pool = user_data["pre_avg_consumption"].values

        def sample_consumption(n):
            return rng.choice(pool, size=n, replace=False)

        # 用 rng.choice 让 A/A 抽样差异可控
        rng = np.random.default_rng(42)
        base_func = lambda n: sample_consumption(n)
        result = run_aa_test(base_func, n_users=500, n_trials=500, metric_type="continuous")

    print(f"\n 试验次数: {result['n_trials']}")
    print(f"\n 【关键指标】")
    print(f" Type I Error (预期 5.0%):")
    print(f"   实测: {result['type_i_error']*100:.2f}%")
    print(f"   {'✓ 合理' if result['type_i_error_pass'] else '⚠ 异常'}")

    print(f"\n P-value 分布（应均匀）:")
    print(f"   中位数: {result['p_value_median']:.4f}  (应接近 0.5)")
    print(f"   均值: {result['p_value_mean']:.4f}  (应接近 0.5)")
    print(f"   KS 均匀性检验 p-value: {result['p_value_dist_uniform']:.4f}")

    print(f"\n 效应分布 (应均值为 0):")
    print(f"   均值: {result['effect_mean']:+.6f}")
    print(f"   std: {result['effect_std']:.6f}")
    print(f"   95% 范围: [{result['effect_95_ci'][0]:+.4f}, {result['effect_95_ci'][1]:+.4f}]")

    print(f"\n 单次 CI 宽度（实验设计参考）:")
    print(f"   平均 CI 宽度: ±${result['avg_ci_width']/2:.2f}")
    print(f"   CI std: ${result['ci_width_std']:.4f}")

    print(f"\n 【A/A 测试结论】")
    if result["type_i_error_pass"]:
        print(f" ✓ 基线噪声正常：Type I Error = {result['type_i_error']*100:.1f}% (期望 5%)")
        print(f"   → 实验结果可信，false positive 不会超预期")
    else:
        print(f" ⚠ 基线噪声异常：Type I Error = {result['type_i_error']*100:.1f}%")
        print(f"   → 需要修正指标 metric 或剔除异常流量")

    print(f"\n 【实验设计建议】")
    print(f" 基线 95% CI 宽度 ±${result['avg_ci_width']/2:.2f}")
    print(f" 欲检出 > ±${result['avg_ci_width']/2:.2f} 的差异更稳")
    print(f" 欲检出 < ±${result['avg_ci_width']/2:.2f} 的差异需要扩大样本量")


import os
import sys


if __name__ == "__main__":
    demo_with_real_data()
