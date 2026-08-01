#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
异常值处理模块
=============
问题: 1% 的高频用户可能贡献 50% GMV，扭曲实验结果
方案: 缩尾处理 (Winsorize) / 百分位切割 / 稳健检验

提供 5 种异常处理方法 + 1 种稳健检验:
  1. raw (原始数据)
  2. winsorize 5% (5% 缩尾)
  3. winsorize 1% (1% 缩尾)
  4. cap 99% (99% 百分位切割)
  5. log_transform (对数变换)
  6. wilcoxon (Wilcoxon 稳健秩检验)

输出对比表: 不同方法的 power + 偏差 + 鲁棒性
"""

from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats


def winsorize(values: np.ndarray, lower_pct: float = 0.05, upper_pct: float = 0.95) -> np.ndarray:
    """缩尾处理: 将低于下分位数 / 高于上分位数的值替换为分位数"""
    lower = np.percentile(values, lower_pct * 100)
    upper = np.percentile(values, upper_pct * 100)
    return np.clip(values, lower, upper)


def cap_top(values: np.ndarray, top_pct: float = 0.99) -> np.ndarray:
    """百分位切割: 将超过 P99 的值替换为 P99"""
    cap_value = np.percentile(values, top_pct * 100)
    return np.minimum(values, cap_value)


def log_transform(values: np.ndarray) -> np.ndarray:
    """对数变换: 适合右偏分布 (如 GMV、消费金额)"""
    return np.log1p(values)


def compare_methods(
    treat_raw: np.ndarray,
    ctrl_raw: np.ndarray,
    n_trials: int = 100,
    baseline_effect_pct: float = 5.0,
    seed_base: int = 20260728,
) -> pd.DataFrame:
    """
    对比不同异常处理方法

    Args:
        treat_raw, ctrl_raw: 实验组 / 对照组原始数据
        n_trials: 蒙特卡洛次数
        baseline_effect_pct: 实验组真实提升（%）
        seed_base: 种子

    Returns:
        DataFrame with methods comparison
    """
    methods = {
        "raw": lambda x: x,
        "winsorize_5%": lambda x: winsorize(x, 0.05, 0.95),
        "winsorize_1%": lambda x: winsorize(x, 0.01, 0.99),
        "cap_99%": lambda x: cap_top(x, 0.99),
        "log_transform": log_transform,
    }

    results = []
    for method_name, transform_fn in methods.items():
        powers = []
        effect_estimates = []
        for trial in range(n_trials):
            rng = np.random.default_rng(seed_base + trial)
            # 在数据上模拟真实提升
            t = treat_raw.copy()
            t = t * (1 + baseline_effect_pct / 100)
            t += rng.normal(0, t.std() * 0.1, len(t))
            c = ctrl_raw.copy()
            c += rng.normal(0, c.std() * 0.1, len(c))

            t_trans = transform_fn(t)
            c_trans = transform_fn(c)

            t_stat, p = stats.ttest_ind(t_trans, c_trans, equal_var=False)
            powers.append(p < 0.05)
            effect_estimates.append(t_trans.mean() - c_trans.mean())

        # 稳健性检验 (无异常值处理的影响)
        t_raw, p_raw = stats.ttest_ind(treat_raw, ctrl_raw, equal_var=False)
        # Wilcoxon
        try:
            _, p_wilcox = stats.mannwhitneyu(treat_raw, ctrl_raw, alternative="two-sided")
        except Exception:
            p_wilcox = 1.0

        results.append({
            "method": method_name,
            "power": 100 * np.mean(powers),
            "avg_effect": float(np.mean(effect_estimates)),
            "effect_std": float(np.std(effect_estimates)),
            "raw_p_value": float(p_raw),
            "robust_p_value (Wilcoxon)": float(p_wilcox),
        })

    return pd.DataFrame(results)


def demo():
    """演示异常处理对真实数据的影响"""
    print("=" * 78)
    print(" 异常值处理对实验结果影响（Kaggle 真实消费数据）".center(40))
    print("=" * 78)

    # 用我们之前实测的"consumption"数据
    path = "/Users/mac/.cache/kagglehub/datasets/computingvictor/transactions-fraud-datasets/versions/1"
    if not os.path.exists(path):
        print(" Kaggle 数据未下载，改用 mock 数据")
        rng = np.random.default_rng(42)
        # 模拟长尾分布 (对数正态)
        treat = rng.lognormal(mean=4, sigma=1.2, size=500)
        ctrl = rng.lognormal(mean=4, sigma=1.2, size=500)
    else:
        # 加载 Kaggle 真实数据
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from did_cuped_consumption import build_user_consumption, load_transactions
        trans = load_transactions()
        user_data = build_user_consumption(trans)
        treat = user_data["pre_avg_consumption"].sample(500, random_state=42).values
        ctrl = user_data["pre_avg_consumption"].sample(500, random_state=43).values

    print(f"\n 实验组: n={len(treat)}, mean=${treat.mean():.2f}, std=${treat.std():.2f}")
    print(f"  实验组 99%分位: ${np.percentile(treat, 99):.2f} (最大值 ${treat.max():.2f})")
    print(f" 对照组: n={len(ctrl)}, mean=${ctrl.mean():.2f}, std=${ctrl.std():.2f}")

    print("\n" + "=" * 78)
    print(" 5 种异常值处理方法 vs 100 次蒙特卡洛".center(50))
    print("=" * 78)
    df = compare_methods(treat, ctrl)
    print()
    print(df.to_string(index=False))


import os
import sys


if __name__ == "__main__":
    demo()
