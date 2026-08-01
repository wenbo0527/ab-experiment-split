#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Beta-Binomial 贝叶斯 AB 测试
===========================
问题: 0/1 事件小流量下，频率派 t 检验 P-value 鲁棒性差
方案: 用 Beta 二项分布描述转化率的后验分布

输出:
  - 实验组/对照组转化率的后验分布
  - P(treat > ctrl) 概率
  - 期望提升 (相对/绝对)
  - 95% 可信区间 (credible interval)

适用场景:
  - 0/1 事件（fraud/注册/付费）
  - 小流量（<1000 用户）
  - 业务方更易理解"概率"而非"P-value"
"""

from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats


def beta_binomial_test(
    n_treat: int,
    x_treat: int,
    n_ctrl: int,
    x_ctrl: int,
    alpha_prior: float = 1.0,
    beta_prior: float = 1.0,
    n_samples: int = 100_000,
    seed: int = 20260728,
) -> Dict:
    """
    Beta-Binomial 贝叶斯 AB 检验

    后验:
      treat ~ Beta(alpha + x_treat, beta + n_treat - x_treat)
      ctrl  ~ Beta(alpha + x_ctrl,  beta + n_ctrl - x_ctrl)

    Args:
        n_treat, x_treat: 实验组用户数、转化数
        n_ctrl, x_ctrl:   对照组用户数、转化数
        alpha_prior, beta_prior: Beta 先验（默认 U(0,1)）
        n_samples: Monte Carlo 采样数

    Returns:
        dict with credible intervals, P(better), expected lift
    """
    rng = np.random.default_rng(seed)

    # 后验分布
    post_treat = stats.beta(alpha_prior + x_treat, beta_prior + n_treat - x_treat)
    post_ctrl = stats.beta(alpha_prior + x_ctrl, beta_prior + n_ctrl - x_ctrl)

    # 后验采样
    samples_t = post_treat.rvs(n_samples, random_state=rng)
    samples_c = post_ctrl.rvs(n_samples, random_state=rng)

    # 后验统计
    treat_mean = float(samples_t.mean())
    ctrl_mean = float(samples_c.mean())
    treat_ci = (
        float(np.percentile(samples_t, 2.5)),
        float(np.percentile(samples_t, 97.5)),
    )
    ctrl_ci = (
        float(np.percentile(samples_c, 2.5)),
        float(np.percentile(samples_c, 97.5)),
    )

    # P(treat > ctrl) - 实验组更好的概率
    p_better = float((samples_t > samples_c).mean())

    # 期望提升
    samples_lift = samples_t - samples_c
    expected_abs_lift = float(samples_lift.mean())
    expected_rel_lift = (
        float((samples_lift / (samples_c + 1e-9)).mean()) if n_ctrl > 0 else 0
    )

    # 提升的 95% 可信区间
    lift_ci = (
        float(np.percentile(samples_lift, 2.5)),
        float(np.percentile(samples_lift, 97.5)),
    )

    return {
        "n_treatment": n_treat,
        "x_treatment": x_treat,
        "n_control": n_ctrl,
        "x_control": x_ctrl,
        "p_treatment_post_mean": treat_mean,
        "p_control_post_mean": ctrl_mean,
        "p_treatment_ci": treat_ci,
        "p_control_ci": ctrl_ci,
        "p_better_treat": p_better,  # 后验 P(treat > ctrl)
        "expected_abs_lift": expected_abs_lift,
        "expected_rel_lift": expected_rel_lift,
        "lift_ci": lift_ci,
        "interpretation": _interpret(p_better, expected_rel_lift),
    }


def _interpret(p_better: float, rel_lift: float) -> str:
    """根据 P(better) 和相对提升给出业务可读的结论"""
    if p_better > 0.95:
        return f"✓ 强证据支持实验组更好 (P={p_better*100:.1f}%, 提升 ≈ {rel_lift*100:+.1f}%)"
    elif p_better > 0.85:
        return f"✓ 中等证据支持实验组 (P={p_better*100:.1f}%, 提升 ≈ {rel_lift*100:+.1f}%)"
    elif p_better > 0.6:
        return f"△ 弱证据，倾向实验组 (P={p_better*100:.1f}%, 提升 ≈ {rel_lift*100:+.1f}%)"
    elif p_better > 0.4:
        return f"△ 无明显差异 (P={p_better*100:.1f}%, 提升 ≈ {rel_lift*100:+.1f}%)"
    else:
        return f"❌ 证据支持对照组 (P={p_better*100:.1f}%, 提升 ≈ {rel_lift*100:+.1f}%)"


def compare_with_ttest(
    n_treat: int,
    x_treat: int,
    n_ctrl: int,
    x_ctrl: int,
    n_simulations: int = 200,
) -> pd.DataFrame:
    """
    对比 Beta-Binomial 和 t 检验的差异

    在多种 effect size 下跑 200 次蒙特卡洛，比较：
      - t 检验的 power
      - 贝叶斯的 P(better) > 0.95 占比
    """
    p_treat = x_treat / n_treat
    base_rate = p_treat

    test_results = {"t_test_significant": [], "bayes_p_better": []}

    for sim in range(n_simulations):
        rng = np.random.default_rng(20260728 + sim)

        # 在多种 effect 下模拟（A/A 基线下 effect=0，A/B 有 effect）
        for effect_addend in [0, 0.005, 0.01, 0.02]:
            # 模拟两个观察组
            treat_obs = rng.binomial(n_treat, base_rate + effect_addend) if effect_addend > 0 else x_treat
            ctrl_obs = x_ctrl

            # t 检验
            p_t = treat_obs / n_treat
            p_c = ctrl_obs / n_ctrl
            p_pool = (treat_obs + ctrl_obs) / (n_treat + n_ctrl)
            se = np.sqrt(p_pool * (1 - p_pool) * (1/n_treat + 1/n_ctrl))
            z = (p_t - p_c) / se if se > 0 else 0
            p_val_t = 2 * (1 - stats.norm.cdf(abs(z)))
            t_sig = p_val_t < 0.05

            # 贝叶斯
            bayes_result = beta_binomial_test(n_treat, treat_obs, n_ctrl, ctrl_obs)
            bayes_sig = bayes_result["p_better_treat"] > 0.95

            test_results["t_test_significant"].append({
                "effect_size": effect_addend,
                "t_significant": t_sig,
                "bayes_p_better_gt_95": bayes_sig,
            })

    df = pd.DataFrame(test_results["t_test_significant"])

    # 汇总
    summary = []
    for effect in df["effect_size"].unique():
        sub = df[df["effect_size"] == effect]
        summary.append({
            "真实 effect": f"+{effect*100:.2f}%" if effect > 0 else "A/A (0)",
            "t 检验 power": f"{sub['t_significant'].mean()*100:.1f}%",
            "贝叶斯 P>95% 占比": f"{sub['bayes_p_better_gt_95'].mean()*100:.1f}%",
        })

    return pd.DataFrame(summary)


def demo():
    """演示贝叶斯 vs t 检验对比"""
    print("=" * 78)
    print(" Beta-Binomial 贝叶斯 AB 测试 演示".center(50))
    print("=" * 78)

    # 示例 1: 单次贝叶斯检验
    print("\n【示例 1】单次检验 (实验组 12% vs 对照组 10%)")
    print("-" * 78)
    result = beta_binomial_test(
        n_treat=1500, x_treat=180,
        n_ctrl=1500, x_ctrl=145,
    )
    print(f"实验组后验转化率: {result['p_treatment_post_mean']*100:.2f}% "
          f"95% CI: ({result['p_treatment_ci'][0]*100:.2f}%, {result['p_treatment_ci'][1]*100:.2f}%)")
    print(f"对照组后验转化率: {result['p_control_post_mean']*100:.2f}% "
          f"95% CI: ({result['p_control_ci'][0]*100:.2f}%, {result['p_control_ci'][1]*100:.2f}%)")
    print(f"\n P(实验组 > 对照组): {result['p_better_treat']*100:.2f}%")
    print(f" 期望绝对提升:     {result['expected_abs_lift']*100:+.2f}%")
    print(f" 期望相对提升:     {result['expected_rel_lift']*100:+.2f}%")
    print(f" 提升 95% CI:      [{result['lift_ci'][0]*100:+.2f}%, {result['lift_ci'][1]*100:+.2f}%]")
    print(f"\n 业务解读: {result['interpretation']}")

    # 示例 2: 小流量场景
    print("\n【示例 2】小流量场景 (实验组 50/200 vs 对照组 30/200)")
    print("-" * 78)
    result2 = beta_binomial_test(n_treat=200, x_treat=50, n_ctrl=200, x_ctrl=30)
    print(f" 转化率后验: 实验组 {result2['p_treatment_post_mean']*100:.2f}% vs 对照 {result2['p_control_post_mean']*100:.2f}%")
    print(f" P(实验组更好): {result2['p_better_treat']*100:.2f}%")
    print(f" 期望提升: {result2['expected_rel_lift']*100:+.2f}%")
    print(f"\n 业务解读: {result2['interpretation']}")

    # 示例 3: 对比表
    print("\n【示例 3】t 检验 vs 贝叶斯: 100 次蒙特卡洛对比")
    print("-" * 78)
    df = compare_with_ttest(n_treat=1500, x_treat=180, n_ctrl=1500, x_ctrl=150, n_simulations=100)
    print()
    print(df.to_string(index=False))


if __name__ == "__main__":
    demo()
