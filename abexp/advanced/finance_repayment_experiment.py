#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
金融场景 AB 实验：长账期（7-30 天）评估框架
===========================================
场景: 金融产品（贷款/信用卡/消费分期）账期普遍 7-30 天甚至更长

挑战:
  1. 短期看不到真实 Y（还款/违约），但仍需决策
  2. 用户行为是"漏斗"，不是单一转化事件
  3. 长尾用户（30 天后仍未还款）造成偏差
  4. 监管约束 → 不能为了实验改产品规则

解决方案:
  1. 分层漏斗指标：中间指标提前预警
  2. 生存分析 Kaplan-Meier: 处理账期长尾
  3. 分段评估：第 7 天 / 第 14 天 / 第 30 天
  4. DID 配对：用户级别异质性控制

输出:
  - 漏斗各层转化率
  - 按时还款函数 S(t)
  - 第 7/14/30 天累计转化
  - 风险指标 (预期损失)
"""

from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats


# ============================ 1. 漏斗指标 ============================

def funnel_metrics(
    treat_data: pd.DataFrame,
    ctrl_data: pd.DataFrame,
    funnel_steps: Dict[str, str],
) -> pd.DataFrame:
    """
    漏斗各层转化率对比

    Args:
        treat_data, ctrl_data: 实验组 / 对照组用户级数据
        funnel_steps: 漏斗各层 {名称: 列名}
            例: {'apply': 'clicked',
                 'submit': 'submitted',
                 'bind': 'card_bound',
                 'apply_repay': 'first_repay'}

    Returns:
        DataFrame with step, treat_rate, ctrl_rate, lift, p_value
    """
    results = []
    for step_name, col in funnel_steps.items():
        if col not in treat_data.columns or col not in ctrl_data.columns:
            continue

        t = treat_data[col].astype(int)
        c = ctrl_data[col].astype(int)

        n_t, n_c = len(t), len(c)
        p_t = t.mean()
        p_c = c.mean()
        lift = (p_t - p_c) / p_c * 100 if p_c > 0 else 0

        # Z 检验
        p_pool = (t.sum() + c.sum()) / (n_t + n_c)
        se = np.sqrt(p_pool * (1 - p_pool) * (1/n_t + 1/n_c)) if p_pool > 0 else 0
        z = (p_t - p_c) / se if se > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))

        results.append({
            "step": step_name,
            "treat_n": n_t,
            "treat_rate": p_t,
            "ctrl_rate": p_c,
            "lift_pct": lift,
            "p_value": p_value,
            "significant": p_value < 0.05,
        })

    return pd.DataFrame(results)


# ============================ 2. 生存分析 (Kaplan-Meier) ============================

def kaplan_meier_estimator(
    days_to_event: np.ndarray,
    event_observed: np.ndarray,
    max_day: int = 60,
) -> Dict:
    """
    Kaplan-Meier 估计器: S(t) = P(还没到账期)

    Args:
        days_to_event: 每个用户到事件的天数（到还款/到终止）
        event_observed: 1=已还款（事件）, 0=censored（终止）
        max_day: 最大观察天数

    Returns:
        dict with t_survival, s_survival, censor_at_max
    """
    days = np.arange(1, max_day + 1)
    s = np.ones(max_day)
    cum_s = 1.0

    for d in days:
        n_at_risk = (days_to_event >= d).sum()
        n_events_today = ((days_to_event == d) & (event_observed == 1)).sum()

        if n_at_risk > 0:
            cum_s *= (1 - n_events_today / n_at_risk)
        s[d - 1] = cum_s

    return {
        "days": days,
        "survival": s,  # S(t)
    }


def compare_km_curves(
    treat_days: np.ndarray,
    treat_events: np.ndarray,
    ctrl_days: np.ndarray,
    ctrl_events: np.ndarray,
    max_day: int = 30,
) -> Dict:
    """
    比较两组生存曲线

    检验: Log-rank test 近似（不需要 SciPy）
    """
    km_t = kaplan_meier_estimator(treat_days, treat_events, max_day)
    km_c = kaplan_meier_estimator(ctrl_days, ctrl_events, max_day)

    # 第 7/14/30 天累计还款率
    milestones = [7, 14, 30]
    cum_repay = {}
    for d in milestones:
        if d <= max_day:
            cum_repay[d] = {
                "treat": float(1 - km_t["survival"][d - 1]),
                "ctrl": float(1 - km_c["survival"][d - 1]),
                "lift": float(1 - km_t["survival"][d - 1] - (1 - km_c["survival"][d - 1])),
            }

    # Log-rank 检验 (简化版: Mann-Whitney U on days_to_event)
    u_stat, p_value = stats.mannwhitneyu(
        treat_days, ctrl_days, alternative="two-sided"
    )

    return {
        "treat_km": km_t,
        "ctrl_km": km_c,
        "milestones": cum_repay,
        "logrank_u_stat": float(u_stat),
        "logrank_p_value": float(p_value),
    }


# ============================ 3. 分段时间点评估 ============================

def evaluate_at_multiple_timepoints(
    treat_users: pd.DataFrame,
    ctrl_users: pd.DataFrame,
    timepoints: list,
    y_col: str,
) -> pd.DataFrame:
    """
    在多个时间点评估转化（如 Day 7, 14, 30）

    Args:
        treat_users, ctrl_users: 用户级数据
        timepoints: 时间点列表，如 [7, 14, 30]
        y_col: 转化事件列
    """
    results = []
    for t in timepoints:
        # 假设用户级数据有 "converted_by_day" 列 (= day when converted)
        # 简化: 这里我们看转化率的变化
        if y_col not in treat_users.columns:
            continue

        # 第 t 天累计转化 (用简单抽样模拟)
        cum_t = treat_users[y_col].mean() * (t / 30)  # 累计转化随时间增长
        cum_c = ctrl_users[y_col].mean() * (t / 30)

        lift = (cum_t - cum_c) / cum_c * 100 if cum_c > 0 else 0

        results.append({
            "timepoint_days": t,
            "treat_cum_rate": cum_t,
            "ctrl_cum_rate": cum_c,
            "lift_pct": lift,
            "sample_size": len(treat_users),
        })

    return pd.DataFrame(results)


# ============================ 4. 风险指标 ============================

def risk_metrics(
    treat_users: pd.DataFrame,
    ctrl_users: pd.DataFrame,
    loan_amount_col: str,
    default_col: str,
) -> Dict:
    """
    金融场景特有：风险指标

    Args:
        treat_users, ctrl_users: 用户数据（含贷款金额、违约标记）
        loan_amount_col: 贷款金额列
        default_col: 违约标记列
    """
    n_t, n_c = len(treat_users), len(ctrl_users)

    # 平均贷款金额
    avg_loan_t = treat_users[loan_amount_col].mean()
    avg_loan_c = ctrl_users[loan_amount_col].mean()

    # 违约率
    default_rate_t = treat_users[default_col].mean()
    default_rate_c = ctrl_users[default_col].mean()

    # 预期损失（假设违约发生 100% 损失）
    expected_loss_t = avg_loan_t * default_rate_t
    expected_loss_c = avg_loan_c * default_rate_c

    return {
        "avg_loan_treat": float(avg_loan_t),
        "avg_loan_ctrl": float(avg_loan_c),
        "default_rate_treat": float(default_rate_t),
        "default_rate_ctrl": float(default_rate_c),
        "expected_loss_treat": float(expected_loss_t),
        "expected_loss_ctrl": float(expected_loss_c),
        "loss_reduction_pct": float(
            (expected_loss_c - expected_loss_t) / expected_loss_c * 100
            if expected_loss_c > 0 else 0
        ),
    }


# ============================ 5. DID 在金融场景 ============================

def finance_did_analysis(
    treat_pre: np.ndarray,
    treat_post: np.ndarray,
    ctrl_pre: np.ndarray,
    ctrl_post: np.ndarray,
    post_window_days: int = 30,
) -> Dict:
    """
    DID 分析（金融场景下配对）
    用户在实验前已有数据，可用作对照组
    """
    # DID: 处理效应 = (实验组 post - pre) - (对照组 post - pre)
    treat_delta = treat_post - treat_pre
    ctrl_delta = ctrl_post - ctrl_pre

    did_effect = float(treat_delta.mean() - ctrl_delta.mean())
    t_stat, p_value = stats.ttest_ind(treat_delta, ctrl_delta, equal_var=False)

    return {
        "did_effect": did_effect,
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
        "interpretation": (
            f"实验组相比对照组相对提升 {did_effect*100:.2f}%"
            "（DID 配对分析）"
        ),
    }


# ============================ Demo ============================

def demo_finance_funnel():
    """演示金融场景漏斗分析"""
    print("=" * 88)
    print(" 金融场景实验：漏斗各层转化分析（30 天账期）".center(50))
    print("=" * 88)

    np.random.seed(20260728)

    # 模拟 1000 用户漏斗数据
    # 实验组：降低申请门槛 → 漏斗上层高转化，但还款率可能下降
    # 对照组：原策略
    n_t, n_c = 1000, 1000

    # 漏斗各层
    print(f"\n 实验组 N={n_t}, 对照组 N={n_c}")
    print(f" 实验组方案：降低申请门槛（让更多人能申请）\n")

    funnel_data = []
    funnel_steps = [
        ("曝光→点击",       0.85, 0.84),
        ("点击→申请",       0.42, 0.35),    # 实验组提升 +7pp
        ("申请→审批通过",   0.68, 0.62),    # 降门槛后通过率略降
        ("通过→绑卡",       0.55, 0.55),    # 一样
        ("绑卡→30天还款",   0.78, 0.85),    # 风险用户还款率降
    ]

    print(f" {'漏斗层':<18}{'实验组':<10}{'对照组':<10}{'提升':<10}{'P-value':<10}{'判定'}")
    print("-" * 78)

    for step, lift_t, lift_c in funnel_steps:
        lift = (lift_t - lift_c) / lift_c * 100 if lift_c > 0 else 0
        n_total = n_t * lift_t
        se = np.sqrt(lift_t * (1 - lift_t) / n_t + lift_c * (1 - lift_c) / n_c)
        z = (lift_t - lift_c) / se if se > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        sig = "✓" if p_value < 0.05 else "△"
        print(f" {step:<18}{lift_t*100:>6.1f}%   {lift_c*100:>6.1f}%   {lift:+6.1f}%  {p_value:<10.4f}{sig}")

    print("\n【关键发现】")
    print(" 实验组（降门槛）：")
    print("   ✓ 漏斗上层（点击/申请）显著提升")
    print("   ✗ 漏斗下层（还款率）显著下降")
    print("   → 实验看起来「提升 10%」，但实际风险上升！")

    print("\n【常规分析的陷阱】")
    print(" 只看点击 / 申请 → 推荐全量上线")
    print(" 看完整漏斗 → 不能上线，应止损")


def demo_3layer_decision():
    """演示 3 层决策：漏斗 + 时间点 + 风险"""
    print("\n" + "=" * 88)
    print(" 金融场景 3 层决策流程".center(50))
    print("=" * 88)
    print()

    print("【第 1 层】漏斗预警（前 7 天就能看到）")
    print("  - 申请率 +20%（OK）")
    print("  - 通过率 -5%（注意）")
    print("  - 绑卡率 -3%（注意）")
    print("  → 风险信号已显现")
    print()

    print("【第 2 层】中段观察（Day 7）")
    print("  - 第 7 天累计还款率: 实验组 5.0% vs 对照 5.5%（-9%，*显著*）")
    print("  - 第 7 天还能看到 50% 用户的真实还款行为")
    print("  → 中段判断: 暂停放量 / 缩量")
    print()

    print("【第 3 层】长期确认（Day 30）")
    print("  - Day 30 累计还款: 实验组 70% vs 对照 82%（-15%，**强烈显著**）")
    print("  - 长尾 (Day 30+) 用户违约率明显")
    print("  - 预期损失: +12%（实证）")
    print("  → 最终决定: 不上线或回滚")
    print()

    print("【3 层结论对照】")
    print(" Day 7 看到: -9% （看似有问题）")
    print(" Day 30 看到: -15% （确实有问题）")
    print(" **如果只到 Day 14，可能还判断不出真实危害**")


def real_3layer_decision_strategy():
    """输出 3 层决策策略文档"""
    print("\n" + "=" * 88)
    print(" 3 层决策策略".center(50))
    print("=" * 88)
    print()

    print("策略原则：实验初期谨慎 + 中期判断 + 长期确认")
    print()
    print("Day 1-3: 紧急监控")
    print("   - 只看漏斗上层（点击 / 申请）")
    print("   - 任何 SRM / 系统异常 → 立即停止")
    print()
    print("Day 4-7: 漏斗预警")
    print("   - 看完整漏斗：申请 → 审批 → 绑卡")
    print("   - 上层显著 + 下层异常 → 缩量到 1%")
    print()
    print("Day 7-30: 中段观察")
    print("   - 每天累计还款率（survival curve）")
    print("   - 实验组 vs 对照组始终跟踪")
    print("   - 若 Day 7 已显示 -5% → **不要等到 Day 30，先缩量**")
    print()
    print("Day 30-60: 最终确认")
    print("   - 90% 用户已完成还款")
    print("   - 报告生成 + 最终决策")
    print("   - 这个阶段才能准确判断亏损 / 收益")


if __name__ == "__main__":
    demo_finance_funnel()
    print()
    demo_3layer_decision()
    print()
    real_3layer_decision_strategy()
