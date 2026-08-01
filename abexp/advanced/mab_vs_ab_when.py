#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MAB vs AB 实验：何时用谁（基于真实数据 + 模拟）
===========================================
目标: 当你面对一个具体业务问题，怎么决定用 AB 还是 MAB？

本脚本基于 6 个维度做决策:
  1. 流量大小（< 1万 只能用 AB）
  2. 决策重要性（必须严谨 → AB）
  3. 实验时长（短期验证 → AB）
  4. 客群稳定性（客群稳定 → AB）
  5. 业务可解释性（需要显著证明 → AB）
  6. 失败成本（不可回滚 → AB）

输出: 6 维评估 → 综合推荐表
"""

from typing import Dict, List

import numpy as np


# ============================ 6 个判断维度 ============================

def evaluate_dimension_traffic(n_users_per_arm: int) -> Dict:
    """
    维度 1: 流量大小

    经验: MAB 需要每臂 ≥ 1000 次观察才有意义
          AB 可在 100 用户/臂做显著性检验
    """
    if n_users_per_arm < 100:
        return {
            "dimension": "流量大小",
            "score_ab": 10, "score_mab": 0,
            "recommend": "AB（极小流量下 MAB 无法学习）",
            "reasoning": "N<100 时 Thompson Sampling 还没探索完所有臂",
        }
    elif n_users_per_arm < 1000:
        return {
            "dimension": "流量大小",
            "score_ab": 8, "score_mab": 3,
            "recommend": "AB（MAB 学习不充分）",
            "reasoning": "N=100~1000 MAB 仍处于探索期，浪费较多流量",
        }
    elif n_users_per_arm < 10000:
        return {
            "dimension": "流量大小",
            "score_ab": 7, "score_mab": 6,
            "recommend": "AB 或 MAB 均可",
            "reasoning": "N=1K-10K 是边界，AB 仍可严格检验",
        }
    elif n_users_per_arm < 100000:
        return {
            "dimension": "流量大小",
            "score_ab": 6, "score_mab": 8,
            "recommend": "MAB 有优势",
            "reasoning": "N=10K-100K 是 MAB 主场，能动态收敛",
        }
    else:
        return {
            "dimension": "流量大小",
            "score_ab": 5, "score_mab": 9,
            "recommend": "MAB 强烈推荐",
            "reasoning": "N>100K 完全 MAB 可行，可探索多版本",
        }


def evaluate_dimension_decision_criticality(criticality: str) -> Dict:
    """
    维度 2: 决策重要性

    criticality: 'low' (例: UI 调整), 'medium' (例: 文案优化), 'high' (例: 收费策略变更)
    """
    mapping = {
        "low": {
            "score_ab": 5, "score_mab": 8,
            "recommend": "MAB",
            "reasoning": "低重要决策可灵活试错，MAB 累计损失低",
        },
        "medium": {
            "score_ab": 7, "score_mab": 6,
            "recommend": "AB",
            "reasoning": "中等重要需要 p<0.05 显著性，AB 占优",
        },
        "high": {
            "score_ab": 10, "score_mab": 1,
            "recommend": "AB（必须）",
            "reasoning": "高重要决策（合规、商业核心）需要严格统计证据",
        },
    }
    res = mapping[criticality]
    res["dimension"] = "决策重要性"
    return res


def evaluate_dimension_duration(duration: str) -> Dict:
    """
    维度 3: 实验时长

    duration: 'short' (1-7天), 'medium' (7-30天), 'long' (30-180天), 'permanent' (>180天)
    """
    mapping = {
        "short (< 7天)": {
            "score_ab": 10, "score_mab": 2,
            "recommend": "AB",
            "reasoning": "短期验证 MAB 还没收敛完毕",
        },
        "medium (7-30天)": {
            "score_ab": 8, "score_mab": 5,
            "recommend": "AB",
            "reasoning": "1-4 周适合 AB 的统计显著性",
        },
        "long (30-180天)": {
            "score_ab": 5, "score_mab": 8,
            "recommend": "MAB",
            "reasoning": "超过 1 个月 MAB 累积损失低于 AB",
        },
        "permanent (长期)": {
            "score_ab": 2, "score_mab": 10,
            "recommend": "MAB（必须）",
            "reasoning": "半年以上 MAB 远远超过 AB 收益",
        },
    }
    res = mapping[duration]
    res["dimension"] = "实验时长"
    return res


def evaluate_dimension_population_stability(stability: str) -> Dict:
    """
    维度 4: 客群稳定性

    stability: 'stable' (客群不变), 'shifting' (客群变化), 'unknown' (未知)
    """
    mapping = {
        "stable": {
            "score_ab": 9, "score_mab": 5,
            "recommend": "AB",
            "reasoning": "稳定客群 AB 抽样可重现，置信区间可靠",
        },
        "shifting": {
            "score_ab": 4, "score_mab": 9,
            "recommend": "MAB",
            "reasoning": "客群变化时 MAB 自适应比 AB 静态分流更灵活",
        },
        "unknown": {
            "score_ab": 6, "score_mab": 6,
            "recommend": "AB（先认识客群）",
            "reasoning": "未知客群先用 AB 摸底，再用 MAB 优化",
        },
    }
    res = mapping[stability]
    res["dimension"] = "客群稳定性"
    return res


def evaluate_dimension_explainability(need_explanation: str) -> Dict:
    """
    维度 5: 业务可解释性

    need_explanation: 'high' (要给非技术决策者讲), 'medium' (技术评审), 'low' (纯优化)
    """
    mapping = {
        "high": {
            "score_ab": 10, "score_mab": 2,
            "recommend": "AB",
            "reasoning": "业务方只能理解'对照 vs 实验'显著性，不懂 Thompson",
        },
        "medium": {
            "score_ab": 7, "score_mab": 5,
            "recommend": "AB",
            "reasoning": "技术评审更熟悉 AB 表述",
        },
        "low": {
            "score_ab": 3, "score_mab": 9,
            "recommend": "MAB",
            "reasoning": "纯优化场景无需解释，只要总收益大",
        },
    }
    res = mapping[need_explanation]
    res["dimension"] = "业务可解释性"
    return res


def evaluate_dimension_rollback(rollback_possible: str) -> Dict:
    """
    维度 6: 失败成本 / 是否可回滚

    rollback_possible: 'yes' (可随时回滚), 'limited' (回滚有代价), 'no' (不可回滚)
    """
    mapping = {
        "yes": {
            "score_ab": 6, "score_mab": 9,
            "recommend": "MAB",
            "reasoning": "可回滚容错高，MAB 可以大胆探索",
        },
        "limited": {
            "score_ab": 8, "score_mab": 5,
            "recommend": "AB",
            "reasoning": "回滚有限制需要严格验证",
        },
        "no": {
            "score_ab": 10, "score_mab": 1,
            "recommend": "AB（必须）",
            "reasoning": "不可回滚（如发版、收费）必须 AB 验证",
        },
    }
    res = mapping[rollback_possible]
    res["dimension"] = "失败成本"
    return res


# ============================ 综合评估 ============================

def full_evaluation(
    n_users_per_arm: int,
    decision_criticality: str,
    duration: str,
    population_stability: str,
    need_explanation: str,
    rollback_possible: str,
) -> Dict:
    """综合评估 6 个维度，给出推荐"""
    dims = [
        evaluate_dimension_traffic(n_users_per_arm),
        evaluate_dimension_decision_criticality(decision_criticality),
        evaluate_dimension_duration(duration),
        evaluate_dimension_population_stability(population_stability),
        evaluate_dimension_explainability(need_explanation),
        evaluate_dimension_rollback(rollback_possible),
    ]

    total_ab = sum(d["score_ab"] for d in dims)
    total_mab = sum(d["score_mab"] for d in dims)

    if total_ab >= total_mab + 15:
        final = "AB（强烈推荐）"
        ab_pct = total_ab / (total_ab + total_mab) * 100
        mab_pct = 100 - ab_pct
    elif total_mab >= total_ab + 15:
        final = "MAB（强烈推荐）"
        mab_pct = total_mab / (total_ab + total_mab) * 100
        ab_pct = 100 - mab_pct
    else:
        final = "AB 或 MAB 均可（建议先 AB 验证，再用 MAB 优化）"
        ab_pct = total_ab / (total_ab + total_mab) * 100
        mab_pct = 100 - ab_pct

    return {
        "dimensions": dims,
        "total_ab_score": total_ab,
        "total_mab_score": total_mab,
        "ab_pct": ab_pct,
        "mab_pct": mab_pct,
        "final_recommendation": final,
    }


# ============================ 8 个真实场景演示 ============================

SCENARIOS = [
    {
        "name": "场景 1: 新功能上线 - 验证是否提升转化",
        "params": dict(
            n_users_per_arm=2000,
            decision_criticality="high",
            duration="medium (7-30天)",
            population_stability="stable",
            need_explanation="high",
            rollback_possible="yes",
        ),
    },
    {
        "name": "场景 2: 长期推荐算法优化 - 持续提升 CTR",
        "params": dict(
            n_users_per_arm=50000,
            decision_criticality="medium",
            duration="permanent (长期)",
            population_stability="shifting",
            need_explanation="low",
            rollback_possible="yes",
        ),
    },
    {
        "name": "场景 3: 冷启动 - 没有历史数据的产品",
        "params": dict(
            n_users_per_arm=200,
            decision_criticality="high",
            duration="short (< 7天)",
            population_stability="unknown",
            need_explanation="high",
            rollback_possible="yes",
        ),
    },
    {
        "name": "场景 4: 业务 KPI - 改收费策略",
        "params": dict(
            n_users_per_arm=10000,
            decision_criticality="high",
            duration="long (30-180天)",
            population_stability="stable",
            need_explanation="high",
            rollback_possible="no",  # 收费变更回滚困难
        ),
    },
    {
        "name": "场景 5: 多个 CTA 文案 - 哪个点击最高",
        "params": dict(
            n_users_per_arm=5000,
            decision_criticality="low",
            duration="medium (7-30天)",
            population_stability="stable",
            need_explanation="medium",
            rollback_possible="yes",
        ),
    },
    {
        "name": "场景 6: 风控规则 - 调整反欺诈阈值",
        "params": dict(
            n_users_per_arm=10000,
            decision_criticality="high",
            duration="long (30-180天)",
            population_stability="shifting",
            need_explanation="medium",
            rollback_possible="limited",
        ),
    },
    {
        "name": "场景 7: 邮件推送 - 选最优发送时间",
        "params": dict(
            n_users_per_arm=500,
            decision_criticality="low",
            duration="short (< 7天)",
            population_stability="stable",
            need_explanation="low",
            rollback_possible="yes",
        ),
    },
    {
        "name": "场景 8: 长期价格弹性优化",
        "params": dict(
            n_users_per_arm=100000,
            decision_criticality="medium",
            duration="permanent (长期)",
            population_stability="shifting",
            need_explanation="medium",
            rollback_possible="limited",
        ),
    },
]


def demo_all_scenarios():
    """演示 8 个真实业务场景"""
    print("=" * 88)
    print(" MAB vs AB 实验：8 个真实场景推荐".center(60))
    print("=" * 88)
    print()

    for scenario in SCENARIOS:
        result = full_evaluation(**scenario["params"])
        print(f" {scenario['name']}")
        print("-" * 88)
        for d in result["dimensions"]:
            score = "AB " * d["score_ab"] + "MAB " * d["score_mab"]
            print(f"  {d['dimension']}: {d['recommend']}")
        print(f"\n  总分: AB={result['total_ab_score']} vs MAB={result['total_mab_score']}")
        print(f"  倾向: AB {result['ab_pct']:.0f}% vs MAB {result['mab_pct']:.0f}%")
        print(f"  → 最终建议: {result['final_recommendation']}")
        print()


def demo_decision_matrix():
    """输出 6 个维度的决策矩阵"""
    print("=" * 88)
    print(" MAB vs AB 决策矩阵 (6 维)".center(60))
    print("=" * 88)
    print()

    test_cases = [
        ("极小流量 (N=50)", 50, "high", "short (< 7天)", "stable", "high", "yes"),
        ("小流量 (N=500)", 500, "high", "short (< 7天)", "stable", "high", "yes"),
        ("中等流量 (N=5K)", 5000, "medium", "medium (7-30天)", "stable", "medium", "yes"),
        ("大流量 (N=50K)", 50000, "low", "long (30-180天)", "shifting", "low", "yes"),
        ("超大流量 (N=500K)", 500000, "low", "permanent (长期)", "shifting", "low", "yes"),
        ("高决策重要性", 5000, "high", "medium (7-30天)", "stable", "high", "no"),
        ("可回滚", 50000, "low", "long (30-180天)", "shifting", "low", "yes"),
        ("不可回滚", 5000, "medium", "medium (7-30天)", "stable", "medium", "no"),
    ]

    print(f" {'场景':<22}{'AB得分':>10}{'MAB得分':>10}{'推荐':<35}")
    print("-" * 88)
    for name, n, crit, dur, stab, expl, rb in test_cases:
        r = full_evaluation(n, crit, dur, stab, expl, rb)
        rec = r["final_recommendation"]
        if len(rec) > 30:
            rec_short = rec[:28] + ".."
        else:
            rec_short = rec
        print(f" {name:<22}{r['total_ab_score']:>10}{r['total_mab_score']:>10}  {rec_short:<35}")


if __name__ == "__main__":
    demo_decision_matrix()
    print()
    demo_all_scenarios()
