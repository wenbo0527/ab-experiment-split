#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多层正交实验分流实验
=====================
字节 DataTester 等工业级 AB 平台的核心机制。

核心思想：
  - 每层用独立的 salt 做哈希，保证层与层之间用户分布独立
  - 数学要求：N(A∩C) = N(A) * N(C) / N_total
  - 作用：让 100% 的流量可以被多个实验同时使用

实验内容：
  1. 实现多层正交分流器
  2. 验证正交性（实测 vs 理论）
  3. 对比单层 vs 多层的偏差
  4. 卡方检验层间独立性
"""

import statistics
from collections import Counter
from typing import Dict, List

import mmh3
import numpy as np
from scipy import stats


# ============================ 多层正交分流器 ============================

class OrthogonalLayerRouter:
    """
    多层正交实验分流器

    每层是一个独立的实验，层与层之间正交：
      - 同层互斥：用户在同一层只能进一个组
      - 跨层正交：用户在每层的分组相互独立

    实现要点：
      - 每层用独立 salt（layer_id）
      - 桶数固定 1000，组数可配置
      - 实时路由，无状态
    """

    def __init__(self, layer_configs: Dict[str, Dict]):
        """
        layer_configs: {
            "recommend": {"num_buckets": 1000, "num_groups": 2, "salt": "rec"},
            "search":    {"num_buckets": 1000, "num_groups": 2, "salt": "sch"},
            "ui":        {"num_buckets": 1000, "num_groups": 3, "salt": "ui"},
        }
        """
        self.layer_configs = layer_configs

    def route(self, user_id: str, layer_name: str) -> int:
        """单层路由：用户 user_id 在 layer_name 层被分到哪个组"""
        cfg = self.layer_configs[layer_name]
        bucket = mmh3.hash(
            f"{user_id}_{cfg['salt']}",
            signed=False,
        ) % cfg["num_buckets"]
        return bucket % cfg["num_groups"]

    def route_all_layers(self, user_id: str) -> Dict[str, int]:
        """路由所有层"""
        return {
            layer: self.route(user_id, layer)
            for layer in self.layer_configs
        }


# ============================ 正交性验证 ============================

def check_orthogonality(
    layer_assignments: Dict[str, List[int]],
    layer_a: str,
    layer_b: str,
) -> Dict[str, float]:
    """
    验证两层之间的正交性

    正交要求：N(A_i ∩ B_j) = N(A_i) * N(B_j) / N_total

    检验方法：卡方检验观察列联表是否独立
    """
    a_groups = layer_assignments[layer_a]
    b_groups = layer_assignments[layer_b]
    n = len(a_groups)

    a_levels = sorted(set(a_groups))
    b_levels = sorted(set(b_groups))

    # 构造列联表
    contingency = np.zeros((len(a_levels), len(b_levels)))
    for a, b in zip(a_groups, b_groups):
        ai = a_levels.index(a)
        bi = b_levels.index(b)
        contingency[ai][bi] += 1

    # 卡方独立性检验
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

    # 计算正交偏差：每个单元格的 (实际 - 期望) / 期望
    deviations = (contingency - expected) / expected * 100

    # 理论正交：expected[i][j] = N(A_i) * N(B_j) / N
    return {
        "n": n,
        "chi2": chi2,
        "p_value": p_value,
        "dof": dof,
        "max_deviation_pct": float(np.max(np.abs(deviations))),
        "mean_deviation_pct": float(np.mean(np.abs(deviations))),
        "contingency": contingency.tolist(),
        "expected": expected.tolist(),
        "orthogonal_pass": p_value > 0.05,
    }


# ============================ 流量复用验证 ============================

def verify_traffic_reuse(
    layer_assignments: Dict[str, List[int]],
    layer_names: List[str],
) -> Dict[str, any]:
    """
    验证多层之间流量复用

    单层：100% 流量分两组 → 50% 进 A 组
    三层正交：A1∩B1∩C1 = 100% × 0.5 × 0.5 × 0.5 = 12.5%
    """
    # 统计每个用户在各层的组合
    n = len(layer_assignments[layer_names[0]])
    combinations = Counter()
    for i in range(n):
        combo = tuple(layer_assignments[layer][i] for layer in layer_names)
        combinations[combo] += 1

    total_combos = len(combinations)
    distribution = sorted(combinations.values(), reverse=True)

    # 理论均匀分布：每个组合期望 n / (G1 * G2 * G3)
    expected_per_combo = n / np.prod([
        len(set(layer_assignments[layer])) for layer in layer_names
    ])

    # 计算分布的均匀性
    max_count = distribution[0]
    min_count = distribution[-1]
    balance_ratio = min_count / max_count if max_count > 0 else 0

    return {
        "n_users": n,
        "n_combinations": total_combos,
        "expected_per_combo": expected_per_combo,
        "max_count": max_count,
        "min_count": min_count,
        "balance_ratio": balance_ratio,
        "top5_combinations": list(combinations.most_common(5)),
    }


# ============================ 实验运行 ============================

def run_orthogonal_experiment(
    n_users: int = 5000,
    n_trials: int = 50,
    seed_base: int = 20260728,
) -> None:
    """主实验：多层正交验证"""

    print("=" * 78)
    print(" 多层正交实验验证 (基于 5000 用户)".center(70))
    print("=" * 78)

    # 配置：3 层实验，每层 2 组
    layer_configs = {
        "L1_recommend": {"num_buckets": 1000, "num_groups": 2, "salt": "rec_v1"},
        "L2_search":    {"num_buckets": 1000, "num_groups": 2, "salt": "sch_v1"},
        "L3_ui":        {"num_buckets": 1000, "num_groups": 2, "salt": "ui_v1"},
    }

    router = OrthogonalLayerRouter(layer_configs)

    # 单次试验数据
    all_orthogonal_results = []
    all_traffic_results = []
    all_layer_deviations = []

    for trial in range(n_trials):
        rng = np.random.default_rng(seed_base + trial * 1000)
        user_ids = [f"u_{rng.integers(0, 10**9):09d}" for _ in range(n_users)]

        # 给每个用户在每层打标签
        layer_assignments = {
            layer: [router.route(uid, layer) for uid in user_ids]
            for layer in layer_configs
        }

        # 验证 L1-L2 正交性
        ortho_12 = check_orthogonality(layer_assignments, "L1_recommend", "L2_search")
        all_orthogonal_results.append(ortho_12)

        # 验证 L1-L3 正交性
        ortho_13 = check_orthogonality(layer_assignments, "L1_recommend", "L3_ui")
        all_orthogonal_results.append(ortho_13)

        # 验证三层流量复用
        traffic = verify_traffic_reuse(
            layer_assignments,
            ["L1_recommend", "L2_search", "L3_ui"],
        )
        all_traffic_results.append(traffic)

        # 各层偏差
        for layer in layer_configs:
            sizes = Counter(layer_assignments[layer])
            expected = n_users / layer_configs[layer]["num_groups"]
            max_diff = max(abs(c - expected) for c in sizes.values()) / expected * 100
            all_layer_deviations.append((layer, max_diff))

    # ====== 报告 ======

    # 1. 正交性验证
    print("\n[1] 两层正交性验证 (卡方独立性检验)")
    print("-" * 78)
    print(" H0: 两层之间独立（正交）")
    print(" H1: 两层之间不独立（不正交）")
    print(" 判定: p > 0.05 → 正交通过\n")

    p_values = [r["p_value"] for r in all_orthogonal_results]
    max_devs = [r["max_deviation_pct"] for r in all_orthogonal_results]
    mean_devs = [r["mean_deviation_pct"] for r in all_orthogonal_results]
    pass_rate = sum(1 for p in p_values if p > 0.05) / len(p_values)

    print(f"   卡方 p-value 中位数 : {statistics.median(p_values):.6f}")
    print(f"   卡方 p-value 最小值 : {min(p_values):.6f}")
    print(f"   正交通过率          : {pass_rate*100:.1f}%  ({sum(1 for p in p_values if p > 0.05)}/{len(p_values)})")
    print(f"   最大列联表偏差      : {statistics.mean(max_devs):.4f}%")
    print(f"   平均列联表偏差      : {statistics.mean(mean_devs):.4f}%")

    # 2. 流量复用验证
    print("\n[2] 三层流量复用验证")
    print("-" * 78)
    print(" 三层各 2 组 → 理论 2×2×2 = 8 种组合")
    print(" 每组合期望用户数 = 5000 / 8 = 625\n")

    balance_ratios = [r["balance_ratio"] for r in all_traffic_results]
    print("   实测组合数  : {all_traffic_results[0]['n_combinations']}/8")
    print(f"   最小/最大组合用户数比 : {statistics.mean(balance_ratios):.4f}")
    print(f"   平衡率中位数 : {statistics.median(balance_ratios):.4f}")
    print("   → 越接近 1.0 越均匀（理想正交 = 1.0）")

    # 3. 各层偏差
    print("\n[3] 各层最大组偏差 (5000 用户实时分流)")
    print("-" * 78)
    for layer in layer_configs:
        devs = [d for lyr, d in all_layer_deviations if lyr == layer]
        print(f"   {layer:<18} 平均偏差 {statistics.mean(devs):.4f}%  "
              f"P95 {np.percentile(devs, 95):.4f}%  "
              f"< 1%: {sum(1 for d in devs if d < 1)/len(devs)*100:.1f}%")

    # 4. 详细示例
    print("\n[4] 详细示例 (取第 0 次试验)")
    print("-" * 78)
    rng = np.random.default_rng(seed_base)
    user_ids = [f"u_{rng.integers(0, 10**9):09d}" for _ in range(100)]  # 100 用户示例
    layer_assignments = {
        layer: [router.route(uid, layer) for uid in user_ids]
        for layer in layer_configs
    }

    ortho = check_orthogonality(layer_assignments, "L1_recommend", "L2_search")
    print("\n   L1 (推荐) × L2 (搜索) 列联表:")
    print(f"   {'':<8}{'L2=0':>10}{'L2=1':>10}{'合计':>10}")
    for i, row in enumerate(ortho["contingency"]):
        a_label = f"L1={i}"
        b0 = int(row[0]) if len(row) > 0 else 0
        b1 = int(row[1]) if len(row) > 1 else 0
        total = int(sum(row))
        print(f"   {a_label:<8}{b0:>10}{b1:>10}{total:>10}")

    print(f"\n   卡方值 = {ortho['chi2']:.4f}")
    print(f"   p-value = {ortho['p_value']:.4f}")
    print(f"   自由度 = {ortho['dof']}")
    if ortho["orthogonal_pass"]:
        print("   结论: ✓ 正交通过（两层独立）")
    else:
        print("   结论: ✗ 正交失败（两层相关）")

    # 5. 实际流量复用示例
    print("\n[5] 三层 8 种组合的实际分布")
    print("-" * 78)
    traffic = verify_traffic_reuse(
        layer_assignments,
        ["L1_recommend", "L2_search", "L3_ui"],
    )
    print("   {'组合 (L1,L2,L3)':<20}{'用户数':<10}{'占比':<10}")
    for combo, count in traffic["top5_combinations"]:
        print("   {str(combo):<20}{count:<10}{count/traffic['n_users']*100:.2f}%")

    # 综合结论
    print("\n" + "=" * 78)
    print(" 综合结论")
    print("=" * 78)

    if pass_rate > 0.95:
        print(f" ✓ 多层正交性验证通过（{pass_rate*100:.1f}%）")
        print(f" ✓ 三层正交实现下，单层偏差 {statistics.mean([d for _, d in all_layer_deviations]):.2f}%")
        print(" ✓ 流量复用：100% 流量可被 3 个实验同时使用")
        print(" → 字节 DataTester 方案在小流量场景下验证可行")
    else:
        print(f" ✗ 正交性未通过（仅 {pass_rate*100:.1f}%）")
    print("=" * 78)


# ============================ 主入口 ============================

def main() -> None:
    run_orthogonal_experiment(
        n_users=5000,
        n_trials=50,
        seed_base=20260728,
    )


if __name__ == "__main__":
    main()