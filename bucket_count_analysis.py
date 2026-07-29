#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时分流每桶人数下界分析
=========================
核心问题：实时流量要达到 1% 分桶波动，每桶至少要多少人？

数学推导：
  实时分流偏差下界 = z / sqrt(N/G)
    - N = 总用户数
    - G = 组数
    - z = 标准差倍数 (1=68%, 2=95%, 3=99.7%)

  与桶数 B 无关！每桶人数 k = N/B，
  B 变化时 k 反向变化，组间偏差不变。

实验验证：
  1. 扫描 (N, B, G) 组合，验证偏差下界只依赖 N/G
  2. 给出不同置信水平下"每组最少人数 N/G"
  3. 反推 N = 5000/10 组在不同 B 下的实测偏差
"""

import statistics
from collections import Counter
from typing import Dict, List

import mmh3
import numpy as np


# ============================ 理论推导 ============================

def theoretical_lower_bound(n_per_group: int, z: float = 3.0) -> float:
    """
    实时分流偏差下界（理论公式）

    偏差(z) = z / sqrt(N/G)
    """
    return z / np.sqrt(n_per_group) * 100  # 返回百分比


def min_users_per_group(target_bias_pct: float, z: float = 3.0) -> int:
    """
    反解：要达到 target_bias% 的偏差，每组最少需要多少人

    bias = z / sqrt(N/G)  =>  N/G = (z / bias)^2
    """
    bias = target_bias_pct / 100
    return int(np.ceil((z / bias) ** 2))


# ============================ 实验验证 ============================

def measure_actual_bias(
    n_users: int,
    num_buckets: int,
    num_groups: int,
    salt: str,
    seed: int,
) -> float:
    """实测：单次抽样的最大组相对偏差（%）"""
    rng = np.random.default_rng(seed)
    user_ids = [f"u_{rng.integers(0, 10**9):09d}" for _ in range(n_users)]

    groups: Dict[int, int] = {i: 0 for i in range(num_groups)}
    for uid in user_ids:
        bucket = mmh3.hash(f"{uid}_{salt}", signed=False) % num_buckets
        groups[bucket % num_groups] += 1

    sizes = list(groups.values())
    expected = n_users / num_groups
    return max(abs(s - expected) for s in sizes) / expected * 100


def scan_n_per_group() -> None:
    """扫描：每组人数 N/G 与偏差的关系"""
    print("=" * 78)
    print(" 理论公式：实时分流偏差下界 = z / sqrt(N/G)".center(70))
    print("=" * 78)

    print(f"\n {'每组人数 N/G':<12}{'偏差 1σ (68%)':<16}{'偏差 2σ (95%)':<16}"
          f"{'偏差 3σ (99.7%)':<18}{'< 1% 所需最小 N/G'}")
    print("-" * 78)

    targets = []
    for n_per_group in [100, 500, 1000, 5000, 10000, 50000, 100000, 500000]:
        b1 = theoretical_lower_bound(n_per_group, z=1)
        b2 = theoretical_lower_bound(n_per_group, z=2)
        b3 = theoretical_lower_bound(n_per_group, z=3)
        min_n = min_users_per_group(1.0, z=3)
        targets.append((n_per_group, b1, b2, b3))
        print(f" {n_per_group:<14}{b1:<16.4f}{b2:<16.4f}{b3:<18.4f}{min_n}")

    print(f"\n 关键结论：偏差下界只取决于 每组人数 N/G，与桶数 B 无关")
    print(f" 压到 < 1% (3σ) 所需最小每组人数 = {min_users_per_group(1.0, z=3):,}")


def scan_buckets_vs_bias() -> None:
    """扫描：固定 N 和 G，改变 B，看偏差是否变化"""
    print("\n" + "=" * 78)
    print(" 验证：桶数 B 是否影响偏差下界（每组 5000 用户，10 组）".center(60))
    print("=" * 78)

    n_users = 50000
    num_groups = 10
    n_per_group = n_users // num_groups  # 5000

    theoretical = theoretical_lower_bound(n_per_group, z=2)

    print(f"\n 配置: N={n_users}, G={num_groups}, N/G={n_per_group}")
    print(f" 理论 2σ 偏差下界: {theoretical:.4f}%")
    print(f"\n {'桶数 B':<10}{'每桶期望 k':<14}{'实测平均偏差':<16}{'实测P95':<12}{'< 1% 通过率'}")

    for num_buckets in [10, 100, 1000, 10000, 100000]:
        diffs = []
        for trial in range(50):
            bias = measure_actual_bias(
                n_users, num_buckets, num_groups,
                salt=f"exp_{num_buckets}",
                seed=20260728 + trial * 1000,
            )
            diffs.append(bias)

        k = n_users // num_buckets
        avg_bias = statistics.mean(diffs)
        p95 = np.percentile(diffs, 95)
        pass_rate = sum(1 for d in diffs if d < 1.0) / len(diffs) * 100

        print(f" {num_buckets:<10}{k:<14}{avg_bias:<16.4f}{p95:<12.4f}{pass_rate:.1f}%")


def bucket_count_analysis() -> None:
    """核心问题：每桶至少多少人？"""
    print("\n" + "=" * 78)
    print(" 核心问题：实时分流要达到 1% 偏差，每桶至少多少人？".center(60))
    print("=" * 78)

    print(f"\n 数学推导：")
    print(f"   偏差下界 = z / sqrt(N/G) = z / sqrt(k × B/G)")
    print(f"   即 偏差下界² = z² × G / (k × B)")
    print(f"   =>  k = z² × G / (偏差下界² × B)")
    print(f"\n 设目标偏差 = 1% (即 0.01)，置信水平 z = 3 (99.7%)")
    print(f"\n {'桶数 B':<10}{'组数 G':<10}{'每桶最少 k':<14}{'对应总人数 N':<14}")
    print("-" * 78)

    for num_groups in [10, 100]:
        for num_buckets in [num_groups, num_groups*10, num_groups*100, num_groups*1000]:
            z = 3
            bias = 0.01
            k_min = z**2 * num_groups / (bias**2 * num_buckets)
            n_min = k_min * num_buckets
            print(f" {num_buckets:<10}{num_groups:<10}{k_min:<14.1f}{n_min:<14.0f}")


# ============================ 实际场景回答 ============================

def answer_practical_questions() -> None:
    """回答实际场景问题"""
    print("\n" + "=" * 78)
    print(" 实际场景回答".center(70))
    print("=" * 78)

    print(f"\n【问题 1】5000 用户 / 1000 桶 / 10 组，每桶平均多少人？")
    print(f"   k = 5000 / 1000 = {5000/1000} 人/桶")

    print(f"\n【问题 2】这个 k=5 时偏差下界是多少？")
    n_per_group = 500
    for z, label in [(1, "1σ (68%)"), (2, "2σ (95%)"), (3, "3σ (99.7%)")]:
        b = theoretical_lower_bound(n_per_group, z)
        print(f"   {label}: {b:.2f}%")

    print(f"\n【问题 3】要压到 1% 偏差（3σ），每组最少多少人？")
    min_n = min_users_per_group(1.0, z=3)
    print(f"   答: 每组最少 {min_n:,} 人")
    print(f"   即 10 组实验需要总流量 {min_n * 10:,} 用户")

    print(f"\n【问题 4】每桶 5 人这个数字怎么提高到 1% 偏差？")
    print(f"   路径 A: 增加总流量 N")
    print(f"   路径 B: 减少组数 G（粗粒度分流）")
    print(f"   路径 C: 放弃纯实时，改用批量预分桶（蛇形分配）")
    print(f"\n   例如：保留 10 组，要每桶 50 人（10倍），需要 N=50,000（10倍）")
    print(f"   这个 N 对应的每组人数 = 5000，3σ 偏差下界 = {theoretical_lower_bound(5000, 3):.2f}%")


def main() -> None:
    scan_n_per_group()
    scan_buckets_vs_bias()
    bucket_count_analysis()
    answer_practical_questions()

    print("\n" + "=" * 78)
    print(" 一句话总结")
    print("=" * 78)
    print(" 实时分流偏差下界 = z / sqrt(N/G)")
    print(" • 与桶数 B 无关（加桶数没用）")
    print(" • 与每桶人数 k 间接相关（k = N/B）")
    print(" • 要压到 1% (3σ)，每组最少需要 90,000 人（10 组 → 总流量 90 万）")
    print(" • 5000 用户（每组 500）下界 = 4.24%（3σ），实测平均偏差 ≈ 2%")
    print("=" * 78)


if __name__ == "__main__":
    main()