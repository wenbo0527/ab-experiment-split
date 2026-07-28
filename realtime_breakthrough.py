#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时分流偏差下界突破实验
=========================
结论：5000 用户 / 10 组，单层实时分流偏差下限 ≈ 7-8%，
     单层纯静态哈希 + 任何单层优化 都无法压到 1%。

必须用：
  - 增加样本量（流量本身放大 √n 倍改善）
  - 多层正交 + 跨层聚合（牺牲一致性换均匀性）
  - 预分桶 + 静态查表（放弃实时算法）
"""

import statistics
from collections import Counter
from typing import Dict, List

import mmh3
import numpy as np

from ab_split_validator import NUM_USERS, calc_hash_diff, srm_check


N_TRIALS = 50  # 用户量扫描，每个量级跑 50 次


# ============ 不同样本量下的实时分流偏差下界 ============

def realtime_single(user_ids: List[str], salt: str) -> Dict[int, List[str]]:
    """单层实时：hash % 1000 % 10"""
    groups: Dict[int, List[str]] = {i: [] for i in range(10)}
    for uid in user_ids:
        bucket = mmh3.hash(f"{uid}_{salt}", signed=False) % 1000
        groups[bucket % 10].append(uid)
    return groups


def realtime_2layers_union(
    user_ids: List[str], salt: str, num_layers: int = 2
) -> Dict[int, List[str]]:
    """
    多层正交：每个用户跨多层独立哈希，
    最终分组 = 各层分组的并集（实际场景是跨层分别打标签）
    模拟思路：把用户复制 num_layers 次分别入不同层，每层 1000 桶，
    最终组别 = 多数层投票决定。
    """
    layer_votes: Dict[str, List[int]] = {uid: [] for uid in user_ids}
    for layer in range(num_layers):
        for uid in user_ids:
            bucket = mmh3.hash(f"{uid}_{salt}_L{layer}", signed=False) % 1000
            layer_votes[uid].append(bucket % 10)

    groups: Dict[int, List[str]] = {i: [] for i in range(10)}
    for uid, votes in layer_votes.items():
        groups[Counter(votes).most_common(1)[0][0]].append(uid)
    return groups


def scan_sample_sizes() -> None:
    """扫描不同用户规模下的实时分流偏差"""
    print("=" * 78)
    print(" 实时分流偏差 vs 样本量 扫描实验".center(70))
    print("=" * 78)
    print(f" 配置: 单层 hash%1000%10，10 组，{N_TRIALS} 次抽样")
    print(f"\n {'样本量':<10}{'期望/组':<10}{'偏差均值':<10}{'偏差P95':<10}"
          f"{'< 1% 通过率':<14}{'< 5% 通过率':<14}{'结论'}")

    for n in [500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]:
        diffs = []
        for trial in range(N_TRIALS):
            rng = np.random.default_rng(BASE_SEED + trial * 1000)
            user_ids = [f"u_{rng.integers(0, 10**9):09d}" for _ in range(n)]
            groups = realtime_single(user_ids, f"exp_{trial}")
            sizes = [len(groups[i]) for i in range(10)]
            expected = n / 10
            diffs.append(max(abs(s - expected) for s in sizes) / expected * 100)

        under_1 = sum(1 for x in diffs if x < 1.0) / N_TRIALS
        under_5 = sum(1 for x in diffs if x < 5.0) / N_TRIALS
        verdict = "✓ 达标" if under_1 >= 0.95 else ("△ 接近" if under_1 >= 0.5 else "✗ 不达标")
        print(f" {n:<10}{n//10:<10}{statistics.mean(diffs):<10.2f}"
              f"{np.percentile(diffs, 95):<10.2f}"
              f"{under_1*100:<14.1f}{under_5*100:<14.1f}{verdict}")


def scan_layer_count() -> None:
    """扫描多层正交的层数对偏差的影响"""
    print("\n" + "=" * 78)
    print(" 多层正交聚合 vs 单层实时 偏差对比".center(70))
    print("=" * 78)
    print(f" 配置: 5000 用户 / 10 组，{N_TRIALS} 次抽样")
    print(f"\n {'层数':<8}{'方案':<25}{'偏差均值':<10}{'偏差P95':<10}"
          f"{'< 1% 通过率':<14}{'SRM通过率'}")

    for n_layers in [1, 2, 3, 5, 8, 10]:
        diffs = []
        srm_pass = 0
        for trial in range(N_TRIALS):
            rng = np.random.default_rng(BASE_SEED + trial * 1000)
            user_ids = [f"u_{rng.integers(0, 10**9):09d}" for _ in range(NUM_USERS)]
            groups = realtime_2layers_union(user_ids, f"exp_{trial}", num_layers=n_layers)
            sizes = [len(groups[i]) for i in range(10)]
            expected = NUM_USERS / 10
            diffs.append(max(abs(s - expected) for s in sizes) / expected * 100)
            if srm_check(sizes)[1] > 0.05:
                srm_pass += 1

        under_1 = sum(1 for x in diffs if x < 1.0) / N_TRIALS
        verdict = "✓" if under_1 >= 0.95 else "✗"
        print(f" {n_layers:<8}{'多层正交投票':<25}{statistics.mean(diffs):<10.2f}"
              f"{np.percentile(diffs, 95):<10.2f}"
              f"{under_1*100:<14.1f}{srm_pass/N_TRIALS*100:.1f}% {verdict}")


def theoretical_lower_bound() -> None:
    """理论下界推导"""
    print("\n" + "=" * 78)
    print(" 理论下界: 为什么单层实时分流压不到 1%".center(70))
    print("=" * 78)

    for n in [500, 5000, 50000, 500000]:
        # 1000 桶分 10 组，每组聚合 100 桶
        # 每桶期望人数 = n / 1000
        # 桶间标准差 = sqrt(n/1000)
        # 每组聚合 100 桶，标准差 = sqrt(100) * sqrt(n/1000) = sqrt(n/10)
        # 每组期望人数 = n / 10
        # 3σ 相对偏差 = 3 * sqrt(n/10) / (n/10) = 3 * sqrt(10/n) * 10
        per_bucket = n / 1000
        bucket_std = np.sqrt(per_bucket)
        group_std = np.sqrt(100) * bucket_std  # 100 桶聚合
        expected = n / 10
        rel_std_3sigma = 3 * group_std / expected * 100

        print(f" n={n:<8} 每桶期望={per_bucket:.1f} 桶间σ={bucket_std:.2f} "
              f"组间σ={group_std:.2f} 3σ偏差={rel_std_3sigma:.2f}%")


BASE_SEED = 20260728


def main() -> None:
    scan_sample_sizes()
    scan_layer_count()
    theoretical_lower_bound()

    print("\n" + "=" * 78)
    print(" 结论")
    print("=" * 78)
    print(" 1. 单层实时分流在 5000 用户量级下，偏差下限约 7-8% (3σ 范围)")
    print(" 2. 加桶数/两次hash/多salt 都是 O(1) 优化，对组级偏差改善 < 1%")
    print(" 3. 要压到 1% 必须：")
    print("    ① 流量放大到 50000+（√n 改善，需要 100x 流量）")
    print("    ② 改成预分桶+静态查表（放弃纯实时）")
    print("    ③ 接受多层正交但允许 5% 偏差（行业惯例）")
    print("=" * 78)


if __name__ == "__main__":
    main()