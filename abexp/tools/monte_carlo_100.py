#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
100次重复抽样 Monte Carlo 实验
==============================
每次独立生成 5000 随机用户ID，用相同的蛇形分配算法分10组，
统计 Hash_diff、最大组偏差、SRM 通过率等指标的分布。

目的：验证算法在随机用户群体下的稳定性（不只是固定ID列表）。
"""

import statistics
from typing import Dict, List

import numpy as np

from abexp.routing.ab_split_validator import (
    NUM_BUCKETS,
    NUM_GROUPS,
    NUM_USERS,
    calc_hash_diff,
    snake_assign,
    hash_to_bucket,
    srm_check,
)


N_TRIALS = 100
BASE_SEED = 20260728


def run_one_trial(trial_id: int, seed: int) -> Dict[str, float]:
    """单次抽样实验"""
    rng = np.random.default_rng(seed)

    # 随机生成 5000 个用户ID（模拟真实场景：每次进入实验的用户不同）
    user_ids = [
        f"u_{rng.integers(0, 10**9):09d}" for _ in range(NUM_USERS)
    ]

    # 分桶
    buckets: Dict[int, List[str]] = {i: [] for i in range(NUM_BUCKETS)}
    for uid in user_ids:
        bid = hash_to_bucket(uid, f"exp_{trial_id}", NUM_BUCKETS)
        buckets[bid].append(uid)

    # 蛇形分配
    groups = snake_assign(buckets, NUM_GROUPS)
    sizes = [len(groups[i]) for i in range(NUM_GROUPS)]
    expected = NUM_USERS / NUM_GROUPS

    # 指标计算
    max_abs_diff = max(abs(s - expected) for s in sizes)
    max_abs_diff_pct = max_abs_diff / expected * 100
    hash_diff = calc_hash_diff(sizes)

    chi2, p, srm_passed, _ = srm_check(sizes)

    # 最大偏差组
    max_group_idx = int(np.argmax(sizes))
    min_group_idx = int(np.argmin(sizes))

    return {
        "trial": trial_id,
        "max_diff_abs": max_abs_diff,
        "max_diff_pct": max_abs_diff_pct,
        "hash_diff": hash_diff,
        "srm_chi2": chi2,
        "srm_p": p,
        "srm_passed": srm_passed,
        "max_size": max(sizes),
        "min_size": min(sizes),
        "max_group": max_group_idx,
        "min_group": min_group_idx,
    }


def summarize(results: List[Dict[str, float]]) -> None:
    """统计汇总"""
    n = len(results)
    max_diff_pcts = [r["max_diff_pct"] for r in results]
    hash_diffs = [r["hash_diff"] for r in results]
    srm_ps = [r["srm_p"] for r in results]
    srm_passed_cnt = sum(1 for r in results if r["srm_passed"])
    hash_passed_cnt = sum(1 for r in results if r["hash_diff"] < 0.01)
    diff_under_1pct = sum(1 for x in max_diff_pcts if x < 1.0)

    print("=" * 78)
    print(" 100次重复抽样 Monte Carlo 实验报告 ".center(70))
    print("=" * 78)
    print(f" 实验配置        : 每次 {NUM_USERS} 用户 / {NUM_BUCKETS} 桶 / {NUM_GROUPS} 组")
    print(f" 抽样次数        : {N_TRIALS}")
    print(" 用户ID生成方式  : 每次随机生成（独立抽样）")
    print("-" * 78)

    # 最大组偏差(%) 分布
    print("\n[指标 1] 最大组人数偏差 (%)")
    print(f"   最小值       : {min(max_diff_pcts):.4f}%")
    print(f"   最大值       : {max(max_diff_pcts):.4f}%")
    print(f"   平均值       : {statistics.mean(max_diff_pcts):.4f}%")
    print(f"   中位数       : {statistics.median(max_diff_pcts):.4f}%")
    print(f"   标准差       : {statistics.pstdev(max_diff_pcts):.4f}%")
    print(f"   P95          : {np.percentile(max_diff_pcts, 95):.4f}%")
    print(f"   P99          : {np.percentile(max_diff_pcts, 99):.4f}%")
    print(f"   < 1% 通过率  : {diff_under_1pct}/{n} = {diff_under_1pct/n*100:.2f}%")

    # Hash_diff 分布
    print("\n[指标 2] Hash_diff (组间人数相对标准差)")
    print(f"   最小值       : {min(hash_diffs):.6f}")
    print(f"   最大值       : {max(hash_diffs):.6f}")
    print(f"   平均值       : {statistics.mean(hash_diffs):.6f}")
    print(f"   中位数       : {statistics.median(hash_diffs):.6f}")
    print(f"   < 0.01 通过率: {hash_passed_cnt}/{n} = {hash_passed_cnt/n*100:.2f}%")

    # SRM 分布
    print("\n[指标 3] SRM 卡方检验 (p-value)")
    print(f"   最小值       : {min(srm_ps):.6f}")
    print(f"   最大值       : {max(srm_ps):.6f}")
    print(f"   中位数       : {statistics.median(srm_ps):.6f}")
    print(f"   < 0.05 通过率: {sum(1 for p in srm_ps if p > 0.05)}/{n} "
          f"= {sum(1 for p in srm_ps if p > 0.05)/n*100:.2f}%")
    print(f"   < 0.001 失败率: {sum(1 for p in srm_ps if p < 0.001)}/{n} "
          f"= {sum(1 for p in srm_ps if p < 0.001)/n*100:.2f}%")
    print(f"   SRM通过率    : {srm_passed_cnt}/{n} = {srm_passed_cnt/n*100:.2f}%")

    # 直方图（最大偏差分布）
    print("\n[指标 4] 最大偏差分布直方图")
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 100]
    hist, _ = np.histogram(max_diff_pcts, bins=bins)
    max_bar = max(hist) if max(hist) > 0 else 1
    for i, count in enumerate(hist):
        bar_len = int(count / max_bar * 40)
        pct_label = f"[{bins[i]:.1f}%, {bins[i+1]:.1f}%)"
        print(f"   {pct_label:<18} {'█' * bar_len:<40} {count}")

    # 综合判定
    all_pass = (
        diff_under_1pct == n
        and hash_passed_cnt == n
        and srm_passed_cnt == n
    )
    print("\n" + "=" * 78)
    if all_pass:
        print(" 综合结论: ✓ 100/100 全部达标，算法稳健可靠")
    else:
        print(f" 综合结论: Hash_diff<1% {hash_passed_cnt}/{n}, "
              f"SRM通过 {srm_passed_cnt}/{n}, "
              f"最大偏差<1% {diff_under_1pct}/{n}")
    print("=" * 78)

    # 前 10 次明细
    print("\n[附录] 前 10 次实验明细")
    print(" {'trial':<6}{'max_size':<10}{'min_size':<10}{'max_diff%':<12}"
          "{'hash_diff':<12}{'SRM p-value':<14}{'verdict'}")
    print("-" * 70)
    for r in results[:10]:
        verdict = "PASS" if r["hash_diff"] < 0.01 and r["srm_passed"] else "FAIL"
        print(f" {r['trial']:<6}{r['max_size']:<10}{r['min_size']:<10}"
              f"{r['max_diff_pct']:<12.4f}{r['hash_diff']:<12.6f}"
              f"{r['srm_p']:<14.6f}{verdict}")


def main() -> None:
    print(f"开始执行 {N_TRIALS} 次重复抽样实验...")
    print(f"基础种子: {BASE_SEED}")

    results: List[Dict[str, float]] = []
    for trial in range(N_TRIALS):
        seed = BASE_SEED + trial * 1000
        result = run_one_trial(trial, seed)
        results.append(result)

    summarize(results)


if __name__ == "__main__":
    main()