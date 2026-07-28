#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时分流偏差补救方案对比实验
=============================
目标：在实时事件流场景下，把组间人数偏差压到 < 1%

测试 4 种方案：
  R0: 基准 - 单次 hash % 1000 % 10（已知偏差 8%）
  R1: 加桶数 - hash % 10000 % 10
  R2: 两次 hash 叠加 - hash1()%1000 XOR hash2()%1000，再 %10
  R3: 多 salt 轮询 - 4 个 salt 取众数
  R4: 桶数+两次 hash - 10000 桶 + 两次独立 hash

每种方案 100 次重复抽样，对比最大偏差均值、< 1% 通过率、SRM 通过率。
"""

import statistics
from collections import Counter
from typing import Dict, List

import mmh3
import numpy as np

from ab_split_validator import NUM_USERS, calc_hash_diff, srm_check


N_TRIALS = 100
BASE_SEED = 20260728


# ============ 实时分流函数 ============

def realtime_r0(user_ids: List[str], salt: str) -> Dict[int, List[str]]:
    """R0: 基准单次哈希"""
    return _static_hash(user_ids, salt, num_buckets=1000)


def realtime_r1(user_ids: List[str], salt: str) -> Dict[int, List[str]]:
    """R1: 1000→10000 桶"""
    return _static_hash(user_ids, salt, num_buckets=10000)


def realtime_r2(user_ids: List[str], salt: str) -> Dict[int, List[str]]:
    """R2: 两次 hash 异或"""
    groups: Dict[int, List[str]] = {i: [] for i in range(10)}
    for uid in user_ids:
        h1 = mmh3.hash(f"{uid}_{salt}_v1", signed=False)
        h2 = mmh3.hash(f"{uid}_{salt}_v2", signed=False)
        bucket = (h1 ^ h2) % 1000
        groups[bucket % 10].append(uid)
    return groups


def realtime_r3(user_ids: List[str], salt: str) -> Dict[int, List[str]]:
    """R3: 4 个 salt 众数投票"""
    salts = [f"{salt}_s{i}" for i in range(4)]
    groups: Dict[int, List[str]] = {i: [] for i in range(10)}
    for uid in user_ids:
        votes = [
            mmh3.hash(f"{uid}_{s}", signed=False) % 10
            for s in salts
        ]
        # 众数（票数最多，平票取最小编号）
        group_id = Counter(votes).most_common(1)[0][0]
        groups[group_id].append(uid)
    return groups


def realtime_r4(user_ids: List[str], salt: str) -> Dict[int, List[str]]:
    """R4: 10000 桶 + 两次 hash 叠加"""
    groups: Dict[int, List[str]] = {i: [] for i in range(10)}
    for uid in user_ids:
        h1 = mmh3.hash(f"{uid}_{salt}_v1", signed=False)
        h2 = mmh3.hash(f"{uid}_{salt}_v2", signed=False)
        bucket = (h1 ^ h2) % 10000
        groups[bucket % 10].append(uid)
    return groups


def _static_hash(user_ids: List[str], salt: str, num_buckets: int) -> Dict[int, List[str]]:
    groups: Dict[int, List[str]] = {i: [] for i in range(10)}
    for uid in user_ids:
        bucket = mmh3.hash(f"{uid}_{salt}", signed=False) % num_buckets
        groups[bucket % 10].append(uid)
    return groups


# ============ 实验执行 ============

def run_trial(trial_id: int, seed: int, strategy_fn) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    user_ids = [f"u_{rng.integers(0, 10**9):09d}" for _ in range(NUM_USERS)]
    salt = f"exp_{trial_id}"

    groups = strategy_fn(user_ids, salt)
    sizes = [len(groups[i]) for i in range(10)]
    expected = NUM_USERS / 10

    return {
        "max_diff_pct": max(abs(s - expected) for s in sizes) / expected * 100,
        "hash_diff": calc_hash_diff(sizes),
        "srm_p": srm_check(sizes)[1],
    }


def summarize(name: str, results: List[Dict[str, float]]) -> Dict[str, float]:
    diffs = [r["max_diff_pct"] for r in results]
    hashes = [r["hash_diff"] for r in results]
    srms = [r["srm_p"] for r in results]

    n = len(results)
    under_1 = sum(1 for x in diffs if x < 1.0)
    hash_pass = sum(1 for h in hashes if h < 0.01)
    srm_pass = sum(1 for p in srms if p > 0.05)

    print(f"\n[{name}]")
    print(f"   最大偏差 平均 : {statistics.mean(diffs):.4f}%")
    print(f"   最大偏差 中位 : {statistics.median(diffs):.4f}%")
    print(f"   最大偏差 P95  : {np.percentile(diffs, 95):.4f}%")
    print(f"   最大偏差 最大 : {max(diffs):.4f}%")
    print(f"   < 1% 通过率   : {under_1}/{n} = {under_1/n*100:.1f}%")
    print(f"   Hash_diff<0.01: {hash_pass}/{n} = {hash_pass/n*100:.1f}%")
    print(f"   SRM 通过率    : {srm_pass}/{n} = {srm_pass/n*100:.1f}%")

    return {
        "name": name,
        "avg_diff": statistics.mean(diffs),
        "pass_rate": under_1 / n,
    }


def main() -> None:
    print(f"开始执行 {N_TRIALS} 次 × 5 策略对比实验...")
    print(f"目标: 实时场景下压到 < 1% 偏差\n")

    strategies = [
        ("R0: 单次 hash % 1000 % 10", realtime_r0),
        ("R1: 加桶数 % 10000 % 10", realtime_r1),
        ("R2: 两次 hash 异或 % 1000", realtime_r2),
        ("R3: 4-salt 众数投票", realtime_r3),
        ("R4: 10000桶 + 两次 hash", realtime_r4),
    ]

    summary_list = []
    for name, fn in strategies:
        results = [run_trial(i, BASE_SEED + i * 1000, fn) for i in range(N_TRIALS)]
        s = summarize(name, results)
        summary_list.append(s)

    # 综合排名
    print("\n" + "=" * 70)
    print(" 综合排名 (按 < 1% 通过率)")
    print("=" * 70)
    ranked = sorted(summary_list, key=lambda x: -x["pass_rate"])
    for i, s in enumerate(ranked, 1):
        bar = "█" * int(s["pass_rate"] * 40)
        marker = " ✓ 达标" if s["pass_rate"] >= 0.95 else " ✗ 不达标"
        print(f"  {i}. {s['name']:<35} {s['avg_diff']:6.2f}%  {bar} {s['pass_rate']*100:5.1f}%{marker}")


if __name__ == "__main__":
    main()