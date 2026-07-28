#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时事件分流 vs 批量分桶 对比实验
================================
对比两种分流策略在相同 5000 用户下的偏差表现：

策略A（批量/蛇形分配）：
  1. 全量哈希分1000桶
  2. 统计每桶人数
  3. 蛇形分配桶到组
  -> 偏差 ≤ 1%

策略B（实时/静态哈希映射）：
  bucket = hash(uid+salt) % 1000
  group  = bucket % 10
  -> 每次请求实时算，无全局视图，偏差受桶间波动影响
"""

import statistics
from typing import Dict, List

import numpy as np

from ab_split_validator import (
    NUM_BUCKETS,
    NUM_GROUPS,
    NUM_USERS,
    calc_hash_diff,
    hash_to_bucket,
    snake_assign,
    srm_check,
)


N_TRIALS = 100
BASE_SEED = 20260728


# ============ 策略A：批量蛇形分配 ============

def strategy_batch(user_ids: List[str], salt: str) -> Dict[int, List[str]]:
    """批量：全量分桶 + 蛇形分配"""
    buckets: Dict[int, List[str]] = {i: [] for i in range(NUM_BUCKETS)}
    for uid in user_ids:
        bid = hash_to_bucket(uid, salt, NUM_BUCKETS)
        buckets[bid].append(uid)
    return snake_assign(buckets, NUM_GROUPS)


# ============ 策略B：实时静态哈希 ============

def strategy_realtime(user_ids: List[str], salt: str) -> Dict[int, List[str]]:
    """实时：每次请求直接 hash -> bucket -> group，无法做蛇形校准"""
    groups: Dict[int, List[str]] = {i: [] for i in range(NUM_GROUPS)}
    for uid in user_ids:
        bucket_id = hash_to_bucket(uid, salt, NUM_BUCKETS)
        group_id = bucket_id % NUM_GROUPS
        groups[group_id].append(uid)
    return groups


# ============ 模拟实时事件流 ============

def simulate_streaming_events(user_ids: List[str], salt: str):
    """
    模拟实时事件流场景：
    - 用户请求逐个到达（在线服务实时处理）
    - 服务端对每个请求做一次 hash 决策，立即返回分组
    - 无第二次机会重哈希，无全局统计
    """
    # 在真实生产中，这里会逐个处理：
    # for event in event_stream:
    #     group = realtime_route(event.user_id, salt)
    # 这里用批量回放模拟
    group_counts = {i: 0 for i in range(NUM_GROUPS)}
    for uid in user_ids:
        bucket_id = hash_to_bucket(uid, salt, NUM_BUCKETS)
        group_id = bucket_id % NUM_GROUPS
        group_counts[group_id] += 1
    return group_counts


def run_trial(trial_id: int, seed: int) -> Dict[str, float]:
    """单次对比实验"""
    rng = np.random.default_rng(seed)
    user_ids = [f"u_{rng.integers(0, 10**9):09d}" for _ in range(NUM_USERS)]
    salt = f"exp_{trial_id}"

    # 策略A
    groups_a = strategy_batch(user_ids, salt)
    sizes_a = [len(groups_a[i]) for i in range(NUM_GROUPS)]

    # 策略B
    groups_b = strategy_realtime(user_ids, salt)
    sizes_b = [len(groups_b[i]) for i in range(NUM_GROUPS)]

    expected = NUM_USERS / NUM_GROUPS

    return {
        "trial": trial_id,
        "a_max_diff_pct": max(abs(s - expected) for s in sizes_a) / expected * 100,
        "a_hash_diff": calc_hash_diff(sizes_a),
        "a_srm_p": srm_check(sizes_a)[1],
        "b_max_diff_pct": max(abs(s - expected) for s in sizes_b) / expected * 100,
        "b_hash_diff": calc_hash_diff(sizes_b),
        "b_srm_p": srm_check(sizes_b)[1],
    }


def summarize(results: List[Dict[str, float]]) -> None:
    n = len(results)

    a_diff = [r["a_max_diff_pct"] for r in results]
    b_diff = [r["b_max_diff_pct"] for r in results]
    a_hash = [r["a_hash_diff"] for r in results]
    b_hash = [r["b_hash_diff"] for r in results]
    a_srm = [r["a_srm_p"] for r in results]
    b_srm = [r["b_srm_p"] for r in results]

    a_under_1 = sum(1 for x in a_diff if x < 1.0)
    b_under_1 = sum(1 for x in b_diff if x < 1.0)
    a_srm_pass = sum(1 for p in a_srm if p > 0.05)
    b_srm_pass = sum(1 for p in b_srm if p > 0.05)
    a_hash_pass = sum(1 for h in a_hash if h < 0.01)
    b_hash_pass = sum(1 for h in b_hash if h < 0.01)

    print("=" * 78)
    print(" 批量蛇形 vs 实时静态哈希 对比实验 (100次抽样) ".center(70))
    print("=" * 78)
    print(f" 配置: {NUM_USERS} 用户 / {NUM_BUCKETS} 桶 / {NUM_GROUPS} 组")

    print("\n[策略 A] 批量蛇形分配 (Batch + Snake Allocation)")
    print(f"   最大偏差 平均   : {statistics.mean(a_diff):.4f}%")
    print(f"   最大偏差 中位数 : {statistics.median(a_diff):.4f}%")
    print(f"   最大偏差 P95    : {np.percentile(a_diff, 95):.4f}%")
    print(f"   最大偏差 P99    : {np.percentile(a_diff, 99):.4f}%")
    print(f"   最大偏差 最大   : {max(a_diff):.4f}%")
    print(f"   Hash_diff  平均 : {statistics.mean(a_hash):.6f}")
    print(f"   < 1% 通过率     : {a_under_1}/{n} = {a_under_1/n*100:.2f}%")
    print(f"   Hash_diff<0.01  : {a_hash_pass}/{n} = {a_hash_pass/n*100:.2f}%")
    print(f"   SRM 通过率      : {a_srm_pass}/{n} = {a_srm_pass/n*100:.2f}%")

    print("\n[策略 B] 实时静态哈希 (Streaming + Static Mapping)")
    print(f"   最大偏差 平均   : {statistics.mean(b_diff):.4f}%")
    print(f"   最大偏差 中位数 : {statistics.median(b_diff):.4f}%")
    print(f"   最大偏差 P95    : {np.percentile(b_diff, 95):.4f}%")
    print(f"   最大偏差 P99    : {np.percentile(b_diff, 99):.4f}%")
    print(f"   最大偏差 最大   : {max(b_diff):.4f}%")
    print(f"   Hash_diff  平均 : {statistics.mean(b_hash):.6f}")
    print(f"   < 1% 通过率     : {b_under_1}/{n} = {b_under_1/n*100:.2f}%")
    print(f"   Hash_diff<0.01  : {b_hash_pass}/{n} = {b_hash_pass/n*100:.2f}%")
    print(f"   SRM 通过率      : {b_srm_pass}/{n} = {b_srm_pass/n*100:.2f}%")

    # 偏差对比表
    print("\n[直接对比] 平均偏差倍数关系")
    ratio = statistics.mean(b_diff) / statistics.mean(a_diff) if statistics.mean(a_diff) > 0 else 0
    print(f"   B/A 偏差倍数 = {ratio:.2f}x")
    print(f"   解读: 实时策略平均偏差是批量策略的 {ratio:.1f} 倍")

    # 越界统计
    print("\n[边界分析] 1% 阈值越界情况")
    print(f"   策略A 越界次数 : {n - a_under_1} 次")
    print(f"   策略B 越界次数 : {n - b_under_1} 次")

    # 实时场景的可补救措施
    print("\n[实时场景补救方案]")
    print("   方案1: 增加桶数 1000→10000（每桶期望人数从5降到0.5，波动√n缩小）")
    print("   方案2: 多salt轮询+加权聚合（牺牲性能换均匀性）")
    print("   方案3: 定期批量重平衡（牺牲一致性换均匀性）")

    print("\n" + "=" * 78)
    print(" 综合结论:")
    print(f"   批量策略: 偏差 {statistics.mean(a_diff):.2f}%，{'达标' if statistics.mean(a_diff) < 1 else '不达标'}")
    print(f"   实时策略: 偏差 {statistics.mean(b_diff):.2f}%，{'达标' if statistics.mean(b_diff) < 1 else '不达标'}")
    print("=" * 78)

    # 前 5 次明细
    print("\n[附录] 前 5 次对比明细")
    print(f" {'trial':<6}{'A_max%':<10}{'A_hash':<10}{'A_SRM':<12}"
          f"{'B_max%':<10}{'B_hash':<10}{'B_SRM'}")
    print("-" * 70)
    for r in results[:5]:
        print(f" {r['trial']:<6}{r['a_max_diff_pct']:<10.4f}{r['a_hash_diff']:<10.6f}"
              f"{r['a_srm_p']:<12.4f}{r['b_max_diff_pct']:<10.4f}"
              f"{r['b_hash_diff']:<10.6f}{r['b_srm_p']:.4f}")


def main() -> None:
    print(f"开始执行 {N_TRIALS} 次对比实验...")
    results = [run_trial(i, BASE_SEED + i * 1000) for i in range(N_TRIALS)]
    summarize(results)


if __name__ == "__main__":
    main()