#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时分流偏差 vs 流量规模验证
==========================
扫描不同用户量下的实测偏差 vs 理论下界。

输出：每个 N 对应的实测平均偏差、P95、理论 95% 置信下界。

注意：
  - 实测平均偏差 < 理论下界（95% 上界 ≈ 1.96σ）
  - 平均偏差接近 1σ ≈ 64% 分位
  - "流量分配偏差" ≠ "MDE"
"""
import numpy as np
import mmh3


def measure_avg_bias(n_users, num_buckets=1000, num_groups=10, n_trials=100):
    """单次抽样的最大组相对偏差（%）"""
    diffs = []
    for trial in range(n_trials):
        rng = np.random.default_rng(20260728 + trial * 1000)
        user_ids = [f"u_{rng.integers(0, 10**9):09d}" for _ in range(n_users)]
        sizes = [0] * num_groups
        for uid in user_ids:
            b = mmh3.hash(f"{uid}_exp_{trial}", signed=False) % num_buckets
            sizes[b % num_groups] += 1
        expected = n_users / num_groups
        max_diff = max(abs(s - expected) for s in sizes) / expected * 100
        diffs.append(max_diff)
    return float(np.mean(diffs)), float(np.percentile(diffs, 95))


def theoretical_lower_bound(n_users, num_groups=10, z=1.96):
    """95% 置信下界 = z / sqrt(N/G) * 100%"""
    return z / np.sqrt(n_users / num_groups) * 100


def main():
    print(f"{'N':>7} {'每组':>6} {'理论下界':>10} {'实测平均':>10} {'实测P95':>10} {'<1%':>6}")
    print("-" * 60)
    for n in [500, 1000, 5000, 10000, 50000, 100000]:
        avg, p95 = measure_avg_bias(n)
        theoretical = theoretical_lower_bound(n)
        pass_rate = 0  # 100 次抽样中 < 1% 的占比
        diffs = []
        for trial in range(100):
            rng = np.random.default_rng(20260728 + trial * 1000)
            user_ids = [f"u_{rng.integers(0, 10**9):09d}" for _ in range(n)]
            sizes = [0] * 10
            for uid in user_ids:
                b = mmh3.hash(f"{uid}_exp_{trial}", signed=False) % 1000
                sizes[b % 10] += 1
            expected = n / 10
            max_diff = max(abs(s - expected) for s in sizes) / expected * 100
            diffs.append(max_diff)
        pass_rate = sum(1 for d in diffs if d < 1.0) / len(diffs) * 100
        print(f"{n:>7} {n//10:>6} {theoretical:>9.2f}% {avg:>9.2f}% {p95:>9.2f}% {pass_rate:>5.0f}%")

    print("\n关键说明：")
    print("  - 理论下界 = 95% 置信上界（z=1.96），不是平均值")
    print("  - 实测平均偏差通常接近 1σ ≈ 64% 分位")
    print("  - 这里的偏差是'流量分配偏差'，不是 MDE")


if __name__ == "__main__":
    main()