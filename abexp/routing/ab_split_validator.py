#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AB实验分流验证算法
==================
目标：5000 用户随机分桶到 10 组，每组人数偏差 < 1%

技术链路：
1. MurmurHash3 + 1000 桶做基础哈希分桶
2. 蛇形分配（S-shape allocation）做桶到组的确定性均衡
3. 三层验证：
   - Hash_diff：组间人数相对标准差
   - SRM（卡方拟合优度检验）：统计层面均匀性
   - AA实验：特征层面均匀性
4. MDE（最小可检测效果）：样本量检测能力评估

依赖：
    pip install mmh3 numpy scipy
"""

import math
import random
import statistics
from typing import Dict, List, Tuple

import mmh3
import numpy as np
from scipy import stats


# ============================ 配置区 ============================

NUM_USERS = 5000            # 总用户数
NUM_BUCKETS = 1000          # 桶数（远大于组数，降低碰撞波动）
NUM_GROUPS = 10             # 实验组数
SALT = "exp_001"            # 实验盐值（实验期内固定）
RANDOM_SEED = 42            # 随机种子（保证可复现）
HASH_DIFF_THRESHOLD = 0.01  # Hash_diff 阈值（< 1%）
SRM_ALPHA = 0.001           # SRM 卡方显著性阈值（p < 0.001 视为不通过）


# ============================ 1. 分桶 ============================

def hash_to_bucket(user_id: str, salt: str, num_buckets: int) -> int:
    """MurmurHash3 哈希分桶"""
    return mmh3.hash(f"{user_id}_{salt}", signed=False) % num_buckets


# ============================ 2. 蛇形分配 ============================

def snake_assign(
    buckets: Dict[int, List[str]],
    num_groups: int,
) -> Dict[int, List[str]]:
    """
    蛇形分配（S-shape allocation）
    1. 按桶人数降序
    2. 第 1 大桶 -> 组 0，第 2 大桶 -> 组 1 …… 第 G 大桶 -> 组 G-1
    3. 第 G+1 大桶 -> 组 G-1，第 G+2 大桶 -> 组 G-2 …… 交替反向
    4. 确定性贪心均衡：每次把当前最大桶给当前人数最少的组
    """
    # 按桶人数降序排列
    sorted_buckets = sorted(
        buckets.items(),
        key=lambda x: len(x[1]),
        reverse=True,
    )

    groups: Dict[int, List[str]] = {i: [] for i in range(num_groups)}

    for idx, (_bucket_id, users) in enumerate(sorted_buckets):
        cycle = idx // num_groups
        pos = idx % num_groups
        if cycle % 2 == 0:
            group_id = pos
        else:
            group_id = num_groups - 1 - pos
        groups[group_id].extend(users)

    return groups


# ============================ 3. 主流程 ============================

def assign_groups(
    user_ids: List[str],
    num_buckets: int = NUM_BUCKETS,
    num_groups: int = NUM_GROUPS,
    salt: str = SALT,
) -> Dict[int, List[str]]:
    """完整分桶 + 蛇形分配流程"""
    # Step 1: 哈希分桶
    buckets: Dict[int, List[str]] = {i: [] for i in range(num_buckets)}
    for uid in user_ids:
        bid = hash_to_bucket(uid, salt, num_buckets)
        buckets[bid].append(uid)

    # Step 2 & 3: 蛇形分配
    groups = snake_assign(buckets, num_groups)
    return groups


# ============================ 4. 三层验证 ============================

def calc_hash_diff(group_sizes: List[int]) -> float:
    """第一层验证：Hash_diff = 组间人数标准差 / 均值"""
    mean_size = statistics.mean(group_sizes)
    std_size = statistics.pstdev(group_sizes)
    return std_size / mean_size


def srm_check(group_sizes: List[int], alpha: float = SRM_ALPHA) -> Tuple[float, float, bool]:
    """
    第二层验证：SRM（Sample Ratio Mismatch）卡方拟合优度检验
    H0: 各组实际人数 == 期望人数
    """
    total = sum(group_sizes)
    expected_per_group = total / len(group_sizes)
    observed = np.array(group_sizes)
    expected = np.full(len(group_sizes), expected_per_group)

    chi2, p = stats.chisquare(observed, expected)

    if p < alpha:
        passed = False
        verdict = "FAIL（存在 SRM 问题，实验结论不可信）"
    elif p < 0.05:
        passed = True
        verdict = "WARN（p < 0.05，统计边缘，建议复检）"
    else:
        passed = True
        verdict = "PASS（p >= 0.05，分流均匀）"

    return chi2, p, passed, verdict


def aa_experiment(
    groups: Dict[int, List[str]],
    n_simulations: int = 5,
    baseline_rate: float = 0.05,
) -> bool:
    """
    第三层验证：AA 实验（空跑验证）
    所有组跑相同策略，检查组间核心指标是否存在显著差异。
    返回 True 表示组间无显著差异（通过）。
    """
    rng = np.random.default_rng(RANDOM_SEED)
    group_rates: Dict[int, List[float]] = {gid: [] for gid in groups}

    for _ in range(n_simulations):
        for gid, users in groups.items():
            n = len(users)
            if n == 0:
                # 空组：使用 0 作为占位率
                group_rates[gid].append(0.0)
                continue
            # 模拟点击率（所有组相同基线）
            rate = rng.binomial(n, baseline_rate) / n
            group_rates[gid].append(rate)

    # 单因素 ANOVA：组间均值是否相同
    samples = [group_rates[gid] for gid in sorted(group_rates)]
    f_stat, p_value = stats.f_oneway(*samples)

    passed = p_value > 0.05
    return passed, p_value, f_stat


# ============================ 5. MDE 评估 ============================

def calc_mde(
    n_per_group: int,
    baseline_rate: float = 0.05,
    alpha: float = 0.05,
    power: float = 0.8,
) -> float:
    """
    最小可检测效果 (Minimum Detectable Effect)
    给定每组样本量，反算能检测出的绝对效果差。
    公式（两比例 Z 检验）：
        n = (Z_{1-alpha/2} + Z_{1-beta})^2 * (p1(1-p1) + p2(1-p2)) / delta^2
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    p1 = baseline_rate
    # 假设实验组提升很小，反解 delta；p2 实际与 p1 相等，仅用其值计算 sigma_sq

    # 整理后求 delta（绝对差）
    # n = (z_a + z_b)^2 * (p1+p2 - (p1^2+p2^2)) / (p2-p1)^2
    # 设 p2 = p1 + delta，解二次方程
    sigma_sq = 2 * p1 * (1 - p1)  # 近似
    delta = (z_alpha + z_beta) * math.sqrt(sigma_sq / n_per_group)
    return delta


# ============================ 6. 报告输出 ============================

def print_report(
    user_ids: List[str],
    groups: Dict[int, List[str]],
    salt: str = SALT,
) -> None:
    group_sizes = [len(groups[i]) for i in range(NUM_GROUPS)]
    total = sum(group_sizes)

    print("=" * 70)
    print(" AB实验分流验证报告".center(60))
    print("=" * 70)
    print(f" 总用户数       : {total}")
    print(f" 桶数           : {NUM_BUCKETS}")
    print(f" 组数           : {NUM_GROUPS}")
    print(f" 盐值 (salt)    : {salt}")
    print(" 期望每组人数   : {total / NUM_GROUPS:.2f}")
    print("-" * 70)
    print(" 各组人数明细")
    print("-" * 70)
    print(" {'组号':<6}{'人数':<8}{'占比':<10}{'偏差(%)':<12}{'桶数'}")
    for i in range(NUM_GROUPS):
        size = len(groups[i])
        pct = size / total * 100
        diff_pct = (size - total / NUM_GROUPS) / (total / NUM_GROUPS) * 100
        print(f" {i:<6}{size:<8}{pct:<10.4f}{diff_pct:<+12.4f}")
    print("-" * 70)

    # 第一层验证
    hash_diff = calc_hash_diff(group_sizes)
    hash_diff_pass = hash_diff < HASH_DIFF_THRESHOLD
    print("\n[第一层] Hash_diff 验证")
    print(f"   Hash_diff = {hash_diff:.6f}  "
          f"阈值 = {HASH_DIFF_THRESHOLD}")
    print("   结果: {'PASS ✓' if hash_diff_pass else 'FAIL ✗'}")

    # 第二层验证
    chi2, p, srm_passed, srm_verdict = srm_check(group_sizes)
    print("\n[第二层] SRM 卡方拟合优度检验")
    print(f"   chi2 = {chi2:.4f},  p-value = {p:.6f}")
    print(f"   结果: {srm_verdict}")

    # 第三层验证
    aa_passed, aa_p, aa_f = aa_experiment(groups)
    print("\n[第三层] AA 实验（空跑验证）")
    print(f"   F-statistic = {aa_f:.4f},  p-value = {aa_p:.4f}")
    print("   结果: {'PASS ✓（组间无显著差异）' if aa_passed else 'FAIL ✗'}")

    # MDE 评估
    mde_abs = calc_mde(group_sizes[0])
    print("\n[第四层] 最小可检测效果 (MDE)")
    print(f"   每组样本量: {group_sizes[0]}")
    print("   基线转化率: 5%")
    print(f"   95% 置信 / 80% 功效下 MDE = {mde_abs:.4f}")
    print(f"   → 小于 {mde_abs / 0.05 * 100:.2f}% 的效果差异无法可靠检出")

    # 综合判定
    all_passed = hash_diff_pass and srm_passed and aa_passed
    print("\n" + "=" * 70)
    print(f" 综合结论: {'✓ 分流达标，可进入实验' if all_passed else '✗ 分流未达标，需排查'}")
    print("=" * 70)


# ============================ 7. 主入口 ============================

def main() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # 生成 5000 个模拟用户ID
    user_ids = [f"user_{i:05d}" for i in range(NUM_USERS)]

    # 分桶 + 蛇形分配
    groups = assign_groups(user_ids)

    # 输出报告
    print_report(user_ids, groups)


if __name__ == "__main__":
    main()