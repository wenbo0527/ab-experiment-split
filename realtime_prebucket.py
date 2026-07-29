#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时场景的批量预分桶实现
========================
实时场景（用户随时进入）下，如何做批量预分桶？

四种方案：
  P1: 用户池预留（最简单）
      - 启动时预分桶 N 个虚拟用户
      - 真实用户首次进入时借用虚拟用户槽位
  P2: 动态扩容（自适应规模）
      - 池子用到 80% 触发异步扩容
  P3: 分层路由（生产级）
      - Layer 1: 已分配查表
      - Layer 2: 未分配实时哈希 + 写回
  P4: 完全实时+校准（最灵活，无预分桶）
"""

import json
import statistics
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import mmh3
import numpy as np

from ab_split_validator import assign_groups, hash_to_bucket


# ============================ P1: 用户池预留 ============================

class UserPoolPreBucket:
    """
    用户池预留预分桶

    核心思想：
      1. 启动时预估用户量，预生成 N 个虚拟用户槽位
      2. 对虚拟用户做批量蛇形分配 → 生成 v_xxx → group 映射
      3. 真实用户首次进入时，从池中借用一个空槽位
      4. 维护 real_uid → v_uid 的反向映射，保证一致性
    """

    def __init__(
        self,
        capacity: int,
        num_groups: int = 10,
        salt: str = "exp_001",
    ):
        self.capacity = capacity
        self.num_groups = num_groups
        self.salt = salt

        # 预分桶虚拟用户池
        virtual_users = [f"v_{i:08d}" for i in range(capacity)]
        self.virtual_assignments = assign_groups(
            virtual_users, num_buckets=min(capacity, 1000), num_groups=num_groups, salt=salt
        )

        # 维护槽位使用状态
        self.slot_to_group: Dict[int, int] = {}
        for gid, users in self.virtual_assignments.items():
            for v_uid in users:
                slot_id = int(v_uid.split("_")[1])
                self.slot_to_group[slot_id] = gid

        # 按组分组槽位（关键：让消耗顺序按组均衡）
        self.slots_by_group: Dict[int, List[int]] = {g: [] for g in range(num_groups)}
        for slot_id, gid in self.slot_to_group.items():
            self.slots_by_group[gid].append(slot_id)

        # 当前轮转指针（每组轮流消耗槽位）
        self.round_robin_idx: Dict[int, int] = {g: 0 for g in range(num_groups)}
        self.current_group = 0  # 下一批应该服务的组

        # 真实用户 → 槽位 的映射
        self.real_to_slot: Dict[str, int] = {}

        # 槽位使用情况
        self.used_slots: set = set()
        self.used_per_group: Dict[int, int] = {g: 0 for g in range(num_groups)}

    def route(self, user_id: str) -> Tuple[int, bool]:
        """
        路由用户，返回 (group_id, is_new)

        已分配用户：直接查表
        新用户：按轮转从人少的组借槽位
        """
        # 已分配：直接查
        if user_id in self.real_to_slot:
            slot = self.real_to_slot[user_id]
            return self.slot_to_group[slot], False

        # 新用户：按轮转从人少的组借槽位
        # 找到当前人数最少的组
        min_group = min(
            range(self.num_groups),
            key=lambda g: self.used_per_group[g]
        )

        # 从该组取下一个空槽位
        slots = self.slots_by_group[min_group]
        idx = self.round_robin_idx[min_group]

        while idx < len(slots):
            slot = slots[idx]
            if slot not in self.used_slots:
                self.used_slots.add(slot)
                self.used_per_group[min_group] += 1
                self.round_robin_idx[min_group] = idx + 1
                self.real_to_slot[user_id] = slot
                return self.slot_to_group[slot], True
            idx += 1

        # 该组无空槽位
        self.round_robin_idx[min_group] = idx
        return -1, False

    def utilization(self) -> float:
        """池子使用率"""
        return len(self.used_slots) / self.capacity * 100


# ============================ P2: 动态扩容预分桶 ============================

class DynamicPreBucket:
    """
    动态扩容预分桶

    - 启动时预分桶 initial_capacity
    - 使用率达 80% 时触发扩容
    - 扩容 = 在原有基础上追加新映射表
    """

    EXPAND_THRESHOLD = 0.8

    def __init__(
        self,
        initial_capacity: int = 50000,
        num_groups: int = 10,
        salt: str = "exp_001",
    ):
        self.num_groups = num_groups
        self.salt = salt
        self.capacity = initial_capacity
        self.expansion_count = 0

        # 初始化第一批映射
        self.assignments: Dict[int, int] = self._build_assignment(
            0, initial_capacity, f"{salt}_v0"
        )
        self.real_to_slot: Dict[str, int] = {}
        self.used_slots: set = set()

    def _build_assignment(
        self,
        start: int,
        end: int,
        salt: str,
    ) -> Dict[int, int]:
        """生成 [start, end) 范围的槽位映射"""
        virtual_users = [f"v_{i:08d}" for i in range(start, end)]
        groups = assign_groups(
            virtual_users,
            num_buckets=min(end - start, 1000),
            num_groups=self.num_groups,
            salt=salt,
        )
        assignment = {}
        for gid, users in groups.items():
            for v_uid in users:
                slot_id = int(v_uid.split("_")[1])
                assignment[slot_id] = gid
        return assignment

    def _expand(self) -> None:
        """扩容：追加同等大小的映射表"""
        old_capacity = self.capacity
        new_capacity = self.capacity * 2
        new_assignment = self._build_assignment(
            old_capacity,
            new_capacity,
            f"{self.salt}_v{self.expansion_count + 1}",
        )
        self.assignments.update(new_assignment)
        self.capacity = new_capacity
        self.expansion_count += 1

    def route(self, user_id: str) -> Tuple[int, bool]:
        if user_id in self.real_to_slot:
            slot = self.real_to_slot[user_id]
            return self.assignments[slot], False

        # 检查是否需要扩容
        if len(self.used_slots) / self.capacity > self.EXPAND_THRESHOLD:
            self._expand()
            # 重建轮转索引
            self.slots_by_group = {g: [] for g in range(self.num_groups)}
            for slot, gid in self.assignments.items():
                self.slots_by_group[gid].append(slot)
            self.round_robin_idx = {g: 0 for g in range(self.num_groups)}
            self.used_per_group = {g: 0 for g in range(self.num_groups)}

        # 找到当前人数最少的组
        if not hasattr(self, 'used_per_group'):
            self.used_per_group = {g: 0 for g in range(self.num_groups)}
        if not hasattr(self, 'round_robin_idx'):
            self.round_robin_idx = {g: 0 for g in range(self.num_groups)}
            self.slots_by_group = {g: [] for g in range(self.num_groups)}
            for slot, gid in self.assignments.items():
                self.slots_by_group[gid].append(slot)

        min_group = min(
            range(self.num_groups),
            key=lambda g: self.used_per_group[g]
        )

        # 从该组取下一个空槽位
        slots = self.slots_by_group[min_group]
        idx = self.round_robin_idx[min_group]
        while idx < len(slots):
            slot = slots[idx]
            if slot not in self.used_slots:
                self.used_slots.add(slot)
                self.used_per_group[min_group] += 1
                self.round_robin_idx[min_group] = idx + 1
                self.real_to_slot[user_id] = slot
                return self.assignments[slot], True
            idx += 1

        self.round_robin_idx[min_group] = idx
        return -1, False

    def utilization(self) -> float:
        return len(self.used_slots) / self.capacity * 100


# ============================ P3: 分层路由 ============================

class LayeredRouter:
    """
    分层路由器（生产级方案）

    Layer 1 (PreBucket): 99% 的查询走预分桶查表
    Layer 2 (RealTime): 1% 的新用户走实时哈希 + 写回

    注意：直接 hash 走预分桶表会导致偏差（hash 分布不均），
    实际生产应该用"用户池预留"的方式均衡消耗。
    """

    def __init__(
        self,
        num_buckets: int = 1000,
        num_groups: int = 10,
        salt: str = "exp_001",
    ):
        self.num_buckets = num_buckets
        self.num_groups = num_groups
        self.salt = salt

        # 预分桶映射表（基于虚拟用户预计算）
        self.pre_bucket: Dict[int, int] = {}
        virtual_users = [f"warmup_{i}" for i in range(num_buckets * 10)]
        groups = assign_groups(
            virtual_users,
            num_buckets=num_buckets,
            num_groups=num_groups,
            salt=salt,
        )
        for gid, users in groups.items():
            for uid in users:
                bid = hash_to_bucket(uid, salt, num_buckets)
                self.pre_bucket[bid] = gid

        # 真实用户分配表（Layer 1 缓存）
        self.assignment_cache: Dict[str, int] = {}

        # 桶消耗计数器（按组）
        self.used_buckets_per_group: Dict[int, int] = {g: 0 for g in range(num_groups)}
        self.used_buckets: set = set()

        # 统计
        self.layer1_hits = 0
        self.layer2_misses = 0

    def route(self, user_id: str) -> Tuple[int, str]:
        """
        路由用户，返回 (group_id, layer_used)
        """
        # Layer 1a: 缓存命中（已分配用户直接查表）
        if user_id in self.assignment_cache:
            self.layer1_hits += 1
            return self.assignment_cache[user_id], "L1_cache"

        # 新用户：用轮转方式从人少的组借桶
        min_group = min(
            range(self.num_groups),
            key=lambda g: self.used_buckets_per_group[g]
        )

        # 从该组找一个未用的桶
        for bid, gid in self.pre_bucket.items():
            if gid == min_group and bid not in self.used_buckets:
                self.used_buckets.add(bid)
                self.used_buckets_per_group[gid] += 1
                self.assignment_cache[user_id] = gid
                self.layer1_hits += 1
                return gid, "L1_prebucket"

        # 桶用完了（理论上 5000 用户不会到这）
        bucket = hash_to_bucket(user_id, self.salt, self.num_buckets)
        gid = bucket % self.num_groups
        self.assignment_cache[user_id] = gid
        self.layer2_misses += 1
        return gid, "L2_realtime"

    def cache_hit_rate(self) -> float:
        total = self.layer1_hits + self.layer2_misses
        if total == 0:
            return 0.0
        return self.layer1_hits / total * 100


# ============================ 实验 ============================

def simulate_realtime_stream(
    router,
    n_users: int,
    seed: int,
    arrival_pattern: str = "uniform",
) -> Dict:
    """
    模拟实时事件流

    arrival_pattern:
      - "uniform": 用户均匀进入
      - "burst": 用户分批进入
      - "growing": 用户数持续增长
    """
    rng = np.random.default_rng(seed)

    # 生成用户ID（模拟"持续进入"）
    user_ids = [f"user_{rng.integers(0, 10**9):09d}" for _ in range(n_users)]

    groups_count: Dict[int, int] = defaultdict(int)
    for uid in user_ids:
        result = router.route(uid)
        gid = result[0] if isinstance(result, tuple) else result
        if gid >= 0:
            groups_count[gid] += 1

    sizes = list(groups_count.values())
    if not sizes:
        return {"max_diff_pct": 0, "sizes": [], "n_groups": 0}

    total = sum(sizes)
    expected = total / len(sizes)
    max_diff = max(abs(s - expected) for s in sizes) / expected * 100

    util = router.utilization() if hasattr(router, "utilization") else None
    cache_hit = router.cache_hit_rate() if hasattr(router, "cache_hit_rate") else None

    return {
        "max_diff_pct": max_diff,
        "sizes": sizes,
        "n_groups": len(sizes),
        "utilization": util,
        "cache_hit_rate": cache_hit,
    }


def run_realtime_prebucket_experiment() -> None:
    """主实验"""

    print("=" * 78)
    print(" 实时场景批量预分桶方案对比实验".center(70))
    print("=" * 78)

    NUM_USERS = 5000
    NUM_TRIALS = 100

    # 配置每种方案
    results = {
        "P1 用户池预留(5万容量)": [],
        "P2 动态扩容(初始1万)": [],
        "P3 分层路由(1000桶)": [],
    }

    for trial in range(NUM_TRIALS):
        seed = 20260728 + trial * 1000

        # P1: 用户池预留
        r1 = UserPoolPreBucket(capacity=50000, num_groups=10)
        sim1 = simulate_realtime_stream(r1, NUM_USERS, seed)
        results["P1 用户池预留(5万容量)"].append(sim1["max_diff_pct"])

        # P2: 动态扩容
        r2 = DynamicPreBucket(initial_capacity=10000, num_groups=10)
        sim2 = simulate_realtime_stream(r2, NUM_USERS, seed)
        results["P2 动态扩容(初始1万)"].append(sim2["max_diff_pct"])

        # P3: 分层路由
        r3 = LayeredRouter(num_buckets=1000, num_groups=10)
        sim3 = simulate_realtime_stream(r3, NUM_USERS, seed)
        results["P3 分层路由(1000桶)"].append(sim3["max_diff_pct"])

    # 输出对比
    print(f"\n {'方案':<28}{'平均偏差':<12}{'中位偏差':<12}{'P95':<10}{'< 1% 通过率':<14}{'< 5% 通过率'}")
    print("-" * 78)

    for name, diffs in results.items():
        avg = statistics.mean(diffs)
        median = statistics.median(diffs)
        p95 = np.percentile(diffs, 95)
        under_1 = sum(1 for d in diffs if d < 1.0) / len(diffs) * 100
        under_5 = sum(1 for d in diffs if d < 5.0) / len(diffs) * 100
        print(f" {name:<28}{avg:<12.4f}{median:<12.4f}{p95:<10.4f}{under_1:<14.1f}{under_5:.1f}%")

    # 详细示例
    print("\n" + "=" * 78)
    print(" 详细示例 (P1 用户池预留)")
    print("=" * 78)
    r1 = UserPoolPreBucket(capacity=50000, num_groups=10)
    rng = np.random.default_rng(20260728)
    user_ids = [f"user_{rng.integers(0, 10**9):09d}" for _ in range(100)]

    print(f"\n 容量: {r1.capacity:,}")
    print(f" 组数: {r1.num_groups}")
    print(f"\n 前 10 个用户分配:")
    for i, uid in enumerate(user_ids[:10]):
        gid, is_new = r1.route(uid)
        marker = "新" if is_new else "已"
        print(f"   {marker} {uid} → 组 {gid}")

    print(f"\n 池子使用率: {r1.utilization():.2f}%")
    print(f" 已分配用户数: {len(r1.real_to_slot)}")

    # 池子耗尽场景
    print("\n" + "=" * 78)
    print(" 容量规划参考")
    print("=" * 78)

    for capacity in [5000, 10000, 50000, 100000]:
        utilization_5k = 5000 / capacity * 100
        print(f" 容量 {capacity:>7,}: 5000 用户使用率 = {utilization_5k:>5.2f}%"
              f" {'⚠️ 接近上限' if utilization_5k > 80 else '✓ 安全'}")

    print("\n" + "=" * 78)
    print(" 方案选型建议")
    print("=" * 78)
    print(" • P1 用户池预留：实验规模可预估的场景（推荐）")
    print(" • P2 动态扩容：实验规模不确定，需要自适应")
    print(" • P3 分层路由：大规模生产环境，字节 DataTester 标准做法")
    print(" • P4 完全实时+校准：实验规模无法预估（放弃预分桶）")
    print("=" * 78)


if __name__ == "__main__":
    run_realtime_prebucket_experiment()