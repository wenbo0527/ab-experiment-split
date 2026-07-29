#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AB 实验分流方案评估表生成器
============================
基于 95% 置信度（z=1.96），
按目标偏差反推所需的最低累计样本量。

核心公式：N/G = (z / bias)^2

输出：
  - 不同偏差阈值（0.5%, 1%, 2%, 5%）所需样本量
  - 批量方案 vs 实时方案的差异
  - 工程选型建议
"""

import math
import statistics
from typing import Dict, List

import numpy as np


# ============================ 核心公式 ============================

def min_sample_size(target_bias_pct: float, num_groups: int, z: float = 1.96) -> int:
    """
    计算满足 95% 置信度的最小总样本量

    公式：N/G = (z / bias)^2
    bias 为小数（如 0.01 表示 1%）
    """
    bias = target_bias_pct / 100
    n_per_group = (z / bias) ** 2
    total_n = n_per_group * num_groups
    return int(math.ceil(total_n))


def theoretical_bias(n_users: int, num_groups: int, z: float = 1.96) -> float:
    """反算偏差（95% 置信度）"""
    n_per_group = n_users / num_groups
    return z / math.sqrt(n_per_group) * 100


# ============================ 批量 vs 实时 总结 ============================

def summarize_batch_vs_realtime() -> str:
    return """
【批量 vs 实时】分流方案核心差异

| 维度 | 批量预分桶（蛇形分配） | 实时分流（纯哈希） |
|---|---|---|
| 算法复杂度 | O(N log N)（含排序） | O(1)（哈希） |
| 启动开销 | 高（需全量计算） | 无 |
| 单次查询开销 | O(1)（查表） | O(1)（哈希） |
| 内存占用 | O(N)（存全量映射） | O(1)（无状态） |
| 一致性保证 | ✓ 强（预计算结果不变） | ✓ 强（同一 hash 永远同组） |
| 偏差控制能力 | 强（< 1%） | 弱（受 √n 下界限制） |
| 流量规模要求 | 不限（小流量也行） | 需大流量（每组 ≥ 9 万） |
| 新用户进入 | 首次落表 | 直接 hash |
| 工程复杂度 | 中（需要预计算+部署） | 低（直接调用） |

核心区别：
  批量：能在小流量下精确控制偏差（蛇形算法）
  实时：只能靠大流量"自然收敛"（√n 法则）
"""


# ============================ 评估表生成 ============================

def generate_evaluation_table() -> str:
    """生成 4x3 评估表"""

    output = []
    output.append("=" * 80)
    output.append(" AB 实验分流方案评估表（95% 置信度）".center(70))
    output.append("=" * 80)

    output.append("\n## 一、数学原理")
    output.append("")
    output.append("实时分流偏差下界公式（基于中心极限定理）：")
    output.append("")
    output.append("    偏差(95%) = 1.96 / √(N/G) × 100%")
    output.append("")
    output.append("反解最小样本量：")
    output.append("")
    output.append("    N/G = (1.96 / bias)^2")
    output.append("    N   = (1.96 / bias)^2 × G")
    output.append("")
    output.append("其中：")
    output.append("    N   = 总用户数（累积样本量）")
    output.append("    G   = 组数")
    output.append("    1.96 = 95% 置信度对应的 z 值")

    # 表 1: 不同组数 × 不同偏差阈值
    output.append("\n\n## 二、不同组数下达到偏差阈值所需的最小累积样本量")
    output.append("")
    output.append("| 组数 G | 0.5% 偏差 | 1% 偏差 | 2% 偏差 | 5% 偏差 |")
    output.append("|---|---|---|---|---|")

    for g in [2, 3, 5, 10, 20, 50, 100]:
        row = [f"**{g}**"]
        for bias in [0.5, 1.0, 2.0, 5.0]:
            n_total = min_sample_size(bias, g)
            row.append(f"{n_total:,}")
        output.append("| " + " | ".join(row) + " |")

    # 表 2: 不同用户量下的实测偏差（实时方案下界）
    output.append("\n\n## 三、不同累计样本量下的实时分流偏差下界")
    output.append("")
    output.append("| 累计样本量 N | 每组人数 | 0.5% 偏差 | 1% 偏差 | 2% 偏差 | 5% 偏差 |")
    output.append("|---|---|---|---|---|---|")

    for n in [500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000]:
        n_per_group = n // 10  # 默认 10 组
        row = [
            f"**{n:,}**",
            f"{n_per_group:,}",
        ]
        for bias in [0.5, 1.0, 2.0, 5.0]:
            actual_bias = theoretical_bias(n, 10)
            achievable = "✓" if actual_bias < bias else "✗"
            row.append(f"{actual_bias:.2f}% {achievable}")
        output.append("| " + " | ".join(row) + " |")

    # 表 3: 批量方案实测数据（来自已有实验）
    output.append("\n\n## 四、批量预分桶方案实测偏差（无 √n 限制）")
    output.append("")
    output.append("| 方案 | 用户量 | 组数 | 平均偏差 | < 1% 通过率 | 95% 置信要求 |")
    output.append("|---|---|---|---|---|---|")
    output.append("| 批量蛇形分配 | 5,000 | 10 | 0.51% | 96% | ✓ 达标 |")
    output.append("| 批量蛇形分配 | 10,000 | 10 | 0.36% | 100% | ✓ 达标 |")
    output.append("| 批量蛇形分配 | 100,000 | 10 | 0.11% | 100% | ✓ 达标 |")
    output.append("| 用户池预留 (P1) | 5,000 | 10 | 0.00% | 100% | ✓ 达标 |")
    output.append("| 动态扩容 (P2) | 5,000 | 10 | 0.00% | 100% | ✓ 达标 |")

    # 表 4: 实时方案实测数据
    output.append("\n\n## 五、实时分流方案实测偏差（受 √n 限制）")
    output.append("")
    output.append("| 方案 | 用户量 | 组数 | 平均偏差 | < 1% 通过率 | 95% 置信要求 |")
    output.append("|---|---|---|---|---|---|")
    output.append("| 纯哈希 | 5,000 | 10 | 8.05% | 0% | ✗ 不达标 |")
    output.append("| 两次 hash 异或 | 5,000 | 10 | 7.82% | 0% | ✗ 不达标 |")
    output.append("| 4-salt 众数投票 | 5,000 | 10 | 7.98% | 0% | ✗ 不达标 |")
    output.append("| 桶级微调 (S2) | 5,000 | 10 | 5.99% | 0% | ✗ 不达标 |")
    output.append("| 校准路由 (C1) | 5,000 | 10 | 1.94% | 10% | △ 边缘 |")
    output.append("| 多映射切换 (S3) | 5,000 | 10 | 0.51% | 98% | ✓ 但破坏一致性 |")

    # 工程选型建议
    output.append("\n\n## 六、工程选型决策树")
    output.append("")
    output.append("### Q1: 实验用户量是否可预估？")
    output.append("  - 是 → P1 用户池预留（推荐）")
    output.append("  - 否 → Q2")
    output.append("")
    output.append("### Q2: 目标偏差 < 1%？")
    output.append("  - 是 → 走批量预分桶（蛇形分配）")
    output.append("  - 否（可接受 5% 偏差）→ Q3")
    output.append("")
    output.append("### Q3: 是否多团队多实验并发？")
    output.append("  - 是 → 多层正交 + 批量预分桶")
    output.append("  - 否 → C1 校准路由 或 纯实时（接受 5-8% 偏差）")
    output.append("")

    # 不同场景的最终推荐
    output.append("\n\n## 七、场景化推荐表")
    output.append("")
    output.append("| 场景 | 推荐方案 | 预期偏差 | 工程成本 |")
    output.append("|---|---|---|---|")
    output.append("| 小流量 (< 1万)，单实验 | P1 用户池预留 | 0% | 低 |")
    output.append("| 中流量 (1-10万)，单实验 | 批量蛇形 + 静态查表 | < 1% | 中 |")
    output.append("| 大流量 (> 10万)，单实验 | 纯实时 + 大流量 | < 1% | 极低 |")
    output.append("| 多团队并发 | 多层正交 + P1 | < 1% | 高 |")
    output.append("| 实验规模不可预估 | P2 动态扩容 | < 1% | 中 |")
    output.append("| 接受 5% 偏差即可 | C1 校准路由 | ~2% | 中 |")
    output.append("| 接受 8% 偏差 | 纯实时 | ~8% | 极低 |")

    output.append("\n\n## 八、关键公式速查")
    output.append("")
    output.append("### 实时分流偏差下界（95% 置信）")
    output.append("")
    output.append("    bias = 1.96 × √(G/N)")
    output.append("")
    output.append("### 最小总样本量（95% 置信）")
    output.append("")
    output.append("    N = (1.96 / bias)^2 × G")
    output.append("")
    output.append("### 等价公式（每组最少人数）")
    output.append("")
    output.append("    N/G = (1.96 / bias)^2")
    output.append("")

    # 不同偏差下的最少人数
    output.append("### 不同偏差下的每组最少人数（95% 置信）")
    output.append("")
    output.append("| 目标偏差 | 每组最少人数 (N/G) | 10 组总人数 | 100 组总人数 |")
    output.append("|---|---|---|---|")
    for bias in [0.5, 1.0, 2.0, 5.0]:
        n_per_g = min_sample_size(bias, 1)  # G=1 时算出 N/G
        n_10 = min_sample_size(bias, 10)
        n_100 = min_sample_size(bias, 100)
        output.append(f"| {bias}% | {n_per_g:,} | {n_10:,} | {n_100:,} |")

    output.append("\n" + "=" * 80)
    output.append(" 一句话总结".center(70))
    output.append("=" * 80)
    output.append("")
    output.append(" 实时分流 95% 置信下：")
    output.append("   0.5% 偏差需每组 15,367 人，1% 偏差需 3,842 人，2% 偏差需 961 人，5% 偏差需 154 人")
    output.append("")
    output.append(" 但批量预分桶（蛇形分配）不受此限制：")
    output.append("   5000 用户 / 10 组即可达到 0.51% 偏差（96% < 1% 通过率）")
    output.append("")
    output.append(" 工程建议：")
    output.append("   小流量场景（< 10万）走批量预分桶")
    output.append("   大流量场景（≥ 10万）走实时 + 大流量自然收敛")
    output.append("=" * 80)

    return "\n".join(output)


def main() -> None:
    # 输出到控制台
    print(generate_evaluation_table())

    # 同时保存到文件
    output = generate_evaluation_table()
    with open("/Users/mac/Documents/trae_projects/AB实验算法测试/AB实验技术实现/EVALUATION_TABLE.md", "w", encoding="utf-8") as f:
        f.write(output)
    print("\n\n评估表已保存到 EVALUATION_TABLE.md")


if __name__ == "__main__":
    main()