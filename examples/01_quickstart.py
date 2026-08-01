"""
Example 01: 5 分钟上手 AB 实验分流

演示：
  1. 用蛇形分配把 5000 用户分成 10 组
  2. 检查 SRM（流量比例失衡量）
  3. 计算 MDE（最小可检测效果）

运行：
    python examples/01_quickstart.py
"""
from __future__ import annotations

from abexp.routing.ab_split_validator import (
    assign_groups,
    srm_check,
    calc_mde,
)

def main() -> None:
    print("=" * 60)
    print(" Example 01: 5 分钟上手 AB 实验分流")
    print("=" * 60)

    # 1. 生成 5000 用户
    user_ids = [f"u_{i:05d}" for i in range(5000)]

    # 2. 蛇形分配到 10 组
    groups = assign_groups(user_ids, num_buckets=1000, num_groups=10, salt="exp_001")
    sizes = [len(v) for v in groups.values()]
    print(f"\n1) 蛇形分配：5000 用户 → 10 组")
    print(f"   各组人数: {sizes}")

    # 3. SRM 检验
    chi2, p, passed, verdict = srm_check(sizes, alpha=0.05)
    print(f"\n2) SRM 检验: chi²={chi2:.4f}, p={p:.4f}, {'✓ 通过' if passed else '✗ 失败'} ({verdict})")

    # 4. MDE（最小可检测效果）
    n_per_group = sizes[0]
    base_rate = 0.05  # 假设 5% 基线转化率
    mde = calc_mde(n_per_group, baseline_rate=base_rate, alpha=0.05, power=0.8)
    print(f"\n3) MDE（基线 5% 转化率, 95% 置信, 80% 功效）")
    print(f"   每组 {n_per_group} 人，能检出的最小相对提升: {mde*100:.2f}%")

    print(f"\n{'='*60}")
    print(" 完整流水线请参考：python examples/02_validation_report.py")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
