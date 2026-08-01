"""
Example 02: 完整实验检验流水线

演示：
  1. 生成 mock 用户数据
  2. 调用 experiment_validation_report 的完整流水线
  3. 获得 Markdown 格式的实验报告

运行：
    python examples/02_validation_report.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from abexp.validation.experiment_validation_report import validate_full_pipeline


def main() -> None:
    print("=" * 60)
    print(" Example 02: 完整实验检验流水线")
    print("=" * 60)

    # 1. 生成 mock 用户数据（3000 用户，2 组）
    rng = np.random.default_rng(42)
    n = 3000
    df = pd.DataFrame({
        "user_id": [f"u_{i:05d}" for i in range(n)],
        "assigned": rng.integers(0, 2, n),  # 0=对照组, 1=实验组
        "age": rng.normal(35, 10, n).clip(18, 70),
        "yearly_income": rng.normal(50000, 15000, n).clip(10000, 200000),
        # 实验组转化率 12%，对照组 9.5%（相对提升 ~26%）
        "converted": [
            int(rng.random() < (0.12 if assigned == 1 else 0.095))
            for assigned in rng.integers(0, 2, n)
        ],
    })
    print(f"\n1) 生成 {n} 用户 mock 数据")
    print(f"   实验组转化率: {df[df.assigned==1].converted.mean()*100:.2f}%")
    print(f"   对照组转化率: {df[df.assigned==0].converted.mean()*100:.2f}%")

    # 2. 调用完整流水线
    print(f"\n2) 调用 validate_full_pipeline ...")
    report = validate_full_pipeline(
        df=df,
        group_col="assigned",
        y_col="converted",
        feature_cols=["age", "yearly_income"],
        experiment_id="EXP_DEMO_001",
        experiment_name="Example-02 演示",
        y_type="binary",
    )

    # 3. 输出报告（前 30 行 + 元数据）
    print(f"\n3) 实验报告（Markdown 格式）：\n")
    for line in report.split("\n")[:30]:
        print(line)
    print(f"\n... (报告共 {len(report.splitlines())} 行)")

    print(f"\n{'='*60}")
    print(" 决策框架参考：python examples/03_mab_decision.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
