"""
Example 03: 6 维决策框架（MAB vs AB）

演示：
  1. 给定业务场景参数（流量、决策重要性、时长、客群稳定性等）
  2. 用 6 维决策框架判断 AB / MAB 推荐
  3. 与表格化决策矩阵对照

运行：
    python examples/03_mab_decision.py
"""
from __future__ import annotations


# 6 维决策框架（与 file mab_vs_ab_when.py 保持一致）
def recommend_ab_or_mab(
    n_users: int,
    decision_importance: str,  # "high" / "medium" / "low"
    duration_days: int,
    audience_stability: str,   # "stable" / "shifting"
    interpretability_required: bool,
    rollback_possible: bool,
) -> str:
    """根据 6 维返回推荐方法。"""
    # 强制 AB（任何一项满足）
    if n_users < 1000:
        return "AB（强制：流量 < 1k，MAB 无法探索）"
    if decision_importance == "high":
        return "AB（强制：决策重要，需严格显著性）"
    if duration_days < 30:
        return "AB（强制：实验时长 < 30 天，主目标是显著性）"
    if not rollback_possible:
        return "AB（强制：不可回滚，必须严格证据）"
    if interpretability_required:
        return "AB（强制：业务方需要可解释结果）"

    # MAB 主场
    if n_users > 100_000 and duration_days > 90 and audience_stability == "shifting":
        return "MAB（转化最大化 + 长期自适应）"
    if duration_days > 90 and rollback_possible:
        return "MAB（可在多个版本间动态选最优）"

    return "AB 或 MAB 均可（建议 AB 起步，验证后切换到 MAB）"


def main() -> None:
    print("=" * 60)
    print(" Example 03: 6 维决策框架")
    print("=" * 60)
    print("""
输入 6 个维度：
  1. 流量大小
  2. 决策重要性
  3. 实验时长
  4. 客群稳定性
  5. 业务可解释性
  6. 失败成本（是否可回滚）
""")

    # 8 个真实场景
    scenarios = [
        # (场景, 流量, 重要性, 时长, 客群, 可解释, 可回滚)
        ("1. 新功能上线验证",   2_000,    "high",   14,  "stable",   True,  True),
        ("2. 推荐 CTR 长期优化",  50_000,   "medium", 120, "shifting", False, True),
        ("3. 冷启动无历史",     200,      "high",   7,   "stable",   True,  True),
        ("4. 改收费策略",        10_000,   "high",   60,  "stable",   True,  False),
        ("5. 多个 CTA 文案",    5_000,    "low",    21,  "stable",   False, True),
        ("6. 风控规则阈值",    10_000,   "high",   90,  "shifting", True,  False),
        ("7. 邮件发送时间",       500,      "low",    7,   "stable",   False, True),
        ("8. 价格弹性优化",     100_000,  "medium", 180, "shifting", False, True),
    ]

    print(f"{'场景':<25} {'流量':>8} {'决策':>6} {'时长':>5} {'客群':>8} {'推荐方法':<35}")
    print("-" * 90)
    for name, n, dec, dur, aud, interp, rb in scenarios:
        rec = recommend_ab_or_mab(n, dec, dur, aud, interp, rb)
        print(f"{name:<25} {n:>8} {dec:>6} {dur:>5} {aud:>8} {rec}")

    print(f"\n{'='*60}")
    print(" 完整 8 个详见：python -m abexp.advanced.mab_vs_ab_when")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
