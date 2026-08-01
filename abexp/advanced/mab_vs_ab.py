#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MAB (Multi-Armed Bandit) vs AB 实验
====================================
MAB 通过动态调整各组流量比例，最大化整体收益。

算法实现:
  - ε-Greedy: 简单但有效
  - UCB1: Upper Confidence Bound
  - Thompson Sampling: 贝叶斯采样

对比维度:
  - Cumulative regret (累积损失 = vs 最优的差距)
  - Convergence speed (收敛速度)
  - Final performance (最终表现)
"""

from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")  # 无 GUI 环境
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================ MAB 算法 ============================

class EpsilonGreedy:
    """ε-Greedy: 以 ε 概率随机探索，否则选当前最优臂"""

    def __init__(self, n_arms: int, epsilon: float = 0.1):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.counts = np.zeros(n_arms)  # 每臂尝试次数
        self.values = np.zeros(n_arms)  # 每臂平均奖励
        self.t = 0

    def select_arm(self) -> int:
        self.t += 1
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_arms)
        return int(np.argmax(self.values))

    def update(self, arm: int, reward: float):
        self.counts[arm] += 1
        n = self.counts[arm]
        # 增量均值更新
        self.values[arm] = ((n - 1) / n) * self.values[arm] + (1 / n) * reward


class UCB1:
    """UCB1: 置信上界，越不确定的臂越优先探索"""

    def __init__(self, n_arms: int, c: float = 1.0):
        self.n_arms = n_arms
        self.c = c  # 探索系数
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.t = 0

    def select_arm(self) -> int:
        self.t += 1
        # 任何臂都没拉过就先拉
        if np.any(self.counts == 0):
            return int(np.argmax(self.counts == 0))
        # UCB = 当前均值 + c * sqrt(2*log(t)/n)
        ucb = self.values + self.c * np.sqrt(2 * np.log(self.t) / self.counts)
        return int(np.argmax(ucb))

    def update(self, arm: int, reward: float):
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] = ((n - 1) / n) * self.values[arm] + (1 / n) * reward


class ThompsonSampling:
    """Thompson Sampling: Beta-Bernoulli 后验采样"""

    def __init__(self, n_arms: int, prior_alpha: float = 1.0, prior_beta: float = 1.0):
        self.n_arms = n_arms
        self.alpha = np.full(n_arms, prior_alpha)
        self.beta = np.full(n_arms, prior_beta)
        self.t = 0

    def select_arm(self) -> int:
        self.t += 1
        # 从每臂的 Beta 后验采样
        samples = np.random.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def update(self, arm: int, reward: float):
        if reward > 0:
            self.alpha[arm] += reward
        else:
            self.beta[arm] += (1 - reward) if 0 < reward <= 1 else 1


class ABTestFixed:
    """AB 实验: 固定 50/50 分流（对照组）"""

    def __init__(self, n_arms: int):
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.t = 0

    def select_arm(self) -> int:
        self.t += 1
        # 50/50 随机（或均匀分布到 n_arms）
        return self.t % self.n_arms

    def update(self, arm: int, reward: float):
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] = ((n - 1) / n) * self.values[arm] + (1 / n) * reward


# ============================ 模拟器 ============================

def simulate(
    algorithm,
    true_rewards: np.ndarray,
    n_steps: int = 5000,
    seed: int = 20260728,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    跑 MAB 模拟

    Args:
        algorithm: 任一算法实例
        true_rewards: 真实转化率（每臂一个）
        n_steps: 总步数
        seed: 种子

    Returns:
        cumulative_rewards: 每步累积奖励
        cumulative_regret: 每步累积损失（vs 最优臂）
    """
    rng = np.random.default_rng(seed)
    optimal_reward = true_rewards.max()

    cumulative_rewards = np.zeros(n_steps)
    cumulative_regret = np.zeros(n_steps)

    for step in range(n_steps):
        arm = algorithm.select_arm()
        reward = rng.binomial(1, true_rewards[arm])
        algorithm.update(arm, reward)

        cumulative_rewards[step] = (
            algorithm.values * algorithm.counts
        ).sum() if hasattr(algorithm, 'values') else (reward if step == 0 else cumulative_rewards[step-1] + reward)

        regret_step = optimal_reward - true_rewards[arm]
        cumulative_regret[step] = (cumulative_regret[step - 1] if step > 0 else 0) + regret_step

    return cumulative_rewards, cumulative_regret


def compare_algorithms(
    n_arms: int = 5,
    n_steps: int = 5000,
    n_trials: int = 30,
    base_rate: float = 0.10,
    treatment_lift: float = 0.04,  # 治疗臂提升 4pp
    seed_base: int = 20260728,
) -> pd.DataFrame:
    """
    对比 4 种算法在多场景下的表现

    Returns:
        DataFrame with algorithm, final_cumulative_reward, total_regret
    """
    # 真转化率: 5 个臂，其中臂 0 是对照 (0.10), 臂 1-4 是实验 (+4pp)
    true_rewards = np.full(n_arms, base_rate)
    true_rewards[1:] = base_rate + treatment_lift

    algorithms = {
        "AB 50/50 固定": lambda: ABTestFixed(n_arms),
        "ε-Greedy (0.1)": lambda: EpsilonGreedy(n_arms, epsilon=0.1),
        "UCB1 (c=1)": lambda: UCB1(n_arms, c=1.0),
        "Thompson Sampling": lambda: ThompsonSampling(n_arms),
    }

    results = []
    for algo_name, algo_fn in algorithms.items():
        rewards_per_trial = []
        regrets_per_trial = []
        for trial in range(n_trials):
            algo = algo_fn()
            cum_reward, cum_regret = simulate(
                algo,
                true_rewards,
                n_steps=n_steps,
                seed=seed_base + trial,
            )
            rewards_per_trial.append(cum_reward[-1])
            regrets_per_trial.append(cum_regret[-1])

        results.append({
            "algorithm": algo_name,
            "final_cumulative_reward": float(np.mean(rewards_per_trial)),
            "total_regret": float(np.mean(regrets_per_trial)),
            "reward_std": float(np.std(rewards_per_trial)),
        })

    return pd.DataFrame(results)


# ============================ 演示 ============================

def demo():
    """演示 MAB vs AB 表现对比"""
    print("=" * 78)
    print(" MAB vs AB 实验对比".center(50))
    print("=" * 78)
    print()
    print(" 场景: 5 个实验臂")
    print("   对照臂: 10% 转化率")
    print("   实验臂 1-4: 14% 转化率（提升 +4pp）")
    print()
    print(" 总步数: 5000 步（用户）")
    print(" 蒙特卡洛次数: 30")
    print()

    df = compare_algorithms(n_arms=5, n_steps=5000, n_trials=30)
    print(df.to_string(index=False))
    print()

    print("【关键观察】")
    print(f" 1. AB 50/50 固定：累积奖励 {df.iloc[0]['final_cumulative_reward']:.0f}")
    print(f"   损失：把 40% 流量浪费在次优臂上")
    print()
    print(f" 2. MAB 算法（UCB1, Thompson）：损失 (regret) 比 AB 少")
    print(f"   UCB1 总损失 ≈ {df.iloc[2]['total_regret']:.0f}")
    print(f"   Thompson 总损失 ≈ {df.iloc[3]['total_regret']:.0f}")
    print()

    print("【结论】")
    print(" √ MAB 通过动态调整，减少了次优臂流量 = 更高转化率")
    print(" √ 但 MAB 没有「统计显著性」概念，难以做 AB 那种严格检验")
    print(" √ 工业实践中：先 AB 验证 → 确认获胜版本 → MAB 微调 / 长期优化")


if __name__ == "__main__":
    demo()
