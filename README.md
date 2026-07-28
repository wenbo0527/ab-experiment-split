# AB 实验分流算法实现

> 5000 用户分 10 组，每组人数偏差 < 1% —— 从分桶到校准到验证的完整算法实现

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 项目背景

现实痛点：50:50 配置的 AB 实验跑出 5% 偏差，甚至单日 29:71 的极端偏离。
看似简单的"随机分两组"，在小样本下会被随机波动推到不可控范围。

**核心目标**：5000 用户分 10 组，每组人数偏差稳定 < 1%

## 技术链路

```
MurmurHash3 哈希分桶 → 蛇形分配校准 → 三层验证 → MDE 检测能力评估
       ↓                    ↓              ↓              ↓
   1000 桶打底        全局贪心均衡    Hash_diff       样本量
   稀释碰撞波动        确定性 < 1%    + SRM + AA     够不够
```

## 快速开始

### 环境要求

```bash
pip install mmh3 numpy scipy
```

### 运行主算法

```bash
python ab_split_validator.py
```

输出包括：
- 各组人数明细
- Hash_diff 验证（< 0.01 阈值）
- SRM 卡方拟合优度检验
- AA 实验空跑验证
- MDE 最小可检测效果

### 运行实验套件

```bash
# 100 次重复抽样稳定性测试
python monte_carlo_100.py

# 批量 vs 实时分流对比
python streaming_vs_batch.py

# 实时分流单层优化对比
python realtime_remedy.py

# 实时分流偏差下界扫描
python realtime_breakthrough.py

# 中间校验 + 动态再均衡方案
python realtime_adaptive.py
```

## 实验结果汇总

> 所有数据均为实测结果。完整实验脚本见各 `.py` 文件，可独立运行复现。

| 方案 | 平均偏差 | P95 偏差 | < 1% 通过率 | SRM 通过率 | 一致性 | 数据来源 |
|---|---|---|---|---|---|---|
| **批量蛇形分配** | **0.51%** | 0.80% | **96%** | 100% | ✓ | `ab_split_validator.py` / `monte_carlo_100.py` |
| 实时单次 hash | 8.05% | 12.60% | 0% | 90% | ✓ | `realtime_remedy.py` |
| 实时两次 hash 异或 | 7.82% | 11.61% | 0% | 93% | ✓ | `realtime_remedy.py` |
| 实时 4-salt 众数投票 | 7.98% | 11.62% | 0% | 97% | ✓ | `realtime_remedy.py` |
| 实时桶级微调 | 5.99% | 8.51% | 0% | 100% | ✓ | `realtime_adaptive.py` |
| 实时多映射切换 | 0.51% | 0.80% | 98% | 100% | ✗ | `realtime_adaptive.py` |
| 预分桶查表（推荐） | 0.51% | 0.80% | 96% | 100% | ✓ | 生产方案（继承批量蛇形） |

**实验配置**：每次抽样 5000 随机用户 ID，每方案 50-100 次独立重复。
**指标定义**：
- 平均偏差 = 100 次抽样中最大组人数偏差的均值
- P95 偏差 = 第 95 百分位的最大组偏差
- < 1% 通过率 = 最大组偏差 < 1% 的抽样占比
- SRM 通过率 = 卡方检验 p > 0.05 的抽样占比

**核心发现**：

1. **批量蛇形分配**是 5000 量级唯一稳定压到 < 1% 的算法
2. **实时分流存在 √n 数学下界**（5000 用户下界 8%）
3. **多映射切换**虽数字达标但破坏一致性，生产不可用
4. **预分桶 + 静态查表**是工业级标准方案

## 核心算法

```python
import mmh3

def assign_groups(user_ids, num_buckets=1000, num_groups=10, salt="exp_001"):
    """蛇形分配：5000 用户分 10 组，偏差 < 1%"""
    # Step 1: 哈希分桶
    buckets = {i: [] for i in range(num_buckets)}
    for uid in user_ids:
        bid = mmh3.hash(f"{uid}_{salt}", signed=False) % num_buckets
        buckets[bid].append(uid)

    # Step 2: 按桶人数降序
    sorted_buckets = sorted(buckets.items(), key=lambda x: len(x[1]), reverse=True)

    # Step 3: 蛇形分配
    groups = {i: [] for i in range(num_groups)}
    for idx, (_, users) in enumerate(sorted_buckets):
        cycle = idx // num_groups
        pos = idx % num_groups
        gid = pos if cycle % 2 == 0 else (num_groups - 1 - pos)
        groups[gid].extend(users)

    return groups
```

## 生产级推荐方案

```bash
# 1. 实验启动前批量预分桶
python ab_split_validator.py  # 输出 user_id → group_id 映射

# 2. 部署到 Redis
# SET exp:001:user_00001 0
# SET exp:001:user_00002 3
# ...

# 3. 运行时 O(1) 查表
group = redis.get(f"exp:{exp_id}:{user_id}")
```

## 文件清单

| 文件 | 作用 |
|---|---|
| `ab_split_validator.py` | 主算法：蛇形分配 + 三层验证 + MDE |
| `monte_carlo_100.py` | 100 次重复抽样稳定性测试 |
| `streaming_vs_batch.py` | 批量 vs 实时分流偏差对比 |
| `realtime_remedy.py` | 实时分流单层优化方案对比 |
| `realtime_breakthrough.py` | 实时分流偏差下界扫描 + 理论推导 |
| `realtime_adaptive.py` | 中间校验 + 动态再均衡方案 |
| `SUMMARY.md` | 完整技术总结文档 |

## 技术原理

详细的算法推导、理论下界证明、实验数据分析，见 [SUMMARY.md](SUMMARY.md)。

### 四道核心坎

| 阶段 | 问题 | 解法 |
|---|---|---|
| 1. 分桶 | 哈希碰撞导致桶间人数不均 | MurmurHash3 + 1000 桶 |
| 2. 校准 | 1000 桶仍兜不住 1% 偏差 | 蛇形分配 |
| 3. 验证 | 人数匀 ≠ 分流对 | Hash_diff + SRM + AA |
| 4. 检测能力 | 偏差控制 ≠ 能检出效果 | MDE 评估 |

## 许可证

MIT License

## 致谢

本项目基于 AB 实验分流的工程实践，参考了字节跳动 DataTester、阿里 A/B 平台等工业级实现思路。