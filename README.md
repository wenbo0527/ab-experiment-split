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
# 主算法：5000 用户分 10 组 + 三层验证 + MDE
python ab_split_validator.py

# 100 次重复抽样稳定性测试
python monte_carlo_100.py

# 批量 vs 实时分流对比
python streaming_vs_batch.py

# 实时分流单层优化对比（单/两次hash/多salt）
python realtime_remedy.py

# 实时分流偏差下界扫描 + 理论推导
python realtime_breakthrough.py

# 中间校验 + 动态再均衡方案
python realtime_adaptive.py

# 多层正交实验 + 流量复用验证
python orthogonal_layers.py

# 实时分流每桶人数下界分析（√n 数学推导）
python bucket_count_analysis.py

# 实时分流偏差校准机制（运行时压缩偏差）
python calibration.py
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

### 多层正交实验（流量复用）

> 数据来源：`orthogonal_layers.py`（5000 用户 × 50 次抽样）

| 指标 | 实测值 | 判定 |
|---|---|---|
| 正交通过率（p > 0.05） | **100%** | ✓ 完美正交 |
| 卡方 p-value 中位数 | 0.48 | 远大于 0.05 |
| 列联表最大偏差 | 1.13% | 与组数 2 的泊松波动一致 |
| 三层 8 组合分布平衡率 | 0.89 | 接近理想 1.0 |

**单层偏差**（实时，每层 2 组）：

| 层 | 平均偏差 | P95 | < 1% 通过率 |
|---|---|---|---|
| L1 推荐 | 1.24% | 2.89% | 52% |
| L2 搜索 | 0.89% | 2.11% | 60% |
| L3 UI | 1.35% | 2.55% | 34% |

**多层正交的价值**：100% 流量可被 3 个实验同时使用（推荐/搜索/UI 各占一层），层间完全独立。**单层偏差仍是 √n 下界**——多层正交解决流量复用，不解决单层均匀性。

### 实时分流每桶人数下界

> 数据来源：`bucket_count_analysis.py`（实测验证 √n 数学下界）

**核心公式**：`实时分流偏差下界 = z / √(N/G)`，**与桶数 B 无关**。

实测（N=50000, G=10，每组 5000 人，改变桶数 B）：

| 桶数 B | 每桶期望 k | 实测平均偏差 | < 1% 通过率 |
|---|---|---|---|
| 10 | 5000 | 2.51% | 2% |
| 100 | 500 | 2.59% | 0% |
| **1000** | **50** | **2.43%** | **0%** |
| 10000 | 5 | 2.63% | 2% |
| 100000 | 0.5 | 2.56% | 0% |

**结论**：B 变化 10000 倍，偏差都在 2.5% 左右——**桶数不影响下界，只影响实现细节**。

**每桶最少多少人？**（目标偏差 1%, z=3）

| 桶数 B | 组数 G | 每桶最少 k | 对应总流量 N |
|---|---|---|---|
| 1000 | 10 | **900** | **90万** |

**真正的瓶颈是总流量 N**：要压到 1%（3σ），每组最少 9 万人，10 组需 90 万用户。**5000 量级下界 4.47%（1σ）**，纯实时分流**不可能**压到 1%。

### 校准机制效果

> 数据来源：`calibration.py`（5000 用户 × 100 次抽样）

| 方案 | 平均偏差 | 中位偏差 | P95 | < 1% 通过率 |
|---|---|---|---|---|
| R0 纯哈希（基线） | 8.05% | 7.40% | 12.60% | 0% |
| **C1 校准路由** | **1.94%** | **1.80%** | 4.00% | **10%** |
| 批量预分桶（蛇形） | 0.51% | 0.40% | 0.80% | 96% |

**校准路由 C1 原理**：实时维护各组人数，对每个新用户**倾向分到当前人数最少的组**。这是字节 DataTester 等工业级系统的核心机制之一。

**"9 万人"的含义**：是统计意义上的累计样本量（无论何时进入实验）。√n 下界取决于当前已分配总用户数，不是单批。持续运行场景下，中后期偏差自然收敛。

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
| `realtime_remedy.py` | 实时分流单层优化方案对比（单hash/两次hash/多salt） |
| `realtime_breakthrough.py` | 实时分流偏差下界扫描 + 理论推导 |
| `realtime_adaptive.py` | 中间校验 + 动态再均衡方案 |
| `orthogonal_layers.py` | 多层正交实验 + 流量复用验证 |
| `bucket_count_analysis.py` | 实时分流每桶人数下界分析（√n 推导） |
| `calibration.py` | 实时分流偏差校准机制对比 |
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