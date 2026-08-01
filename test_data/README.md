# Test Data / 测试数据

> **本目录作用**：存放"演示用"的真实数据样本，让任何人都能跑本项目看效果，**无需下载 348MB Kaggle 全量数据**。

---

## 总览

```
test_data/
├── README.md                       # 本文件
├── sample_users.csv                # 100 用户级数据（Kaggle 子集）
├── sample_transactions.csv         # 2000 笔交易样本
├── sample_user_history.csv         # 用户消费聚合（用于 CUPED）
├── sample_split_output.csv         # 5000 用户的 4 种算法分组
└── reports/                        # 各脚本的输出样本
    ├── experiment_validation_report.md  # 一份完整的实验报告
    ├── cuped_results.csv           # CUPED 在 fraud vs consumption 对比
    ├── mab_vs_ab_results.csv       # MAB vs AB 算法对比
    └── sr_check.csv                # SR 校验样本
```

**总大小**：约 277 KB（可在任何 GitHub 网络下瞬间 pull/fetch）

---

## 数据来源

| 文件 | 来源 | 大小 | 行数 |
|---|---|---|---|
| sample_users.csv | Kaggle `transactions-fraud-datasets` 子集 | 2.7 KB | 100 |
| sample_transactions.csv | Kaggle 子集 | 183 KB | 2000 |
| sample_user_history.csv | 从 transactions 聚合 | 5.5 KB | 109 |
| sample_split_output.csv | 模拟 4 种算法 | 83 KB | 5000 |
| reports/* | 演示各脚本输出 | < 3 KB | N/A |

**如果已有 Kaggle 数据缓存** → 自动用 Kaggle 真实子集
**如果没有** → 用 mock 数据（同样格式，可比对）

---

## 如何生成

```bash
# 一键生成（30 秒）
python generate_test_data.py
```

输出会显示在终端，最后一行是总大小（~280KB）。

---

## 如何使用

### 1. 算法脚本测试

```python
import pandas as pd

# 加载样本
df = pd.read_csv("test_data/sample_users.csv")
# 传到 split_validator 等脚本
```

### 2. 实验报告 demo

```bash
# 输出实验报告
cat test_data/reports/experiment_validation_report.md
```

### 3. CUPED 算法对比

```bash
# 看 fraud vs consumption 的真实表现差异
cat test_data/reports/cuped_results.csv
```

---

## 为什么不全量推送？

GitHub 单文件限制 100MB。Kaggle 全量数据 ~348 MB，超过限制。
即便去掉下载限制，**没人想从 GitHub clone 1.5GB 数据**——网速比 Kaggle 慢得多。

我们的策略是：
- **小样本**（< 10 MB）→ 推到 GitHub（这个目录）
- **全量数据** → 用户跑 `kagglehub.dataset_download()` 自动获取（首次）

---

## 完整数据（kagglehub，348MB）

```python
import kagglehub
path = kagglehub.dataset_download("computingvictor/transactions-fraud-datasets")
```

数据集会自动下载到：
```
~/.cache/kagglehub/datasets/computingvictor/transactions-fraud-datasets/
```

`generate_test_data.py` 检测到本地缓存时自动用 Kaggle 数据，否则用 mock 数据。

---

## 注意

- 所有测试数据**仅用于演示**，不构成投资建议
- 原始数据归原作者所有
- 本项目用于研究和教学
