# 项目结构索引 / Project Structure

> **从这开始**：本文件帮你 30 秒内定位任何脚本的用途和相互关系。

---

## 快速导航（按角色）

### 🟢 我想跑代码 → 看「5 分钟上手」

### 📘 我想查算法 → 看「算法索引表」

### 📖 我想读原理 → 看 README.md（不在此文件）

### 🏗️ 我想做架构决策 → 看「架构图」

### 📋 我想看版本 → 看 CHANGELOG.md

---

## 5 分钟上手（Quickstart）

```bash
# 1. 安装依赖
pip install mmh3 numpy scipy pandas kagglehub

# 2. 跑核心验证（5 分钟，无 Kaggle 数据）
python ab_split_validator.py        # 蛇形批量分配
python realtime_prebucket.py        # P1 用户池预留
python experiment_validation_report.py  # 实验报告生成

# 3. 生成测试数据子集（< 1 秒，9 个文件 ~280KB）
python generate_test_data.py
cat test_data/README.md             # 查看测试数据说明

# 4. 跑数据驱动验证（需要 Kaggle 真实数据，首次自动下载 348MB）
python did_cuped_kaggle.py          # fraud 场景 CUPED
python did_cuped_consumption.py     # consumption 场景 CUPED
python full_scale_validation.py     # 全量 2000 用户 + 客群资质
```

更多命令见 README.md。

---

## 算法索引表（按主题分类）

### A 流量分配层（6 个脚本）

| 脚本 | 作用 | 对应课题 |
|---|---|---|
| `ab_split_validator.py` | 蛇形批量分配 + 三层校验 | 课题 1, 2 |
| `realtime_prebucket.py` | **P1 用户池预留**（0% 流量偏差，工业级主力） | 课题 4 |
| `realtime_calibration.py` | 校准路由 C1（贪心均衡） | 课题 3 |
| `realtime_remedy.py` | 实时分流 5 种方案横向对比 | 课题 3 |
| `realtime_breakthrough.py` | √n 数学下界扫描 + 5000 用户难点 | 课题 2 |
| `realtime_adaptive.py` | 中间校验 + 动态再均衡 | 课题 3 |
| `orthogonal_layers.py` | 多层正交实验 + 流量复用 | 课题 5 |
| `bucket_count_analysis.py` | 桶数对偏差影响的纯分析 | 课题 2 |
| `calibration.py` | 校准算法底层数学 | 课题 3 |
| `streaming_vs_batch.py` | 实时 vs 批量架构对比 | 课题 6 |

### B 数据分析层（5 个脚本）

| 脚本 | 作用 | 对应课题 |
|---|---|---|
| `did_cuped_analysis.py` | DID/CUPED 基础模拟（mock 数据） | 课题 7 |
| `did_cuped_kaggle.py` | **真实 fraud 数据** + 5 种方法对比 | 课题 7 |
| `did_cuped_consumption.py` | **真实消费数据** + CUPED 方缩减 93.7% | 课题 11 |
| `beta_binomial.py` | Beta-Binomial 贝叶斯 AB | 课题 11 |
| `outlier_handling.py` | 5 种异常值处理 + Wilcoxon | 课题 7 |

### C 实验检验层（4 个脚本）

| 脚本 | 作用 | 对应课题 |
|---|---|---|
| `experiment_validation_report.py` | **完整流水线**：SRM + ANOVA + 显著性 + Markdown | 课题 12, 13 |
| `aa_test.py` | A/A 基线噪声验证 | 课题 7 |
| `stratified_bucketing.py` | age × income 分层预分桶 | 课题 7 |
| `seasonal_early_stop.py` | 季节性 CUPED + mSPRT 早期停止 | 课题 7, 10 |

### D 进阶主题（5 个脚本）

| 脚本 | 作用 | 对应课题 |
|---|---|---|
| `mab_vs_ab.py` | 4 种 MAB 算法 vs AB | 课题 14 |
| `mab_vs_ab_when.py` | **6 维决策框架** + 8 真实场景 | 课题 14, 16 |
| `ab_rampup_strategy.py` | 灰度发布 1%→100% + 自动决策 | 课题 15 |
| `finance_repayment_experiment.py` | **金融场景**（7-30 天账期）| 课题 17 |
| `full_scale_validation.py` | 全量 2000 用户 + 客群资质 | 课题 2 |

### E 评估工具（4 个脚本）

| 脚本 | 作用 | 对应课题 |
|---|---|---|
| `monte_carlo_100.py` | 100 次重复抽样稳定性 | 课题 1 |
| `bias_vs_traffic.py` | 流量-偏差联合实测 | 课题 2 |
| `sample_size_table.py` | 评估表生成器 | 课题 7 |
| `generate_test_data.py` | 生成测试数据样本（无需下载 348MB） | 通用 |

### F 测试数据 `test_data/` 目录

| 文件 | 内容 | 大小 |
|---|---|---|
| `README.md` | 测试数据说明 | 4KB |
| `sample_users.csv` | 100 用户级数据 | 2.7KB |
| `sample_transactions.csv` | 2000 笔交易样本 | 184KB |
| `sample_user_history.csv` | 用户消费聚合（CUPED 用） | 5.5KB |
| `sample_split_output.csv` | 5000 用户的 4 种算法分组 | 83KB |
| `reports/experiment_validation_report.md` | 示例完整实验报告 | 1.9KB |
| `reports/cuped_results.csv` | CUPED fraud vs consumption 对比 | 0.2KB |
| `reports/mab_vs_ab_results.csv` | MAB vs AB 算法对比 | 0.1KB |
| `reports/sr_check.csv` | SR 校验样本 | 0.2KB |

**总大小**：约 280 KB（任何 GitHub 网络瞬间 clone）

详情见 [test_data/README.md](test_data/README.md)

---

## 文件大小表

| 类别 | 总行数 | 总大小 |
|---|---|---|
| 流量分配 | ~8500 行 | ~250KB |
| 数据分析 | ~5800 行 | ~150KB |
| 实验检验 | ~5500 行 | ~140KB |
| 进阶主题 | ~6500 行 | ~175KB |
| 评估工具 | ~5500 行 | ~140KB |
| **README.md** | **3000+** | **113KB** |

---

## 与 README.md 的对应关系

```
README.md 三层结构          本项目对应
─────────────────────────────────────────
📘 代码验证层          →   Quickstart + 算法索引表
📋 业务视角详解        →   (README.md 内置)
📖 文字解释层          →   各课题在 README 中
📘📖📋 双层交叉导航    →   本文件作为索引
```

**进入 README 的入口**：
- 入口一：直接看 `📘 代码验证层`（最快）
- 入口二：看 `📋 业务视角详解`（业务方优先）
- 入口三：按 Part I/II/III 看课题（数据科学家）

---

## 文档清单

| 文档 | 作用 |
|---|---|
| [README.md](README.md) | 主文档，13 个课题 + 业务章节 |
| [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) | 本文件 - 文件索引 |
| [ARCHITECTURE.md](ARCHITECTURE.md) | 架构图 + 模块依赖 |
| [CHANGELOG.md](CHANGELOG.md) | 版本演进史 |
| [SUMMARY.md](SUMMARY.md) | 早期摘要（**已归档**，仅供参考）|
| [EVALUATION_TABLE.md](EVALUATION_TABLE.md) | 早期评估表（**已归档**）|

---

## 依赖关系

```
核心算法（基础）：
    mmh3 + numpy + scipy

数据驱动（部分脚本需要）：
    + pandas + kagglehub

完整环境：
    pip install mmh3 numpy scipy pandas kagglehub
```

---

## 推荐阅读顺序

**完全新手**（第一次接触 AB 实验）：
```
1. README.md 顶部"三层结构"导览
2. README.md 的「📋 业务视角详解」
3. README.md 的「📖 Part I」课题 1-8（流量分配）
4. 跑 ab_split_validator.py 验证算法
```

**有 AB 经验**（想深入）：
```
1. README.md 顶部"快速验证层"（跑代码）
2. README.md 的「📖 Part II」数据分析
3. 跑 did_cuped_kaggle.py 看真实数据
4. PROJECT_STRUCTURE.md 找进阶脚本
```

**做决策**（PM/业务负责人）：
```
1. README.md 的「📋 业务视角详解」（红绿灯判断法）
2. mab_vs_ab_when.py 输出 8 真实场景推荐
3. ab_rampup_strategy.py 看开量流程
```

---

## 版本

本项目当前版本：**v0.5.0** （2026 年 7 月）

完整版本演进见 [CHANGELOG.md](CHANGELOG.md)。
