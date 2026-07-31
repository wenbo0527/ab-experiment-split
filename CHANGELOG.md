# 版本演进史 / Changelog

> 本项目版本演进透明记录。每次新功能/重大变化都按时间倒序记录。

---

## v0.5.0 (2026-07-31) · 当前版本

**核心主题**：从 60% 到 80%+ 覆盖度、MAB vs AB 决策框架、金融场景专题

### 新增能力（P0/P1 全部完成）

#### 课题专题
- 📋 **业务视角详解**（独立章节）
  - 大白话讲 AB 实验
  - 7 个业务方必问问题
  - 红绿灯判断法
  - 4 个场景化示例
- 课题 14: MAB vs AB 实验（4 种算法）
- 课题 15: AB 实验开量策略（灰度发布 1%→100%）
- 课题 16: MAB vs AB 决策框架（6 维）
- 课题 17: 金融场景实验设计（30 天账期）

#### 新增脚本
- `mab_vs_ab.py`：4 种 MAB 算法（AB 50/50 / ε-Greedy / UCB1 / Thompson Sampling）
- `mab_vs_ab_when.py`：6 维决策框架 + 8 个真实业务场景
- `ab_rampup_strategy.py`：5 阶段灰度 + 自动决策引擎
- `finance_repayment_experiment.py`：5 大金融场景能力（漏斗/KM/分时段/风险/DID）
- `outlier_handling.py`：5 种异常处理 + Wilcoxon
- `aa_test.py`：500-1000 次 A/A 基线验证
- `beta_binomial.py`：贝叶斯 AB（业务方友好输出）
- `stratified_bucketing.py`：年龄 × 收入三维分层
- `seasonal_early_stop.py`：季节性 CUPED + mSPRT 早期停止

#### 项目文档
- `PROJECT_STRUCTURE.md`：文件索引（NEW）
- `ARCHITECTURE.md`：架构图 + 模块边界（NEW）
- `CHANGELOG.md`：本文件（NEW）
- 旧文档归档提示

### 改进
- README 从 700 行 扩到 3000+ 行
- 章节结构：从 2 层（代码/文字）→ 3 层（代码/业务/文字）
- 颠覆性发现：mock vs Kaggle 真实数据，揭示 CUPED 在 fraud/consumption 上的真实表现差异

### 关键数据
- P1 用户池：0% 流量偏差（工业级）
- 客群分层：偏差 12.5% → 6.9%
- CUPED fraud 场景：方缩减 0.1%（真实）
- CUPED consumption 场景：方缩减 93.7%（真实）
- 整体覆盖度：60% → 80%

---

## v0.4.0 (2026-07-29)

**核心主题**：从分流算法到数据分析 + 实验检验

### 新增
- Part I 课题 1-8（流量分配层）
- Part II 课题 9-11（数据分析层）
- Part III 课题 12-13（实验检验层）
- 三大测试案例（Kaggle 真实数据）

### 脚本
- 13 个核心脚本（流量 + 分析 + 检验）
- 真实数据 vs mock 数据双重验证

### 文档
- README 主体（约 1500 行）
- 三层结构（代码/业务/文字）

---

## v0.3.0 (2026-07-25)

**核心主题**：流量分配的"4 种算法"对比 + √n 数学下界

### 新增
- P1 用户池预留（realtime_prebucket.py）
- 校准路由 C1（realtime_calibration.py）
- 多层正交实验（orthogonal_layers.py）
- √n 数学下界分析（realtime_breakthrough.py）

### 关键数据
- 纯 hash 5000 用户：8.05% 流量偏差
- P1 用户池：实测 0% 流量偏差
- 校准 C1：实测 0.20% 流量偏差
- 多层正交：100% 实验正交率

---

## v0.2.0 (2026-07-20)

**核心主题**：从单点算法到横向对比

### 新增
- 蛇形批量分配（ab_split_validator.py）
- 实时 5 种方案对比（realtime_remedy.py）
- 中间校验 + 动态再均衡（realtime_adaptive.py）

---

## v0.1.0 (2026-07-15)

**核心主题**：从 0 到 1 的 AB 实验方案

### 初始能力
- 纯 hash 分流实现
- 蛇形分配原理 + 实现
- 100 次蒙特卡洛稳定性测试（monte_carlo_100.py）
- 早期文档：SUMMARY.md / EVALUATION_TABLE.md

---

## 未来版本

### v0.6.0（计划中）
- 实验 ROI 评估（业务方关心）
- 长期效果跟踪 7/30/90 天
- 实验档案库（历史经验复用）

### v1.0（生产级）
- Redis 流式路由
- 实时监控仪表盘
- Slack/邮件自动推送
- 多实验冲突管理

### v1.5（SaaS 化）
- 后端 API
- 数据可视化 web UI
- 多租户权限

---

## 重大改进日期表

| 日期 | 版本 | 主要变化 |
|---|---|---|
| 2026-07-31 | v0.5.0 | 业务章节 + MAB + 开量 + 金融场景 |
| 2026-07-29 | v0.4.0 | 数据分析 + 实验检验 + 真实数据 |
| 2026-07-25 | v0.3.0 | P1/C1/正交 4 种算法对比 |
| 2026-07-20 | v0.2.0 | 蛇形 + 实时方案横向对比 |
| 2026-07-15 | v0.1.0 | 初始方案 |

---

## 贡献者

- 主要作者：trAEas AI
- 数据来源：Kaggle `computingvictor/transactions-fraud-datasets`

---

## 协议

MIT License - 详见 [LICENSE](LICENSE)
