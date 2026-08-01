# AB 实验分流算法

> **5000 用户 / 10 组，每组偏差 < 1% 的完整实现、实验验证与工程实践**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/Version-v0.5.0-blue.svg)](docs/CHANGELOG.md)
[![Scripts](https://img.shields.io/badge/Scripts-27+-green.svg)](docs/PROJECT_STRUCTURE.md)

本项目记录了一个完整的工程问题研究过程：从实际工作中踩到的 AB 实验分坑出发，通过数学推导 + 实测验证 + 多种方案对比，最终给出可工业部署的方案。

---

## �� 一键上手（30 秒看到效果）

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 跑 3 个最小示例
python examples/01_quickstart.py          # 蛇形分配 + SRM + MDE
python examples/02_validation_report.py   # 完整实验报告
python examples/03_mab_decision.py        # 6 维决策框架

# 3. 跑核心算法（无需 Kaggle 数据）
python scripts/realtime_prebucket.py      # P1 用户池 0% 偏差
python scripts/ab_split_validator.py      # 蛇形分配 + 三层验证

# 4. 跑数据驱动验证（首次自动下载 348MB Kaggle 数据）
python scripts/did_cuped_kaggle.py        # fraud 场景 CUPED
python scripts/did_cuped_consumption.py   # consumption 场景 CUPED
```

> **旧命令兼容**：原本根目录下的 `python xxx.py` 命令已迁移到 `scripts/xxx.py`，旧 README 链接全部仍能跑。

---

## �� 文档导航（按角色快速上手）

| 我是谁 | 看哪 | 用途 |
|---|---|---|
| �� **工程师**（先跑代码）| [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) | 文件索引 + 一键复现命令 |
| �� **算法学习者** | [docs/README.md](docs/README.md) | 13 个课题 + 三层结构 |
| ��️ **架构师** | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 模块边界 + 数据流图 |
| �� **业务方 / PM** | [docs/README.md#业务视角详解](docs/README.md) | 红绿灯判断法 |
| �� **回顾历史** | [docs/CHANGELOG.md](docs/CHANGELOG.md) | 版本演进 |
| �� **找特定脚本** | [docs/PROJECT_STRUCTURE.md#算法索引表](docs/PROJECT_STRUCTURE.md) | 算法分类索引 |
| ��️ **归档旧文档** | [docs/archive/](docs/archive/) | 早期 SUMMARY / EVALUATION_TABLE |

---

## ��️ 项目结构

```
AB实验技术实现/
├── abexp/                       # 核心 Python 包
│   ├── routing/                 # 流量分配（10 个脚本）
│   ├── analysis/                # 数据分析（5 个脚本）
│   ├── validation/              # 实验检验（5 个脚本）
│   ├── advanced/                # 进阶主题（4 个脚本）
│   └── tools/                   # 工具（3 个脚本）
├── examples/                    # 3 个最小可运行示例
├── tests/                       # smoke test
├── scripts/                     # 旧命令兼容（27 个薄包装）
├── docs/                        # 完整文档
│   ├── README.md                # 主文档（13 个课题）
│   ├── PROJECT_STRUCTURE.md     # 文件索引
│   ├── ARCHITECTURE.md          # 架构图
│   ├── CHANGELOG.md             # 版本演进
│   └── archive/                 # 归档
├── test_data/                   # 测试数据
├── pyproject.toml               # 包元数据
├── requirements.txt             # 依赖
├── LICENSE                      # MIT
└── README.md                    # 本文件（index）
```

### 公开 API（建议用法）

```python
from abexp.routing.ab_split_validator import assign_groups, srm_check, calc_mde
from abexp.validation.experiment_validation_report import validate_full_pipeline
from abexp.advanced.mab_vs_ab_when import recommend
```

---

## �� 本项目亮点

```
✅ 真实数据验证        2000 用户 / 13M+ 笔交易（Kaggle 真实数据）
✅ 工业级算法         P1 用户池 0% 流量偏差（实测）
✅ 完整流水线         SRM + ANOVA + 显著性 + Markdown 报告
✅ 业务友好           红绿灯判断法（业务方也能用）
✅ 透明可复现         100% 脚本独立可跑，无黑盒
✅ 跨学科专题         金融场景 30 天账期专题
✅ 完整开源化         pyproject.toml + tests + examples
✅ 旧命令兼容         python scripts/xxx.py 仍可跑
```

---

## ❓ 常见问题

### Q: 旧命令 `python ab_split_validator.py` 还能跑吗？
**A**: 能。已迁移到 `python scripts/ab_split_validator.py`，输出与原版完全一致。

### Q: 新代码怎么 import？
**A**: 直接从 `abexp` 包导入：
```python
from abexp.routing.ab_split_validator import assign_groups
from abexp.advanced.mab_vs_ab_when import recommend
```

### Q: 跑测试？
**A**: `pytest tests/`

### Q: 这个项目能上生产吗？
**A**: 不能直接上。它是**原理 + 验证 + 参考实现**。完整答案见 [docs/ARCHITECTURE.md#常见架构问题](docs/ARCHITECTURE.md)。

---

## �� 许可证

MIT License — 详见 [LICENSE](LICENSE)
