# sgmacro

[![Tests](https://github.com/shyfox-01/sgmacro/actions/workflows/tests.yml/badge.svg)](https://github.com/shyfox-01/sgmacro/actions/workflows/tests.yml)
![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

[English](README.md)

用于新加坡住房研究的数据整理与报告工具。

sgmacro 从 SingStat 和 data.gov.sg 读取住房及宏观经济数据，检查数据定义和更新情况，
输出 CSV/JSON 表格与分析报告。报告中的每个数值都带有来源和计算记录，可以回到原始数据核对。

仓库包含三个住房指数和十三个候选预测变量。它完成的是建模前的数据准备，
目前没有训练房价预测模型。

[示例报告](reports/2026-10-02_d26e13cf/report.md) ·
[设计与变量选择](docs/design.md) ·
[命令说明](docs/usage.md)

![示例报告中的住房指数](reports/2026-10-02_d26e13cf/charts/targets.svg)

## 运行示例

测试环境为 Python 3.12，程序要求 Python 3.12 或更高版本。

```bash
git clone https://github.com/shyfox-01/sgmacro.git
cd sgmacro
```

**Windows / PowerShell**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]" -c constraints.txt
.\.venv\Scripts\sgmacro.exe run --as-of 2026-10-02 --offline --llm cache
```

**macOS / Linux**

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]" -c constraints.txt
.venv/bin/sgmacro run --as-of 2026-10-02 --offline --llm cache
```

安装好依赖后，这个示例不需要联网、API key 或运行中的模型。
程序读取仓库内的数据快照和已缓存的摘要。

在本地浏览器打开 [report.html](reports/2026-10-02_d26e13cf/report.html)，
可以点击数值查看来源、对应时期和计算过程。
GitHub 页面可以直接阅读 [Markdown 报告](reports/2026-10-02_d26e13cf/report.md)。

| 位置 | 内容 |
| --- | --- |
| `reports/<run-id>/` | Markdown、HTML 报告及 SVG 图表 |
| `data/processed/<run-id>/` | 观测值、序列定义、验证结果、信号及相关性诊断 |
| `data/snapshots/` | 上游响应、发布方元数据、抓取时间和文件哈希 |
| `runs/<run-id>/` | 运行清单和执行日志 |

运行 ID 由输入和报告内容决定。同一请求的重放使用同一目录；
报告内容不同则生成不同的 ID。

## 数据范围

| 目标指标 | 覆盖范围 |
| --- | --- |
| URA 私人住宅价格指数 | 私宅售价，全部住宅类型 |
| URA 私人住宅租金指数 | 全岛非有地私宅租金 |
| HDB 转售价格指数 | HDB 转售交易 |

十三个候选变量覆盖融资成本、债券收益率、住房信贷、GDP、失业、通胀、
交易量、库存、空置率、在建供应、人口和住房政策。
[序列注册表](config/sources.yaml)记录每个变量的经济解释、来源标识、
单位、发布滞后和预期影响方向。

大部分输入来自公开 API。住房政策指标来自
[官方公告整理表](config/policy_events.yaml)，统计收紧与放松的轮次，需要人工维护。

示例数据抓取于 **2026 年 10 月 2 日**，并非持续更新的数据服务。
数据来源与使用条款见 [NOTICE.zh-CN.md](NOTICE.zh-CN.md)。

## 工作流程

```text
序列注册表
  -> 下载并保存快照
  -> 解析和逐序列验证
  -> 计算变化、信号和领先滞后相关性
  -> 整理数值及来源记录
  -> 输出表格、图表和报告
```

验证针对实际使用的序列，检查单位、频率、缺失值、重复时期和过期数据。
报告日期、声明的发布滞后，以及已保存快照的抓取时间共同决定可用数据。

注册表还保留三个负面测试样例：已停止更新的租金序列、过期的住房贷款序列，
以及单位故意写错的人口序列。测试要求程序正确拒绝这三个样例。

语言模型用于两个可选功能：

- **发现与修复**：agent 搜索新变量或替代数据来源，代码验证提案，
  用户逐项决定是否加入注册表。
- **报告摘要**：模型根据已计算的事实组织文字。检查器拒绝缺乏引用、
  包含数值或方向不一致的陈述；有效陈述不足时使用模板。

常规数据流程不会调用发现 agent。连接方法与限制见[模型配置](docs/models.md)。

## 常用命令

以下命令假设已经激活虚拟环境。也可以使用
`.venv/bin/sgmacro` 或 `.\.venv\Scripts\sgmacro.exe` 的完整相对路径。

```bash
# 按指定报告日期抓取最新数据
sgmacro run --as-of YYYY-MM-DD --refetch --llm off

# 只查看验证结果
sgmacro validate --as-of 2026-10-02

# 从原始数据重新计算示例中的一个数值
sgmacro trace F:331a4bda --run 2026-10-02_d26e13cf

# 发现新的候选变量，需要连接模型
sgmacro discover --brief predictors

# 为失效序列寻找替代来源
sgmacro discover --brief repair --series mas_housing_loans_discontinued
```

请把 `YYYY-MM-DD` 替换为实际日期。提案批准方式与离线行为见[命令说明](docs/usage.md)。

## 验证

```powershell
.\.venv\Scripts\python.exe -m pytest
```

macOS/Linux 使用 `.venv/bin/python -m pytest`。

离线测试覆盖解析、按单位计算、发布日期、数据源故障、提案验证与采纳、
模型输出检查和报告重放。测试会从原始快照独立重算示例中的 49 个直接依赖原始观测的数值，
并比较重新生成的报告与仓库内的示例文件。

[discovery/](discovery/) 保留本地模型的运行记录；
[docs/evidence/](docs/evidence/) 保存一次摘要生成对比。
托管模型后端使用模拟服务器测试，未完成付费云端模型的实测。

## 限制

- 相关性与信号描述数据关系，尚未通过样本外预测验证这些变量的预测价值。
- 仓库中的快照来自同一个抓取日期。较早报告日期的运行仍可能包含后来修订的数据，
  程序会对此发出提示。
- 发布滞后与信号阈值采用声明值；当前没有覆盖 HDB 租金和分地区的住房目标指标。
- 政策轮次统计不能表示每项措施的实际力度。
- 文字通过检查，只表示与声明规则一致，不代表经济解释已经得到证明。

## 许可

项目代码与文档使用 [MIT License](LICENSE)。
原始数据遵循发布方的使用条款，详见 [NOTICE.zh-CN.md](NOTICE.zh-CN.md)。
