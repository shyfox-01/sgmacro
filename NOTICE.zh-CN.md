# 数据来源与归属

[English](NOTICE.md)

MIT 许可证适用于本项目的代码与文档，不替代上游数据、模型权重或第三方依赖的使用条款。

## 公开数据

仓库包含[数据来源清单](docs/data-sources.md)中列出的数据，
于 2026 年 10 月 2 日从 [data.gov.sg](https://data.gov.sg/) 和
[SingStat Table Builder](https://tablebuilder.singstat.gov.sg/) 获取。
这些数据适用 [Singapore Open Data Licence](https://data.gov.sg/open-data-licence)，
SingStat 的[使用条款](https://www.singstat.gov.sg/terms-of-use)也适用。

原始发布方包括新加坡市区重建局、建屋发展局、金融管理局、人力部和统计局。
每份快照记录来源、发布方、抓取时间和文件哈希。

快照和目录是保存的历史样例，并非当前数据服务。处理后的表格、计算与文字分析
由本项目生成，不应归属于数据发布方。本项目未获得政府机构背书。

`tests/fixtures/` 中的测试数据来自裁剪后的上游响应或人工构造的样例。
仓库保留部分模型会话及摘要响应供核对和重放，不包含模型权重。

## 政策事件

`config/policy_events.yaml` 记录政策日期、工具及官方公告链接，需要人工维护。
所链接的公告仍遵循其发布方的使用条款。

## 开发工具

开发过程中使用了 Claude 和 ChatGPT / Codex。
模型生成的代码和文字仍受文档所述检查与限制约束。
