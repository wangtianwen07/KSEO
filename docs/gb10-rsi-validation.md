# GB10 RSI 工程验证记录

日期：2026-11-17。基础提交：`f94bf2f1bb3ec6a409eb11ec284187c081fc02c3`。版本：`0.6.0.dev0`。执行环境为 WSL Ubuntu、Python 3.12.3、Git；没有启动模型或占用 GPU。

| 检查 | 结果 |
| --- | --- |
| `PYTHONPATH=src python3 -m unittest discover -s tests -v` | 180 项全量测试通过。 |
| 最后增加经验摘要长度限制后的 `test_rsi.py` | 22 项专项测试通过；包含一个新增测试，与全量测试有重叠，不累计声称 202 项。 |
| Grid、Agent、pipeline 三个 README CPU 示例 | 全部 `COMPLETED`，均为 synthetic。 |
| `kseo rsi demo` | `COMPLETED`；包含错误候选拒绝、多层累计 patch、E2E 与最终审计。 |
| `rsi references` | 实际检索 12 条公开 PR/Issue，保留来源与版本信息；缓存后的 CLI 重放也通过。 |
| `rsi enable` + 含真实参考快照的 CPU run | 合同派生与运行通过，结果仍为 synthetic。 |
| `rsi prepare` 使用占位 brief | 正确保留草稿，报告缺模型访问，未误标为真实就绪。 |
| `git diff --check` | 通过。 |

新增测试覆盖 UCT 回传、失败编译反馈、固定 oracle、冻结提取清单、累计补丁与 E2E tree 一致性、中断恢复和不可返还的候选预算、资料快照与工具 manifest、跨硬件经验过滤、摘要长度上限、GB10 身份拒绝、成对计时噪声门禁、逐 token 缺失/重复/错误 scorer/大 logit 偏差及原始产物篡改。

首次使用 Windows 自动 CRLF checkout 在 WSL 运行时，4 项原有 patch 示例失败；将工作副本恢复 LF 后原有 pipeline 19 项通过。本分支为 Python 源码加入 `text eol=lf`，避免同类跨环境换行问题，冻结数据文件仍保留原始字节。

公开 GitHub API 首次遇到匿名额度耗尽（HTTP 403）；使用本机已有 GitHub 身份后完成只读检索。token 没有写入源码、快照或报告。检索加入一小时缓存以减少限流后的重复请求，并对动态查询结果增加 GLM 标题筛选。PR 状态只表示抓取时外部状态，不作为本机验证。

本地 WSL 缺少 `ensurepip/python3-venv`，没有完成新的隔离 venv 安装检查；所有上述 CLI 均通过源码 `PYTHONPATH=src` 执行。CI 已增加已安装 `kseo rsi demo`，沿用 Python 3.10/3.12 矩阵。此工作副本采用 sparse checkout，未下载既有 LiveCodeBench 大型压缩分片；此次测试未对这些完整分片执行重新组装或模型评分。

真实 GB10 的 kernel 编译、完整模型正确性、teacher-forced scorer、量化格式适配、速度提升及跨部署经验成功迁移尚未实测。实测任务必须提供真实 adapter 与冻结合同，并通过 [GB10 接入协议](gb10-rsi.md) 中的所有门禁；本文不构成硬件性能证明。
