# 架构解析

## 1. 顶层分层

| 模块 | 说明 | 关键入口 |
| ---- | ---- | -------- |
| `scripts/` | 行情、资讯、链上等外部数据任务。各脚本遵循“读取配置 → 拉取 → 清洗/导出”的轻量模式。 | `fetch_ccxt_ohlcv.py`, `clean_ohlcv.py`, `dex_indexer.py`, `news_harvester.py` |
| `freqtrade/` | 上游 freqtrade 项目的副本，在 `scripts/` 中新增 LLM/自动化特征挖掘脚本。 | `scripts/freqai_expression_agent.py`, `scripts/freqai_auto_agent.py` |
| `src/agent_market/` | 本项目扩展的 Python 包，集中封装 LLM 调用、提示词生成等共享逻辑。 | `freqai/llm.py` |
| `docs/` | 使用说明、架构解析、LLM 流水线操作指南。 | `llm_pipeline.md`, `ARCHITECTURE.md` |

## 2. 数据脚本解析

### 2.1 行情处理 (`scripts/`)

- `fetch_ccxt_ohlcv.py`：基于 CCXT 抓取实时行情。
- `fetch_binance_bulk.py`：批量下载 Binance Data Vision ZIP，并生成校验报告。
- `clean_ohlcv.py`：将原始数据标准化为统一的 `data/clean/` 输出格式，支持缺失值填补与指标附加。
- `dq_report.py`：输出缺失区间、异常波动等质量指标。

整体设计以 YAML 配置(`conf/*.yaml`) 控制资产范围与参数，使脚本可组合、可替换。

### 2.2 资讯/链上 (`news_harvester.py`, `x_stream.py`, `dex_indexer.py`)

- 统一入口通过 `.env` 提供 API Key/RPC。
- 输出采用 CSV/Parquet，便于接入后续分析或入库流程。

## 3. FreqAI LLM 因子流水线

```
features.json ──> (LLM / 模板 / gplearn) ──> expressions.json ──> freqtrade backtesting ──> 结果压缩包 + 摘要
```

### 3.1 `freqai_expression_agent.py`

- 解析 `user_data/freqai_features.json` 重建基础指标。
- `agent_market.freqai.llm` 提供默认 LLM 配置、提示词、候选解析。
- 若 LLM 调用失败，会自动回退到模板组合 & gplearn 的遗传编程搜索。

### 3.2 `freqai_auto_agent.py`

- 执行流程：
  1. `aggregate_features`：对多交易对按指标聚合、打分。
  2. `build_expression_payload`：调用 LLM 生成表达式，按稳定性、夏普、复杂度评分。
  3. `compute_auto_timerange`：根据数据文件决定训练/回测时间窗。
  4. `run_backtest`：重用 freqtrade CLI，并将 `.last_result.json` 对应的 zip 复制到 `auto_agent/<timestamp>/`。
  5. `summarize_result`：解压或读取 JSON，输出交易笔数、收益、胜率。
- 所有 LLM 相关参数可直接通过命令行覆盖。

### 3.3 LLM 模块 (`src/agent_market/freqai/llm.py`)

- `LLMConfig`: 数据类封装 base url / 模型 / 温度 / 重试等参数。
- `build_prompt`: 拼装含特征说明、允许函数列表的提示词。
- `request_completion`: 包含重试、异常提示、token 消耗统计。
- `extract_candidates`: 清洗 LLM 返回的表达式列表，消除重复并保留元数据描述。

## 4. 回测结果组织

- freqtrade 默认输出到 `user_data/backtest_results/`，本项目新增自动目录 `user_data/backtest_results/auto_agent/<timestamp>/`，并保留原始 zip 以便复现。
- `_resolve_result_path` 会优先解析 `.last_result.json`，若缺失则回退到目录内最新 zip/JSON 文件。

## 5. 下一步优化建议

1. **测试覆盖**：为关键脚本（LLM 因子、数据清洗）添加 pytest + 快速烟雾测试，保障重构后的稳定性。
2. **配置统一**：考虑将 `conf/` 与 `user_data/config_freqai.json` 合并为单一 YAML 管理入口，更易于批量调整参数。
3. **任务编排**：可引入 `invoke` 或 `poetry scripts`，整合常用命令为标准化 CLI，减少命令行参数重复输入。
4. **监控与缓存**：为 LLM 调用增加磁盘缓存与速率限流，避免高频重复请求导致的成本浪费。

