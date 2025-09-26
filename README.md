# 项目说明

本仓库包含两个可直接运行的数据流程：

1. **加密货币行情回测管线**：使用 CCXT 拉取 BinanceUS 行情、配合自研清洗脚本与 vectorbt 回测。
2. **新闻 + X（原 Twitter）数据采集管线**：合规接入 RSS/News Sitemap 与 X API，统一清洗并输出分析报表。

---

## 环境准备

```powershell
python -m venv venv
# Windows PowerShell
.\venv\Scripts\Activate.ps1
pip install -U -r requirements.txt
```

> 提示：首次安装耗时较长，后续重复执行会快速完成。

---

## 加密货币行情与回测流程

| 步骤 | 命令 | 说明 |
| --- | --- | --- |
| 1 | `python scripts/fetch_ccxt_ohlcv.py --conf conf/symbols.yaml` | 通过 CCXT 增量拉取 BinanceUS 现货 1h K 线，自动剔除未收盘蜡烛，输出到 `data/raw/exchange=...` |
| 2 | `python scripts/fetch_binance_bulk.py --conf conf/symbols.yaml --limit-months 4` | 可选：下载 Binance Data Vision 月度 ZIP，校验 CHECKSUM 后补齐历史 |
| 3 | `python scripts/clean_ohlcv.py --conf conf/symbols.yaml` | 合并 CCXT 与历史包，去重、排序、按月分区写入 `data/clean/exchange=...` |
| 4 | `python scripts/dq_report.py --mode ohlcv --conf conf/symbols.yaml` | 生成行情数据覆盖率报告（缺口、最大间隔、起止时间等） |
| 5 | `python backtests/vbt_ma_rsi.py --conf conf/symbols.yaml` | 使用 vectorbt 的 MA+RSI 策略网格回测，结果保存到 `backtests/results/*.parquet` |

**默认配置 `conf/symbols.yaml`**

```yaml
exchange: binanceus
type: spot
symbols:
  - BTC/USDT
  - ETH/USDT
timeframes:
  - 1h
start: "2024-01-01"
end: null
store_as: parquet
```

如需增加交易对或时间粒度，直接修改上述文件并重新执行流程。

---

## 新闻 + X 数据采集流程

| 步骤 | 命令 | 说明 |
| --- | --- | --- |
| 1 | `python scripts/news_harvester.py --config conf/feeds.yaml` | 异步抓取 RSS 与 News Sitemap，遵循 robots.txt，使用 trafilatura 抽取正文，输出 `data/raw/news/news_*.parquet` |
| 2 | `python scripts/x_recent_search.py --max-results 20` | 调用 X Recent Search API（需在 `.env` 填写 `X_BEARER_TOKEN`），未配置时会提示并跳过 |
| 3 | `python scripts/x_stream.py --max-messages 20` | 调用 X Filtered Stream（需 `X_STREAM_TOKEN` 或 `X_BEARER_TOKEN`），可根据 `conf/x_rules.yaml` 自动同步规则 |
| 4 | `python scripts/normalize.py` | 合并新闻与 X 数据，利用 SimHash 近重过滤，输出统一结构 `data/clean/news/all.parquet` |
| 5 | `python scripts/dq_report.py --mode news` | 统计来源、语言、发布时间区间、重复 URL 等质量指标 |

**配置文件示例**

- `conf/feeds.yaml`：定义 RSS、News Sitemap、是否启用 NewsAPI/GDELT。
- `conf/x_rules.yaml`：X 过滤流规则，支持多主题。
- `.env`：存放 X/NewsAPI/GDELT 等密钥（示例已提供占位字段）。

> X 平台严格限制数据用途与返回量，必须遵守官方开发者协议，收到删除事件需及时同步删除本地数据。

---

## 数据输出结构

- `data/raw/news/`：新闻原始 parquet（含正文、摘要、元数据、原始 JSON 文本）。
- `data/raw/x/`：Recent Search `jsonl` 与 Filtered Stream `ndjson`。
- `data/clean/news/all.parquet`：统一后的新闻/X 数据集，字段包括 `source`、`url`、`title`、`published_at`、`text`、`tags`、`raw_json` 等。
- `data/clean/exchange=.../`：按月分区的行情数据。
- `backtests/results/*.parquet`：vectorbt 回测指标表。

---

## 常见扩展

1. **调度**：结合 `cron`、`APScheduler` 或 Prefect 实现定时采集与回测。
2. **分析**：使用 DuckDB/Polars 直接读取 parquet，或落地至 PostgreSQL + OpenSearch 并在 Superset/Metabase 搭建仪表盘。
3. **NLP 富化**：在 `scripts/normalize.py` 中扩展 spaCy/HF 模型实现实体抽取、情感分析、摘要等。
4. **风险控制**：在回测脚本中加入仓位管理、滑点敏感度测试、分层走样验证。

---

## 合规提醒

- **X API**：仅使用官方端点，遵守速率与内容使用政策，禁止抓取网页端或用于训练基础模型。
- **新闻站点**：采集前检查 robots.txt，必要时与站点签订或申请额外授权。
- **数据删除**：收到 X Compliance 通知或版权要求时，需同步删除本地存档。

---

如需将上述流程打包成最小可运行模板（含配置示例、调度脚本或可视化示例），欢迎继续告知需求。
