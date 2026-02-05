# Plan Gap（`plan.md`）— 仅 PARTIAL/MISSING

Generated: 2026-02-05

Source: `docs/plan_gap_planmd.md`（全量逐章核对，不省略）

---

说明：本文件是从 `docs/plan_gap_planmd.md` 中抽取所有包含 `**PARTIAL**` / `**MISSING**` 的章节块，用于只看差距列表与证据的场景。

---

## (L8) Agent Market Pro 工程 Proposal（Factor Compiler DSL + 微观结构特征 + TCA + Flow 扩展）

- Status: **PARTIAL**
- Evidence:
  - 已有 Flow/Jobs/Results 基座：`src/agent_market/agent_flow.py`, `server/api/routes/*`
  - 已实现 Phase1 的 `micro_feature` / `tca`：`src/agent_market/microstructure/`, `src/agent_market/tca/`
  - 已实现 KuCoin `capture`（fixture+live）：`scripts/micro_capture.py`, `src/agent_market/microstructure/capture/`
- Gaps: Factor Compiler 类型系统/完整算子库/高级 scoring；微观结构特征表完整覆盖；TCA orders/fills/impact 等深度指标（见下文各章节）

---

#### (L180) 3.3.1 核心类型
- Status: **PARTIAL**（已新增最小 `FactorType`/`infer_expr_type`；尚缺完整 Type System/规则与更细语义类型）

---

#### (L190) 3.3.2 类型属性（必须携带）
- Status: **PARTIAL**
- Notes:
  - 最小 `FactorType` 已携带 `freq/timezone/availability_delay_ms/lookback` 字段，但尚未形成“强制携带 + 强约束校验”的闭环。

---

#### (L268) 3.5.2 类型检查（Typecheck）
- Status: **PARTIAL**（已有最小 type inference + API preflight；尚缺完整 Type System + 规则库）

---

#### (L274) 3.5.3 时间安全（Time-safety）
- Status: **PARTIAL**
- Notes:
  - 已将 lookahead 防护下沉为“可执行层强制”（`shift(n>=0)`）+ 结构检查（负 shift 直接 fail）。
  - walk-forward 的 purge/embargo 已有最小可运行实现（`eval_protocol`），但尚未成为训练/回测的强制 gate。

##### 3.5.3 条目逐项核对（不省略）

| Item | Status | Notes |
|---|---|---|
| 禁止 `lag(x, -k)` | **DONE** | `ExpressionEngine` validator + runtime 强制 `shift(n>=0)`；另有结构检查 `check_no_negative_shift()` |
| label 必须显式 `future_return(h)` | **DONE** | `src/agent_market/freqai/training/labels.py` + pipeline 统一入口 |
| 训练/评测阶段自动 purge/embargo | **PARTIAL** | 已有最小 `eval_protocol`（walk-forward + purge/embargo）但未全链路强制 |
| `availability_delay_ms` 可交易性约束 | **PARTIAL** | 已加入 best-effort gate（`check_time_safety(..., min_delay_ms=...)`）；真实延迟建模仍缺 |

---

### (L352) 4.2 订单流（Order Flow）与成交流（Trades）
- Status: **PARTIAL**
- Notes: 已实现从 KuCoin `match` 计算 trade_sign/vwap/ofi/arrival_intensity/buy_vol/sell_vol，并对齐到 LOB 时间戳；`ofi_w` 仍为 signed-volume proxy（非 L2 delta-OFI）。

---

### (L364) 4.3 执行与 adverse selection proxy
- Status: **PARTIAL**
- Notes: 已补齐执行/毒性 proxy 特征（best-effort）；仍缺真实校准与更严格的执行回放对齐。

#### 4.x 计划表格中每个 feature 的逐项核对（不省略）

**4.1 LOB shape**

| Feature | Status | Notes |
|---|---|---|
| `mid` | **DONE** | 由 `lob_state.parquet` 产出并在 microstructure features 透传 |
| `spread` | **DONE** | 同上 |
| `rel_spread` | **DONE** | 同上 |
| `microprice` | **DONE** | `src/agent_market/microstructure/features/core_features.py` |
| `depth_bid_L` | **DONE** | 实现为 `depth_bid_{L}`（参数化 levels） |
| `depth_ask_L` | **DONE** | 实现为 `depth_ask_{L}`（参数化 levels） |
| `imbalance_L` | **DONE** | 实现为 `imbalance_{L}`（参数化 levels） |
| `slope_bid_L` | **DONE** | 实现为 `slope_bid_{L}`（best-effort size~distance slope） |
| `convexity_L` | **DONE (proxy)** | `convexity_{L}`（best-effort；基于多档 size 分布的二阶形态） |

**4.2 Trades / Order Flow**

| Feature | Status | Notes |
|---|---|---|
| `trade_sign` | **DONE** | KuCoin `match.data.side` 直接提供买卖方向 |
| `buy_vol_w` | **DONE** | trades rolling buy volume（`buy_vol_{w}`） |
| `sell_vol_w` | **DONE** | trades rolling sell volume（`sell_vol_{w}`） |
| `ofi_w` | **PARTIAL** | 当前为 trades signed-volume proxy（非 L2 delta-OFI） |
| `vwap_w` | **DONE** | trades rolling vwap |
| `rv_w` | **DONE** | realized volatility on `mid` pct change（`rv_{w}`） |
| `arrival_intensity_w` | **DONE** | trades rolling count / window_sec |

**4.3 Execution / toxicity proxies**

| Feature | Status | Notes |
|---|---|---|
| `expected_slippage_proxy` | **DONE (proxy)** | best-effort：基于 spread/波动等启发式 |
| `fill_prob_proxy` | **DONE (proxy)** | best-effort：基于 limit/impact 等启发式（缺列降级） |
| `toxicity_proxy` | **DONE (proxy)** | best-effort：基于价格冲击/不平衡等启发式 |

---

### (L382) 5.1 TCA 报告顶层 JSON Schema（v1）
- Status: **PARTIAL**
- Evidence:
  - 已实现 plan.md v1 schema 位置：`src/agent_market/tca/schema.py`
  - CLI：`scripts/tca_report.py`
- Gaps:
  - 订单/成交（orders/fills）与 arrival_mid/impact 等需要接入真实执行/LOB 基准，当前多为 `null/[]` 占位。

#### 5.1.1 plan.md v1 schema 字段级核对（不省略）

> 下面按 `plan.md` JSON 示例的结构逐项核对。

| Path | Status | Notes |
|---|---|---|
| `schema_version` | **DONE** | 输出 `"1.0"` |
| `meta.run_id` | **DONE** | 与 `run_id` 一致 |
| `meta.generated_at` | **DONE** | 与 `generated_at` 一致 |
| `meta.exchange` | **PARTIAL** | 字段存在，常为 `null`（取决于 backtest 元信息） |
| `meta.market` | **PARTIAL** | 字段存在，常为 `null`（spot/perp 未建模） |
| `meta.symbols[]` | **DONE** | 从 backtest trades 的 `pair` 汇总（可能为 `BTC/USDT` 格式） |
| `meta.time_range.start/end` | **PARTIAL** | 字段存在，best-effort（取决于 backtest 元信息） |
| `meta.timeframe` | **PARTIAL** | 字段存在，best-effort（取决于 backtest 元信息） |
| `meta.data_sources[]` | **DONE** | 写入 `["freqtrade_backtest"]` |
| `meta.strategy.name/params` | **DONE** | 写入 strategy name + 空 params |
| `orders[]` | **PARTIAL** | best-effort：从 backtest `trade['orders']` 提取（缺失则为 `[]`） |
| `fills[]` | **PARTIAL** | best-effort：从 backtest `trade['orders']` 提取（缺失则为 `[]`） |
| `benchmarks.arrival_mid` | **PARTIAL** | 输出 definition + `value_series_ref=null` |
| `benchmarks.vwap` | **PARTIAL** | 输出 definition + `value_series_ref=null` |
| `costs.implementation_shortfall.total` | **PARTIAL** | 最小 IS proxy：以 fees 为 `quote_ccy`；有 fills 时可算 `fees_bps` |
| `costs.implementation_shortfall.by_component` | **PARTIAL** | `spread/delay/market_impact` 以 `0.0` proxy 补齐；fees 写入 bps+quote_ccy |
| `costs.slippage_distribution` | **PARTIAL** | orders 存在时输出 `0.0` 占位，否则为 `null` |
| `costs.fill.fill_rate` | **PARTIAL** | orders 存在时输出 `1.0` 占位，否则为 `null` |
| `costs.fill.avg_fill_latency_ms` | **PARTIAL** | orders 存在时输出 `0.0` 占位，否则为 `null` |
| `costs.fill.cancel_rate` | **PARTIAL** | orders 存在时输出 `0.0` 占位，否则为 `null` |
| `diagnostics.regime` | **PARTIAL** | 占位（bucket 为 `null`） |
| `diagnostics.notes[]` | **PARTIAL** | 输出 `[]` |
| `diagnostics.plots[]` | **PARTIAL** | 输出 `[]` |
| `diagnostics.participation` | **PARTIAL** | best-effort：OHLCV volume participation proxy（缺数据时为 `null`） |

---

### (L469) 5.2 TCA 指标定义建议（v1 必含）
- Status: **PARTIAL**
- Evidence:
  - `fees_total` 已聚合；IS total 以 fees 为最小 proxy；spread/delay/impact 以 `0.0` proxy 补齐；participation 提供 OHLCV volume proxy。

#### 5.2.1 “v1 必含”指标逐项核对（不省略）

| Metric | Status | Notes |
|---|---|---|
| Implementation Shortfall（IS） | **PARTIAL** | 最小 proxy：以 fees 为 total（缺 arrival benchmark 与更完整分解） |
| Spread cost（taker vs maker） | **PARTIAL** | 0.0 proxy（缺 maker/taker 标记或推断） |
| Delay cost | **PARTIAL** | 0.0 proxy（缺 decision/submit 时间戳） |
| Market impact | **PARTIAL** | 0.0 proxy（缺冲击模型） |
| Fees | **PARTIAL** | 已聚合 fee_open/fee_close/funding_fees |
| Fill quality | **PARTIAL** | 字段占位已输出；需真实 fills/延迟统计 |
| Participation/footprint | **PARTIAL** | OHLCV volume proxy（缺更严格的市场成交量对齐） |

---

### (L551) 8.1 LLM 输出：必须是 `FactorSpec` JSON
- Status: **PARTIAL**
- Evidence:
  - 已新增 prompt assets（`factor_spec.system.md`/fewshot）与离线 parse/validate 入口（`src/agent_market/freqai/llm.py`）
  - 现有 expression agent 仍以 `expressions[]` 为主输出（未强制切换为 FactorSpec）

---

### (L556) 8.2 反馈闭环
- Status: **PARTIAL**
- Evidence:
  - 现有 backtest_summary → `user_data/llm_feedback/latest_backtest_summary.json`
  - 缺少 factor_eval 的 ScoreReport 反馈闭环

---

## (L565) 9. 评测规范（让你过审、不被喷、也更容易赚钱）
- Status: **PARTIAL**
- Notes:
  - 已有最小 walk-forward（purge/embargo）与成本入账（fees）评测协议（`eval_protocol`）；仍缺容量/impact/参与率等更完整指标。

---

### (L576) 10.1 30 天（最小可卖：Execution Intelligence Lite）
- Status: **PARTIAL**
- Notes: `micro_feature`/`tca` 已有；`capture`/`lob_rebuild` 已接入 Flow；TCA v1 已补齐 orders/fills 与最小 IS（fees 入账），但 arrival_mid/impact/delay 等仍缺。

---

### (L582) 10.2 60 天（Factor Compiler v1）
- Status: **PARTIAL**

---

## (L595) 11. 交付清单（你最终应该在 repo 里看到什么）

### 1) Factor Compiler
- Status: **PARTIAL**

### 2) 微观结构
- Status: **PARTIAL**（capture/LOB/features registry 已有；多交易所/更完整表格仍缺）

### 3) TCA
- Status: **PARTIAL**（schema v1 位置齐全；orders/fills 与最小 IS 已补齐；impact/delay/spread 分解仍缺）

### 4) Flow
- Status: **PARTIAL**（Flow 已接入 capture/lob/factor/report；仍缺与 plan.md 建议的 data/results 路径完全对齐）
