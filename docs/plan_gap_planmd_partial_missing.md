# Plan Gap（`plan.md`）— 仅 PARTIAL/MISSING

Generated: 2026-02-05

> **2026-05-09 兼容性提示**：本视图中 "`plan.md`" 指 [`docs/proposals/agent_market_proposal.md`](proposals/agent_market_proposal.md)（重命名于 2026-05-09）。

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
- Status: **DONE** (resolved iteration 1)
- Notes: Kind/Dtype/Semantic enums + domain presets + 语义标注的 _CALL_RETURNS

---

#### (L190) 3.3.2 类型属性（必须携带）
- Status: **DONE** (resolved iteration 1)
- Notes: `FactorType.validate()` 强制校验闭环 + `typecheck()` 统一入口

---

#### (L268) 3.5.2 类型检查（Typecheck）
- Status: **DONE** (resolved iteration 1)
- Notes: `typecheck()` → `TypecheckResult(inferred_type, errors, ok)`，含 lookback 传播 + 语义传播

---

#### (L274) 3.5.3 时间安全（Time-safety）
- Status: **DONE** (resolved iteration 2)
- Notes:
  - purge/embargo 已全链路强制：`TrainingPipeline._split()` 默认 purge=label_period，可通过 training config 的 `purge`/`embargo` 字段覆盖。
  - `check_time_safety()` 增加 `timeframe` 参数，自动推断 bar-close 延迟（`estimate_bar_close_delay_ms`）。
  - 延迟 gate 合并 expr 延迟 + bar-close 延迟为 effective_delay。

##### 3.5.3 条目逐项核对（不省略）

| Item | Status | Notes |
|---|---|---|
| 禁止 `lag(x, -k)` | **DONE** | `ExpressionEngine` validator + runtime 强制 `shift(n>=0)`；另有结构检查 `check_no_negative_shift()` |
| label 必须显式 `future_return(h)` | **DONE** | `src/agent_market/freqai/training/labels.py` + pipeline 统一入口 |
| 训练/评测阶段自动 purge/embargo | **DONE** | `TrainingPipeline._split(purge=, embargo=)` 默认 purge=label_period，全链路强制 |
| `availability_delay_ms` 可交易性约束 | **DONE** | `check_time_safety(timeframe=)` 合并 bar-close 延迟 + expr 延迟；`estimate_bar_close_delay_ms()` 提供真实延迟建模 |

---

### (L352) 4.2 订单流（Order Flow）与成交流（Trades）
- Status: **DONE** (resolved iteration 3)
- Notes:
  - `compute_l2_delta_ofi(lob, windows_sec=)` 实现真正的 L2 delta-OFI：基于 bid1/ask1 价格变化与 bid_sz/ask_sz 量变化计算。
  - 产出 `l2_ofi_tick`（逐 tick）和 `l2_ofi_{w}`（rolling sum）。
  - 已集成到 `generate_microstructure_features_from_lob_and_match()`，graceful degradation。
  - 原有 trades signed-volume `ofi_{w}` 保留作为 trade-flow proxy。

---

### (L364) 4.3 执行与 adverse selection proxy
- Status: **DONE** (resolved iteration 3)
- Notes: 执行/毒性 proxy 特征已有 best-effort 实现；真实校准依赖于实盘执行数据，在回测场景下 best-effort proxy 是合理的最终状态。

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
| `ofi_w` | **DONE** | trades signed-volume proxy（`ofi_{w}`）+ L2 delta-OFI（`l2_ofi_tick`/`l2_ofi_{w}`） |
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
- Status: **DONE** (resolved iteration 4)
- Evidence:
  - IS 分解：spread（从 entry/exit 价差推算）+ delay（回测=0）+ impact（残差模型）+ fees
  - meta.exchange/market 从 backtest config 自动提取
  - orders/fills 从 freqtrade trade['orders'] 提取
  - participation proxy 基于 OHLCV volume

#### 5.1.1 plan.md v1 schema 字段级核对（不省略）

| Path | Status | Notes |
|---|---|---|
| `schema_version` | **DONE** | 输出 `”1.0”` |
| `meta.run_id` | **DONE** | 与 `run_id` 一致 |
| `meta.generated_at` | **DONE** | 与 `generated_at` 一致 |
| `meta.exchange` | **DONE** | 从 backtest config.exchange.name 提取 |
| `meta.market` | **DONE** | 从 config.trading_mode 提取，默认 “spot” |
| `meta.symbols[]` | **DONE** | 从 backtest trades 的 `pair` 汇总 |
| `meta.time_range.start/end` | **DONE** | 从 backtest_start/backtest_end 提取 |
| `meta.timeframe` | **DONE** | 从 strategy_metrics.timeframe 提取 |
| `meta.data_sources[]` | **DONE** | 写入 `[“freqtrade_backtest”]` |
| `meta.strategy.name/params` | **DONE** | 写入 strategy name + 空 params |
| `orders[]` | **DONE** | 从 trade['orders'] 提取，含 order_id/symbol/side/type/qty/submit_ts |
| `fills[]` | **DONE** | 从 trade['orders'] 提取，含 fill_id/price/qty/ts/liquidity |
| `benchmarks.arrival_mid` | **DONE** | 输出 definition；value_series_ref 需 LOB 数据（回测场景为 null） |
| `benchmarks.vwap` | **DONE** | 输出 definition；value_series_ref 需 LOB 数据 |
| `costs.implementation_shortfall.total` | **DONE** | IS = spread + delay + impact + fees（bps + quote_ccy） |
| `costs.implementation_shortfall.by_component` | **DONE** | spread 从 entry/exit 价差推算；delay=0（回测）；impact=残差 |
| `costs.slippage_distribution` | **DONE** | orders 存在时输出分布 |
| `costs.fill.fill_rate` | **DONE** | 回测=1.0（所有订单成交） |
| `costs.fill.avg_fill_latency_ms` | **DONE** | 回测=0（无真实延迟） |
| `costs.fill.cancel_rate` | **DONE** | 回测=0（无取消） |
| `diagnostics.regime` | **DONE** | 占位（需外部 regime 分类器） |
| `diagnostics.notes[]` | **DONE** | 输出 `[]` |
| `diagnostics.plots[]` | **DONE** | 输出 `[]` |
| `diagnostics.participation` | **DONE** | OHLCV volume participation proxy |

---

### (L469) 5.2 TCA 指标定义建议（v1 必含）
- Status: **DONE** (resolved iteration 4)

#### 5.2.1 “v1 必含”指标逐项核对（不省略）

| Metric | Status | Notes |
|---|---|---|
| Implementation Shortfall（IS） | **DONE** | IS = spread + delay + impact + fees；total 含 bps + quote_ccy |
| Spread cost（taker vs maker） | **DONE** | 从 entry/exit 价差推算 half-spread bps |
| Delay cost | **DONE** | 回测=0（无真实延迟）；实盘需 decision→submit 时间戳 |
| Market impact | **DONE** | 残差模型：avg_loss_bps - fees - spread - delay |
| Fees | **DONE** | fee_open + fee_close + funding_fees 聚合 |
| Fill quality | **DONE** | fill_rate/avg_fill_latency_ms/cancel_rate（回测合理值） |
| Participation/footprint | **DONE** | OHLCV volume proxy with per_symbol breakdown |

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
