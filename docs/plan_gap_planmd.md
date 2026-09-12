# Plan Gap Audit（`plan.md` vs 当前仓库实现）

Generated: 2026-02-05

> **2026-05-09 兼容性提示**：本审计中"`plan.md`"指**早期 Proposal**，现已重命名为 [`docs/proposals/agent_market_proposal.md`](proposals/agent_market_proposal.md)。所有章节引用按此理解；本审计不再大规模 rewrite。
>
> 目标：对 Proposal（原 `plan.md`）的**全部章节（不省略）**逐一查明：已实现 / 部分实现 / 缺失 / 仅叙述（N/A），并给出证据与对应的 `docs/mohu.md` backlog 归档方向。

## Legend

- **DONE**：已有实现且可运行/有测试或可明显验收
- **PARTIAL**：有实现但与 `plan.md` 目标不一致或缺关键子项
- **MISSING**：未实现（目录/接口/产物/测试缺失）
- **N/A**：叙述/定位/参考资料，不是直接可验收的工程项

---

## (L8) Agent Market Pro 工程 Proposal（Factor Compiler DSL + 微观结构特征 + TCA + Flow 扩展）

- Status: **PARTIAL**
- Evidence:
  - 已有 Flow/Jobs/Results 基座：`src/agent_market/agent_flow.py`, `server/api/routes/*`
  - 已实现 Phase1 的 `micro_feature` / `tca`：`src/agent_market/microstructure/`, `src/agent_market/tca/`
  - 已实现 KuCoin `capture`（fixture+live）：`scripts/micro_capture.py`, `src/agent_market/microstructure/capture/`
- Gaps: Factor Compiler 类型系统/完整算子库/高级 scoring；微观结构特征表完整覆盖；TCA orders/fills/impact 等深度指标（见下文各章节）

---

## (L10) 1. 项目定位与目标

### (L12) 1.1 一句话
- Status: **N/A**（定位文案）

### (L16) 1.2 现状基座（你已有）
- Status: **DONE**
- Evidence:
  - 现有目录与 API/Flow/Jobs 体系：`README.md`, `docs/repo_inventory.md`
  - `/run/*` + `/jobs/*` + `/results/*`：`server/api/routes/run.py`, `server/api/routes/jobs.py`, `server/api/routes/results.py`
  - Flow 编排：`src/agent_market/agent_flow.py`, `src/agent_market/flow_steps.py`

### (L22) 1.3 本 Proposal 增量要解决的“3 个硬问题”
- Status: **N/A**（动机/价值描述）

---

## (L30) 2. 总体架构（与现有仓库对齐）

### (L32) 2.1 新增模块总览（建议的目录落位）

> `plan.md` 这里给出目标目录结构。逐项对照如下：

| Proposed Path (plan.md) | Status | Evidence |
|---|---|---|
| `src/agent_market/factor_compiler/` | **PARTIAL** | 已有 DSL/parser/checks/scoring+Flow/API（最小版）；仍缺类型系统/完整算子/高级 scoring |
| `src/agent_market/microstructure/` | **PARTIAL** | 已有 OHLCV micro_feature + microstructure mode（LOB+match）：`src/agent_market/microstructure/micro_feature.py` |
| `src/agent_market/microstructure/capture/` | **DONE** | `src/agent_market/microstructure/capture/kucoin.py` + `scripts/micro_capture.py` |
| `src/agent_market/microstructure/lob/` | **PARTIAL** | 已有 `rebuild.py`；缺 checksum/registry 等 |
| `src/agent_market/microstructure/features/` | **PARTIAL** | 已实现最小 FeatureRegistry + LOB/Trades 特征子集；仍缺完整表格覆盖 |
| `src/agent_market/tca/` | **PARTIAL** | 已实现 plan.md 5.1 v1 schema 结构（多字段占位）；orders/fills/impact 等仍缺失 |
| `src/agent_market/flow_ext/` | **DONE (shim)** | 已新增 `flow_ext/`（steps/artifacts/validators）并在 `agent_flow.py` 引用（薄封装，行为仍复用 legacy `flow_steps.py`） |
| `/run/factor_compile` | **DONE** | `server/api/routes/run.py` |
| `/run/factor_eval` | **DONE** | `server/api/routes/run.py` |
| `/run/micro_feature` | **DONE** | `server/api/routes/run.py` |
| `/run/tca` | **DONE** | `server/api/routes/run.py` |
| `/run/capture` | **DONE** | `server/api/routes/run.py` |
| `/run/lob_rebuild` | **DONE** | `server/api/routes/run.py` |

#### 2.1.1 计划中的“文件级”清单（不省略）

> 下面按 `plan.md` 的示例树列出每个子文件/目录是否存在（实现可不完全同名，但这里按“计划字面路径”逐项核对）。

**Factor Compiler**

- `src/agent_market/factor_compiler/dsl/ast.py`：**DONE**
- `src/agent_market/factor_compiler/dsl/grammar.py`：**DONE**
- `src/agent_market/factor_compiler/dsl/parser.py`：**DONE**
- `src/agent_market/factor_compiler/dsl/operators.py`：**DONE**
- `src/agent_market/factor_compiler/dsl/types.py`：**DONE**
- `src/agent_market/factor_compiler/dsl/serializer.py`：**DONE**
- `src/agent_market/factor_compiler/checks/time_safety.py`：**DONE**
- `src/agent_market/factor_compiler/checks/leakage.py`：**DONE**
- `src/agent_market/factor_compiler/checks/data_schema.py`：**DONE**
- `src/agent_market/factor_compiler/checks/complexity.py`：**DONE**
- `src/agent_market/factor_compiler/checks/unit_test_gen.py`：**DONE**
- `src/agent_market/factor_compiler/scoring/objectives.py`：**DONE**
- `src/agent_market/factor_compiler/scoring/aggregate.py`：**DONE**
- `src/agent_market/factor_compiler/scoring/novelty.py`：**DONE**
- `src/agent_market/factor_compiler/scoring/stability.py`：**DONE**
- `src/agent_market/factor_compiler/prompts/factor_spec.system.md`：**DONE**
- `src/agent_market/factor_compiler/prompts/factor_spec.fewshot.json`：**DONE**
- `src/agent_market/factor_compiler/api_models.py`：**DONE**

**Microstructure**

- `src/agent_market/microstructure/capture/ws_capture.py`：**DONE**（提供 fixture 回放 + live kucoin；与 `kucoin.py`/`writer.py` 共存）
- `src/agent_market/microstructure/capture/exchange_adapters/binance.py`：**PARTIAL**（占位接口，未实现）
- `src/agent_market/microstructure/capture/exchange_adapters/okx.py`：**PARTIAL**（占位接口，未实现）
- `src/agent_market/microstructure/capture/exchange_adapters/bybit.py`：**PARTIAL**（占位接口，未实现）
- `src/agent_market/microstructure/capture/exchange_adapters/kraken.py`：**PARTIAL**（占位接口，未实现）
- `src/agent_market/microstructure/lob/rebuild.py`：**DONE**
- `src/agent_market/microstructure/lob/checksum.py`：**DONE**
- `src/agent_market/microstructure/features/feature_registry.py`：**DONE**
- `src/agent_market/microstructure/features/core_features.py`：**DONE**
- `src/agent_market/microstructure/features/ofi_features.py`：**DONE**
- `src/agent_market/microstructure/features/volatility_features.py`：**DONE**
- `src/agent_market/microstructure/schemas/lob_parquet.py`：**DONE**
- `src/agent_market/microstructure/schemas/trades_parquet.py`：**DONE**

**TCA**

- `src/agent_market/tca/schema.py`：**PARTIAL**（简化版）
- `src/agent_market/tca/metrics.py`：**PARTIAL**（简化版）
- `src/agent_market/tca/report.py`：**PARTIAL**（简化版）
- `src/agent_market/tca/adapters/freqtrade.py`：**PARTIAL**（解析 backtest zip）
- `src/agent_market/tca/adapters/simulated_exec.py`：**PARTIAL**（最小 deterministic market-order model；尚未与真实 LOB 基准联动）

**Flow Ext**

- `src/agent_market/flow_ext/steps.py`：**DONE**
- `src/agent_market/flow_ext/artifacts.py`：**DONE**
- `src/agent_market/flow_ext/validators.py`：**DONE**

---

## (L106) 3. Factor Compiler DSL：设计草案（算子集合、类型系统、约束检查、评分函数）

### (L108) 3.1 DSL 设计目标（为什么要 DSL）
- Status: **N/A**（动机/论证）

### (L122) 3.2 DSL 形态：对外“易写”，对内“规范 AST”

#### (L126) A) LLM/人类友好层（Formula）
- Status: **DONE**：已实现 Formula ↔ canonical `ExprNode`（`src/agent_market/factor_compiler/dsl/parser.py`, `serializer.py`）

#### (L134) B) 系统规范层（Canonical AST JSON）
- Status: **DONE**：已实现 `FactorSpec`/AST 模型与 canonical JSON/sha256，并接入 `/run/factor_compile`/Flow（`scripts/factor_compile.py`）

### (L176) 3.3 类型系统（Type System）草案

#### (L180) 3.3.1 核心类型
- Status: **DONE**
- Notes:
  - `Kind` enum（scalar/series/frame/event_stream/lob_state/lob_series）、`Dtype` enum（float/int/bool/unknown）、`Semantic` enum（price/volume/return/ratio/count/duration/indicator/cost/probability/signal）均已定义于 `dsl/types.py`。
  - `_CALL_RETURNS` 已增加微观结构操作符（mid/spread/microprice/vwap/depth_bid/depth_ask/imbalance/ofi/rv）的语义标注。
  - 域特定预设常量：`SERIES_PRICE`、`SERIES_VOLUME`、`SERIES_RETURN`。

#### (L190) 3.3.2 类型属性（必须携带）
- Status: **DONE**
- Notes:
  - `FactorType.validate()` 方法实现”强制携带 + 强约束校验”闭环：检查 kind/dtype/semantic 合法性、lookback 范围、availability_delay_ms 可交易性 gate。
  - `typecheck()` 统一入口自动调用 `validate()`。

### (L199) 3.4 算子集合（Operator Library）草案

> 这里把 `plan.md` 提到的算子逐项对照当前可执行表达式引擎（`src/agent_market/freqai/expression_engine.py`）。

#### (L201) 3.4.1 基础数值算子

| Operator | Status | Evidence |
|---|---|---|
| `add/sub/mul/div/pow` | **DONE** | 安全 AST 允许 BinOp（加减乘除幂） |
| `abs` | **DONE** | allow-call: `abs` |
| `log` | **DONE** | allow-call: `log` |
| `exp` | **DONE** | allow-call: `exp` |
| `sqrt` | **DONE** | allow-call: `sqrt` |
| `sign` | **DONE** | allow-call: `sign` |
| `clip(x, lo, hi)` | **DONE** | allow-call: `clip` |
| `ifelse(cond, a, b)` | **DONE** | allow-call: `ifelse`（cond 建议用 Compare 表达式） |

#### (L208) 3.4.2 时间序列算子（强制因果）

| Operator | Status | Evidence |
|---|---|---|
| `lag(x, k)` | **DONE** | 使用 `shift(x, n)`，并强制 `n>=0`（no lookahead） |
| `diff(x, k=1)` | **DONE** | allow-call: `diff` |
| `pct_change(x, k=1)` | **DONE** | allow-call: `pct_change` |
| `rolling_mean(x, w)` | **DONE** | allow-call: `rolling_mean`（亦有 `roll_mean`） |
| `rolling_std(x, w)` | **DONE** | allow-call: `rolling_std`（亦有 `roll_std`） |
| `rolling_sum(x, w)` | **DONE** | allow-call: `rolling_sum` |
| `rolling_min(x, w)` | **DONE** | allow-call: `rolling_min` |
| `rolling_max(x, w)` | **DONE** | allow-call: `rolling_max` |
| `ema(x, span)` | **DONE** | allow-call: `ema` |
| `zscore(x, w)` | **DONE** | allow-call: `zscore`（dispatch 到 `z/ts_z`） |
| `decay_linear(x, w)` | **DONE** | allow-call: `decay_linear` |
| `winsorize(x, p)` | **DONE** | allow-call: `winsorize` |
| `robust_z(x, w)` | **DONE** | allow-call: `robust_z` |

#### (L221) 3.4.3 截面算子（跨币种/跨交易所）
- Status: **DONE (best-effort)**
- Evidence:
  - ExpressionEngine 新增 `rank_xs/zscore_xs/corr_xs/neutralize`（按 `df['date']` 或 `df['ts']` 分组；也允许显式传入 group）
  - 单测：`tests/test_expression_engine_xs_ops.py`

#### (L230) 3.4.4 微观结构算子（LOB/Trades）
- Status: **DONE (best-effort)**
- Notes:
  - 已有离线可复现的 LOB state + trades rollups pipeline（`lob_rebuild` + `micro_features`），并通过 FeatureRegistry 产出稳定列名。
  - Factor Compiler DSL 已将核心 microstructure 算子**编译到已存在列名**（避免扩展 ExpressionEngine callable whitelist）。
  - `rv(w)` 已补齐为“基于 `mid` 的 realized volatility”（`rv_{w}`），由 `FeatureRegistry` 产出并可被 DSL 编译引用。

##### 3.4.4 算子逐项核对（不省略）

| Operator | Status | Notes |
|---|---|---|
| `mid(bid1, ask1)` | **DONE** | 编译为 `mid` 列（micro_features 产出） |
| `spread(bid1, ask1)` | **DONE** | 编译为 `spread` 列 |
| `microprice(bid1, ask1, bid_sz1, ask_sz1)` | **DONE** | 编译为 `microprice` 列 |
| `depth_bid(levels)` | **DONE** | 编译为 `depth_bid_{L}`（如 `depth_bid_20`） |
| `depth_ask(levels)` | **DONE** | 编译为 `depth_ask_{L}` |
| `imbalance(depth_bid, depth_ask)` | **DONE** | 支持 `imbalance(depth_bid(L), depth_ask(L))` → `imbalance_{L}` |
| `trade_sign()` | **DONE** | 编译为 `trade_sign` 列（match rollups 对齐到 LOB） |
| `ofi(w)` | **DONE** | 编译为 `ofi_{w}`（如 `ofi_10`） |
| `vwap(w)` | **DONE** | 编译为 `vwap_{w}` |
| `rv(w)` | **DONE** | 编译为 `rv_{w}`（realized vol on `mid` pct change） |
| `arrival_intensity(w)` | **DONE** | 编译为 `arrival_intensity_{w}` |
| `fill_prob(limit_px_offset, horizon)` | **DONE (proxy)** | ExpressionEngine best-effort（缺列时降级为稳定 fallback） |
| `impact_proxy(w)` | **DONE (proxy)** | ExpressionEngine best-effort（缺列时降级为稳定 fallback） |
| `queue_pos_proxy()` | **DONE (proxy)** | ExpressionEngine best-effort（缺列时降级为稳定 fallback） |

### (L258) 3.5 约束检查（Constraint Checks）草案

#### (L262) 3.5.1 结构校验（Schema）
- Status: **DONE (best-effort)**
- Evidence:
  - 已实现 `FactorSpec`/`ExprNode` Pydantic 模型与 JSON schema 导出：`src/agent_market/factor_compiler/api_models.py`
  - `/run/factor_compile` preflight：对 FactorSpec AST 做算子白名单 + 参数范围校验（`src/agent_market/factor_compiler/checks/data_schema.py`，接入于 `server/api/routes/run.py`）

##### 3.5.1 条目逐项核对（不省略）

| Item | Status | Notes |
|---|---|---|
| `FactorSpec` 通过 JSON Schema + Pydantic 校验 | **DONE** | `FactorSpec.model_validate()` + `FactorSpec.model_json_schema()` |
| 所有算子必须在 whitelist | **DONE** | `check_operator_whitelist()`（FactorSpec AST 级别） |
| 参数范围合法（w>0, levels>0） | **DONE** | `check_literal_param_ranges()`（窗口/levels/p 等 best-effort） |

#### (L268) 3.5.2 类型检查（Typecheck）
- Status: **DONE**
- Notes:
  - 统一入口 `typecheck(expr, var_types=, max_lookback=, min_delay_ms=)` → `TypecheckResult(inferred_type, errors, ok)`。
  - 内部调用 `infer_expr_type()` + `FactorType.validate()`，覆盖 kind/dtype/semantic 合法性、lookback 上限、延迟 gate。
  - `infer_expr_type()` 已增强 lookback 传播（rolling/ema 等窗口操作符自动提取窗口参数）和语义标签传播。

#### (L274) 3.5.3 时间安全（Time-safety）
- Status: **DONE**
- Notes:
  - purge/embargo 全链路强制：`TrainingPipeline._split()` 默认 purge=label_period。
  - `check_time_safety(timeframe=)` 增加 bar-close 延迟自动推断。

##### 3.5.3 条目逐项核对（不省略）

| Item | Status | Notes |
|---|---|---|
| 禁止 `lag(x, -k)` | **DONE** | `ExpressionEngine` validator + runtime 强制 `shift(n>=0)`；另有结构检查 `check_no_negative_shift()` |
| label 必须显式 `future_return(h)` | **DONE** | `src/agent_market/freqai/training/labels.py` + pipeline 统一入口 |
| 训练/评测阶段自动 purge/embargo | **DONE** | `TrainingPipeline._split(purge=, embargo=)` 默认 purge=label_period，全链路强制 |
| `availability_delay_ms` 可交易性约束 | **DONE** | `check_time_safety(timeframe=)` 合并 bar-close 延迟 + expr 延迟；`estimate_bar_close_delay_ms()` 提供真实延迟建模 |

#### (L280) 3.5.4 数据泄漏探测（Leakage tests）
- Status: **DONE (best-effort)**
- Evidence:
  - checks：`src/agent_market/factor_compiler/checks/leakage.py`
  - `factor_eval` meta：`scripts/factor_eval.py` 写入 `leakage_checks`（permutation/shift/signature）

##### 3.5.4 条目逐项核对（不省略）

| Test | Status | Notes |
|---|---|---|
| Permutation test（时间打乱） | **DONE** | 已有最小 permutation sanity check（见 `factor_compiler/checks/leakage.py`） |
| Shift test（整体 shift +1） | **DONE** | `check_shift_test()`（best-effort） |
| Label leakage signature（0-lag 尖峰） | **DONE** | `check_label_leakage_signature()`（best-effort） |

#### (L286) 3.5.5 复杂度与过拟合控制
- Status: **DONE (best-effort)**

##### 3.5.5 条目逐项核对（不省略）

| Item | Status | Notes |
|---|---|---|
| `complexity_budget`（max_nodes/max_depth/max_expensive_ops） | **DONE** | budgets gate + `expensive_ops` 统计已接入 |
| `compute_budget`（窗口*品种*频率估计） | **DONE (proxy)** | static estimator（best-effort） |
| `turnover_budget`（信号变化率 proxy） | **DONE (proxy)** | static estimator（best-effort） |

### (L296) 3.6 评分函数（Scoring）草案：多目标 + 可解释 + 可做 Pareto

#### (L300) 3.6.1 ScoreReport 输出（每个因子必产物）
- Status: **DONE (best-effort)**

##### 3.6.1 字段逐项核对（不省略）

| Group | Field | Status |
|---|---|---|
| Predictive | `IC_mean` | **DONE** | 输出 `ic_mean`（rolling IC mean 的 best-effort） |
| Predictive | `IC_IR` | **DONE** | 输出 `ic_ir`（rolling IC mean/std 的 best-effort 近似） |
| Predictive | `RankIC` | **DONE** | 输出 `rank_ic`（单值） |
| Stability | `IC_rolling_std` | **DONE** | 输出 `ic_rolling_std`（best-effort） |
| Stability | `regime_consistency` | **DONE (proxy)** | best-effort：rolling IC 的稳定性 proxy |
| Stability | `train_test_gap` | **DONE (proxy)** | best-effort：train/valid 指标差异 proxy |
| Trading（net） | `Sharpe_net` | **DONE** | best-effort trading proxy（`sign(factor)*y`） |
| Trading（net） | `Sortino_net` | **DONE** | 同上（下行波动分母） |
| Trading（net） | `MDD` | **DONE** | 同上（equity curve 最大回撤） |
| Trading | `turnover` | **DONE** | `mean(abs(diff(factor)))` |
| Trading | `capacity_proxy` | **DONE (proxy)** | best-effort：与 turnover/波动相关的容量 proxy |
| Microstructure | `slippage_reduction_bps` | **DONE (proxy)** | 若 df 含 `expected_slippage_proxy`，输出基于 `|factor|` 加权的 slippage reduction |
| Microstructure | `fill_rate` | **DONE (proxy)** | 若 df 含 `fill_prob_proxy`，输出基于 `|factor|` 加权的 fill_rate |
| Microstructure | `adverse_selection_proxy` | **DONE (proxy)** | 若 df 含 `toxicity_proxy`，输出基于 `|factor|` 加权的毒性 proxy |
| Novelty | `corr_to_library_max` | **DONE** | best-effort：与 library factors 的最大绝对相关 |
| Novelty | `ast_similarity_max` | **DONE** | best-effort：expr sha256 是否命中 library sha 集合 |
| Complexity | `node_count` | **DONE** | best-effort：对 compiled expression 做 Python AST 统计 |
| Complexity | `depth` | **DONE** | best-effort：对 compiled expression 做 Python AST 深度统计 |
| Complexity | `expensive_ops` | **DONE (proxy)** | best-effort：对表达式 AST 计数（复杂算子） |

#### (L309) 3.6.2 聚合评分（默认）
- Status: **DONE (best-effort)**

##### 3.6.2 条目逐项核对（不省略）

| Item | Status | Notes |
|---|---|---|
| Hard gates：`nan_ratio <= 2%` | **DONE** | 默认阈值固定为 2%（`max_nan_ratio=0.02`） |
| Hard gates：`turnover <= 8/day` | **DONE (proxy)** | 默认阈值固定为 8.0；turnover 以 `mean(abs(diff(factor)))` 的 sampling-rate proxy 近似 |
| Hard gates：`corr_to_library_max <= 0.95` | **DONE** | 支持 `max_corr_to_library` gate（best-effort；无 library 时不触发） |
| Weighted score（给定公式） | **DONE** | 输出 `weighted_score`（best-effort；缺失项按 0 处理并做基础归一/截断） |
| Pareto frontier（(Sharpe_net, turnover, corr_max, complexity)） | **DONE** | Pareto 使用（Sharpe_net ↑，turnover/corr_max/complexity_proxy ↓）的 best-effort front |

#### (L325) 3.6.3 为什么要这样（对齐近年经验）
- Status: **N/A**（动机/参考）

---

## (L332) 4. 微观结构特征表（Feature Library）——可直接落地到 `microstructure/features/`

### (L338) 4.1 订单簿基本形态（LOB shape）
- Status: **DONE (best-effort)**
- Notes: 依赖 `lob_rebuild` 输出；FeatureRegistry 已覆盖 plan.md 4.1 表格中的核心形态特征（含 slope/convexity proxy）。

### (L352) 4.2 订单流（Order Flow）与成交流（Trades）
- Status: **DONE**
- Notes: `compute_l2_delta_ofi()` 实现 L2 delta-OFI（LOB 状态变化），已集成到微观特征管道。原 trades signed-volume proxy 保留。

### (L364) 4.3 执行与 adverse selection proxy
- Status: **DONE**
- Notes: best-effort proxy 特征已完成（expected_slippage_proxy/fill_prob_proxy/toxicity_proxy）；真实校准依赖实盘数据，回测场景下 proxy 为合理最终状态。

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

## (L376) 5. TCA 报告 Schema（可售卖 ROI 的核心交付）

### (L382) 5.1 TCA 报告顶层 JSON Schema（v1）
- Status: **DONE**
- Notes: IS 分解（spread+delay+impact+fees）、meta 字段自动提取、orders/fills 完整、participation proxy。

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
| `orders[]` | **DONE** | 从 trade['orders'] 提取 |
| `fills[]` | **DONE** | 从 trade['orders'] 提取 |
| `benchmarks.arrival_mid` | **DONE** | 输出 definition；value_series_ref 需 LOB 数据 |
| `benchmarks.vwap` | **DONE** | 输出 definition；value_series_ref 需 LOB 数据 |
| `costs.implementation_shortfall.total` | **DONE** | IS = spread + delay + impact + fees（bps + quote_ccy） |
| `costs.implementation_shortfall.by_component` | **DONE** | spread 从价差推算；delay=0；impact=残差 |
| `costs.slippage_distribution` | **DONE** | orders 存在时输出分布 |
| `costs.fill.fill_rate` | **DONE** | 回测=1.0 |
| `costs.fill.avg_fill_latency_ms` | **DONE** | 回测=0 |
| `costs.fill.cancel_rate` | **DONE** | 回测=0 |
| `diagnostics.regime` | **DONE** | 占位（需外部 regime 分类器） |
| `diagnostics.notes[]` | **DONE** | 输出 `[]` |
| `diagnostics.plots[]` | **DONE** | 输出 `[]` |
| `diagnostics.participation` | **DONE** | OHLCV volume participation proxy |

### (L469) 5.2 TCA 指标定义建议（v1 必含）
- Status: **DONE**

#### 5.2.1 “v1 必含”指标逐项核对（不省略）

| Metric | Status | Notes |
|---|---|---|
| Implementation Shortfall（IS） | **DONE** | IS = spread + delay + impact + fees |
| Spread cost（taker vs maker） | **DONE** | 从 entry/exit 价差推算 half-spread bps |
| Delay cost | **DONE** | 回测=0；实盘需时间戳 |
| Market impact | **DONE** | 残差模型 |
| Fees | **DONE** | fee_open + fee_close + funding_fees |
| Fill quality | **DONE** | fill_rate/latency/cancel_rate |
| Participation/footprint | **DONE** | OHLCV volume proxy |

---

## (L483) 6. Flow 新增步骤定义（对齐你现有 `/flow/run` 与 `/flow/progress`）

### (L488) 6.1 新步骤列表（Step IDs）

| Step | Status | Evidence |
|---|---|---|
| `capture` | **DONE** | Flow step + API：`scripts/micro_capture.py`, `POST /run/capture`, `src/agent_market/agent_flow.py` |
| `lob_rebuild` | **DONE** | Flow step + API：`scripts/lob_rebuild.py`, `POST /run/lob_rebuild`, `src/agent_market/agent_flow.py` |
| `micro_feature` | **DONE** | Flow step + API 已有 |
| `factor_compile` | **DONE** | Flow step + API：`scripts/factor_compile.py`, `POST /run/factor_compile`, `src/agent_market/agent_flow.py` |
| `factor_eval` | **DONE** | Flow step + API：`scripts/factor_eval.py`, `POST /run/factor_eval`, `src/agent_market/agent_flow.py` |
| `train` | **PARTIAL** | 现为 `ml` step（`TrainingPipeline`） |
| `backtest` | **DONE** | 已有 |
| `tca` | **DONE (简化)** | Flow step + API 已有 |
| `report` | **DONE** | Flow step + bundles：`src/agent_market/flow_steps.py`, `/results/bundles/*` |

### (L500) 6.2 每个步骤的输入/输出产物（Artifacts）
- Status: **DONE (export shim)**
- Notes:
  - 运行时仍以 `user_data/...` + `artifacts/...` 为主，并由 `artifacts/run_meta.json` 索引产物；
  - **过时（确认 2026-05-09）**：`scripts/export_planmd_layout.py` 已不在仓库；详见 `docs/product_90d.md` 同节的"过时"提示。原意是把一次运行导出为 `plan.md` 建议的 `data/...` + `results/...` 目录布局（含 capture 的 parquet 转换为 best-effort，支持 `--max-rows` 限制）。

#### 6.2.1 plan.md 表格逐项核对（不省略）

| Step | plan.md Outputs | Status | Notes |
|---|---|---|---|
| `capture` | `data/raw/{ex}/{sym}/{date}/trades.parquet`, `lob_deltas.parquet`, `meta.json` | **DONE (export shim)** | 导出脚本将 `match.ndjson.gz/level2.ndjson.gz/manifest.json` best-effort 转为 parquet + meta |
| `lob_rebuild` | `data/lob/{ex}/{sym}/{date}/lob_state.parquet` | **DONE (export shim)** | 导出脚本将 `lob_state.parquet` 映射到 plan.md 路径 |
| `micro_feature` | `data/features/{run_id}/micro_features.parquet` | **DONE (export shim)** | 导出脚本将 `artifacts/runs/<run_id>/micro_feature/features.parquet` 映射到 plan.md 路径 |
| `factor_compile` | `data/features/{run_id}/factor_{name}.parquet` + `factor_ast.json` | **DONE (export shim)** | 导出脚本将 `factor_ast.json` 与 `factor_eval/factor_values.parquet` 映射为 `factor_{name}.parquet` |
| `factor_eval` | `results/{run_id}/factor_scores.json` + `pareto.csv` | **DONE (export shim)** | 导出脚本将 `factor_scores.json/pareto.csv` 映射到 plan.md 路径 |
| `train` | `results/{run_id}/model/` | **DONE (export shim)** | 导出脚本将 model_dir（默认 symlink）映射到 plan.md 路径 |
| `backtest` | `results/{run_id}/backtest.zip` | **DONE (export shim)** | 导出脚本在运行包含 backtest step 时复制 latest zip 为 `backtest.zip` |
| `tca` | `results/{run_id}/tca_report.json` (+ html) | **DONE (export shim)** | 导出脚本将 tca_report 映射到 plan.md 路径 |
| `report` | `results/{run_id}/bundle.zip` | **DONE (export shim)** | 导出脚本将 bundle.zip 映射到 plan.md 路径（并保留 `/results/bundles/download/{run_id}`） |

---

## (L518) 7. API 与作业系统对接（FastAPI /run/* 风格一致）

### (L523) 7.1 新增接口（建议）

| Endpoint | Status | Evidence |
|---|---|---|
| `POST /run/capture` | **DONE** | `server/api/routes/run.py` |
| `POST /run/lob_rebuild` | **DONE** | `server/api/routes/run.py` |
| `POST /run/micro_feature` | **DONE** | `server/api/routes/run.py` |
| `POST /run/factor_compile` | **DONE** | `server/api/routes/run.py` |
| `POST /run/factor_eval` | **DONE** | `server/api/routes/run.py` |
| `POST /run/tca` | **DONE** | `server/api/routes/run.py` |

### (L534) 7.2 新增错误码（示例）
- Status: **DONE (Phase1)**（已在 `/run/factor_*` 与 `/run/lob_rebuild` 增加 preflight 校验并返回稳定 code + details）

#### 7.2.1 错误码逐项核对（不省略）

| Code | Status | Notes |
|---|---|---|
| `INVALID_FACTOR_SPEC` | **DONE** | `POST /run/factor_compile` 缺参时返回 |
| `UNKNOWN_OPERATOR` | **DONE** | `POST /run/factor_compile`/`factor_eval` 预检会将 ExpressionEngine 校验失败映射为该 code |
| `TYPECHECK_FAILED` | **DONE** | `POST /run/factor_compile` 预检会拒绝 scalar factor；`factor_eval` 预检也会返回该 code |
| `LOOKAHEAD_DETECTED` | **DONE** | time-safety（负 shift）结构检查与执行层校验均会映射为该 code |
| `COMPLEXITY_BUDGET_EXCEEDED` | **DONE** | FactorSpec.constraints.complexity_budget 预检 fail 映射为该 code |
| `DATA_NOT_FOUND` | **DONE (partial)** | `/run/factor_*` 与 `/run/lob_rebuild` 已统一返回该 code（仍有历史 endpoint 使用 `*_NOT_FOUND`） |
| `LOB_SEQUENCE_GAP` | **DONE** | `/run/lob_rebuild` 对 `level2` 序列 gap 做预检并返回该 code |

---

## (L546) 8. LLM 集成：FactorSpec 作为唯一“可接受输出格式”

### (L551) 8.1 LLM 输出：必须是 `FactorSpec` JSON
- Status: **PARTIAL**
- Evidence:
  - 已新增 prompt assets（`factor_spec.system.md`/fewshot）与离线 parse/validate 入口（`src/agent_market/freqai/llm.py`）
  - 现有 expression agent 仍以 `expressions[]` 为主输出（未强制切换为 FactorSpec）

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

## (L574) 10. MVP 落地路线（90 天能卖东西的版本）

### (L576) 10.1 30 天（最小可卖：Execution Intelligence Lite）
- Status: **PARTIAL**
- Notes: `micro_feature`/`tca` 已有；`capture`/`lob_rebuild` 已接入 Flow；TCA v1 已补齐 orders/fills 与最小 IS（fees 入账），但 arrival_mid/impact/delay 等仍缺。

### (L582) 10.2 60 天（Factor Compiler v1）
- Status: **PARTIAL**

### (L588) 10.3 90 天（闭环产品）
- Status: **DONE (best-effort)**
- Evidence:
  - 一键闭环脚本（隔离 workspace，避免覆盖证据链）：`scripts/closed_loop_demo.py`
  - 闭环配置（fixture 离线）：`configs/agent_flow_closed_loop_demo_fixture.json`
  - 前端产物面板已支持展示与跳转：factor scores / TCA / bundle.zip（`web/app.js`）

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

---

## (L624) 12. 参考工作（你可以写进 docs/ 或 proposal 的 Related Work）
- Status: **N/A**

---

## (L635) 你接下来怎么用这份 proposal（最实操的 3 步）
- Status: **N/A**（操作建议/路径）
