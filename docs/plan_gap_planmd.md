# Plan Gap Audit（`plan.md` vs 当前仓库实现）

Generated: 2026-02-05

> 目标：对 `plan.md` 的**全部章节（不省略）**逐一查明：已实现 / 部分实现 / 缺失 / 仅叙述（N/A），并给出证据与对应的 `docs/mohu.md` backlog 归档方向。

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
- Status: **PARTIAL**（已新增最小 `FactorType`/`infer_expr_type`；尚缺完整 Type System/规则与更细语义类型）

#### (L190) 3.3.2 类型属性（必须携带）
- Status: **PARTIAL**
- Notes:
  - 最小 `FactorType` 已携带 `freq/timezone/availability_delay_ms/lookback` 字段，但尚未形成“强制携带 + 强约束校验”的闭环。

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
- Status: **PARTIAL**
- Notes:
  - 已有离线可复现的 LOB state + trades rollups pipeline（`lob_rebuild` + `micro_features`），并通过 FeatureRegistry 产出稳定列名。
  - Factor Compiler DSL 已将核心 microstructure 算子**编译到已存在列名**（避免扩展 ExpressionEngine callable whitelist）。
  - 仍缺：`rv(w)` 的 mid/trades 版本（当前 `rv_*` 主要来自 OHLCV proxy）。

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
| `rv(w)` | **PARTIAL** | 目前仅 OHLCV rolling vol（`rv_12/24/72`），非 mid/trades |
| `arrival_intensity(w)` | **DONE** | 编译为 `arrival_intensity_{w}` |
| `fill_prob(limit_px_offset, horizon)` | **DONE (proxy)** | ExpressionEngine best-effort（缺列时降级为稳定 fallback） |
| `impact_proxy(w)` | **DONE (proxy)** | ExpressionEngine best-effort（缺列时降级为稳定 fallback） |
| `queue_pos_proxy()` | **DONE (proxy)** | ExpressionEngine best-effort（缺列时降级为稳定 fallback） |

### (L258) 3.5 约束检查（Constraint Checks）草案

#### (L262) 3.5.1 结构校验（Schema）
- Status: **PARTIAL**
- Evidence:
  - 已实现 `FactorSpec`/`ExprNode` Pydantic 模型与 JSON schema 导出：`src/agent_market/factor_compiler/api_models.py`

##### 3.5.1 条目逐项核对（不省略）

| Item | Status | Notes |
|---|---|---|
| `FactorSpec` 通过 JSON Schema + Pydantic 校验 | **DONE** | `FactorSpec.model_validate()` + `FactorSpec.model_json_schema()` |
| 所有算子必须在 whitelist | **PARTIAL** | 表达式层有 whitelist（callable names），但无 FactorSpec 层 |
| 参数范围合法（w>0, levels>0） | **PARTIAL** | 部分算子 int() 强转但不做范围校验 |

#### (L268) 3.5.2 类型检查（Typecheck）
- Status: **PARTIAL**（已有最小 type inference + API preflight；尚缺完整 Type System + 规则库）

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

#### (L280) 3.5.4 数据泄漏探测（Leakage tests）
- Status: **PARTIAL**

##### 3.5.4 条目逐项核对（不省略）

| Test | Status | Notes |
|---|---|---|
| Permutation test（时间打乱） | **DONE** | 已有最小 permutation sanity check（见 `factor_compiler/checks/leakage.py`） |
| Shift test（整体 shift +1） | **DONE** | `check_shift_test()`（best-effort） |
| Label leakage signature（0-lag 尖峰） | **DONE** | `check_label_leakage_signature()`（best-effort） |

#### (L286) 3.5.5 复杂度与过拟合控制
- Status: **PARTIAL**

##### 3.5.5 条目逐项核对（不省略）

| Item | Status | Notes |
|---|---|---|
| `complexity_budget`（max_nodes/max_depth/max_expensive_ops） | **DONE** | budgets gate + `expensive_ops` 统计已接入 |
| `compute_budget`（窗口*品种*频率估计） | **DONE (proxy)** | static estimator（best-effort） |
| `turnover_budget`（信号变化率 proxy） | **DONE (proxy)** | static estimator（best-effort） |

### (L296) 3.6 评分函数（Scoring）草案：多目标 + 可解释 + 可做 Pareto

#### (L300) 3.6.1 ScoreReport 输出（每个因子必产物）
- Status: **PARTIAL**

##### 3.6.1 字段逐项核对（不省略）

| Group | Field | Status |
|---|---|---|
| Predictive | `IC_mean` | **PARTIAL** | 目前以全样本相关（单值）近似 `IC_mean` |
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
| Microstructure | `slippage_reduction_bps` | **PARTIAL** | 字段已补齐（占位 `null`） |
| Microstructure | `fill_rate` | **PARTIAL** | 字段已补齐（占位 `null`） |
| Microstructure | `adverse_selection_proxy` | **PARTIAL** | 字段已补齐（占位 `null`） |
| Novelty | `corr_to_library_max` | **DONE** | best-effort：与 library factors 的最大绝对相关 |
| Novelty | `ast_similarity_max` | **DONE** | best-effort：expr sha256 是否命中 library sha 集合 |
| Complexity | `node_count` | **DONE** | best-effort：对 compiled expression 做 Python AST 统计 |
| Complexity | `depth` | **DONE** | best-effort：对 compiled expression 做 Python AST 深度统计 |
| Complexity | `expensive_ops` | **DONE (proxy)** | best-effort：对表达式 AST 计数（复杂算子） |

#### (L309) 3.6.2 聚合评分（默认）
- Status: **PARTIAL**

##### 3.6.2 条目逐项核对（不省略）

| Item | Status | Notes |
|---|---|---|
| Hard gates：`nan_ratio <= 2%` | **PARTIAL** | 已实现 nan_ratio gate（阈值可配置）；默认未固定为 2% |
| Hard gates：`turnover <= 8/day` | **PARTIAL** | 已实现 turnover gate（阈值可配置）；含义仍需与采样频率对齐 |
| Hard gates：`corr_to_library_max <= 0.95` | **DONE** | 支持 `max_corr_to_library` gate（best-effort；无 library 时不触发） |
| Weighted score（给定公式） | **DONE** | 输出 `weighted_score`（best-effort；缺失项按 0 处理并做基础归一/截断） |
| Pareto frontier（(Sharpe_net, turnover, corr_max, complexity)） | **PARTIAL** | 已实现简化 Pareto（IC_abs/RankIC_abs + turnover + nan_ratio），未含 Sharpe/corr/complexity |

#### (L325) 3.6.3 为什么要这样（对齐近年经验）
- Status: **N/A**（动机/参考）

---

## (L332) 4. 微观结构特征表（Feature Library）——可直接落地到 `microstructure/features/`

### (L338) 4.1 订单簿基本形态（LOB shape）
- Status: **PARTIAL**
- Notes: 依赖 `lob_rebuild` 输出；已实现最小 LOB 派生特征（mid/spread/rel_spread/microprice/depth/imbalance）。

### (L352) 4.2 订单流（Order Flow）与成交流（Trades）
- Status: **PARTIAL**
- Notes: 已实现从 KuCoin `match` 计算 trade_sign/vwap/ofi/arrival_intensity 并对齐到 LOB 时间戳；缺 buy/sell vol 等扩展。

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
| `rv_w` | **PARTIAL** | 目前仅 OHLCV rolling vol（`rv_12/24/72`），非 mid/trades 版本 |
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
- Status: **PARTIAL**
- Notes:
  - 计划建议路径为 `data/...` 与 `results/...`；当前仓库使用 `user_data/...` + `artifacts/...`，并由 `artifacts/run_meta.json` 索引产物。

#### 6.2.1 plan.md 表格逐项核对（不省略）

| Step | plan.md Outputs | Status | Notes |
|---|---|---|---|
| `capture` | `data/raw/{ex}/{sym}/{date}/trades.parquet`, `lob_deltas.parquet`, `meta.json` | **PARTIAL** | 当前输出 `*.ndjson.gz` + `manifest.json`（路径不同、格式不同） |
| `lob_rebuild` | `data/lob/{ex}/{sym}/{date}/lob_state.parquet` | **PARTIAL** | 当前输出 `out_dir/lob_state.parquet` + `rebuild_report.json`（路径/命名不同） |
| `micro_feature` | `data/features/{run_id}/micro_features.parquet` | **PARTIAL** | 当前为 `artifacts/runs/<run_id>/micro_feature/features.parquet` |
| `factor_compile` | `data/features/{run_id}/factor_{name}.parquet` + `factor_ast.json` | **PARTIAL** | 当前为 `artifacts/runs/<run_id>/factor_compile/*`（spec/ast/expression），未产出 factor_{name}.parquet |
| `factor_eval` | `results/{run_id}/factor_scores.json` + `pareto.csv` | **PARTIAL** | 当前为 `artifacts/runs/<run_id>/factor_eval/factor_scores.json` + `pareto.csv` |
| `train` | `results/{run_id}/model/` | **PARTIAL** | 当前模型产物在 `artifacts/models/...` |
| `backtest` | `results/{run_id}/backtest.zip` | **PARTIAL** | 当前为 `user_data/backtest_results/backtest-result-*.zip` |
| `tca` | `results/{run_id}/tca_report.json` (+ html) | **PARTIAL** | 当前为 `artifacts/runs/<run_id>/tca/tca_report.json` |
| `report` | `results/{run_id}/bundle.zip` | **PARTIAL** | 当前为 `artifacts/runs/<run_id>/bundle/bundle.zip`（并提供 `/results/bundles/download/{run_id}`） |

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
- Status: **MISSING**

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
