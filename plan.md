下面给你一份**“工程级、可落地（偏实现）”完整 Proposal**：把你现有的 Agent Market（FastAPI + Flow + LLM 表达式生成 + 特征工程 + ML/RL + 回测）升级为 **Factor Compiler（受约束 DSL） + 微观结构特征库 + TCA 报告 + Flow 新步骤** 的可售卖产品化方案。

> 我会严格对齐你仓库当前暴露出来的结构与接口（`configs/ scripts/ server/ src/agent_market/ web/`；以及 `/run/feature /run/expression /run/train /run/rl_train /run/backtest /flow/run ...` 这套作业/结果体系）。([GitHub][1])
> 注意：这份是“系统与接口级别的实现提案”，不是论文；不会承诺收益，但会把“可计量 ROI（执行成本/TCA）+ 可复现研究效率（因子编译与评测闭环）”作为赚钱抓手。

---

# Agent Market Pro 工程 Proposal（Factor Compiler DSL + 微观结构特征 + TCA + Flow 扩展）

## 1. 项目定位与目标

### 1.1 一句话

把 Agent Market 从“LLM 帮我写表达式并回测”升级为 **“可审计的因子编译器 + 可复现评测流水线 + 微观结构/执行智能（TCA）产品”**，从而更容易形成**可卖的 ROI**（节省 bps / 降低滑点 / 提升研究效率）。

### 1.2 现状基座（你已有）

* 目录：`configs/ scripts/ server/ src/agent_market/ tests/ web/`，已形成“工作台”结构。([GitHub][1])
* FastAPI 统一接口：`/run/feature /run/expression /run/train /run/rl_train /run/backtest /flow/run`，以及 `jobs/status/logs/cancel`、结果聚合/画廊等。([GitHub][1])
* 这意味着你已经具备 StrategyOps 的核心：**编排（Flow）+ 异步作业（jobs）+ 结果管理（results）**。([GitHub][1])

### 1.3 本 Proposal 增量要解决的“3 个硬问题”

1. **LLM 产物不可信** → 用 DSL 约束 + 静态检查 + 自动单测，变成“可编译、可审计”的因子程序。
2. **回测容易被喷（泄漏/成本/容量）** → 把时间安全与成本入账写成系统级评测规范，并固化到 pipeline。
3. **想赚真钱要有可卖 ROI** → 把微观结构数据/特征与 **TCA（交易成本归因）**作为产品化 wedge（机构最愿意付费的部分）。

---

## 2. 总体架构（与现有仓库对齐）

### 2.1 新增模块总览（建议的目录落位）

你现有核心在 `src/agent_market/`，我们新增 4 个子域（每个都可独立迭代、可插拔）：

```
src/agent_market/
  factor_compiler/          # Factor Compiler 核心（DSL/检查/编译/评分/提示词）
    dsl/
      ast.py
      grammar.py
      parser.py
      operators.py
      types.py
      serializer.py
    checks/
      time_safety.py
      leakage.py
      data_schema.py
      complexity.py
      unit_test_gen.py
    scoring/
      objectives.py
      aggregate.py
      novelty.py
      stability.py
    prompts/
      factor_spec.system.md
      factor_spec.fewshot.json
    api_models.py            # pydantic schemas（FactorSpec、ScoreReport）

  microstructure/            # 订单簿/成交采集、重建、特征库
    capture/
      ws_capture.py
      exchange_adapters/
        binance.py
        okx.py
        bybit.py
        kraken.py
    lob/
      rebuild.py             # snapshot+delta → LOB 状态序列
      checksum.py
    features/
      feature_registry.py
      core_features.py
      ofi_features.py
      volatility_features.py
    schemas/
      lob_parquet.py
      trades_parquet.py

  tca/                       # TCA 报告生成
    schema.py                # TCA JSON schema + pydantic
    metrics.py               # IS/Spread/Impact/Delay/Fill 等
    report.py                # 输出 JSON + HTML(可选)
    adapters/
      freqtrade.py
      simulated_exec.py

  flow_ext/                  # Flow 扩展：新增步骤定义、产物命名、依赖关系
    steps.py
    artifacts.py
    validators.py
```

同时对应新增：

* `configs/`：增加若干 JSON 配置模板（下面给样例）
* `scripts/`：增加可单步运行脚本（capture、compile、tca）
* `server/`：增加 `/run/factor_compile`、`/run/micro_feature`、`/run/tca` 等路由（或扩展现有 `/run/feature`）

> 你当前的 API 体系与 Flow 已经非常适合用“新增 job kind + 输出产物 zip + results 画廊”来接入新模块。([GitHub][1])

---

## 3. Factor Compiler DSL：设计草案（算子集合、类型系统、约束检查、评分函数）

### 3.1 DSL 设计目标（为什么要 DSL）

LLM 直接输出 Python 表达式的问题：

* 容易“未来函数/泄漏”
* 不可审计（你不知道它用到哪些字段、窗口、shift）
* 不可控复杂度（过拟合/拥挤/计算爆炸）

DSL 的目标是：
**“把因子定义变成一个受约束、可静态分析、可编译、可自动生成单测的程序”**。
这也是近几年 LLM 因子挖掘系统（如 FAMA、Chain-of-Alpha、AlphaAgent）共同走向的工程化方向：把公式/因子落到 AST/算子库、并围绕反馈闭环迭代。([ACL Anthology][2])

---

### 3.2 DSL 形态：对外“易写”，对内“规范 AST”

建议两层表示：

#### A) LLM/人类友好层（Formula）

示例（文本表达式）：

```
zscore(imbalance(depth_bid(20), depth_ask(20)), 120)
```

#### B) 系统规范层（Canonical AST JSON）

你内部只认 AST JSON（便于校验、去重、做原创性检查）：

```json
{
  "name": "lob_imbalance_z_120_l20",
  "version": "1.0",
  "hypothesis": "订单簿不平衡在短期内影响价格的微小漂移",
  "expr": {
    "op": "zscore",
    "args": [
      {
        "op": "imbalance",
        "args": [
          {"op": "depth_bid", "args": [20]},
          {"op": "depth_ask", "args": [20]}
        ]
      },
      120
    ]
  },
  "constraints": {
    "max_lookback": 2000,
    "min_delay_ms": 0,
    "max_turnover": 8.0,
    "complexity_budget": {"max_nodes": 40, "max_depth": 8}
  },
  "tests": [
    {"type": "no_lookahead", "horizon": 1},
    {"type": "nan_rate", "max_nan_ratio": 0.02}
  ],
  "meta": {
    "timeframe": "1s",
    "universe": ["BTC/USDT", "ETH/USDT"],
    "data_sources": ["trades", "lob_l2"]
  }
}
```

---

### 3.3 类型系统（Type System）草案

DSL 不是“所有东西都是 float”；必须显式表达“数据形态与时间对齐”。

#### 3.3.1 核心类型

* `Scalar[T]`：常量（int/float/bool）
* `Series[T]`：按时间索引的序列（float/int/bool）
* `Frame[T]`：多列时间序列（OHLCV、特征矩阵）
* `EventStream`：逐笔事件流（trades/updates）
* `LOBState`：某时刻的 L2 订单簿状态（含多档 depth）
* `LOBSeries`：LOBState 的时间序列（可重采样）
* `Price`、`Volume`、`Return`：语义子类型（便于约束检查）

#### 3.3.2 类型属性（必须携带）

* `freq`: 采样频率（1s/100ms/1m…）
* `timezone`: 统一 UTC
* `availability_delay_ms`: 数据到达延迟（用于线上/仿真与“可交易性”约束）
* `lookback`: 运算要求的最小历史长度（用于缺失与 warm-up）

---

### 3.4 算子集合（Operator Library）草案

#### 3.4.1 基础数值算子

* `add/sub/mul/div/pow`
* `abs/log/exp/sqrt/sign`
* `clip(x, lo, hi)`
* `ifelse(cond, a, b)`

#### 3.4.2 时间序列算子（强制因果）

* `lag(x, k)`：只能 k>=0（k=0 允许）
* `diff(x, k=1)`
* `pct_change(x, k=1)`
* `rolling_mean(x, w)`
* `rolling_std(x, w)`
* `rolling_sum/min/max(x, w)`
* `ema(x, span)`
* `zscore(x, w)`：等价 `(x-mean)/std`
* `decay_linear(x, w)`：线性衰减
* `winsorize(x, p)` / `robust_z(x, w)`

#### 3.4.3 截面算子（跨币种/跨交易所）

（对 crypto 你可以按“币种集合/交易所集合/行业标签(可选)”做 group）

* `rank_xs(x, group=None)`
* `zscore_xs(x, group=None)`
* `neutralize(x, against=[...])`（线性残差）
* `corr_xs(x, y)`（用于拥挤/相关性约束）

#### 3.4.4 微观结构算子（LOB/Trades）

**LOB 基础**

* `mid(bid1, ask1)`
* `spread(bid1, ask1)`
* `microprice(bid1, ask1, bid_sz1, ask_sz1)`
* `depth_bid(levels)` / `depth_ask(levels)`
* `imbalance(depth_bid, depth_ask)`（常用定义：`(bid-ask)/(bid+ask)`）

**订单流/成交流**

* `trade_sign()`：基于 tick rule 或 quote rule（实现细节在 microstructure 模块）
* `ofi(w)`：Order Flow Imbalance（按窗口聚合）
* `vwap(w)`：成交量加权价格
* `rv(w)`：realized volatility（高频收益平方和）
* `arrival_intensity(w)`：到达强度（trades 或 updates）

**执行/成本 proxy（用于 TCA/执行策略）**

* `fill_prob(limit_px_offset, horizon)`
* `impact_proxy(w)`（例如基于短期 mid 变动与主动成交量）
* `queue_pos_proxy()`（如果没有 L3，则用 L2+成交推断）

> 这里建议把算子库做成 **注册表**（`FeatureRegistry`），方便 marketplace/插件化。

---

### 3.5 约束检查（Constraint Checks）草案

Factor Compiler 的关键不是“能算”，而是**能证明它没在胡来**。

#### 3.5.1 结构校验（Schema）

* `FactorSpec`（上面的 JSON）必须通过 JSON Schema + Pydantic 校验
* 所有算子必须在 whitelist
* 参数范围必须合法（窗口 w>0，levels>0 等）

#### 3.5.2 类型检查（Typecheck）

* `rolling_mean` 只接受 `Series[float]`
* `depth_bid` 输出 `Series[float]`
* `microprice` 需要 price+size 输入

#### 3.5.3 时间安全（Time-safety）

* 禁止 `lag(x, -k)`（任何“未来 shift”）
* 任何目标（label）必须显式用 `future_return(h)` 且在训练/评测阶段自动 **purge/embargo**（避免 leakage）
* 如果算子用到了 `t` 时刻的 LOB/成交，必须在 `availability_delay_ms` 约束下可交易

#### 3.5.4 数据泄漏探测（Leakage tests）

* **Permutation test**：对输入做时间打乱，若因子仍强相关 → 可疑
* **Shift test**：对输入整体 shift +1 bar，若性能不降或升 → 可疑
* **Label leakage signature**：因子与 label 的相关在 0-lag 异常尖峰 → 可疑

#### 3.5.5 复杂度与过拟合控制

* `complexity_budget`: `max_nodes/max_depth/max_expensive_ops`
* `compute_budget`: 估计计算量（窗口*品种*频率）
* `turnover_budget`: 估计换手（用信号变化率 proxy）

> AlphaAgent 强调用 AST 结构做原创性与复杂度控制，你可以直接复用这个“AST 约束”的思想做工程化 guardrail。([arXiv][3])

---

### 3.6 评分函数（Scoring）草案：多目标 + 可解释 + 可做 Pareto

你不要单一指标（IC 或 Sharpe）——会被喷，也不利于变现。

#### 3.6.1 ScoreReport 输出（每个因子必产物）

* Predictive：`IC_mean / IC_IR / RankIC`
* Stability：`IC_rolling_std / regime_consistency / train_test_gap`
* Trading（成本后）：`Sharpe_net / Sortino_net / MDD / turnover / capacity_proxy`
* Microstructure/Execution（可选）：`slippage_reduction_bps / fill_rate / adverse_selection_proxy`
* Novelty：`corr_to_library_max / ast_similarity_max`
* Complexity：`node_count / depth / expensive_ops`

#### 3.6.2 聚合评分（默认）

建议用 **“加权+硬约束门槛+Pareto”** 三段式：

1. Hard gates（不过线直接淘汰）

   * `nan_ratio <= 2%`
   * `turnover <= 8/day`（可配置）
   * `corr_to_library_max <= 0.95`
2. Weighted score

   * `Score = 0.35*IC_IR + 0.25*Sharpe_net + 0.15*Stability - 0.15*TurnoverPenalty - 0.10*ComplexityPenalty`
3. Pareto frontier

   * 输出前沿集合：`(Sharpe_net, turnover, corr_max, complexity)`

#### 3.6.3 为什么要这样（对齐近年经验）

* FAMA/Chain-of-Alpha 这类系统本质都是“生成→评测→反馈→再生成”，没有工程化评分与约束就跑不稳。([ACL Anthology][2])
* StockBench 这种 contamination-free 的评测提醒你：别迷信“会说就会赚”，一定要把评测做严。([arXiv][4])

---

## 4. 微观结构特征表（Feature Library）——可直接落地到 `microstructure/features/`

下面给你一份“够用且能扩展”的 **微观结构特征表（v1）**。每个特征都标注：输入、公式、输出、典型用途、坑点。

> 说明：若你暂时只有 L2（聚合深度），先做 L2 版本；L3 后续再加（Kraken 有 level3）。([arXiv][5])

### 4.1 订单簿基本形态（LOB shape）

| Feature       | 输入                         | 公式/定义                                 | 输出            | 用途       | 坑点             |
| ------------- | -------------------------- | ------------------------------------- | ------------- | -------- | -------------- |
| `mid`         | bid1, ask1                 | (bid1+ask1)/2                         | Series[Price] | 基础价格     | 注意价格精度         |
| `spread`      | bid1, ask1                 | ask1-bid1                             | Series[Price] | 流动性      | 跨交易所不可直接比      |
| `rel_spread`  | bid1, ask1                 | (ask-bid)/mid                         | Series[float] | 可比性更强    | mid=0 保护       |
| `microprice`  | bid1, ask1, bidSz1, askSz1 | (ask*bidSz + bid*askSz)/(bidSz+askSz) | Series[Price] | 短期方向/不平衡 | size=0 保护      |
| `depth_bid_L` | L2 levels                  | sum_{i<=L} bidSz_i                    | Series[Vol]   | 供给强弱     | 需要统一档位定义       |
| `depth_ask_L` | L2 levels                  | sum_{i<=L} askSz_i                    | Series[Vol]   | 需求强弱     | 同上             |
| `imbalance_L` | depth_bid_L, depth_ask_L   | (bid-ask)/(bid+ask)                   | Series[float] | 经典预测/执行  | (bid+ask)=0 保护 |
| `slope_bid_L` | bidPx_i, bidSz_i           | 拟合 size~distance 的斜率                  | Series[float] | 盘形陡峭度    | 需稳健回归          |
| `convexity_L` | depth across levels        | 二阶形态指标                                | Series[float] | 流动性形态    | 噪声大要平滑         |

### 4.2 订单流（Order Flow）与成交流（Trades）

| Feature               | 输入                         | 定义                         | 输出            | 用途         | 坑点        |
| --------------------- | -------------------------- | -------------------------- | ------------- | ---------- | --------- |
| `trade_sign`          | trades                     | tick/quote rule            | Series[int]   | 主动买卖识别     | 加密可能有异常成交 |
| `buy_vol_w`           | trades                     | sum(size * 1[sign=+]) in w | Series[Vol]   | 买压         | 需时间窗对齐    |
| `sell_vol_w`          | trades                     | sum(size * 1[sign=-]) in w | Series[Vol]   | 卖压         | 同上        |
| `ofi_w`               | L2 deltas or trades+quotes | 常见 OFI 定义：ΔbidSz - ΔaskSz  | Series[float] | 微观结构 alpha | 需正确处理撤单   |
| `vwap_w`              | trades                     | sum(px*vol)/sum(vol)       | Series[Price] | 执行基准       | 空窗处理      |
| `rv_w`                | mid 或 last                 | sum(r_t^2) in w            | Series[float] | 波动状态       | 高频噪声要滤    |
| `arrival_intensity_w` | trades 或 book updates      | count(events)/w            | Series[float] | 市场活跃度      | 断线会污染     |

### 4.3 执行与 adverse selection proxy

| Feature                   | 输入                          | 定义        | 输出            | 用途      | 坑点      |
| ------------------------- | --------------------------- | --------- | ------------- | ------- | ------- |
| `expected_slippage_proxy` | spread, imbalance, rv       | 经验模型/回归   | Series[float] | 预估滑点    | 需分交易所校准 |
| `fill_prob_proxy`         | depth, imbalance, intensity | 经验模型      | Series[float] | 限价单成交概率 | 需要历史标注  |
| `toxicity_proxy`          | ofi, price response         | 订单流→价格冲击比 | Series[float] | 识别“有毒流” | 定义要固定   |

> 这张表在工程上对应：每个 feature 都是一个注册函数（输入 schema、输出列名、依赖数据源、默认窗口），可以被 Factor Compiler 调用，也可以直接被 `/run/micro_feature` 调用。

---

## 5. TCA 报告 Schema（可售卖 ROI 的核心交付）

TCA（Transaction Cost Analysis）要解决的问题不是“策略赚不赚”，而是：
**“执行环节到底亏在哪：点差？冲击？延迟？排队？滑点分布？”**
这在机构采购里非常硬核、可定价。

### 5.1 TCA 报告顶层 JSON Schema（v1）

建议输出一个 `tca_report.json`（同时可渲染 HTML）。

```json
{
  "schema_version": "1.0",
  "meta": {
    "run_id": "uuid",
    "generated_at": "2026-02-04T00:00:00Z",
    "exchange": "OKX",
    "market": "perp|spot",
    "symbols": ["BTC-USDT-SWAP"],
    "time_range": {"start": "...", "end": "..."},
    "timeframe": "1s",
    "data_sources": ["trades", "lob_l2"],
    "strategy": {
      "name": "exec_rule_v1",
      "params": {"max_participation": 0.05}
    }
  },

  "orders": [
    {
      "order_id": "string",
      "symbol": "BTC-USDT-SWAP",
      "side": "BUY|SELL",
      "type": "MKT|LMT",
      "qty": 1.23,
      "limit_px": 65000.0,
      "submit_ts": "...",
      "cancel_ts": null
    }
  ],

  "fills": [
    {
      "order_id": "string",
      "fill_id": "string",
      "symbol": "BTC-USDT-SWAP",
      "side": "BUY|SELL",
      "qty": 0.5,
      "price": 65010.0,
      "ts": "...",
      "fee": 0.8,
      "liquidity": "TAKER|MAKER"
    }
  ],

  "benchmarks": {
    "arrival_mid": {"definition": "mid at submit_ts", "value_series_ref": "..." },
    "vwap": {"definition": "VWAP over exec window", "value_series_ref": "..."}
  },

  "costs": {
    "implementation_shortfall": {
      "total": {"bps": 3.2, "quote_ccy": 123.4},
      "by_component": {
        "spread": {"bps": 1.1},
        "delay": {"bps": 0.4},
        "market_impact": {"bps": 1.3},
        "fees": {"bps": 0.4}
      }
    },
    "slippage_distribution": {
      "p50_bps": 1.2,
      "p90_bps": 6.5,
      "p99_bps": 15.0
    },
    "fill": {
      "fill_rate": 0.93,
      "avg_fill_latency_ms": 120,
      "cancel_rate": 0.05
    }
  },

  "diagnostics": {
    "regime": {"vol_bucket": "high", "liquidity_bucket": "low"},
    "notes": ["..."],
    "plots": [
      {"name": "slippage_hist", "path": "tca/slippage_hist.png"},
      {"name": "cost_breakdown", "path": "tca/cost_breakdown.png"}
    ]
  }
}
```

### 5.2 TCA 指标定义建议（v1 必含）

* **Implementation Shortfall（IS）**：相对 arrival mid 或 arrival quote 的实际成本
* **Spread cost**：吃掉点差的成本（taker vs maker 分解）
* **Delay cost**：从 decision 到 submit 的价格漂移
* **Market impact**：成交引发的 mid 变化（需要简化模型也可以）
* **Fees**：手续费与返佣
* **Fill quality**：成交率、成交延迟、撤单率
* **Participation / footprint**：成交量占比（容量与冲击 proxy）

> 你后续做 RL 执行时，TCA 这套 schema 就是“离线训练/线上监控”的统一 KPI。

---

## 6. Flow 新增步骤定义（对齐你现有 `/flow/run` 与 `/flow/progress`）

你现在 Flow 典型步骤是：`feature → expression → ml → backtest`（可加 portfolio/hyperopt/rl_train）。([GitHub][1])
我们新增微观结构/Factor Compiler/TCA 的步骤，但不破坏旧流程。

### 6.1 新步骤列表（Step IDs）

1. `capture`（可选）：WebSocket 采集 trades + L2 book（落盘）
2. `lob_rebuild`：snapshot+delta → 可查询的 LOBSeries（按 1s/100ms 重采样）
3. `micro_feature`：生成微观结构特征矩阵（写入 features 文件）
4. `factor_compile`：LLM 输出 FactorSpec → DSL AST → 编译为可执行特征
5. `factor_eval`：跑 ScoreReport（多目标评分 + gates + Pareto）
6. `train`（复用已有）：ML 训练
7. `backtest`（复用已有）：回测
8. `tca`：基于 backtest fills 或 simulated execution 输出 TCA 报告
9. `report`（可选）：汇总一份“因子→策略→TCA”的 HTML/zip 产物

### 6.2 每个步骤的输入/输出产物（Artifacts）

建议统一产物规范，便于 `/results/gallery` 与聚合接口复用（你已有结果体系）。([GitHub][1])

| Step           | Inputs                      | Outputs（建议路径）                                                                  |
| -------------- | --------------------------- | ------------------------------------------------------------------------------ |
| capture        | exchange/symbols/ws config  | `data/raw/{ex}/{sym}/{date}/trades.parquet`, `lob_deltas.parquet`, `meta.json` |
| lob_rebuild    | raw deltas + snapshots      | `data/lob/{ex}/{sym}/{date}/lob_state.parquet`                                 |
| micro_feature  | lob_state + trades          | `data/features/{run_id}/micro_features.parquet`                                |
| factor_compile | FactorSpec JSON             | `data/features/{run_id}/factor_{name}.parquet` + `factor_ast.json`             |
| factor_eval    | features + label            | `results/{run_id}/factor_scores.json` + `pareto.csv`                           |
| train          | config                      | `results/{run_id}/model/`                                                      |
| backtest       | model + config              | `results/{run_id}/backtest.zip`                                                |
| tca            | backtest fills + lob/trades | `results/{run_id}/tca_report.json` (+ html)                                    |
| report         | all above                   | `results/{run_id}/bundle.zip`                                                  |

---

## 7. API 与作业系统对接（FastAPI /run/* 风格一致）

你现在 `/run/*` 返回统一的 `{status, job_id, kind, cmd}`，并由 `/jobs/{id}/status|logs|cancel` 管理。([GitHub][1])
新增接口建议保持同一风格：

### 7.1 新增接口（建议）

* `POST /run/capture`：启动采集（后台 job）
* `POST /run/lob_rebuild`：订单簿重建
* `POST /run/micro_feature`：微观结构特征生成
* `POST /run/factor_compile`：FactorSpec → 编译特征
* `POST /run/factor_eval`：评分/筛选
* `POST /run/tca`：TCA 报告生成

或者更“少接口”的方式：扩展 `POST /run/feature` 增加 `mode: freqai|micro|factor_compiled`；但工程上更推荐拆开便于权限与资源隔离。

### 7.2 新增错误码（示例）

* `INVALID_FACTOR_SPEC`
* `UNKNOWN_OPERATOR`
* `TYPECHECK_FAILED`
* `LOOKAHEAD_DETECTED`
* `COMPLEXITY_BUDGET_EXCEEDED`
* `DATA_NOT_FOUND`
* `LOB_SEQUENCE_GAP`

---

## 8. LLM 集成：FactorSpec 作为唯一“可接受输出格式”

你已有 `/run/expression` 与 `/results/prepare-feedback` 的 LLM/反馈路径。([GitHub][1])
我们把它升级为：

### 8.1 LLM 输出：必须是 `FactorSpec` JSON

* 你给 LLM 的 system prompt 强制它输出 JSON
* 允许它附带自然语言 rationale，但不能替代 JSON

### 8.2 反馈闭环

`factor_eval` 输出的 ScoreReport 直接喂给 LLM（通过你现有的 feedback 准备接口能力延展），形成：

* 失败归因（泄漏/成本吞噬/不稳定/太复杂/与已有因子太像）
* 下一轮生成时加入“反同质化与复杂度惩罚”（AlphaAgent 思路）([arXiv][3])

---

## 9. 评测规范（让你过审、不被喷、也更容易赚钱）

1. **时间安全**：walk-forward + purge/embargo（系统强制）
2. **成本入账**：手续费 + 滑点/冲击 proxy（至少在 backtest 与 TCA 中入账）
3. **容量报告**：参与率/成交量占比约束下曲线
4. **污染控制**：尽量用“最近数据 + 可复现切分”避免“模型背题”，StockBench 的 contamination-free 思路对你非常关键。([arXiv][4])

---

## 10. MVP 落地路线（90 天能卖东西的版本）

### 10.1 30 天（最小可卖：Execution Intelligence Lite）

* 完成 `capture → micro_feature → tca` 三步
* 输出稳定的 `tca_report.json + html`
* 你就可以去找小团队/做市朋友做试点：**节省 bps**就是价值

### 10.2 60 天（Factor Compiler v1）

* FactorSpec + DSL parser + typecheck + no-lookahead + complexity budget
* factor_eval + Pareto 输出
* 这时你已经有“因子工厂雏形”

### 10.3 90 天（闭环产品）

* Flow 一键跑：`micro_feature → factor_compile → factor_eval → train → backtest → tca → report`
* 前端 Flow 增加节点（可视化因子评分、TCA 分解）

---

## 11. 交付清单（你最终应该在 repo 里看到什么）

1. **Factor Compiler**

* `FactorSpec` JSON schema（pydantic + jsonschema）
* DSL AST + 算子库 + 类型系统 + 静态检查
* 代码生成器（pandas/polars）+ 自动单测生成
* ScoreReport 与 Pareto 输出

2. **微观结构**

* WebSocket 采集器（至少 1 个交易所适配器）
* LOB 重建（snapshot+delta 校准）
* 特征注册表（feature registry）

3. **TCA**

* `tca_report.json` schema
* 基础指标实现 + HTML 渲染（可选）
* freqtrade/backtest 结果适配器（如果你用它作为回测引擎）

4. **Flow**

* 新步骤定义（steps + artifacts）
* configs 模板 + scripts 一键跑
* tests：最小 e2e（小数据集）保证可复现

---

## 12. 参考工作（你可以写进 docs/ 或 proposal 的 Related Work）

* FAMA（Neural-Symbolic 因子挖掘 agent）([ACL Anthology][2])
* Chain-of-Alpha（双链路自动公式 alpha 生成与优化）([arXiv][6])
* AlphaAgent（原创性/复杂度/假设一致性约束，抗 alpha decay）([arXiv][3])
* LOBench（订单簿表征学习 benchmark 思路，强调可复现）([arXiv][5])
* RL 执行：同时使用 market/limit orders 的执行优化框架（你后续 RL 执行模块可对标）([arXiv][7])
* StockBench（污染可控的交易 agent 评测思路，提醒“别只看回测曲线”）([arXiv][4])

---

# 你接下来怎么用这份 proposal（最实操的 3 步）

1. **先做 TCA + micro_feature**：这是最容易变现的 wedge。
2. **再做 FactorSpec + DSL 最小闭环**：先支持 20–30 个算子 + 5 个硬约束检查就够跑起来。
3. **最后把它接进 Flow**：把“结果资产化”（factor scores、tca report、bundle.zip）接入你现有 results/gallery/aggregate 体系。([GitHub][1])

---

如果你希望我把它进一步“落地到代码级别”，我可以在这份提案基础上继续给你两样东西（不需要你再解释）：

1. **FactorSpec 的 JSON Schema（draft-07 完整版）+ Pydantic 模型代码**
2. **一个最小可运行的算子库清单（30 个）+ AST parser/serializer 伪代码 + no-lookahead 静态检查算法**

你直接说一句：**“按 v1 最小可运行版本给我 schema + 30 算子 + 检查算法”**，我就按这个 proposal 的结构给你输出可直接拷进仓库的工程规格。

[1]: https://github.com/shatianming5/Agent_market.git "GitHub - shatianming5/Agent_market"
[2]: https://aclanthology.org/2024.findings-acl.233/?utm_source=chatgpt.com "Can Large Language Models Mine Interpretable Financial ..."
[3]: https://arxiv.org/abs/2502.16789?utm_source=chatgpt.com "AlphaAgent: LLM-Driven Alpha Mining with Regularized Exploration to Counteract Alpha Decay"
[4]: https://arxiv.org/abs/2510.02209?utm_source=chatgpt.com "StockBench: Can LLM Agents Trade Stocks Profitably In Real-world Markets?"
[5]: https://arxiv.org/abs/2505.02139?utm_source=chatgpt.com "Representation Learning of Limit Order Book: A Comprehensive Study and Benchmarking"
[6]: https://arxiv.org/abs/2508.06312?utm_source=chatgpt.com "Chain-of-Alpha: Unleashing the Power of Large Language ..."
[7]: https://arxiv.org/abs/2507.06345?utm_source=chatgpt.com "Reinforcement Learning for Trade Execution with Market and Limit Orders"
