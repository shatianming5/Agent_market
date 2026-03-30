# Strategy Miner Recovery Report (2026-03-05)

目标：将 strategy miner 从连续 `best_reward=-inf` 恢复到**稳定产出有效 leaderboard**（非 `-inf`，且包含可运行、可回测的策略），并为后续收益质量提升提供修复优先级与路线图。

## A. 根因审计（runs: b628eb800705 / 285dd2073f4d / 8c32a9157438）

### A1) 审计方法
- 数据来源：`artifacts/runs/<run_id>/strategy_miner/checkpoint.json`
- “每轮(round)”口径：checkpoint 中每个 `candidates[i]` 视为一次产出轮次（对应迭代 i）。
- 分类规则（按优先级）：
  1. `opencode empty_response_body`：`diagnosis` 含 `empty_response_body`
  2. **语法校验失败**：`validation_passed=false`（包括 SyntaxError / IStrategy 继承校验等静态校验失败）
  3. **回测执行失败**：静态校验通过但无回测摘要，且诊断为 backtest 执行错误/超时
  4. **结果解析失败**：回测执行完成但结果 zip/JSON 解析失败（或找不到结果 zip）

### A2) 失败类别占比（每轮）

| run_id | rounds | opencode empty_response_body | 语法校验失败 | 回测执行失败 | 结果解析失败 |
|---|---:|---:|---:|---:|---:|
| b628eb800705 | 5 | 0 (0.0%) | 5 (100.0%) | 0 (0.0%) | 0 (0.0%) |
| 285dd2073f4d | 5 | 0 (0.0%) | 5 (100.0%) | 0 (0.0%) | 0 (0.0%) |
| 8c32a9157438 | 5 | 0 (0.0%) | 5 (100.0%) | 0 (0.0%) | 0 (0.0%) |
| **Overall** | **15** | **0 (0.0%)** | **15 (100.0%)** | **0 (0.0%)** | **0 (0.0%)** |

### A3) 关键发现（可复现证据）
三组 runs 的失败都在**静态校验阶段**被拦截，典型错误是：
- strategy 文件第 1 行出现 OpenCode 风格的工具标记（如 `<read .../>` / `<write ...>` / `<bash .../>`），导致 Python `SyntaxError`。
- 该问题在 “生成” 与 “repair” 两条链路中都会复现（prompt 明确允许工具标记；当模型未实际执行工具调用时，会把工具标记当成普通文本输出；当前实现把该文本直接写入 `.py`）。

## 修复优先级（从“恢复可用”到“提升质量”）

### P0（必须先做，否则永远 -inf）
1. **生成输出清洗/提取**：对 LLM 输出中的工具标记/代码围栏做鲁棒提取，确保写入 `.py` 的始终是纯 Python 代码。
2. **本地 auto-fix 兜底**：遇到 `SyntaxError` / 常见结构错误时先本地轻修（去工具标记、补 import、补 pass、修继承），再走 LLM repair。

### P1（稳定产出非 -inf leaderboard）
3. **provider 重试与降级链**：`opencode` 失败或产出不可用时，自动 fallback 到 `openai-compatible (glm-4-flash)`；不再有 template 兜底（no-template enforced）。
4. **多候选并行**：每迭代至少生成 3 个候选（可配置），并各自独立验证/回测；避免单点失败导致整轮无有效产出。
5. **回测失败可诊断可恢复**：将失败分为依赖缺失/参数错误/策略路径错误等，并在 diagnosis 中给出明确可执行建议（必要时驱动 repair prompt 更精准）。

### P2（质量门禁与反未来）
6. **扩展 look-ahead 检查**：除 `shift(-1)` / `rolling(center=True)` 外，增加常见未来泄露模式检测（如 `bfill/backfill`、`diff(-1)`、`pct_change(-k)`、`np.roll(...,-1)` 等）。
7. **最小可交易性门槛 + 惩罚**：除 `min trades / max DD / winrate` 硬门槛外，引入过拟合惩罚（例如“极少交易 + 极高收益/胜率”的组合惩罚、代码阈值/常数过多的复杂度惩罚等），并确保 leaderboard 仅纳入 `constraints_ok=true` 的候选。

## 下一步执行概览（对应本次需求 B→F）
- B：实现 fallback 链、多候选并行、增强 repair 与 backtest 失败分类
- C：扩展 look-ahead + 交易性/过拟合门禁；leaderboard 仅收录 constraints_ok
- D：新增 `configs/strategy_miner_recovery.json` + `scripts/run_strategy_miner_recovery.py`
- E：补测 + 真实 recovery 实跑一轮，确保产生非 `-inf` 的有效结果
