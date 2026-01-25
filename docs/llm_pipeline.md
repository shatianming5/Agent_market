# LLM 表达式生成与因子挖掘（Top-N + 进化算法）

本项目的表达式生成脚本为 `scripts/freqai_expression_agent.py`，支持：
- LLM 生成表达式（可选）
- 基于历史数据的因子打分与筛选（`--mine --top-n N`）
- 进化算法（默认开启；可用 `--no-evolve` 关闭）

所有产物默认写入仓库根目录的 `user_data/`。

## 0. LLM 配置（可选）

在根目录 `.env`（或环境变量）中设置（OpenAI 兼容接口）：

```ini
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
OPENAI_API_KEY=<YOUR_API_KEY>
```

兼容变量（可选）：也支持 `LLM_BASE_URL/LLM_MODEL/LLM_API_KEY`（优先级高于 `OPENAI_*`）。

也可以在命令行中用 `--llm-base-url/--llm-model/--llm-api-key` 覆盖。

## 1. 生成基础特征

```bash
./.venv/bin/python scripts/freqai_feature_agent.py \
  --config user_data/config_freqai.json \
  --output user_data/freqai_features_real.json \
  --timeframe 1h
```

## 2. 表达式生成 + 因子挖掘（Top-N + 进化算法）

```bash
./.venv/bin/python scripts/freqai_expression_agent.py \
  --config user_data/config_freqai.json \
  --output user_data/freqai_expressions_selected.json \
  --timeframe 1h \
  --mine --top-n 30
```

常用参数：
- `--llm-enabled`：启用 LLM 生成候选表达式
- `--no-evolve`：关闭进化算法（默认开启）
- `--evolve-population/--evolve-generations/--evolve-max-depth`：控制进化搜索规模与复杂度

产物：
- Top-N：`user_data/freqai_expressions_selected.json`
- 全量打分候选：`user_data/freqai_expressions_selected_scored_all.json`

## 3. 推荐：用 Agent Flow 逐步调试

```bash
./.venv/bin/python scripts/agent_flow.py --config configs/agent_flow_example.json --steps feature expression
```
