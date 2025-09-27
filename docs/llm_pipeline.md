# LLM 因子挖掘工作流

本文档说明如何通过 `freqai_expression_agent.py` 与 `freqai_auto_agent.py` 启用大语言模型（LLM）生成复合特征表达式，并跑通完整的 FreqAI 回测流水线。

## 1. 准备 LLM API

- 基础地址：`https://api.zhizengzeng.com/v1`
- 推荐模型：`gpt-3.5-turbo`（成本较低）
- 将 API Key 通过命令行参数 `--llm-api-key` 传入，或设置环境变量 `LLM_API_KEY`。

## 2. 单独生成表达式

```bash
conda run -n freqtrade python scripts/freqai_expression_agent.py \
    --feature-file user_data/freqai_features.json \
    --output user_data/freqai_expressions.json \
    --timeframe 1h \
    --llm-enabled \
    --llm-api-key <YOUR_API_KEY>
```

常用参数：
- `--llm-model`：默认 `gpt-3.5-turbo`
- `--llm-count`：单次生成的表达式数量
- `--llm-temperature`：采样温度（默认 0.2）

若 LLM 调用失败，脚本会自动回退到模板+gplearn 生成逻辑。

## 3. 一键跑通特征 + 表达式 + 回测

```bash
conda run --no-capture-output -n freqtrade \
  python scripts/freqai_auto_agent.py \
    --config user_data/config_freqai.json \
    --timeframe 1h \
    --llm-enabled \
    --llm-api-key <YOUR_API_KEY> \
    --top-expressions 40 \
    --expression-combo-top 5
```

脚本流程：
1. 聚合多交易对的基础特征，写入 `user_data/freqai_features.json`
2. 调用 LLM 生成候选表达式，评分后写入 `user_data/freqai_expressions.json`
3. 计算自动时间区间并运行 `freqtrade backtesting`
4. 将回测压缩结果复制到 `user_data/backtest_results/auto_agent/<timestamp>/`
5. 输出交易统计（笔数、总收益、胜率等）

示例输出片段：
```
[llm] valid expressions 5
[summary] FreqAIExampleStrategy 交易笔数: 49, 总收益: -8.5543, 胜率: 40.82%
[完成] 回测结果位置: user_data\backtest_results\auto_agent\20250927-202513\backtest-result-2025-09-27_16-25-24.zip
```

## 4. 常见问题

- `ModuleNotFoundError: talib`：请在 `freqtrade` 环境中执行命令。
- `未找到回测结果文件`：确认回测已成功完成，并检查 `user_data/backtest_results/.last_result.json` 是否产生最新 zip。
- 网络报错：LLM 请求失败时会自动回退到模板/遗传编程生成，但建议检查 API key 和 base url。

