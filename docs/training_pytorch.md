# PyTorch 深度学习训练（MLP）

本页介绍如何通过统一训练流水线（TrainingPipeline）使用 PyTorch MLP 进行训练与评估。

## 1. 依赖

建议在现有 conda 环境中安装 CPU 版 torch（如需 GPU 请按 CUDA 版本选择对应轮子）：

```
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

如果你使用的是清华/阿里镜像，请替换上面的 index-url。也可使用 conda：

```
conda install pytorch cpuonly -c pytorch
```

## 2. 准备数据与特征

请先确保已通过 `freqai_feature_agent.py` 生成特征文件，并完成历史数据下载：
- 特征：`user_data/freqai_features.json`
- 数据目录：`freqtrade/user_data/data/binanceus/*-1h.feather`

## 3. 运行训练（两种方式）

- 方式 A：使用配置文件（推荐）

```
python scripts/train_pipeline.py --config configs/train_pytorch_mlp.json
```

- 方式 B：命令行参数

```
python scripts/train_pipeline.py \
  --feature-file user_data/freqai_features.json \
  --data-dir freqtrade/user_data/data \
  --exchange binanceus \
  --timeframe 1h \
  --pairs "BTC/USDT" \
  --model pytorch_mlp \
  --params '{"hidden_dims":[128,64,32],"dropout":0.1,"epochs":25,"batch_size":128,"learning_rate":0.001,"use_cuda":false}' \
  --validation-ratio 0.2 \
  --model-dir artifacts/models/pytorch_mlp_demo
```

## 4. 产出

- 模型权重：`artifacts/models/pytorch_mlp_demo/pytorch_mlp.pt`
- 训练摘要：`artifacts/models/pytorch_mlp_demo/training_summary.json`
  - 包含：模型名、特征列、训练/验证集大小、RMSE 指标、模型路径等

## 5. 参数说明（常用）

- hidden_dims: 隐藏层维度列表（如 [128,64,32]）
- dropout: Dropout 比例（0~1）
- epochs: 训练轮数（默认 20~50）
- batch_size: 批大小（64~256）
- learning_rate: 学习率（如 1e-3）
- use_cuda: 是否使用 CUDA（默认 false）

## 6. 与回测衔接（可选）

当前训练摘要主要用于离线评估；如需将预测注入策略，请在策略中加载模型并使用同样的特征构造进行推断，或扩展 Pipeline 在训练后自动触发 freqtrade backtesting 并合并回测指标（参考 `src/agent_market/freqai/training/pipeline.py`）。

