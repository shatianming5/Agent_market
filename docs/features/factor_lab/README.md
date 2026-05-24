# Factor Lab 研究 CLI

[返回主 README](../../../README.md) · [返回功能索引](../README.md)

Factor Lab 是 crypto / Freqtrade 研究方向的统一 CLI，覆盖数据下载、特征合并、因子挖掘、验证、回测、rank portfolio、LEAN 桥接、RL、组合搜索和 Factor Hub。

## 代码入口

| 位置 | 用途 |
|---|---|
| `scripts/factor_lab.py` | 统一 CLI |
| `src/agent_market/factor_lab/` | 研究核心模块 |
| `user_data/freqai_expressions*.json` | 因子表达式库 |
| `artifacts/factor_lab/` | Factor Lab 状态和中间产物 |
| `artifacts/rank_portfolio/` | rank portfolio 输出 |

## 子命令分类

| 子命令 | 功能 |
|---|---|
| `data` | 下载 KuCoin / OKX / Bybit / Binance / funding 数据 |
| `features` | 合并 mtf4h、cross-sectional、pair、funding、micro、ohlcv_micro 特征 |
| `features-restore` | 恢复 feature feather 备份 |
| `mine` | 迭代因子挖掘，支持 IC、组合、LLM、LEAN gate 等模式 |
| `mine-export` | 导出挖掘结果 Top-N |
| `mine-lean-gate` | 对挖掘候选做 LEAN gate |
| `factor-report` | 生成 IC / turnover / decay 诊断 |
| `exposure-report` | 因子收益 exposure attribution |
| `cache` | 查看或清理 Factor Lab cache |
| `memory-audit` | 审计 factor memory 覆盖、snoop 标签、重复和可交易性 |
| `validate` | 子区间稳定性、随机 baseline、相关性审计 |
| `backtest` | walk-forward 回测 |
| `rank-export` | 导出 rank portfolio 因子和 per-pair 信号 |
| `rank-backtest` | rank portfolio research backtest |
| `rank-sweep` | 扫 top-k、gross-cap 等配置 |
| `lean-export` / `lean-backtest` / `lean-compare` | 本地 LEAN 桥接验证 |
| `strategy-loop*` | agentic rank/factor strategy loop |
| `rl` | PPO / recurrent PPO / BC train/eval |
| `combo` | 因子组合搜索 |
| `deploy` | 因子库部署管理 |
| `hub` | Factor Hub 初始化、迁移、服务、UI |

## 常用命令

```bash
python scripts/factor_lab.py data okx-futures
python scripts/factor_lab.py features all
python scripts/factor_lab.py mine --tag exp1 --rounds 50
python scripts/factor_lab.py mine-export --tag exp1 --n 30
python scripts/factor_lab.py validate user_data/freqai_expressions_exp1.json
python scripts/factor_lab.py backtest --tag exp1 --train-months 6
```

Rank portfolio：

```bash
python scripts/factor_lab.py rank-export --tag exp1 --n 50 --risk-profile aggressive
python scripts/factor_lab.py rank-backtest --tag exp1 --venue okx --top-k 3 --gross-cap 10
python scripts/factor_lab.py rank-sweep --tag exp1 --venue okx
```

Deploy：

```bash
python scripts/factor_lab.py deploy list
python scripts/factor_lab.py deploy current
python scripts/factor_lab.py deploy switch freqai_expressions_exp1.json
python scripts/factor_lab.py deploy describe
```

## 数据和产物

| 路径 | 说明 |
|---|---|
| `user_data/data/kucoin/*.feather` | KuCoin spot OHLCV |
| `user_data/data/okx/futures/*-futures.feather` | OKX futures OHLCV |
| `user_data/freqai_expressions.json` | 当前部署因子库 |
| `user_data/freqai_expressions_<tag>.json` | 因子库快照 |
| `artifacts/factor_lab/<subcmd>/` | 挖掘、验证、报告等产物 |
| `artifacts/rank_portfolio/<tag>/signals/` | rank portfolio per-pair 信号 |

## 依赖

Factor Lab 需要完整依赖；只安装 server 最小依赖时会缺 `pandas`、训练库或 backtest 运行时。

```bash
pip install -c constraints.txt -r requirements-full.txt
```

## 验证

```bash
python scripts/factor_lab.py --help
pytest tests/test_rank_portfolio.py -q
```

