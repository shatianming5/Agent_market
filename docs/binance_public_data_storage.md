# Binance Public Data：存储与下载管理（AgentMarket）

目标：
- 统一存储结构（可增量/可断点/可校验）
- 能下载 Binance public data（klines / trades / aggTrades）的**可下载部分**
- 生成 manifest，避免重复下载、便于审计与后续 ETL

> 重要："全部" 体量巨大（全交易对 + 1s/1m/… + 多年），建议先从 BTC/ETH + 1m/1h 起跑；脚本支持分批扩展。

## 目录结构
默认根目录：`data/binance_public/`

```
data/binance_public/
  manifests/
    manifest.sqlite          # 下载清单（断点续传/状态/校验）
    manifest.jsonl           # 追加式日志（可选）
  spot/
    monthly/
      klines/<SYMBOL>/<INTERVAL>/*.zip
      trades/<SYMBOL>/*.zip
      aggTrades/<SYMBOL>/*.zip
    daily/...
  um/...
  cm/...
```

## 可下载的“全部”指什么
Binance 公共数据站点可下载的类型主要是：
- `klines`（多周期）
- `trades`
- `aggTrades`
并且按 `spot/um/cm`、`daily/monthly`、`symbol` 分目录。

脚本会：
- 发现并枚举目录（可选：从 exchangeInfo/或用户提供 symbol 列表）
- 逐文件下载 zip + `.CHECKSUM`
- 校验 sha256
- 更新 manifest（成功/失败/重试次数/文件大小/时间戳）

## 一键下载（建议分批）
示例：现货、月度、BTCUSDT/ETHUSDT、klines 1m 和 1h，时间 2020-01 到 2020-12

```bash
python scripts/binance_public_data/download.py \
  --market spot \
  --freq monthly \
  --symbols BTCUSDT,ETHUSDT \
  --datasets klines,trades,aggTrades \
  --intervals 1m,1h \
  --start 2020-01 --end 2020-12
```

## 继续下载（断点续传）
重复执行同一命令即可：
- 已完成的会跳过
- 未完成的会继续

## 存储清理
- 不建议删除 zip 原始包（可复现实验）
- 如需节省空间：后续增加 `extract_to_parquet.py` 后，可将 zip 转换并选择保留/删除
