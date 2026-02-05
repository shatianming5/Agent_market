# 90 天闭环产品（Closed Loop）

> 目标：一键跑通 `micro_feature → factor_compile → factor_eval → train → backtest → tca → report`，并在前端可查看关键产物（factor scores / TCA / bundle.zip）。

## 一键运行（推荐：隔离 workspace，避免覆盖证据链）

```bash
python scripts/closed_loop_demo.py
```

运行完成后会打印 `run_id`、隔离后的 `artifacts_root` 与 `user_data_root`。产物索引文件为：

- `artifacts_root/run_meta.json`
- `artifacts_root/runs/<run_id>/run_meta.json`

## 直接运行（使用仓库默认 `artifacts/` 与 `user_data/`）

```bash
python scripts/agent_flow.py --config configs/agent_flow_closed_loop_demo_fixture.json --steps capture lob_rebuild micro_feature factor_compile factor_eval ml backtest tca report
```

## 导出为 plan.md 建议目录布局（`data/` + `results/`）

```bash
python scripts/export_planmd_layout.py --run-id <run_id>
```

如需限制 capture 原始数据转换规模（避免大文件内存压力），可加：

```bash
python scripts/export_planmd_layout.py --run-id <run_id> --max-rows 200000
```

## 前端查看

```bash
uvicorn server.main:app --host 0.0.0.0 --port 8000
```

打开 `http://127.0.0.1:8000/web/index.html`，在“产物检查”面板可跳转：

- factor scores（`/flow/factor-scores/<run_id>`）
- TCA report（`/flow/tca/<run_id>`）
- bundle.zip 下载（`/results/bundles/download/<run_id>`）

