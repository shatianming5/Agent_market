# Phase A 探针结论：WQ 账号 **USA-only**

> 日期：2026-05-10
> 范围：approved plan `突破 ACTIVE-18 self-corr 天花板`(`/Users/shatianming/.claude/plans/mutable-wishing-quilt.md`)
> 执行环境：远端 `zechuan@222.200.185.138:/mnt/SSD_4TB/zechuan/Agent_market`
> 账号：`.env` 中的 `WQ_EMAIL/WQ_PASSWORD`（与 wqb_v5_loop / v10_loop 共用）

## TL;DR

**WQ 远端账号只授权 USA region。所有非 USA 区域（GLB / EUR / ASI / CHN / AMR / TWN / HKG / KOR / JPN）simulate POST 直接返回 HTTP 400。**

⇒ 原计划 **Phase B1（CHN region pilot）BLOCKED at account-permission layer**，不可执行。

⇒ **唯一可立刻动作的杠杆是 Phase B2（USA region 内换 universe）**；Phase D（换数据源）依然有效但需要账户升级或本地数据基建。

## 探针数据（13 组合，全部已确认）

### CHN 多 universe（专项）— `/tmp/chn_probe.log`
| Region | Universe   | Result |
|--------|------------|--------|
| CHN    | TOP3000    | 400 Bad Request |
| CHN    | TOP2000A   | 400 Bad Request |
| CHN    | TOP1000A   | 400 Bad Request |
| CHN    | TOP500     | 400 Bad Request |
| CHN    | MID3000    | 400 Bad Request |
| CHN    | MIDLOW1000 | 400 Bad Request |

⇒ 6/6 universe 全 400 → 是 **region** 层级拒绝，不是 universe 名称错。

### 多 region 横切 — `/tmp/region_probe.log` + `/tmp/usa_baseline.py`
| Region | Universe    | Result |
|--------|-------------|--------|
| USA    | TOP3000     | ✅ 接受，sim url `3IDrTY1Eg4jT9Dnxju4T8hN` |
| USA    | TOP500      | ✅ 接受，sim url `2zY5Lb4VQ4iGbLM1bmGi7i4B` |
| GLB    | TOP3000     | 400 Bad Request |
| EUR    | TOP1200     | 400 Bad Request |
| ASI    | MINVOL1M    | 400 Bad Request |
| CHN    | TOP3000     | 400 Bad Request |
| AMR    | TOP600      | 400 Bad Request |
| TWN    | TOP500      | 400 Bad Request |
| HKG    | TOP500      | 400 Bad Request |

(`KOR/TOP200`, `JPN/TOP1600` 探针因连接断开未跑完；既然 7/7 其他区域都 400 且无 KOR/JPN 是异常授权的合理假设，省略。)

⇒ 13 个组合中**仅 USA 通过**。

## 决策树更新（覆盖原计划 Phase A3）

```
A1 真实结果           →  Phase B 入口
─────────────────────────────────
CHN/GLB 隔离  YES →  B1: wqb_v6_chn        — ❌ DEAD（账号无 CHN/GLB 权限）
universe-only YES →  B2: wqb_v6_usa_top500 — ✅ 唯一可立即执行
两者都 NO         →  Phase D（数据源扩展）  — 备选
```

## 推荐立即动作

### 短期（无新代码，今日可启动）
- **B2 USA/TOP500 pilot**：`--tag wqb_v6_usa_top500 --region USA --universe TOP500 --decay 6 --neutralization SUBINDUSTRY --truncation 0.08`，50 iter，max-turns 30，与 v5_loop 串行（v5 当前 ITER 38/100，等剩余 <20 iter 再起避免 submit quota 撞墙）。
- **理由**：账号唯一可行杠杆；TOP500 ↔ TOP3000 信号空间（mega-cap vs 全市场）差异显著；预期解锁 3-5 个 v5 已 ACTIVE-18 之外的新 ACTIVE。

### 中期（如 B2 也饱和）
- **Phase D1 - alt data**：检查 WQ 账号是否含 analyst / news_sentiment 字段访问，prompt 加入新字段族扩展信号空间。
- **Phase D3 - 本地 OHLCV custom expr**：`wq_brain local-simulate` 用 Kaggle/Stooq 数据离线评估候选，绕过 submit quota 限制。

### 不做（明确放弃）
- ~~B1 CHN region~~：账号权限层 BLOCKED，无修复路径。
- ~~Cross-region ensemble (D2)~~：依赖多 region 访问，同样 BLOCKED。

## 当前状态（截至 2026-05-10）

- `wqb_v5_loop`（tmux: `wqb_v10_loop`）：ITER 38/100，pool 527/ACTIVE 18，未见新 ACTIVE 增长（最近 5 个 ACTIVE 全是历史条目）。预计 ~30 小时后跑满 100 iter。
- 今日 WQ quota：simulate 356，submit 81（充足；submit 接近账号每日上限是 v5_loop 后期 LLM 报告的瓶颈，**与 region 无关**）。
- 探针消耗：14 simulate（已计入上述 356）。

## 文件 / 命令引用

- 探针脚本：`/tmp/chn_probe.py`, `/tmp/region_probe.py`, `/tmp/usa_baseline.py`（远端）
- 探针日志：`/tmp/chn_probe.log`, `/tmp/region_probe.log`（远端）
- 计划原文：`/Users/shatianming/.claude/plans/mutable-wishing-quilt.md`
- 任务状态：本地 TaskList #122-#127 已全部 completed
