#!/usr/bin/env bash
set -Eeuo pipefail
cd /mnt/SSD_4TB/zechuan/Agent_market
set -a; source .env; set +a
for i in $(seq 1 50); do
  echo === ITER $i / 50 START $(date +%H:%M:%S) ===
  PYTHONPATH=src python3 scripts/wq_brain.py agent --tag wqb_v6_usa_top500 --region USA --universe TOP500 --decay 6 --neutralization SUBINDUSTRY --truncation 0.08 --cli opencode --model MiniMax-M2.7-highspeed --max-turns 30 --auto-submit --timeout-sec 4500 || true
  echo === ITER $i / 50 END $(date +%H:%M:%S) ===
done 2>&1 | tee -a logs/wqb_v6_usa_top500.log
