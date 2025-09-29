#!/usr/bin/env python
from __future__ import annotations

from server.main import HyperoptReq, run_hyperopt  # type: ignore


def main() -> None:
    req = HyperoptReq(
        config='configs/config_freqai_multi.json',
        strategy='ExpressionLongStrategy',
        strategy_path='freqtrade/user_data/strategies',
        timerange='20210101-20210430',
        spaces='buy sell protection',
        hyperopt_loss='SharpeHyperOptLoss',
        epochs=1,
        freqaimodel='LightGBMRegressor',
        job_workers=-1,
    )
    out = run_hyperopt(req)
    print(out)


if __name__ == '__main__':
    main()

