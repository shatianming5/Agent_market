import zipfile
import pandas as pd
from pathlib import Path
zip_path = Path('user_data/backtest_results/expr_sweeper/20250928-080022/000_volatility_1/backtest-result-2025-09-28_04-00-29.zip')
with zipfile.ZipFile(zip_path) as zf:
    with zf.open('backtest-result-2025-09-28_04-00-29_market_change.feather') as fp:
        df = pd.read_feather(fp)
print(df.columns.tolist())
print(df[['llm_signal','llm_exit_ratio']].describe())
print(df['llm_signal'].quantile([0.7,0.8,0.9,0.95,0.99]))
