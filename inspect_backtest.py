import zipfile
import pandas as pd
from pathlib import Path
zip_path = Path('user_data/backtest_results/expr_sweeper/20250928-073050/000_volatility_1/backtest-result-2025-09-28_03-30-57.zip')
with zipfile.ZipFile(zip_path) as zf:
    json_member = next(name for name in zf.namelist() if name.endswith('.json') and 'config' not in name)
    data = zf.read(json_member)
    # load csv? there maybe data file? need to inspect
    members = zf.namelist()
    print(members)
    if any(name.endswith('freqai_data.feather') for name in members):
        name = next(name for name in members if name.endswith('freqai_data.feather'))
        with zf.open(name) as fp:
            df = pd.read_feather(fp)
        print(df.columns.tolist())
        print(df[['llm_entry_ratio','llm_signal']].describe())
        if 'prediction' in df.columns:
            print(df['prediction'].describe())
