import zipfile
from pathlib import Path
zip_path = Path('user_data/backtest_results/expr_sweeper/20250928-073050/000_volatility_1/backtest-result-2025-09-28_03-30-57.zip')
with zipfile.ZipFile(zip_path) as zf:
    data = zf.read('backtest-result-2025-09-28_03-30-57_FreqAIExampleStrategy.py').decode('utf-8').splitlines()
for line in data[:40]:
    print(line)
