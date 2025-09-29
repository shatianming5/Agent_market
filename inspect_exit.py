import zipfile
from pathlib import Path
zip_path = Path('user_data/backtest_results/expr_sweeper/20250928-080022/000_volatility_1/backtest-result-2025-09-28_04-00-29.zip')
with zipfile.ZipFile(zip_path) as zf:
    code = zf.read('backtest-result-2025-09-28_04-00-29_FreqAIExampleStrategy.py').decode('utf-8')
start = code.index('def populate_exit_trend')
print('\n'.join(code[start:start+300].splitlines()))
