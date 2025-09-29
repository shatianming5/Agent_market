import zipfile
from pathlib import Path
zip_path = Path('user_data/backtest_results/expr_sweeper/20250928-075120/000_volatility_1/backtest-result-2025-09-28_03-50-13.zip')
with zipfile.ZipFile(zip_path) as zf:
    code = zf.read('backtest-result-2025-09-28_03-50-13_FreqAIExampleStrategy.py').decode('utf-8')
print('\n'.join(code.splitlines()[300:350]))
