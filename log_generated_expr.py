from __future__ import annotations

import pathlib

path = pathlib.Path('freqtrade/user_data/strategies/FreqAIExampleStrategy.py')
text = path.read_text(encoding='utf-8')
marker = "        return dataframe, generated, metadata_map\n"
if text.count(marker) != 1:
    raise SystemExit('marker mismatch')
text = text.replace(marker, "        logger.info(\"evaluate expressions -> %d features\", len(generated))\n        return dataframe, generated, metadata_map\n")
path.write_text(text, encoding='utf-8')
