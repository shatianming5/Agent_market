"""python -m agent_market.wq_brain → forward to scripts/wq_brain.py CLI."""
import sys
from pathlib import Path

# Allow running as module from repo root
_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from scripts.wq_brain import main  # type: ignore[import]

if __name__ == "__main__":
    main()
