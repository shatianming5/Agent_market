from __future__ import annotations

from typing import Sequence

import sys
from pathlib import Path


def _import_smoke() -> None:
    root = Path(__file__).resolve().parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    # Local import after path adjustment
    from scripts.diagnostics.server_smoke import main as smoke_main  # type: ignore

    smoke_main()


def main(_: Sequence[str] | None = None) -> None:
    """Run the lightweight server smoke test."""
    _import_smoke()


if __name__ == "__main__":
    main()

