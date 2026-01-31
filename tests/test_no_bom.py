from __future__ import annotations

import subprocess
from pathlib import Path


def test_repo_files_do_not_start_with_utf8_bom() -> None:
    bom = b"\xef\xbb\xbf"
    files = (
        subprocess.check_output(["git", "ls-files"])
        .decode("utf-8")
        .splitlines()
    )
    offenders: list[str] = []
    for rel in files:
        path = Path(rel)
        try:
            head = path.read_bytes()[:3]
        except Exception:
            continue
        if head == bom:
            offenders.append(rel)

    assert not offenders, "Files start with UTF-8 BOM:\n" + "\n".join(offenders)

