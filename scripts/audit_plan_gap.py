#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional


ROOT = Path(__file__).resolve().parents[1]


_HEADING_RE = re.compile(r"^(#{2,4})\s+\(L\d+\)\s+.+$")
_STATUS_RE = re.compile(r"Status:\s+\*\*(PARTIAL|MISSING)\*\*")


@dataclass(frozen=True, slots=True)
class _Block:
    heading: str
    text: str


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _split_blocks(doc: str) -> List[_Block]:
    lines = doc.splitlines()
    starts: list[int] = []
    for idx, line in enumerate(lines):
        if _HEADING_RE.match(line.strip()):
            starts.append(idx)
    if not starts:
        return []

    starts.append(len(lines))
    blocks: list[_Block] = []
    for a, b in zip(starts, starts[1:]):
        chunk_lines = list(lines[a:b])
        # Drop trailing separators/whitespace so we can use a single stable separator
        # when rendering the extracted view.
        while chunk_lines and not chunk_lines[-1].strip():
            chunk_lines.pop()
        while chunk_lines and chunk_lines[-1].strip() == "---":
            chunk_lines.pop()
            while chunk_lines and not chunk_lines[-1].strip():
                chunk_lines.pop()

        chunk = "\n".join(chunk_lines).rstrip() + "\n"
        first = chunk_lines[0].strip() if chunk_lines else lines[a].strip()
        blocks.append(_Block(heading=first, text=chunk))
    return blocks


def _extract_partial_missing_blocks(full_doc: str) -> List[_Block]:
    blocks = _split_blocks(full_doc)
    out: list[_Block] = []
    for blk in blocks:
        if _STATUS_RE.search(blk.text):
            out.append(blk)
    return out


def _render_partial_missing(blocks: Iterable[_Block], *, source_name: str) -> str:
    generated = datetime.now().strftime("%Y-%m-%d")
    parts: list[str] = []
    parts.append("# Plan Gap（`plan.md`）— 仅 PARTIAL/MISSING")
    parts.append("")
    parts.append(f"Generated: {generated}")
    parts.append("")
    parts.append(f"Source: `{source_name}`（全量逐章核对，不省略）")
    parts.append("")
    parts.append("---")
    parts.append("")
    parts.append("说明：本文件是从 `docs/plan_gap_planmd.md` 中抽取所有包含 `**PARTIAL**` / `**MISSING**` 的章节块，用于只看差距列表与证据的场景。")
    parts.append("")
    parts.append("---")
    parts.append("")

    first = True
    for blk in blocks:
        if not first:
            parts.append("---")
            parts.append("")
        first = False
        parts.append(blk.text.rstrip())
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Regenerate plan.md gap extracts (PARTIAL/MISSING) from the full audit doc.")
    parser.add_argument("--full", default="docs/plan_gap_planmd.md", help="Full audit markdown path")
    parser.add_argument(
        "--partial",
        default="docs/plan_gap_planmd_partial_missing.md",
        help="Output partial/missing extract markdown path",
    )
    parser.add_argument("--check-only", action="store_true", help="Do not write files; fail if output differs")
    args = parser.parse_args(argv)

    full_path = (ROOT / str(args.full)).resolve() if not Path(str(args.full)).is_absolute() else Path(str(args.full)).resolve()
    partial_path = (ROOT / str(args.partial)).resolve() if not Path(str(args.partial)).is_absolute() else Path(str(args.partial)).resolve()

    if not full_path.exists():
        raise SystemExit(f"Full audit doc not found: {full_path}")

    full_doc = _read_text(full_path)
    blocks = _extract_partial_missing_blocks(full_doc)
    rendered = _render_partial_missing(blocks, source_name=str(Path(args.full)))

    if args.check_only:
        existing = partial_path.read_text(encoding="utf-8") if partial_path.exists() else ""
        if existing != rendered:
            raise SystemExit(f"[audit_plan_gap] OUT_OF_DATE: {partial_path}")
        print(f"[audit_plan_gap] OK: {partial_path} is up to date")
        return 0

    partial_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path.write_text(rendered, encoding="utf-8")
    print(f"[audit_plan_gap] wrote: {partial_path.relative_to(ROOT)}")
    print(f"[audit_plan_gap] blocks: {len(blocks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
