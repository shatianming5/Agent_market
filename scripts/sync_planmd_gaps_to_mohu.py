#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


ROOT = Path(__file__).resolve().parents[1]

_HEADING_RE = re.compile(r"^(#{2,4})\s+\(L(?P<lineno>\d+)\)\s+(?P<title>.+)$")
_STATUS_RE = re.compile(r"Status:\s+\*\*(?P<status>PARTIAL|MISSING)\*\*")

_MARK_START = "<!-- planmd-gaps:start -->"
_MARK_END = "<!-- planmd-gaps:end -->"


@dataclass(frozen=True, slots=True)
class _GapItem:
    lineno: int
    heading: str
    title: str
    status: str


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _parse_gap_items(partial_doc: str) -> List[_GapItem]:
    lines = partial_doc.splitlines()
    items: list[_GapItem] = []
    for idx, line in enumerate(lines):
        m = _HEADING_RE.match(line.strip())
        if not m:
            continue
        # Find status within a small window after the heading.
        window = "\n".join(lines[idx : min(len(lines), idx + 30)])
        sm = _STATUS_RE.search(window)
        status = sm.group("status") if sm else "PARTIAL"
        lineno = int(m.group("lineno"))
        title = m.group("title").strip()
        items.append(
            _GapItem(
                lineno=lineno,
                heading=line.strip(),
                title=title,
                status=status,
            )
        )
    return items


def _render_section(items: List[_GapItem]) -> str:
    out: list[str] = []
    out.append("## Proposal Gap Backlog（plan.md）")
    out.append("")
    out.append("> Auto-generated from `docs/plan_gap_planmd_partial_missing.md`（仅 PARTIAL/MISSING 抽取视图）。")
    out.append("> 请勿在 marker 内手工编辑；请更新 `docs/plan_gap_planmd.md` 后再重新生成。")
    out.append("")
    out.append(_MARK_START)
    if not items:
        out.append("- (empty)")
    else:
        for idx, it in enumerate(items, start=1):
            out.append(f"- [ ] PlanMD-{idx:03d}: {it.heading.lstrip('# ').strip()}")
            out.append(f"  - Status: **{it.status}**")
            out.append(f"  - PlanRef: `docs/plan_gap_planmd.md` (L{it.lineno})")
            out.append("  - Acceptance: 对应章节在 `docs/plan_gap_planmd.md` 中变为 **DONE** 且 `pytest -q` 通过。")
    out.append(_MARK_END)
    out.append("")
    return "\n".join(out)


def _upsert_marked_section(mohu: str, section_md: str) -> str:
    header = "## Proposal Gap Backlog（plan.md）"
    lines = mohu.splitlines()
    sec_lines = section_md.strip("\n").splitlines()

    # If markers exist, replace the entire section (including the heading/preamble)
    # to keep the operation idempotent even if a previous version duplicated headers.
    if _MARK_START in mohu and _MARK_END in mohu:
        try:
            start_idx = next(i for i, ln in enumerate(lines) if ln.strip() == _MARK_START)
            end_idx = next(i for i, ln in enumerate(lines) if ln.strip() == _MARK_END and i >= start_idx)
        except StopIteration:
            start_idx = -1
            end_idx = -1

        if start_idx >= 0 and end_idx >= 0:
            section_start = start_idx
            found_header = False
            for i in range(start_idx - 1, -1, -1):
                if lines[i].strip() == header:
                    section_start = i
                    found_header = True
                    continue
                if found_header and lines[i].startswith("## ") and lines[i].strip() != header:
                    break

            section_end = end_idx + 1
            # Consume trailing blank lines after the marker end to keep formatting stable.
            while section_end < len(lines) and not lines[section_end].strip():
                section_end += 1

            before = "\n".join(lines[:section_start]).rstrip()
            after = "\n".join(lines[section_end:]).lstrip()
            if before:
                return before + "\n\n" + "\n".join(sec_lines).rstrip() + "\n\n" + after + ("\n" if after else "\n")
            return "\n".join(sec_lines).rstrip() + "\n\n" + after + ("\n" if after else "\n")

    # Append at end.
    return mohu.rstrip() + "\n\n" + section_md.strip() + "\n"


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Sync plan.md gap items into docs/mohu.md as a Proposal backlog section.")
    parser.add_argument("--partial", default="docs/plan_gap_planmd_partial_missing.md", help="Input partial gap doc")
    parser.add_argument("--mohu", default="docs/mohu.md", help="Target mohu markdown")
    parser.add_argument("--dry-run", action="store_true", help="Print summary only (no writes)")
    args = parser.parse_args(argv)

    partial_path = (ROOT / str(args.partial)).resolve() if not Path(str(args.partial)).is_absolute() else Path(str(args.partial)).resolve()
    mohu_path = (ROOT / str(args.mohu)).resolve() if not Path(str(args.mohu)).is_absolute() else Path(str(args.mohu)).resolve()

    if not partial_path.exists():
        raise SystemExit(f"Partial gap doc not found: {partial_path}")
    if not mohu_path.exists():
        raise SystemExit(f"Mohu doc not found: {mohu_path}")

    partial_doc = _read(partial_path)
    items = _parse_gap_items(partial_doc)
    section_md = _render_section(items)

    if args.dry_run:
        print(f"[sync_planmd_gaps_to_mohu] items: {len(items)}")
        return 0

    mohu_doc = _read(mohu_path)
    updated = _upsert_marked_section(mohu_doc, section_md)
    _write(mohu_path, updated)
    print(f"[sync_planmd_gaps_to_mohu] wrote: {mohu_path.relative_to(ROOT)}")
    print(f"[sync_planmd_gaps_to_mohu] items: {len(items)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
