"""LLMOutputValidator tests — expression / CLI / JSON validation."""
from __future__ import annotations

import pytest

from agent_market.wq_brain.llm_validator import (
    KNOWN_SUBCOMMANDS,
    ValidationResult,
    _closest_subcommand,
    _extract_json,
    format_recovery_hint,
    validate_cli_invocation,
    validate_emit_response,
    validate_expression,
)


# ── ValidationResult ────────────────────────────────────────────────────


def test_result_truthy_when_ok():
    assert bool(ValidationResult(ok=True))
    assert not bool(ValidationResult(ok=False))


# ── validate_expression ─────────────────────────────────────────────────


def test_validate_expression_simple_ok():
    r = validate_expression("rank(close)")
    assert r.ok
    assert r.parsed is not None


def test_validate_expression_compound_ok():
    r = validate_expression(
        "rank(ts_corr(close, volume, 20)) + 0.5 * rank(sales / assets)"
    )
    assert r.ok


def test_validate_expression_empty_rejected():
    r = validate_expression("")
    assert not r.ok
    assert "empty" in r.error
    assert r.hint


def test_validate_expression_whitespace_only_rejected():
    r = validate_expression("    \n  \t ")
    assert not r.ok


def test_validate_expression_unbalanced_paren_rejected():
    r = validate_expression("rank(ts_corr(close, volume, 20)")  # missing close-paren
    assert not r.ok
    assert r.hint  # has recovery hint


def test_validate_expression_garbled_rejected():
    r = validate_expression("@@@ this is not fastexpr")
    assert not r.ok


# ── validate_cli_invocation ─────────────────────────────────────────────


def test_validate_cli_python_form_ok():
    r = validate_cli_invocation(
        "python3 scripts/wq_brain.py simulate --expr 'rank(close)' --tag X"
    )
    assert r.ok
    assert r.parsed["subcommand"] == "simulate"
    assert "--tag" in r.parsed["args"]


def test_validate_cli_bare_form_ok():
    r = validate_cli_invocation("wq_brain pool list --tag wqb")
    assert r.ok
    assert r.parsed["subcommand"] == "pool"


def test_validate_cli_unknown_subcommand_rejected():
    r = validate_cli_invocation("python3 scripts/wq_brain.py simulat --expr X")
    assert not r.ok
    # Levenshtein-1 should pick up "simulate" as the suggestion
    assert "simulate" in r.hint.lower()


def test_validate_cli_empty_rejected():
    assert not validate_cli_invocation("").ok
    assert not validate_cli_invocation("   ").ok


def test_validate_cli_non_wq_command_rejected():
    r = validate_cli_invocation("ls -la /tmp")
    assert not r.ok
    assert "wq_brain" in r.hint


def test_validate_cli_unbalanced_quotes_rejected():
    r = validate_cli_invocation(
        "python3 scripts/wq_brain.py simulate --expr 'rank(close)"
    )
    assert not r.ok
    assert "quotes" in r.hint.lower() or "tokenisation" in r.error.lower()


def test_validate_cli_pool_requires_subsubcommand():
    r = validate_cli_invocation("python3 scripts/wq_brain.py pool")
    assert not r.ok
    assert "pool" in r.hint.lower()


def test_validate_cli_pool_unknown_subsubcommand_rejected():
    r = validate_cli_invocation("python3 scripts/wq_brain.py pool fly")
    assert not r.ok


def test_validate_cli_pool_known_subsubcommand_ok():
    r = validate_cli_invocation("python3 scripts/wq_brain.py pool list --tag X")
    assert r.ok


def test_known_subcommands_includes_recent_additions():
    # Sanity that the validator is up-to-date with calibrate-local /
    # seed-calibration / kaggle-import / audit-data added recently.
    for sub in ("calibrate-local", "seed-calibration", "audit-data",
                "kaggle-fetch", "kaggle-import"):
        assert sub in KNOWN_SUBCOMMANDS


def test_closest_subcommand_prefix_match():
    assert _closest_subcommand("simu") == "simulate"


def test_closest_subcommand_typo_match():
    assert _closest_subcommand("simulat") == "simulate"


def test_closest_subcommand_unmatched_returns_none():
    assert _closest_subcommand("aklsdjflkasdj") is None


# ── validate_emit_response ──────────────────────────────────────────────


def test_validate_emit_pure_json_ok():
    r = validate_emit_response('{"ok": true, "alpha_id": "X1"}')
    assert r.ok
    assert r.parsed["alpha_id"] == "X1"


def test_validate_emit_required_fields_pass():
    r = validate_emit_response(
        '{"ok": true, "alpha_id": "X1", "wq_response": {}}',
        required_fields=["alpha_id", "wq_response"],
    )
    assert r.ok


def test_validate_emit_required_fields_fail():
    r = validate_emit_response(
        '{"ok": true, "alpha_id": "X1"}',
        required_fields=["alpha_id", "wq_response"],
    )
    assert not r.ok
    assert "wq_response" in r.error


def test_validate_emit_handles_markdown_fenced_json():
    text = "```json\n{\"ok\": false, \"error\": \"boom\"}\n```"
    r = validate_emit_response(text)
    assert r.ok  # parsed correctly even though wrapped
    assert r.parsed["error"] == "boom"


def test_validate_emit_handles_chatty_preamble():
    text = "Here is the result:\n\n{\"ok\": true, \"alpha_id\": \"AAA\"}\nDone."
    r = validate_emit_response(text)
    assert r.ok
    assert r.parsed["alpha_id"] == "AAA"


def test_validate_emit_no_json_rejected():
    r = validate_emit_response("This is plain text without json")
    assert not r.ok
    assert "JSON" in r.error


def test_validate_emit_missing_ok_field():
    r = validate_emit_response('{"alpha_id": "X1"}')
    assert not r.ok
    assert "'ok'" in r.error


def test_extract_json_returns_none_for_garbage():
    assert _extract_json("hello world") is None


def test_extract_json_handles_object_inside_text():
    parsed = _extract_json("prefix {\"key\": 1} suffix")
    assert parsed == {"key": 1}


def test_extract_json_picks_first_valid_object_among_many():
    """When the text has 2 JSON objects, return only the FIRST.

    The previous greedy regex `\\{[\\s\\S]*\\}` would span both objects and
    silently fail; raw_decode returns the first one cleanly.
    """
    text = '{"first": 1} then some prose then {"second": 2}'
    parsed = _extract_json(text)
    assert parsed == {"first": 1}


def test_extract_json_skips_invalid_leading_brace():
    """If the first `{` doesn't open a valid object, raw_decode advances."""
    text = '{ this is not json } {"good": true}'
    parsed = _extract_json(text)
    assert parsed == {"good": True}


def test_extract_json_handles_nested_objects():
    parsed = _extract_json('prefix {"outer": {"inner": 42}} suffix')
    assert parsed == {"outer": {"inner": 42}}


def test_extract_json_returns_none_when_only_arrays():
    """Top-level array isn't a dict — extract_json should keep looking."""
    assert _extract_json("[1, 2, 3]") is None


# ── format_recovery_hint ────────────────────────────────────────────────


def test_format_recovery_hint_empty_for_ok():
    r = ValidationResult(ok=True)
    assert format_recovery_hint(r) == ""


def test_format_recovery_hint_renders_markdown_for_failure():
    r = ValidationResult(
        ok=False, kind="cli",
        error="unknown subcommand",
        hint="try `simulate`",
    )
    md = format_recovery_hint(r)
    assert "❌" in md
    assert "Cli" in md  # title-cased kind
    assert "unknown subcommand" in md
    assert "simulate" in md
