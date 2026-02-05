from __future__ import annotations


def test_llm_factor_spec_prompt_assets_and_validation() -> None:
    from agent_market.freqai.llm import load_factor_spec_prompt_assets, parse_factor_spec

    assets = load_factor_spec_prompt_assets()
    assert isinstance(assets.get("system_prompt"), str) and assets["system_prompt"].strip()
    fewshot = assets.get("fewshot") or {}
    assert isinstance(fewshot, dict)
    examples = fewshot.get("examples")
    assert isinstance(examples, list) and examples

    spec = parse_factor_spec(__import__("json").dumps(examples[0], ensure_ascii=False))
    assert getattr(spec, "name", None)

