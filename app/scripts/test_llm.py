#!/usr/bin/env python
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agent_market.freqai import llm as llm_utils  # type: ignore


def main() -> None:
    cfg = llm_utils.LLMConfig(
        base_url=os.environ.get('LLM_BASE_URL', llm_utils.DEFAULT_BASE_URL),
        api_key=os.environ.get('LLM_API_KEY', llm_utils.DEFAULT_API_KEY),
        model=os.environ.get('LLM_MODEL', llm_utils.DEFAULT_MODEL),
        temperature=0.0,
        max_tokens=128,
        count=1,
        retries=1,
        timeout=float(os.environ.get('LLM_TIMEOUT', '20')),
    )
    prompt = "{""expressions"": [{""name"": ""probe"", ""expression"": ""z(feat_rsi_14)""}] }"
    try:
        content, usage = llm_utils.request_completion(prompt, cfg)
        print('OK content length:', len(content))
        if usage:
            print('usage:', usage)
    except Exception as exc:  # noqa: BLE001
        print('LLM ERROR:', type(exc).__name__, str(exc)[:200])
        sys.exit(2)


if __name__ == '__main__':
    main()

