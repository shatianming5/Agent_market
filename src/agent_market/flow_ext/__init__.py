"""flow_ext — agent_flow 主流水线的步骤分发与 artifacts 助手。

模块组织：

  * ``step_dispatch`` — 把 agent_flow.py 解析出的 steps 列表分派到具体 handler
  * ``steps``         — 具体 step 实现（feature / expression / ml / backtest / ...）
  * ``artifacts``     — 每步 artifacts 落盘到 ``artifacts/runs/<run_id>/``

直接调用入口在 ``scripts/agent_flow.py`` → ``src/agent_market/agent_flow.py``。
"""
from __future__ import annotations

from . import artifacts, steps

__all__ = ["artifacts", "steps"]
