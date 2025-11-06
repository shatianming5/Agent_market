from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


class WorkflowError(RuntimeError):
    """Base exception for workflow-related failures."""

    def __init__(self, message: str, workflow: str, node: Optional[str] = None) -> None:
        super().__init__(message)
        self.workflow = workflow
        self.node = node


@dataclass
class NodeErrorContext:
    workflow: str
    node_id: str
    node_type: str


class NodeExecutionError(WorkflowError):
    """Raised when a node fails and the failure is not handled by on_error branches."""

    def __init__(self, ctx: NodeErrorContext, original: Exception) -> None:
        super().__init__(
            f"Node '{ctx.node_id}' ({ctx.node_type}) failed in workflow '{ctx.workflow}': {original}",
            workflow=ctx.workflow,
            node=ctx.node_id,
        )
        self.original = original
        self.context = ctx


class WorkflowValidationError(WorkflowError):
    """Emitted when the workflow definition fails semantic validation."""

    def __init__(self, message: str, workflow: str) -> None:
        super().__init__(message, workflow=workflow, node=None)
