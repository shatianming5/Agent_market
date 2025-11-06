from __future__ import annotations

from typing import Dict

from .models import WORKFLOW_SCHEMA_VERSION, WorkflowModel


def workflow_schema(version: str = WORKFLOW_SCHEMA_VERSION) -> Dict:
    """Return the JSON schema definition for the requested workflow version."""
    if version != WORKFLOW_SCHEMA_VERSION:
        raise ValueError(f"Unsupported workflow schema version: {version}")
    return WorkflowModel.model_json_schema()


__all__ = ["WORKFLOW_SCHEMA_VERSION", "workflow_schema"]
