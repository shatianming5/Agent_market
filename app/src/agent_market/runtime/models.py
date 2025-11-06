from __future__ import annotations

from typing import Annotated, Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

WORKFLOW_SCHEMA_VERSION = "workflow/v1"


class BaseNode(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    id: str = Field(..., description="Unique identifier of the node within the workflow.")
    type: str = Field(..., description="Discriminator identifying the node kind.")
    next: Optional[str] = Field(
        default=None,
        description="Node id to jump to when the current node succeeds. Nodes may override the transition behaviour (e.g. condition nodes).",
    )
    on_error: Optional[str] = Field(
        default=None,
        description="Optional node id to transition to when execution raises an exception.",
    )
    assign: Optional[str] = Field(
        default=None,
        description="Optional context key used to persist the node result.",
    )


class TaskNode(BaseNode):
    type: Literal["task"]
    action: str = Field(..., description="Callable name to execute for this node.")
    inputs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Mapping of argument name to value specification. Supports literal values as well as {$ref, $expr} helpers.",
    )
    append_to: Optional[str] = Field(
        default=None,
        description="If provided, the node result is appended to this list stored in context.",
    )


class ConditionNode(BaseNode):
    type: Literal["condition"]
    test: str = Field(..., description="Boolean expression evaluated against the workflow context.")
    on_true: Optional[str] = Field(
        default=None, description="Node id to follow when the expression evaluates to truthy."
    )
    on_false: Optional[str] = Field(
        default=None, description="Node id to follow when the expression evaluates to falsy."
    )

    @field_validator("next")
    @classmethod
    def disallow_next(cls, value: Optional[str]) -> Optional[str]:
        if value is not None:
            raise ValueError("condition nodes must use on_true/on_false instead of next")
        return value


class WorkflowStub(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    version: Literal[WORKFLOW_SCHEMA_VERSION] = Field(
        default=WORKFLOW_SCHEMA_VERSION, description="Schema version of the sub workflow."
    )
    name: str = Field(..., description="Human-readable name of the sub workflow.")
    start: str = Field(..., description="Entry node id of the sub workflow.")
    nodes: List[Dict[str, Any]] = Field(..., min_length=1)
    variables: Dict[str, Any] = Field(
        default_factory=dict,
        description="Initial context variables merged before running the sub workflow.",
    )

    @model_validator(mode="after")
    def validate_nodes(self) -> WorkflowStub:
        ids = [n.get("id") for n in self.nodes if isinstance(n, dict)]
        if len(ids) != len(set(ids)):
            raise ValueError("node ids must be unique within a sub workflow")
        if self.start not in set(ids):
            raise ValueError(f"start node '{self.start}' is not defined in sub workflow nodes")
        return self


class LoopNode(BaseNode):
    type: Literal["loop"]
    iterate: Optional[str] = Field(
        default=None,
        description="Expression producing an iterable. Each item is bound to 'item' for loop body.",
    )
    condition: Optional[str] = Field(
        default=None,
        description="Boolean expression evaluated before each iteration (while-style loop).",
    )
    item: Optional[str] = Field(
        default=None,
        description="Name of the loop variable exposed to the loop body.",
    )
    body: WorkflowStub = Field(..., description="Workflow executed for each loop iteration.")
    max_iterations: Optional[int] = Field(
        default=None,
        ge=1,
        description="Safety limit preventing infinite loops. Required when using condition-based loops.",
    )
    accumulate: Optional[str] = Field(
        default=None,
        description="Context key used to collect per-iteration results. Defaults to assign target.",
    )

    @model_validator(mode="after")
    def validate_iteration_mode(self) -> LoopNode:
        if not self.iterate and not self.condition:
            raise ValueError("loop node requires either 'iterate' or 'condition'")
        if self.iterate and not self.item:
            raise ValueError("'item' is required when using iterate-based loops")
        if self.condition and self.max_iterations is None:
            raise ValueError("max_iterations must be specified for condition-based loops")
        return self


class SubflowNode(BaseNode):
    type: Literal["subflow"]
    workflow: WorkflowStub = Field(..., description="Nested workflow executed as a single node.")
    inputs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional context updates applied before running the subflow.",
    )
    propagate: bool = Field(
        default=True,
        description="When false, subflow context changes are not merged back into the parent context.",
    )


NodeUnion = Annotated[TaskNode | ConditionNode | LoopNode | SubflowNode, Field(discriminator="type")]


class WorkflowModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    version: Literal[WORKFLOW_SCHEMA_VERSION] = Field(
        default=WORKFLOW_SCHEMA_VERSION, description="Schema version identifier."
    )
    name: str = Field(..., description="Human-readable workflow name.")
    description: Optional[str] = Field(default=None, description="Workflow purpose.")
    start: str = Field(..., description="Entry node id.")
    variables: Dict[str, Any] = Field(
        default_factory=dict,
        description="Initial context variables merged into execution context.",
    )
    nodes: List[NodeUnion] = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_topology(self) -> WorkflowModel:
        ids = [n.id for n in self.nodes]
        if len(ids) != len(set(ids)):
            raise ValueError("node ids must be unique within a workflow")
        if self.start not in set(ids):
            raise ValueError(f"start node '{self.start}' is not defined in nodes")
        return self

    def node_map(self) -> Dict[str, BaseNode]:
        return {node.id: node for node in self.nodes}


def parse_workflow(data: Dict[str, Any]) -> WorkflowModel:
    """Parse a workflow definition into a strongly typed model."""
    return WorkflowModel.model_validate(data)
