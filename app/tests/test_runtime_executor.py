from __future__ import annotations

import pytest

from typing import Callable

from agent_market.runtime import (
    ExecutionResult,
    NodeExecutionError,
    WorkflowExecutor,
    workflow_schema,
)
from agent_market.runtime.schema import WORKFLOW_SCHEMA_VERSION


def _fail() -> None:
    raise ValueError("boom")


def _build_executor(extra_registry: dict[str, Callable[..., object]] | None = None) -> WorkflowExecutor:
    registry = {
        "tests.runtime.add": lambda a, b: a + b,
        "tests.runtime.multiply": lambda value, factor=1: value * factor,
        "tests.runtime.echo": lambda value: value,
        "tests.runtime.format_pair": lambda pair: f"{pair}-ok",
        "tests.runtime.identity": lambda value: value,
        "tests.runtime.fail": _fail,
    }
    if extra_registry:
        registry.update(extra_registry)
    return WorkflowExecutor(registry=registry)


def _workflow(name: str, start: str, nodes: list[dict], variables: dict | None = None) -> dict:
    wf = {
        "version": WORKFLOW_SCHEMA_VERSION,
        "name": name,
        "start": start,
        "nodes": nodes,
    }
    if variables:
        wf["variables"] = variables
    return wf


def test_sequential_task_execution():
    executor = _build_executor()
    workflow = _workflow(
        name="sequential",
        start="sum",
        variables={"factor": 3},
        nodes=[
            {
                "id": "sum",
                "type": "task",
                "action": "tests.runtime.add",
                "inputs": {"a": {"$ref": "base"}, "b": 4},
                "assign": "total",
                "next": "double",
            },
            {
                "id": "double",
                "type": "task",
                "action": "tests.runtime.multiply",
                "inputs": {"value": {"$ref": "total"}, "factor": {"$ref": "factor"}},
                "assign": "doubled",
            },
        ],
    )

    result = executor.execute(workflow, context={"base": 2})

    assert isinstance(result, ExecutionResult)
    assert result.context["total"] == 6
    assert result.context["doubled"] == 18
    assert result.last_result == 18
    assert result.history == ["sequential:sum", "sequential:double"]


def test_condition_branching():
    executor = _build_executor()
    workflow = _workflow(
        name="conditional",
        start="sum",
        nodes=[
            {
                "id": "sum",
                "type": "task",
                "action": "tests.runtime.add",
                "inputs": {"a": 2, "b": 3},
                "assign": "total",
                "next": "guard",
            },
            {
                "id": "guard",
                "type": "condition",
                "test": "ctx['total'] > 4",
                "on_true": "success",
                "on_false": "failure",
            },
            {
                "id": "success",
                "type": "task",
                "action": "tests.runtime.echo",
                "inputs": {"value": {"$ref": "total"}},
                "assign": "result",
            },
            {
                "id": "failure",
                "type": "task",
                "action": "tests.runtime.echo",
                "inputs": {"value": "LOW"},
                "assign": "result",
            },
        ],
    )

    result = executor.execute(workflow)

    assert result.context["result"] == 5
    assert result.history == [
        "conditional:sum",
        "conditional:guard",
        "conditional:success",
    ]


def test_loop_iteration_accumulates_results():
    executor = _build_executor()

    loop_body = _workflow(
        name="loop-body",
        start="collect",
        nodes=[
            {
                "id": "collect",
                "type": "task",
                "action": "tests.runtime.format_pair",
                "inputs": {"pair": {"$ref": "pair"}},
                "assign": "last_loop_value",
            }
        ],
    )

    workflow = _workflow(
        name="loop-parent",
        start="seed",
        nodes=[
            {
                "id": "seed",
                "type": "task",
                "action": "tests.runtime.identity",
                "inputs": {"value": 0},
                "assign": "seed_value",
                "next": "loop",
            },
            {
                "id": "loop",
                "type": "loop",
                "iterate": "ctx['pairs']",
                "item": "pair",
                "body": loop_body,
                "accumulate": "processed_pairs",
            },
        ],
    )

    result = executor.execute(workflow, context={"pairs": ["BTC", "ETH"]})

    assert result.context["processed_pairs"] == ["BTC-ok", "ETH-ok"]
    assert result.context["last_loop_value"] == "ETH-ok"
    assert result.last_result == ["BTC-ok", "ETH-ok"]


def test_subflow_propagation():
    executor = _build_executor()

    subflow = _workflow(
        name="child",
        start="add",
        nodes=[
            {
                "id": "add",
                "type": "task",
                "action": "tests.runtime.add",
                "inputs": {"a": {"$ref": "base"}, "b": {"$ref": "delta"}},
                "assign": "child_total",
            }
        ],
    )

    workflow = _workflow(
        name="parent",
        start="trigger",
        nodes=[
            {
                "id": "trigger",
                "type": "subflow",
                "workflow": subflow,
                "inputs": {"delta": {"$ref": "increment"}},
                "assign": "sub_result",
                "propagate": False,
            }
        ],
    )

    result = executor.execute(workflow, context={"base": 5, "increment": 7})

    assert result.context["sub_result"] == 12
    assert "child_total" not in result.context

    # When propagate is True, child context merges back.
    workflow["nodes"][0]["propagate"] = True
    result = executor.execute(workflow, context={"base": 5, "increment": 7})
    assert result.context["child_total"] == 12


def test_error_branch_handles_failure():
    executor = _build_executor()
    workflow = _workflow(
        name="with-error",
        start="danger",
        nodes=[
            {
                "id": "danger",
                "type": "task",
                "action": "tests.runtime.fail",
                "on_error": "recover",
            },
            {
                "id": "recover",
                "type": "task",
                "action": "tests.runtime.echo",
                "inputs": {"value": {"$expr": "ctx['_last_error'].__class__.__name__"}},
                "assign": "recovery_message",
            },
        ],
    )

    result = executor.execute(workflow)

    assert result.context["recovery_message"] == "ValueError"
    assert "_last_error" in result.context


def test_unhandled_failure_raises_node_error():
    executor = _build_executor()
    workflow = _workflow(
        name="unhandled",
        start="danger",
        nodes=[
            {
                "id": "danger",
                "type": "task",
                "action": "tests.runtime.fail",
            }
        ],
    )

    with pytest.raises(NodeExecutionError):
        executor.execute(workflow)


def test_schema_generation_matches_version():
    schema = workflow_schema()
    assert schema["title"] == "WorkflowModel"
    assert schema["properties"]["version"]["const"] == WORKFLOW_SCHEMA_VERSION
