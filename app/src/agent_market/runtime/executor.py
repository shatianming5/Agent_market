from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional

from .exceptions import NodeErrorContext, NodeExecutionError, WorkflowError, WorkflowValidationError
from .models import ConditionNode, LoopNode, SubflowNode, TaskNode, parse_workflow


Resolver = Callable[[str], Callable[..., Any]]


def _import_callable(name: str) -> Callable[..., Any]:
    """Resolve dotted import targets into callables."""
    if ":" in name:
        module_name, attr = name.split(":", 1)
    else:
        module_name, attr = name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def default_resolver(registry: Optional[Mapping[str, Callable[..., Any]]] = None) -> Resolver:
    registry = registry or {}

    def resolve(name: str) -> Callable[..., Any]:
        if name in registry:
            return registry[name]
        return _import_callable(name)

    return resolve


def _safe_eval(expr: str, ctx: MutableMapping[str, Any]) -> Any:
    """Evaluate a restricted Python expression against the workflow context."""
    allowed_globals = {
        "__builtins__": {},
        "len": len,
        "min": min,
        "max": max,
        "sum": sum,
        "any": any,
        "all": all,
        "sorted": sorted,
    }
    # Expose context under two convenient aliases.
    local_ctx = {"ctx": ctx, "context": ctx}
    code = compile(expr, "<workflow_expr>", "eval")
    return eval(code, allowed_globals, local_ctx)


def _resolve_value(value: Any, ctx: MutableMapping[str, Any]) -> Any:
    """Resolve node input values supporting {$ref, $expr} markers."""
    if isinstance(value, dict):
        if "$ref" in value:
            ref = value["$ref"]
            if ref not in ctx:
                raise KeyError(f"Context does not contain key '{ref}'")
            return ctx[ref]
        if "$expr" in value:
            expr = value["$expr"]
            if not isinstance(expr, str):
                raise TypeError("$expr must be a string")
            return _safe_eval(expr, ctx)
        return {k: _resolve_value(v, ctx) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_value(v, ctx) for v in value]
    return value


@dataclass
class ExecutionResult:
    context: Dict[str, Any]
    history: List[str] = field(default_factory=list)
    last_result: Any = None


class WorkflowExecutor:
    """Runtime executor for declarative workflow definitions."""

    def __init__(
        self,
        registry: Optional[Mapping[str, Callable[..., Any]]] = None,
        resolver: Optional[Resolver] = None,
        max_depth: int = 16,
    ) -> None:
        self._registry = dict(registry or {})
        self._resolve_callable = resolver or default_resolver(self._registry)
        self._max_depth = max_depth

    def execute(self, workflow: Mapping[str, Any], context: Optional[Dict[str, Any]] = None) -> ExecutionResult:
        data: Dict[str, Any]
        if hasattr(workflow, "model_dump"):
            data = workflow.model_dump()  # type: ignore[assignment]
        elif hasattr(workflow, "dict"):
            data = workflow.dict()  # type: ignore[assignment]
        else:
            data = dict(workflow)
        model = parse_workflow(data)
        return self._execute_model(model, context=context)

    def _execute_model(
        self,
        workflow,
        context: Optional[Dict[str, Any]] = None,
        *,
        depth: int = 0,
        history: Optional[List[str]] = None,
    ) -> ExecutionResult:
        if depth > self._max_depth:
            raise WorkflowError("maximum workflow depth exceeded", workflow.name)
        history = history or []
        if context is None:
            ctx: Dict[str, Any] = {}
        elif isinstance(context, dict):
            ctx = context
        else:
            ctx = dict(context)
        ctx.update(workflow.variables or {})
        node_map = workflow.node_map()
        current = workflow.start
        last_result: Any = None

        while current:
            node = node_map.get(current)
            if node is None:
                raise WorkflowValidationError(
                    f"workflow references undefined node '{current}'", workflow=workflow.name
                )
            history.append(f"{workflow.name}:{node.id}")
            try:
                if isinstance(node, TaskNode):
                    next_node, last_result = self._run_task(node, ctx)
                elif isinstance(node, ConditionNode):
                    next_node, last_result = self._run_condition(node, ctx)
                elif isinstance(node, LoopNode):
                    next_node, last_result = self._run_loop(node, ctx, depth, history)
                elif isinstance(node, SubflowNode):
                    next_node, last_result = self._run_subflow(node, ctx, depth, history)
                else:
                    raise WorkflowValidationError(
                        f"unsupported node type '{node.type}' encountered at runtime",
                        workflow=workflow.name,
                    )
            except Exception as exc:  # pragma: no cover - defensive; tests cover branches
                if node.on_error:
                    ctx["_last_error"] = exc
                    current = node.on_error
                    continue
                raise NodeExecutionError(
                    NodeErrorContext(workflow=workflow.name, node_id=node.id, node_type=node.type),
                    exc,
                )

            current = next_node

        return ExecutionResult(context=ctx, history=history, last_result=last_result)

    def _run_task(self, node: TaskNode, ctx: MutableMapping[str, Any]) -> tuple[Optional[str], Any]:
        action = self._resolve_callable(node.action)
        kwargs = {name: _resolve_value(value, ctx) for name, value in (node.inputs or {}).items()}
        result = action(**kwargs)
        if node.assign:
            ctx[node.assign] = result
        target_list = node.append_to or None
        if target_list:
            container = ctx.setdefault(target_list, [])
            if not isinstance(container, list):
                raise TypeError(f"context key '{target_list}' is not a list; cannot append result")
            container.append(result)
        return node.next, result

    def _run_condition(
        self, node: ConditionNode, ctx: MutableMapping[str, Any]
    ) -> tuple[Optional[str], Any]:
        outcome = _safe_eval(node.test, ctx)
        branch = node.on_true if outcome else node.on_false
        return branch, outcome

    def _run_loop(
        self,
        node: LoopNode,
        ctx: MutableMapping[str, Any],
        depth: int,
        history: List[str],
    ) -> tuple[Optional[str], Any]:
        loop_results: List[Any] = []
        body_model = parse_workflow(node.body.model_dump())

        if node.iterate is not None:
            iterable = _resolve_value({"$expr": node.iterate}, ctx)
            if not isinstance(iterable, Iterable):
                raise TypeError(f"loop iterate expression for node '{node.id}' must yield an iterable")
            for index, item in enumerate(iterable):
                if node.max_iterations is not None and index >= node.max_iterations:
                    break
                ctx[node.item] = item  # type: ignore[index]
                sub_result = self._execute_model(body_model, context=ctx, depth=depth + 1, history=history)
                loop_results.append(sub_result.last_result)
            ctx.pop(node.item, None)  # type: ignore[arg-type]
        else:
            iterations = 0
            while _safe_eval(node.condition, ctx):  # type: ignore[arg-type]
                if node.max_iterations is not None and iterations >= node.max_iterations:
                    break
                sub_result = self._execute_model(body_model, context=ctx, depth=depth + 1, history=history)
                loop_results.append(sub_result.last_result)
                iterations += 1

        target = node.accumulate or node.assign
        if target:
            ctx[target] = list(loop_results)
        if node.assign and target != node.assign:
            ctx[node.assign] = loop_results[-1] if loop_results else None

        return node.next, loop_results

    def _run_subflow(
        self,
        node: SubflowNode,
        ctx: MutableMapping[str, Any],
        depth: int,
        history: List[str],
    ) -> tuple[Optional[str], Any]:
        inputs = {name: _resolve_value(value, ctx) for name, value in (node.inputs or {}).items()}
        sub_context = ctx.copy()
        sub_context.update(inputs)
        sub_model = parse_workflow(node.workflow.model_dump())
        sub_result = self._execute_model(
            sub_model, context=sub_context, depth=depth + 1, history=history
        )
        if node.propagate:
            ctx.update(sub_result.context)
        if node.assign:
            ctx[node.assign] = sub_result.last_result
        return node.next, sub_result.last_result
