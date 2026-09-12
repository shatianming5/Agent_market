"""factor_compiler — 因子规格 (FactorSpec) DSL 编译 + 检查 + 评分。

输入为结构化 ``FactorSpec`` (含 ``ExprNode`` AST + 元数据 + 测试用例)；
输出为可执行 + scored 的因子。

子包：

  * ``dsl``      — 表达式 AST、解析、序列化（``FormulaParseError`` / ``FormulaSerializeError``）
  * ``checks``   — 复杂度 / 安全 / 数据可用性等静态检查
  * ``scoring``  — 评分聚合（``score_factors_to_artifacts``）

顶层导出 ``api_models`` 中的核心 dataclass：``FactorSpec``、``FactorMeta``、
``FactorConstraints``、``FactorTest``、``ExprNode`` + ``write_factor_spec_schema``。
"""
from __future__ import annotations

from .api_models import (
    ExprNode,
    FactorConstraints,
    FactorMeta,
    FactorSpec,
    FactorTest,
    write_factor_spec_schema,
)

__all__ = [
    "ExprNode",
    "FactorConstraints",
    "FactorMeta",
    "FactorSpec",
    "FactorTest",
    "write_factor_spec_schema",
]

