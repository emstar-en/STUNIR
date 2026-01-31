"""STUNIR Semantic IR Expression Nodes.

DO-178C Level A Compliant
Expression node definitions.
"""

from typing import List, Optional, Union
from pydantic import BaseModel, Field

from .ir_types import (
    NodeID, IRName, IRNodeKind,
    BinaryOperator, UnaryOperator
)
from .nodes import IRNodeBase, TypeReference


class ExpressionNode(IRNodeBase):
    """Base expression node."""
    type: TypeReference  # Required for expressions


class IntegerLiteral(ExpressionNode):
    """Integer literal expression."""
    kind: IRNodeKind = IRNodeKind.INTEGER_LITERAL
    value: int
    radix: int = Field(default=10, ge=2, le=16)


class FloatLiteral(ExpressionNode):
    """Float literal expression."""
    kind: IRNodeKind = IRNodeKind.FLOAT_LITERAL
    value: float


class StringLiteral(ExpressionNode):
    """String literal expression."""
    kind: IRNodeKind = IRNodeKind.STRING_LITERAL
    value: str


class BoolLiteral(ExpressionNode):
    """Boolean literal expression."""
    kind: IRNodeKind = IRNodeKind.BOOL_LITERAL
    value: bool


class VarRef(ExpressionNode):
    """Variable reference expression."""
    kind: IRNodeKind = IRNodeKind.VAR_REF
    name: IRName
    binding: Optional[NodeID] = None


class BinaryExpr(ExpressionNode):
    """Binary operation expression."""
    kind: IRNodeKind = IRNodeKind.BINARY_EXPR
    op: BinaryOperator
    left: Union[NodeID, "ExpressionNode"]
    right: Union[NodeID, "ExpressionNode"]


class UnaryExpr(ExpressionNode):
    """Unary operation expression."""
    kind: IRNodeKind = IRNodeKind.UNARY_EXPR
    op: UnaryOperator
    operand: Union[NodeID, "ExpressionNode"]


class FunctionCall(ExpressionNode):
    """Function call expression."""
    kind: IRNodeKind = IRNodeKind.FUNCTION_CALL
    function: Union[IRName, NodeID]
    arguments: List[Union[NodeID, "ExpressionNode"]] = Field(default_factory=list)


class MemberExpr(ExpressionNode):
    """Member access expression."""
    kind: IRNodeKind = IRNodeKind.MEMBER_EXPR
    object: Union[NodeID, "ExpressionNode"]
    member: IRName
    is_arrow: bool = False


class ArrayAccess(ExpressionNode):
    """Array access expression."""
    kind: IRNodeKind = IRNodeKind.ARRAY_ACCESS
    array: Union[NodeID, "ExpressionNode"]
    index: Union[NodeID, "ExpressionNode"]


class CastExpr(ExpressionNode):
    """Cast expression."""
    kind: IRNodeKind = IRNodeKind.CAST_EXPR
    operand: Union[NodeID, "ExpressionNode"]
    target_type: TypeReference


class TernaryExpr(ExpressionNode):
    """Ternary conditional expression."""
    kind: IRNodeKind = IRNodeKind.TERNARY_EXPR
    condition: Union[NodeID, "ExpressionNode"]
    then_expr: Union[NodeID, "ExpressionNode"]
    else_expr: Union[NodeID, "ExpressionNode"]


# Enable forward references
BinaryExpr.model_rebuild()
UnaryExpr.model_rebuild()
FunctionCall.model_rebuild()
MemberExpr.model_rebuild()
ArrayAccess.model_rebuild()
CastExpr.model_rebuild()
TernaryExpr.model_rebuild()
