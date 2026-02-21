"""STUNIR Semantic IR Statement Nodes.

DO-178C Level A Compliant
Statement node definitions.
"""

from typing import List, Optional, Union
from pydantic import BaseModel, Field, conint

from .ir_types import (
    NodeID, IRName, IRNodeKind,
    StorageClass, MutabilityKind
)
from .nodes import IRNodeBase, TypeReference


class StatementNode(IRNodeBase):
    """Base statement node."""
    pass


class BlockStmt(StatementNode):
    """Block statement."""
    kind: IRNodeKind = IRNodeKind.BLOCK_STMT
    statements: List[Union[NodeID, "StatementNode"]] = Field(default_factory=list)
    scope_id: Optional[IRName] = None


class ExprStmt(StatementNode):
    """Expression statement."""
    kind: IRNodeKind = IRNodeKind.EXPR_STMT
    expression: Union[NodeID, "ExpressionNode"]


class IfStmt(StatementNode):
    """If statement."""
    kind: IRNodeKind = IRNodeKind.IF_STMT
    condition: Union[NodeID, "ExpressionNode"]
    then_branch: Union[NodeID, "StatementNode"]
    else_branch: Optional[Union[NodeID, "StatementNode"]] = None


class WhileStmt(StatementNode):
    """While loop statement."""
    kind: IRNodeKind = IRNodeKind.WHILE_STMT
    condition: Union[NodeID, "ExpressionNode"]
    body: Union[NodeID, "StatementNode"]
    loop_bound: conint(ge=0) = 0  # 0 means unbounded
    unroll: bool = False


class ForStmt(StatementNode):
    """For loop statement."""
    kind: IRNodeKind = IRNodeKind.FOR_STMT
    init: Union[NodeID, "StatementNode"]
    condition: Union[NodeID, "ExpressionNode"]
    increment: Union[NodeID, "StatementNode"]
    body: Union[NodeID, "StatementNode"]
    loop_bound: conint(ge=0) = 0
    unroll: bool = False
    vectorize: bool = False


class ReturnStmt(StatementNode):
    """Return statement."""
    kind: IRNodeKind = IRNodeKind.RETURN_STMT
    value: Optional[Union[NodeID, "ExpressionNode"]] = None


class BreakStmt(StatementNode):
    """Break statement."""
    kind: IRNodeKind = IRNodeKind.BREAK_STMT


class ContinueStmt(StatementNode):
    """Continue statement."""
    kind: IRNodeKind = IRNodeKind.CONTINUE_STMT


class VarDeclStmt(StatementNode):
    """Variable declaration statement."""
    kind: IRNodeKind = IRNodeKind.VAR_DECL_STMT
    name: IRName
    var_type: Optional[TypeReference] = None
    initializer: Optional[Union[NodeID, "ExpressionNode"]] = None
    storage: StorageClass = StorageClass.AUTO
    mutability: MutabilityKind = MutabilityKind.MUTABLE


class AssignStmt(StatementNode):
    """Assignment statement."""
    kind: IRNodeKind = IRNodeKind.ASSIGN_STMT
    target: Union[NodeID, "ExpressionNode"]
    value: Union[NodeID, "ExpressionNode"]


# Enable forward references
from .expressions import ExpressionNode  # noqa: E402

BlockStmt.model_rebuild()
ExprStmt.model_rebuild()
IfStmt.model_rebuild()
WhileStmt.model_rebuild()
ForStmt.model_rebuild()
ReturnStmt.model_rebuild()
VarDeclStmt.model_rebuild()
AssignStmt.model_rebuild()
