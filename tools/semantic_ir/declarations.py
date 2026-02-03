"""STUNIR Semantic IR Declaration Nodes.

DO-178C Level A Compliant
Declaration node definitions.
"""

from typing import List, Optional, Union
from pydantic import BaseModel, Field, conint

from .ir_types import (
    NodeID, IRName, IRNodeKind,
    StorageClass, MutabilityKind,
    VisibilityKind, InlineHint
)
from .nodes import IRNodeBase, TypeReference


class DeclarationNode(IRNodeBase):
    """Base declaration node."""
    name: IRName
    visibility: VisibilityKind = VisibilityKind.PUBLIC


class Parameter(BaseModel):
    """Function parameter."""
    name: IRName
    param_type: TypeReference
    default: Optional[Union[NodeID, "ExpressionNode"]] = None


class FunctionDecl(DeclarationNode):
    """Function declaration."""
    kind: IRNodeKind = IRNodeKind.FUNCTION_DECL
    return_type: TypeReference
    parameters: List[Parameter] = Field(default_factory=list)
    body: Optional[Union[NodeID, "StatementNode"]] = None
    inline: InlineHint = InlineHint.NONE
    is_pure: bool = False
    stack_usage: conint(ge=0) = 0
    priority: int = 0
    interrupt_vector: conint(ge=0) = 0  # 0 means not an interrupt handler
    entry_point: bool = False


class TypeDecl(DeclarationNode):
    """Type declaration."""
    kind: IRNodeKind = IRNodeKind.TYPE_DECL
    type_definition: TypeReference


class ConstDecl(DeclarationNode):
    """Constant declaration."""
    kind: IRNodeKind = IRNodeKind.CONST_DECL
    const_type: Optional[TypeReference] = None
    value: Union[NodeID, "ExpressionNode"]
    compile_time: bool = True


class VarDecl(DeclarationNode):
    """Variable declaration."""
    kind: IRNodeKind = IRNodeKind.VAR_DECL
    var_type: Optional[TypeReference] = None
    initializer: Optional[Union[NodeID, "ExpressionNode"]] = None
    storage: StorageClass = StorageClass.AUTO
    mutability: MutabilityKind = MutabilityKind.MUTABLE
    alignment: conint(ge=1) = 1
    is_volatile: bool = False


# Enable forward references
from .expressions import ExpressionNode  # noqa: E402
from .statements import StatementNode  # noqa: E402

Parameter.model_rebuild()
FunctionDecl.model_rebuild()
ConstDecl.model_rebuild()
VarDecl.model_rebuild()
