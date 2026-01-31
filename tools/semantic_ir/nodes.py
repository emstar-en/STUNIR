"""STUNIR Semantic IR Node Structures.

DO-178C Level A Compliant
Base node types and type references.
"""

from enum import Enum
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

from .ir_types import NodeID, IRHash, IRName, SourceLocation, IRNodeKind, IRPrimitiveType


class TypeKind(str, Enum):
    """Type kind discriminator."""
    PRIMITIVE = "primitive_type"
    ARRAY = "array_type"
    POINTER = "pointer_type"
    STRUCT = "struct_type"
    FUNCTION = "function_type"
    REF = "type_ref"


class TypeReference(BaseModel):
    """Base type reference."""
    kind: TypeKind

    class Config:
        use_enum_values = True


class PrimitiveTypeRef(TypeReference):
    """Primitive type reference."""
    kind: TypeKind = TypeKind.PRIMITIVE
    primitive: IRPrimitiveType


class TypeRef(TypeReference):
    """Named type reference."""
    kind: TypeKind = TypeKind.REF
    name: IRName
    binding: Optional[NodeID] = None


class IRNodeBase(BaseModel):
    """Base IR node structure."""
    node_id: NodeID
    kind: IRNodeKind
    location: Optional[SourceLocation] = None
    type: Optional[TypeReference] = None
    attributes: Dict[str, Any] = Field(default_factory=dict)
    hash: Optional[IRHash] = None

    class Config:
        use_enum_values = True
        arbitrary_types_allowed = True
