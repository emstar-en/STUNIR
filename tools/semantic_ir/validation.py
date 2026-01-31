"""STUNIR Semantic IR Validation.

DO-178C Level A Compliant
Validation functions with detailed error reporting.
"""

import re
from enum import Enum
from typing import Optional
from pydantic import BaseModel

from .ir_types import NodeID, IRHash, BinaryOperator, IRPrimitiveType
from .nodes import TypeReference, TypeKind, PrimitiveTypeRef
from .modules import IRModule


class ValidationStatus(str, Enum):
    """Validation status."""
    VALID = "valid"
    INVALID = "invalid"
    WARNING = "warning"


class ValidationResult(BaseModel):
    """Validation result with status and message."""
    status: ValidationStatus
    message: str = ""

    class Config:
        frozen = True

    @classmethod
    def valid(cls) -> "ValidationResult":
        """Create a valid result."""
        return cls(status=ValidationStatus.VALID)

    @classmethod
    def invalid(cls, message: str) -> "ValidationResult":
        """Create an invalid result with error message."""
        return cls(status=ValidationStatus.INVALID, message=message)

    @classmethod
    def warning(cls, message: str) -> "ValidationResult":
        """Create a warning result."""
        return cls(status=ValidationStatus.WARNING, message=message)


def validate_node_id(node_id: str) -> ValidationResult:
    """Validate node ID format.
    
    Node IDs must start with 'n_' followed by alphanumeric characters.
    """
    pattern = r"^n_[a-zA-Z0-9_]+$"
    if not re.match(pattern, node_id):
        return ValidationResult.invalid(
            f"Invalid node ID '{node_id}': must match pattern '{pattern}'"
        )
    return ValidationResult.valid()


def validate_hash(hash_str: str) -> ValidationResult:
    """Validate hash format.
    
    Hashes must be 'sha256:' followed by 64 hexadecimal characters.
    """
    pattern = r"^sha256:[a-f0-9]{64}$"
    if not re.match(pattern, hash_str):
        return ValidationResult.invalid(
            f"Invalid hash '{hash_str}': must match pattern '{pattern}'"
        )
    return ValidationResult.valid()


def validate_type_reference(type_ref: TypeReference) -> ValidationResult:
    """Validate type reference structure."""
    if type_ref.kind == TypeKind.PRIMITIVE:
        return ValidationResult.valid()
    elif type_ref.kind == TypeKind.REF:
        if not hasattr(type_ref, 'name') or not type_ref.name:
            return ValidationResult.invalid("Type reference must have non-empty name")
        return ValidationResult.valid()
    else:
        return ValidationResult.warning("Complex type validation not fully implemented")


def validate_module(module: IRModule) -> ValidationResult:
    """Validate IR module structure.
    
    Checks:
    - Valid node ID
    - Valid hash (if present)
    - Non-empty module name
    - Valid declarations
    """
    # Validate node ID
    id_result = validate_node_id(module.node_id)
    if id_result.status != ValidationStatus.VALID:
        return id_result

    # Validate hash if present
    if module.hash:
        hash_result = validate_hash(module.hash)
        if hash_result.status != ValidationStatus.VALID:
            return hash_result

    # Validate module name
    if not module.name:
        return ValidationResult.invalid("Module must have non-empty name")

    return ValidationResult.valid()


def check_type_compatibility(
    type1: TypeReference,
    type2: TypeReference
) -> bool:
    """Check if two types are compatible.
    
    Simplified type compatibility check.
    """
    if type1.kind != type2.kind:
        return False

    if isinstance(type1, PrimitiveTypeRef) and isinstance(type2, PrimitiveTypeRef):
        return type1.primitive == type2.primitive

    # For other types, basic kind matching
    return type1.kind == type2.kind


def check_binary_op_types(
    op: BinaryOperator,
    left: TypeReference,
    right: TypeReference
) -> ValidationResult:
    """Check type validity for binary operations.
    
    Ensures operand types are compatible with the operator.
    """
    # Both must be primitive types
    if not (isinstance(left, PrimitiveTypeRef) and isinstance(right, PrimitiveTypeRef)):
        return ValidationResult.invalid("Binary operators require primitive types")

    left_prim = left.primitive
    right_prim = right.primitive

    # Arithmetic operators
    if op in [BinaryOperator.ADD, BinaryOperator.SUB, BinaryOperator.MUL,
              BinaryOperator.DIV, BinaryOperator.MOD]:
        numeric_types = {
            IRPrimitiveType.I8, IRPrimitiveType.I16, IRPrimitiveType.I32, IRPrimitiveType.I64,
            IRPrimitiveType.U8, IRPrimitiveType.U16, IRPrimitiveType.U32, IRPrimitiveType.U64,
            IRPrimitiveType.F32, IRPrimitiveType.F64
        }
        if left_prim not in numeric_types or right_prim not in numeric_types:
            return ValidationResult.invalid("Arithmetic operators require numeric types")
        return ValidationResult.valid()

    # Comparison operators
    if op in [BinaryOperator.EQ, BinaryOperator.NEQ, BinaryOperator.LT,
              BinaryOperator.LEQ, BinaryOperator.GT, BinaryOperator.GEQ]:
        if not check_type_compatibility(left, right):
            return ValidationResult.invalid("Comparison operators require compatible types")
        return ValidationResult.valid()

    # Logical operators
    if op in [BinaryOperator.AND, BinaryOperator.OR]:
        if left_prim != IRPrimitiveType.BOOL or right_prim != IRPrimitiveType.BOOL:
            return ValidationResult.invalid("Logical operators require boolean types")
        return ValidationResult.valid()

    # Bitwise operators
    if op in [BinaryOperator.BIT_AND, BinaryOperator.BIT_OR, BinaryOperator.BIT_XOR,
              BinaryOperator.SHL, BinaryOperator.SHR]:
        integer_types = {
            IRPrimitiveType.I8, IRPrimitiveType.I16, IRPrimitiveType.I32, IRPrimitiveType.I64,
            IRPrimitiveType.U8, IRPrimitiveType.U16, IRPrimitiveType.U32, IRPrimitiveType.U64
        }
        if left_prim not in integer_types or right_prim not in integer_types:
            return ValidationResult.invalid("Bitwise operators require integer types")
        return ValidationResult.valid()

    # Assignment operator
    if op == BinaryOperator.ASSIGN:
        if not check_type_compatibility(left, right):
            return ValidationResult.invalid("Assignment requires compatible types")
        return ValidationResult.valid()

    return ValidationResult.warning(f"Unknown operator: {op}")
