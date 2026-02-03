"""Test type system functionality.

Tests:
- Primitive types
- Type compatibility
- Binary operator type checking
- Type serialization
"""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tools"))

from semantic_ir.ir_types import (
    IRPrimitiveType, BinaryOperator, UnaryOperator,
    StorageClass, VisibilityKind, MutabilityKind
)
from semantic_ir.nodes import PrimitiveTypeRef, TypeKind
from semantic_ir.validation import check_type_compatibility, check_binary_op_types, ValidationStatus


class TestPrimitiveTypes:
    """Test primitive type definitions."""

    def test_integer_types(self):
        """Test integer type enumeration."""
        assert IRPrimitiveType.I8 == "i8"
        assert IRPrimitiveType.I16 == "i16"
        assert IRPrimitiveType.I32 == "i32"
        assert IRPrimitiveType.I64 == "i64"
        assert IRPrimitiveType.U8 == "u8"
        assert IRPrimitiveType.U16 == "u16"
        assert IRPrimitiveType.U32 == "u32"
        assert IRPrimitiveType.U64 == "u64"
    
    def test_float_types(self):
        """Test floating-point type enumeration."""
        assert IRPrimitiveType.F32 == "f32"
        assert IRPrimitiveType.F64 == "f64"
    
    def test_other_types(self):
        """Test other primitive types."""
        assert IRPrimitiveType.VOID == "void"
        assert IRPrimitiveType.BOOL == "bool"
        assert IRPrimitiveType.STRING == "string"
        assert IRPrimitiveType.CHAR == "char"


class TestTypeCompatibility:
    """Test type compatibility checking."""

    def test_same_primitive_types_compatible(self):
        """Test that same primitive types are compatible."""
        type_i32_1 = PrimitiveTypeRef(kind=TypeKind.PRIMITIVE, primitive=IRPrimitiveType.I32)
        type_i32_2 = PrimitiveTypeRef(kind=TypeKind.PRIMITIVE, primitive=IRPrimitiveType.I32)
        
        assert check_type_compatibility(type_i32_1, type_i32_2)
    
    def test_different_primitive_types_incompatible(self):
        """Test that different primitive types are incompatible."""
        type_i32 = PrimitiveTypeRef(kind=TypeKind.PRIMITIVE, primitive=IRPrimitiveType.I32)
        type_f32 = PrimitiveTypeRef(kind=TypeKind.PRIMITIVE, primitive=IRPrimitiveType.F32)
        
        assert not check_type_compatibility(type_i32, type_f32)


class TestBinaryOperatorTypes:
    """Test binary operator type checking."""

    def test_arithmetic_operators_with_numeric_types(self):
        """Test arithmetic operators with numeric types."""
        type_i32 = PrimitiveTypeRef(kind=TypeKind.PRIMITIVE, primitive=IRPrimitiveType.I32)
        type_f32 = PrimitiveTypeRef(kind=TypeKind.PRIMITIVE, primitive=IRPrimitiveType.F32)
        
        result = check_binary_op_types(BinaryOperator.ADD, type_i32, type_i32)
        assert result.status == ValidationStatus.VALID
        
        result = check_binary_op_types(BinaryOperator.MUL, type_f32, type_f32)
        assert result.status == ValidationStatus.VALID
    
    def test_logical_operators_with_bool_types(self):
        """Test logical operators with boolean types."""
        type_bool = PrimitiveTypeRef(kind=TypeKind.PRIMITIVE, primitive=IRPrimitiveType.BOOL)
        
        result = check_binary_op_types(BinaryOperator.AND, type_bool, type_bool)
        assert result.status == ValidationStatus.VALID
        
        result = check_binary_op_types(BinaryOperator.OR, type_bool, type_bool)
        assert result.status == ValidationStatus.VALID
    
    def test_logical_operators_with_non_bool_types_invalid(self):
        """Test logical operators with non-boolean types fail."""
        type_i32 = PrimitiveTypeRef(kind=TypeKind.PRIMITIVE, primitive=IRPrimitiveType.I32)
        
        result = check_binary_op_types(BinaryOperator.AND, type_i32, type_i32)
        assert result.status == ValidationStatus.INVALID
