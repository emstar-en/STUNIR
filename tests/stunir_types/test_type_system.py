#!/usr/bin/env python3
"""Comprehensive tests for STUNIR type system - corrected version.

Tests all type system functionality to achieve high coverage.
"""

import pytest
from tools.stunir_types.type_system import (
    TypeKind, Ownership, Mutability, Lifetime,
    STUNIRType, VoidType, UnitType, BoolType, IntType, FloatType, CharType,
    StringType, PointerType, ReferenceType, ArrayType, SliceType,
    StructType, UnionType, EnumType, TaggedUnionType, FunctionType,
    OptionalType, ResultType, TupleType, RecursiveType, TypeVar, GenericType,
    StructField, EnumVariant, TaggedVariant
)


class TestTypeKind:
    """Test TypeKind enum."""
    
    def test_type_kind_values(self):
        """Test that all type kinds are defined."""
        assert TypeKind.VOID
        assert TypeKind.BOOL
        assert TypeKind.INT
        assert TypeKind.FLOAT
        assert TypeKind.CHAR
        assert TypeKind.STRING
        assert TypeKind.POINTER
        assert TypeKind.REFERENCE
        assert TypeKind.ARRAY
        assert TypeKind.SLICE
        assert TypeKind.STRUCT
        assert TypeKind.UNION
        assert TypeKind.ENUM
        assert TypeKind.TAGGED_UNION
        assert TypeKind.FUNCTION
        assert TypeKind.CLOSURE
        assert TypeKind.GENERIC
        assert TypeKind.TYPE_VAR
        assert TypeKind.OPAQUE
        assert TypeKind.RECURSIVE
        assert TypeKind.OPTIONAL
        assert TypeKind.RESULT
        assert TypeKind.TUPLE
        assert TypeKind.UNIT


class TestOwnership:
    """Test Ownership enum."""
    
    def test_ownership_values(self):
        """Test ownership semantics."""
        assert Ownership.OWNED
        assert Ownership.BORROWED
        assert Ownership.BORROWED_MUT
        assert Ownership.COPY
        assert Ownership.STATIC


class TestMutability:
    """Test Mutability enum."""
    
    def test_mutability_values(self):
        """Test mutability values."""
        assert Mutability.IMMUTABLE
        assert Mutability.MUTABLE
        assert Mutability.CONST


class TestLifetime:
    """Test Lifetime dataclass."""
    
    def test_lifetime_default(self):
        """Test default lifetime."""
        lt = Lifetime()
        assert lt.name == "'a"
        assert not lt.is_static
        assert str(lt) == "'a"
    
    def test_lifetime_custom(self):
        """Test custom lifetime."""
        lt = Lifetime(name="'b")
        assert lt.name == "'b"
        assert str(lt) == "'b"
    
    def test_lifetime_static(self):
        """Test static lifetime."""
        lt = Lifetime(is_static=True)
        assert lt.is_static
        assert str(lt) == "'static"


class TestVoidType:
    """Test VoidType."""
    
    def test_void_type_kind(self):
        """Test void type kind."""
        void = VoidType()
        assert void.kind == TypeKind.VOID
    
    def test_void_type_ir(self):
        """Test void type IR representation."""
        void = VoidType()
        assert void.to_ir() == {'type': 'void'}
    
    def test_void_type_str(self):
        """Test void type string representation."""
        void = VoidType()
        assert str(void) == 'void'
    
    def test_void_is_primitive(self):
        """Test that void is primitive."""
        void = VoidType()
        assert void.is_primitive()
        assert not void.is_pointer_like()
        assert not void.is_compound()


class TestUnitType:
    """Test UnitType."""
    
    def test_unit_type_kind(self):
        """Test unit type kind."""
        unit = UnitType()
        assert unit.kind == TypeKind.UNIT
    
    def test_unit_type_ir(self):
        """Test unit type IR representation."""
        unit = UnitType()
        assert unit.to_ir() == {'type': 'unit'}
    
    def test_unit_type_str(self):
        """Test unit type string representation."""
        unit = UnitType()
        assert str(unit) == '()'
    
    def test_unit_is_primitive(self):
        """Test that unit is primitive."""
        unit = UnitType()
        assert unit.is_primitive()


class TestBoolType:
    """Test BoolType."""
    
    def test_bool_type_kind(self):
        """Test bool type kind."""
        bool_type = BoolType()
        assert bool_type.kind == TypeKind.BOOL
    
    def test_bool_type_ir(self):
        """Test bool type IR representation."""
        bool_type = BoolType()
        assert bool_type.to_ir() == {'type': 'bool'}
    
    def test_bool_type_str(self):
        """Test bool type string representation."""
        bool_type = BoolType()
        assert str(bool_type) == 'bool'
    
    def test_bool_is_primitive(self):
        """Test that bool is primitive."""
        bool_type = BoolType()
        assert bool_type.is_primitive()


class TestIntType:
    """Test IntType."""
    
    def test_int_type_default(self):
        """Test default int type (i32)."""
        int_type = IntType()
        assert int_type.kind == TypeKind.INT
        assert int_type.bits == 32
        assert int_type.signed
        assert str(int_type) == 'i32'
    
    def test_int_type_i64(self):
        """Test 64-bit signed int."""
        int_type = IntType(bits=64, signed=True)
        assert int_type.bits == 64
        assert int_type.signed
        assert str(int_type) == 'i64'
    
    def test_int_type_u32(self):
        """Test 32-bit unsigned int."""
        int_type = IntType(bits=32, signed=False)
        assert not int_type.signed
        assert str(int_type) == 'u32'
    
    def test_int_type_u8(self):
        """Test 8-bit unsigned int."""
        int_type = IntType(bits=8, signed=False)
        assert int_type.bits == 8
        assert str(int_type) == 'u8'
    
    def test_int_type_ir(self):
        """Test int type IR representation."""
        int_type = IntType(bits=16, signed=True)
        ir = int_type.to_ir()
        assert ir == {'type': 'int', 'bits': 16, 'signed': True}
    
    def test_int_is_primitive(self):
        """Test that int is primitive."""
        int_type = IntType()
        assert int_type.is_primitive()


class TestFloatType:
    """Test FloatType."""
    
    def test_float_type_default(self):
        """Test default float type (f64)."""
        float_type = FloatType()
        assert float_type.kind == TypeKind.FLOAT
        assert float_type.bits == 64
        assert str(float_type) == 'f64'
    
    def test_float_type_f32(self):
        """Test 32-bit float."""
        float_type = FloatType(bits=32)
        assert float_type.bits == 32
        assert str(float_type) == 'f32'
    
    def test_float_type_ir(self):
        """Test float type IR representation."""
        float_type = FloatType(bits=32)
        ir = float_type.to_ir()
        assert ir == {'type': 'float', 'bits': 32}
    
    def test_float_is_primitive(self):
        """Test that float is primitive."""
        float_type = FloatType()
        assert float_type.is_primitive()


class TestCharType:
    """Test CharType."""
    
    def test_char_type_default(self):
        """Test default char type (unicode)."""
        char_type = CharType()
        assert char_type.kind == TypeKind.CHAR
        assert char_type.unicode
    
    def test_char_type_ascii(self):
        """Test ASCII char type."""
        char_type = CharType(unicode=False)
        assert not char_type.unicode
    
    def test_char_type_ir(self):
        """Test char type IR representation."""
        char_type = CharType(unicode=True)
        ir = char_type.to_ir()
        assert 'type' in ir
    
    def test_char_is_primitive(self):
        """Test that char is primitive."""
        char_type = CharType()
        assert char_type.is_primitive()


class TestStringType:
    """Test StringType."""
    
    def test_string_type_kind(self):
        """Test string type kind."""
        string_type = StringType()
        assert string_type.kind == TypeKind.STRING
    
    def test_string_type_owned(self):
        """Test owned string."""
        string_type = StringType(owned=True)
        assert string_type.owned
        assert str(string_type) == 'String'
    
    def test_string_type_borrowed(self):
        """Test borrowed string."""
        string_type = StringType(owned=False)
        assert not string_type.owned
        assert str(string_type) == '&str'
    
    def test_string_type_ir(self):
        """Test string type IR representation."""
        string_type = StringType(owned=True)
        ir = string_type.to_ir()
        assert ir['type'] == 'string'
        assert ir['owned'] == True


class TestPointerType:
    """Test PointerType."""
    
    def test_pointer_to_int(self):
        """Test pointer to int."""
        int_type = IntType(bits=32, signed=True)
        ptr = PointerType(pointee=int_type)
        assert ptr.kind == TypeKind.POINTER
        assert ptr.pointee == int_type
    
    def test_pointer_mutability(self):
        """Test pointer mutability."""
        int_type = IntType()
        ptr_mut = PointerType(pointee=int_type, mutability=Mutability.MUTABLE)
        ptr_const = PointerType(pointee=int_type, mutability=Mutability.CONST)
        assert ptr_mut.mutability == Mutability.MUTABLE
        assert ptr_const.mutability == Mutability.CONST
    
    def test_pointer_nullable(self):
        """Test nullable pointer."""
        int_type = IntType()
        ptr_nullable = PointerType(pointee=int_type, nullable=True)
        ptr_nonnull = PointerType(pointee=int_type, nullable=False)
        assert ptr_nullable.nullable
        assert not ptr_nonnull.nullable
    
    def test_pointer_is_pointer_like(self):
        """Test that pointer is pointer-like."""
        ptr = PointerType(pointee=IntType())
        assert ptr.is_pointer_like()
        assert not ptr.is_primitive()
        assert not ptr.is_compound()
    
    def test_pointer_ir(self):
        """Test pointer IR representation."""
        int_type = IntType()
        ptr = PointerType(pointee=int_type)
        ir = ptr.to_ir()
        assert ir['type'] == 'pointer'


class TestReferenceType:
    """Test ReferenceType."""
    
    def test_reference_to_int(self):
        """Test reference to int."""
        int_type = IntType()
        ref = ReferenceType(referent=int_type)
        assert ref.kind == TypeKind.REFERENCE
        assert ref.referent == int_type
    
    def test_reference_mutability(self):
        """Test reference mutability."""
        int_type = IntType()
        ref_mut = ReferenceType(referent=int_type, mutability=Mutability.MUTABLE)
        ref_const = ReferenceType(referent=int_type, mutability=Mutability.IMMUTABLE)
        assert ref_mut.mutability == Mutability.MUTABLE
        assert ref_const.mutability == Mutability.IMMUTABLE
    
    def test_reference_lifetime(self):
        """Test reference with lifetime."""
        int_type = IntType()
        lifetime = Lifetime(name="'a")
        ref = ReferenceType(referent=int_type, lifetime=lifetime)
        assert ref.lifetime == lifetime
    
    def test_reference_is_pointer_like(self):
        """Test that reference is pointer-like."""
        ref = ReferenceType(referent=IntType())
        assert ref.is_pointer_like()


class TestArrayType:
    """Test ArrayType."""
    
    def test_array_fixed_size(self):
        """Test fixed-size array."""
        int_type = IntType()
        arr = ArrayType(element=int_type, size=10)
        assert arr.kind == TypeKind.ARRAY
        assert arr.element == int_type
        assert arr.size == 10
    
    def test_array_dynamic(self):
        """Test dynamic array."""
        int_type = IntType()
        arr = ArrayType(element=int_type, size=None)
        assert arr.size is None
    
    def test_array_ir(self):
        """Test array IR representation."""
        int_type = IntType()
        arr = ArrayType(element=int_type, size=5)
        ir = arr.to_ir()
        assert ir['type'] == 'array'


class TestSliceType:
    """Test SliceType."""
    
    def test_slice_type(self):
        """Test slice type."""
        int_type = IntType()
        slice_type = SliceType(element=int_type)
        assert slice_type.kind == TypeKind.SLICE
        assert slice_type.element == int_type
    
    def test_slice_mutable(self):
        """Test mutable slice."""
        int_type = IntType()
        slice_type = SliceType(element=int_type, mutability=Mutability.MUTABLE)
        assert slice_type.mutability == Mutability.MUTABLE
    
    def test_slice_lifetime(self):
        """Test slice with lifetime."""
        int_type = IntType()
        lifetime = Lifetime(name="'a")
        slice_type = SliceType(element=int_type, lifetime=lifetime)
        assert slice_type.lifetime == lifetime
    
    def test_slice_is_pointer_like(self):
        """Test that slice is pointer-like."""
        slice_type = SliceType(element=IntType())
        assert slice_type.is_pointer_like()


class TestStructType:
    """Test StructType."""
    
    def test_struct_basic(self):
        """Test basic struct."""
        fields = [
            StructField(name="x", type=IntType()),
            StructField(name="y", type=IntType())
        ]
        struct = StructType(name="Point", fields=fields)
        assert struct.kind == TypeKind.STRUCT
        assert struct.name == "Point"
        assert len(struct.fields) == 2
    
    def test_struct_packed(self):
        """Test packed struct."""
        fields = [StructField(name="a", type=IntType())]
        struct = StructType(name="Packed", fields=fields, packed=True)
        assert struct.packed
    
    def test_struct_generic(self):
        """Test generic struct."""
        fields = [StructField(name="value", type=IntType())]
        struct = StructType(name="Container", fields=fields, generics=["T"])
        assert len(struct.generics) == 1
        assert struct.generics[0] == "T"
    
    def test_struct_is_compound(self):
        """Test that struct is compound."""
        struct = StructType(name="S", fields=[])
        assert struct.is_compound()
        assert not struct.is_primitive()
        assert not struct.is_pointer_like()


class TestUnionType:
    """Test UnionType."""
    
    def test_union_basic(self):
        """Test basic union."""
        variants = [
            StructField(name="i", type=IntType()),
            StructField(name="f", type=FloatType())
        ]
        union = UnionType(name="Value", variants=variants)
        assert union.kind == TypeKind.UNION
        assert union.name == "Value"
        assert len(union.variants) == 2
    
    def test_union_is_compound(self):
        """Test that union is compound."""
        union = UnionType(name="U", variants=[])
        assert union.is_compound()


class TestEnumType:
    """Test EnumType."""
    
    def test_enum_basic(self):
        """Test basic enum."""
        variants = [
            EnumVariant(name="Red"),
            EnumVariant(name="Green"),
            EnumVariant(name="Blue")
        ]
        enum = EnumType(name="Color", variants=variants)
        assert enum.kind == TypeKind.ENUM
        assert enum.name == "Color"
        assert len(enum.variants) == 3
    
    def test_enum_with_values(self):
        """Test enum with explicit values."""
        variants = [
            EnumVariant(name="A", value=1),
            EnumVariant(name="B", value=2)
        ]
        enum = EnumType(name="E", variants=variants)
        assert enum.variants[0].value == 1
        assert enum.variants[1].value == 2
    
    def test_enum_is_compound(self):
        """Test that enum is compound."""
        enum = EnumType(name="E", variants=[])
        assert enum.is_compound()


class TestTaggedUnionType:
    """Test TaggedUnionType (Rust enum, Haskell ADT)."""
    
    def test_tagged_union_basic(self):
        """Test basic tagged union."""
        variants = [
            TaggedVariant(name="None"),
            TaggedVariant(name="Some", fields=[StructField(name="value", type=IntType())])
        ]
        tagged = TaggedUnionType(name="Option", variants=variants)
        assert tagged.kind == TypeKind.TAGGED_UNION
        assert tagged.name == "Option"
    
    def test_tagged_union_generic(self):
        """Test generic tagged union."""
        variants = [TaggedVariant(name="None")]
        tagged = TaggedUnionType(name="Option", variants=variants, generics=["T"])
        assert len(tagged.generics) == 1
    
    def test_tagged_union_is_compound(self):
        """Test that tagged union is compound."""
        tagged = TaggedUnionType(name="T", variants=[])
        assert tagged.is_compound()


class TestFunctionType:
    """Test FunctionType."""
    
    def test_function_basic(self):
        """Test basic function type."""
        params = [IntType(), IntType()]
        return_type = IntType()
        func = FunctionType(params=params, returns=return_type)
        assert func.kind == TypeKind.FUNCTION
        assert len(func.params) == 2
        assert func.returns == return_type
    
    def test_function_variadic(self):
        """Test variadic function."""
        func = FunctionType(params=[IntType()], returns=VoidType(), variadic=True)
        assert func.variadic
    
    def test_function_void_return(self):
        """Test function returning void."""
        func = FunctionType(params=[], returns=VoidType())
        assert func.returns.kind == TypeKind.VOID


class TestOptionalType:
    """Test OptionalType."""
    
    def test_optional_basic(self):
        """Test optional type."""
        optional = OptionalType(inner=IntType())
        assert optional.kind == TypeKind.OPTIONAL
        assert isinstance(optional.inner, IntType)
    
    def test_optional_nested(self):
        """Test nested optional."""
        inner_optional = OptionalType(inner=IntType())
        outer_optional = OptionalType(inner=inner_optional)
        assert outer_optional.inner.kind == TypeKind.OPTIONAL


class TestResultType:
    """Test ResultType."""
    
    def test_result_basic(self):
        """Test result type."""
        result = ResultType(ok_type=IntType(), err_type=StringType())
        assert result.kind == TypeKind.RESULT
        assert isinstance(result.ok_type, IntType)
        assert isinstance(result.err_type, StringType)
    
    def test_result_with_unit(self):
        """Test result with unit error type."""
        result = ResultType(ok_type=IntType(), err_type=UnitType())
        assert result.err_type.kind == TypeKind.UNIT


class TestTupleType:
    """Test TupleType."""
    
    def test_tuple_basic(self):
        """Test tuple type."""
        elements = [IntType(), FloatType(), BoolType()]
        tuple_type = TupleType(elements=elements)
        assert tuple_type.kind == TypeKind.TUPLE
        assert len(tuple_type.elements) == 3
    
    def test_tuple_empty(self):
        """Test empty tuple (equivalent to unit)."""
        tuple_type = TupleType(elements=[])
        assert tuple_type.kind == TypeKind.TUPLE
        assert len(tuple_type.elements) == 0
    
    def test_tuple_is_compound(self):
        """Test that tuple is compound."""
        tuple_type = TupleType(elements=[IntType()])
        assert tuple_type.is_compound()


class TestTypeVar:
    """Test TypeVar (generic type variable)."""
    
    def test_type_var_basic(self):
        """Test type variable."""
        type_var = TypeVar(name="T")
        assert type_var.kind == TypeKind.TYPE_VAR
        assert type_var.name == "T"
    
    def test_type_var_with_constraints(self):
        """Test type variable with constraints."""
        constraints = ["Ord", "Clone"]
        type_var = TypeVar(name="T", constraints=constraints)
        assert len(type_var.constraints) == 2


class TestGenericType:
    """Test GenericType."""
    
    def test_generic_basic(self):
        """Test generic type."""
        type_args = [IntType()]
        generic = GenericType(base="Vec", args=type_args)
        assert generic.kind == TypeKind.GENERIC
        assert generic.base == "Vec"
        assert len(generic.args) == 1
    
    def test_generic_multiple_args(self):
        """Test generic with multiple type arguments."""
        type_args = [IntType(), StringType()]
        generic = GenericType(base="HashMap", args=type_args)
        assert len(generic.args) == 2


class TestRecursiveType:
    """Test RecursiveType."""
    
    def test_recursive_basic(self):
        """Test recursive type."""
        recursive = RecursiveType(name="List")
        assert recursive.kind == TypeKind.RECURSIVE
        assert recursive.name == "List"
    
    def test_recursive_with_inner(self):
        """Test recursive type with inner type."""
        inner = OptionalType(inner=IntType())
        recursive = RecursiveType(name="Tree", inner=inner)
        assert recursive.inner is not None


class TestStructField:
    """Test StructField dataclass."""
    
    def test_field_basic(self):
        """Test basic field."""
        field = StructField(name="x", type=IntType())
        assert field.name == "x"
        assert isinstance(field.type, IntType)
    
    def test_field_with_visibility(self):
        """Test field with visibility."""
        field = StructField(name="x", type=IntType(), visibility="private")
        assert field.visibility == "private"
    
    def test_field_with_offset(self):
        """Test field with offset."""
        field = StructField(name="x", type=IntType(), offset=4)
        assert field.offset == 4


class TestEnumVariant:
    """Test EnumVariant dataclass."""
    
    def test_enum_variant_basic(self):
        """Test basic enum variant."""
        variant = EnumVariant(name="A")
        assert variant.name == "A"
        assert variant.value is None
    
    def test_enum_variant_with_value(self):
        """Test enum variant with value."""
        variant = EnumVariant(name="B", value=42)
        assert variant.value == 42


class TestTaggedVariant:
    """Test TaggedVariant dataclass."""
    
    def test_tagged_variant_basic(self):
        """Test basic tagged variant."""
        variant = TaggedVariant(name="A")
        assert variant.name == "A"
    
    def test_tagged_variant_with_fields(self):
        """Test tagged variant with fields."""
        fields = [StructField(name="x", type=IntType())]
        variant = TaggedVariant(name="C", fields=fields)
        assert len(variant.fields) == 1


# Integration tests
class TestTypeSystemIntegration:
    """Integration tests for type system."""
    
    def test_complex_nested_type(self):
        """Test complex nested type structure."""
        # Vec<Option<&'a mut i32>>
        inner = IntType(bits=32, signed=True)
        lifetime = Lifetime(name="'a")
        ref = ReferenceType(referent=inner, mutability=Mutability.MUTABLE, lifetime=lifetime)
        optional = OptionalType(inner=ref)
        vec = GenericType(base="Vec", args=[optional])
        
        assert vec.kind == TypeKind.GENERIC
        assert optional.kind == TypeKind.OPTIONAL
        assert ref.kind == TypeKind.REFERENCE
    
    def test_function_with_complex_types(self):
        """Test function with complex parameter and return types."""
        param1 = PointerType(pointee=StructType(name="S", fields=[]))
        param2 = ArrayType(element=IntType(), size=10)
        return_type = ResultType(ok_type=BoolType(), err_type=StringType())
        
        func = FunctionType(params=[param1, param2], returns=return_type)
        assert len(func.params) == 2
        assert func.returns.kind == TypeKind.RESULT
    
    def test_recursive_data_structure(self):
        """Test recursive data structure (e.g., linked list)."""
        # struct Node { value: i32, next: Option<Box<Node>> }
        node_recursive = RecursiveType(name="Node")
        
        # Build fields
        value_field = StructField(name="value", type=IntType())
        # In practice, you'd need proper handling of recursive types
        next_field = StructField(name="next", type=OptionalType(inner=node_recursive))
        
        assert node_recursive.kind == TypeKind.RECURSIVE
        assert next_field.type.kind == TypeKind.OPTIONAL


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
