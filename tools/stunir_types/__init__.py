"""STUNIR Type System Module.

Provides a comprehensive type system for cross-language code generation,
including type mapping, type inference, and type checking.
"""

from .type_system import (
    # Enums
    TypeKind, Ownership, Mutability, Lifetime,
    # Base class
    STUNIRType,
    # Type classes
    VoidType, UnitType, BoolType, IntType, FloatType, CharType, StringType,
    PointerType, ReferenceType, ArrayType, SliceType,
    StructType, StructField, UnionType,
    EnumType, EnumVariant, TaggedUnionType, TaggedVariant,
    FunctionType, ClosureType,
    GenericType, TypeVar, OpaqueType, RecursiveType,
    OptionalType, ResultType, TupleType,
    # Registry and parser
    TypeRegistry, parse_type,
    # Common instances
    VOID, UNIT, BOOL,
    I8, I16, I32, I64, U8, U16, U32, U64,
    F32, F64, CHAR, STRING, STR
)

from .type_mapper import (
    TargetLanguage, MappedType, TypeMapper, create_type_mapper
)

from .type_inference import (
    TypeErrorKind, TypeError, TypeConstraint,
    TypeScope, TypeInferenceEngine, TypeChecker
)

__all__ = [
    # Type system
    'TypeKind', 'Ownership', 'Mutability', 'Lifetime',
    'STUNIRType',
    'VoidType', 'UnitType', 'BoolType', 'IntType', 'FloatType', 'CharType', 'StringType',
    'PointerType', 'ReferenceType', 'ArrayType', 'SliceType',
    'StructType', 'StructField', 'UnionType',
    'EnumType', 'EnumVariant', 'TaggedUnionType', 'TaggedVariant',
    'FunctionType', 'ClosureType',
    'GenericType', 'TypeVar', 'OpaqueType', 'RecursiveType',
    'OptionalType', 'ResultType', 'TupleType',
    'TypeRegistry', 'parse_type',
    'VOID', 'UNIT', 'BOOL',
    'I8', 'I16', 'I32', 'I64', 'U8', 'U16', 'U32', 'U64',
    'F32', 'F64', 'CHAR', 'STRING', 'STR',
    # Type mapper
    'TargetLanguage', 'MappedType', 'TypeMapper', 'create_type_mapper',
    # Type inference
    'TypeErrorKind', 'TypeError', 'TypeConstraint',
    'TypeScope', 'TypeInferenceEngine', 'TypeChecker'
]
