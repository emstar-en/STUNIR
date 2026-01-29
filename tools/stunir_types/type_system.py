#!/usr/bin/env python3
"""STUNIR Type System Module.

Provides a comprehensive type system for cross-language code generation,
supporting complex types including pointers, arrays, structs, enums,
function pointers, generics, and recursive types.

This module is part of the STUNIR code generation enhancement suite.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Any, Tuple, Union


class TypeKind(Enum):
    """Kinds of types in the STUNIR type system."""
    VOID = auto()
    BOOL = auto()
    INT = auto()
    FLOAT = auto()
    CHAR = auto()
    STRING = auto()
    POINTER = auto()
    REFERENCE = auto()
    ARRAY = auto()
    SLICE = auto()
    STRUCT = auto()
    UNION = auto()
    ENUM = auto()
    TAGGED_UNION = auto()  # Rust enum, Haskell ADT
    FUNCTION = auto()
    CLOSURE = auto()
    GENERIC = auto()
    TYPE_VAR = auto()  # Generic type variable
    OPAQUE = auto()
    RECURSIVE = auto()
    OPTIONAL = auto()  # Option<T>, Maybe T
    RESULT = auto()   # Result<T, E>
    TUPLE = auto()
    UNIT = auto()  # () in Rust, () in Haskell


class Ownership(Enum):
    """Ownership semantics (primarily for Rust)."""
    OWNED = auto()
    BORROWED = auto()
    BORROWED_MUT = auto()
    COPY = auto()
    STATIC = auto()


class Mutability(Enum):
    """Mutability of references/variables."""
    IMMUTABLE = auto()
    MUTABLE = auto()
    CONST = auto()  # C const


@dataclass
class Lifetime:
    """Represents a Rust lifetime."""
    name: str = "'a"
    is_static: bool = False
    
    def __str__(self) -> str:
        return "'static" if self.is_static else self.name


class STUNIRType(ABC):
    """Base class for all STUNIR types."""
    
    @property
    @abstractmethod
    def kind(self) -> TypeKind:
        """Return the kind of this type."""
        pass
    
    @abstractmethod
    def to_ir(self) -> Dict[str, Any]:
        """Convert type to IR representation."""
        pass
    
    @abstractmethod
    def __str__(self) -> str:
        """String representation of the type."""
        pass
    
    def is_primitive(self) -> bool:
        """Check if this is a primitive type."""
        return self.kind in (
            TypeKind.VOID, TypeKind.BOOL, TypeKind.INT,
            TypeKind.FLOAT, TypeKind.CHAR, TypeKind.UNIT
        )
    
    def is_pointer_like(self) -> bool:
        """Check if this type is pointer-like."""
        return self.kind in (
            TypeKind.POINTER, TypeKind.REFERENCE, TypeKind.SLICE
        )
    
    def is_compound(self) -> bool:
        """Check if this is a compound type."""
        return self.kind in (
            TypeKind.STRUCT, TypeKind.UNION, TypeKind.ENUM,
            TypeKind.TAGGED_UNION, TypeKind.TUPLE
        )


@dataclass
class VoidType(STUNIRType):
    """Void/unit type."""
    
    @property
    def kind(self) -> TypeKind:
        return TypeKind.VOID
    
    def to_ir(self) -> Dict[str, Any]:
        return {'type': 'void'}
    
    def __str__(self) -> str:
        return 'void'


@dataclass
class UnitType(STUNIRType):
    """Unit type () - Rust/Haskell."""
    
    @property
    def kind(self) -> TypeKind:
        return TypeKind.UNIT
    
    def to_ir(self) -> Dict[str, Any]:
        return {'type': 'unit'}
    
    def __str__(self) -> str:
        return '()'


@dataclass
class BoolType(STUNIRType):
    """Boolean type."""
    
    @property
    def kind(self) -> TypeKind:
        return TypeKind.BOOL
    
    def to_ir(self) -> Dict[str, Any]:
        return {'type': 'bool'}
    
    def __str__(self) -> str:
        return 'bool'


@dataclass
class IntType(STUNIRType):
    """Integer type with size and signedness."""
    bits: int = 32
    signed: bool = True
    
    @property
    def kind(self) -> TypeKind:
        return TypeKind.INT
    
    def to_ir(self) -> Dict[str, Any]:
        return {
            'type': 'int',
            'bits': self.bits,
            'signed': self.signed
        }
    
    def __str__(self) -> str:
        prefix = 'i' if self.signed else 'u'
        return f'{prefix}{self.bits}'


@dataclass
class FloatType(STUNIRType):
    """Floating-point type."""
    bits: int = 64
    
    @property
    def kind(self) -> TypeKind:
        return TypeKind.FLOAT
    
    def to_ir(self) -> Dict[str, Any]:
        return {'type': 'float', 'bits': self.bits}
    
    def __str__(self) -> str:
        return f'f{self.bits}'


@dataclass
class CharType(STUNIRType):
    """Character type."""
    unicode: bool = True
    
    @property
    def kind(self) -> TypeKind:
        return TypeKind.CHAR
    
    def to_ir(self) -> Dict[str, Any]:
        return {'type': 'char', 'unicode': self.unicode}
    
    def __str__(self) -> str:
        return 'char'


@dataclass
class StringType(STUNIRType):
    """String type."""
    owned: bool = True  # String vs &str
    
    @property
    def kind(self) -> TypeKind:
        return TypeKind.STRING
    
    def to_ir(self) -> Dict[str, Any]:
        return {'type': 'string', 'owned': self.owned}
    
    def __str__(self) -> str:
        return 'String' if self.owned else '&str'


@dataclass
class PointerType(STUNIRType):
    """Pointer type (raw pointer)."""
    pointee: STUNIRType
    mutability: Mutability = Mutability.MUTABLE
    nullable: bool = True
    
    @property
    def kind(self) -> TypeKind:
        return TypeKind.POINTER
    
    def to_ir(self) -> Dict[str, Any]:
        return {
            'type': 'pointer',
            'pointee': self.pointee.to_ir(),
            'mutable': self.mutability == Mutability.MUTABLE,
            'nullable': self.nullable
        }
    
    def __str__(self) -> str:
        mut = 'mut' if self.mutability == Mutability.MUTABLE else 'const'
        return f'*{mut} {self.pointee}'


@dataclass
class ReferenceType(STUNIRType):
    """Reference type (Rust references, C++ references)."""
    referent: STUNIRType
    mutability: Mutability = Mutability.IMMUTABLE
    lifetime: Optional[Lifetime] = None
    
    @property
    def kind(self) -> TypeKind:
        return TypeKind.REFERENCE
    
    def to_ir(self) -> Dict[str, Any]:
        ir = {
            'type': 'reference',
            'referent': self.referent.to_ir(),
            'mutable': self.mutability == Mutability.MUTABLE
        }
        if self.lifetime:
            ir['lifetime'] = str(self.lifetime)
        return ir
    
    def __str__(self) -> str:
        lifetime_str = f'{self.lifetime} ' if self.lifetime else ''
        mut = 'mut ' if self.mutability == Mutability.MUTABLE else ''
        return f'&{lifetime_str}{mut}{self.referent}'


@dataclass
class ArrayType(STUNIRType):
    """Fixed-size array type."""
    element: STUNIRType
    size: Optional[int] = None  # None for dynamic
    
    @property
    def kind(self) -> TypeKind:
        return TypeKind.ARRAY
    
    def to_ir(self) -> Dict[str, Any]:
        ir = {
            'type': 'array',
            'element': self.element.to_ir()
        }
        if self.size is not None:
            ir['size'] = self.size
        return ir
    
    def __str__(self) -> str:
        if self.size is not None:
            return f'[{self.element}; {self.size}]'
        return f'[{self.element}]'


@dataclass
class SliceType(STUNIRType):
    """Slice type (Rust &[T])."""
    element: STUNIRType
    mutability: Mutability = Mutability.IMMUTABLE
    lifetime: Optional[Lifetime] = None
    
    @property
    def kind(self) -> TypeKind:
        return TypeKind.SLICE
    
    def to_ir(self) -> Dict[str, Any]:
        ir = {
            'type': 'slice',
            'element': self.element.to_ir(),
            'mutable': self.mutability == Mutability.MUTABLE
        }
        if self.lifetime:
            ir['lifetime'] = str(self.lifetime)
        return ir
    
    def __str__(self) -> str:
        lifetime_str = f'{self.lifetime} ' if self.lifetime else ''
        mut = 'mut ' if self.mutability == Mutability.MUTABLE else ''
        return f'&{lifetime_str}{mut}[{self.element}]'


@dataclass
class StructField:
    """A field in a struct."""
    name: str
    type: STUNIRType
    visibility: str = 'public'
    offset: Optional[int] = None  # For C structs
    
    def to_ir(self) -> Dict[str, Any]:
        ir = {
            'name': self.name,
            'type': self.type.to_ir(),
            'visibility': self.visibility
        }
        if self.offset is not None:
            ir['offset'] = self.offset
        return ir


@dataclass
class StructType(STUNIRType):
    """Struct/record type."""
    name: str
    fields: List[StructField] = field(default_factory=list)
    generics: List[str] = field(default_factory=list)
    packed: bool = False  # C packed struct
    
    @property
    def kind(self) -> TypeKind:
        return TypeKind.STRUCT
    
    def to_ir(self) -> Dict[str, Any]:
        return {
            'type': 'struct',
            'name': self.name,
            'fields': [f.to_ir() for f in self.fields],
            'generics': self.generics,
            'packed': self.packed
        }
    
    def __str__(self) -> str:
        if self.generics:
            generics_str = ', '.join(self.generics)
            return f'{self.name}<{generics_str}>'
        return self.name
    
    def get_field(self, name: str) -> Optional[StructField]:
        """Get field by name."""
        for f in self.fields:
            if f.name == name:
                return f
        return None


@dataclass
class UnionType(STUNIRType):
    """C union type."""
    name: str
    variants: List[StructField] = field(default_factory=list)
    
    @property
    def kind(self) -> TypeKind:
        return TypeKind.UNION
    
    def to_ir(self) -> Dict[str, Any]:
        return {
            'type': 'union',
            'name': self.name,
            'variants': [v.to_ir() for v in self.variants]
        }
    
    def __str__(self) -> str:
        return f'union {self.name}'


@dataclass
class EnumVariant:
    """A variant in an enum."""
    name: str
    value: Optional[int] = None
    
    def to_ir(self) -> Dict[str, Any]:
        ir = {'name': self.name}
        if self.value is not None:
            ir['value'] = self.value
        return ir


@dataclass
class EnumType(STUNIRType):
    """Simple enum type (C-style)."""
    name: str
    variants: List[EnumVariant] = field(default_factory=list)
    
    @property
    def kind(self) -> TypeKind:
        return TypeKind.ENUM
    
    def to_ir(self) -> Dict[str, Any]:
        return {
            'type': 'enum',
            'name': self.name,
            'variants': [v.to_ir() for v in self.variants]
        }
    
    def __str__(self) -> str:
        return f'enum {self.name}'


@dataclass
class TaggedVariant:
    """A variant in a tagged union (Rust enum, Haskell ADT)."""
    name: str
    fields: List[STUNIRType] = field(default_factory=list)
    named_fields: Optional[Dict[str, STUNIRType]] = None
    
    def to_ir(self) -> Dict[str, Any]:
        ir = {'name': self.name}
        if self.fields:
            ir['fields'] = [f.to_ir() for f in self.fields]
        if self.named_fields:
            ir['named_fields'] = {
                k: v.to_ir() for k, v in self.named_fields.items()
            }
        return ir


@dataclass
class TaggedUnionType(STUNIRType):
    """Tagged union type (Rust enum, Haskell ADT)."""
    name: str
    variants: List[TaggedVariant] = field(default_factory=list)
    generics: List[str] = field(default_factory=list)
    
    @property
    def kind(self) -> TypeKind:
        return TypeKind.TAGGED_UNION
    
    def to_ir(self) -> Dict[str, Any]:
        return {
            'type': 'tagged_union',
            'name': self.name,
            'variants': [v.to_ir() for v in self.variants],
            'generics': self.generics
        }
    
    def __str__(self) -> str:
        if self.generics:
            generics_str = ', '.join(self.generics)
            return f'{self.name}<{generics_str}>'
        return self.name


@dataclass
class FunctionType(STUNIRType):
    """Function pointer type."""
    params: List[STUNIRType] = field(default_factory=list)
    returns: STUNIRType = field(default_factory=VoidType)
    variadic: bool = False
    
    @property
    def kind(self) -> TypeKind:
        return TypeKind.FUNCTION
    
    def to_ir(self) -> Dict[str, Any]:
        return {
            'type': 'function',
            'params': [p.to_ir() for p in self.params],
            'returns': self.returns.to_ir(),
            'variadic': self.variadic
        }
    
    def __str__(self) -> str:
        params_str = ', '.join(str(p) for p in self.params)
        if self.variadic:
            params_str += ', ...'
        return f'fn({params_str}) -> {self.returns}'


@dataclass
class ClosureType(STUNIRType):
    """Closure type (captures environment)."""
    params: List[STUNIRType] = field(default_factory=list)
    returns: STUNIRType = field(default_factory=VoidType)
    captures: List[Tuple[str, STUNIRType, Ownership]] = field(default_factory=list)
    
    @property
    def kind(self) -> TypeKind:
        return TypeKind.CLOSURE
    
    def to_ir(self) -> Dict[str, Any]:
        return {
            'type': 'closure',
            'params': [p.to_ir() for p in self.params],
            'returns': self.returns.to_ir(),
            'captures': [
                {'name': n, 'type': t.to_ir(), 'ownership': o.name}
                for n, t, o in self.captures
            ]
        }
    
    def __str__(self) -> str:
        params_str = ', '.join(str(p) for p in self.params)
        return f'Fn({params_str}) -> {self.returns}'


@dataclass
class GenericType(STUNIRType):
    """Generic/parametric type."""
    base: str
    args: List[STUNIRType] = field(default_factory=list)
    
    @property
    def kind(self) -> TypeKind:
        return TypeKind.GENERIC
    
    def to_ir(self) -> Dict[str, Any]:
        return {
            'type': 'generic',
            'base': self.base,
            'args': [a.to_ir() for a in self.args]
        }
    
    def __str__(self) -> str:
        if self.args:
            args_str = ', '.join(str(a) for a in self.args)
            return f'{self.base}<{args_str}>'
        return self.base


@dataclass
class TypeVar(STUNIRType):
    """Type variable (for generics)."""
    name: str
    constraints: List[str] = field(default_factory=list)
    
    @property
    def kind(self) -> TypeKind:
        return TypeKind.TYPE_VAR
    
    def to_ir(self) -> Dict[str, Any]:
        return {
            'type': 'type_var',
            'name': self.name,
            'constraints': self.constraints
        }
    
    def __str__(self) -> str:
        if self.constraints:
            constraints_str = ' + '.join(self.constraints)
            return f'{self.name}: {constraints_str}'
        return self.name


@dataclass
class OpaqueType(STUNIRType):
    """Opaque type (unknown internal structure)."""
    name: str
    
    @property
    def kind(self) -> TypeKind:
        return TypeKind.OPAQUE
    
    def to_ir(self) -> Dict[str, Any]:
        return {'type': 'opaque', 'name': self.name}
    
    def __str__(self) -> str:
        return f'opaque {self.name}'


@dataclass
class RecursiveType(STUNIRType):
    """Recursive type (self-referential)."""
    name: str
    inner: Optional[STUNIRType] = None  # Set after creation
    
    @property
    def kind(self) -> TypeKind:
        return TypeKind.RECURSIVE
    
    def to_ir(self) -> Dict[str, Any]:
        ir = {'type': 'recursive', 'name': self.name}
        if self.inner:
            ir['inner'] = self.inner.to_ir()
        return ir
    
    def __str__(self) -> str:
        return f'rec {self.name}'


@dataclass
class OptionalType(STUNIRType):
    """Optional type (Option<T>, Maybe T)."""
    inner: STUNIRType
    
    @property
    def kind(self) -> TypeKind:
        return TypeKind.OPTIONAL
    
    def to_ir(self) -> Dict[str, Any]:
        return {
            'type': 'optional',
            'inner': self.inner.to_ir()
        }
    
    def __str__(self) -> str:
        return f'Option<{self.inner}>'


@dataclass
class ResultType(STUNIRType):
    """Result type (Result<T, E>)."""
    ok_type: STUNIRType
    err_type: STUNIRType
    
    @property
    def kind(self) -> TypeKind:
        return TypeKind.RESULT
    
    def to_ir(self) -> Dict[str, Any]:
        return {
            'type': 'result',
            'ok': self.ok_type.to_ir(),
            'err': self.err_type.to_ir()
        }
    
    def __str__(self) -> str:
        return f'Result<{self.ok_type}, {self.err_type}>'


@dataclass
class TupleType(STUNIRType):
    """Tuple type."""
    elements: List[STUNIRType] = field(default_factory=list)
    
    @property
    def kind(self) -> TypeKind:
        return TypeKind.TUPLE
    
    def to_ir(self) -> Dict[str, Any]:
        return {
            'type': 'tuple',
            'elements': [e.to_ir() for e in self.elements]
        }
    
    def __str__(self) -> str:
        elements_str = ', '.join(str(e) for e in self.elements)
        return f'({elements_str})'


class TypeRegistry:
    """Registry for managing named types."""
    
    def __init__(self):
        self.types: Dict[str, STUNIRType] = {}
        self._init_builtins()
    
    def _init_builtins(self) -> None:
        """Initialize built-in types."""
        self.types['void'] = VoidType()
        self.types['()'] = UnitType()
        self.types['bool'] = BoolType()
        self.types['i8'] = IntType(bits=8, signed=True)
        self.types['i16'] = IntType(bits=16, signed=True)
        self.types['i32'] = IntType(bits=32, signed=True)
        self.types['i64'] = IntType(bits=64, signed=True)
        self.types['u8'] = IntType(bits=8, signed=False)
        self.types['u16'] = IntType(bits=16, signed=False)
        self.types['u32'] = IntType(bits=32, signed=False)
        self.types['u64'] = IntType(bits=64, signed=False)
        self.types['f32'] = FloatType(bits=32)
        self.types['f64'] = FloatType(bits=64)
        self.types['char'] = CharType()
        self.types['String'] = StringType(owned=True)
        self.types['str'] = StringType(owned=False)
        
        # C type aliases
        self.types['int'] = self.types['i32']
        self.types['long'] = self.types['i64']
        self.types['short'] = self.types['i16']
        self.types['char_c'] = IntType(bits=8, signed=True)
        self.types['unsigned'] = self.types['u32']
        self.types['size_t'] = self.types['u64']
        self.types['ssize_t'] = self.types['i64']
        self.types['float'] = self.types['f32']
        self.types['double'] = self.types['f64']
    
    def register(self, name: str, typ: STUNIRType) -> None:
        """Register a named type."""
        self.types[name] = typ
    
    def get(self, name: str) -> Optional[STUNIRType]:
        """Get a type by name."""
        return self.types.get(name)
    
    def has(self, name: str) -> bool:
        """Check if a type is registered."""
        return name in self.types
    
    def all_types(self) -> Dict[str, STUNIRType]:
        """Get all registered types."""
        return dict(self.types)


def parse_type(type_str: str, registry: Optional[TypeRegistry] = None) -> STUNIRType:
    """Parse a type string into a STUNIRType."""
    type_str = type_str.strip()
    
    if registry is None:
        registry = TypeRegistry()
    
    # Check registry first
    if registry.has(type_str):
        return registry.get(type_str)
    
    # Pointer type
    if type_str.startswith('*'):
        rest = type_str[1:].strip()
        if rest.startswith('mut '):
            pointee = parse_type(rest[4:], registry)
            return PointerType(pointee, Mutability.MUTABLE)
        elif rest.startswith('const '):
            pointee = parse_type(rest[6:], registry)
            return PointerType(pointee, Mutability.CONST)
        else:
            pointee = parse_type(rest, registry)
            return PointerType(pointee)
    
    # Reference type
    if type_str.startswith('&'):
        rest = type_str[1:].strip()
        if rest.startswith('mut '):
            referent = parse_type(rest[4:], registry)
            return ReferenceType(referent, Mutability.MUTABLE)
        else:
            referent = parse_type(rest, registry)
            return ReferenceType(referent)
    
    # Array type [T; N] or [T]
    if type_str.startswith('[') and type_str.endswith(']'):
        inner = type_str[1:-1]
        if ';' in inner:
            elem_str, size_str = inner.rsplit(';', 1)
            element = parse_type(elem_str.strip(), registry)
            size = int(size_str.strip())
            return ArrayType(element, size)
        else:
            element = parse_type(inner.strip(), registry)
            return ArrayType(element)
    
    # Tuple type (T, U, ...)
    if type_str.startswith('(') and type_str.endswith(')'):
        inner = type_str[1:-1]
        if not inner:
            return UnitType()
        elements = [parse_type(e.strip(), registry) for e in inner.split(',')]
        return TupleType(elements)
    
    # Generic type Name<T, U>
    if '<' in type_str and type_str.endswith('>'):
        base_end = type_str.index('<')
        base = type_str[:base_end]
        args_str = type_str[base_end+1:-1]
        args = [parse_type(a.strip(), registry) for a in args_str.split(',')]
        
        # Special cases
        if base == 'Option':
            return OptionalType(args[0])
        elif base == 'Result' and len(args) >= 2:
            return ResultType(args[0], args[1])
        
        return GenericType(base, args)
    
    # Function type fn(T, U) -> R
    if type_str.startswith('fn('):
        # Parse params and return type
        paren_end = type_str.index(')')
        params_str = type_str[3:paren_end]
        params = []
        if params_str:
            params = [parse_type(p.strip(), registry) for p in params_str.split(',')]
        
        returns = VoidType()
        if '->' in type_str:
            ret_str = type_str[type_str.index('->') + 2:].strip()
            returns = parse_type(ret_str, registry)
        
        return FunctionType(params, returns)
    
    # Default: try as simple type
    return registry.get(type_str) or OpaqueType(type_str)


# Common type instances for convenience
VOID = VoidType()
UNIT = UnitType()
BOOL = BoolType()
I8 = IntType(bits=8, signed=True)
I16 = IntType(bits=16, signed=True)
I32 = IntType(bits=32, signed=True)
I64 = IntType(bits=64, signed=True)
U8 = IntType(bits=8, signed=False)
U16 = IntType(bits=16, signed=False)
U32 = IntType(bits=32, signed=False)
U64 = IntType(bits=64, signed=False)
F32 = FloatType(bits=32)
F64 = FloatType(bits=64)
CHAR = CharType()
STRING = StringType(owned=True)
STR = StringType(owned=False)


__all__ = [
    # Enums
    'TypeKind', 'Ownership', 'Mutability', 'Lifetime',
    # Base class
    'STUNIRType',
    # Type classes
    'VoidType', 'UnitType', 'BoolType', 'IntType', 'FloatType', 'CharType', 'StringType',
    'PointerType', 'ReferenceType', 'ArrayType', 'SliceType',
    'StructType', 'StructField', 'UnionType',
    'EnumType', 'EnumVariant', 'TaggedUnionType', 'TaggedVariant',
    'FunctionType', 'ClosureType',
    'GenericType', 'TypeVar', 'OpaqueType', 'RecursiveType',
    'OptionalType', 'ResultType', 'TupleType',
    # Registry and parser
    'TypeRegistry', 'parse_type',
    # Common instances
    'VOID', 'UNIT', 'BOOL',
    'I8', 'I16', 'I32', 'I64', 'U8', 'U16', 'U32', 'U64',
    'F32', 'F64', 'CHAR', 'STRING', 'STR'
]
