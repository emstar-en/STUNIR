#!/usr/bin/env python3
"""STUNIR Systems IR - Strong type system constructs.

This module defines IR nodes for strong type system features
including subtypes, derived types, records, arrays, enums,
and generic type parameters.

Usage:
    from ir.systems.types import SubtypeDecl, RecordType, ArrayType
    
    # Create a subtype with constraint
    positive = SubtypeDecl(
        name='Positive',
        base_type=TypeRef(name='Integer'),
        constraint=RangeConstraint(lower=Literal(1), upper=None)
    )
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from ir.systems.systems_ir import (
    SystemsNode, TypeRef, Expr, Visibility, 
    ComponentDecl, Discriminant, Declaration
)


# =============================================================================
# Base Type Declaration
# =============================================================================

@dataclass
class TypeDecl(SystemsNode):
    """Base class for type declarations."""
    name: str = ''
    visibility: Visibility = Visibility.PUBLIC
    is_private: bool = False  # Private type completion
    is_limited: bool = False  # Ada limited type
    kind: str = 'type_decl'


# =============================================================================
# Constraints
# =============================================================================

@dataclass
class RangeConstraint(SystemsNode):
    """Range constraint for subtypes.
    
    Ada: range Lower .. Upper
    """
    lower: Optional[Expr] = None
    upper: Optional[Expr] = None
    kind: str = 'range_constraint'


@dataclass
class DigitsConstraint(SystemsNode):
    """Digits constraint for floating-point types (Ada).
    
    Ada: digits D [range Lower .. Upper]
    """
    digits: Expr = None
    range_constraint: Optional[RangeConstraint] = None
    kind: str = 'digits_constraint'


@dataclass
class DeltaConstraint(SystemsNode):
    """Delta constraint for fixed-point types (Ada).
    
    Ada: delta D [range Lower .. Upper]
    """
    delta: Expr = None
    range_constraint: Optional[RangeConstraint] = None
    kind: str = 'delta_constraint'


@dataclass
class IndexConstraint(SystemsNode):
    """Index constraint for array subtypes.
    
    Ada: (Index_Range1, Index_Range2, ...)
    """
    index_ranges: List[Expr] = field(default_factory=list)  # RangeExpr
    kind: str = 'index_constraint'


@dataclass
class DiscriminantConstraint(SystemsNode):
    """Discriminant constraint for record subtypes.
    
    Ada: (Disc1 => Value1, Disc2 => Value2)
    """
    values: Dict[str, Expr] = field(default_factory=dict)
    kind: str = 'discriminant_constraint'


# =============================================================================
# Subtypes and Derived Types
# =============================================================================

@dataclass
class SubtypeDecl(TypeDecl):
    """Subtype declaration.
    
    Ada: subtype Name is Base_Type [constraint];
    D: alias Name = Base_Type;  (limited support)
    
    Examples:
        Ada: subtype Positive is Integer range 1 .. Integer'Last;
             subtype Index is Integer range 1 .. 100;
    """
    base_type: TypeRef = None
    constraint: Optional[SystemsNode] = None  # RangeConstraint, etc.
    kind: str = 'subtype_decl'


@dataclass
class DerivedTypeDecl(TypeDecl):
    """Derived type declaration.
    
    Ada: type Name is [abstract] new Parent [with ...] [and Interfaces];
    
    Examples:
        Ada: type My_Int is new Integer;
             type Extended is new Parent with record
                Extra_Field : Integer;
             end record;
    """
    parent_type: TypeRef = None
    constraint: Optional[SystemsNode] = None
    extension: Optional['RecordExtension'] = None
    interfaces: List[TypeRef] = field(default_factory=list)
    is_abstract: bool = False
    kind: str = 'derived_type_decl'


@dataclass
class RecordExtension(SystemsNode):
    """Record extension part.
    
    Ada: with record ... end record;
    """
    components: List[ComponentDecl] = field(default_factory=list)
    is_null_extension: bool = False  # with null record
    kind: str = 'record_extension'


# =============================================================================
# Record Types
# =============================================================================

@dataclass
class RecordType(TypeDecl):
    """Record/struct type declaration.
    
    Ada: type Name is [abstract] [tagged] [limited] record ... end record;
    D: struct Name { ... }
    """
    discriminants: List[Discriminant] = field(default_factory=list)
    components: List[ComponentDecl] = field(default_factory=list)
    variant_part: Optional['VariantPart'] = None
    is_tagged: bool = False  # Ada tagged type
    is_abstract: bool = False
    parent_type: Optional[TypeRef] = None  # For derived records
    interfaces: List[TypeRef] = field(default_factory=list)
    # D invariant
    invariants: List['Contract'] = field(default_factory=list)
    kind: str = 'record_type'


@dataclass
class VariantPart(SystemsNode):
    """Variant part of a discriminated record (Ada).
    
    Ada: case Discriminant is
            when Choice1 => ...
            when Choice2 => ...
         end case;
    """
    discriminant_name: str = ''
    variants: List['Variant'] = field(default_factory=list)
    kind: str = 'variant_part'


@dataclass
class Variant(SystemsNode):
    """Single variant in a variant part."""
    choices: List[Expr] = field(default_factory=list)
    components: List[ComponentDecl] = field(default_factory=list)
    kind: str = 'variant'


@dataclass
class NullRecord(TypeDecl):
    """Null record type.
    
    Ada: type Name is null record;
    D: struct Name { }
    """
    kind: str = 'null_record'


# =============================================================================
# Array Types
# =============================================================================

@dataclass
class ArrayType(TypeDecl):
    """Array type declaration.
    
    Ada: type Name is array (Index_Type [, ...]) of Element_Type;
    D: alias Name = ElementType[Size];
    
    Examples:
        Ada: type Vector is array (1 .. 10) of Integer;
             type Matrix is array (Positive range <>, Positive range <>) of Float;
        D: alias Vector = int[10];
    """
    index_types: List[TypeRef] = field(default_factory=list)
    element_type: TypeRef = None
    index_constraints: List[Expr] = field(default_factory=list)  # For constrained arrays
    is_unconstrained: bool = False  # Ada unconstrained array (<>)
    is_aliased_components: bool = False  # Ada aliased components
    kind: str = 'array_type'


# =============================================================================
# Enumeration Types
# =============================================================================

@dataclass
class EnumType(TypeDecl):
    """Enumeration type declaration.
    
    Ada: type Name is (Enum1, Enum2, ...);
    D: enum Name { val1, val2, ... }
    
    Examples:
        Ada: type Color is (Red, Green, Blue);
        D: enum Color { Red, Green, Blue }
    """
    literals: List['EnumLiteral'] = field(default_factory=list)
    kind: str = 'enum_type'


@dataclass
class EnumLiteral(SystemsNode):
    """Enumeration literal."""
    name: str = ''
    value: Optional[Expr] = None  # Explicit value (D)
    position: Optional[int] = None  # Position value (Ada 'Pos)
    kind: str = 'enum_literal'


# =============================================================================
# Modular and Integer Types (Ada)
# =============================================================================

@dataclass
class IntegerType(TypeDecl):
    """Signed integer type declaration (Ada).
    
    Ada: type Name is range Lower .. Upper;
    """
    range_constraint: RangeConstraint = None
    kind: str = 'integer_type'


@dataclass
class ModularType(TypeDecl):
    """Modular (unsigned) integer type declaration (Ada).
    
    Ada: type Name is mod Modulus;
    
    Examples:
        Ada: type Byte is mod 256;
             type Word is mod 2**16;
    """
    modulus: Expr = None
    kind: str = 'modular_type'


# =============================================================================
# Fixed and Floating Point Types (Ada)
# =============================================================================

@dataclass
class FloatingPointType(TypeDecl):
    """Floating-point type declaration (Ada).
    
    Ada: type Name is digits D [range Lower .. Upper];
    """
    digits_constraint: DigitsConstraint = None
    kind: str = 'floating_point_type'


@dataclass
class FixedPointType(TypeDecl):
    """Fixed-point type declaration (Ada).
    
    Ada: type Name is delta D range Lower .. Upper;
    """
    delta_constraint: DeltaConstraint = None
    kind: str = 'fixed_point_type'


@dataclass
class DecimalType(TypeDecl):
    """Decimal fixed-point type declaration (Ada).
    
    Ada: type Name is delta D digits N [range Lower .. Upper];
    """
    delta: Expr = None
    digits: Expr = None
    range_constraint: Optional[RangeConstraint] = None
    kind: str = 'decimal_type'


# =============================================================================
# Interface Types (Ada)
# =============================================================================

@dataclass
class InterfaceType(TypeDecl):
    """Interface type declaration (Ada).
    
    Ada: type Name is [limited|synchronized|task|protected] interface
         [and Interface1 and Interface2];
    """
    parent_interfaces: List[TypeRef] = field(default_factory=list)
    is_limited: bool = False
    is_synchronized: bool = False
    is_task_interface: bool = False
    is_protected_interface: bool = False
    kind: str = 'interface_type'


# =============================================================================
# Alias/Typedef
# =============================================================================

@dataclass
class TypeAlias(TypeDecl):
    """Type alias declaration.
    
    Ada: subtype Name is Existing_Type;  (without constraint)
    D: alias Name = ExistingType;
    """
    aliased_type: TypeRef = None
    kind: str = 'type_alias'


# =============================================================================
# Class Types (D)
# =============================================================================

@dataclass
class ClassType(TypeDecl):
    """Class type declaration (D).
    
    D: class Name : ParentClass, Interface1 { ... }
    """
    parent_class: Optional[TypeRef] = None
    interfaces: List[TypeRef] = field(default_factory=list)
    components: List[ComponentDecl] = field(default_factory=list)
    methods: List['Subprogram'] = field(default_factory=list)
    invariants: List['Contract'] = field(default_factory=list)
    is_final: bool = False
    is_abstract: bool = False
    kind: str = 'class_type'


# =============================================================================
# Incomplete Type Declaration
# =============================================================================

@dataclass
class IncompleteType(TypeDecl):
    """Incomplete type declaration (forward declaration).
    
    Ada: type Name;
         type Name is tagged;  -- incomplete tagged
    """
    is_tagged: bool = False
    kind: str = 'incomplete_type'


# =============================================================================
# Private Type Declaration (Ada)
# =============================================================================

@dataclass
class PrivateTypeDecl(TypeDecl):
    """Private type declaration (Ada).
    
    Ada: type Name is [abstract] [tagged] [limited] private;
    """
    is_tagged: bool = False
    is_abstract: bool = False
    kind: str = 'private_type_decl'


# Forward reference
Contract = 'Contract'
Subprogram = 'Subprogram'
