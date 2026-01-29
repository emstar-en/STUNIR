#!/usr/bin/env python3
"""STUNIR Functional IR - Algebraic Data Types.

This module defines algebraic data type (ADT) constructs including
sum types, product types, newtypes, and records.

Usage:
    from ir.functional.adt import DataType, DataConstructor, TypeParameter
    
    # Define Maybe type
    maybe = DataType(
        name='Maybe',
        type_params=[TypeParameter(name='a')],
        constructors=[
            DataConstructor(name='Nothing'),
            DataConstructor(name='Just', fields=[TypeVar(name='a')])
        ]
    )
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Union
from ir.functional.functional_ir import FunctionalNode, TypeExpr, TypeVar, Expr


@dataclass
class TypeParameter:
    """Type parameter for polymorphic types.
    
    Attributes:
        name: Parameter name (e.g., 'a', 'b')
        kind_annotation: Optional kind annotation for Haskell
    """
    name: str = ''
    kind_annotation: Optional[str] = None  # For Haskell kind annotations
    
    def to_dict(self) -> dict:
        result = {'name': self.name}
        if self.kind_annotation:
            result['kind_annotation'] = self.kind_annotation
        return result


@dataclass
class DataConstructor:
    """Data constructor definition.
    
    Represents a constructor for an algebraic data type.
    
    Attributes:
        name: Constructor name (e.g., 'Just', 'Cons')
        fields: Types of constructor fields
        field_names: Optional names for record-style constructors
    """
    name: str = ''
    fields: List[TypeExpr] = field(default_factory=list)
    field_names: Optional[List[str]] = None  # For record syntax
    
    def to_dict(self) -> dict:
        result = {'name': self.name}
        if self.fields:
            result['fields'] = [f.to_dict() for f in self.fields]
        if self.field_names:
            result['field_names'] = self.field_names
        return result
    
    def arity(self) -> int:
        """Return number of fields."""
        return len(self.fields)
    
    def is_record(self) -> bool:
        """Check if this is a record-style constructor."""
        return self.field_names is not None


@dataclass
class DataType(FunctionalNode):
    """Algebraic data type definition.
    
    Represents a sum type (data in Haskell, type in OCaml).
    
    Attributes:
        name: Type name
        type_params: Type parameters for polymorphism
        constructors: List of data constructors
        deriving: Haskell deriving clauses
    """
    name: str = ''
    type_params: List[TypeParameter] = field(default_factory=list)
    constructors: List[DataConstructor] = field(default_factory=list)
    deriving: List[str] = field(default_factory=list)  # Haskell deriving
    kind: str = 'data_type'
    
    def is_enum(self) -> bool:
        """Check if this is an enumeration (all nullary constructors)."""
        return all(con.arity() == 0 for con in self.constructors)
    
    def constructor_names(self) -> List[str]:
        """Get list of constructor names."""
        return [con.name for con in self.constructors]


@dataclass
class TypeAlias(FunctionalNode):
    """Type alias (type synonym).
    
    Creates an alias for an existing type.
    """
    name: str = ''
    type_params: List[TypeParameter] = field(default_factory=list)
    target: TypeExpr = None
    kind: str = 'type_alias'


@dataclass
class NewType(FunctionalNode):
    """Newtype definition (Haskell-specific).
    
    Creates a new type with zero runtime overhead.
    """
    name: str = ''
    type_params: List[TypeParameter] = field(default_factory=list)
    constructor: DataConstructor = None
    deriving: List[str] = field(default_factory=list)
    kind: str = 'newtype'


@dataclass
class RecordField:
    """Field in a record type.
    
    Attributes:
        name: Field name
        field_type: Field type
        mutable: Whether field is mutable (OCaml)
    """
    name: str = ''
    field_type: TypeExpr = None
    mutable: bool = False  # OCaml mutable fields
    
    def to_dict(self) -> dict:
        result = {
            'name': self.name,
        }
        if self.field_type:
            result['field_type'] = self.field_type.to_dict()
        if self.mutable:
            result['mutable'] = self.mutable
        return result


@dataclass
class RecordType(FunctionalNode):
    """Record type (OCaml-style).
    
    Defines a product type with named fields.
    """
    name: str = ''
    type_params: List[TypeParameter] = field(default_factory=list)
    fields: List[RecordField] = field(default_factory=list)
    kind: str = 'record_type'
    
    def field_names(self) -> List[str]:
        """Get list of field names."""
        return [f.name for f in self.fields]
    
    def has_mutable_fields(self) -> bool:
        """Check if any fields are mutable."""
        return any(f.mutable for f in self.fields)


# =============================================================================
# Type Class Definitions (Haskell-specific)
# =============================================================================

@dataclass
class MethodSignature:
    """Type class method signature.
    
    Attributes:
        name: Method name
        type_signature: Method type
    """
    name: str = ''
    type_signature: TypeExpr = None
    
    def to_dict(self) -> dict:
        result = {'name': self.name}
        if self.type_signature:
            result['type_signature'] = self.type_signature.to_dict()
        return result


@dataclass
class TypeClass(FunctionalNode):
    """Type class definition (Haskell-specific).
    
    Defines a type class with methods.
    """
    name: str = ''
    type_params: List[TypeParameter] = field(default_factory=list)
    superclasses: List[str] = field(default_factory=list)
    methods: List[MethodSignature] = field(default_factory=list)
    default_implementations: Dict[str, Expr] = field(default_factory=dict)
    kind: str = 'type_class'


@dataclass
class TypeClassInstance(FunctionalNode):
    """Type class instance.
    
    Provides implementations of type class methods for a specific type.
    """
    class_name: str = ''
    type_args: List[TypeExpr] = field(default_factory=list)
    implementations: Dict[str, Expr] = field(default_factory=dict)
    kind: str = 'type_class_instance'


# =============================================================================
# Module Definitions
# =============================================================================

@dataclass
class Import:
    """Module import.
    
    Attributes:
        module: Module name to import
        qualified: Whether import is qualified
        alias: Optional alias name
        items: Specific items to import (None = all)
        hiding: Items to hide from import
    """
    module: str = ''
    qualified: bool = False
    alias: Optional[str] = None
    items: Optional[List[str]] = None  # None = import all
    hiding: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        result = {'module': self.module}
        if self.qualified:
            result['qualified'] = self.qualified
        if self.alias:
            result['alias'] = self.alias
        if self.items:
            result['items'] = self.items
        if self.hiding:
            result['hiding'] = self.hiding
        return result


@dataclass
class FunctionClause:
    """Single clause of a function definition (pattern matching).
    
    Represents one equation in a function definition.
    """
    patterns: List['Pattern'] = field(default_factory=list)
    guard: Optional[Expr] = None
    body: Expr = None
    
    def to_dict(self) -> dict:
        result = {}
        if self.patterns:
            result['patterns'] = [p.to_dict() for p in self.patterns]
        if self.guard:
            result['guard'] = self.guard.to_dict()
        if self.body:
            result['body'] = self.body.to_dict()
        return result


from ir.functional.functional_ir import LetExpr


@dataclass
class FunctionDef(FunctionalNode):
    """Function definition.
    
    A named function with optional type signature and pattern-matching clauses.
    """
    name: str = ''
    type_signature: Optional[TypeExpr] = None
    clauses: List[FunctionClause] = field(default_factory=list)
    where_bindings: List[LetExpr] = field(default_factory=list)
    kind: str = 'function_def'


@dataclass
class Module(FunctionalNode):
    """Module definition.
    
    Represents a complete module with imports, types, and functions.
    """
    name: str = ''
    exports: List[str] = field(default_factory=list)
    imports: List[Import] = field(default_factory=list)
    type_definitions: List[Union[DataType, TypeAlias, NewType, RecordType]] = field(default_factory=list)
    type_classes: List[TypeClass] = field(default_factory=list)
    instances: List[TypeClassInstance] = field(default_factory=list)
    functions: List[FunctionDef] = field(default_factory=list)
    kind: str = 'module'


# Import Pattern for type hints
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ir.functional.pattern import Pattern
