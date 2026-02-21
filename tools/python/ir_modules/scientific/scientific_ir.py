#!/usr/bin/env python3
"""STUNIR Scientific IR - Core scientific programming constructs.

This module defines IR nodes for scientific and legacy programming
languages like Fortran and Pascal, including arrays, numerical
computing, and structured programming constructs.

Usage:
    from ir.scientific.scientific_ir import Module, Subprogram, Parameter
    from ir.scientific import Visibility, Intent
    
    # Create a module with function
    mod = Module(
        name='math_utils',
        subprograms=[
            Subprogram(
                name='add',
                is_function=True,
                parameters=[Parameter(name='a', type_ref=TypeRef(name='f64'))]
            )
        ]
    )
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from abc import ABC
from enum import Enum


# =============================================================================
# Enumerations
# =============================================================================

class Visibility(Enum):
    """Visibility specifiers for modules and types."""
    PUBLIC = 'public'
    PRIVATE = 'private'


class Intent(Enum):
    """Fortran intent attributes for parameters."""
    IN = 'in'
    OUT = 'out'
    INOUT = 'inout'


class ParameterMode(Enum):
    """Pascal parameter passing modes."""
    VALUE = 'value'      # By value (default)
    VAR = 'var'          # By reference
    CONST = 'const'      # By reference, read-only
    OUT = 'out'          # Output only


class ArrayOrder(Enum):
    """Array storage order."""
    COLUMN_MAJOR = 'column_major'  # Fortran default
    ROW_MAJOR = 'row_major'        # Pascal/C default


# =============================================================================
# Base Class
# =============================================================================

@dataclass
class ScientificNode(ABC):
    """Base class for all scientific IR nodes."""
    kind: str = 'scientific_node'
    
    def to_dict(self) -> dict:
        """Convert node to dictionary representation."""
        result = {'kind': self.kind}
        for key, value in self.__dict__.items():
            if key != 'kind' and value is not None:
                if isinstance(value, ScientificNode):
                    result[key] = value.to_dict()
                elif isinstance(value, list):
                    result[key] = [
                        v.to_dict() if isinstance(v, ScientificNode) else v
                        for v in value
                    ]
                elif isinstance(value, dict):
                    result[key] = {
                        k: (v.to_dict() if isinstance(v, ScientificNode) else v)
                        for k, v in value.items()
                    }
                elif isinstance(value, Enum):
                    result[key] = value.value
                else:
                    result[key] = value
        return result


# =============================================================================
# Type References
# =============================================================================

@dataclass
class TypeRef(ScientificNode):
    """Reference to a type.
    
    Can represent primitive types, named types, or parameterized types.
    """
    name: str = ''
    kind_param: Optional[int] = None  # Fortran KIND parameter
    len_param: Optional[Union[int, str]] = None  # CHARACTER LEN
    is_pointer: bool = False
    is_allocatable: bool = False
    kind: str = 'type_ref'


# =============================================================================
# Module and Program Structures
# =============================================================================

@dataclass
class Import(ScientificNode):
    """Module import/use statement."""
    module_name: str = ''
    only: List[str] = field(default_factory=list)  # ONLY clause items
    rename: Dict[str, str] = field(default_factory=dict)  # local => original
    kind: str = 'import'


@dataclass
class Module(ScientificNode):
    """Module/unit definition (Fortran MODULE, Pascal UNIT)."""
    name: str = ''
    imports: List[Import] = field(default_factory=list)
    exports: List[str] = field(default_factory=list)  # PUBLIC items
    types: List['TypeDecl'] = field(default_factory=list)
    variables: List['VariableDecl'] = field(default_factory=list)
    subprograms: List['Subprogram'] = field(default_factory=list)
    visibility: Visibility = Visibility.PUBLIC
    is_submodule: bool = False  # Fortran submodule
    parent_module: Optional[str] = None  # Parent module for submodule
    kind: str = 'module'


@dataclass
class Program(ScientificNode):
    """Main program definition."""
    name: str = ''
    uses: List[Import] = field(default_factory=list)
    types: List['TypeDecl'] = field(default_factory=list)
    variables: List['VariableDecl'] = field(default_factory=list)
    subprograms: List['Subprogram'] = field(default_factory=list)
    body: List['Statement'] = field(default_factory=list)
    kind: str = 'program'


# =============================================================================
# Subprograms
# =============================================================================

@dataclass
class Parameter(ScientificNode):
    """Subprogram parameter."""
    name: str = ''
    type_ref: TypeRef = None
    intent: Intent = Intent.IN           # Fortran
    mode: ParameterMode = ParameterMode.VALUE  # Pascal
    is_optional: bool = False
    default_value: Optional['Expr'] = None
    kind: str = 'parameter'


@dataclass
class Interface(ScientificNode):
    """Fortran interface block."""
    name: Optional[str] = None  # Named interface for generics
    procedures: List[str] = field(default_factory=list)
    is_abstract: bool = False
    kind: str = 'interface'


@dataclass
class Subprogram(ScientificNode):
    """Subroutine or function definition."""
    name: str = ''
    is_function: bool = False
    parameters: List[Parameter] = field(default_factory=list)
    return_type: Optional[TypeRef] = None
    result_var: Optional[str] = None  # Fortran RESULT clause
    local_vars: List['VariableDecl'] = field(default_factory=list)
    local_types: List['TypeDecl'] = field(default_factory=list)
    body: List['Statement'] = field(default_factory=list)
    is_pure: bool = False       # Fortran PURE
    is_elemental: bool = False  # Fortran ELEMENTAL
    is_recursive: bool = False  # Fortran RECURSIVE
    interface: Optional[Interface] = None
    visibility: Visibility = Visibility.PUBLIC
    kind: str = 'subprogram'


# =============================================================================
# Variable Declarations
# =============================================================================

@dataclass
class VariableDecl(ScientificNode):
    """Variable declaration."""
    name: str = ''
    type_ref: TypeRef = None
    initial_value: Optional['Expr'] = None
    is_constant: bool = False      # Fortran PARAMETER / Pascal const
    is_save: bool = False          # Fortran SAVE
    is_volatile: bool = False      # Fortran VOLATILE
    is_target: bool = False        # Fortran TARGET
    visibility: Visibility = Visibility.PUBLIC
    kind: str = 'variable_decl'


@dataclass
class ConstantDecl(ScientificNode):
    """Named constant declaration."""
    name: str = ''
    type_ref: Optional[TypeRef] = None
    value: 'Expr' = None
    kind: str = 'constant_decl'


# =============================================================================
# Type Declarations
# =============================================================================

@dataclass
class TypeDecl(ScientificNode):
    """Type declaration base."""
    name: str = ''
    visibility: Visibility = Visibility.PUBLIC
    kind: str = 'type_decl'


@dataclass
class FieldDecl(ScientificNode):
    """Field declaration in record/derived type."""
    name: str = ''
    type_ref: TypeRef = None
    is_pointer: bool = False
    is_allocatable: bool = False
    default_value: Optional['Expr'] = None
    visibility: Visibility = Visibility.PUBLIC
    kind: str = 'field_decl'


@dataclass
class RecordType(ScientificNode):
    """Record/derived type definition."""
    name: str = ''
    fields: List[FieldDecl] = field(default_factory=list)
    is_sequence: bool = False   # Fortran SEQUENCE
    is_bind_c: bool = False     # Fortran BIND(C)
    extends: Optional[str] = None  # Fortran EXTENDS
    visibility: Visibility = Visibility.PUBLIC
    kind: str = 'record_type'


@dataclass
class RecordVariant(ScientificNode):
    """Pascal variant part of a variant record."""
    tag_values: List['Expr'] = field(default_factory=list)
    fields: List[FieldDecl] = field(default_factory=list)
    kind: str = 'record_variant'


@dataclass
class VariantRecord(ScientificNode):
    """Pascal variant record type."""
    name: str = ''
    fixed_fields: List[FieldDecl] = field(default_factory=list)
    tag_field: Optional[FieldDecl] = None
    variants: List[RecordVariant] = field(default_factory=list)
    kind: str = 'variant_record'


@dataclass
class EnumType(ScientificNode):
    """Enumeration type."""
    name: str = ''
    values: List[str] = field(default_factory=list)
    kind: str = 'enum_type'


@dataclass
class SetType(ScientificNode):
    """Pascal set type."""
    name: str = ''
    base_type: TypeRef = None
    kind: str = 'set_type'


@dataclass
class RangeType(ScientificNode):
    """Subrange type."""
    name: str = ''
    base_type: Optional[TypeRef] = None
    lower: 'Expr' = None
    upper: 'Expr' = None
    kind: str = 'range_type'


@dataclass
class PointerType(ScientificNode):
    """Pointer type."""
    name: str = ''
    target_type: TypeRef = None
    kind: str = 'pointer_type'


@dataclass
class FileType(ScientificNode):
    """Pascal file type."""
    name: str = ''
    element_type: Optional[TypeRef] = None  # None for TEXT files
    kind: str = 'file_type'


# =============================================================================
# Object Pascal Constructs
# =============================================================================

@dataclass
class MethodDecl(ScientificNode):
    """Object Pascal method declaration."""
    name: str = ''
    is_function: bool = False
    parameters: List[Parameter] = field(default_factory=list)
    return_type: Optional[TypeRef] = None
    is_virtual: bool = False
    is_abstract: bool = False
    is_override: bool = False
    is_static: bool = False
    is_class_method: bool = False
    visibility: Visibility = Visibility.PUBLIC
    kind: str = 'method_decl'


@dataclass
class PropertyDecl(ScientificNode):
    """Object Pascal property declaration."""
    name: str = ''
    type_ref: TypeRef = None
    getter: Optional[str] = None
    setter: Optional[str] = None
    is_default: bool = False
    visibility: Visibility = Visibility.PUBLIC
    kind: str = 'property_decl'


@dataclass
class ClassType(ScientificNode):
    """Object Pascal class definition."""
    name: str = ''
    parent: Optional[str] = None
    fields: List[FieldDecl] = field(default_factory=list)
    methods: List[MethodDecl] = field(default_factory=list)
    properties: List[PropertyDecl] = field(default_factory=list)
    is_abstract: bool = False
    kind: str = 'class_type'


# =============================================================================
# Statements
# =============================================================================

Statement = Union['Assignment', 'IfStatement', 'ForLoop', 'WhileLoop',
                  'RepeatLoop', 'CaseStatement', 'CallStatement',
                  'ReturnStatement', 'BlockStatement', 'NullStatement',
                  'ExitStatement', 'ContinueStatement', 'GotoStatement',
                  'WithStatement', 'TryStatement']


@dataclass
class Assignment(ScientificNode):
    """Assignment statement."""
    target: 'Expr' = None
    value: 'Expr' = None
    kind: str = 'assignment'


@dataclass
class IfStatement(ScientificNode):
    """If-then-else statement."""
    condition: 'Expr' = None
    then_body: List[Statement] = field(default_factory=list)
    elseif_parts: List['ElseIfPart'] = field(default_factory=list)
    else_body: List[Statement] = field(default_factory=list)
    kind: str = 'if_statement'


@dataclass
class ElseIfPart(ScientificNode):
    """Else-if clause."""
    condition: 'Expr' = None
    body: List[Statement] = field(default_factory=list)
    kind: str = 'elseif_part'


@dataclass
class ForLoop(ScientificNode):
    """Counting for loop."""
    variable: str = ''
    start: 'Expr' = None
    end: 'Expr' = None
    step: Optional['Expr'] = None
    body: List[Statement] = field(default_factory=list)
    is_downto: bool = False     # Pascal DOWNTO
    kind: str = 'for_loop'


@dataclass
class WhileLoop(ScientificNode):
    """While loop."""
    condition: 'Expr' = None
    body: List[Statement] = field(default_factory=list)
    kind: str = 'while_loop'


@dataclass
class RepeatLoop(ScientificNode):
    """Pascal repeat-until loop."""
    condition: 'Expr' = None
    body: List[Statement] = field(default_factory=list)
    kind: str = 'repeat_loop'


@dataclass
class CaseItem(ScientificNode):
    """Case/when branch."""
    values: List['Expr'] = field(default_factory=list)
    body: List[Statement] = field(default_factory=list)
    kind: str = 'case_item'


@dataclass
class CaseStatement(ScientificNode):
    """Case/select statement."""
    selector: 'Expr' = None
    cases: List[CaseItem] = field(default_factory=list)
    default_body: List[Statement] = field(default_factory=list)
    kind: str = 'case_statement'


@dataclass
class CallStatement(ScientificNode):
    """Procedure/subroutine call statement."""
    name: str = ''
    arguments: List['Expr'] = field(default_factory=list)
    kind: str = 'call_statement'


@dataclass
class ReturnStatement(ScientificNode):
    """Return from subprogram."""
    value: Optional['Expr'] = None
    kind: str = 'return_statement'


@dataclass
class BlockStatement(ScientificNode):
    """Block/compound statement."""
    declarations: List[VariableDecl] = field(default_factory=list)
    body: List[Statement] = field(default_factory=list)
    kind: str = 'block_statement'


@dataclass
class NullStatement(ScientificNode):
    """Null/no-op statement."""
    kind: str = 'null_statement'


@dataclass
class ExitStatement(ScientificNode):
    """Exit from loop (Fortran EXIT, Pascal break)."""
    loop_name: Optional[str] = None  # Fortran named loop
    kind: str = 'exit_statement'


@dataclass
class ContinueStatement(ScientificNode):
    """Continue to next iteration (Fortran CYCLE, Pascal continue)."""
    loop_name: Optional[str] = None
    kind: str = 'continue_statement'


@dataclass
class GotoStatement(ScientificNode):
    """Goto statement."""
    label: str = ''
    kind: str = 'goto_statement'


@dataclass
class LabeledStatement(ScientificNode):
    """Labeled statement."""
    label: str = ''
    statement: Statement = None
    kind: str = 'labeled_statement'


@dataclass
class WithStatement(ScientificNode):
    """Pascal WITH statement."""
    record_vars: List['Expr'] = field(default_factory=list)
    body: List[Statement] = field(default_factory=list)
    kind: str = 'with_statement'


@dataclass
class ExceptionHandler(ScientificNode):
    """Exception handler clause."""
    exception_type: Optional[str] = None
    variable: Optional[str] = None
    body: List[Statement] = field(default_factory=list)
    kind: str = 'exception_handler'


@dataclass
class TryStatement(ScientificNode):
    """Try-except/finally statement."""
    try_body: List[Statement] = field(default_factory=list)
    handlers: List[ExceptionHandler] = field(default_factory=list)
    finally_body: List[Statement] = field(default_factory=list)
    kind: str = 'try_statement'


@dataclass
class RaiseStatement(ScientificNode):
    """Raise/throw exception."""
    exception_expr: Optional['Expr'] = None
    kind: str = 'raise_statement'


# =============================================================================
# File I/O Statements
# =============================================================================

@dataclass
class OpenStatement(ScientificNode):
    """Fortran OPEN or Pascal file open."""
    unit: 'Expr' = None
    file_name: Optional['Expr'] = None
    status: Optional[str] = None  # 'OLD', 'NEW', etc.
    access: Optional[str] = None  # 'SEQUENTIAL', 'DIRECT'
    form: Optional[str] = None    # 'FORMATTED', 'UNFORMATTED'
    kind: str = 'open_statement'


@dataclass
class CloseStatement(ScientificNode):
    """Close file."""
    unit: 'Expr' = None
    kind: str = 'close_statement'


@dataclass
class ReadStatement(ScientificNode):
    """Read statement."""
    unit: Optional['Expr'] = None  # None for stdin
    format_spec: Optional['Expr'] = None
    items: List['Expr'] = field(default_factory=list)
    kind: str = 'read_statement'


@dataclass
class WriteStatement(ScientificNode):
    """Write/print statement."""
    unit: Optional['Expr'] = None  # None for stdout
    format_spec: Optional['Expr'] = None
    items: List['Expr'] = field(default_factory=list)
    kind: str = 'write_statement'


# =============================================================================
# Expressions
# =============================================================================

Expr = Union['Literal', 'VarRef', 'BinaryOp', 'UnaryOp', 'FunctionCall',
             'ArrayAccess', 'FieldAccess', 'SetExpr', 'SetOp', 'RangeExpr',
             'TypeCast', 'SizeOfExpr', 'AllocateExpr']


@dataclass
class Literal(ScientificNode):
    """Literal value."""
    value: Any = None
    type_hint: Optional[str] = None  # 'integer', 'real', 'string', etc.
    kind_param: Optional[int] = None  # Fortran KIND
    kind: str = 'literal'


@dataclass
class VarRef(ScientificNode):
    """Variable reference."""
    name: str = ''
    kind: str = 'var_ref'


@dataclass
class BinaryOp(ScientificNode):
    """Binary operation."""
    op: str = ''                # '+', '-', '*', '/', '**', etc.
    left: 'Expr' = None
    right: 'Expr' = None
    kind: str = 'binary_op'


@dataclass
class UnaryOp(ScientificNode):
    """Unary operation."""
    op: str = ''                # '-', 'NOT', '.NOT.', etc.
    operand: 'Expr' = None
    kind: str = 'unary_op'


@dataclass
class FunctionCall(ScientificNode):
    """Function call expression."""
    name: str = ''
    arguments: List['Expr'] = field(default_factory=list)
    kind: str = 'function_call'


@dataclass
class ArrayAccess(ScientificNode):
    """Array element access."""
    array: 'Expr' = None
    indices: List['Expr'] = field(default_factory=list)
    kind: str = 'array_access'


@dataclass
class FieldAccess(ScientificNode):
    """Record/object field access."""
    record: 'Expr' = None
    field_name: str = ''
    kind: str = 'field_access'


@dataclass
class SetExpr(ScientificNode):
    """Pascal set literal expression."""
    elements: List['Expr'] = field(default_factory=list)
    kind: str = 'set_expr'


@dataclass
class SetOp(ScientificNode):
    """Pascal set operation."""
    op: str = ''                # 'union', 'intersection', 'difference', 'in'
    left: 'Expr' = None
    right: 'Expr' = None
    kind: str = 'set_op'


@dataclass
class RangeExpr(ScientificNode):
    """Range expression (1..10)."""
    start: 'Expr' = None
    end: 'Expr' = None
    kind: str = 'range_expr'


@dataclass
class TypeCast(ScientificNode):
    """Type cast/conversion."""
    target_type: TypeRef = None
    expr: 'Expr' = None
    kind: str = 'type_cast'


@dataclass
class SizeOfExpr(ScientificNode):
    """Size of type/variable expression."""
    target: Union[TypeRef, 'Expr'] = None
    kind: str = 'sizeof_expr'


@dataclass
class AllocateExpr(ScientificNode):
    """Allocation expression (Pascal New, Fortran ALLOCATE)."""
    type_ref: TypeRef = None
    size: Optional['Expr'] = None
    kind: str = 'allocate_expr'


@dataclass
class DeallocateExpr(ScientificNode):
    """Deallocation expression."""
    target: 'Expr' = None
    kind: str = 'deallocate_expr'


@dataclass
class PointerDeref(ScientificNode):
    """Pointer dereference."""
    pointer: 'Expr' = None
    kind: str = 'pointer_deref'


@dataclass
class AddressOf(ScientificNode):
    """Address-of operator."""
    operand: 'Expr' = None
    kind: str = 'address_of'


# Fortran-specific allocation statement
@dataclass
class AllocateStatement(ScientificNode):
    """Fortran ALLOCATE statement."""
    allocations: List['AllocationSpec'] = field(default_factory=list)
    stat_var: Optional[str] = None
    errmsg_var: Optional[str] = None
    source_expr: Optional['Expr'] = None
    kind: str = 'allocate_statement'


@dataclass
class AllocationSpec(ScientificNode):
    """Single allocation specification."""
    target: str = ''
    shape: List['Expr'] = field(default_factory=list)
    kind: str = 'allocation_spec'


@dataclass
class DeallocateStatement(ScientificNode):
    """Fortran DEALLOCATE statement."""
    variables: List[str] = field(default_factory=list)
    stat_var: Optional[str] = None
    errmsg_var: Optional[str] = None
    kind: str = 'deallocate_statement'
