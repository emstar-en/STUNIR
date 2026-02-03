#!/usr/bin/env python3
"""STUNIR Systems IR - Core systems programming constructs.

This module defines IR nodes for systems programming languages
like Ada and D, including strong typing, memory management,
concurrency, and formal verification.

Usage:
    from ir.systems.systems_ir import Package, Subprogram, Parameter
    from ir.systems import Mode, Visibility
    
    # Create a subprogram with contracts
    sub = Subprogram(
        name='Add',
        parameters=[Parameter(name='A', type_ref=TypeRef(name='Integer'))],
        return_type=TypeRef(name='Integer')
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
    """Visibility specifiers."""
    PUBLIC = 'public'
    PRIVATE = 'private'
    PROTECTED = 'protected'
    LIMITED = 'limited'  # Ada limited visibility


class Mode(Enum):
    """Parameter modes (Ada-style)."""
    IN = 'in'
    OUT = 'out'
    IN_OUT = 'in_out'
    ACCESS = 'access'  # Ada access parameters


class SafetyLevel(Enum):
    """D memory safety levels."""
    SAFE = 'safe'
    TRUSTED = 'trusted'
    SYSTEM = 'system'


# =============================================================================
# Base Classes
# =============================================================================

@dataclass
class SystemsNode(ABC):
    """Base class for all systems IR nodes."""
    kind: str = 'systems_node'
    
    def to_dict(self) -> dict:
        """Convert node to dictionary representation."""
        result = {'kind': self.kind}
        for key, value in self.__dict__.items():
            if key != 'kind' and value is not None:
                if isinstance(value, SystemsNode):
                    result[key] = value.to_dict()
                elif isinstance(value, list):
                    result[key] = [
                        v.to_dict() if isinstance(v, SystemsNode) else v
                        for v in value
                    ]
                elif isinstance(value, dict):
                    result[key] = {
                        k: (v.to_dict() if isinstance(v, SystemsNode) else v)
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
class TypeRef(SystemsNode):
    """Reference to a type.
    
    Can represent primitive types, named types, or parameterized types.
    """
    name: str = ''
    type_args: List['TypeRef'] = field(default_factory=list)
    is_access: bool = False  # Pointer/access type
    not_null: bool = False  # Not null access
    is_constant: bool = False  # Constant type
    kind: str = 'type_ref'


# =============================================================================
# Expressions
# =============================================================================

@dataclass
class Expr(SystemsNode):
    """Base class for expressions."""
    type_ref: Optional[TypeRef] = None
    kind: str = 'expr'


@dataclass
class Literal(Expr):
    """Literal value expression."""
    value: Any = None
    literal_type: str = 'int'  # 'int', 'float', 'bool', 'string', 'char'
    kind: str = 'literal'


@dataclass
class VarExpr(Expr):
    """Variable reference expression."""
    name: str = ''
    kind: str = 'var_expr'


@dataclass
class BinaryOp(Expr):
    """Binary operation expression."""
    op: str = ''
    left: Expr = None
    right: Expr = None
    kind: str = 'binary_op'


@dataclass
class UnaryOp(Expr):
    """Unary operation expression."""
    op: str = ''
    operand: Expr = None
    kind: str = 'unary_op'


@dataclass
class CallExpr(Expr):
    """Function/procedure call expression."""
    name: str = ''
    arguments: List[Expr] = field(default_factory=list)
    kind: str = 'call_expr'


@dataclass
class MemberAccess(Expr):
    """Member/field access expression."""
    target: Expr = None
    member: str = ''
    kind: str = 'member_access'


@dataclass
class IndexExpr(Expr):
    """Array/string index expression."""
    target: Expr = None
    index: Expr = None
    kind: str = 'index_expr'


@dataclass
class IfExpr(Expr):
    """Conditional expression (if-then-else)."""
    condition: Expr = None
    then_expr: Expr = None
    else_expr: Expr = None
    kind: str = 'if_expr'


@dataclass
class CastExpr(Expr):
    """Type cast expression."""
    target_type: TypeRef = None
    expr: Expr = None
    kind: str = 'cast_expr'


@dataclass
class RangeExpr(Expr):
    """Range expression (Ada: 1 .. 10, D: 0..10)."""
    start: Expr = None
    end: Expr = None
    inclusive: bool = True  # Whether end is inclusive
    kind: str = 'range_expr'


@dataclass
class AggregateExpr(Expr):
    """Aggregate/initializer expression.
    
    Ada: (Field1 => Value1, Field2 => Value2)
    D: MyStruct(field1: value1, field2: value2)
    """
    components: Dict[str, Expr] = field(default_factory=dict)
    positional: List[Expr] = field(default_factory=list)
    kind: str = 'aggregate_expr'


@dataclass
class AttributeExpr(Expr):
    """Attribute expression (Ada: X'First, X'Last, X'Length)."""
    target: Expr = None
    attribute: str = ''
    arguments: List[Expr] = field(default_factory=list)
    kind: str = 'attribute_expr'


@dataclass
class QualifiedExpr(Expr):
    """Qualified expression (Ada: Type'(expr))."""
    type_mark: TypeRef = None
    expr: Expr = None
    kind: str = 'qualified_expr'


# =============================================================================
# Statements
# =============================================================================

@dataclass
class Statement(SystemsNode):
    """Base class for statements."""
    kind: str = 'statement'


@dataclass
class Assignment(Statement):
    """Assignment statement."""
    target: Expr = None
    value: Expr = None
    kind: str = 'assignment'


@dataclass
class IfStatement(Statement):
    """If statement with optional elsif and else."""
    condition: Expr = None
    then_body: List[Statement] = field(default_factory=list)
    elsif_parts: List['ElsifPart'] = field(default_factory=list)
    else_body: Optional[List[Statement]] = None
    kind: str = 'if_statement'


@dataclass
class ElsifPart(SystemsNode):
    """Elsif part of an if statement."""
    condition: Expr = None
    body: List[Statement] = field(default_factory=list)
    kind: str = 'elsif_part'


@dataclass
class CaseStatement(Statement):
    """Case/switch statement."""
    selector: Expr = None
    alternatives: List['CaseAlternative'] = field(default_factory=list)
    kind: str = 'case_statement'


@dataclass
class CaseAlternative(SystemsNode):
    """Case alternative with choices and statements."""
    choices: List[Expr] = field(default_factory=list)  # Can be ranges
    body: List[Statement] = field(default_factory=list)
    kind: str = 'case_alternative'


@dataclass
class WhileLoop(Statement):
    """While loop statement."""
    condition: Expr = None
    body: List[Statement] = field(default_factory=list)
    loop_name: Optional[str] = None  # Ada loop label
    kind: str = 'while_loop'


@dataclass
class ForLoop(Statement):
    """For loop statement."""
    variable: str = ''
    range_expr: Expr = None  # RangeExpr or iterable
    reverse: bool = False  # Ada: reverse iteration
    body: List[Statement] = field(default_factory=list)
    loop_name: Optional[str] = None
    kind: str = 'for_loop'


@dataclass
class BasicLoop(Statement):
    """Basic loop (Ada: loop ... end loop)."""
    body: List[Statement] = field(default_factory=list)
    loop_name: Optional[str] = None
    kind: str = 'basic_loop'


@dataclass
class ExitStatement(Statement):
    """Exit/break statement."""
    loop_name: Optional[str] = None  # Target loop
    condition: Optional[Expr] = None  # exit when condition
    kind: str = 'exit_statement'


@dataclass
class ReturnStatement(Statement):
    """Return statement."""
    value: Optional[Expr] = None
    kind: str = 'return_statement'


@dataclass
class NullStatement(Statement):
    """Null/no-op statement."""
    kind: str = 'null_statement'


@dataclass
class BlockStatement(Statement):
    """Block statement with declarations."""
    name: Optional[str] = None
    declarations: List['Declaration'] = field(default_factory=list)
    body: List[Statement] = field(default_factory=list)
    kind: str = 'block_statement'


@dataclass
class RaiseStatement(Statement):
    """Raise/throw exception statement."""
    exception_name: Optional[str] = None
    message: Optional[Expr] = None
    kind: str = 'raise_statement'


@dataclass
class CallStatement(Statement):
    """Procedure/function call as statement."""
    call: CallExpr = None
    kind: str = 'call_statement'


# =============================================================================
# Exception Handling
# =============================================================================

@dataclass
class ExceptionHandler(SystemsNode):
    """Exception handler clause."""
    exception_names: List[str] = field(default_factory=list)
    choice_parameter: Optional[str] = None  # Occurrence parameter
    body: List[Statement] = field(default_factory=list)
    kind: str = 'exception_handler'


@dataclass
class TryStatement(Statement):
    """Try-except/when statement."""
    body: List[Statement] = field(default_factory=list)
    handlers: List[ExceptionHandler] = field(default_factory=list)
    finally_body: Optional[List[Statement]] = None
    kind: str = 'try_statement'


# =============================================================================
# Declarations
# =============================================================================

@dataclass
class Declaration(SystemsNode):
    """Base class for declarations."""
    kind: str = 'declaration'


@dataclass
class VariableDecl(Declaration):
    """Variable declaration."""
    name: str = ''
    type_ref: TypeRef = None
    initializer: Optional[Expr] = None
    is_constant: bool = False
    is_aliased: bool = False  # Ada aliased
    visibility: Visibility = Visibility.PUBLIC
    is_ghost: bool = False  # SPARK ghost
    kind: str = 'variable_decl'


@dataclass
class ConstantDecl(Declaration):
    """Constant declaration."""
    name: str = ''
    type_ref: Optional[TypeRef] = None  # May be inferred
    value: Expr = None
    visibility: Visibility = Visibility.PUBLIC
    kind: str = 'constant_decl'


@dataclass
class ComponentDecl(SystemsNode):
    """Record/struct component (field) declaration."""
    name: str = ''
    type_ref: TypeRef = None
    default_value: Optional[Expr] = None
    kind: str = 'component_decl'


@dataclass
class Discriminant(SystemsNode):
    """Discriminant declaration (Ada)."""
    name: str = ''
    type_ref: TypeRef = None
    default_value: Optional[Expr] = None
    kind: str = 'discriminant'


# =============================================================================
# Parameters
# =============================================================================

@dataclass
class Parameter(SystemsNode):
    """Subprogram parameter."""
    name: str = ''
    type_ref: TypeRef = None
    mode: Mode = Mode.IN
    default_value: Optional[Expr] = None
    is_aliased: bool = False
    kind: str = 'parameter'


# =============================================================================
# Subprograms
# =============================================================================

@dataclass
class Subprogram(SystemsNode):
    """Function or Procedure definition.
    
    Ada: procedure/function with contracts
    D: function with in/out/invariant
    """
    name: str = ''
    parameters: List[Parameter] = field(default_factory=list)
    return_type: Optional[TypeRef] = None  # None for procedures
    body: List[Statement] = field(default_factory=list)
    
    # Contracts (from verification.py, referenced here)
    preconditions: List['Contract'] = field(default_factory=list)
    postconditions: List['Contract'] = field(default_factory=list)
    contract_cases: List['ContractCase'] = field(default_factory=list)
    
    # SPARK annotations
    global_spec: Optional['GlobalSpec'] = None
    depends_spec: Optional['DependsSpec'] = None
    
    # Properties
    visibility: Visibility = Visibility.PUBLIC
    is_pure: bool = False
    is_inline: bool = False
    is_abstract: bool = False
    is_overriding: bool = False
    spark_mode: bool = False
    
    # D-specific
    safety_level: SafetyLevel = SafetyLevel.SYSTEM
    is_nothrow: bool = False
    is_nogc: bool = False
    is_ctfe: bool = False  # CTFE-capable
    
    # Local declarations
    local_declarations: List[Declaration] = field(default_factory=list)
    
    kind: str = 'subprogram'


# =============================================================================
# Imports
# =============================================================================

@dataclass
class Import(SystemsNode):
    """Import/with clause.
    
    Ada: with Package; use Package;
    D: import module; import module : symbol;
    """
    module: str = ''
    use_clause: bool = False  # Ada: add use clause
    selective_imports: List[str] = field(default_factory=list)  # D: selective imports
    renamed_as: Optional[str] = None  # Renaming
    is_private: bool = False  # Private with (Ada)
    is_limited: bool = False  # Limited with (Ada)
    kind: str = 'import'


# =============================================================================
# Package/Module
# =============================================================================

@dataclass
class Package(SystemsNode):
    """Package/Module definition (Ada packages, D modules).
    
    Ada: package specification and body
    D: module with imports and declarations
    """
    name: str = ''
    imports: List[Import] = field(default_factory=list)
    types: List['TypeDecl'] = field(default_factory=list)
    constants: List[ConstantDecl] = field(default_factory=list)
    variables: List[VariableDecl] = field(default_factory=list)
    subprograms: List[Subprogram] = field(default_factory=list)
    tasks: List['TaskType'] = field(default_factory=list)
    protected_types: List['ProtectedType'] = field(default_factory=list)
    generics: List['GenericUnit'] = field(default_factory=list)
    child_packages: List['Package'] = field(default_factory=list)
    exceptions: List['ExceptionDecl'] = field(default_factory=list)
    
    # Package properties
    visibility: Visibility = Visibility.PUBLIC
    spark_mode: bool = False
    is_pure: bool = False  # Ada Pure package
    is_preelaborate: bool = False  # Ada Preelaborate
    
    # Private section
    private_types: List['TypeDecl'] = field(default_factory=list)
    private_constants: List[ConstantDecl] = field(default_factory=list)
    private_variables: List[VariableDecl] = field(default_factory=list)
    
    # Body (for package bodies)
    body_declarations: List[Declaration] = field(default_factory=list)
    initialization: List[Statement] = field(default_factory=list)
    finalization: List[Statement] = field(default_factory=list)
    
    kind: str = 'package'


@dataclass
class ExceptionDecl(Declaration):
    """Exception declaration (Ada)."""
    name: str = ''
    kind: str = 'exception_decl'


# Forward references for type annotations
Contract = 'Contract'
ContractCase = 'ContractCase'
GlobalSpec = 'GlobalSpec'
DependsSpec = 'DependsSpec'
TypeDecl = 'TypeDecl'
TaskType = 'TaskType'
ProtectedType = 'ProtectedType'
GenericUnit = 'GenericUnit'
