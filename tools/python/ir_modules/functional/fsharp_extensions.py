#!/usr/bin/env python3
"""STUNIR Functional IR - F# Extensions.

This module defines F#-specific IR constructs including computation expressions,
units of measure, active patterns, and .NET interoperability constructs.

Usage:
    from ir.functional.fsharp_extensions import (
        ComputationExpr, MeasureType, ActivePattern, ClassDef
    )
    
    # Create an async computation expression
    async_expr = ComputationExpr(
        builder='async',
        body=[ComputationReturn(value=some_expr)]
    )
"""

from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict

from ir.functional.functional_ir import FunctionalNode, Expr, TypeExpr
from ir.functional.pattern import Pattern
from ir.functional.adt import TypeParameter


# =============================================================================
# Computation Expressions (async, seq, query, etc.)
# =============================================================================

@dataclass
class ComputationStatement(FunctionalNode):
    """Base class for computation expression statements."""
    kind: str = 'computation_statement'


@dataclass
class ComputationLet(ComputationStatement):
    """let! x = expr or let x = expr in computation expression.
    
    Attributes:
        pattern: Pattern to bind
        value: Expression to evaluate
        is_bang: True for let! (monadic bind), False for let
    """
    pattern: Pattern = None
    value: Expr = None
    is_bang: bool = False
    kind: str = 'computation_let'


@dataclass
class ComputationDo(ComputationStatement):
    """do! expr or do expr in computation expression.
    
    Attributes:
        expr: Expression to execute
        is_bang: True for do! (monadic action), False for do
    """
    expr: Expr = None
    is_bang: bool = False
    kind: str = 'computation_do'


@dataclass
class ComputationReturn(ComputationStatement):
    """return expr or return! expr in computation expression.
    
    Attributes:
        value: Expression to return
        is_bang: True for return! (from nested computation)
    """
    value: Expr = None
    is_bang: bool = False
    kind: str = 'computation_return'


@dataclass
class ComputationYield(ComputationStatement):
    """yield expr or yield! expr (for seq expressions).
    
    Attributes:
        value: Expression to yield
        is_bang: True for yield! (flatten nested sequence)
    """
    value: Expr = None
    is_bang: bool = False
    kind: str = 'computation_yield'


@dataclass
class ComputationFor(ComputationStatement):
    """for pattern in expr do body in computation expression.
    
    Attributes:
        pattern: Loop variable pattern
        source: Source expression to iterate
        body: Body statements
    """
    pattern: Pattern = None
    source: Expr = None
    body: List['ComputationStatement'] = field(default_factory=list)
    kind: str = 'computation_for'


@dataclass
class ComputationWhile(ComputationStatement):
    """while cond do body in computation expression.
    
    Attributes:
        condition: Loop condition
        body: Body statements
    """
    condition: Expr = None
    body: List['ComputationStatement'] = field(default_factory=list)
    kind: str = 'computation_while'


@dataclass
class ComputationIf(ComputationStatement):
    """if cond then ... else ... in computation expression.
    
    Attributes:
        condition: Condition expression
        then_body: Then branch statements
        else_body: Optional else branch statements
    """
    condition: Expr = None
    then_body: List['ComputationStatement'] = field(default_factory=list)
    else_body: List['ComputationStatement'] = field(default_factory=list)
    kind: str = 'computation_if'


@dataclass
class ComputationTry(ComputationStatement):
    """try ... with ... in computation expression.
    
    Attributes:
        body: Try body statements
        handlers: Exception handlers as (pattern, statements) pairs
        finally_body: Optional finally statements
    """
    body: List['ComputationStatement'] = field(default_factory=list)
    handlers: List[tuple] = field(default_factory=list)  # [(pattern, statements), ...]
    finally_body: List['ComputationStatement'] = field(default_factory=list)
    kind: str = 'computation_try'


@dataclass
class ComputationExpr(Expr):
    """F# computation expression (async, seq, query, etc.).
    
    Represents computation expressions like:
    - async { let! x = fetchAsync(); return x }
    - seq { for i in 1..10 do yield i * i }
    - query { for x in xs do where (x > 5); select x }
    
    Attributes:
        builder: Builder name ('async', 'seq', 'query', or custom)
        body: List of computation statements
    """
    builder: str = ''
    body: List[ComputationStatement] = field(default_factory=list)
    kind: str = 'computation_expr'


# =============================================================================
# Units of Measure
# =============================================================================

@dataclass
class MeasureExpr(FunctionalNode):
    """Base class for measure expressions."""
    kind: str = 'measure_expr'


@dataclass
class MeasureUnit(MeasureExpr):
    """Single unit of measure (e.g., m, kg, s).
    
    Attributes:
        name: Unit name
    """
    name: str = ''
    kind: str = 'measure_unit'


@dataclass
class MeasureProd(MeasureExpr):
    """Product of measures (e.g., m * kg).
    
    Attributes:
        left: Left measure
        right: Right measure
    """
    left: 'MeasureExpr' = None
    right: 'MeasureExpr' = None
    kind: str = 'measure_prod'


@dataclass
class MeasureDiv(MeasureExpr):
    """Division of measures (e.g., m / s for velocity).
    
    Attributes:
        numerator: Numerator measure
        denominator: Denominator measure
    """
    numerator: 'MeasureExpr' = None
    denominator: 'MeasureExpr' = None
    kind: str = 'measure_div'


@dataclass
class MeasurePow(MeasureExpr):
    """Power of measure (e.g., m^2 for area).
    
    Attributes:
        base: Base measure
        power: Integer exponent
    """
    base: 'MeasureExpr' = None
    power: int = 1
    kind: str = 'measure_pow'


@dataclass
class MeasureOne(MeasureExpr):
    """Dimensionless measure (1 or unitless)."""
    kind: str = 'measure_one'


@dataclass
class MeasureType(TypeExpr):
    """Type with unit of measure annotation.
    
    Represents types like float<m>, int<kg>, float<m/s>.
    
    Attributes:
        base_type: Underlying numeric type (float, int, etc.)
        measure: Measure expression
    """
    base_type: TypeExpr = None
    measure: MeasureExpr = None
    kind: str = 'measure_type'


@dataclass
class MeasureDeclaration(FunctionalNode):
    """Unit of measure declaration.
    
    Represents [<Measure>] type declarations.
    
    Attributes:
        name: Measure name
        base_measure: Optional base measure for derived units
        conversion_factor: Optional conversion factor
    """
    name: str = ''
    base_measure: Optional[MeasureExpr] = None
    conversion_factor: Optional[float] = None
    kind: str = 'measure_declaration'


# =============================================================================
# Active Patterns
# =============================================================================

@dataclass
class ActivePattern(FunctionalNode):
    """Active pattern definition.
    
    Represents both total and partial active patterns:
    - Total: let (|Even|Odd|) n = if n % 2 = 0 then Even else Odd
    - Partial: let (|Integer|_|) s = try Some(int s) with _ -> None
    
    Attributes:
        cases: List of case names
        is_partial: True for partial patterns (with |_|)
        params: Parameter names
        body: Pattern body expression
    """
    cases: List[str] = field(default_factory=list)
    is_partial: bool = False
    params: List[str] = field(default_factory=list)
    body: Expr = None
    kind: str = 'active_pattern'


@dataclass
class ActivePatternMatch(Pattern):
    """Active pattern usage in pattern matching.
    
    Attributes:
        pattern_name: Name of the active pattern case
        args: Arguments to the pattern
    """
    pattern_name: str = ''
    args: List[Pattern] = field(default_factory=list)
    kind: str = 'active_pattern_match'


@dataclass
class ParameterizedActivePattern(FunctionalNode):
    """Parameterized active pattern.
    
    let (|DivisibleBy|_|) divisor n = if n % divisor = 0 then Some() else None
    
    Attributes:
        cases: Case names
        is_partial: Whether partial
        pattern_params: Parameters that appear in the pattern
        additional_params: Additional parameters
        body: Pattern body
    """
    cases: List[str] = field(default_factory=list)
    is_partial: bool = False
    pattern_params: List[str] = field(default_factory=list)
    additional_params: List[str] = field(default_factory=list)
    body: Expr = None
    kind: str = 'parameterized_active_pattern'


# =============================================================================
# .NET Interoperability Constructs
# =============================================================================

@dataclass
class Attribute(FunctionalNode):
    """F# attribute (e.g., [<Serializable>], [<CLIEvent>]).
    
    Attributes:
        name: Attribute name (without [< >])
        args: Attribute arguments (positional and named)
    """
    name: str = ''
    args: List[Any] = field(default_factory=list)
    named_args: Dict[str, Any] = field(default_factory=dict)
    kind: str = 'attribute'


@dataclass
class ClassMember(FunctionalNode):
    """Base class for class members."""
    attributes: List[Attribute] = field(default_factory=list)
    kind: str = 'class_member'


@dataclass
class ClassField(ClassMember):
    """Class field (let binding inside type).
    
    Attributes:
        name: Field name
        field_type: Field type (optional if inferred)
        is_mutable: Whether field is mutable
        default_value: Optional default value
        is_static: Whether field is static
    """
    name: str = ''
    field_type: Optional[TypeExpr] = None
    is_mutable: bool = False
    default_value: Optional[Expr] = None
    is_static: bool = False
    kind: str = 'class_field'


@dataclass
class ClassMethod(ClassMember):
    """Class method (member function).
    
    Attributes:
        name: Method name
        params: Parameter list as (name, type) tuples
        return_type: Return type
        body: Method body
        is_static: Whether method is static
        is_override: Whether method overrides base
        is_abstract: Whether method is abstract
        is_virtual: Whether method is virtual
    """
    name: str = ''
    params: List[tuple] = field(default_factory=list)  # [(name, type), ...]
    return_type: Optional[TypeExpr] = None
    body: Optional[Expr] = None
    is_static: bool = False
    is_override: bool = False
    is_abstract: bool = False
    is_virtual: bool = False
    kind: str = 'class_method'


@dataclass
class ClassProperty(ClassMember):
    """Class property (member with get/set).
    
    Attributes:
        name: Property name
        property_type: Property type
        getter: Getter expression (None if no getter)
        setter: Setter expression (None if no setter)
        is_static: Whether property is static
        is_override: Whether property overrides base
    """
    name: str = ''
    property_type: Optional[TypeExpr] = None
    getter: Optional[Expr] = None
    setter: Optional[Expr] = None
    is_static: bool = False
    is_override: bool = False
    kind: str = 'class_property'


@dataclass
class ClassEvent(ClassMember):
    """Class event (CLIEvent).
    
    Attributes:
        name: Event name
        event_type: Event type (typically IEvent<...>)
    """
    name: str = ''
    event_type: Optional[TypeExpr] = None
    kind: str = 'class_event'


@dataclass
class Constructor(FunctionalNode):
    """Class constructor.
    
    Attributes:
        params: Constructor parameters
        body: Constructor body (do bindings)
        call_base: Whether to call base constructor
        base_args: Arguments to pass to base constructor
    """
    params: List[tuple] = field(default_factory=list)
    body: List[Expr] = field(default_factory=list)
    call_base: bool = False
    base_args: List[Expr] = field(default_factory=list)
    kind: str = 'constructor'


@dataclass
class ClassDef(FunctionalNode):
    """F# class definition for .NET interop.
    
    Attributes:
        name: Class name
        type_params: Type parameters
        attributes: Class attributes
        base_class: Base class (if any)
        interfaces: Implemented interfaces
        primary_constructor: Primary constructor params
        additional_constructors: Additional constructors
        members: Class members (fields, methods, properties)
    """
    name: str = ''
    type_params: List[TypeParameter] = field(default_factory=list)
    attributes: List[Attribute] = field(default_factory=list)
    base_class: Optional[str] = None
    interfaces: List[str] = field(default_factory=list)
    primary_constructor: List[tuple] = field(default_factory=list)
    additional_constructors: List[Constructor] = field(default_factory=list)
    members: List[ClassMember] = field(default_factory=list)
    kind: str = 'class_def'


@dataclass
class InterfaceMember(FunctionalNode):
    """Interface member signature.
    
    Attributes:
        name: Member name
        member_type: Member type signature
    """
    name: str = ''
    member_type: TypeExpr = None
    kind: str = 'interface_member'


@dataclass
class InterfaceDef(FunctionalNode):
    """F# interface definition.
    
    Attributes:
        name: Interface name
        type_params: Type parameters
        attributes: Interface attributes
        base_interfaces: Extended interfaces
        members: Interface members
    """
    name: str = ''
    type_params: List[TypeParameter] = field(default_factory=list)
    attributes: List[Attribute] = field(default_factory=list)
    base_interfaces: List[str] = field(default_factory=list)
    members: List[InterfaceMember] = field(default_factory=list)
    kind: str = 'interface_def'


@dataclass
class StructDef(FunctionalNode):
    """F# struct definition ([<Struct>] type).
    
    Attributes:
        name: Struct name
        type_params: Type parameters
        fields: Struct fields
    """
    name: str = ''
    type_params: List[TypeParameter] = field(default_factory=list)
    fields: List[ClassField] = field(default_factory=list)
    kind: str = 'struct_def'


# =============================================================================
# Type Provider Support
# =============================================================================

@dataclass
class TypeProviderRef(TypeExpr):
    """Reference to a type provider generated type.
    
    Attributes:
        provider: Provider name (e.g., 'SqlDataConnection')
        args: Provider arguments (connection string, etc.)
        accessed_type: The type path within the provider
    """
    provider: str = ''
    args: List[str] = field(default_factory=list)
    accessed_type: str = ''
    kind: str = 'type_provider_ref'


# =============================================================================
# Additional F# Constructs
# =============================================================================

@dataclass
class ObjectExpr(Expr):
    """Object expression for creating anonymous interface implementations.
    
    { new IInterface with
        member x.Method() = ...
    }
    
    Attributes:
        interface_type: Interface being implemented
        members: Member implementations
    """
    interface_type: TypeExpr = None
    base_call: Optional[tuple] = None  # (base_type, args)
    members: List[ClassMember] = field(default_factory=list)
    kind: str = 'object_expr'


@dataclass
class UseExpr(Expr):
    """use binding for automatic disposal.
    
    use resource = createResource()
    
    Attributes:
        name: Resource binding name
        value: Resource expression
        body: Body expression using the resource
    """
    name: str = ''
    value: Expr = None
    body: Expr = None
    kind: str = 'use_expr'


@dataclass
class UsingExpr(Expr):
    """using! in computation expressions.
    
    Attributes:
        name: Resource binding name
        value: Resource expression
        body: Body computation statements
    """
    name: str = ''
    value: Expr = None
    body: List[ComputationStatement] = field(default_factory=list)
    kind: str = 'using_expr'


@dataclass
class QuotationExpr(Expr):
    """F# code quotation.
    
    <@ expr @> for typed quotations
    <@@ expr @@> for untyped quotations
    
    Attributes:
        body: Quoted expression
        is_typed: True for <@ @>, False for <@@ @@>
    """
    body: Expr = None
    is_typed: bool = True
    kind: str = 'quotation_expr'


@dataclass
class PipelineExpr(Expr):
    """Pipeline expression (|> or <|).
    
    Attributes:
        left: Left operand
        right: Right operand (function)
        direction: 'forward' for |>, 'backward' for <|
    """
    left: Expr = None
    right: Expr = None
    direction: str = 'forward'  # 'forward' for |>, 'backward' for <|
    kind: str = 'pipeline_expr'


@dataclass
class CompositionExpr(Expr):
    """Function composition (>> or <<).
    
    Attributes:
        left: Left function
        right: Right function
        direction: 'forward' for >>, 'backward' for <<
    """
    left: Expr = None
    right: Expr = None
    direction: str = 'forward'  # 'forward' for >>, 'backward' for <<
    kind: str = 'composition_expr'
