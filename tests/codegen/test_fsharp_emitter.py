#!/usr/bin/env python3
"""Tests for STUNIR F# Emitter.

This module contains comprehensive tests for the F# code emitter,
including tests for:
- Basic expressions and types
- Discriminated unions
- Record types
- Pattern matching
- Computation expressions
- Units of measure
- Active patterns
- .NET interoperability

Usage:
    pytest tests/codegen/test_fsharp_emitter.py -v
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from targets.functional.fsharp_emitter import FSharpEmitter
from ir.functional import (
    # Types
    TypeExpr, TypeVar, TypeCon, FunctionType, TupleType, ListType,
    # Expressions
    Expr, LiteralExpr, VarExpr, AppExpr, LambdaExpr, LetExpr, IfExpr,
    CaseBranch, CaseExpr, ListExpr, TupleExpr, BinaryOpExpr, UnaryOpExpr,
    # Patterns
    Pattern, WildcardPattern, VarPattern, LiteralPattern,
    ConstructorPattern, TuplePattern, ListPattern, AsPattern, OrPattern,
    # ADTs
    TypeParameter, DataConstructor, DataType, TypeAlias,
    RecordField, RecordType,
    Import, FunctionClause, FunctionDef, Module,
    # F# specific
    ComputationExpr, ComputationLet, ComputationDo, ComputationReturn,
    ComputationYield, ComputationFor, ComputationWhile,
    MeasureType, MeasureUnit, MeasureProd, MeasureDiv, MeasurePow, MeasureDeclaration,
    ActivePattern, ActivePatternMatch,
    Attribute, ClassField, ClassMethod, ClassProperty, ClassDef,
    InterfaceMember, InterfaceDef,
    ObjectExpr, UseExpr, PipelineExpr, CompositionExpr,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def emitter():
    """Create a fresh FSharpEmitter instance."""
    return FSharpEmitter()


@pytest.fixture
def emitter_with_config():
    """Create an FSharpEmitter with custom config."""
    return FSharpEmitter(config={'indent_size': 2, 'generate_fsproj': True})


# =============================================================================
# Test: Basic Expressions
# =============================================================================

class TestBasicExpressions:
    """Tests for basic expression emission."""
    
    def test_literal_int(self, emitter):
        """TC-FS-001: Integer literal emission."""
        expr = LiteralExpr(value=42, literal_type='int')
        result = emitter._emit_expr(expr)
        assert result == "42"
    
    def test_literal_float(self, emitter):
        """TC-FS-002: Float literal emission."""
        expr = LiteralExpr(value=3.14, literal_type='float')
        result = emitter._emit_expr(expr)
        assert "3.14" in result
    
    def test_literal_bool_true(self, emitter):
        """TC-FS-003: Boolean true literal."""
        expr = LiteralExpr(value=True, literal_type='bool')
        result = emitter._emit_expr(expr)
        assert result == "true"
    
    def test_literal_bool_false(self, emitter):
        """TC-FS-004: Boolean false literal."""
        expr = LiteralExpr(value=False, literal_type='bool')
        result = emitter._emit_expr(expr)
        assert result == "false"
    
    def test_literal_string(self, emitter):
        """TC-FS-005: String literal emission."""
        expr = LiteralExpr(value="hello", literal_type='string')
        result = emitter._emit_expr(expr)
        assert result == '"hello"'
    
    def test_literal_char(self, emitter):
        """TC-FS-006: Character literal emission."""
        expr = LiteralExpr(value='a', literal_type='char')
        result = emitter._emit_expr(expr)
        assert result == "'a'"
    
    def test_variable(self, emitter):
        """TC-FS-007: Variable expression."""
        expr = VarExpr(name='x')
        result = emitter._emit_expr(expr)
        assert result == "x"
    
    def test_binary_op_add(self, emitter):
        """TC-FS-008: Binary addition."""
        expr = BinaryOpExpr(
            op='+',
            left=LiteralExpr(value=1, literal_type='int'),
            right=LiteralExpr(value=2, literal_type='int')
        )
        result = emitter._emit_expr(expr)
        assert '1 + 2' in result
    
    def test_binary_op_equal(self, emitter):
        """TC-FS-009: Equality comparison (== maps to =)."""
        expr = BinaryOpExpr(
            op='==',
            left=VarExpr(name='x'),
            right=LiteralExpr(value=0, literal_type='int')
        )
        result = emitter._emit_expr(expr)
        assert '=' in result
        assert 'x' in result
    
    def test_binary_op_not_equal(self, emitter):
        """TC-FS-010: Not-equal comparison."""
        expr = BinaryOpExpr(
            op='!=',
            left=VarExpr(name='x'),
            right=VarExpr(name='y')
        )
        result = emitter._emit_expr(expr)
        assert '<>' in result
    
    def test_unary_not(self, emitter):
        """TC-FS-011: Unary not operator."""
        expr = UnaryOpExpr(op='not', operand=VarExpr(name='flag'))
        result = emitter._emit_expr(expr)
        assert 'not' in result
        assert 'flag' in result
    
    def test_lambda(self, emitter):
        """TC-FS-012: Lambda expression."""
        expr = LambdaExpr(param='x', body=VarExpr(name='x'))
        result = emitter._emit_expr(expr)
        assert 'fun x ->' in result
    
    def test_let_binding(self, emitter):
        """TC-FS-013: Let binding expression."""
        expr = LetExpr(
            name='x',
            value=LiteralExpr(value=10, literal_type='int'),
            body=BinaryOpExpr(
                op='+',
                left=VarExpr(name='x'),
                right=LiteralExpr(value=1, literal_type='int')
            )
        )
        result = emitter._emit_expr(expr)
        assert 'let x = 10' in result
        assert 'in' in result
    
    def test_if_expression(self, emitter):
        """TC-FS-014: If-then-else expression."""
        expr = IfExpr(
            condition=BinaryOpExpr(
                op='>',
                left=VarExpr(name='x'),
                right=LiteralExpr(value=0, literal_type='int')
            ),
            then_branch=LiteralExpr(value="positive", literal_type='string'),
            else_branch=LiteralExpr(value="non-positive", literal_type='string')
        )
        result = emitter._emit_expr(expr)
        assert 'if' in result
        assert 'then' in result
        assert 'else' in result
    
    def test_tuple_expression(self, emitter):
        """TC-FS-015: Tuple expression."""
        expr = TupleExpr(elements=[
            LiteralExpr(value=1, literal_type='int'),
            LiteralExpr(value="hello", literal_type='string')
        ])
        result = emitter._emit_expr(expr)
        assert '(' in result
        assert ',' in result
        assert ')' in result
    
    def test_list_expression(self, emitter):
        """TC-FS-016: List expression."""
        expr = ListExpr(elements=[
            LiteralExpr(value=1, literal_type='int'),
            LiteralExpr(value=2, literal_type='int'),
            LiteralExpr(value=3, literal_type='int')
        ])
        result = emitter._emit_expr(expr)
        assert '[' in result
        assert ';' in result  # F# uses semicolons
        assert ']' in result


# =============================================================================
# Test: Type Emission
# =============================================================================

class TestTypeEmission:
    """Tests for type expression emission."""
    
    def test_type_var(self, emitter):
        """TC-FS-017: Type variable emission."""
        type_expr = TypeVar(name='a')
        result = emitter._emit_type(type_expr)
        assert result == "'a"
    
    def test_type_con_int(self, emitter):
        """TC-FS-018: Int type emission."""
        type_expr = TypeCon(name='int')
        result = emitter._emit_type(type_expr)
        assert result == "int"
    
    def test_type_con_list(self, emitter):
        """TC-FS-019: List type emission (postfix)."""
        type_expr = TypeCon(name='list', args=[TypeVar(name='a')])
        result = emitter._emit_type(type_expr)
        assert "'a list" in result
    
    def test_type_con_option(self, emitter):
        """TC-FS-020: Option type emission (postfix)."""
        type_expr = TypeCon(name='option', args=[TypeCon(name='int')])
        result = emitter._emit_type(type_expr)
        assert "int option" in result
    
    def test_function_type(self, emitter):
        """TC-FS-021: Function type emission."""
        type_expr = FunctionType(
            param_type=TypeCon(name='int'),
            return_type=TypeCon(name='int')
        )
        result = emitter._emit_type(type_expr)
        assert '->' in result
        assert 'int' in result
    
    def test_tuple_type(self, emitter):
        """TC-FS-022: Tuple type emission."""
        type_expr = TupleType(elements=[
            TypeCon(name='int'),
            TypeCon(name='string')
        ])
        result = emitter._emit_type(type_expr)
        assert '*' in result


# =============================================================================
# Test: Functions
# =============================================================================

class TestFunctions:
    """Tests for function definition emission."""
    
    def test_simple_function(self, emitter):
        """TC-FS-023: Simple function emission."""
        func = FunctionDef(
            name='add',
            clauses=[FunctionClause(
                patterns=[VarPattern(name='x'), VarPattern(name='y')],
                body=BinaryOpExpr(
                    op='+',
                    left=VarExpr(name='x'),
                    right=VarExpr(name='y')
                )
            )]
        )
        result = emitter._emit_function(func)
        assert 'let add x y =' in result
    
    def test_recursive_function(self, emitter):
        """TC-FS-024: Recursive function emission."""
        func = FunctionDef(
            name='factorial',
            clauses=[FunctionClause(
                patterns=[VarPattern(name='n')],
                body=IfExpr(
                    condition=BinaryOpExpr(
                        op='=',
                        left=VarExpr(name='n'),
                        right=LiteralExpr(value=0, literal_type='int')
                    ),
                    then_branch=LiteralExpr(value=1, literal_type='int'),
                    else_branch=BinaryOpExpr(
                        op='*',
                        left=VarExpr(name='n'),
                        right=AppExpr(
                            func=VarExpr(name='factorial'),
                            arg=BinaryOpExpr(
                                op='-',
                                left=VarExpr(name='n'),
                                right=LiteralExpr(value=1, literal_type='int')
                            )
                        )
                    )
                )
            )]
        )
        result = emitter._emit_function(func)
        assert 'let rec factorial' in result
    
    def test_pattern_matching_function(self, emitter):
        """TC-FS-025: Pattern matching function."""
        func = FunctionDef(
            name='describe',
            clauses=[
                FunctionClause(
                    patterns=[ConstructorPattern(constructor='None')],
                    body=LiteralExpr(value="Nothing", literal_type='string')
                ),
                FunctionClause(
                    patterns=[ConstructorPattern(
                        constructor='Some',
                        args=[VarPattern(name='x')]
                    )],
                    body=VarExpr(name='x')
                )
            ]
        )
        result = emitter._emit_function(func)
        assert 'match' in result
        assert '| None ->' in result
        assert '| Some(x) ->' in result


# =============================================================================
# Test: Discriminated Unions
# =============================================================================

class TestDiscriminatedUnions:
    """Tests for discriminated union emission."""
    
    def test_simple_union(self, emitter):
        """TC-FS-026: Simple discriminated union."""
        data = DataType(
            name='Color',
            constructors=[
                DataConstructor(name='Red'),
                DataConstructor(name='Green'),
                DataConstructor(name='Blue')
            ]
        )
        result = emitter._emit_discriminated_union(data)
        assert 'type Color =' in result
        assert '| Red' in result
        assert '| Green' in result
        assert '| Blue' in result
    
    def test_union_with_data(self, emitter):
        """TC-FS-027: Union with data."""
        data = DataType(
            name='Shape',
            constructors=[
                DataConstructor(name='Circle', fields=[TypeCon(name='float')]),
                DataConstructor(name='Rectangle', fields=[
                    TypeCon(name='float'),
                    TypeCon(name='float')
                ])
            ]
        )
        result = emitter._emit_discriminated_union(data)
        assert 'type Shape =' in result
        assert '| Circle of float' in result
        assert '| Rectangle of float * float' in result
    
    def test_generic_union(self, emitter):
        """TC-FS-028: Generic discriminated union."""
        data = DataType(
            name='Option',
            type_params=[TypeParameter(name='a')],
            constructors=[
                DataConstructor(name='None'),
                DataConstructor(name='Some', fields=[TypeVar(name='a')])
            ]
        )
        result = emitter._emit_discriminated_union(data)
        assert "type Option<'a> =" in result
        assert '| None' in result
        assert "| Some of 'a" in result
    
    def test_union_with_named_fields(self, emitter):
        """TC-FS-029: Union with named fields."""
        data = DataType(
            name='Person',
            constructors=[
                DataConstructor(
                    name='Person',
                    fields=[TypeCon(name='string'), TypeCon(name='int')],
                    field_names=['name', 'age']
                )
            ]
        )
        result = emitter._emit_discriminated_union(data)
        assert 'name: string' in result
        assert 'age: int' in result


# =============================================================================
# Test: Records
# =============================================================================

class TestRecords:
    """Tests for record type emission."""
    
    def test_simple_record(self, emitter):
        """TC-FS-030: Simple record type."""
        record = RecordType(
            name='Point',
            fields=[
                RecordField(name='X', field_type=TypeCon(name='float')),
                RecordField(name='Y', field_type=TypeCon(name='float'))
            ]
        )
        result = emitter._emit_record_type(record)
        assert 'type Point =' in result
        assert 'X: float' in result
        assert 'Y: float' in result
    
    def test_record_with_mutable_field(self, emitter):
        """TC-FS-031: Record with mutable field."""
        record = RecordType(
            name='Counter',
            fields=[
                RecordField(name='Value', field_type=TypeCon(name='int'), mutable=True)
            ]
        )
        result = emitter._emit_record_type(record)
        assert 'mutable Value' in result
    
    def test_generic_record(self, emitter):
        """TC-FS-032: Generic record type."""
        record = RecordType(
            name='Pair',
            type_params=[TypeParameter(name='a'), TypeParameter(name='b')],
            fields=[
                RecordField(name='First', field_type=TypeVar(name='a')),
                RecordField(name='Second', field_type=TypeVar(name='b'))
            ]
        )
        result = emitter._emit_record_type(record)
        assert "type Pair<'a, 'b> =" in result


# =============================================================================
# Test: Computation Expressions
# =============================================================================

class TestComputationExpressions:
    """Tests for computation expression emission."""
    
    def test_async_basic(self, emitter):
        """TC-FS-033: Basic async computation expression."""
        comp = ComputationExpr(
            builder='async',
            body=[
                ComputationLet(
                    pattern=VarPattern(name='data'),
                    value=AppExpr(
                        func=VarExpr(name='fetchAsync'),
                        arg=None
                    ),
                    is_bang=True
                ),
                ComputationReturn(
                    value=VarExpr(name='data'),
                    is_bang=False
                )
            ]
        )
        result = emitter._emit_computation_expr(comp)
        assert 'async {' in result
        assert 'let! data' in result
        assert 'return data' in result
    
    def test_seq_with_yield(self, emitter):
        """TC-FS-034: Seq computation with yield."""
        comp = ComputationExpr(
            builder='seq',
            body=[
                ComputationYield(
                    value=LiteralExpr(value=1, literal_type='int'),
                    is_bang=False
                ),
                ComputationYield(
                    value=LiteralExpr(value=2, literal_type='int'),
                    is_bang=False
                )
            ]
        )
        result = emitter._emit_computation_expr(comp)
        assert 'seq {' in result
        assert 'yield 1' in result
        assert 'yield 2' in result
    
    def test_seq_with_for(self, emitter):
        """TC-FS-035: Seq computation with for loop."""
        comp = ComputationExpr(
            builder='seq',
            body=[
                ComputationFor(
                    pattern=VarPattern(name='i'),
                    source=ListExpr(elements=[
                        LiteralExpr(value=1, literal_type='int'),
                        LiteralExpr(value=2, literal_type='int'),
                        LiteralExpr(value=3, literal_type='int')
                    ]),
                    body=[
                        ComputationYield(
                            value=BinaryOpExpr(
                                op='*',
                                left=VarExpr(name='i'),
                                right=VarExpr(name='i')
                            ),
                            is_bang=False
                        )
                    ]
                )
            ]
        )
        result = emitter._emit_computation_expr(comp)
        assert 'seq {' in result
        assert 'for i in' in result
        assert 'yield' in result
    
    def test_async_with_do_bang(self, emitter):
        """TC-FS-036: Async with do!."""
        comp = ComputationExpr(
            builder='async',
            body=[
                ComputationDo(
                    expr=AppExpr(
                        func=VarExpr(name='Async.Sleep'),
                        arg=LiteralExpr(value=1000, literal_type='int')
                    ),
                    is_bang=True
                ),
                ComputationReturn(
                    value=LiteralExpr(value="done", literal_type='string'),
                    is_bang=False
                )
            ]
        )
        result = emitter._emit_computation_expr(comp)
        assert 'async {' in result
        assert 'do!' in result
        assert 'return' in result


# =============================================================================
# Test: Units of Measure
# =============================================================================

class TestUnitsOfMeasure:
    """Tests for units of measure emission."""
    
    def test_simple_measure_declaration(self, emitter):
        """TC-FS-037: Simple measure declaration."""
        measure = MeasureDeclaration(name='m')
        result = emitter._emit_measure_declaration(measure)
        assert '[<Measure>] type m' in result
    
    def test_measure_type_simple(self, emitter):
        """TC-FS-038: Simple measure type."""
        measure_type = MeasureType(
            base_type=TypeCon(name='float'),
            measure=MeasureUnit(name='m')
        )
        result = emitter._emit_measure_type(measure_type)
        assert 'float<m>' in result
    
    def test_measure_type_division(self, emitter):
        """TC-FS-039: Measure type with division (velocity)."""
        measure_type = MeasureType(
            base_type=TypeCon(name='float'),
            measure=MeasureDiv(
                numerator=MeasureUnit(name='m'),
                denominator=MeasureUnit(name='s')
            )
        )
        result = emitter._emit_measure_type(measure_type)
        assert 'float<m / s>' in result
    
    def test_measure_type_product(self, emitter):
        """TC-FS-040: Measure type with product."""
        measure_type = MeasureType(
            base_type=TypeCon(name='float'),
            measure=MeasureProd(
                left=MeasureUnit(name='kg'),
                right=MeasureUnit(name='m')
            )
        )
        result = emitter._emit_measure_type(measure_type)
        assert 'float<kg * m>' in result
    
    def test_measure_type_power(self, emitter):
        """TC-FS-041: Measure type with power."""
        measure_type = MeasureType(
            base_type=TypeCon(name='float'),
            measure=MeasurePow(
                base=MeasureUnit(name='m'),
                power=2
            )
        )
        result = emitter._emit_measure_type(measure_type)
        assert 'float<m^2>' in result


# =============================================================================
# Test: Active Patterns
# =============================================================================

class TestActivePatterns:
    """Tests for active pattern emission."""
    
    def test_total_active_pattern(self, emitter):
        """TC-FS-042: Total active pattern."""
        ap = ActivePattern(
            cases=['Even', 'Odd'],
            is_partial=False,
            params=['n'],
            body=IfExpr(
                condition=BinaryOpExpr(
                    op='=',
                    left=BinaryOpExpr(
                        op='%',
                        left=VarExpr(name='n'),
                        right=LiteralExpr(value=2, literal_type='int')
                    ),
                    right=LiteralExpr(value=0, literal_type='int')
                ),
                then_branch=VarExpr(name='Even'),
                else_branch=VarExpr(name='Odd')
            )
        )
        result = emitter._emit_active_pattern(ap)
        assert '(|Even|Odd|)' in result
        assert 'n' in result
    
    def test_partial_active_pattern(self, emitter):
        """TC-FS-043: Partial active pattern."""
        ap = ActivePattern(
            cases=['Positive'],
            is_partial=True,
            params=['x'],
            body=VarExpr(name='checkPositive')
        )
        result = emitter._emit_active_pattern(ap)
        assert '(|Positive|_|)' in result


# =============================================================================
# Test: .NET Interop
# =============================================================================

class TestDotNetInterop:
    """Tests for .NET interoperability emission."""
    
    def test_simple_class(self, emitter):
        """TC-FS-044: Simple class definition."""
        cls = ClassDef(
            name='Counter',
            members=[
                ClassField(
                    name='count',
                    field_type=TypeCon(name='int'),
                    is_mutable=True,
                    default_value=LiteralExpr(value=0, literal_type='int')
                ),
                ClassMethod(
                    name='Increment',
                    params=[],
                    body=BinaryOpExpr(
                        op='+',
                        left=VarExpr(name='count'),
                        right=LiteralExpr(value=1, literal_type='int')
                    )
                )
            ]
        )
        result = emitter._emit_class(cls)
        assert 'type Counter()' in result
        assert 'let mutable count = 0' in result
        assert 'member' in result
        assert 'Increment' in result
    
    def test_class_with_attribute(self, emitter):
        """TC-FS-045: Class with attribute."""
        cls = ClassDef(
            name='SerializableData',
            attributes=[Attribute(name='Serializable')],
            members=[]
        )
        result = emitter._emit_class(cls)
        assert '[<Serializable>]' in result
    
    def test_interface_definition(self, emitter):
        """TC-FS-046: Interface definition."""
        iface = InterfaceDef(
            name='ICounter',
            members=[
                InterfaceMember(name='Count', member_type=TypeCon(name='int')),
                InterfaceMember(
                    name='Increment',
                    member_type=FunctionType(
                        param_type=TypeCon(name='unit'),
                        return_type=TypeCon(name='unit')
                    )
                )
            ]
        )
        result = emitter._emit_interface(iface)
        assert 'type ICounter =' in result
        assert 'abstract member Count' in result
        assert 'abstract member Increment' in result
    
    def test_class_property(self, emitter):
        """TC-FS-047: Class property."""
        prop = ClassProperty(
            name='Value',
            property_type=TypeCon(name='int'),
            getter=VarExpr(name='_value'),
            setter=VarExpr(name='_value')
        )
        result = emitter._emit_class_member(prop)
        assert 'member' in result
        assert 'Value' in result
        assert 'get()' in result
        assert 'set(value)' in result


# =============================================================================
# Test: Module Emission
# =============================================================================

class TestModuleEmission:
    """Tests for complete module emission."""
    
    def test_simple_module(self, emitter):
        """TC-FS-048: Simple module emission."""
        module = Module(
            name='Sample',
            imports=[Import(module='System')],
            functions=[
                FunctionDef(
                    name='add',
                    clauses=[FunctionClause(
                        patterns=[VarPattern(name='x'), VarPattern(name='y')],
                        body=BinaryOpExpr(
                            op='+',
                            left=VarExpr(name='x'),
                            right=VarExpr(name='y')
                        )
                    )]
                )
            ]
        )
        result = emitter.emit_module(module)
        assert 'module Sample' in result
        assert 'open System' in result
        assert 'let add' in result
    
    def test_namespaced_module(self, emitter):
        """TC-FS-049: Namespaced module."""
        module = Module(
            name='MyCompany.MyApp.Utils',
            functions=[]
        )
        result = emitter.emit_module(module)
        assert 'namespace MyCompany.MyApp' in result
        assert 'module Utils' in result
    
    def test_module_with_types(self, emitter):
        """TC-FS-050: Module with type definitions."""
        module = Module(
            name='Types',
            type_definitions=[
                DataType(
                    name='Result',
                    type_params=[TypeParameter(name='a'), TypeParameter(name='e')],
                    constructors=[
                        DataConstructor(name='Ok', fields=[TypeVar(name='a')]),
                        DataConstructor(name='Error', fields=[TypeVar(name='e')])
                    ]
                ),
                RecordType(
                    name='Config',
                    fields=[
                        RecordField(name='Host', field_type=TypeCon(name='string')),
                        RecordField(name='Port', field_type=TypeCon(name='int'))
                    ]
                )
            ]
        )
        result = emitter.emit_module(module)
        assert 'module Types' in result
        assert "type Result<'a, 'e>" in result
        assert 'type Config =' in result


# =============================================================================
# Test: Pattern Emission
# =============================================================================

class TestPatternEmission:
    """Tests for pattern emission."""
    
    def test_wildcard_pattern(self, emitter):
        """TC-FS-051: Wildcard pattern."""
        pattern = WildcardPattern()
        result = emitter._emit_pattern(pattern)
        assert result == "_"
    
    def test_variable_pattern(self, emitter):
        """TC-FS-052: Variable pattern."""
        pattern = VarPattern(name='x')
        result = emitter._emit_pattern(pattern)
        assert result == "x"
    
    def test_constructor_pattern_nullary(self, emitter):
        """TC-FS-053: Nullary constructor pattern."""
        pattern = ConstructorPattern(constructor='None')
        result = emitter._emit_pattern(pattern)
        assert result == "None"
    
    def test_constructor_pattern_with_args(self, emitter):
        """TC-FS-054: Constructor pattern with arguments."""
        pattern = ConstructorPattern(
            constructor='Some',
            args=[VarPattern(name='x')]
        )
        result = emitter._emit_pattern(pattern)
        assert 'Some(x)' in result
    
    def test_tuple_pattern(self, emitter):
        """TC-FS-055: Tuple pattern."""
        pattern = TuplePattern(elements=[
            VarPattern(name='a'),
            VarPattern(name='b')
        ])
        result = emitter._emit_pattern(pattern)
        assert '(a, b)' in result
    
    def test_list_pattern(self, emitter):
        """TC-FS-056: List pattern."""
        pattern = ListPattern(elements=[
            VarPattern(name='x'),
            VarPattern(name='y')
        ])
        result = emitter._emit_pattern(pattern)
        assert '[x; y]' in result
    
    def test_cons_pattern(self, emitter):
        """TC-FS-057: Cons pattern."""
        pattern = ListPattern(
            elements=[VarPattern(name='h')],
            rest=VarPattern(name='t')
        )
        result = emitter._emit_pattern(pattern)
        assert 'h::t' in result
    
    def test_as_pattern(self, emitter):
        """TC-FS-058: As pattern."""
        pattern = AsPattern(
            name='all',
            pattern=ConstructorPattern(
                constructor='Some',
                args=[VarPattern(name='x')]
            )
        )
        result = emitter._emit_pattern(pattern)
        assert 'as all' in result
    
    def test_or_pattern(self, emitter):
        """TC-FS-059: Or pattern."""
        pattern = OrPattern(
            left=LiteralPattern(value=0, literal_type='int'),
            right=LiteralPattern(value=1, literal_type='int')
        )
        result = emitter._emit_pattern(pattern)
        assert '|' in result


# =============================================================================
# Test: Additional F# Constructs
# =============================================================================

class TestAdditionalConstructs:
    """Tests for additional F# constructs."""
    
    def test_pipeline_forward(self, emitter):
        """TC-FS-060: Forward pipeline."""
        expr = PipelineExpr(
            left=LiteralExpr(value=5, literal_type='int'),
            right=VarExpr(name='double'),
            direction='forward'
        )
        result = emitter._emit_pipeline_expr(expr)
        assert '|>' in result
    
    def test_pipeline_backward(self, emitter):
        """TC-FS-061: Backward pipeline."""
        expr = PipelineExpr(
            left=VarExpr(name='print'),
            right=LiteralExpr(value="hello", literal_type='string'),
            direction='backward'
        )
        result = emitter._emit_pipeline_expr(expr)
        assert '<|' in result
    
    def test_composition_forward(self, emitter):
        """TC-FS-062: Forward composition."""
        expr = CompositionExpr(
            left=VarExpr(name='f'),
            right=VarExpr(name='g'),
            direction='forward'
        )
        result = emitter._emit_composition_expr(expr)
        assert '>>' in result
    
    def test_use_expression(self, emitter):
        """TC-FS-063: Use expression."""
        expr = UseExpr(
            name='file',
            value=AppExpr(
                func=VarExpr(name='openFile'),
                arg=LiteralExpr(value="test.txt", literal_type='string')
            ),
            body=AppExpr(
                func=VarExpr(name='read'),
                arg=VarExpr(name='file')
            )
        )
        result = emitter._emit_use_expr(expr)
        assert 'use file =' in result
        assert 'in' in result


# =============================================================================
# Test: fsproj Generation
# =============================================================================

class TestFsprojGeneration:
    """Tests for F# project file generation."""
    
    def test_fsproj_generation(self, emitter):
        """TC-FS-064: Generate fsproj file."""
        result = emitter.emit_fsproj('MyProject', ['Module1.fs', 'Module2.fs'])
        assert '<Project Sdk="Microsoft.NET.Sdk">' in result
        assert '<TargetFramework>net8.0</TargetFramework>' in result
        assert '<Compile Include="Module1.fs" />' in result
        assert '<Compile Include="Module2.fs" />' in result


# =============================================================================
# Test: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_empty_list(self, emitter):
        """TC-FS-065: Empty list."""
        expr = ListExpr(elements=[])
        result = emitter._emit_expr(expr)
        assert result == "[]"
    
    def test_empty_module(self, emitter):
        """TC-FS-066: Empty module."""
        module = Module(name='Empty')
        result = emitter.emit_module(module)
        assert 'module Empty' in result
    
    def test_none_expression(self, emitter):
        """TC-FS-067: None expression."""
        result = emitter._emit_expr(None)
        assert result == "()"
    
    def test_none_pattern(self, emitter):
        """TC-FS-068: None pattern."""
        result = emitter._emit_pattern(None)
        assert result == "_"
    
    def test_string_escaping(self, emitter):
        """TC-FS-069: String with special characters."""
        expr = LiteralExpr(value='hello "world"', literal_type='string')
        result = emitter._emit_expr(expr)
        assert '\\"' in result  # Escaped quotes


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
