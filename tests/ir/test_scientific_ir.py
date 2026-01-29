#!/usr/bin/env python3
"""Tests for STUNIR Scientific IR.

Tests cover:
- Module and Program structures
- Array types and operations
- Numerical primitives
- Parallel constructs
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from ir.scientific import (
    # Core
    Module, Program, Import, Subprogram, Parameter,
    TypeRef, VariableDecl, ConstantDecl,
    Visibility, Intent, ParameterMode, ArrayOrder,
    # Types
    RecordType, VariantRecord, RecordVariant, FieldDecl,
    EnumType, SetType, RangeType,
    # Arrays
    ArrayType, ArrayDimension, ArraySlice, SliceSpec,
    ArrayIntrinsic, ArrayConstructor, ImpliedDo,
    PascalArrayType, IndexRange, DynamicArray,
    # Statements
    Assignment, IfStatement, ForLoop, WhileLoop,
    CaseStatement, CaseItem, CallStatement, ReturnStatement,
    # Expressions
    Literal, VarRef, BinaryOp, UnaryOp, FunctionCall,
    ArrayAccess, FieldAccess, SetExpr, SetOp, RangeExpr,
    # Numerical
    NumericType, ComplexType, ComplexLiteral,
    MathIntrinsic, INTRINSIC_MAP,
    DoConcurrent, LoopIndex, LocalitySpec, ReduceSpec,
    Coarray, SyncAll,
)


class TestModuleStructures:
    """Test module and program structures."""
    
    def test_module_creation(self):
        """Test basic module creation."""
        mod = Module(
            name='math_utils',
            exports=['add', 'multiply'],
            visibility=Visibility.PUBLIC
        )
        
        assert mod.name == 'math_utils'
        assert mod.exports == ['add', 'multiply']
        assert mod.visibility == Visibility.PUBLIC
        assert mod.kind == 'module'
    
    def test_module_to_dict(self):
        """Test module serialization."""
        mod = Module(
            name='test_module',
            imports=[Import(module_name='base_module')],
            exports=['func1']
        )
        
        d = mod.to_dict()
        assert d['kind'] == 'module'
        assert d['name'] == 'test_module'
        assert len(d['imports']) == 1
        assert d['imports'][0]['module_name'] == 'base_module'
    
    def test_program_creation(self):
        """Test program creation."""
        prog = Program(
            name='main_program',
            uses=[Import(module_name='utils')],
            body=[CallStatement(name='init')]
        )
        
        assert prog.name == 'main_program'
        assert len(prog.uses) == 1
        assert len(prog.body) == 1
        assert prog.kind == 'program'
    
    def test_submodule(self):
        """Test Fortran submodule."""
        sub = Module(
            name='math_impl',
            is_submodule=True,
            parent_module='math_utils'
        )
        
        assert sub.is_submodule
        assert sub.parent_module == 'math_utils'


class TestSubprograms:
    """Test subprogram definitions."""
    
    def test_function_creation(self):
        """Test function creation."""
        func = Subprogram(
            name='add',
            is_function=True,
            parameters=[
                Parameter(name='a', type_ref=TypeRef(name='f64'), intent=Intent.IN),
                Parameter(name='b', type_ref=TypeRef(name='f64'), intent=Intent.IN)
            ],
            return_type=TypeRef(name='f64'),
            is_pure=True
        )
        
        assert func.name == 'add'
        assert func.is_function
        assert len(func.parameters) == 2
        assert func.is_pure
        assert func.kind == 'subprogram'
    
    def test_subroutine_creation(self):
        """Test subroutine creation."""
        sub = Subprogram(
            name='swap',
            is_function=False,
            parameters=[
                Parameter(name='a', type_ref=TypeRef(name='i32'), intent=Intent.INOUT),
                Parameter(name='b', type_ref=TypeRef(name='i32'), intent=Intent.INOUT)
            ]
        )
        
        assert sub.name == 'swap'
        assert not sub.is_function
        assert sub.parameters[0].intent == Intent.INOUT
    
    def test_pascal_parameter_modes(self):
        """Test Pascal parameter modes."""
        param_value = Parameter(name='x', mode=ParameterMode.VALUE)
        param_var = Parameter(name='y', mode=ParameterMode.VAR)
        param_const = Parameter(name='z', mode=ParameterMode.CONST)
        
        assert param_value.mode == ParameterMode.VALUE
        assert param_var.mode == ParameterMode.VAR
        assert param_const.mode == ParameterMode.CONST


class TestArrayTypes:
    """Test array types and operations."""
    
    def test_array_type_creation(self):
        """Test array type with dimensions."""
        arr = ArrayType(
            element_type=TypeRef(name='f64'),
            dimensions=[
                ArrayDimension(lower=Literal(value=1), upper=Literal(value=10)),
                ArrayDimension(lower=Literal(value=1), upper=Literal(value=10))
            ],
            order=ArrayOrder.COLUMN_MAJOR
        )
        
        assert arr.kind == 'array_type'
        assert len(arr.dimensions) == 2
        assert arr.order == ArrayOrder.COLUMN_MAJOR
    
    def test_allocatable_array(self):
        """Test allocatable array."""
        arr = ArrayType(
            element_type=TypeRef(name='f64'),
            dimensions=[
                ArrayDimension(is_deferred=True),
                ArrayDimension(is_deferred=True)
            ],
            allocatable=True
        )
        
        assert arr.allocatable
        assert arr.dimensions[0].is_deferred
    
    def test_array_slice(self):
        """Test array slicing."""
        slice_op = ArraySlice(
            array=VarRef(name='matrix'),
            slices=[
                SliceSpec(start=Literal(value=1), stop=Literal(value=5)),
                SliceSpec()  # Full dimension
            ]
        )
        
        assert slice_op.kind == 'array_slice'
        assert len(slice_op.slices) == 2
        assert slice_op.slices[1].is_full_slice()
    
    def test_array_constructor(self):
        """Test array constructor."""
        constr = ArrayConstructor(
            elements=[
                Literal(value=1),
                Literal(value=2),
                Literal(value=3)
            ]
        )
        
        assert constr.kind == 'array_constructor'
        assert len(constr.elements) == 3
    
    def test_implied_do(self):
        """Test implied DO loop."""
        impl = ImpliedDo(
            expr=BinaryOp(op='*', left=VarRef(name='i'), right=Literal(value=2)),
            variable='i',
            start=Literal(value=1),
            end=Literal(value=10)
        )
        
        assert impl.kind == 'implied_do'
        assert impl.variable == 'i'
    
    def test_array_intrinsic(self):
        """Test array intrinsic function."""
        intr = ArrayIntrinsic(
            name='sum',
            array=VarRef(name='arr'),
            dim=Literal(value=1)
        )
        
        assert intr.kind == 'array_intrinsic'
        assert intr.name == 'sum'


class TestPascalTypes:
    """Test Pascal-specific types."""
    
    def test_pascal_array_type(self):
        """Test Pascal array with index ranges."""
        arr = PascalArrayType(
            element_type=TypeRef(name='i32'),
            index_ranges=[
                IndexRange(lower=Literal(value=1), upper=Literal(value=100))
            ]
        )
        
        assert arr.kind == 'pascal_array_type'
        assert len(arr.index_ranges) == 1
    
    def test_dynamic_array(self):
        """Test Pascal dynamic array."""
        dyn = DynamicArray(
            element_type=TypeRef(name='f64'),
            dimensions=2
        )
        
        assert dyn.kind == 'dynamic_array'
        assert dyn.dimensions == 2
    
    def test_set_type(self):
        """Test Pascal set type."""
        st = SetType(
            name='TCharSet',
            base_type=TypeRef(name='char')
        )
        
        assert st.kind == 'set_type'
        assert st.name == 'TCharSet'
    
    def test_variant_record(self):
        """Test Pascal variant record."""
        vr = VariantRecord(
            name='TShape',
            fixed_fields=[
                FieldDecl(name='X', type_ref=TypeRef(name='f64')),
                FieldDecl(name='Y', type_ref=TypeRef(name='f64'))
            ],
            tag_field=FieldDecl(name='Kind', type_ref=TypeRef(name='i32')),
            variants=[
                RecordVariant(
                    tag_values=[Literal(value=1)],
                    fields=[FieldDecl(name='Radius', type_ref=TypeRef(name='f64'))]
                )
            ]
        )
        
        assert vr.kind == 'variant_record'
        assert len(vr.fixed_fields) == 2
        assert len(vr.variants) == 1


class TestRecordTypes:
    """Test record/derived type definitions."""
    
    def test_record_type_creation(self):
        """Test basic record type."""
        rec = RecordType(
            name='Point',
            fields=[
                FieldDecl(name='x', type_ref=TypeRef(name='f64')),
                FieldDecl(name='y', type_ref=TypeRef(name='f64'))
            ]
        )
        
        assert rec.name == 'Point'
        assert len(rec.fields) == 2
        assert rec.kind == 'record_type'
    
    def test_fortran_derived_type(self):
        """Test Fortran derived type with EXTENDS."""
        rec = RecordType(
            name='Point3D',
            extends='Point',
            fields=[
                FieldDecl(name='z', type_ref=TypeRef(name='f64'))
            ]
        )
        
        assert rec.extends == 'Point'
    
    def test_enum_type(self):
        """Test enumeration type."""
        enum = EnumType(
            name='Color',
            values=['Red', 'Green', 'Blue']
        )
        
        assert enum.kind == 'enum_type'
        assert len(enum.values) == 3


class TestNumericalTypes:
    """Test numerical computing types."""
    
    def test_numeric_type(self):
        """Test numeric type with KIND."""
        num = NumericType(
            base='real',
            kind_param=8
        )
        
        assert num.base == 'real'
        assert num.kind_param == 8
    
    def test_complex_type(self):
        """Test complex type."""
        cplx = ComplexType(
            component_type=NumericType(base='real', kind_param=8)
        )
        
        assert cplx.kind == 'complex_type'
    
    def test_complex_literal(self):
        """Test complex number literal."""
        lit = ComplexLiteral(
            real_part=Literal(value=1.0),
            imag_part=Literal(value=2.0)
        )
        
        assert lit.kind == 'complex_literal'
    
    def test_math_intrinsic(self):
        """Test math intrinsic function."""
        intr = MathIntrinsic(
            name='sin',
            arguments=[VarRef(name='x')]
        )
        
        assert intr.fortran_name() == 'SIN'
        assert intr.pascal_name() == 'Sin'
    
    def test_intrinsic_map(self):
        """Test intrinsic mapping."""
        assert 'sin' in INTRINSIC_MAP
        assert INTRINSIC_MAP['sin']['fortran'] == 'SIN'
        assert INTRINSIC_MAP['sin']['pascal'] == 'Sin'


class TestParallelConstructs:
    """Test Fortran parallel constructs."""
    
    def test_do_concurrent(self):
        """Test DO CONCURRENT loop."""
        dc = DoConcurrent(
            indices=[
                LoopIndex(variable='i', start=Literal(value=1), end=VarRef(name='n'))
            ],
            locality=LocalitySpec(
                local_vars=['temp'],
                shared=['result']
            ),
            body=[
                Assignment(
                    target=ArrayAccess(array=VarRef(name='result'), indices=[VarRef(name='i')]),
                    value=BinaryOp(op='*', left=VarRef(name='i'), right=Literal(value=2))
                )
            ]
        )
        
        assert dc.kind == 'do_concurrent'
        assert len(dc.indices) == 1
        assert dc.locality.local_vars == ['temp']
    
    def test_reduce_spec(self):
        """Test reduction specification."""
        rs = ReduceSpec(op='+', variable='sum')
        
        assert rs.op == '+'
        assert rs.variable == 'sum'
    
    def test_coarray(self):
        """Test coarray declaration."""
        ca = Coarray(
            name='shared_data',
            element_type=TypeRef(name='f64'),
            dimensions=[ArrayDimension(lower=Literal(value=1), upper=Literal(value=100))],
            codimensions=[ArrayDimension(is_assumed=True)]
        )
        
        assert ca.kind == 'coarray'
        assert ca.name == 'shared_data'
    
    def test_sync_all(self):
        """Test SYNC ALL statement."""
        sync = SyncAll(stat_var='ierr')
        
        assert sync.kind == 'sync_all'
        assert sync.stat_var == 'ierr'


class TestStatements:
    """Test statement types."""
    
    def test_assignment(self):
        """Test assignment statement."""
        stmt = Assignment(
            target=VarRef(name='x'),
            value=Literal(value=42)
        )
        
        assert stmt.kind == 'assignment'
    
    def test_if_statement(self):
        """Test if statement."""
        stmt = IfStatement(
            condition=BinaryOp(op='>', left=VarRef(name='x'), right=Literal(value=0)),
            then_body=[CallStatement(name='positive')],
            else_body=[CallStatement(name='negative')]
        )
        
        assert stmt.kind == 'if_statement'
        assert len(stmt.then_body) == 1
    
    def test_for_loop(self):
        """Test for loop."""
        loop = ForLoop(
            variable='i',
            start=Literal(value=1),
            end=Literal(value=10),
            body=[CallStatement(name='process', arguments=[VarRef(name='i')])]
        )
        
        assert loop.kind == 'for_loop'
        assert loop.variable == 'i'
    
    def test_case_statement(self):
        """Test case statement."""
        case = CaseStatement(
            selector=VarRef(name='option'),
            cases=[
                CaseItem(values=[Literal(value=1)], body=[CallStatement(name='opt1')]),
                CaseItem(values=[Literal(value=2)], body=[CallStatement(name='opt2')])
            ],
            default_body=[CallStatement(name='default')]
        )
        
        assert case.kind == 'case_statement'
        assert len(case.cases) == 2


class TestExpressions:
    """Test expression types."""
    
    def test_literal(self):
        """Test literal values."""
        int_lit = Literal(value=42, type_hint='integer')
        float_lit = Literal(value=3.14, type_hint='real')
        bool_lit = Literal(value=True, type_hint='logical')
        str_lit = Literal(value='hello', type_hint='string')
        
        assert int_lit.value == 42
        assert float_lit.value == 3.14
        assert bool_lit.value is True
        assert str_lit.value == 'hello'
    
    def test_binary_op(self):
        """Test binary operations."""
        add = BinaryOp(op='+', left=VarRef(name='a'), right=VarRef(name='b'))
        mul = BinaryOp(op='*', left=add, right=Literal(value=2))
        
        assert add.kind == 'binary_op'
        assert mul.left.op == '+'
    
    def test_function_call(self):
        """Test function call."""
        call = FunctionCall(
            name='sqrt',
            arguments=[VarRef(name='x')]
        )
        
        assert call.kind == 'function_call'
        assert call.name == 'sqrt'
    
    def test_array_access(self):
        """Test array element access."""
        access = ArrayAccess(
            array=VarRef(name='matrix'),
            indices=[VarRef(name='i'), VarRef(name='j')]
        )
        
        assert access.kind == 'array_access'
        assert len(access.indices) == 2
    
    def test_field_access(self):
        """Test record field access."""
        access = FieldAccess(
            record=VarRef(name='point'),
            field_name='x'
        )
        
        assert access.kind == 'field_access'
        assert access.field_name == 'x'
    
    def test_set_expression(self):
        """Test Pascal set expression."""
        set_expr = SetExpr(
            elements=[Literal(value=1), Literal(value=2), Literal(value=3)]
        )
        
        assert set_expr.kind == 'set_expr'
        assert len(set_expr.elements) == 3
    
    def test_set_operation(self):
        """Test Pascal set operation."""
        set_op = SetOp(
            op='union',
            left=VarRef(name='setA'),
            right=VarRef(name='setB')
        )
        
        assert set_op.kind == 'set_op'
        assert set_op.op == 'union'


class TestSerialization:
    """Test to_dict serialization."""
    
    def test_module_serialization(self):
        """Test complete module serialization."""
        mod = Module(
            name='test',
            subprograms=[
                Subprogram(
                    name='add',
                    is_function=True,
                    parameters=[
                        Parameter(name='a', type_ref=TypeRef(name='i32'))
                    ],
                    return_type=TypeRef(name='i32'),
                    body=[
                        ReturnStatement(value=VarRef(name='a'))
                    ]
                )
            ]
        )
        
        d = mod.to_dict()
        
        assert d['kind'] == 'module'
        assert d['name'] == 'test'
        assert len(d['subprograms']) == 1
        assert d['subprograms'][0]['name'] == 'add'
        assert d['subprograms'][0]['is_function'] is True
    
    def test_nested_expression_serialization(self):
        """Test nested expression serialization."""
        expr = BinaryOp(
            op='+',
            left=BinaryOp(op='*', left=VarRef(name='a'), right=VarRef(name='b')),
            right=Literal(value=10)
        )
        
        d = expr.to_dict()
        
        assert d['kind'] == 'binary_op'
        assert d['op'] == '+'
        assert d['left']['kind'] == 'binary_op'
        assert d['left']['op'] == '*'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
