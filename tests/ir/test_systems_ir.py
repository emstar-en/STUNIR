#!/usr/bin/env python3
"""Tests for STUNIR Systems IR.

Tests cover:
- Core IR node creation
- Type system constructs
- Memory management
- Concurrency primitives
- SPARK verification annotations
- IR serialization (to_dict)
"""

import sys
import pytest
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ir.systems import (
    # Core
    Package, Import, Subprogram, Parameter, Mode, Visibility, SafetyLevel,
    TypeRef, Statement, Declaration,
    # Expressions
    Literal, VarExpr, BinaryOp, UnaryOp, CallExpr, RangeExpr,
    # Statements
    Assignment, IfStatement, WhileLoop, ForLoop, ReturnStatement,
    # Types
    TypeDecl, SubtypeDecl, RecordType, ArrayType, EnumType, EnumLiteral,
    ComponentDecl, RangeConstraint, IntegerType, ModularType,
    # Concurrency
    TaskType, Entry, ProtectedType, AcceptStatement, SelectStatement,
    AcceptAlternative, DelayAlternative,
    # Verification
    Contract, ContractCase, GlobalSpec, DependsSpec,
    LoopInvariant, LoopVariant, VariantExpr, GhostVariable,
    AssertPragma, QuantifiedExpr,
    # Memory
    AccessType, Allocator, Deallocate, AddressOf,
    VariableDecl, ConstantDecl,
)


class TestSystemsIRBasics:
    """Test basic IR node creation."""
    
    def test_package_creation(self):
        """Test creating a package."""
        pkg = Package(name='Test_Package', spark_mode=True)
        assert pkg.name == 'Test_Package'
        assert pkg.spark_mode == True
        assert pkg.kind == 'package'
    
    def test_package_with_imports(self):
        """Test package with imports."""
        pkg = Package(
            name='Math_Utils',
            imports=[
                Import(module='Ada.Text_IO', use_clause=True),
                Import(module='Interfaces', use_clause=False),
            ]
        )
        assert len(pkg.imports) == 2
        assert pkg.imports[0].use_clause == True
    
    def test_type_ref(self):
        """Test type reference."""
        int_type = TypeRef(name='Integer')
        assert int_type.name == 'Integer'
        
        access_type = TypeRef(name='Integer', is_access=True, not_null=True)
        assert access_type.is_access == True
        assert access_type.not_null == True
    
    def test_subprogram_creation(self):
        """Test subprogram creation."""
        sub = Subprogram(
            name='Add',
            parameters=[
                Parameter(name='A', type_ref=TypeRef(name='Integer'), mode=Mode.IN),
                Parameter(name='B', type_ref=TypeRef(name='Integer'), mode=Mode.IN),
            ],
            return_type=TypeRef(name='Integer'),
            body=[ReturnStatement(value=BinaryOp('+', VarExpr(name='A'), VarExpr(name='B')))]
        )
        assert sub.name == 'Add'
        assert len(sub.parameters) == 2
        assert sub.return_type.name == 'Integer'
        assert len(sub.body) == 1


class TestTypeSystem:
    """Test type system IR nodes."""
    
    def test_subtype_with_range(self):
        """Test subtype with range constraint."""
        positive = SubtypeDecl(
            name='Positive',
            base_type=TypeRef(name='Integer'),
            constraint=RangeConstraint(
                lower=Literal(value=1, literal_type='int'),
                upper=None  # No upper bound (Integer'Last)
            )
        )
        assert positive.name == 'Positive'
        assert positive.base_type.name == 'Integer'
        assert positive.constraint.lower.value == 1
    
    def test_record_type(self):
        """Test record type with components."""
        point = RecordType(
            name='Point',
            components=[
                ComponentDecl(name='X', type_ref=TypeRef(name='Integer')),
                ComponentDecl(name='Y', type_ref=TypeRef(name='Integer')),
            ],
            is_tagged=False
        )
        assert point.name == 'Point'
        assert len(point.components) == 2
        assert point.components[0].name == 'X'
    
    def test_array_type(self):
        """Test array type."""
        vector = ArrayType(
            name='Vector',
            index_types=[TypeRef(name='Index')],
            element_type=TypeRef(name='Float'),
            is_unconstrained=False
        )
        assert vector.name == 'Vector'
        assert vector.element_type.name == 'Float'
    
    def test_enum_type(self):
        """Test enumeration type."""
        color = EnumType(
            name='Color',
            literals=[
                EnumLiteral(name='Red'),
                EnumLiteral(name='Green'),
                EnumLiteral(name='Blue'),
            ]
        )
        assert color.name == 'Color'
        assert len(color.literals) == 3
        assert color.literals[0].name == 'Red'
    
    def test_modular_type(self):
        """Test modular (unsigned) type."""
        byte = ModularType(
            name='Byte',
            modulus=Literal(value=256, literal_type='int')
        )
        assert byte.name == 'Byte'
        assert byte.modulus.value == 256


class TestMemoryManagement:
    """Test memory management IR nodes."""
    
    def test_access_type(self):
        """Test access type."""
        int_ptr = AccessType(
            target_type=TypeRef(name='Integer'),
            not_null=True
        )
        assert int_ptr.not_null == True
        assert int_ptr.target_type.name == 'Integer'
    
    def test_allocator(self):
        """Test allocator (new)."""
        alloc = Allocator(
            type_ref=TypeRef(name='Integer'),
            initializer=Literal(value=42, literal_type='int')
        )
        assert alloc.type_ref.name == 'Integer'
        assert alloc.initializer.value == 42
    
    def test_deallocate(self):
        """Test deallocate."""
        dealloc = Deallocate(
            target=VarExpr(name='Ptr'),
            deallocation_proc='Free'
        )
        assert dealloc.target.name == 'Ptr'
        assert dealloc.deallocation_proc == 'Free'
    
    def test_address_of(self):
        """Test address-of operation."""
        addr = AddressOf(target=VarExpr(name='X'))
        assert addr.target.name == 'X'
        assert addr.kind == 'address_of'


class TestConcurrency:
    """Test concurrency IR nodes."""
    
    def test_task_type(self):
        """Test task type creation."""
        task = TaskType(
            name='Worker',
            entries=[
                Entry(name='Start'),
                Entry(name='Stop'),
            ]
        )
        assert task.name == 'Worker'
        assert len(task.entries) == 2
        assert task.entries[0].name == 'Start'
    
    def test_entry_with_parameters(self):
        """Test entry with parameters."""
        entry = Entry(
            name='Put',
            parameters=[
                Parameter(name='Item', type_ref=TypeRef(name='Integer'), mode=Mode.IN)
            ]
        )
        assert entry.name == 'Put'
        assert len(entry.parameters) == 1
    
    def test_protected_type(self):
        """Test protected type."""
        counter = ProtectedType(
            name='Counter',
            procedures=[Subprogram(name='Increment')],
            functions=[Subprogram(name='Get', return_type=TypeRef(name='Integer'))],
            private_components=[
                ComponentDecl(name='Value', type_ref=TypeRef(name='Integer'))
            ]
        )
        assert counter.name == 'Counter'
        assert len(counter.procedures) == 1
        assert len(counter.functions) == 1
        assert len(counter.private_components) == 1
    
    def test_accept_statement(self):
        """Test accept statement."""
        accept = AcceptStatement(
            entry_name='Start',
            body=[Assignment(target=VarExpr(name='Running'), value=Literal(value=True, literal_type='bool'))]
        )
        assert accept.entry_name == 'Start'
        assert len(accept.body) == 1


class TestVerification:
    """Test SPARK verification IR nodes."""
    
    def test_contract_precondition(self):
        """Test precondition contract."""
        pre = Contract(
            condition=BinaryOp('>', VarExpr(name='X'), Literal(value=0, literal_type='int')),
            message="X must be positive"
        )
        assert pre.condition is not None
        assert pre.message == "X must be positive"
    
    def test_contract_case(self):
        """Test contract case."""
        case = ContractCase(
            guard=BinaryOp('>=', VarExpr(name='X'), Literal(value=0, literal_type='int')),
            consequence=BinaryOp('=', VarExpr(name='Result'), VarExpr(name='X'))
        )
        assert case.guard is not None
        assert case.consequence is not None
    
    def test_global_spec(self):
        """Test Global specification."""
        global_spec = GlobalSpec(
            inputs=['X', 'Y'],
            outputs=['Z'],
            in_outs=['Counter']
        )
        assert 'X' in global_spec.inputs
        assert 'Z' in global_spec.outputs
        assert 'Counter' in global_spec.in_outs
    
    def test_depends_spec(self):
        """Test Depends specification."""
        depends = DependsSpec(
            dependencies={
                'Z': ['X', 'Y'],
                'Counter': ['Counter'],
            }
        )
        assert depends.dependencies['Z'] == ['X', 'Y']
        assert 'Counter' in depends.dependencies['Counter']
    
    def test_loop_invariant(self):
        """Test loop invariant."""
        inv = LoopInvariant(
            condition=BinaryOp('<=', VarExpr(name='I'), VarExpr(name='N'))
        )
        assert inv.condition is not None
        assert inv.kind == 'loop_invariant'
    
    def test_loop_variant(self):
        """Test loop variant (termination)."""
        variant = LoopVariant(
            expressions=[
                VariantExpr(
                    expr=BinaryOp('-', VarExpr(name='N'), VarExpr(name='I')),
                    direction='decreases'
                )
            ]
        )
        assert len(variant.expressions) == 1
        assert variant.expressions[0].direction == 'decreases'
    
    def test_ghost_variable(self):
        """Test ghost variable."""
        ghost = GhostVariable(
            name='Ghost_Sum',
            type_ref=TypeRef(name='Integer'),
            initializer=Literal(value=0, literal_type='int')
        )
        assert ghost.name == 'Ghost_Sum'
        assert ghost.kind == 'ghost_variable'
    
    def test_quantified_expression(self):
        """Test quantified expression."""
        quant = QuantifiedExpr(
            quantifier='all',
            variable='I',
            range_expr=RangeExpr(
                start=Literal(value=1, literal_type='int'),
                end=VarExpr(name='N')
            ),
            condition=BinaryOp('>=', CallExpr(name='A', arguments=[VarExpr(name='I')]), Literal(value=0, literal_type='int'))
        )
        assert quant.quantifier == 'all'
        assert quant.variable == 'I'


class TestSerialization:
    """Test IR serialization."""
    
    def test_to_dict_basic(self):
        """Test basic to_dict conversion."""
        lit = Literal(value=42, literal_type='int')
        d = lit.to_dict()
        assert d['kind'] == 'literal'
        assert d['value'] == 42
        assert d['literal_type'] == 'int'
    
    def test_to_dict_nested(self):
        """Test nested to_dict conversion."""
        binop = BinaryOp(
            op='+',
            left=Literal(value=1, literal_type='int'),
            right=Literal(value=2, literal_type='int')
        )
        d = binop.to_dict()
        assert d['kind'] == 'binary_op'
        assert d['op'] == '+'
        assert d['left']['kind'] == 'literal'
        assert d['right']['value'] == 2
    
    def test_to_dict_package(self):
        """Test package to_dict conversion."""
        pkg = Package(
            name='Test',
            spark_mode=True,
            subprograms=[
                Subprogram(name='Foo', return_type=TypeRef(name='Integer'))
            ]
        )
        d = pkg.to_dict()
        assert d['kind'] == 'package'
        assert d['name'] == 'Test'
        assert d['spark_mode'] == True
        assert len(d['subprograms']) == 1
        assert d['subprograms'][0]['name'] == 'Foo'


class TestSubprogramContracts:
    """Test subprogram with full SPARK contracts."""
    
    def test_subprogram_with_contracts(self):
        """Test subprogram with complete SPARK contracts."""
        abs_value = Subprogram(
            name='Abs_Value',
            parameters=[
                Parameter(name='X', type_ref=TypeRef(name='Integer'), mode=Mode.IN)
            ],
            return_type=TypeRef(name='Natural'),
            spark_mode=True,
            preconditions=[
                Contract(condition=BinaryOp('/=', VarExpr(name='X'), Literal(value=-2147483648, literal_type='int')))
            ],
            postconditions=[
                Contract(condition=BinaryOp('>=', VarExpr(name='Result'), Literal(value=0, literal_type='int')))
            ],
            contract_cases=[
                ContractCase(
                    guard=BinaryOp('>=', VarExpr(name='X'), Literal(value=0, literal_type='int')),
                    consequence=BinaryOp('=', VarExpr(name='Result'), VarExpr(name='X'))
                ),
                ContractCase(
                    guard=BinaryOp('<', VarExpr(name='X'), Literal(value=0, literal_type='int')),
                    consequence=BinaryOp('=', VarExpr(name='Result'), UnaryOp('-', VarExpr(name='X')))
                )
            ],
            global_spec=GlobalSpec(is_null=True),
            depends_spec=DependsSpec(dependencies={'Result': ['X']})
        )
        
        assert abs_value.name == 'Abs_Value'
        assert abs_value.spark_mode == True
        assert len(abs_value.preconditions) == 1
        assert len(abs_value.postconditions) == 1
        assert len(abs_value.contract_cases) == 2
        assert abs_value.global_spec.is_null == True
        assert 'X' in abs_value.depends_spec.dependencies['Result']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
