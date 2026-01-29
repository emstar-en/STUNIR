#!/usr/bin/env python3
"""Tests for STUNIR Systems Language Emitters.

Tests cover:
- Ada emitter with SPARK support
- D emitter with templates and contracts
- Code generation quality
- Manifest generation
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
    Assignment, IfStatement, ElsifPart, WhileLoop, ForLoop, ReturnStatement,
    BlockStatement, CallStatement,
    # Types
    SubtypeDecl, RecordType, ArrayType, EnumType, EnumLiteral,
    ComponentDecl, RangeConstraint, IntegerType, ModularType,
    # Concurrency
    TaskType, Entry, ProtectedType, AcceptStatement, SelectStatement,
    AcceptAlternative, DelayAlternative, TerminateAlternative,
    DelayStatement,
    # Verification
    Contract, ContractCase, GlobalSpec, DependsSpec,
    LoopInvariant, LoopVariant, VariantExpr,
    AssertPragma, QuantifiedExpr,
    # Memory
    AccessType, Allocator,
    VariableDecl, ConstantDecl,
)
from targets.systems.ada_emitter import AdaEmitter
from targets.systems.d_emitter import DEmitter


class TestAdaEmitterBasics:
    """Test basic Ada code generation."""
    
    @pytest.fixture
    def emitter(self):
        return AdaEmitter()
    
    def test_empty_package(self, emitter):
        """Test empty package specification."""
        pkg = Package(name='Empty_Package')
        spec = emitter.emit_package_spec(pkg)
        
        assert 'package Empty_Package is' in spec
        assert 'end Empty_Package;' in spec
    
    def test_package_with_spark_mode(self, emitter):
        """Test package with SPARK mode enabled."""
        pkg = Package(name='SPARK_Package', spark_mode=True)
        spec = emitter.emit_package_spec(pkg)
        
        assert 'pragma SPARK_Mode (On);' in spec
        assert 'package SPARK_Package is' in spec
    
    def test_package_with_imports(self, emitter):
        """Test package with imports."""
        pkg = Package(
            name='IO_Package',
            imports=[
                Import(module='Ada.Text_IO', use_clause=True),
                Import(module='Interfaces', use_clause=False),
            ]
        )
        spec = emitter.emit_package_spec(pkg)
        
        assert 'with Ada.Text_IO;' in spec
        assert 'use Ada.Text_IO;' in spec
        assert 'with Interfaces;' in spec
    
    def test_type_mapping(self, emitter):
        """Test IR type to Ada type mapping."""
        assert emitter._map_type(TypeRef(name='i32')) == 'Interfaces.Integer_32'
        assert emitter._map_type(TypeRef(name='f64')) == 'Long_Float'
        assert emitter._map_type(TypeRef(name='bool')) == 'Boolean'
        assert emitter._map_type(TypeRef(name='string')) == 'String'


class TestAdaTypeDeclarations:
    """Test Ada type declaration emission."""
    
    @pytest.fixture
    def emitter(self):
        return AdaEmitter()
    
    def test_subtype_declaration(self, emitter):
        """Test subtype declaration."""
        pkg = Package(
            name='Types_Pkg',
            types=[
                SubtypeDecl(
                    name='Positive',
                    base_type=TypeRef(name='Integer'),
                    constraint=RangeConstraint(
                        lower=Literal(value=1, literal_type='int'),
                        upper=Literal(value=100, literal_type='int')
                    )
                )
            ]
        )
        spec = emitter.emit_package_spec(pkg)
        
        assert 'subtype Positive is Integer range 1 .. 100;' in spec
    
    def test_record_type(self, emitter):
        """Test record type declaration."""
        pkg = Package(
            name='Records_Pkg',
            types=[
                RecordType(
                    name='Point',
                    components=[
                        ComponentDecl(name='X', type_ref=TypeRef(name='Integer')),
                        ComponentDecl(name='Y', type_ref=TypeRef(name='Integer')),
                    ]
                )
            ]
        )
        spec = emitter.emit_package_spec(pkg)
        
        assert 'type Point is record' in spec
        assert 'X : Integer;' in spec
        assert 'Y : Integer;' in spec
        assert 'end record;' in spec
    
    def test_enum_type(self, emitter):
        """Test enumeration type."""
        pkg = Package(
            name='Enums_Pkg',
            types=[
                EnumType(
                    name='Color',
                    literals=[
                        EnumLiteral(name='Red'),
                        EnumLiteral(name='Green'),
                        EnumLiteral(name='Blue'),
                    ]
                )
            ]
        )
        spec = emitter.emit_package_spec(pkg)
        
        assert 'type Color is (Red, Green, Blue);' in spec


class TestAdaSubprograms:
    """Test Ada subprogram emission."""
    
    @pytest.fixture
    def emitter(self):
        return AdaEmitter()
    
    def test_procedure_spec(self, emitter):
        """Test procedure specification."""
        pkg = Package(
            name='Procs_Pkg',
            subprograms=[
                Subprogram(
                    name='Greet',
                    parameters=[],
                    body=[]
                )
            ]
        )
        spec = emitter.emit_package_spec(pkg)
        
        assert 'procedure Greet;' in spec
    
    def test_function_spec(self, emitter):
        """Test function specification."""
        pkg = Package(
            name='Funcs_Pkg',
            subprograms=[
                Subprogram(
                    name='Add',
                    parameters=[
                        Parameter(name='A', type_ref=TypeRef(name='Integer'), mode=Mode.IN),
                        Parameter(name='B', type_ref=TypeRef(name='Integer'), mode=Mode.IN),
                    ],
                    return_type=TypeRef(name='Integer')
                )
            ]
        )
        spec = emitter.emit_package_spec(pkg)
        
        assert 'function Add' in spec
        assert 'A : Integer' in spec
        assert 'B : Integer' in spec
        assert 'return Integer' in spec
    
    def test_function_body(self, emitter):
        """Test function body emission."""
        pkg = Package(
            name='Funcs_Pkg',
            subprograms=[
                Subprogram(
                    name='Add',
                    parameters=[
                        Parameter(name='A', type_ref=TypeRef(name='Integer')),
                        Parameter(name='B', type_ref=TypeRef(name='Integer')),
                    ],
                    return_type=TypeRef(name='Integer'),
                    body=[ReturnStatement(value=BinaryOp(op='+', left=VarExpr(name='A'), right=VarExpr(name='B')))]
                )
            ]
        )
        body = emitter.emit_package_body(pkg)
        
        assert 'function Add' in body
        assert 'return (A + B);' in body
        assert 'end Add;' in body


class TestAdaSPARKAnnotations:
    """Test Ada SPARK annotation emission."""
    
    @pytest.fixture
    def emitter(self):
        return AdaEmitter()
    
    def test_precondition(self, emitter):
        """Test precondition emission."""
        pkg = Package(
            name='Contracts_Pkg',
            subprograms=[
                Subprogram(
                    name='Divide',
                    parameters=[
                        Parameter(name='A', type_ref=TypeRef(name='Integer')),
                        Parameter(name='B', type_ref=TypeRef(name='Integer')),
                    ],
                    return_type=TypeRef(name='Integer'),
                    preconditions=[
                        Contract(condition=BinaryOp(op='/=', left=VarExpr(name='B'), right=Literal(value=0, literal_type='int')))
                    ]
                )
            ]
        )
        spec = emitter.emit_package_spec(pkg)
        
        assert 'Pre =>' in spec
        assert 'B /= 0' in spec
    
    def test_global_spec(self, emitter):
        """Test Global specification emission."""
        pkg = Package(
            name='Global_Pkg',
            subprograms=[
                Subprogram(
                    name='Process',
                    spark_mode=True,
                    global_spec=GlobalSpec(
                        inputs=['Input_Data'],
                        outputs=['Output_Data']
                    )
                )
            ]
        )
        spec = emitter.emit_package_spec(pkg)
        
        assert 'Global =>' in spec
        assert 'Input =>' in spec
        assert 'Output =>' in spec
    
    def test_depends_spec(self, emitter):
        """Test Depends specification emission."""
        pkg = Package(
            name='Depends_Pkg',
            subprograms=[
                Subprogram(
                    name='Compute',
                    return_type=TypeRef(name='Integer'),
                    depends_spec=DependsSpec(
                        dependencies={'Result': ['X', 'Y']}
                    )
                )
            ]
        )
        spec = emitter.emit_package_spec(pkg)
        
        assert 'Depends =>' in spec


class TestAdaConcurrency:
    """Test Ada concurrency emission."""
    
    @pytest.fixture
    def emitter(self):
        return AdaEmitter()
    
    def test_task_type_spec(self, emitter):
        """Test task type specification."""
        pkg = Package(
            name='Tasks_Pkg',
            tasks=[
                TaskType(
                    name='Worker_Task',
                    entries=[
                        Entry(name='Start'),
                        Entry(name='Stop'),
                    ]
                )
            ]
        )
        spec = emitter.emit_package_spec(pkg)
        
        assert 'task type Worker_Task is' in spec
        assert 'entry Start;' in spec
        assert 'entry Stop;' in spec
        assert 'end Worker_Task;' in spec
    
    def test_protected_type_spec(self, emitter):
        """Test protected type specification."""
        pkg = Package(
            name='Protected_Pkg',
            protected_types=[
                ProtectedType(
                    name='Counter',
                    procedures=[Subprogram(name='Increment')],
                    functions=[Subprogram(name='Get', return_type=TypeRef(name='Integer'))],
                    private_components=[
                        ComponentDecl(
                            name='Value',
                            type_ref=TypeRef(name='Integer'),
                            default_value=Literal(value=0, literal_type='int')
                        )
                    ]
                )
            ]
        )
        spec = emitter.emit_package_spec(pkg)
        
        assert 'protected type Counter is' in spec
        assert 'procedure Increment;' in spec
        assert 'function Get' in spec
        assert 'return Integer' in spec
        assert 'private' in spec
        assert 'Value : Integer := 0;' in spec


class TestDEmitterBasics:
    """Test basic D code generation."""
    
    @pytest.fixture
    def emitter(self):
        return DEmitter()
    
    def test_empty_module(self, emitter):
        """Test empty module generation."""
        pkg = Package(name='empty_module')
        code = emitter.emit_module(pkg)
        
        assert 'module empty_module;' in code
    
    def test_module_with_imports(self, emitter):
        """Test module with imports."""
        pkg = Package(
            name='io_module',
            imports=[
                Import(module='std.stdio'),
                Import(module='std.algorithm', selective_imports=['sort', 'filter']),
            ]
        )
        code = emitter.emit_module(pkg)
        
        assert 'import std_stdio;' in code
        assert 'import std_algorithm : sort, filter;' in code
    
    def test_type_mapping(self, emitter):
        """Test IR type to D type mapping."""
        assert emitter._map_type(TypeRef(name='i32')) == 'int'
        assert emitter._map_type(TypeRef(name='i64')) == 'long'
        assert emitter._map_type(TypeRef(name='f64')) == 'double'
        assert emitter._map_type(TypeRef(name='bool')) == 'bool'


class TestDTypeDeclarations:
    """Test D type declaration emission."""
    
    @pytest.fixture
    def emitter(self):
        return DEmitter()
    
    def test_struct_definition(self, emitter):
        """Test struct definition."""
        pkg = Package(
            name='structs_module',
            types=[
                RecordType(
                    name='Point',
                    components=[
                        ComponentDecl(name='x', type_ref=TypeRef(name='i32')),
                        ComponentDecl(name='y', type_ref=TypeRef(name='i32')),
                    ]
                )
            ]
        )
        code = emitter.emit_module(pkg)
        
        assert 'struct Point {' in code
        assert 'int x;' in code
        assert 'int y;' in code
    
    def test_enum_definition(self, emitter):
        """Test enum definition."""
        pkg = Package(
            name='enums_module',
            types=[
                EnumType(
                    name='Color',
                    literals=[
                        EnumLiteral(name='Red'),
                        EnumLiteral(name='Green'),
                        EnumLiteral(name='Blue'),
                    ]
                )
            ]
        )
        code = emitter.emit_module(pkg)
        
        assert 'enum Color {' in code
        assert 'Red' in code
        assert 'Green' in code
        assert 'Blue' in code


class TestDFunctions:
    """Test D function emission."""
    
    @pytest.fixture
    def emitter(self):
        return DEmitter()
    
    def test_basic_function(self, emitter):
        """Test basic function emission."""
        pkg = Package(
            name='funcs_module',
            subprograms=[
                Subprogram(
                    name='add',
                    parameters=[
                        Parameter(name='a', type_ref=TypeRef(name='i32')),
                        Parameter(name='b', type_ref=TypeRef(name='i32')),
                    ],
                    return_type=TypeRef(name='i32'),
                    body=[ReturnStatement(value=BinaryOp(op='+', left=VarExpr(name='a'), right=VarExpr(name='b')))]
                )
            ]
        )
        code = emitter.emit_module(pkg)
        
        assert 'int add(int a, int b)' in code
        assert 'return (a + b);' in code
    
    def test_function_with_safety(self, emitter):
        """Test function with safety attributes."""
        pkg = Package(
            name='safe_module',
            subprograms=[
                Subprogram(
                    name='safe_fn',
                    return_type=TypeRef(name='i32'),
                    safety_level=SafetyLevel.SAFE,
                    is_pure=True,
                    is_nothrow=True,
                    body=[ReturnStatement(value=Literal(value=42, literal_type='int'))]
                )
            ]
        )
        code = emitter.emit_module(pkg)
        
        assert '@safe' in code
        assert 'pure' in code
        assert 'nothrow' in code


class TestDContracts:
    """Test D contract emission."""
    
    @pytest.fixture
    def emitter(self):
        return DEmitter()
    
    def test_precondition(self, emitter):
        """Test precondition (in contract) emission."""
        pkg = Package(
            name='contracts_module',
            subprograms=[
                Subprogram(
                    name='divide',
                    parameters=[
                        Parameter(name='a', type_ref=TypeRef(name='i32')),
                        Parameter(name='b', type_ref=TypeRef(name='i32')),
                    ],
                    return_type=TypeRef(name='i32'),
                    preconditions=[
                        Contract(condition=BinaryOp(op='!=', left=VarExpr(name='b'), right=Literal(value=0, literal_type='int')))
                    ],
                    body=[ReturnStatement(value=BinaryOp(op='/', left=VarExpr(name='a'), right=VarExpr(name='b')))]
                )
            ]
        )
        code = emitter.emit_module(pkg)
        
        assert 'in (' in code
        assert '(b != 0)' in code
    
    def test_postcondition(self, emitter):
        """Test postcondition (out contract) emission."""
        pkg = Package(
            name='post_module',
            subprograms=[
                Subprogram(
                    name='abs_val',
                    parameters=[
                        Parameter(name='x', type_ref=TypeRef(name='i32')),
                    ],
                    return_type=TypeRef(name='i32'),
                    postconditions=[
                        Contract(condition=BinaryOp(op='>=', left=VarExpr(name='result'), right=Literal(value=0, literal_type='int')))
                    ],
                    body=[ReturnStatement(value=VarExpr(name='x'))]
                )
            ]
        )
        code = emitter.emit_module(pkg)
        
        assert 'out (' in code
        assert 'result' in code


class TestManifestGeneration:
    """Test manifest generation."""
    
    def test_ada_manifest(self):
        """Test Ada manifest generation."""
        emitter = AdaEmitter()
        pkg = Package(name='Test_Pkg')
        result = emitter.emit(pkg)
        
        assert 'spec_code' in dir(result)
        assert 'body_code' in dir(result)
        assert 'manifest' in dir(result)
        assert result.manifest['schema'] == 'stunir.manifest.targets.ada.v1'
        assert 'manifest_hash' in result.manifest
    
    def test_d_manifest(self):
        """Test D manifest generation."""
        emitter = DEmitter()
        pkg = Package(name='test_module')
        result = emitter.emit(pkg)
        
        assert 'code' in dir(result)
        assert 'manifest' in dir(result)
        assert result.manifest['schema'] == 'stunir.manifest.targets.d.v1'
        assert 'manifest_hash' in result.manifest


class TestCompleteExamples:
    """Test complete code generation examples."""
    
    def test_ada_spark_verified_function(self):
        """Test complete Ada SPARK verified function."""
        emitter = AdaEmitter()
        pkg = Package(
            name='Math_Utils',
            spark_mode=True,
            subprograms=[
                Subprogram(
                    name='Abs_Value',
                    parameters=[
                        Parameter(name='X', type_ref=TypeRef(name='Integer'))
                    ],
                    return_type=TypeRef(name='Natural'),
                    spark_mode=True,
                    preconditions=[
                        Contract(condition=BinaryOp(op='/=', left=VarExpr(name='X'), right=Literal(value=-2147483648, literal_type='int')))
                    ],
                    global_spec=GlobalSpec(is_null=True),
                    body=[
                        IfStatement(
                            condition=BinaryOp(op='>=', left=VarExpr(name='X'), right=Literal(value=0, literal_type='int')),
                            then_body=[ReturnStatement(value=VarExpr(name='X'))],
                            else_body=[ReturnStatement(value=UnaryOp(op='-', operand=VarExpr(name='X')))]
                        )
                    ]
                )
            ]
        )
        
        spec, body = emitter.emit_package(pkg)
        
        # Check specification
        assert 'pragma SPARK_Mode (On);' in spec
        assert 'function Abs_Value' in spec
        assert 'Pre =>' in spec
        assert 'Global => null' in spec
        
        # Check body
        assert 'if (X >= 0) then' in body
        assert 'return X;' in body
    
    def test_d_template_function(self):
        """Test D function with safety attributes."""
        emitter = DEmitter()
        pkg = Package(
            name='utils',
            subprograms=[
                Subprogram(
                    name='max',
                    parameters=[
                        Parameter(name='a', type_ref=TypeRef(name='i32')),
                        Parameter(name='b', type_ref=TypeRef(name='i32')),
                    ],
                    return_type=TypeRef(name='i32'),
                    safety_level=SafetyLevel.SAFE,
                    is_pure=True,
                    is_nothrow=True,
                    body=[
                        IfStatement(
                            condition=BinaryOp(op='>', left=VarExpr(name='a'), right=VarExpr(name='b')),
                            then_body=[ReturnStatement(value=VarExpr(name='a'))],
                            else_body=[ReturnStatement(value=VarExpr(name='b'))]
                        )
                    ]
                )
            ]
        )
        
        code = emitter.emit_module(pkg)
        
        assert 'int max(int a, int b)' in code
        assert '@safe' in code
        assert 'pure' in code
        assert 'nothrow' in code
        assert 'if ((a > b))' in code
        assert 'return a;' in code
        assert 'return b;' in code


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
