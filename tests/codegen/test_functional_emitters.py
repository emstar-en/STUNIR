#!/usr/bin/env python3
"""Tests for STUNIR Functional Language Emitters.

This module tests the Haskell and OCaml emitters including:
- Data type emission
- Function emission
- Pattern matching
- Type classes (Haskell)
- Modules and functors (OCaml)
"""

import sys
import pytest
from pathlib import Path

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from targets.functional import HaskellEmitter, OCamlEmitter
from ir.functional import (
    # Types
    TypeVar, TypeCon, FunctionType, TupleType, ListType,
    # Expressions
    LiteralExpr, VarExpr, AppExpr, LambdaExpr, LetExpr, IfExpr,
    CaseBranch, CaseExpr, ListGenerator, ListExpr, TupleExpr,
    BinaryOpExpr, UnaryOpExpr,
    DoExpr, DoBindStatement, DoLetStatement, DoExprStatement,
    # Patterns
    WildcardPattern, VarPattern, LiteralPattern, ConstructorPattern,
    TuplePattern, ListPattern, AsPattern, OrPattern,
    # ADTs
    TypeParameter, DataConstructor, DataType, TypeAlias, NewType,
    RecordField, RecordType,
    MethodSignature, TypeClass, TypeClassInstance,
    Import, FunctionClause, FunctionDef, Module,
)


# =============================================================================
# Haskell Emitter Tests
# =============================================================================

class TestHaskellEmitter:
    """Tests for Haskell code emitter."""
    
    @pytest.fixture
    def emitter(self):
        return HaskellEmitter()
    
    def test_emit_literal_int(self, emitter):
        """Test integer literal emission."""
        result = emitter._emit_literal_value(42, 'int')
        assert result == '42'
    
    def test_emit_literal_bool(self, emitter):
        """Test boolean literal emission."""
        assert emitter._emit_literal_value(True, 'bool') == 'True'
        assert emitter._emit_literal_value(False, 'bool') == 'False'
    
    def test_emit_literal_string(self, emitter):
        """Test string literal emission."""
        result = emitter._emit_literal_value('hello', 'string')
        assert result == '"hello"'
    
    def test_emit_literal_char(self, emitter):
        """Test char literal emission."""
        result = emitter._emit_literal_value('a', 'char')
        assert result == "'a'"
    
    def test_emit_type_simple(self, emitter):
        """Test simple type emission."""
        int_t = TypeCon(name='Int')
        assert emitter._emit_type(int_t) == 'Int'
        
        bool_t = TypeCon(name='Bool')
        assert emitter._emit_type(bool_t) == 'Bool'
    
    def test_emit_type_var(self, emitter):
        """Test type variable emission."""
        t = TypeVar(name='a')
        assert emitter._emit_type(t) == 'a'
    
    def test_emit_function_type(self, emitter):
        """Test function type emission."""
        ft = FunctionType(
            param_type=TypeCon(name='Int'),
            return_type=TypeCon(name='Bool')
        )
        result = emitter._emit_type(ft)
        assert 'Int' in result
        assert 'Bool' in result
        assert '->' in result
    
    def test_emit_pattern_wildcard(self, emitter):
        """Test wildcard pattern emission."""
        p = WildcardPattern()
        assert emitter._emit_pattern(p) == '_'
    
    def test_emit_pattern_var(self, emitter):
        """Test variable pattern emission."""
        p = VarPattern(name='x')
        assert emitter._emit_pattern(p) == 'x'
    
    def test_emit_pattern_constructor(self, emitter):
        """Test constructor pattern emission."""
        p = ConstructorPattern(constructor='Just', args=[VarPattern(name='x')])
        result = emitter._emit_pattern(p)
        assert 'Just' in result
        assert 'x' in result
    
    def test_emit_pattern_list_cons(self, emitter):
        """Test list cons pattern emission."""
        p = ListPattern(elements=[VarPattern(name='h')], rest=VarPattern(name='t'))
        result = emitter._emit_pattern(p)
        assert 'h' in result
        assert 't' in result
        assert ':' in result
    
    def test_emit_expr_var(self, emitter):
        """Test variable expression emission."""
        e = VarExpr(name='x')
        assert emitter._emit_expr(e) == 'x'
    
    def test_emit_expr_app(self, emitter):
        """Test function application emission."""
        e = AppExpr(func=VarExpr(name='f'), arg=VarExpr(name='x'))
        result = emitter._emit_expr(e)
        assert 'f' in result
        assert 'x' in result
    
    def test_emit_expr_lambda(self, emitter):
        """Test lambda expression emission."""
        e = LambdaExpr(param='x', body=VarExpr(name='x'))
        result = emitter._emit_expr(e)
        assert '\\x' in result
        assert '->' in result
    
    def test_emit_expr_let(self, emitter):
        """Test let expression emission."""
        e = LetExpr(
            name='x',
            value=LiteralExpr(value=5, literal_type='int'),
            body=VarExpr(name='x')
        )
        result = emitter._emit_expr(e)
        assert 'let' in result
        assert 'x' in result
        assert 'in' in result
    
    def test_emit_expr_if(self, emitter):
        """Test if expression emission."""
        e = IfExpr(
            condition=LiteralExpr(value=True, literal_type='bool'),
            then_branch=LiteralExpr(value=1, literal_type='int'),
            else_branch=LiteralExpr(value=2, literal_type='int')
        )
        result = emitter._emit_expr(e)
        assert 'if' in result
        assert 'then' in result
        assert 'else' in result
    
    def test_emit_expr_case(self, emitter):
        """Test case expression emission."""
        e = CaseExpr(
            scrutinee=VarExpr(name='x'),
            branches=[
                CaseBranch(pattern=LiteralPattern(value=0, literal_type='int'), body=LiteralExpr(value=1, literal_type='int')),
                CaseBranch(pattern=VarPattern(name='n'), body=VarExpr(name='n'))
            ]
        )
        result = emitter._emit_expr(e)
        assert 'case' in result
        assert 'of' in result
        assert '->' in result
    
    def test_emit_expr_list(self, emitter):
        """Test list expression emission."""
        e = ListExpr(elements=[
            LiteralExpr(value=1, literal_type='int'),
            LiteralExpr(value=2, literal_type='int'),
            LiteralExpr(value=3, literal_type='int')
        ])
        result = emitter._emit_expr(e)
        assert '[' in result
        assert '1' in result
        assert '2' in result
        assert '3' in result
    
    def test_emit_expr_tuple(self, emitter):
        """Test tuple expression emission."""
        e = TupleExpr(elements=[
            LiteralExpr(value=1, literal_type='int'),
            LiteralExpr(value=True, literal_type='bool')
        ])
        result = emitter._emit_expr(e)
        assert '(' in result
        assert '1' in result
        assert 'True' in result
    
    def test_emit_expr_binary_op(self, emitter):
        """Test binary operation emission."""
        e = BinaryOpExpr(
            op='+',
            left=LiteralExpr(value=1, literal_type='int'),
            right=LiteralExpr(value=2, literal_type='int')
        )
        result = emitter._emit_expr(e)
        assert '+' in result
    
    def test_emit_data_type_maybe(self, emitter):
        """Test Maybe data type emission."""
        maybe = DataType(
            name='Maybe',
            type_params=[TypeParameter(name='a')],
            constructors=[
                DataConstructor(name='Nothing'),
                DataConstructor(name='Just', fields=[TypeVar(name='a')])
            ],
            deriving=['Eq', 'Show']
        )
        result = emitter._emit_data_type(maybe)
        assert 'data Maybe a' in result
        assert 'Nothing' in result
        assert 'Just a' in result
        assert 'deriving (Eq, Show)' in result
    
    def test_emit_data_type_tree(self, emitter):
        """Test Tree data type emission."""
        tree = DataType(
            name='Tree',
            type_params=[TypeParameter(name='a')],
            constructors=[
                DataConstructor(name='Leaf', fields=[TypeVar(name='a')]),
                DataConstructor(
                    name='Node',
                    fields=[
                        TypeCon(name='Tree', args=[TypeVar(name='a')]),
                        TypeVar(name='a'),
                        TypeCon(name='Tree', args=[TypeVar(name='a')])
                    ]
                )
            ],
            deriving=['Show']
        )
        result = emitter._emit_data_type(tree)
        assert 'data Tree a' in result
        assert 'Leaf a' in result
        assert 'Node' in result
    
    def test_emit_type_class(self, emitter):
        """Test type class emission."""
        functor = TypeClass(
            name='Functor',
            type_params=[TypeParameter(name='f')],
            methods=[
                MethodSignature(
                    name='fmap',
                    type_signature=FunctionType(
                        param_type=FunctionType(
                            param_type=TypeVar(name='a'),
                            return_type=TypeVar(name='b')
                        ),
                        return_type=FunctionType(
                            param_type=TypeCon(name='f', args=[TypeVar(name='a')]),
                            return_type=TypeCon(name='f', args=[TypeVar(name='b')])
                        )
                    )
                )
            ]
        )
        result = emitter._emit_type_class(functor)
        assert 'class Functor f where' in result
        assert 'fmap ::' in result
    
    def test_emit_function_identity(self, emitter):
        """Test identity function emission."""
        identity = FunctionDef(
            name='identity',
            type_signature=FunctionType(
                param_type=TypeVar(name='a'),
                return_type=TypeVar(name='a')
            ),
            clauses=[FunctionClause(
                patterns=[VarPattern(name='x')],
                body=VarExpr(name='x')
            )]
        )
        result = emitter._emit_function(identity)
        assert 'identity ::' in result
        assert 'identity x = x' in result
    
    def test_emit_function_length(self, emitter):
        """Test list length function emission."""
        length = FunctionDef(
            name='length',
            clauses=[
                FunctionClause(
                    patterns=[ListPattern(elements=[])],
                    body=LiteralExpr(value=0, literal_type='int')
                ),
                FunctionClause(
                    patterns=[ListPattern(elements=[WildcardPattern()], rest=VarPattern(name='xs'))],
                    body=BinaryOpExpr(
                        op='+',
                        left=LiteralExpr(value=1, literal_type='int'),
                        right=AppExpr(func=VarExpr(name='length'), arg=VarExpr(name='xs'))
                    )
                )
            ]
        )
        result = emitter._emit_function(length)
        assert 'length []' in result
        assert '= 0' in result
        assert '_:xs' in result or '_ : xs' in result
    
    def test_emit_do_notation(self, emitter):
        """Test do notation emission."""
        do_expr = DoExpr(statements=[
            DoBindStatement(
                pattern=VarPattern(name='x'),
                action=AppExpr(func=VarExpr(name='getLine'), arg=None)
            ),
            DoLetStatement(name='y', value=AppExpr(
                func=VarExpr(name='read'),
                arg=VarExpr(name='x')
            )),
            DoExprStatement(expr=AppExpr(
                func=VarExpr(name='return'),
                arg=VarExpr(name='y')
            ))
        ])
        result = emitter._emit_do(do_expr)
        assert 'do' in result
        assert 'x <- ' in result
        assert 'let y =' in result
    
    def test_emit_module(self, emitter):
        """Test complete module emission."""
        module = Module(
            name='Example',
            exports=['identity', 'Maybe'],
            imports=[
                Import(module='Prelude'),
                Import(module='Data.List', qualified=True, alias='L')
            ],
            type_definitions=[
                DataType(
                    name='Maybe',
                    type_params=[TypeParameter(name='a')],
                    constructors=[
                        DataConstructor(name='Nothing'),
                        DataConstructor(name='Just', fields=[TypeVar(name='a')])
                    ]
                )
            ],
            functions=[
                FunctionDef(
                    name='identity',
                    clauses=[FunctionClause(
                        patterns=[VarPattern(name='x')],
                        body=VarExpr(name='x')
                    )]
                )
            ]
        )
        result = emitter.emit_module(module)
        assert 'module Example' in result
        assert 'identity, Maybe' in result
        assert 'import Prelude' in result
        assert 'import qualified Data.List as L' in result
        assert 'data Maybe' in result
        assert 'identity x = x' in result


# =============================================================================
# OCaml Emitter Tests
# =============================================================================

class TestOCamlEmitter:
    """Tests for OCaml code emitter."""
    
    @pytest.fixture
    def emitter(self):
        return OCamlEmitter()
    
    def test_emit_literal_int(self, emitter):
        """Test integer literal emission."""
        result = emitter._emit_literal_value(42, 'int')
        assert result == '42'
    
    def test_emit_literal_bool(self, emitter):
        """Test boolean literal emission."""
        assert emitter._emit_literal_value(True, 'bool') == 'true'
        assert emitter._emit_literal_value(False, 'bool') == 'false'
    
    def test_emit_literal_float(self, emitter):
        """Test float literal emission."""
        result = emitter._emit_literal_value(3.14, 'float')
        assert '3.14' in result
    
    def test_emit_type_simple(self, emitter):
        """Test simple type emission."""
        int_t = TypeCon(name='Int')
        result = emitter._emit_type(int_t)
        assert result == 'int'
        
        bool_t = TypeCon(name='Bool')
        result = emitter._emit_type(bool_t)
        assert result == 'bool'
    
    def test_emit_type_var(self, emitter):
        """Test type variable emission."""
        t = TypeVar(name='a')
        result = emitter._emit_type(t)
        assert result == "'a"
    
    def test_emit_list_type(self, emitter):
        """Test list type emission."""
        lst_t = ListType(element_type=TypeCon(name='Int'))
        result = emitter._emit_type(lst_t)
        assert 'list' in result
        assert 'int' in result
    
    def test_emit_pattern_wildcard(self, emitter):
        """Test wildcard pattern emission."""
        p = WildcardPattern()
        assert emitter._emit_pattern(p) == '_'
    
    def test_emit_pattern_var(self, emitter):
        """Test variable pattern emission."""
        p = VarPattern(name='x')
        assert emitter._emit_pattern(p) == 'x'
    
    def test_emit_pattern_or(self, emitter):
        """Test or pattern emission (OCaml-specific)."""
        p = OrPattern(
            left=LiteralPattern(value=0, literal_type='int'),
            right=LiteralPattern(value=1, literal_type='int')
        )
        result = emitter._emit_pattern(p)
        assert '|' in result
        assert '0' in result
        assert '1' in result
    
    def test_emit_expr_lambda(self, emitter):
        """Test lambda expression emission."""
        e = LambdaExpr(param='x', body=VarExpr(name='x'))
        result = emitter._emit_expr(e)
        assert 'fun x' in result
        assert '->' in result
    
    def test_emit_expr_let(self, emitter):
        """Test let expression emission."""
        e = LetExpr(
            name='x',
            value=LiteralExpr(value=5, literal_type='int'),
            body=VarExpr(name='x')
        )
        result = emitter._emit_expr(e)
        assert 'let x =' in result
        assert 'in' in result
    
    def test_emit_expr_match(self, emitter):
        """Test match expression emission."""
        e = CaseExpr(
            scrutinee=VarExpr(name='x'),
            branches=[
                CaseBranch(pattern=LiteralPattern(value=0, literal_type='int'), body=LiteralExpr(value=1, literal_type='int')),
                CaseBranch(pattern=VarPattern(name='n'), body=VarExpr(name='n'))
            ]
        )
        result = emitter._emit_match(e)
        assert 'match x with' in result
        assert '| 0 ->' in result
        assert '| n ->' in result
    
    def test_emit_expr_list(self, emitter):
        """Test list expression emission."""
        e = ListExpr(elements=[
            LiteralExpr(value=1, literal_type='int'),
            LiteralExpr(value=2, literal_type='int')
        ])
        result = emitter._emit_expr(e)
        assert '[1; 2]' in result
    
    def test_emit_variant_type(self, emitter):
        """Test variant type emission."""
        option = DataType(
            name='option',
            type_params=[TypeParameter(name='a')],
            constructors=[
                DataConstructor(name='None'),
                DataConstructor(name='Some', fields=[TypeVar(name='a')])
            ]
        )
        result = emitter._emit_variant_type(option)
        assert "type ('a) option =" in result
        assert '| None' in result
        assert '| Some of' in result
    
    def test_emit_record_type(self, emitter):
        """Test record type emission."""
        person = RecordType(
            name='person',
            fields=[
                RecordField(name='name', field_type=TypeCon(name='String')),
                RecordField(name='age', field_type=TypeCon(name='Int'), mutable=True)
            ]
        )
        result = emitter._emit_record_type(person)
        assert 'type person = {' in result
        assert 'name: string' in result
        assert 'mutable age: int' in result
    
    def test_emit_function_identity(self, emitter):
        """Test identity function emission."""
        identity = FunctionDef(
            name='identity',
            clauses=[FunctionClause(
                patterns=[VarPattern(name='x')],
                body=VarExpr(name='x')
            )]
        )
        result = emitter._emit_function(identity)
        assert 'let identity x = x' in result
    
    def test_emit_function_recursive(self, emitter):
        """Test recursive function emission."""
        length = FunctionDef(
            name='length',
            clauses=[
                FunctionClause(
                    patterns=[ListPattern(elements=[])],
                    body=LiteralExpr(value=0, literal_type='int')
                ),
                FunctionClause(
                    patterns=[ListPattern(elements=[WildcardPattern()], rest=VarPattern(name='xs'))],
                    body=BinaryOpExpr(
                        op='+',
                        left=LiteralExpr(value=1, literal_type='int'),
                        right=AppExpr(func=VarExpr(name='length'), arg=VarExpr(name='xs'))
                    )
                )
            ]
        )
        result = emitter._emit_function(length)
        assert 'let rec length' in result
        assert 'match' in result
    
    def test_emit_functor(self, emitter):
        """Test functor emission."""
        body = Module(
            name='SetImpl',
            functions=[FunctionDef(
                name='empty',
                clauses=[FunctionClause(
                    patterns=[],
                    body=ListExpr(elements=[])
                )]
            )]
        )
        result = emitter.emit_functor('MakeSet', 'Ord', 'OrderedType', body)
        assert 'module MakeSet (Ord : OrderedType) = struct' in result
        assert 'let empty' in result
        assert 'end' in result
    
    def test_emit_ref_operations(self, emitter):
        """Test reference operations emission."""
        # Reference creation
        ref_code = emitter.emit_ref_operations('counter', LiteralExpr(value=0, literal_type='int'))
        assert 'let counter = ref 0' in ref_code
        
        # Assignment
        assign_code = emitter.emit_assignment('counter', LiteralExpr(value=1, literal_type='int'))
        assert 'counter := 1' in assign_code
        
        # Dereference
        deref_code = emitter.emit_deref('counter')
        assert '!counter' in deref_code
    
    def test_emit_module(self, emitter):
        """Test complete module emission."""
        module = Module(
            name='Example',
            imports=[
                Import(module='List'),
                Import(module='Array', qualified=True, alias='A')
            ],
            type_definitions=[
                DataType(
                    name='option',
                    type_params=[TypeParameter(name='a')],
                    constructors=[
                        DataConstructor(name='None'),
                        DataConstructor(name='Some', fields=[TypeVar(name='a')])
                    ]
                )
            ],
            functions=[
                FunctionDef(
                    name='identity',
                    clauses=[FunctionClause(
                        patterns=[VarPattern(name='x')],
                        body=VarExpr(name='x')
                    )]
                )
            ]
        )
        result = emitter.emit_module(module)
        assert 'open List' in result
        assert 'module A = Array' in result
        assert "type ('a) option =" in result
        assert 'let identity x = x' in result


# =============================================================================
# Integration Tests
# =============================================================================

class TestEmitterIntegration:
    """Integration tests for both emitters."""
    
    def test_map_function_haskell(self):
        """Test emitting map function in Haskell."""
        emitter = HaskellEmitter()
        
        map_func = FunctionDef(
            name='map',
            type_signature=FunctionType(
                param_type=FunctionType(
                    param_type=TypeVar(name='a'),
                    return_type=TypeVar(name='b')
                ),
                return_type=FunctionType(
                    param_type=ListType(element_type=TypeVar(name='a')),
                    return_type=ListType(element_type=TypeVar(name='b'))
                )
            ),
            clauses=[
                FunctionClause(
                    patterns=[WildcardPattern(), ListPattern(elements=[])],
                    body=ListExpr(elements=[])
                ),
                FunctionClause(
                    patterns=[
                        VarPattern(name='f'),
                        ListPattern(elements=[VarPattern(name='x')], rest=VarPattern(name='xs'))
                    ],
                    body=BinaryOpExpr(
                        op=':',
                        left=AppExpr(func=VarExpr(name='f'), arg=VarExpr(name='x')),
                        right=AppExpr(
                            func=AppExpr(func=VarExpr(name='map'), arg=VarExpr(name='f')),
                            arg=VarExpr(name='xs')
                        )
                    )
                )
            ]
        )
        result = emitter._emit_function(map_func)
        assert 'map ::' in result
        assert 'map _ []' in result
        assert ':' in result
    
    def test_filter_function_ocaml(self):
        """Test emitting filter function in OCaml."""
        emitter = OCamlEmitter()
        
        filter_func = FunctionDef(
            name='filter',
            clauses=[
                FunctionClause(
                    patterns=[WildcardPattern(), ListPattern(elements=[])],
                    body=ListExpr(elements=[])
                ),
                FunctionClause(
                    patterns=[
                        VarPattern(name='p'),
                        ListPattern(elements=[VarPattern(name='x')], rest=VarPattern(name='xs'))
                    ],
                    body=IfExpr(
                        condition=AppExpr(func=VarExpr(name='p'), arg=VarExpr(name='x')),
                        then_branch=BinaryOpExpr(
                            op='::',
                            left=VarExpr(name='x'),
                            right=AppExpr(
                                func=AppExpr(func=VarExpr(name='filter'), arg=VarExpr(name='p')),
                                arg=VarExpr(name='xs')
                            )
                        ),
                        else_branch=AppExpr(
                            func=AppExpr(func=VarExpr(name='filter'), arg=VarExpr(name='p')),
                            arg=VarExpr(name='xs')
                        )
                    )
                )
            ]
        )
        result = emitter._emit_function(filter_func)
        assert 'let rec filter' in result
        assert 'if' in result
        assert 'then' in result
        assert 'else' in result
    
    def test_tree_operations(self):
        """Test tree data type and operations."""
        tree = DataType(
            name='Tree',
            type_params=[TypeParameter(name='a')],
            constructors=[
                DataConstructor(name='Leaf', fields=[TypeVar(name='a')]),
                DataConstructor(
                    name='Node',
                    fields=[
                        TypeCon(name='Tree', args=[TypeVar(name='a')]),
                        TypeVar(name='a'),
                        TypeCon(name='Tree', args=[TypeVar(name='a')])
                    ]
                )
            ]
        )
        
        # Haskell
        hs_emitter = HaskellEmitter()
        hs_result = hs_emitter._emit_data_type(tree)
        assert 'data Tree a' in hs_result
        assert 'Leaf a' in hs_result
        assert 'Node' in hs_result
        
        # OCaml - need to adapt for OCaml naming conventions
        ml_tree = DataType(
            name='tree',
            type_params=[TypeParameter(name='a')],
            constructors=[
                DataConstructor(name='Leaf', fields=[TypeVar(name='a')]),
                DataConstructor(
                    name='Node',
                    fields=[
                        TypeCon(name='tree', args=[TypeVar(name='a')]),
                        TypeVar(name='a'),
                        TypeCon(name='tree', args=[TypeVar(name='a')])
                    ]
                )
            ]
        )
        ml_emitter = OCamlEmitter()
        ml_result = ml_emitter._emit_variant_type(ml_tree)
        assert "type ('a) tree =" in ml_result
        assert '| Leaf of' in ml_result
        assert '| Node of' in ml_result


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
