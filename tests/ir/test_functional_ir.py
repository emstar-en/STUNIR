#!/usr/bin/env python3
"""Tests for STUNIR Functional IR.

This module tests the functional programming IR constructs including:
- Expressions and types
- Pattern matching
- Algebraic data types
- Type inference
"""

import sys
import pytest
from pathlib import Path

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ir.functional import (
    # Type expressions
    TypeVar, TypeCon, FunctionType, TupleType, ListType,
    # Expressions
    LiteralExpr, VarExpr, AppExpr, LambdaExpr, LetExpr, IfExpr,
    CaseBranch, CaseExpr, ListGenerator, ListExpr, TupleExpr,
    BinaryOpExpr, UnaryOpExpr, DoExpr, DoBindStatement, DoLetStatement,
    # Patterns
    WildcardPattern, VarPattern, LiteralPattern, ConstructorPattern,
    TuplePattern, ListPattern, AsPattern,
    get_pattern_variables, is_exhaustive, simplify_pattern,
    # ADTs
    TypeParameter, DataConstructor, DataType, TypeAlias, NewType,
    RecordField, RecordType, FunctionClause, FunctionDef, Module, Import,
    MethodSignature, TypeClass, TypeClassInstance,
    # Type system
    TypeEnvironment, TypeInference, TypeError, free_type_vars, type_to_string,
    # Utilities
    make_app, make_lambda, make_let_chain,
)


# =============================================================================
# Type Expression Tests
# =============================================================================

class TestTypeExpressions:
    """Tests for type expression nodes."""
    
    def test_type_var(self):
        """Test type variable creation."""
        t = TypeVar(name='a')
        assert t.name == 'a'
        assert t.kind == 'type_var'
    
    def test_type_con_simple(self):
        """Test simple type constructor."""
        t = TypeCon(name='Int')
        assert t.name == 'Int'
        assert t.args == []
        assert t.kind == 'type_con'
    
    def test_type_con_with_args(self):
        """Test type constructor with arguments."""
        a = TypeVar(name='a')
        list_t = TypeCon(name='List', args=[a])
        assert list_t.name == 'List'
        assert len(list_t.args) == 1
        assert list_t.args[0].name == 'a'
    
    def test_function_type(self):
        """Test function type."""
        int_t = TypeCon(name='Int')
        bool_t = TypeCon(name='Bool')
        func_t = FunctionType(param_type=int_t, return_type=bool_t)
        assert func_t.kind == 'function_type'
        assert func_t.param_type.name == 'Int'
        assert func_t.return_type.name == 'Bool'
    
    def test_tuple_type(self):
        """Test tuple type."""
        int_t = TypeCon(name='Int')
        bool_t = TypeCon(name='Bool')
        tuple_t = TupleType(elements=[int_t, bool_t])
        assert tuple_t.kind == 'tuple_type'
        assert len(tuple_t.elements) == 2
    
    def test_list_type(self):
        """Test list type."""
        int_t = TypeCon(name='Int')
        list_t = ListType(element_type=int_t)
        assert list_t.kind == 'list_type'
        assert list_t.element_type.name == 'Int'


# =============================================================================
# Expression Tests
# =============================================================================

class TestExpressions:
    """Tests for expression nodes."""
    
    def test_literal_int(self):
        """Test integer literal."""
        e = LiteralExpr(value=42, literal_type='int')
        assert e.value == 42
        assert e.literal_type == 'int'
        assert e.kind == 'literal'
    
    def test_literal_bool(self):
        """Test boolean literal."""
        e = LiteralExpr(value=True, literal_type='bool')
        assert e.value is True
        assert e.literal_type == 'bool'
    
    def test_literal_string(self):
        """Test string literal."""
        e = LiteralExpr(value='hello', literal_type='string')
        assert e.value == 'hello'
        assert e.literal_type == 'string'
    
    def test_var_expr(self):
        """Test variable expression."""
        e = VarExpr(name='x')
        assert e.name == 'x'
        assert e.kind == 'var'
    
    def test_app_expr(self):
        """Test function application."""
        f = VarExpr(name='f')
        x = VarExpr(name='x')
        app = AppExpr(func=f, arg=x)
        assert app.kind == 'app'
        assert app.func.name == 'f'
        assert app.arg.name == 'x'
    
    def test_lambda_expr(self):
        """Test lambda expression."""
        body = VarExpr(name='x')
        lam = LambdaExpr(param='x', body=body)
        assert lam.kind == 'lambda'
        assert lam.param == 'x'
        assert lam.body.name == 'x'
    
    def test_let_expr(self):
        """Test let expression."""
        val = LiteralExpr(value=5, literal_type='int')
        body = VarExpr(name='x')
        let = LetExpr(name='x', value=val, body=body)
        assert let.kind == 'let'
        assert let.name == 'x'
        assert not let.is_recursive
    
    def test_let_rec_expr(self):
        """Test recursive let expression."""
        body = VarExpr(name='fact')
        let = LetExpr(name='fact', value=body, body=body, is_recursive=True)
        assert let.is_recursive is True
    
    def test_if_expr(self):
        """Test if expression."""
        cond = LiteralExpr(value=True, literal_type='bool')
        then_b = LiteralExpr(value=1, literal_type='int')
        else_b = LiteralExpr(value=2, literal_type='int')
        if_e = IfExpr(condition=cond, then_branch=then_b, else_branch=else_b)
        assert if_e.kind == 'if'
    
    def test_case_expr(self):
        """Test case expression."""
        scrutinee = VarExpr(name='x')
        branch = CaseBranch(
            pattern=VarPattern(name='y'),
            body=VarExpr(name='y')
        )
        case = CaseExpr(scrutinee=scrutinee, branches=[branch])
        assert case.kind == 'case'
        assert len(case.branches) == 1
    
    def test_list_expr(self):
        """Test list expression."""
        elems = [
            LiteralExpr(value=1, literal_type='int'),
            LiteralExpr(value=2, literal_type='int'),
        ]
        lst = ListExpr(elements=elems)
        assert lst.kind == 'list'
        assert len(lst.elements) == 2
    
    def test_tuple_expr(self):
        """Test tuple expression."""
        elems = [
            LiteralExpr(value=1, literal_type='int'),
            LiteralExpr(value=True, literal_type='bool'),
        ]
        tup = TupleExpr(elements=elems)
        assert tup.kind == 'tuple'
        assert len(tup.elements) == 2
    
    def test_binary_op(self):
        """Test binary operation."""
        left = LiteralExpr(value=1, literal_type='int')
        right = LiteralExpr(value=2, literal_type='int')
        binop = BinaryOpExpr(op='+', left=left, right=right)
        assert binop.kind == 'binary_op'
        assert binop.op == '+'
    
    def test_unary_op(self):
        """Test unary operation."""
        operand = LiteralExpr(value=True, literal_type='bool')
        unop = UnaryOpExpr(op='not', operand=operand)
        assert unop.kind == 'unary_op'
        assert unop.op == 'not'


# =============================================================================
# Pattern Matching Tests
# =============================================================================

class TestPatternMatching:
    """Tests for pattern matching definitions."""
    
    def test_wildcard_pattern(self):
        """Test wildcard pattern."""
        p = WildcardPattern()
        assert p.kind == 'wildcard'
    
    def test_var_pattern(self):
        """Test variable pattern."""
        p = VarPattern(name='x')
        assert p.name == 'x'
        assert p.kind == 'var_pattern'
    
    def test_literal_pattern(self):
        """Test literal pattern."""
        p = LiteralPattern(value=42, literal_type='int')
        assert p.value == 42
        assert p.kind == 'literal_pattern'
    
    def test_constructor_pattern(self):
        """Test constructor pattern."""
        inner = VarPattern(name='x')
        p = ConstructorPattern(constructor='Just', args=[inner])
        assert p.constructor == 'Just'
        assert len(p.args) == 1
        assert p.kind == 'constructor_pattern'
    
    def test_tuple_pattern(self):
        """Test tuple pattern."""
        a = VarPattern(name='a')
        b = VarPattern(name='b')
        p = TuplePattern(elements=[a, b])
        assert len(p.elements) == 2
        assert p.kind == 'tuple_pattern'
    
    def test_list_pattern(self):
        """Test list pattern."""
        head = VarPattern(name='h')
        tail = VarPattern(name='t')
        p = ListPattern(elements=[head], rest=tail)
        assert len(p.elements) == 1
        assert p.rest.name == 't'
        assert p.kind == 'list_pattern'
    
    def test_as_pattern(self):
        """Test as pattern."""
        inner = ConstructorPattern(constructor='Just', args=[VarPattern(name='x')])
        p = AsPattern(name='whole', pattern=inner)
        assert p.name == 'whole'
        assert p.kind == 'as_pattern'
    
    def test_get_pattern_variables(self):
        """Test extracting variables from patterns."""
        # Tuple pattern (a, b)
        p = TuplePattern(elements=[
            VarPattern(name='a'),
            VarPattern(name='b')
        ])
        vars = get_pattern_variables(p)
        assert 'a' in vars
        assert 'b' in vars
        
        # Constructor pattern Just x
        p2 = ConstructorPattern(constructor='Just', args=[VarPattern(name='x')])
        vars2 = get_pattern_variables(p2)
        assert 'x' in vars2
        
        # Wildcard
        p3 = WildcardPattern()
        vars3 = get_pattern_variables(p3)
        assert vars3 == []
    
    def test_is_exhaustive(self):
        """Test exhaustiveness check."""
        constructors = ['Just', 'Nothing']
        
        # Wildcard is always exhaustive
        patterns = [WildcardPattern()]
        assert is_exhaustive(patterns, constructors) is True
        
        # All constructors covered
        patterns2 = [
            ConstructorPattern(constructor='Just', args=[WildcardPattern()]),
            ConstructorPattern(constructor='Nothing')
        ]
        assert is_exhaustive(patterns2, constructors) is True


# =============================================================================
# Algebraic Data Type Tests
# =============================================================================

class TestADTs:
    """Tests for algebraic data type definitions."""
    
    def test_data_type_maybe(self):
        """Test Maybe type definition."""
        maybe = DataType(
            name='Maybe',
            type_params=[TypeParameter(name='a')],
            constructors=[
                DataConstructor(name='Nothing'),
                DataConstructor(name='Just', fields=[TypeVar(name='a')])
            ],
            deriving=['Eq', 'Show']
        )
        assert maybe.name == 'Maybe'
        assert len(maybe.constructors) == 2
        assert maybe.constructors[0].name == 'Nothing'
        assert maybe.constructors[1].name == 'Just'
        assert maybe.deriving == ['Eq', 'Show']
    
    def test_data_type_list(self):
        """Test List type definition."""
        list_t = DataType(
            name='List',
            type_params=[TypeParameter(name='a')],
            constructors=[
                DataConstructor(name='Nil'),
                DataConstructor(
                    name='Cons',
                    fields=[TypeVar(name='a'), TypeCon(name='List', args=[TypeVar(name='a')])]
                )
            ]
        )
        assert list_t.name == 'List'
        assert list_t.constructors[1].name == 'Cons'
        assert len(list_t.constructors[1].fields) == 2
    
    def test_data_type_tree(self):
        """Test Tree type definition."""
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
        assert tree.name == 'Tree'
        assert not tree.is_enum()
        assert tree.constructor_names() == ['Leaf', 'Node']
    
    def test_type_alias(self):
        """Test type alias."""
        alias = TypeAlias(
            name='IntList',
            target=TypeCon(name='List', args=[TypeCon(name='Int')])
        )
        assert alias.name == 'IntList'
        assert alias.kind == 'type_alias'
    
    def test_newtype(self):
        """Test newtype definition."""
        newtype = NewType(
            name='Age',
            constructor=DataConstructor(name='Age', fields=[TypeCon(name='Int')]),
            deriving=['Eq', 'Ord']
        )
        assert newtype.name == 'Age'
        assert newtype.kind == 'newtype'
    
    def test_record_type(self):
        """Test record type."""
        person = RecordType(
            name='Person',
            fields=[
                RecordField(name='name', field_type=TypeCon(name='String')),
                RecordField(name='age', field_type=TypeCon(name='Int'), mutable=True)
            ]
        )
        assert person.name == 'Person'
        assert person.field_names() == ['name', 'age']
        assert person.has_mutable_fields() is True
    
    def test_type_class(self):
        """Test type class definition."""
        eq_class = TypeClass(
            name='Eq',
            type_params=[TypeParameter(name='a')],
            methods=[
                MethodSignature(
                    name='eq',
                    type_signature=FunctionType(
                        param_type=TypeVar(name='a'),
                        return_type=FunctionType(
                            param_type=TypeVar(name='a'),
                            return_type=TypeCon(name='Bool')
                        )
                    )
                )
            ]
        )
        assert eq_class.name == 'Eq'
        assert len(eq_class.methods) == 1
    
    def test_type_class_instance(self):
        """Test type class instance."""
        instance = TypeClassInstance(
            class_name='Eq',
            type_args=[TypeCon(name='Int')],
            implementations={
                'eq': LambdaExpr(
                    param='a',
                    body=LambdaExpr(
                        param='b',
                        body=BinaryOpExpr(
                            op='==',
                            left=VarExpr(name='a'),
                            right=VarExpr(name='b')
                        )
                    )
                )
            }
        )
        assert instance.class_name == 'Eq'
        assert 'eq' in instance.implementations


# =============================================================================
# Type Inference Tests
# =============================================================================

class TestTypeInference:
    """Tests for type inference."""
    
    def test_infer_literal_int(self):
        """Test inferring type of integer literal."""
        inference = TypeInference()
        env = TypeEnvironment()
        
        lit = LiteralExpr(value=42, literal_type='int')
        t = inference.infer(lit, env)
        assert isinstance(t, TypeCon)
        assert t.name == 'Int'
    
    def test_infer_literal_bool(self):
        """Test inferring type of boolean literal."""
        inference = TypeInference()
        env = TypeEnvironment()
        
        lit = LiteralExpr(value=True, literal_type='bool')
        t = inference.infer(lit, env)
        assert t.name == 'Bool'
    
    def test_infer_variable(self):
        """Test inferring type of variable."""
        inference = TypeInference()
        env = TypeEnvironment()
        env = env.extend('x', TypeCon(name='Int'))
        
        var = VarExpr(name='x')
        t = inference.infer(var, env)
        assert t.name == 'Int'
    
    def test_infer_unbound_variable(self):
        """Test error for unbound variable."""
        inference = TypeInference()
        env = TypeEnvironment()
        
        var = VarExpr(name='undefined')
        with pytest.raises(TypeError):
            inference.infer(var, env)
    
    def test_infer_lambda(self):
        """Test inferring type of lambda."""
        inference = TypeInference()
        env = TypeEnvironment()
        
        # \x -> x (identity function)
        lam = LambdaExpr(param='x', body=VarExpr(name='x'))
        t = inference.infer(lam, env)
        assert isinstance(t, FunctionType)
    
    def test_infer_application(self):
        """Test inferring type of function application."""
        inference = TypeInference()
        env = TypeEnvironment()
        
        # Add 'succ :: Int -> Int' to environment
        succ_type = FunctionType(
            param_type=TypeCon(name='Int'),
            return_type=TypeCon(name='Int')
        )
        env = env.extend('succ', succ_type)
        
        # succ 5
        app = AppExpr(
            func=VarExpr(name='succ'),
            arg=LiteralExpr(value=5, literal_type='int')
        )
        t = inference.infer(app, env)
        t = inference.apply_substitution(t)
        assert isinstance(t, TypeCon)
        assert t.name == 'Int'
    
    def test_infer_if_expr(self):
        """Test inferring type of if expression."""
        inference = TypeInference()
        env = TypeEnvironment()
        
        if_expr = IfExpr(
            condition=LiteralExpr(value=True, literal_type='bool'),
            then_branch=LiteralExpr(value=1, literal_type='int'),
            else_branch=LiteralExpr(value=2, literal_type='int')
        )
        t = inference.infer(if_expr, env)
        t = inference.apply_substitution(t)
        assert t.name == 'Int'
    
    def test_infer_let(self):
        """Test inferring type of let expression."""
        inference = TypeInference()
        env = TypeEnvironment()
        
        # let x = 5 in x
        let_expr = LetExpr(
            name='x',
            value=LiteralExpr(value=5, literal_type='int'),
            body=VarExpr(name='x')
        )
        t = inference.infer(let_expr, env)
        t = inference.apply_substitution(t)
        assert t.name == 'Int'
    
    def test_infer_list(self):
        """Test inferring type of list."""
        inference = TypeInference()
        env = TypeEnvironment()
        
        lst = ListExpr(elements=[
            LiteralExpr(value=1, literal_type='int'),
            LiteralExpr(value=2, literal_type='int')
        ])
        t = inference.infer(lst, env)
        assert isinstance(t, ListType)
        t_elem = inference.apply_substitution(t.element_type)
        assert t_elem.name == 'Int'
    
    def test_infer_tuple(self):
        """Test inferring type of tuple."""
        inference = TypeInference()
        env = TypeEnvironment()
        
        tup = TupleExpr(elements=[
            LiteralExpr(value=1, literal_type='int'),
            LiteralExpr(value=True, literal_type='bool')
        ])
        t = inference.infer(tup, env)
        assert isinstance(t, TupleType)
        assert len(t.elements) == 2
    
    def test_infer_binary_op_arithmetic(self):
        """Test inferring type of arithmetic operations."""
        inference = TypeInference()
        env = TypeEnvironment()
        
        add = BinaryOpExpr(
            op='+',
            left=LiteralExpr(value=1, literal_type='int'),
            right=LiteralExpr(value=2, literal_type='int')
        )
        t = inference.infer(add, env)
        t = inference.apply_substitution(t)
        assert t.name == 'Int'
    
    def test_infer_binary_op_comparison(self):
        """Test inferring type of comparison operations."""
        inference = TypeInference()
        env = TypeEnvironment()
        
        lt = BinaryOpExpr(
            op='<',
            left=LiteralExpr(value=1, literal_type='int'),
            right=LiteralExpr(value=2, literal_type='int')
        )
        t = inference.infer(lt, env)
        assert t.name == 'Bool'
    
    def test_unify_type_vars(self):
        """Test unification of type variables."""
        inference = TypeInference()
        
        t1 = TypeVar(name='a')
        t2 = TypeCon(name='Int')
        
        assert inference.unify(t1, t2) is True
        resolved = inference.apply_substitution(t1)
        assert resolved.name == 'Int'
    
    def test_unify_function_types(self):
        """Test unification of function types."""
        inference = TypeInference()
        
        f1 = FunctionType(
            param_type=TypeVar(name='a'),
            return_type=TypeVar(name='b')
        )
        f2 = FunctionType(
            param_type=TypeCon(name='Int'),
            return_type=TypeCon(name='Bool')
        )
        
        assert inference.unify(f1, f2) is True
        resolved_param = inference.apply_substitution(f1.param_type)
        resolved_return = inference.apply_substitution(f1.return_type)
        assert resolved_param.name == 'Int'
        assert resolved_return.name == 'Bool'
    
    def test_free_type_vars(self):
        """Test finding free type variables."""
        # Int has no free type vars
        assert free_type_vars(TypeCon(name='Int')) == set()
        
        # Type variable has itself as free
        assert free_type_vars(TypeVar(name='a')) == {'a'}
        
        # Function type
        ft = FunctionType(
            param_type=TypeVar(name='a'),
            return_type=TypeVar(name='b')
        )
        assert free_type_vars(ft) == {'a', 'b'}
    
    def test_type_to_string(self):
        """Test type pretty printing."""
        int_t = TypeCon(name='Int')
        assert type_to_string(int_t) == 'Int'
        
        func_t = FunctionType(
            param_type=TypeCon(name='Int'),
            return_type=TypeCon(name='Bool')
        )
        assert 'Int' in type_to_string(func_t)
        assert 'Bool' in type_to_string(func_t)
        assert '->' in type_to_string(func_t)


# =============================================================================
# Utility Function Tests
# =============================================================================

class TestUtilities:
    """Tests for utility functions."""
    
    def test_make_app(self):
        """Test curried application builder."""
        f = VarExpr(name='f')
        x = VarExpr(name='x')
        y = VarExpr(name='y')
        
        app = make_app(f, x, y)
        assert isinstance(app, AppExpr)
        assert isinstance(app.func, AppExpr)
    
    def test_make_lambda(self):
        """Test curried lambda builder."""
        body = VarExpr(name='x')
        lam = make_lambda(['x', 'y'], body)
        
        assert isinstance(lam, LambdaExpr)
        assert lam.param == 'x'
        assert isinstance(lam.body, LambdaExpr)
        assert lam.body.param == 'y'
    
    def test_make_let_chain(self):
        """Test let chain builder."""
        body = VarExpr(name='y')
        bindings = [
            ('x', LiteralExpr(value=1, literal_type='int')),
            ('y', BinaryOpExpr(op='+', left=VarExpr(name='x'), right=LiteralExpr(value=1, literal_type='int')))
        ]
        
        let = make_let_chain(bindings, body)
        assert isinstance(let, LetExpr)
        assert let.name == 'x'
        assert isinstance(let.body, LetExpr)
        assert let.body.name == 'y'


# =============================================================================
# Module Definition Tests
# =============================================================================

class TestModuleDefinition:
    """Tests for module definitions."""
    
    def test_function_def(self):
        """Test function definition."""
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
        assert identity.name == 'identity'
        assert len(identity.clauses) == 1
    
    def test_module(self):
        """Test module definition."""
        module = Module(
            name='Example',
            exports=['identity', 'Maybe'],
            imports=[Import(module='Prelude')],
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
        assert module.name == 'Example'
        assert len(module.exports) == 2
        assert len(module.type_definitions) == 1
        assert len(module.functions) == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
