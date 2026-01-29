#!/usr/bin/env python3
"""Tests for STUNIR Symbolic IR Extensions.

Part of Phase 5A: Core Lisp Implementation.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tools.ir.symbolic_ir import (
    SymbolicIRExtension,
    Symbol, Atom, SList, Cons, Quote, Quasiquote,
    Unquote, UnquoteSplicing, Lambda, Macro,
    sexpr, sym, quote, quasiquote, unquote, unquote_splicing,
    SYMBOLIC_KINDS
)


class TestSymbolClass:
    """Test Symbol class."""
    
    def test_symbol_creation(self):
        """Test basic symbol creation."""
        s = Symbol("hello")
        assert s.name == "hello"
        assert s.package is None
    
    def test_symbol_with_package(self):
        """Test symbol with package qualifier."""
        s = Symbol("format", "cl")
        assert s.name == "format"
        assert s.package == "cl"
        assert str(s) == "cl:format"
    
    def test_symbol_to_dict(self):
        """Test symbol to dictionary conversion."""
        s = Symbol("test")
        d = s.to_dict()
        assert d == {'kind': 'symbol', 'name': 'test'}
    
    def test_symbol_from_dict(self):
        """Test symbol creation from dictionary."""
        d = {'kind': 'symbol', 'name': 'test', 'package': 'pkg'}
        s = Symbol.from_dict(d)
        assert s.name == 'test'
        assert s.package == 'pkg'


class TestAtomClass:
    """Test Atom class."""
    
    def test_atom_integer(self):
        """Test integer atom."""
        a = Atom(42)
        assert a.value == 42
        assert str(a) == "42"
    
    def test_atom_float(self):
        """Test float atom."""
        a = Atom(3.14)
        assert a.value == 3.14
    
    def test_atom_string(self):
        """Test string atom."""
        a = Atom("hello")
        assert str(a) == '"hello"'
    
    def test_atom_boolean(self):
        """Test boolean atom."""
        assert str(Atom(True)) == "t"
        assert str(Atom(False)) == "nil"
    
    def test_atom_nil(self):
        """Test nil atom."""
        assert str(Atom(None)) == "nil"


class TestSListClass:
    """Test SList class."""
    
    def test_slist_creation(self):
        """Test basic list creation."""
        lst = SList([Symbol("a"), Symbol("b"), Symbol("c")])
        assert len(lst) == 3
        assert str(lst) == "(a b c)"
    
    def test_slist_to_dict(self):
        """Test list to dictionary conversion."""
        lst = SList([Symbol("add"), Atom(1), Atom(2)])
        d = lst.to_dict()
        assert d['kind'] == 'list'
        assert len(d['elements']) == 3
    
    def test_slist_indexing(self):
        """Test list indexing."""
        lst = SList([Atom(1), Atom(2), Atom(3)])
        assert lst[0].value == 1
        assert lst[2].value == 3


class TestQuoteClass:
    """Test Quote class."""
    
    def test_quote_creation(self):
        """Test quote creation."""
        q = Quote(Symbol("hello"))
        assert str(q) == "'hello"
    
    def test_quote_to_dict(self):
        """Test quote to dictionary conversion."""
        q = Quote(SList([Symbol("a"), Symbol("b")]))
        d = q.to_dict()
        assert d['kind'] == 'quote'
        assert d['value']['kind'] == 'list'


class TestLambdaClass:
    """Test Lambda class."""
    
    def test_lambda_creation(self):
        """Test lambda creation."""
        lam = Lambda(
            params=[{'name': 'x', 'type': 'i32'}],
            body=[{'kind': 'return', 'value': {'kind': 'var', 'name': 'x'}}]
        )
        assert len(lam.params) == 1
        assert len(lam.body) == 1
    
    def test_lambda_with_rest_param(self):
        """Test lambda with rest parameter."""
        lam = Lambda(
            params=[{'name': 'first'}],
            rest_param='rest',
            body=[]
        )
        assert lam.rest_param == 'rest'
    
    def test_lambda_to_dict(self):
        """Test lambda to dictionary conversion."""
        lam = Lambda(params=[], body=[])
        d = lam.to_dict()
        assert d['kind'] == 'lambda'
        assert 'params' in d
        assert 'body' in d


class TestMacroClass:
    """Test Macro class."""
    
    def test_macro_creation(self):
        """Test macro creation."""
        mac = Macro(
            name='when',
            params=[{'name': 'cond'}, {'name': 'body'}],
            body=[{'kind': 'return', 'value': None}]
        )
        assert mac.name == 'when'
        assert len(mac.params) == 2


class TestSymbolicIRExtension:
    """Test SymbolicIRExtension class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.ext = SymbolicIRExtension()
    
    def test_validate_basic_ir(self):
        """TC-SYM-001: Test basic IR validation."""
        ir = {
            "module": "test",
            "functions": [{
                "name": "get_symbol",
                "body": [{"kind": "return", "value": {"kind": "symbol", "name": "hello"}}]
            }]
        }
        assert self.ext.validate(ir) is True
        assert self.ext.has_symbolic_features(ir) is True
    
    def test_validate_quote(self):
        """TC-SYM-002: Test quote validation."""
        ir = {
            "module": "test",
            "functions": [{
                "name": "quoted_list",
                "body": [{
                    "kind": "return",
                    "value": {
                        "kind": "quote",
                        "value": {
                            "kind": "list",
                            "elements": [
                                {"kind": "symbol", "name": "a"},
                                {"kind": "symbol", "name": "b"}
                            ]
                        }
                    }
                }]
            }]
        }
        assert self.ext.validate(ir) is True
    
    def test_validate_lambda(self):
        """TC-SYM-003: Test lambda validation."""
        ir = {
            "module": "test",
            "functions": [{
                "name": "make_adder",
                "params": [{"name": "n", "type": "i32"}],
                "body": [{
                    "kind": "return",
                    "value": {
                        "kind": "lambda",
                        "params": [{"name": "x", "type": "i32"}],
                        "body": [{
                            "kind": "return",
                            "value": {
                                "kind": "binary_op",
                                "op": "+",
                                "left": {"kind": "var", "name": "n"},
                                "right": {"kind": "var", "name": "x"}
                            }
                        }]
                    }
                }]
            }]
        }
        assert self.ext.validate(ir) is True
    
    def test_has_symbolic_features_false(self):
        """Test detection of non-symbolic IR."""
        ir = {
            "module": "test",
            "functions": [{
                "name": "add",
                "body": [{"kind": "return", "value": {"kind": "literal", "value": 42}}]
            }]
        }
        assert self.ext.has_symbolic_features(ir) is False
    
    def test_extract_macros(self):
        """Test macro extraction."""
        ir = {
            "module": "test",
            "definitions": [
                {"kind": "defmacro", "name": "when", "params": [], "body": []},
                {"kind": "function", "name": "foo"}
            ],
            "macros": [
                {"kind": "defmacro", "name": "unless", "params": [], "body": []}
            ]
        }
        macros = self.ext.extract_macros(ir)
        assert len(macros) == 2
        names = [m['name'] for m in macros]
        assert 'when' in names
        assert 'unless' in names
    
    def test_extract_lambdas(self):
        """Test lambda extraction."""
        ir = {
            "module": "test",
            "functions": [{
                "name": "test",
                "body": [
                    {"kind": "lambda", "params": [{"name": "x"}], "body": []},
                    {"kind": "return", "value": {
                        "kind": "lambda", "params": [{"name": "y"}], "body": []
                    }}
                ]
            }]
        }
        lambdas = self.ext.extract_lambdas(ir)
        assert len(lambdas) == 2
    
    def test_extract_quotes(self):
        """Test quote extraction."""
        ir = {
            "module": "test",
            "functions": [{
                "name": "test",
                "body": [
                    {"kind": "quote", "value": {"kind": "symbol", "name": "a"}},
                    {"kind": "quasiquote", "value": {"kind": "symbol", "name": "b"}}
                ]
            }]
        }
        quotes = self.ext.extract_quotes(ir)
        assert len(quotes) == 2
    
    def test_validate_invalid_symbol(self):
        """Test validation rejects invalid symbol."""
        ir = {
            "module": "test",
            "functions": [{
                "name": "test",
                "body": [{"kind": "symbol"}]  # Missing 'name'
            }]
        }
        with pytest.raises(ValueError):
            self.ext.validate(ir)
    
    def test_validate_invalid_lambda(self):
        """Test validation rejects invalid lambda."""
        ir = {
            "module": "test",
            "functions": [{
                "name": "test",
                "body": [{"kind": "lambda"}]  # Missing params and body
            }]
        }
        with pytest.raises(ValueError):
            self.ext.validate(ir)
    
    def test_to_sexpression_string(self):
        """Test S-expression string conversion."""
        data = {
            "kind": "list",
            "elements": [
                {"kind": "symbol", "name": "add"},
                1,
                2
            ]
        }
        result = self.ext.to_sexpression_string(data)
        assert result == "(add 1 2)"
    
    def test_transform_quotes(self):
        """Test quote transformation."""
        expr = {
            "kind": "quote",
            "value": {
                "kind": "list",
                "elements": [
                    {"kind": "symbol", "name": "a"}
                ]
            }
        }
        result = self.ext.transform_quotes(expr)
        assert result['kind'] == 'quote'
        assert result['value']['kind'] == 'list'


class TestConvenienceFunctions:
    """Test convenience builder functions."""
    
    def test_sexpr_builder(self):
        """Test sexpr builder function."""
        expr = sexpr("add", 1, 2)
        assert len(expr) == 3
        assert isinstance(expr[0], Symbol)
        assert isinstance(expr[1], Atom)
    
    def test_sym_builder(self):
        """Test sym builder function."""
        s = sym("hello", "cl")
        assert s.name == "hello"
        assert s.package == "cl"
    
    def test_quote_builder(self):
        """Test quote builder function."""
        q = quote(sym("hello"))
        assert isinstance(q, Quote)
    
    def test_quasiquote_builder(self):
        """Test quasiquote builder function."""
        qq = quasiquote(sexpr("list", unquote(sym("x"))))
        assert isinstance(qq, Quasiquote)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_module(self):
        """Test empty module validation."""
        ext = SymbolicIRExtension()
        ir = {"module": "empty"}
        assert ext.validate(ir) is True
        assert ext.has_symbolic_features(ir) is False
    
    def test_deeply_nested_symbols(self):
        """Test deeply nested symbolic expressions."""
        ext = SymbolicIRExtension()
        ir = {
            "module": "test",
            "functions": [{
                "name": "nested",
                "body": [{
                    "kind": "quote",
                    "value": {
                        "kind": "list",
                        "elements": [{
                            "kind": "list",
                            "elements": [{
                                "kind": "list",
                                "elements": [{"kind": "symbol", "name": "deep"}]
                            }]
                        }]
                    }
                }]
            }]
        }
        assert ext.validate(ir) is True
        assert ext.has_symbolic_features(ir) is True
    
    def test_cons_cell(self):
        """Test cons cell handling."""
        ext = SymbolicIRExtension()
        ir = {
            "module": "test",
            "functions": [{
                "name": "pair",
                "body": [{
                    "kind": "cons",
                    "car": {"kind": "symbol", "name": "a"},
                    "cdr": {"kind": "symbol", "name": "b"}
                }]
            }]
        }
        assert ext.validate(ir) is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
