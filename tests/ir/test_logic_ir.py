#!/usr/bin/env python3
"""Tests for STUNIR Logic IR Extensions.

Comprehensive test suite for logic programming constructs including
terms, predicates, clauses, and the unification algorithm.

Part of Phase 5C-1: Logic Programming Foundation.
"""

import pytest
import sys
import os

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from tools.ir.logic_ir import (
    # Terms
    Variable, Atom, Number, StringTerm, Compound, ListTerm, Anonymous,
    # Clauses
    Fact, Rule, Goal, Query, Predicate,
    # Substitution and unification
    Substitution, unify, UnificationError,
    # Enums and helpers
    TermKind, GoalKind, LOGIC_KINDS, term_from_dict,
    # Extension
    LogicIRExtension,
)


class TestTermClasses:
    """Tests for term class functionality."""
    
    def test_variable_creation(self):
        """Test Variable creation and properties."""
        x = Variable("X")
        assert x.name == "X"
        assert str(x) == "X"
        assert x.to_dict() == {'kind': 'variable', 'name': 'X'}
    
    def test_variable_equality(self):
        """Test Variable equality."""
        x1 = Variable("X")
        x2 = Variable("X")
        y = Variable("Y")
        
        assert x1 == x2
        assert x1 != y
        assert hash(x1) == hash(x2)
    
    def test_variable_anonymous(self):
        """Test anonymous variable detection."""
        x = Variable("X")
        anon = Variable("_")
        underscore_var = Variable("_Temp")
        
        assert not x.is_anonymous()
        assert anon.is_anonymous()
        assert underscore_var.is_anonymous()
    
    def test_atom_creation(self):
        """Test Atom creation and properties."""
        hello = Atom("hello")
        assert hello.value == "hello"
        assert str(hello) == "hello"
        assert hello.to_dict() == {'kind': 'atom', 'value': 'hello'}
    
    def test_atom_equality(self):
        """Test Atom equality."""
        a1 = Atom("foo")
        a2 = Atom("foo")
        b = Atom("bar")
        
        assert a1 == a2
        assert a1 != b
    
    def test_number_creation(self):
        """Test Number creation."""
        n_int = Number(42)
        n_float = Number(3.14)
        
        assert n_int.value == 42
        assert n_float.value == 3.14
        assert n_int.to_dict() == 42
        assert n_float.to_dict() == 3.14
    
    def test_compound_creation(self):
        """Test Compound term creation."""
        # f(X, a, 42)
        x = Variable("X")
        a = Atom("a")
        n = Number(42)
        
        f = Compound("f", [x, a, n])
        
        assert f.functor == "f"
        assert f.arity == 3
        assert f.args == [x, a, n]
        assert str(f) == "f(X, a, 42)"
    
    def test_compound_nested(self):
        """Test nested compound terms."""
        # f(g(X), Y)
        x = Variable("X")
        y = Variable("Y")
        g = Compound("g", [x])
        f = Compound("f", [g, y])
        
        assert f.arity == 2
        assert f.args[0].functor == "g"
        
        # Get all variables
        vars = f.get_variables()
        assert len(vars) == 2
        assert Variable("X") in vars
        assert Variable("Y") in vars
    
    def test_list_term_proper(self):
        """Test proper list creation."""
        # [a, b, c]
        lst = ListTerm([Atom("a"), Atom("b"), Atom("c")])
        
        assert lst.is_proper()
        assert len(lst.elements) == 3
        assert str(lst) == "[a, b, c]"
    
    def test_list_term_head_tail(self):
        """Test head/tail list pattern."""
        # [H|T]
        h = Variable("H")
        t = Variable("T")
        lst = ListTerm([h], tail=t)
        
        assert not lst.is_proper()
        assert str(lst) == "[H|T]"
    
    def test_list_term_empty(self):
        """Test empty list."""
        empty = ListTerm([])
        assert empty.is_proper()
        assert str(empty) == "[]"
    
    def test_anonymous_uniqueness(self):
        """Test that anonymous variables are unique."""
        a1 = Anonymous()
        a2 = Anonymous()
        
        # Each anonymous variable is unique
        assert a1 != a2
        assert hash(a1) != hash(a2)


class TestSubstitution:
    """Tests for Substitution class."""
    
    def test_empty_substitution(self):
        """Test empty substitution."""
        subst = Substitution()
        assert subst.is_empty()
        assert len(subst) == 0
    
    def test_bind(self):
        """Test binding variables."""
        x = Variable("X")
        a = Atom("hello")
        
        subst = Substitution()
        subst2 = subst.bind(x, a)
        
        # Original unchanged
        assert subst.is_empty()
        # New has binding
        assert subst2.get(x) == a
        assert x in subst2
    
    def test_apply_substitution(self):
        """Test applying substitution to terms."""
        x = Variable("X")
        y = Variable("Y")
        a = Atom("hello")
        
        subst = Substitution({x: a})
        
        # Variable gets replaced
        result = x.apply_substitution(subst)
        assert result == a
        
        # Unbound variable unchanged
        result2 = y.apply_substitution(subst)
        assert result2 == y
    
    def test_compose_substitutions(self):
        """Test composing substitutions."""
        x = Variable("X")
        y = Variable("Y")
        a = Atom("a")
        
        # {X -> Y} ∘ {Y -> a} should give {X -> a, Y -> a}
        subst1 = Substitution({x: y})
        subst2 = Substitution({y: a})
        
        composed = subst1.compose(subst2)
        
        # X should map to a (through Y)
        assert composed.get(x) == a
        assert composed.get(y) == a


class TestUnification:
    """Tests for unification algorithm."""
    
    def test_variable_atom_unification(self):
        """TC-LOG-001: Test basic variable unification."""
        x = Variable("X")
        a = Atom("hello")
        
        subst = unify(x, a)
        assert subst.get(x) == a
        
        # Apply substitution
        result = x.apply_substitution(subst)
        assert result == a
    
    def test_variable_variable_unification(self):
        """Test unifying two variables."""
        x = Variable("X")
        y = Variable("Y")
        
        subst = unify(x, y)
        
        # One should be bound to the other
        assert subst.get(x) == y or subst.get(y) == x
    
    def test_compound_unification(self):
        """TC-LOG-002: Test compound term unification."""
        # f(X, b) unifies with f(a, Y)
        x = Variable("X")
        y = Variable("Y")
        
        t1 = Compound("f", [x, Atom("b")])
        t2 = Compound("f", [Atom("a"), y])
        
        subst = unify(t1, t2)
        
        assert subst.get(x) == Atom("a")
        assert subst.get(y) == Atom("b")
    
    def test_nested_compound_unification(self):
        """Test unifying nested compounds."""
        # f(g(X), Y) unifies with f(g(1), h(Z))
        x = Variable("X")
        y = Variable("Y")
        z = Variable("Z")
        
        t1 = Compound("f", [Compound("g", [x]), y])
        t2 = Compound("f", [Compound("g", [Number(1)]), Compound("h", [z])])
        
        subst = unify(t1, t2)
        
        assert subst.get(x) == Number(1)
        # Y binds to h(Z)
        y_val = subst.get(y)
        assert isinstance(y_val, Compound)
        assert y_val.functor == "h"
    
    def test_occurs_check(self):
        """TC-LOG-003: Test occurs check prevents infinite terms."""
        x = Variable("X")
        # X cannot unify with f(X) - would create infinite term
        t = Compound("f", [x])
        
        with pytest.raises(UnificationError):
            unify(x, t, occurs_check=True)
    
    def test_occurs_check_disabled(self):
        """Test that occurs check can be disabled."""
        x = Variable("X")
        t = Compound("f", [x])
        
        # Without occurs check, it "succeeds" (unsound but sometimes useful)
        subst = unify(x, t, occurs_check=False)
        assert subst.get(x) == t
    
    def test_list_unification_simple(self):
        """Test simple list unification."""
        # [a, b] unifies with [a, b]
        lst1 = ListTerm([Atom("a"), Atom("b")])
        lst2 = ListTerm([Atom("a"), Atom("b")])
        
        subst = unify(lst1, lst2)
        assert subst.is_empty()  # No bindings needed
    
    def test_list_unification_with_variables(self):
        """TC-LOG-004: Test list unification with head/tail."""
        h = Variable("H")
        t = Variable("T")
        
        # [H|T] unifies with [1, 2, 3]
        pattern = ListTerm([h], tail=t)
        list_term = ListTerm([Number(1), Number(2), Number(3)])
        
        subst = unify(pattern, list_term)
        
        # H should be 1
        assert subst.get(h) == Number(1)
        # T should be [2, 3]
        t_val = subst.get(t)
        assert isinstance(t_val, ListTerm)
        assert len(t_val.elements) == 2
    
    def test_unification_failure_functor(self):
        """Test unification fails on different functors."""
        t1 = Compound("f", [Atom("a")])
        t2 = Compound("g", [Atom("a")])
        
        with pytest.raises(UnificationError):
            unify(t1, t2)
    
    def test_unification_failure_arity(self):
        """Test unification fails on different arities."""
        t1 = Compound("f", [Atom("a")])
        t2 = Compound("f", [Atom("a"), Atom("b")])
        
        with pytest.raises(UnificationError):
            unify(t1, t2)
    
    def test_unification_failure_atoms(self):
        """Test unification fails on different atoms."""
        a1 = Atom("foo")
        a2 = Atom("bar")
        
        with pytest.raises(UnificationError):
            unify(a1, a2)
    
    def test_anonymous_variable_unification(self):
        """Test anonymous variables always succeed."""
        anon = Anonymous()
        
        # Anonymous unifies with anything
        subst1 = unify(anon, Atom("test"))
        assert subst1.is_empty()
        
        subst2 = unify(anon, Compound("f", [Variable("X")]))
        assert subst2.is_empty()


class TestClauseClasses:
    """Tests for Fact, Rule, Query classes."""
    
    def test_fact_creation(self):
        """Test Fact creation."""
        # parent(tom, bob)
        fact = Fact("parent", [Atom("tom"), Atom("bob")])
        
        assert fact.predicate == "parent"
        assert fact.arity == 2
        assert str(fact) == "parent(tom, bob)."
    
    def test_fact_from_dict(self):
        """Test Fact parsing from dictionary."""
        data = {
            "kind": "fact",
            "predicate": "likes",
            "args": [
                {"kind": "atom", "value": "mary"},
                {"kind": "atom", "value": "food"}
            ]
        }
        
        fact = Fact.from_dict(data)
        assert fact.predicate == "likes"
        assert fact.arity == 2
    
    def test_rule_creation(self):
        """Test Rule creation."""
        # grandparent(X, Z) :- parent(X, Y), parent(Y, Z)
        x = Variable("X")
        y = Variable("Y")
        z = Variable("Z")
        
        head = Compound("grandparent", [x, z])
        body = [
            Goal.call(Compound("parent", [x, y])),
            Goal.call(Compound("parent", [y, z]))
        ]
        
        rule = Rule(head, body)
        
        assert rule.predicate == "grandparent"
        assert rule.arity == 2
        assert len(rule.body) == 2
    
    def test_rule_from_dict(self):
        """TC-LOG-005: Test parsing facts and rules from IR."""
        data = {
            "kind": "rule",
            "head": {
                "kind": "compound",
                "functor": "grandparent",
                "args": [
                    {"kind": "variable", "name": "X"},
                    {"kind": "variable", "name": "Z"}
                ]
            },
            "body": [
                {
                    "kind": "compound",
                    "functor": "parent",
                    "args": [
                        {"kind": "variable", "name": "X"},
                        {"kind": "variable", "name": "Y"}
                    ]
                },
                {
                    "kind": "compound",
                    "functor": "parent",
                    "args": [
                        {"kind": "variable", "name": "Y"},
                        {"kind": "variable", "name": "Z"}
                    ]
                }
            ]
        }
        
        rule = Rule.from_dict(data)
        assert rule.predicate == "grandparent"
        assert len(rule.body) == 2
    
    def test_query_creation(self):
        """Test Query creation."""
        # ?- parent(tom, X)
        goal = Goal.call(Compound("parent", [Atom("tom"), Variable("X")]))
        query = Query([goal])
        
        assert len(query.goals) == 1
    
    def test_goal_cut(self):
        """Test cut goal."""
        cut = Goal.cut()
        assert cut.kind == GoalKind.CUT
    
    def test_goal_negation(self):
        """Test negation goal."""
        inner = Goal.call(Compound("member", [Variable("X"), Variable("L")]))
        neg = Goal.negation(inner)
        
        assert neg.kind == GoalKind.NEGATION
        assert len(neg.goals) == 1
    
    def test_goal_unification(self):
        """Test unification goal."""
        x = Variable("X")
        y = Variable("Y")
        goal = Goal.unification(x, y)
        
        assert goal.kind == GoalKind.UNIFICATION
        assert goal.left == x
        assert goal.right == y


class TestPredicateClass:
    """Tests for Predicate class."""
    
    def test_predicate_creation(self):
        """Test Predicate creation."""
        pred = Predicate("append", 3)
        assert pred.name == "append"
        assert pred.arity == 3
        assert pred.indicator == "append/3"
    
    def test_predicate_add_clauses(self):
        """Test adding clauses to predicate."""
        pred = Predicate("parent", 2)
        
        fact1 = Fact("parent", [Atom("tom"), Atom("bob")])
        fact2 = Fact("parent", [Atom("tom"), Atom("liz")])
        
        pred.add_clause(fact1)
        pred.add_clause(fact2)
        
        assert len(pred.clauses) == 2
    
    def test_predicate_properties(self):
        """Test predicate properties."""
        pred = Predicate("data", 1, is_dynamic=True)
        
        assert pred.is_dynamic
        assert not pred.is_multifile


class TestLogicIRExtension:
    """Tests for LogicIRExtension class."""
    
    def test_has_logic_features(self):
        """Test detecting logic features in IR."""
        ext = LogicIRExtension()
        
        # IR with logic features
        logic_ir = {
            "module": "test",
            "clauses": [
                {"kind": "fact", "predicate": "test", "args": []}
            ]
        }
        
        assert ext.has_logic_features(logic_ir)
        
        # IR without logic features
        plain_ir = {
            "module": "test",
            "functions": []
        }
        
        assert not ext.has_logic_features(plain_ir)
    
    def test_extract_predicates(self):
        """TC-LOG-005: Test extracting predicates from IR."""
        ext = LogicIRExtension()
        
        ir = {
            "module": "family",
            "clauses": [
                {"kind": "fact", "predicate": "parent", "args": [
                    {"kind": "atom", "value": "tom"},
                    {"kind": "atom", "value": "bob"}
                ]},
                {"kind": "rule",
                 "head": {"kind": "compound", "functor": "grandparent", "args": [
                     {"kind": "variable", "name": "X"},
                     {"kind": "variable", "name": "Z"}
                 ]},
                 "body": [
                     {"kind": "compound", "functor": "parent", "args": [
                         {"kind": "variable", "name": "X"},
                         {"kind": "variable", "name": "Y"}
                     ]},
                     {"kind": "compound", "functor": "parent", "args": [
                         {"kind": "variable", "name": "Y"},
                         {"kind": "variable", "name": "Z"}
                     ]}
                 ]}
            ]
        }
        
        predicates = ext.extract_predicates(ir)
        
        assert ("parent", 2) in predicates
        assert ("grandparent", 2) in predicates
        assert len(predicates[("parent", 2)].clauses) == 1
    
    def test_extract_queries(self):
        """TC-LOG-006: Test extracting queries from IR."""
        ext = LogicIRExtension()
        
        ir = {
            "module": "test",
            "clauses": [],
            "queries": [
                {"kind": "query", "goals": [
                    {"kind": "compound", "functor": "grandparent", "args": [
                        {"kind": "atom", "value": "tom"},
                        {"kind": "variable", "name": "X"}
                    ]}
                ]}
            ]
        }
        
        queries = ext.extract_queries(ir)
        
        assert len(queries) == 1
        assert len(queries[0].goals) == 1
    
    def test_validate_valid_ir(self):
        """Test validation of valid IR."""
        ext = LogicIRExtension()
        
        ir = {
            "module": "test",
            "clauses": [
                {"kind": "fact", "predicate": "test", "args": []}
            ]
        }
        
        assert ext.validate(ir) is True
    
    def test_validate_invalid_ir(self):
        """Test validation catches invalid IR."""
        ext = LogicIRExtension()
        
        # Not a dict
        with pytest.raises(ValueError):
            ext.validate("not a dict")
    
    def test_extract_dynamic_predicates(self):
        """Test extracting dynamic predicate declarations."""
        ext = LogicIRExtension()
        
        ir = {
            "module": "test",
            "clauses": [],
            "dynamic": [
                {"predicate": "data", "arity": 2}
            ]
        }
        
        predicates = ext.extract_predicates(ir)
        
        assert ("data", 2) in predicates
        assert predicates[("data", 2)].is_dynamic


class TestTermFromDict:
    """Tests for term_from_dict function."""
    
    def test_parse_variable(self):
        """Test parsing variable from dict."""
        data = {"kind": "variable", "name": "X"}
        term = term_from_dict(data)
        
        assert isinstance(term, Variable)
        assert term.name == "X"
    
    def test_parse_atom(self):
        """Test parsing atom from dict."""
        data = {"kind": "atom", "value": "hello"}
        term = term_from_dict(data)
        
        assert isinstance(term, Atom)
        assert term.value == "hello"
    
    def test_parse_number(self):
        """Test parsing number."""
        # Bare number
        term1 = term_from_dict(42)
        assert isinstance(term1, Number)
        assert term1.value == 42
        
        # Float
        term2 = term_from_dict(3.14)
        assert isinstance(term2, Number)
        assert term2.value == 3.14
    
    def test_parse_compound(self):
        """Test parsing compound term."""
        data = {
            "kind": "compound",
            "functor": "f",
            "args": [
                {"kind": "variable", "name": "X"},
                {"kind": "atom", "value": "a"}
            ]
        }
        term = term_from_dict(data)
        
        assert isinstance(term, Compound)
        assert term.functor == "f"
        assert term.arity == 2
    
    def test_parse_list(self):
        """Test parsing list term."""
        data = {
            "kind": "list_term",
            "elements": [
                {"kind": "atom", "value": "a"},
                {"kind": "atom", "value": "b"}
            ]
        }
        term = term_from_dict(data)
        
        assert isinstance(term, ListTerm)
        assert len(term.elements) == 2
    
    def test_parse_list_with_tail(self):
        """Test parsing list with tail."""
        data = {
            "kind": "list_term",
            "elements": [{"kind": "variable", "name": "H"}],
            "tail": {"kind": "variable", "name": "T"}
        }
        term = term_from_dict(data)
        
        assert isinstance(term, ListTerm)
        assert not term.is_proper()
        assert isinstance(term.tail, Variable)


class TestToDict:
    """Tests for to_dict methods (round-trip)."""
    
    def test_variable_roundtrip(self):
        """Test Variable dict round-trip."""
        original = Variable("X")
        data = original.to_dict()
        restored = term_from_dict(data)
        
        assert original == restored
    
    def test_compound_roundtrip(self):
        """Test Compound dict round-trip."""
        original = Compound("f", [
            Variable("X"),
            Atom("a"),
            Number(42)
        ])
        data = original.to_dict()
        restored = term_from_dict(data)
        
        assert original == restored
    
    def test_list_roundtrip(self):
        """Test ListTerm dict round-trip."""
        original = ListTerm([
            Atom("a"),
            Atom("b"),
            Atom("c")
        ])
        data = original.to_dict()
        restored = term_from_dict(data)
        
        assert original == restored
    
    def test_fact_roundtrip(self):
        """Test Fact dict round-trip."""
        original = Fact("parent", [Atom("tom"), Atom("bob")])
        data = original.to_dict()
        restored = Fact.from_dict(data)
        
        assert original.predicate == restored.predicate
        assert original.arity == restored.arity


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
