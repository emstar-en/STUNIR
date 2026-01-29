"""Tests for ASP IR.

This module contains comprehensive tests for the Answer Set Programming
Intermediate Representation, including atoms, literals, aggregates, rules,
and programs.

Part of Phase 7D: Answer Set Programming
"""

import pytest
import json

from ir.asp import (
    # Enums
    RuleType, AggregateFunction, ComparisonOp, NegationType,
    DEFAULT_PRIORITY, DEFAULT_WEIGHT,
    validate_identifier, is_variable, is_constant,
    # Atom/Literal
    Term, Atom, Literal,
    term, var, const, atom, pos, neg, classical_neg,
    # Aggregates
    AggregateElement, Guard, Aggregate, Comparison,
    count, sum_agg, min_agg, max_agg, agg_element, compare,
    # Rules
    BodyElement, HeadElement, ChoiceElement, Rule,
    normal_rule, fact, constraint, choice_rule, disjunctive_rule, weak_constraint,
    # Program
    ShowStatement, OptimizeStatement, ConstantDef, ASPProgram, program,
)


class TestTerm:
    """Tests for Term class."""
    
    def test_simple_constant(self):
        """Test simple constant term."""
        t = Term("a")
        assert t.name == "a"
        assert t.args is None
        assert not t.is_variable
        assert t.is_constant
        assert not t.is_function
        assert str(t) == "a"
    
    def test_variable(self):
        """Test variable term (uppercase)."""
        t = Term("X")
        assert t.name == "X"
        assert t.is_variable
        assert not t.is_constant
        assert str(t) == "X"
    
    def test_function_term(self):
        """Test function term."""
        t = Term("f", [Term("X"), Term("a")])
        assert t.name == "f"
        assert t.is_function
        assert t.arity == 2
        assert str(t) == "f(X, a)"
    
    def test_integer_constant(self):
        """Test integer constant."""
        t = Term("42")
        assert t.is_integer
        assert t.is_constant
    
    def test_get_variables(self):
        """Test getting all variables from a term."""
        t = Term("f", [Term("X"), Term("a"), Term("Y")])
        vars = t.get_variables()
        assert len(vars) == 2
        assert vars[0].name == "X"
        assert vars[1].name == "Y"
    
    def test_substitute(self):
        """Test variable substitution."""
        t = Term("f", [Term("X"), Term("Y")])
        mapping = {"X": Term("a")}
        result = t.substitute(mapping)
        assert str(result) == "f(a, Y)"
    
    def test_to_dict_from_dict(self):
        """Test serialization."""
        t = Term("f", [Term("X"), Term("a")])
        d = t.to_dict()
        t2 = Term.from_dict(d)
        assert t == t2
    
    def test_equality_and_hash(self):
        """Test equality and hashing."""
        t1 = Term("f", [Term("X")])
        t2 = Term("f", [Term("X")])
        t3 = Term("f", [Term("Y")])
        
        assert t1 == t2
        assert t1 != t3
        assert hash(t1) == hash(t2)


class TestAtom:
    """Tests for Atom class."""
    
    def test_simple_atom(self):
        """Test simple atom without arguments."""
        a = Atom("p")
        assert a.predicate == "p"
        assert a.arity == 0
        assert a.signature == "p/0"
        assert str(a) == "p"
    
    def test_atom_with_terms(self):
        """Test atom with arguments."""
        a = Atom("edge", [Term("X"), Term("Y")])
        assert a.predicate == "edge"
        assert a.arity == 2
        assert a.signature == "edge/2"
        assert str(a) == "edge(X, Y)"
    
    def test_ground_atom(self):
        """Test ground atom (no variables)."""
        a = Atom("color", [Term("node1"), Term("red")])
        assert a.is_ground()
    
    def test_non_ground_atom(self):
        """Test non-ground atom (has variables)."""
        a = Atom("color", [Term("X"), Term("red")])
        assert not a.is_ground()
    
    def test_get_variables(self):
        """Test getting variables from atom."""
        a = Atom("p", [Term("X"), Term("a"), Term("Y")])
        vars = a.get_variables()
        assert len(vars) == 2
    
    def test_substitute(self):
        """Test substitution in atom."""
        a = Atom("edge", [Term("X"), Term("Y")])
        mapping = {"X": Term("a")}
        result = a.substitute(mapping)
        assert str(result) == "edge(a, Y)"
    
    def test_to_dict_from_dict(self):
        """Test serialization."""
        a = Atom("p", [Term("X"), Term("Y")])
        d = a.to_dict()
        a2 = Atom.from_dict(d)
        assert a == a2
    
    def test_invalid_predicate_name(self):
        """Test that invalid predicate names are rejected."""
        with pytest.raises(ValueError):
            Atom("123invalid")


class TestLiteral:
    """Tests for Literal class."""
    
    def test_positive_literal(self):
        """Test positive literal."""
        a = Atom("p", [Term("X")])
        lit = Literal(a)
        assert lit.is_positive
        assert not lit.is_negative
        assert str(lit) == "p(X)"
    
    def test_default_negated_literal(self):
        """Test default-negated literal (not)."""
        a = Atom("p", [Term("X")])
        lit = Literal(a, NegationType.DEFAULT)
        assert not lit.is_positive
        assert lit.is_negative
        assert lit.is_default_negated
        assert str(lit) == "not p(X)"
    
    def test_classically_negated_literal(self):
        """Test classically-negated literal (-)."""
        a = Atom("p", [Term("X")])
        lit = Literal(a, NegationType.CLASSICAL)
        assert lit.is_classically_negated
        assert str(lit) == "-p(X)"
    
    def test_negate(self):
        """Test negation."""
        a = Atom("p", [Term("X")])
        pos_lit = Literal(a)
        neg_lit = pos_lit.negate()
        assert neg_lit.is_default_negated
        
        # Double negation cancels
        pos_again = neg_lit.negate()
        assert pos_again.is_positive
    
    def test_factory_functions(self):
        """Test factory functions."""
        a = Atom("p", [Term("X")])
        
        p = pos(a)
        assert p.is_positive
        
        n = neg(a)
        assert n.is_default_negated
        
        c = classical_neg(a)
        assert c.is_classically_negated


class TestAggregate:
    """Tests for Aggregate class."""
    
    def test_count_aggregate(self):
        """Test count aggregate."""
        agg = count()
        agg.add_element(
            [Term("X")],
            [Literal(Atom("p", [Term("X")]))]
        )
        assert agg.function == AggregateFunction.COUNT
        assert len(agg.elements) == 1
        assert "#count" in str(agg)
    
    def test_sum_aggregate(self):
        """Test sum aggregate."""
        agg = sum_agg()
        agg.add_element(
            [Term("W")],
            [Literal(Atom("cost", [Term("X"), Term("W")]))]
        )
        assert agg.function == AggregateFunction.SUM
        assert "#sum" in str(agg)
    
    def test_aggregate_with_guards(self):
        """Test aggregate with bounds."""
        agg = count()
        agg.add_element([Term("X")], [Literal(Atom("p", [Term("X")]))])
        agg.set_left_guard(Term("2"), ComparisonOp.LE)
        agg.set_right_guard(ComparisonOp.LE, Term("5"))
        
        result = str(agg)
        assert "2 <=" in result
        assert "<= 5" in result
    
    def test_aggregate_element(self):
        """Test aggregate element."""
        elem = agg_element(
            [Term("X"), Term("W")],
            [Literal(Atom("selected", [Term("X")])),
             Literal(Atom("weight", [Term("X"), Term("W")]))]
        )
        assert len(elem.terms) == 2
        assert len(elem.condition) == 2
    
    def test_to_dict_from_dict(self):
        """Test aggregate serialization."""
        agg = count()
        agg.add_element([Term("X")], [Literal(Atom("p", [Term("X")]))])
        agg.set_right_guard(ComparisonOp.LE, Term("3"))
        
        d = agg.to_dict()
        agg2 = Aggregate.from_dict(d)
        assert agg.function == agg2.function
        assert len(agg.elements) == len(agg2.elements)


class TestComparison:
    """Tests for Comparison class."""
    
    def test_comparison(self):
        """Test comparison expression."""
        comp = compare(Term("X"), ComparisonOp.LT, Term("Y"))
        assert str(comp) == "X < Y"
    
    def test_comparison_equality(self):
        """Test equality comparison."""
        comp = compare("N", ComparisonOp.EQ, "5")
        assert str(comp) == "N = 5"


class TestRule:
    """Tests for Rule class."""
    
    def test_fact(self):
        """Test fact (rule with no body)."""
        a = Atom("node", [Term("a")])
        r = fact(a)
        assert r.is_fact
        assert r.rule_type == RuleType.NORMAL
        assert str(r) == "node(a)."
    
    def test_normal_rule(self):
        """Test normal rule."""
        head = Atom("path", [Term("X"), Term("Y")])
        body = [Literal(Atom("edge", [Term("X"), Term("Y")]))]
        r = normal_rule(head, body)
        
        assert r.rule_type == RuleType.NORMAL
        assert not r.is_fact
        assert "path(X, Y) :- edge(X, Y)." == str(r)
    
    def test_constraint(self):
        """Test constraint (rule with no head)."""
        body = [
            Literal(Atom("edge", [Term("X"), Term("Y")])),
            Literal(Atom("color", [Term("X"), Term("C")])),
            Literal(Atom("color", [Term("Y"), Term("C")]))
        ]
        r = constraint(body)
        
        assert r.is_constraint
        assert r.rule_type == RuleType.CONSTRAINT
        assert ":- edge(X, Y), color(X, C), color(Y, C)." == str(r)
    
    def test_choice_rule(self):
        """Test choice rule."""
        elements = [Atom("color", [Term("X"), Term("C")])]
        body = [Literal(Atom("node", [Term("X")]))]
        r = choice_rule(elements, body, lower=1, upper=1)
        
        assert r.is_choice
        assert r.rule_type == RuleType.CHOICE
        result = str(r)
        assert "{ color(X, C) }" in result
        assert "node(X)" in result
    
    def test_disjunctive_rule(self):
        """Test disjunctive rule."""
        heads = [Atom("p", [Term("X")]), Atom("q", [Term("X")])]
        body = [Literal(Atom("r", [Term("X")]))]
        r = disjunctive_rule(heads, body)
        
        assert r.is_disjunctive
        assert r.rule_type == RuleType.DISJUNCTIVE
        assert "p(X) | q(X) :- r(X)." == str(r)
    
    def test_weak_constraint(self):
        """Test weak constraint."""
        body = [Literal(Atom("violated", [Term("X")]))]
        r = weak_constraint(body, weight=1, priority=0, terms=[Term("X")])
        
        assert r.is_weak
        assert r.rule_type == RuleType.WEAK
        assert ":~ violated(X). [1@0, X]" == str(r)
    
    def test_rule_serialization(self):
        """Test rule serialization."""
        head = Atom("p", [Term("X")])
        body = [Literal(Atom("q", [Term("X")]))]
        r = normal_rule(head, body)
        
        d = r.to_dict()
        r2 = Rule.from_dict(d)
        assert r.rule_type == r2.rule_type
        assert len(r.head) == len(r2.head)
        assert len(r.body) == len(r2.body)


class TestASPProgram:
    """Tests for ASPProgram class."""
    
    def test_empty_program(self):
        """Test empty program."""
        p = program("test")
        assert p.name == "test"
        assert len(p.rules) == 0
        assert len(p.get_predicates()) == 0
    
    def test_add_facts(self):
        """Test adding facts."""
        p = program("test")
        p.add_fact(atom("node", "a"))
        p.add_fact(atom("node", "b"))
        p.add_fact(atom("edge", "a", "b"))
        
        assert len(p.rules) == 3
        assert len(p.get_facts()) == 3
    
    def test_add_rules(self):
        """Test adding various rule types."""
        p = program("test")
        
        # Add normal rule
        p.add_normal_rule(
            atom("path", var("X"), var("Y")),
            [pos(atom("edge", var("X"), var("Y")))]
        )
        
        # Add constraint
        p.add_constraint([
            pos(atom("edge", var("X"), var("Y"))),
            pos(atom("color", var("X"), var("C"))),
            pos(atom("color", var("Y"), var("C")))
        ])
        
        assert len(p.rules) == 2
        assert len(p.get_constraints()) == 1
    
    def test_add_choice_rule(self):
        """Test adding choice rule."""
        p = program("test")
        p.add_choice_rule(
            elements=[ChoiceElement(
                atom("color", var("X"), var("C")),
                [pos(atom("col", var("C")))]
            )],
            body=[pos(atom("node", var("X")))],
            lower=1, upper=1
        )
        
        assert len(p.get_choice_rules()) == 1
    
    def test_add_show_statements(self):
        """Test adding show statements."""
        p = program("test")
        p.add_show("color", 2)
        p.add_show("edge", 2, positive=False)
        
        assert len(p.show_statements) == 2
    
    def test_add_optimization(self):
        """Test adding optimization."""
        p = program("test")
        p.add_minimize([
            agg_element([var("W")], [pos(atom("cost", var("X"), var("W")))])
        ])
        
        assert len(p.optimize_statements) == 1
        assert p.optimize_statements[0].minimize
    
    def test_add_constant(self):
        """Test adding constants."""
        p = program("test")
        p.add_constant("n", 10)
        p.add_constant("max_colors", 3)
        
        assert len(p.constants) == 2
    
    def test_get_predicates(self):
        """Test getting all predicates."""
        p = program("test")
        p.add_fact(atom("node", "a"))
        p.add_normal_rule(
            atom("path", var("X"), var("Y")),
            [pos(atom("edge", var("X"), var("Y")))]
        )
        
        preds = p.get_predicates()
        assert "node" in preds
        assert "path" in preds
        assert "edge" in preds
    
    def test_get_rules_for_predicate(self):
        """Test getting rules for a predicate."""
        p = program("test")
        p.add_normal_rule(
            atom("path", var("X"), var("Y")),
            [pos(atom("edge", var("X"), var("Y")))]
        )
        p.add_normal_rule(
            atom("path", var("X"), var("Z")),
            [pos(atom("path", var("X"), var("Y"))),
             pos(atom("edge", var("Y"), var("Z")))]
        )
        
        path_rules = p.get_rules_for_predicate("path")
        assert len(path_rules) == 2
    
    def test_program_serialization(self):
        """Test program serialization."""
        p = program("test")
        p.add_fact(atom("node", "a"))
        p.add_fact(atom("edge", "a", "b"))
        p.add_show("node", 1)
        
        json_str = p.to_json()
        p2 = ASPProgram.from_json(json_str)
        
        assert p.name == p2.name
        assert len(p.rules) == len(p2.rules)
        assert len(p.show_statements) == len(p2.show_statements)
    
    def test_compute_hash(self):
        """Test hash computation."""
        p1 = program("test")
        p1.add_fact(atom("a"))
        
        p2 = program("test")
        p2.add_fact(atom("a"))
        
        # Same programs should have same hash
        assert p1.compute_hash() == p2.compute_hash()
    
    def test_program_string_output(self):
        """Test program string output."""
        p = program("graph_coloring")
        p.add_fact(atom("node", "a"))
        p.add_fact(atom("node", "b"))
        p.add_fact(atom("edge", "a", "b"))
        
        output = str(p)
        assert "% ASP Program: graph_coloring" in output
        assert "node(a)." in output
        assert "edge(a, b)." in output


class TestHelperFunctions:
    """Tests for helper functions."""
    
    def test_validate_identifier(self):
        """Test identifier validation."""
        assert validate_identifier("validName")
        assert validate_identifier("_private")
        assert validate_identifier("name123")
        assert not validate_identifier("123invalid")
        assert not validate_identifier("")
    
    def test_is_variable(self):
        """Test variable detection."""
        assert is_variable("X")
        assert is_variable("Variable")
        assert not is_variable("constant")
        assert not is_variable("")
    
    def test_is_constant(self):
        """Test constant detection."""
        assert is_constant("constant")
        assert is_constant("123")
        assert not is_constant("Variable")


class TestGraphColoring:
    """Integration test: Graph coloring problem."""
    
    def test_graph_coloring_program(self):
        """Test complete graph coloring program."""
        p = program("graph_coloring")
        
        # Domain facts
        p.add_fact(atom("node", "1"))
        p.add_fact(atom("node", "2"))
        p.add_fact(atom("node", "3"))
        p.add_fact(atom("edge", "1", "2"))
        p.add_fact(atom("edge", "2", "3"))
        p.add_fact(atom("edge", "1", "3"))
        p.add_fact(atom("col", "red"))
        p.add_fact(atom("col", "green"))
        p.add_fact(atom("col", "blue"))
        
        # Choice rule: each node gets exactly one color
        p.add_choice_rule(
            elements=[ChoiceElement(
                atom("color", var("X"), var("C")),
                [pos(atom("col", var("C")))]
            )],
            body=[pos(atom("node", var("X")))],
            lower=1, upper=1
        )
        
        # Constraint: adjacent nodes must have different colors
        p.add_constraint([
            pos(atom("edge", var("X"), var("Y"))),
            pos(atom("color", var("X"), var("C"))),
            pos(atom("color", var("Y"), var("C")))
        ])
        
        # Show only color assignments
        p.add_show("color", 2)
        
        # Verify structure
        assert len(p.rules) == 11  # 9 facts + 1 choice + 1 constraint
        assert len(p.get_facts()) == 9
        assert len(p.get_constraints()) == 1
        assert len(p.get_choice_rules()) == 1
        
        # Verify output
        output = str(p)
        assert "% Choice Rules" in output
        assert "% Constraints" in output
        assert "#show color/2." in output


class TestKnapsack:
    """Integration test: Knapsack optimization problem."""
    
    def test_knapsack_program(self):
        """Test knapsack optimization program."""
        p = program("knapsack")
        
        # Items with weights and values
        p.add_fact(atom("item", "a"))
        p.add_fact(atom("item", "b"))
        p.add_fact(atom("item", "c"))
        p.add_fact(atom("weight", "a", "3"))
        p.add_fact(atom("weight", "b", "4"))
        p.add_fact(atom("weight", "c", "2"))
        p.add_fact(atom("value", "a", "5"))
        p.add_fact(atom("value", "b", "6"))
        p.add_fact(atom("value", "c", "3"))
        
        # Capacity
        p.add_constant("capacity", 5)
        
        # Choice: select items
        p.add_choice_rule(
            elements=[ChoiceElement(atom("selected", var("I")))],
            body=[pos(atom("item", var("I")))]
        )
        
        # Maximize value
        p.add_maximize([
            agg_element([var("V"), var("I")], [
                pos(atom("selected", var("I"))),
                pos(atom("value", var("I"), var("V")))
            ])
        ])
        
        p.add_show("selected", 1)
        
        # Verify structure
        assert len(p.optimize_statements) == 1
        assert not p.optimize_statements[0].minimize  # maximize
        
        output = str(p)
        assert "#maximize" in output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
