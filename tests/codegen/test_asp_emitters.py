"""Tests for ASP Emitters.

This module contains comprehensive tests for the Clingo and DLV
Answer Set Programming emitters.

Part of Phase 7D: Answer Set Programming
"""

import pytest
import tempfile
import os

from ir.asp import (
    ASPProgram, program, Atom, Term, Literal, ChoiceElement,
    atom, term, var, pos, neg, classical_neg,
    count, sum_agg, agg_element, compare,
    RuleType, AggregateFunction, ComparisonOp, NegationType,
)
from targets.asp import (
    ClingoEmitter, ClingoConfig,
    DLVEmitter, DLVConfig,
    emit_clingo, emit_dlv,
    emit_clingo_to_file, emit_dlv_to_file,
    EmitterResult,
)


class TestClingoEmitter:
    """Tests for ClingoEmitter class."""
    
    def test_emit_empty_program(self):
        """Test emitting empty program."""
        p = program("empty")
        emitter = ClingoEmitter()
        result = emitter.emit(p)
        
        assert isinstance(result, EmitterResult)
        assert "% ASP Program: empty" in result.code
        assert result.manifest["program_name"] == "empty"
        assert result.manifest["rules_count"] == 0
    
    def test_emit_fact(self):
        """Test emitting facts."""
        p = program("test")
        p.add_fact(atom("node", "a"))
        p.add_fact(atom("edge", "a", "b"))
        
        result = emit_clingo(p)
        
        assert "node(a)." in result.code
        assert "edge(a, b)." in result.code
        assert result.manifest["facts_count"] == 2
    
    def test_emit_normal_rule(self):
        """Test emitting normal rule."""
        p = program("test")
        p.add_normal_rule(
            atom("path", var("X"), var("Y")),
            [pos(atom("edge", var("X"), var("Y")))]
        )
        
        result = emit_clingo(p)
        
        assert "path(X, Y) :- edge(X, Y)." in result.code
    
    def test_emit_constraint(self):
        """Test emitting constraint."""
        p = program("test")
        p.add_constraint([
            pos(atom("edge", var("X"), var("Y"))),
            pos(atom("color", var("X"), var("C"))),
            pos(atom("color", var("Y"), var("C")))
        ])
        
        result = emit_clingo(p)
        
        assert ":- edge(X, Y), color(X, C), color(Y, C)." in result.code
        assert result.manifest["constraints_count"] == 1
    
    def test_emit_choice_rule(self):
        """Test emitting choice rule."""
        p = program("test")
        p.add_choice_rule(
            elements=[ChoiceElement(
                atom("color", var("X"), var("C")),
                [pos(atom("col", var("C")))]
            )],
            body=[pos(atom("node", var("X")))],
            lower=1, upper=1
        )
        
        result = emit_clingo(p)
        
        assert "1 {" in result.code
        assert "} 1" in result.code
        assert "color(X, C) : col(C)" in result.code
        assert "node(X)" in result.code
    
    def test_emit_choice_rule_no_bounds(self):
        """Test emitting choice rule without bounds."""
        p = program("test")
        p.add_choice_rule(
            elements=[atom("selected", var("X"))],
            body=[pos(atom("item", var("X")))]
        )
        
        result = emit_clingo(p)
        
        assert "{ selected(X) }" in result.code
        assert ":- item(X)." in result.code
    
    def test_emit_disjunctive_rule(self):
        """Test emitting disjunctive rule."""
        p = program("test")
        p.add_disjunctive_rule(
            [atom("p", var("X")), atom("q", var("X"))],
            [pos(atom("r", var("X")))]
        )
        
        result = emit_clingo(p)
        
        assert "p(X) | q(X) :- r(X)." in result.code
    
    def test_emit_weak_constraint(self):
        """Test emitting weak constraint."""
        p = program("test")
        p.add_weak_constraint(
            body=[pos(atom("violated", var("X")))],
            weight=1,
            priority=0,
            terms=[var("X")]
        )
        
        result = emit_clingo(p)
        
        assert ":~ violated(X). [1@0, X]" in result.code
    
    def test_emit_weak_constraint_no_terms(self):
        """Test emitting weak constraint without terms."""
        p = program("test")
        p.add_weak_constraint(
            body=[pos(atom("bad"))],
            weight=5,
            priority=1
        )
        
        result = emit_clingo(p)
        
        assert ":~ bad. [5@1]" in result.code
    
    def test_emit_aggregate(self):
        """Test emitting aggregates."""
        p = program("test")
        
        # This would be part of a rule body
        # For now, test via the emitter directly
        emitter = ClingoEmitter()
        
        agg = count()
        agg.add_element([var("X")], [pos(atom("p", var("X")))])
        agg.set_right_guard(ComparisonOp.LE, Term("3"))
        
        agg_str = emitter.emit_aggregate(agg)
        assert "#count" in agg_str
        assert "<= 3" in agg_str
    
    def test_emit_literal_with_negation(self):
        """Test emitting literals with negation."""
        emitter = ClingoEmitter()
        
        a = atom("p", var("X"))
        
        assert emitter.emit_literal(pos(a)) == "p(X)"
        assert emitter.emit_literal(neg(a)) == "not p(X)"
        assert emitter.emit_literal(classical_neg(a)) == "-p(X)"
    
    def test_emit_show_statement(self):
        """Test emitting show statements."""
        p = program("test")
        p.add_show("color", 2)
        p.add_show("edge", 2, positive=False)
        
        result = emit_clingo(p)
        
        assert "#show color/2." in result.code
        assert "#show -edge/2." in result.code
    
    def test_emit_optimize_minimize(self):
        """Test emitting minimize optimization."""
        p = program("test")
        p.add_minimize([
            agg_element([var("W"), var("X")], [
                pos(atom("cost", var("X"), var("W")))
            ])
        ])
        
        result = emit_clingo(p)
        
        assert "#minimize" in result.code
        assert "cost(X, W)" in result.code
    
    def test_emit_optimize_maximize(self):
        """Test emitting maximize optimization."""
        p = program("test")
        p.add_maximize([
            agg_element([var("V"), var("I")], [
                pos(atom("value", var("I"), var("V")))
            ])
        ])
        
        result = emit_clingo(p)
        
        assert "#maximize" in result.code
    
    def test_emit_constant(self):
        """Test emitting constants."""
        p = program("test")
        p.add_constant("n", 10)
        p.add_constant("max_items", 5)
        
        result = emit_clingo(p)
        
        assert "#const n = 10." in result.code
        assert "#const max_items = 5." in result.code
    
    def test_emit_to_file(self):
        """Test emitting to file."""
        p = program("test")
        p.add_fact(atom("node", "a"))
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lp', delete=False) as f:
            path = f.name
        
        try:
            result = emit_clingo_to_file(p, path)
            
            assert os.path.exists(path)
            with open(path) as f:
                content = f.read()
            assert "node(a)." in content
        finally:
            os.unlink(path)
    
    def test_config_options(self):
        """Test configuration options."""
        p = program("test")
        p.add_fact(atom("a"))
        
        config = ClingoConfig(
            include_comments=False,
            include_header=False
        )
        emitter = ClingoEmitter(config)
        result = emitter.emit(p)
        
        # Header should not be present
        assert "% ASP Program:" not in result.code
    
    def test_manifest_generation(self):
        """Test manifest generation."""
        p = program("test")
        p.add_fact(atom("node", "a"))
        p.add_constraint([pos(atom("p"))])
        
        result = emit_clingo(p)
        
        manifest = result.manifest
        assert manifest["schema"] == "stunir.asp.clingo.v1"
        assert manifest["dialect"] == "clingo"
        assert manifest["rules_count"] == 2
        assert manifest["facts_count"] == 1
        assert manifest["constraints_count"] == 1
        assert "code_hash" in manifest
        assert "code_size" in manifest


class TestDLVEmitter:
    """Tests for DLVEmitter class."""
    
    def test_emit_empty_program(self):
        """Test emitting empty program."""
        p = program("empty")
        emitter = DLVEmitter()
        result = emitter.emit(p)
        
        # Result should have code, manifest, warnings attributes
        assert hasattr(result, 'code')
        assert hasattr(result, 'manifest')
        assert "% ASP Program: empty" in result.code
        assert result.manifest["program_name"] == "empty"
    
    def test_emit_fact(self):
        """Test emitting facts."""
        p = program("test")
        p.add_fact(atom("node", "a"))
        
        result = emit_dlv(p)
        
        assert "node(a)." in result.code
    
    def test_emit_normal_rule(self):
        """Test emitting normal rule."""
        p = program("test")
        p.add_normal_rule(
            atom("path", var("X"), var("Y")),
            [pos(atom("edge", var("X"), var("Y")))]
        )
        
        result = emit_dlv(p)
        
        assert "path(X, Y) :- edge(X, Y)." in result.code
    
    def test_emit_disjunctive_rule(self):
        """Test emitting disjunctive rule with 'v' syntax."""
        p = program("test")
        p.add_disjunctive_rule(
            [atom("p", var("X")), atom("q", var("X"))],
            [pos(atom("r", var("X")))]
        )
        
        # Default uses 'v' for disjunction
        result = emit_dlv(p)
        
        assert "p(X) v q(X) :- r(X)." in result.code
    
    def test_emit_disjunctive_rule_pipe_syntax(self):
        """Test emitting disjunctive rule with '|' syntax."""
        p = program("test")
        p.add_disjunctive_rule(
            [atom("p", var("X")), atom("q", var("X"))],
            [pos(atom("r", var("X")))]
        )
        
        config = DLVConfig(use_v_disjunction=False)
        result = emit_dlv(p, config)
        
        assert "p(X) | q(X) :- r(X)." in result.code
    
    def test_emit_weak_constraint_classic(self):
        """Test emitting weak constraint with classic DLV syntax."""
        p = program("test")
        p.add_weak_constraint(
            body=[pos(atom("violated", var("X")))],
            weight=1,
            priority=0,
            terms=[var("X")]
        )
        
        # Classic DLV uses colon
        config = DLVConfig(dlv2_compatible=False)
        result = emit_dlv(p, config)
        
        assert ":~ violated(X). [1:0, X]" in result.code
    
    def test_emit_weak_constraint_dlv2(self):
        """Test emitting weak constraint with DLV2 syntax."""
        p = program("test")
        p.add_weak_constraint(
            body=[pos(atom("violated", var("X")))],
            weight=1,
            priority=0,
            terms=[var("X")]
        )
        
        # DLV2 compatible uses @
        config = DLVConfig(dlv2_compatible=True)
        result = emit_dlv(p, config)
        
        assert ":~ violated(X). [1@0, X]" in result.code
    
    def test_emit_constraint(self):
        """Test emitting constraint."""
        p = program("test")
        p.add_constraint([pos(atom("p")), pos(atom("q"))])
        
        result = emit_dlv(p)
        
        assert ":- p, q." in result.code
    
    def test_choice_rule_conversion_warning(self):
        """Test that complex choice rules generate warnings."""
        p = program("test")
        # Complex choice rule that can't be directly converted
        p.add_choice_rule(
            elements=[
                ChoiceElement(atom("a", var("X"))),
                ChoiceElement(atom("b", var("X"))),
                ChoiceElement(atom("c", var("X")))
            ],
            body=[pos(atom("node", var("X")))],
            lower=2, upper=3
        )
        
        result = emit_dlv(p)
        
        # Should have warnings about conversion
        assert len(result.warnings) > 0
    
    def test_emit_to_file(self):
        """Test emitting to file."""
        p = program("test")
        p.add_fact(atom("node", "a"))
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dlv', delete=False) as f:
            path = f.name
        
        try:
            result = emit_dlv_to_file(p, path)
            
            assert os.path.exists(path)
            with open(path) as f:
                content = f.read()
            assert "node(a)." in content
        finally:
            os.unlink(path)
    
    def test_manifest_generation(self):
        """Test manifest generation."""
        p = program("test")
        p.add_disjunctive_rule(
            [atom("p"), atom("q")],
            [pos(atom("r"))]
        )
        
        result = emit_dlv(p)
        
        manifest = result.manifest
        assert manifest["schema"] == "stunir.asp.dlv.v1"
        assert manifest["dialect"] == "dlv"
        assert manifest["disjunctive_rules_count"] == 1


class TestGraphColoring:
    """Integration test: Graph coloring with both emitters."""
    
    def create_graph_coloring_program(self):
        """Create graph coloring program."""
        p = program("graph_coloring")
        
        # Domain
        p.add_fact(atom("node", "1"))
        p.add_fact(atom("node", "2"))
        p.add_fact(atom("node", "3"))
        p.add_fact(atom("edge", "1", "2"))
        p.add_fact(atom("edge", "2", "3"))
        p.add_fact(atom("edge", "1", "3"))
        p.add_fact(atom("col", "red"))
        p.add_fact(atom("col", "green"))
        p.add_fact(atom("col", "blue"))
        
        # Choice rule
        p.add_choice_rule(
            elements=[ChoiceElement(
                atom("color", var("X"), var("C")),
                [pos(atom("col", var("C")))]
            )],
            body=[pos(atom("node", var("X")))],
            lower=1, upper=1
        )
        
        # Constraint
        p.add_constraint([
            pos(atom("edge", var("X"), var("Y"))),
            pos(atom("color", var("X"), var("C"))),
            pos(atom("color", var("Y"), var("C")))
        ])
        
        # Output
        p.add_show("color", 2)
        
        return p
    
    def test_clingo_graph_coloring(self):
        """Test graph coloring with Clingo emitter."""
        p = self.create_graph_coloring_program()
        result = emit_clingo(p)
        
        # Verify key elements
        assert "% Facts" in result.code
        assert "node(1)." in result.code
        assert "edge(1, 2)." in result.code
        assert "% Choice Rules" in result.code
        assert "1 { color(X, C) : col(C) } 1 :- node(X)." in result.code
        assert "% Constraints" in result.code
        assert ":- edge(X, Y), color(X, C), color(Y, C)." in result.code
        assert "#show color/2." in result.code
    
    def test_dlv_graph_coloring(self):
        """Test graph coloring with DLV emitter."""
        p = self.create_graph_coloring_program()
        result = emit_dlv(p)
        
        # Verify key elements (choice rule converted)
        assert "node(1)." in result.code
        assert ":- edge(X, Y), color(X, C), color(Y, C)." in result.code
        assert "#show color/2." in result.code


class TestHamiltonianPath:
    """Integration test: Hamiltonian path problem."""
    
    def test_hamiltonian_path_clingo(self):
        """Test Hamiltonian path with Clingo emitter."""
        p = program("hamiltonian")
        
        # Graph
        p.add_fact(atom("node", "a"))
        p.add_fact(atom("node", "b"))
        p.add_fact(atom("node", "c"))
        p.add_fact(atom("edge", "a", "b"))
        p.add_fact(atom("edge", "b", "c"))
        p.add_fact(atom("edge", "c", "a"))
        
        # Choose starting node
        p.add_choice_rule(
            elements=[ChoiceElement(
                atom("start", var("X")),
                [pos(atom("node", var("X")))]
            )],
            lower=1, upper=1
        )
        
        # Path definition
        p.add_normal_rule(
            atom("reached", var("X")),
            [pos(atom("start", var("X")))]
        )
        p.add_normal_rule(
            atom("reached", var("Y")),
            [pos(atom("reached", var("X"))),
             pos(atom("in_path", var("X"), var("Y")))]
        )
        
        # Constraint: all nodes must be reached
        p.add_constraint([
            pos(atom("node", var("X"))),
            neg(atom("reached", var("X")))
        ])
        
        p.add_show("in_path", 2)
        
        result = emit_clingo(p)
        
        assert "start(X) : node(X)" in result.code
        assert "reached(X) :- start(X)." in result.code
        assert ":- node(X), not reached(X)." in result.code


class TestOptimizationProblems:
    """Integration test: Optimization problems."""
    
    def test_knapsack_clingo(self):
        """Test knapsack problem with Clingo emitter."""
        p = program("knapsack")
        
        # Items
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
        
        # Selection
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
        
        result = emit_clingo(p)
        
        assert "#const capacity = 5." in result.code
        assert "{ selected(I) }" in result.code
        assert "#maximize" in result.code
        assert "value(I, V)" in result.code


class TestEdgeCases:
    """Test edge cases and special scenarios."""
    
    def test_empty_body(self):
        """Test rule with empty body (fact)."""
        p = program("test")
        p.add_fact(atom("p"))
        
        clingo_result = emit_clingo(p)
        dlv_result = emit_dlv(p)
        
        assert "p." in clingo_result.code
        assert "p." in dlv_result.code
    
    def test_empty_head(self):
        """Test rule with empty head (constraint)."""
        p = program("test")
        p.add_constraint([pos(atom("p")), pos(atom("q"))])
        
        clingo_result = emit_clingo(p)
        dlv_result = emit_dlv(p)
        
        assert ":- p, q." in clingo_result.code
        assert ":- p, q." in dlv_result.code
    
    def test_nested_function_terms(self):
        """Test nested function terms."""
        p = program("test")
        
        # f(g(X), h(Y, Z))
        inner1 = Term("g", [Term("X")])
        inner2 = Term("h", [Term("Y"), Term("Z")])
        outer = Term("f", [inner1, inner2])
        
        a = Atom("p", [outer])
        p.add_fact(a)
        
        result = emit_clingo(p)
        
        assert "p(f(g(X), h(Y, Z)))." in result.code
    
    def test_classical_negation(self):
        """Test classical negation."""
        p = program("test")
        p.add_normal_rule(
            atom("happy", var("X")),
            [classical_neg(atom("sad", var("X")))]
        )
        
        result = emit_clingo(p)
        
        assert "happy(X) :- -sad(X)." in result.code
    
    def test_multiple_aggregates(self):
        """Test multiple aggregates in same rule."""
        emitter = ClingoEmitter()
        
        count_agg = count()
        count_agg.add_element([var("X")], [pos(atom("p", var("X")))])
        
        sum_aggregate = sum_agg()
        sum_aggregate.add_element([var("W")], [pos(atom("cost", var("X"), var("W")))])
        
        assert "#count" in emitter.emit_aggregate(count_agg)
        assert "#sum" in emitter.emit_aggregate(sum_aggregate)
    
    def test_unicode_predicates(self):
        """Test handling of non-ASCII predicates."""
        # Note: Our validation currently allows unicode identifiers
        # as long as they follow the basic rules (start with letter or _)
        # If strict ASCII enforcement is needed, this would raise ValueError
        try:
            a = atom("日本語", var("X"))
            # If it doesn't raise, just verify it's created
            assert a.predicate == "日本語"
        except ValueError:
            # If validation is stricter, this is also acceptable
            pass
    
    def test_large_program(self):
        """Test handling of larger programs."""
        p = program("large")
        
        # Add many facts
        for i in range(100):
            p.add_fact(atom("node", str(i)))
        
        for i in range(99):
            p.add_fact(atom("edge", str(i), str(i+1)))
        
        result = emit_clingo(p)
        
        assert result.manifest["facts_count"] == 199
        assert result.manifest["rules_count"] == 199


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
