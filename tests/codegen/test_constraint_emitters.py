"""Tests for Constraint Programming emitters.

This module tests the constraint emitters including:
- MiniZinc emitter
- CHR emitter
"""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ir.constraints import (
    ConstraintModel, Variable, ArrayVariable, IndexSet, Domain, Parameter,
    RelationalConstraint, LogicalConstraint, GlobalConstraint,
    Objective, VariableType, ConstraintType, ObjectiveType,
    VariableRef, Literal, BinaryOp,
    eq, ne, lt, le, gt, ge, alldifferent, conjunction,
)

from targets.constraints import MiniZincEmitter, CHREmitter
from targets.constraints.chr_emitter import (
    emit_simplification_rule,
    emit_propagation_rule,
    emit_simpagation_rule
)


class TestMiniZincEmitter:
    """Tests for MiniZinc emitter."""
    
    @pytest.fixture
    def emitter(self):
        return MiniZincEmitter()
    
    def test_simple_model(self, emitter):
        """Test simple model emission."""
        model = ConstraintModel("simple")
        model.add_int_variable("x", 1, 10)
        model.add_int_variable("y", 1, 10)
        model.add_constraint(eq(VariableRef("x"), VariableRef("y")))
        model.set_objective(Objective.satisfy())
        
        result = emitter.emit(model)
        
        assert "% Model: simple" in result.code
        assert "var 1..10: x;" in result.code
        assert "var 1..10: y;" in result.code
        assert "constraint x = y;" in result.code
        assert "solve satisfy;" in result.code
        
        # Check manifest
        assert result.manifest["model_name"] == "simple"
        assert result.manifest["statistics"]["variables"] == 2
        assert result.manifest["statistics"]["constraints"] == 1
    
    def test_arithmetic_constraints(self, emitter):
        """Test arithmetic constraint emission."""
        model = ConstraintModel("arithmetic")
        model.add_int_variable("x", 1, 10)
        model.add_int_variable("y", 1, 10)
        
        model.add_constraint(ne(VariableRef("x"), VariableRef("y")))
        model.add_constraint(lt(VariableRef("x"), Literal(5)))
        model.add_constraint(ge(VariableRef("y"), Literal(3)))
        
        result = emitter.emit(model)
        
        assert "x != y" in result.code
        assert "x < 5" in result.code
        assert "y >= 3" in result.code
    
    def test_alldifferent_constraint(self, emitter):
        """Test alldifferent constraint emission."""
        model = ConstraintModel("alldiff")
        for i in range(4):
            model.add_int_variable(f"x{i}", 1, 4)
        
        vars_list = [VariableRef(f"x{i}") for i in range(4)]
        model.add_constraint(alldifferent(vars_list))
        
        result = emitter.emit(model)
        
        assert 'include "alldifferent.mzn"' in result.code
        assert "alldifferent([x0, x1, x2, x3])" in result.code
    
    def test_array_variable(self, emitter):
        """Test array variable emission."""
        model = ConstraintModel("array_test")
        model.add_int_array("arr", 5, 1, 10)
        
        result = emitter.emit(model)
        
        assert "array[1..5] of var 1..10: arr;" in result.code
    
    def test_minimize_objective(self, emitter):
        """Test minimize objective emission."""
        model = ConstraintModel("minimize")
        model.add_int_variable("cost", 0, 100)
        model.minimize(VariableRef("cost"))
        
        result = emitter.emit(model)
        
        assert "solve minimize cost;" in result.code
    
    def test_maximize_objective(self, emitter):
        """Test maximize objective emission."""
        model = ConstraintModel("maximize")
        model.add_int_variable("profit", 0, 100)
        model.maximize(VariableRef("profit"))
        
        result = emitter.emit(model)
        
        assert "solve maximize profit;" in result.code
    
    def test_bool_domain(self, emitter):
        """Test boolean domain emission."""
        model = ConstraintModel("bool_test")
        model.add_bool_variable("flag")
        
        result = emitter.emit(model)
        
        assert "var bool: flag;" in result.code
    
    def test_parameters(self, emitter):
        """Test parameter emission."""
        model = ConstraintModel("params")
        model.add_parameter("n", 8, VariableType.INT)
        model.add_int_variable("x", 1, 8)
        
        result = emitter.emit(model)
        
        assert "int: n = 8;" in result.code
    
    def test_complex_expression(self, emitter):
        """Test complex expression emission."""
        model = ConstraintModel("complex")
        model.add_int_variable("x", 1, 10)
        model.add_int_variable("y", 1, 10)
        
        # x + 5 = y
        left = BinaryOp("+", VariableRef("x"), Literal(5))
        model.add_constraint(RelationalConstraint(
            ConstraintType.EQ, 
            left=left, 
            right=VariableRef("y")
        ))
        
        result = emitter.emit(model)
        
        assert "(x + 5) = y" in result.code
    
    def test_nqueens_model(self, emitter):
        """Test N-Queens model emission."""
        n = 4
        model = ConstraintModel("nqueens")
        
        for i in range(1, n + 1):
            model.add_int_variable(f"q{i}", 1, n)
        
        vars_list = [VariableRef(f"q{i}") for i in range(1, n + 1)]
        model.add_constraint(alldifferent(vars_list))
        
        model.set_objective(Objective.satisfy())
        
        result = emitter.emit(model)
        
        assert "% Model: nqueens" in result.code
        assert "alldifferent" in result.code
        assert "solve satisfy;" in result.code
        assert result.manifest["statistics"]["variables"] == n
    
    def test_manifest_generation(self, emitter):
        """Test manifest generation."""
        model = ConstraintModel("manifest_test")
        model.add_int_variable("x", 1, 10)
        
        result = emitter.emit(model)
        
        assert "schema" in result.manifest
        assert "minizinc" in result.manifest["schema"]
        assert "hash" in result.manifest["output"]
        assert "size" in result.manifest["output"]


class TestCHREmitter:
    """Tests for CHR emitter."""
    
    @pytest.fixture
    def emitter(self):
        return CHREmitter()
    
    def test_simple_model(self, emitter):
        """Test simple model emission."""
        model = ConstraintModel("simple")
        model.add_int_variable("x", 1, 10)
        model.add_int_variable("y", 1, 10)
        model.add_constraint(eq(VariableRef("x"), VariableRef("y")))
        
        result = emitter.emit(model)
        
        assert "%% Model: simple" in result.code
        assert ":- use_module(library(chr))." in result.code
        assert ":- chr_constraint" in result.code
        assert "solve :-" in result.code
    
    def test_domain_rules(self, emitter):
        """Test domain rules emission."""
        model = ConstraintModel("domain")
        model.add_int_variable("x", 1, 5)
        
        result = emitter.emit(model)
        
        assert "domain(X, D)" in result.code
        assert "make_domain" in result.code
    
    def test_alldifferent_rules(self, emitter):
        """Test alldifferent rules emission."""
        model = ConstraintModel("alldiff")
        for i in range(3):
            model.add_int_variable(f"x{i}", 1, 3)
        
        vars_list = [VariableRef(f"x{i}") for i in range(3)]
        model.add_constraint(alldifferent(vars_list))
        
        result = emitter.emit(model)
        
        assert "alldifferent([]) <=> true." in result.code
        assert "exclude" in result.code
    
    def test_manifest_generation(self, emitter):
        """Test manifest generation."""
        model = ConstraintModel("chr_manifest")
        model.add_int_variable("x", 1, 10)
        
        result = emitter.emit(model)
        
        assert "schema" in result.manifest
        assert "chr" in result.manifest["schema"]
        assert result.manifest["model_name"] == "chr_manifest"


class TestCHRRuleHelpers:
    """Tests for CHR rule helper functions."""
    
    def test_simplification_rule(self):
        """Test simplification rule generation."""
        rule = emit_simplification_rule(
            "test", 
            "eq(X, Y)", 
            "ground(Y)", 
            "X = Y"
        )
        assert rule == "eq(X, Y) <=> ground(Y) | X = Y."
    
    def test_simplification_rule_no_guard(self):
        """Test simplification rule without guard."""
        rule = emit_simplification_rule(
            "test",
            "eq(X, X)",
            "",
            "true"
        )
        assert rule == "eq(X, X) <=> true."
    
    def test_propagation_rule(self):
        """Test propagation rule generation."""
        rule = emit_propagation_rule(
            "test",
            "lt(X, Y)",
            "",
            "neq(X, Y)"
        )
        assert rule == "lt(X, Y) ==> neq(X, Y)."
    
    def test_simpagation_rule(self):
        """Test simpagation rule generation."""
        rule = emit_simpagation_rule(
            "domain(X, D1)",
            "domain(X, D2)",
            "intersection(D1, D2, D)",
            "domain(X, D)"
        )
        assert "domain(X, D1) \\ domain(X, D2) <=>" in rule


class TestExampleProblems:
    """Test complete example problems."""
    
    def test_send_more_money_minizinc(self):
        """Test SEND+MORE=MONEY puzzle in MiniZinc."""
        model = ConstraintModel("send_more_money")
        
        # Variables for each digit
        for letter in ['S', 'E', 'N', 'D', 'M', 'O', 'R', 'Y']:
            model.add_int_variable(letter, 0, 9)
        
        # All different
        vars_list = [VariableRef(c) for c in ['S', 'E', 'N', 'D', 'M', 'O', 'R', 'Y']]
        model.add_constraint(alldifferent(vars_list))
        
        # Leading digits non-zero
        model.add_constraint(gt(VariableRef('S'), Literal(0)))
        model.add_constraint(gt(VariableRef('M'), Literal(0)))
        
        emitter = MiniZincEmitter()
        result = emitter.emit(model)
        
        assert "alldifferent" in result.code
        assert "S > 0" in result.code
        assert "M > 0" in result.code
    
    def test_sudoku_structure(self):
        """Test Sudoku puzzle structure in MiniZinc."""
        model = ConstraintModel("sudoku")
        
        # 9x9 grid with values 1-9
        for i in range(9):
            for j in range(9):
                model.add_int_variable(f"cell_{i}_{j}", 1, 9)
        
        # Row constraints
        for i in range(9):
            row_vars = [VariableRef(f"cell_{i}_{j}") for j in range(9)]
            model.add_constraint(alldifferent(row_vars))
        
        # Column constraints
        for j in range(9):
            col_vars = [VariableRef(f"cell_{i}_{j}") for i in range(9)]
            model.add_constraint(alldifferent(col_vars))
        
        emitter = MiniZincEmitter()
        result = emitter.emit(model)
        
        assert result.manifest["statistics"]["variables"] == 81
        assert result.manifest["statistics"]["constraints"] == 18  # 9 rows + 9 cols


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
