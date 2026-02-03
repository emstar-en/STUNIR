"""Tests for Constraint Programming IR.

This module tests the constraint IR classes including:
- Variables and domains
- Constraints (relational, logical, global)
- Objective functions
- Constraint models
"""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ir.constraints import (
    # Enums
    VariableType, DomainType, ConstraintType, ObjectiveType,
    SearchStrategy, ValueChoice,
    # Results and errors
    ConstraintEmitterResult, InvalidDomainError,
    # Variables
    Variable, ArrayVariable, IndexSet, Parameter,
    # Domains
    Domain,
    # Expressions
    VariableRef, Literal, ArrayAccess, BinaryOp, UnaryOp, FunctionCall,
    # Constraints
    RelationalConstraint, LogicalConstraint, GlobalConstraint,
    eq, ne, lt, le, gt, ge, alldifferent, conjunction, disjunction, negation, implies,
    # Model
    Objective, SearchAnnotation, ConstraintModel,
)


class TestDomain:
    """Tests for Domain class."""
    
    def test_int_range_domain(self):
        """Test integer range domain creation."""
        dom = Domain.int_range(1, 10)
        assert dom.domain_type == DomainType.RANGE
        assert dom.lower == 1
        assert dom.upper == 10
        assert dom.size() == 10
    
    def test_float_range_domain(self):
        """Test float range domain creation."""
        dom = Domain.float_range(0.0, 1.0)
        assert dom.domain_type == DomainType.RANGE
        assert dom.lower == 0.0
        assert dom.upper == 1.0
        assert dom.size() is None  # Float ranges are infinite
    
    def test_bool_domain(self):
        """Test boolean domain creation."""
        dom = Domain.bool_domain()
        assert dom.domain_type == DomainType.BOOL
        assert dom.size() == 2
    
    def test_set_domain(self):
        """Test explicit set domain creation."""
        dom = Domain.set_domain({1, 3, 5, 7})
        assert dom.domain_type == DomainType.SET
        assert dom.size() == 4
        assert dom.contains(3)
        assert not dom.contains(2)
    
    def test_unbounded_domain(self):
        """Test unbounded domain creation."""
        dom = Domain.unbounded()
        assert dom.domain_type == DomainType.UNBOUNDED
        assert dom.size() is None
    
    def test_invalid_range(self):
        """Test that invalid range raises error."""
        with pytest.raises(InvalidDomainError):
            Domain.int_range(10, 1)  # lower > upper
    
    def test_empty_set_domain(self):
        """Test that empty set domain raises error."""
        with pytest.raises(InvalidDomainError):
            Domain.set_domain(set())
    
    def test_domain_contains(self):
        """Test domain containment check."""
        dom = Domain.int_range(1, 10)
        assert dom.contains(5)
        assert not dom.contains(0)
        assert not dom.contains(11)
    
    def test_domain_to_minizinc(self):
        """Test MiniZinc syntax generation."""
        dom = Domain.int_range(1, 10)
        assert dom.to_minizinc() == "1..10"
        
        dom = Domain.bool_domain()
        assert dom.to_minizinc() == "bool"


class TestVariable:
    """Tests for Variable class."""
    
    def test_variable_creation(self):
        """Test basic variable creation."""
        var = Variable("x", VariableType.INT, Domain.int_range(1, 10))
        assert var.name == "x"
        assert var.var_type == VariableType.INT
        assert var.domain.size() == 10
    
    def test_variable_equality(self):
        """Test variable equality based on name."""
        var1 = Variable("x", VariableType.INT, Domain.int_range(1, 10))
        var2 = Variable("x", VariableType.INT, Domain.int_range(1, 20))
        var3 = Variable("y", VariableType.INT, Domain.int_range(1, 10))
        
        assert var1 == var2  # Same name
        assert var1 != var3  # Different name
    
    def test_variable_hash(self):
        """Test variable hashing."""
        var = Variable("x", VariableType.INT, Domain.int_range(1, 10))
        var_set = {var}
        assert var in var_set


class TestIndexSet:
    """Tests for IndexSet class."""
    
    def test_index_set_1d(self):
        """Test 1D index set."""
        idx = IndexSet([(1, 10)])
        assert idx.dimensions == 1
        assert idx.size == 10
    
    def test_index_set_2d(self):
        """Test 2D index set."""
        idx = IndexSet([(1, 3), (1, 4)])
        assert idx.dimensions == 2
        assert idx.size == 12
    
    def test_index_set_to_minizinc(self):
        """Test MiniZinc syntax."""
        idx = IndexSet([(1, 10)])
        assert idx.to_minizinc() == "1..10"
        
        idx = IndexSet([(1, 3), (1, 4)])
        assert idx.to_minizinc() == "1..3, 1..4"


class TestArrayVariable:
    """Tests for ArrayVariable class."""
    
    def test_array_variable_creation(self):
        """Test array variable creation."""
        arr = ArrayVariable(
            "queens", 
            VariableType.INT,
            IndexSet([(1, 8)]),
            Domain.int_range(1, 8)
        )
        assert arr.name == "queens"
        assert arr.size == 8
        assert arr.element_domain.size() == 8


class TestExpression:
    """Tests for Expression classes."""
    
    def test_variable_ref(self):
        """Test variable reference."""
        ref = VariableRef("x")
        assert ref.name == "x"
        assert ref.kind == "var"
        assert str(ref) == "x"
    
    def test_literal_int(self):
        """Test integer literal."""
        lit = Literal(42)
        assert lit.value == 42
        assert lit.kind == "literal"
        assert str(lit) == "42"
    
    def test_literal_bool(self):
        """Test boolean literal."""
        lit_true = Literal(True)
        lit_false = Literal(False)
        assert str(lit_true) == "true"
        assert str(lit_false) == "false"
    
    def test_binary_op(self):
        """Test binary operation."""
        left = VariableRef("x")
        right = Literal(5)
        op = BinaryOp("+", left, right)
        assert op.kind == "binary_op"
        assert str(op) == "(x + 5)"
    
    def test_unary_op(self):
        """Test unary operation."""
        operand = VariableRef("x")
        op = UnaryOp("-", operand)
        assert op.kind == "unary_op"
        assert str(op) == "-x"
    
    def test_array_access(self):
        """Test array access."""
        acc = ArrayAccess("arr", [Literal(1), Literal(2)])
        assert acc.kind == "array_access"
        assert str(acc) == "arr[1, 2]"
    
    def test_function_call(self):
        """Test function call."""
        call = FunctionCall("abs", [VariableRef("x")])
        assert call.kind == "call"
        assert str(call) == "abs(x)"


class TestConstraint:
    """Tests for Constraint classes."""
    
    def test_relational_eq(self):
        """Test equality constraint."""
        c = eq(VariableRef("x"), Literal(5))
        assert c.constraint_type == ConstraintType.EQ
        assert str(c) == "x = 5"
    
    def test_relational_ne(self):
        """Test not-equal constraint."""
        c = ne(VariableRef("x"), VariableRef("y"))
        assert c.constraint_type == ConstraintType.NE
        assert str(c) == "x != y"
    
    def test_relational_lt(self):
        """Test less-than constraint."""
        c = lt(VariableRef("x"), VariableRef("y"))
        assert c.constraint_type == ConstraintType.LT
        assert str(c) == "x < y"
    
    def test_alldifferent(self):
        """Test alldifferent constraint."""
        c = alldifferent([VariableRef("x"), VariableRef("y"), VariableRef("z")])
        assert c.constraint_type == ConstraintType.ALLDIFFERENT
        assert "alldifferent" in str(c)
    
    def test_conjunction(self):
        """Test conjunction (AND) constraint."""
        c1 = eq(VariableRef("x"), Literal(5))
        c2 = lt(VariableRef("y"), Literal(10))
        c = conjunction([c1, c2])
        assert c.constraint_type == ConstraintType.AND
    
    def test_disjunction(self):
        """Test disjunction (OR) constraint."""
        c1 = eq(VariableRef("x"), Literal(5))
        c2 = eq(VariableRef("x"), Literal(10))
        c = disjunction([c1, c2])
        assert c.constraint_type == ConstraintType.OR
    
    def test_negation(self):
        """Test negation constraint."""
        c = negation(eq(VariableRef("x"), Literal(5)))
        assert c.constraint_type == ConstraintType.NOT
    
    def test_implies(self):
        """Test implication constraint."""
        c1 = eq(VariableRef("x"), Literal(5))
        c2 = gt(VariableRef("y"), Literal(0))
        c = implies(c1, c2)
        assert c.constraint_type == ConstraintType.IMPLIES


class TestObjective:
    """Tests for Objective class."""
    
    def test_satisfy_objective(self):
        """Test satisfaction objective."""
        obj = Objective.satisfy()
        assert obj.objective_type == ObjectiveType.SATISFY
        assert obj.expression is None
    
    def test_minimize_objective(self):
        """Test minimize objective."""
        obj = Objective.minimize(VariableRef("cost"))
        assert obj.objective_type == ObjectiveType.MINIMIZE
        assert obj.expression is not None
    
    def test_maximize_objective(self):
        """Test maximize objective."""
        obj = Objective.maximize(VariableRef("profit"))
        assert obj.objective_type == ObjectiveType.MAXIMIZE
        assert obj.expression is not None


class TestSearchAnnotation:
    """Tests for SearchAnnotation class."""
    
    def test_search_annotation(self):
        """Test search annotation."""
        ann = SearchAnnotation(
            ["x", "y", "z"],
            SearchStrategy.FIRST_FAIL,
            ValueChoice.INDOMAIN_MIN
        )
        mzn = ann.to_minizinc()
        assert "first_fail" in mzn
        assert "indomain_min" in mzn


class TestConstraintModel:
    """Tests for ConstraintModel class."""
    
    def test_model_creation(self):
        """Test basic model creation."""
        model = ConstraintModel("test")
        assert model.name == "test"
        assert len(model.variables) == 0
        assert len(model.constraints) == 0
    
    def test_add_variable(self):
        """Test adding a variable."""
        model = ConstraintModel("test")
        var = model.add_variable("x", VariableType.INT, Domain.int_range(1, 10))
        assert len(model.variables) == 1
        assert var.name == "x"
    
    def test_add_int_variable(self):
        """Test adding an integer variable."""
        model = ConstraintModel("test")
        var = model.add_int_variable("x", 1, 10)
        assert var.var_type == VariableType.INT
        assert var.domain.lower == 1
        assert var.domain.upper == 10
    
    def test_add_bool_variable(self):
        """Test adding a boolean variable."""
        model = ConstraintModel("test")
        var = model.add_bool_variable("b")
        assert var.var_type == VariableType.BOOL
    
    def test_add_array(self):
        """Test adding an array."""
        model = ConstraintModel("test")
        arr = model.add_int_array("arr", 10, 1, 10)
        assert len(model.arrays) == 1
        assert arr.size == 10
    
    def test_add_constraint(self):
        """Test adding a constraint."""
        model = ConstraintModel("test")
        model.add_int_variable("x", 1, 10)
        model.add_int_variable("y", 1, 10)
        model.add_constraint(eq(VariableRef("x"), VariableRef("y")))
        assert len(model.constraints) == 1
    
    def test_set_objective(self):
        """Test setting objective."""
        model = ConstraintModel("test")
        model.add_int_variable("cost", 0, 100)
        model.minimize(VariableRef("cost"))
        assert model.objective.objective_type == ObjectiveType.MINIMIZE
    
    def test_get_variable(self):
        """Test getting a variable by name."""
        model = ConstraintModel("test")
        model.add_int_variable("x", 1, 10)
        var = model.get_variable("x")
        assert var is not None
        assert var.name == "x"
        
        var = model.get_variable("nonexistent")
        assert var is None
    
    def test_validate_model(self):
        """Test model validation."""
        model = ConstraintModel("test")
        errors = model.validate()
        assert len(errors) == 0
        
        # Test with empty name
        model2 = ConstraintModel("")
        errors = model2.validate()
        assert len(errors) > 0
    
    def test_nqueens_model(self):
        """Test creating N-Queens model."""
        n = 8
        model = ConstraintModel("nqueens")
        
        # Add queen position variables
        for i in range(1, n + 1):
            model.add_int_variable(f"q{i}", 1, n)
        
        # Add alldifferent constraint
        vars_list = [VariableRef(f"q{i}") for i in range(1, n + 1)]
        model.add_constraint(alldifferent(vars_list))
        
        # Set satisfaction objective
        model.set_objective(Objective.satisfy())
        
        assert len(model.variables) == n
        assert len(model.constraints) == 1
        assert model.objective.objective_type == ObjectiveType.SATISFY


class TestConstraintEmitterResult:
    """Tests for ConstraintEmitterResult class."""
    
    def test_result_creation(self):
        """Test result creation."""
        result = ConstraintEmitterResult(
            code="var 1..10: x;",
            manifest={"schema": "test"},
            warnings=[]
        )
        assert "var" in result.code
        assert result.manifest["schema"] == "test"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
