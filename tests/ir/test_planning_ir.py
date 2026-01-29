"""Tests for Planning IR.

Tests for the planning intermediate representation classes
including domains, problems, actions, predicates, and formulas.
"""

import pytest
import sys
import os

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from ir.planning import (
    # Enums
    PDDLRequirement, FormulaType, EffectType,
    # Types
    TypeDef, Parameter, Predicate, Function, Atom, FunctionApplication,
    Formula, Effect, Action,
    ObjectDef, DerivedPredicate, Domain,
    InitialState, Metric, Problem,
    # Exceptions
    PlanningError, InvalidActionError, InvalidPredicateError,
    InvalidDomainError, InvalidProblemError,
    # Result
    PlanningEmitterResult,
)


class TestTypeDef:
    """Tests for TypeDef class."""
    
    def test_simple_type(self):
        """Test creating a simple type."""
        t = TypeDef("block")
        assert t.name == "block"
        assert t.parent == "object"
    
    def test_type_with_parent(self):
        """Test creating a type with custom parent."""
        t = TypeDef("city", "location")
        assert t.name == "city"
        assert t.parent == "location"
    
    def test_empty_name_raises(self):
        """Test that empty name raises ValueError."""
        with pytest.raises(ValueError):
            TypeDef("")
    
    def test_invalid_name_raises(self):
        """Test that invalid name raises ValueError."""
        with pytest.raises(ValueError):
            TypeDef("123invalid")


class TestParameter:
    """Tests for Parameter class."""
    
    def test_simple_parameter(self):
        """Test creating a simple parameter."""
        p = Parameter("?x")
        assert p.name == "?x"
        assert p.param_type == "object"
    
    def test_typed_parameter(self):
        """Test creating a typed parameter."""
        p = Parameter("?loc", "location")
        assert p.name == "?loc"
        assert p.param_type == "location"
    
    def test_auto_prefix(self):
        """Test that ? is auto-prefixed."""
        p = Parameter("x")
        assert p.name == "?x"
    
    def test_empty_name_raises(self):
        """Test that empty name raises ValueError."""
        with pytest.raises(ValueError):
            Parameter("")


class TestPredicate:
    """Tests for Predicate class."""
    
    def test_nullary_predicate(self):
        """Test creating a predicate with no parameters."""
        p = Predicate("arm-empty")
        assert p.name == "arm-empty"
        assert p.arity() == 0
    
    def test_unary_predicate(self):
        """Test creating a unary predicate."""
        p = Predicate("clear", [Parameter("?x", "block")])
        assert p.name == "clear"
        assert p.arity() == 1
    
    def test_binary_predicate(self):
        """Test creating a binary predicate."""
        p = Predicate("on", [
            Parameter("?x", "block"),
            Parameter("?y", "block")
        ])
        assert p.name == "on"
        assert p.arity() == 2
    
    def test_get_signature(self):
        """Test getting predicate signature."""
        p = Predicate("at", [
            Parameter("?obj", "object"),
            Parameter("?loc", "location")
        ])
        sig = p.get_signature()
        assert "at" in sig
        assert "object" in sig
        assert "location" in sig


class TestAtom:
    """Tests for Atom class."""
    
    def test_ground_atom(self):
        """Test creating a ground atom."""
        a = Atom("at", ["robot1", "loc1"])
        assert a.predicate == "at"
        assert a.arguments == ["robot1", "loc1"]
        assert a.is_ground()
    
    def test_lifted_atom(self):
        """Test creating a lifted atom."""
        a = Atom("at", ["?obj", "?loc"])
        assert not a.is_ground()
    
    def test_arity(self):
        """Test atom arity."""
        a = Atom("on", ["?x", "?y"])
        assert a.arity() == 2


class TestFormula:
    """Tests for Formula class."""
    
    def test_make_atom(self):
        """Test creating atomic formula."""
        f = Formula.make_atom("clear", "?x")
        assert f.formula_type == FormulaType.ATOM
        assert f.atom is not None
        assert f.atom.predicate == "clear"
    
    def test_make_and(self):
        """Test creating conjunction."""
        f1 = Formula.make_atom("clear", "?x")
        f2 = Formula.make_atom("ontable", "?x")
        f = Formula.make_and(f1, f2)
        assert f.formula_type == FormulaType.AND
        assert len(f.children) == 2
    
    def test_make_and_single(self):
        """Test that single-child AND returns the child."""
        f1 = Formula.make_atom("clear", "?x")
        f = Formula.make_and(f1)
        assert f.formula_type == FormulaType.ATOM
    
    def test_make_or(self):
        """Test creating disjunction."""
        f1 = Formula.make_atom("a")
        f2 = Formula.make_atom("b")
        f = Formula.make_or(f1, f2)
        assert f.formula_type == FormulaType.OR
        assert len(f.children) == 2
    
    def test_make_not(self):
        """Test creating negation."""
        f1 = Formula.make_atom("holding", "?x")
        f = Formula.make_not(f1)
        assert f.formula_type == FormulaType.NOT
        assert len(f.children) == 1
    
    def test_make_imply(self):
        """Test creating implication."""
        ant = Formula.make_atom("a")
        cons = Formula.make_atom("b")
        f = Formula.make_imply(ant, cons)
        assert f.formula_type == FormulaType.IMPLY
        assert len(f.children) == 2
    
    def test_make_exists(self):
        """Test creating existential formula."""
        body = Formula.make_atom("at", "?obj", "?loc")
        f = Formula.make_exists([Parameter("?obj", "object")], body)
        assert f.formula_type == FormulaType.EXISTS
        assert len(f.variables) == 1
    
    def test_make_forall(self):
        """Test creating universal formula."""
        body = Formula.make_atom("safe", "?loc")
        f = Formula.make_forall([Parameter("?loc", "location")], body)
        assert f.formula_type == FormulaType.FORALL
        assert len(f.variables) == 1
    
    def test_make_equals(self):
        """Test creating equality formula."""
        f = Formula.make_equals("?x", "?y")
        assert f.formula_type == FormulaType.EQUALS
        assert f.left_term == "?x"
        assert f.right_term == "?y"
    
    def test_is_atom(self):
        """Test is_atom check."""
        f = Formula.make_atom("p")
        assert f.is_atom()
        f2 = Formula.make_not(f)
        assert not f2.is_atom()
    
    def test_is_compound(self):
        """Test is_compound check."""
        f = Formula.make_and(
            Formula.make_atom("a"),
            Formula.make_atom("b")
        )
        assert f.is_compound()
    
    def test_is_quantified(self):
        """Test is_quantified check."""
        f = Formula.make_exists([Parameter("?x")], Formula.make_atom("p", "?x"))
        assert f.is_quantified()


class TestEffect:
    """Tests for Effect class."""
    
    def test_make_positive(self):
        """Test creating positive effect."""
        e = Effect.make_positive("holding", "?x")
        assert e.effect_type == EffectType.POSITIVE
        assert e.formula is not None
    
    def test_make_negative(self):
        """Test creating negative effect."""
        e = Effect.make_negative("clear", "?x")
        assert e.effect_type == EffectType.NEGATIVE
    
    def test_make_compound(self):
        """Test creating compound effect."""
        e1 = Effect.make_positive("a")
        e2 = Effect.make_negative("b")
        e = Effect.make_compound(e1, e2)
        assert e.effect_type == EffectType.COMPOUND
        assert len(e.children) == 2
    
    def test_make_conditional(self):
        """Test creating conditional effect."""
        cond = Formula.make_atom("heavy", "?x")
        eff = Effect.make_positive("slow")
        e = Effect.make_conditional(cond, eff)
        assert e.effect_type == EffectType.CONDITIONAL
        assert e.condition is not None
    
    def test_make_increase(self):
        """Test creating increase effect."""
        e = Effect.make_increase("total-cost", [], 1)
        assert e.effect_type == EffectType.INCREASE
        assert e.function_app.function == "total-cost"
        assert e.value == 1
    
    def test_make_decrease(self):
        """Test creating decrease effect."""
        e = Effect.make_decrease("fuel", ["?v"], 10)
        assert e.effect_type == EffectType.DECREASE


class TestAction:
    """Tests for Action class."""
    
    def test_simple_action(self):
        """Test creating a simple action."""
        a = Action(name="move")
        assert a.name == "move"
        assert a.parameters == []
    
    def test_action_with_parameters(self):
        """Test action with typed parameters."""
        a = Action(
            name="move",
            parameters=[
                Parameter("?from", "location"),
                Parameter("?to", "location")
            ]
        )
        assert len(a.parameters) == 2
        assert a.get_parameter_names() == ["?from", "?to"]
        assert a.get_parameter_types() == ["location", "location"]
    
    def test_action_with_precondition(self):
        """Test action with precondition."""
        a = Action(
            name="pick-up",
            parameters=[Parameter("?x", "block")],
            precondition=Formula.make_and(
                Formula.make_atom("clear", "?x"),
                Formula.make_atom("arm-empty")
            )
        )
        assert a.precondition is not None
    
    def test_action_with_effect(self):
        """Test action with effect."""
        a = Action(
            name="pick-up",
            parameters=[Parameter("?x", "block")],
            effect=Effect.make_compound(
                Effect.make_positive("holding", "?x"),
                Effect.make_negative("arm-empty")
            )
        )
        assert a.effect is not None
    
    def test_empty_name_raises(self):
        """Test that empty action name raises."""
        with pytest.raises(ValueError):
            Action(name="")


class TestDomain:
    """Tests for Domain class."""
    
    def test_minimal_domain(self):
        """Test creating minimal domain."""
        d = Domain(name="test-domain")
        assert d.name == "test-domain"
        assert d.types == []
        assert d.predicates == []
        assert d.actions == []
    
    def test_domain_with_types(self):
        """Test domain with types."""
        d = Domain(name="typed-domain")
        d.add_type("location")
        d.add_type("vehicle")
        assert len(d.types) == 2
    
    def test_domain_with_predicates(self):
        """Test domain with predicates."""
        d = Domain(name="pred-domain")
        d.add_predicate(Predicate("at", [
            Parameter("?obj", "object"),
            Parameter("?loc", "location")
        ]))
        assert len(d.predicates) == 1
    
    def test_domain_fluent_interface(self):
        """Test domain fluent interface."""
        d = (Domain(name="fluent")
             .add_type("block")
             .add_predicate(Predicate("on", [
                 Parameter("?x", "block"),
                 Parameter("?y", "block")
             ]))
             .add_requirement(PDDLRequirement.STRIPS)
             .add_requirement(PDDLRequirement.TYPING))
        
        assert len(d.types) == 1
        assert len(d.predicates) == 1
        assert len(d.requirements) == 2
    
    def test_validate_empty_name(self):
        """Test that empty name raises ValueError on construction."""
        with pytest.raises(ValueError, match="empty"):
            Domain(name="")
    
    def test_validate_duplicate_types(self):
        """Test validation catches duplicate types."""
        d = Domain(name="test")
        d.add_type("block")
        d.add_type("block")
        errors = d.validate()
        assert any("duplicate" in e.lower() for e in errors)
    
    def test_validate_unknown_type_in_predicate(self):
        """Test validation catches unknown type references."""
        d = Domain(name="test", requirements=[PDDLRequirement.TYPING])
        d.add_predicate(Predicate("at", [Parameter("?x", "unknown_type")]))
        errors = d.validate()
        assert any("unknown" in e.lower() for e in errors)
    
    def test_infer_requirements(self):
        """Test requirement inference."""
        d = Domain(name="test")
        d.add_type("block")
        reqs = d.infer_requirements()
        assert PDDLRequirement.STRIPS in reqs
        assert PDDLRequirement.TYPING in reqs


class TestProblem:
    """Tests for Problem class."""
    
    def test_minimal_problem(self):
        """Test creating minimal problem."""
        p = Problem(name="test-prob", domain_name="test-domain")
        assert p.name == "test-prob"
        assert p.domain_name == "test-domain"
    
    def test_problem_with_objects(self):
        """Test problem with objects."""
        p = Problem(name="prob", domain_name="dom")
        p.add_object("b1", "block")
        p.add_object("b2", "block")
        assert len(p.objects) == 2
    
    def test_problem_add_objects_batch(self):
        """Test adding multiple objects."""
        p = Problem(name="prob", domain_name="dom")
        p.add_objects(["b1", "b2", "b3"], "block")
        assert len(p.objects) == 3
    
    def test_problem_with_init(self):
        """Test problem with initial state."""
        p = Problem(name="prob", domain_name="dom")
        p.init.add_fact("clear", "b1")
        p.init.add_fact("on", "b1", "b2")
        assert len(p.init.facts) == 2
    
    def test_problem_with_goal(self):
        """Test problem with goal."""
        p = Problem(name="prob", domain_name="dom")
        p.set_goal(Formula.make_atom("achieved"))
        assert p.goal is not None
    
    def test_problem_with_metric(self):
        """Test problem with metric."""
        p = Problem(name="prob", domain_name="dom")
        p.set_metric("minimize", "total-cost")
        assert p.metric is not None
        assert p.metric.direction == "minimize"
    
    def test_validate_empty_goal(self):
        """Test validation catches empty goal."""
        p = Problem(name="prob", domain_name="dom")
        errors = p.validate()
        assert any("goal" in e.lower() for e in errors)
    
    def test_validate_duplicate_objects(self):
        """Test validation catches duplicate objects."""
        p = Problem(name="prob", domain_name="dom")
        p.add_object("b1", "block")
        p.add_object("b1", "block")
        p.set_goal(Formula.make_atom("done"))
        errors = p.validate()
        assert any("duplicate" in e.lower() for e in errors)
    
    def test_get_objects_by_type(self):
        """Test filtering objects by type."""
        p = Problem(name="prob", domain_name="dom")
        p.add_objects(["b1", "b2"], "block")
        p.add_objects(["l1", "l2", "l3"], "location")
        
        blocks = p.get_objects_by_type("block")
        locs = p.get_objects_by_type("location")
        
        assert len(blocks) == 2
        assert len(locs) == 3


class TestInitialState:
    """Tests for InitialState class."""
    
    def test_add_fact(self):
        """Test adding facts."""
        init = InitialState()
        init.add_fact("clear", "b1")
        init.add_fact("on", "b1", "b2")
        assert len(init.facts) == 2
    
    def test_add_numeric(self):
        """Test adding numeric values."""
        init = InitialState()
        init.add_numeric("total-cost", 0)
        init.add_numeric("fuel truck1", 100)
        assert len(init.numeric_values) == 2
    
    def test_add_timed_literal(self):
        """Test adding timed literals."""
        init = InitialState()
        init.add_timed_literal(10.0, "available", "resource1")
        assert len(init.timed_literals) == 1


class TestPlanningEmitterResult:
    """Tests for PlanningEmitterResult class."""
    
    def test_basic_result(self):
        """Test creating basic result."""
        result = PlanningEmitterResult(domain_code="(domain test)")
        assert result.domain_code == "(domain test)"
        assert result.problem_code == ""
    
    def test_result_with_problem(self):
        """Test result with problem code."""
        result = PlanningEmitterResult(
            domain_code="(domain test)",
            problem_code="(problem test-p)"
        )
        assert result.problem_code == "(problem test-p)"


class TestBlocksWorld:
    """Integration tests using Blocks World domain."""
    
    @pytest.fixture
    def blocks_domain(self):
        """Create Blocks World domain."""
        domain = Domain(
            name="blocks-world",
            requirements=[PDDLRequirement.STRIPS, PDDLRequirement.TYPING]
        )
        domain.add_type("block")
        
        # Predicates
        domain.add_predicate(Predicate("on", [
            Parameter("?x", "block"),
            Parameter("?y", "block")
        ]))
        domain.add_predicate(Predicate("ontable", [Parameter("?x", "block")]))
        domain.add_predicate(Predicate("clear", [Parameter("?x", "block")]))
        domain.add_predicate(Predicate("holding", [Parameter("?x", "block")]))
        domain.add_predicate(Predicate("arm-empty", []))
        
        # Actions
        domain.add_action(Action(
            name="pick-up",
            parameters=[Parameter("?x", "block")],
            precondition=Formula.make_and(
                Formula.make_atom("clear", "?x"),
                Formula.make_atom("ontable", "?x"),
                Formula.make_atom("arm-empty")
            ),
            effect=Effect.make_compound(
                Effect.make_positive("holding", "?x"),
                Effect.make_negative("ontable", "?x"),
                Effect.make_negative("clear", "?x"),
                Effect.make_negative("arm-empty")
            )
        ))
        
        return domain
    
    @pytest.fixture
    def blocks_problem(self, blocks_domain):
        """Create Blocks World problem."""
        problem = Problem(
            name="blocks-4-0",
            domain_name="blocks-world"
        )
        problem.add_objects(["b1", "b2", "b3", "b4"], "block")
        
        problem.init.add_fact("clear", "b1")
        problem.init.add_fact("on", "b1", "b2")
        problem.init.add_fact("on", "b2", "b3")
        problem.init.add_fact("ontable", "b3")
        problem.init.add_fact("ontable", "b4")
        problem.init.add_fact("clear", "b4")
        problem.init.add_fact("arm-empty")
        
        problem.set_goal(Formula.make_and(
            Formula.make_atom("on", "b4", "b3"),
            Formula.make_atom("on", "b3", "b2"),
            Formula.make_atom("on", "b2", "b1")
        ))
        
        return problem
    
    def test_domain_structure(self, blocks_domain):
        """Test Blocks World domain structure."""
        assert blocks_domain.name == "blocks-world"
        assert len(blocks_domain.types) == 1
        assert len(blocks_domain.predicates) == 5
        assert len(blocks_domain.actions) == 1
    
    def test_domain_validation(self, blocks_domain):
        """Test Blocks World domain validates."""
        errors = blocks_domain.validate()
        assert len(errors) == 0
    
    def test_problem_structure(self, blocks_problem):
        """Test Blocks World problem structure."""
        assert blocks_problem.name == "blocks-4-0"
        assert len(blocks_problem.objects) == 4
        assert len(blocks_problem.init.facts) == 7
        assert blocks_problem.goal is not None
    
    def test_problem_validation(self, blocks_domain, blocks_problem):
        """Test Blocks World problem validates."""
        errors = blocks_problem.validate(blocks_domain)
        assert len(errors) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
