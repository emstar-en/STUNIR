"""Tests for PDDL Emitter.

Tests for the PDDL planning domain/problem emitter.
"""

import pytest
import sys
import os

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from ir.planning import (
    Domain, Problem, Action, Predicate, Function, Parameter,
    Formula, Effect, TypeDef, ObjectDef, Atom, InitialState, Metric,
    PDDLRequirement, FormulaType, EffectType,
    PlanningEmitterResult,
)
from targets.planning import PDDLEmitter


class TestPDDLEmitterBasics:
    """Basic PDDL emitter tests."""
    
    @pytest.fixture
    def emitter(self):
        """Create emitter instance."""
        return PDDLEmitter()
    
    def test_minimal_domain(self, emitter):
        """Test emitting minimal domain."""
        domain = Domain(name="minimal")
        result = emitter.emit_domain(domain)
        
        assert "(define (domain minimal)" in result.domain_code
        assert result.manifest['domain']['name'] == "minimal"
    
    def test_domain_with_requirements(self, emitter):
        """Test emitting domain with requirements."""
        domain = Domain(
            name="typed",
            requirements=[PDDLRequirement.STRIPS, PDDLRequirement.TYPING]
        )
        result = emitter.emit_domain(domain)
        
        assert "(:requirements" in result.domain_code
        assert ":strips" in result.domain_code
        assert ":typing" in result.domain_code
    
    def test_domain_with_types(self, emitter):
        """Test emitting domain with types."""
        domain = Domain(
            name="typed-domain",
            requirements=[PDDLRequirement.TYPING]
        )
        domain.add_type("location")
        domain.add_type("vehicle")
        domain.add_type("truck", "vehicle")
        
        result = emitter.emit_domain(domain)
        
        assert "(:types" in result.domain_code
        assert "location" in result.domain_code
        assert "truck - vehicle" in result.domain_code
    
    def test_domain_with_predicates(self, emitter):
        """Test emitting domain with predicates."""
        domain = Domain(name="pred-domain")
        domain.add_predicate(Predicate("at", [
            Parameter("?obj", "object"),
            Parameter("?loc", "location")
        ]))
        domain.add_predicate(Predicate("clear", [Parameter("?x", "block")]))
        domain.add_predicate(Predicate("arm-empty", []))
        
        result = emitter.emit_domain(domain)
        
        assert "(:predicates" in result.domain_code
        assert "(at" in result.domain_code
        assert "(clear" in result.domain_code
        assert "(arm-empty)" in result.domain_code
    
    def test_domain_with_functions(self, emitter):
        """Test emitting domain with functions."""
        domain = Domain(
            name="numeric-domain",
            requirements=[PDDLRequirement.NUMERIC_FLUENTS]
        )
        domain.add_function(Function("total-cost", []))
        domain.add_function(Function("distance", [
            Parameter("?from", "location"),
            Parameter("?to", "location")
        ]))
        
        result = emitter.emit_domain(domain)
        
        assert "(:functions" in result.domain_code
        assert "(total-cost)" in result.domain_code
        assert "(distance" in result.domain_code


class TestPDDLEmitterActions:
    """Tests for PDDL action emission."""
    
    @pytest.fixture
    def emitter(self):
        return PDDLEmitter()
    
    def test_simple_action(self, emitter):
        """Test emitting simple action."""
        domain = Domain(name="test")
        domain.add_action(Action(
            name="noop",
            parameters=[]
        ))
        
        result = emitter.emit_domain(domain)
        
        assert "(:action noop" in result.domain_code
        assert ":parameters ()" in result.domain_code
    
    def test_action_with_parameters(self, emitter):
        """Test action with typed parameters."""
        domain = Domain(name="test", requirements=[PDDLRequirement.TYPING])
        domain.add_type("location")
        domain.add_action(Action(
            name="move",
            parameters=[
                Parameter("?from", "location"),
                Parameter("?to", "location")
            ]
        ))
        
        result = emitter.emit_domain(domain)
        
        assert "(:action move" in result.domain_code
        assert "?from" in result.domain_code
        assert "?to" in result.domain_code
        assert "location" in result.domain_code
    
    def test_action_with_precondition(self, emitter):
        """Test action with precondition."""
        domain = Domain(name="test")
        domain.add_action(Action(
            name="pick-up",
            parameters=[Parameter("?x", "block")],
            precondition=Formula.make_and(
                Formula.make_atom("clear", "?x"),
                Formula.make_atom("arm-empty")
            )
        ))
        
        result = emitter.emit_domain(domain)
        
        assert ":precondition" in result.domain_code
        assert "(and" in result.domain_code
        assert "(clear ?x)" in result.domain_code
        assert "(arm-empty)" in result.domain_code
    
    def test_action_with_negative_precondition(self, emitter):
        """Test action with negative precondition."""
        domain = Domain(name="test", requirements=[PDDLRequirement.NEGATIVE_PRECONDITIONS])
        domain.add_action(Action(
            name="test-act",
            parameters=[Parameter("?x")],
            precondition=Formula.make_not(Formula.make_atom("busy", "?x"))
        ))
        
        result = emitter.emit_domain(domain)
        
        assert "(not (busy ?x))" in result.domain_code
    
    def test_action_with_effect(self, emitter):
        """Test action with effect."""
        domain = Domain(name="test")
        domain.add_action(Action(
            name="pick-up",
            parameters=[Parameter("?x", "block")],
            effect=Effect.make_compound(
                Effect.make_positive("holding", "?x"),
                Effect.make_negative("clear", "?x")
            )
        ))
        
        result = emitter.emit_domain(domain)
        
        assert ":effect" in result.domain_code
        assert "(holding ?x)" in result.domain_code
        assert "(not (clear ?x))" in result.domain_code


class TestPDDLEmitterFormulas:
    """Tests for PDDL formula emission."""
    
    @pytest.fixture
    def emitter(self):
        return PDDLEmitter()
    
    def test_disjunction(self, emitter):
        """Test emitting disjunction."""
        domain = Domain(name="test", requirements=[PDDLRequirement.DISJUNCTIVE_PRECONDITIONS])
        domain.add_action(Action(
            name="test-act",
            parameters=[Parameter("?x")],
            precondition=Formula.make_or(
                Formula.make_atom("a", "?x"),
                Formula.make_atom("b", "?x")
            )
        ))
        
        result = emitter.emit_domain(domain)
        
        assert "(or" in result.domain_code
        assert "(a ?x)" in result.domain_code
        assert "(b ?x)" in result.domain_code
    
    def test_implication(self, emitter):
        """Test emitting implication."""
        domain = Domain(name="test")
        domain.add_action(Action(
            name="test-act",
            parameters=[Parameter("?x")],
            precondition=Formula.make_imply(
                Formula.make_atom("a", "?x"),
                Formula.make_atom("b", "?x")
            )
        ))
        
        result = emitter.emit_domain(domain)
        
        assert "(imply" in result.domain_code
    
    def test_existential(self, emitter):
        """Test emitting existential formula."""
        domain = Domain(name="test", requirements=[PDDLRequirement.EXISTENTIAL_PRECONDITIONS])
        domain.add_action(Action(
            name="test-act",
            parameters=[],
            precondition=Formula.make_exists(
                [Parameter("?x", "block")],
                Formula.make_atom("clear", "?x")
            )
        ))
        
        result = emitter.emit_domain(domain)
        
        assert "(exists" in result.domain_code
        assert "?x - block" in result.domain_code
    
    def test_universal(self, emitter):
        """Test emitting universal formula."""
        domain = Domain(name="test", requirements=[PDDLRequirement.UNIVERSAL_PRECONDITIONS])
        domain.add_action(Action(
            name="test-act",
            parameters=[],
            precondition=Formula.make_forall(
                [Parameter("?x", "block")],
                Formula.make_atom("safe", "?x")
            )
        ))
        
        result = emitter.emit_domain(domain)
        
        assert "(forall" in result.domain_code
    
    def test_equality(self, emitter):
        """Test emitting equality formula."""
        domain = Domain(name="test", requirements=[PDDLRequirement.EQUALITY])
        domain.add_action(Action(
            name="test-act",
            parameters=[Parameter("?x"), Parameter("?y")],
            precondition=Formula.make_not(Formula.make_equals("?x", "?y"))
        ))
        
        result = emitter.emit_domain(domain)
        
        assert "(= ?x ?y)" in result.domain_code


class TestPDDLEmitterEffects:
    """Tests for PDDL effect emission."""
    
    @pytest.fixture
    def emitter(self):
        return PDDLEmitter()
    
    def test_conditional_effect(self, emitter):
        """Test emitting conditional effect."""
        domain = Domain(name="test", requirements=[PDDLRequirement.CONDITIONAL_EFFECTS])
        domain.add_action(Action(
            name="test-act",
            parameters=[Parameter("?x")],
            effect=Effect.make_conditional(
                Formula.make_atom("heavy", "?x"),
                Effect.make_positive("slow")
            )
        ))
        
        result = emitter.emit_domain(domain)
        
        assert "(when" in result.domain_code
        assert "(heavy ?x)" in result.domain_code
        assert "(slow)" in result.domain_code
    
    def test_forall_effect(self, emitter):
        """Test emitting universal effect."""
        domain = Domain(name="test")
        domain.add_action(Action(
            name="reset-all",
            parameters=[],
            effect=Effect.make_forall(
                [Parameter("?x", "block")],
                Effect.make_negative("marked", "?x")
            )
        ))
        
        result = emitter.emit_domain(domain)
        
        assert "(forall" in result.domain_code
        assert "(not (marked ?x))" in result.domain_code
    
    def test_numeric_increase(self, emitter):
        """Test emitting numeric increase effect."""
        domain = Domain(name="test", requirements=[PDDLRequirement.NUMERIC_FLUENTS])
        domain.add_action(Action(
            name="test-act",
            parameters=[],
            effect=Effect.make_increase("total-cost", [], 1)
        ))
        
        result = emitter.emit_domain(domain)
        
        assert "(increase (total-cost) 1)" in result.domain_code
    
    def test_numeric_decrease(self, emitter):
        """Test emitting numeric decrease effect."""
        domain = Domain(name="test", requirements=[PDDLRequirement.NUMERIC_FLUENTS])
        domain.add_action(Action(
            name="consume-fuel",
            parameters=[Parameter("?v", "vehicle")],
            effect=Effect.make_decrease("fuel", ["?v"], 10)
        ))
        
        result = emitter.emit_domain(domain)
        
        assert "(decrease (fuel ?v) 10)" in result.domain_code


class TestPDDLEmitterProblem:
    """Tests for PDDL problem emission."""
    
    @pytest.fixture
    def emitter(self):
        return PDDLEmitter()
    
    def test_minimal_problem(self, emitter):
        """Test emitting minimal problem."""
        problem = Problem(name="prob1", domain_name="test-domain")
        problem.set_goal(Formula.make_atom("done"))
        
        result = emitter.emit_problem(problem)
        
        assert "(define (problem prob1)" in result.problem_code
        assert "(:domain test-domain)" in result.problem_code
    
    def test_problem_with_objects(self, emitter):
        """Test problem with objects."""
        problem = Problem(name="prob1", domain_name="test")
        problem.add_objects(["b1", "b2", "b3"], "block")
        problem.add_objects(["l1", "l2"], "location")
        problem.set_goal(Formula.make_atom("done"))
        
        result = emitter.emit_problem(problem)
        
        assert "(:objects" in result.problem_code
        assert "b1" in result.problem_code
        assert "block" in result.problem_code
        assert "location" in result.problem_code
    
    def test_problem_with_init(self, emitter):
        """Test problem with initial state."""
        problem = Problem(name="prob1", domain_name="test")
        problem.init.add_fact("clear", "b1")
        problem.init.add_fact("on", "b1", "b2")
        problem.init.add_fact("arm-empty")
        problem.set_goal(Formula.make_atom("done"))
        
        result = emitter.emit_problem(problem)
        
        assert "(:init" in result.problem_code
        assert "(clear b1)" in result.problem_code
        assert "(on b1 b2)" in result.problem_code
        assert "(arm-empty)" in result.problem_code
    
    def test_problem_with_numeric_init(self, emitter):
        """Test problem with numeric initial values."""
        problem = Problem(name="prob1", domain_name="test")
        problem.init.add_numeric("total-cost", 0)
        problem.init.add_numeric("fuel truck1", 100)
        problem.set_goal(Formula.make_atom("done"))
        
        result = emitter.emit_problem(problem)
        
        assert "(= (total-cost) 0)" in result.problem_code
        assert "(= (fuel truck1) 100)" in result.problem_code
    
    def test_problem_with_goal(self, emitter):
        """Test problem with goal."""
        problem = Problem(name="prob1", domain_name="test")
        problem.set_goal(Formula.make_and(
            Formula.make_atom("on", "b1", "b2"),
            Formula.make_atom("clear", "b1")
        ))
        
        result = emitter.emit_problem(problem)
        
        assert "(:goal" in result.problem_code
        assert "(and" in result.problem_code
        assert "(on b1 b2)" in result.problem_code
    
    def test_problem_with_metric(self, emitter):
        """Test problem with metric."""
        problem = Problem(name="prob1", domain_name="test")
        problem.set_goal(Formula.make_atom("done"))
        problem.set_metric("minimize", "total-cost")
        
        result = emitter.emit_problem(problem)
        
        assert "(:metric minimize (total-cost))" in result.problem_code


class TestPDDLEmitterCombined:
    """Tests for combined domain and problem emission."""
    
    @pytest.fixture
    def emitter(self):
        return PDDLEmitter()
    
    def test_emit_both(self, emitter):
        """Test emitting domain and problem together."""
        domain = Domain(name="test-domain")
        domain.add_predicate(Predicate("done", []))
        
        problem = Problem(name="test-prob", domain_name="test-domain")
        problem.set_goal(Formula.make_atom("done"))
        
        result = emitter.emit(domain, problem)
        
        assert result.domain_code != ""
        assert result.problem_code != ""
        assert "test-domain" in result.domain_code
        assert "test-prob" in result.problem_code
    
    def test_manifest_combined(self, emitter):
        """Test manifest for combined emission."""
        domain = Domain(name="test-domain")
        problem = Problem(name="test-prob", domain_name="test-domain")
        problem.set_goal(Formula.make_atom("done"))
        
        result = emitter.emit(domain, problem)
        
        assert 'domain' in result.manifest
        assert 'problem' in result.manifest or 'problem_output' in result.manifest
        assert 'manifest_hash' in result.manifest


class TestBlocksWorldIntegration:
    """Integration tests with Blocks World domain."""
    
    @pytest.fixture
    def emitter(self):
        return PDDLEmitter()
    
    @pytest.fixture
    def blocks_domain(self):
        """Create complete Blocks World domain."""
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
        
        # Pick-up action
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
        
        # Put-down action
        domain.add_action(Action(
            name="put-down",
            parameters=[Parameter("?x", "block")],
            precondition=Formula.make_atom("holding", "?x"),
            effect=Effect.make_compound(
                Effect.make_positive("ontable", "?x"),
                Effect.make_positive("clear", "?x"),
                Effect.make_positive("arm-empty"),
                Effect.make_negative("holding", "?x")
            )
        ))
        
        # Stack action
        domain.add_action(Action(
            name="stack",
            parameters=[Parameter("?x", "block"), Parameter("?y", "block")],
            precondition=Formula.make_and(
                Formula.make_atom("holding", "?x"),
                Formula.make_atom("clear", "?y")
            ),
            effect=Effect.make_compound(
                Effect.make_positive("on", "?x", "?y"),
                Effect.make_positive("clear", "?x"),
                Effect.make_positive("arm-empty"),
                Effect.make_negative("holding", "?x"),
                Effect.make_negative("clear", "?y")
            )
        ))
        
        # Unstack action
        domain.add_action(Action(
            name="unstack",
            parameters=[Parameter("?x", "block"), Parameter("?y", "block")],
            precondition=Formula.make_and(
                Formula.make_atom("on", "?x", "?y"),
                Formula.make_atom("clear", "?x"),
                Formula.make_atom("arm-empty")
            ),
            effect=Effect.make_compound(
                Effect.make_positive("holding", "?x"),
                Effect.make_positive("clear", "?y"),
                Effect.make_negative("on", "?x", "?y"),
                Effect.make_negative("clear", "?x"),
                Effect.make_negative("arm-empty")
            )
        ))
        
        return domain
    
    @pytest.fixture
    def blocks_problem(self):
        """Create Blocks World problem."""
        problem = Problem(
            name="blocks-4-0",
            domain_name="blocks-world"
        )
        problem.add_objects(["b1", "b2", "b3", "b4"], "block")
        
        # Initial state: b1 on b2, b2 on b3, b3 and b4 on table
        problem.init.add_fact("clear", "b1")
        problem.init.add_fact("on", "b1", "b2")
        problem.init.add_fact("on", "b2", "b3")
        problem.init.add_fact("ontable", "b3")
        problem.init.add_fact("ontable", "b4")
        problem.init.add_fact("clear", "b4")
        problem.init.add_fact("arm-empty")
        
        # Goal: b4 on b3 on b2 on b1
        problem.set_goal(Formula.make_and(
            Formula.make_atom("on", "b4", "b3"),
            Formula.make_atom("on", "b3", "b2"),
            Formula.make_atom("on", "b2", "b1")
        ))
        
        return problem
    
    def test_blocks_domain_emission(self, emitter, blocks_domain):
        """Test Blocks World domain emission."""
        result = emitter.emit_domain(blocks_domain)
        
        # Check structure
        assert "(define (domain blocks-world)" in result.domain_code
        assert "(:requirements :strips :typing)" in result.domain_code
        assert "(:types" in result.domain_code
        assert "block" in result.domain_code
        assert "(:predicates" in result.domain_code
        
        # Check all predicates
        assert "(on" in result.domain_code
        assert "(ontable" in result.domain_code
        assert "(clear" in result.domain_code
        assert "(holding" in result.domain_code
        assert "(arm-empty)" in result.domain_code
        
        # Check all actions
        assert "(:action pick-up" in result.domain_code
        assert "(:action put-down" in result.domain_code
        assert "(:action stack" in result.domain_code
        assert "(:action unstack" in result.domain_code
        
        # Check manifest
        assert result.manifest['domain']['name'] == "blocks-world"
        assert result.manifest['domain']['actions'] == 4
        assert result.manifest['domain']['predicates'] == 5
    
    def test_blocks_problem_emission(self, emitter, blocks_problem):
        """Test Blocks World problem emission."""
        result = emitter.emit_problem(blocks_problem)
        
        # Check structure
        assert "(define (problem blocks-4-0)" in result.problem_code
        assert "(:domain blocks-world)" in result.problem_code
        assert "(:objects" in result.problem_code
        assert "(:init" in result.problem_code
        assert "(:goal" in result.problem_code
        
        # Check objects
        assert "b1" in result.problem_code
        assert "b2" in result.problem_code
        assert "b3" in result.problem_code
        assert "b4" in result.problem_code
        
        # Check init facts
        assert "(clear b1)" in result.problem_code
        assert "(on b1 b2)" in result.problem_code
        assert "(arm-empty)" in result.problem_code
        
        # Check manifest
        assert result.manifest['problem']['name'] == "blocks-4-0"
        assert result.manifest['problem']['objects'] == 4
    
    def test_blocks_full_emission(self, emitter, blocks_domain, blocks_problem):
        """Test full Blocks World emission."""
        result = emitter.emit(blocks_domain, blocks_problem)
        
        assert result.domain_code != ""
        assert result.problem_code != ""
        assert 'manifest_hash' in result.manifest


class TestLogisticsIntegration:
    """Integration tests with Logistics domain."""
    
    @pytest.fixture
    def emitter(self):
        return PDDLEmitter()
    
    @pytest.fixture
    def logistics_domain(self):
        """Create simplified Logistics domain."""
        domain = Domain(
            name="logistics",
            requirements=[PDDLRequirement.STRIPS, PDDLRequirement.TYPING]
        )
        
        # Types
        domain.add_type("location")
        domain.add_type("city")
        domain.add_type("thing")
        domain.add_type("package", "thing")
        domain.add_type("vehicle", "thing")
        domain.add_type("truck", "vehicle")
        domain.add_type("airplane", "vehicle")
        domain.add_type("airport", "location")
        
        # Predicates
        domain.add_predicate(Predicate("in-city", [
            Parameter("?loc", "location"),
            Parameter("?city", "city")
        ]))
        domain.add_predicate(Predicate("at", [
            Parameter("?obj", "thing"),
            Parameter("?loc", "location")
        ]))
        domain.add_predicate(Predicate("in", [
            Parameter("?pkg", "package"),
            Parameter("?veh", "vehicle")
        ]))
        
        # Load truck action
        domain.add_action(Action(
            name="load-truck",
            parameters=[
                Parameter("?pkg", "package"),
                Parameter("?truck", "truck"),
                Parameter("?loc", "location")
            ],
            precondition=Formula.make_and(
                Formula.make_atom("at", "?pkg", "?loc"),
                Formula.make_atom("at", "?truck", "?loc")
            ),
            effect=Effect.make_compound(
                Effect.make_positive("in", "?pkg", "?truck"),
                Effect.make_negative("at", "?pkg", "?loc")
            )
        ))
        
        # Drive truck action
        domain.add_action(Action(
            name="drive-truck",
            parameters=[
                Parameter("?truck", "truck"),
                Parameter("?from", "location"),
                Parameter("?to", "location"),
                Parameter("?city", "city")
            ],
            precondition=Formula.make_and(
                Formula.make_atom("at", "?truck", "?from"),
                Formula.make_atom("in-city", "?from", "?city"),
                Formula.make_atom("in-city", "?to", "?city")
            ),
            effect=Effect.make_compound(
                Effect.make_positive("at", "?truck", "?to"),
                Effect.make_negative("at", "?truck", "?from")
            )
        ))
        
        return domain
    
    def test_logistics_domain_emission(self, emitter, logistics_domain):
        """Test Logistics domain emission."""
        result = emitter.emit_domain(logistics_domain)
        
        # Check type hierarchy (truck and airplane are both subtypes of vehicle)
        assert "vehicle" in result.domain_code
        assert "truck" in result.domain_code
        assert "package" in result.domain_code
        
        # Check predicates
        assert "(in-city" in result.domain_code
        assert "(at" in result.domain_code
        assert "(in" in result.domain_code
        
        # Check actions
        assert "(:action load-truck" in result.domain_code
        assert "(:action drive-truck" in result.domain_code


class TestGripperIntegration:
    """Integration tests with Gripper domain."""
    
    @pytest.fixture
    def emitter(self):
        return PDDLEmitter()
    
    @pytest.fixture
    def gripper_domain(self):
        """Create Gripper domain."""
        domain = Domain(
            name="gripper",
            requirements=[PDDLRequirement.STRIPS, PDDLRequirement.TYPING]
        )
        
        # Types
        domain.add_type("ball")
        domain.add_type("gripper")
        domain.add_type("room")
        
        # Predicates
        domain.add_predicate(Predicate("at-robby", [Parameter("?r", "room")]))
        domain.add_predicate(Predicate("at", [
            Parameter("?b", "ball"),
            Parameter("?r", "room")
        ]))
        domain.add_predicate(Predicate("free", [Parameter("?g", "gripper")]))
        domain.add_predicate(Predicate("carry", [
            Parameter("?b", "ball"),
            Parameter("?g", "gripper")
        ]))
        
        # Pick action
        domain.add_action(Action(
            name="pick",
            parameters=[
                Parameter("?b", "ball"),
                Parameter("?r", "room"),
                Parameter("?g", "gripper")
            ],
            precondition=Formula.make_and(
                Formula.make_atom("at", "?b", "?r"),
                Formula.make_atom("at-robby", "?r"),
                Formula.make_atom("free", "?g")
            ),
            effect=Effect.make_compound(
                Effect.make_positive("carry", "?b", "?g"),
                Effect.make_negative("at", "?b", "?r"),
                Effect.make_negative("free", "?g")
            )
        ))
        
        # Drop action
        domain.add_action(Action(
            name="drop",
            parameters=[
                Parameter("?b", "ball"),
                Parameter("?r", "room"),
                Parameter("?g", "gripper")
            ],
            precondition=Formula.make_and(
                Formula.make_atom("carry", "?b", "?g"),
                Formula.make_atom("at-robby", "?r")
            ),
            effect=Effect.make_compound(
                Effect.make_positive("at", "?b", "?r"),
                Effect.make_positive("free", "?g"),
                Effect.make_negative("carry", "?b", "?g")
            )
        ))
        
        # Move action
        domain.add_action(Action(
            name="move",
            parameters=[
                Parameter("?from", "room"),
                Parameter("?to", "room")
            ],
            precondition=Formula.make_atom("at-robby", "?from"),
            effect=Effect.make_compound(
                Effect.make_positive("at-robby", "?to"),
                Effect.make_negative("at-robby", "?from")
            )
        ))
        
        return domain
    
    def test_gripper_domain_emission(self, emitter, gripper_domain):
        """Test Gripper domain emission."""
        result = emitter.emit_domain(gripper_domain)
        
        # Check structure
        assert "(define (domain gripper)" in result.domain_code
        assert "(:action pick" in result.domain_code
        assert "(:action drop" in result.domain_code
        assert "(:action move" in result.domain_code
        
        # Check manifest
        assert result.manifest['domain']['actions'] == 3
        assert result.manifest['domain']['predicates'] == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
