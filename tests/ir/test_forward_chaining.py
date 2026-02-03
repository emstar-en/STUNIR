"""Tests for Forward Chaining Engine.

This module tests the forward chaining inference engine including
rule activation, conflict resolution, and action execution.
"""

import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ir.rules import (
    Fact, FactTemplate,
    Rule, RuleBase,
    PatternCondition, TestCondition, CompositeCondition,
    AssertAction, RetractAction, ModifyAction, PrintoutAction, HaltAction,
    LiteralPattern, VariablePattern,
    ForwardChainingEngine, Agenda, Activation,
    ConflictResolutionStrategy, ConditionType,
)


class TestAgenda:
    """Tests for Agenda class."""
    
    def test_agenda_add_activation(self):
        """Test adding activations to agenda."""
        agenda = Agenda()
        rule = Rule("test", [], [], salience=10)
        
        agenda.add_activation(rule, {}, [1])
        assert len(agenda) == 1
    
    def test_agenda_salience_ordering(self):
        """Test activations are ordered by salience."""
        agenda = Agenda()
        low_rule = Rule("low", [], [], salience=10)
        high_rule = Rule("high", [], [], salience=100)
        
        agenda.add_activation(low_rule, {}, [1])
        agenda.add_activation(high_rule, {}, [2])
        
        # High salience should come first
        first = agenda.get_next()
        assert first.rule.name == "high"
    
    def test_agenda_remove_activation(self):
        """Test removing activations."""
        agenda = Agenda()
        rule = Rule("test", [], [])
        
        agenda.add_activation(rule, {}, [1, 2])
        assert len(agenda) == 1
        
        agenda.remove_activation("test", [1])
        assert len(agenda) == 0
    
    def test_agenda_no_duplicates(self):
        """Test duplicate activations not added."""
        agenda = Agenda()
        rule = Rule("test", [], [])
        
        agenda.add_activation(rule, {}, [1])
        agenda.add_activation(rule, {}, [1])  # Duplicate
        
        assert len(agenda) == 1


class TestForwardChainingEngine:
    """Tests for ForwardChainingEngine class."""
    
    def test_engine_reset(self):
        """Test engine reset initializes facts."""
        rb = RuleBase(
            name="test",
            initial_facts=[Fact(values=("initial", "fact"))]
        )
        engine = ForwardChainingEngine(rb)
        engine.reset()
        
        assert len(engine.working_memory) == 1
    
    def test_simple_rule_firing(self):
        """Test a simple rule fires correctly."""
        rb = RuleBase(
            name="test",
            rules=[
                Rule(
                    name="greet",
                    conditions=[
                        PatternCondition(
                            template_name=None,
                            patterns=[],
                            ordered_patterns=[LiteralPattern("person"), VariablePattern("name")]
                        )
                    ],
                    actions=[
                        AssertAction(
                            fact_template=None,
                            ordered_values=["greeted", "?name"]
                        )
                    ]
                )
            ]
        )
        
        engine = ForwardChainingEngine(rb)
        engine.reset()
        
        # Assert a triggering fact
        engine.working_memory.assert_fact(Fact(values=("person", "John")))
        
        # Run the engine
        fired = engine.run()
        
        assert fired >= 1
        
        # Check that new fact was asserted
        all_facts = list(engine.working_memory.get_all_facts())
        values_list = [f.values for f in all_facts if f.is_ordered()]
        assert ("greeted", "John") in values_list
    
    def test_salience_based_firing_order(self):
        """Test rules fire in salience order."""
        fired_order = []
        
        rb = RuleBase(
            name="test",
            rules=[
                Rule(
                    name="low-priority",
                    conditions=[
                        PatternCondition(None, [], [LiteralPattern("trigger")])
                    ],
                    actions=[
                        AssertAction(None, {}, ["low-fired"])
                    ],
                    salience=10
                ),
                Rule(
                    name="high-priority",
                    conditions=[
                        PatternCondition(None, [], [LiteralPattern("trigger")])
                    ],
                    actions=[
                        AssertAction(None, {}, ["high-fired"])
                    ],
                    salience=100
                )
            ]
        )
        
        engine = ForwardChainingEngine(rb)
        engine.reset()
        engine.working_memory.assert_fact(Fact(values=("trigger",)))
        engine.run()
        
        # High priority should fire first
        fired_rules = engine.get_fired_rules()
        assert fired_rules[0][0] == "high-priority"
    
    def test_rule_chaining(self):
        """Test rules can trigger other rules."""
        rb = RuleBase(
            name="animal-id",
            rules=[
                Rule(
                    name="identify-mammal",
                    conditions=[
                        PatternCondition(None, [], [LiteralPattern("has"), VariablePattern("x"), LiteralPattern("hair")])
                    ],
                    actions=[
                        AssertAction(None, {}, ["is", "?x", "mammal"])
                    ]
                ),
                Rule(
                    name="identify-dog",
                    conditions=[
                        PatternCondition(None, [], [LiteralPattern("is"), VariablePattern("x"), LiteralPattern("mammal")]),
                        PatternCondition(None, [], [LiteralPattern("says"), VariablePattern("x"), LiteralPattern("bark")])
                    ],
                    actions=[
                        AssertAction(None, {}, ["is", "?x", "dog"])
                    ]
                )
            ]
        )
        
        engine = ForwardChainingEngine(rb)
        engine.reset()
        
        # Assert initial facts
        engine.working_memory.assert_fact(Fact(values=("has", "fido", "hair")))
        engine.working_memory.assert_fact(Fact(values=("says", "fido", "bark")))
        
        engine.run()
        
        # Check that dog was identified through chaining
        all_facts = list(engine.working_memory.get_all_facts())
        values_list = [f.values for f in all_facts if f.is_ordered()]
        assert ("is", "fido", "mammal") in values_list
        assert ("is", "fido", "dog") in values_list
    
    def test_retract_action(self):
        """Test retract action removes facts."""
        rb = RuleBase(
            name="test",
            rules=[
                Rule(
                    name="consume",
                    conditions=[
                        PatternCondition(
                            template_name=None,
                            patterns=[],
                            ordered_patterns=[LiteralPattern("consume-me")],
                            binding_name="f"
                        )
                    ],
                    actions=[
                        RetractAction("f"),
                        AssertAction(None, {}, ["consumed"])
                    ]
                )
            ]
        )
        
        engine = ForwardChainingEngine(rb)
        engine.reset()
        engine.working_memory.assert_fact(Fact(values=("consume-me",)))
        engine.run()
        
        # Original fact should be retracted
        all_facts = list(engine.working_memory.get_all_facts())
        values_list = [f.values for f in all_facts if f.is_ordered()]
        assert ("consume-me",) not in values_list
        assert ("consumed",) in values_list
    
    def test_halt_action(self):
        """Test halt action stops execution."""
        rb = RuleBase(
            name="test",
            rules=[
                Rule(
                    name="halt-rule",
                    conditions=[
                        PatternCondition(None, [], [LiteralPattern("halt-trigger")])
                    ],
                    actions=[
                        AssertAction(None, {}, ["before-halt"]),
                        HaltAction(),
                        AssertAction(None, {}, ["after-halt"])  # Should not execute
                    ],
                    salience=100
                ),
                Rule(
                    name="other-rule",
                    conditions=[
                        PatternCondition(None, [], [LiteralPattern("halt-trigger")])
                    ],
                    actions=[
                        AssertAction(None, {}, ["other-fired"])
                    ],
                    salience=10  # Lower priority, should not fire after halt
                )
            ]
        )
        
        engine = ForwardChainingEngine(rb)
        engine.reset()
        engine.working_memory.assert_fact(Fact(values=("halt-trigger",)))
        engine.run()
        
        all_facts = list(engine.working_memory.get_all_facts())
        values_list = [f.values for f in all_facts if f.is_ordered()]
        
        assert ("before-halt",) in values_list
        assert ("after-halt",) not in values_list
        assert ("other-fired",) not in values_list
    
    def test_template_facts(self):
        """Test rules work with template facts."""
        rb = RuleBase(
            name="test",
            templates=[
                FactTemplate(
                    name="person",
                    slots=[("name", "string"), ("age", "i32")]
                )
            ],
            rules=[
                Rule(
                    name="adult-check",
                    conditions=[
                        PatternCondition(
                            template_name="person",
                            patterns=[
                                ("name", VariablePattern("n")),
                                ("age", VariablePattern("a"))
                            ]
                        ),
                        TestCondition("?a >= 18")
                    ],
                    actions=[
                        AssertAction(None, {}, ["adult", "?n"])
                    ]
                )
            ]
        )
        
        engine = ForwardChainingEngine(rb)
        engine.reset()
        engine.working_memory.assert_fact(
            Fact(template_name="person", slots={"name": "John", "age": 25})
        )
        engine.working_memory.assert_fact(
            Fact(template_name="person", slots={"name": "Child", "age": 10})
        )
        engine.run()
        
        all_facts = list(engine.working_memory.get_all_facts())
        values_list = [f.values for f in all_facts if f.is_ordered()]
        
        assert ("adult", "John") in values_list
        assert ("adult", "Child") not in values_list
    
    def test_engine_statistics(self):
        """Test engine statistics reporting."""
        rb = RuleBase(
            name="test",
            rules=[Rule("r1", [], [])]
        )
        engine = ForwardChainingEngine(rb)
        engine.reset()
        engine.working_memory.assert_fact(Fact(values=("test",)))
        
        stats = engine.get_statistics()
        assert "facts_count" in stats
        assert "rules_count" in stats
        assert stats["rules_count"] == 1
    
    def test_step_execution(self):
        """Test single-step execution."""
        rb = RuleBase(
            name="test",
            rules=[
                Rule(
                    name="step-rule",
                    conditions=[
                        PatternCondition(None, [], [LiteralPattern("step")])
                    ],
                    actions=[
                        AssertAction(None, {}, ["stepped"])
                    ]
                )
            ]
        )
        
        engine = ForwardChainingEngine(rb)
        engine.reset()
        engine.working_memory.assert_fact(Fact(values=("step",)))
        
        # Execute one step
        result = engine.step()
        assert result is True
        
        # No more rules to fire
        result = engine.step()
        assert result is False


class TestNOTCondition:
    """Tests for NOT (negated) conditions."""
    
    def test_not_condition(self):
        """Test NOT condition succeeds when fact is absent."""
        rb = RuleBase(
            name="test",
            rules=[
                Rule(
                    name="no-blocker",
                    conditions=[
                        PatternCondition(None, [], [LiteralPattern("trigger")]),
                        CompositeCondition(
                            operator=ConditionType.NOT,
                            children=[
                                PatternCondition(None, [], [LiteralPattern("blocker")])
                            ]
                        )
                    ],
                    actions=[
                        AssertAction(None, {}, ["fired"])
                    ]
                )
            ]
        )
        
        engine = ForwardChainingEngine(rb)
        engine.reset()
        engine.working_memory.assert_fact(Fact(values=("trigger",)))
        # No "blocker" fact
        engine.run()
        
        all_facts = list(engine.working_memory.get_all_facts())
        values_list = [f.values for f in all_facts if f.is_ordered()]
        assert ("fired",) in values_list
    
    def test_not_condition_blocked(self):
        """Test NOT condition fails when fact is present.
        
        Note: The blocker must be asserted BEFORE the trigger to prevent
        activation, since activations are created at assertion time.
        """
        rb = RuleBase(
            name="test",
            rules=[
                Rule(
                    name="no-blocker",
                    conditions=[
                        PatternCondition(None, [], [LiteralPattern("trigger")]),
                        CompositeCondition(
                            operator=ConditionType.NOT,
                            children=[
                                PatternCondition(None, [], [LiteralPattern("blocker")])
                            ]
                        )
                    ],
                    actions=[
                        AssertAction(None, {}, ["fired"])
                    ]
                )
            ]
        )
        
        engine = ForwardChainingEngine(rb)
        engine.reset()
        # Assert blocker BEFORE trigger so NOT condition is evaluated with blocker present
        engine.working_memory.assert_fact(Fact(values=("blocker",)))
        engine.working_memory.assert_fact(Fact(values=("trigger",)))
        engine.run()
        
        all_facts = list(engine.working_memory.get_all_facts())
        values_list = [f.values for f in all_facts if f.is_ordered()]
        assert ("fired",) not in values_list


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
