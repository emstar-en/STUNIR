"""Tests for Rule-based IR.

This module tests the core Rule IR data structures including
facts, templates, rules, and pattern matching.
"""

import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ir.rules import (
    PatternType, ConditionType, ActionType, ConflictResolutionStrategy,
    Fact, FactTemplate,
    Rule, RuleBase,
    PatternCondition, TestCondition, CompositeCondition,
    AssertAction, RetractAction, ModifyAction, BindAction, PrintoutAction,
    LiteralPattern, VariablePattern, WildcardPattern, MultifieldPattern,
    PatternMatcher,
    WorkingMemory,
    FunctionDef,
)


class TestFact:
    """Tests for Fact class."""
    
    def test_ordered_fact_creation(self):
        """Test creating an ordered fact."""
        fact = Fact(values=("animal", "dog", "fido"))
        assert fact.is_ordered()
        assert not fact.is_template()
        assert fact.values == ("animal", "dog", "fido")
    
    def test_template_fact_creation(self):
        """Test creating a template fact."""
        fact = Fact(
            template_name="person",
            slots={"name": "John", "age": 30}
        )
        assert fact.is_template()
        assert not fact.is_ordered()
        assert fact.slots["name"] == "John"
        assert fact.slots["age"] == 30
    
    def test_fact_hash(self):
        """Test fact hashing is deterministic."""
        fact1 = Fact(template_name="person", slots={"name": "John", "age": 30})
        fact2 = Fact(template_name="person", slots={"name": "John", "age": 30})
        assert fact1.get_hash() == fact2.get_hash()
    
    def test_fact_equality(self):
        """Test fact equality based on content."""
        fact1 = Fact(template_name="person", slots={"name": "John"})
        fact2 = Fact(template_name="person", slots={"name": "John"})
        assert fact1 == fact2
    
    def test_fact_str(self):
        """Test fact string representation."""
        fact = Fact(fact_id=1, values=("hello", "world"))
        assert "f-1" in str(fact)
        assert "hello" in str(fact)


class TestFactTemplate:
    """Tests for FactTemplate class."""
    
    def test_template_creation(self):
        """Test creating a fact template."""
        template = FactTemplate(
            name="person",
            slots=[("name", "string"), ("age", "i32")],
            default_values={"age": 0}
        )
        assert template.name == "person"
        assert len(template.slots) == 2
        assert template.default_values.get("age") == 0
    
    def test_template_validate_fact(self):
        """Test template validates facts."""
        template = FactTemplate(
            name="person",
            slots=[("name", "string"), ("age", "i32")]
        )
        valid_fact = Fact(template_name="person", slots={"name": "John", "age": 30})
        invalid_fact = Fact(template_name="animal", slots={"species": "dog"})
        
        assert template.validate_fact(valid_fact)
        assert not template.validate_fact(invalid_fact)
    
    def test_template_create_fact(self):
        """Test creating fact from template."""
        template = FactTemplate(
            name="person",
            slots=[("name", "string"), ("age", "i32")],
            default_values={"age": 0}
        )
        fact = template.create_fact(name="Jane")
        assert fact.template_name == "person"
        assert fact.slots["name"] == "Jane"
        assert fact.slots["age"] == 0  # Default value


class TestPatternMatching:
    """Tests for pattern matching."""
    
    def test_literal_pattern_match(self):
        """Test literal pattern matching."""
        matcher = PatternMatcher()
        pattern = PatternCondition(
            template_name=None,
            patterns=[],
            ordered_patterns=[LiteralPattern("hello"), LiteralPattern("world")]
        )
        fact = Fact(values=("hello", "world"))
        
        result = matcher.match_pattern(pattern, fact, {})
        assert result is not None
    
    def test_literal_pattern_no_match(self):
        """Test literal pattern not matching."""
        matcher = PatternMatcher()
        pattern = PatternCondition(
            template_name=None,
            patterns=[],
            ordered_patterns=[LiteralPattern("hello"), LiteralPattern("world")]
        )
        fact = Fact(values=("hello", "there"))
        
        result = matcher.match_pattern(pattern, fact, {})
        assert result is None
    
    def test_variable_pattern_binding(self):
        """Test variable pattern binds values."""
        matcher = PatternMatcher()
        pattern = PatternCondition(
            template_name=None,
            patterns=[],
            ordered_patterns=[LiteralPattern("person"), VariablePattern("name")]
        )
        fact = Fact(values=("person", "John"))
        
        result = matcher.match_pattern(pattern, fact, {})
        assert result is not None
        assert result["name"] == "John"
    
    def test_variable_pattern_consistency(self):
        """Test same variable must match same value."""
        matcher = PatternMatcher()
        pattern = PatternCondition(
            template_name=None,
            patterns=[],
            ordered_patterns=[VariablePattern("x"), VariablePattern("x")]
        )
        
        fact_same = Fact(values=("a", "a"))
        fact_diff = Fact(values=("a", "b"))
        
        assert matcher.match_pattern(pattern, fact_same, {}) is not None
        assert matcher.match_pattern(pattern, fact_diff, {}) is None
    
    def test_wildcard_pattern(self):
        """Test wildcard matches anything."""
        matcher = PatternMatcher()
        pattern = PatternCondition(
            template_name=None,
            patterns=[],
            ordered_patterns=[LiteralPattern("test"), WildcardPattern()]
        )
        
        fact1 = Fact(values=("test", "anything"))
        fact2 = Fact(values=("test", 42))
        
        assert matcher.match_pattern(pattern, fact1, {}) is not None
        assert matcher.match_pattern(pattern, fact2, {}) is not None
    
    def test_template_pattern_match(self):
        """Test template pattern matching with slots."""
        matcher = PatternMatcher()
        pattern = PatternCondition(
            template_name="person",
            patterns=[("name", VariablePattern("n")), ("age", VariablePattern("a"))]
        )
        fact = Fact(template_name="person", slots={"name": "John", "age": 30})
        
        result = matcher.match_pattern(pattern, fact, {})
        assert result is not None
        assert result["n"] == "John"
        assert result["a"] == 30


class TestWorkingMemory:
    """Tests for WorkingMemory class."""
    
    def test_assert_fact(self):
        """Test asserting facts."""
        wm = WorkingMemory()
        fact = Fact(values=("test", "fact"))
        
        fact_id = wm.assert_fact(fact)
        assert fact_id == 1
        assert len(wm) == 1
    
    def test_duplicate_fact_not_added(self):
        """Test duplicate facts return same ID."""
        wm = WorkingMemory()
        fact1 = Fact(values=("test", "fact"))
        fact2 = Fact(values=("test", "fact"))
        
        id1 = wm.assert_fact(fact1)
        id2 = wm.assert_fact(fact2)
        
        assert id1 == id2
        assert len(wm) == 1
    
    def test_retract_fact(self):
        """Test retracting facts."""
        wm = WorkingMemory()
        fact = Fact(values=("test",))
        fact_id = wm.assert_fact(fact)
        
        assert wm.retract_fact(fact_id)
        assert len(wm) == 0
        assert not wm.retract_fact(fact_id)  # Already retracted
    
    def test_modify_fact(self):
        """Test modifying facts."""
        wm = WorkingMemory()
        fact = Fact(template_name="person", slots={"name": "John", "age": 30})
        fact_id = wm.assert_fact(fact)
        
        assert wm.modify_fact(fact_id, {"age": 31})
        modified = wm.get_fact(fact_id)
        assert modified.slots["age"] == 31
    
    def test_get_facts_by_template(self):
        """Test getting facts by template."""
        wm = WorkingMemory()
        wm.assert_fact(Fact(template_name="person", slots={"name": "John"}))
        wm.assert_fact(Fact(template_name="person", slots={"name": "Jane"}))
        wm.assert_fact(Fact(template_name="animal", slots={"species": "dog"}))
        
        person_facts = list(wm.get_facts_by_template("person"))
        assert len(person_facts) == 2
    
    def test_fact_listener(self):
        """Test fact change listeners."""
        wm = WorkingMemory()
        events = []
        
        def listener(event, fact):
            events.append((event, fact.values if fact.is_ordered() else fact.slots))
        
        wm.add_listener(listener)
        wm.assert_fact(Fact(values=("test",)))
        
        assert len(events) == 1
        assert events[0][0] == "assert"


class TestRule:
    """Tests for Rule class."""
    
    def test_rule_creation(self):
        """Test creating a rule."""
        rule = Rule(
            name="test-rule",
            conditions=[
                PatternCondition(None, [], [LiteralPattern("test")])
            ],
            actions=[
                PrintoutAction(items=["Hello", "crlf"])
            ],
            salience=10
        )
        assert rule.name == "test-rule"
        assert rule.salience == 10
        assert len(rule.conditions) == 1
        assert len(rule.actions) == 1
    
    def test_rule_bound_variables(self):
        """Test getting bound variables from rule."""
        rule = Rule(
            name="test",
            conditions=[
                PatternCondition(
                    template_name="person",
                    patterns=[("name", VariablePattern("n"))],
                    binding_name="f"
                )
            ],
            actions=[]
        )
        
        bound = rule.get_bound_variables()
        assert "n" in bound
        assert "f" in bound


class TestRuleBase:
    """Tests for RuleBase class."""
    
    def test_rulebase_creation(self):
        """Test creating a rule base."""
        rb = RuleBase(
            name="test-kb",
            templates=[
                FactTemplate("person", [("name", "string")])
            ],
            rules=[
                Rule("r1", [], [])
            ]
        )
        assert rb.name == "test-kb"
        assert len(rb.templates) == 1
        assert len(rb.rules) == 1
    
    def test_rulebase_add_rule(self):
        """Test adding rules to rule base."""
        rb = RuleBase(name="test")
        rb.add_rule(Rule("r1", [], []))
        rb.add_rule(Rule("r2", [], []))
        
        assert len(rb.rules) == 2
        assert rb.get_rule_by_name("r1") is not None
        assert rb.get_rule_by_name("r3") is None
    
    def test_rulebase_statistics(self):
        """Test rule base statistics."""
        rb = RuleBase(
            name="test",
            rules=[Rule("r1", [], []), Rule("r2", [], [])],
            templates=[FactTemplate("t1", [])],
            functions={"f1": FunctionDef("f1", [], "body")}
        )
        
        stats = rb.get_statistics()
        assert stats["rules"] == 2
        assert stats["templates"] == 1
        assert stats["functions"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
