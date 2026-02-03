"""Tests for Expert System Emitters.

This module tests the CLIPS and Jess emitters.
"""

import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ir.rules import (
    Fact, FactTemplate,
    Rule, RuleBase,
    PatternCondition, TestCondition,
    AssertAction, RetractAction, ModifyAction, PrintoutAction,
    LiteralPattern, VariablePattern, WildcardPattern,
    FunctionDef,
)

from targets.expert_systems import CLIPSEmitter, JessEmitter


class TestCLIPSEmitter:
    """Tests for CLIPS emitter."""
    
    def test_basic_emission(self):
        """Test basic CLIPS emission."""
        rb = RuleBase(name="test")
        emitter = CLIPSEmitter()
        
        result = emitter.emit(rb)
        
        assert "CLIPS Expert System: test" in result.code
        assert result.manifest["dialect"] == "clips"
    
    def test_template_emission(self):
        """Test deftemplate emission."""
        rb = RuleBase(
            name="test",
            templates=[
                FactTemplate(
                    name="person",
                    slots=[("name", "string"), ("age", "i32")],
                    default_values={"age": 0}
                )
            ]
        )
        emitter = CLIPSEmitter()
        result = emitter.emit(rb)
        
        assert "(deftemplate person" in result.code
        assert "(slot name" in result.code
        assert "(type STRING)" in result.code
        assert "(slot age" in result.code
        assert "(type INTEGER)" in result.code
        assert "(default 0)" in result.code
    
    def test_rule_emission(self):
        """Test defrule emission."""
        rb = RuleBase(
            name="test",
            templates=[
                FactTemplate("person", [("name", "string")])
            ],
            rules=[
                Rule(
                    name="greet",
                    conditions=[
                        PatternCondition(
                            template_name="person",
                            patterns=[("name", VariablePattern("n"))]
                        )
                    ],
                    actions=[
                        PrintoutAction(items=["Hello ", "?n", "crlf"])
                    ],
                    salience=10,
                    documentation="Greet a person"
                )
            ]
        )
        emitter = CLIPSEmitter()
        result = emitter.emit(rb)
        
        assert "(defrule greet" in result.code
        assert '"Greet a person"' in result.code
        assert "(salience 10)" in result.code
        assert "(person (name ?n))" in result.code
        assert "=>" in result.code
        assert '(printout t "Hello " ?n crlf)' in result.code
    
    def test_initial_facts_emission(self):
        """Test deffacts emission."""
        rb = RuleBase(
            name="test",
            templates=[
                FactTemplate("person", [("name", "string")])
            ],
            initial_facts=[
                Fact(template_name="person", slots={"name": "John"})
            ]
        )
        emitter = CLIPSEmitter()
        result = emitter.emit(rb)
        
        assert "(deffacts test-initial-facts" in result.code
        assert '(person (name "John"))' in result.code
    
    def test_function_emission(self):
        """Test deffunction emission."""
        rb = RuleBase(
            name="test",
            functions={
                "square": FunctionDef(
                    name="square",
                    parameters=["x"],
                    body="(* ?x ?x)"
                )
            }
        )
        emitter = CLIPSEmitter()
        result = emitter.emit(rb)
        
        assert "(deffunction square (?x)" in result.code
        assert "(* ?x ?x)" in result.code
    
    def test_ordered_fact_pattern(self):
        """Test ordered fact pattern emission."""
        rb = RuleBase(
            name="test",
            rules=[
                Rule(
                    name="ordered-test",
                    conditions=[
                        PatternCondition(
                            template_name=None,
                            patterns=[],
                            ordered_patterns=[
                                LiteralPattern("fact"),
                                VariablePattern("x"),
                                WildcardPattern()
                            ]
                        )
                    ],
                    actions=[]
                )
            ]
        )
        emitter = CLIPSEmitter()
        result = emitter.emit(rb)
        
        assert "(fact ?x ?)" in result.code
    
    def test_actions_emission(self):
        """Test various action emissions."""
        rb = RuleBase(
            name="test",
            templates=[
                FactTemplate("data", [("value", "i32")])
            ],
            rules=[
                Rule(
                    name="action-test",
                    conditions=[
                        PatternCondition(
                            template_name="data",
                            patterns=[("value", VariablePattern("v"))],
                            binding_name="f"
                        )
                    ],
                    actions=[
                        AssertAction("data", {"value": 42}),
                        ModifyAction("f", {"value": 100}),
                        RetractAction("f")
                    ]
                )
            ]
        )
        emitter = CLIPSEmitter()
        result = emitter.emit(rb)
        
        assert "(assert (data (value 42)))" in result.code
        assert "(modify ?f (value 100))" in result.code
        assert "(retract ?f)" in result.code
    
    def test_manifest_generation(self):
        """Test manifest is generated correctly."""
        rb = RuleBase(
            name="test",
            rules=[Rule("r1", [], []), Rule("r2", [], [])],
            templates=[FactTemplate("t1", [])]
        )
        emitter = CLIPSEmitter()
        result = emitter.emit(rb)
        
        assert result.manifest["statistics"]["rules"] == 2
        assert result.manifest["statistics"]["templates"] == 1
        assert "hash" in result.manifest["output"]
        assert "manifest_hash" in result.manifest


class TestJessEmitter:
    """Tests for Jess emitter."""
    
    def test_basic_emission(self):
        """Test basic Jess emission."""
        rb = RuleBase(name="test")
        emitter = JessEmitter()
        
        result = emitter.emit(rb)
        
        assert "Jess Expert System: test" in result.code
        assert result.manifest["dialect"] == "jess"
    
    def test_template_emission(self):
        """Test deftemplate emission in Jess."""
        rb = RuleBase(
            name="test",
            templates=[
                FactTemplate(
                    name="person",
                    slots=[("name", "string"), ("age", "i64")],
                    default_values={"age": 0}
                )
            ]
        )
        emitter = JessEmitter()
        result = emitter.emit(rb)
        
        assert "(deftemplate person" in result.code
        assert "(slot name" in result.code
        assert "(type STRING)" in result.code
        assert "(slot age" in result.code
        assert "(type LONG)" in result.code  # Jess uses LONG for i64
    
    def test_java_integration_disabled(self):
        """Test Jess without Java integration."""
        rb = RuleBase(name="test")
        emitter = JessEmitter(java_integration=False)
        
        result = emitter.emit(rb)
        
        assert "Java Integration Enabled" not in result.code
        assert result.manifest["jess_info"]["java_integration"] is False
    
    def test_java_integration_enabled(self):
        """Test Jess with Java integration."""
        rb = RuleBase(name="test")
        emitter = JessEmitter(java_integration=True)
        emitter.add_java_import("java.util.ArrayList")
        
        result = emitter.emit(rb)
        
        assert "Java Integration Enabled" in result.code
        assert "(import java.util.ArrayList)" in result.code
        assert result.manifest["jess_info"]["java_integration"] is True
    
    def test_java_new_generation(self):
        """Test Java constructor call generation."""
        emitter = JessEmitter(java_integration=True)
        
        code = emitter.emit_java_new("ArrayList")
        assert code == "(new ArrayList)"
        
        code = emitter.emit_java_new("ArrayList", 10)
        assert code == "(new ArrayList 10)"
    
    def test_java_call_generation(self):
        """Test Java method call generation."""
        emitter = JessEmitter(java_integration=True)
        
        code = emitter.emit_java_call("list", "add", "item")
        assert code == '(call ?list add "item")'
        
        code = emitter.emit_java_call("list", "size")
        assert code == "(call ?list size)"
    
    def test_java_static_call_generation(self):
        """Test static Java method call generation."""
        emitter = JessEmitter(java_integration=True)
        
        code = emitter.emit_java_static_call("Math", "sqrt", 16)
        assert code == "(call Math sqrt 16)"
    
    def test_rule_with_auto_focus(self):
        """Test rule with auto-focus declaration."""
        rb = RuleBase(
            name="test",
            rules=[
                Rule(
                    name="auto-focus-rule",
                    conditions=[],
                    actions=[],
                    auto_focus=True,
                    salience=50
                )
            ]
        )
        emitter = JessEmitter()
        result = emitter.emit(rb)
        
        assert "(auto-focus TRUE)" in result.code
        assert "(salience 50)" in result.code
    
    def test_manifest_has_jess_info(self):
        """Test manifest includes Jess-specific info."""
        rb = RuleBase(
            name="test",
            rules=[Rule("r1", [], [])]
        )
        emitter = JessEmitter(java_integration=True)
        emitter.add_java_import("java.util.*")
        result = emitter.emit(rb)
        
        assert "jess_info" in result.manifest
        assert result.manifest["jess_info"]["requires_jess_version"] == "7.0+"
        assert "java.util.*" in result.manifest["jess_info"]["java_imports"]


class TestEmitterIntegration:
    """Integration tests for emitters."""
    
    def test_animal_identification_clips(self):
        """Test complete animal identification example in CLIPS."""
        rb = RuleBase(
            name="animal-identification",
            templates=[
                FactTemplate("animal", [("name", "symbol")])
            ],
            rules=[
                Rule(
                    name="identify-dog",
                    conditions=[
                        PatternCondition(
                            template_name="animal",
                            patterns=[("name", VariablePattern("x"))]
                        ),
                        PatternCondition(
                            template_name=None,
                            patterns=[],
                            ordered_patterns=[
                                LiteralPattern("says"),
                                VariablePattern("x"),
                                LiteralPattern("bark")
                            ]
                        )
                    ],
                    actions=[
                        PrintoutAction(items=["?x", " is a dog", "crlf"])
                    ],
                    documentation="Identify dogs by their bark"
                )
            ],
            initial_facts=[
                Fact(template_name="animal", slots={"name": "fido"}),
                Fact(values=("says", "fido", "bark"))
            ]
        )
        
        emitter = CLIPSEmitter()
        result = emitter.emit(rb)
        
        # Verify complete example
        assert "(deftemplate animal" in result.code
        assert "(defrule identify-dog" in result.code
        assert "(deffacts animal-identification-initial-facts" in result.code
        assert result.manifest["statistics"]["rules"] == 1
        assert result.manifest["statistics"]["templates"] == 1
        assert result.manifest["statistics"]["initial_facts"] == 2
    
    def test_same_rulebase_different_emitters(self):
        """Test same rulebase emits to both CLIPS and Jess."""
        rb = RuleBase(
            name="test",
            templates=[
                FactTemplate("data", [("value", "i32")])
            ],
            rules=[
                Rule(
                    name="simple-rule",
                    conditions=[
                        PatternCondition(
                            template_name="data",
                            patterns=[("value", VariablePattern("v"))]
                        )
                    ],
                    actions=[
                        PrintoutAction(items=["Value: ", "?v", "crlf"])
                    ]
                )
            ]
        )
        
        clips_emitter = CLIPSEmitter()
        jess_emitter = JessEmitter()
        
        clips_result = clips_emitter.emit(rb)
        jess_result = jess_emitter.emit(rb)
        
        # Both should have similar structure
        assert "(deftemplate data" in clips_result.code
        assert "(deftemplate data" in jess_result.code
        assert "(defrule simple-rule" in clips_result.code
        assert "(defrule simple-rule" in jess_result.code
        
        # But different metadata
        assert clips_result.manifest["dialect"] == "clips"
        assert jess_result.manifest["dialect"] == "jess"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
