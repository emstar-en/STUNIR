#!/usr/bin/env python3
"""Tests for STUNIR Tau Prolog Emitter.

Comprehensive test suite for Tau Prolog code generation including
module declarations, facts, rules, JavaScript interop, and DOM predicates.

Part of Phase 5D-4: Extended Prolog Targets (Tau Prolog).
"""

import pytest
import sys
import os
import re

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from targets.prolog.tau_prolog.emitter import (
    TauPrologEmitter, TauPrologConfig, TauPrologEmitterResult
)
from targets.prolog.tau_prolog.types import (
    TauPrologTypeMapper, TAU_PROLOG_TYPES, DOM_PREDICATES, JS_PREDICATES,
    TAU_LIBRARIES, LISTS_PREDICATES
)


class TestTauPrologEmitterBasic:
    """Basic emitter functionality tests."""
    
    def test_emitter_creation(self):
        """Test emitter can be created with default config."""
        emitter = TauPrologEmitter()
        assert emitter.config.module_prefix == "stunir"
        assert emitter.config.emit_module is True
        assert emitter.config.enable_js_interop is True
    
    def test_emitter_with_custom_config(self):
        """Test emitter with custom configuration."""
        config = TauPrologConfig(
            module_prefix="myapp",
            emit_comments=False,
            emit_type_hints=False,
            target_runtime="node"
        )
        emitter = TauPrologEmitter(config)
        assert emitter.config.module_prefix == "myapp"
        assert emitter.config.emit_comments is False
        assert emitter.config.target_runtime == "node"
    
    def test_emit_result_structure(self):
        """Test TauPrologEmitterResult has correct structure."""
        ir = {"module": "test", "clauses": []}
        emitter = TauPrologEmitter()
        result = emitter.emit(ir)
        
        assert isinstance(result, TauPrologEmitterResult)
        assert isinstance(result.code, str)
        assert isinstance(result.module_name, str)
        assert isinstance(result.predicates, list)
        assert isinstance(result.sha256, str)
        assert isinstance(result.emit_time, float)
        assert isinstance(result.libraries_used, list)
        assert isinstance(result.has_dom, bool)
        assert isinstance(result.has_js_interop, bool)
        assert isinstance(result.target_runtime, str)
    
    def test_dialect_and_extension(self):
        """Test dialect and file extension constants."""
        emitter = TauPrologEmitter()
        assert emitter.DIALECT == "tau-prolog"
        assert emitter.FILE_EXTENSION == ".pl"


class TestSimpleFactEmission:
    """Tests for emitting simple facts."""
    
    def test_simple_fact_emission(self):
        """TC-TAU-001: Test emitting simple facts."""
        ir = {
            "module": "family",
            "clauses": [
                {"kind": "fact", "predicate": "parent", "args": [
                    {"kind": "atom", "value": "tom"},
                    {"kind": "atom", "value": "bob"}
                ]},
                {"kind": "fact", "predicate": "parent", "args": [
                    {"kind": "atom", "value": "tom"},
                    {"kind": "atom", "value": "liz"}
                ]}
            ]
        }
        
        emitter = TauPrologEmitter()
        result = emitter.emit(ir)
        
        assert "parent(tom, bob)." in result.code
        assert "parent(tom, liz)." in result.code
        assert result.module_name == "stunir_family"
    
    def test_fact_no_args(self):
        """Test fact with no arguments."""
        ir = {
            "module": "test",
            "clauses": [
                {"kind": "fact", "predicate": "sunny", "args": []}
            ]
        }
        
        emitter = TauPrologEmitter()
        result = emitter.emit(ir)
        
        assert "sunny." in result.code
    
    def test_fact_with_numbers(self):
        """Test fact with numeric arguments."""
        ir = {
            "module": "math",
            "clauses": [
                {"kind": "fact", "predicate": "add", "args": [
                    {"kind": "number", "value": 1},
                    {"kind": "number", "value": 2},
                    {"kind": "number", "value": 3}
                ]}
            ]
        }
        
        emitter = TauPrologEmitter()
        result = emitter.emit(ir)
        
        assert "add(1, 2, 3)." in result.code
    
    def test_fact_with_strings(self):
        """Test fact with string arguments."""
        ir = {
            "module": "text",
            "clauses": [
                {"kind": "fact", "predicate": "message", "args": [
                    {"kind": "string_term", "value": "Hello, World!"}
                ]}
            ]
        }
        
        emitter = TauPrologEmitter()
        result = emitter.emit(ir)
        
        assert 'message("Hello, World!").' in result.code


class TestRuleEmission:
    """Tests for emitting rules."""
    
    def test_simple_rule(self):
        """TC-TAU-002: Test emitting simple rules."""
        ir = {
            "module": "family",
            "clauses": [
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
        
        emitter = TauPrologEmitter()
        result = emitter.emit(ir)
        
        assert "grandparent(X, Z) :-" in result.code
        assert "parent(X, Y)" in result.code
        assert "parent(Y, Z)" in result.code
    
    def test_rule_with_single_goal(self):
        """Test rule with single body goal."""
        ir = {
            "module": "test",
            "clauses": [
                {"kind": "rule",
                 "head": {"kind": "compound", "functor": "human", "args": [
                     {"kind": "variable", "name": "X"}
                 ]},
                 "body": [
                     {"kind": "compound", "functor": "mortal", "args": [
                         {"kind": "variable", "name": "X"}
                     ]}
                 ]}
            ]
        }
        
        emitter = TauPrologEmitter()
        result = emitter.emit(ir)
        
        assert "human(X) :-" in result.code
        assert "mortal(X)" in result.code


class TestModuleDeclaration:
    """Tests for module declarations."""
    
    def test_empty_module(self):
        """Test module with no exports."""
        config = TauPrologConfig(emit_comments=False)
        emitter = TauPrologEmitter(config)
        
        ir = {"module": "empty", "clauses": []}
        result = emitter.emit(ir)
        
        assert ":- module(stunir_empty, [])." in result.code
    
    def test_module_with_exports(self):
        """Test module with explicit exports."""
        ir = {
            "module": "test",
            "exports": [
                {"predicate": "hello", "arity": 1},
                {"predicate": "world", "arity": 0}
            ],
            "clauses": []
        }
        
        emitter = TauPrologEmitter()
        result = emitter.emit(ir)
        
        assert "hello/1" in result.code
        assert "world/0" in result.code
    
    def test_module_prefix(self):
        """Test custom module prefix."""
        config = TauPrologConfig(module_prefix="myapp", emit_comments=False)
        emitter = TauPrologEmitter(config)
        
        ir = {"module": "demo", "clauses": []}
        result = emitter.emit(ir)
        
        assert ":- module(myapp_demo" in result.code


class TestLibraryImports:
    """Tests for library imports."""
    
    def test_default_libraries(self):
        """TC-TAU-003: Test default library inclusion."""
        config = TauPrologConfig(default_libraries=['lists'])
        emitter = TauPrologEmitter(config)
        
        ir = {"module": "test", "clauses": []}
        result = emitter.emit(ir)
        
        assert ":- use_module(library(lists))." in result.code
        assert "lists" in result.libraries_used
    
    def test_dom_library_detection(self):
        """Test DOM library auto-detection."""
        config = TauPrologConfig(enable_dom=True, default_libraries=[])
        emitter = TauPrologEmitter(config)
        
        ir = {
            "module": "ui",
            "clauses": [
                {"kind": "rule",
                 "head": {"kind": "compound", "functor": "update", "args": []},
                 "body": [
                     {"kind": "compound", "functor": "get_by_id", "args": [
                         {"kind": "atom", "value": "output"},
                         {"kind": "variable", "name": "E"}
                     ]}
                 ]}
            ]
        }
        
        result = emitter.emit(ir)
        
        assert ":- use_module(library(dom))." in result.code
        assert "dom" in result.libraries_used
        assert result.has_dom is True
    
    def test_js_library_detection(self):
        """Test JavaScript library auto-detection."""
        config = TauPrologConfig(enable_js_interop=True, default_libraries=[])
        emitter = TauPrologEmitter(config)
        
        ir = {
            "module": "jstest",
            "clauses": [
                {"kind": "rule",
                 "head": {"kind": "compound", "functor": "test", "args": []},
                 "body": [
                     {"kind": "compound", "functor": "global", "args": [
                         {"kind": "atom", "value": "console"},
                         {"kind": "variable", "name": "C"}
                     ]}
                 ]}
            ]
        }
        
        result = emitter.emit(ir)
        
        assert ":- use_module(library(js))." in result.code
        assert "js" in result.libraries_used
        assert result.has_js_interop is True


class TestDOMPredicates:
    """Tests for DOM predicate emission."""
    
    def test_get_by_id(self):
        """TC-TAU-004: Test get_by_id predicate."""
        config = TauPrologConfig(enable_dom=True, default_libraries=[])
        emitter = TauPrologEmitter(config)
        
        ir = {
            "module": "ui",
            "clauses": [
                {"kind": "rule",
                 "head": {"kind": "compound", "functor": "find_element", "args": [
                     {"kind": "variable", "name": "E"}
                 ]},
                 "body": [
                     {"kind": "compound", "functor": "get_by_id", "args": [
                         {"kind": "atom", "value": "main"},
                         {"kind": "variable", "name": "E"}
                     ]}
                 ]}
            ]
        }
        
        result = emitter.emit(ir)
        
        assert "get_by_id(main, E)" in result.code
        assert result.has_dom is True
    
    def test_set_html(self):
        """Test set_html predicate."""
        config = TauPrologConfig(enable_dom=True, default_libraries=[])
        emitter = TauPrologEmitter(config)
        
        ir = {
            "module": "ui",
            "clauses": [
                {"kind": "rule",
                 "head": {"kind": "compound", "functor": "set_content", "args": [
                     {"kind": "variable", "name": "E"},
                     {"kind": "variable", "name": "Text"}
                 ]},
                 "body": [
                     {"kind": "compound", "functor": "set_html", "args": [
                         {"kind": "variable", "name": "E"},
                         {"kind": "variable", "name": "Text"}
                     ]}
                 ]}
            ]
        }
        
        result = emitter.emit(ir)
        
        assert "set_html(E, Text)" in result.code


class TestJavaScriptInterop:
    """Tests for JavaScript interoperability."""
    
    def test_global_predicate(self):
        """TC-TAU-005: Test global predicate for JS objects."""
        config = TauPrologConfig(enable_js_interop=True, default_libraries=[])
        emitter = TauPrologEmitter(config)
        
        ir = {
            "module": "jstest",
            "clauses": [
                {"kind": "rule",
                 "head": {"kind": "compound", "functor": "get_console", "args": [
                     {"kind": "variable", "name": "C"}
                 ]},
                 "body": [
                     {"kind": "compound", "functor": "global", "args": [
                         {"kind": "atom", "value": "console"},
                         {"kind": "variable", "name": "C"}
                     ]}
                 ]}
            ]
        }
        
        result = emitter.emit(ir)
        
        assert "global(console, C)" in result.code
        assert result.has_js_interop is True
    
    def test_apply_predicate(self):
        """Test apply predicate for JS method calls."""
        config = TauPrologConfig(enable_js_interop=True, default_libraries=[])
        emitter = TauPrologEmitter(config)
        
        ir = {
            "module": "jstest",
            "clauses": [
                {"kind": "rule",
                 "head": {"kind": "compound", "functor": "log_message", "args": [
                     {"kind": "variable", "name": "Msg"}
                 ]},
                 "body": [
                     {"kind": "compound", "functor": "global", "args": [
                         {"kind": "atom", "value": "console"},
                         {"kind": "variable", "name": "C"}
                     ]},
                     {"kind": "compound", "functor": "apply", "args": [
                         {"kind": "variable", "name": "C"},
                         {"kind": "atom", "value": "log"},
                         {"kind": "list_term", "elements": [
                             {"kind": "variable", "name": "Msg"}
                         ]},
                         {"kind": "anonymous"}
                     ]}
                 ]}
            ]
        }
        
        result = emitter.emit(ir)
        
        assert "global(console, C)" in result.code
        assert "apply(C, log, [Msg], _)" in result.code
    
    def test_prop_predicate(self):
        """Test prop predicate for JS property access."""
        config = TauPrologConfig(enable_js_interop=True, default_libraries=[])
        emitter = TauPrologEmitter(config)
        
        ir = {
            "module": "jstest",
            "clauses": [
                {"kind": "rule",
                 "head": {"kind": "compound", "functor": "get_title", "args": [
                     {"kind": "variable", "name": "T"}
                 ]},
                 "body": [
                     {"kind": "compound", "functor": "global", "args": [
                         {"kind": "atom", "value": "document"},
                         {"kind": "variable", "name": "Doc"}
                     ]},
                     {"kind": "compound", "functor": "prop", "args": [
                         {"kind": "variable", "name": "Doc"},
                         {"kind": "atom", "value": "title"},
                         {"kind": "variable", "name": "T"}
                     ]}
                 ]}
            ]
        }
        
        result = emitter.emit(ir)
        
        assert "prop(Doc, title, T)" in result.code


class TestJSLoaderGeneration:
    """Tests for JavaScript loader generation."""
    
    def test_browser_loader(self):
        """TC-TAU-006: Test browser JavaScript loader generation."""
        config = TauPrologConfig(
            emit_loader_js=True,
            target_runtime="browser",
            default_libraries=['lists']
        )
        emitter = TauPrologEmitter(config)
        
        ir = {
            "module": "demo",
            "clauses": [
                {"kind": "fact", "predicate": "hello", "args": [
                    {"kind": "atom", "value": "world"}
                ]}
            ]
        }
        
        result = emitter.emit(ir)
        
        assert result.loader_js is not None
        assert "pl.create()" in result.loader_js
        assert "session.consult" in result.loader_js
        assert "stunir_demo" in result.loader_js
        assert "window.stunirSession" in result.loader_js
    
    def test_node_loader(self):
        """Test Node.js JavaScript loader generation."""
        config = TauPrologConfig(
            emit_loader_js=True,
            target_runtime="node",
            default_libraries=['lists']
        )
        emitter = TauPrologEmitter(config)
        
        ir = {
            "module": "demo",
            "clauses": [
                {"kind": "fact", "predicate": "hello", "args": [
                    {"kind": "atom", "value": "world"}
                ]}
            ]
        }
        
        result = emitter.emit(ir)
        
        assert result.loader_js is not None
        assert 'require("tau-prolog")' in result.loader_js
        assert "module.exports" in result.loader_js
        assert result.target_runtime == "node"


class TestListTerms:
    """Tests for list term emission."""
    
    def test_empty_list(self):
        """Test empty list emission."""
        ir = {
            "module": "test",
            "clauses": [
                {"kind": "fact", "predicate": "empty", "args": [
                    {"kind": "list_term", "elements": []}
                ]}
            ]
        }
        
        emitter = TauPrologEmitter()
        result = emitter.emit(ir)
        
        assert "empty([])." in result.code
    
    def test_simple_list(self):
        """Test simple list emission."""
        ir = {
            "module": "test",
            "clauses": [
                {"kind": "fact", "predicate": "nums", "args": [
                    {"kind": "list_term", "elements": [
                        {"kind": "number", "value": 1},
                        {"kind": "number", "value": 2},
                        {"kind": "number", "value": 3}
                    ]}
                ]}
            ]
        }
        
        emitter = TauPrologEmitter()
        result = emitter.emit(ir)
        
        assert "nums([1, 2, 3])." in result.code
    
    def test_list_with_tail(self):
        """Test list with tail variable [H|T]."""
        ir = {
            "module": "test",
            "clauses": [
                {"kind": "rule",
                 "head": {"kind": "compound", "functor": "first", "args": [
                     {"kind": "list_term", "elements": [
                         {"kind": "variable", "name": "H"}
                     ], "tail": {"kind": "variable", "name": "T"}},
                     {"kind": "variable", "name": "H"}
                 ]},
                 "body": []}
            ]
        }
        
        emitter = TauPrologEmitter()
        result = emitter.emit(ir)
        
        assert "[H|T]" in result.code


class TestTypeMapper:
    """Tests for TauPrologTypeMapper."""
    
    def test_basic_type_mapping(self):
        """Test basic type mapping."""
        mapper = TauPrologTypeMapper()
        
        assert mapper.map_type('i32') == 'integer'
        assert mapper.map_type('f64') == 'float'
        assert mapper.map_type('string') == 'atom'
        assert mapper.map_type('bool') == 'boolean'
        assert mapper.map_type('list') == 'list'
    
    def test_js_type_mapping(self):
        """Test JavaScript type mapping."""
        mapper = TauPrologTypeMapper()
        
        assert mapper.map_type('js_object') == 'term'
        assert mapper.map_type('js_function') == 'term'
        assert mapper.map_type('js_array') == 'list'
    
    def test_dom_predicate_detection(self):
        """Test DOM predicate detection."""
        mapper = TauPrologTypeMapper()
        
        assert mapper.is_dom_predicate('get_by_id') is True
        assert mapper.is_dom_predicate('set_html') is True
        assert mapper.is_dom_predicate('not_a_dom_pred') is False
    
    def test_js_predicate_detection(self):
        """Test JS predicate detection."""
        mapper = TauPrologTypeMapper()
        
        assert mapper.is_js_predicate('apply') is True
        assert mapper.is_js_predicate('global') is True
        assert mapper.is_js_predicate('prop') is True
        assert mapper.is_js_predicate('not_a_js_pred') is False
    
    def test_lists_predicate_detection(self):
        """Test lists predicate detection."""
        mapper = TauPrologTypeMapper()
        
        assert mapper.is_lists_predicate('append') is True
        assert mapper.is_lists_predicate('member') is True
        assert mapper.is_lists_predicate('not_a_list_pred') is False
    
    def test_required_libraries_detection(self):
        """Test required libraries detection."""
        mapper = TauPrologTypeMapper()
        
        predicates = {'get_by_id', 'set_html', 'append', 'global'}
        libs = mapper.get_required_libraries(predicates)
        
        assert 'dom' in libs
        assert 'js' in libs
        assert 'lists' in libs


class TestDynamicPredicates:
    """Tests for dynamic predicate declarations."""
    
    def test_dynamic_declaration(self):
        """Test dynamic predicate declaration."""
        ir = {
            "module": "kb",
            "dynamic": [
                {"predicate": "fact", "arity": 1},
                {"predicate": "rule", "arity": 2}
            ],
            "clauses": []
        }
        
        emitter = TauPrologEmitter()
        result = emitter.emit(ir)
        
        assert ":- dynamic(fact/1)." in result.code
        assert ":- dynamic(rule/2)." in result.code


class TestSpecialCharacters:
    """Tests for special character handling."""
    
    def test_atom_with_spaces(self):
        """Test atom requiring quotes."""
        ir = {
            "module": "test",
            "clauses": [
                {"kind": "fact", "predicate": "name", "args": [
                    {"kind": "atom", "value": "hello world"}
                ]}
            ]
        }
        
        emitter = TauPrologEmitter()
        result = emitter.emit(ir)
        
        assert "name('hello world')." in result.code
    
    def test_atom_with_quote(self):
        """Test atom containing quote character."""
        ir = {
            "module": "test",
            "clauses": [
                {"kind": "fact", "predicate": "text", "args": [
                    {"kind": "atom", "value": "it's"}
                ]}
            ]
        }
        
        emitter = TauPrologEmitter()
        result = emitter.emit(ir)
        
        assert "text('it\\'s')." in result.code
    
    def test_uppercase_atom(self):
        """Test atom starting with uppercase."""
        ir = {
            "module": "test",
            "clauses": [
                {"kind": "fact", "predicate": "name", "args": [
                    {"kind": "atom", "value": "John"}
                ]}
            ]
        }
        
        emitter = TauPrologEmitter()
        result = emitter.emit(ir)
        
        assert "name('John')." in result.code


class TestGoalEmission:
    """Tests for goal emission."""
    
    def test_cut(self):
        """Test cut emission."""
        ir = {
            "module": "test",
            "clauses": [
                {"kind": "rule",
                 "head": {"kind": "compound", "functor": "first", "args": [
                     {"kind": "variable", "name": "X"},
                     {"kind": "variable", "name": "Y"}
                 ]},
                 "body": [
                     {"kind": "compound", "functor": "member", "args": [
                         {"kind": "variable", "name": "X"},
                         {"kind": "variable", "name": "Y"}
                     ]},
                     {"kind": "cut"}
                 ]}
            ]
        }
        
        emitter = TauPrologEmitter()
        result = emitter.emit(ir)
        
        assert "!" in result.code
    
    def test_negation(self):
        """Test negation-as-failure emission."""
        ir = {
            "module": "test",
            "clauses": [
                {"kind": "rule",
                 "head": {"kind": "compound", "functor": "not_member", "args": [
                     {"kind": "variable", "name": "X"},
                     {"kind": "variable", "name": "L"}
                 ]},
                 "body": [
                     {"kind": "negation",
                      "goal": {"kind": "compound", "functor": "member", "args": [
                          {"kind": "variable", "name": "X"},
                          {"kind": "variable", "name": "L"}
                      ]}}
                 ]}
            ]
        }
        
        emitter = TauPrologEmitter()
        result = emitter.emit(ir)
        
        assert "\\+ member(X, L)" in result.code


class TestHeaderAndComments:
    """Tests for header and comment generation."""
    
    def test_header_generation(self):
        """Test header comment generation."""
        config = TauPrologConfig(emit_comments=True, emit_timestamps=True)
        emitter = TauPrologEmitter(config)
        
        ir = {"module": "test", "clauses": []}
        result = emitter.emit(ir)
        
        assert "STUNIR Generated Tau Prolog Module" in result.code
        assert "Module: test" in result.code
        assert "Tau Prolog: JavaScript-based Prolog" in result.code
        assert "https://tau-prolog.org/" in result.code
    
    def test_no_comments(self):
        """Test emission without comments."""
        config = TauPrologConfig(emit_comments=False)
        emitter = TauPrologEmitter(config)
        
        ir = {"module": "test", "clauses": []}
        result = emitter.emit(ir)
        
        assert "/*" not in result.code
        assert "STUNIR Generated" not in result.code


class TestTargetRuntime:
    """Tests for target runtime settings."""
    
    def test_browser_runtime(self):
        """Test browser target runtime."""
        config = TauPrologConfig(target_runtime="browser")
        emitter = TauPrologEmitter(config)
        
        ir = {"module": "test", "clauses": []}
        result = emitter.emit(ir)
        
        assert result.target_runtime == "browser"
        assert "Target Runtime: browser" in result.code
    
    def test_node_runtime(self):
        """Test Node.js target runtime."""
        config = TauPrologConfig(target_runtime="node")
        emitter = TauPrologEmitter(config)
        
        ir = {"module": "test", "clauses": []}
        result = emitter.emit(ir)
        
        assert result.target_runtime == "node"
        assert "Target Runtime: node" in result.code


class TestConstants:
    """Tests for module constants."""
    
    def test_tau_libraries(self):
        """Test TAU_LIBRARIES constant."""
        assert 'lists' in TAU_LIBRARIES
        assert 'dom' in TAU_LIBRARIES
        assert 'js' in TAU_LIBRARIES
        assert 'format' in TAU_LIBRARIES
    
    def test_dom_predicates(self):
        """Test DOM_PREDICATES constant."""
        assert 'get_by_id' in DOM_PREDICATES
        assert 'set_html' in DOM_PREDICATES
        assert 'append_child' in DOM_PREDICATES
        assert DOM_PREDICATES['get_by_id'] == 2
    
    def test_js_predicates(self):
        """Test JS_PREDICATES constant."""
        assert 'apply' in JS_PREDICATES
        assert 'global' in JS_PREDICATES
        assert 'prop' in JS_PREDICATES
        assert JS_PREDICATES['apply'] == 4


class TestEmitterResult:
    """Tests for TauPrologEmitterResult."""
    
    def test_to_dict(self):
        """Test to_dict conversion."""
        ir = {
            "module": "test",
            "clauses": [
                {"kind": "fact", "predicate": "hello", "args": []}
            ]
        }
        
        emitter = TauPrologEmitter()
        result = emitter.emit(ir)
        
        d = result.to_dict()
        
        assert 'code' in d
        assert 'module_name' in d
        assert 'predicates' in d
        assert 'sha256' in d
        assert 'emit_time' in d
        assert 'libraries_used' in d
        assert 'has_dom' in d
        assert 'has_js_interop' in d
        assert 'target_runtime' in d


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
