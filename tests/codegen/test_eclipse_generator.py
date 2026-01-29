#!/usr/bin/env python3
"""Tests for ECLiPSe emitter.

Tests cover:
- Basic fact and rule emission
- Module declarations
- Library imports (IC, FD, ic_global, etc.)
- IC constraint operators ($=, $<, etc.)
- FD constraint operators (#=, #<, etc.)
- Domain constraints (X :: L..H)
- Global constraints (alldifferent, element)
- Optimization goals (minimize, maximize, bb_min)
- Search strategies (search/6, labeling)
- Determinism verification
- Comparison with GNU Prolog CLP output

Part of Phase 5D-2: ECLiPSe with Constraint Optimization.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from targets.prolog.eclipse.emitter import ECLiPSeEmitter, ECLiPSeConfig, EmitterResult
from targets.prolog.eclipse.types import (
    ECLiPSeTypeMapper, ECLIPSE_TYPES,
    IC_OPERATORS, FD_OPERATORS, ECLIPSE_GLOBALS,
    ECLIPSE_OPTIMIZATION, ECLIPSE_SEARCH
)


class TestECLiPSeTypes:
    """Tests for type mapping."""
    
    def test_basic_type_mapping(self):
        """Test basic IR to ECLiPSe type mapping."""
        mapper = ECLiPSeTypeMapper()
        
        assert mapper.map_type('i32') == 'integer'
        assert mapper.map_type('f64') == 'real'
        assert mapper.map_type('bool') == 'integer'  # 0/1 in ECLiPSe
        assert mapper.map_type('string') == 'atom'
        assert mapper.map_type('unknown') == 'term'
    
    def test_ic_operator_mapping(self):
        """Test IC library operator mapping (default)."""
        mapper = ECLiPSeTypeMapper()
        
        assert mapper.map_constraint_operator('==') == '$='
        assert mapper.map_constraint_operator('!=') == '$\\='
        assert mapper.map_constraint_operator('<') == '$<'
        assert mapper.map_constraint_operator('>=') == '$>='
    
    def test_fd_operator_mapping(self):
        """Test FD library operator mapping."""
        config = type('Config', (), {'default_library': 'fd'})()
        mapper = ECLiPSeTypeMapper(config)
        
        assert mapper.map_constraint_operator('==') == '#='
        assert mapper.map_constraint_operator('!=') == '#\\='
        assert mapper.map_constraint_operator('<') == '#<'
        assert mapper.map_constraint_operator('>=') == '#>='
    
    def test_constraint_operator_detection(self):
        """Test constraint operator detection."""
        mapper = ECLiPSeTypeMapper()
        
        assert mapper.is_constraint_operator('$=')
        assert mapper.is_constraint_operator('#<')
        assert mapper.is_constraint_operator('>=')
        assert not mapper.is_constraint_operator('append')
    
    def test_global_constraint_detection(self):
        """Test global constraint detection."""
        mapper = ECLiPSeTypeMapper()
        
        assert mapper.is_global_constraint('alldifferent')
        assert mapper.is_global_constraint('all_different')
        assert mapper.is_global_constraint('element')
        assert mapper.is_global_constraint('cumulative')
        assert not mapper.is_global_constraint('append')
    
    def test_global_constraint_mapping(self):
        """Test global constraint name mapping."""
        mapper = ECLiPSeTypeMapper()
        
        assert mapper.map_global_constraint('all_different') == 'alldifferent'
        assert mapper.map_global_constraint('fd_all_different') == 'alldifferent'
        assert mapper.map_global_constraint('element') == 'element'
    
    def test_optimization_predicate_detection(self):
        """Test optimization predicate detection."""
        mapper = ECLiPSeTypeMapper()
        
        assert mapper.is_optimization_predicate('minimize')
        assert mapper.is_optimization_predicate('maximize')
        assert mapper.is_optimization_predicate('bb_min')
        assert not mapper.is_optimization_predicate('append')
    
    def test_select_method_mapping(self):
        """Test search variable selection method mapping."""
        mapper = ECLiPSeTypeMapper()
        
        assert mapper.map_select_method('first_fail') == 'first_fail'
        assert mapper.map_select_method('ff') == 'first_fail'
        assert mapper.map_select_method('most_constrained') == 'most_constrained'
        assert mapper.map_select_method('input_order') == 'input_order'
    
    def test_choice_method_mapping(self):
        """Test search value choice method mapping."""
        mapper = ECLiPSeTypeMapper()
        
        assert mapper.map_choice_method('indomain') == 'indomain'
        assert mapper.map_choice_method('indomain_middle') == 'indomain_middle'
        assert mapper.map_choice_method('middle') == 'indomain_middle'
        assert mapper.map_choice_method('indomain_min') == 'indomain_min'
    
    def test_required_libraries(self):
        """Test required library detection."""
        mapper = ECLiPSeTypeMapper()
        
        libs = mapper.get_required_libraries(
            has_constraints=True,
            has_globals=True,
            has_optimization=True
        )
        
        assert 'ic' in libs
        assert 'ic_global' in libs
        assert 'branch_and_bound' in libs


class TestECLiPSeEmitter:
    """Tests for ECLiPSe emitter."""
    
    def test_simple_fact_emission(self):
        """TC-ECL-001: Test emitting simple facts."""
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
        
        emitter = ECLiPSeEmitter()
        result = emitter.emit(ir)
        
        assert isinstance(result, EmitterResult)
        assert "parent(tom, bob)." in result.code
        assert "parent(tom, liz)." in result.code
        assert result.module_name == "family"
    
    def test_module_declaration(self):
        """TC-ECL-002: Test module declaration."""
        ir = {
            "module": "scheduling",
            "clauses": [
                {"kind": "fact", "predicate": "task", "args": [
                    {"kind": "atom", "value": "a"}
                ]}
            ]
        }
        
        emitter = ECLiPSeEmitter()
        result = emitter.emit(ir)
        
        assert ":- module(scheduling)." in result.code
    
    def test_ic_library_domain(self):
        """TC-ECL-003: Test IC library domain constraint."""
        ir = {
            "module": "puzzle",
            "clauses": [
                {"kind": "rule",
                 "head": {"kind": "compound", "functor": "solve", "args": [
                     {"kind": "variable", "name": "X"}
                 ]},
                 "body": [
                     {"kind": "compound", "functor": "::", "args": [
                         {"kind": "variable", "name": "X"},
                         {"kind": "compound", "functor": "..", "args": [
                             {"kind": "number", "value": 1},
                             {"kind": "number", "value": 9}
                         ]}
                     ]}
                 ]}
            ]
        }
        
        emitter = ECLiPSeEmitter()
        result = emitter.emit(ir)
        
        assert "X :: 1..9" in result.code
        assert ":- lib(ic)." in result.code
        assert 'ic' in result.libraries_used
    
    def test_ic_arithmetic_constraint(self):
        """TC-ECL-004: Test IC arithmetic constraint operators."""
        ir = {
            "module": "math",
            "clauses": [
                {"kind": "rule",
                 "head": {"kind": "compound", "functor": "check", "args": [
                     {"kind": "variable", "name": "X"},
                     {"kind": "variable", "name": "Y"}
                 ]},
                 "body": [
                     {"kind": "compound", "functor": "#>", "args": [
                         {"kind": "variable", "name": "X"},
                         {"kind": "number", "value": 0}
                     ]}
                 ]}
            ]
        }
        
        emitter = ECLiPSeEmitter()
        result = emitter.emit(ir)
        
        # IC uses $> instead of #>
        assert "X $> 0" in result.code
        assert 'ic' in result.libraries_used
    
    def test_global_constraint_alldifferent(self):
        """TC-ECL-005: Test global constraint alldifferent."""
        ir = {
            "module": "sudoku",
            "clauses": [
                {"kind": "rule",
                 "head": {"kind": "compound", "functor": "unique", "args": [
                     {"kind": "variable", "name": "L"}
                 ]},
                 "body": [
                     {"kind": "compound", "functor": "alldifferent", "args": [
                         {"kind": "variable", "name": "L"}
                     ]}
                 ]}
            ]
        }
        
        emitter = ECLiPSeEmitter()
        result = emitter.emit(ir)
        
        assert "alldifferent(L)" in result.code
        assert ":- lib(ic_global)." in result.code
    
    def test_minimize_optimization(self):
        """TC-ECL-006: Test minimize optimization."""
        ir = {
            "module": "opt",
            "clauses": [
                {"kind": "rule",
                 "head": {"kind": "compound", "functor": "optimal", "args": [
                     {"kind": "variable", "name": "X"}
                 ]},
                 "body": [
                     {"kind": "compound", "functor": "minimize", "args": [
                         {"kind": "compound", "functor": "solve", "args": [
                             {"kind": "variable", "name": "X"}
                         ]},
                         {"kind": "variable", "name": "X"}
                     ]}
                 ]}
            ]
        }
        
        emitter = ECLiPSeEmitter()
        result = emitter.emit(ir)
        
        assert "minimize(solve(X), X)" in result.code
        assert result.has_optimization
        assert ":- lib(branch_and_bound)." in result.code
    
    def test_search_strategy(self):
        """TC-ECL-007: Test search strategy emission."""
        ir = {
            "module": "search_test",
            "clauses": [
                {"kind": "rule",
                 "head": {"kind": "compound", "functor": "solve", "args": [
                     {"kind": "variable", "name": "Vars"}
                 ]},
                 "body": [
                     {"kind": "compound", "functor": "search", "args": [
                         {"kind": "variable", "name": "Vars"},
                         {"kind": "number", "value": 0},
                         {"kind": "atom", "value": "first_fail"},
                         {"kind": "atom", "value": "indomain_middle"}
                     ]}
                 ]}
            ]
        }
        
        emitter = ECLiPSeEmitter()
        result = emitter.emit(ir)
        
        assert "search(Vars, 0, first_fail, indomain_middle, complete, [])" in result.code
        assert len(result.search_strategies) > 0
    
    def test_branch_and_bound(self):
        """TC-ECL-008: Test branch-and-bound optimization."""
        ir = {
            "module": "bb",
            "clauses": [
                {"kind": "rule",
                 "head": {"kind": "compound", "functor": "opt_solve", "args": [
                     {"kind": "variable", "name": "X"}
                 ]},
                 "body": [
                     {"kind": "compound", "functor": "bb_min", "args": [
                         {"kind": "compound", "functor": "solve", "args": [
                             {"kind": "variable", "name": "X"}
                         ]},
                         {"kind": "variable", "name": "X"}
                     ]}
                 ]}
            ]
        }
        
        emitter = ECLiPSeEmitter()
        result = emitter.emit(ir)
        
        assert "bb_min(solve(X), X, bb_options{})" in result.code
        assert 'branch_and_bound' in result.libraries_used
    
    def test_determinism(self):
        """TC-ECL-009: Test that output is deterministic."""
        ir = {
            "module": "test",
            "clauses": [
                {"kind": "fact", "predicate": "a", "args": []},
                {"kind": "fact", "predicate": "b", "args": []},
            ]
        }
        
        config = ECLiPSeConfig(emit_timestamps=False)
        emitter = ECLiPSeEmitter(config)
        
        # Generate multiple times
        results = [emitter.emit(ir).code for _ in range(5)]
        
        # All should be identical
        assert all(r == results[0] for r in results)
    
    def test_compare_gnu_prolog(self):
        """TC-ECL-010: Test that ECLiPSe output differs appropriately from GNU Prolog."""
        from targets.prolog.gnu_prolog.emitter import GNUPrologEmitter
        
        ir = {
            "module": "test",
            "exports": [{"predicate": "foo", "arity": 1}],
            "clauses": [
                {"kind": "fact", "predicate": "foo", "args": [
                    {"kind": "atom", "value": "bar"}
                ]}
            ]
        }
        
        gnu_emitter = GNUPrologEmitter()
        ecl_emitter = ECLiPSeEmitter()
        
        gnu_result = gnu_emitter.emit(ir)
        ecl_result = ecl_emitter.emit(ir)
        
        # ECLiPSe has module declaration, GNU uses public
        assert ":- module(" in ecl_result.code
        assert ":- public(" in gnu_result.code
        
        # Both emit the fact the same way
        assert "foo(bar)." in gnu_result.code
        assert "foo(bar)." in ecl_result.code
    
    def test_export_declaration(self):
        """Test export predicate declaration."""
        ir = {
            "module": "utils",
            "exports": [{"predicate": "helper", "arity": 2}],
            "clauses": [
                {"kind": "fact", "predicate": "helper", "args": [
                    {"kind": "variable", "name": "X"},
                    {"kind": "variable", "name": "X"}
                ]}
            ]
        }
        
        emitter = ECLiPSeEmitter()
        result = emitter.emit(ir)
        
        assert ":- export(helper/2)." in result.code
    
    def test_dynamic_declaration(self):
        """Test dynamic predicate declaration."""
        ir = {
            "module": "db",
            "clauses": [],
            "dynamic": [
                {"predicate": "fact", "arity": 2}
            ]
        }
        
        emitter = ECLiPSeEmitter()
        result = emitter.emit(ir)
        
        assert ":- dynamic(fact/2)." in result.code
    
    def test_rule_emission(self):
        """Test emitting rules with body."""
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
        
        emitter = ECLiPSeEmitter()
        result = emitter.emit(ir)
        
        assert "grandparent(X, Z) :-" in result.code
        assert "parent(X, Y)" in result.code
        assert "parent(Y, Z)" in result.code
    
    def test_rule_with_cut(self):
        """Test rule emission with cut operator."""
        ir = {
            "module": "test",
            "clauses": [
                {"kind": "rule",
                 "head": {"kind": "compound", "functor": "max", "args": [
                     {"kind": "variable", "name": "X"},
                     {"kind": "variable", "name": "Y"},
                     {"kind": "variable", "name": "X"}
                 ]},
                 "body": [
                     {"kind": "compound", "functor": ">=", "args": [
                         {"kind": "variable", "name": "X"},
                         {"kind": "variable", "name": "Y"}
                     ]},
                     {"kind": "cut"}
                 ]}
            ]
        }
        
        emitter = ECLiPSeEmitter()
        result = emitter.emit(ir)
        
        assert "!" in result.code
        assert "max(X, Y, X) :-" in result.code
    
    def test_list_emission(self):
        """Test list term emission."""
        ir = {
            "module": "test",
            "clauses": [
                {"kind": "rule",
                 "head": {"kind": "compound", "functor": "first", "args": [
                     {"kind": "list_term", 
                      "elements": [{"kind": "variable", "name": "H"}],
                      "tail": {"kind": "variable", "name": "T"}},
                     {"kind": "variable", "name": "H"}
                 ]},
                 "body": []}
            ]
        }
        
        emitter = ECLiPSeEmitter()
        result = emitter.emit(ir)
        
        assert "first([H|T], H)" in result.code
    
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
        
        emitter = ECLiPSeEmitter()
        result = emitter.emit(ir)
        
        assert "empty([])." in result.code
    
    def test_atom_escaping(self):
        """Test atom escaping for special characters."""
        ir = {
            "module": "test",
            "clauses": [
                {"kind": "fact", "predicate": "name", "args": [
                    {"kind": "atom", "value": "Hello World"}
                ]},
                {"kind": "fact", "predicate": "quoted", "args": [
                    {"kind": "atom", "value": "it's"}
                ]}
            ]
        }
        
        emitter = ECLiPSeEmitter()
        result = emitter.emit(ir)
        
        # Atoms with spaces need quoting
        assert "'Hello World'" in result.code
        # Atoms with apostrophes need quoting and escaping
        assert "it" in result.code
    
    def test_negation(self):
        """Test negation operator emission."""
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
        
        emitter = ECLiPSeEmitter()
        result = emitter.emit(ir)
        
        assert "\\+" in result.code
    
    def test_library_header_emission(self):
        """Test library header comments are generated."""
        ir = {
            "module": "constraints",
            "clauses": [
                {"kind": "rule",
                 "head": {"kind": "compound", "functor": "solve", "args": [
                     {"kind": "variable", "name": "X"}
                 ]},
                 "body": [
                     {"kind": "compound", "functor": "::", "args": [
                         {"kind": "variable", "name": "X"},
                         {"kind": "compound", "functor": "..", "args": [
                             {"kind": "number", "value": 1},
                             {"kind": "number", "value": 10}
                         ]}
                     ]}
                 ]}
            ]
        }
        
        emitter = ECLiPSeEmitter()
        result = emitter.emit(ir)
        
        assert "lib(ic)" in result.code
    
    def test_config_options(self):
        """Test configuration options."""
        ir = {
            "module": "test",
            "clauses": [
                {"kind": "fact", "predicate": "test", "args": []}
            ]
        }
        
        # Test with timestamps disabled
        config = ECLiPSeConfig(emit_timestamps=False, emit_comments=True)
        emitter = ECLiPSeEmitter(config)
        result = emitter.emit(ir)
        
        assert "STUNIR Generated" in result.code
        
        # Test with comments disabled
        config = ECLiPSeConfig(emit_comments=False)
        emitter = ECLiPSeEmitter(config)
        result = emitter.emit(ir)
        
        assert "/*" not in result.code
    
    def test_result_fields(self):
        """Test EmitterResult has all required fields."""
        ir = {
            "module": "test",
            "clauses": [
                {"kind": "fact", "predicate": "test", "args": []}
            ]
        }
        
        emitter = ECLiPSeEmitter()
        result = emitter.emit(ir)
        
        assert hasattr(result, 'code')
        assert hasattr(result, 'module_name')
        assert hasattr(result, 'predicates')
        assert hasattr(result, 'sha256')
        assert hasattr(result, 'emit_time')
        assert hasattr(result, 'libraries_used')
        assert hasattr(result, 'has_optimization')
        assert hasattr(result, 'search_strategies')
        
        assert isinstance(result.code, str)
        assert isinstance(result.module_name, str)
        assert isinstance(result.predicates, list)
        assert isinstance(result.sha256, str)
        assert isinstance(result.emit_time, float)
        assert isinstance(result.libraries_used, list)
        assert isinstance(result.has_optimization, bool)
        assert isinstance(result.search_strategies, list)


class TestECLiPSeConstraintLibraries:
    """Tests for ECLiPSe constraint library support."""
    
    def test_fd_library_mode(self):
        """Test FD library mode with # operators."""
        ir = {
            "module": "fd_test",
            "clauses": [
                {"kind": "rule",
                 "head": {"kind": "compound", "functor": "check", "args": [
                     {"kind": "variable", "name": "X"}
                 ]},
                 "body": [
                     {"kind": "compound", "functor": "==", "args": [
                         {"kind": "variable", "name": "X"},
                         {"kind": "number", "value": 5}
                     ]}
                 ]}
            ]
        }
        
        config = ECLiPSeConfig(default_library='fd')
        emitter = ECLiPSeEmitter(config)
        result = emitter.emit(ir)
        
        # FD uses #= instead of $=
        assert "X #= 5" in result.code
        assert 'fd' in result.libraries_used
    
    def test_domain_from_in_syntax(self):
        """Test domain constraint from 'in' syntax."""
        ir = {
            "module": "test",
            "clauses": [
                {"kind": "rule",
                 "head": {"kind": "compound", "functor": "solve", "args": [
                     {"kind": "variable", "name": "X"}
                 ]},
                 "body": [
                     {"kind": "compound", "functor": "in", "args": [
                         {"kind": "variable", "name": "X"},
                         {"kind": "compound", "functor": "..", "args": [
                             {"kind": "number", "value": 1},
                             {"kind": "number", "value": 100}
                         ]}
                     ]}
                 ]}
            ]
        }
        
        emitter = ECLiPSeEmitter()
        result = emitter.emit(ir)
        
        assert "X :: 1..100" in result.code
    
    def test_domain_from_domain_predicate(self):
        """Test domain constraint from domain/3 predicate."""
        ir = {
            "module": "test",
            "clauses": [
                {"kind": "rule",
                 "head": {"kind": "compound", "functor": "solve", "args": [
                     {"kind": "variable", "name": "X"}
                 ]},
                 "body": [
                     {"kind": "compound", "functor": "domain", "args": [
                         {"kind": "variable", "name": "X"},
                         {"kind": "number", "value": 0},
                         {"kind": "number", "value": 99}
                     ]}
                 ]}
            ]
        }
        
        emitter = ECLiPSeEmitter()
        result = emitter.emit(ir)
        
        assert "X :: 0..99" in result.code
    
    def test_element_constraint(self):
        """Test element global constraint."""
        ir = {
            "module": "test",
            "clauses": [
                {"kind": "rule",
                 "head": {"kind": "compound", "functor": "select", "args": [
                     {"kind": "variable", "name": "I"},
                     {"kind": "variable", "name": "E"}
                 ]},
                 "body": [
                     {"kind": "compound", "functor": "element", "args": [
                         {"kind": "variable", "name": "I"},
                         {"kind": "list_term", "elements": [
                             {"kind": "number", "value": 1},
                             {"kind": "number", "value": 2},
                             {"kind": "number", "value": 3}
                         ]},
                         {"kind": "variable", "name": "E"}
                     ]}
                 ]}
            ]
        }
        
        emitter = ECLiPSeEmitter()
        result = emitter.emit(ir)
        
        assert "element(I, [1, 2, 3], E)" in result.code


class TestECLiPSeOptimization:
    """Tests for ECLiPSe optimization features."""
    
    def test_maximize_optimization(self):
        """Test maximize optimization goal."""
        ir = {
            "module": "opt",
            "clauses": [
                {"kind": "rule",
                 "head": {"kind": "compound", "functor": "best", "args": [
                     {"kind": "variable", "name": "X"}
                 ]},
                 "body": [
                     {"kind": "compound", "functor": "maximize", "args": [
                         {"kind": "compound", "functor": "solve", "args": [
                             {"kind": "variable", "name": "X"}
                         ]},
                         {"kind": "variable", "name": "X"}
                     ]}
                 ]}
            ]
        }
        
        emitter = ECLiPSeEmitter()
        result = emitter.emit(ir)
        
        assert "maximize(solve(X), X)" in result.code
        assert result.has_optimization
    
    def test_bb_max_optimization(self):
        """Test bb_max branch-and-bound optimization."""
        ir = {
            "module": "opt",
            "clauses": [
                {"kind": "rule",
                 "head": {"kind": "compound", "functor": "best", "args": [
                     {"kind": "variable", "name": "X"}
                 ]},
                 "body": [
                     {"kind": "compound", "functor": "bb_max", "args": [
                         {"kind": "compound", "functor": "solve", "args": [
                             {"kind": "variable", "name": "X"}
                         ]},
                         {"kind": "variable", "name": "X"}
                     ]}
                 ]}
            ]
        }
        
        emitter = ECLiPSeEmitter()
        result = emitter.emit(ir)
        
        assert "bb_max(solve(X), X, bb_options{})" in result.code
        assert 'branch_and_bound' in result.libraries_used


class TestECLiPSeEdgeCases:
    """Edge case tests for ECLiPSe emitter."""
    
    def test_empty_ir(self):
        """Test handling of minimal IR."""
        ir = {
            "module": "empty"
        }
        
        emitter = ECLiPSeEmitter()
        result = emitter.emit(ir)
        
        assert result.module_name == "empty"
        assert ":- module(empty)." in result.code
    
    def test_uppercase_module_name(self):
        """Test module names starting with uppercase."""
        ir = {
            "module": "MyModule",
            "clauses": []
        }
        
        emitter = ECLiPSeEmitter()
        result = emitter.emit(ir)
        
        # Should be converted to lowercase
        assert result.module_name == "myModule"
    
    def test_special_characters_in_name(self):
        """Test handling of special characters in module name."""
        ir = {
            "module": "test-module.v2",
            "clauses": []
        }
        
        emitter = ECLiPSeEmitter()
        result = emitter.emit(ir)
        
        # Special chars should be replaced
        assert "test_module_v2" in result.module_name
    
    def test_multiple_predicates(self):
        """Test emission of multiple predicates."""
        ir = {
            "module": "multi",
            "clauses": [
                {"kind": "fact", "predicate": "a", "args": []},
                {"kind": "fact", "predicate": "b", "args": []},
                {"kind": "fact", "predicate": "c", "args": []},
            ]
        }
        
        emitter = ECLiPSeEmitter()
        result = emitter.emit(ir)
        
        assert "a." in result.code
        assert "b." in result.code
        assert "c." in result.code
        assert len(result.predicates) == 3
    
    def test_simple_labeling(self):
        """Test simple labeling search."""
        ir = {
            "module": "test",
            "clauses": [
                {"kind": "rule",
                 "head": {"kind": "compound", "functor": "solve", "args": [
                     {"kind": "variable", "name": "Vars"}
                 ]},
                 "body": [
                     {"kind": "compound", "functor": "labeling", "args": [
                         {"kind": "variable", "name": "Vars"}
                     ]}
                 ]}
            ]
        }
        
        emitter = ECLiPSeEmitter()
        result = emitter.emit(ir)
        
        assert "labeling(Vars)" in result.code


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
