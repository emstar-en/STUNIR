#!/usr/bin/env python3
"""Tests for Parser Generator interface and common functionality.

Tests:
- ParserGeneratorResult creation and methods
- Error recovery generation
- Grammar validation for parsing
"""

import pytest
import sys
import os

# Add the repository root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ir.parser.parser_generator import (
    ParserGeneratorResult, ErrorRecoveryStrategy,
    generate_error_recovery, validate_grammar_for_parsing
)
from ir.parser.parse_table import ParserType, ParseTable, LL1Table


class TestParserGeneratorResult:
    """Test ParserGeneratorResult class."""
    
    def test_result_creation(self):
        """Test creating a parser generator result."""
        table = ParseTable()
        result = ParserGeneratorResult(
            parse_table=table,
            parser_type=ParserType.LALR1
        )
        
        assert result.parse_table == table
        assert result.parser_type == ParserType.LALR1
        assert not result.has_conflicts()
        assert result.is_successful()
    
    def test_result_with_conflicts(self):
        """Test result with conflicts."""
        from ir.parser.parse_table import Conflict, Action
        from ir.grammar.symbol import Symbol, SymbolType
        
        table = ParseTable()
        sym = Symbol("a", SymbolType.TERMINAL)
        conflict = Conflict(0, sym, Action.shift(1), Action.reduce(0))
        
        result = ParserGeneratorResult(
            parse_table=table,
            parser_type=ParserType.LALR1,
            conflicts=[conflict]
        )
        
        assert result.has_conflicts()
        assert not result.is_successful()
        assert len(result.conflicts) == 1
    
    def test_add_warning(self):
        """Test adding warnings."""
        result = ParserGeneratorResult(
            parse_table=ParseTable(),
            parser_type=ParserType.LALR1
        )
        
        result.add_warning("Test warning")
        assert "Test warning" in result.warnings
    
    def test_add_info(self):
        """Test adding info."""
        result = ParserGeneratorResult(
            parse_table=ParseTable(),
            parser_type=ParserType.LALR1
        )
        
        result.add_info("state_count", 10)
        assert result.info["state_count"] == 10
    
    def test_result_str(self):
        """Test string representation."""
        result = ParserGeneratorResult(
            parse_table=ParseTable(),
            parser_type=ParserType.LALR1
        )
        
        str_repr = str(result)
        assert "LALR1" in str_repr
        assert "Successful" in str_repr


class TestErrorRecoveryStrategy:
    """Test error recovery strategy generation."""
    
    def test_panic_mode_recovery(self):
        """Test panic mode error recovery."""
        from ir.grammar.grammar_ir import Grammar, GrammarType
        from ir.grammar.symbol import Symbol, SymbolType
        from ir.grammar.production import ProductionRule
        
        S = Symbol("S", SymbolType.NONTERMINAL)
        semi = Symbol(";", SymbolType.TERMINAL)
        a = Symbol("a", SymbolType.TERMINAL)
        
        grammar = Grammar("test", GrammarType.BNF, S)
        grammar.add_production(ProductionRule(S, (a, semi)))
        
        recovery = generate_error_recovery(grammar, ErrorRecoveryStrategy.PANIC_MODE)
        
        assert recovery["strategy"] == "PANIC_MODE"
        assert ";" in recovery["sync_tokens"]
        assert recovery["use_follow_sets"]
    
    def test_phrase_level_recovery(self):
        """Test phrase level error recovery."""
        from ir.grammar.grammar_ir import Grammar, GrammarType
        from ir.grammar.symbol import Symbol, SymbolType
        from ir.grammar.production import ProductionRule
        
        S = Symbol("S", SymbolType.NONTERMINAL)
        a = Symbol("a", SymbolType.TERMINAL)
        
        grammar = Grammar("test", GrammarType.BNF, S)
        grammar.add_production(ProductionRule(S, (a,)))
        
        recovery = generate_error_recovery(grammar, ErrorRecoveryStrategy.PHRASE_LEVEL)
        
        assert recovery["strategy"] == "PHRASE_LEVEL"
        assert "insert_candidates" in recovery
    
    def test_error_productions_recovery(self):
        """Test error productions recovery."""
        from ir.grammar.grammar_ir import Grammar, GrammarType
        from ir.grammar.symbol import Symbol, SymbolType
        from ir.grammar.production import ProductionRule
        
        S = Symbol("S", SymbolType.NONTERMINAL)
        a = Symbol("a", SymbolType.TERMINAL)
        
        grammar = Grammar("test", GrammarType.BNF, S)
        grammar.add_production(ProductionRule(S, (a,)))
        
        recovery = generate_error_recovery(grammar, ErrorRecoveryStrategy.ERROR_PRODUCTIONS)
        
        assert recovery["strategy"] == "ERROR_PRODUCTIONS"
        assert recovery["needs_grammar_modification"]


class TestGrammarValidation:
    """Test grammar validation for parsing."""
    
    def test_valid_grammar(self):
        """Test validating a correct grammar."""
        from ir.grammar.grammar_ir import Grammar, GrammarType
        from ir.grammar.symbol import Symbol, SymbolType
        from ir.grammar.production import ProductionRule
        
        S = Symbol("S", SymbolType.NONTERMINAL)
        a = Symbol("a", SymbolType.TERMINAL)
        
        grammar = Grammar("test", GrammarType.BNF, S)
        grammar.add_production(ProductionRule(S, (a,)))
        
        issues = validate_grammar_for_parsing(grammar)
        assert len(issues) == 0
    
    def test_empty_grammar(self):
        """Test validating empty grammar."""
        from ir.grammar.grammar_ir import Grammar, GrammarType
        from ir.grammar.symbol import Symbol, SymbolType
        
        S = Symbol("S", SymbolType.NONTERMINAL)
        grammar = Grammar("test", GrammarType.BNF, S)
        
        issues = validate_grammar_for_parsing(grammar)
        assert any("no productions" in issue.lower() for issue in issues)
    
    def test_missing_start_productions(self):
        """Test grammar with no productions for start symbol."""
        from ir.grammar.grammar_ir import Grammar, GrammarType
        from ir.grammar.symbol import Symbol, SymbolType
        from ir.grammar.production import ProductionRule
        
        S = Symbol("S", SymbolType.NONTERMINAL)
        A = Symbol("A", SymbolType.NONTERMINAL)
        a = Symbol("a", SymbolType.TERMINAL)
        
        grammar = Grammar("test", GrammarType.BNF, S)
        grammar.add_production(ProductionRule(A, (a,)))
        
        issues = validate_grammar_for_parsing(grammar)
        assert any("start symbol" in issue.lower() for issue in issues)
    
    def test_unreachable_nonterminals(self):
        """Test grammar with unreachable nonterminals."""
        from ir.grammar.grammar_ir import Grammar, GrammarType
        from ir.grammar.symbol import Symbol, SymbolType
        from ir.grammar.production import ProductionRule
        
        S = Symbol("S", SymbolType.NONTERMINAL)
        A = Symbol("A", SymbolType.NONTERMINAL)
        B = Symbol("B", SymbolType.NONTERMINAL)  # Unreachable
        a = Symbol("a", SymbolType.TERMINAL)
        
        grammar = Grammar("test", GrammarType.BNF, S)
        grammar.add_production(ProductionRule(S, (A,)))
        grammar.add_production(ProductionRule(A, (a,)))
        grammar.add_production(ProductionRule(B, (a,)))  # Unreachable
        
        issues = validate_grammar_for_parsing(grammar)
        assert any("unreachable" in issue.lower() for issue in issues)


class TestAST:
    """Test AST node generation."""
    
    def test_ast_node_spec_creation(self):
        """Test creating AST node spec."""
        from ir.parser.ast_node import ASTNodeSpec
        
        node = ASTNodeSpec("BinaryExpr")
        node.add_field("left", "Expr")
        node.add_field("op", "Token")
        node.add_field("right", "Expr")
        
        assert node.name == "BinaryExpr"
        assert node.field_count() == 3
        assert node.get_field("left") == ("left", "Expr")
    
    def test_ast_schema_creation(self):
        """Test creating AST schema."""
        from ir.parser.ast_node import ASTSchema, ASTNodeSpec
        
        schema = ASTSchema()
        schema.add_node(ASTNodeSpec("Expr", is_abstract=True))
        schema.add_node(ASTNodeSpec("BinaryExpr", base_class="Expr"))
        schema.add_node(ASTNodeSpec("Literal", base_class="Expr"))
        
        assert schema.node_count() == 3
        assert schema.get_node("Expr") is not None
        assert len(schema.get_abstract_nodes()) == 1
        assert len(schema.get_concrete_nodes()) == 2
    
    def test_ast_schema_validation(self):
        """Test AST schema validation."""
        from ir.parser.ast_node import ASTSchema, ASTNodeSpec
        
        schema = ASTSchema()
        schema.add_node(ASTNodeSpec("Child", base_class="NonExistent"))
        
        errors = schema.validate()
        assert len(errors) > 0
        assert any("unknown base class" in e.lower() for e in errors)
    
    def test_generate_ast_schema(self):
        """Test generating AST schema from grammar."""
        from ir.grammar.grammar_ir import Grammar, GrammarType
        from ir.grammar.symbol import Symbol, SymbolType
        from ir.grammar.production import ProductionRule
        from ir.parser.ast_node import generate_ast_schema
        
        E = Symbol("expr", SymbolType.NONTERMINAL)
        num = Symbol("num", SymbolType.TERMINAL)
        
        grammar = Grammar("test", GrammarType.BNF, E)
        grammar.add_production(ProductionRule(E, (num,)))
        
        schema = generate_ast_schema(grammar)
        
        assert schema.node_count() > 0
        # Should have ExprNode
        expr_node = schema.get_node("ExprNode")
        assert expr_node is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
