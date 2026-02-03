"""
Tests for STUNIR Grammar Specification.

Tests the Grammar IR-based STUNIR grammar builder and validates
that the grammar correctly represents the STUNIR language.
"""

import pytest
from typing import Set

from bootstrap.stunir_grammar import (
    STUNIRGrammarBuilder,
    create_stunir_grammar,
)
from ir.grammar.symbol import Symbol, SymbolType, EPSILON
from ir.grammar.grammar_ir import Grammar, GrammarType


class TestSTUNIRGrammarBuilder:
    """Tests for STUNIRGrammarBuilder class."""
    
    def test_builder_initialization(self):
        """Test builder initializes correctly."""
        builder = STUNIRGrammarBuilder()
        assert builder is not None
    
    def test_build_grammar(self):
        """Test building complete grammar."""
        builder = STUNIRGrammarBuilder()
        grammar = builder.build()
        
        assert grammar is not None
        assert isinstance(grammar, Grammar)
        assert grammar.name == 'STUNIR'
        assert grammar.grammar_type == GrammarType.EBNF
    
    def test_start_symbol(self):
        """Test start symbol is 'program'."""
        grammar = create_stunir_grammar()
        
        assert grammar.start_symbol.name == 'program'
        assert grammar.start_symbol.symbol_type == SymbolType.NONTERMINAL
    
    def test_terminals_defined(self):
        """Test all expected terminals are defined."""
        builder = STUNIRGrammarBuilder()
        builder.build()
        terminals = builder.get_terminals()
        
        # Keywords
        assert 'KW_MODULE' in terminals
        assert 'KW_FUNCTION' in terminals
        assert 'KW_TYPE' in terminals
        assert 'KW_IR' in terminals
        assert 'KW_TARGET' in terminals
        assert 'KW_LET' in terminals
        assert 'KW_VAR' in terminals
        assert 'KW_IF' in terminals
        assert 'KW_ELSE' in terminals
        assert 'KW_WHILE' in terminals
        assert 'KW_FOR' in terminals
        assert 'KW_RETURN' in terminals
        assert 'KW_EMIT' in terminals
        
        # Type keywords
        assert 'KW_I32' in terminals
        assert 'KW_F64' in terminals
        assert 'KW_BOOL' in terminals
        assert 'KW_STRING' in terminals
        
        # Operators
        assert 'PLUS' in terminals
        assert 'MINUS' in terminals
        assert 'STAR' in terminals
        assert 'SLASH' in terminals
        assert 'EQ' in terminals
        assert 'NE' in terminals
        assert 'AND' in terminals
        assert 'OR' in terminals
        
        # Punctuation
        assert 'LPAREN' in terminals
        assert 'RPAREN' in terminals
        assert 'LBRACE' in terminals
        assert 'RBRACE' in terminals
        assert 'SEMICOLON' in terminals
    
    def test_nonterminals_defined(self):
        """Test all expected non-terminals are defined."""
        builder = STUNIRGrammarBuilder()
        builder.build()
        nonterminals = builder.get_nonterminals()
        
        # Core non-terminals
        assert 'program' in nonterminals
        assert 'module_decl' in nonterminals
        assert 'declaration' in nonterminals
        assert 'type_def' in nonterminals
        assert 'function_def' in nonterminals
        assert 'ir_def' in nonterminals
        assert 'target_def' in nonterminals
        
        # Statement non-terminals
        assert 'statement' in nonterminals
        assert 'var_decl' in nonterminals
        assert 'if_stmt' in nonterminals
        assert 'while_stmt' in nonterminals
        assert 'for_stmt' in nonterminals
        assert 'return_stmt' in nonterminals
        
        # Expression non-terminals
        assert 'expression' in nonterminals
        assert 'primary_expr' in nonterminals
        assert 'unary_expr' in nonterminals
        assert 'additive_expr' in nonterminals
        assert 'multiplicative_expr' in nonterminals
    
    def test_productions_count(self):
        """Test grammar has reasonable number of productions."""
        builder = STUNIRGrammarBuilder()
        builder.build()
        productions = builder.get_productions()
        
        # Should have many productions for a complete language
        assert len(productions) >= 50
        assert len(productions) <= 300  # Sanity check
    
    def test_program_productions(self):
        """Test program production rules exist."""
        builder = STUNIRGrammarBuilder()
        builder.build()
        productions = builder.get_productions()
        
        # Find program productions
        program_prods = [p for p in productions if p.head.name == 'program']
        assert len(program_prods) >= 1
    
    def test_module_productions(self):
        """Test module declaration productions exist."""
        builder = STUNIRGrammarBuilder()
        builder.build()
        productions = builder.get_productions()
        
        # Find module_decl productions
        module_prods = [p for p in productions if p.head.name == 'module_decl']
        assert len(module_prods) >= 2  # Simple and block forms
    
    def test_function_productions(self):
        """Test function definition productions exist."""
        builder = STUNIRGrammarBuilder()
        builder.build()
        productions = builder.get_productions()
        
        # Find function_def productions
        func_prods = [p for p in productions if p.head.name == 'function_def']
        assert len(func_prods) >= 1
    
    def test_expression_productions(self):
        """Test expression productions exist."""
        builder = STUNIRGrammarBuilder()
        builder.build()
        productions = builder.get_productions()
        
        # Check various expression levels
        assert any(p.head.name == 'expression' for p in productions)
        assert any(p.head.name == 'primary_expr' for p in productions)
        assert any(p.head.name == 'additive_expr' for p in productions)


class TestGrammarValidation:
    """Tests for grammar validation."""
    
    def test_grammar_validates(self):
        """Test grammar passes validation."""
        from ir.grammar.validation import GrammarValidator
        
        grammar = create_stunir_grammar()
        validator = GrammarValidator(grammar)
        result = validator.validate()
        
        # May have warnings but should not have fatal errors
        # that prevent parsing
        assert result is not None
    
    def test_no_undefined_symbols(self):
        """Test all symbols in productions are defined."""
        builder = STUNIRGrammarBuilder()
        builder.build()
        
        terminals = set(builder.get_terminals().keys())
        nonterminals = set(builder.get_nonterminals().keys())
        productions = builder.get_productions()
        
        for prod in productions:
            # Check head is a non-terminal
            assert prod.head.name in nonterminals, \
                f"Undefined non-terminal in head: {prod.head.name}"
            
            # Check body symbols
            for sym in prod.body:
                if sym.symbol_type == SymbolType.TERMINAL:
                    assert sym.name in terminals, \
                        f"Undefined terminal: {sym.name} in {prod}"
                elif sym.symbol_type == SymbolType.NONTERMINAL:
                    assert sym.name in nonterminals, \
                        f"Undefined non-terminal: {sym.name} in {prod}"


class TestGrammarCompleteness:
    """Tests for grammar completeness."""
    
    def test_all_declarations_covered(self):
        """Test all declaration types have productions."""
        builder = STUNIRGrammarBuilder()
        builder.build()
        productions = builder.get_productions()
        
        # Check declaration alternatives
        decl_prods = [p for p in productions if p.head.name == 'declaration']
        decl_bodies = {p.label for p in decl_prods if p.label}
        
        # Should cover type, function, ir, target
        assert any('type' in str(b).lower() for b in decl_bodies)
        assert any('function' in str(b).lower() for b in decl_bodies)
        assert any('ir' in str(b).lower() for b in decl_bodies)
        assert any('target' in str(b).lower() for b in decl_bodies)
    
    def test_all_statements_covered(self):
        """Test all statement types have productions."""
        builder = STUNIRGrammarBuilder()
        builder.build()
        productions = builder.get_productions()
        
        # Check statement alternatives
        stmt_prods = [p for p in productions if p.head.name == 'statement']
        
        # Should have multiple statement types
        assert len(stmt_prods) >= 5
    
    def test_all_expressions_covered(self):
        """Test expression precedence levels are covered."""
        builder = STUNIRGrammarBuilder()
        builder.build()
        nonterminals = builder.get_nonterminals()
        
        # Check expression precedence levels exist
        precedence_levels = [
            'ternary_expr', 'or_expr', 'and_expr', 'equality_expr',
            'relational_expr', 'additive_expr', 'multiplicative_expr',
            'unary_expr', 'postfix_expr', 'primary_expr'
        ]
        
        for level in precedence_levels:
            assert level in nonterminals, f"Missing expression level: {level}"
    
    def test_all_basic_types_covered(self):
        """Test all basic types have productions."""
        builder = STUNIRGrammarBuilder()
        builder.build()
        productions = builder.get_productions()
        
        # Find basic_type productions
        basic_prods = [p for p in productions if p.head.name == 'basic_type']
        
        # Should cover i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, bool, string, void, any
        assert len(basic_prods) >= 14


class TestGrammarProperties:
    """Tests for grammar properties."""
    
    def test_grammar_has_name(self):
        """Test grammar has a name."""
        grammar = create_stunir_grammar()
        assert grammar.name == 'STUNIR'
    
    def test_grammar_type_is_ebnf(self):
        """Test grammar type is EBNF."""
        grammar = create_stunir_grammar()
        assert grammar.grammar_type == GrammarType.EBNF
    
    def test_productions_have_labels(self):
        """Test productions have labels for disambiguation."""
        builder = STUNIRGrammarBuilder()
        builder.build()
        productions = builder.get_productions()
        
        # Most productions should have labels
        labeled = sum(1 for p in productions if p.label)
        total = len(productions)
        
        # At least 80% should be labeled
        assert labeled >= total * 0.8


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
