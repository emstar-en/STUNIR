#!/usr/bin/env python3
"""Tests for Parser Emitters.

Tests:
- Python parser emitter
- Rust parser emitter
- C parser emitter
- Table-driven parser emitter
"""

import pytest
import json
import sys
import os

# Add the repository root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ir.grammar.grammar_ir import Grammar, GrammarType
from ir.grammar.symbol import terminal, nonterminal
from ir.grammar.production import ProductionRule

from ir.parser.lr_parser import LRParserGenerator
from ir.parser.ll_parser import LLParserGenerator
from ir.parser.parse_table import ParserType
from ir.parser.ast_node import ASTSchema, ASTNodeSpec

from targets.parser.base import ParserEmitterResult
from targets.parser.python_parser import PythonParserEmitter
from targets.parser.rust_parser import RustParserEmitter
from targets.parser.c_parser import CParserEmitter
from targets.parser.table_driven import TableDrivenEmitter


@pytest.fixture
def simple_grammar():
    """Create a simple grammar for testing."""
    S = nonterminal("S")
    num = terminal("num")
    
    grammar = Grammar("simple", GrammarType.BNF, S)
    grammar.add_production(ProductionRule(S, (num,)))
    
    return grammar


@pytest.fixture
def expression_grammar():
    """Create an expression grammar for testing."""
    E = nonterminal("E")
    T = nonterminal("T")
    plus = terminal("+")
    num = terminal("num")
    
    grammar = Grammar("expr", GrammarType.BNF, E)
    grammar.add_production(ProductionRule(E, (E, plus, T)))
    grammar.add_production(ProductionRule(E, (T,)))
    grammar.add_production(ProductionRule(T, (num,)))
    
    return grammar


@pytest.fixture
def lr_result(simple_grammar):
    """Generate LR parser result for testing."""
    gen = LRParserGenerator(ParserType.LALR1)
    return gen.generate(simple_grammar)


@pytest.fixture
def ll_result():
    """Generate LL parser result for testing."""
    S = nonterminal("S")
    a = terminal("a")
    b = terminal("b")
    
    grammar = Grammar("ll_test", GrammarType.BNF, S)
    grammar.add_production(ProductionRule(S, (a,)))
    grammar.add_production(ProductionRule(S, (b,)))
    
    gen = LLParserGenerator()
    return gen.generate(grammar), grammar


class TestPythonParserEmitter:
    """Test Python parser code emitter."""
    
    def test_emitter_creation(self):
        """Test creating Python emitter."""
        emitter = PythonParserEmitter()
        assert emitter.LANGUAGE == "python"
        assert emitter.FILE_EXTENSION == ".py"
    
    def test_emit_lr_parser(self, lr_result, simple_grammar):
        """Test emitting LR parser in Python."""
        emitter = PythonParserEmitter()
        result = emitter.emit(lr_result, simple_grammar)
        
        assert isinstance(result, ParserEmitterResult)
        assert "class Parser" in result.code
        assert "ACTION" in result.code or "action" in result.code.lower()
        assert "PRODUCTIONS" in result.code
    
    def test_emit_ll_parser(self, ll_result):
        """Test emitting LL parser in Python."""
        parser_result, grammar = ll_result
        emitter = PythonParserEmitter()
        result = emitter.emit(parser_result, grammar)
        
        assert "class Parser" in result.code
        assert "LL_TABLE" in result.code or "ll_table" in result.code.lower()
    
    def test_emit_ast_nodes(self, lr_result, simple_grammar):
        """Test emitting AST nodes in Python."""
        emitter = PythonParserEmitter()
        result = emitter.emit(lr_result, simple_grammar)
        
        if lr_result.ast_schema:
            assert "@dataclass" in result.ast_code
            assert "class" in result.ast_code
    
    def test_manifest_generation(self, lr_result, simple_grammar):
        """Test manifest generation."""
        emitter = PythonParserEmitter()
        result = emitter.emit(lr_result, simple_grammar)
        
        assert "schema" in result.manifest
        assert "parser_code_hash" in result.manifest
        assert "python" in result.manifest["schema"]
    
    def test_emit_token_class(self, lr_result, simple_grammar):
        """Test Token class emission."""
        emitter = PythonParserEmitter()
        result = emitter.emit(lr_result, simple_grammar)
        
        assert "class Token" in result.code
        assert "type:" in result.code or "token_type" in result.code.lower()


class TestRustParserEmitter:
    """Test Rust parser code emitter."""
    
    def test_emitter_creation(self):
        """Test creating Rust emitter."""
        emitter = RustParserEmitter()
        assert emitter.LANGUAGE == "rust"
        assert emitter.FILE_EXTENSION == ".rs"
    
    def test_emit_lr_parser(self, lr_result, simple_grammar):
        """Test emitting LR parser in Rust."""
        emitter = RustParserEmitter()
        result = emitter.emit(lr_result, simple_grammar)
        
        assert isinstance(result, ParserEmitterResult)
        assert "struct Parser" in result.code or "pub struct Parser" in result.code
        assert "fn get_action" in result.code or "fn parse" in result.code
    
    def test_emit_cargo_toml(self, lr_result, simple_grammar):
        """Test Cargo.toml generation."""
        emitter = RustParserEmitter()
        result = emitter.emit(lr_result, simple_grammar)
        
        assert "Cargo.toml" in result.auxiliary_files
        cargo = result.auxiliary_files["Cargo.toml"]
        assert "[package]" in cargo
        assert "edition" in cargo
    
    def test_token_enum(self, lr_result, simple_grammar):
        """Test TokenType enum generation."""
        emitter = RustParserEmitter()
        result = emitter.emit(lr_result, simple_grammar)
        
        assert "enum TokenType" in result.code or "pub enum TokenType" in result.code
        assert "Eof" in result.code
    
    def test_action_enum(self, lr_result, simple_grammar):
        """Test Action enum generation."""
        emitter = RustParserEmitter()
        result = emitter.emit(lr_result, simple_grammar)
        
        assert "enum Action" in result.code
        assert "Shift" in result.code
        assert "Reduce" in result.code


class TestCParserEmitter:
    """Test C parser code emitter."""
    
    def test_emitter_creation(self):
        """Test creating C emitter."""
        emitter = CParserEmitter()
        assert emitter.LANGUAGE == "c"
        assert emitter.FILE_EXTENSION == ".c"
    
    def test_emit_lr_parser(self, lr_result, simple_grammar):
        """Test emitting LR parser in C."""
        emitter = CParserEmitter()
        result = emitter.emit(lr_result, simple_grammar)
        
        assert isinstance(result, ParserEmitterResult)
        assert "#include" in result.code
        assert "Parser" in result.code
    
    def test_emit_header_file(self, lr_result, simple_grammar):
        """Test header file generation."""
        emitter = CParserEmitter()
        result = emitter.emit(lr_result, simple_grammar)
        
        assert "parser.h" in result.auxiliary_files
        header = result.auxiliary_files["parser.h"]
        assert "#ifndef" in header
        assert "#define" in header
        assert "typedef" in header
    
    def test_emit_makefile(self, lr_result, simple_grammar):
        """Test Makefile generation."""
        emitter = CParserEmitter()
        result = emitter.emit(lr_result, simple_grammar)
        
        assert "Makefile" in result.auxiliary_files
        makefile = result.auxiliary_files["Makefile"]
        assert "CC" in makefile or "gcc" in makefile
        assert "clean" in makefile
    
    def test_token_types_enum(self, lr_result, simple_grammar):
        """Test TokenType enum generation in header."""
        emitter = CParserEmitter()
        result = emitter.emit(lr_result, simple_grammar)
        
        header = result.auxiliary_files["parser.h"]
        assert "typedef enum" in header
        assert "TokenType" in header
        assert "TOK_EOF" in header
    
    def test_c89_standard(self, lr_result, simple_grammar):
        """Test C89 standard compliance option."""
        emitter = CParserEmitter(config={'c_standard': 'c89'})
        result = emitter.emit(lr_result, simple_grammar)
        
        makefile = result.auxiliary_files["Makefile"]
        assert "-ansi" in makefile or "c89" in makefile


class TestTableDrivenEmitter:
    """Test table-driven parser emitter."""
    
    def test_emitter_creation(self):
        """Test creating table-driven emitter."""
        emitter = TableDrivenEmitter()
        assert emitter.LANGUAGE == "table_driven"
        assert emitter.FILE_EXTENSION == ".json"
    
    def test_emit_lr_tables(self, lr_result, simple_grammar):
        """Test emitting LR tables as JSON."""
        emitter = TableDrivenEmitter()
        result = emitter.emit(lr_result, simple_grammar)
        
        # Code should be valid JSON
        data = json.loads(result.code)
        
        assert "schema" in data
        assert "grammar_name" in data
        assert "table_type" in data
        assert data["table_type"] == "LR"
    
    def test_emit_ll_tables(self, ll_result):
        """Test emitting LL tables as JSON."""
        parser_result, grammar = ll_result
        emitter = TableDrivenEmitter()
        result = emitter.emit(parser_result, grammar)
        
        data = json.loads(result.code)
        
        assert data["table_type"] == "LL"
        assert "table" in data
    
    def test_compact_option(self, lr_result, simple_grammar):
        """Test compact JSON option."""
        emitter = TableDrivenEmitter(config={'compact': True})
        result = emitter.emit(lr_result, simple_grammar)
        
        # Compact JSON should have no indentation
        assert "\n  " not in result.code
    
    def test_debug_info(self, lr_result, simple_grammar):
        """Test debug info inclusion."""
        emitter = TableDrivenEmitter(config={'include_debug': True})
        result = emitter.emit(lr_result, simple_grammar)
        
        data = json.loads(result.code)
        assert "debug" in data
        assert "state_count" in data["debug"]
    
    def test_runtime_code(self, lr_result, simple_grammar):
        """Test runtime code generation."""
        emitter = TableDrivenEmitter()
        result = emitter.emit(lr_result, simple_grammar)
        
        assert "parser_runtime.py" in result.auxiliary_files
        runtime = result.auxiliary_files["parser_runtime.py"]
        assert "class TableDrivenParser" in runtime
    
    def test_productions_in_output(self, lr_result, simple_grammar):
        """Test productions in JSON output."""
        emitter = TableDrivenEmitter()
        result = emitter.emit(lr_result, simple_grammar)
        
        data = json.loads(result.code)
        assert "productions" in data
        assert len(data["productions"]) > 0
    
    def test_ast_schema_json(self, lr_result, simple_grammar):
        """Test AST schema as JSON."""
        emitter = TableDrivenEmitter()
        result = emitter.emit(lr_result, simple_grammar)
        
        if result.ast_code:
            ast_data = json.loads(result.ast_code)
            assert "schema" in ast_data
            assert "nodes" in ast_data


class TestEmitterCommon:
    """Test common emitter functionality."""
    
    def test_manifest_has_hash(self, lr_result, simple_grammar):
        """Test manifest has hash."""
        for emitter_class in [PythonParserEmitter, RustParserEmitter, 
                              CParserEmitter, TableDrivenEmitter]:
            emitter = emitter_class()
            result = emitter.emit(lr_result, simple_grammar)
            
            assert "manifest_hash" in result.manifest
            assert "parser_code_hash" in result.manifest
    
    def test_total_size(self, lr_result, simple_grammar):
        """Test total size calculation."""
        emitter = CParserEmitter()
        result = emitter.emit(lr_result, simple_grammar)
        
        expected_size = len(result.code) + len(result.ast_code)
        for content in result.auxiliary_files.values():
            expected_size += len(content)
        
        assert result.total_size() == expected_size
    
    def test_warnings_collection(self, lr_result, simple_grammar):
        """Test warning collection."""
        emitter = PythonParserEmitter()
        result = emitter.emit(lr_result, simple_grammar)
        
        # Result should have warnings list (even if empty)
        assert isinstance(result.warnings, list)


class TestEmitterResult:
    """Test ParserEmitterResult class."""
    
    def test_result_creation(self):
        """Test creating emitter result."""
        result = ParserEmitterResult(
            code="test code",
            ast_code="test ast",
            manifest={"test": "manifest"}
        )
        
        assert result.code == "test code"
        assert result.ast_code == "test ast"
        assert result.manifest == {"test": "manifest"}
    
    def test_add_warning(self):
        """Test adding warning."""
        result = ParserEmitterResult(
            code="", ast_code="", manifest={}
        )
        
        result.add_warning("Test warning")
        assert "Test warning" in result.warnings
    
    def test_add_auxiliary_file(self):
        """Test adding auxiliary file."""
        result = ParserEmitterResult(
            code="", ast_code="", manifest={}
        )
        
        result.add_auxiliary_file("test.txt", "test content")
        assert "test.txt" in result.auxiliary_files
        assert result.auxiliary_files["test.txt"] == "test content"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
