"""
Tests for Lexer Emitters.
"""

import pytest
import json
import sys
import os
import tempfile

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ir.lexer.token_spec import TokenSpec, LexerSpec
from ir.lexer.lexer_generator import LexerGenerator
from targets.lexer import (
    PythonLexerEmitter,
    RustLexerEmitter,
    CLexerEmitter,
    TableDrivenEmitter,
    CompactTableEmitter
)


@pytest.fixture
def simple_spec():
    """Simple lexer specification for testing."""
    return LexerSpec("Simple", [
        TokenSpec("INT", "[0-9]+"),
        TokenSpec("PLUS", "\\+"),
        TokenSpec("MINUS", "-"),
        TokenSpec("ID", "[a-zA-Z_][a-zA-Z0-9_]*"),
        TokenSpec("WS", "[ \\t\\n]+", skip=True)
    ])


@pytest.fixture
def generated_dfa(simple_spec):
    """Generate DFA from simple spec."""
    gen = LexerGenerator(simple_spec)
    gen.generate()
    return gen


class TestPythonLexerEmitter:
    """Test Python lexer emitter."""
    
    def test_emit_produces_code(self, generated_dfa, simple_spec):
        """Emitter produces Python code."""
        emitter = PythonLexerEmitter()
        code = emitter.emit(simple_spec, generated_dfa.minimized_dfa, generated_dfa.table)
        
        assert "class SimpleLexer" in code
        assert "def tokenize" in code
        assert "_TRANSITIONS" in code
        assert "class Token" in code
    
    def test_emit_includes_token_types(self, generated_dfa, simple_spec):
        """Emitted code includes token type constants."""
        emitter = PythonLexerEmitter()
        code = emitter.emit(simple_spec, generated_dfa.minimized_dfa, generated_dfa.table)
        
        assert "class TokenType" in code
        assert 'INT = "INT"' in code
        assert 'PLUS = "PLUS"' in code
    
    def test_emit_includes_skip_tokens(self, generated_dfa, simple_spec):
        """Emitted code handles skip tokens."""
        emitter = PythonLexerEmitter()
        code = emitter.emit(simple_spec, generated_dfa.minimized_dfa, generated_dfa.table)
        
        assert "SKIP_TOKENS" in code
        assert '"WS"' in code
    
    def test_generated_code_executes(self, generated_dfa, simple_spec):
        """Generated Python code executes without errors."""
        emitter = PythonLexerEmitter()
        code = emitter.emit(simple_spec, generated_dfa.minimized_dfa, generated_dfa.table)
        
        # Execute the generated code
        exec_globals = {}
        exec(code, exec_globals)
        
        # Check classes are defined
        assert "SimpleLexer" in exec_globals
        assert "Token" in exec_globals
        assert "TokenType" in exec_globals
    
    def test_generated_lexer_tokenizes(self, generated_dfa, simple_spec):
        """Generated lexer actually tokenizes input."""
        emitter = PythonLexerEmitter()
        code = emitter.emit(simple_spec, generated_dfa.minimized_dfa, generated_dfa.table)
        
        # Execute and use lexer
        exec_globals = {}
        exec(code, exec_globals)
        
        LexerClass = exec_globals["SimpleLexer"]
        lexer = LexerClass("123 + abc")
        tokens = lexer.tokenize()
        
        assert len(tokens) == 3
        assert tokens[0].type == "INT"
        assert tokens[1].type == "PLUS"
        assert tokens[2].type == "ID"


class TestRustLexerEmitter:
    """Test Rust lexer emitter."""
    
    def test_emit_produces_code(self, generated_dfa, simple_spec):
        """Emitter produces Rust code."""
        emitter = RustLexerEmitter()
        code = emitter.emit(simple_spec, generated_dfa.minimized_dfa, generated_dfa.table)
        
        assert "pub struct Token" in code
        assert "pub enum TokenType" in code
        assert "pub struct SimpleLexer" in code
        assert "fn tokenize" in code
    
    def test_emit_includes_token_variants(self, generated_dfa, simple_spec):
        """Emitted code includes TokenType variants."""
        emitter = RustLexerEmitter()
        code = emitter.emit(simple_spec, generated_dfa.minimized_dfa, generated_dfa.table)
        
        assert "INT," in code
        assert "PLUS," in code
        assert "ID," in code
    
    def test_emit_includes_transitions(self, generated_dfa, simple_spec):
        """Emitted code includes DFA transitions."""
        emitter = RustLexerEmitter()
        code = emitter.emit(simple_spec, generated_dfa.minimized_dfa, generated_dfa.table)
        
        assert "static TRANSITIONS" in code
        assert "const START_STATE" in code
    
    def test_emit_includes_test_module(self, generated_dfa, simple_spec):
        """Emitted code includes test module."""
        emitter = RustLexerEmitter()
        code = emitter.emit(simple_spec, generated_dfa.minimized_dfa, generated_dfa.table)
        
        assert "#[cfg(test)]" in code
        assert "mod tests" in code


class TestCLexerEmitter:
    """Test C lexer emitter."""
    
    def test_emit_produces_header_and_source(self, generated_dfa, simple_spec):
        """Emitter produces both header and source."""
        emitter = CLexerEmitter()
        code = emitter.emit(simple_spec, generated_dfa.minimized_dfa, generated_dfa.table)
        
        assert "simple_lexer.h" in code
        assert "simple_lexer.c" in code
    
    def test_emit_includes_token_enum(self, generated_dfa, simple_spec):
        """Emitted code includes token type enum."""
        emitter = CLexerEmitter()
        code = emitter.emit(simple_spec, generated_dfa.minimized_dfa, generated_dfa.table)
        
        assert "typedef enum" in code
        assert "TOKEN_INT" in code
        assert "TOKEN_PLUS" in code
    
    def test_emit_includes_struct(self, generated_dfa, simple_spec):
        """Emitted code includes Token and Lexer structs."""
        emitter = CLexerEmitter()
        code = emitter.emit(simple_spec, generated_dfa.minimized_dfa, generated_dfa.table)
        
        assert "SimpleToken" in code
        assert "SimpleLexer" in code
    
    def test_emit_includes_functions(self, generated_dfa, simple_spec):
        """Emitted code includes API functions."""
        emitter = CLexerEmitter()
        code = emitter.emit(simple_spec, generated_dfa.minimized_dfa, generated_dfa.table)
        
        assert "simple_lexer_init" in code
        assert "simple_lexer_next_token" in code
        assert "simple_lexer_is_skip_token" in code
    
    def test_emit_includes_transitions(self, generated_dfa, simple_spec):
        """Emitted code includes DFA transitions."""
        emitter = CLexerEmitter()
        code = emitter.emit(simple_spec, generated_dfa.minimized_dfa, generated_dfa.table)
        
        assert "static const int32_t transitions[]" in code
        assert "#define START_STATE" in code
    
    def test_header_guards(self, generated_dfa, simple_spec):
        """Header has proper include guards."""
        emitter = CLexerEmitter()
        code = emitter.emit(simple_spec, generated_dfa.minimized_dfa, generated_dfa.table)
        
        assert "#ifndef SIMPLE_LEXER_H" in code
        assert "#define SIMPLE_LEXER_H" in code
        assert "#endif" in code


class TestTableDrivenEmitter:
    """Test table-driven (JSON) emitter."""
    
    def test_emit_produces_valid_json(self, generated_dfa, simple_spec):
        """Emitter produces valid JSON."""
        emitter = TableDrivenEmitter()
        output = emitter.emit(simple_spec, generated_dfa.minimized_dfa, generated_dfa.table)
        
        # Should parse as JSON
        data = json.loads(output)
        assert data is not None
    
    def test_emit_includes_schema(self, generated_dfa, simple_spec):
        """Emitted JSON includes schema version."""
        emitter = TableDrivenEmitter()
        output = emitter.emit(simple_spec, generated_dfa.minimized_dfa, generated_dfa.table)
        
        data = json.loads(output)
        assert data["schema"] == "stunir.lexer.table.v1"
    
    def test_emit_includes_tokens(self, generated_dfa, simple_spec):
        """Emitted JSON includes token definitions."""
        emitter = TableDrivenEmitter()
        output = emitter.emit(simple_spec, generated_dfa.minimized_dfa, generated_dfa.table)
        
        data = json.loads(output)
        assert "tokens" in data
        assert len(data["tokens"]) == 5
        
        token_names = [t["name"] for t in data["tokens"]]
        assert "INT" in token_names
        assert "WS" in token_names
    
    def test_emit_includes_dfa(self, generated_dfa, simple_spec):
        """Emitted JSON includes DFA structure."""
        emitter = TableDrivenEmitter()
        output = emitter.emit(simple_spec, generated_dfa.minimized_dfa, generated_dfa.table)
        
        data = json.loads(output)
        assert "dfa" in data
        assert "start_state" in data["dfa"]
        assert "num_states" in data["dfa"]
        assert "transitions" in data["dfa"]
        assert "accept_states" in data["dfa"]
    
    def test_emit_includes_table(self, generated_dfa, simple_spec):
        """Emitted JSON includes transition table."""
        emitter = TableDrivenEmitter()
        output = emitter.emit(simple_spec, generated_dfa.minimized_dfa, generated_dfa.table)
        
        data = json.loads(output)
        assert "table" in data
        assert "symbol_to_index" in data["table"]
        assert "transitions" in data["table"]
        assert "accept_table" in data["table"]
    
    def test_emit_includes_content_hash(self, generated_dfa, simple_spec):
        """Emitted JSON includes content hash."""
        emitter = TableDrivenEmitter()
        output = emitter.emit(simple_spec, generated_dfa.minimized_dfa, generated_dfa.table)
        
        data = json.loads(output)
        assert "content_hash" in data
        # SHA256 hash is 64 hex characters
        assert len(data["content_hash"]) == 64
    
    def test_emit_pretty(self, generated_dfa, simple_spec):
        """Pretty print produces readable JSON."""
        emitter = TableDrivenEmitter()
        output = emitter.emit_pretty(simple_spec, generated_dfa.minimized_dfa, generated_dfa.table)
        
        # Pretty print has indentation
        assert "\n  " in output
        
        # Still valid JSON
        data = json.loads(output)
        assert data is not None


class TestCompactTableEmitter:
    """Test compact table emitter."""
    
    def test_emit_produces_valid_json(self, generated_dfa, simple_spec):
        """Emitter produces valid JSON."""
        emitter = CompactTableEmitter()
        output = emitter.emit(simple_spec, generated_dfa.minimized_dfa, generated_dfa.table)
        
        data = json.loads(output)
        assert data is not None
    
    def test_emit_is_compact(self, generated_dfa, simple_spec):
        """Compact output is smaller than full output."""
        compact_emitter = CompactTableEmitter()
        full_emitter = TableDrivenEmitter()
        
        compact = compact_emitter.emit(simple_spec, generated_dfa.minimized_dfa, generated_dfa.table)
        full = full_emitter.emit(simple_spec, generated_dfa.minimized_dfa, generated_dfa.table)
        
        assert len(compact) < len(full)
    
    def test_emit_uses_short_keys(self, generated_dfa, simple_spec):
        """Compact output uses short key names."""
        emitter = CompactTableEmitter()
        output = emitter.emit(simple_spec, generated_dfa.minimized_dfa, generated_dfa.table)
        
        data = json.loads(output)
        # Uses short keys
        assert "v" in data  # version
        assert "n" in data  # name
        assert "t" in data  # tokens
        assert "d" in data  # dfa
    
    def test_emit_includes_essential_data(self, generated_dfa, simple_spec):
        """Compact output includes essential data."""
        emitter = CompactTableEmitter()
        output = emitter.emit(simple_spec, generated_dfa.minimized_dfa, generated_dfa.table)
        
        data = json.loads(output)
        assert data["n"] == "Simple"  # lexer name
        assert "INT" in data["t"]  # tokens include INT


class TestEmitterManifest:
    """Test emitter manifest generation."""
    
    def test_python_manifest(self, generated_dfa, simple_spec):
        """Python emitter produces manifest."""
        emitter = PythonLexerEmitter()
        manifest = emitter.get_manifest(simple_spec, generated_dfa.minimized_dfa)
        
        assert manifest["lexer_name"] == "Simple"
        assert manifest["num_tokens"] == 5
        assert "WS" in manifest["skip_tokens"]
        assert manifest["emitter"] == "PythonLexerEmitter"
    
    def test_rust_manifest(self, generated_dfa, simple_spec):
        """Rust emitter produces manifest."""
        emitter = RustLexerEmitter()
        manifest = emitter.get_manifest(simple_spec, generated_dfa.minimized_dfa)
        
        assert manifest["emitter"] == "RustLexerEmitter"
    
    def test_c_manifest(self, generated_dfa, simple_spec):
        """C emitter produces manifest."""
        emitter = CLexerEmitter()
        manifest = emitter.get_manifest(simple_spec, generated_dfa.minimized_dfa)
        
        assert manifest["emitter"] == "CLexerEmitter"


class TestEmitterIntegration:
    """Integration tests for emitters."""
    
    def test_emit_and_execute_python(self):
        """Full integration: generate, emit, and execute Python lexer."""
        spec = LexerSpec("Calc", [
            TokenSpec("NUM", "[0-9]+"),
            TokenSpec("PLUS", "\\+"),
            TokenSpec("MINUS", "-"),
            TokenSpec("MUL", "\\*"),
            TokenSpec("DIV", "/"),
            TokenSpec("LPAREN", "\\("),
            TokenSpec("RPAREN", "\\)"),
            TokenSpec("WS", "[ \\t]+", skip=True)
        ])
        
        gen = LexerGenerator(spec)
        gen.generate()
        
        emitter = PythonLexerEmitter()
        code = emitter.emit(spec, gen.minimized_dfa, gen.table)
        
        # Execute and test
        exec_globals = {}
        exec(code, exec_globals)
        
        lexer = exec_globals["CalcLexer"]("1 + 2 * (3 - 4) / 5")
        tokens = lexer.tokenize()
        
        types = [t.type for t in tokens]
        expected = ["NUM", "PLUS", "NUM", "MUL", "LPAREN", "NUM", "MINUS", "NUM", "RPAREN", "DIV", "NUM"]
        assert types == expected
    
    def test_all_emitters_produce_output(self):
        """All emitters produce non-empty output."""
        spec = LexerSpec("Test", [
            TokenSpec("A", "a"),
            TokenSpec("B", "b")
        ])
        
        gen = LexerGenerator(spec)
        gen.generate()
        
        emitters = [
            PythonLexerEmitter(),
            RustLexerEmitter(),
            CLexerEmitter(),
            TableDrivenEmitter(),
            CompactTableEmitter()
        ]
        
        for emitter in emitters:
            output = emitter.emit(spec, gen.minimized_dfa, gen.table)
            assert len(output) > 0, f"{emitter.__class__.__name__} produced empty output"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
