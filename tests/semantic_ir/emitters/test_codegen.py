"""Test code generation utilities."""

import pytest
from tools.semantic_ir.emitters.codegen import CodeGenerator
from tools.semantic_ir.emitters.types import IRDataType


class TestCodeGenerator:
    """Test CodeGenerator utilities."""

    def test_sanitize_identifier_valid(self):
        """Test sanitizing valid identifiers."""
        assert CodeGenerator.sanitize_identifier("valid_name") == "valid_name"
        assert CodeGenerator.sanitize_identifier("ValidName123") == "ValidName123"
        assert CodeGenerator.sanitize_identifier("_private") == "_private"

    def test_sanitize_identifier_invalid(self):
        """Test sanitizing invalid identifiers."""
        assert CodeGenerator.sanitize_identifier("123invalid") == "_123invalid"
        assert CodeGenerator.sanitize_identifier("name-with-dash") == "name_with_dash"
        assert CodeGenerator.sanitize_identifier("name.with.dots") == "name_with_dots"
        assert CodeGenerator.sanitize_identifier("name with spaces") == "name_with_spaces"

    def test_escape_string_c(self):
        """Test C-style string escaping."""
        assert CodeGenerator.escape_string('test', 'c') == 'test'
        assert CodeGenerator.escape_string('line1\nline2', 'c') == 'line1\\nline2'
        assert CodeGenerator.escape_string('tab\there', 'c') == 'tab\\there'
        assert CodeGenerator.escape_string('quote"test', 'c') == 'quote\\"test'
        assert CodeGenerator.escape_string('back\\slash', 'c') == 'back\\\\slash'

    def test_generate_include_guard(self):
        """Test C include guard generation."""
        start, end = CodeGenerator.generate_include_guard("test.h")
        
        assert "#ifndef" in start
        assert "#define" in start
        assert "STUNIR_" in start
        assert "#endif" in end

    def test_map_type_to_c(self):
        """Test IR type to C type mapping."""
        assert CodeGenerator.map_type_to_language(IRDataType.I32, "c") == "int32_t"
        assert CodeGenerator.map_type_to_language(IRDataType.U64, "c") == "uint64_t"
        assert CodeGenerator.map_type_to_language(IRDataType.F32, "c") == "float"
        assert CodeGenerator.map_type_to_language(IRDataType.BOOL, "c") == "bool"
        assert CodeGenerator.map_type_to_language(IRDataType.STRING, "c") == "char*"

    def test_map_type_to_rust(self):
        """Test IR type to Rust type mapping."""
        assert CodeGenerator.map_type_to_language(IRDataType.I32, "rust") == "i32"
        assert CodeGenerator.map_type_to_language(IRDataType.U64, "rust") == "u64"
        assert CodeGenerator.map_type_to_language(IRDataType.F32, "rust") == "f32"
        assert CodeGenerator.map_type_to_language(IRDataType.BOOL, "rust") == "bool"
        assert CodeGenerator.map_type_to_language(IRDataType.STRING, "rust") == "String"

    def test_map_type_to_python(self):
        """Test IR type to Python type mapping."""
        assert CodeGenerator.map_type_to_language(IRDataType.I32, "python") == "int"
        assert CodeGenerator.map_type_to_language(IRDataType.F64, "python") == "float"
        assert CodeGenerator.map_type_to_language(IRDataType.BOOL, "python") == "bool"
        assert CodeGenerator.map_type_to_language(IRDataType.STRING, "python") == "str"

    def test_generate_function_signature_c(self):
        """Test C function signature generation."""
        sig = CodeGenerator.generate_function_signature(
            "test_func",
            [("x", "int32_t"), ("y", "int32_t")],
            "int32_t",
            "c"
        )
        assert sig == "int32_t test_func(int32_t x, int32_t y)"

    def test_generate_function_signature_python(self):
        """Test Python function signature generation."""
        sig = CodeGenerator.generate_function_signature(
            "test_func",
            [("x", "int"), ("y", "int")],
            "int",
            "python"
        )
        assert sig == "def test_func(x: int, y: int) -> int:"

    def test_generate_function_signature_rust(self):
        """Test Rust function signature generation."""
        sig = CodeGenerator.generate_function_signature(
            "test_func",
            [("x", "i32"), ("y", "i32")],
            "i32",
            "rust"
        )
        assert sig == "fn test_func(x: i32, y: i32) -> i32"

    def test_format_comment_c(self):
        """Test C-style comment formatting."""
        lines = CodeGenerator.format_comment("Single line", style="c")
        assert lines[0] == "/*"
        assert " * Single line" in lines
        assert lines[-1] == " */"

    def test_format_comment_cpp(self):
        """Test C++-style comment formatting."""
        lines = CodeGenerator.format_comment("Single line", style="cpp")
        assert all(line.startswith("//") for line in lines)

    def test_format_comment_python(self):
        """Test Python docstring formatting."""
        lines = CodeGenerator.format_comment("Single line", style="python")
        assert lines[0] == '"""'
        assert lines[-1] == '"""'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
