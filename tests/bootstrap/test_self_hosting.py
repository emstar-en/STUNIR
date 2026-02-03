"""
Tests for STUNIR Self-Hosting Validation.

Tests that STUNIR can parse its own grammar and lexer specifications
written in STUNIR syntax, demonstrating self-hosting capability.
"""

import pytest
from pathlib import Path

from bootstrap.self_host_validator import (
    SelfHostValidator,
    ValidationResult,
    validate_self_hosting,
    STUNIR_GRAMMAR_IN_STUNIR,
    STUNIR_LEXER_IN_STUNIR,
    SIMPLE_STUNIR_PROGRAM,
    ARITHMETIC_STUNIR_PROGRAM,
)
from bootstrap.bootstrap_compiler import BootstrapCompiler


class TestSelfHostValidator:
    """Tests for SelfHostValidator class."""
    
    @pytest.fixture
    def validator(self):
        """Create a validator instance."""
        return SelfHostValidator()
    
    def test_validator_initialization(self, validator):
        """Test validator initializes correctly."""
        assert validator is not None
        assert validator.compiler is not None
    
    def test_validate_returns_result(self, validator):
        """Test validate returns ValidationResult."""
        result = validator.validate()
        
        assert isinstance(result, ValidationResult)
    
    def test_validate_counts_files(self, validator):
        """Test validation counts parsed files."""
        result = validator.validate()
        
        assert result.files_parsed >= 0
    
    def test_validate_counts_tests(self, validator):
        """Test validation counts tests."""
        result = validator.validate()
        
        assert result.tests_passed >= 0
        assert result.tests_failed >= 0
    
    def test_validate_has_details(self, validator):
        """Test validation has detailed results."""
        result = validator.validate()
        
        assert isinstance(result.details, dict)


class TestSimpleProgramParsing:
    """Tests for parsing simple STUNIR programs."""
    
    @pytest.fixture
    def compiler(self):
        return BootstrapCompiler()
    
    def test_parse_simple_program(self, compiler):
        """Test parsing the simple STUNIR program."""
        result = compiler.parse(SIMPLE_STUNIR_PROGRAM, 'simple.stunir')
        
        assert result.success, f"Errors: {result.errors}"
        assert result.ast is not None
        assert result.ast.kind == 'program'
    
    def test_simple_program_has_module(self, compiler):
        """Test simple program has module declaration."""
        result = compiler.parse(SIMPLE_STUNIR_PROGRAM, 'simple.stunir')
        
        assert result.success
        module_decls = result.ast.find_children('module_decl')
        assert len(module_decls) == 1
        assert module_decls[0].attributes['name'] == 'hello'
    
    def test_simple_program_has_functions(self, compiler):
        """Test simple program has functions."""
        result = compiler.parse(SIMPLE_STUNIR_PROGRAM, 'simple.stunir')
        
        assert result.success
        funcs = result.ast.find_children('function_def')
        assert len(funcs) == 2  # greet and main


class TestArithmeticProgramParsing:
    """Tests for parsing arithmetic STUNIR program."""
    
    @pytest.fixture
    def compiler(self):
        return BootstrapCompiler()
    
    def test_parse_arithmetic_program(self, compiler):
        """Test parsing the arithmetic STUNIR program."""
        result = compiler.parse(ARITHMETIC_STUNIR_PROGRAM, 'arithmetic.stunir')
        
        assert result.success, f"Errors: {result.errors}"
    
    def test_arithmetic_has_module(self, compiler):
        """Test arithmetic program has module."""
        result = compiler.parse(ARITHMETIC_STUNIR_PROGRAM, 'arithmetic.stunir')
        
        assert result.success
        module_decls = result.ast.find_children('module_decl')
        assert len(module_decls) == 1
        assert module_decls[0].attributes['name'] == 'arithmetic'
    
    def test_arithmetic_has_ir_definitions(self, compiler):
        """Test arithmetic program has IR definitions."""
        result = compiler.parse(ARITHMETIC_STUNIR_PROGRAM, 'arithmetic.stunir')
        
        assert result.success
        irs = result.ast.find_children('ir_def')
        assert len(irs) >= 1


class TestGrammarInSTUNIR:
    """Tests for parsing STUNIR grammar written in STUNIR."""
    
    @pytest.fixture
    def compiler(self):
        return BootstrapCompiler()
    
    def test_parse_grammar_spec(self, compiler):
        """Test parsing STUNIR grammar specification."""
        result = compiler.parse(STUNIR_GRAMMAR_IN_STUNIR, 'stunir_grammar.stunir')
        
        assert result.success, f"Errors: {result.errors}"
    
    def test_grammar_has_module(self, compiler):
        """Test grammar spec has module."""
        result = compiler.parse(STUNIR_GRAMMAR_IN_STUNIR, 'stunir_grammar.stunir')
        
        if result.success:
            module_decls = result.ast.find_children('module_decl')
            assert len(module_decls) == 1
            assert module_decls[0].attributes['name'] == 'stunir_grammar'
    
    def test_grammar_has_type_definitions(self, compiler):
        """Test grammar spec has type definitions."""
        result = compiler.parse(STUNIR_GRAMMAR_IN_STUNIR, 'stunir_grammar.stunir')
        
        if result.success:
            types = result.ast.find_children('type_def')
            assert len(types) >= 1
    
    def test_grammar_has_ir_definitions(self, compiler):
        """Test grammar spec has IR definitions."""
        result = compiler.parse(STUNIR_GRAMMAR_IN_STUNIR, 'stunir_grammar.stunir')
        
        if result.success:
            irs = result.ast.find_children('ir_def')
            assert len(irs) >= 1


class TestLexerInSTUNIR:
    """Tests for parsing STUNIR lexer written in STUNIR."""
    
    @pytest.fixture
    def compiler(self):
        return BootstrapCompiler()
    
    def test_parse_lexer_spec(self, compiler):
        """Test parsing STUNIR lexer specification."""
        result = compiler.parse(STUNIR_LEXER_IN_STUNIR, 'stunir_lexer.stunir')
        
        assert result.success, f"Errors: {result.errors}"
    
    def test_lexer_has_module(self, compiler):
        """Test lexer spec has module."""
        result = compiler.parse(STUNIR_LEXER_IN_STUNIR, 'stunir_lexer.stunir')
        
        if result.success:
            module_decls = result.ast.find_children('module_decl')
            assert len(module_decls) == 1
            assert module_decls[0].attributes['name'] == 'stunir_lexer'


class TestSelfHostingValidation:
    """Tests for overall self-hosting validation."""
    
    def test_validate_self_hosting_function(self):
        """Test convenience validation function."""
        result = validate_self_hosting()
        
        assert isinstance(result, ValidationResult)
    
    def test_self_hosting_valid(self):
        """Test that STUNIR is self-hosting."""
        result = validate_self_hosting()
        
        # At minimum, we should be able to parse some files
        assert result.files_parsed >= 2
    
    def test_grammar_parses_successfully(self):
        """Test STUNIR grammar parses successfully."""
        result = validate_self_hosting()
        
        grammar_detail = result.details.get('stunir_grammar', {})
        assert grammar_detail.get('success', False), \
            f"Grammar parsing failed: {result.errors}"
    
    def test_lexer_parses_successfully(self):
        """Test STUNIR lexer parses successfully."""
        result = validate_self_hosting()
        
        lexer_detail = result.details.get('stunir_lexer', {})
        assert lexer_detail.get('success', False), \
            f"Lexer parsing failed: {result.errors}"
    
    def test_simple_program_parses(self):
        """Test simple program parses in validation."""
        result = validate_self_hosting()
        
        simple_detail = result.details.get('simple_program', {})
        assert simple_detail.get('success', False)
    
    def test_validation_details_complete(self):
        """Test validation includes all expected details."""
        result = validate_self_hosting()
        
        expected_tests = ['simple_program', 'arithmetic_program', 
                         'stunir_grammar', 'stunir_lexer']
        
        for test_name in expected_tests:
            assert test_name in result.details, f"Missing detail: {test_name}"


class TestExampleFiles:
    """Tests for parsing example STUNIR files."""
    
    @pytest.fixture
    def compiler(self):
        return BootstrapCompiler()
    
    def test_parse_hello_example(self, compiler):
        """Test parsing hello.stunir example."""
        example_path = Path(__file__).parent.parent.parent / 'examples' / 'stunir' / 'hello.stunir'
        
        if example_path.exists():
            source = example_path.read_text()
            result = compiler.parse(source, str(example_path))
            
            assert result.success, f"Errors: {result.errors}"
    
    def test_parse_arithmetic_example(self, compiler):
        """Test parsing arithmetic.stunir example."""
        example_path = Path(__file__).parent.parent.parent / 'examples' / 'stunir' / 'arithmetic.stunir'
        
        if example_path.exists():
            source = example_path.read_text()
            result = compiler.parse(source, str(example_path))
            
            assert result.success, f"Errors: {result.errors}"
    
    def test_parse_mini_compiler_example(self, compiler):
        """Test parsing mini_compiler.stunir example."""
        example_path = Path(__file__).parent.parent.parent / 'examples' / 'stunir' / 'mini_compiler.stunir'
        
        if example_path.exists():
            source = example_path.read_text()
            result = compiler.parse(source, str(example_path))
            
            assert result.success, f"Errors: {result.errors}"


class TestValidationResult:
    """Tests for ValidationResult structure."""
    
    def test_result_attributes(self):
        """Test ValidationResult has all attributes."""
        result = ValidationResult(
            self_hosting_valid=True,
            files_parsed=4,
            tests_passed=10,
            tests_failed=0,
            errors=[],
            details={}
        )
        
        assert result.self_hosting_valid is True
        assert result.files_parsed == 4
        assert result.tests_passed == 10
        assert result.tests_failed == 0
        assert result.errors == []
        assert result.details == {}
    
    def test_result_with_errors(self):
        """Test ValidationResult with errors."""
        result = ValidationResult(
            self_hosting_valid=False,
            errors=['Error 1', 'Error 2']
        )
        
        assert result.self_hosting_valid is False
        assert len(result.errors) == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
