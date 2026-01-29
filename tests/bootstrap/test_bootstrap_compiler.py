"""
Tests for STUNIR Bootstrap Compiler.

Tests the bootstrap compiler's ability to parse STUNIR source code
and produce ASTs.
"""

import pytest

from bootstrap.bootstrap_compiler import (
    BootstrapCompiler,
    BootstrapResult,
    STUNIRASTNode,
    STUNIRToken,
    RecursiveDescentParser,
    SimpleLexer,
    STUNIRParseError,
)


class TestBootstrapCompiler:
    """Tests for BootstrapCompiler class."""
    
    @pytest.fixture
    def compiler(self):
        """Create a bootstrap compiler instance."""
        return BootstrapCompiler()
    
    def test_compiler_initialization(self, compiler):
        """Test compiler initializes correctly."""
        assert compiler is not None
    
    def test_parse_empty_module(self, compiler):
        """Test parsing empty module."""
        source = "module empty;"
        result = compiler.parse(source)
        
        assert result.success
        assert result.ast is not None
        assert result.ast.kind == 'program'
    
    def test_parse_module_with_block(self, compiler):
        """Test parsing module with block."""
        source = "module test { }"
        result = compiler.parse(source)
        
        assert result.success
        assert result.ast.kind == 'program'
    
    def test_parse_simple_function(self, compiler):
        """Test parsing simple function."""
        source = """
        module test;
        function main(): i32 {
            return 0;
        }
        """
        result = compiler.parse(source)
        
        assert result.success
        assert len(result.ast.children) >= 2  # module + function
    
    def test_parse_function_with_params(self, compiler):
        """Test parsing function with parameters."""
        source = """
        module test;
        function add(a: i32, b: i32): i32 {
            return a + b;
        }
        """
        result = compiler.parse(source)
        
        assert result.success
        
        # Find function definition
        funcs = result.ast.find_children('function_def')
        assert len(funcs) == 1
        assert funcs[0].attributes['name'] == 'add'
    
    def test_parse_type_definition(self, compiler):
        """Test parsing type definition."""
        source = """
        module test;
        type Point {
            x: i32;
            y: i32;
        }
        """
        result = compiler.parse(source)
        
        assert result.success
        
        # Find type definition
        types = result.ast.find_children('type_def')
        assert len(types) == 1
        assert types[0].attributes['name'] == 'Point'
    
    def test_parse_type_alias(self, compiler):
        """Test parsing type alias."""
        source = """
        module test;
        type MyInt = i32;
        """
        result = compiler.parse(source)
        
        assert result.success
    
    def test_parse_ir_definition(self, compiler):
        """Test parsing IR definition."""
        source = """
        module test;
        ir BinaryOp {
            op: string;
            child left: Expr;
            child right: Expr;
        }
        """
        result = compiler.parse(source)
        
        assert result.success
        
        # Find IR definition
        irs = result.ast.find_children('ir_def')
        assert len(irs) == 1
        assert irs[0].attributes['name'] == 'BinaryOp'
    
    def test_parse_target_definition(self, compiler):
        """Test parsing target definition."""
        source = """
        module test;
        target Python {
            extension: ".py";
        }
        """
        result = compiler.parse(source)
        
        assert result.success
        
        # Find target definition
        targets = result.ast.find_children('target_def')
        assert len(targets) == 1
        assert targets[0].attributes['name'] == 'Python'
    
    def test_parse_var_declarations(self, compiler):
        """Test parsing variable declarations."""
        source = """
        module test;
        function main(): i32 {
            let x = 42;
            var y: i32;
            var z: i32 = 0;
            return x;
        }
        """
        result = compiler.parse(source)
        
        assert result.success
    
    def test_parse_if_statement(self, compiler):
        """Test parsing if statement."""
        source = """
        module test;
        function main(): i32 {
            if x > 0 {
                return 1;
            } else {
                return 0;
            }
        }
        """
        result = compiler.parse(source)
        
        assert result.success
    
    def test_parse_while_statement(self, compiler):
        """Test parsing while statement."""
        source = """
        module test;
        function main(): i32 {
            while x > 0 {
                x = x - 1;
            }
            return 0;
        }
        """
        result = compiler.parse(source)
        
        assert result.success
    
    def test_parse_for_statement(self, compiler):
        """Test parsing for statement."""
        source = """
        module test;
        function main(): i32 {
            for item in items {
                total = total + item;
            }
            return total;
        }
        """
        result = compiler.parse(source)
        
        assert result.success
    
    def test_parse_match_statement(self, compiler):
        """Test parsing match statement."""
        source = """
        module test;
        function main(): i32 {
            match x {
                0 => 1,
                1 => 2,
                _ => 0,
            }
        }
        """
        result = compiler.parse(source)
        
        assert result.success
    
    def test_parse_expressions(self, compiler):
        """Test parsing various expressions."""
        source = """
        module test;
        function main(): i32 {
            let a = 1 + 2 * 3;
            let b = (1 + 2) * 3;
            let c = a < b && b > 0;
            let d = x ? 1 : 0;
            return a + b;
        }
        """
        result = compiler.parse(source)
        
        assert result.success
    
    def test_parse_array_literal(self, compiler):
        """Test parsing array literal."""
        source = """
        module test;
        function main(): i32 {
            let arr = [1, 2, 3];
            return arr[0];
        }
        """
        result = compiler.parse(source)
        
        assert result.success
    
    def test_parse_object_literal(self, compiler):
        """Test parsing object literal."""
        source = """
        module test;
        function main(): i32 {
            let obj = {x: 1, y: 2};
            return obj.x;
        }
        """
        result = compiler.parse(source)
        
        assert result.success
    
    def test_parse_function_call(self, compiler):
        """Test parsing function call."""
        source = """
        module test;
        function main(): i32 {
            let result = add(1, 2);
            return result;
        }
        """
        result = compiler.parse(source)
        
        assert result.success
    
    def test_parse_import_declaration(self, compiler):
        """Test parsing import declaration."""
        source = """
        module test {
            import std.io;
            import utils as u;
        }
        """
        result = compiler.parse(source)
        
        assert result.success
    
    def test_parse_export_declaration(self, compiler):
        """Test parsing export declaration."""
        source = """
        module test {
            export main, helper;
        }
        """
        result = compiler.parse(source)
        
        assert result.success
    
    def test_parse_emit_statement(self, compiler):
        """Test parsing emit statement."""
        source = """
        module test;
        target Python {
            emit BinaryOp(node: BinaryOp) {
                emit node.left;
                emit " + ";
                emit node.right;
            }
        }
        """
        result = compiler.parse(source)
        
        assert result.success


class TestCompilerErrors:
    """Tests for compiler error handling."""
    
    @pytest.fixture
    def compiler(self):
        return BootstrapCompiler()
    
    def test_missing_module(self, compiler):
        """Test error on missing module declaration."""
        source = "function main(): i32 { return 0; }"
        result = compiler.parse(source)
        
        # Should fail - no module declaration
        assert not result.success or len(result.errors) > 0
    
    def test_unclosed_brace(self, compiler):
        """Test error on unclosed brace."""
        source = """
        module test;
        function main(): i32 {
            return 0;
        """
        result = compiler.parse(source)
        
        assert not result.success
        assert len(result.errors) > 0
    
    def test_unexpected_token(self, compiler):
        """Test error on unexpected token."""
        source = """
        module test;
        function main(): i32 {
            let x = ;
            return 0;
        }
        """
        result = compiler.parse(source)
        
        assert not result.success


class TestASTStructure:
    """Tests for AST structure."""
    
    @pytest.fixture
    def compiler(self):
        return BootstrapCompiler()
    
    def test_ast_node_kind(self, compiler):
        """Test AST node has correct kind."""
        source = "module test;"
        result = compiler.parse(source)
        
        assert result.ast.kind == 'program'
    
    def test_ast_node_children(self, compiler):
        """Test AST node has children."""
        source = """
        module test;
        function main(): i32 {
            return 0;
        }
        """
        result = compiler.parse(source)
        
        assert len(result.ast.children) >= 2
    
    def test_ast_node_attributes(self, compiler):
        """Test AST node has attributes."""
        source = """
        module test;
        function main(): i32 {
            return 0;
        }
        """
        result = compiler.parse(source)
        
        module_decl = result.ast.children[0]
        assert 'name' in module_decl.attributes
        assert module_decl.attributes['name'] == 'test'
    
    def test_ast_node_location(self, compiler):
        """Test AST node has location."""
        source = "module test;"
        result = compiler.parse(source)
        
        # Root should have location
        assert result.ast.location is not None or \
               result.ast.children[0].location is not None
    
    def test_find_children(self, compiler):
        """Test finding children by kind."""
        source = """
        module test;
        function a(): i32 { return 0; }
        function b(): i32 { return 1; }
        """
        result = compiler.parse(source)
        
        funcs = result.ast.find_children('function_def')
        assert len(funcs) == 2
    
    def test_get_attr(self, compiler):
        """Test getting attribute with default."""
        node = STUNIRASTNode('test')
        
        assert node.get_attr('missing') is None
        assert node.get_attr('missing', 'default') == 'default'
    
    def test_set_attr(self, compiler):
        """Test setting attribute."""
        node = STUNIRASTNode('test')
        node.set_attr('name', 'value')
        
        assert node.get_attr('name') == 'value'


class TestBootstrapResult:
    """Tests for BootstrapResult structure."""
    
    @pytest.fixture
    def compiler(self):
        return BootstrapCompiler()
    
    def test_result_success(self, compiler):
        """Test result success flag."""
        source = "module test;"
        result = compiler.parse(source)
        
        assert result.success is True
    
    def test_result_tokens(self, compiler):
        """Test result contains tokens."""
        source = "module test;"
        result = compiler.parse(source)
        
        assert len(result.tokens) > 0
    
    def test_result_ast(self, compiler):
        """Test result contains AST."""
        source = "module test;"
        result = compiler.parse(source)
        
        assert result.ast is not None
    
    def test_result_errors_on_failure(self, compiler):
        """Test result contains errors on failure."""
        source = "module ;"  # Invalid - missing name
        result = compiler.parse(source)
        
        if not result.success:
            assert len(result.errors) > 0


class TestValidateAST:
    """Tests for AST validation."""
    
    @pytest.fixture
    def compiler(self):
        return BootstrapCompiler()
    
    def test_validate_valid_ast(self, compiler):
        """Test validating a valid AST."""
        source = "module test;"
        result = compiler.parse(source)
        
        errors = compiler.validate_ast(result.ast)
        assert len(errors) == 0
    
    def test_validate_missing_module(self, compiler):
        """Test validating AST without module."""
        # Create AST without module
        ast = STUNIRASTNode('program')
        
        errors = compiler.validate_ast(ast)
        assert len(errors) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
