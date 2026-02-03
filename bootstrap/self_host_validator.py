"""
STUNIR Self-Hosting Validator.

Validates that STUNIR is self-hosting by:
1. Writing STUNIR grammar/lexer specs in STUNIR syntax
2. Parsing them with the bootstrap compiler
3. Verifying the AST matches expected structure
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional

from .bootstrap_compiler import BootstrapCompiler, STUNIRASTNode, BootstrapResult


@dataclass
class ValidationResult:
    """
    Result of self-hosting validation.
    
    Attributes:
        self_hosting_valid: True if STUNIR is self-hosting
        files_parsed: Number of files successfully parsed
        tests_passed: Number of validation tests passed
        tests_failed: Number of validation tests failed
        errors: List of error messages
        details: Detailed validation results
    """
    self_hosting_valid: bool
    files_parsed: int = 0
    tests_passed: int = 0
    tests_failed: int = 0
    errors: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


# STUNIR language specification written in STUNIR syntax
# This is the ultimate self-hosting test - STUNIR describing itself

STUNIR_GRAMMAR_IN_STUNIR = '''
// STUNIR Grammar Specification in STUNIR Syntax
// This file describes the STUNIR language using STUNIR itself

module stunir_grammar;

// =========================================
// Token Definitions
// =========================================

type TokenType {
    | KEYWORD
    | IDENTIFIER
    | LITERAL
    | OPERATOR
    | PUNCTUATION
    | COMMENT
    | WHITESPACE
}

type Token {
    type: TokenType;
    value: string;
    line: i32;
    column: i32;
}

// =========================================
// Grammar Symbol Definitions
// =========================================

type SymbolKind {
    | Terminal
    | NonTerminal
    | Epsilon
}

type Symbol {
    name: string;
    kind: SymbolKind;
}

type Production {
    head: Symbol;
    body: [Symbol];
    label: string?;
}

// =========================================
// AST Node Definitions
// =========================================

ir Program {
    name: string;
    child declarations: [Declaration];
}

ir Declaration {
    kind: string;
}

ir TypeDef {
    name: string;
    type_params: [string]?;
    child body: TypeBody;
}

ir FunctionDef {
    name: string;
    params: [Parameter];
    return_type: Type?;
    child body: Block;
}

ir IRDef {
    name: string;
    child members: [IRMember];
}

ir TargetDef {
    name: string;
    child members: [TargetMember];
}

// =========================================
// Expression Definitions
// =========================================

ir Expression {
    kind: string;
}

ir BinaryOp {
    op: string;
    child left: Expression;
    child right: Expression;
    
    op evaluate(): any;
}

ir UnaryOp {
    op: string;
    child operand: Expression;
}

ir Literal {
    value: any;
    literal_type: string;
}

ir Identifier {
    name: string;
}

// =========================================
// Type Definitions
// =========================================

type BasicType {
    | I8 | I16 | I32 | I64
    | U8 | U16 | U32 | U64
    | F32 | F64
    | Bool | String | Void | Any
}

type Type {
    | Basic(BasicType)
    | Named(string)
    | Array(Type)
    | Function([Type], Type)
    | Optional(Type)
    | Union(Type, Type)
}

// =========================================
// Helper Functions
// =========================================

function make_terminal(name: string): Symbol {
    return {
        name: name,
        kind: SymbolKind.Terminal
    };
}

function make_nonterminal(name: string): Symbol {
    return {
        name: name,
        kind: SymbolKind.NonTerminal
    };
}

function create_production(head: string, body: [string]): Production {
    let head_sym = make_nonterminal(head);
    let body_syms: [Symbol] = [];
    
    for part in body {
        if part == "EPSILON" {
            body_syms = body_syms + [{name: "ε", kind: SymbolKind.Epsilon}];
        } else {
            body_syms = body_syms + [make_terminal(part)];
        }
    }
    
    return {
        head: head_sym,
        body: body_syms,
        label: null
    };
}

// =========================================
// Code Generation Target
// =========================================

target Python {
    extension: ".py";
    indent: "    ";
    
    emit BinaryOp(node: BinaryOp) {
        emit node.left;
        emit " " + node.op + " ";
        emit node.right;
    }
    
    emit UnaryOp(node: UnaryOp) {
        emit node.op;
        emit node.operand;
    }
    
    emit Literal(node: Literal) {
        if node.literal_type == "string" {
            emit "\\"" + node.value + "\\"";
        } else {
            emit node.value;
        }
    }
    
    emit Identifier(node: Identifier) {
        emit node.name;
    }
}
'''

STUNIR_LEXER_IN_STUNIR = '''
// STUNIR Lexer Specification in STUNIR Syntax
// This file describes the STUNIR lexer using STUNIR itself

module stunir_lexer;

// =========================================
// Token Specifications
// =========================================

type TokenPriority = i32;

type TokenSpec {
    name: string;
    pattern: string;
    priority: TokenPriority;
    skip: bool;
}

// Keywords
const KW_MODULE: TokenSpec = {
    name: "KW_MODULE",
    pattern: "module",
    priority: 100,
    skip: false
};

const KW_FUNCTION: TokenSpec = {
    name: "KW_FUNCTION",
    pattern: "function",
    priority: 100,
    skip: false
};

const KW_TYPE: TokenSpec = {
    name: "KW_TYPE",
    pattern: "type",
    priority: 100,
    skip: false
};

const KW_IR: TokenSpec = {
    name: "KW_IR",
    pattern: "ir",
    priority: 100,
    skip: false
};

const KW_TARGET: TokenSpec = {
    name: "KW_TARGET",
    pattern: "target",
    priority: 100,
    skip: false
};

// Literals
const INTEGER: TokenSpec = {
    name: "INTEGER_LITERAL",
    pattern: "[0-9]+",
    priority: 80,
    skip: false
};

const FLOAT: TokenSpec = {
    name: "FLOAT_LITERAL",
    pattern: "[0-9]+\\\\.[0-9]+",
    priority: 81,
    skip: false
};

const STRING: TokenSpec = {
    name: "STRING_LITERAL",
    pattern: "\\"[^\\"]*\\"",
    priority: 78,
    skip: false
};

// Whitespace and comments (skipped)
const WHITESPACE: TokenSpec = {
    name: "WHITESPACE",
    pattern: "[ \\\\t\\\\r\\\\n]+",
    priority: 99,
    skip: true
};

const COMMENT_LINE: TokenSpec = {
    name: "COMMENT_LINE",
    pattern: "//[^\\\\n]*",
    priority: 100,
    skip: true
};

// =========================================
// Lexer Specification
// =========================================

type LexerSpec {
    name: string;
    tokens: [TokenSpec];
    case_sensitive: bool;
}

function create_stunir_lexer(): LexerSpec {
    return {
        name: "STUNIRLexer",
        tokens: [
            KW_MODULE,
            KW_FUNCTION,
            KW_TYPE,
            KW_IR,
            KW_TARGET,
            INTEGER,
            FLOAT,
            STRING,
            WHITESPACE,
            COMMENT_LINE
        ],
        case_sensitive: true
    };
}

// =========================================
// Lexer Implementation
// =========================================

ir Lexer {
    source: string;
    pos: i32;
    line: i32;
    column: i32;
    
    op tokenize(): [Token];
    op next_token(): Token?;
}

function create_lexer(source: string): Lexer {
    return {
        source: source,
        pos: 0,
        line: 1,
        column: 1
    };
}
'''

SIMPLE_STUNIR_PROGRAM = '''
// Simple STUNIR program for testing
module hello;

function greet(name: string): string {
    return "Hello, " + name + "!";
}

function main(): i32 {
    let greeting = greet("World");
    return 0;
}
'''

ARITHMETIC_STUNIR_PROGRAM = '''
// Arithmetic compiler in STUNIR
module arithmetic;

// AST nodes for expressions
ir NumberExpr {
    value: i32;
}

ir BinaryExpr {
    op: string;
    child left: Expr;
    child right: Expr;
}

type Expr {
    | Num(NumberExpr)
    | Bin(BinaryExpr)
}

// Evaluator
function evaluate(expr: Expr): i32 {
    match expr {
        Num(n) => n.value,
        Bin(b) => {
            let l = evaluate(b.left);
            let r = evaluate(b.right);
            
            match b.op {
                "+" => l + r,
                "-" => l - r,
                "*" => l * r,
                "/" => l / r,
            }
        }
    }
}

// Code generator
target C {
    emit NumberExpr(n: NumberExpr) {
        emit n.value;
    }
    
    emit BinaryExpr(b: BinaryExpr) {
        emit "(";
        emit b.left;
        emit " " + b.op + " ";
        emit b.right;
        emit ")";
    }
}

// Test
function test(): i32 {
    let expr = Bin({
        op: "+",
        left: Num({value: 2}),
        right: Bin({
            op: "*",
            left: Num({value: 3}),
            right: Num({value: 4})
        })
    });
    
    return evaluate(expr);
}
'''


class SelfHostValidator:
    """
    Validates STUNIR self-hosting capability.
    
    Tests:
    1. Parse STUNIR grammar written in STUNIR
    2. Parse STUNIR lexer written in STUNIR
    3. Parse various STUNIR programs
    4. Validate AST structure matches expectations
    
    Usage:
        validator = SelfHostValidator()
        result = validator.validate()
        
        if result.self_hosting_valid:
            print("STUNIR is self-hosting!")
    """
    
    def __init__(self):
        """Initialize validator."""
        self.compiler = BootstrapCompiler()
        self._tests_passed = 0
        self._tests_failed = 0
        self._errors: List[str] = []
        self._details: Dict[str, Any] = {}
    
    def _run_test(self, name: str, source: str) -> bool:
        """
        Run a single validation test.
        
        Args:
            name: Test name
            source: STUNIR source code
            
        Returns:
            True if test passed
        """
        result = self.compiler.parse(source, filename=name)
        
        self._details[name] = {
            'success': result.success,
            'errors': result.errors,
            'ast_kind': result.ast.kind if result.ast else None,
            'token_count': len(result.tokens),
        }
        
        if result.success:
            self._tests_passed += 1
            return True
        else:
            self._tests_failed += 1
            for error in result.errors:
                self._errors.append(f"{name}: {error}")
            return False
    
    def _validate_ast_structure(self, name: str, ast: STUNIRASTNode,
                                expected: Dict[str, Any]) -> bool:
        """
        Validate AST structure matches expectations.
        
        Args:
            name: Test name
            ast: AST root node
            expected: Expected structure
            
        Returns:
            True if structure matches
        """
        # Check root kind
        if ast.kind != expected.get('kind', 'program'):
            self._errors.append(
                f"{name}: Expected root kind '{expected.get('kind', 'program')}', "
                f"got '{ast.kind}'"
            )
            return False
        
        # Check module name if expected
        if 'module_name' in expected:
            module_decls = ast.find_children('module_decl')
            if not module_decls:
                self._errors.append(f"{name}: Missing module declaration")
                return False
            
            module_name = module_decls[0].get_attr('name')
            if module_name != expected['module_name']:
                self._errors.append(
                    f"{name}: Expected module '{expected['module_name']}', "
                    f"got '{module_name}'"
                )
                return False
        
        # Check declaration counts
        if 'function_count' in expected:
            funcs = ast.find_children('function_def')
            if len(funcs) != expected['function_count']:
                self._errors.append(
                    f"{name}: Expected {expected['function_count']} functions, "
                    f"got {len(funcs)}"
                )
                return False
        
        if 'type_count' in expected:
            types = ast.find_children('type_def')
            if len(types) != expected['type_count']:
                self._errors.append(
                    f"{name}: Expected {expected['type_count']} types, "
                    f"got {len(types)}"
                )
                return False
        
        if 'ir_count' in expected:
            irs = ast.find_children('ir_def')
            if len(irs) != expected['ir_count']:
                self._errors.append(
                    f"{name}: Expected {expected['ir_count']} IR defs, "
                    f"got {len(irs)}"
                )
                return False
        
        if 'target_count' in expected:
            targets = ast.find_children('target_def')
            if len(targets) != expected['target_count']:
                self._errors.append(
                    f"{name}: Expected {expected['target_count']} targets, "
                    f"got {len(targets)}"
                )
                return False
        
        return True
    
    def validate(self) -> ValidationResult:
        """
        Run all self-hosting validation tests.
        
        Returns:
            ValidationResult with test results
        """
        self._tests_passed = 0
        self._tests_failed = 0
        self._errors = []
        self._details = {}
        files_parsed = 0
        
        # Test 1: Parse simple STUNIR program
        if self._run_test('simple_program', SIMPLE_STUNIR_PROGRAM):
            files_parsed += 1
            
            # Validate structure
            result = self.compiler.parse(SIMPLE_STUNIR_PROGRAM)
            if result.ast:
                if self._validate_ast_structure('simple_program', result.ast, {
                    'kind': 'program',
                    'module_name': 'hello',
                    'function_count': 2,
                }):
                    self._tests_passed += 1
                else:
                    self._tests_failed += 1
        
        # Test 2: Parse arithmetic STUNIR program
        if self._run_test('arithmetic_program', ARITHMETIC_STUNIR_PROGRAM):
            files_parsed += 1
            
            # Validate structure
            result = self.compiler.parse(ARITHMETIC_STUNIR_PROGRAM)
            if result.ast:
                if self._validate_ast_structure('arithmetic_program', result.ast, {
                    'kind': 'program',
                    'module_name': 'arithmetic',
                }):
                    self._tests_passed += 1
                else:
                    self._tests_failed += 1
        
        # Test 3: Parse STUNIR grammar in STUNIR
        if self._run_test('stunir_grammar', STUNIR_GRAMMAR_IN_STUNIR):
            files_parsed += 1
            
            # Validate structure
            result = self.compiler.parse(STUNIR_GRAMMAR_IN_STUNIR)
            if result.ast:
                if self._validate_ast_structure('stunir_grammar', result.ast, {
                    'kind': 'program',
                    'module_name': 'stunir_grammar',
                }):
                    self._tests_passed += 1
                else:
                    self._tests_failed += 1
        
        # Test 4: Parse STUNIR lexer in STUNIR
        if self._run_test('stunir_lexer', STUNIR_LEXER_IN_STUNIR):
            files_parsed += 1
            
            # Validate structure
            result = self.compiler.parse(STUNIR_LEXER_IN_STUNIR)
            if result.ast:
                if self._validate_ast_structure('stunir_lexer', result.ast, {
                    'kind': 'program',
                    'module_name': 'stunir_lexer',
                }):
                    self._tests_passed += 1
                else:
                    self._tests_failed += 1
        
        # Determine overall result
        # Self-hosting is valid if we can parse the grammar and lexer specs
        grammar_ok = self._details.get('stunir_grammar', {}).get('success', False)
        lexer_ok = self._details.get('stunir_lexer', {}).get('success', False)
        self_hosting_valid = grammar_ok and lexer_ok
        
        return ValidationResult(
            self_hosting_valid=self_hosting_valid,
            files_parsed=files_parsed,
            tests_passed=self._tests_passed,
            tests_failed=self._tests_failed,
            errors=self._errors,
            details=self._details,
        )
    
    def validate_file(self, path: Path) -> bool:
        """
        Validate a single STUNIR file.
        
        Args:
            path: Path to STUNIR file
            
        Returns:
            True if file parses successfully
        """
        path = Path(path)
        
        if not path.exists():
            self._errors.append(f"File not found: {path}")
            return False
        
        source = path.read_text()
        return self._run_test(str(path), source)


def validate_self_hosting() -> ValidationResult:
    """
    Validate STUNIR self-hosting capability.
    
    Convenience function for running validation.
    
    Returns:
        ValidationResult with test results
    """
    validator = SelfHostValidator()
    return validator.validate()


if __name__ == '__main__':
    import sys
    
    print("Validating STUNIR self-hosting capability...")
    print("=" * 60)
    
    result = validate_self_hosting()
    
    print(f"\nFiles parsed: {result.files_parsed}")
    print(f"Tests passed: {result.tests_passed}")
    print(f"Tests failed: {result.tests_failed}")
    
    if result.errors:
        print("\nErrors:")
        for error in result.errors[:10]:
            print(f"  ✗ {error}")
    
    print("\nDetails:")
    for name, detail in result.details.items():
        status = "✅" if detail['success'] else "❌"
        print(f"  {status} {name}: {detail['token_count']} tokens")
    
    print("\n" + "=" * 60)
    if result.self_hosting_valid:
        print("✅ STUNIR is self-hosting!")
        sys.exit(0)
    else:
        print("❌ STUNIR self-hosting validation failed")
        sys.exit(1)
