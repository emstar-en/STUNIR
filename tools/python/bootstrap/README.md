# STUNIR Bootstrap Package

**Phase 6D: STUNIR Self-Specification and Bootstrap Compilation**

This package enables STUNIR to be self-hosting by providing:
- STUNIR grammar specification using Grammar IR (Phase 6A)
- STUNIR lexer specification using LexerSpec (Phase 6C)
- Tool generator for creating parser and lexer
- Bootstrap compiler for parsing STUNIR source files
- Self-hosting validation

## Quick Start

```python
from bootstrap import BootstrapCompiler

# Parse STUNIR source code
compiler = BootstrapCompiler()
result = compiler.parse("""
module hello;

function main(): i32 {
    return 0;
}
""")

if result.success:
    print(f"Parsed successfully!")
    print(f"Module: {result.ast.children[0].attributes['name']}")
else:
    for error in result.errors:
        print(f"Error: {error}")
```

## Package Structure

```
bootstrap/
├── __init__.py              # Package exports
├── stunir_grammar.py        # STUNIR grammar specification
├── stunir_lexer.py          # STUNIR lexer specification
├── generate_tools.py        # Tool generation script
├── bootstrap_compiler.py    # Bootstrap compiler implementation
├── self_host_validator.py   # Self-hosting validation
└── README.md                # This file
```

## Components

### STUNIRGrammarBuilder

Builds the STUNIR grammar using the Grammar IR framework:

```python
from bootstrap import STUNIRGrammarBuilder

builder = STUNIRGrammarBuilder()
grammar = builder.build()

print(f"Grammar: {grammar.name}")
print(f"Start symbol: {grammar.start_symbol.name}")
print(f"Productions: {len(builder.get_productions())}")
```

### STUNIRLexerBuilder

Builds the STUNIR lexer specification:

```python
from bootstrap import STUNIRLexerBuilder, STUNIR_KEYWORDS

builder = STUNIRLexerBuilder()
spec = builder.build()

print(f"Lexer: {spec.name}")
print(f"Tokens: {len(spec.tokens)}")
print(f"Keywords: {len(STUNIR_KEYWORDS)}")
```

### ToolGenerator

Generates parser and lexer code from specifications:

```python
from bootstrap import ToolGenerator

generator = ToolGenerator()
result = generator.generate_all()

if result.success:
    print(f"Parser saved to: {result.parser_path}")
    print(f"Lexer saved to: {result.lexer_path}")
```

### BootstrapCompiler

Parses STUNIR source files:

```python
from bootstrap import BootstrapCompiler
from pathlib import Path

compiler = BootstrapCompiler()

# Parse from string
result = compiler.parse(source_code)

# Parse from file
result = compiler.parse_file(Path("program.stunir"))
```

### SelfHostValidator

Validates STUNIR self-hosting capability:

```python
from bootstrap import SelfHostValidator

validator = SelfHostValidator()
result = validator.validate()

if result.self_hosting_valid:
    print("STUNIR is self-hosting!")
else:
    for error in result.errors:
        print(f"Error: {error}")
```

## STUNIR Language

STUNIR is a domain-specific language for specifying compilers and code generators.

### Module Declaration

```stunir
module mymodule;

// Or with a body:
module mymodule {
    import std.io;
    export main;
}
```

### Type Definitions

```stunir
// Type alias
type MyInt = i32;

// Struct type
type Point {
    x: i32;
    y: i32;
}

// Variant type (sum type)
type Option<T> {
    | Some(T)
    | None
}
```

### Function Definitions

```stunir
function add(a: i32, b: i32): i32 {
    return a + b;
}

function greet(name: string, greeting: string = "Hello"): string {
    return greeting + ", " + name + "!";
}
```

### IR Node Definitions

```stunir
ir BinaryOp {
    op: string;
    child left: Expr;
    child right: Expr;
    op evaluate(): i32;
}
```

### Target Definitions

```stunir
target Python {
    extension: ".py";
    
    emit BinaryOp(node: BinaryOp) {
        emit node.left;
        emit " " + node.op + " ";
        emit node.right;
    }
}
```

### Statements

```stunir
// Variable declarations
let x = 42;
var y: i32;
var z: i32 = 0;

// Control flow
if condition {
    // ...
} else {
    // ...
}

while condition {
    // ...
}

for item in items {
    // ...
}

match value {
    0 => "zero",
    1 => "one",
    _ => "other",
}

return result;
emit code;
```

### Expressions

```stunir
// Arithmetic
a + b * c - d / e % f

// Comparison
a == b && c != d
a < b || c >= d

// Ternary
condition ? then_value : else_value

// Literals
42          // integer
3.14        // float
"hello"     // string
true        // boolean
null        // null
[1, 2, 3]   // array
{x: 1, y: 2} // object

// Access
obj.field
arr[index]
func(args)
```

## Bootstrap Process

The bootstrap process enables STUNIR to be self-hosting:

1. **Stage 0**: Define grammar/lexer specs manually (stunir_grammar.py, stunir_lexer.py)
2. **Stage 1**: Generate parser/lexer from specs (generate_tools.py)
3. **Stage 2**: Parse STUNIR source with generated tools (bootstrap_compiler.py)
4. **Stage 3**: Validate by parsing STUNIR's own specs (self_host_validator.py)

## Testing

Run tests with pytest:

```bash
pytest tests/bootstrap/ -v
```

Test coverage:
- Grammar specification tests
- Lexer specification tests
- Bootstrap compiler tests
- Self-hosting validation tests

## See Also

- [STUNIR Language Specification](../docs/STUNIR_LANGUAGE_SPEC.md)
- [Example Programs](../examples/stunir/)
- [Phase 6A: Grammar IR](../ir/grammar/README.md)
- [Phase 6B: Parser Generation](../ir/parser/README.md)
- [Phase 6C: Lexer Generation](../ir/lexer/README.md)
