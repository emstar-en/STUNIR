# STUNIR Example Programs

This directory contains example programs written in the STUNIR language,
demonstrating various features and capabilities.

## Files

### hello.stunir

A simple "Hello World" program demonstrating:
- Module declaration
- Function definitions
- Type aliases
- Basic expressions

### arithmetic.stunir

An arithmetic expression compiler demonstrating:
- IR node definitions
- Type variants (sum types)
- Pattern matching
- Multiple code generation targets (C, Python)
- Expression evaluation

### mini_compiler.stunir

A complete mini-compiler demonstrating:
- Full AST definition for a source language
- IR instruction set
- Semantic analysis (type checking)
- IR generation
- Target code generation (x86-64 assembly)

## Running Examples

Use the bootstrap compiler to parse these examples:

```python
from bootstrap import BootstrapCompiler

compiler = BootstrapCompiler()

# Parse hello.stunir
with open('examples/stunir/hello.stunir') as f:
    result = compiler.parse(f.read())
    
if result.success:
    print("Parse successful!")
    print(f"Module: {result.ast.children[0].attributes['name']}")
else:
    print("Parse errors:")
    for error in result.errors:
        print(f"  {error}")
```

## STUNIR Language Features Used

| Feature | hello | arithmetic | mini_compiler |
|---------|-------|------------|---------------|
| module | ✓ | ✓ | ✓ |
| function | ✓ | ✓ | ✓ |
| type (alias) | ✓ | ✓ | ✓ |
| type (variant) | | ✓ | ✓ |
| ir | | ✓ | ✓ |
| target | | ✓ | ✓ |
| match | | ✓ | ✓ |
| if/else | | ✓ | ✓ |
| for | | | ✓ |
| while | | | ✓ |
| let/var | ✓ | ✓ | ✓ |
| return | ✓ | ✓ | ✓ |
| emit | | ✓ | ✓ |
