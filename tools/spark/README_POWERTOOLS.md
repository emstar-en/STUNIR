# STUNIR Powertools - Unix Philosophy Implementation

## Overview

The STUNIR toolchain follows the Unix philosophy: **small, focused, composable programs** that do one thing well. The monolithic `spec_to_ir` and `ir_to_code` tools have been deprecated in favor of a collection of 70+ specialized powertools.

## Directory Structure

```
src/
├── codegen/        # Code generation utilities (12 tools)
├── core/           # Pipeline orchestration (19 tools)
├── detection/      # Language and feature detection (5 tools)
├── emitters/       # Language-specific code emitters (1 tool)
├── files/          # File I/O operations (4 tools)
├── functions/      # Function extraction and indexing (6 tools)
├── ir/             # IR manipulation and optimization (8 tools)
├── json/           # JSON processing (12 tools)
├── semantic_ir/    # Semantic IR handling (1 tool)
├── spec/           # Spec validation and assembly (9 tools)
├── types/          # Type definitions and utilities (4 tools)
├── utils/          # General utilities (8 tools)
├── validation/     # Schema validation (4 tools)
├── verification/   # Receipt and hash generation (3 tools)
└── deprecated/     # Legacy monolithic tools
```

## Building

The `powertools.gpr` GNAT project file builds all powertools:

```bash
gprbuild -P powertools.gpr
```

This will compile 137 Ada source files and produce 8 main executables in `bin/`:

- `code_emitter` - Emit code from IR
- `ir_converter` - Convert between IR formats
- `pipeline_driver` - Orchestrate full pipeline
- `spec_assembler` - Assemble specs from parts
- `receipt_link` - Link receipts for attestation
- `code_slice` - Extract code slices
- `code_index` - Index code functions
- `spec_assemble` - Assemble specification files

## Pipeline Orchestration

Instead of one monolithic tool, the pipeline is composed of small utilities:

### Example: Spec → IR → Code

```bash
# Extract functions from source
./bin/code_index input.cpp > funcs.json

# Validate spec schema
json_validate < spec.json

# Convert spec to IR
./bin/ir_converter --from-spec spec.json > output.ir

# Optimize IR
ir_optimize < output.ir > optimized.ir

# Generate code
./bin/code_emitter --lang rust optimized.ir > output.rs

# Generate receipt
receipt_generate output.rs > receipt.json
```

## AI Orchestration

Each tool has:
- **Deterministic behavior**: Same input → same output
- **Clear contracts**: Well-defined I/O formats
- **Exit codes**: 0=success, 1=error, 2=validation failure
- **JSON I/O**: Machine-readable inputs and outputs

AI models can discover and chain these tools by:
1. Reading tool contracts (`--describe` flag planned)
2. Composing pipelines based on user intent
3. Validating each step with receipts

## Development Status

- **Powertools**: 70+ tools organized, not yet fully built/tested
- **Monolithic Tools**: Deprecated, moved to `src/deprecated/`
- **Build System**: `powertools.gpr` created, needs testing
- **Receipts**: Generation logic exists, integration pending

## Next Steps

1. Build all powertools: `gprbuild -P powertools.gpr`
2. Test individual tools
3. Create orchestration examples
4. Add `--describe` flags for AI discovery
5. Generate receipts for deterministic attestation
