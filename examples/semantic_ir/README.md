# STUNIR Semantic IR Examples

This directory contains example Semantic IR files demonstrating various target categories and programming paradigms.

## Examples

### 1. simple_function.json
**Target:** Native  
**Description:** Basic function that adds two integers.  
**Key Features:**
- Simple binary expression
- Primitive types (i32)
- Return statement
- Basic module structure

### 2. embedded_startup.json
**Target:** Embedded, Realtime, Safety-Critical  
**Safety Level:** DO-178C Level A  
**Description:** Embedded system startup code with interrupt handler.  
**Key Features:**
- Entry point function
- Interrupt handler with vector specification
- Stack usage annotations
- Memory section attributes
- Safety-critical metadata

### 3. gpu_kernel.json
**Target:** GPU  
**Description:** GPU kernel for vector addition.  
**Key Features:**
- Kernel execution model
- Workgroup size specification
- Global memory address space
- Pointer types with mutability

### 4. wasm_module.json
**Target:** WebAssembly  
**Description:** WASM module with exported function.  
**Key Features:**
- WASM import/export declarations
- WASM type annotations
- Module import from environment

### 5. lisp_expression.json
**Target:** Functional  
**Description:** Recursive factorial function.  
**Key Features:**
- Ternary conditional expression
- Recursive function call
- Pure function annotation
- Tail recursion hint

## Validation

All examples can be validated using the Semantic IR validator:

```bash
python tools/semantic_ir/validator.py examples/semantic_ir/simple_function.json
```

## Target Categories Covered

- **Native**: General-purpose native code generation
- **Embedded**: Microcontroller and embedded systems
- **Realtime**: Real-time operating systems
- **Safety-Critical**: DO-178C Level A certified systems
- **GPU**: Graphics processing units (CUDA, OpenCL, Vulkan)
- **WASM**: WebAssembly for browser and edge computing
- **Functional**: Functional programming languages (Lisp, Scheme, Haskell)

## Schema Compliance

All examples comply with the Semantic IR Schema v1.0 defined in `schemas/semantic_ir/`.
