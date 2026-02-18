# STUNIR Alpha Release: Spec-to-Code/Environment Pipeline Plan

**Date**: 2025-02-18
**Objective**: Working pipeline from spec_v1.json → semantic_ir.json → **any target environment**
**Target Outputs**: C++, Python, Rust, x86-64 Assembly, ARM Assembly (all first-class IR targets)
**Target Platforms**: Linux (x86-64), Windows (x86-64), macOS (x86-64/ARM64)
**Status**: Execution Plan Ready

---

## Executive Summary

**Goal**: Demonstrate STUNIR's **confluence principle** - one spec generates code for ANY environment through deterministic IR-to-target emission.

**STUNIR Confluence**: Because all emitters work from the same canonical IR, generating assembly is **just as direct** as generating C++ or Python. The pipeline is:

```
spec_v1.json → semantic_ir.json → {C++, Python, Rust, x86 ASM, ARM ASM, WASM, ...}
                                    ↓
                                  All targets are equal
```

**Scope**:
- ✅ Input: spec_v1.json (function signatures, types, modules)
- ✅ Pipeline: spec → IR → **direct target emission** (any language/assembly)
- ✅ Output Options:
  - **High-level code**: C++, Python, Rust (for human readability/modification)
  - **Assembly/Machine code**: x86-64, ARM, MIPS (for direct execution)
  - **Intermediate formats**: WASM, LLVM IR, bytecode
- ✅ Verification: Hash verification and receipt generation
- ✅ Cross-platform: Multiple architectures and operating systems

**Key Principle - Direct Assembly Generation**:
- ❌ **NOT**: spec → IR → C++ → compile → assembly → binary
- ✅ **YES**: spec → IR → **assembly directly** → assemble → binary
- ✅ **ALSO**: spec → IR → C++/Python/Rust (for readable code)

Assembly is **not a compilation artifact** - it's a **first-class IR target** that can be emitted directly, just like any other language.

**Out of Scope (Post-Alpha)**:
- ❌ Source code extraction/parsing (use pre-made specs)
- ❌ Exotic targets (FPGA, GPU shaders - though emitters exist)
- ❌ Advanced optimizations (register allocation, dead code elimination)
- ❌ IDE integrations
- ❌ Cross-compilation tooling (x86 → ARM cross-assembler)

---

## Phase Breakdown

### Phase 1: Spec → IR Conversion ✅ (Should be working)

**Tools Involved**:
- `stunir_spec_to_ir_main` - Main converter
- `ir_converter` - Core orchestrator
- `module_to_ir` - Convert module structure
- `func_to_ir` - Convert function signatures
- `type_resolver` - Resolve type dependencies
- `ir_validate` - Validate IR structure

**Input**: `spec_v1.json`
```json
{
  "manifest_version": "1.0",
  "module": {
    "name": "example",
    "functions": [
      {
        "name": "add",
        "return_type": "int",
        "parameters": [
          {"name": "a", "type": "int"},
          {"name": "b", "type": "int"}
        ]
      }
    ]
  }
}
```

**Output**: `semantic_ir.json`
```json
{
  "version": "1.0",
  "module": {
    "name": "example",
    "declarations": [...],
    "types": [...]
  }
}
```

**Validation Steps**:
1. Load spec_v1.json
2. Run `stunir_spec_to_ir_main spec_v1.json --output semantic_ir.json`
3. Verify IR structure with `ir_validate semantic_ir.json`
4. Check all functions/types present
5. Verify type resolution complete

---

### Phase 2: IR → C++ Code Generation + Binary

**Tools Involved**:
- `stunir_ir_to_code_main` - Main code generator
- `code_emitter` - Core orchestrator
- `sig_gen_cpp` - Generate C++ signatures
- `cpp_header_gen` - Generate .h files
- `cpp_impl_gen` - Generate .cpp files
- `code_gen_preamble` - Add headers/includes
- `code_write` - Write output files
- **stunir-emitters-assembly** - Assembly generation (optional)
- **External**: g++/clang for compilation

**Target Output Structure**:
```
output/cpp/
├── include/
│   └── example.h       // Function declarations
├── src/
│   ├── example.cpp     // Function implementations
│   └── main.cpp        // Test entry point
├── CMakeLists.txt      // Build configuration
├── bin/
│   ├── example_linux   // ELF binary (Linux)
│   ├── example.exe     // PE binary (Windows)
│   └── example_macos   // Mach-O binary (macOS)
└── .stunir_receipt.json
```

**Generated C++ Code**:
```cpp
// example.h
#ifndef EXAMPLE_H
#define EXAMPLE_H

int add(int a, int b);

#endif

// example.cpp
#include "example.h"

int add(int a, int b) {
    return a + b;  // Actual implementation
}

// main.cpp (generated for testing)
#include <iostream>
#include "example.h"

int main() {
    std::cout << "Testing add(2, 3): " << add(2, 3) << std::endl;
    return 0;
}
```

**Compilation to Binary**:
```bash
# Linux (ELF)
g++ -o output/cpp/bin/example_linux output/cpp/src/example.cpp output/cpp/src/main.cpp -I output/cpp/include -O2

# Windows (PE) - on Windows or cross-compile
g++ -o output/cpp/bin/example.exe output/cpp/src/example.cpp output/cpp/src/main.cpp -I output/cpp/include -O2

# macOS (Mach-O)
clang++ -o output/cpp/bin/example_macos output/cpp/src/example.cpp output/cpp/src/main.cpp -I output/cpp/include -O2
```

**Validation Steps**:
1. Run `stunir_ir_to_code_main semantic_ir.json --lang cpp --output output/cpp`
2. Verify file structure created
3. Compile: `g++ -o output/cpp/bin/example output/cpp/src/*.cpp -I output/cpp/include`
4. Check binary exists: `file output/cpp/bin/example` (should show ELF/PE/Mach-O)
5. Execute: `./output/cpp/bin/example` (should run without errors)
6. Verify function signatures with `nm output/cpp/bin/example | grep add`

---

### Phase 2b: IR → Assembly Code Generation (Direct Target)

**Tools Involved**:
- `stunir_ir_to_code_main` - Main code generator
- `code_emitter` - Core orchestrator
- **stunir-emitters-assembly** - Direct assembly generation from IR
- **stunir-emitters-systems** - Systems programming patterns
- **External**: NASM/YASM (x86-64), GNU as (ARM), or platform assembler

**Target Output Structure**:
```
output/assembly/
├── x86_64/
│   ├── example.asm     // x86-64 assembly (NASM syntax)
│   ├── example.s       // x86-64 assembly (AT&T syntax)
│   └── example.o       // Assembled object file
├── arm64/
│   ├── example.s       // ARM64 assembly
│   └── example.o       // Assembled object file
├── bin/
│   ├── example_x86_64  // x86-64 executable
│   └── example_arm64   // ARM64 executable
└── .stunir_receipt.json
```

**Generated x86-64 Assembly** (Direct from IR):
```nasm
; Generated directly from STUNIR IR
; Target: x86-64 Linux (System V ABI)

section .text
global add

add:
    ; Function: add(int a, int b) -> int
    ; Arguments: RDI = a, RSI = b
    ; Return: RAX

    push rbp
    mov rbp, rsp

    mov rax, rdi        ; Load a into rax
    add rax, rsi        ; Add b to rax

    pop rbp
    ret

global main
main:
    ; Test harness
    push rbp
    mov rbp, rsp

    mov rdi, 2          ; First argument: 2
    mov rsi, 3          ; Second argument: 3
    call add

    ; rax now contains 5
    mov rdi, rax        ; Exit code = result
    mov rax, 60         ; sys_exit
    syscall
```

**Assembly and Linking**:
```bash
# x86-64 (Linux)
nasm -f elf64 output/assembly/x86_64/example.asm -o output/assembly/x86_64/example.o
ld output/assembly/x86_64/example.o -o output/assembly/bin/example_x86_64

# Alternative: AT&T syntax with GNU as
as output/assembly/x86_64/example.s -o output/assembly/x86_64/example.o
ld output/assembly/x86_64/example.o -o output/assembly/bin/example_x86_64

# ARM64 (Linux)
as output/assembly/arm64/example.s -o output/assembly/arm64/example.o
ld output/assembly/arm64/example.o -o output/assembly/bin/example_arm64
```

**Validation Steps**:
1. Run `stunir_ir_to_code_main semantic_ir.json --lang assembly --arch x86_64 --output output/assembly/x86_64`
2. Verify assembly syntax: `nasm -f elf64 output/assembly/x86_64/example.asm -o /dev/null` (syntax check)
3. Assemble: `nasm -f elf64 output/assembly/x86_64/example.asm -o output/assembly/x86_64/example.o`
4. Link: `ld output/assembly/x86_64/example.o -o output/assembly/bin/example_x86_64`
5. Check binary format: `file output/assembly/bin/example_x86_64` (should show ELF 64-bit)
6. Execute: `./output/assembly/bin/example_x86_64; echo $?` (should exit with code 5)
7. Disassemble to verify: `objdump -d output/assembly/bin/example_x86_64`
8. Verify symbols: `nm output/assembly/bin/example_x86_64 | grep add`

**Why This Matters - Confluence in Action**:
- ✅ **No intermediate C/C++** - IR emits assembly directly
- ✅ **Platform-specific optimization** at generation time
- ✅ **Minimal dependencies** - only assembler needed, no compiler
- ✅ **Maximum control** - direct hardware-level output
- ✅ **Proves confluence** - same IR, different target, deterministic output
- ✅ **Same pipeline** - assembly is generated exactly like Python or Rust

---

### Phase 3: IR → Python Code Generation + Executable

**Tools Involved**:
- `stunir_ir_to_code_main` - Main code generator
- `code_emitter` - Core orchestrator
- `sig_gen_python` - Generate Python signatures
- `code_gen_func_sig` - Function signature generator
- `code_gen_func_body` - Function body generator
- `code_write` - Write output files
- **External**: PyInstaller or Nuitka for binary generation

**Target Output Structure**:
```
output/python/
├── example/
│   ├── __init__.py
│   └── example.py      // Module with functions
├── __main__.py         // Entry point for testing
├── setup.py            // Package setup
├── requirements.txt    // Dependencies
└── dist/
    ├── example_linux   // Standalone binary (Linux)
    ├── example.exe     // Standalone binary (Windows)
    └── example_macos   // Standalone binary (macOS)
```

**Generated Python Code**:
```python
# example.py
"""Generated from STUNIR spec_v1"""

def add(a: int, b: int) -> int:
    """
    Add two integers.

    Args:
        a: First integer
        b: Second integer

    Returns:
        Sum of a and b
    """
    return a + b

# __main__.py (generated for testing)
from example.example import add

if __name__ == "__main__":
    print(f"Testing add(2, 3): {add(2, 3)}")
```

**Compilation to Binary**:
```bash
# Option 1: PyInstaller (simpler)
cd output/python
pyinstaller --onefile --name example_linux __main__.py

# Option 2: Nuitka (faster, native)
cd output/python
python -m nuitka --standalone --onefile --output-filename=example_linux __main__.py
```

**Validation Steps**:
1. Run `stunir_ir_to_code_main semantic_ir.json --lang python --output output/python`
2. Verify file structure created
3. Test syntax: `python -m py_compile output/python/example/example.py`
4. Install PyInstaller: `pip install pyinstaller`
5. Build binary: `cd output/python && pyinstaller --onefile __main__.py`
6. Check binary exists: `file output/python/dist/example` (should show ELF/PE)
7. Execute: `./output/python/dist/example` (should print result)

---

### Phase 4: IR → Rust Code Generation + Binary

**Tools Involved**:
- `stunir_ir_to_code_main` - Main code generator
- `code_emitter` - Core orchestrator
- `sig_gen_rust` - Generate Rust signatures
- `code_gen_func_sig` - Function signature generator
- `code_gen_func_body` - Function body generator
- `code_write` - Write output files
- **External**: cargo build (native Rust compiler)

**Target Output Structure**:
```
output/rust/
├── src/
│   ├── lib.rs          // Library module
│   └── main.rs         // Binary entry point
├── Cargo.toml          // Package manifest
├── target/
│   └── release/
│       ├── example     // Native binary (Linux/macOS)
│       └── example.exe // Native binary (Windows)
└── README.md
```

**Generated Rust Code**:
```rust
// lib.rs
//! Generated from STUNIR spec_v1

/// Add two integers
///
/// # Arguments
/// * `a` - First integer
/// * `b` - Second integer
///
/// # Returns
/// Sum of a and b
pub fn add(a: i32, b: i32) -> i32 {
    a + b  // Actual implementation
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        assert_eq!(add(2, 3), 5);
    }
}

// main.rs (generated for testing)
use example::add;

fn main() {
    println!("Testing add(2, 3): {}", add(2, 3));
}
```

**Compilation to Binary**:
```bash
cd output/rust

# Debug build
cargo build

# Release build (optimized)
cargo build --release

# Binary will be at: target/release/example (or example.exe on Windows)
```

**Validation Steps**:
1. Run `stunir_ir_to_code_main semantic_ir.json --lang rust --output output/rust`
2. Verify file structure created
3. Check syntax: `cd output/rust && cargo check`
4. Run tests: `cargo test` (library tests)
5. Build binary: `cargo build --release`
6. Check binary exists: `file target/release/example` (should show ELF/PE/Mach-O)
7. Execute: `./target/release/example` (should print result)
8. Verify symbols: `nm target/release/example | grep add`

---

### Phase 5: Verification & Receipt Generation

**Tools Involved**:
- `hash_compute` - Compute file hashes
- `receipt_generate` - Generate verification receipts
- `toolchain_verify` - Verify toolchain.lock
- `stunir_receipt_link_main` - Link receipts

**Output**:
```
output/
├── cpp/
│   └── .stunir_receipt.json
├── python/
│   └── .stunir_receipt.json
├── rust/
│   └── .stunir_receipt.json
└── toolchain.lock
```

**Receipt Format**:
```json
{
  "timestamp": "2025-02-18T12:00:00Z",
  "input_spec": "spec_v1.json",
  "input_hash": "sha256:abc123...",
  "output_files": [
    {
      "path": "src/lib.rs",
      "hash": "sha256:def456..."
    }
  ],
  "toolchain_version": "0.1.0-alpha",
  "tools_used": [
    "stunir_spec_to_ir_main",
    "stunir_ir_to_code_main"
  ]
}
```

**Validation Steps**:
1. Generate receipts for all output directories
2. Link receipts into toolchain.lock
3. Verify hash integrity
4. Check tool version consistency

---

## Implementation Checklist

### Prerequisites
- [ ] **Build all missing powertools** (19 tools not in build)
  - func_to_ir, module_to_ir
  - code_gen_func_sig, code_gen_func_body, code_gen_preamble, code_write
  - sig_gen_cpp, sig_gen_python, sig_gen_rust (verify in build)
  
- [ ] **Remove duplicate tools** (7 tools)
  - json_validator, json_read, json_write
  - file_hash, type_map_cpp, type_resolve, spec_validate

- [ ] **Verify core orchestrators compile**
  - stunir_spec_to_ir_main
  - stunir_ir_to_code_main
  - pipeline_driver_main (optional)

### Step-by-Step Execution Plan

#### Step 1: Environment Setup (1 session)
```powershell
# Navigate to spark directory
cd tools/spark

# Verify GNAT/SPARK installation
gprbuild --version
gnatprove --version

# Check current build status
gprbuild -P powertools.gpr -p
```

#### Step 2: Update Build System (1 session)
```powershell
# Edit powertools.gpr to add missing 19 tools
# Remove 7 duplicate .adb files
# Rebuild
gprbuild -P powertools.gpr -p -j0

# Verify all 66 tools build successfully
ls bin/ | Measure-Object
```

#### Step 3: Create Test Spec (1 session)
```powershell
# Create test_data/simple_spec_v1.json
# Contains: 2-3 functions with basic types (int, string, bool)
# Example: add(int, int) -> int, concat(string, string) -> string
```

#### Step 4: Test Spec → IR (1 session)
```powershell
# Run spec-to-IR converter
./bin/stunir_spec_to_ir_main test_data/simple_spec_v1.json --output test_output/semantic_ir.json

# Validate IR
./bin/ir_validate test_output/semantic_ir.json

# Check IR structure
Get-Content test_output/semantic_ir.json | ConvertFrom-Json | Format-List
```

#### Step 5: Test IR → C++ (1 session)
```powershell
# Generate C++ code
./bin/stunir_ir_to_code_main test_output/semantic_ir.json --lang cpp --output test_output/cpp
```

#### Step 9: Integration Test Script (1 session)
```bash
#!/bin/bash
# test_pipeline_full.sh - Full pipeline test with assembly + high-level targets

set -e

echo "=== STUNIR Alpha Pipeline Test (Spec → All Targets) ==="

# Clean
rm -rf test_output
mkdir -p test_output

# Phase 1: Spec → IR
echo "Phase 1: Converting spec to IR..."
./bin/stunir_spec_to_ir_main test_data/simple_spec_v1.json --output test_output/semantic_ir.json
./bin/ir_validate test_output/semantic_ir.json

# Phase 2a: IR → Assembly (x86-64) - DIRECT TARGET
echo "Phase 2a: Generating x86-64 assembly (direct from IR)..."
./bin/stunir_ir_to_code_main test_output/semantic_ir.json --lang assembly --arch x86_64 --output test_output/assembly/x86_64
(cd test_output/assembly/x86_64 && nasm -f elf64 example.asm -o example.o && ld example.o -o ../bin/example_x86_64)
echo "Testing assembly binary..."
test_output/assembly/bin/example_x86_64
echo "Exit code: $?"

# Phase 2b: IR → C++ → Binary
echo "Phase 2b: Generating C++ code and compiling..."
./bin/stunir_ir_to_code_main test_output/semantic_ir.json --lang cpp --output test_output/cpp
(cd test_output/cpp && mkdir -p bin && g++ -o bin/example src/*.cpp -I include -O2)
echo "Testing C++ binary..."
test_output/cpp/bin/example

# Phase 3: IR → Python → Executable
echo "Phase 3: Generating Python code and building executable..."
./bin/stunir_ir_to_code_main test_output/semantic_ir.json --lang python --output test_output/python
(cd test_output/python && pyinstaller --onefile --name example __main__.py)
echo "Testing Python executable..."
test_output/python/dist/example

# Phase 4: IR → Rust → Binary
echo "Phase 4: Generating Rust code and compiling..."
./bin/stunir_ir_to_code_main test_output/semantic_ir.json --lang rust --output test_output/rust
(cd test_output/rust && cargo build --release)
echo "Testing Rust binary..."
test_output/rust/target/release/example

# Phase 5: Verification
echo "Phase 5: Generating receipts..."
./bin/receipt_generate test_output/assembly --input test_data/simple_spec_v1.json --include-binaries
./bin/receipt_generate test_output/cpp --input test_data/simple_spec_v1.json --include-binaries
./bin/receipt_generate test_output/python --input test_data/simple_spec_v1.json --include-binaries
./bin/receipt_generate test_output/rust --input test_data/simple_spec_v1.json --include-binaries
./bin/stunir_receipt_link_main test_output/**/.stunir_receipt.json --output test_output/toolchain.lock
./bin/toolchain_verify test_output/toolchain.lock

echo ""
echo "=== Binary Information ==="
echo "Assembly: $(file test_output/assembly/bin/example_x86_64)"
echo "C++:      $(file test_output/cpp/bin/example)"
echo "Python:   $(file test_output/python/dist/example)"
echo "Rust:     $(file test_output/rust/target/release/example)"
echo ""
echo "=== Binary Sizes ==="
ls -lh test_output/assembly/bin/example_x86_64 test_output/cpp/bin/example test_output/python/dist/example test_output/rust/target/release/example

echo ""
echo "✅ All phases completed successfully!"
echo "✅ All targets generated from same IR!"
echo "✅ Assembly generated directly (no C/C++ intermediate)!"
echo "✅ Confluence principle demonstrated!"
```

#### Step 10: Documentation (1 session)
```markdown
# Create docs/QUICKSTART.md with:
1. Installation instructions (compilers, assemblers, tools)
2. Simple example workflow (spec → IR → assembly/code)
3. Command reference for all target types
4. Expected outputs (assembly, source, binaries)
5. Platform-specific notes
6. Understanding confluence (why assembly is direct)
7. Troubleshooting
```

#### Step 5: Test IR → C++ (1 session)
```powershell
# Generate C++ code
./bin/stunir_ir_to_code_main test_output/semantic_ir.json --lang cpp --output test_output/cpp

# Verify structure
tree test_output/cpp

# Compile test
cd test_output/cpp
g++ -c src/*.cpp -I include
# Should compile with no errors
```

#### Step 6: Test IR → Python (1 session)
```powershell
# Generate Python code
./bin/stunir_ir_to_code_main test_output/semantic_ir.json --lang python --output test_output/python

# Verify syntax
python -m py_compile test_output/python/**/*.py

# Import test
python -c "import sys; sys.path.insert(0, 'test_output/python'); import example"
```

#### Step 7: Test IR → Rust (1 session)
```powershell
# Generate Rust code
./bin/stunir_ir_to_code_main test_output/semantic_ir.json --lang rust --output test_output/rust

# Verify compilation
cd test_output/rust
cargo check
cargo test
```

#### Step 8: Receipt Generation (1 session)
```powershell
# Generate receipts for all outputs
./bin/receipt_generate test_output/cpp --input test_data/simple_spec_v1.json
./bin/receipt_generate test_output/python --input test_data/simple_spec_v1.json
./bin/receipt_generate test_output/rust --input test_data/simple_spec_v1.json

# Link receipts
./bin/stunir_receipt_link_main test_output/**/.stunir_receipt.json --output test_output/toolchain.lock

# Verify
./bin/toolchain_verify test_output/toolchain.lock
```

#### Step 9: Integration Test Script (1 session)
```bash
#!/bin/bash
# test_pipeline.sh - Full pipeline test

set -e

echo "=== STUNIR Alpha Pipeline Test ==="

# Clean
rm -rf test_output
mkdir -p test_output

# Phase 1: Spec → IR
echo "Phase 1: Converting spec to IR..."
./bin/stunir_spec_to_ir_main test_data/simple_spec_v1.json --output test_output/semantic_ir.json
./bin/ir_validate test_output/semantic_ir.json

# Phase 2: IR → C++
echo "Phase 2: Generating C++ code..."
./bin/stunir_ir_to_code_main test_output/semantic_ir.json --lang cpp --output test_output/cpp
(cd test_output/cpp && g++ -c src/*.cpp -I include)

# Phase 3: IR → Python
echo "Phase 3: Generating Python code..."
./bin/stunir_ir_to_code_main test_output/semantic_ir.json --lang python --output test_output/python
python -m py_compile test_output/python/**/*.py

# Phase 4: IR → Rust
echo "Phase 4: Generating Rust code..."
./bin/stunir_ir_to_code_main test_output/semantic_ir.json --lang rust --output test_output/rust
(cd test_output/rust && cargo check)

# Phase 5: Verification
echo "Phase 5: Generating receipts..."
./bin/receipt_generate test_output/cpp --input test_data/simple_spec_v1.json
./bin/receipt_generate test_output/python --input test_data/simple_spec_v1.json
./bin/receipt_generate test_output/rust --input test_data/simple_spec_v1.json
./bin/stunir_receipt_link_main test_output/**/.stunir_receipt.json --output test_output/toolchain.lock
./bin/toolchain_verify test_output/toolchain.lock

echo "✅ All phases completed successfully!"
```

#### Step 10: Documentation (1 session)
```markdown
# Create docs/QUICKSTART.md with:
1. Installation instructions
2. Simple example workflow
3. Command reference
4. Expected outputs
5. Troubleshooting
```

---

## Success Criteria

### Must Have (Alpha Release Blocker)
- ✅ All 66 powertools compile successfully
- ✅ Spec → IR conversion works for simple spec
- ✅ **IR → Assembly generates valid x86-64/ARM assembly directly (demonstrates confluence)**
- ✅ **Assembly code assembles to working binary**
- ✅ **Assembly binary executes successfully**
- ✅ IR → C++ generates compilable code
- ✅ **C++ code compiles to working binary (ELF/PE/Mach-O)**
- ✅ **C++ binary executes successfully**
- ✅ IR → Python generates valid Python
- ✅ **Python code builds to standalone executable**
- ✅ **Python executable runs successfully**
- ✅ IR → Rust generates compilable code
- ✅ **Rust code compiles to optimized binary**
- ✅ **Rust binary executes successfully**
- ✅ **All binaries verified with `file` command (ELF/PE/Mach-O)**
- ✅ **All binaries tested for correct output**
- ✅ **Assembly proves IR-to-target confluence (no C/C++ intermediate)**
- ✅ Receipt generation includes binary hashes for all targets
- ✅ Basic documentation (QUICKSTART.md with assembly + high-level examples)
- ✅ Integration test script passes (spec → IR → assembly/C++/Python/Rust → binaries)

### Should Have (Improve Alpha)
- ⚠️ Support for structs/custom types in all targets (assembly, C++, Python, Rust)
- ⚠️ Support for multiple modules with linking
- ⚠️ Error handling and reporting in generated code
- ⚠️ Build system generation (CMakeLists.txt, setup.py, Cargo.toml, NASM project files)
- ⚠️ Multiple assembly architectures (x86-64, ARM64, RISC-V)
- ⚠️ Assembly syntax options (NASM, AT&T, Intel)
- ⚠️ Binary stripping and optimization flags
- ⚠️ Cross-platform build scripts (works on Linux, Windows, macOS)
- ⚠️ Symbol verification (nm/objdump checks)
- ⚠️ Assembly ABI compliance verification (System V, Windows x64 calling conventions)

### Nice to Have (Post-Alpha)
- ⭕ Cross-compilation support (x86 → ARM, Linux → Windows)
- ⭕ Assembly output analysis and inspection tools
- ⭕ Binary size optimization and comparison across targets
- ⭕ Debug symbol generation (DWARF, PDB) for all targets
- ⭕ Static/dynamic linking options
- ⭕ Multiple file outputs with proper linking
- ⭕ Binary benchmarking and performance metrics
- ⭕ Assembly optimization passes (register allocation, peephole optimization)
- ⭕ WASM target generation (another direct IR target)
- ⭕ Additional assembly architectures (MIPS, PowerPC, etc.)

---

## Risk Assessment

### High Risk
1. **IR Converter Not Complete** - module_to_ir, func_to_ir not in build
   - Mitigation: Add to build immediately, test with simple specs
   
2. **Code Generator Integration** - Unclear how emitters work together
   - Mitigation: Start with simplest case (single function), expand gradually

3. **Type Mapping Incomplete** - Complex types may not translate
   - Mitigation: Start with primitives only (int, string, bool, float)

### Medium Risk
4. **Build System Generation** - May need manual CMakeLists.txt/Cargo.toml
   - Mitigation: Create templates, expand later

5. **Error Handling** - Tools may fail silently
   - Mitigation: Add logging, check exit codes

### Low Risk
6. **Documentation** - Time consuming but not technical
   - Mitigation: Keep minimal for alpha, expand later

---

## Timeline Estimate

**Total**: ~10 sessions (assuming 1-2 hour sessions)

1. Environment setup & verification: 1 session
2. Build system updates: 1-2 sessions
3. Create test specs: 1 session
4. Spec → IR testing: 1 session
5. IR → C++ testing: 1-2 sessions
6. IR → Python testing: 1 session
7. IR → Rust testing: 1 session
8. Receipt generation: 1 session
9. Integration test: 1 session
10. Documentation: 1 session

**Optimistic**: 8 sessions  
**Realistic**: 10-12 sessions  
**Pessimistic**: 15 sessions (if major issues found)

---

## Example Test Spec

**File**: `test_data/simple_spec_v1.json`

```json
{
  "manifest_version": "1.0",
  "metadata": {
    "name": "simple_example",
    "version": "0.1.0",
    "description": "Simple test for alpha pipeline"
  },
  "module": {
    "name": "simple_example",
    "functions": [
      {
        "name": "add",
        "signature": "add(int, int) -> int",
        "return_type": "int",
        "parameters": [
          {
            "name": "a",
            "type": "int",
            "description": "First number"
          },
          {
            "name": "b",
            "type": "int",
            "description": "Second number"
          }
        ],
        "description": "Add two integers"
      },
      {
        "name": "concat",
        "signature": "concat(string, string) -> string",
        "return_type": "string",
        "parameters": [
          {
            "name": "s1",
            "type": "string",
            "description": "First string"
          },
          {
            "name": "s2",
            "type": "string",
            "description": "Second string"
          }
        ],
        "description": "Concatenate two strings"
      },
      {
        "name": "is_positive",
        "signature": "is_positive(int) -> bool",
        "return_type": "bool",
        "parameters": [
          {
            "name": "n",
            "type": "int",
            "description": "Number to check"
          }
        ],
        "description": "Check if number is positive"
      }
    ],
    "types": [
      {
        "name": "int",
        "kind": "primitive",
        "size": 32
      },
      {
        "name": "string",
        "kind": "primitive"
      },
      {
        "name": "bool",
        "kind": "primitive"
      }
    ]
  }
}
```

---

## Commands Quick Reference

```powershell
# Build everything
gprbuild -P powertools.gpr -p -j0

# Run full pipeline
./bin/stunir_spec_to_ir_main input.json --output ir.json
./bin/ir_validate ir.json
./bin/stunir_ir_to_code_main ir.json --lang cpp --output out/cpp
./bin/stunir_ir_to_code_main ir.json --lang python --output out/python
./bin/stunir_ir_to_code_main ir.json --lang rust --output out/rust

# Generate receipts
./bin/receipt_generate out/cpp --input input.json
./bin/receipt_generate out/python --input input.json
./bin/receipt_generate out/rust --input input.json

# Link and verify
./bin/stunir_receipt_link_main out/**/.stunir_receipt.json --output toolchain.lock
./bin/toolchain_verify toolchain.lock
```

---

## Next Steps

1. **Review this plan** - Ensure scope and approach are correct
2. **Start new session** - Begin with Step 1 (Environment Setup)
3. **Execute sequentially** - Follow steps 1-10 in order
4. **Document issues** - Track problems and solutions
5. **Update plan** - Refine as you discover what works

---

**Status**: ✅ **PLAN READY FOR EXECUTION**  
**Recommendation**: Start with environment verification and build updates  
**Expected Outcome**: Working spec-to-code pipeline for C++, Python, Rust
