# STUNIR Targets API

> Part of `docs/api/1068`

## Target Emitter Interface

All target emitters follow a common interface:

```python
class BaseEmitter:
    def emit(self, ir_data: dict, output_dir: str) -> dict:
        """Emit target-specific code from IR."""
        pass
    
    def generate_build_script(self, output_dir: str) -> str:
        """Generate build script for target."""
        pass
    
    def generate_manifest(self, output_dir: str) -> dict:
        """Generate manifest for emitted files."""
        pass
```

---

## ASM/IR Target (`targets/asm/ir/`)

### `emitter.py`

**Functions:**
- `emit_asm_ir(ir_data, output_dir)` - Emit ASM/IR format
- `generate_manifest(output_dir)` - Create target manifest

**Output:**
- `.dcbor` files in `asm/ir/`
- Canonical JSON manifest

---

## Polyglot Targets

### Rust (`targets/polyglot/rust/`)

**Emitter:** `emitter.py`

**Output:**
- `Cargo.toml` - Package manifest
- `src/lib.rs` - Library entry point
- Type-safe IR mapping

**Build:**
```bash
cargo build --release
```

### C89 (`targets/polyglot/c89/`)

**Emitter:** `emitter.py`

**Output:**
- `.c` source files
- `.h` header files
- `Makefile` with `-ansi` flag

### C99 (`targets/polyglot/c99/`)

**Emitter:** `emitter.py`

**Output:**
- `.c` source files
- `.h` header files
- `Makefile` with `-std=c99` flag

---

## Assembly Targets

### x86 (`targets/assembly/x86/`)

**Emitter:** `emitter.py`

**Output:**
- `.asm` files (Intel syntax)
- NASM build scripts
- 32/64-bit support

### ARM (`targets/assembly/arm/`)

**Emitter:** `emitter.py`

**Output:**
- `.s` files (ARM syntax)
- ARM toolchain build scripts
- AAPCS calling convention

---

## Specialized Targets

### WASM (`targets/wasm/`)
WebAssembly target for browser/WASI runtime.

### Bytecode (`targets/bytecode/`)
Virtual machine bytecode target.

### GPU (`targets/gpu/`)
GPU compute shader generation.

### FPGA (`targets/fpga/`)
FPGA synthesis target.

### Embedded (`targets/embedded/`)
Embedded systems with minimal runtime.

### Mobile (`targets/mobile/`)
Mobile platform optimizations.

---

## Target Selection

```bash
# Via build script
./scripts/build.sh --target=rust
./scripts/build.sh --target=c89
./scripts/build.sh --target=x86

# Via Python
python -m tools.emitters.emit_code --target rust --ir ir.json --output out/
```

---

## Related
- [API Overview](README.md)
- [Architecture Components](../architecture/components.md)

---
*STUNIR Targets API v1.0*
