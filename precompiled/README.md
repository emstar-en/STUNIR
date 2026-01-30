# STUNIR Precompiled Binaries

This directory contains precompiled binaries for STUNIR tools, eliminating the need to install compilers for typical usage.

## Directory Structure

```
precompiled/
├── README.md                  # This file
└── linux-x86_64/
    └── spark/
        └── bin/
            ├── stunir_spec_to_ir_main    # Spec → IR converter
            ├── stunir_ir_to_code_main    # IR → Code emitter
            └── embedded_emitter_main     # Embedded target emitter
```

## Supported Platforms

| Platform | Architecture | Status |
|----------|--------------|--------|
| Linux    | x86_64       | ✅ Available |
| macOS    | x86_64       | ❌ Not yet available |
| macOS    | arm64        | ❌ Not yet available |
| Windows  | x86_64       | ❌ Not yet available |

## Runtime Requirements

The precompiled Ada SPARK binaries require the GNAT runtime library (`libgnat`):

```bash
# Debian/Ubuntu (Bookworm)
sudo apt-get install libgnat-12

# Debian/Ubuntu (older)
sudo apt-get install libgnat-11  # or libgnat-10

# Fedora/RHEL
sudo dnf install libgnat

# Note: Installing only libgnat is MUCH smaller than full GNAT compiler
# libgnat-12 is ~5MB vs gnat compiler ~100MB+
```

**If you get "libgnat not found" errors**, install the runtime library above. This is the only dependency needed - you do NOT need the full GNAT compiler to use precompiled binaries.

## Usage

### Using Precompiled Binaries (Recommended)

The build script automatically uses precompiled binaries when available:

```bash
# Automatic detection
./scripts/build.sh

# Force precompiled SPARK binaries
export STUNIR_USE_PRECOMPILED=1
./scripts/build.sh
```

### Manual Usage

```bash
# Spec to IR
./precompiled/linux-x86_64/spark/bin/stunir_spec_to_ir_main \
    --spec-root spec/ \
    --out asm/spec_ir.json

# IR to Code
./precompiled/linux-x86_64/spark/bin/stunir_ir_to_code_main \
    --input asm/spec_ir.json \
    --output asm/output.py \
    --target python
```

## Building from Source

If you need to rebuild from source (e.g., for a different platform):

```bash
# Install GNAT compiler
sudo apt-get install gnat gprbuild

# Build core tools
cd tools/spark
gprbuild -P stunir_tools.gpr

# Build emitters
cd targets/spark
gprbuild -P stunir_emitters.gpr
```

## Binary Manifest

| Binary | Source | Description |
|--------|--------|-------------|
| `stunir_spec_to_ir_main` | `tools/spark/` | Converts spec JSON to IR manifest |
| `stunir_ir_to_code_main` | `tools/spark/` | Generates code from IR |
| `embedded_emitter_main` | `targets/spark/` | Embedded C code emitter |

## Version Information

- Built with: GNAT 12.2.0
- Target: x86_64-linux-gnu
- Build date: 2026-01-30
- SPARK verification: Level 2

## License

MIT License - Copyright (c) 2026 STUNIR Project
