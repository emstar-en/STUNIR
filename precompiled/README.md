# STUNIR Precompiled Binaries

This directory contains precompiled binaries for STUNIR tools, eliminating the need to install compilers for typical usage.

## Directory Structure

```
precompiled/
├── README.md                  # This file
└── linux-x86_64/
    └── spark/
        └── bin/
            ├── ir_converter_main         # Spec → IR converter (PRIMARY)
            ├── code_emitter_main         # IR → Code emitter (PRIMARY)
            ├── pipeline_driver_main      # Full pipeline orchestrator
            └── embedded_emitter_main     # Embedded target emitter
```

> **Note:** The old tool names (`stunir_spec_to_ir_main`, `stunir_ir_to_code_main`) are deprecated.
> They may still exist for backward compatibility but should not be used for new work.
> See `docs/archive/spark_deprecated/README.md` for the deprecation schedule.

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
# Spec to IR (use ir_converter_main)
./precompiled/linux-x86_64/spark/bin/ir_converter_main \
    --spec-root spec/ \
    --out asm/spec_ir.json

# IR to Code (use code_emitter_main)
./precompiled/linux-x86_64/spark/bin/code_emitter_main \
    --input asm/spec_ir.json \
    --output asm/output.py \
    --target python

# Full pipeline (use pipeline_driver_main)
./precompiled/linux-x86_64/spark/bin/pipeline_driver_main \
    --config pipeline.json
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
| `ir_converter_main` | `tools/spark/src/core/` | Converts spec JSON to IR manifest (PRIMARY) |
| `code_emitter_main` | `tools/spark/src/core/` | Generates code from IR (PRIMARY) |
| `pipeline_driver_main` | `tools/spark/src/core/` | Orchestrates full pipeline execution |
| `embedded_emitter_main` | `targets/spark/` | Embedded C code emitter |

**Deprecated binaries (do not use for new work):**

| Binary | Replacement |
|--------|-------------|
| `stunir_spec_to_ir_main` | `ir_converter_main` |
| `stunir_ir_to_code_main` | `code_emitter_main` |

## Version Information

- Built with: GNAT 12.2.0
- Target: x86_64-linux-gnu
- Build date: 2026-01-30
- SPARK verification: Level 2

## License

MIT License - Copyright (c) 2026 STUNIR Project
