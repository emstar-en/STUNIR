# STUNIR Embedded Target

Bare-metal embedded C emitter for STUNIR.

## Overview

This emitter converts STUNIR IR to C89-compliant code optimized for
resource-constrained embedded systems without dynamic memory allocation.

## Usage

```bash
python emitter.py <ir.json> --output=<output_dir> [--arch=arm|avr|mips|riscv]
```

## Output Files

- `<module>.h` - Header with function prototypes
- `<module>.c` - C implementation
- `startup.c` - Reset handler and startup code
- `<module>.ld` - Linker script
- `Makefile` - Build system
- `config.h` - Architecture configuration
- `manifest.json` - Deterministic file manifest
- `README.md` - Generated documentation

## Features

- No malloc/free (static allocation only)
- C89 compliant with fixed-width types
- Configurable stack/heap sizes
- Architecture-specific optimizations
- Startup code and linker scripts

## Supported Architectures

- ARM Cortex-M (arm-none-eabi)
- AVR (avr-gcc)
- MIPS (mips-elf)
- RISC-V (riscv32-unknown-elf)

## Schema

`stunir.embedded.{arch}.v1`
