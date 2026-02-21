# STUNIR ARM Assembly Target

This target emits ARM (32-bit) and ARM64/AArch64 (64-bit) assembly from STUNIR IR.

## Purpose

The ARM target provides:
- Native assembly code generation for ARM processors
- Support for both 32-bit ARM and 64-bit ARM64
- AAPCS calling convention compliance
- GNU assembler (as) compatible output

## Usage

```bash
# 32-bit ARM
python3 emitter.py <input.ir.json> --output=<output_dir>

# 64-bit ARM64/AArch64
python3 emitter.py <input.ir.json> --output=<output_dir> --64bit
```

## Output Structure

```
output_dir/
├── module.s            # Assembly source
├── build.sh            # Build script
├── manifest.json       # STUNIR manifest
└── README.md           # Documentation
```

## Build Requirements

- GNU ARM toolchain (`arm-linux-gnueabi-*` or `aarch64-linux-gnu-*`)
- Or native compilation on ARM/ARM64 system

## Architecture Details

| Mode | Word Size | Registers | Calling Convention |
|------|-----------|-----------|-------------------|
| ARM | 32-bit | r0-r12, sp, lr, pc | AAPCS |
| ARM64 | 64-bit | x0-x30, sp | AAPCS64 |

## Files

- `emitter.py` - ARM emitter implementation
- `../base.py` - Shared assembly emitter base

## Schema

`stunir.target.arm.v1` / `stunir.target.arm64.v1`
