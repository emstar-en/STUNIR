# STUNIR x86 Assembly Target

This target emits x86 (32-bit) and x86-64 (64-bit) assembly from STUNIR IR.

## Purpose

The x86 target provides:
- Native assembly code generation
- Support for both 32-bit and 64-bit modes
- Intel syntax assembly
- NASM-compatible output

## Usage

```bash
# 32-bit x86
python3 emitter.py <input.ir.json> --output=<output_dir>

# 64-bit x86-64
python3 emitter.py <input.ir.json> --output=<output_dir> --64bit
```

## Output Structure

```
output_dir/
├── module.asm          # Assembly source
├── build.sh            # Build script
├── manifest.json       # STUNIR manifest
└── README.md           # Documentation
```

## Build Requirements

- NASM (Netwide Assembler)
- GNU ld linker

## Architecture Details

| Mode | Word Size | Registers | Calling Convention |
|------|-----------|-----------|-------------------|
| x86 | 32-bit | eax, ebx, ecx, edx, ... | cdecl |
| x86-64 | 64-bit | rax, rbx, rcx, rdx, ... | System V AMD64 |

## Files

- `emitter.py` - x86 emitter implementation
- `../base.py` - Shared assembly emitter base

## Schema

`stunir.target.x86.v1` / `stunir.target.x86_64.v1`
