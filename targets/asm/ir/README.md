# STUNIR ASM/IR Target

This target emits STUNIR IR in an assembly-like format suitable for debugging,
verification, and low-level analysis.

## Purpose

The ASM/IR format provides:
- Human-readable representation of STUNIR IR
- Deterministic output for verification
- Canonical JSON counterpart for machine processing

## Usage

```bash
python3 emitter.py <input.ir.json> --output=<output_dir>
```

## Output Format

### ASM-IR Syntax

```
; STUNIR ASM-IR Format
.module example

.func main() -> i32
{
    local.i32 x = 0
    add x, x, 1
    return x
}
```

### Instructions

| Instruction | Description |
|-------------|-------------|
| `local.T var = val` | Declare local variable |
| `store dst, src` | Store value |
| `load dst, src` | Load value |
| `add dst, l, r` | Addition |
| `sub dst, l, r` | Subtraction |
| `mul dst, l, r` | Multiplication |
| `cmp.op l, r` | Comparison |
| `br label` | Unconditional branch |
| `br_if cond, label` | Conditional branch |
| `call func(args)` | Function call |
| `return val` | Return from function |

## Files

- `emitter.py` - ASM/IR emitter implementation
- `manifest.json` - Output manifest (after emission)

## Schema

`stunir.asm.ir.v1`
