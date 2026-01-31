# Polyglot Code Examples

## C89 (ANSI C)

**Standard:** ISO/IEC 9899:1990  
**Compatibility:** Universal (all C compilers)

### Compilation

```bash
gcc -std=c89 -pedantic -Wall -Wextra output.c89 -o program_c89
```

**Features:**
- Maximum portability
- Compatible with ancient compilers
- Fixed-width types via typedef

## C99

**Standard:** ISO/IEC 9899:1999  
**Compatibility:** Modern C compilers

### Compilation

```bash
gcc -std=c99 -Wall -Wextra output.c99 -o program_c99
```

**Features:**
- `<stdint.h>` for fixed-width types
- `<stdbool.h>` for boolean type
- Inline functions
- VLAs (optional)

## Rust

**Edition:** 2021  
**Safety:** Memory-safe by design

### Compilation

```bash
rustc --edition 2021 -O output.rs -o program_rust
```

**Features:**
- Zero-cost abstractions
- Memory safety without GC
- Thread safety
- Ownership system

### Comparison

| Feature | C89 | C99 | Rust |
|---------|-----|-----|------|
| Fixed-width types | Manual | ✅ | ✅ |
| Boolean type | Manual | ✅ | ✅ |
| Memory safety | Manual | Manual | ✅ |
| Thread safety | Manual | Manual | ✅ |
| Portability | ✅✅✅ | ✅✅ | ✅ |
| Performance | ✅✅ | ✅✅ | ✅✅✅ |
