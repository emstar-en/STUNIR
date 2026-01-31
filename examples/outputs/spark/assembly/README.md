# Assembly Code Examples

## x86_64 Example (Intel Syntax)

**Target:** x86_64 processors  
**Syntax:** Intel (NASM-compatible)

### Assembly

```bash
# Assemble
nasm -f elf64 x86_64.asm -o x86_64.o

# Link
ld -o program x86_64.o

# Run
./program
```

## ARM Example

**Target:** ARM Cortex-A processors  
**Syntax:** ARM UAL

### Assembly

```bash
# Assemble
arm-linux-gnueabihf-as arm.asm -o arm.o

# Link
arm-linux-gnueabihf-ld -o program arm.o

# Run on ARM device or emulator
qemu-arm ./program
```

### Features

- Maximum performance
- Minimal code size
- Direct hardware control
- Verified safety properties
