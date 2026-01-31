# Embedded Code Examples

## ARM Cortex-M Example

**Target:** ARM Cortex-M4  
**Compiler:** arm-none-eabi-gcc  
**Flash:** 256KB  
**RAM:** 64KB

### Files Generated

- `arm_cortex_m.c` - Application code
- `startup.c` - Startup code
- `linker.ld` - Linker script

### Compilation

```bash
arm-none-eabi-gcc -mcpu=cortex-m4 -mthumb \
  -O2 -Wall -Wextra \
  -T linker.ld \
  startup.c arm_cortex_m.c \
  -o firmware.elf

# Generate binary
arm-none-eabi-objcopy -O binary firmware.elf firmware.bin

# Flash to device
st-flash write firmware.bin 0x08000000
```

### Features

- Zero-cost abstractions
- Minimal runtime overhead
- Predictable memory usage
- DO-178C Level A compliant
