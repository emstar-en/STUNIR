# STUNIR Generated Makefile
# Target: x86_64 minimal OS
# DO-178C Level A Compliance

# Toolchain configuration
AS = nasm
CC = gcc
LD = ld

# Directories
ASM_DIR = asm
C_DIR = c
BUILD_DIR = build
OBJ_DIR = $(BUILD_DIR)/obj

# Output files
KERNEL_ELF = $(BUILD_DIR)/kernel.elf
KERNEL_BIN = $(BUILD_DIR)/kernel.bin

# Compiler flags for bare-metal x86_64
CFLAGS = -ffreestanding -fno-stack-protector -fno-pic -mno-red-zone \
         -mno-mmx -mno-sse -mno-sse2 -nostdinc -nostdlib \
         -Wall -Wextra -O2 -mcmodel=kernel -I$(C_DIR)

# Linker flags  
LDFLAGS = -n -nostdlib -T $(BUILD_DIR)/linker.ld

# Source files
C_SOURCES = $(wildcard $(C_DIR)/*.c)

# Object files
OBJS = $(OBJ_DIR)/boot.o \
       $(OBJ_DIR)/interrupts_asm.o \
       $(OBJ_DIR)/serial.o \
       $(OBJ_DIR)/pmm.o \
       $(OBJ_DIR)/timer.o \
       $(OBJ_DIR)/interrupts.o \
       $(OBJ_DIR)/scheduler.o \
       $(OBJ_DIR)/syscalls.o \
       $(OBJ_DIR)/init.o

.PHONY: all clean run debug iso

all: $(OBJ_DIR) $(KERNEL_BIN)
	@echo "========================================"
	@echo "STUNIR OS built successfully!"
	@echo "Kernel: $(KERNEL_ELF)"
	@ls -la $(KERNEL_ELF)
	@echo "========================================"

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# Boot code - combined 32-bit entry with 64-bit transition
$(OBJ_DIR)/boot.o: $(ASM_DIR)/boot.asm
	$(AS) -f elf64 -w-zext-reloc $< -o $@

# Interrupt handlers (64-bit)
$(OBJ_DIR)/interrupts_asm.o: $(ASM_DIR)/interrupts.asm
	$(AS) -f elf64 $< -o $@

# C sources
$(OBJ_DIR)/%.o: $(C_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

# Link kernel
$(KERNEL_ELF): $(OBJS)
	$(LD) $(LDFLAGS) -o $@ $(OBJS)

# Create flat binary
$(KERNEL_BIN): $(KERNEL_ELF)
	objcopy -O binary $< $@

# Run in QEMU
run: $(KERNEL_ELF)
	@echo "Starting QEMU... (Press Ctrl-A X to exit)"
	qemu-system-x86_64 \
		-kernel $(KERNEL_ELF) \
		-serial stdio \
		-no-reboot \
		-m 128M \
		-display none \
		-d guest_errors 2>&1 | head -100

# Run with timeout for testing
test: $(KERNEL_ELF)
	@echo "Running QEMU test (5 second timeout)..."
	timeout 5s qemu-system-x86_64 \
		-kernel $(KERNEL_ELF) \
		-serial stdio \
		-no-reboot \
		-no-shutdown \
		-m 128M \
		-display none 2>&1 || true

# Run with debug output
debug: $(KERNEL_ELF)
	qemu-system-x86_64 \
		-kernel $(KERNEL_ELF) \
		-serial stdio \
		-no-reboot \
		-m 128M \
		-display none \
		-d int,cpu_reset \
		-D qemu_debug.log

clean:
	rm -rf $(OBJ_DIR) $(KERNEL_ELF) $(KERNEL_BIN) $(BUILD_DIR)/iso $(BUILD_DIR)/stunir_os.iso

# Show help
help:
	@echo "STUNIR OS Build System"
	@echo "====================="
	@echo "make          - Build the kernel"
	@echo "make run      - Build and run in QEMU"
	@echo "make test     - Build and test with timeout"
	@echo "make debug    - Build and run with debug output"
	@echo "make clean    - Remove build artifacts"
