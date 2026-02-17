#!/bin/bash
# STUNIR Generated x86 Build Script
set -e

# Assemble
nasm -f elf32 -o module.o module.asm

# Link
ld -m elf_i386 -o module module.o

echo "Built: module (x86-32)"
