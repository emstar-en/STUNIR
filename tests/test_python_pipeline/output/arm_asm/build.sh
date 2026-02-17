#!/bin/bash
# STUNIR Generated ARM32 Build Script
set -e

# Assemble (requires cross-compiler or ARM system)
arm-linux-gnueabi-as -o module.o module.s

# Link
arm-linux-gnueabi-ld -o module module.o

echo "Built: module (ARM32)"
