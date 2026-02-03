#!/bin/bash
# STUNIR WASM Build Script
# Compiles WAT to WASM binary

set -e

WAT_FILE="module.wat"
WASM_FILE="module.wasm"

# Check for wat2wasm (from WABT toolkit)
if command -v wat2wasm &> /dev/null; then
    echo "Compiling $WAT_FILE -> $WASM_FILE"
    wat2wasm "$WAT_FILE" -o "$WASM_FILE"
    echo "Generated: $WASM_FILE"
else
    echo "Error: wat2wasm not found. Install WABT toolkit."
    echo "  macOS: brew install wabt"
    echo "  Linux: apt install wabt"
    exit 1
fi
