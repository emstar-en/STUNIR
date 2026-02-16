#!/bin/bash
# STUNIR ROCm Kernel Build Script
# Build all kernel patterns

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check for hipcc
if ! command -v hipcc &> /dev/null; then
    echo "Error: hipcc not found. Install ROCm first."
    echo "See: https://rocm.docs.amd.com/en/latest/"
    exit 1
fi

# Build options
OPTIMIZE="-O3"
DEBUG=""
VERBOSE=0

for arg in "$@"; do
    case $arg in
        --debug)
            OPTIMIZE="-O0"
            DEBUG="-g -DDEBUG"
            ;;
        --verbose)
            VERBOSE=1
            ;;
        --help)
            echo "Usage: $0 [--debug] [--verbose] [--help]"
            echo "  --debug    Build with debug symbols"
            echo "  --verbose  Show compilation commands"
            exit 0
            ;;
    esac
done

KERNELS=("conv2d" "fft" "sparse_matvec" "transpose")

echo "=== STUNIR ROCm Kernel Build ==="
echo "Compiler: $(hipcc --version | head -1)"
echo ""

for kernel in "${KERNELS[@]}"; do
    if [ -f "${kernel}.hip" ]; then
        echo "Building ${kernel}..."
        CMD="hipcc $OPTIMIZE $DEBUG -o $kernel ${kernel}.hip"
        if [ $VERBOSE -eq 1 ]; then
            echo "  $CMD"
        fi
        $CMD
        echo "  -> $kernel built successfully"
    else
        echo "Warning: ${kernel}.hip not found"
    fi
done

echo ""
echo "Build complete!"
echo "Run with: ./<kernel_name>"
