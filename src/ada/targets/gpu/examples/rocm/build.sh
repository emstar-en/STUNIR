#!/bin/bash
# STUNIR ROCm Examples Build Script
#
# Builds all ROCm/HIP examples for AMD or NVIDIA GPUs.
# Usage:
#   ./build.sh              # Build for AMD GPU (default)
#   ./build.sh --nvidia     # Build for NVIDIA GPU
#   ./build.sh --clean      # Clean build artifacts
#   ./build.sh --all        # Build all examples

set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
BUILD_DIR="$SCRIPT_DIR/build"

NVIDIA_MODE=0
CLEAN_MODE=0
BUILD_ALL=1
DEBUG_MODE=0

for arg in "$@"; do
    case $arg in
        --nvidia)
            NVIDIA_MODE=1
            ;;
        --clean)
            CLEAN_MODE=1
            ;;
        --debug)
            DEBUG_MODE=1
            ;;
        vector_add|matmul|reduction)
            BUILD_ALL=0
            BUILD_TARGET=$arg
            ;;
    esac
done

if [ $CLEAN_MODE -eq 1 ]; then
    echo "Cleaning build artifacts..."
    rm -rf "$BUILD_DIR"
    echo "Done."
    exit 0
fi

if ! command -v hipcc &> /dev/null; then
    echo "Error: hipcc not found."
    echo ""
    echo "Install ROCm for AMD GPUs:"
    echo "  https://rocm.docs.amd.com/en/latest/"
    echo ""
    echo "Or install HIP on NVIDIA:"
    echo "  https://rocm.docs.amd.com/en/latest/install/hip.html"
    exit 1
fi

mkdir -p "$BUILD_DIR"

BUILD_FLAGS="-O3 -Wall"
if [ $DEBUG_MODE -eq 1 ]; then
    BUILD_FLAGS="-g -O0 -Wall"
fi

if [ $NVIDIA_MODE -eq 1 ]; then
    PLATFORM_FLAGS="--platform nvidia"
    echo "Target: NVIDIA GPU (CUDA backend)"
else
    PLATFORM_FLAGS=""
    echo "Target: AMD GPU (ROCm backend)"
fi

build_example() {
    local name=$1
    local src="$SCRIPT_DIR/${name}.hip"
    local out="$BUILD_DIR/${name}"
    
    if [ ! -f "$src" ]; then
        echo "Error: Source file not found: $src"
        return 1
    fi
    
    echo "Building: $name"
    hipcc $PLATFORM_FLAGS $BUILD_FLAGS -o "$out" "$src"
    echo "  -> $out"
}

echo "=== STUNIR ROCm Examples Build ==="
echo "Build directory: $BUILD_DIR"
echo ""

if [ $BUILD_ALL -eq 1 ]; then
    build_example "vector_add"
    build_example "matmul"
    build_example "reduction"
else
    build_example "$BUILD_TARGET"
fi

echo ""
echo "Build complete!"
echo ""
echo "Run examples:"
echo "  $BUILD_DIR/vector_add"
echo "  $BUILD_DIR/matmul"
echo "  $BUILD_DIR/reduction"
