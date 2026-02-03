#!/bin/bash
# Build STUNIR ROCm Benchmarks

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ! command -v hipcc &> /dev/null; then
    echo "Error: hipcc not found"
    exit 1
fi

echo "=== Building STUNIR ROCm Benchmarks ==="

FLAGS="-O3 -Wall"
DEBUG=""

for arg in "$@"; do
    case $arg in
        --debug)
            FLAGS="-O0 -g"
            DEBUG="-DDEBUG"
            ;;
        --profile)
            FLAGS="$FLAGS -DPROFILE"
            ;;
    esac
done

echo "Building kernel_benchmarks..."
hipcc $FLAGS $DEBUG -o kernel_benchmarks kernel_benchmarks.hip

echo ""
echo "Build complete!"
echo "Run: ./kernel_benchmarks"
