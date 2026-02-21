#!/bin/bash
# Build STUNIR Multi-GPU Support

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ! command -v hipcc &> /dev/null; then
    echo "Error: hipcc not found"
    exit 1
fi

echo "=== Building STUNIR Multi-GPU Support ==="

hipcc -O3 -o multi_gpu_example multi_gpu_example.hip

echo "Build complete!"
echo "Run: ./multi_gpu_example"
