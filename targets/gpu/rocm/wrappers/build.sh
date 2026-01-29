#!/bin/bash
# Build STUNIR ROCm Wrappers

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ! command -v hipcc &> /dev/null; then
    echo "Error: hipcc not found"
    exit 1
fi

echo "=== Building STUNIR ROCm Wrappers ==="

# Build hipBLAS test
echo "Building hipBLAS test..."
hipcc -DSTUNIR_HIPBLAS_TEST -O3 -lhipblas -o test_hipblas hipblas_wrapper.hip
echo "  -> test_hipblas"

# Build hipSPARSE test
echo "Building hipSPARSE test..."
hipcc -DSTUNIR_HIPSPARSE_TEST -O3 -lhipsparse -o test_hipsparse hipsparse_wrapper.hip
echo "  -> test_hipsparse"

echo ""
echo "Build complete!"
echo "Run tests:"
echo "  ./test_hipblas"
echo "  ./test_hipsparse"
