#!/bin/bash
set -e
echo "Building Haskell-First pipeline (ISSUE.HASKELL.0001)"

# Haskell canonical (priority 1)
cd tools/native/haskell && cabal build
cd ../ir_canonicalizer && cabal build

# Conformance cascade (Rust/Python MUST match)
echo "Haskell reference established. Ready for ISSUE.CONFORMANCE.0001"

echo "✓ Haskell-First build complete"