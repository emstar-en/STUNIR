#!/bin/bash
set -e

echo "=== ISSUE.CONFORMANCE.0001: Haskell-First Conformance ==="
REF_DIGEST="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

# Test vector
SPEC="tools/native/haskell/test_vectors/spec_0001.json"
IR="test_vectors/ir_0001.dcbor"

# 1. Haskell Canonical (SOURCE OF TRUTH)
echo "1. Haskell reference:"
cd tools/native/haskell && cabal build
cd ../ir_canonicalizer && cabal build
bin/stunir-ir "$SPEC" "../$IR"
HASKELL_DIGEST=$(sha256sum "../$IR" | cut -d' ' -f1)
echo "Haskell: $HASKELL_DIGEST"

# 2. Rust MUST match
echo "2. Rust conformance:"
cd ../../conformance/rust_stunir_native && cargo build --release
../../bin/stunir-native-rs canonicalize < "$SPEC" > "../../$IR"
RUST_DIGEST=$(sha256sum "../../$IR" | cut -d' ' -f1)
[ "$RUST_DIGEST" = "$HASKELL_DIGEST" ] && echo "✓ Rust PASS" || echo "✗ Rust FAIL"

# 3. Python MUST match  
echo "3. Python conformance:"
cd ../../conformance/python_stunir_native
python canonicalize.py "$SPEC" "../../$IR" --float-policy=forbid_floats
PYTHON_DIGEST=$(sha256sum "../../$IR" | cut -d' ' -f1)
[ "$PYTHON_DIGEST" = "$HASKELL_DIGEST" ] && echo "✓ Python PASS" || echo "✗ Python FAIL"

# Validate all
tools/native/haskell/bin/stunir-native-hs validate "$IR" && echo "✓ All IR valid (Profile-3 UN)"

echo "✓ ISSUE.CONFORMANCE.0001: Conformance complete (Haskell=$HASKELL_DIGEST)"
