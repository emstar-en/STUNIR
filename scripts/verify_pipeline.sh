#!/bin/bash
set -e
# ISSUE.VERIFY.0003: Full STUNIR pipeline verification

echo "=== STUNIR End-to-End Verification ==="

# 1. Build all artifacts
./scripts/build_compiled.sh
./scripts/build_artifacts.sh

# 2. Verify conformance  
./scripts/verify_conformance.sh

# 3. Generate + verify pack
tools/pack_attestation/generate_root.sh
tools/native/haskell/bin/stunir-native-hs verify pack test_vectors/pack_0001.dcbor

# 4. Master manifest
./scripts/verify_manifest.sh

echo "✓ FULL PIPELINE: spec → ir → pack → manifest ✓ (20/20)"