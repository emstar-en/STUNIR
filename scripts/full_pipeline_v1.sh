#!/bin/bash
set -e

echo "=== STUNIR Haskell-First Full Pipeline (V1) ==="

scripts/dispatch.sh ISSUE.HASKELL.0002
scripts/canonicalize.sh
scripts/test_conformance.sh
tools/pack_attestation/generate_root.sh
tools/native/haskell/bin/stunir-native-hs verify pack test_vectors/pack_0001.dcbor

echo "✓ V1 FULL PIPELINE: spec.json → root_attestation.dcbor ✓"