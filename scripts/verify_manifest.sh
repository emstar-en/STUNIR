#!/bin/bash
set -e
# ISSUE.MANIFEST.0001: Master manifest verification

tools/native/haskell/bin/stunir-native-hs verify manifest manifests/stunir_complete_v1.json
echo "✓ Master manifest: COMPLETE (20/20 issues)"