#!/bin/bash
set -e

# STUNIR Bootstrap
# ----------------
# Compiles the native toolchain so the build pipeline can use it.

STUNIR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=== STUNIR Bootstrap ==="

# Detect Language Preference (Default to Haskell as per v2 spec)
LANG="haskell"
if [ -d "$STUNIR_ROOT/tools/native/rust" ]; then
    # If user explicitly wants rust or has rust but not haskell configured...
    # For now, we stick to the Haskell path unless told otherwise.
    :
fi

if [ "$LANG" == "haskell" ]; then
    echo "Bootstrapping Haskell Native Core..."
    cd "$STUNIR_ROOT/tools/native/haskell/stunir-native"

    # Check for cabal
    if ! command -v cabal &> /dev/null; then
        echo "Error: 'cabal' not found. Cannot bootstrap Haskell core."
        exit 1
    fi

    cabal update
    cabal build

    # Locate binary
    BIN_PATH=$(cabal list-bin stunir-native)
    echo "Built binary at: $BIN_PATH"

    # Verify it works
    "$BIN_PATH" version
else
    echo "Bootstrapping Rust Native Core..."
    cd "$STUNIR_ROOT/tools/native/rust/stunir-native"
    cargo build --release
fi

echo "Bootstrap Complete."
