#!/bin/bash
set -euo pipefail

# STUNIR Native Core Builder
# Builds the Haskell Native Core from tools/native/haskell/stunir-native

echo "🔨 [STUNIR] Building Native Core..."

# Ensure we are in the project root
if [ ! -d "tools/native/haskell/stunir-native" ]; then
    echo "❌ Error: Cannot find tools/native/haskell/stunir-native. Run from project root."
    exit 1
fi

# Create output directory
mkdir -p build/bin

# Build Haskell Core
echo "   -> Compiling Haskell sources..."
cd tools/native/haskell/stunir-native

# Check for cabal
if ! command -v cabal &> /dev/null; then
    echo "❌ Error: 'cabal' not found. Please install GHC/Cabal."
    exit 1
fi

cabal update
cabal build

# Find and copy binary
# We use 'cabal list-bin' to find the exact path, handling architecture/version differences
BIN_PATH=$(cabal list-bin exe:stunir-native)

if [ -f "$BIN_PATH" ]; then
    echo "   -> Binary found at: $BIN_PATH"
    cp "$BIN_PATH" ../../../../build/bin/stunir-native
    chmod +x ../../../../build/bin/stunir-native
    echo "✅ [STUNIR] Native Core installed to build/bin/stunir-native"
else
    echo "❌ Error: Could not locate built binary."
    exit 1
fi
