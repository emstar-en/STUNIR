#!/usr/bin/env bash
# scripts/build.sh
# STUNIR Master Build Script (Polyglot Dispatch)

# Source core library
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/stunir_core.sh"

# --- Configuration ---
LOCKFILE="build/local_toolchain.lock.json"

# --- 1. Bootstrap / Discovery ---
if [ ! -f "$LOCKFILE" ]; then
    log_info "Lockfile not found. Running discovery..."
    bash "$SCRIPT_DIR/discovery.sh"
fi

# --- 2. Orchestration ---
log_info "Starting Build Orchestration..."

# --- 3. Dispatch to Semantic Engine ---

# Strategy: Prefer Native (Rust/Haskell) -> Fallback to Python -> Fallback to Shell

NATIVE_BIN=""

# Check for Rust binary
if [ -f "tools/native/rust/target/debug/stunir-rust" ]; then
    NATIVE_BIN="tools/native/rust/target/debug/stunir-rust"
elif [ -f "tools/native/rust/target/release/stunir-rust" ]; then
    NATIVE_BIN="tools/native/rust/target/release/stunir-rust"
fi

# (Optional) Check for Haskell binary if Rust not found
# Haskell paths are tricky with cabal, usually we'd expect it installed to a bin dir.
# For now, we stick to Rust as primary native detection.

if [ -n "$NATIVE_BIN" ]; then
    log_info "Native Engine detected: $NATIVE_BIN"

    # Example Pipeline Execution
    mkdir -p build/ir build/prov

    # 1. Spec -> IR
    log_info "[Native] Generating IR..."
    # In real usage, we'd loop over specs. Here we mock with a test spec if exists.
    if [ -f "spec/main.stunir" ]; then
        "$NATIVE_BIN" spec-to-ir --in-json "spec/main.stunir" --out-ir "build/ir/main.ir"
    fi

    # 2. Gen Provenance
    log_info "[Native] Generating Provenance..."
    # Mock Epoch
    echo '{"timestamp": "now"}' > build/epoch.json
    if [ -f "build/ir/main.ir" ]; then
        "$NATIVE_BIN" gen-provenance --in-ir "build/ir/main.ir" --epoch-json "build/epoch.json" --out-prov "build/prov/main.prov"
    fi

else
    # Fallback to Python
    PYTHON_PATH=$(grep '"python":' "$LOCKFILE" | grep -o '"path": "[^"]*"' | cut -d'"' -f4)

    if [ -n "$PYTHON_PATH" ] && [ -x "$PYTHON_PATH" ]; then
        log_info "Python detected ($PYTHON_PATH). Dispatching to Semantic Engine..."
        # "$PYTHON_PATH" scripts/core/spec_to_ir.py ...
        log_info "(Simulation) Running Python Semantic Engine..."
    else
        log_warn "No Semantic Engine found (Native or Python). Falling back to Shell-Only Verification Mode."
    fi
fi

# --- 4. Manifest Generation ---
mkdir -p dist
log_info "Generating build manifest..."
generate_manifest "scripts" "dist/manifest_scripts.txt"

log_info "Build Complete."
