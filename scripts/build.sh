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

NATIVE_BIN=""
PYTHON_MINIMAL="tools/python/stunir_minimal.py"

# Priority 1: Compiled Native (Rust)
if [ -f "tools/native/rust/target/debug/stunir-rust" ]; then
    NATIVE_BIN="tools/native/rust/target/debug/stunir-rust"
elif [ -f "tools/native/rust/target/release/stunir-rust" ]; then
    NATIVE_BIN="tools/native/rust/target/release/stunir-rust"
fi

# Priority 2: Minimal Python (if Native missing)
if [ -z "$NATIVE_BIN" ] && [ -f "$PYTHON_MINIMAL" ]; then
    # Check if we have python3
    PYTHON_PATH=$(grep '"python":' "$LOCKFILE" | grep -o '"path": "[^"]*"' | cut -d'"' -f4)
    if [ -n "$PYTHON_PATH" ]; then
        log_info "Native binary not found. Using Minimal Python Pipeline ($PYTHON_MINIMAL)..."
        NATIVE_BIN="$PYTHON_PATH $PYTHON_MINIMAL"
    fi
fi

# Execute Generation
if [ -n "$NATIVE_BIN" ]; then
    log_info "Semantic Engine: $NATIVE_BIN"

    mkdir -p build/ir build/prov

    # 1. Spec -> IR
    log_info "[Engine] Generating IR..."
    if [ -f "spec/main.stunir" ]; then
        $NATIVE_BIN spec-to-ir --in-json "spec/main.stunir" --out-ir "build/ir/main.ir"
    fi

    # 2. Gen Provenance
    log_info "[Engine] Generating Provenance..."
    echo '{"timestamp": "now"}' > build/epoch.json
    if [ -f "build/ir/main.ir" ]; then
        $NATIVE_BIN gen-provenance --in-ir "build/ir/main.ir" --epoch-json "build/epoch.json" --out-prov "build/prov/main.prov"
    fi

else
    log_warn "No Semantic Engine found (Native or Python). Skipping Generation."
fi

# --- 4. Shell-Native Verification ---
log_info "Running Shell-Native Verification..."
if [ -f "build/ir/main.ir" ] && [ -f "build/prov/main.prov" ]; then
    bash "$SCRIPT_DIR/verify_shell.sh" "build/ir/main.ir" "build/prov/main.prov"
else
    log_warn "Artifacts missing, skipping verification."
fi

# --- 5. Manifest Generation ---
mkdir -p dist
log_info "Generating build manifest..."
generate_manifest "scripts" "dist/manifest_scripts.txt"

log_info "Build Complete."
