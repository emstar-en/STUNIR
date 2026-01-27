#!/bin/sh
# STUNIR Polyglot Build Entrypoint
# Automatically dispatches to the best available runtime.

set -u

# --- Configuration ---
# Allow override via env var: STUNIR_PROFILE=native|python|shell
PROFILE="${STUNIR_PROFILE:-auto}"
SPEC_ROOT="spec"
OUT_IR="asm/spec_ir.json"
OUT_PY="asm/output.py"
LOCK_FILE="local_toolchain.lock.json"
# Correct path to the Rust binary we built
NATIVE_BIN="tools/native/rust/stunir-native/target/release/stunir-native"

log() { echo "[STUNIR] $1"; }

# --- Pre-flight ---
if [ ! -d "$SPEC_ROOT" ]; then
    log "Creating default spec directory..."
    mkdir -p "$SPEC_ROOT"
fi

# --- Detection ---
detect_runtime() {
    if [ "$PROFILE" = "native" ]; then
        echo "native"
        return
    elif [ "$PROFILE" = "shell" ]; then
        echo "shell"
        return
    elif [ "$PROFILE" = "python" ]; then
        echo "python"
        return
    fi

    # Auto-detection priority: Native -> Python -> Shell
    if [ -x "$NATIVE_BIN" ]; then
        echo "native"
    elif command -v python3 >/dev/null 2>&1; then
        echo "python"
    else
        echo "shell"
    fi
}

# --- Dispatch ---
RUNTIME=$(detect_runtime)
log "Runtime selected: $RUNTIME"

# 1. Discovery Phase (Always run shell manifest first to generate lockfile)
log "Running Toolchain Discovery..."
chmod +x scripts/lib/*.sh 2>/dev/null || true
./scripts/lib/manifest.sh

case "$RUNTIME" in
    native)
        if [ ! -x "$NATIVE_BIN" ]; then
            log "Error: Native binary not found at $NATIVE_BIN"
            log "Please run: cd tools/native/rust/stunir-native && cargo build --release"
            exit 1
        fi
        
        log "Using Native Core: $NATIVE_BIN"
        
        # 1. Spec -> IR (Native Directory Scan)
        "$NATIVE_BIN" spec-to-ir --input "$SPEC_ROOT" --output "$OUT_IR"
        
        # 2. IR -> Code
        "$NATIVE_BIN" emit --input "$OUT_IR" --target python --output "$OUT_PY"
        
        # 3. Generate IR Manifest (Issue: native/haskell/1205)
        # Creates deterministic manifest with SHA256 hashes
        if [ -d "asm/ir" ]; then
            log "Generating IR Bundle Manifest..."
            mkdir -p receipts
            "$NATIVE_BIN" gen-ir-manifest --ir-dir asm/ir --out receipts/ir_manifest.json 2>/dev/null || \
                log "Note: gen-ir-manifest requires Haskell native build"
        fi
        
        log "Build complete (Native)"
        ;;

    python)
        # Original Python path
        python3 -B tools/spec_to_ir.py \
            --spec-root "$SPEC_ROOT" \
            --out "$OUT_IR" \
            --lockfile "$LOCK_FILE"
        
        # Issue: ISSUE.IR.0001 - Emit dCBOR IR artifacts
        log "Emitting dCBOR IR artifacts..."
        ./scripts/lib/emit_dcbor.sh "$OUT_IR" asm/ir 2>/dev/null || log "Note: dCBOR emission skipped"
        
        # Generate IR manifest
        if [ -d "asm/ir" ]; then
            log "Generating IR Bundle Manifest..."
            mkdir -p receipts
            # Use Python fallback for manifest if Haskell not available
            python3 -c "
import os, json, hashlib
ir_dir = 'asm/ir'
entries = []
for f in sorted(os.listdir(ir_dir)):
    if f.endswith('.dcbor'):
        path = os.path.join(ir_dir, f)
        with open(path, 'rb') as fp:
            content = fp.read()
        entries.append({
            'filename': f,
            'sha256': hashlib.sha256(content).hexdigest(),
            'size': len(content)
        })
manifest = {
    'schema': 'stunir.ir_manifest.v2',
    'version': '1.0.0',
    'ir_count': len(entries),
    'files': entries
}
os.makedirs('receipts', exist_ok=True)
with open('receipts/ir_manifest.json', 'w') as f:
    json.dump(manifest, f, indent=2, sort_keys=True)
print(f'Generated receipts/ir_manifest.json with {len(entries)} files')
" 2>/dev/null || log "Note: IR manifest generation skipped"
        fi
        
        log "Build complete (Python)"
        ;;

    shell)
        # Shell-Native path
        ./scripts/lib/runner.sh
        
        # Issue: ISSUE.IR.0001 - Emit dCBOR IR artifacts for shell path too
        if [ -f "$OUT_IR" ]; then
            log "Emitting dCBOR IR artifacts..."
            ./scripts/lib/emit_dcbor.sh "$OUT_IR" asm/ir 2>/dev/null || log "Note: dCBOR emission skipped"
        fi
        
        log "Build complete (Shell)"
        ;;

    *)
        log "Error: Unknown runtime profile '$RUNTIME'"
        exit 1
        ;;
esac
