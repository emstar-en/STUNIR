#!/bin/sh
# STUNIR Polyglot Build Entrypoint
# PRIMARY: Ada SPARK tools are the DEFAULT implementation
# 
# Auto-detection priority: 
#   1. Precompiled SPARK binaries (no compiler needed)
#   2. Built SPARK binaries (requires GNAT)
#   3. Native (Rust)
#   4. Python (reference only)
#   5. Shell (minimal)
#
# Override via: STUNIR_PROFILE=spark|native|python|shell
# Force precompiled: STUNIR_USE_PRECOMPILED=1

set -u

# --- Configuration ---
# Allow override via env var: STUNIR_PROFILE=spark|native|python|shell
PROFILE="${STUNIR_PROFILE:-auto}"
USE_PRECOMPILED="${STUNIR_USE_PRECOMPILED:-auto}"
SPEC_ROOT="spec"
OUT_IR="asm/spec_ir.json"
OUT_PY="asm/output.py"
LOCK_FILE="local_toolchain.lock.json"

# Detect platform for precompiled binaries
detect_platform() {
    case "$(uname -s)" in
        Linux)  echo "linux" ;;
        Darwin) echo "macos" ;;
        *)      echo "unknown" ;;
    esac
}

detect_arch() {
    case "$(uname -m)" in
        x86_64|amd64) echo "x86_64" ;;
        aarch64|arm64) echo "arm64" ;;
        *)             echo "unknown" ;;
    esac
}

PLATFORM="$(detect_platform)"
ARCH="$(detect_arch)"
PRECOMPILED_DIR="precompiled/${PLATFORM}-${ARCH}/spark/bin"

# Tool paths - prefer precompiled, fall back to built
# Only use tools from stunir_tools.gpr Main list (updated binaries)
# Deprecated tools (stunir_spec_to_ir_main, stunir_ir_to_code_main) are in bin/deprecated/
if [ -x "$PRECOMPILED_DIR/ir_converter_main" ] && [ "$USE_PRECOMPILED" != "0" ]; then
    SPARK_IR_CONVERTER="$PRECOMPILED_DIR/ir_converter_main"
    SPARK_CODE_EMITTER="$PRECOMPILED_DIR/code_emitter_main"
    SPARK_PIPELINE_DRIVER="$PRECOMPILED_DIR/pipeline_driver_main"
    USING_PRECOMPILED=1
else
    SPARK_IR_CONVERTER="tools/spark/bin/ir_converter_main"
    SPARK_CODE_EMITTER="tools/spark/bin/code_emitter_main"
    SPARK_PIPELINE_DRIVER="tools/spark/bin/pipeline_driver_main"
    USING_PRECOMPILED=0
fi
# Check for deprecated tools and warn
if [ -x "tools/spark/bin/stunir_spec_to_ir_main" ] || [ -x "tools/spark/bin/stunir_ir_to_code_main" ]; then
    warn "Deprecated tools found in bin/. Run: cd tools/spark/bin && ./deprecate_extras.cmd"
fi
NATIVE_BIN="tools/native/rust/stunir-native/target/release/stunir-native"

log() { echo "[STUNIR] $1"; }
warn() { echo "[STUNIR][WARN] $1"; }
error() { echo "[STUNIR][ERROR] $1" >&2; }

# --- Pre-flight ---
if [ ! -d "$SPEC_ROOT" ]; then
    log "Creating default spec directory..."
    mkdir -p "$SPEC_ROOT"
fi

# --- Detection ---
detect_runtime() {
    # Allow explicit override
    if [ "$PROFILE" = "spark" ]; then
        echo "spark"
        return
    elif [ "$PROFILE" = "native" ]; then
        echo "native"
        return
    elif [ "$PROFILE" = "python" ]; then
        echo "python"
        return
    elif [ "$PROFILE" = "shell" ]; then
        echo "shell"
        return
    fi

    # Auto-detection priority: Precompiled SPARK -> Built SPARK -> Native -> Python -> Shell
    # SPARK (Ada) is the PRIMARY and PREFERRED runtime
    if [ -x "$SPARK_IR_CONVERTER" ] && [ -x "$SPARK_CODE_EMITTER" ]; then
        if [ "$USING_PRECOMPILED" = "1" ]; then
            echo "[STUNIR] Using precompiled SPARK binaries (no GNAT compiler needed)" >&2
        fi
        echo "spark"
    elif [ -x "$NATIVE_BIN" ]; then
        echo "native"
    elif command -v python3 >/dev/null 2>&1; then
        echo "[STUNIR][WARN] Falling back to Python reference implementation (not recommended for production)" >&2
        echo "python"
    else
        echo "[STUNIR][WARN] Falling back to Shell (minimal functionality)" >&2
        echo "shell"
    fi
}

# --- Build SPARK tools if needed ---
build_spark_tools() {
    if [ ! -x "$SPARK_IR_CONVERTER" ] || [ ! -x "$SPARK_CODE_EMITTER" ]; then
        log "Building Ada SPARK tools..."
        if command -v gprbuild >/dev/null 2>&1; then
            (cd tools/spark && gprbuild -P stunir_tools.gpr) || {
                warn "Failed to build SPARK tools, falling back..."
                return 1
            }
            log "SPARK tools built successfully"
            return 0
        else
            warn "gprbuild not found, cannot build SPARK tools"
            return 1
        fi
    fi
    return 0
}

# --- Dispatch ---
# Try to build SPARK tools if auto-detecting
if [ "$PROFILE" = "auto" ]; then
    build_spark_tools || true
fi

RUNTIME=$(detect_runtime)
log "Runtime selected: $RUNTIME"
log "NOTE: Ada SPARK is the PRIMARY implementation. Python is reference only."

# 1. Discovery Phase (Always run shell manifest first to generate lockfile)
log "Running Toolchain Discovery..."
chmod +x scripts/lib/*.sh 2>/dev/null || true
./scripts/lib/manifest.sh 2>/dev/null || true

case "$RUNTIME" in
    spark)
        # PRIMARY: Ada SPARK implementation
        if [ ! -x "$SPARK_IR_CONVERTER" ]; then
            error "SPARK binary not found at $SPARK_IR_CONVERTER"
            error "Please run: cd tools/spark && gprbuild -P stunir_tools.gpr"
            exit 1
        fi
        
        log "Using Ada SPARK Core (PRIMARY implementation)"
        
        # 1. Spec -> IR (SPARK)
        "$SPARK_IR_CONVERTER" --spec-root "$SPEC_ROOT" --out "$OUT_IR" --lockfile "$LOCK_FILE"
        
        # 2. IR -> Code (SPARK)
        if [ -x "$SPARK_CODE_EMITTER" ]; then
            "$SPARK_CODE_EMITTER" --input "$OUT_IR" --output "$OUT_PY" --target python
        fi
        
        # 3. Generate IR Manifest
        if [ -d "asm/ir" ]; then
            log "Generating IR Bundle Manifest..."
            mkdir -p receipts
            # Use SPARK or fall back to shell
            ./scripts/lib/gen_ir_manifest.sh asm/ir receipts/ir_manifest.json 2>/dev/null || \
                log "Note: IR manifest generation using fallback"
        fi
        
        log "Build complete (Ada SPARK - PRIMARY)"
        ;;

    native)
        # Secondary: Rust/Haskell native implementation
        if [ ! -x "$NATIVE_BIN" ]; then
            error "Native binary not found at $NATIVE_BIN"
            error "Please run: cd tools/native/rust/stunir-native && cargo build --release"
            exit 1
        fi
        
        log "Using Native Core: $NATIVE_BIN"
        warn "Consider using Ada SPARK (PRIMARY) instead: STUNIR_PROFILE=spark"
        
        # 1. Spec -> IR (Native Directory Scan)
        "$NATIVE_BIN" spec-to-ir --input "$SPEC_ROOT" --output "$OUT_IR"
        
        # 2. IR -> Code
        "$NATIVE_BIN" emit --input "$OUT_IR" --target python --output "$OUT_PY"
        
        # 3. Generate IR Manifest
        if [ -d "asm/ir" ]; then
            log "Generating IR Bundle Manifest..."
            mkdir -p receipts
            "$NATIVE_BIN" gen-ir-manifest --ir-dir asm/ir --out receipts/ir_manifest.json 2>/dev/null || \
                log "Note: gen-ir-manifest requires Haskell native build"
        fi
        
        log "Build complete (Native)"
        ;;

    python)
        # REFERENCE ONLY: Python implementation (NOT recommended for production)
        warn "=============================================="
        warn "USING PYTHON REFERENCE IMPLEMENTATION"
        warn "This is NOT recommended for production use."
        warn "Python tools are for REFERENCE/READABILITY only."
        warn "For production, use Ada SPARK: STUNIR_PROFILE=spark"
        warn "=============================================="
        
        python3 -B tools/spec_to_ir.py \
            --spec-root "$SPEC_ROOT" \
            --out "$OUT_IR" \
            --lockfile "$LOCK_FILE"
        
        # Emit dCBOR IR artifacts
        log "Emitting dCBOR IR artifacts..."
        ./scripts/lib/emit_dcbor.sh "$OUT_IR" asm/ir 2>/dev/null || log "Note: dCBOR emission skipped"
        
        # Generate IR manifest
        if [ -d "asm/ir" ]; then
            log "Generating IR Bundle Manifest..."
            mkdir -p receipts
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
        
        log "Build complete (Python - REFERENCE ONLY)"
        warn "For production, rebuild with: STUNIR_PROFILE=spark ./scripts/build.sh"
        ;;

    shell)
        # Shell-Native path (minimal functionality)
        warn "Using Shell fallback (minimal functionality)"
        warn "For full functionality, use Ada SPARK: STUNIR_PROFILE=spark"
        
        ./scripts/lib/runner.sh
        
        # Emit dCBOR IR artifacts for shell path too
        if [ -f "$OUT_IR" ]; then
            log "Emitting dCBOR IR artifacts..."
            ./scripts/lib/emit_dcbor.sh "$OUT_IR" asm/ir 2>/dev/null || log "Note: dCBOR emission skipped"
        fi
        
        log "Build complete (Shell)"
        ;;

    *)
        error "Unknown runtime profile '$RUNTIME'"
        exit 1
        ;;
esac

# Final summary
echo ""
log "=============================================="
log "Build Summary"
log "=============================================="
log "Runtime: $RUNTIME"
if [ "$RUNTIME" = "spark" ] && [ "$USING_PRECOMPILED" = "1" ]; then
    log "Binary Source: Precompiled (no GNAT compiler required)"
elif [ "$RUNTIME" = "spark" ]; then
    log "Binary Source: Built from source"
fi
log "Spec Root: $SPEC_ROOT"
log "Output IR: $OUT_IR"
if [ "$RUNTIME" != "spark" ]; then
    warn "NOTE: For production use, switch to Ada SPARK:"
    warn "  STUNIR_PROFILE=spark ./scripts/build.sh"
fi
log "=============================================="
