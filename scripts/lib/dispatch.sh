#!/bin/bash

# STUNIR Dispatcher
# -----------------
# Routes commands to the best available implementation:
# 1. Native Binary (stunir-native) - Preferred for speed and strictness
# 2. Python Scripts (tools/*.py)   - Fallback for development
# 3. Shell Functions (lib/*.sh)    - Last resort / bootstrapping

# Locate Native Binary
# We look in the standard build location or PATH
NATIVE_BIN="$STUNIR_ROOT/tools/native/haskell/stunir-native/dist-newstyle/build/x86_64-linux/ghc-9.4.8/stunir-native-0.5.0.0/x/stunir-native/build/stunir-native/stunir-native"
# Also check for Rust binary location
if [ ! -f "$NATIVE_BIN" ]; then
    NATIVE_BIN="$STUNIR_ROOT/tools/native/rust/stunir-native/target/release/stunir-native"
fi

stunir_dispatch() {
    local cmd="$1"
    shift

    # Strategy 1: Native Binary
    if [ -x "$NATIVE_BIN" ]; then
        # echo "[DEBUG] Dispatching to Native: $cmd" >&2
        "$NATIVE_BIN" "$cmd" "$@"
        return $?
    fi

    # Strategy 2: Python Tools
    # Map kebab-case commands to python scripts
    # e.g. spec-to-ir -> tools/spec_to_ir.py
    local py_script="$STUNIR_ROOT/tools/${cmd//-/_}.py"
    if [ -f "$py_script" ]; then
        # echo "[DEBUG] Dispatching to Python: $cmd" >&2
        python3 "$py_script" "$@"
        return $?
    fi

    # Strategy 3: Shell Fallback
    # Load shell library if needed
    local sh_lib="$STUNIR_ROOT/scripts/lib/${cmd//-/_}.sh"
    if [ -f "$sh_lib" ]; then
        source "$sh_lib"
        "stunir_${cmd//-/_}" "$@"
        return $?
    fi

    echo "Error: No implementation found for command '$cmd'" >&2
    echo "  Checked Native: $NATIVE_BIN" >&2
    echo "  Checked Python: $py_script" >&2
    echo "  Checked Shell:  $sh_lib" >&2
    exit 127
}
