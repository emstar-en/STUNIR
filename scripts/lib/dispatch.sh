#!/usr/bin/env bash
# scripts/lib/dispatch.sh
# STUNIR Polyglot Dispatcher
# Routes commands to the appropriate backend with priority:
# 1. Native Binary (Haskell/Rust) - The Trust Anchor
# 2. Python Tools - The Prototyping/Legacy Path
# 3. Shell Fallback - The Bootstrap/Verification Path

# Configuration
# Users can override STUNIR_NATIVE_BIN to point to a specific binary (e.g. dist-newstyle location)
: "${STUNIR_NATIVE_BIN:=build/bin/stunir-native}"
: "${STUNIR_PYTHON_BIN:=python3}"

# Check if native dispatch is viable
can_dispatch_native() {
    if [[ "${STUNIR_FORCE_NO_NATIVE:-0}" == "1" ]]; then
        return 1
    fi
    if [[ -x "$STUNIR_NATIVE_BIN" ]]; then
        return 0
    fi
    return 1
}

# Main Dispatch Function
stunir_dispatch() {
    local cmd="$1"
    shift

    # 1. Native Route (Haskell/Rust)
    if can_dispatch_native; then
        # Convert snake_case command to kebab-case for CLI if needed
        # e.g. "gen_receipt" -> "gen-receipt"
        local native_cmd="${cmd//_/-}"

        # Execute and return exit code directly
        "$STUNIR_NATIVE_BIN" "$native_cmd" "$@"
        return $?
    fi

    # 2. Python Route
    # Maps "command" to "tools/command.py"
    local py_tool="tools/${cmd}.py"
    if [[ -f "$py_tool" && "${STUNIR_FORCE_NO_PYTHON:-0}" != "1" ]]; then
        "$STUNIR_PYTHON_BIN" "$py_tool" "$@"
        return $?
    fi

    # 3. Shell/Error Route
    # If we are here, we have no implementation.
    # In the future, shell functions can be registered here.
    echo "STUNIR DISPATCH ERROR: No implementation found for '$cmd'" >&2
    echo "  Checked Native: $STUNIR_NATIVE_BIN (Found: $(if [[ -x "$STUNIR_NATIVE_BIN" ]]; then echo "Yes"; else echo "No"; fi))" >&2
    echo "  Checked Python: $py_tool (Found: $(if [[ -f "$py_tool" ]]; then echo "Yes"; else echo "No"; fi))" >&2
    return 127
}

# Export for use in build scripts
export -f stunir_dispatch
export -f can_dispatch_native
