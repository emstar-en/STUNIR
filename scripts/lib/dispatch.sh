#!/usr/bin/env bash
# scripts/lib/dispatch.sh
# STUNIR Polyglot Dispatcher

# Configuration
: "${STUNIR_NATIVE_BIN:=build/bin/stunir-native}"
: "${STUNIR_PYTHON_BIN:=python3}"

# Load Shell Libraries
source scripts/lib/json_canon.sh
source scripts/lib/receipt.sh

can_dispatch_native() {
    if [[ "${STUNIR_FORCE_NO_NATIVE:-0}" == "1" ]]; then return 1; fi
    if [[ -x "$STUNIR_NATIVE_BIN" ]]; then return 0; fi
    return 1
}

stunir_dispatch() {
    local cmd="$1"
    shift

    # 1. Native Route
    if can_dispatch_native; then
        local native_cmd="${cmd//_/-}"
        "$STUNIR_NATIVE_BIN" "$native_cmd" "$@"
        return $?
    fi

    # 2. Python Route
    local py_tool="tools/${cmd}.py"
    if [[ -f "$py_tool" && "${STUNIR_FORCE_NO_PYTHON:-0}" != "1" ]]; then
        "$STUNIR_PYTHON_BIN" "$py_tool" "$@"
        return $?
    fi

    # 3. Shell Fallback Route
    # Map "gen_receipt" -> "cmd_gen_receipt"
    local shell_func="cmd_${cmd}"
    if declare -f "$shell_func" > /dev/null; then
        "$shell_func" "$@"
        return $?
    fi

    echo "STUNIR DISPATCH ERROR: No implementation found for '$cmd'" >&2
    return 127
}

export -f stunir_dispatch
export -f can_dispatch_native
