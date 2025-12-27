#!/usr/bin/env bash
# scripts/lib/dispatch.sh
# Polyglot Dispatcher: Routes logical operations to the best available implementation.
# Priority: Native Binary > Python Script > Shell Function

# 1. Detect Capabilities
HAS_NATIVE=0
if [[ -n "${STUNIR_BIN:-}" ]] && [[ -x "$STUNIR_BIN" ]]; then
    HAS_NATIVE=1
fi

HAS_PYTHON=0
if [[ -n "${STUNIR_TOOL_PYTHON:-}" ]]; then
    HAS_PYTHON=1
fi

# 2. Dispatch Function
# Usage: stunir_dispatch <logical_operation> [args...]
# Example: stunir_dispatch record_receipt --target foo.json ...
stunir_dispatch() {
    local op="$1"
    shift

    # --- STRATEGY A: Native Binary ---
    if [[ "$HAS_NATIVE" == "1" ]]; then
        # Convert snake_case op to kebab-case command (e.g., record_receipt -> record-receipt)
        local cmd="${op//_/-}"
        # Execute and return
        "$STUNIR_BIN" "$cmd" "$@"
        return $?
    fi

    # --- STRATEGY B: Python Toolchain ---
    if [[ "$HAS_PYTHON" == "1" ]]; then
        # Map logical op to python script path
        local script="tools/${op}.py"

        if [[ -f "$script" ]]; then
            "$STUNIR_TOOL_PYTHON" "$script" "$@"
            return $?
        else
            echo "WARNING: Python implementation for '$op' not found at '$script'" >&2
            # Fall through to shell
        fi
    fi

    # --- STRATEGY C: Shell Native (Profile 3) ---
    # Look for a shell library that might implement this
    # Convention: scripts/lib/<op>.sh or scripts/lib/core.sh

    # Specific library check (e.g., scripts/lib/receipt.sh for record_receipt)
    # We strip the prefix to find the category if needed, but for now let's try a direct map or common libs.

    if [[ "$op" == "record_receipt" ]]; then
        if [[ -f "scripts/lib/receipt.sh" ]]; then
            source "scripts/lib/receipt.sh"
            stunir_shell_record_receipt "$@"
            return $?
        fi
    elif [[ "$op" == "epoch" ]]; then
        if [[ -f "scripts/lib/epoch.sh" ]]; then
            source "scripts/lib/epoch.sh"
            stunir_shell_epoch "$@"
            return $?
        fi
    fi

    # --- FAILURE ---
    echo "CRITICAL ERROR: No implementation found for operation '$op'." >&2
    echo "  Native Binary: ${STUNIR_BIN:-Not Found}" >&2
    echo "  Python: ${STUNIR_TOOL_PYTHON:-Not Found}" >&2
    echo "  Shell Lib: scripts/lib/${op}.sh (Not Found)" >&2
    exit 99
}

# Export for subshells
export -f stunir_dispatch
