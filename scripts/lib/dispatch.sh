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
stunir_dispatch() {
    local op="$1"
    shift

    # --- STRATEGY A: Native Binary ---
    if [[ "$HAS_NATIVE" == "1" ]]; then
        local cmd="${op//_/-}"
        "$STUNIR_BIN" "$cmd" "$@"
        return $?
    fi

    # --- STRATEGY B: Python Toolchain ---
    if [[ "$HAS_PYTHON" == "1" ]]; then
        local script="tools/${op}.py"
        if [[ -f "$script" ]]; then
            "$STUNIR_TOOL_PYTHON" "$script" "$@"
            return $?
        fi
    fi

    # --- STRATEGY C: Shell Native (Profile 3) ---
    # Map operations to shell functions

    case "$op" in
        epoch)
            if [[ -f "scripts/lib/epoch.sh" ]]; then
                source "scripts/lib/epoch.sh"
                stunir_shell_epoch "$@"
                return $?
            fi
            ;;
        record_receipt)
            if [[ -f "scripts/lib/receipt.sh" ]]; then
                source "scripts/lib/receipt.sh"
                stunir_shell_record_receipt "$@"
                return $?
            fi
            ;;
        spec_to_ir)
            if [[ -f "scripts/lib/spec_to_ir.sh" ]]; then
                source "scripts/lib/spec_to_ir.sh"
                stunir_shell_spec_to_ir "$@"
                return $?
            fi
            ;;
        spec_to_ir_files)
            if [[ -f "scripts/lib/spec_to_ir_files.sh" ]]; then
                source "scripts/lib/spec_to_ir_files.sh"
                stunir_shell_spec_to_ir_files "$@"
                return $?
            fi
            ;;
        gen_provenance)
            if [[ -f "scripts/lib/gen_provenance.sh" ]]; then
                source "scripts/lib/gen_provenance.sh"
                stunir_shell_gen_provenance "$@"
                return $?
            fi
            ;;
    esac

    # --- FAILURE ---
    echo "CRITICAL ERROR: No implementation found for operation '$op'." >&2
    echo "  Native Binary: ${STUNIR_BIN:-Not Found}" >&2
    echo "  Python: ${STUNIR_TOOL_PYTHON:-Not Found}" >&2
    echo "  Shell Lib: scripts/lib/${op}.sh (Not Found)" >&2
    exit 99
}
export -f stunir_dispatch
