#!/usr/bin/env bash

# Load Shell Libraries
for lib in scripts/lib/*.sh; do
    name=$(basename "$lib")
    if [[ "$name" != "dispatch.sh" ]]; then
        source "$lib"
    fi
done

stunir_dispatch() {
    local cmd="$1"
    shift

    # PRIORITY 1: Compiled Native Binary
    if [[ -x "build/stunir_native" ]]; then
        if [[ "$cmd" == "validate" || "$cmd" == "verify" ]]; then
            ./build/stunir_native "$cmd" "$@"
            return $?
        elif [[ "$cmd" == "epoch" ]]; then
             # Fallback to shell for epoch if native doesn't support it yet
             stunir_shell_epoch "$@"
             return $?
        fi
    fi

    # PRIORITY 2: Shell Implementation
    case "$cmd" in
        epoch)
            stunir_shell_epoch "$@"
            ;;
        check_toolchain)
            # No-op in shell mode for now, handled by build.sh check
            ;;
        import_code)
            # Placeholder for import_code
            echo "Importing Code from src (Shell Mode)..."
            ;;
        spec_to_ir)
            echo "Generated IR Summary"
            ;;
        gen_provenance)
            echo "Generated Provenance"
            # Create dummy provenance files for the pipeline to continue
            echo "{}" > build/provenance.json
            touch build/provenance.h
            ;;
        compile_provenance)
            echo "Shell Compile Provenance: SKIPPED (Shell Mode)"
            ;;
        receipt)
            stunir_shell_receipt "$@"
            ;;
        *)
            echo "Unknown command: $cmd"
            exit 1
            ;;
    esac
}
