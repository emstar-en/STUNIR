#!/usr/bin/env bash

# Load Shell Libraries
if [[ -f "scripts/lib/json_canon.sh" ]]; then
    source scripts/lib/json_canon.sh
fi

for lib in scripts/lib/*.sh; do
    name=$(basename "$lib")
    if [[ "$name" != "dispatch.sh" && "$name" != "json_canon.sh" ]]; then
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
            # No-op in shell mode
            ;;
        import_code)
            local out_spec=""
            while [[ $# -gt 0 ]]; do
                case "$1" in
                    --out-spec) out_spec="$2"; shift 2 ;;
                    *) shift ;;
                esac
            done
            echo "Importing Code from src (Shell Mode)..."
            if [[ -n "$out_spec" ]]; then
                stunir_canon_echo '{"kind":"spec","modules":[]}' > "$out_spec"
            fi
            ;;
        spec_to_ir)
            local out_ir=""
            while [[ $# -gt 0 ]]; do
                case "$1" in
                    --out) out_ir="$2"; shift 2 ;;
                    *) shift ;;
                esac
            done
            echo "Generated IR Summary"
            if [[ -n "$out_ir" ]]; then
                stunir_canon_echo '{}' > "$out_ir"
            fi
            ;;
        gen_provenance)
            local out_json=""
            local out_header=""
            local epoch_val=0
            local epoch_src="UNKNOWN"
            
            while [[ $# -gt 0 ]]; do
                case "$1" in
                    --out-json) out_json="$2"; shift 2 ;;
                    --out-header) out_header="$2"; shift 2 ;;
                    --epoch) epoch_val="$2"; shift 2 ;;
                    --epoch-source) epoch_src="$2"; shift 2 ;;
                    *) shift ;;
                esac
            done
            echo "Generated Provenance"
            if [[ -n "$out_json" ]]; then
                # Construct JSON string manually to ensure structure, then canonicalize
                # Note: epoch_val is int, epoch_src is string
                local json_str="{\"build_epoch\": $epoch_val, \"epoch_source\": \"$epoch_src\"}"
                stunir_canon_echo "$json_str" > "$out_json"
            fi
            if [[ -n "$out_header" ]]; then
                echo "#define STUNIR_EPOCH $epoch_val" > "$out_header"
            fi
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
