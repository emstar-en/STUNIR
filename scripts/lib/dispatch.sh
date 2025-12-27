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
        if [[ "$cmd" == "epoch" ]]; then
             ./build/stunir_native epoch "$@"
             return $?
        elif [[ "$cmd" == "import_code" ]]; then
             ./build/stunir_native import-code "$@"
             return $?
        elif [[ "$cmd" == "spec_to_ir" ]]; then
             ./build/stunir_native spec-to-ir "$@"
             return $?
        elif [[ "$cmd" == "gen_provenance" ]]; then
             # Pass all args, including ignored ones
             ./build/stunir_native gen-provenance "$@"
             return $?
        elif [[ "$cmd" == "compile_provenance" ]]; then
             ./build/stunir_native compile-provenance "$@"
             return $?
        elif [[ "$cmd" == "validate" || "$cmd" == "verify" ]]; then
            ./build/stunir_native "$cmd" "$@"
            return $?
        fi
    fi

    # PRIORITY 2: Shell/Python Implementation
    case "$cmd" in
        epoch)
            stunir_shell_epoch "$@"
            ;;
        check_toolchain)
            # No-op in shell mode
            ;;
        import_code)
            local input_root=""
            local out_spec=""
            while [[ $# -gt 0 ]]; do
                case "$1" in
                    --input-root) input_root="$2"; shift 2 ;;
                    --out-spec) out_spec="$2"; shift 2 ;;
                    *) shift ;;
                esac
            done
            
            echo "Importing Code from $input_root (Python Fallback)..."
            if command -v python3 >/dev/null 2>&1; then
                python3 tools/import_spec.py --input-root "$input_root" --out-spec "$out_spec"
            else
                echo "ERROR: Python3 required for import_code"
                exit 1
            fi
            ;;
        spec_to_ir)
            local out_ir=""
            local spec_root=""
            while [[ $# -gt 0 ]]; do
                case "$1" in
                    --out) out_ir="$2"; shift 2 ;;
                    --spec-root) spec_root="$2"; shift 2 ;;
                    *) shift ;;
                esac
            done
            echo "Generating IR from $spec_root (Python Fallback)..."
            
            if command -v python3 >/dev/null 2>&1; then
                python3 tools/spec_to_ir.py --spec-root "$spec_root" --out "$out_ir"
            else
                echo "ERROR: Python3 required for spec_to_ir"
                exit 1
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
                    --spec-root) shift 2 ;;
                    --asm-root) shift 2 ;;
                    *) shift ;;
                esac
            done
            echo "Generated Provenance"
            
            if command -v python3 >/dev/null 2>&1; then
                python3 tools/gen_provenance.py \
                    --epoch "$epoch_val" \
                    --epoch-source "$epoch_src" \
                    --out-json "$out_json" \
                    --out-header "$out_header"
            else
                local json_str="{\"build_epoch\": $epoch_val, \"epoch_source\": \"$epoch_src\"}"
                stunir_canon_echo "$json_str" > "$out_json"
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
