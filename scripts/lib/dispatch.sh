#!/usr/bin/env bash

# Load Shell Libraries
for lib in scripts/lib/*.sh; do
    name=$(basename "$lib")
    if [[ "$name" != "dispatch.sh" && "$name" != "toolchain.sh" ]]; then
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
        elif [[ "$cmd" == "spec_to_ir" ]]; then
            # Adapter for spec-to-ir
            local in_json=""
            local out_ir=""
            while [[ $# -gt 0 ]]; do
                case "$1" in
                    --epoch-json|--in-json) in_json="$2"; shift 2 ;;
                    --out|--out-ir) out_ir="$2"; shift 2 ;;
                    *) shift ;;
                esac
            done
            ./build/stunir_native spec-to-ir --in-json "$in_json" --out-ir "$out_ir"
            return $?
        elif [[ "$cmd" == "gen_provenance" ]]; then
            # Adapter for gen-provenance
            local in_ir=""
            local out_prov=""
            local epoch_json="build/epoch.json"
            while [[ $# -gt 0 ]]; do
                case "$1" in
                    --in-ir) in_ir="$2"; shift 2 ;;
                    --out-prov|--out-json) out_prov="$2"; shift 2 ;;
                    --epoch-json) epoch_json="$2"; shift 2 ;;
                    *) shift ;;
                esac
            done
            ./build/stunir_native gen-provenance --in-ir "$in_ir" --epoch-json "$epoch_json" --out-prov "$out_prov"
            return $?
        elif [[ "$cmd" == "check_toolchain" ]]; then
            # Adapter for check-toolchain
            local lockfile=""
            while [[ $# -gt 0 ]]; do
                case "$1" in
                    --lockfile) lockfile="$2"; shift 2 ;;
                    *) shift ;;
                esac
            done
            ./build/stunir_native check-toolchain --lockfile "$lockfile"
            return $?
        fi
    fi

    # PRIORITY 2: Shell Function (Fallback)
    if [[ "$cmd" == "import_code" || "$cmd" == "compile_provenance" || "$cmd" == "receipt" ]]; then
        local sh_func="stunir_shell_${cmd}"
        if type "$sh_func" &>/dev/null; then
            "$sh_func" "$@"
            return $?
        fi
    fi

    # PRIORITY 3: Python Tool (Fallback)
    local py_tool="tools/${cmd}.py"
    if [[ -f "$py_tool" ]]; then
        python3 "$py_tool" "$@"
        return $?
    fi

    echo "ERROR: No handler found for $cmd" >&2
    exit 127
}
