#!/usr/bin/env bash
# scripts/lib/epoch.sh
# Shell-native implementation of epoch management.

# Load JSON helper if available
if [[ -f "scripts/lib/json_canon.sh" ]]; then
    source "scripts/lib/json_canon.sh"
fi

stunir_shell_epoch() {
    local out_json=""
    local set_epoch=""
    local print_epoch=0

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --out-json) out_json="$2"; shift 2 ;;
            --set-epoch) set_epoch="$2"; shift 2 ;;
            --print-epoch) print_epoch=1; shift ;;
            *) shift ;;
        esac
    done

    # 1. Determine Epoch Value
    local epoch_val=""
    local source=""

    if [[ -n "$set_epoch" ]]; then
        epoch_val="$set_epoch"
        source="ENV_OR_FLAG"
    elif [[ -n "${SOURCE_DATE_EPOCH:-}" ]]; then
        epoch_val="$SOURCE_DATE_EPOCH"
        source="SOURCE_DATE_EPOCH"
    else
        # Fallback to current time
        epoch_val="$(date +%s)"
        source="CURRENT_TIME"
    fi

    # 2. Output JSON
    if [[ -n "$out_json" ]]; then
        # Ensure directory exists
        mkdir -p "$(dirname "$out_json")"

        if type stunir_json_simple_object >/dev/null 2>&1; then
            stunir_json_simple_object "selected_epoch" "$epoch_val" "source" "$source" > "$out_json"
        else
            # Manual fallback if json_canon not loaded
            echo "{"selected_epoch":"$epoch_val","source":"$source"}" > "$out_json"
        fi
    fi

    # 3. Print if requested
    if [[ "$print_epoch" == "1" ]]; then
        echo "$epoch_val"
    fi
}
