#!/usr/bin/env bash
# scripts/lib/spec_to_ir.sh
# Shell implementation of Spec -> IR Text conversion.
# Dependencies: find, sort, sha256sum (or shasum), grep, sed, awk

stunir_shell_spec_to_ir() {
    local spec_root="spec"
    local out_file=""
    local epoch_json=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --spec-root) spec_root="$2"; shift 2 ;;
            --out) out_file="$2"; shift 2 ;;
            --epoch-json) epoch_json="$2"; shift 2 ;;
            *) shift ;;
        esac
    done

    if [[ -z "$out_file" ]]; then
        echo "Error: --out required for spec_to_ir" >&2
        return 1
    fi

    mkdir -p "$(dirname "$out_file")"

    # 1. Header
    echo "; STUNIR IR SUMMARY v1" > "$out_file"

    # Epoch Info (Simple grep extraction)
    local ep_sel="?"
    local ep_src="?"
    if [[ -f "$epoch_json" ]]; then
        # Try to extract with grep/sed if jq is missing
        # Pattern: "selected_epoch": 12345
        ep_sel=$(grep -o '"selected_epoch"[[:space:]]*:[[:space:]]*[0-9]*' "$epoch_json" | cut -d: -f2 | tr -d ' "')
        # Pattern: "source": "..."
        ep_src=$(grep -o '"source"[[:space:]]*:[[:space:]]*"[^"]*"' "$epoch_json" | cut -d: -f2 | tr -d ' "')
    fi
    echo "; epoch.selected=${ep_sel:-?} source=${ep_src:-?}" >> "$out_file"

    # 2. Process Files
    # Find all .json files in spec_root, sorted
    find "$spec_root" -type f -name "*.json" | sort | while read -r fpath; do
        # Relative path
        # ${fpath#$spec_root/} removes the prefix. Ensure we handle trailing slash if present.
        local rel_path="${fpath#$spec_root/}"
        if [[ "$rel_path" == "$fpath" ]]; then
             # spec_root didn't have trailing slash, try adding one
             rel_path="${fpath#$spec_root}"
             rel_path="${rel_path#/}" # remove leading slash
        fi

        # SHA256
        local sha=""
        if command -v sha256sum >/dev/null 2>&1; then
            sha=$(sha256sum "$fpath" | awk '{print $1}')
        elif command -v shasum >/dev/null 2>&1; then
            sha=$(shasum -a 256 "$fpath" | awk '{print $1}')
        else
            sha="unknown"
        fi

        # Extract ID and Name (Naive grep/sed approach)
        # This is fragile but fits "legacy power tools" without jq.
        # We look for "id": "..." and "name": "..." (or "title")

        local ident="-"
        local name="-"

        # Grep for "id": "value"
        local id_match=$(grep -o '"id"[[:space:]]*:[[:space:]]*"[^"]*"' "$fpath" | head -n1 | cut -d: -f2 | tr -d ' "')
        if [[ -n "$id_match" ]]; then ident="$id_match"; fi

        # Grep for "name": "value"
        local name_match=$(grep -o '"name"[[:space:]]*:[[:space:]]*"[^"]*"' "$fpath" | head -n1 | cut -d: -f2 | tr -d ' "')
        if [[ -z "$name_match" ]]; then
             # Fallback to title
             name_match=$(grep -o '"title"[[:space:]]*:[[:space:]]*"[^"]*"' "$fpath" | head -n1 | cut -d: -f2 | tr -d ' "')
        fi
        if [[ -n "$name_match" ]]; then name="$name_match"; fi

        echo "FILE $rel_path SHA256 $sha ID $ident NAME $name" >> "$out_file"
    done

    echo "Generated IR: $out_file"
}
