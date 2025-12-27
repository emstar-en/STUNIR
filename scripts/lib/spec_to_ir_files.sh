#!/usr/bin/env bash
# scripts/lib/spec_to_ir_files.sh
# Shell implementation of IR File Splitting.
# NOTE: In Shell Mode (Profile 3), we cannot easily generate dCBOR.
#       Instead, we treat the Canonical JSON as the IR format.
#       This causes divergence from Python Mode (which uses dCBOR).

stunir_shell_spec_to_ir_files() {
    local spec_root="spec"
    local out_root=""
    local manifest_out=""
    local bundle_out=""
    local bundle_manifest_out=""
    local epoch_json=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --spec-root) spec_root="$2"; shift 2 ;;
            --out-root) out_root="$2"; shift 2 ;;
            --manifest-out) manifest_out="$2"; shift 2 ;;
            --bundle-out) bundle_out="$2"; shift 2 ;;
            --bundle-manifest-out) bundle_manifest_out="$2"; shift 2 ;;
            --epoch-json) epoch_json="$2"; shift 2 ;;
            *) shift ;;
        esac
    done

    if [[ -z "$out_root" ]]; then
        echo "Error: --out-root required" >&2
        return 1
    fi

    mkdir -p "$out_root"
    mkdir -p "$(dirname "$manifest_out")"
    mkdir -p "$(dirname "$bundle_out")"

    echo "INFO: Running Shell-Native spec_to_ir_files..."
    echo "WARN: Generating JSON IR (Profile 3), not dCBOR (Profile 1)."

    # 1. Process Files
    # We simply copy/normalize the JSON specs to the IR directory.
    # In a real implementation, we would use json_canon.sh here.

    # Start Manifest JSON
    echo "{" > "$manifest_out"
    echo '  "files": {' >> "$manifest_out"

    local first=1

    # Find and sort files
    find "$spec_root" -type f -name "*.json" | sort | while read -r fpath; do
        local rel_path="${fpath#$spec_root/}"
        if [[ "$rel_path" == "$fpath" ]]; then rel_path="${fpath#$spec_root}"; rel_path="${rel_path#/}"; fi

        # Output filename: replace / with _ to flatten, or keep structure?
        # Python version keeps structure but changes ext to .dcbor
        # We will change ext to .ir.json to indicate it's the IR version
        local out_rel="${rel_path%.json}.ir.json"
        local out_path="$out_root/$out_rel"

        mkdir -p "$(dirname "$out_path")"

        # Copy (and ideally canonicalize)
        # For now, simple copy. Future: pipe through json_canon
        cp "$fpath" "$out_path"

        # Hash
        local sha=""
        if command -v sha256sum >/dev/null 2>&1; then
            sha=$(sha256sum "$out_path" | awk '{print $1}')
        else
            sha=$(shasum -a 256 "$out_path" | awk '{print $1}')
        fi

        # Add to manifest
        if [[ "$first" == "0" ]]; then echo "," >> "$manifest_out"; fi
        first=0
        echo -n "    "$out_rel": "$sha"" >> "$manifest_out"

    done

    echo "" >> "$manifest_out"
    echo "  }," >> "$manifest_out"

    # Epoch Info
    local ep_sel="0"
    if [[ -f "$epoch_json" ]]; then
        ep_sel=$(grep -o '"selected_epoch"[[:space:]]*:[[:space:]]*[0-9]*' "$epoch_json" | cut -d: -f2 | tr -d ' "')
    fi
    echo "  "epoch": ${ep_sel:-0}" >> "$manifest_out"
    echo "}" >> "$manifest_out"

    # 2. Bundle (Stub for Shell)
    # Shell concatenation of JSONs is messy. We leave it empty for now.
    touch "$bundle_out"
    echo "{}" > "$bundle_manifest_out"

    echo "Generated IR Manifest: $manifest_out"
}
