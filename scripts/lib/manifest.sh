#!/bin/sh
# STUNIR Toolchain Discovery & Locking
# Generates local_toolchain.lock.json

. "$(dirname "$0")/core.sh"
. "$(dirname "$0")/json.sh"

LOCKFILE="local_toolchain.lock.json"

discover_tool() {
    name="$1"
    binary="$2"

    path=$(command -v "$binary" 2>/dev/null)
    if [ -n "$path" ]; then
        hash=$(calc_sha256 "$path")
        log_info "Found $name: $path ($hash)"

        json_obj_start
        json_key_str "name" "$name"
        json_key_str "path" "$path"
        json_key_str "sha256" "$hash"
        json_key_str "status" "OK"
        json_obj_end
    else
        log_warn "Tool not found: $name"
        json_obj_start
        json_key_str "name" "$name"
        json_key_str "status" "MISSING"
        json_obj_end
    fi
}

generate_lockfile() {
    log_info "Generating toolchain lockfile..."

    # Initialize variable
    json_init "$LOCKFILE"

    # Write Header manually
    echo "{" > "$LOCKFILE"
    echo '  "version": "1.0",' >> "$LOCKFILE"
    echo '  "type": "toolchain_lock",' >> "$LOCKFILE"
    printf '  "tools": [' >> "$LOCKFILE"

    # Reset helper state for the array items
    _JSON_FIRST_ITEM=1

    discover_tool "git" "git"
    discover_tool "bash" "bash"
    discover_tool "python" "python3"
    discover_tool "sh" "sh"

    # Close Array and Object
    printf "\n  ]\n}\n" >> "$LOCKFILE"

    log_info "Lockfile written to $LOCKFILE"
}

# Execute if run directly
if [ "$(basename "$0")" = "manifest.sh" ]; then
    generate_lockfile
fi
