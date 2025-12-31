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

    json_init "$LOCKFILE"
    json_obj_start
    json_key_str "version" "1.0"
    json_key_str "type" "toolchain_lock"

    json_key_str "tools" "" # Header for array (hacky but works with our simple builder)
    # Actually, let's do it properly with array support if we had it, 
    # but for now we'll just manually construct the array structure
    # Re-initializing to use manual structure for the array part

    echo "{" > "$LOCKFILE"
    echo '  "version": "1.0",' >> "$LOCKFILE"
    echo '  "type": "toolchain_lock",' >> "$LOCKFILE"
    echo '  "tools": [' >> "$LOCKFILE"

    # Reset helper state for the array
    _JSON_FILE="$LOCKFILE"
    _JSON_FIRST_ITEM=1

    discover_tool "git" "git"
    discover_tool "bash" "bash"
    discover_tool "python" "python3"
    discover_tool "sh" "sh"

    echo "" >> "$LOCKFILE"
    echo "  ]" >> "$LOCKFILE"
    echo "}" >> "$LOCKFILE"

    log_info "Lockfile written to $LOCKFILE"
}
