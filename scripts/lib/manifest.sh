#!/usr/bin/env bash
# STUNIR Toolchain Discovery (Shell-Native)

source "$STUNIR_ROOT/scripts/lib/json.sh"

stunir_generate_lockfile() {
    local build_dir="$STUNIR_ROOT/build"
    local lockfile="$build_dir/local_toolchain.lock.json"

    mkdir -p "$build_dir"

    stunir_log "Scanning environment for required tools..."

    # Initialize JSON object
    json_init
    json_start_object
    json_key_val "schema_version" "1.0.0"
    json_key_val "generated_at" "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    json_key_val "host_os" "$(uname -s)"

    json_key_start_array "tools"

    # --- Scan Tools ---

    # 1. Git
    _scan_tool "git" "git" "--version"
    json_add_comma

    # 2. Python (Optional but recommended)
    if command -v python3 >/dev/null; then
        _scan_tool "python" "python3" "--version"
    elif command -v python >/dev/null; then
        _scan_tool "python" "python" "--version"
    else
        _scan_tool_missing "python"
    fi
    json_add_comma

    # 3. Bash (Self)
    _scan_tool "bash" "bash" "--version" | head -n 1

    json_end_array
    json_end_object

    # Write to file
    echo "$JSON_BUFFER" > "$lockfile"
    stunir_ok "Toolchain lockfile generated at: $lockfile"
}

_scan_tool() {
    local name=$1
    local bin=$2
    local ver_arg=$3

    local path=$(command -v "$bin")
    local version=$("$bin" $ver_arg 2>&1 | head -n 1 | tr -d '"')

    # Calculate Hash (Try sha256sum, then shasum)
    local hash=""
    if command -v sha256sum >/dev/null; then
        hash=$(sha256sum "$path" | awk '{print $1}')
    elif command -v shasum >/dev/null; then
        hash=$(shasum -a 256 "$path" | awk '{print $1}')
    else
        hash="unknown_no_hasher"
    fi

    stunir_log "Found $name: $path ($hash)"

    json_start_object
    json_key_val "name" "$name"
    json_key_val "path" "$path"
    json_key_val "version" "$version"
    json_key_val "sha256" "$hash"
    json_key_val "status" "OK"
    json_end_object
}

_scan_tool_missing() {
    local name=$1
    stunir_warn "Tool '$name' not found."

    json_start_object
    json_key_val "name" "$name"
    json_key_val "status" "MISSING"
    json_end_object
}
