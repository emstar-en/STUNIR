#!/usr/bin/env bash
# scripts/lib/load_toolchain.sh
# Parses build/local_toolchain.lock.json and exports STUNIR_TOOL_<NAME> variables.

LOCKFILE="${1:-build/local_toolchain.lock.json}"

if [[ ! -f "$LOCKFILE" ]]; then
    echo "Warning: Toolchain lockfile not found at $LOCKFILE. Skipping injection." >&2
    return 0
fi

# Parse JSON using grep/sed (Shell-Native compatible for the specific format of discover_toolchain.sh)
# We expect: "toolname": { ... "path": "..." ... }

current_tool=""

while read -r line; do
    # Clean whitespace
    line=$(echo "$line" | sed 's/^[ 	]*//;s/[ 	]*$//')

    # Match tool start: "name": {
    if [[ "$line" =~ ^"([a-zA-Z0-9_-]+)":\ \{$ ]]; then
        # Extract name, remove quotes and colon-brace
        current_tool=$(echo "$line" | sed 's/^"//;s/": {$//')
    fi

    # Match path: "path": "..."
    if [[ -n "$current_tool" && "$line" =~ ^"path":\ "(.*)",$ ]]; then
        tool_path=$(echo "$line" | sed 's/^"path": "//;s/",$//')

        # Convert to uppercase for variable name
        var_name="STUNIR_TOOL_$(echo "$current_tool" | tr '[:lower:]' '[:upper:]' | tr '-' '_')"

        # Export
        export "$var_name"="$tool_path"

        # Reset
        current_tool=""
    fi
done < "$LOCKFILE"

# Aliases
if [[ -n "${STUNIR_TOOL_PYTHON3:-}" && -z "${STUNIR_TOOL_PYTHON:-}" ]]; then
    export STUNIR_TOOL_PYTHON="$STUNIR_TOOL_PYTHON3"
fi
