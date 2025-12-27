#!/usr/bin/env bash
# scripts/lib/dispatch.sh
# The core dispatcher for STUNIR's polyglot architecture.
# It decides whether to call a Python tool or a Shell function.

# Load all shell libraries if they exist
for lib in scripts/lib/*.sh; do
    if [[ "$lib" != "scripts/lib/dispatch.sh" && "$lib" != "scripts/lib/toolchain.sh" ]]; then
        source "$lib"
    fi
done

stunir_dispatch() {
    local command="$1"
    shift

    # 1. Try Python Tool first (Profile 1 & 2)
    # We look for tools/{command}.py
    local py_tool="tools/${command}.py"

    if [[ -f "$py_tool" && -n "${STUNIR_TOOL_PYTHON:-}" ]]; then
        # echo "Dispatching to Python: $py_tool $@" >&2
        "$STUNIR_TOOL_PYTHON" "$py_tool" "$@"
        return $?
    fi

    # 2. Try Shell Function fallback (Profile 3)
    # We look for a function named stunir_shell_{command}
    local sh_func="stunir_shell_${command}"

    if type "$sh_func" &>/dev/null; then
        # echo "Dispatching to Shell: $sh_func $@" >&2
        "$sh_func" "$@"
        return $?
    fi

    # 3. Failure
    echo "ERROR: No handler found for command '$command'" >&2
    echo "  Checked Python: $py_tool" >&2
    echo "  Checked Shell:  $sh_func" >&2
    exit 127
}
