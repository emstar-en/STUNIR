#!/usr/bin/env bash
# scripts/lib/toolchain.sh
# Sets up the STUNIR toolchain environment variables.

# Default to python3 if not set
export STUNIR_TOOL_PYTHON="${STUNIR_TOOL_PYTHON:-python3}"

# Ensure tools/ directory is in python path if needed (though scripts usually call by path)
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

# Mock function for tool discovery if needed
stunir_tool_discovery() {
    echo "Toolchain discovery..."
}
