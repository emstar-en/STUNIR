#!/usr/bin/env bash
# STUNIR Polyglot Build Dispatcher
#
# This script serves as the entry point for the STUNIR build pipeline.
# It implements the "Models Propose, Tools Commit" philosophy by dispatching
# to the appropriate runtime (Python, Native Binary, or Shell-Only) based on
# the environment and availability.

set -e

# --- Configuration ---
STUNIR_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STUNIR_PROFILE="${STUNIR_PROFILE:-default}" # Options: default, native, shell
STUNIR_STRICT="${STUNIR_STRICT:-0}"

# --- Colors ---
if [ -t 1 ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    NC='\033[0m' # No Color
else
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    NC=''
fi

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_err()  { echo -e "${RED}[ERROR]${NC} $1"; }
log_ok()   { echo -e "${GREEN}[OK]${NC} $1"; }

# --- Discovery Phase ---

detect_python() {
    if command -v python3 >/dev/null 2>&1; then
        echo "python3"
    elif command -v python >/dev/null 2>&1; then
        echo "python"
    else
        echo ""
    fi
}

detect_native() {
    if [ -x "$STUNIR_ROOT/bin/stunir-cli" ]; then
        echo "$STUNIR_ROOT/bin/stunir-cli"
    else
        echo ""
    fi
}

# --- Dispatch Logic ---

log_info "Initializing STUNIR Build Pipeline..."
log_info "Root: $STUNIR_ROOT"
log_info "Profile: $STUNIR_PROFILE"

# 1. Check for Native Binary (Profile 2 - Pre-Compiled)
# Priority: Highest (if explicitly requested or available)
NATIVE_BIN=$(detect_native)
if [ "$STUNIR_PROFILE" = "native" ]; then
    if [ -n "$NATIVE_BIN" ]; then
        log_info "Dispatching to Native Core ($NATIVE_BIN)..."
        exec "$NATIVE_BIN" build "$@"
    else
        log_err "Profile 'native' requested but 'bin/stunir-cli' not found."
        exit 1
    fi
fi

# 2. Check for Python (Profile 1 - Standard)
# Priority: High (unless Shell-Only requested)
PYTHON_BIN=$(detect_python)
if [ "$STUNIR_PROFILE" != "shell" ] && [ -n "$PYTHON_BIN" ]; then
    log_info "Python detected: $PYTHON_BIN"

    # Ensure PYTHONPATH includes the repo root
    export PYTHONPATH="$STUNIR_ROOT:$PYTHONPATH"

    # Check if the python module exists
    if [ -d "$STUNIR_ROOT/stunir" ]; then
         log_info "Dispatching to Python Toolchain..."
         exec "$PYTHON_BIN" -m stunir.main build "$@"
    else
         log_warn "Python detected but 'stunir' module not found. Falling back to shell..."
         # Force fallback to shell mode
         STUNIR_PROFILE="shell"
    fi
fi

# 3. Fallback to Shell-Native (Profile 3 - Minimal/Verification)
# Priority: Fallback (or explicit request)
if [ "$STUNIR_PROFILE" = "shell" ] || [ -z "$PYTHON_BIN" ]; then
    if [ -z "$PYTHON_BIN" ]; then
        log_warn "No Python detected. Falling back to Shell-Native mode."
    else
        log_info "Shell-Only profile active."
    fi

    log_info "Dispatching to Shell-Native Library..."

    # Check for Shell Library
    SHELL_LIB="$STUNIR_ROOT/scripts/lib/core.sh"
    if [ ! -f "$SHELL_LIB" ]; then
        log_err "Shell library not found at $SHELL_LIB"
        log_err "Cannot proceed in Shell-Only mode."
        exit 1
    fi

    # Source the shell core and run
    exec "$STUNIR_ROOT/scripts/lib/runner.sh" "$@"
fi

log_err "Unexpected state: No suitable runtime found."
exit 1
