#!/bin/sh
# STUNIR Shell-Native Runner (Profile 3)
# Handles orchestration and verification when no Semantic Engine is available.

set -u

log() { echo "[STUNIR-SHELL] $1"; }

log "Starting Shell-Native Runner..."

# 1. Check Environment
if [ -f "local_toolchain.lock.json" ]; then
    log "Toolchain lockfile found."
else
    log "Warning: No toolchain lockfile found."
fi

# 2. Check for Artifacts
# In Shell Mode, we cannot compile JSON -> Code. We can only manage existing artifacts.
ARTIFACT="asm/output.py"

if [ -f "$ARTIFACT" ]; then
    log "Artifact found: $ARTIFACT"
    log "Verifying artifact integrity..."
    # TODO: Implement hash verification against a manifest
    log "Verification passed (simulated)."
else
    log "ERROR: Artifact '$ARTIFACT' not found."
    log "The Shell-Native profile cannot compile JSON specs into code."
    log "Please use the 'native' or 'python' profile to generate the artifacts first:"
    log "  STUNIR_PROFILE=native ./scripts/build.sh"
    exit 1
fi

log "Shell-Native Run Complete."
