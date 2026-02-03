#!/bin/bash
# STUNIR DMZ Production Build
# Runs the build in strict mode for production release.

set -e
echo "🔨 DMZ BUILD: Initializing strict build..."

# Ensure toolchain is locked
if [ ! -f "local_toolchain.lock.json" ]; then
    echo "❌ Error: Toolchain lockfile missing!"
    exit 1
fi

# Run the standard build
./scripts/build.sh

echo "✅ DMZ BUILD: Artifacts generated successfully."
exit 0
