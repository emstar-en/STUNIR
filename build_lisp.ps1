$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath "scripts/build.sh")) {
  throw "scripts/build.sh not found. Run this from the STUNIR repo root."
}

# Portable Common Lisp emission (no runtime required)
$env:STUNIR_OUTPUT_TARGETS = "lisp"

bash "scripts/build.sh"
