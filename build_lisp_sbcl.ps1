$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath "scripts/build.sh")) {
  throw "scripts/build.sh not found. Run this from the STUNIR repo root."
}

# SBCL-backed Common Lisp variant (requires accepted sbcl dependency receipt)
$env:STUNIR_OUTPUT_TARGETS = "lisp_sbcl"
if (-not $env:STUNIR_REQUIRE_DEPS) { $env:STUNIR_REQUIRE_DEPS = "1" }

bash "scripts/build.sh"
