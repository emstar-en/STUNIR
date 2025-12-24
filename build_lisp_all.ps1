$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath "scripts/build.sh")) {
  throw "scripts/build.sh not found. Run this from the STUNIR repo root."
}

# Emit both portable Lisp and SBCL-backed Lisp.
# If you want to allow SBCL to be missing (skip lisp_sbcl), set STUNIR_REQUIRE_DEPS=0 before running.
$env:STUNIR_OUTPUT_TARGETS = "lisp,lisp_sbcl"
if (-not $env:STUNIR_REQUIRE_DEPS) { $env:STUNIR_REQUIRE_DEPS = "1" }

bash "scripts/build.sh"
