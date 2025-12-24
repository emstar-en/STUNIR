$ErrorActionPreference = "Stop"

# Preset: build_smt_z3
# Targets: smt_z3

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$helper = Resolve-Path (Join-Path $here "..")

& (Join-Path $helper "build_targets.ps1") "smt_z3" --require-deps
