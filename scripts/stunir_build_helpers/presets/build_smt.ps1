$ErrorActionPreference = "Stop"

# Preset: build_smt
# Targets: smt

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$helper = Resolve-Path (Join-Path $here "..")

& (Join-Path $helper "build_targets.ps1") "smt"
