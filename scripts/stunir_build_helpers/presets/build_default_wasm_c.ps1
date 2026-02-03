$ErrorActionPreference = "Stop"

# Preset: build_default_wasm_c
# Targets: wasm,c

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$helper = Resolve-Path (Join-Path $here "..")

& (Join-Path $helper "build_targets.ps1") "wasm,c"
