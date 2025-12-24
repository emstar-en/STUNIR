$ErrorActionPreference = "Stop"

# Preset: build_python_cpython
# Targets: python_cpython

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$helper = Resolve-Path (Join-Path $here "..")

& (Join-Path $helper "build_targets.ps1") "python_cpython" --require-deps
