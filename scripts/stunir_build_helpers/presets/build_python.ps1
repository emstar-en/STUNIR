$ErrorActionPreference = "Stop"

# Preset: build_python
# Targets: python

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$helper = Resolve-Path (Join-Path $here "..")

& (Join-Path $helper "build_targets.ps1") "python"
