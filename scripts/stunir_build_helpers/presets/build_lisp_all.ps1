$ErrorActionPreference = "Stop"

# Preset: build_lisp_all
# Targets: lisp,lisp_sbcl

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$helper = Resolve-Path (Join-Path $here "..")

& (Join-Path $helper "build_targets.ps1") "lisp,lisp_sbcl" --require-deps
