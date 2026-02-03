$ErrorActionPreference = "Stop"

function Show-Usage {
  Write-Error "Usage: build_targets.ps1 <targets_csv> [--require-deps|--allow-missing-deps]"
}

if (-not (Test-Path -LiteralPath "scripts/build.sh")) {
  throw "scripts/build.sh not found. Run this from the STUNIR repo root."
}

if ($args.Count -lt 1) {
  Show-Usage
  exit 2
}

$targets = $args[0]
$flags = @()
if ($args.Count -gt 1) { $flags = $args[1..($args.Count-1)] }

if (-not $env:STUNIR_REQUIRE_DEPS) { $env:STUNIR_REQUIRE_DEPS = "0" }

foreach ($f in $flags) {
  switch ($f) {
    "--require-deps" { $env:STUNIR_REQUIRE_DEPS = "1" }
    "--allow-missing-deps" { $env:STUNIR_REQUIRE_DEPS = "0" }
    "-h" { Show-Usage; exit 0 }
    "--help" { Show-Usage; exit 0 }
    default { throw "Unknown argument: $f" }
  }
}

$env:STUNIR_OUTPUT_TARGETS = $targets

bash "scripts/build.sh"
