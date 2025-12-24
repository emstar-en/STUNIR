param(
  [string]$Root = ".",
  [string]$Attestation = "root_attestation.txt",
  [string]$PubKey = ""
)

$attestPath = Join-Path $Root $Attestation
$objDir = Join-Path $Root "objects/sha256"

if (!(Test-Path -LiteralPath $attestPath)) {
  Write-Error "missing $attestPath"; exit 1
}
if (!(Test-Path -LiteralPath $objDir)) {
  Write-Error "missing $objDir"; exit 1
}

function Is-Sha256Digest([string]$d) {
  return $d -match '^sha256:[0-9a-f]{64}$'
}

$lines = Get-Content -LiteralPath $attestPath
$firstSeen = $false
$irCount = 0

foreach ($raw in $lines) {
  $line = $raw.TrimEnd("`r")
  if ($line.Length -eq 0) { continue }
  if ($line.StartsWith("#")) { continue }

  if (-not $firstSeen) {
    if ($line -ne "stunir.pack.root_attestation_text.v0") {
      Write-Error "bad version line: $line"; exit 1
    }
    $firstSeen = $true
    continue
  }

  $tokens = $line -split '\s+'
  $rtype = $tokens[0]

  if ($rtype -eq "epoch") { continue }

  if (@("ir","input","receipt","artifact") -contains $rtype) {
    if ($tokens.Length -lt 3) { Write-Error "malformed line: $line"; exit 1 }
    $digest = $tokens[1]
    if (-not (Is-Sha256Digest $digest)) { Write-Error "bad digest: $digest"; exit 1 }
    $hex = $digest.Substring(7)
    $objPath = Join-Path $objDir $hex
    if (!(Test-Path -LiteralPath $objPath)) { Write-Error "missing object: $objPath"; exit 1 }
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $objPath).Hash.ToLowerInvariant()
    if ($actual -ne $hex) {
      Write-Error "hash mismatch for $objPath`nexpected $hex`nactual   $actual"; exit 1
    }
    if ($rtype -eq "ir") { $irCount += 1 }
    continue
  }

  Write-Error "unknown record type: $rtype"; exit 1
}

if (-not $firstSeen) { Write-Error "no version line found"; exit 1 }
if ($irCount -ne 1) { Write-Error "expected exactly 1 ir record, got $irCount"; exit 1 }

$sigPath = Join-Path $Root "root_attestation.txt.sig"
if ((Test-Path -LiteralPath $sigPath) -and ($PubKey -ne "")) {
  $openssl = Get-Command "openssl" -ErrorAction SilentlyContinue
  if ($null -ne $openssl) {
    & openssl dgst -sha256 -verify $PubKey -signature $sigPath $attestPath | Out-Null
    if ($LASTEXITCODE -ne 0) { Write-Error "signature verification failed"; exit 1 }
    Write-Output "OK (integrity + signature)"; exit 0
  } else {
    Write-Warning "signature present but openssl not available; integrity OK"
  }
}

Write-Output "OK (integrity)"
