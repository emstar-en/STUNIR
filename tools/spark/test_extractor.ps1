# Test script for spark_extract_main.exe
$ErrorActionPreference = "Continue"

$sparkDir = "C:\Users\MSTAR\AppData\Roaming\AbacusAI\Agent Workspaces\stunir\tools\spark"
$exe = Join-Path $sparkDir "bin\spark_extract_main.exe"
$inputFile = Join-Path $sparkDir "src\core\code_emitter_main.adb"
$outputFile = Join-Path $sparkDir "test_output.json"

Write-Host "=== SPARK Extractor Test ===" -ForegroundColor Cyan
Write-Host "Exe: $exe"
Write-Host "Input: $inputFile"
Write-Host "Output: $outputFile"

# Check exe exists
if (-not (Test-Path $exe)) {
    Write-Host "ERROR: Exe not found!" -ForegroundColor Red
    exit 1
}
Write-Host "Exe exists: YES" -ForegroundColor Green

# Check input exists
if (-not (Test-Path $inputFile)) {
    Write-Host "ERROR: Input file not found!" -ForegroundColor Red
    exit 1
}
Write-Host "Input exists: YES" -ForegroundColor Green

# Remove old output files
Remove-Item -Path "$outputFile*" -Force -ErrorAction SilentlyContinue

# Run extractor
Write-Host "`n=== Running Extractor ===" -ForegroundColor Cyan
$process = Start-Process -FilePath $exe `
    -ArgumentList "-i", $inputFile, "-o", $outputFile, "--lang", "spark" `
    -NoNewWindow -Wait -PassThru `
    -RedirectStandardOutput (Join-Path $sparkDir "test_stdout.txt") `
    -RedirectStandardError (Join-Path $sparkDir "test_stderr.txt")

Write-Host "Exit Code: $($process.ExitCode)"

# Show stderr
$stderrFile = Join-Path $sparkDir "test_stderr.txt"
if (Test-Path $stderrFile) {
    $stderr = Get-Content $stderrFile -Raw
    Write-Host "`n=== STDERR ===" -ForegroundColor Yellow
    Write-Host $stderr
}

# Show stdout
$stdoutFile = Join-Path $sparkDir "test_stdout.txt"
if (Test-Path $stdoutFile) {
    $stdout = Get-Content $stdoutFile -Raw
    Write-Host "`n=== STDOUT ===" -ForegroundColor Yellow
    Write-Host $stdout
}

# Check output files
Write-Host "`n=== Output Files ===" -ForegroundColor Cyan
Get-ChildItem -Path $sparkDir -Filter "test_output*" | ForEach-Object {
    Write-Host "$($_.Name) - $($_.Length) bytes"
}

# Show output JSON if exists
if (Test-Path $outputFile) {
    Write-Host "`n=== Output JSON ===" -ForegroundColor Green
    Get-Content $outputFile
} else {
    Write-Host "`n=== Output JSON NOT FOUND ===" -ForegroundColor Red
}

# Check for marker files
$startedFile = "$outputFile.started.txt"
$errorFile = "$outputFile.error.txt"
$okFile = "$outputFile.ok.txt"

Write-Host "`n=== Marker Files ===" -ForegroundColor Cyan
if (Test-Path $startedFile) {
    Write-Host "STARTED: YES" -ForegroundColor Green
    Get-Content $startedFile
} else {
    Write-Host "STARTED: NO" -ForegroundColor Red
}

if (Test-Path $errorFile) {
    Write-Host "ERROR: YES" -ForegroundColor Red
    Get-Content $errorFile
} else {
    Write-Host "ERROR: NO" -ForegroundColor Green
}

if (Test-Path $okFile) {
    Write-Host "OK: YES" -ForegroundColor Green
    Get-Content $okFile
} else {
    Write-Host "OK: NO" -ForegroundColor Yellow
}
