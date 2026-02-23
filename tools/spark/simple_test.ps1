# Simple test
$output = "Hello from PowerShell at " + (Get-Date -Format "HH:mm:ss")
$output | Out-File -FilePath "simple_test_output.txt" -Encoding UTF8
Write-Host $output
Write-Host "File written to simple_test_output.txt"
