@echo off
cd /d "C:\Users\MSTAR\AppData\Roaming\AbacusAI\Agent Workspaces\stunir\tools\spark"
echo Running spark_extract_main.exe...
bin\spark_extract_main.exe -i test_data\golden_test.ads -o test_data\golden_test_extraction.json --lang spark
echo Exit code: %ERRORLEVEL%
echo.
echo Output files:
dir test_data\golden_test_extraction.json* 2>nul
echo.
if exist test_data\golden_test_extraction.json (
    echo === JSON Content ===
    type test_data\golden_test_extraction.json
)
