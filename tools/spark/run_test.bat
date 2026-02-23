@echo off
cd /d "C:\Users\MSTAR\AppData\Roaming\AbacusAI\Agent Workspaces\stunir\tools\spark"
echo === Running spark_extract_main.exe ===
echo.
bin\spark_extract_main.exe -i src\core\code_emitter_main.adb -o test_output.json --lang spark 2>&1
echo.
echo === Exit Code: %ERRORLEVEL% ===
echo.
echo === Files created ===
dir test_output* 2>nul
echo.
echo === Output JSON ===
type test_output.json 2>nul || echo NO OUTPUT FILE
echo.
echo === Started marker ===
type test_output.json.started.txt 2>nul || echo NO STARTED FILE
echo.
echo === Error marker ===
type test_output.json.error.txt 2>nul || echo NO ERROR FILE
echo.
echo === OK marker ===
type test_output.json.ok.txt 2>nul || echo NO OK FILE
