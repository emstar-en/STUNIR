@echo off
cd /d "C:\Users\MSTAR\AppData\Roaming\AbacusAI\Agent Workspaces\stunir\tools\spark"

echo === Test 1: No args === > test_results.txt
bin\spark_extract_main.exe >> test_results.txt 2>&1
echo Exit: %ERRORLEVEL% >> test_results.txt

echo. >> test_results.txt
echo === Test 2: Golden test === >> test_results.txt
bin\spark_extract_main.exe -i test_data\golden_test.ads -o test_data\golden_test_extraction.json --lang spark >> test_results.txt 2>&1
echo Exit: %ERRORLEVEL% >> test_results.txt

echo. >> test_results.txt
echo === Files created === >> test_results.txt
dir test_data\golden_test_extraction.json* >> test_results.txt 2>&1

echo. >> test_results.txt
echo === JSON content === >> test_results.txt
type test_data\golden_test_extraction.json >> test_results.txt 2>&1

echo Done. Results in test_results.txt
