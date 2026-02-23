@echo off
setlocal enabledelayedexpansion

cd /d "C:\Users\MSTAR\AppData\Roaming\AbacusAI\Agent Workspaces\stunir\tools\spark"

echo === Golden Test for SPARK Extractor ===
echo.

set EXE=bin\spark_extract_main.exe
set INPUT=test_data\golden_test.ads
set OUTPUT=test_data\golden_test_extraction.json

echo Exe: %EXE%
echo Input: %INPUT%
echo Output: %OUTPUT%
echo.

if not exist "%EXE%" (
    echo ERROR: Exe not found!
    exit /b 1
)

if not exist "%INPUT%" (
    echo ERROR: Input not found!
    exit /b 1
)

echo === Running Extractor ===
"%EXE%" -i "%INPUT%" -o "%OUTPUT%" --lang spark
set EXITCODE=%ERRORLEVEL%
echo Exit Code: %EXITCODE%
echo.

echo === Checking Output ===
if exist "%OUTPUT%" (
    echo Output file created:
    type "%OUTPUT%"
) else (
    echo Output file NOT created!
)
echo.

echo === Checking Marker Files ===
if exist "%OUTPUT%.started.txt" (
    echo STARTED marker:
    type "%OUTPUT%.started.txt"
) else (
    echo STARTED marker NOT found
)

if exist "%OUTPUT%.ok.txt" (
    echo OK marker:
    type "%OUTPUT%.ok.txt"
) else (
    echo OK marker NOT found
)

if exist "%OUTPUT%.error.txt" (
    echo ERROR marker:
    type "%OUTPUT%.error.txt"
) else (
    echo ERROR marker NOT found
)

echo.
echo === Test Complete ===
