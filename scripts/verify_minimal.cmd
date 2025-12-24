@echo off
setlocal enabledelayedexpansion

set ROOT=.
if not "%~1"=="" set ROOT=%~1

set ATTEST=%ROOT%oot_attestation.txt
set OBJDIR=%ROOT%\objects\sha256

if not exist "%ATTEST%" (
  echo missing %ATTEST%
  exit /b 1
)
if not exist "%OBJDIR%" (
  echo missing %OBJDIR%
  exit /b 1
)

set FIRSTSEEN=0
set IRCOUNT=0

for /f "usebackq delims=" %%L in ("%ATTEST%") do (
  set LINE=%%L
  if "!LINE!"=="" goto :continue
  if "!LINE:~0,1!"=="#" goto :continue

  if !FIRSTSEEN!==0 (
    if not "!LINE!"=="stunir.pack.root_attestation_text.v0" (
      echo bad version line: !LINE!
      exit /b 1
    )
    set FIRSTSEEN=1
    goto :continue
  )

  for /f "tokens=1,2,3" %%A in ("!LINE!") do (
    set RTYPE=%%A
    set DIGEST=%%B
    set MT=%%C
  )

  if "!RTYPE!"=="epoch" goto :continue

  if "!RTYPE!"=="ir" (
    set /a IRCOUNT+=1
  )

  if "!RTYPE!"=="ir" goto :checkdigest
  if "!RTYPE!"=="input" goto :checkdigest
  if "!RTYPE!"=="receipt" goto :checkdigest
  if "!RTYPE!"=="artifact" goto :checkdigest

  echo unknown record type: !RTYPE!
  exit /b 1

  :checkdigest
  echo !DIGEST! | findstr /r "^sha256:[0-9a-f][0-9a-f]*$" >nul
  if errorlevel 1 (
    echo bad digest: !DIGEST!
    exit /b 1
  )
  set HEX=!DIGEST:~7!
  if not exist "%OBJDIR%\!HEX!" (
    echo missing object: %OBJDIR%\!HEX!
    exit /b 1
  )

  for /f "tokens=1" %%H in ('certutil -hashfile "%OBJDIR%\!HEX!" SHA256 ^| findstr /r "^[0-9A-F][0-9A-F]*$"') do (
    set ACTUAL=%%H
    goto :gotHash
  )
  :gotHash
  set ACTUAL=!ACTUAL!
  set ACTUAL=!ACTUAL: =!
  set ACTUAL=!ACTUAL:~0,64!
  for %%Z in (!ACTUAL!) do set ACTUAL=%%Z
  set ACTUAL=!ACTUAL!
  rem lowercasing is non-trivial in cmd; compare case-insensitive
  if /i not "!ACTUAL!"=="!HEX!" (
    echo hash mismatch for %OBJDIR%\!HEX!
    echo expected !HEX!
    echo actual   !ACTUAL!
    exit /b 1
  )

  :continue
)

if %FIRSTSEEN%==0 (
  echo no version line found
  exit /b 1
)

if not %IRCOUNT%==1 (
  echo expected exactly 1 ir record, got %IRCOUNT%
  exit /b 1
)

echo OK (integrity)
exit /b 0
