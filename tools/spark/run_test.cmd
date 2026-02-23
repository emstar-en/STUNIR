@echo off
cd /d "c:\Users\MSTAR\AppData\Roaming\AbacusAI\Agent Workspaces\STUNIR\tools\spark"
powershell -NoProfile -ExecutionPolicy Bypass -File "self_refine_test.ps1"
echo Exit code: %ERRORLEVEL%
