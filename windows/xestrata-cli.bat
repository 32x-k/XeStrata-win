@echo off
rem SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
rem SPDX-License-Identifier: LGPL-3.0-or-later
rem xestrata-cli.bat: what xestrata-cli.exe does, for a zip without the .NET launchers (tools/package/windows_bundle.py
rem without --launcher): setup.py in app\ with the command line given, run by the bundled Python in python\, on this
rem console.  Double-clicked, it installs a model and starts it; later runs just start it.
setlocal
cd /d "%~dp0"
if not exist "python\python.exe" goto broken
if not exist "app\setup.py" goto broken
if not exist "app\PACKAGED" goto broken
rem the Microsoft Visual C++ runtime the engine and Python's packages need (not in the zip: the user installs it)
for %%D in (vcruntime140.dll vcruntime140_1.dll msvcp140.dll msvcp140_atomic_wait.dll) do (
  if not exist "%SystemRoot%\System32\%%D" set "MISSING=%%D" & goto vcrt
)
rem the bundled Python sees its own packages only, not a Python installed on the PC nor the user's site-packages
set "PYTHONHOME="
set "PYTHONPATH="
set "PYTHONNOUSERSITE=1"
"python\python.exe" "app\setup.py" %*
if errorlevel 1 pause
exit /b %errorlevel%

:broken
echo  This XeStrata folder is incomplete (python\python.exe, app\setup.py or app\PACKAGED is missing).
echo  Unpack the zip again.
pause
exit /b 1

:vcrt
echo  The Microsoft Visual C++ runtime is not installed (%MISSING% is missing).
echo  Install the Visual C++ Redistributable (x64) from https://aka.ms/vs/17/release/vc_redist.x64.exe and run this again.
pause
exit /b 1
