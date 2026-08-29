@echo off
rem ForzaSqueegee - double-click launcher.
rem
rem Everything runs from a private Python in this folder (runtime\) - the
rem system Python, Anaconda, PYTHONPATH, or packages installed elsewhere can
rem never touch it, and nothing has to be installed on the PC beforehand.
rem The runtime is the python.org embeddable build: its ._pth file makes the
rem interpreter ignore PYTHONPATH/PYTHONHOME and the registry entirely.
rem
rem Messages here are ASCII on purpose - this file is parsed in the console
rem codepage, which differs per PC. Korean output comes from the Python side.
setlocal
cd /d "%~dp0"
title ForzaSqueegee

rem Cleared for the legacy .venv path below; the runtime ignores them anyway.
set "PYTHONPATH="
set "PYTHONHOME="
set "PYTHONSTARTUP="

set "RT=%~dp0runtime"
set "RPY=%RT%\python.exe"
set "RPYW=%RT%\pythonw.exe"

rem Pinned runtime - checked by SHA-256 after download. When bumping, update
rem all three lines together (and re-bake the standard 10 images, pyproject).
set "PYVER=3.12.10"
set "PYURL=https://www.python.org/ftp/python/3.12.10/python-3.12.10-embed-amd64.zip"
set "PYSHA=4acbed6dd1c744b0376e3b1cf57ce906f9dc9e95e68824584c8099a63025a3c3"

if not exist "%RPY%" goto :legacy
"%RPY%" -c "import sys" >nul 2>&1
if errorlevel 1 goto :getruntime
goto :deps

:legacy
rem A .venv left by an older version keeps working while it still matches the
rem pins exactly - those users lose nothing. Anyone else gets the runtime.
set "VPY=%~dp0.venv\Scripts\python.exe"
if not exist "%VPY%" goto :getruntime
"%VPY%" tools\check_env.py --quiet >nul 2>&1
if errorlevel 1 goto :getruntime
start "" "%~dp0.venv\Scripts\pythonw.exe" -m forzasqueegee gui %*
exit /b 0

:getruntime
echo [ForzaSqueegee] Setting up the private Python runtime (first run only).
echo   Downloading Python %PYVER% embeddable, about 11 MB...
rem The zip lands inside this folder too - TEMP can be blocked by AV.
set "PYZIP=%~dp0runtime.zip"
curl.exe -fsSL --retry 3 -o "%PYZIP%" "%PYURL%"
if errorlevel 1 (
  echo   Download failed. Check the internet connection and run this again.
  pause
  exit /b 1
)
certutil -hashfile "%PYZIP%" SHA256 | find /i "%PYSHA%" >nul
if errorlevel 1 (
  echo   The downloaded file failed its SHA-256 check. Run this again.
  del "%PYZIP%" >nul 2>&1
  pause
  exit /b 1
)
if exist "%RT%" rmdir /s /q "%RT%"
mkdir "%RT%"
tar.exe -xf "%PYZIP%" -C "%RT%"
if errorlevel 1 (
  echo   Could not unpack the runtime. Run this again.
  rmdir /s /q "%RT%" >nul 2>&1
  pause
  exit /b 1
)
del "%PYZIP%" >nul 2>&1

:deps
rem Fast pin check every launch (metadata only, well under a second). The
rem full setup - pip bootstrap, pinned packages, import check - runs only
rem when something is missing or the wrong version.
"%RPY%" tools\check_env.py --quiet >nul 2>&1
if not errorlevel 1 goto :go
"%RPY%" tools\bootstrap_env.py
if errorlevel 1 (
  echo.
  echo [ForzaSqueegee] Setup did not finish. See the messages above.
  pause
  exit /b 1
)

:go
start "" "%RPYW%" -m forzasqueegee gui %*
exit /b 0
