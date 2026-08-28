@echo off
rem ForzaSqueegee - 더블클릭 실행기.
rem 창을 띄우면 프로그램이 스스로 관리자 권한을 묻는다 (게임 프로세스를 못 열 때만).
rem 거절해도 창은 뜬다 - 오버레이와 창 조작은 권한이 필요 없다.
cd /d "%~dp0"
title ForzaSqueegee

rem ── 바탕 파이썬 찾기 ──────────────────────────────────────────────
rem "실행해서 3.12~3.14인지"로 판단한다. where python은 못 쓴다: 파이썬이 없는
rem PC에도 스토어 유도용 가짜 python.exe(WindowsApps)가 있어 where가 성공해
rem 버리고, "Add python.exe to PATH"를 안 켜고 설치한 PC는 python은 안 잡혀도
rem py 런처(기본 설치)로는 잡힌다. 상한이 있는 것은 꾸러미 판을 못 박아서다.
set "PY=python"
python -c "import sys; raise SystemExit(0 if (3,12) <= sys.version_info < (3,15) else 1)" >nul 2>&1
if not errorlevel 1 goto :pyok
set "PY=py -3"
py -3 -c "import sys; raise SystemExit(0 if (3,12) <= sys.version_info < (3,15) else 1)" >nul 2>&1
if not errorlevel 1 goto :pyok
echo [ForzaSqueegee] 파이썬 3.12 ~ 3.14 가 필요합니다.
echo   https://www.python.org/downloads/ 에서 설치하세요.
echo   설치할 때 "Add python.exe to PATH"를 꼭 켜세요.
echo.
pause
exit /b 1
:pyok

rem ── 전용 파이썬(.venv) ────────────────────────────────────────────
rem 꾸러미는 **이 폴더 안에만** 깐다. 시스템 파이썬에 깔면 받는 사람이 쓰던
rem numpy 판을 갈아치우게 되고(고정 판이라 반드시 그렇게 된다), 반대로 다른
rem 꾸러미가 충돌하는 판을 물고 있으면 설치가 통째로 실패한다. 폴더를 지우면
rem 흔적도 안 남는다.
set "VPY=%~dp0.venv\Scripts\python.exe"
set "VPYW=%~dp0.venv\Scripts\pythonw.exe"
"%VPY%" -c "import sys" >nul 2>&1
if errorlevel 1 (
  echo [ForzaSqueegee] 이 폴더에 전용 파이썬 환경을 만듭니다 ^(.venv^)
  rmdir /s /q ".venv" >nul 2>&1
  %PY% -m venv ".venv"
  if errorlevel 1 (
    echo   환경을 못 만들었습니다. 위 메시지를 확인하세요.
    pause
    exit /b 1
  )
  "%VPY%" -m pip install --quiet --upgrade pip
)

rem ── 꾸러미 판 대조 ────────────────────────────────────────────────
rem "불러와지나"가 아니라 **판이 맞나**를 본다 - 옛 판이 깔려 있어도 임포트는
rem 다 되므로, 그것으로 판단하면 고정한 판이 한 번도 서지 않는다.
"%VPY%" tools\check_env.py --quiet
if errorlevel 1 (
  echo [ForzaSqueegee] 필요한 꾸러미를 설치해야 합니다.
  "%VPY%" tools\check_env.py
  echo   PySide6 / opencv-python / numpy / Pillow / onnxruntime
  echo   내려받기 약 310MB, 설치 후 약 890MB ^(이 폴더의 .venv 안^)
  choice /c YN /m "지금 설치할까요"
  if errorlevel 2 exit /b 1
  "%VPY%" -m pip install -e .
  if errorlevel 1 (
    echo 설치에 실패했습니다. 위 메시지를 확인하세요.
    pause
    exit /b 1
  )
)

rem 설치 뒤에도 안 맞으면 여기서 **보이게** 실패시킨다 - pythonw로 띄우면
rem 콘솔이 없어 오류가 통째로 묻힌다 (아무 일도 안 일어난 것처럼 보인다).
"%VPY%" tools\check_env.py
if errorlevel 1 (
  echo.
  echo [ForzaSqueegee] 꾸러미 판이 맞지 않습니다. 위 목록을 확인하세요.
  echo   수동 설치: "%VPY%" -m pip install -e .
  pause
  exit /b 1
)

start "" "%VPYW%" -m forzasqueegee gui %*
exit /b 0
