r"""FLS(ForzaLiveryStudio) 편집기를 띄운다 — 우리가 구운 파일을 물려서.

KFPS 편집기는 브라우저 안에서 도는 것이라 서버가 필요했지만(`kfpseditor.py`),
FLS는 **네이티브 창**이라 그냥 실행하면 된다. 받는 인자는 딱 둘이다
(`main.cpp openStartupFiles`):

- `*.3so` — 편집기 프로젝트 (그룹이든 리버리든)
- `C_livery` — 리버리 컨테이너 파일 (폴더가 아니라 **파일** 경로여야 한다)

`C_group`은 인자로 못 연다 (FLS가 프로젝트로 안 친다) — 그래서 도안을 열
때는 늘 `.3so`를 굽는다. 게임에 넣을 `C_group` 폴더는 그것대로 따로 쓴다.

## 어디 있나

1. `vendor/fls-editor/ForzaLiveryStudio.exe` (동봉본 — `tools/get_fls.py`가 받는다)
2. 못 박아 저장한 경로 (`work/state/flsdir.json`)
3. 환경변수 `FS_FLS_EXE`
4. 흔한 설치 자리 (Program Files·바탕화면·다운로드)

없으면 `None`이고, 부르는 쪽이 "받으시겠습니까"를 묻는다 — 우리 파일 내보내기
자체는 FLS가 없어도 다 된다 (게임이 읽는 것은 파일이지 편집기가 아니다).
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from .paths import data_root, work_root

EXE_NAME = "ForzaLiveryStudio.exe"
ENV = "FS_FLS_EXE"
VENDOR = data_root() / "vendor" / "fls-editor"
SETTINGS = work_root() / "state" / "flsdir.json"
# FLS가 처음 뜰 때 묻는 `.3so` 파일 연결 — 우리가 띄우는 길에서는 모달이
# 창을 가리기만 한다. "이미 물어봤다"만 적어 둔다 (연결 자체는 안 건드린다).
_REG_KEY = r"Software\ForzaTools\ForzaLiveryStudio\system"
_REG_VALUE = "associationPrompted"
# 게임 설치 폴더 — 차 id가 실린 리버리를 열면 FLS가 **그 차 모델을 자동으로
# 불러오려** 하고, 설정에 폴더가 없으면 창을 띄우기도 전에 모달로 묻는다
# (2026-08-26 실측: 창이 영영 안 뜬 것처럼 보였다). 우리는 그 폴더를 이미
# 아니까 비어 있을 때만 적어 준다 — 사람이 고른 값은 안 건드린다.
_REG_UI = r"Software\ForzaTools\ForzaLiveryStudio\ui\behavior"
_REG_GAME = "gameFolder"
# [Itasha] 메뉴가 부르는 엔진 — 우리 자신이다 (`cli.design.cmd_flsedit`).
# 편집기는 이것을 `QSettings("itasha/command")`로 읽고, 없으면 메뉴가 그 사실을
# 제목에 적고 꺼진 채로 선다. 그래서 **띄울 때마다 적어 준다** — 저장소를 옮기거나
# 파이썬을 바꿔도 다음 실행에서 자리가 맞는다.
_REG_ITASHA = r"Software\ForzaTools\ForzaLiveryStudio\itasha"


def _candidates() -> list[Path]:
    """찾는 순서 — **못 박은 것이 이긴다**. 환경변수(이번 실행) → 저장해 둔
    경로 → 동봉본 → 흔한 설치 자리. 동봉본이 먼저면 직접 빌드한 판을 가리켜도
    검사가 동봉본으로 도는 사고가 난다 (2026-08-26 실측)."""
    out: list[Path] = []
    env = os.environ.get(ENV)
    if env:
        out.append(Path(env))
    saved = saved_path()
    if saved:
        out.append(saved)
    out.append(VENDOR / EXE_NAME)
    home = Path.home()
    for base in (Path(os.environ.get("ProgramFiles", r"C:\Program Files")),
                 Path(os.environ.get("LOCALAPPDATA", str(home))),
                 home / "Desktop", home / "Downloads"):
        out.append(base / "ForzaLiveryStudio" / EXE_NAME)
        out.append(base / "ForzaLiveryStudio" / "Release" / EXE_NAME)
    return out


def saved_path() -> Path | None:
    try:
        raw = json.loads(SETTINGS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    p = raw.get("exe")
    return Path(p) if p else None


def set_path(path: str | Path | None) -> Path | None:
    """FLS 실행 파일을 못 박아 저장한다. `None`이면 저장을 지운다."""
    if path is None:
        SETTINGS.unlink(missing_ok=True)
        return None
    p = Path(path)
    if p.is_dir():
        p = p / EXE_NAME
    if not p.is_file():
        raise ValueError(f"{EXE_NAME}이 아니다 — {path}")
    SETTINGS.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS.write_text(json.dumps({"exe": str(p)}, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    return p


def find_exe() -> Path | None:
    for cand in _candidates():
        try:
            if cand.is_file():
                return cand
        except OSError:
            continue
    return None


def available() -> bool:
    return find_exe() is not None


def _quiet_association() -> None:
    """`.3so` 연결 묻기를 한 번 지나간 것으로 표시 (모달이 창을 가린다)."""
    try:
        import winreg

        with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, _REG_KEY, 0,
                                winreg.KEY_READ | winreg.KEY_WRITE) as k:
            try:
                if winreg.QueryValueEx(k, _REG_VALUE)[0] in ("true", 1, "1"):
                    return
            except OSError:
                pass
            winreg.SetValueEx(k, _REG_VALUE, 0, winreg.REG_SZ, "true")
    except (ImportError, OSError):
        pass


def _seed_game_folder() -> str | None:
    """FLS 설정의 게임 폴더가 비었으면 우리가 아는 자리를 적어 준다.

    이걸 안 하면 차 id가 실린 리버리를 열 때 FLS가 **창을 띄우기 전에** 폴더를
    묻는 모달을 세운다 — 사람 눈에는 안 뜬 것처럼 보인다. 적어 두면 묻지도
    않고 그 차 모델까지 자동으로 올라와 3D 미리보기가 바로 선다."""
    try:
        import winreg

        from .game.carfiles import install_dir

        root = install_dir()
        if root is None:
            return None
        with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, _REG_UI, 0,
                                winreg.KEY_READ | winreg.KEY_WRITE) as k:
            try:
                cur = winreg.QueryValueEx(k, _REG_GAME)[0]
            except OSError:
                cur = ""
            if str(cur).strip():
                return str(cur)             # 사람이 고른 값이 이긴다
            winreg.SetValueEx(k, _REG_GAME, 0, winreg.REG_SZ, str(root))
        return str(root)
    except (ImportError, OSError):
        return None


def engine_command() -> list[str]:
    """[Itasha] 메뉴가 실행할 명령줄.

    `pythonw.exe`로 떠 있으면 **콘솔 없는 판**을 그대로 쓴다 — 엔진이 창을
    띄우지 않아야 편집기 위로 검은 창이 튀지 않는다."""
    import sys

    exe = Path(sys.executable)
    quiet = exe.with_name("pythonw.exe")
    if quiet.is_file():
        exe = quiet
    return [str(exe), "-m", "forzasqueegee", "flsedit"]


def register_engine() -> bool:
    """편집기 설정에 우리 자신을 이타샤 엔진으로 적는다.

    `QSettings`의 윈도 저장 형태 그대로다 — 문자열 목록은 `REG_MULTI_SZ`,
    문자열은 `REG_SZ`. 사람이 손으로 바꿔 둔 명령이 있어도 **덮어쓴다**: 이
    값은 우리가 어디 설치돼 있나를 적는 자리이지 취향이 아니다."""
    try:
        import winreg

        from .paths import data_root

        with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, _REG_ITASHA, 0,
                                winreg.KEY_READ | winreg.KEY_WRITE) as k:
            winreg.SetValueEx(k, "command", 0, winreg.REG_MULTI_SZ,
                              engine_command())
            winreg.SetValueEx(k, "cwd", 0, winreg.REG_SZ, str(data_root()))
        return True
    except (ImportError, OSError):
        return False


def open_file(path: str | Path | None = None, *,
              exe: str | Path | None = None) -> subprocess.Popen:
    """FLS를 그 파일로 띄운다 (`None`이면 **빈 프로젝트**로).

    이미 떠 있어도 **새 창**이다 (FLS는 한 창에 프로젝트 하나다 —
    `docs/MANUAL.md`)."""
    binary = Path(exe) if exe else find_exe()
    if binary is None:
        raise FileNotFoundError(
            f"{EXE_NAME}을 못 찾았다 — vendor/fls-editor/에 두거나 "
            f"`python tools/get_fls.py`로 받으세요")
    p = None
    if path is not None:
        # **절대 경로여야 한다** — FLS는 인자를 제 작업 폴더 기준으로 푼다
        # (상대 경로를 주면 조용히 못 찾고 빈 프로젝트로 뜬다, 2026-08-26 실측)
        p = Path(path).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"열 파일이 없다 — {p}")
    _quiet_association()
    _seed_game_folder()
    register_engine()
    cwd = str(p.parent) if p is not None else str(binary.parent)
    return subprocess.Popen([str(binary)] + ([str(p)] if p is not None else []),
                            cwd=cwd)
