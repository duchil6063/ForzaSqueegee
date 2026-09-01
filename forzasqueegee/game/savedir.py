"""게임 **저장 컨테이너 뿌리** — 넣으면 게임 그리드에 뜨는 그 자리.

`LayerGroup_<이름>/` 폴더를 여기에 놓으면 게임이 '내 비닐 그룹'에 그대로
띄운다 (창을 한 번도 안 건드리는 파일 노선의 종착지다). 설치 폴더
(`carfiles` — `media/Cars`가 있는 곳)와는 **다른 자리**다: 게임이 D: 스팀
라이브러리에 있어도 저장은 Xbox 앱이 관리하는

    <드라이브>:/XboxGames/GameSave/pgs/<계정>/current/ContainersRoot

에 산다. 그래서 찾는 법도 따로다.

찾는 순서는 설치 폴더와 같은 규칙이다 — **못 박은 것이 이기고 자동 탐색이
맨 뒤다**. 자동 탐색은 두 갈래를 본다: FLS 편집기가 적어 둔 자리(QSettings
`import/*Directory`, HKCU 레지스트리)와 드라이브 훑기.

`ContainersRoot`라는 **이름이 판정자**다 (`media/Cars`가 설치 폴더의
판정자이듯). 옆 칸을 고르기 쉬운 자리라 한두 칸 안쪽까지는 되짚어 준다.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from ..i18n import msg
from .carfiles import read_settings, write_setting

ENV_DIR = "FS_FH6_SAVE_DIR"
ROOT_NAME = "ContainersRoot"
_MARKS = ("layergroup_", "livery_", "baselivery_")   # 컨테이너 뿌리에 사는 것들
_OVERRIDE: Path | None = None          # 이번 실행만


def _root_of(p: str | Path | None) -> Path | None:
    """이 경로가 컨테이너 뿌리인가 — 아니면 None.

    폴더 고르기는 `pgs/<계정>`이나 `current`에서 멈추기 쉬우므로 그 안쪽까지
    짚어 본다 (사람이 저장 자리를 고른 것은 맞다).
    """
    if not p:
        return None
    q = Path(p).expanduser()
    for c in (q, q / ROOT_NAME, q / "current" / ROOT_NAME):
        try:
            if c.is_dir() and c.name.lower() == ROOT_NAME.lower():
                return c
        except OSError:
            continue
    return None


def _fls_dirs() -> list[Path]:
    """FLS 편집기가 적어 둔 자리들 (QSettings = HKCU 레지스트리).

    편집기로 한 번이라도 내보냈으면 여기에 남는다 — 드라이브를 훑기 전에
    본다 (사람이 실제로 쓴 자리라 훑기보다 정확하다).
    """
    try:
        import winreg
    except ImportError:                                   # 윈도가 아니다
        return []
    out: list[Path] = []
    try:
        with winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\ForzaTools\ForzaLiveryStudio\import") as k:
            for name in ("itashaGroupDirectory", "itashaExportDirectory",
                         "exportFolderDirectory", "exportNestedDirectory"):
                try:
                    got = winreg.QueryValueEx(k, name)[0]
                except OSError:
                    continue
                if got:
                    out.append(Path(str(got)))
    except OSError:
        return []
    return out


def _drives() -> list[Path]:
    """훑을 드라이브들."""
    try:
        return [Path(d) for d in os.listdrives()]         # 3.12+ · 윈도
    except (AttributeError, OSError):
        return [Path(f"{c}:\\") for c in "CDEFGHIJKLMNOPQRSTUVWXYZ"]


def _has_marks(p: Path) -> int:
    """이 폴더에 저장 컨테이너가 실제로 사는가 (1/0) — 후보 줄 세우기용."""
    try:
        with os.scandir(p) as it:
            for e in it:
                if e.name.lower().startswith(_MARKS):
                    return 1
    except OSError:
        pass
    return 0


def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


@lru_cache(maxsize=1)
def scan() -> tuple[Path, ...]:
    """드라이브 훑기 — 컨테이너 뿌리 후보들, **쓸 만한 것이 앞**이다.

    `current`는 재파스 지점(정션)이라 훑기가 못 따라갈 수 있어 번호 폴더
    (`pgs/<계정>/52/ContainersRoot`)도 같이 본다. 둘은 같은 자리를 가리키므로
    **`current` 쪽 이름을 남긴다** — 게임이 판을 갈아도 그 이름은 안 바뀐다.
    """
    best: dict[str, Path] = {}
    for d in _drives():
        pgs = d / "XboxGames" / "GameSave" / "pgs"
        try:
            if not pgs.is_dir():
                continue
            cands = [*pgs.glob(f"*/current/{ROOT_NAME}"),
                     *pgs.glob(f"*/*/{ROOT_NAME}")]
        except OSError:
            continue
        for c in cands:
            if not c.is_dir():
                continue
            try:
                key = str(c.resolve()).lower()
            except OSError:
                key = str(c).lower()
            old = best.get(key)
            if old is None or (c.parent.name.lower() == "current"
                               and old.parent.name.lower() != "current"):
                best[key] = c
    return tuple(sorted(best.values(),
                        key=lambda p: (_has_marks(p), _mtime(p)), reverse=True))


def saved_dir() -> Path | None:
    """저장 파일에 적힌 폴더 (검사 없이 적힌 그대로). 없으면 None."""
    got = read_settings().get("save_dir")
    return Path(got) if got else None


def resolve() -> tuple[Path | None, str]:
    """컨테이너 뿌리 + **어디서 나온 자리인가** (사람에게 그대로 보여 줄 한 줄).

    못 박은 것이 이긴다: 이번 실행 → 환경변수 `FS_FH6_SAVE_DIR` → 저장해 둔
    폴더 → FLS 편집기가 적어 둔 자리 → 드라이브 훑기.
    """
    bad = ""
    for src, cand in ((msg("이번 실행"), _OVERRIDE),
                      (ENV_DIR, os.environ.get(ENV_DIR)),
                      (msg("저장해 둔 폴더"), saved_dir())):
        if not cand:
            continue
        root = _root_of(cand)
        if root is not None:
            return root, src + bad
        # 못 박았는데 그 자리가 아니면 **말은 해 준다** (carfiles와 같은 규칙).
        bad = msg(" ({src} `{cand}`에 {name}이 없다)",
                  src=src, cand=cand, name=ROOT_NAME)
    for cand in _fls_dirs():
        root = _root_of(cand)
        if root is not None:
            return root, msg("FLS 편집기가 적어 둔 자리") + bad
    got = scan()
    if got:
        return got[0], msg("드라이브 자동 탐색") + bad
    return None, msg("못 찾았다") + bad


def save_root() -> Path | None:
    """게임 저장 컨테이너 뿌리. 못 찾으면 None."""
    return resolve()[0]


def use_dir(path: str | Path | None) -> Path | None:
    """이번 실행만 이 자리를 쓴다 — 저장하지 않는다."""
    global _OVERRIDE
    root = None if path is None else _root_of(path)
    if path is not None and root is None:
        raise ValueError(msg("게임 저장 컨테이너 뿌리가 아니다 ({name}이 없다) — {path}",
                             name=ROOT_NAME, path=path))
    _OVERRIDE = root
    return root


def set_save_dir(path: str | Path | None) -> Path | None:
    """컨테이너 뿌리를 못 박아 **저장한다**. `None`이면 지운다 (자동 탐색으로)."""
    if path is None:
        write_setting("save_dir", None)
        return None
    root = _root_of(path)
    if root is None:
        raise ValueError(msg("게임 저장 컨테이너 뿌리가 아니다 ({name}이 없다) — {path}",
                             name=ROOT_NAME, path=path))
    write_setting("save_dir", str(root))
    return root
