"""관리자 권한으로 자신을 다시 띄운다 — 메모리 주입에 SeDebugPrivilege가 필요하다.

제품은 **실행하자마자** 권한을 묻는다 (사용자 지시). 거절해도 죽지 않는다:
오버레이와 창 조작은 권한이 없어도 되므로 그대로 일반 권한으로 간다.

되풀이 방지는 **인자**로 한다 — 승격된 프로세스는 부모 환경변수를 물려받지
않으므로(AppInfo 서비스가 띄운다) 표시를 환경에 못 남긴다. 다시 띄울 때
`--no-admin`을 앞에 붙이고, 그 인자가 있으면 두 번 묻지 않는다.
"""

from __future__ import annotations

import ctypes
import os
import sys


class _Null:
    """콘솔 없이 뜬 pythonw에서 `print`가 터지지 않게 받아 두는 자리."""

    def write(self, s: str) -> int:
        return len(s)

    def flush(self) -> None:
        pass


def ensure_std_streams() -> None:
    """콘솔이 없거나(`pythonw`) 코드페이지가 좁을 때 `print`가 죽지 않게 한다.

    두 가지가 다 실제로 났다 — `pythonw`는 stdout이 **None**이라 `print` 한 줄에
    죽고, cp949 콘솔로 파이프를 물리면 로그의 `—` 하나에 UnicodeEncodeError가 난다.
    """
    if sys.stdout is None:
        sys.stdout = _Null()
    if sys.stderr is None:
        sys.stderr = _Null()
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except Exception:        # noqa: BLE001 — 못 바꿔도 그냥 간다
                pass


def is_admin() -> bool:
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:            # noqa: BLE001 — 윈도가 아니면 그냥 아니다
        return False


def _quote(a: str) -> str:
    return '"' + a.replace('"', r"\"") + '"'


def relaunch_as_admin() -> bool:
    """UAC를 띄워 같은 인자로 다시 실행. 뜨면 True (부른 쪽은 끝내야 한다)."""
    # `--no-admin`은 **전역 옵션**이라 하위 명령보다 앞이어야 한다
    exe = sys.executable
    args = ["-m", "forzasqueegee", "--no-admin", *sys.argv[1:]]
    try:
        rc = ctypes.windll.shell32.ShellExecuteW(
            None, "runas", exe, " ".join(_quote(a) for a in args), os.getcwd(), 1)
    except Exception:            # noqa: BLE001
        return False
    return int(rc) > 32          # 32 이하는 실패 (5 = 사용자가 거절)


def game_reachable() -> bool | None:
    """지금 권한으로 게임 프로세스를 **열 수 있나**. 게임이 없으면 None.

    주입에 필요한 것은 관리자 권한 자체가 아니라 대상 프로세스를 여는 권한이다.
    게임이 승격 안 된 채로 돌면 같은 사용자·같은 무결성 수준이라 `OpenProcess`가
    그냥 열린다 (2026-08-13 실측: 비승격 셸에서 3,000장 주입). 그러니 **열어
    보고** 정하면 UAC를 괜히 띄우지 않는다.
    """
    try:
        import ctypes as _c

        from .game.inject import PROCESS_ALL, find_pid

        pid = find_pid()
        if not pid:
            return None
        k32 = _c.WinDLL("kernel32", use_last_error=True)
        h = k32.OpenProcess(PROCESS_ALL, False, pid)
        if h:
            k32.CloseHandle(h)
            return True
        return False
    except Exception:            # noqa: BLE001 — 못 재면 모른다고 한다
        return None


def need_admin() -> bool:
    """UAC를 띄워야 하나. **게임을 못 열 때만** True.

    게임이 안 떠 있으면 재 볼 대상이 없으므로 안 띄운다 — 띄워 봐야 그 판단을
    다시 해야 하고, 사용자는 게임을 켠 뒤에 다시 부른다.
    """
    if is_admin():
        return False
    return game_reachable() is False


def ensure_admin() -> bool:
    """관리자면 True. 아니면 UAC를 띄우고, 뜨면 이 프로세스를 끝낸다.

    거절하면 False를 주고 **그대로 계속 간다** — 주입만 못 쓴다."""
    if is_admin():
        return True
    if relaunch_as_admin():
        sys.exit(0)
    return False
