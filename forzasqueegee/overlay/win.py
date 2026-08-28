"""Win32 창 유틸: FH6 클라이언트 rect 조회, 오버레이 클릭 통과 설정."""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wt

user32 = ctypes.windll.user32

GWL_EXSTYLE = -20
WS_EX_TRANSPARENT = 0x20
WS_EX_LAYERED = 0x80000
WS_EX_NOACTIVATE = 0x8000000
WS_EX_TOOLWINDOW = 0x80

FH6_TITLE = "Forza Horizon 6"


def set_dpi_aware() -> None:
    try:
        user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except Exception:
        pass


def find_window_client_rect(title: str = FH6_TITLE) -> tuple[int, int, int, int] | None:
    """제목이 일치하는 보이는 창의 클라이언트 rect (화면좌표 x, y, w, h). 없으면 None.

    창 위치/크기는 수시로 바뀔 수 있으므로 매번 재조회할 것.
    """
    found: list[int] = []

    @ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
    def enum_cb(hwnd, _):
        buf = ctypes.create_unicode_buffer(256)
        user32.GetWindowTextW(hwnd, buf, 256)
        if user32.IsWindowVisible(hwnd) and buf.value == title:
            found.append(hwnd)
        return True

    user32.EnumWindows(enum_cb, 0)
    if not found:
        return None
    hwnd = found[0]
    if user32.IsIconic(hwnd):  # 최소화 상태
        return None
    r = wt.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(r))
    pt = wt.POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(pt))
    if r.right <= 0 or r.bottom <= 0:
        return None
    return pt.x, pt.y, r.right, r.bottom


def make_click_through(hwnd: int) -> None:
    """오버레이 창을 클릭 통과 + 비활성(포커스 안 뺏음)으로."""
    ex = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
    user32.SetWindowLongW(
        hwnd, GWL_EXSTYLE,
        ex | WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW)
