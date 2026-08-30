"""FH6 창 캡처 + SendInput 키 입력 (스캔코드).

실측 기반 주의사항:
- 키 드롭 존재 → 모든 조작은 캡처/OCR 검증 폐루프로
- 이미 포그라운드면 Alt 트릭 생략 (Alt 래치로 다음 키 씹힘)
- 창 rect·캡처 크기는 매번 재조회
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import sys
import time

import numpy as np

from ..i18n import msg

user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32

FH6_TITLE = "Forza Horizon 6"

SCANCODES = {
    "esc": 0x01, "1": 0x02, "2": 0x03, "3": 0x04, "4": 0x05, "5": 0x06,
    "backspace": 0x0E, "tab": 0x0F, "q": 0x10, "w": 0x11, "e": 0x12, "r": 0x13,
    "t": 0x14, "y": 0x15, "p": 0x19, "enter": 0x1C,
    "a": 0x1E, "s": 0x1F, "d": 0x20, "x": 0x2D, "m": 0x32, "space": 0x39,
    "up": 0xC8, "down": 0xD0, "left": 0xCB, "right": 0xCD,
    "pgup": 0xC9, "pgdn": 0xD1, "home": 0xC7, "end": 0xCF,
}
EXTENDED = {"up", "down", "left", "right", "pgup", "pgdn", "home", "end"}

_PUL = ctypes.POINTER(ctypes.c_ulong)


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wt.WORD), ("wScan", wt.WORD), ("dwFlags", wt.DWORD),
                ("time", wt.DWORD), ("dwExtraInfo", _PUL)]


class _INPUT(ctypes.Structure):
    class _U(ctypes.Union):
        _fields_ = [("ki", _KEYBDINPUT), ("pad", ctypes.c_byte * 32)]
    _anonymous_ = ("u",)
    _fields_ = [("type", wt.DWORD), ("u", _U)]


def set_dpi_aware() -> None:
    try:
        user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except Exception:
        pass


def find_hwnd(title: str = FH6_TITLE) -> int | None:
    found: list[int] = []

    @ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
    def cb(hwnd, _):
        buf = ctypes.create_unicode_buffer(256)
        user32.GetWindowTextW(hwnd, buf, 256)
        if user32.IsWindowVisible(hwnd) and buf.value == title:
            found.append(hwnd)
        return True

    user32.EnumWindows(cb, 0)
    return found[0] if found else None


def is_foreground(hwnd: int) -> bool:
    return user32.GetForegroundWindow() == hwnd


def _send_scan(scan: int, up: bool, ext: bool) -> None:
    flags = 0x8 | (0x2 if up else 0) | (0x1 if ext else 0)  # SCANCODE|KEYUP|EXTENDED
    inp = _INPUT(type=1)
    inp.ki = _KEYBDINPUT(0, (scan & 0x7F) if ext else scan, flags, 0, None)
    user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(_INPUT))


def press(key: str, hold_s: float = 0.09) -> None:
    """키 1회 탭. 반영 보장은 없음 — 호출측에서 검증 폐루프 필수.

    hold_s 기본 90ms: 40ms는 변형 편집의 Enter(커밋)가 무시되는 것을 실측.
    """
    sc, ext = SCANCODES[key], key in EXTENDED
    _send_scan(sc, False, ext)
    time.sleep(hold_s)
    _send_scan(sc, True, ext)


def press_batch(key: str, n: int, hold_s: float = 0.004, gap_s: float = 0.004) -> None:
    """키 n회 배치 탭 (기본 4ms/4ms).

    실측: 변형 편집 축은 100% 반영. HSB 슬라이더도 전량 반영되나 1회당 내부값
    0.005(표시 스텝의 절반, 표시는 절사) → 호출측 반 스텝 환산 + OCR 검증 필수.

    **더 빨리 보내도 빨라지지 않는다** (56차 실측). 1ms/1ms는 전송 자체는 2.6배
    빠르고 반영률도 1.000이지만, 표시 롤이 그만큼 뒤처져 "전송~검증 완료" 총
    시간이 되레 는다 — 회전 240스텝: 4ms 3.48s vs 1ms 4.88s (120스텝 1.86 vs
    2.52, 60스텝 1.01 vs 1.33). 화살표의 실효 상한은 게임 소비 속도 ~65 step/s다.
    따라서 화살표는 **줄이는 것**이지 빨리 보내는 것이 아니다 (코스 홀드로 최대한
    좁히고 남은 것만 화살표).
    """
    sc, ext = SCANCODES[key], key in EXTENDED
    for _ in range(n):
        _send_scan(sc, False, ext)
        time.sleep(hold_s)
        _send_scan(sc, True, ext)
        time.sleep(gap_s)


def type_text(text: str, gap_s: float = 0.04) -> None:
    """텍스트 입력 대화상자에 유니코드 문자열 입력 (KEYEVENTF_UNICODE).

    저장 파일 이름 등 OS 스타일 텍스트 필드용. 반영 검증은 호출측(캡처)에서.
    """
    for ch in text:
        for up in (False, True):
            flags = 0x0004 | (0x0002 if up else 0)  # UNICODE|KEYUP
            inp = _INPUT(type=1)
            inp.ki = _KEYBDINPUT(0, ord(ch), flags, 0, None)
            user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(_INPUT))
        time.sleep(gap_s)


def hold(key: str, dur_s: float) -> None:
    """키 홀드(연속 변경용). 실제 변화량은 OCR로 확인할 것 (간헐 끊김 실측됨)."""
    sc, ext = SCANCODES[key], key in EXTENDED
    _send_scan(sc, False, ext)
    time.sleep(dur_s)
    _send_scan(sc, True, ext)


class _GUITHREADINFO(ctypes.Structure):
    _fields_ = [("cbSize", ctypes.c_uint32), ("flags", ctypes.c_uint32),
                ("hwndActive", ctypes.c_void_p), ("hwndFocus", ctypes.c_void_p),
                ("hwndCapture", ctypes.c_void_p), ("hwndMenuOwner", ctypes.c_void_p),
                ("hwndMoveSize", ctypes.c_void_p), ("hwndCaret", ctypes.c_void_p),
                ("rcCaret", ctypes.c_long * 4)]


_GUI_MENUMODE = 0x4 | 0x8  # GUI_INMENUMODE | GUI_SYSTEMMENUMODE


def menu_latched() -> bool:
    """포그라운드 창이 시스템 메뉴 모드에 걸려 있는가 (= 키 입력 전량 무시)."""
    gti = _GUITHREADINFO()
    gti.cbSize = ctypes.sizeof(_GUITHREADINFO)
    if not user32.GetGUIThreadInfo(0, ctypes.byref(gti)):
        return False
    return bool(gti.flags & _GUI_MENUMODE)


def unlatch_menu(tries: int = 3) -> bool:
    """Alt 래치 해제. 걸려 있지 않으면 아무것도 안 함.

    42차 실측: Alt가 눌린 채로 남으면(포커스 트릭 잔재·사용자 Alt+Tab) FH6가
    시스템 메뉴 모드로 들어가 **모든 키 입력을 삼킨다**. 화면 캡처는 정상이라
    driver는 "단계 전이 실패"만 반복 — 원인 규명이 어렵다. Esc는 안 듣고
    Alt 탭 1회로 풀린다 (실측).
    """
    for _ in range(tries):
        if not menu_latched():
            return True
        _send_scan(0x38, False, False)  # Alt 탭 = 메뉴 모드 종료
        time.sleep(0.05)
        _send_scan(0x38, True, False)
        time.sleep(0.5)
    return not menu_latched()


def focus(hwnd: int) -> bool:
    """FH6를 포그라운드로. 이미 포그라운드면 Alt 래치만 확인 (트릭은 생략)."""
    if is_foreground(hwnd):
        return unlatch_menu()
    user32.ShowWindow(hwnd, 9)  # SW_RESTORE
    _send_scan(0x38, False, False)  # Alt down: 포그라운드 잠금 해제 트릭
    user32.SetForegroundWindow(hwnd)
    _send_scan(0x38, True, False)
    time.sleep(0.5)
    return is_foreground(hwnd) and unlatch_menu()


_MOUSEEVENTF_LEFTDOWN = 0x02
_MOUSEEVENTF_LEFTUP = 0x04


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wt.LONG), ("dy", wt.LONG), ("mouseData", wt.DWORD),
                ("dwFlags", wt.DWORD), ("time", wt.DWORD), ("dwExtraInfo", _PUL)]


class _MINPUT(ctypes.Structure):
    class _U(ctypes.Union):
        _fields_ = [("mi", _MOUSEINPUT), ("pad", ctypes.c_byte * 32)]
    _anonymous_ = ("u",)
    _fields_ = [("type", wt.DWORD), ("u", _U)]


def _send_button(flags: int) -> None:
    inp = _MINPUT(type=0)
    inp.mi = _MOUSEINPUT(0, 0, 0, flags, 0, None)
    user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(_MINPUT))


def to_screen(hwnd: int, cx: float, cy: float) -> tuple[int, int]:
    pt = wt.POINT(int(cx), int(cy))
    user32.ClientToScreen(hwnd, ctypes.byref(pt))
    return pt.x, pt.y


def click(hwnd: int, cx: float, cy: float) -> None:
    """클라이언트 좌표 (cx,cy) 좌클릭. 반영 보장 없음 — 호출측 검증 필수."""
    x, y = to_screen(hwnd, cx, cy)
    user32.SetCursorPos(x, y)
    time.sleep(0.05)
    _send_button(_MOUSEEVENTF_LEFTDOWN)
    time.sleep(0.06)
    _send_button(_MOUSEEVENTF_LEFTUP)


def drag(hwnd: int, cx1: float, cy1: float, cx2: float, cy2: float,
         steps: int = 12, delay: float = 0.015) -> None:
    """클라이언트 좌표 드래그 (보간 이동)."""
    x1, y1 = to_screen(hwnd, cx1, cy1)
    x2, y2 = to_screen(hwnd, cx2, cy2)
    user32.SetCursorPos(x1, y1)
    time.sleep(0.08)
    _send_button(_MOUSEEVENTF_LEFTDOWN)
    time.sleep(0.08)
    for i in range(1, steps + 1):
        t = i / steps
        user32.SetCursorPos(int(x1 + (x2 - x1) * t), int(y1 + (y2 - y1) * t))
        time.sleep(delay)
    time.sleep(0.08)
    _send_button(_MOUSEEVENTF_LEFTUP)


class _BIH(ctypes.Structure):
    _fields_ = [("biSize", wt.DWORD), ("biWidth", wt.LONG), ("biHeight", wt.LONG),
                ("biPlanes", wt.WORD), ("biBitCount", wt.WORD), ("biCompression", wt.DWORD),
                ("biSizeImage", wt.DWORD), ("biXPelsPerMeter", wt.LONG),
                ("biYPelsPerMeter", wt.LONG), ("biClrUsed", wt.DWORD), ("biClrImportant", wt.DWORD)]


STATS = {"captures": 0}  # 계측용 (bench/진단에서 읽음)


def client_size(hwnd: int) -> tuple[int, int]:
    r = wt.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(r))
    return r.right, r.bottom


_CAP: dict[tuple[int, int, int], tuple[int, int]] = {}


def _mem_target(hwnd: int, w: int, h: int) -> tuple[int, int]:
    """(메모리 DC, 비트맵) 캐시 — 창 크기가 그대로면 재사용.

    캡처 1회당 CreateCompatibleDC/Bitmap + Delete는 수 ms다. 창 rect는 매번
    재조회하되(크기 가변) 크기가 같으면 GDI 객체는 다시 만들지 않는다.
    """
    key = (hwnd, w, h)
    ent = _CAP.get(key)
    if ent is not None:
        return ent
    for k in [k for k in _CAP if k[0] == hwnd]:  # 크기 바뀜 — 옛 객체 해제
        mem, bmp = _CAP.pop(k)
        gdi32.DeleteObject(bmp)
        gdi32.DeleteDC(mem)
    scr = user32.GetDC(0)
    mem = gdi32.CreateCompatibleDC(scr)
    bmp = gdi32.CreateCompatibleBitmap(scr, w, h)
    user32.ReleaseDC(0, scr)
    gdi32.SelectObject(mem, bmp)
    _CAP[key] = (mem, bmp)
    return mem, bmp


# PrintWindow가 **새까만 프레임만** 줄 때가 있다. 원인은 하나로 밝혀졌다
# (2026-08-26 실측): **게임이 우리보다 높은 권한으로 돈다** — 스팀을 관리자로
# 켜면 그렇게 된다. 그때 낮은 쪽에서 부른 PrintWindow는 전 프레임 0을 주고,
# 같은 이유로 **합성 입력도 UIPI에 조용히 막힌다**. 그래서 화면 뜨기로 갈아타
# 그림은 얻더라도 폐루프는 어차피 못 돈다 — 갈아탈 때 그 진단을 한 번 찍는다.
#
# 그래도 갈아타는 것은 캡처만 쓰는 길(진단·계측)이 있어서다. 판정은 전체 프레임
# 한 번으로 하고 그 뒤로는 갈아탄 길만 쓴다 — 매 프레임 PrintWindow를 헛돌리면
# 폐루프가 17ms씩 손해다. 화면 뜨기는 **창이 안 가려져 있어야** 하므로 기본은
# PrintWindow 그대로다.
_SCREEN_GRAB: set[int] = set()


def _blt_screen(hwnd: int, w: int, h: int, y0: int, nh: int) -> np.ndarray:
    """바탕화면 DC에서 창 클라이언트 영역의 [y0, y0+nh) 행을 떠 온다."""
    pt = wt.POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(pt))
    mem, bmp = _mem_target(hwnd, w, h)
    scr = user32.GetDC(0)
    gdi32.BitBlt(mem, 0, 0, w, nh, scr, pt.x, pt.y + y0, 0x00CC0020)  # SRCCOPY
    user32.ReleaseDC(0, scr)
    bih = _BIH(ctypes.sizeof(_BIH), w, h, 1, 32, 0, 0, 0, 0, 0, 0)
    buf = ctypes.create_string_buffer(w * nh * 4)
    # BitBlt은 대상 DC의 위쪽부터 채웠다 — 비트맵 기준으로는 맨 아래 nh줄이다
    gdi32.GetDIBits(mem, bmp, h - nh, nh, buf, ctypes.byref(bih), 0)
    a = np.frombuffer(buf, np.uint8).reshape(nh, w, 4)[::-1]
    return np.ascontiguousarray(a[:, :, 2::-1])


def capture(hwnd: int, rows: tuple[int, int] | None = None) -> np.ndarray:
    """클라이언트 영역 RGB 캡처 (PrintWindow PW_RENDERFULLCONTENT — 가려져도 동작).

    rows=(y0,y1) 지정 시 **그 행 구간만** 내려받는다. PrintWindow는 늘 전체를
    렌더하지만(17ms) GetDIBits+RGB 변환이 전체 프레임에서 19ms라 값 박스처럼
    좁은 띠만 읽는 폐루프는 캡처 비용이 절반이 된다 (56차 실측).
    반환 배열의 행 0 = 클라이언트 y0.

    PrintWindow가 새까만 프레임만 주는 기계에서는 **바탕화면 뜨기로 갈아탄다**
    (`_SCREEN_GRAB`) — 그때는 창이 가려지면 안 된다.
    """
    r = wt.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(r))
    w, h = r.right, r.bottom
    STATS["captures"] += 1
    y0, y1 = (0, h) if rows is None else (max(0, rows[0]), min(h, rows[1]))
    nh = max(1, y1 - y0)
    if hwnd in _SCREEN_GRAB:
        return _blt_screen(hwnd, w, h, y0, nh)
    mem, bmp = _mem_target(hwnd, w, h)
    user32.PrintWindow(hwnd, mem, 3)  # PW_CLIENTONLY | PW_RENDERFULLCONTENT
    # 상향식(biHeight>0) DIB: 스캔라인 0 = 맨 아래 행 → [h-y1, h-y0) 구간 요청 후 뒤집기
    bih = _BIH(ctypes.sizeof(_BIH), w, h, 1, 32, 0, 0, 0, 0, 0, 0)
    buf = ctypes.create_string_buffer(w * nh * 4)
    gdi32.GetDIBits(mem, bmp, h - y1, nh, buf, ctypes.byref(bih), 0)
    a = np.frombuffer(buf, np.uint8).reshape(nh, w, 4)[::-1]
    out = np.ascontiguousarray(a[:, :, 2::-1])  # BGRA → RGB
    if out.any() or rows is not None:
        return out
    # 전체 프레임이 통째로 0 — 화면에서 떠 보고, 거기 그림이 있으면 갈아탄다
    # (띠 요청으로는 판정하지 않는다: 검은 띠는 얼마든지 있을 수 있다)
    alt = _blt_screen(hwnd, w, h, y0, nh)
    if alt.any():
        _SCREEN_GRAB.add(hwnd)
        print(msg("[io] PrintWindow가 검은 프레임만 준다 — 화면 뜨기로 갈아탄다. "
                  "게임이 우리보다 높은 권한으로 도는 것이 원인이고(스팀을 관리자로 "
                  "켠 경우), 그러면 **키·마우스 입력도 막힌다**. 스팀과 게임을 일반 "
                  "권한으로 다시 켤 것."), file=sys.stderr)
        return alt
    return out
