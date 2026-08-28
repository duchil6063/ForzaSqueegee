"""변형 편집 수치 표시 OCR.

에디터 변형 도구의 두 값 박스(X/가로, Y/세로)에 표시되는 수치를 판독한다.
- 소수 2자리 고정 → 숫자 글리프만 이어붙여 /100 (점 파싱 불요, '7.'류 병합 회피)
- 글리프 분할: 2D 연결성분(점·부호가 이웃 숫자와 열 겹침 가능하므로 열 투영 금지)
- 분류: 24×16 정규화 후 NCC (absdiff는 '0'↔'6' 혼동 실측 → NCC/XOR 무오류)
- 음수: 높이 낮고 폭 넓은 글리프 = '-'
- 값 변경 직후 롤 애니메이션 → read_stable()로 연속 2회 동일 판독 대기

템플릿: templates/digits.npz — 1776×999 클라이언트에서 수집. NCC 정규화 판독은
클라이언트 720~1350 높이 실측으로 크기 무관 확인.
"""

from __future__ import annotations

import time
from functools import cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import io as gio

# 수치 표시 영역 (클라이언트 크기 대비 비율, 1776×999 실측: x 68..418, y 148..193)
VAL_RECT_REL = (68 / 1776, 148 / 999, 418 / 1776, 193 / 999)
# 크롭(350×45 기준) 내 각 박스의 숫자 영역
BOX_REL = {"x": (44 / 350, 132 / 350), "y": (224 / 350, 312 / 350)}
REF_CROP_W, REF_CROP_H = 350, 45

# HSB 미세 조정 화면: 슬라이더 3행 값 표시 (검은 글씨/흰 배경 — 변형 화면과 극성 반대)
# 1776×999 실측: 값 글리프 h=280..296, s=366..382, b=452..468 (행 간격 86px)
HSB_RECT_REL = {
    "h": (300 / 1776, 274 / 999, 372 / 1776, 302 / 999),
    "s": (300 / 1776, 360 / 999, 372 / 1776, 388 / 999),
    "b": (300 / 1776, 446 / 999, 372 / 1776, 474 / 999),
}
HSB_REF_CROP_H = 28


@cache
def _templates() -> dict[str, np.ndarray]:
    z = np.load(Path(__file__).parent / "templates" / "digits.npz")
    return {chr(int(k[2:])): z[k] for k in z.files}


def _norm(bmp: np.ndarray, size=(24, 16)) -> np.ndarray:
    im = Image.fromarray((bmp * 255).astype(np.uint8)).resize((size[1], size[0]), Image.LANCZOS)
    return np.asarray(im, np.float32) / 255.0


def _ncc_dist(n: np.ndarray, t: np.ndarray) -> float:
    a, b = n - n.mean(), t - t.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return 1.0 - float((a * b).sum() / d) if d > 0 else 1.0


def _glyphs(gray: np.ndarray, x0: int, x1: int, dark_on_light: bool = False):
    """[x0:x1] 영역의 글리프 (bbox, bool비트맵) 목록, 왼→오 (2D 연결성분)."""
    m = ((gray[:, x0:x1] < 90) if dark_on_light else (gray[:, x0:x1] > 180)).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a < 2:
            continue
        out.append(((x, y, x + w, y + h), lab[y:y + h, x:x + w] == i))
    return sorted(out, key=lambda g: g[0][0])


def val_rows(h: int) -> tuple[int, int]:
    """값 박스 띠의 클라이언트 행 구간 — gio.capture(rows=) 부분 캡처용."""
    _, ry0, _, ry1 = VAL_RECT_REL
    return int(ry0 * h), int(ry1 * h)


def read_value(img: np.ndarray, box: str = "x") -> float | None:
    """캡처(RGB 전체)에서 값 박스('x'|'y') 수치 판독. 실패 시 None."""
    h, w = img.shape[:2]
    rx0, ry0, rx1, ry1 = VAL_RECT_REL
    return _read_crop(img[int(ry0 * h):int(ry1 * h), int(rx0 * w):int(rx1 * w)], box)


def read_value_band(band: np.ndarray, box: str = "x") -> float | None:
    """값 박스 띠(val_rows 구간 부분 캡처)에서 수치 판독. 실패 시 None."""
    w = band.shape[1]
    rx0, _, rx1, _ = VAL_RECT_REL
    return _read_crop(band[:, int(rx0 * w):int(rx1 * w)], box)


def _read_crop(crop: np.ndarray, box: str) -> float | None:
    gray = np.asarray(Image.fromarray(crop).convert("L"))
    ch, cw = gray.shape
    scale = ch / REF_CROP_H
    b0, b1 = BOX_REL[box]
    gl = _glyphs(gray, int(b0 * cw), int(b1 * cw))
    if not gl:
        return None
    tpl = _templates()
    # 구분자 판별은 글리프 최대 높이 대비 상대 문턱 — 소수 2자리 고정이라 '.'뿐
    # 아니라 쉼표 소수점 로케일(',', 높이 ~8px로 6px 절대 문턱을 넘음)도 걸러진다.
    gmax = max(y1 - y0 for (_, y0, _, y1), _ in gl)
    digits, neg = "", False
    for (x0, y0, x1, y1), bmp in gl:
        gw, gh = x1 - x0, y1 - y0
        if gh < 0.45 * gmax or gh <= 6 * scale:
            if gw > 4 * scale:
                neg = True
            continue  # 소수점 구분자 무시
        n = _norm(bmp)
        digits += min("0123456789", key=lambda c: _ncc_dist(n, tpl[c]))
    if not digits:
        return None
    v = int(digits) / 100.0
    return -v if neg else v


def read_stable(hwnd: int, box: str = "x", tries: int = 12, delay: float = 0.06) -> float | None:
    """롤 애니메이션 대기: 연속 2회 동일 판독이 나올 때까지.

    표본 간격 = delay + 캡처(17ms). 롤 1스텝보다 길어야 조기 정착 오판이 없다.
    """
    _, h = gio.client_size(hwnd)
    rows = val_rows(h)
    prev: float | None = None
    for _ in range(tries):
        v = read_value_band(gio.capture(hwnd, rows=rows), box)
        if v is not None and v == prev:
            return v
        prev = v
        time.sleep(delay)
    return None


def read_stable_multi(hwnd: int, boxes: tuple[str, ...] = ("x", "y"),
                      tries: int = 30, delay: float = 0.06) -> dict[str, float] | None:
    """두 값 칸을 **한 캡처로** 함께 안정 판독 (연속 2회 동일).

    같은 도구의 두 축(이동 x·y, 크기 sx·sy)은 값 칸이 한 띠에 같이 들어온다 —
    따로 읽으면 롤 정착 대기를 두 번 무는데, 함께 읽으면 한 번이다.
    """
    _, h = gio.client_size(hwnd)
    rows = val_rows(h)
    prev: dict[str, float | None] | None = None
    for _ in range(tries):
        band = gio.capture(hwnd, rows=rows)
        cur = {b: read_value_band(band, b) for b in boxes}
        if prev == cur and all(v is not None for v in cur.values()):
            return {k: float(v) for k, v in cur.items()}
        prev = cur
        time.sleep(delay)
    return None


def read_hsb_value(img: np.ndarray, key: str) -> float | None:
    """HSB 미세 조정 화면에서 슬라이더('h'|'s'|'b') 값 판독. 실패 시 None.

    검은 글씨/흰 배경(극성 반전) 외에는 변형 값 박스와 동일 파이프라인.
    항상 0.00~1.00, 음수·부호 없음.
    """
    h, w = img.shape[:2]
    rx0, ry0, rx1, ry1 = HSB_RECT_REL[key]
    crop = img[int(ry0 * h):int(ry1 * h), int(rx0 * w):int(rx1 * w)]
    gray = np.asarray(Image.fromarray(crop).convert("L"))
    scale = gray.shape[0] / HSB_REF_CROP_H
    gl = _glyphs(gray, 0, gray.shape[1], dark_on_light=True)
    if not gl:
        return None
    tpl = _templates()
    gmax = max(y1 - y0 for (_, y0, _, y1), _ in gl)
    digits = ""
    for (x0, y0, x1, y1), bmp in gl:
        if (y1 - y0) < 0.45 * gmax or (y1 - y0) <= 6 * scale:
            continue  # 소수점 구분자 무시 (상대 문턱 — 쉼표 로케일 포함)
        n = _norm(bmp)
        digits += min("0123456789", key=lambda c: _ncc_dist(n, tpl[c]))
    if not digits:
        return None
    return int(digits) / 100.0


def read_hsb_stable(hwnd: int, key: str, tries: int = 10, delay: float = 0.06) -> float | None:
    """HSB 값 안정 판독 (연속 2회 동일)."""
    prev: float | None = None
    for _ in range(tries):
        v = read_hsb_value(gio.capture(hwnd), key)
        if v is not None and v == prev:
            return v
        prev = v
        time.sleep(delay)
    return None


# 레이어 카운터 "N / M" 크롭 (클라 비율) — 에디터마다 자리가 다르다.
# 차체 에디터는 비닐 에디터보다 한 칸 아래다 (2026-08-17 실측, 1600×899:
# 라임 N x111.., 흰 "/ M", y 770..787). x 하한은 왼쪽의 **흰 아이콘 상자**
# (x 60..100)를 피한 값이다.
#
# 오른쪽은 못 자른다 — 자릿수가 늘면 "R 옵션" 배지가 같이 밀린다 (0/3000일 때
# 배지 x206, 2835/3000일 때 x237). 그래서 넓게 잡고 **글리프 무리로 가른다**:
# 숫자 사이 틈은 2~4px인데 상한과 배지 사이는 27px이라 첫 무리가 곧 "/ M"이다.
VINYL_COUNT_REL = (0.05, 0.790, 0.22, 0.840)
BODY_COUNT_REL = (0.0656, 0.8498, 0.2000, 0.8810)
GLYPH_GAP_REL = 0.0075   # 이보다 먼 글리프는 다른 무리 (배지·라벨)


def _count_glyphs(crop: np.ndarray, mask: np.ndarray,
                  boxes: bool = False) -> list:
    """카운터 크롭에서 글리프 비트맵 목록 (왼→오). 잡티·낮은 조각은 버린다.

    `boxes=True`면 (x0, x1, 비트맵) 튜플을 낸다 (무리 가르기용).
    """
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    gl = []
    for i in range(1, n):
        x, y, gw, gh, a = stats[i]
        if a < 8 or gh < 0.3 * crop.shape[0]:
            continue
        gl.append((int(x), int(x + gw), lab[y:y + gh, x:x + gw] == i))
    gl.sort(key=lambda t: t[0])
    return gl if boxes else [g[2] for g in gl]


def _count_digits(glyphs: list[np.ndarray]) -> int | None:
    if not glyphs:
        return None
    tpl = _templates()
    digits = "".join(min("0123456789", key=lambda c: _ncc_dist(_norm(bmp), tpl[c]))
                     for bmp in glyphs)
    return int(digits) if digits else None


def _slot_glyphs(crop: np.ndarray) -> list[np.ndarray]:
    """비닐 그룹 **저장 슬롯 그리드** 셀 우하단 장수의 숫자 글리프 (왼→오).

    좌측 정보 패널이 없는 화면이라(장수가 셀 우하단 아이콘 옆에 있다) 일반
    `_count_glyphs`가 안 맞는다: ① 4자리는 숫자가 작게 렌더돼 상대 높이 임계에
    걸리고 ② 레이어 겹침 아이콘이 숫자로 오독되고 ③ 콤마가 낀다. 그래서 **절대
    픽셀 크기**로 가른다 — 숫자는 높이 10~24·세로 길쭉(gw<gh), 아이콘은 큰
    정사각(24~42), 콤마는 낮은 조각(<10). 아이콘 오른쪽만 숫자로 본다(썸네일
    조각이 아이콘 왼쪽에 흰점으로 잡히는 것을 버린다 — 2026-08-24 실측).
    """
    mask = (crop.min(axis=2) > 200).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    icon_right = 0
    for i in range(1, n):
        x, y, gw, gh, a = stats[i]
        if 24 <= gh <= 42 and 22 <= gw <= 42:       # 레이어 겹침 아이콘
            icon_right = max(icon_right, x + gw)
    gl = []
    for i in range(1, n):
        x, y, gw, gh, a = stats[i]
        if gh < 10 or gh > 24 or gw >= gh or x < icon_right:
            continue                                 # 콤마·아이콘·좌측 잡티 제외
        gl.append((int(x), lab[y:y + gh, x:x + gw] == i))
    gl.sort(key=lambda t: t[0])
    return [g[1] for g in gl]


def read_slot_count(crop: np.ndarray) -> int | None:
    """저장 슬롯 셀 우하단 장수 크롭 → 정수 (콤마 4자리까지). 실패 시 None."""
    return _count_digits(_slot_glyphs(crop))


def _count_crop(img: np.ndarray, rel: tuple[float, float, float, float]) -> np.ndarray:
    h, w = img.shape[:2]
    x0, y0, x1, y1 = rel
    return img[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]


def _lime_mask(crop: np.ndarray) -> np.ndarray:
    r, g, b = (crop[:, :, i].astype(int) for i in range(3))
    del r
    return (g > 150) & (b < 100) & (g > b + 100)


def read_layer_count(img: np.ndarray) -> int | None:
    """레이어 리스트 좌하단 "N / 3000"의 **N**. 실패 시 None.

    **색으로 가른다** — 현재 수는 라임(191,241,3)이고 "/ 3000"은 흰색이다
    (실측). 자릿수가 늘면 글자가 오른쪽으로 밀리므로 위치로 자를 수 없다.
    소수점이 없어 그대로 정수로 읽는다."""
    crop = _count_crop(img, VINYL_COUNT_REL)
    return _count_digits(_count_glyphs(crop, _lime_mask(crop)))


def read_body_count(img: np.ndarray) -> int | None:
    """차체 에디터 면 카운터의 **현재 장수**(라임). 실패 시 None."""
    crop = _count_crop(img, BODY_COUNT_REL)
    return _count_digits(_count_glyphs(crop, _lime_mask(crop)))


def read_body_cap(img: np.ndarray) -> int | None:
    """차체 에디터 면 카운터의 **상한**(흰 "/ M"의 M). 실패 시 None.

    흰 글리프의 **첫 무리**만 쓴다 (뒤 무리는 "R 옵션" 배지·라벨이다). 그 무리의
    맨 왼쪽 글리프는 구분자 '/'인데 숫자와 높이가 같아 모양으로는 못 거르므로
    **자리로** 버린다 — 표기가 언제나 "/ M"이라 첫 글리프가 곧 구분자다.
    """
    crop = _count_crop(img, BODY_COUNT_REL)
    gl = _count_glyphs(crop, crop.min(axis=2) > 200, boxes=True)
    if len(gl) < 2:
        return None
    gap = GLYPH_GAP_REL * img.shape[1]
    first = [gl[0]]
    for prev, cur in zip(gl, gl[1:]):
        if cur[0] - prev[1] > gap:
            break
        first.append(cur)
    return _count_digits([g[2] for g in first[1:]]) if len(first) >= 2 else None


def read_layer_count_stable(hwnd: int, tries: int = 8,
                            delay: float = 0.05) -> int | None:
    """레이어 수 안정 판독 (연속 2회 동일)."""
    prev: int | None = None
    for _ in range(tries):
        v = read_layer_count(gio.capture(hwnd))
        if v is not None and v == prev:
            return v
        prev = v
        time.sleep(delay)
    return None
