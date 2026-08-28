"""레이어 계획 렌더 — 미리보기 합성 + 인게임 합성 모델 유틸.

앞부분(`render_plan`)은 카탈로그 도형 전부를 표시 공간에서 그리는 미리보기다.
뒷부분은 painter 노선(`galatea`)·검증 도구가 쓰는 인게임 합성 모델 조각들:

- `_ell_mask` — **게임이 실제로 그리는 A_02 48각형** 래스터. galatea의 채점
  마스크가 이것이다 (`galatea.geometry.ellipse_mask`가 위임).
- 색공간 변환 — sRGB↔선형, 미리곱. FH6는 레이어 알파를 **선형(감마 해제)
  공간**에서 섞는다 (60차 인게임 실측: 25장 캡처 대조에서 sRGB 합성은 평균
  색오차 11.2/255, 선형 합성은 0.82/255). 레이어에 적히는 색은 언제나
  **sRGB**다 (게임 입력값).
- `_from_layer` — 플랜 레이어 → 이미지 px 도형 [x, y, rx, ry, ang] 환산.
- `_replay` — 플랜을 게임과 같은 선형 공간에서 재합성한 **인게임 예측**
  (`tools/verify_alpha_canvas`가 캡처 대조에 쓴다).
"""

from __future__ import annotations

import cv2
import numpy as np

from .catalog import Catalog, default_catalog_path
from .model import UNITS_PER_SCALE, Layer, LayerPlan

_BASE_HEIGHT_UNITS = 900.0  # 캔버스 세로 유닛 (오버레이 기본 배율과 동일)
_SHAPE = "A_02"             # 카탈로그 원 = 회전 타원의 몸통


def render_plan(plan: LayerPlan, catalog: Catalog, scale: int = 1, pad: int = 0,
                bg: int = 255) -> np.ndarray:
    """계획을 원본 이미지 크기(×scale)로 렌더 (RGB, 기본 흰 배경).

    pad: 이미지 rect 사방에 추가할 px — 경계 밖 돌출·마스크 밴드 검증용.
    bg: 배경 회색값. **뺄셈 마스크가 드러내는 색도 이 값이다** (마스크의 정의가
        "먼저 그려진 것을 잘라 배경을 노출"이라 배경을 바꾸면 같이 바뀐다).
        검은 배경 렌더를 같이 뜨면 알파를 되풀 수 있다 (`verify_inject_canvas`).
    """
    w, h = plan.image_size
    out = np.full(((h + 2 * pad) * scale, (w + 2 * pad) * scale, 3), bg, np.uint8)
    for layer in plan.layers:
        _draw_layer(out, layer, plan, catalog, scale, pad, bg)
    return out


def _is_font(shape: str) -> bool:
    from .textvinyl import is_font

    return is_font(shape)


def _draw_layer(img: np.ndarray, layer: Layer, plan: LayerPlan, catalog: Catalog,
                scale: int, pad: int = 0, bg: int = 255) -> None:
    sh = catalog[layer.shape]
    w, h = plan.image_size
    upp = plan.units_per_px
    rot = np.radians(layer.rot)
    c, s = np.cos(rot), np.sin(rot)
    # 뺄셈 마스크 = 먼저 그려진 것 전부 잘라 배경 노출 → 배경색으로 칠함
    color = (bg, bg, bg) if layer.mask else layer.rgb()

    # **글꼴 글리프의 em 상자 표식은 안 그린다.** 알파 0인 삼각형 4개가 ±1
    # 귀퉁이에 박혀 있어 게임은 안 그리는데, 그대로 칠하면 미리보기와 수동
    # 오버레이에 없는 도형이 뜬다 (`engine.textvinyl` 문서).
    loops = sh.loops
    if _is_font(layer.shape):
        from .textvinyl import ink_loops
        loops = tuple(ink_loops(catalog, layer.shape))
    polys = []
    for loop in loops:
        pts = loop * np.array([layer.sx, layer.sy], np.float32) * UNITS_PER_SCALE
        if layer.skew:
            # 전단 — 회전 **전** 도형 좌표에서 x += skew·y (Tab 뒤집기가 부호를
            # 반전시킨다는 실측과 맞는 꼴이다: x→-x 면 같은 모양을 내려면 k→-k)
            pts = pts + np.stack([pts[:, 1] * layer.skew,
                                  np.zeros(len(pts), np.float32)], axis=1)
        # 캔버스 y-up CCW 회전
        pts = pts @ np.array([[c, s], [-s, c]], np.float32)
        pts += np.array([layer.x, layer.y], np.float32)
        # 캔버스 유닛 → 이미지 px
        px = pts[:, 0] / upp + w / 2 + pad
        py = h / 2 - pts[:, 1] / upp + pad
        polys.append(np.stack([px, py], axis=1) * scale)

    a = float(np.clip(layer.alpha, 0.0, 100.0)) / 100.0
    if sh.gradient is not None and not layer.mask:
        # 그라데이션 도형 (39차 인게임 실측): 실루엣 안 픽셀별 알파 =
        # 프로파일[로컬 정규화 좌표] × 레이어 알파, 소스오버 합성
        m = np.zeros(img.shape[:2], np.uint8)
        for p in polys:
            mm = np.zeros_like(m)
            cv2.fillPoly(mm, [np.round(p).astype(np.int32)], 1)
            m ^= mm
        sel = m.astype(bool)
        if not sel.any():
            return
        ys, xs = np.nonzero(sel)
        ux = (xs.astype(np.float32) / scale - w / 2 - pad) * upp
        uy = (h / 2 + pad - ys.astype(np.float32) / scale) * upp
        lx = (ux - layer.x) * c + (uy - layer.y) * s  # 역회전 (y-up)
        ly = -(ux - layer.x) * s + (uy - layer.y) * c
        if layer.skew:
            lx = lx - ly * layer.skew                 # 역전단
        nx = lx / (layer.sx * UNITS_PER_SCALE)
        ny = ly / (layer.sy * UNITS_PER_SCALE)
        g = sh.gradient
        coord = np.abs(nx) if g["kind"] == "linear" else np.hypot(nx, ny)
        prof = np.asarray(g["profile"], np.float32)
        atex = np.interp(coord * g["bin_scale"], np.arange(len(prof)), prof,
                         right=0.0).astype(np.float32)
        aa = (a * atex)[:, None]
        img[ys, xs] = (img[ys, xs].astype(np.float32) * (1.0 - aa)
                       + np.array(layer.rgb(), np.float32)[None] * aa
                       ).astype(np.uint8)
        return
    if a >= 0.995:
        if len(polys) == 1:
            cv2.fillPoly(img, [np.round(polys[0]).astype(np.int32)], color, cv2.LINE_AA)
        else:  # 짝홀 규칙(구멍) — 마스크 XOR 후 적용
            m = np.zeros(img.shape[:2], np.uint8)
            for p in polys:
                mm = np.zeros_like(m)
                cv2.fillPoly(mm, [np.round(p).astype(np.int32)], 1)
                m ^= mm
            img[m.astype(bool)] = color
        return
    # 반투명 레이어: 소스오버 합성. 게임 투명도 축은
    # 8비트 알파(표시값 = alpha/255×100, 검증 문서 4절) — 표시값 그대로 합성
    m = np.zeros(img.shape[:2], np.uint8)
    for p in polys:
        mm = np.zeros_like(m)
        cv2.fillPoly(mm, [np.round(p).astype(np.int32)], 1)
        m ^= mm
    sel = m.astype(bool)
    img[sel] = (img[sel].astype(np.float32) * (1.0 - a)
                + np.array(color, np.float32) * a).astype(np.uint8)


# -------------------------------------------------------------- 색 공간 변환
# 게임이 섞는 곳(선형)과 게임에 적는 곳(sRGB)을 오간다. 전부 0~255 눈금이다.

def _srgb_to_lin(c) -> np.ndarray:
    x = np.asarray(c, np.float64) / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4) * 255.0


def _lin_to_srgb(c) -> np.ndarray:
    x = np.clip(np.asarray(c, np.float64) / 255.0, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92,
                    1.055 * x ** (1 / 2.4) - 0.055) * 255.0


def _unpremult(p: np.ndarray) -> np.ndarray:
    """투명 화소는 **0**으로 둔다 — `where=`만 주고 `out=`을 안 주면 그 자리가
    초기화 안 된 쓰레기로 남는다. 대개 유한값이라 뒤에 곱하는 알파 0에 지워지지만
    NaN이 걸리면 그 화소가 통째로 NaN이 되어 지나가던 계산이 조용히 무너진다."""
    a = p[..., 3:4] / 255.0
    return np.divide(p[..., :3], np.maximum(a, 1e-6),
                     out=np.zeros_like(p[..., :3]), where=a > 1e-6)


def _prem_to_srgb(p: np.ndarray) -> np.ndarray:
    """미리곱 선형 → 미리곱 sRGB (표시 공간 — 미리보기·채점 보고용)."""
    a = p[..., 3:4] / 255.0
    return np.dstack([_lin_to_srgb(_unpremult(p)) * a, p[..., 3]]).astype(np.float32)


# ------------------------------------------------- 인게임 A_02 래스터·환산

_UNIT: np.ndarray | None = None  # A_02 정규화 윤곽 (48각형)


def _unit() -> np.ndarray:
    """카탈로그 A_02의 ±1 윤곽. **게임이 그리는 그 다각형**이다 (48각형) —
    참 타원으로 채점하면 작은 도형에서 면적이 최대 20% 어긋나 플랜 렌더와
    시뮬레이션이 갈린다 (58차 실측 IoU 0.78~0.96)."""
    global _UNIT
    if _UNIT is None:
        _UNIT = np.asarray(Catalog(default_catalog_path())[_SHAPE].loops[0],
                           np.float32)
    return _UNIT


def _ell_mask(w: int, h: int, cx: float, cy: float, rx: float, ry: float,
              ang: float):
    """이미지 좌표계 회전 타원(=A_02 도형)의 bbox 한정 마스크. (mask, x0, y0).

    기하는 `_draw_layer`와 같은 식이다 — 도형 윤곽 × (rx, ry) →
    캔버스 y-up CCW 회전(rot = −ang) → 이미지 좌표."""
    r = np.radians(-ang)
    c, s = np.cos(r), np.sin(r)
    p = _unit() * np.array([rx, ry], np.float32)
    p = p @ np.array([[c, s], [-s, c]], np.float32)
    px = p[:, 0] + cx
    py = cy - p[:, 1]
    x0 = max(0, int(np.floor(px.min())))
    y0 = max(0, int(np.floor(py.min())))
    x1 = min(w, int(np.ceil(px.max())) + 1)
    y1 = min(h, int(np.ceil(py.max())) + 1)
    if x1 - x0 < 1 or y1 - y0 < 1:
        return None
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(m, [np.round(np.stack([px - x0, py - y0], axis=1)).astype(np.int32)], 1)
    return m, x0, y0


def _from_layer(lay: Layer, upp: float, w: int, h: int):
    s = np.array([lay.x / upp + w / 2, h / 2 - lay.y / upp,
                  lay.sx * UNITS_PER_SCALE / upp, lay.sy * UNITS_PER_SCALE / upp,
                  (-lay.rot) % 360.0], np.float64)
    return s, np.array(lay.rgb(), np.float32)


def _replay(plan: LayerPlan, w: int, h: int, upp: float) -> np.ndarray:
    """레이어 목록을 처음부터 다시 합성한 **선형** 미리곱 RGBA 캔버스.

    게임과 같은 공간에서 섞으므로 이것이 인게임 예측이다. 표시용으로 쓰려면
    `_prem_to_srgb`를 거칠 것. `fp_bg`(전면 배경 사각형 — `kfpsjson`이 들여온
    플랜에 있다)만 전면 합성이고 나머지는 A_02 타원으로 그린다."""
    cur = np.zeros((h, w, 4), np.float32)
    for lay in plan.layers:
        la = float(np.clip(lay.alpha, 0, 100)) / 100.0
        col4 = np.array([*_srgb_to_lin(lay.rgb()), 255.0], np.float32)
        if lay.label == "fp_bg":
            cur[:] = (1.0 - la) * cur + la * col4
            continue
        s, _ = _from_layer(lay, upp, w, h)
        mm = _ell_mask(w, h, s[0], s[1], s[2], s[3], s[4])
        if mm is None:
            continue
        m, x0, y0 = mm
        box = cur[y0:y0 + m.shape[0], x0:x0 + m.shape[1]]
        mb = m.astype(bool)
        box[mb] = (1.0 - la) * box[mb] + la * col4
    return cur
