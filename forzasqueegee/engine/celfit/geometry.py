"""px 기하 — 레이어 하나를 **화면 픽셀**로 옮기는 식 하나.

여기 있는 폴리곤 식이 `render._draw_layer`와 같아야 "플랜 렌더 = 채점 결과"가
성립한다 — 식이 갈리면 그 차이가 곧 새 오차원이다. 배치·채점·수리·메움이
전부 이 한 벌을 쓴다.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..catalog import Catalog
from ..model import UNITS_PER_SCALE, Layer


def _min_span(upp: float) -> float:
    """양자화 최소 도형(스케일 0.01)의 반폭, px — 이보다 작은 것은 못 그린다."""
    return 0.01 * UNITS_PER_SCALE / upp


def _poly_px(cat: Catalog, lay: Layer, upp: float, w: int, h: int,
             ox: int = 0, oy: int = 0) -> list[np.ndarray]:
    """레이어 → 이미지 px 폴리곤 (render._draw_layer와 같은 식, ROI 오프셋)."""
    sh = cat[lay.shape]
    rot = np.radians(lay.rot)
    c, s = np.cos(rot), np.sin(rot)
    polys = []
    for loop in sh.loops:
        pts = loop * np.array([lay.sx, lay.sy], np.float32) * UNITS_PER_SCALE
        if lay.skew:
            pts = pts + np.stack([pts[:, 1] * lay.skew,
                                  np.zeros(len(pts), np.float32)], axis=1)
        pts = pts @ np.array([[c, s], [-s, c]], np.float32)
        pts += np.array([lay.x, lay.y], np.float32)
        px = pts[:, 0] / upp + w / 2 - ox
        py = h / 2 - pts[:, 1] / upp - oy
        polys.append(np.stack([px, py], axis=1))
    return polys


def _mask_px(cat: Catalog, lay: Layer, upp: float, w: int, h: int,
             roi: tuple[int, int, int, int]) -> np.ndarray:
    """ROI(x0,y0,x1,y1) 안 도형 마스크 (짝홀 규칙)."""
    x0, y0, x1, y1 = roi
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    for p in _poly_px(cat, lay, upp, w, h, x0, y0):
        mm = np.zeros_like(m)
        cv2.fillPoly(mm, [np.round(p).astype(np.int32)], 1)
        m ^= mm
    return m.astype(bool)


def _layer(shape: str, cx: float, cy: float, a: float, b: float, theta: float,
           skew: float, color: tuple[int, int, int], upp: float,
           w: int, h: int, label: str = "cel", stroke: int = -1,
           ext: tuple[float, float] | None = None,
           rot_off: float = 0.0) -> Layer:
    """px 파라미터(중심·반길이 a·반폭 b·이미지 각 theta) → 게임 레이어.

    이미지 y-down의 각 theta는 캔버스 y-up에서 -theta다. 획은 label="ink" —
    pruneplan이 ε-프루닝에서 보호한다 (선 마디 하나가 빠지면 그 자리가 빈다).
    `stroke`는 획 그룹 id — 한 획에서 나온 마디끼리 같은 값을 준다.

    `ext`는 도형의 로컬 반길이 (x, y)다 — 기본은 정규화 ±1 도형의 값
    (`UNITS_PER_SCALE`)이라 안 주면 종전과 같다. **가는 획 도형**은 짧은 축의
    반길이가 그보다 작아서, 같은 스케일 스텝으로 훨씬 가는 폭을 낼 수 있다
    (`descriptor.ShapeDesc.min_width_px`). `rot_off`는 그 도형의 긴 축이
    로컬 y일 때 더하는 각(도)이다.
    """
    ex, ey = ext or (UNITS_PER_SCALE, UNITS_PER_SCALE)
    # rot_off 90°는 도형의 **로컬 y가 획 방향**이라는 뜻이다 — 그러면 길이는
    # y축이, 폭은 x축이 맡는다. 짧은 축에 폭을 실어야 최소 스케일이 낼 수 있는
    # 폭이 그 축의 반길이만큼 가늘어진다 (그것이 가는 도형을 쓰는 이유다)
    la, lb = (b, a) if abs(rot_off % 180.0 - 90.0) < 1e-6 else (a, b)
    return Layer(shape=shape,
                 x=(cx - w / 2) * upp, y=(h / 2 - cy) * upp,
                 sx=max(0.01, la * upp / max(ex, 1e-6)),
                 sy=max(0.01, lb * upp / max(ey, 1e-6)),
                 rot=(-np.degrees(theta) + rot_off) % 360.0, skew=skew,
                 color=tuple(int(v) for v in color), alpha=100.0,
                 label=label, stroke=stroke)


def _grad_alpha(cat: Catalog, lay: Layer, upp: float, w: int, h: int,
                rx0: int, ry0: int, bx0: int, by0: int,
                shape: tuple[int, int]) -> np.ndarray | None:
    """그라디언트 도형의 **픽셀별 알파** (bbox 창) — 단색이면 None.

    채점기가 폴리곤 안을 1로 세면 알파 프로파일이 있는 도형을 **덮은 것으로
    과대 계산**한다 (반투명한 자리도 이득으로 친다). 그래서 렌더러와 **같은
    식**을 쓴다 — 식이 갈리면 그 차이가 곧 새 오차원이다
    (`render._draw_layer`의 그라디언트 분기와 한 쌍으로 고칠 것).
    """
    g = cat[lay.shape].gradient
    if g is None or lay.mask:
        return None
    bh, bw = shape
    ys, xs = np.mgrid[by0:by0 + bh, bx0:bx0 + bw]
    ux = ((rx0 + xs).astype(np.float32) - w / 2) * upp
    uy = (h / 2 - (ry0 + ys).astype(np.float32)) * upp
    rot = np.radians(lay.rot)
    c, s = np.cos(rot), np.sin(rot)
    lx = (ux - lay.x) * c + (uy - lay.y) * s          # 역회전 (y-up)
    ly = -(ux - lay.x) * s + (uy - lay.y) * c
    if lay.skew:
        lx = lx - ly * lay.skew                        # 역전단
    nx = lx / (lay.sx * UNITS_PER_SCALE)
    ny = ly / (lay.sy * UNITS_PER_SCALE)
    coord = np.abs(nx) if g["kind"] == "linear" else np.hypot(nx, ny)
    prof = np.asarray(g["profile"], np.float32)
    return np.interp(coord * g["bin_scale"], np.arange(len(prof)), prof,
                     right=0.0).astype(np.float32)


def _ink_cover(layers: list[Layer], cat: Catalog, upp: float,
               w: int, h: int) -> np.ndarray:
    """획 레이어가 실제로 덮는 픽셀 지도 — 면 배치의 공짜 자리 (`_Scorer` ink).

    선 **지도**(`cel.line_mask`)가 아니라 배치된 **레이어**에서 받는다: 3분류로
    안 그은 선, 예산에 밀린 선, 최소 도형보다 굵게 나간 획이 전부 반영돼야
    "정말 가려지는가"가 맞다.
    """
    out = np.zeros((h, w), np.uint8)
    lay_m = one = None
    for lay in layers:
        polys = _poly_px(cat, lay, upp, w, h)
        if len(polys) == 1:
            cv2.fillPoly(out, [np.round(polys[0]).astype(np.int32)], 1)
            continue
        if lay_m is None:                  # 구멍 있는 도형만 짝홀 규칙으로 따로
            lay_m, one = (np.zeros((h, w), np.uint8) for _ in range(2))
        lay_m[:] = 0
        for p in polys:
            one[:] = 0
            cv2.fillPoly(one, [np.round(p).astype(np.int32)], 1)
            lay_m ^= one
        out |= lay_m
    return out.astype(bool)
