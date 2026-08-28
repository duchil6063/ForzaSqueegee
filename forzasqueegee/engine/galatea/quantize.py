# SPDX-License-Identifier: MIT
# kloudys-forza-painter-suite(MIT)의 생성 파이프라인을 이식한 파생물이다.
# Portions © 2021 AE (forza-painter) · Sam Twidale (geometrize-lib) ·
# Michael Fogleman (Primitive). 전문·유래는 THIRD_PARTY_NOTICES.md.
"""게임 입력 스텝 양자화 (수정 2) — 도형 ↔ 게임 레이어.

KFPS 원본에는 없는 층이다. 원본은 색·기하를 RGB·px로 다루고 최종 JSON을
정수 px로 반올림한다 — 그들의 임포터가 RGB 레코드를 메모리에 직접 쓰기
때문이다. 우리 plan.json은 **변형**이 게임 입력값(이동 0.5유닛·스케일 0.01·
회전 0.1°)이 정본이라 채점·산출을 그 그리드로 스냅한다 (플랜 렌더 = 인게임
원칙 — 사용자 확정 2026-08-25).

색은 스냅하지 않는다 — 정본이 RGB 바이트다 (레코드 +0x74 그대로, KFPS와
같다). 게임 HSB 입력 UI를 쓰는 창 조작만 적용 시점에 `Layer.hsb()`로
변환한다 (사용자 확정 2026-08-25: 인게임 적용에 필요한 경우에만 변환).
그래서 원본과의 색 차이는 float→바이트 반올림뿐이다.
"""

from __future__ import annotations

import numpy as np

from ..catalog import Catalog, default_catalog_path
from ..render import _BASE_HEIGHT_UNITS, _SHAPE
from ..model import UNITS_PER_SCALE, Layer
from .base import RECTANGLE, ROTATED_ELLIPSE, ROTATED_RECTANGLE
from .geometry import copy_shape


def _shape_to_layer(shape: dict, full_w: int, full_h: int, upp: float) -> Layer | None:
    """체크포인트 도형 → 게임 레이어 (양자화 포함).

    KFPS 사각형 data는 [중심x, 중심y, 폭, 높이, 회전] (V2 `rectangle_mask`·
    미리보기 렌더 실측). 타원은 [중심x, 중심y, rx, ry, 회전]. 회전은 이미지
    좌표(y-down) 도(deg)라 우리 캔버스(y-up) rot와 부호 반전이다.
    """
    t = int(shape.get("type", ROTATED_ELLIPSE))
    d = [float(v) for v in shape.get("data", [])]
    if len(d) < 4:
        return None
    col = list(shape.get("color", [0, 0, 0, 255]))
    a255 = float(col[3]) if len(col) > 3 else 255.0
    alpha_pct = round(float(np.clip(a255, 0.0, 255.0)) / 255.0 * 100.0, 2)
    rgb = tuple(int(np.clip(round(float(v)), 0, 255)) for v in col[:3])
    cx, cy = d[0], d[1]
    rot = d[4] if len(d) >= 5 else 0.0
    if t in (RECTANGLE, ROTATED_RECTANGLE):
        cat = Catalog(default_catalog_path())
        return Layer(
            shape=cat.square,
            x=(cx - full_w / 2) * upp, y=(full_h / 2 - cy) * upp,
            sx=max(0.01, d[2] * upp / (2 * UNITS_PER_SCALE)),
            sy=max(0.01, d[3] * upp / (2 * UNITS_PER_SCALE)),
            rot=(-rot) % 360.0, color=rgb, alpha=alpha_pct,
            label="fp").quantized()
    return Layer(
        shape=_SHAPE,
        x=(cx - full_w / 2) * upp, y=(full_h / 2 - cy) * upp,
        sx=max(0.01, d[2] * upp / UNITS_PER_SCALE),
        sy=max(0.01, d[3] * upp / UNITS_PER_SCALE),
        rot=(-rot) % 360.0, color=rgb, alpha=alpha_pct,
        label="fp").quantized()


def _layer_to_shape(lay: Layer, shape_type: int, full_w: int, full_h: int,
                    upp: float, score=0) -> dict:
    """게임 레이어 → 체크포인트 도형 (px, 게임 그리드 위의 실수)."""
    r, g, b = lay.rgb()
    a255 = float(np.clip(lay.alpha, 0.0, 100.0)) / 100.0 * 255.0
    cx = lay.x / upp + full_w / 2
    cy = full_h / 2 - lay.y / upp
    rot = (-lay.rot) % 360.0
    if shape_type in (RECTANGLE, ROTATED_RECTANGLE):
        w_px = lay.sx * 2 * UNITS_PER_SCALE / upp
        h_px = lay.sy * 2 * UNITS_PER_SCALE / upp
        return {"type": shape_type, "color": [r, g, b, a255],
                "data": [cx, cy, max(1.0, w_px), max(1.0, h_px), rot],
                "score": score}
    rx = lay.sx * UNITS_PER_SCALE / upp
    ry = lay.sy * UNITS_PER_SCALE / upp
    return {"type": shape_type, "color": [r, g, b, a255],
            "data": [cx, cy, max(0.5, rx), max(0.5, ry), rot],
            "score": score}


def quantize_shape(shape: dict, full_w: int, full_h: int, upp: float) -> dict:
    """도형 하나를 게임 입력 스텝 그리드로 스냅 (색은 바이트 반올림). 멱등."""
    lay = _shape_to_layer(shape, full_w, full_h, upp)
    if lay is None:
        return copy_shape(shape)
    return _layer_to_shape(lay, int(shape.get("type", ROTATED_ELLIPSE)),
                           full_w, full_h, upp, score=shape.get("score", 0))


def quantize_shapes(shapes: list[dict], full_w: int, full_h: int) -> list[dict]:
    upp = _BASE_HEIGHT_UNITS / full_h
    return [quantize_shape(shape, full_w, full_h, upp) for shape in shapes]
