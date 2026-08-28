# SPDX-License-Identifier: MIT
# kloudys-forza-painter-suite(MIT)의 생성 파이프라인을 이식한 파생물이다.
# Portions © 2021 AE (forza-painter) · Sam Twidale (geometrize-lib) ·
# Michael Fogleman (Primitive). 전문·유래는 THIRD_PARTY_NOTICES.md.
"""최종 도형 목록 → LayerPlan 변환."""

from __future__ import annotations

from ..render import _BASE_HEIGHT_UNITS
from ..model import LayerPlan
from .quantize import _shape_to_layer


def plan_from_shapes(shapes: list[dict], full_w: int, full_h: int,
                     source_image: str = "") -> LayerPlan:
    """최종 도형 목록(양자화 완료) → LayerPlan.

    도형은 이미 게임 그리드 위라 여기의 `.quantized()`는 멱등 확인일 뿐이다.
    KFPS 사각형 data는 중심 기준 [cx, cy, 폭, 높이, rot]다 — `kfpsjson`의
    legacy 들여오기도 같은 `_shape_to_layer` 해석을 쓴다.
    """
    upp = _BASE_HEIGHT_UNITS / full_h
    plan = LayerPlan(source_image=source_image, image_size=(full_w, full_h),
                     units_per_px=upp)
    for shape in shapes:
        lay = _shape_to_layer(shape, full_w, full_h, upp)
        if lay is not None and float(lay.alpha) > 0:
            plan.layers.append(lay)
    return plan
