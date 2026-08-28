# SPDX-License-Identifier: MIT
# kloudys-forza-painter-suite(MIT)의 생성 파이프라인을 이식한 파생물이다.
# Portions © 2021 AE (forza-painter) · Sam Twidale (geometrize-lib) ·
# Michael Fogleman (Primitive). 전문·유래는 THIRD_PARTY_NOTICES.md.
"""후보 미리보기 렌더 — 원본: forza_generator_v2.render_import_preview.

체커보드 배경 + 소스오버 합성은 원본 그대로이고, 도형 래스터만 shape_mask
(A_02 48각형 — 수정 1)를 탄다.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .geometry import shape_mask


def render_import_preview(background: dict, shapes: list[dict], width: int, height: int) -> Image.Image:
    checker = np.zeros((height, width, 3), dtype=np.float32)
    bg_rgba = [int(v) for v in list(background.get("color", [0, 0, 0, 0]))[:4]]
    if len(bg_rgba) < 4:
        bg_rgba += [0] * (4 - len(bg_rgba))
    bg_r, bg_g, bg_b, bg_a = bg_rgba
    premul = np.zeros((height, width, 3), dtype=np.float32)
    alpha_canvas = np.zeros((height, width), dtype=np.float32)
    if bg_a > 0:
        base_alpha = max(0.0, min(1.0, bg_a / 255.0))
        premul[:, :] = np.array((bg_r, bg_g, bg_b), dtype=np.float32) * base_alpha
        alpha_canvas[:, :] = base_alpha
        checker[:, :] = (38, 38, 38)
    else:
        checker[:, :] = (38, 38, 38)
        tile = 32
        for y in range(0, height, tile):
            for x in range(0, width, tile):
                if ((x // tile) + (y // tile)) % 2 == 0:
                    checker[y : y + tile, x : x + tile] = (58, 58, 58)

    for shape in shapes:
        color = list(shape.get("color", [0, 0, 0, 255]))[:4]
        if len(color) < 4 or float(color[3]) <= 0:
            continue
        r, g, b, a = color
        data = list(shape.get("data", []))
        if len(data) < 4:
            continue
        bbox, m = shape_mask(shape, width, height)
        x0, x1, y0, y1 = bbox
        if x1 < x0 or y1 < y0 or not np.any(m):
            continue
        alpha = max(0.0, min(1.0, float(a) / 255.0))
        if alpha <= 0.0:
            continue
        src_rgb = np.array((float(r), float(g), float(b)), dtype=np.float32)
        premul_sub = premul[y0 : y1 + 1, x0 : x1 + 1]
        alpha_sub = alpha_canvas[y0 : y1 + 1, x0 : x1 + 1]
        old_alpha = alpha_sub[m]
        premul_sub[m] = src_rgb * alpha + premul_sub[m] * (1.0 - alpha)
        alpha_sub[m] = alpha + old_alpha * (1.0 - alpha)

    out = premul + checker * (1.0 - alpha_canvas[..., None])
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")
