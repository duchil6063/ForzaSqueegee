"""경계 밖 4밴드 마스크 — 이미지 rect 밖으로 돌출한 페인트를 덮는 사각 뺄셈 마스크.

인게임 검증 완료(13차): 마스크 스코프가 전역이므로 반드시 플랜 말미에 둘 것.
스케일 캡 없음·가로/세로 독립이라 밴드 크기 제약 없음.
"""

from __future__ import annotations

import numpy as np

from .catalog import Catalog
from .model import UNITS_PER_SCALE, Layer, LayerPlan

_SQUARE = "A_01"
BORDER_MARGIN = 4.0  # 돌출 bbox 바깥 여유
BORDER_INSET = 1.0  # 이미지 안쪽 겹침 — 양자화(이동 0.5·스케일 0.01)로 경계선이 새는 것 방지
_BORDER_EPS = 0.5  # 이 이하 돌출은 무시 (이동 스텝 1개 미만)


def layer_bbox(layer: Layer, catalog: Catalog) -> tuple[float, float, float, float]:
    """레이어의 캔버스 유닛 bbox (minx, miny, maxx, maxy) — 렌더와 같은 변환."""
    rot = np.radians(layer.rot)
    c, s = np.cos(rot), np.sin(rot)
    pts = np.concatenate(catalog[layer.shape].loops).astype(np.float32)
    pts = pts * np.array([layer.sx, layer.sy], np.float32) * UNITS_PER_SCALE
    pts = pts @ np.array([[c, s], [-s, c]], np.float32)
    pts += np.array([layer.x, layer.y], np.float32)
    return (float(pts[:, 0].min()), float(pts[:, 1].min()),
            float(pts[:, 0].max()), float(pts[:, 1].max()))


def border_mask_layers(plan: LayerPlan, catalog: Catalog,
                       margin: float = BORDER_MARGIN) -> list[Layer]:
    """이미지 rect 밖 돌출을 덮는 상/하/좌/우 사각(A_01) 마스크 밴드.

    페인트 레이어 전체의 돌출 bbox → 변별 돌출량 산출, 돌출한 변만 밴드 생성.
    밴드는 바깥 전체 rect 길이로 깔아 모서리를 이중 커버(마스크 중복 컷은 무해).
    """
    w, h = plan.image_size
    hw = w * plan.units_per_px / 2
    hh = h * plan.units_per_px / 2
    paint = [l for l in plan.layers if not l.mask]
    if not paint:
        return []
    boxes = np.array([layer_bbox(l, catalog) for l in paint])
    minx, miny = boxes[:, 0].min(), boxes[:, 1].min()
    maxx, maxy = boxes[:, 2].max(), boxes[:, 3].max()
    prot = {"left": max(0.0, -minx - hw), "right": max(0.0, maxx - hw),
            "bottom": max(0.0, -miny - hh), "top": max(0.0, maxy - hh)}
    # 바깥 rect (돌출 + 여유)
    lx, rx = hw + prot["left"] + margin, hw + prot["right"] + margin
    by, ty = hh + prot["bottom"] + margin, hh + prot["top"] + margin
    rects = {  # (x0, y0, x1, y1) — 상/하 밴드는 전체 폭, 좌/우 밴드는 전체 높이
        "top": (-lx, hh - BORDER_INSET, rx, ty),
        "bottom": (-lx, -by, rx, -(hh - BORDER_INSET)),
        "left": (-lx, -by, -(hw - BORDER_INSET), ty),
        "right": (hw - BORDER_INSET, -by, rx, ty),
    }
    bands = []
    for side, (x0, y0, x1, y1) in rects.items():
        if prot[side] <= _BORDER_EPS:
            continue
        bands.append(Layer(
            shape=_SQUARE,
            x=(x0 + x1) / 2, y=(y0 + y1) / 2,
            sx=(x1 - x0) / (2 * UNITS_PER_SCALE),
            sy=(y1 - y0) / (2 * UNITS_PER_SCALE),
            color=(255, 255, 255),
            label="mask",
            mask=True,
        ).quantized())
    return bands
