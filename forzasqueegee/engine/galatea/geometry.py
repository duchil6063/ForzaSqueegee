# SPDX-License-Identifier: MIT
# kloudys-forza-painter-suite(MIT)의 생성 파이프라인을 이식한 파생물이다.
# Portions © 2021 AE (forza-painter) · Sam Twidale (geometrize-lib) ·
# Michael Fogleman (Primitive). 전문·유래는 THIRD_PARTY_NOTICES.md.
"""도형 기하 — 마스크·bbox·캔버스 경계.

원본: forza_generator_v2.py. 사각형·경계·맞춤은 원본 그대로이고 두 곳이
다르다 (전부 "채점 마스크 = 게임이 실제로 그리는 것" 원칙, 수정 1):

- `ellipse_mask` — 참 타원 + 크기 보정(`compensated_ellipse_size`) 대신
  게임이 그리는 A_02 48각형 (`render._ell_mask`).
- `raw_rect_bbox`/`rotated_rect_bbox` — 원본은 사각형에도 타원 bbox 공식을
  써서 회전 사각형 모서리가 채점에서 잘렸다 (45°에서 면적 -7.6% 실측).
"""

from __future__ import annotations

import math

import numpy as np

from ..render import _ell_mask
from .base import RECTANGLE, ROTATED_ELLIPSE, ROTATED_RECTANGLE


def copy_shape(shape: dict) -> dict:
    out = {
        "type": int(shape.get("type", ROTATED_ELLIPSE)),
        "color": list(shape.get("color", [0, 0, 0, 255])),
        "data": list(shape.get("data", [])),
        "score": shape.get("score", 0),
    }
    return out


def scale_shape(shape: dict, sx: float, sy: float) -> dict:
    scaled = {
        "type": int(shape.get("type", ROTATED_ELLIPSE)),
        "color": list(shape.get("color", [0, 0, 0, 255])),
    }
    data = list(shape.get("data", []))
    if len(data) < 4:
        raise ValueError("shape missing data")
    cx, cy, rx, ry = data[:4]
    rot = data[4] if len(data) >= 5 else 0
    scaled["data"] = [
        float(cx) * sx,
        float(cy) * sy,
        max(0.5, float(rx) * sx),
        max(0.5, float(ry) * sy),
        float(rot),
    ]
    return scaled


def unscale_shape_f(shape: dict, sx: float, sy: float) -> dict:
    """채점 공간 → 전체 해상도 px, **반올림 없이**.

    원본 `unscale_shape`는 정수 px로 반올림했다 — 우리 최종 그리드는 게임
    입력 스텝이므로 여기서는 실수로 되돌리고 `quantize_shape`가 스냅한다.
    """
    data = list(shape.get("data", []))
    if len(data) < 4:
        raise ValueError("shape missing data")
    x, y, rx, ry = [float(v) for v in data[:4]]
    rot = float(data[4]) if len(data) >= 5 else 0.0
    out = {
        "type": int(shape.get("type", ROTATED_ELLIPSE)),
        "color": list(shape.get("color", [0, 0, 0, 255])),
        "data": [
            x / max(sx, 1e-6),
            y / max(sy, 1e-6),
            max(0.5, rx / max(sx, 1e-6)),
            max(0.5, ry / max(sy, 1e-6)),
            rot % 360.0,
        ],
        "score": shape.get("score", 0),
    }
    return out


def target_has_alpha_boundary(target_rgba: np.ndarray) -> bool:
    if target_rgba.ndim < 3 or target_rgba.shape[2] < 4:
        return False
    alpha = target_rgba[..., 3]
    return bool(alpha.size and int(np.min(alpha)) < 250)


def raw_rotated_bbox(cx: float, cy: float, rx: float, ry: float, rot_deg: float) -> tuple[float, float, float, float]:
    theta = math.radians(rot_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    ex = math.sqrt((rx * rx * cos_t * cos_t) + (ry * ry * sin_t * sin_t))
    ey = math.sqrt((rx * rx * sin_t * sin_t) + (ry * ry * cos_t * cos_t))
    return cx - ex, cx + ex, cy - ey, cy + ey


def raw_rect_bbox(cx: float, cy: float, hw: float, hh: float, rot_deg: float) -> tuple[float, float, float, float]:
    """회전 **사각형**의 참 bbox (반폭 hw·반높이 hh).

    원본은 사각형에도 타원 bbox 공식(`raw_rotated_bbox`)을 써서 회전 사각형의
    네 모서리가 bbox 밖으로 잘렸다 (45°에서 면적 -7.6% 실측). 게임은 A_01
    사각형을 통째로 그리고 KFPS 자신의 미리보기 렌더(fillConvexPoly)도 통째로
    그린다 — "채점 마스크 = 게임이 실제로 그리는 것" 원칙(수정 1)에 맞춰
    참 bbox로 고친다.
    """
    theta = math.radians(rot_deg)
    cos_t = abs(math.cos(theta))
    sin_t = abs(math.sin(theta))
    ex = hw * cos_t + hh * sin_t
    ey = hw * sin_t + hh * cos_t
    return cx - ex, cx + ex, cy - ey, cy + ey


def rotated_bbox(cx: float, cy: float, rx: float, ry: float, rot_deg: float, width: int, height: int) -> tuple[int, int, int, int]:
    raw_x0, raw_x1, raw_y0, raw_y1 = raw_rotated_bbox(cx, cy, rx, ry, rot_deg)
    x0 = max(0, int(math.floor(raw_x0 - 1)))
    x1 = min(width - 1, int(math.ceil(raw_x1 + 1)))
    y0 = max(0, int(math.floor(raw_y0 - 1)))
    y1 = min(height - 1, int(math.ceil(raw_y1 + 1)))
    return x0, x1, y0, y1


def rotated_rect_bbox(cx: float, cy: float, hw: float, hh: float, rot_deg: float, width: int, height: int) -> tuple[int, int, int, int]:
    raw_x0, raw_x1, raw_y0, raw_y1 = raw_rect_bbox(cx, cy, hw, hh, rot_deg)
    x0 = max(0, int(math.floor(raw_x0 - 1)))
    x1 = min(width - 1, int(math.ceil(raw_x1 + 1)))
    y0 = max(0, int(math.floor(raw_y0 - 1)))
    y1 = min(height - 1, int(math.ceil(raw_y1 + 1)))
    return x0, x1, y0, y1


def rectangle_mask(shape: dict, width: int, height: int) -> tuple[tuple[int, int, int, int], np.ndarray]:
    data = list(shape.get("data", []))
    if len(data) < 4:
        return (0, -1, 0, -1), np.zeros((0, 0), dtype=bool)
    cx, cy, w, h = [float(v) for v in data[:4]]
    rot_deg = float(data[4]) if len(data) >= 5 else 0.0
    rx = max(0.5, w * 0.5)
    ry = max(0.5, h * 0.5)
    x0, x1, y0, y1 = rotated_rect_bbox(cx, cy, rx, ry, rot_deg, width, height)
    if x1 < x0 or y1 < y0:
        return (x0, x1, y0, y1), np.zeros((0, 0), dtype=bool)
    xs = np.arange(x0, x1 + 1, dtype=np.float32) + 0.5
    ys = np.arange(y0, y1 + 1, dtype=np.float32) + 0.5
    xx, yy = np.meshgrid(xs, ys)
    theta = math.radians(rot_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    dx = xx - cx
    dy = yy - cy
    xr = dx * cos_t + dy * sin_t
    yr = -dx * sin_t + dy * cos_t
    return (x0, x1, y0, y1), (np.abs(xr) <= rx) & (np.abs(yr) <= ry)


def ellipse_mask(shape: dict, width: int, height: int) -> tuple[tuple[int, int, int, int], np.ndarray]:
    """**수정 1** — 참 타원 방정식 대신 게임이 실제로 그리는 A_02 48각형.

    래스터는 `render._ell_mask` 그대로다 (fillPoly, 정점 반올림) — 그쪽이
    인게임 실측으로 검증된 함수라 여기서 다시 만들지 않는다. 원본의
    `compensated_ellipse_size`(대형·고종횡비 축소 보정)는 참 타원으로 게임
    렌더를 근사하던 장치라 실물 다각형 앞에서는 뺀다.
    """
    data = list(shape.get("data", []))
    if len(data) < 4:
        return (0, -1, 0, -1), np.zeros((0, 0), dtype=bool)
    cx, cy, rx, ry = [float(v) for v in data[:4]]
    rot_deg = float(data[4]) if len(data) >= 5 else 0.0
    mm = _ell_mask(width, height, cx, cy, max(0.5, rx), max(0.5, ry), rot_deg)
    if mm is None:
        return (0, -1, 0, -1), np.zeros((0, 0), dtype=bool)
    m, x0, y0 = mm
    return (x0, x0 + m.shape[1] - 1, y0, y0 + m.shape[0] - 1), m.astype(bool)


def shape_mask(shape: dict, width: int, height: int) -> tuple[tuple[int, int, int, int], np.ndarray]:
    if int(shape.get("type", ROTATED_ELLIPSE)) in (RECTANGLE, ROTATED_RECTANGLE):
        return rectangle_mask(shape, width, height)
    return ellipse_mask(shape, width, height)


def shape_bbox(shape: dict, width: int, height: int) -> tuple[int, int, int, int]:
    data = list(shape.get("data", []))
    if len(data) < 4:
        return (0, -1, 0, -1)
    cx, cy, w, h = [float(v) for v in data[:4]]
    rot = float(data[4]) if len(data) >= 5 else 0.0
    if int(shape.get("type", ROTATED_ELLIPSE)) in (RECTANGLE, ROTATED_RECTANGLE):
        return rotated_rect_bbox(cx, cy, max(0.5, w * 0.5), max(0.5, h * 0.5), rot, width, height)
    # 48각형은 외접 타원 안에 있으므로 해석적 타원 bbox가 안전한 덮개다
    return rotated_bbox(cx, cy, max(0.5, w), max(0.5, h), rot, width, height)


def shape_raw_bbox(shape: dict) -> tuple[float, float, float, float] | None:
    data = list(shape.get("data", []))
    if len(data) < 4:
        return None
    cx, cy, w, h = [float(v) for v in data[:4]]
    rot = float(data[4]) if len(data) >= 5 else 0.0
    if int(shape.get("type", ROTATED_ELLIPSE)) in (RECTANGLE, ROTATED_RECTANGLE):
        return raw_rect_bbox(cx, cy, max(0.5, w * 0.5), max(0.5, h * 0.5), rot)
    return raw_rotated_bbox(cx, cy, max(0.5, w), max(0.5, h), rot)


def canvas_edge_context(target_rgba: np.ndarray) -> dict | None:
    if not target_has_alpha_boundary(target_rgba):
        return None
    height, width = target_rgba.shape[:2]
    if height <= 0 or width <= 0:
        return None
    alpha = target_rgba[..., 3] if target_rgba.ndim >= 3 and target_rgba.shape[2] >= 4 else None
    if alpha is None:
        return None
    strip = max(2, min(12, int(round(min(height, width) * 0.01))))
    visible = alpha > 8
    return {
        "left": np.max(visible[:, :strip], axis=1),
        "right": np.max(visible[:, width - strip :], axis=1),
        "top": np.max(visible[:strip, :], axis=0),
        "bottom": np.max(visible[height - strip :, :], axis=0),
        "strip": strip,
    }


def edge_side_allows_overhang(
    edge_context: dict | None,
    side: str,
    span_start: float,
    span_end: float,
    length: int,
    min_visible_fraction: float = 0.08,
) -> bool:
    if not edge_context:
        return False
    edge = edge_context.get(side)
    if edge is None or length <= 0:
        return False
    start = max(0, min(length - 1, int(math.floor(span_start))))
    end = max(0, min(length - 1, int(math.ceil(span_end))))
    if end < start:
        return False
    span = edge[start : end + 1]
    if span.size == 0:
        return False
    return (float(np.count_nonzero(span)) / float(span.size)) >= min_visible_fraction


def shape_boundary_penalty(shape: dict, width: int, height: int, enabled: bool, edge_context: dict | None = None) -> float:
    if not enabled:
        return 0.0
    bbox = shape_raw_bbox(shape)
    if bbox is None:
        return 0.0
    x0, x1, y0, y1 = bbox
    bbox_w = max(0.0, x1 - x0)
    bbox_h = max(0.0, y1 - y0)
    if bbox_w <= 0.0 or bbox_h <= 0.0:
        return 0.0
    outside_area = 0.0
    if x0 < 0.0 and not edge_side_allows_overhang(edge_context, "left", y0, y1, height):
        outside_area += min(-x0, bbox_w) * bbox_h
    if x1 > float(width) and not edge_side_allows_overhang(edge_context, "right", y0, y1, height):
        outside_area += min(x1 - float(width), bbox_w) * bbox_h
    if y0 < 0.0 and not edge_side_allows_overhang(edge_context, "top", x0, x1, width):
        outside_area += min(-y0, bbox_h) * bbox_w
    if y1 > float(height) and not edge_side_allows_overhang(edge_context, "bottom", x0, x1, width):
        outside_area += min(y1 - float(height), bbox_h) * bbox_w
    outside_area = min(outside_area, bbox_w * bbox_h)
    if outside_area <= 0.25:
        return 0.0
    color = list(shape.get("color", [0, 0, 0, 255]))
    alpha = float(color[3]) / 255.0 if len(color) >= 4 else 1.0
    alpha = max(0.0, min(1.0, alpha))
    if alpha <= 0.0:
        return 0.0
    return outside_area * alpha * alpha * float(255.0 * 255.0 * 3.0 * 8.0)


def shape_boundary_penalties(shapes: list[dict], width: int, height: int, enabled: bool, edge_context: dict | None = None) -> np.ndarray:
    if not enabled:
        return np.zeros(len(shapes), dtype=np.float64)
    return np.array([shape_boundary_penalty(shape, width, height, True, edge_context) for shape in shapes], dtype=np.float64)


def fit_shape_inside_canvas(shape: dict, width: int, height: int, edge_context: dict | None = None) -> dict:
    fitted = copy_shape(shape)
    data = list(fitted.get("data", []))
    if len(data) < 4:
        return fitted
    for _ in range(8):
        bbox = shape_raw_bbox(fitted)
        if bbox is None:
            break
        x0, x1, y0, y1 = bbox
        allow_left = edge_side_allows_overhang(edge_context, "left", y0, y1, height)
        allow_right = edge_side_allows_overhang(edge_context, "right", y0, y1, height)
        allow_top = edge_side_allows_overhang(edge_context, "top", x0, x1, width)
        allow_bottom = edge_side_allows_overhang(edge_context, "bottom", x0, x1, width)
        violate_left = x0 < 0.0 and not allow_left
        violate_right = x1 > float(width) and not allow_right
        violate_top = y0 < 0.0 and not allow_top
        violate_bottom = y1 > float(height) and not allow_bottom
        if not (violate_left or violate_right or violate_top or violate_bottom):
            break
        bbox_w = max(1.0, x1 - x0)
        bbox_h = max(1.0, y1 - y0)
        scale_limits = [1.0]
        if violate_left and violate_right:
            scale_limits.append(max(1.0, float(width) - 1.0) / bbox_w)
        if violate_top and violate_bottom:
            scale_limits.append(max(1.0, float(height) - 1.0) / bbox_h)
        scale = min(scale_limits)
        if scale < 0.999:
            data = list(fitted["data"])
            data[2] = max(1.0, float(data[2]) * scale)
            data[3] = max(1.0, float(data[3]) * scale)
            fitted["data"] = data
            bbox = shape_raw_bbox(fitted)
            if bbox is None:
                break
            x0, x1, y0, y1 = bbox
            allow_left = edge_side_allows_overhang(edge_context, "left", y0, y1, height)
            allow_right = edge_side_allows_overhang(edge_context, "right", y0, y1, height)
            allow_top = edge_side_allows_overhang(edge_context, "top", x0, x1, width)
            allow_bottom = edge_side_allows_overhang(edge_context, "bottom", x0, x1, width)
        dx = 0.0
        dy = 0.0
        if x0 < 0.0 and not allow_left:
            dx = -x0
        if x1 > float(width) and not allow_right:
            dx = min(dx, float(width) - x1) if dx else float(width) - x1
        if y0 < 0.0 and not allow_top:
            dy = -y0
        if y1 > float(height) and not allow_bottom:
            dy = min(dy, float(height) - y1) if dy else float(height) - y1
        data = list(fitted["data"])
        data[0] = float(data[0]) + dx
        data[1] = float(data[1]) + dy
        fitted["data"] = data
    return fitted
