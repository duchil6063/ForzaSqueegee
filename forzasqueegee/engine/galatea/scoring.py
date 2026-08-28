# SPDX-License-Identifier: MIT
# kloudys-forza-painter-suite(MIT)의 생성 파이프라인을 이식한 파생물이다.
# Portions © 2021 AE (forza-painter) · Sam Twidale (geometrize-lib) ·
# Michael Fogleman (Primitive). 전문·유래는 THIRD_PARTY_NOTICES.md.
"""채점·프루닝 — 중요도 가중 렌더 채점, 목표 장수 프루닝, 가려진 레이어 정리.

원본: forza_generator_v2.py(build_importance_map·render_and_score·
render_and_score_region·normalized_error·score_shape_list·
remove_lowest_ranked_batch·prune_to_target·visible_shape_pixels·
remove_fully_covered_layers) — 전부 원본 그대로. 마스크만 geometry의
A_02 48각형(수정 1)을 탄다.
"""

from __future__ import annotations

import numpy as np

from .geometry import (canvas_edge_context, copy_shape, shape_bbox,
                       shape_boundary_penalties, shape_mask)


def build_importance_map(target_rgba: np.ndarray) -> np.ndarray:
    """에지·알파 절단·채도 세부·선화 쪽으로 채점 가중 (원본 그대로)."""
    height, width = target_rgba.shape[:2]
    if height <= 0 or width <= 0:
        return np.ones((max(1, height), max(1, width)), dtype=np.float32)

    rgba = target_rgba.astype(np.float32)
    rgb = rgba[..., :3]
    alpha = np.clip(rgba[..., 3] / 255.0, 0.0, 1.0)
    luma = (rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114) / 255.0
    maxc = rgb.max(axis=2) / 255.0
    minc = rgb.min(axis=2) / 255.0
    saturation = maxc - minc

    gx = np.zeros_like(luma, dtype=np.float32)
    gy = np.zeros_like(luma, dtype=np.float32)
    gx[:, 1:] = np.abs(luma[:, 1:] - luma[:, :-1])
    gy[1:, :] = np.abs(luma[1:, :] - luma[:-1, :])
    edge = np.maximum(gx, gy)

    agx = np.zeros_like(alpha, dtype=np.float32)
    agy = np.zeros_like(alpha, dtype=np.float32)
    agx[:, 1:] = np.abs(alpha[:, 1:] - alpha[:, :-1])
    agy[1:, :] = np.abs(alpha[1:, :] - alpha[:-1, :])
    alpha_edge = np.maximum(agx, agy)

    linework = np.clip((0.48 - luma) / 0.48, 0.0, 1.0) * np.clip(saturation * 1.35, 0.0, 1.0) * alpha
    highlights = np.clip((luma - 0.78) / 0.22, 0.0, 1.0) * np.clip(saturation * 1.15, 0.0, 1.0) * alpha
    visible = np.where(alpha > 0.02, 1.0, 0.55).astype(np.float32)

    importance = (
        1.0
        + np.clip(edge * 9.0, 0.0, 2.6)
        + np.clip(alpha_edge * 7.5, 0.0, 2.8)
        + np.clip(saturation * 0.55, 0.0, 0.75) * alpha
        + linework * 1.35
        + highlights * 0.70
    ) * visible

    padded = np.pad(importance, 1, mode="edge")
    dilated = importance.copy()
    for dy in range(3):
        for dx in range(3):
            dilated = np.maximum(dilated, padded[dy : dy + height, dx : dx + width] * 0.92)
    return np.clip(dilated, 0.55, 5.25).astype(np.float32)


def render_and_score(
    background: dict,
    shapes: list[dict],
    target_rgba: np.ndarray,
    enforce_canvas_boundary: bool = False,
    importance_map: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    height, width = target_rgba.shape[:2]
    edge_context = canvas_edge_context(target_rgba) if enforce_canvas_boundary else None
    bg_rgba = np.array(list(background.get("color", [0, 0, 0, 0]))[:4], dtype=np.float32)
    top_rgb = np.empty((height, width, 3), dtype=np.float32)
    top_rgb[:] = bg_rgba[:3]
    under_rgb = np.empty((height, width, 3), dtype=np.float32)
    under_rgb[:] = bg_rgba[:3]
    top_alpha = np.full((height, width), max(0.0, min(1.0, float(bg_rgba[3]) / 255.0)), dtype=np.float32)
    under_alpha = np.full((height, width), max(0.0, min(1.0, float(bg_rgba[3]) / 255.0)), dtype=np.float32)
    top_idx = np.full((height, width), -1, dtype=np.int32)

    for idx, shape in enumerate(shapes):
        rgba = list(shape.get("color", [0, 0, 0, 255]))[:4]
        color = np.array(rgba[:3], dtype=np.float32)
        alpha = float(rgba[3]) / 255.0 if len(rgba) >= 4 else 1.0
        alpha = max(0.0, min(1.0, alpha))
        bbox, mask = shape_mask(shape, width, height)
        x0, x1, y0, y1 = bbox
        if x1 < x0 or y1 < y0 or not np.any(mask):
            continue
        top_sub = top_rgb[y0 : y1 + 1, x0 : x1 + 1]
        under_sub = under_rgb[y0 : y1 + 1, x0 : x1 + 1]
        top_alpha_sub = top_alpha[y0 : y1 + 1, x0 : x1 + 1]
        under_alpha_sub = under_alpha[y0 : y1 + 1, x0 : x1 + 1]
        idx_sub = top_idx[y0 : y1 + 1, x0 : x1 + 1]
        old_top = top_sub.copy()
        old_alpha = top_alpha_sub.copy()
        under_sub[mask] = old_top[mask]
        under_alpha_sub[mask] = old_alpha[mask]
        if alpha >= 1.0:
            top_sub[mask] = color
        elif alpha > 0.0:
            top_sub[mask] = old_top[mask] * (1.0 - alpha) + color * alpha
        if alpha > 0.0:
            top_alpha_sub[mask] = old_alpha[mask] + alpha * (1.0 - old_alpha[mask])
        idx_sub[mask] = idx

    target_rgb = target_rgba[..., :3].astype(np.float32)
    target_alpha = np.clip(target_rgba[..., 3].astype(np.float32) / 255.0, 0.0, 1.0)
    rgb_weight = np.maximum(target_alpha, 0.08)
    transparent_boost = np.where(target_alpha < 0.02, 3.25, np.where(target_alpha < 0.15, 2.35, 1.0)).astype(np.float32)
    rgb_top = np.square(top_rgb - target_rgb).sum(axis=2) * rgb_weight
    rgb_under = np.square(under_rgb - target_rgb).sum(axis=2) * rgb_weight
    alpha_scale = float(255.0 * 255.0 * 3.0 * 1.10)
    spill_scale = float(255.0 * 255.0 * 3.0 * 0.95)
    alpha_top = np.square(top_alpha - target_alpha) * alpha_scale
    alpha_under = np.square(under_alpha - target_alpha) * alpha_scale
    spill_top = np.square(np.maximum(0.0, top_alpha - target_alpha)) * transparent_boost * spill_scale
    spill_under = np.square(np.maximum(0.0, under_alpha - target_alpha)) * transparent_boost * spill_scale
    if importance_map is not None and importance_map.shape == target_alpha.shape:
        importance = np.clip(importance_map.astype(np.float32), 0.25, 8.0)
    else:
        importance = np.ones_like(target_alpha, dtype=np.float32)
    diff_top = (rgb_top + alpha_top + spill_top) * importance
    diff_under = (rgb_under + alpha_under + spill_under) * importance
    contrib_map = diff_under - diff_top
    valid = top_idx >= 0
    if np.any(valid):
        contributions = np.bincount(
            top_idx[valid].ravel(),
            weights=contrib_map[valid].ravel(),
            minlength=len(shapes),
        ).astype(np.float64)
    else:
        contributions = np.zeros(len(shapes), dtype=np.float64)
    boundary_penalties = shape_boundary_penalties(shapes, width, height, enforce_canvas_boundary, edge_context)
    if len(boundary_penalties):
        contributions -= boundary_penalties
    pixel_weights = np.where(target_alpha > 0.02, 1.0, 0.35).astype(np.float32)
    total_error = float((diff_top * pixel_weights).sum() + boundary_penalties.sum())
    scored_pixels = float((pixel_weights * importance).sum())
    return top_rgb, top_alpha, top_idx, diff_top, contributions, total_error, scored_pixels


def bbox_intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ax0, ax1, ay0, ay1 = a
    bx0, bx1, by0, by1 = b
    return ax0 <= bx1 and ax1 >= bx0 and ay0 <= by1 and ay1 >= by0


def render_and_score_region(
    background: dict,
    shapes: list[dict],
    target_rgba: np.ndarray,
    bbox: tuple[int, int, int, int],
    enforce_canvas_boundary: bool = False,
    importance_map: np.ndarray | None = None,
):
    x0, x1, y0, y1 = bbox
    if x1 < x0 or y1 < y0:
        return render_and_score(background, [], target_rgba[:1, :1], enforce_canvas_boundary=False)

    height, width = target_rgba.shape[:2]
    x0 = max(0, min(width - 1, int(x0)))
    x1 = max(0, min(width - 1, int(x1)))
    y0 = max(0, min(height - 1, int(y0)))
    y1 = max(0, min(height - 1, int(y1)))
    if x1 < x0 or y1 < y0:
        return render_and_score(background, [], target_rgba[:1, :1], enforce_canvas_boundary=False)

    crop_bbox = (x0, x1, y0, y1)
    crop_w = x1 - x0 + 1
    crop_h = y1 - y0 + 1
    crop_shapes = []
    for shape in shapes:
        current_bbox = shape_bbox(shape, width, height)
        if not bbox_intersects(current_bbox, crop_bbox):
            continue
        shifted = copy_shape(shape)
        shifted["data"] = list(shifted["data"])
        shifted["data"][0] = float(shifted["data"][0]) - x0
        shifted["data"][1] = float(shifted["data"][1]) - y0
        crop_shapes.append(shifted)

    crop_bg = dict(background)
    crop_bg["data"] = [0, 0, crop_w, crop_h]
    crop_importance = importance_map[y0 : y1 + 1, x0 : x1 + 1] if importance_map is not None else None
    return render_and_score(
        crop_bg,
        crop_shapes,
        target_rgba[y0 : y1 + 1, x0 : x1 + 1],
        enforce_canvas_boundary=False,
        importance_map=crop_importance,
    )


def normalized_error(total_error: float, scored_pixels: float) -> float:
    denom = max(1.0, scored_pixels * 4.0)
    return total_error / float(denom)


def score_shape_list(
    background: dict,
    shapes: list[dict],
    target_rgba: np.ndarray,
    enforce_canvas_boundary: bool = False,
    importance_map: np.ndarray | None = None,
) -> float:
    _, _, _, _, _, total_error, scored_pixels = render_and_score(
        background,
        shapes,
        target_rgba,
        enforce_canvas_boundary=enforce_canvas_boundary,
        importance_map=importance_map,
    )
    return normalized_error(total_error, scored_pixels)


def remove_lowest_ranked_batch(
    background: dict,
    shapes: list[dict],
    target_rgba: np.ndarray,
    ranked_indices: np.ndarray,
    requested_remove_count: int,
    current_error: float,
    enforce_canvas_boundary: bool = False,
    importance_map: np.ndarray | None = None,
) -> tuple[list[dict], float]:
    if requested_remove_count <= 0 or len(shapes) <= 1:
        return shapes, current_error

    requested_remove_count = max(1, min(int(requested_remove_count), len(shapes) - 1))
    tolerance = max(0.0025, current_error * 0.0018)
    trial_sizes = []
    size = requested_remove_count
    while size >= 1:
        if size not in trial_sizes:
            trial_sizes.append(size)
        if size == 1:
            break
        size = max(1, size // 2)

    for remove_count in trial_sizes:
        remove_idx = set(int(i) for i in ranked_indices[:remove_count])
        candidate = [shape for idx, shape in enumerate(shapes) if idx not in remove_idx]
        candidate_error = score_shape_list(
            background,
            candidate,
            target_rgba,
            enforce_canvas_boundary=enforce_canvas_boundary,
            importance_map=importance_map,
        )
        if candidate_error <= current_error + tolerance or remove_count == 1:
            return candidate, candidate_error
    return shapes, current_error


def prune_to_target(
    background: dict,
    shapes: list[dict],
    target_rgba: np.ndarray,
    target_count: int,
    enforce_canvas_boundary: bool = False,
    importance_map: np.ndarray | None = None,
) -> tuple[list[dict], float]:
    working = list(shapes)
    working, _, _ = remove_fully_covered_layers(
        background,
        working,
        target_rgba,
        enforce_canvas_boundary=enforce_canvas_boundary,
        importance_map=importance_map,
        max_batch=96,
        removal_limit=max(0, len(working) - target_count),
    )
    if len(working) <= target_count:
        _, _, _, _, _, total_error, scored_pixels = render_and_score(
            background,
            working,
            target_rgba,
            enforce_canvas_boundary=enforce_canvas_boundary,
            importance_map=importance_map,
        )
        return working, normalized_error(total_error, scored_pixels)

    while len(working) > target_count:
        _, _, _, _, contributions, total_error, scored_pixels = render_and_score(
            background,
            working,
            target_rgba,
            enforce_canvas_boundary=enforce_canvas_boundary,
            importance_map=importance_map,
        )
        current_error = normalized_error(total_error, scored_pixels)
        excess = len(working) - target_count
        order = np.argsort(contributions)
        zeroish = int(np.count_nonzero(contributions[order] <= 1e-6))
        if zeroish > 0:
            remove_count = min(excess, zeroish)
        else:
            # 큰 눈먼 배치는 머리카락·에지를 깎는다 — 적당히 떼고 검증해 줄인다
            remove_count = min(excess, max(1, min(72, excess // 8 if excess > 24 else 1)))
        working, current_error = remove_lowest_ranked_batch(
            background,
            working,
            target_rgba,
            order,
            remove_count,
            current_error,
            enforce_canvas_boundary=enforce_canvas_boundary,
            importance_map=importance_map,
        )

    _, _, _, _, _, total_error, scored_pixels = render_and_score(
        background,
        working,
        target_rgba,
        enforce_canvas_boundary=enforce_canvas_boundary,
        importance_map=importance_map,
    )
    return working, normalized_error(total_error, scored_pixels)


def visible_shape_pixels(top_idx: np.ndarray, shape_count: int) -> np.ndarray:
    valid = top_idx >= 0
    if not np.any(valid):
        return np.zeros(shape_count, dtype=np.int64)
    return np.bincount(top_idx[valid].ravel(), minlength=shape_count).astype(np.int64)


def remove_fully_covered_layers(
    background: dict,
    shapes: list[dict],
    target_rgba: np.ndarray,
    enforce_canvas_boundary: bool = False,
    importance_map: np.ndarray | None = None,
    max_batch: int = 64,
    removal_limit: int | None = None,
) -> tuple[list[dict], float, dict]:
    if not shapes:
        return [], 0.0, {"removed": 0, "before": 0, "after": 0, "score_before": 0.0, "score_after": 0.0}
    if removal_limit is not None and removal_limit <= 0:
        current_error = score_shape_list(
            background,
            shapes,
            target_rgba,
            enforce_canvas_boundary=enforce_canvas_boundary,
            importance_map=importance_map,
        )
        return list(shapes), current_error, {
            "removed": 0,
            "rejected": 0,
            "before": len(shapes),
            "after": len(shapes),
            "score_before": current_error,
            "score_after": current_error,
            "skipped": "candidate is already at or under the target layer budget",
        }

    working = list(shapes)
    initial_count = len(working)
    current_error = score_shape_list(
        background,
        working,
        target_rgba,
        enforce_canvas_boundary=enforce_canvas_boundary,
        importance_map=importance_map,
    )
    removed_total = 0
    rejected_total = 0

    while working:
        _, _, top_idx, _, _, _, _ = render_and_score(
            background,
            working,
            target_rgba,
            enforce_canvas_boundary=enforce_canvas_boundary,
            importance_map=importance_map,
        )
        visible_pixels = visible_shape_pixels(top_idx, len(working))
        hidden_indices = [idx for idx, pixels in enumerate(visible_pixels) if int(pixels) <= 0]
        if not hidden_indices:
            break

        progress = False
        while hidden_indices:
            remaining_allowed = None if removal_limit is None else max(0, int(removal_limit) - removed_total)
            if remaining_allowed is not None and remaining_allowed <= 0:
                hidden_indices = []
                break
            batch_size = max(1, min(max_batch, len(hidden_indices)))
            if remaining_allowed is not None:
                batch_size = min(batch_size, remaining_allowed)
            batch = hidden_indices[:batch_size]
            remove_idx = set(batch)
            candidate = [shape for idx, shape in enumerate(working) if idx not in remove_idx]
            candidate_error = score_shape_list(
                background,
                candidate,
                target_rgba,
                enforce_canvas_boundary=enforce_canvas_boundary,
                importance_map=importance_map,
            )
            tolerance = max(0.0015, current_error * 0.00075)
            if candidate_error <= current_error + tolerance:
                working = candidate
                current_error = candidate_error
                removed_total += len(remove_idx)
                progress = True
                break
            if len(batch) == 1:
                rejected_total += 1
                hidden_indices = hidden_indices[1:]
                continue
            max_batch = max(1, len(batch) // 2)

        if not progress and not hidden_indices:
            break

    return working, current_error, {
        "removed": removed_total,
        "rejected": rejected_total,
        "before": initial_count,
        "after": len(working),
        "score_before": score_shape_list(
            background,
            shapes,
            target_rgba,
            enforce_canvas_boundary=enforce_canvas_boundary,
            importance_map=importance_map,
        ),
        "score_after": current_error,
    }
