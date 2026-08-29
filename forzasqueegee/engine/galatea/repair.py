# SPDX-License-Identifier: MIT
# kloudys-forza-painter-suite(MIT)의 생성 파이프라인을 이식한 파생물이다.
# Portions © 2021 AE (forza-painter) · Sam Twidale (geometrize-lib) ·
# Michael Fogleman (Primitive). 전문·유래는 THIRD_PARTY_NOTICES.md.
"""국소 수리·도형 가족 교체·평면색 안정화·캔버스 경계 강제.

원본: forza_generator_v2.py — 로직·문턱 전부 원본 그대로에 두 접목이 있다
(수정 2): `repair_shapes`는 국소 승자를 채택하기 전에 게임 그리드로 스냅해
전역 재채점에 넣고(양자화가 이득을 잠식하면 기존 채택 검사에서 떨어진다),
`stabilize_flat_region_colors`는 스냅하는 색을 HSB 스텝 그리드로 적는다.
"""

from __future__ import annotations

import math

import numpy as np

from .base import ELLIPSE, RECTANGLE, ROTATED_ELLIPSE, ROTATED_RECTANGLE
from .geometry import (canvas_edge_context, copy_shape, fit_shape_inside_canvas,
                       rotated_bbox, rotated_rect_bbox, shape_boundary_penalties,
                       shape_boundary_penalty, shape_mask)
from .scoring import (normalized_error, render_and_score,
                      render_and_score_region)


def force_opaque_drawables(shapes: list[dict]) -> list[dict]:
    out = []
    for shape in shapes:
        fixed = copy_shape(shape)
        color = list(fixed.get("color", [0, 0, 0, 255]))
        if len(color) < 4:
            color += [255] * (4 - len(color))
        if float(color[3]) > 0:
            color[3] = 255
        fixed["color"] = color
        out.append(fixed)
    return out


def stabilize_flat_region_colors(
    shapes: list[dict],
    target_rgba: np.ndarray,
    force_opaque: bool = True,
    min_pixels: int = 48,
) -> tuple[list[dict], dict]:
    """평면 아트의 큰 저분산 도형을 지배 원색으로 스냅 (원본 그대로).

    바꾸는 색은 **HSB 스텝 그리드**로 스냅해 적는다 (수정 2 — 원본은 원색
    중앙값 그대로 적는다).
    """
    if not shapes:
        return shapes, {"enabled": True, "changed": 0, "checked": 0, "skipped": 0}

    height, width = target_rgba.shape[:2]
    target = np.clip(target_rgba, 0, 255).astype(np.uint8)
    stabilized: list[dict] = []
    changed = 0
    checked = 0
    skipped = 0
    protected_large_underpaint = 0
    protected_extreme_snap = 0
    canvas_pixels = max(1, width * height)
    total_shapes = max(1, len(shapes))
    for shape_index, shape in enumerate(shapes):
        fixed = copy_shape(shape)
        color = list(fixed.get("color", [0, 0, 0, 255]))
        if len(color) < 4 or float(color[3]) <= 0:
            stabilized.append(fixed)
            skipped += 1
            continue
        bbox, mask = shape_mask(fixed, width, height)
        x0, x1, y0, y1 = bbox
        if x1 < x0 or y1 < y0 or not np.any(mask):
            stabilized.append(fixed)
            skipped += 1
            continue
        local = target[y0 : y1 + 1, x0 : x1 + 1]
        visible = mask & (local[..., 3] > 32)
        pixel_count = int(np.count_nonzero(visible))
        if pixel_count < min_pixels:
            stabilized.append(fixed)
            skipped += 1
            continue
        checked += 1
        layer_fraction = float(shape_index) / float(max(1, total_shapes - 1))
        pixel_fraction = float(pixel_count) / float(canvas_pixels)
        rgb = local[..., :3][visible].astype(np.uint8)
        channel_std = float(np.mean(np.std(rgb.astype(np.float32), axis=0))) if rgb.size else 999.0
        bins = (rgb // 10).astype(np.uint16)
        packed = (bins[:, 0] << 10) | (bins[:, 1] << 5) | bins[:, 2]
        values, counts = np.unique(packed, return_counts=True)
        if values.size == 0:
            stabilized.append(fixed)
            skipped += 1
            continue
        best = int(np.argmax(counts))
        dominant_fraction = float(counts[best]) / float(max(1, pixel_count))
        if dominant_fraction < 0.46 or (dominant_fraction < 0.62 and channel_std > 22.0):
            stabilized.append(fixed)
            skipped += 1
            continue
        dominant_rgb = rgb[packed == values[best]]
        if dominant_rgb.size == 0:
            stabilized.append(fixed)
            skipped += 1
            continue
        new_rgb = [int(round(v)) for v in np.median(dominant_rgb, axis=0)]
        old_rgb = [int(v) for v in color[:3]]
        old_luma = sum(old_rgb) / 3.0
        new_luma = sum(new_rgb) / 3.0
        snap_delta = sum(abs(a - b) for a, b in zip(old_rgb, new_rgb))
        snaps_to_extreme = new_luma <= 18.0 or new_luma >= 237.0
        old_was_mid = 48.0 < old_luma < 207.0

        # 큰 초기 레이어는 밑칠이다 — 강한 근거 없이 흑백으로 스냅하지 않는다
        if pixel_fraction >= 0.010 and layer_fraction <= 0.12:
            if dominant_fraction < 0.88 or channel_std > 8.0:
                stabilized.append(fixed)
                skipped += 1
                protected_large_underpaint += 1
                continue
        elif pixel_fraction >= 0.004 and layer_fraction <= 0.22:
            if dominant_fraction < 0.78 or channel_std > 14.0:
                stabilized.append(fixed)
                skipped += 1
                protected_large_underpaint += 1
                continue

        # 중간 회색 → 순흑/순백 스냅은 번짐 실패 모드 — 압도적일 때만 허용
        if snaps_to_extreme and old_was_mid and snap_delta >= 90:
            if dominant_fraction < 0.82 or channel_std > 10.0:
                stabilized.append(fixed)
                skipped += 1
                protected_extreme_snap += 1
                continue

        if snap_delta >= 3:
            color[:3] = new_rgb
            if force_opaque:
                color[3] = 255
            fixed["color"] = color
            changed += 1
        stabilized.append(fixed)

    return stabilized, {
        "enabled": True,
        "changed": changed,
        "checked": checked,
        "skipped": skipped,
        "min_pixels": min_pixels,
        "protected_large_underpaint": protected_large_underpaint,
        "protected_extreme_snap": protected_extreme_snap,
    }


def shape_visual_extents(shape: dict) -> tuple[float, float, float, float, float] | None:
    data = list(shape.get("data", []))
    if len(data) < 4:
        return None
    cx, cy, a, b = [float(v) for v in data[:4]]
    rot = float(data[4]) if len(data) >= 5 else 0.0
    if int(shape.get("type", ROTATED_ELLIPSE)) in (RECTANGLE, ROTATED_RECTANGLE):
        half_w = max(0.5, a * 0.5)
        half_h = max(0.5, b * 0.5)
    else:
        half_w = max(0.5, a)
        half_h = max(0.5, b)
    return cx, cy, half_w, half_h, rot


def shape_family_variant(shape: dict, shape_type: int) -> dict | None:
    extents = shape_visual_extents(shape)
    if extents is None:
        return None
    cx, cy, half_w, half_h, rot = extents
    variant = copy_shape(shape)
    variant["type"] = shape_type
    if shape_type == RECTANGLE:
        variant["data"] = [cx, cy, max(1.0, half_w * 2.0), max(1.0, half_h * 2.0)]
    elif shape_type == ROTATED_RECTANGLE:
        variant["data"] = [cx, cy, max(1.0, half_w * 2.0), max(1.0, half_h * 2.0), rot % 360.0]
    elif shape_type == ELLIPSE:
        variant["data"] = [cx, cy, max(1.0, half_w), max(1.0, half_h), 0.0]
    elif shape_type == ROTATED_ELLIPSE:
        variant["data"] = [cx, cy, max(1.0, half_w), max(1.0, half_h), rot % 360.0]
    else:
        return None
    return variant


def shape_family_variants(shape: dict, prefer_smooth_shapes: bool = False) -> list[dict]:
    current_type = int(shape.get("type", ROTATED_ELLIPSE))
    extents = shape_visual_extents(shape)
    if extents is None:
        return []
    _, _, half_w, half_h, rot = extents
    aspect = max(half_w, half_h) / max(1.0, min(half_w, half_h))
    near_axis = abs((rot % 90.0)) <= 2.0 or abs((rot % 90.0) - 90.0) <= 2.0

    if prefer_smooth_shapes:
        candidates = [ROTATED_ELLIPSE, ELLIPSE, ROTATED_RECTANGLE]
        if near_axis:
            candidates.append(RECTANGLE)
        if aspect >= 4.0:
            candidates = [ROTATED_ELLIPSE, ROTATED_RECTANGLE, ELLIPSE, RECTANGLE]
    else:
        candidates = [ROTATED_ELLIPSE, ROTATED_RECTANGLE]
        if near_axis:
            candidates.extend([ELLIPSE, RECTANGLE])
        if aspect <= 1.20:
            candidates = [ROTATED_ELLIPSE, ELLIPSE, ROTATED_RECTANGLE, RECTANGLE]
        elif aspect >= 2.20:
            candidates = [ROTATED_RECTANGLE, ROTATED_ELLIPSE, RECTANGLE, ELLIPSE]

    out = []
    seen: set[int] = set()
    for shape_type in candidates:
        if shape_type == current_type or shape_type in seen:
            continue
        seen.add(shape_type)
        variant = shape_family_variant(shape, shape_type)
        if variant is not None:
            out.append(variant)
    return out


def local_error_value(diff_map: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
    x0, x1, y0, y1 = bbox
    if x1 < x0 or y1 < y0:
        return 0.0
    sub = diff_map[y0 : y1 + 1, x0 : x1 + 1]
    if sub.size == 0:
        return 0.0
    return float(sub.mean())


def shape_homogeneity_penalty(shape: dict, target_rgba: np.ndarray) -> float:
    height, width = target_rgba.shape[:2]
    bbox, mask = shape_mask(shape, width, height)
    x0, x1, y0, y1 = bbox
    if x1 < x0 or y1 < y0 or not np.any(mask):
        return 0.0
    target_mask = target_rgba[..., 3] > 0.5
    sub_target_mask = target_mask[y0 : y1 + 1, x0 : x1 + 1]
    valid = mask & sub_target_mask
    if not np.any(valid):
        return 0.0
    sub_rgb = target_rgba[y0 : y1 + 1, x0 : x1 + 1, :3].astype(np.float32)
    pixels = sub_rgb[valid]
    if len(pixels) < 4:
        return 0.0
    mean = pixels.mean(axis=0)
    sq = np.square(pixels - mean).sum(axis=1)
    return float(sq.mean())


def expanded_shape_bbox(shape: dict, width: int, height: int, move_step: float, radius_step: float) -> tuple[int, int, int, int]:
    data = list(shape.get("data", []))
    if len(data) < 4:
        return (0, width - 1, 0, height - 1)
    cx, cy, rx, ry = [float(v) for v in data[:4]]
    rot_deg = float(data[4]) if len(data) >= 5 else 0.0
    shape_type = int(shape.get("type", ROTATED_ELLIPSE))
    if shape_type in (RECTANGLE, ROTATED_RECTANGLE):
        half_w = max(0.5, (rx + radius_step * 2.0) * 0.5) + move_step
        half_h = max(0.5, (ry + radius_step * 2.0) * 0.5) + move_step
        x0, x1, y0, y1 = rotated_rect_bbox(cx, cy, half_w, half_h, rot_deg, width, height)
    else:
        x0, x1, y0, y1 = rotated_bbox(cx, cy, rx + radius_step + move_step, ry + radius_step + move_step, rot_deg, width, height)
    margin = max(4, int(math.ceil(max(move_step, radius_step))))
    return (
        max(0, x0 - margin),
        min(width - 1, x1 + margin),
        max(0, y0 - margin),
        min(height - 1, y1 + margin),
    )


def rank_repair_targets(
    shapes: list[dict],
    top_idx: np.ndarray,
    diff_top: np.ndarray,
    top_alpha: np.ndarray,
    target_rgba: np.ndarray,
    max_shapes: int,
    enforce_canvas_boundary: bool = False,
    importance_map: np.ndarray | None = None,
) -> list[int]:
    target_alpha = np.clip(target_rgba[..., 3].astype(np.float32) / 255.0, 0.0, 1.0)
    valid = top_idx >= 0
    if not np.any(valid):
        return []
    shape_error = np.bincount(top_idx[valid].ravel(), weights=diff_top[valid].ravel(), minlength=len(shapes)).astype(np.float64)
    if importance_map is not None and importance_map.shape == target_alpha.shape:
        visible_weights = np.clip(importance_map.astype(np.float32), 0.25, 8.0)
        visible_pixels = np.bincount(top_idx[valid].ravel(), weights=visible_weights[valid].ravel(), minlength=len(shapes)).astype(np.float64)
    else:
        visible_pixels = np.bincount(top_idx[valid].ravel(), minlength=len(shapes)).astype(np.float64)
    spill = np.maximum(0.0, top_alpha - target_alpha)
    spill_pixels = np.bincount(top_idx[valid].ravel(), weights=spill[valid].ravel(), minlength=len(shapes)).astype(np.float64)
    edge_context = canvas_edge_context(target_rgba) if enforce_canvas_boundary else None
    boundary_penalty = shape_boundary_penalties(shapes, target_rgba.shape[1], target_rgba.shape[0], enforce_canvas_boundary, edge_context)
    area = np.array([max(1.0, float(shape["data"][2]) * float(shape["data"][3])) for shape in shapes], dtype=np.float64)
    homogeneity = np.array([shape_homogeneity_penalty(shape, target_rgba) for shape in shapes], dtype=np.float64)
    index = np.arange(len(shapes), dtype=np.float64)
    early_weight = np.where(index < 250, 1.8 - (index / 250.0) * 0.8, 1.0)
    size_weight = np.sqrt(np.maximum(1.0, area))
    score = (
        (shape_error * 1.05)
        + (spill_pixels * size_weight * 9500.0 * early_weight)
        + (homogeneity * size_weight * early_weight)
        + (boundary_penalty * 1.75)
    ) * (1.0 + np.sqrt(np.maximum(1.0, visible_pixels)) / 6.0) / np.sqrt(area)
    ranked = np.argsort(score)[::-1]
    ranked = [
        int(idx)
        for idx in ranked
        if (shape_error[idx] > 0 or homogeneity[idx] > 0 or spill_pixels[idx] > 0 or boundary_penalty[idx] > 0)
        and (visible_pixels[idx] > 0 or boundary_penalty[idx] > 0)
    ]
    return ranked[:max_shapes]


def repair_shapes(
    background: dict,
    shapes: list[dict],
    target_rgba: np.ndarray,
    max_shapes: int = 8,
    rounds: int = 1,
    enforce_canvas_boundary: bool = False,
    prefer_smooth_shapes: bool = False,
    importance_map: np.ndarray | None = None,
    allow_alpha_repair: bool = True,
    quantize=None,
) -> tuple[list[dict], float, dict]:
    """국소 수리 (원본 그대로) + **양자화 채택 게이트** (수정 2).

    `quantize`는 채점 공간 도형 → 게임 그리드 스냅 함수다 (부르는 쪽이
    스케일 폐포로 만든다). 국소 승자를 **양자화해서** 전역 재채점에 넣으므로,
    양자화가 이득을 잠식하는 후보는 기존 채택 검사에서 그대로 떨어진다 —
    양자화 후 재채점, 이득 잠식이면 기각이다.
    """
    if not shapes:
        _, _, _, _, _, total_error, scored_pixels = render_and_score(
            background,
            shapes,
            target_rgba,
            enforce_canvas_boundary=enforce_canvas_boundary,
            importance_map=importance_map,
        )
        error = normalized_error(total_error, scored_pixels)
        return shapes, error, {"enabled": True, "touched": 0, "improvements": 0, "before": error, "after": error}

    working = [copy_shape(shape) for shape in shapes]
    _, top_alpha, top_idx, diff_top, _, total_error, scored_pixels = render_and_score(
        background,
        working,
        target_rgba,
        enforce_canvas_boundary=enforce_canvas_boundary,
        importance_map=importance_map,
    )
    best_error = normalized_error(total_error, scored_pixels)
    before_error = best_error
    edge_context = canvas_edge_context(target_rgba) if enforce_canvas_boundary else None

    improvements = 0
    family_changes = 0
    boundary_fits = 0
    quantize_rejects = 0
    touched_indices: set[int] = set()
    for _round in range(rounds):
        ranked = rank_repair_targets(
            working,
            top_idx,
            diff_top,
            top_alpha,
            target_rgba,
            max_shapes,
            enforce_canvas_boundary=enforce_canvas_boundary,
            importance_map=importance_map,
        )
        changed = False
        for idx in ranked:
            if idx >= len(working):
                continue
            touched_indices.add(idx)
            shape = working[idx]
            data = list(shape.get("data", []))
            if len(data) < 4:
                continue
            x, y, rx, ry = [float(v) for v in data[:4]]
            rot = float(data[4]) if len(data) >= 5 else 0.0
            move_step = max(1.0, round(max(rx, ry) * 0.014))
            radius_step = max(1.0, round(max(rx, ry) * 0.03))
            rot_step = 2.0
            local_bbox = expanded_shape_bbox(shape, target_rgba.shape[1], target_rgba.shape[0], move_step, radius_step)
            local_best_score = local_error_value(diff_top, local_bbox)
            if enforce_canvas_boundary:
                local_area = max(1.0, float((local_bbox[1] - local_bbox[0] + 1) * (local_bbox[3] - local_bbox[2] + 1)))
                local_best_score += shape_boundary_penalty(shape, target_rgba.shape[1], target_rgba.shape[0], True, edge_context) / local_area
            alpha0 = float(shape.get("color", [0, 0, 0, 255])[3])
            alpha_step = max(10.0, round(alpha0 * 0.12))
            proposals = [
                (False,  move_step, 0.0, 0.0, 0.0, 0.0),
                (False, -move_step, 0.0, 0.0, 0.0, 0.0),
                (False, 0.0,  move_step, 0.0, 0.0, 0.0),
                (False, 0.0, -move_step, 0.0, 0.0, 0.0),
                (False,  move_step * 0.5, 0.0, 0.0, 0.0, 0.0),
                (False, -move_step * 0.5, 0.0, 0.0, 0.0, 0.0),
                (False, 0.0,  move_step * 0.5, 0.0, 0.0, 0.0),
                (False, 0.0, -move_step * 0.5, 0.0, 0.0, 0.0),
                (False, 0.0, 0.0, -radius_step, 0.0, 0.0),
                (False, 0.0, 0.0, 0.0, -radius_step, 0.0),
                (False, 0.0, 0.0, -radius_step, -radius_step, 0.0),
                (False, 0.0, 0.0, 0.0, 0.0,  rot_step),
                (False, 0.0, 0.0, 0.0, 0.0, -rot_step),
                (False, 0.0, 0.0, -radius_step, 0.0, rot_step),
                (False, 0.0, 0.0, 0.0, -radius_step, -rot_step),
            ]
            if allow_alpha_repair:
                proposals.extend([
                    (False, 0.0, 0.0, -radius_step, -radius_step, 0.0, -alpha_step),
                    (False, 0.0, 0.0, 0.0, 0.0, 0.0, -alpha_step),
                    (False, 0.0, 0.0, 0.0, 0.0, 0.0, -alpha_step * 2.0),
                    (False,  move_step * 0.5, 0.0, -radius_step, 0.0, 0.0, -alpha_step),
                    (False, -move_step * 0.5, 0.0, -radius_step, 0.0, 0.0, -alpha_step),
                    (False, 0.0,  move_step * 0.5, 0.0, -radius_step, 0.0, -alpha_step),
                    (False, 0.0, -move_step * 0.5, 0.0, -radius_step, 0.0, -alpha_step),
                ])

            local_best = copy_shape(shape)
            original_local_score = local_best_score
            local_best_deleted = False
            trial_shapes: list[dict | None] = shape_family_variants(shape, prefer_smooth_shapes=prefer_smooth_shapes)
            if enforce_canvas_boundary and shape_boundary_penalty(shape, target_rgba.shape[1], target_rgba.shape[0], True, edge_context) > 0:
                trial_shapes.append(fit_shape_inside_canvas(shape, target_rgba.shape[1], target_rgba.shape[0], edge_context))
            for proposal in proposals:
                if len(proposal) == 6:
                    delete_shape, dx, dy, drx, dry, drot = proposal
                    dalpha = 0.0
                else:
                    delete_shape, dx, dy, drx, dry, drot, dalpha = proposal
                if delete_shape:
                    trial_shapes.append(None)
                    continue
                trial = copy_shape(shape)
                trial["data"] = [
                    x + dx,
                    y + dy,
                    max(1.0, rx + drx),
                    max(1.0, ry + dry),
                    (rot + drot) % 360.0,
                ]
                trial["color"] = list(trial.get("color", [0, 0, 0, 255]))
                if len(trial["color"]) < 4:
                    trial["color"] += [255] * (4 - len(trial["color"]))
                trial["color"][3] = int(max(0, min(255, round(alpha0 + dalpha)))) if allow_alpha_repair else int(max(1, min(255, round(alpha0))))
                trial_shapes.append(trial)

            for trial in trial_shapes:
                prev = working[idx]
                if trial is None:
                    del working[idx]
                else:
                    working[idx] = trial
                _, _, _, trial_diff, _, _, _ = render_and_score_region(
                    background,
                    working,
                    target_rgba,
                    local_bbox,
                    importance_map=importance_map,
                )
                trial_local_score = float(trial_diff.mean()) if trial_diff.size else local_best_score
                if enforce_canvas_boundary and trial is not None:
                    local_area = max(1.0, float((local_bbox[1] - local_bbox[0] + 1) * (local_bbox[3] - local_bbox[2] + 1)))
                    trial_local_score += shape_boundary_penalty(trial, target_rgba.shape[1], target_rgba.shape[0], True, edge_context) / local_area
                if trial_local_score + 1e-9 < local_best_score:
                    local_best_score = trial_local_score
                    local_best = None if trial is None else copy_shape(trial)
                    local_best_deleted = trial is None
                if trial is None:
                    working.insert(idx, prev)
                else:
                    working[idx] = prev

            if local_best_score + 1e-9 < original_local_score:
                prev = working[idx]
                if not local_best_deleted and local_best is not None and quantize is not None:
                    # 채택 전에 게임 그리드로 스냅 — 아래 전역 검사가 양자화
                    # 잠식까지 본다 (잠식이면 원상 복구된다)
                    local_best = quantize(local_best)
                changed_family = bool(
                    not local_best_deleted
                    and local_best is not None
                    and int(local_best.get("type", ROTATED_ELLIPSE)) != int(prev.get("type", ROTATED_ELLIPSE))
                )
                if local_best_deleted:
                    del working[idx]
                else:
                    working[idx] = local_best
                _, trial_alpha, trial_top_idx, trial_diff, _, trial_total_error, trial_scored_pixels = render_and_score(
                    background,
                    working,
                    target_rgba,
                    enforce_canvas_boundary=enforce_canvas_boundary,
                    importance_map=importance_map,
                )
                trial_error = normalized_error(trial_total_error, trial_scored_pixels)
                if trial_error <= best_error + 1e-6 or trial_error + 1e-9 < best_error:
                    best_error = trial_error
                    top_alpha = trial_alpha
                    top_idx = trial_top_idx
                    diff_top = trial_diff
                    improvements += 1
                    if changed_family:
                        family_changes += 1
                    if enforce_canvas_boundary and shape_boundary_penalty(prev, target_rgba.shape[1], target_rgba.shape[0], True, edge_context) > 0:
                        boundary_fits += 1
                    changed = True
                else:
                    if quantize is not None and not local_best_deleted:
                        quantize_rejects += 1
                    if local_best_deleted:
                        working.insert(idx, prev)
                    else:
                        working[idx] = prev

        if not changed:
            break

    summary = {
        "enabled": True,
        "touched": len(touched_indices),
        "improvements": improvements,
        "family_changes": family_changes,
        "boundary_fits": boundary_fits,
        "quantize_rejects": quantize_rejects,
        "alpha_repair": allow_alpha_repair,
        "canvas_boundary_enforced": enforce_canvas_boundary,
        "before": before_error,
        "after": best_error,
    }
    return working, best_error, summary


def enforce_shapes_inside_canvas(
    background: dict,
    shapes: list[dict],
    target_rgba: np.ndarray,
    importance_map: np.ndarray | None = None,
) -> tuple[list[dict], float, dict]:
    width = target_rgba.shape[1]
    height = target_rgba.shape[0]
    edge_context = canvas_edge_context(target_rgba)
    working: list[dict] = []
    fitted_count = 0
    remaining_penalty = 0.0
    for shape in shapes:
        before_penalty = shape_boundary_penalty(shape, width, height, True, edge_context)
        if before_penalty > 0.0:
            fitted = fit_shape_inside_canvas(shape, width, height, edge_context)
            after_penalty = shape_boundary_penalty(fitted, width, height, True, edge_context)
            if after_penalty < before_penalty:
                working.append(fitted)
                fitted_count += 1
                remaining_penalty += after_penalty
                continue
            remaining_penalty += before_penalty
        working.append(copy_shape(shape))
    _, _, _, _, _, total_error, scored_pixels = render_and_score(
        background,
        working,
        target_rgba,
        enforce_canvas_boundary=True,
        importance_map=importance_map,
    )
    error = normalized_error(total_error, scored_pixels)
    return working, error, {
        "enabled": True,
        "fitted": fitted_count,
        "remaining_penalty": remaining_penalty,
    }
