# SPDX-License-Identifier: MIT
# kloudys-forza-painter-suite(MIT)의 생성 파이프라인을 이식한 파생물이다.
# Portions © 2021 AE (forza-painter) · Sam Twidale (geometrize-lib) ·
# Michael Fogleman (Primitive). 전문·유래는 THIRD_PARTY_NOTICES.md.
"""원화 전처리 + 원화 분류 + 세부 열지도.

원본: forza_generator_v2.py(resize_source_for_generation·
remove_alpha_fringe_noise·apply_preprocess·apply_logo_hard_edges·
source_art_profile·downscale_rgba) + detail_heatmap.py(normalize_map·
build_detail_heatmap·heatmap_to_rgba·apply_detail_guidance) — 전부 원본 그대로.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def resize_source_for_generation(image_path: Path, max_resolution: int) -> np.ndarray:
    img = Image.open(image_path).convert("RGBA")
    w, h = img.size
    max_dim = max(w, h)
    if max_resolution > 0 and max_dim > max_resolution:
        scale = max_resolution / float(max_dim)
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        img = img.resize((nw, nh), Image.Resampling.BICUBIC)
    return np.asarray(img, dtype=np.float32)


def remove_alpha_fringe_noise(rgba: np.ndarray) -> tuple[np.ndarray, dict]:
    """배경 제거기가 남긴 저알파 부스러기 정리 (원본 독스트링 요약)."""
    if rgba.ndim != 3 or rgba.shape[2] < 4:
        return rgba, {"enabled": False, "changed": False, "removed_pixels": 0}

    alpha = np.clip(rgba[..., 3], 0, 255).astype(np.uint8)
    total = int(alpha.size)
    if total == 0 or int(np.min(alpha)) >= 250:
        return rgba, {"enabled": True, "changed": False, "removed_pixels": 0, "removed_fraction": 0.0}

    hard_drop = alpha <= 16
    core = alpha >= 96
    if np.any(core):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        near_core = cv2.dilate(core.astype(np.uint8), kernel, iterations=1).astype(bool)
        soft_haze = (alpha < 48) & ~near_core
        drop = hard_drop | soft_haze
    else:
        drop = hard_drop

    removed = int(np.count_nonzero(drop & (alpha > 0)))
    if removed <= 0:
        return rgba, {"enabled": True, "changed": False, "removed_pixels": 0, "removed_fraction": 0.0}

    cleaned = rgba.copy()
    cleaned[drop, 3] = 0.0
    cleaned[drop, :3] = 0.0
    return cleaned, {
        "enabled": True,
        "changed": True,
        "removed_pixels": removed,
        "removed_fraction": round(removed / float(max(1, total)), 6),
        "hard_threshold": 16,
        "soft_threshold": 48,
        "core_threshold": 96,
    }


def apply_preprocess(rgba: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return rgba

    rgb = np.clip(rgba[..., :3], 0, 255).astype(np.uint8)
    alpha = np.clip(rgba[..., 3], 0, 255).astype(np.uint8)

    if mode == "luma_bands":
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        l = lab[..., 0].astype(np.float32)
        levels = 64.0
        step = 256.0 / levels
        lq = np.floor(l / step) * step + step * 0.5
        blur = cv2.GaussianBlur(l, (0, 0), 1.1)
        gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
        edge = np.sqrt(gx * gx + gy * gy)
        edge = np.clip((edge - 3.0) / 18.0, 0.0, 1.0)
        band_weight = 0.16 + edge * 0.34
        l_out = lq * band_weight + l * (1.0 - band_weight)
        l_out = (l_out - 128.0) * 1.005 + 128.0
        lab[..., 0] = np.clip(l_out, 0, 255).astype(np.uint8)
        rgb_out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    else:
        raise ValueError(f"unsupported preprocess mode: {mode}")

    out = np.dstack([rgb_out, alpha]).astype(np.float32)
    return out


def apply_logo_hard_edges(rgba: np.ndarray, alpha_threshold: int = 96) -> np.ndarray:
    """투명 로고의 반투명 테두리를 불투명으로 스냅 (원본 그대로)."""
    rgb = np.clip(rgba[..., :3], 0, 255).astype(np.uint8)
    alpha = np.clip(rgba[..., 3], 0, 255).astype(np.uint8)
    visible = alpha >= int(alpha_threshold)
    soft_visible = visible & (alpha < 245)
    solid = alpha >= 245
    rgb_out = rgb.copy()
    if np.any(soft_visible) and np.any(solid):
        repair_mask = soft_visible.astype(np.uint8) * 255
        rgb_out = cv2.inpaint(rgb_out, repair_mask, 3, cv2.INPAINT_TELEA)
    alpha_out = np.where(visible, 255, 0).astype(np.uint8)
    return np.dstack([rgb_out, alpha_out]).astype(np.float32)


def source_art_profile(rgba: np.ndarray) -> dict:
    rgb = np.clip(rgba[..., :3], 0, 255).astype(np.float32)
    alpha = np.clip(rgba[..., 3], 0, 255).astype(np.float32)
    visible = alpha > 16.0
    visible_count = int(np.count_nonzero(visible))
    total_pixels = int(alpha.size)
    alpha_coverage = visible_count / float(max(1, total_pixels))
    if visible_count == 0:
        return {
            "alpha_coverage": 0.0,
            "edge_density": 0.0,
            "luma_std": 0.0,
            "white_fraction": 0.0,
            "category": "empty_alpha",
            "recommended_shape_mode": "mixed_character_art",
            "recommended_luma_prep": "none",
            "recommendation": "No visible pixels were detected.",
        }

    luma = rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722
    luma_visible = luma[visible]
    alpha_visible = alpha[visible]
    alpha_norm = (alpha / 255.0).astype(np.float32)
    luma_masked = (luma * alpha_norm).astype(np.float32)
    gx = cv2.Sobel(luma_masked, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(luma_masked, cv2.CV_32F, 0, 1, ksize=3)
    agx = cv2.Sobel(alpha_norm, cv2.CV_32F, 1, 0, ksize=3)
    agy = cv2.Sobel(alpha_norm, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt(gx * gx + gy * gy) + np.sqrt(agx * agx + agy * agy) * 255.0
    edge_density = float(np.count_nonzero(edge[visible] > 35.0) / float(max(1, visible_count)))
    luma_std = float(np.std(luma_visible))
    white_fraction = float(np.count_nonzero((luma_visible > 232.0) & (alpha_visible > 128.0)) / float(max(1, visible_count)))

    if alpha_coverage < 0.18 and white_fraction > 0.45:
        category = "sparse_white_line_art"
        recommended_shape_mode = "mixed_edge_bias"
        recommended_luma = "none"
        recommendation = "Sparse white transparent line art: use edge-biased shapes, enough layers, and usually leave Luma Prep off."
    elif edge_density >= 0.34 and luma_std >= 55.0:
        category = "flat_crisp_livery"
        recommended_shape_mode = "mixed_edge_bias"
        recommended_luma = "luma_bands"
        recommendation = "Flat crisp livery art: edge-biased shapes and Luma Prep usually preserve borders and broad color regions best."
    elif alpha_coverage < 0.70 and edge_density < 0.30 and luma_std < 65.0:
        category = "soft_gradient_character"
        recommended_shape_mode = "mixed_soft_detail"
        recommended_luma = "none"
        recommendation = "Soft transparent character art: soft-detail or smart-detail weighting without Luma Prep usually avoids posterized gradients, hard rectangle blocks, and over-smoothed hair."
    else:
        category = "general_art"
        recommended_shape_mode = "mixed_smart_detail"
        recommended_luma = "none"
        recommendation = "General art: smart-detail weighting without Luma Prep is the safer default; enable Luma Prep manually for flat logo/livery sources."

    return {
        "alpha_coverage": round(alpha_coverage, 4),
        "edge_density": round(edge_density, 4),
        "luma_std": round(luma_std, 4),
        "white_fraction": round(white_fraction, 4),
        "category": category,
        "recommended_shape_mode": recommended_shape_mode,
        "recommended_luma_prep": recommended_luma,
        "recommendation": recommendation,
    }


def downscale_rgba(arr: np.ndarray, max_dim: int) -> np.ndarray:
    h, w = arr.shape[:2]
    if max_dim <= 0 or max(h, w) <= max_dim:
        return arr
    scale = max_dim / float(max(h, w))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGBA")
    img = img.resize((nw, nh), Image.Resampling.BICUBIC)
    return np.asarray(img, dtype=np.float32)


# ------------------------------------------------------------ 세부 열지도

def normalize_map(values: np.ndarray, visible: np.ndarray | None = None) -> np.ndarray:
    data = np.nan_to_num(values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if visible is not None and np.any(visible):
        sample = data[visible]
    else:
        sample = data.reshape(-1)
    high = float(np.percentile(sample, 97.5)) if sample.size else 1.0
    if high <= 1e-6:
        return np.zeros_like(data, dtype=np.float32)
    return np.clip(data / high, 0.0, 1.0).astype(np.float32)


def build_detail_heatmap(rgba: np.ndarray) -> np.ndarray:
    if rgba.ndim != 3 or rgba.shape[2] < 4:
        raise ValueError("detail heatmap expects an RGBA image")
    height, width = rgba.shape[:2]
    if height <= 0 or width <= 0:
        return np.zeros((max(1, height), max(1, width)), dtype=np.float32)

    rgb = np.clip(rgba[..., :3], 0, 255).astype(np.float32)
    alpha = np.clip(rgba[..., 3] / 255.0, 0.0, 1.0).astype(np.float32)
    visible = alpha > 0.04
    luma = (rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114) / 255.0
    maxc = rgb.max(axis=2) / 255.0
    minc = rgb.min(axis=2) / 255.0
    saturation = np.clip(maxc - minc, 0.0, 1.0)

    blur_luma = cv2.GaussianBlur(luma, (0, 0), 1.0)
    gx = cv2.Sobel(blur_luma, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur_luma, cv2.CV_32F, 0, 1, ksize=3)
    luma_edge = normalize_map(np.sqrt(gx * gx + gy * gy), visible)

    color_edges = []
    for channel in range(3):
        channel_data = cv2.GaussianBlur(rgb[..., channel] / 255.0, (0, 0), 0.85)
        cgx = cv2.Sobel(channel_data, cv2.CV_32F, 1, 0, ksize=3)
        cgy = cv2.Sobel(channel_data, cv2.CV_32F, 0, 1, ksize=3)
        color_edges.append(cgx * cgx + cgy * cgy)
    color_edge = normalize_map(np.sqrt(np.maximum.reduce(color_edges)), visible)

    agx = cv2.Sobel(alpha, cv2.CV_32F, 1, 0, ksize=3)
    agy = cv2.Sobel(alpha, cv2.CV_32F, 0, 1, ksize=3)
    alpha_edge = normalize_map(np.sqrt(agx * agx + agy * agy), alpha > 0.0)

    local_mean = cv2.GaussianBlur(luma, (0, 0), 3.0)
    local_contrast = normalize_map(np.abs(luma - local_mean), visible)
    linework = np.clip((0.62 - luma) / 0.62, 0.0, 1.0) * np.maximum(luma_edge, color_edge) * alpha
    highlights = np.clip((luma - 0.78) / 0.22, 0.0, 1.0) * np.maximum(luma_edge, color_edge) * alpha

    heat = (
        luma_edge * 0.28
        + color_edge * 0.24
        + alpha_edge * 0.20
        + local_contrast * 0.16
        + linework * 0.18
        + highlights * 0.10
        + saturation * np.maximum(luma_edge, color_edge) * 0.14
    )
    heat *= np.where(visible, 1.0, 0.15).astype(np.float32)

    kernel = np.ones((3, 3), np.uint8)
    heat = cv2.dilate(heat.astype(np.float32), kernel, iterations=1)
    heat = cv2.GaussianBlur(heat, (0, 0), 0.8)
    return normalize_map(heat, visible)


def heatmap_to_rgba(mask: np.ndarray, source_rgba: np.ndarray | None = None) -> np.ndarray:
    heat = np.clip(mask.astype(np.float32), 0.0, 1.0)
    height, width = heat.shape[:2]
    blue = np.clip((1.0 - heat) * 150.0, 0, 255)
    red = np.clip(heat * 255.0, 0, 255)
    green = np.clip(np.minimum(heat, 1.0 - heat) * 300.0, 0, 255)
    alpha = np.full((height, width), 255.0, dtype=np.float32)
    if source_rgba is not None and source_rgba.ndim == 3 and source_rgba.shape[:2] == heat.shape:
        src_alpha = np.clip(source_rgba[..., 3], 0, 255).astype(np.float32)
        alpha = np.where(src_alpha > 8, 255.0, 80.0)
    return np.dstack([red, green, blue, alpha]).astype(np.uint8)


def apply_detail_guidance(rgba: np.ndarray, mask: np.ndarray, strength: float = 0.32) -> np.ndarray:
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0:
        return rgba.copy()
    rgb = np.clip(rgba[..., :3], 0, 255).astype(np.float32)
    alpha = np.clip(rgba[..., 3], 0, 255).astype(np.float32)
    heat = np.clip(mask.astype(np.float32), 0.0, 1.0)[..., None]

    blur = cv2.GaussianBlur(rgb, (0, 0), 1.15)
    sharp = np.clip(rgb + (rgb - blur) * (0.85 * strength), 0, 255)
    gray = (rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114)[..., None]
    color_boost = np.clip(gray + (rgb - gray) * (1.0 + 0.18 * strength), 0, 255)
    guided = rgb * (1.0 - heat * strength) + ((sharp * 0.72) + (color_boost * 0.28)) * (heat * strength)
    return np.dstack([np.clip(guided, 0, 255), alpha]).astype(np.float32)
