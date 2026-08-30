# SPDX-License-Identifier: MIT
# kloudys-forza-painter-suite(MIT)의 생성 파이프라인을 이식한 파생물이다.
# Portions © 2021 AE (forza-painter) · Sam Twidale (geometrize-lib) ·
# Michael Fogleman (Primitive). 전문·유래는 THIRD_PARTY_NOTICES.md.
"""본체 — 이미지 → KFPS 동일 로직 도안 (원시 생성 + Finalize Checkpoints).

원본: forza_generator_v2.main() + generator_backend.build_generator_command의
기본값. 후보 선택(select_candidate·select_top_checkpoint_indices)도 원본
그대로다. 승격 정책만 우리 몫이다 (본문 주석).
"""

from __future__ import annotations

import json
import math
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ...i18n import msg
from ...paths import run_file
from ..render import _BASE_HEIGHT_UNITS
from .base import (CHECKPOINTS_DIR_NAME, FINALS_DIR_NAME, GENERATOR_BIN,
                   LIVE_PREVIEW_EVERY, PRESETS, PREVIEWS_DIR_NAME,
                   REPAIR_CANDIDATE_LIMIT, REPORTS_DIR_NAME, SCORE_SIZE,
                   SETTINGS_DIR)
from .checkpoints import (background_shape, canvas_size_from_payload,
                          checkpoint_tag_for_candidate, collect_candidate_jsons,
                          drawable_shapes, normalize_payload,
                          raw_checkpoint_number, shape_type_counts,
                          stem_from_image, synthesize_missing_checkpoints,
                          v2_json_path_for_tag, v2_preview_path_for_tag)
from .geometry import (scale_shape, target_has_alpha_boundary,
                       unscale_shape_f)
from .prep import (apply_detail_guidance, apply_logo_hard_edges,
                   apply_preprocess, build_detail_heatmap, downscale_rgba,
                   heatmap_to_rgba, remove_alpha_fringe_noise,
                   resize_source_for_generation, source_art_profile)
from .preview import render_import_preview
from .quantize import quantize_shape, quantize_shapes
from .repair import (enforce_shapes_inside_canvas, force_opaque_drawables,
                     repair_shapes, stabilize_flat_region_colors)
from .runner import run_generator
from .scoring import (build_importance_map, normalized_error, prune_to_target,
                      remove_fully_covered_layers, render_and_score)
from .settings import (checkpoint_step_from_save_at, normalized_save_at_text,
                       parse_bool, parse_ini, parse_save_points,
                       write_v2_settings)
from .store import require_saved_file, save_json, save_png
from .plan import plan_from_shapes

ROTATED_ELLIPSE = 16


def select_candidate(results: list[dict], tolerance: float) -> tuple[dict, dict]:
    best_accuracy = min(results, key=lambda item: item["error"])
    if tolerance <= 0:
        return best_accuracy, best_accuracy
    threshold = best_accuracy["error"] * (1.0 + max(0.0, tolerance))
    within = [item for item in results if item["error"] <= threshold]
    selected = min(within, key=lambda item: (item["final_drawables"], item["error"]))
    return best_accuracy, selected


def select_top_checkpoint_indices(records: list[dict], limit: int) -> set[int]:
    if limit <= 0 or len(records) <= limit:
        return set(range(len(records)))
    selected = {
        int(item["index"])
        for item in sorted(records, key=lambda item: (item["base_error"], item["raw_drawables"]))[:limit]
    }
    latest = max(records, key=lambda item: (item["raw_drawables"], item["candidate"]))
    selected.add(int(latest["index"]))
    return selected


def generate(image: str | Path, out_dir: str | Path, *,
             shapes: int = 0, preset: str = "shaded", seed: int = 0,
             repair: bool | None = None, luma: bool | None = None,
             heatmap: bool = False, heatmap_strength: float = 0.10,
             boost: bool = False,
             overshoot_ratio: float = 1.0, overshoot_max_extra: int = 0,
             enable_prune: bool = False, efficiency_tolerance: float = 0.0,
             score_size: int = SCORE_SIZE,
             repair_candidate_limit: int = REPAIR_CANDIDATE_LIMIT,
             finalize_only: bool = False,
             log=None, progress=None) -> dict:
    """이미지 → KFPS 동일 로직 도안. 반환: 요약 dict (pipeline이 report에 싣는다).

    KFPS 브리지 기본값 그대로다: `shapes`=0이면 프리셋 stopAt, `repair`/`luma`
    None이면 프리셋 값(v2EnableRepair·v2PreprocessMode), overshoot 꺼짐,
    후보 미리보기 전부 렌더. `progress(0..1, 단계)`는 창의 진행 막대용 —
    예외(Cancelled)는 안 잡고 전파한다. 출력 폴더에 `STOP` 파일이 생기면
    원시 생성을 우아하게 멈추고 있는 체크포인트로 마무리한다.
    """
    if log is None:
        def log(s: str) -> None:
            print(s, flush=True)
    t0 = time.time()
    image_path = Path(image).resolve()
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not GENERATOR_BIN.is_file() and not finalize_only:
        raise RuntimeError(msg("GPU 생성기가 없다: {path}", path=GENERATOR_BIN))
    preset_file = SETTINGS_DIR / PRESETS.get(preset, PRESETS["shaded"])
    if not preset_file.is_file():
        raise RuntimeError(msg("프리셋이 없다: {path}", path=preset_file))

    checkpoint_dir = out / CHECKPOINTS_DIR_NAME
    finals_dir = out / FINALS_DIR_NAME
    reports_dir = out / REPORTS_DIR_NAME
    previews_dir = out / PREVIEWS_DIR_NAME
    for folder in (checkpoint_dir, finals_dir, reports_dir, previews_dir):
        folder.mkdir(parents=True, exist_ok=True)
    stem = stem_from_image(image_path)
    stop_file = out / "STOP"
    if stop_file.exists():
        try:
            stop_file.unlink()
        except OSError:
            pass

    # ── 설정 합성 (generator_backend 기본값 그대로) ──
    base_settings = parse_ini(preset_file)
    if shapes and int(shapes) > 0:
        base_settings["stopAt"] = str(int(shapes))
    target_shapes = int(base_settings.get("stopAt", "3000"))
    if target_shapes < 1:
        raise ValueError("target shape count must be positive")
    base_settings["saveAt"] = normalized_save_at_text(base_settings.get("saveAt", ""), target_shapes)
    checkpoint_step = int(checkpoint_step_from_save_at(base_settings.get("saveAt", ""), target_shapes))
    if boost:
        rnd = int(base_settings.get("randomSamples", "0") or 0) * 2
        mut = int(base_settings.get("mutatedSamples", "0") or 0) * 2
        retries = int(base_settings.get("maxNoImproveRetries", "0") or 0)
        base_settings["randomSamples"] = str(max(1, min(rnd, 2_400_000)))
        base_settings["mutatedSamples"] = str(max(1, min(mut, 140_000)))
        if retries:
            base_settings["maxNoImproveRetries"] = str(max(1, min(retries * 2, 96)))
    preprocess_mode = base_settings.get("v2PreprocessMode", "none")
    if luma is not None:
        preprocess_mode = "luma_bands" if luma else "none"
    if preprocess_mode not in ("none", "luma_bands"):
        preprocess_mode = "none"
    setting_repair = parse_bool(base_settings.get("v2EnableRepair"), False)
    repair_enabled = setting_repair if repair is None else bool(repair)
    detail_heatmap_mode = "auto" if heatmap else "off"

    drawable_target_shapes = target_shapes  # reserved import layers = 0 (브리지 고정)
    overshoot_extra = min(int(overshoot_max_extra),
                          max(0, int(math.ceil(target_shapes * max(0.0, overshoot_ratio - 1.0)))))
    if overshoot_extra == 0 and overshoot_ratio > 1.0:
        overshoot_extra = min(int(overshoot_max_extra), max(1, int(round(target_shapes * 0.08))))
    raw_stop = target_shapes + overshoot_extra
    shape_mode = str(base_settings.get("shapeMode", "")).strip().lower()
    force_opaque_shapes = parse_bool(base_settings.get("forceOpaqueShapes"), False)
    logo_hard_edges = parse_bool(base_settings.get("logoHardEdges"), False)
    prefer_smooth_repair = (
        shape_mode in {"mixed_soft_detail", "mixed_character_art", "mixed_smart_detail"}
        and preprocess_mode == "none"
    )

    v2_settings_path = reports_dir / f"{stem}.v2.settings.ini"
    write_v2_settings(base_settings, v2_settings_path, target_shapes, raw_stop,
                      checkpoint_step, LIVE_PREVIEW_EVERY)

    # ── 전처리 (V2 순서 그대로) ──
    max_resolution = int(base_settings.get("maxResolution", "0") or 0)
    source_rgba = resize_source_for_generation(image_path, max_resolution)
    source_rgba, alpha_cleanup = remove_alpha_fringe_noise(source_rgba)
    art_profile = source_art_profile(source_rgba)
    prepared_rgba = apply_logo_hard_edges(source_rgba) if logo_hard_edges else source_rgba
    processed_rgba = apply_preprocess(prepared_rgba, preprocess_mode)
    detail_heatmap = None
    detail_strength = float(np.clip(heatmap_strength, 0.0, 1.0))
    if detail_heatmap_mode == "auto":
        detail_heatmap = build_detail_heatmap(processed_rgba)
        Image.fromarray(heatmap_to_rgba(detail_heatmap, processed_rgba), mode="RGBA").save(
            previews_dir / f"{stem}.detail-heatmap.png")
        if detail_strength > 0:
            processed_rgba = apply_detail_guidance(processed_rgba, detail_heatmap, detail_strength)
    generation_image_path = image_path
    if alpha_cleanup.get("changed") or logo_hard_edges or preprocess_mode != "none" or (
            detail_heatmap is not None and detail_strength > 0):
        preprocess_output_path = previews_dir / f"{stem}.preprocessed.png"
        Image.fromarray(np.clip(processed_rgba, 0, 255).astype(np.uint8), mode="RGBA").save(preprocess_output_path)
        generation_image_path = preprocess_output_path

    log(msg("KFPS 동일 로직 생성 — 프리셋 {preset} ({file})",
            preset=preset, file=preset_file.name))
    log(msg("  목표 {target}장 · 원시 상한 {raw_stop} · 해상도 {resolution} · "
            "루마 {luma} · 수리 {repair} · 씨드 {seed}",
            target=target_shapes, raw_stop=raw_stop, resolution=max_resolution,
            luma=preprocess_mode,
            repair=msg("켬") if repair_enabled else msg("끔"),
            seed=seed or msg("무작위")))
    log(msg("  원화 분류: {category} — {recommendation}",
            category=art_profile["category"],
            recommendation=art_profile["recommendation"]))
    if alpha_cleanup.get("changed"):
        log(msg("  알파 부스러기 정리: {px}px", px=alpha_cleanup["removed_pixels"]))

    # ── 원시 생성 (GPU) ──
    raw_generator_error = None
    if finalize_only:
        interrupted = True
        log(msg("체크포인트 재사용 — 원시 생성 없이 마무리만 다시 돈다"))
    else:
        if progress:
            progress(0.0, msg("GPU 원시 생성"))
        try:
            interrupted = run_generator(
                generation_image_path, v2_settings_path, checkpoint_dir, previews_dir,
                stem, stop_file=stop_file, seed=int(seed or 0), log=log,
                progress=(lambda n, total: progress(0.02 + 0.48 * n / max(1, total),
                                                    msg("GPU {n}/{total}장",
                                                        n=n, total=total)))
                if progress else None)
        except subprocess.CalledProcessError as exc:
            recoverable = collect_candidate_jsons(checkpoint_dir, stem, max_checkpoint=raw_stop, log=log)
            if not recoverable:
                raise RuntimeError(
                    msg("GPU 생성기 실패 (exit {code}) — OpenCL/Vulkan 드라이버가 있는 "
                        "GPU가 필요하다", code=exc.returncode)) from exc
            interrupted = True
            raw_generator_error = f"exit {exc.returncode}"
            log(msg("원시 생성기가 비정상 종료 — 있는 체크포인트 {n}개로 마무리한다",
                    n=len(recoverable)))

    # ── Finalize Checkpoints (V2 이식 + 수정 1·2) ──
    requested_checkpoints = parse_save_points(base_settings.get("saveAt", ""), raw_stop)
    synthesize_missing_checkpoints(checkpoint_dir, stem, requested_checkpoints, raw_stop, log=log)
    raw_candidates = collect_candidate_jsons(checkpoint_dir, stem, max_checkpoint=raw_stop, log=log)
    if not raw_candidates:
        raise RuntimeError(msg("체크포인트가 하나도 없다 — 원시 생성이 전혀 저장하지 못했다"))
    log(msg("마무리: 체크포인트 {n}개 채점·프루닝·최종화", n=len(raw_candidates)))

    score_rgba = downscale_rgba(source_rgba, score_size)
    score_importance = build_importance_map(score_rgba)
    if detail_heatmap is not None:
        score_heatmap = cv2.resize(detail_heatmap, (score_rgba.shape[1], score_rgba.shape[0]),
                                   interpolation=cv2.INTER_AREA)
        score_importance = np.clip(score_importance * (1.0 + score_heatmap * (detail_strength * 0.85)),
                                   0.55, 7.0).astype(np.float32)
    enforce_canvas_boundary = target_has_alpha_boundary(source_rgba)

    candidate_records = []
    for index, candidate_path in enumerate(raw_candidates):
        if progress:
            progress(0.52 + 0.18 * index / max(1, len(raw_candidates)),
                     msg("채점 {n}/{total}", n=index + 1, total=len(raw_candidates)))
        try:
            payload = normalize_payload(candidate_path)
            background = background_shape(payload)
            raw_generator_name = str(payload.get("generator", "") or "")
            is_modern_raw = raw_generator_name.lower().startswith(("kloudysgeneratorv6", "kloudysgeneratorv7"))
            checkpoint_tag = checkpoint_tag_for_candidate(candidate_path, stem)
            checkpoint_number = raw_checkpoint_number(candidate_path, stem)
            full_w, full_h = canvas_size_from_payload(payload)

            # 수정 2 — 읽자마자 전 도형을 게임 입력 스텝으로 스냅한다.
            # 이후 채점·프루닝·수리·산출 전부 이 값(=인게임 값)이다.
            drawables = quantize_shapes(drawable_shapes(payload), full_w, full_h)
            raw_count = len(drawables)
            if checkpoint_number is not None and checkpoint_number > raw_stop:
                if checkpoint_number == raw_stop + 1 and raw_count <= raw_stop:
                    checkpoint_tag = str(raw_stop)

            score_h, score_w = score_rgba.shape[:2]
            sx = score_w / float(max(1, full_w))
            sy = score_h / float(max(1, full_h))
            scaled_bg = dict(background)
            scaled_bg["color"] = list(background.get("color", [0, 0, 0, 0]))

            scaled_drawables = [scale_shape(shape, sx, sy) for shape in drawables]
            should_prune = enable_prune or raw_count > drawable_target_shapes
            if should_prune:
                kept_scaled, error = prune_to_target(
                    scaled_bg, scaled_drawables, score_rgba, drawable_target_shapes,
                    enforce_canvas_boundary=enforce_canvas_boundary,
                    importance_map=score_importance)
                scaled_map = {id(shape): idx for idx, shape in enumerate(scaled_drawables)}
                kept_original = [drawables[scaled_map[id(shape)]] for shape in kept_scaled]
                final_count = len(kept_original)
            else:
                _, _, _, _, _, total_error, scored_pixels = render_and_score(
                    scaled_bg, scaled_drawables, score_rgba,
                    enforce_canvas_boundary=enforce_canvas_boundary,
                    importance_map=score_importance)
                error = normalized_error(total_error, scored_pixels)
                kept_original = list(drawables)
                final_count = len(kept_original)
        except Exception as exc:
            log(msg("후보 건너뜀 {name}: {kind}: {error}", name=candidate_path.name,
                    kind=type(exc).__name__, error=exc))
            continue
        candidate_records.append({
            "index": index,
            "candidate_path": candidate_path,
            "candidate": candidate_path.name,
            "background": background,
            "raw_drawables": raw_count,
            "base_drawables": final_count,
            "base_shapes": kept_original,
            "base_error": error,
            "canvas_size": [full_w, full_h],
            "scale": [sx, sy],
            "checkpoint_tag": checkpoint_tag,
            "v6_raw": is_modern_raw,
        })
    if not candidate_records:
        raise RuntimeError(msg("검증을 통과한 체크포인트가 없다"))

    repair_indices = select_top_checkpoint_indices(candidate_records, repair_candidate_limit) if repair_enabled else set()

    results = []
    for result_index, record in enumerate(candidate_records, start=1):
        if progress:
            progress(0.70 + 0.28 * (result_index - 1) / max(1, len(candidate_records)),
                     msg("최종화 {n}/{total}", n=result_index,
                         total=len(candidate_records)))
        candidate_path = record["candidate_path"]
        background = record["background"]
        raw_count = record["raw_drawables"]
        full_w, full_h = record["canvas_size"]
        sx, sy = record["scale"]
        upp = _BASE_HEIGHT_UNITS / full_h

        def unscale_quantized(shape: dict) -> dict:
            """채점 공간 → 전체 px, 게임 그리드로 스냅 (원본 int 반올림 대체)."""
            return quantize_shape(unscale_shape_f(shape, sx, sy), full_w, full_h, upp)

        def quantize_scaled(shape: dict) -> dict:
            """채점 공간 도형을 게임 그리드에 스냅한 채점 공간 도형으로."""
            return scale_shape(unscale_quantized(shape), sx, sy)

        refinement = {
            "enabled": False, "touched": 0, "improvements": 0,
            "before": record["base_error"], "after": record["base_error"],
        }
        final_shapes = list(record["base_shapes"])
        final_error = record["base_error"]
        repair_applied = repair_enabled and (not bool(record.get("v6_raw"))) and int(record["index"]) in repair_indices
        if repair_applied:
            try:
                scaled_bg = dict(background)
                scaled_bg["color"] = list(background.get("color", [0, 0, 0, 0]))
                scaled_selected = [scale_shape(shape, sx, sy) for shape in final_shapes]
                refined_scaled, refined_error, refinement = repair_shapes(
                    scaled_bg, scaled_selected, score_rgba,
                    max_shapes=18 if enforce_canvas_boundary else 8,
                    rounds=2 if enforce_canvas_boundary else 1,
                    enforce_canvas_boundary=enforce_canvas_boundary,
                    prefer_smooth_shapes=prefer_smooth_repair,
                    importance_map=score_importance,
                    allow_alpha_repair=(not force_opaque_shapes and not prefer_smooth_repair),
                    quantize=quantize_scaled)
                final_shapes = [unscale_quantized(shape) for shape in refined_scaled]
                if force_opaque_shapes:
                    final_shapes = force_opaque_drawables(final_shapes)
                final_error = refined_error
                refinement = dict(refinement)
                refinement["prefer_smooth_shapes"] = prefer_smooth_repair
                refinement["force_opaque_shapes"] = force_opaque_shapes
            except Exception as exc:
                refinement = dict(refinement)
                refinement.update({"enabled": True, "failed": True,
                                   "error": f"{type(exc).__name__}: {exc}"})
                log(msg("수리 실패 {name}: {error} — 프루닝 결과를 그대로 쓴다",
                        name=candidate_path.name, error=exc))
        if enforce_canvas_boundary:
            try:
                scaled_bg = dict(background)
                scaled_bg["color"] = list(background.get("color", [0, 0, 0, 0]))
                scaled_selected = [scale_shape(shape, sx, sy) for shape in final_shapes]
                bounded_scaled, bounded_error, boundary_summary = enforce_shapes_inside_canvas(
                    scaled_bg, scaled_selected, score_rgba, importance_map=score_importance)
                if boundary_summary.get("fitted", 0):
                    final_shapes = [unscale_quantized(shape) for shape in bounded_scaled]
                    final_error = bounded_error
                refinement = dict(refinement)
                refinement["canvas_boundary"] = boundary_summary
            except Exception as exc:
                refinement = dict(refinement)
                refinement["canvas_boundary_failed"] = f"{type(exc).__name__}: {exc}"
        try:
            scaled_bg = dict(background)
            scaled_bg["color"] = list(background.get("color", [0, 0, 0, 0]))
            if force_opaque_shapes:
                final_shapes = force_opaque_drawables(final_shapes)
            scaled_selected = [scale_shape(shape, sx, sy) for shape in final_shapes]
            flat_color_summary = {"enabled": False, "skipped": "not a flat opaque preset"}
            should_stabilize_flat_colors = bool(
                force_opaque_shapes
                and (preprocess_mode == "luma_bands"
                     or any(token in shape_mode for token in ("flat", "edge", "logo", "livery"))))
            if should_stabilize_flat_colors:
                stabilized_scaled, flat_color_summary = stabilize_flat_region_colors(
                    scaled_selected, score_rgba, force_opaque=True)
                if flat_color_summary.get("changed", 0):
                    final_shapes = [unscale_quantized(shape) for shape in stabilized_scaled]
                    scaled_selected = stabilized_scaled
                    log(msg("  평면색 안정화: {n}장 색 스냅 ({name})",
                            n=flat_color_summary["changed"],
                            name=candidate_path.name))
            refinement = dict(refinement)
            refinement["flat_color_stabilization"] = flat_color_summary
            cleanup_budget = max(0, len(final_shapes) - drawable_target_shapes)
            cleaned_scaled, cleaned_error, covered_cleanup = remove_fully_covered_layers(
                scaled_bg, scaled_selected, score_rgba,
                enforce_canvas_boundary=enforce_canvas_boundary,
                importance_map=score_importance,
                removal_limit=cleanup_budget)
            final_error = cleaned_error
            if covered_cleanup.get("removed", 0):
                final_shapes = [unscale_quantized(shape) for shape in cleaned_scaled]
                log(msg("  가려진 레이어 정리: {n}장 제거 ({name})",
                        n=covered_cleanup["removed"], name=candidate_path.name))
            refinement = dict(refinement)
            refinement["covered_layer_cleanup"] = covered_cleanup
        except Exception as exc:
            refinement = dict(refinement)
            refinement["covered_layer_cleanup_failed"] = f"{type(exc).__name__}: {exc}"
        if len(final_shapes) > drawable_target_shapes:
            try:
                scaled_bg = dict(background)
                scaled_bg["color"] = list(background.get("color", [0, 0, 0, 0]))
                scaled_selected = [scale_shape(shape, sx, sy) for shape in final_shapes]
                capped_scaled, capped_error = prune_to_target(
                    scaled_bg, scaled_selected, score_rgba, drawable_target_shapes,
                    enforce_canvas_boundary=enforce_canvas_boundary,
                    importance_map=score_importance)
                final_shapes = [unscale_quantized(shape) for shape in capped_scaled]
                final_error = capped_error
            except Exception as exc:
                final_shapes = final_shapes[:drawable_target_shapes]
                final_error = record["base_error"]
                refinement = dict(refinement)
                refinement["hard_cap_failed"] = f"{type(exc).__name__}: {exc}"
            refinement = dict(refinement)
            refinement["after_hard_cap"] = final_error
            try:
                scaled_bg = dict(background)
                scaled_bg["color"] = list(background.get("color", [0, 0, 0, 0]))
                scaled_selected = [scale_shape(shape, sx, sy) for shape in final_shapes]
                cleaned_scaled, cleaned_error, post_cap_cleanup = remove_fully_covered_layers(
                    scaled_bg, scaled_selected, score_rgba,
                    enforce_canvas_boundary=enforce_canvas_boundary,
                    importance_map=score_importance,
                    removal_limit=0)
                if post_cap_cleanup.get("removed", 0):
                    final_shapes = [unscale_quantized(shape) for shape in cleaned_scaled]
                    final_error = cleaned_error
                refinement["covered_layer_cleanup_after_cap"] = post_cap_cleanup
            except Exception as exc:
                refinement["covered_layer_cleanup_after_cap_failed"] = f"{type(exc).__name__}: {exc}"
        checkpoint_tag = record["checkpoint_tag"]
        final_json_path = v2_json_path_for_tag(out, stem, checkpoint_tag)
        final_preview_path = v2_preview_path_for_tag(out, stem, checkpoint_tag)
        if force_opaque_shapes:
            final_shapes = force_opaque_drawables(final_shapes)
        # 산출은 JSON 직렬화를 위해 값을 다듬는다 (그리드 값 자체는 불변)
        cleaned_payload_shapes = [
            {"type": int(shape.get("type", ROTATED_ELLIPSE)),
             "data": [round(float(v), 3) for v in shape.get("data", [])],
             "color": [int(round(float(v))) if i < 3 else round(float(v), 2)
                       for i, v in enumerate(list(shape.get("color", [0, 0, 0, 255]))[:4])],
             "score": shape.get("score", 0)}
            for shape in final_shapes
        ]
        save_json(final_json_path, {"shapes": cleaned_payload_shapes})
        require_saved_file(final_json_path, "Final checkpoint JSON")
        preview = render_import_preview(background, final_shapes, full_w, full_h)
        save_png(final_preview_path, preview)
        results.append({
            "candidate": candidate_path.name,
            "raw_drawables": raw_count,
            "final_drawables": len(final_shapes),
            "error": final_error,
            "base_error": record["base_error"],
            "kept_shapes": final_shapes,
            "canvas_size": [full_w, full_h],
            "refinement": refinement,
            "repair_applied": repair_applied,
            "v2_json": str(final_json_path),
            "v2_preview": str(final_preview_path),
            "checkpoint_tag": checkpoint_tag,
        })
        log(msg("  후보 {name}: 원시 {raw} → 최종 {final}장, 오차 {error:.6f}",
                name=candidate_path.name, raw=raw_count, final=len(final_shapes),
                error=final_error))

    best_accuracy, _tolerant = select_candidate(results, efficiency_tolerance)
    # 승격은 **요청 장수 체크포인트**(최신)다. KFPS는 후보 전부를 늘어놓고
    # 사람이 고르는 구조라 자동 선택은 우리 몫인데, 오차 수치는 소프트 알파
    # 매트에서 장수와 역행한다 — 불투명 도형이 반투명 가장자리를 덮는 알파
    # 벌점이 장수 증가분을 잠식해서다 (2026-08-25 실측: 벤티 500장 5118 <
    # 1500장 5547인데 육안은 1500장이 명백히 낫다 — 라이어 페그·머리단·
    # 나비매듭이 500장에는 없다). 요청 장수가 사용자의 선택이므로 그쪽을
    # 승격하고, 오차 최소 후보는 보고서에 병기한다 — 다른 체크포인트는
    # finals/*.v2.json에서 `kfpsimport`로 언제든 플랜이 된다.
    selected = max(results, key=lambda item: (item["raw_drawables"], item["candidate"]))
    if selected is not best_accuracy:
        log(msg("  참고: 오차 최소는 {name} ({n}장, {error:.1f})"
                " — finals/에서 kfpsimport로 바꿔 쓸 수 있다",
                name=best_accuracy["candidate"],
                n=best_accuracy["final_drawables"],
                error=best_accuracy["error"]))

    # ── 승격: 요청 장수 후보 → 도안 + 프리뷰 + KFPS JSON ──
    full_w, full_h = selected["canvas_size"]
    plan = plan_from_shapes(selected["kept_shapes"], full_w, full_h,
                            source_image=str(image_path))
    plan.save(run_file(out, "plan.json"))
    from ..catalog import Catalog, default_catalog_path
    from ..kfpsjson import export_typecode
    from ..render import render_plan
    cat = Catalog(default_catalog_path())
    render = cv2.resize(render_plan(plan, cat, scale=2), (full_w, full_h),
                        interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(render, cv2.COLOR_RGB2BGR))
    if ok:
        run_file(out, "preview.png").write_bytes(buf.tobytes())
    kfps_data, kfps_stats = export_typecode(plan, cat)
    run_file(out, "kfps.json").write_text(
        json.dumps(kfps_data, ensure_ascii=False), encoding="utf-8")

    report = {
        "pipeline": "kfps_galatea_v2",
        "generator": GENERATOR_BIN.name,
        "preset": preset,
        "preset_file": preset_file.name,
        "target_shapes": target_shapes,
        "raw_stop": raw_stop,
        "seed": int(seed or 0),
        "preprocess_mode": preprocess_mode,
        "detail_heatmap": detail_heatmap_mode,
        "repair_enabled": repair_enabled,
        "interrupted": bool(interrupted),
        "raw_generator_error": raw_generator_error,
        "score_size": score_size,
        "ingame_fidelity": {
            "mask": "A_02 48각형 (render._ell_mask)",
            "quantized": "이동 0.5유닛 · 스케일 0.01 · 회전 0.1° · HSB 0.01 · 투명도 0.01",
        },
        "alpha_cleanup": alpha_cleanup,
        "source_art_profile": art_profile,
        "selected": {
            "candidate": selected["candidate"],
            "checkpoint_tag": selected["checkpoint_tag"],
            "final_drawables": selected["final_drawables"],
            "error": selected["error"],
            "shape_types": shape_type_counts(selected["kept_shapes"]),
        },
        "best_accuracy": {
            "candidate": best_accuracy["candidate"],
            "final_drawables": best_accuracy["final_drawables"],
            "error": best_accuracy["error"],
        },
        "kfps_export": kfps_stats,
        "candidates": [
            {"candidate": item["candidate"], "checkpoint_tag": item["checkpoint_tag"],
             "raw_drawables": item["raw_drawables"], "final_drawables": item["final_drawables"],
             "error": item["error"], "base_error": item["base_error"],
             "repair_applied": item["repair_applied"], "v2_json": item["v2_json"],
             "refinement": item["refinement"]}
            for item in sorted(results, key=lambda item: (item["raw_drawables"], item["candidate"]))
        ],
        "sec": round(time.time() - t0, 1),
    }
    save_json(reports_dir / f"{stem}.v2.report.json", report)
    log(msg("선택: {name} → {n}장, 오차 {error:.6f} ({sec}s)",
            name=selected["candidate"], n=selected["final_drawables"],
            error=selected["error"], sec=report["sec"]))
    if stop_file.exists():
        try:
            stop_file.unlink()
        except OSError:
            pass
    return report
