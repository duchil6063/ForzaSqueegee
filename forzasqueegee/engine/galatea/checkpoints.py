# SPDX-License-Identifier: MIT
# kloudys-forza-painter-suite(MIT)의 생성 파이프라인을 이식한 파생물이다.
# Portions © 2021 AE (forza-painter) · Sam Twidale (geometrize-lib) ·
# Michael Fogleman (Primitive). 전문·유래는 THIRD_PARTY_NOTICES.md.
"""체크포인트 JSON 발굴·복원·이름 규약.

원본: forza_generator_v2.py(normalize_payload·drawable_shapes·
background_shape·canvas_size_from_payload·raw_checkpoint_number·
collect_candidate_jsons·synthesize_missing_checkpoints·경로 헬퍼) —
로그 문자열(한국어)만 다르고 로직은 원본 그대로.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from ...i18n import msg
from .base import ELLIPSE, RECTANGLE, ROTATED_ELLIPSE, ROTATED_RECTANGLE
from .base import FINALS_DIR_NAME, PREVIEWS_DIR_NAME
from .store import save_json


def normalize_payload(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "shapes" not in payload:
        raise ValueError(f"{path} is not a generator geometry JSON")
    return payload


def drawable_shapes(payload: dict) -> list[dict]:
    out = []
    for shape in payload.get("shapes", [])[1:]:
        color = shape.get("color", [0, 0, 0, 0])
        if len(color) < 4 or int(color[3]) <= 0:
            continue
        out.append(shape)
    return out


def shape_type_name(shape_type: int) -> str:
    if shape_type == RECTANGLE:
        return "rectangle"
    if shape_type == ROTATED_RECTANGLE:
        return "rotated_rectangle"
    if shape_type == ELLIPSE:
        return "ellipse"
    if shape_type == ROTATED_ELLIPSE:
        return "rotated_ellipse"
    return f"type_{shape_type}"


def shape_type_counts(shapes: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for shape in shapes:
        name = shape_type_name(int(shape.get("type", ROTATED_ELLIPSE)))
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))


def background_shape(payload: dict) -> dict:
    shapes = payload.get("shapes", [])
    if not shapes:
        raise ValueError("geometry payload has no shapes")
    return shapes[0]


def canvas_size_from_payload(payload: dict) -> tuple[int, int]:
    bg = background_shape(payload)
    data = bg.get("data", [])
    if len(data) >= 4:
        return max(1, int(data[2])), max(1, int(data[3]))
    raise ValueError("background shape is missing canvas size")


def raw_checkpoint_number(path: Path, stem: str) -> int | None:
    core = path.stem
    if not core.startswith(f"{stem}."):
        return None
    tag = core[len(stem) + 1 :]
    if not tag.isdigit():
        return None
    return int(tag)


def collect_candidate_jsons(out_dir: Path, stem: str, max_checkpoint: int | None = None,
                            log=print) -> list[Path]:
    paths = []
    for path in out_dir.glob(f"{stem}*.json"):
        name = path.name
        if ".v2." in name or ".fh6." in name or ".report." in name or re.search(r"\.\d+v2\.json$", name) or name.endswith(".v2.json"):
            continue
        checkpoint = raw_checkpoint_number(path, stem)
        if max_checkpoint is not None and checkpoint is not None and checkpoint > max_checkpoint:
            log(msg("넘침 체크포인트 {name}: {checkpoint} > 요청 상한 {cap} — 마무리가 검증 후 상한으로 자른다",
                    name=name, checkpoint=checkpoint, cap=max_checkpoint))
        paths.append(path)
    paths = sorted(set(paths), key=lambda path: candidate_json_sort_key(path, stem))
    final_path = out_dir / f"{stem}.json"
    if final_path in paths and max_checkpoint is not None and (out_dir / f"{stem}.{max_checkpoint}.json") in paths:
        paths = [path for path in paths if path != final_path]
    return paths


def drawable_count_from_payload(payload: dict) -> int:
    return len(drawable_shapes(payload))


def first_drawable_shapes(payload: dict, count: int) -> list[dict]:
    selected = []
    for shape in payload.get("shapes", [])[1:]:
        color = shape.get("color", [0, 0, 0, 0])
        if len(color) >= 4 and int(color[3]) > 0:
            selected.append(shape)
            if len(selected) >= count:
                break
    return selected


def synthesize_missing_checkpoints(out_dir: Path, stem: str, requested_points: list[int],
                                   max_checkpoint: int, log=print) -> None:
    requested = sorted({int(point) for point in requested_points if 0 < int(point) <= max_checkpoint})
    if not requested:
        return

    available: list[tuple[int, Path, dict]] = []
    for path in collect_candidate_jsons(out_dir, stem, max_checkpoint=max_checkpoint, log=log):
        try:
            payload = normalize_payload(path)
            count = drawable_count_from_payload(payload)
        except Exception as exc:
            log(msg("읽을 수 없는 체크포인트 건너뜀 {name}: {error}",
                    name=path.name, error=exc))
            continue
        if count > 0:
            available.append((count, path, payload))
    if not available:
        return
    available.sort(key=lambda item: (item[0], item[1].name.lower()))

    existing_by_number = {
        checkpoint: path
        for count, path, _payload in available
        if (checkpoint := raw_checkpoint_number(path, stem)) is not None
    }
    for point in requested:
        existing = existing_by_number.get(point)
        if existing is not None and existing.exists():
            continue
        source = next(((count, path, payload) for count, path, payload in available if count >= point), None)
        if source is None:
            continue
        source_count, source_path, source_payload = source
        selected_shapes = first_drawable_shapes(source_payload, point)
        if len(selected_shapes) < point:
            continue
        synthesized = dict(source_payload)
        synthesized["shapes"] = [background_shape(source_payload)] + selected_shapes
        dest = out_dir / f"{stem}.{point}.json"
        save_json(dest, synthesized)
        available.append((point, dest, synthesized))
        available.sort(key=lambda item: (item[0], item[1].name.lower()))
        existing_by_number[point] = dest
        log(msg("빠진 체크포인트 {point} 복원: {source}({count}장) 앞부분 절단 → {dest}",
                point=point, source=source_path.name, count=source_count,
                dest=dest.name))


def candidate_json_sort_key(path: Path, stem: str) -> tuple[int, int, str]:
    checkpoint = raw_checkpoint_number(path, stem)
    if checkpoint is None:
        return (1, 0, path.name.lower())
    return (0, checkpoint, path.name.lower())


def checkpoint_tag_for_candidate(candidate_path: Path, stem: str) -> str:
    core = candidate_path.stem
    if core == stem:
        return "final"
    if core.startswith(f"{stem}."):
        return core[len(stem) + 1 :]
    return core


def v2_json_path_for_tag(out_dir: Path, stem: str, tag: str) -> Path:
    tag = re.sub(r"[^A-Za-z0-9_-]+", "", tag).strip() or "final"
    return out_dir / FINALS_DIR_NAME / f"{stem}.{tag}v2.json"


def v2_preview_path_for_tag(out_dir: Path, stem: str, tag: str) -> Path:
    tag = re.sub(r"[^A-Za-z0-9_-]+", "", tag).strip() or "final"
    return out_dir / PREVIEWS_DIR_NAME / f"{stem}.preview.{tag}v2.png"


def stem_from_image(path: Path) -> str:
    # Go 생성기는 출력 베이스의 점을 확장자 구분자로 본다 — 점을 빼서 안전하게
    return re.sub(r"[^A-Za-z0-9_-]+", "_", path.stem).strip("_") or "image"
