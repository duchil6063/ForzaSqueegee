# SPDX-License-Identifier: MIT
# kloudys-forza-painter-suite(MIT)의 생성 파이프라인을 이식한 파생물이다.
# Portions © 2021 AE (forza-painter) · Sam Twidale (geometrize-lib) ·
# Michael Fogleman (Primitive). 전문·유래는 THIRD_PARTY_NOTICES.md.
"""프리셋 ini 해석·저장점 계산·V2 임시 설정 작성.

원본: forza_generator_v2.py(parse_ini·build_save_points·parse_save_points·
parse_bool·write_v2_settings) + generator_backend.py(normalized_save_at_text·
checkpoint_step_from_save_at) — 전부 원본 그대로.
"""

from __future__ import annotations

import re
from pathlib import Path


def parse_ini(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith(";") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def build_save_points(target: int, stop_at: int, checkpoint_step: int) -> str:
    points = set()
    step = max(1, checkpoint_step)
    n = step
    while n < stop_at:
        points.add(n)
        n += step
    points.add(target)
    points.add(stop_at)
    return ",".join(str(n) for n in sorted(points))


def parse_save_points(value: str, stop_at: int) -> list[int]:
    points = []
    for part in re.split(r"[,;\s]+", str(value or "")):
        if not part.strip():
            continue
        try:
            point = int(part)
        except ValueError:
            continue
        if 0 < point <= stop_at:
            points.append(point)
    return sorted(set(points))


def parse_bool(value, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off", ""}:
        return False
    return default


def normalized_save_at_text(value, target_count: int) -> str:
    """generator_backend.normalized_save_at_text — 목표 장수를 saveAt에 넣는다."""
    points = set()
    for part in re.split(r"[,;\s]+", str(value or "")):
        if not part.strip():
            continue
        try:
            point = int(part)
        except ValueError:
            continue
        if point > 0:
            points.add(point)
    points.add(max(1, int(target_count)))
    return ",".join(str(point) for point in sorted(points))


def checkpoint_step_from_save_at(save_at_text, target_count: int) -> str:
    """generator_backend.checkpoint_step_from_save_at — saveAt 최소 간격."""
    save_at_points = []
    for part in re.split(r"[,;\s]+", str(save_at_text or "")):
        if not part.strip():
            continue
        try:
            point = int(part)
        except ValueError:
            continue
        if point > 0:
            save_at_points.append(point)
    if len(save_at_points) >= 2:
        ordered = sorted(set(save_at_points))
        deltas = [b - a for a, b in zip(ordered, ordered[1:]) if b > a]
        return str(min(deltas) if deltas else ordered[0])
    if save_at_points:
        return str(save_at_points[0])
    return "250" if target_count <= 1000 else "500"


def write_v2_settings(
    base_settings: dict[str, str],
    out_path: Path,
    target: int,
    stop_at: int,
    checkpoint_step: int,
    live_preview_every: int,
) -> None:
    values = dict(base_settings)
    values["description"] = f"V2 settings targeting {target} template layers"
    values.setdefault("shapeMode", "mixed_ellipses")
    values["stopAt"] = str(stop_at)
    preview_every = max(1, min(int(live_preview_every or checkpoint_step or 100), max(1, stop_at)))
    explicit_points = parse_save_points(values.get("saveAt", ""), stop_at)
    if explicit_points:
        explicit_points = sorted(set(explicit_points + [target, stop_at]))
        values["saveAt"] = ",".join(str(point) for point in explicit_points)
        values["saveEvery"] = str(preview_every)
        values["previewEvery"] = str(preview_every)
    else:
        values["saveAt"] = build_save_points(target, stop_at, checkpoint_step)
        values["saveEvery"] = str(preview_every)
        values["previewEvery"] = str(preview_every)

    ordered_keys = [
        "description",
        "detailMode",
        "maxPreviewSize",
        "maxResolution",
        "maxThreads",
        "mutatedSamples",
        "forceOpaqueShapes",
        "logoHardEdges",
        "posterizeLevels",
        "previewEvery",
        "randomSamples",
        "saveAt",
        "saveEvery",
        "shapeMode",
        "stopAt",
    ]
    lines = []
    seen = set()
    for key in ordered_keys:
        if key in values:
            lines.append(f"{key} = {values[key]}")
            seen.add(key)
    for key, value in values.items():
        if key not in seen:
            lines.append(f"{key} = {value}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
