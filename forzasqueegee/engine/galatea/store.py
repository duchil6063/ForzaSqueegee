# SPDX-License-Identifier: MIT
# kloudys-forza-painter-suite(MIT)의 생성 파이프라인을 이식한 파생물이다.
# Portions © 2021 AE (forza-painter) · Sam Twidale (geometrize-lib) ·
# Michael Fogleman (Primitive). 전문·유래는 THIRD_PARTY_NOTICES.md.
"""원자적 저장 (JSON·PNG) — 원본: forza_generator_v2.py 그대로."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from PIL import Image


def replace_atomic(temp_path: Path, destination: Path) -> None:
    last_error = None
    for attempt in range(6):
        try:
            os.replace(temp_path, destination)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(0.05 * (attempt + 1))
    if last_error is not None:
        raise last_error


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temp_path.open("w", encoding="utf-8", newline="\n") as output:
            output.write(json.dumps(payload, indent=2) + "\n")
            output.flush()
            os.fsync(output.fileno())
        replace_atomic(temp_path, path)
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass


def save_png(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        image.save(temp_path, format="PNG")
        with temp_path.open("r+b") as output:
            os.fsync(output.fileno())
        replace_atomic(temp_path, path)
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass


def require_saved_file(path: Path, label: str) -> None:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise RuntimeError(f"{label} was not saved: {path}") from exc
    if size <= 0:
        raise RuntimeError(f"{label} was saved as an empty file: {path}")
