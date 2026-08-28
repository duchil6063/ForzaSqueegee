# SPDX-License-Identifier: MIT
# kloudys-forza-painter-suite(MIT)의 생성 파이프라인을 이식한 파생물이다.
# Portions © 2021 AE (forza-painter) · Sam Twidale (geometrize-lib) ·
# Michael Fogleman (Primitive). 전문·유래는 THIRD_PARTY_NOTICES.md.
"""공통 상수·벤더 경로 — galatea 패키지의 밑판.

도형 종별 숫자는 KFPS 체크포인트 JSON의 `type`(geometrize 유래)이고,
실행 파일·프리셋은 `vendor/galatea/`의 KFPS 동봉본이다 (출처·해시는
vendor/galatea/README.md).
"""

from __future__ import annotations

from pathlib import Path

# geometrize 도형 종별 (KFPS 체크포인트 JSON의 type)
RECTANGLE = 1
ROTATED_RECTANGLE = 2
ELLIPSE = 8
ROTATED_ELLIPSE = 16

_ROOT = Path(__file__).resolve().parents[3]
VENDOR_DIR = _ROOT / "vendor" / "galatea"
GENERATOR_BIN = VENDOR_DIR / "KloudysGalateaGenesis.exe"
SETTINGS_DIR = VENDOR_DIR / "settings"
# 프리셋 3종 (KFPS 원본 ini 그대로). shaded가 KFPS UI 기본값이고 우리 기본이다
PRESETS = {
    "shaded": "b.shaded-art.ini",
    "flat": "a.flat-colors.ini",
    "gradients": "c.gradients.ini",
}

FINALS_DIR_NAME = "finals"
CHECKPOINTS_DIR_NAME = "checkpoints"
REPORTS_DIR_NAME = "reports"
PREVIEWS_DIR_NAME = "previews"
LIVE_PREVIEW_EVERY = 100     # KFPS 브리지 고정값
SCORE_SIZE = 640             # V2 기본 채점 해상도
REPAIR_CANDIDATE_LIMIT = 4   # V2 기본 — 상위 4 + 최신 체크포인트만 수리
