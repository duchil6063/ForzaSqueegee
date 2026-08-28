"""즐겨찾기 색 스택 모델 — 게임 프로필의 즐겨찾기 리스트를 로컬로 추적.

게임 리스트는 프로필 전역·영구(크래시에도 유지). 등록(HSB 화면 Y)은 prepend,
정확 동일 색 재등록은 기존 항목이 맨 위로 이동(dedupe). 이 모델은 자동화가
등록한 색만 추적하고, 미지 항목(수동 등록·크래시 정크)은 아래로 밀릴 뿐이다.
모델이 실제와 어긋나면 fav_pick의 행 색 대조가 감지 → 호출측이 forget 후
HSB 폴백 + 재등록으로 자기 치유한다.
"""

from __future__ import annotations

import colorsys
import json
from pathlib import Path

HSB = tuple[float, float, float]


def default_path() -> Path:
    from ..paths import work_file

    return work_file("state", "fav_stack.json")


def _key(hsb) -> HSB:
    return tuple(round(float(v), 2) for v in hsb)


def hsb_rgb(hsb) -> tuple[float, float, float]:
    """게임 스와치 RGB (표준 HSV 변환, 인게임 실측 일치 — 2026-08-02)."""
    r, g, b = colorsys.hsv_to_rgb(*_key(hsb))
    return (r * 255.0, g * 255.0, b * 255.0)


class FavStack:
    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else default_path()
        self.stack: list[HSB] = []  # index 0 = 게임 리스트 맨 위
        if self.path.exists():
            self.stack = [_key(c) for c in json.loads(self.path.read_text(encoding="utf-8"))]

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.stack), encoding="utf-8")

    def index(self, hsb) -> int | None:
        try:
            return self.stack.index(_key(hsb))
        except ValueError:
            return None

    def register(self, hsb) -> None:
        """게임에 Y 등록한 직후 호출 — 맨 위 삽입(있으면 이동)."""
        k = _key(hsb)
        if k in self.stack:
            self.stack.remove(k)
        self.stack.insert(0, k)
        self._save()

    def forget(self, hsb) -> None:
        """검증 실패한 색을 모델에서 제거 (다음엔 HSB 경로 + 재등록)."""
        k = _key(hsb)
        if k in self.stack:
            self.stack.remove(k)
            self._save()
