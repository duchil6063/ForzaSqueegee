"""vinyl_catalog.json 로더 + 도형 기하 유틸.

카탈로그 좌표: y-up, loops는 외곽선 폐루프(구멍 포함 가능).

**native 상자 보정** (55차, `catalog/shape_native.json` — 게임 에셋의
BBox 청크에서 뽑은 표): 원본 카탈로그는 도형마다 **축별로** ±1로
정규화해 실제 크기·종횡비·중심을 잃었다. 게임은 도형을 설계 크기 그대로
(스케일 1.0 = 128유닛이 상자 0.5) **설계 원점을 레이어 좌표에 맞춰** 그린다.
그래서 카탈로그 그대로 sx·sy를 계산해 찍으면 크기가 어긋나고(1,282/1,480종),
상자가 원점 대칭이 아닌 도형은 위치까지 밀린다(29종, 최대 도형 높이의 17%).

    보정 loops = pts × [nx, ny] + [tx, ty]

로드 시점에 loops에 넣어 이후 `loop × (sx,sy) × 128`을 쓰는 곳
(render·sortplan·bordermask·overlay)과 loops bbox에서 배치를 역산하는 곳이
전부 자동으로 맞는다. `rasterize`는 loops의 bbox를 기준으로 펴므로
**도형 대조 결과는 보정 전과 같다**.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class CatShape:
    name: str
    group: str
    area: float
    # 각 (N,2) float32. **±1이 아니다** — 대개 ±1이지만 꽃·물감·결정 계열은
    # ±1.65까지 간다 (2026-08-22 실측: I_12 1.666 · G_01 1.657 · I_31 1.635),
    # 스월은 세로가 ±0.17뿐이다. 크기를 유닛으로 정해 놓는 쪽은 `reach`로
    # 나눠야 도형이 바뀌어도 같은 크기로 선다.
    loops: tuple[np.ndarray, ...]
    # 그라데이션 도형(39차 인게임 실측): {"kind": "radial"|"linear",
    # "profile": [80bin], "bin_scale": 64} — 알파 = profile[좌표×bin_scale].
    # radial은 r=√(x²+y²), linear는 |x| (도형 로컬 정규화 좌표). None = 불투명
    gradient: dict | None = None
    # 정점 알파(에셋 COLOR 속성)에서 나온 두 칸.
    # opaque=False면 인게임이 **도안보다 옅게** 그린다. alpha_area = 면적 가중
    # 평균 알파(0~1) = 실제로 올라가는 잉크의 몫
    opaque: bool = True
    alpha_area: float = 1.0

    @property
    def reach(self) -> float:
        """이 도형이 실제로 뻗는 **최대 반경** (로컬 좌표). 빈 도형은 1.0.

        스케일 1이면 도형이 `reach × UNITS_PER_SCALE` 유닛까지 뻗는다. 크기를
        "인물 높이의 몇 할"처럼 유닛으로 정하는 쪽은 이 값으로 나눠야 계열이
        달라져도 같은 크기가 나온다 — 안 나누면 꽃·물감이 별보다 1.65배 크게
        선다 (2026-08-22 판정: 인물만 한 꽃이 얼굴을 덮었다).
        """
        if not self.loops:
            return 1.0
        r = max(float(np.abs(l).max()) for l in self.loops if len(l))
        return r if r > 1e-6 else 1.0

    def rasterize(self, size: int = 128) -> np.ndarray:
        """짝홀 규칙으로 채운 bool 마스크 (size×size).

        **loops의 bbox**를 정사각 래스터에 편다 — native 상자 보정을 넣어도
        도형 대조(IoU) 결과가 변하지 않도록 하기 위한 것이다. 보정 전에는 모든
        도형이 정확히 ±1이라 이 식이 옛 식과 같은 그림을 낸다.
        """
        if not self.loops:
            return np.zeros((size, size), bool)   # 빈 도형 (카탈로그에 소수 있다)
        pts_all = np.concatenate(self.loops, axis=0)
        lo = pts_all.min(axis=0)
        span = np.maximum(pts_all.max(axis=0) - lo, 1e-6)
        acc = np.zeros((size, size), np.uint8)
        for loop in self.loops:
            pts = np.round((loop - lo) / span * (size - 1)).astype(np.int32)
            pts[:, 1] = size - 1 - pts[:, 1]  # y-up → 이미지 y-down
            m = np.zeros((size, size), np.uint8)
            cv2.fillPoly(m, [pts], 1)
            acc ^= m
        return acc.astype(bool)


class Catalog:
    def __init__(self, path: str | Path, native: bool = True):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        nat = {}
        npath = Path(path).parent / "shape_native.json"
        if native and npath.exists():
            nat = json.loads(npath.read_text(encoding="utf-8"))["native"]

        def _loops(entry):
            v = nat.get(entry["name"], (1.0, 1.0, 0.0, 0.0))
            k, t = np.asarray(v[:2], np.float32), np.asarray(v[2:], np.float32)
            return tuple(np.asarray(l["pts"], np.float32) * k + t for l in entry["loops"])

        self.shapes: dict[str, CatShape] = {}
        for e in data["shapes"]:
            self.shapes[e["name"]] = CatShape(
                e["name"], e["group"], e["area"], _loops(e), None,
                bool(e.get("opaque", True)), float(e.get("alpha_area", 1.0)))
        # 그라데이션 도형 (39차 실측, 있으면 추가 로드): 같은 폴더
        # gradient_catalog.json — 실루엣 loops + 알파 프로파일
        gpath = Path(path).parent / "gradient_catalog.json"
        if gpath.exists():
            gdata = json.loads(gpath.read_text(encoding="utf-8"))
            for e in gdata["shapes"]:
                self.shapes[e["name"]] = CatShape(
                    e["name"], e["group"], e["area"], _loops(e), e["gradient"],
                    False, 0.0)

    def __getitem__(self, name: str) -> CatShape:
        return self.shapes[name]

    def _best_iou(self, ideal: np.ndarray, candidates: list[CatShape]) -> str:
        best, best_iou = "", -1.0
        for sh in candidates:
            m = sh.rasterize(ideal.shape[0])
            iou = np.logical_and(m, ideal).sum() / max(1, np.logical_or(m, ideal).sum())
            if iou > best_iou:
                best, best_iou = sh.name, iou
        return best

    @cached_property
    def square(self) -> str:
        """꽉 찬 정사각형 도형 이름 (얇은 직사각형/스트로크용으로도 사용)."""
        size = 128
        ideal = np.ones((size, size), bool)
        cands = [s for s in self.shapes.values() if len(s.loops) == 1 and abs(s.area - 4.0) < 0.2]
        return self._best_iou(ideal, cands) if cands else self._best_iou(ideal, list(self.shapes.values()))

    @cached_property
    def circle(self) -> str:
        """원에 가장 가까운 단일 루프 도형 이름 (타원 근사용)."""
        size = 128
        yy, xx = np.mgrid[0:size, 0:size]
        c = (size - 1) / 2
        ideal = (xx - c) ** 2 + (yy - c) ** 2 <= c**2
        cands = [
            s
            for s in self.shapes.values()
            if len(s.loops) == 1 and len(s.loops[0]) >= 12 and 2.8 < s.area < 3.5
        ]
        return self._best_iou(ideal, cands) if cands else self._best_iou(ideal, list(self.shapes.values()))


def default_catalog_path() -> Path:
    return Path(__file__).resolve().parents[2] / "catalog" / "vinyl_catalog.json"
