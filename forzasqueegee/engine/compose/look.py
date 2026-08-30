"""도안 읽기 — 플랜 한 장이 면에서 **어떤 생김새인가**."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from ...i18n import msg
from ..catalog import Catalog, default_catalog_path
from ..model import UNITS_PER_SCALE, Layer, LayerPlan, rgb_to_hsb


@dataclass
class Look:
    """도안의 **생김새** — 구성이 이걸 보고 자리·크기를 정한다."""

    box: tuple[float, float, float, float]     # 잉크 상자 (캔버스 유닛)
    palette: list[tuple[int, int, int]]        # 넓은 면적 순 색
    layers: int
    # 잉크의 **볼록 껍질** (캔버스 유닛, N×2). 기울여 앉힐 때 크기를 이걸로 잰다 —
    # 사각형 공식(`w·|cos|+h·|sin|`)은 그림이 상자를 다 안 채우기 때문에 회전한
    # 그림을 과대평가한다 (실측: 65° 근방에서 폭 1.46배·높이 1.76배). 과대평가는
    # 그대로 축소로 이어져 인물이 작아진다.
    hull: np.ndarray | None = None
    # `palette`와 **짝 맞춘 면적 몫** (합이 1은 아니다 — 팔레트가 상위 12색이다).
    # 색이 도안에서 얼마나 넓은가는 순위보다 훨씬 많은 것을 말한다: 2위가 1위의
    # 절반인 도안과 1위와 맞먹는 도안은 "지배색이 있나"가 서로 다르다.
    weights: list[float] = field(default_factory=list)
    # **흰 바탕이 삼키는 잉크의 몫** — 밝고 흐린(근백) 잉크의 면적 비율.
    # 베이스 흰/검을 가르는 자다 (`base_paint`).
    pale: float = 0.0

    @property
    def w(self) -> float:
        return self.box[2] - self.box[0]

    @property
    def h(self) -> float:
        return self.box[3] - self.box[1]

    @property
    def aspect(self) -> float:
        return self.w / max(1e-6, self.h)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2)

    @property
    def kind(self) -> str:
        """`tall`(전신) · `bust`(상반신·정방) · `wide`(가로) — 자리 규칙의 갈림길."""
        if self.aspect < 0.62:
            return "tall"
        if self.aspect <= 1.45:
            return "bust"
        return "wide"


def layer_points(l: Layer, cat: Catalog) -> np.ndarray:
    """레이어가 캔버스에서 차지하는 점들 (유닛). `engine.render`와 같은 식이다."""
    sh = cat.shapes.get(l.shape)
    if sh is None or not sh.loops:
        return np.zeros((0, 2), np.float32)
    pts = np.concatenate(sh.loops, axis=0) * np.array([l.sx, l.sy], np.float32)
    pts = pts * UNITS_PER_SCALE
    if l.skew:
        pts = pts + np.stack([pts[:, 1] * l.skew, np.zeros(len(pts), np.float32)], 1)
    if l.rot:
        r = math.radians(l.rot)
        c, s = math.cos(r), math.sin(r)
        pts = pts @ np.array([[c, s], [-s, c]], np.float32)
    return pts + np.array([l.x, l.y], np.float32)


def look(plan: LayerPlan, cat: Catalog | None = None,
         exclude_labels: tuple[str, ...] = ()) -> Look:
    """도안의 잉크 상자와 팔레트. **마스크 레이어는 상자에서 뺀다** (잉크가 아니다).

    `exclude_labels`는 상자에서 뺄 레이어 분류다. 배경 띠(`itasha_stripe`)를 빼는
    자리다 — 띠는 일부러 면 밖으로 흘리므로 그것까지 넣어 크기를 맞추면 **인물이
    그만큼 작아진다** (띠가 인물을 밀어낸다).
    """
    cat = cat or Catalog(default_catalog_path())
    lo = np.array([1e9, 1e9], np.float32)
    hi = np.array([-1e9, -1e9], np.float32)
    area: dict[tuple[int, int, int], float] = {}
    cloud: list[np.ndarray] = []
    for l in plan.layers:
        if l.label in exclude_labels:
            continue
        pts = layer_points(l, cat)
        if len(pts) == 0:
            continue
        if not l.mask:
            lo = np.minimum(lo, pts.min(axis=0))
            hi = np.maximum(hi, pts.max(axis=0))
            cloud.append(pts)
            sh = cat.shapes.get(l.shape)
            a = abs(sh.area) * abs(l.sx * l.sy) if sh else 0.0
            key = l.rgb()
            area[key] = area.get(key, 0.0) + a
    if hi[0] < lo[0]:
        raise ValueError(msg("빈 도안 (잉크 레이어가 없다)"))
    ranked = sorted(area.items(), key=lambda kv: -kv[1])
    pal = [c for c, _a in ranked]
    total = sum(area.values()) or 1.0
    hull = None
    if cloud:
        all_pts = np.concatenate(cloud, 0).astype(np.float32)
        hull = cv2.convexHull(all_pts.reshape(-1, 1, 2)).reshape(-1, 2)
    return Look(box=(float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])),
                palette=pal[:12], layers=len(plan.layers), hull=hull,
                weights=[a / total for _c, a in ranked[:12]],
                pale=sum(a for c, a in area.items()
                         if _is_pale(c)) / total)


# ---- 근백 잉크 — 흰 바탕에 녹는 색 (2026-08-22 렌더 실측) ----
# 흰 차에서 사라지는 것은 **밝고 흐린** 잉크다. 어두운 잉크도, 진한 유채색도 흰
# 위에서 산다 — 분홍 머리·파란 재킷은 명도가 높은데도 읽힌다.
PALE_B = 0.72              # 이보다 밝고


PALE_S = 0.25              # 이보다 흐리면 흰 바탕에 녹는다


def _is_pale(c: tuple[int, int, int]) -> bool:
    _h, s, b = rgb_to_hsb(*c)
    return b > PALE_B and s < PALE_S


def rot_ink_box(lk: Look, deg: float, mirror: bool = False
                ) -> tuple[float, float, float, float]:
    """표시 변환 `R(deg)·M`을 먹인 잉크의 **면 좌표 상자** (배율 1에서).

    돌린 뒤의 상자는 크기뿐 아니라 **중심**도 안 돌린 상자와 다르다 — 그림이
    상자를 다 안 채우기 때문이다. 배치는 상자 중심을 면 상자 중심에 맞추므로
    (`place_in_rect`) 둘 다 여기서 나와야 어긋나지 않는다.

    껍질이 없으면(빈 도안·폴백) 사각형 공식으로 물러난다. 회전 규약은
    `layer_points`와 같다: 점 p → p·[[c, s], [−s, c]].
    """
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    mx = -1.0 if mirror else 1.0
    if lk.hull is None or len(lk.hull) < 3:
        ca, sa = abs(c), abs(s)
        w, h = lk.w * ca + lk.h * sa, lk.w * sa + lk.h * ca
        lcx, lcy = lk.center
        lcx *= mx
        rcx, rcy = lcx * c - lcy * s, lcx * s + lcy * c
        return (rcx - w / 2, rcy - h / 2, rcx + w / 2, rcy + h / 2)
    q = (lk.hull * np.array([mx, 1.0], np.float32)) @ np.array(
        [[c, s], [-s, c]], np.float32)
    return (float(q[:, 0].min()), float(q[:, 1].min()),
            float(q[:, 0].max()), float(q[:, 1].max()))


def rot_ink(lk: Look, deg: float) -> tuple[float, float]:
    """`deg`만큼 돌린 도안의 **잉크 크기** (폭, 높이) — 껍질이 있으면 실측이다."""
    if not deg:
        return lk.w, lk.h
    b = rot_ink_box(lk, deg)
    return b[2] - b[0], b[3] - b[1]


def person_ink(lk: Look, tilt: float, mirror: bool = False) -> tuple[float, float]:
    """면에서 인물이 덮을 **잉크 크기** — 미러를 반영한다.

    미러 면의 표시 변환은 `R(rot)·M`이고 `R(−θ)·M = M·R(θ)`이므로 오른쪽 면의
    상자는 왼쪽 면 상자의 x반전이다 — **크기가 같다.** 미러를 빼고 재면 좌우가
    다른 배율을 받는다 (껍질이 좌우 대칭이 아니기 때문이다).
    """
    b = rot_ink_box(lk, tilt, mirror)
    return b[2] - b[0], b[3] - b[1]
