"""차 한 대의 **면 지도** — 구도가 패널 하나에서 끝나지 않게 하는 밑판.

## 왜

지금까지 옆면은 제 안에서 완결된 구도를 짜고, 다른 면은 그 **결과의 부스러기**
를 받았다: 같은 색 세 벌, 같은 모티프 어휘, 로커 띠 하나, 그리고 이음새 너머로
투영한 무리의 뿌리(`build._anchor`). 큰 색면 자체는 옆면에서 잘리고 끝난다 —
차를 돌아가며 보면 옆면에는 구도가 있고 리어에는 띠와 조각만 있다.

빠진 것은 **이음새를 건너는 자**다. `game.fold`가 면 사이 아핀 변환을 이미
쥐고 있는데(`Fold.A`·`b`), 그것을 쓰는 자리가 뿌리 투영 한 곳뿐이었다. 같은
변환으로 **방향**과 **높이**도 건널 수 있다: 옆면에서 v 높이로 각 θ로 달리는
띠가 리어에서는 어느 높이·어느 각인가.

## 무엇을 담나

    surfaces    면 지도 (`game.surface.SurfaceMap`)
    seams       면에서 나가는 이음새들 — 변환·이음선·건너편 면
    lines       차체 선 — 벨트라인 · 사이드실 · 루프라인 · 휠아치 (아는 면만)
    drawable    실제로 그려지는 마스크 (유리·아치를 뺀 것)

그리고 건너는 자 셋:

    carry_point(src, dst, p)      점을 이웃 면 좌표로
    carry_dir(src, dst, d)        방향을 이웃 면 좌표로 (아핀의 선형부)
    carry_band(src, dst, v, ang)  띠의 (높이, 각)을 이웃 면으로

한 번에 전체를 갈아엎지 않는다 — 기존 `folds`·`rigs`·면 투영 위에 얹는 얇은
층이고, 쓰는 쪽은 필요한 것만 물어본다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ...game import fold as gfold, surface as gsurf
from .folds import _all_folds


@dataclass(frozen=True)
class Seam:
    """면 하나에서 이웃 면으로 나가는 **이음새** — 변환과 그 이음선."""

    src: str
    dst: str
    fold: gfold.Fold

    @property
    def axis(self) -> str:
        return self.fold.axis

    @property
    def edge(self) -> float:
        """이음선 (src 유닛) — `axis`가 'u'면 u 값, 'v'면 v 값."""
        return float(self.fold.edge)

    @property
    def sign(self) -> float:
        return float(self.fold.sign)

    def point(self, u: float, v: float) -> tuple[float, float]:
        pu, pv = self.fold.to(u, v)
        return float(pu), float(pv)

    def direction(self, d: tuple[float, float]) -> tuple[float, float]:
        """방향 하나를 dst 유닛으로 — 아핀의 **선형부만** 먹인다 (평행이동 없음).

        차체 면끼리의 `A`는 부호 순열이라(`game.fold`) 방향이 축을 바꿔 가며
        그대로 건너간다. 유리 이음새는 비등방이라 길이가 안 보존되므로 다시
        단위벡터로 만든다.
        """
        a = self.fold.A
        x = float(a[0, 0]) * d[0] + float(a[0, 1]) * d[1]
        y = float(a[1, 0]) * d[0] + float(a[1, 1]) * d[1]
        n = math.hypot(x, y)
        return (x / n, y / n) if n > 1e-9 else (1.0, 0.0)


@dataclass(frozen=True)
class BodyLines:
    """면 하나가 아는 **차체 선** (면 유닛). 모르는 것은 None이다."""

    belt: float | None = None          # 벨트라인 (v)
    sill: float | None = None          # 사이드실 (v)
    roof: float | None = None          # 루프라인 (v)
    rocker: float | None = None        # 하부 투톤의 윗선 (v)
    arches: tuple[tuple[float, float], ...] = ()   # 휠아치 중심 (u, v)
    rear_dir: float = 1.0              # +u가 차 뒤면 +1


@dataclass
class VehicleAtlas:
    surfaces: dict[str, gsurf.SurfaceMap] = field(default_factory=dict)
    seams: dict[str, tuple[Seam, ...]] = field(default_factory=dict)
    lines: dict[str, BodyLines] = field(default_factory=dict)

    # ---- 잇기 ----
    def seam_to(self, src: str, dst: str) -> Seam | None:
        return next((s for s in self.seams.get(src, ()) if s.dst == dst), None)

    def neighbours(self, src: str) -> tuple[str, ...]:
        return tuple(s.dst for s in self.seams.get(src, ()))

    def carry_point(self, src: str, dst: str, p: tuple[float, float]
                    ) -> tuple[float, float] | None:
        s = self.seam_to(src, dst)
        return s.point(p[0], p[1]) if s is not None else None

    def carry_dir(self, src: str, dst: str, d: tuple[float, float]
                  ) -> tuple[float, float] | None:
        s = self.seam_to(src, dst)
        return s.direction(d) if s is not None else None

    def carry_band(self, src: str, dst: str, v: float, ang: float
                   ) -> tuple[float, float] | None:
        """띠 하나를 이웃 면으로 — 되돌림은 (그 면에서의 높이 v', 각 ang').

        띠의 **이음선 위 점**을 건너보내 높이를 얻고, 방향을 건너보내 각을 얻는다.
        그래야 두 면의 띠가 이음새에서 같은 자리·같은 기울기로 만난다 (사람이
        만든 리버리는 면을 정밀하게 잇는 대신 이 둘을 맞추고 전환을 이음새에
        숨긴다 — `surfshapes.flow_shapes` 문서와 같은 관찰이다).
        """
        s = self.seam_to(src, dst)
        if s is None:
            return None
        # 이음선 위에서 띠의 중심선이 지나는 점
        if s.axis == "u":
            pu, pv = s.point(s.edge, v)
        else:
            # 이음선이 v 상수선이면 띠의 높이가 곧 이음선이라 자리가 아니라
            # **길이 방향**이 건너간다 — 그 면에서는 띠가 세로로 선다.
            pu, pv = s.point(0.0, s.edge)
        d = s.direction((math.cos(math.radians(ang)), math.sin(math.radians(ang))))
        sm = self.surfaces.get(dst)
        if sm is not None:
            u0, v0, u1, v1 = sm.paint
            if not (v0 - 0.35 * (v1 - v0) <= pv <= v1 + 0.35 * (v1 - v0)):
                return None                        # 건너간 높이가 면 밖이다
        return float(pv), math.degrees(math.atan2(d[1], d[0]))


def _lines_for(name: str, maps: dict, rigs: dict) -> BodyLines:
    r = rigs.get(name)
    if r is None:
        return BodyLines()
    g = r.geom
    wheels = tuple((float(w[0]), float(w[1]))
                   for w in (getattr(g, "wheels", None) or ())
                   if len(w) >= 2)
    sill = float(getattr(g, "sill", 0.0) or 0.0)
    belt = float(getattr(g, "belt", 0.0) or 0.0)
    from .bands import ROCKER_FRAC
    rocker = sill + ROCKER_FRAC * (belt - sill) if belt > sill else None
    return BodyLines(belt=belt or None, sill=sill or None,
                     roof=float(getattr(g, "roof", 0.0) or 0.0) or None,
                     rocker=rocker, arches=wheels,
                     rear_dir=float(getattr(r, "rear_dir", 1.0)))


def build_atlas(maps: dict[str, gsurf.SurfaceMap], rigs: dict,
                media: str | None = None) -> VehicleAtlas:
    """면 지도 + 옆면 뼈대 → 차 한 대의 지도.

    이음새는 `folds._all_folds`가 이미 푸는 것을 담기만 한다 (차체 모서리는
    껍질이, 유리는 실측 이음새·필러 프로필이 낸다). 못 푸는 차는 그 면의
    이음새가 빈 튜플이고, 쓰는 쪽은 이어 붙이기를 접는다.
    """
    at = VehicleAtlas(surfaces=dict(maps))
    for name in maps:
        try:
            got = [f for f in _all_folds(name, maps, rigs, media=media)
                   if f.dst in maps]
        except Exception:                          # 지도가 모자란 차 — 이 면은 못 잇는다
            got = []
        at.seams[name] = tuple(Seam(src=name, dst=f.dst, fold=f) for f in got)
        at.lines[name] = _lines_for(name, maps, rigs)
    return at
