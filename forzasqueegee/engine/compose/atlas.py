"""차 한 대의 **면 지도** — 구도가 패널 하나에서 끝나지 않게 하는 밑판.

## 왜

지금까지 옆면은 제 안에서 완결된 구도를 짜고, 다른 면은 그 **결과의 부스러기**
를 받았다: 같은 색 세 벌, 같은 모티프 어휘, 로커 띠 하나, 그리고 이음새 너머로
투영한 무리의 뿌리(`build._anchor`). 큰 색면 자체는 옆면에서 잘리고 끝난다 —
차를 돌아가며 보면 옆면에는 구도가 있고 리어에는 띠와 조각만 있다.

빠진 것은 **이음새를 건너는 자**다. `game.fold`가 면 사이 변환을 이미 쥐고
있는데(`Fold.A`·`b`와 조각별 이음선 `Fold.segments`), 그것을 쓰는 자리가 뿌리
투영 한 곳뿐이었다.

## 무엇을 담나

    surfaces    면 지도 (`game.surface.SurfaceMap`)
    seams       면에서 나가는 이음새들 — 변환·이음선·건너편 면
    lines       차체 선 — 벨트라인 · 사이드실 · 루프라인 · 휠아치 (아는 면만)

**건너는 일 자체는 여기 없다** — `compose.seams`가 한다 (정책·기하·못 이을 때의
끊기). 이 모듈은 그 자가 물어볼 지도를 한 번만 풀어 두는 자리다. 한 번에 전체를
갈아엎지 않는다: 기존 `folds`·`rigs`·면 투영 위에 얹는 얇은 층이고, 쓰는 쪽은
필요한 것만 물어본다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ...game import fold as gfold, surface as gsurf
from .folds import _all_folds


@dataclass(frozen=True)
class Seam:
    """면 하나에서 이웃 면으로 나가는 **이음새** — 그 변환 (`game.fold.Fold`).

    변환은 조각별 이음선까지 들고 있다 (`Fold.segments`). 그것으로 무엇을 할지는
    `compose.seams`가 정한다.
    """

    src: str
    dst: str
    fold: gfold.Fold


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
