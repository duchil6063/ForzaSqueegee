"""리듬 — 조각을 **하나의 흐름**으로 놓는다.

## 왜

옛 산포는 황금각 나선에서 후보를 뜨고 뭉치는 자리에 가까운 순으로 골랐다
(`scatter.scatter_motifs`). 자리는 고르게 퍼지고 크기는 거리로 층을 갈랐지만,
결과는 "뿌린 것"으로 읽힌다 — 조각들이 **어디서 와서 어디로 가는지**가 없기
때문이다. 사람이 그린 무리는 큰 것 하나에서 시작해 잔것으로 잦아들며 한 방향
으로 흐른다 (레퍼런스 여덟 장 전부).

## 곡선 하나

리듬은 곡선 하나다: 원점에서 출발해 일정한 비율로 **꺾이고**(곡률), 걸음이
**벌어지고**(간격 등비), 조각이 **잦아든다**(크기 등비). 곁가지는 곡선에서
법선 방향으로 조금씩 벗어나 한 줄로 서지 않게 한다 (황금각 위상 — 결정적이다).

    origin      어디서 나오나 (인물 곁 · 큰 색면 가장자리)
    direction   어디로 가나
    curvature   전체 길이에 걸쳐 몇 도 꺾이나
    spacing     첫 걸음과 그 등비 (1.0이면 등간격 = 기계, 1.2면 벌어진다)
    size_decay  조각 크기의 등비 (0.8이면 다섯째가 첫째의 0.41배)
    rot_step    조각이 걸음마다 몇 도 도나
    lateral     곡선에서 벗어나는 폭 (크기 대비)
    strands     가닥 수 (둘이면 굵은 줄기 + 짧은 곁줄기)

자리 검사(`place_ok`)에 걸린 걸음은 **건너뛰되 걸음 수는 센다** — 그래야 크기가
거리를 따라 잦아드는 관계가 안 깨진다. 살아남은 것이 너무 적으면 부르는 쪽이
옛 산포로 물러난다 (`design._scatter` — 황금각은 결정적 폴백으로 남는다).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# 곁가지의 황금각 — 곡선에서 벗어나는 몫이 되풀이되지 않게 (결정적).
GOLDEN = 2.399963


# 걸음 하나가 조각 크기의 몇 배인가 (첫 걸음). 1.0이면 외접 원이 딱 닿는다 —
# 레퍼런스의 무리는 서로 모서리를 물므로 그보다 촘촘하다.
STEP_OF_SIZE = 0.82


@dataclass(frozen=True)
class RhythmCurve:
    origin: tuple[float, float]
    direction: tuple[float, float]
    n: int
    size0: float
    curvature: float = 0.0          # 전체 길이에 걸친 회전 (도)
    length: float = 1.0             # 곡률을 나누는 기준 길이 (유닛)
    spacing_growth: float = 1.16    # 걸음의 등비
    size_decay: float = 0.80        # 크기의 등비
    rot0: float = 0.0
    rot_step: float = 37.0
    lateral: float = 0.55           # 곡선에서 벗어나는 폭 (조각 크기 대비)
    strands: int = 1
    phase: float = 0.0
    step0: float = 0.0              # 첫 걸음 (0이면 `STEP_OF_SIZE × size0`)


@dataclass(frozen=True)
class Beat:
    """리듬의 한 걸음 — 자리·크기·각·순서."""

    x: float
    y: float
    size: float
    rot: float
    i: int                          # 몇 번째 걸음인가 (0이 가장 크다)
    strand: int


def beats(c: RhythmCurve) -> list[Beat]:
    """곡선을 걸어 나오는 걸음들 (결정적).

    가닥이 여럿이면 둘째부터는 첫 가닥의 중간에서 갈라져 나가고 더 짧다 —
    무리가 한 줄로 서지 않으면서도 한 흐름으로 읽힌다.
    """
    out: list[Beat] = []
    for s in range(max(1, c.strands)):
        # 곁가지는 줄기의 1/3 지점에서 갈라져 각이 조금 벌어지고 잔것부터 시작한다
        frac = 0.0 if s == 0 else 0.30 + 0.18 * (s - 1)
        n = c.n if s == 0 else max(2, int(c.n * 0.45))
        size = c.size0 * (c.size_decay ** (frac * c.n)) * (1.0 if s == 0 else 0.62)
        base_step = c.step0 if c.step0 > 0 else STEP_OF_SIZE * c.size0
        gap = base_step * (c.spacing_growth ** (frac * c.n)) * (1.0 if s == 0 else 0.7)
        head = math.degrees(math.atan2(c.direction[1], c.direction[0])) \
            + c.curvature * frac + (0.0 if s == 0 else (-22.0 if s % 2 else 22.0))
        x, y = c.origin
        # 곁가지의 출발점은 줄기를 그만큼 따라간 자리
        if s:
            hh = math.degrees(math.atan2(c.direction[1], c.direction[0]))
            d0 = frac * c.length
            r0 = math.radians(hh + c.curvature * frac * 0.5)
            x, y = x + math.cos(r0) * d0, y + math.sin(r0) * d0
        for i in range(n):
            r = math.radians(head)
            nx, ny = -math.sin(r), math.cos(r)
            off = c.lateral * size * math.sin(c.phase + GOLDEN * (i + 3 * s))
            out.append(Beat(x=x + nx * off, y=y + ny * off, size=size,
                            rot=(c.rot0 + c.rot_step * (i + 2 * s)) % 360.0,
                            i=i, strand=s))
            x += math.cos(r) * gap
            y += math.sin(r) * gap
            # 곡률은 **길이에 대해** 도는 비율이다 — 걸음이 벌어져도 호가 같다
            head += c.curvature * gap / max(1e-6, c.length)
            gap *= c.spacing_growth
            size *= c.size_decay
    return out


def tier_of(i: int, n: int) -> int:
    """걸음 번호 → 층 (0이 최대형). 크기는 이미 등비로 잦아드므로 층은 **기록**이다.

    `score._cluster_stats`와 `critic.rhythm`이 층을 읽는다 (고아 판정·층 수).
    """
    if i == 0:
        return 0
    if i <= 2:
        return 1
    if i <= 5:
        return 2
    return 3


def curve_for(*, origin: tuple[float, float], direction: tuple[float, float],
              reach: float, n: int, size0: float, angularity: float,
              phase: float = 0.0, strands: int = 1) -> RhythmCurve:
    """구도 값에서 곡선 한 벌 — 뾰족한 그림은 곧게, 둥근 그림은 휘게.

    `reach`는 원점에서 갈 수 있는 거리(프레임 끝까지)다 — 걸음의 등비를 그
    안에 들어오게 잡는다.
    """
    # 등비 급수의 합이 `reach`를 넘지 않게 첫 걸음을 잡는다:
    #   Σ g·r^i = g·(r^n − 1)/(r − 1)
    r = 1.14
    step0 = STEP_OF_SIZE * size0
    total = step0 * (r ** n - 1) / (r - 1) if abs(r - 1) > 1e-6 else step0 * n
    if total > reach > 0:
        step0 *= reach / total
    # 곡률 — 둥근 인상일수록 휜다 (뾰족한 그림에 휜 무리를 놓으면 따로 논다)
    curve = (1.0 - max(0.0, min(1.0, angularity))) * 46.0
    return RhythmCurve(
        origin=origin, direction=direction, n=max(1, n), size0=size0,
        curvature=curve, length=max(1e-6, reach),
        spacing_growth=r, size_decay=0.82, step0=step0,
        rot0=(phase * 57.29578) % 360.0, rot_step=37.0,
        lateral=0.62, strands=max(1, strands), phase=phase)
