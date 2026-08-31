"""상자 산술과 자잘한 자 — 이 패키지의 밑판 (아무것도 안 쓴다)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from ...game import surface as gsurf


# 캔버스 유닛 1 = 면 유닛 몇인가 (불러온 그룹 스케일 1.0에서). 실측으로 확정하는
# 값이라 면 지도 파일이 쥐고, 없으면 1.0으로 본다 — 배치 검증이 이 값을 잰다.
DEFAULT_GROUP_UNIT = 1.0


# 비닐 캔버스는 **긴 변 900유닛 고정**이다 (`engine.celfit`·`game.inject`가 같은
# 값을 쓴다 — 도안의 `units_per_px`도 긴 변을 900에 맞춘다). 캔버스 밖에 앉은
# 레이어는 저장·불러오기 뒤에 **게임이 안 그린다.**
CANVAS_UNITS = 900.0


def major_axis(xs: np.ndarray, ys: np.ndarray) -> tuple[tuple[float, float], float]:
    """점 구름의 **장축 단위벡터와 장/단축 비** — 2차 모멘트의 닫힌 식.

    ## 왜 `np.linalg.eigh`를 안 쓰나

    LAPACK은 스레드 수에 따라 마지막 비트가 흔들린다. 구도 점수는 상위 후보가
    0.001 안에 몰리는 일이 흔해서 그 흔들림이 그대로 **순위를 뒤집고**, 같은
    입력이 프로세스마다 다른 파일을 냈다 (2026-09-01 실측: 한 프로세스 안에서는
    세 번 다 같은데 프로세스를 새로 뜨면 `deco.json` 해시가 두 값 사이를 오갔다.
    옛 주석이 공분산을 명시적 합으로 쌓으라 한 것과 같은 사정이고, 남은 구멍이
    고윳값 분해였다).

    2×2 대칭 행렬은 고윳값 분해에 닫힌 식이 있다 — 각 θ = ½·atan2(2b, a−c)가
    장축이고 고윳값은 (a+c)/2 ± √(((a−c)/2)² + b²)다. 정확하고, 더 싸고,
    스레드와 무관하다.

    점이 모자라면 세로축 `(0, 1)`과 비 1.0.
    """
    n = len(xs)
    if n < 3:
        return (0.0, 1.0), 1.0
    x = xs.astype(np.float64) - xs.mean()
    y = ys.astype(np.float64) - ys.mean()
    a = float((x * x).sum()) / n
    b = float((x * y).sum()) / n
    c = float((y * y).sum()) / n
    th = 0.5 * math.atan2(2.0 * b, a - c)
    mid = 0.5 * (a + c)
    rad = math.hypot(0.5 * (a - c), b)
    hi, lo = mid + rad, max(mid - rad, 1e-12)
    return (math.cos(th), math.sin(th)), math.sqrt(hi / lo)


def _union(a: tuple[float, float, float, float],
           b: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _overlap(a: tuple[float, float, float, float],
             b: tuple[float, float, float, float]) -> float:
    """두 상자가 겹치는 넓이. 안 겹치면 0 — 어느 이웃이 이 면의 앵커냐를 가른다."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return w * h if w > 0 and h > 0 else 0.0


def _gap(a: tuple[float, float, float, float],
         b: tuple[float, float, float, float]) -> float:
    """상자 둘 사이의 **틈** (겹치면 0). 투영이 여럿일 때 가까운 것을 고르는 자."""
    du = max(0.0, max(a[0] - b[2], b[0] - a[2]))
    dv = max(0.0, max(a[1] - b[3], b[1] - a[3]))
    return math.hypot(du, dv)


def _clamp_box(box: tuple[float, float, float, float],
               into: tuple[float, float, float, float]
               ) -> tuple[float, float, float, float]:
    """상자를 `into` 안으로 **크기를 지키며** 민다 (안 들어가면 그만큼 줄인다).

    이음새 너머로 투영된 도안 상자를 그 면의 뿌리로 쓰는 자리에서 쓴다 — 투영은
    대개 패널 밖이라, 밀어 넣어야 무리가 이음새 가장자리에서 자란다.
    """
    hw = min((box[2] - box[0]) / 2, (into[2] - into[0]) / 2)
    hh = min((box[3] - box[1]) / 2, (into[3] - into[1]) / 2)
    cu = min(max((box[0] + box[2]) / 2, into[0] + hw), into[2] - hw)
    cv = min(max((box[1] + box[3]) / 2, into[1] + hh), into[3] - hh)
    return (cu - hw, cv - hh, cu + hw, cv + hh)


def _face_phase(name: str) -> float:
    """면마다 다른 **나선 위상** — 같은 배열이 면마다 되풀이되지 않게.

    옛 산포는 면마다 같은 황금각 나선을 상자 크기만 바꿔 복제했다: 한 구성의
    front·rear·window_left·window_right가 도형 순서도 회전값도 전부 같았다
    (`A_08 0° · A_18 174.5° · G_01 349° …`). 차를 한 바퀴 돌면 같은 무늬가 네 번
    나온다. 이름으로 위상을 흔들면 결정성은 그대로고(같은 차 → 같은 그림) 면끼리
    안 닮는다.
    """
    return (sum(ord(c) * (i + 1) for i, c in enumerate(name)) % 360) * math.pi / 180.0


def _group_unit(car: str | None) -> float:
    """면 지도가 쥔 `group_unit` (캔버스 유닛 → 면 유닛). 없으면 1.0."""
    if not car:
        return DEFAULT_GROUP_UNIT
    p = gsurf.map_dir() / f"{gsurf.car_slug(car)}.json"
    if not p.exists():
        return DEFAULT_GROUP_UNIT
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return DEFAULT_GROUP_UNIT
    return float(raw.get("group_unit") or DEFAULT_GROUP_UNIT)


def _rel(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()
