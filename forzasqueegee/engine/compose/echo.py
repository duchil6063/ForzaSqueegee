"""그래픽 에코 — 캐릭터 안의 **시각 언어를 차체에 되풀이**한다.

"파랑이면 별, 분홍이면 꽃"은 색 규칙이지 형태 규칙이 아니다. 사람이 만든
리버리는 인물의 결을 차체에 이어 붙인다 — 머리카락 방향을 닮은 사선 조각,
뾰족한 실루엣을 닮은 샤드, 의상의 어두운 블록을 닮은 잔 사각. 여기서 내는
것은 산포 모티프와 **별개의 잔 조각**들이고, 무엇을 낼지는 도안 뜻이 정한다:

- **결 조각(streak)** — 결이 한 방향이면 (`texture_coherence`) 그 방향의 가는
  막대 (A_20 — 한쪽이 가늘어지는 바).
- **샤드(shard)** — 인상이 뾰족하면 (`impression == sharp`) 삼각 파편 (A_37·A_03).
- **블록(block)** — 어두운 무채 씨앗이 있고 빽빽하면 작은 사각 블록.
"""

from __future__ import annotations

import math

import numpy as np

from ..catalog import Catalog
from ..model import UNITS_PER_SCALE, Layer
from .field import CompositionField
from .intent import DesignIntent
from .roles import RolePalette


LABEL = "itasha_echo"


STREAK = ("A_20", "A_01")
SHARD = ("A_37", "A_03", "A_04")
BLOCK = ("A_01",)


def _spots(fld: CompositionField, n: int, phase: float, along: tuple[float, float],
           min_sep: float) -> list[tuple[float, float]]:
    """장식 구역에서 **흐름을 따라** 띄엄띄엄 고른 자리들 (결정적)."""
    g = fld.grid
    ys, xs = np.where(fld.decoration > 0.45)
    if len(xs) == 0:
        return []
    px = g.x0 + (xs + 0.5) * g.cell
    py = g.y_top - (ys + 0.5) * g.cell
    vcx, vcy = fld.visual_center
    # 흐름 방향으로 인물에서 떨어진 정도로 정렬, 황금각으로 골라 몰리지 않게
    key = (px - vcx) * along[0] + (py - vcy) * along[1]
    order = np.argsort(key)
    pick: list[tuple[float, float]] = []
    step = max(1, len(order) // max(1, n * 5))
    i = int(phase * step) % max(1, step)
    while i < len(order) and len(pick) < n:
        x, y = float(px[order[i]]), float(py[order[i]])
        if all(math.hypot(x - a, y - b) >= min_sep for a, b in pick):
            pick.append((x, y))
        i += step
    return pick


def echo_layers(fld: CompositionField, it: DesignIntent, pal: RolePalette,
                cat: Catalog, n: int = 6, phase: float = 0.0) -> list[Layer]:
    if n <= 0:
        return []
    ch = fld.char_h
    out: list[Layer] = []
    kinds: list[str] = []
    if it.flow_coherence >= 0.25:
        kinds.append("streak")
    if it.impression == "sharp" or it.angularity >= 0.45:
        kinds.append("shard")
    if it.dark_neutral_rgb is not None and it.density > 0.55:
        kinds.append("block")
    if not kinds:
        kinds.append("shard" if it.angularity >= 0.3 else "streak")
    kind = kinds[0]
    # 결 조각의 각은 **베드 축**이다 — 결 방향 측정은 세로 획이 많은 셀 그림에서
    # 거의 늘 세로로 나와 (11장 실측) 조각이 판을 가로지르는 막대가 된다.
    from .bed import slab_axis

    tx, ty = slab_axis(fld)
    if kind == "streak" and it.flow_coherence >= 0.40:
        tx, ty = fld.texture
    ang = math.degrees(math.atan2(ty, tx))
    spots = _spots(fld, n, phase, fld.flow, min_sep=0.22 * ch)
    colors = (pal.highlight, pal.secondary, pal.primary)
    for j, (x, y) in enumerate(spots):
        k = 0.06 + 0.10 * ((j * 7 % 5) / 4.0)          # 인물 높이의 6~16%
        col = colors[j % len(colors)]
        if kind == "streak":
            name = STREAK[j % len(STREAK)]
            length = ch * (0.35 + 0.45 * ((j * 3 % 4) / 3.0))
            out.append(Layer(shape=name, x=x, y=y,
                             sx=k * ch * 0.18 / UNITS_PER_SCALE,
                             sy=length / 2 / UNITS_PER_SCALE,
                             rot=(ang - 90.0 + (j * 11 % 7 - 3)) % 360.0,
                             color=col, alpha=92.0, label=LABEL))
        elif kind == "shard":
            name = SHARD[j % len(SHARD)]
            sh = cat.shapes.get(name)
            reach = sh.reach if sh is not None else 1.0
            size = k * ch * 1.6
            out.append(Layer(shape=name, x=x, y=y,
                             sx=size / 2 / UNITS_PER_SCALE / reach,
                             sy=size * 1.6 / 2 / UNITS_PER_SCALE / reach,
                             rot=(ang - 90.0 + 180.0 * (j % 2) + (j * 13 % 21 - 10)) % 360.0,
                             color=col, alpha=100.0, label=LABEL))
        else:
            size = k * ch * 1.2
            out.append(Layer(shape=BLOCK[0], x=x, y=y,
                             sx=size / 2 / UNITS_PER_SCALE,
                             sy=size * (0.6 + 0.4 * (j % 2)) / 2 / UNITS_PER_SCALE,
                             rot=(ang + (j * 5 % 3) * 4.0) % 360.0,
                             color=pal.dark if j % 3 else col, alpha=100.0, label=LABEL))
    return out
