"""지붕 블랙아웃 — 윗면의 후드 뒤 구간을 검정으로 덮는다."""

from __future__ import annotations

import numpy as np

from ...game import surface as gsurf
from ..catalog import Catalog
from ..model import UNITS_PER_SCALE
from .bands import TEETH_OVERLAP
from .surfshapes import band_tiles, surface_sx_cap


# ---- 지붕 블랙아웃 (영상 문법 18 · HINATA 검정 지붕 · 수이세이 검정 펜더) ----
# 사람이 만든 이타샤의 베이스는 전부 투톤이다 — 가장 흔한 꼴이 지붕·필러 블랙.
# 도료로는 못 칠한다 — 도색 부품 그리드 아홉 칸을 실측했더니 차체 패널은
# 후드·윙·미러뿐이고 **지붕·필러 칸이 없다** (`catalog/paint_parts.json`).
# 그래서 윗면 비닐로 지붕 띠를 덮는 것이 유일한 길이다. 후드(최대 덩어리)는
# 남긴다 — 후드 인물·아트가 올라가는 자리다.
ROOF_DARK = (16, 17, 20)


ROOF_MIN_FRAC = 0.05           # 이 몫보다 작은 덩어리는 안 덮는다 (몰딩 조각)


# 투톤 경계를 뜯는 조각 수 (앞쪽 한 변). 레퍼런스의 투톤 경계는 곧은 선이
# 아니다 — Evo IX의 빨강↔검정은 찢긴 가장자리이고, 곧게 두면 도색 견본이 된다.
ROOF_TEETH = 4


# 지붕 투톤 경계의 **진폭** — 덮는 구간 길이(차 길이 방향)의 몫이다.
ROOF_TEETH_AMP = 0.16


# 판의 **앞선**이 구간 앞끝에서 물러나는 몫 — 뜯는 조각이 이 자리를 걸터앉아
# 곧은 선을 지운다. 여기만 물러난다: 앞선은 투톤 경계라 그림의 일부다.
ROOF_HEAD_IN = 0.02


# 판의 **뒷선**이 구간 뒤끝을 넘는 몫. 뒤는 그림이 아니라 그냥 끝이라 곧은
# 단면이 남으면 안 된다 — 넘긴 몫은 유리 구멍(또는 차 끝)이 자른다. 옛 값은
# 앞뒤를 같이 2% 물렸고(0.96), 그 바람에 판과 뒷유리 사이에 본색 실선이 남아
# 판이 "지붕에 얹은 검은 사각"으로 읽혔다.
ROOF_TAIL_OVER = 0.06


def hood_index(segs: list[tuple[float, float, float, float]],
               hood_u: float | None) -> int:
    """구간 목록에서 **후드 구간의 인덱스**. 근거가 없으면 0(맨 앞)이다.

    맨 앞 구간을 후드로 보는 것은 축 규약에서 나온 어림이고, 앞 스플리터·노즈
    벤트가 짧은 구간을 하나 더 내는 차에서는 틀린다 (실측 6/106). 설치 파일의
    후드 로케이터가 있으면 그 점이 든 구간이 후드다 (`game.locators.hood_u` —
    이미 갈린 106대에서 94% 일치).
    """
    if hood_u is None:
        return 0
    for i, s in enumerate(segs):
        if s[0] <= hood_u <= s[2]:
            return i
    return 0


def roof_blackout(smap: gsurf.SurfaceMap,
                  shapes: tuple[str, ...] | None = None,
                  hood_u: float | None = None,
                  cat: Catalog | None = None) -> list[dict]:
    """윗면에서 **후드 뒤 구간**(지붕·리어데크)을 검정 사각으로 덮는다.

    덩어리 나누기로는 못 가른다 — A필러 도색이 후드와 지붕을 잇는 차가 많다
    (인테그라: 윗면 전체가 한 덩어리). 대신 **중앙 밴드의 u-프로파일**로 가른다:
    차 가운데 띠(|v| < 0.35×반높이)의 도색 비율을 u마다 재면 유리(앞유리·선루프·
    뒷유리)가 구멍으로 나타난다. 맨 앞(-u가 앞 — 아틀라스 축 규약) 구간이
    후드이고, 첫 구멍 뒤 구간들이 지붕·데크다.
    """
    segs = top_segments(smap)
    if len(segs) < 2:                              # 못 가른다 — 안 덮는다
        return []
    hi = hood_index(segs, hood_u)
    hood_h = segs[hi][3] - segs[hi][1]
    cap_u = surface_sx_cap(smap)
    cap_v = max(0.5, smap.height / 2 / UNITS_PER_SCALE)
    out: list[dict] = []
    first = True
    for bu0, bv0, bu1, bv1 in segs[hi + 1:hi + 3]:  # 후드 구간 뒤의 최대 둘
        if (bu1 - bu0) < ROOF_MIN_FRAC * smap.width:
            continue
        # **지붕처럼 좁은 구간만** 덮는다 — 후드만큼 넓은 구간은 트렁크 데크다
        # (레퍼런스의 블랙은 지붕·필러이고 데크는 본색 + 모티프다)
        if (bv1 - bv0) > 0.82 * hood_h:
            continue
        # 세로(차 폭)로는 **넘치게** 덮는다 — 구간 상자는 도색 55% 이상인 행만
        # 담으므로 지붕 양 옆에 본색 띠가 남아 블랙아웃이 지붕 위에 얹힌 검은
        # 판으로 보인다. 넘긴 몫은 면 마스크가 알아서 자른다 (레퍼런스의 검은
        # 지붕은 필러까지 통째다 — HINATA·수이세이).
        # 앞선은 구간 앞끝 안쪽(뜯는 조각이 덮는다), 뒷선은 구간 뒤끝 **너머**다.
        # 넘기는 몫은 **다음 구간까지 남은 틈의 절반**을 안 넘는다 — 넘으면
        # 검정이 데크 앞머리를 물어 투톤 경계가 데크로 내려온다.
        seg = bu1 - bu0
        gap = next((a - bu1 for a, _b, _c, _d in segs if a > bu1), None)
        tail = ROOF_TAIL_OVER * seg
        if gap is not None:
            tail = min(tail, 0.5 * gap)
        pu0, pu1 = bu0 + ROOF_HEAD_IN * seg, bu1 + tail
        sy = min(1.18 * (bv1 - bv0) / 2 / UNITS_PER_SCALE, cap_v)
        for tx, tsx in band_tiles(pu0, pu1, cap_u):
            out.append({"shape": "A_01",
                        "x": round(tx, 1), "y": round((bv0 + bv1) / 2, 1),
                        "sx": round(tsx, 3), "sy": round(sy, 3),
                        "rot": 0.0, "rgb": list(ROOF_DARK)})
        if not first:
            continue
        first = False
        # 앞쪽(후드 쪽) 경계만 뜯는다 — 뒤쪽은 데크 끝이라 안 보인다. 여기서
        # 밴드가 달리는 축은 **v(차 폭)**이고 진폭은 u(차 길이)다 — 로커와 축만
        # 바뀔 뿐 자는 같다 (가로는 겹치게, 진폭은 조금만). 등방으로 재던 옛
        # 자는 지붕 폭의 0.21배를 후드 쪽으로 물어 투톤 경계가 톱니바퀴가 됐다.
        vocab = shapes or ("A_21",)
        step = (bv1 - bv0) / ROOF_TEETH
        for i in range(ROOF_TEETH):
            k = 0.55 + 0.45 * (i % 3) / 2.0
            name = vocab[i % len(vocab)]
            reach = (cat.shapes[name].reach
                     if cat is not None and name in cat.shapes else 1.0)
            out.append({"shape": name,
                        "x": round(bu0 + 0.06 * (bu1 - bu0)
                                   * ((i * 3 % 5) / 4.0), 1),
                        "y": round(bv0 + step * (i + 0.5), 1),
                        "sx": round(ROOF_TEETH_AMP * (bu1 - bu0) / 2 * k
                                    / UNITS_PER_SCALE / reach, 3),
                        "sy": round(TEETH_OVERLAP * step / 2 * k
                                    / UNITS_PER_SCALE / reach, 3),
                        "rot": round((17.0 * i) % 24.0 - 12.0, 1),
                        "rgb": list(ROOF_DARK)})
    return out


def top_segments(smap: gsurf.SurfaceMap) -> list[tuple[float, float, float, float]]:
    """윗면을 **유리 구멍으로 가른 구간 상자들** (u 오름차순 = 앞→뒤).

    첫 구간이 후드다 (top의 앞 = -u — 아틀라스 축 규약·프로브 카메라 규약
    일치). 각 상자의 v범위는 그 구간에서 행 도색 비율 55% 이상인 행들이다.
    """
    m = smap.mask
    if m.size <= 1 or not m.any():
        return []
    mh, mw = m.shape
    u0, v0, u1, v1 = smap.paint
    band = m[int(mh * 0.325):int(mh * 0.675), :]   # 중앙 밴드
    solid = band.mean(axis=0) > 0.55
    raw: list[tuple[int, int]] = []
    start = None
    for i, s in enumerate(solid):
        if s and start is None:
            start = i
        elif not s and start is not None:
            raw.append((start, i))
            start = None
    if start is not None:
        raw.append((start, mw))
    min_w = max(3, int(0.05 * mw))
    raw = [(a, b) for a, b in raw if b - a >= min_w]
    # 유리 구멍이 없는 마스크 (설치 마스크는 유리 위까지 통째로 칠해지는 차종이
    # 있다 — 인테그라 실측) → **실루엣 허리**로 가른다: 그린하우스는 후드·데크
    # 보다 좁다. 폭 프로파일이 문턱 아래로 꺼지는 구간이 지붕(+필러)이다.
    if len(raw) == 1:
        a0, b0 = raw[0]
        widths = m[:, a0:b0].sum(axis=0).astype(float)
        thr = 0.80 * float(widths.max())
        narrow = widths < thr
        # 앞(후드)에서 처음 좁아지는 곳 ~ 뒤에서 마지막 좁은 곳
        idx = np.where(narrow)[0]
        # 앞뒤 끝의 라운딩(범퍼 곡선)은 무시한다 — 가운데 40~90% 구간에서 찾는다
        idx = idx[(idx > 0.25 * (b0 - a0)) & (idx < 0.97 * (b0 - a0))]
        if len(idx) >= max(3, int(0.06 * (b0 - a0))):
            cut1, cut2 = int(idx.min()), int(idx.max())
            raw = [(a0, a0 + cut1), (a0 + cut1, a0 + cut2)]
            if b0 - (a0 + cut2) >= min_w:
                raw.append((a0 + cut2, b0))
    upp = (u1 - u0) / mw
    out: list[tuple[float, float, float, float]] = []
    for a, b in raw:
        if b - a < min_w:
            continue
        rows = np.where(m[:, a:b].mean(axis=1) > 0.55)[0]
        if len(rows) < 2:
            continue
        out.append((u0 + a * upp, v1 - rows.max() * (v1 - v0) / mh,
                    u0 + b * upp, v1 - rows.min() * (v1 - v0) / mh))
    return out
