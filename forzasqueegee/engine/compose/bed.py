"""캐릭터 베드 — 인물 뒤에 받쳐 주는 **큰 색면**.

사람이 만든 이타샤의 인물은 맨 도색 위에 홀로 서지 않는다 — 린 시부야의
남색 판, EVELYNE의 검은 바탕 위 흰 백합 판, 코토네의 자홍 판이 인물 뒤를
받친다. 실루엣 전체를 두껍게 두르는 아웃라인이 아니라 **차체 디자인의
일부처럼** 보이는 판이어야 한다: 포즈 축과 차의 흐름을 따르는 사선판,
흐름 쪽으로 뻗는 쐐기, 실루엣을 품는 덩어리.

꼴은 계열이 고르고(`families.Family.bed`), 크기는 `level`이며, 얼마나 커도
되는지는 점수가 조인다 (`score` — 베드가 인물을 먹으면 가독성이 깎인다).
좌표는 꾸밈 캔버스(프레임) 좌표다 — 그룹 한 장(`deco.json`)에 맨 아래로 깔린다.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from ..catalog import Catalog
from ..model import UNITS_PER_SCALE, Layer
from .bands import ROCKER_FRAC, _teeth
from .field import CompositionField
from .roles import RolePalette


LABEL = "itasha_bed"


# 베드 판이 수평에서 기울 수 있는 상한 (도). 포즈 축을 그대로 따르면 세운
# 인물에서 판이 세로로 서고 눕힌 인물에서는 "기울인 사진틀"이 된다 — 차체
# 그래픽은 차 길이 방향으로 달리되 살짝 기운다 (레퍼런스의 판·띠 실측 8~25°).
BED_TILT_MAX = 22.0


# 키라인(실루엣 후광)의 두께 — 인물 높이 대비. 레퍼런스의 스티커 컷 흰 테는
# 인물 높이의 2~4%다.
KEYLINE_FRAC = 0.035


# 키라인을 원으로 덮을 때의 상한 장수 — 잔 원이 더 있어 봐야 테가 안 달라진다.
KEYLINE_MAX = 90


# 판 뒤끝을 사선으로 재단하는 몫 (판 높이 대비 가로 길이). 곧게 끝나는 사각은
# 스티커로 읽힌다 — 레퍼런스의 판·띠는 예외 없이 사선이나 뜯긴 끝이다.
CHAMFER = 0.85


# 판 높이의 상한 (차체 밴드 대비) — 밴드를 다 덮으면 판이 아니라 두 번째 베이스다.
BED_H_MAX = 0.82


def _rect(x: float, y: float, w: float, h: float, rot: float,
          color: tuple[int, int, int], cat: Catalog, alpha: float = 100.0) -> Layer:
    return Layer(shape=cat.square, x=x, y=y, sx=w / 2 / UNITS_PER_SCALE,
                 sy=h / 2 / UNITS_PER_SCALE, rot=rot % 360.0, color=color,
                 alpha=alpha, label=LABEL)


def _ellipse(x: float, y: float, w: float, h: float, rot: float,
             color: tuple[int, int, int], cat: Catalog, alpha: float = 100.0) -> Layer:
    sh = cat.shapes.get(cat.circle)
    reach = sh.reach if sh is not None else 1.0
    return Layer(shape=cat.circle, x=x, y=y, sx=w / 2 / UNITS_PER_SCALE / reach,
                 sy=h / 2 / UNITS_PER_SCALE / reach, rot=rot % 360.0, color=color,
                 alpha=alpha, label=LABEL)


def slab_axis(fld: CompositionField) -> tuple[float, float]:
    """베드가 따르는 축 — 포즈 장축과 흐름의 섞임, 수평에서 `BED_TILT_MAX` 안.

    흐름 쪽이 +다. 포즈 축이 세로(세운 인물)면 섞임이 가팔라지는데 그것을 그대로
    쓰면 판이 세로로 선다 — 상한으로 눕힌다.
    """
    ax, ay = fld.axis
    fx, fy = fld.flow
    if ax * fx + ay * fy < 0:                    # 축은 부호가 없다 — 흐름과 같은 쪽으로
        ax, ay = -ax, -ay
    sx, sy = 0.45 * ax + 0.55 * fx, 0.45 * ay + 0.55 * fy
    ang = math.degrees(math.atan2(sy, sx))
    if sx < 0:
        ang = math.degrees(math.atan2(-sy, -sx))
    ang = max(-BED_TILT_MAX, min(BED_TILT_MAX, ang))
    r = math.radians(ang)
    return math.cos(r), math.sin(r)


# 기운 판의 세로 뻗음(길이 × sin)이 밴드의 이 몫을 넘지 않게 기울기를 눕힌다.
# 900유닛 판은 5°가 한계이고 인물만 한 판은 22°까지 간다 — 레퍼런스도 그렇다
# (긴 띠는 거의 수평, 가파른 것은 짧은 쐐기).
BED_RISE_MAX = 0.55


def _tilt_for(ang: float, length: float, band: float) -> tuple[float, tuple[float, float]]:
    """길이 `length`의 판이 가질 수 있는 기울기 (도) — 세로 뻗음을 밴드 몫으로 묶는다."""
    lim = math.degrees(math.asin(max(0.0, min(1.0, BED_RISE_MAX * band / max(1e-6, length)))))
    a = max(-lim, min(lim, ang))
    r = math.radians(a)
    return a, (math.cos(r), math.sin(r))


def _flow_reach(fld: CompositionField, d: tuple[float, float], cx: float, cy: float) -> float:
    """`(cx, cy)`에서 축 `d`를 따라 프레임 끝까지의 거리 (흐름 쪽)."""
    fx0, _y0, fx1, _y1 = fld.frame_box
    edge = fx1 if fld.flow[0] >= 0 else fx0
    if abs(d[0]) < 1e-6:
        return 0.5 * (fx1 - fx0)
    return abs((edge - cx) / d[0])


def _support_extent(fld: CompositionField, d: tuple[float, float]) -> tuple[float, float, float]:
    """지지 구역을 축 `d`에 투영한 (중심, 반길이, 반폭) — 판이 덮을 범위."""
    ys, xs = np.where(fld.support > 0.5)
    if len(xs) == 0:
        cx, cy = fld.visual_center
        return 0.0, 0.5 * fld.char_w, 0.4 * fld.char_h
    g = fld.grid
    px = g.x0 + (xs + 0.5) * g.cell
    py = g.y_top - (ys + 0.5) * g.cell
    vcx, vcy = fld.visual_center
    along = (px - vcx) * d[0] + (py - vcy) * d[1]
    across = -(px - vcx) * d[1] + (py - vcy) * d[0]
    lo, hi = float(np.percentile(along, 4)), float(np.percentile(along, 96))
    wlo, whi = float(np.percentile(across, 6)), float(np.percentile(across, 94))
    return (lo + hi) / 2, (hi - lo) / 2, max(0.25 * fld.char_h, (whi - wlo) / 2)


def _chamfer(rect: Layer, fs: float, d: tuple[float, float],
             frac: float = CHAMFER) -> Layer:
    """판의 **뒤끝(흐름 반대쪽)을 사선으로 자르는** 뺄셈 마스크 삼각형.

    게임의 뺄셈 마스크는 먼저 그려진 레이어를 도형만큼 잘라 도색을 드러낸다
    (`Layer.mask`). 직각삼각형(A_04 — 직각이 (−1,−1))의 직각을 판의 뒤끝
    아래 모서리에 두고 빗변이 판을 가로지르게 놓는다.
    """
    w, h = rect.sx * 2 * UNITS_PER_SCALE, rect.sy * 2 * UNITS_PER_SCALE
    xd = (-fs * d[0], -fs * d[1])                # 뒤쪽 = 흐름 반대
    n = (-xd[1], xd[0])                          # xd의 반시계 수직
    # 판의 뒤끝 아래 모서리 — n이 아래를 보면 위 모서리가 되므로 부호로 판다
    down = n if n[1] < 0 else (-n[0], -n[1])
    bx = rect.x + xd[0] * w / 2 + down[0] * h / 2
    by = rect.y + xd[1] * w / 2 + down[1] * h / 2
    c = frac * h
    # 삼각형의 x축은 **판 안쪽**(흐름 쪽), y축은 위쪽 — 직각이 (bx, by)에 온다
    ax_ = (-xd[0], -xd[1])
    up = (-down[0], -down[1])
    cx = bx + ax_[0] * c / 2 + up[0] * h / 2
    cy = by + ax_[1] * c / 2 + up[1] * h / 2
    rot = math.degrees(math.atan2(ax_[1], ax_[0]))
    # 회전 뒤 로컬 +y가 `up`이 아니면 (거울 꼴) sy를 뒤집는다
    ly = (-math.sin(math.radians(rot)), math.cos(math.radians(rot)))
    flip = -1.0 if (ly[0] * up[0] + ly[1] * up[1]) < 0 else 1.0
    return Layer(shape="A_04", x=cx, y=cy, sx=c / 2 / UNITS_PER_SCALE,
                 sy=flip * h / 2 / UNITS_PER_SCALE, rot=rot % 360.0,
                 color=(0, 0, 0), mask=True, label=LABEL)


def _lift_above_rocker(layers: list[Layer], fld: CompositionField,
                       rocker: bool) -> list[Layer]:
    """판이 로커 띠 속으로 꺼지지 않게 **위로 민다** — 밴드 안에서.

    기울인 판의 낮은 끝이 로커에 잠기면 판과 로커가 한 덩이로 붙어 "검은 하부가
    앞으로 치솟는" 꼴이 된다. 판의 최저점을 로커 윗선 위로 올린다 (밴드 위끝을
    넘지 않는 만큼만).
    """
    fx0, fy0, fx1, fy1 = fld.frame_box
    floor = fy0 + (ROCKER_FRAC * (fy1 - fy0) if rocker else 0.0)
    out = []
    for l in layers:
        if l.mask:
            out.append(l)
            continue
        w, h = abs(l.sx) * 2 * UNITS_PER_SCALE, abs(l.sy) * 2 * UNITS_PER_SCALE
        r = math.radians(l.rot)
        sn, cs = abs(math.sin(r)), abs(math.cos(r))
        # 밴드(로커 위 ~ 벨트라인)에 안 들면 **높이를 줄인다** — 기운 판은 세로
        # 뻗음이 w·|sin| + h·|cos| 이라 그것이 밴드 높이를 넘지 않게
        room_all = fy1 - floor
        if w * sn + h * cs > room_all and cs > 1e-6:
            h = max(0.10 * (fy1 - fy0), (room_all - w * sn) / cs)
            l = Layer(**{**l.__dict__, "sy": math.copysign(h / 2 / UNITS_PER_SCALE, l.sy)})
        drop = 0.5 * (w * sn + h * cs)
        low, high = l.y - drop, l.y + drop
        dy = 0.0
        if low < floor:
            dy = floor - low
        elif high > fy1:
            dy = fy1 - high
        if dy:
            l = Layer(**{**l.__dict__, "y": l.y + dy})
        out.append(l)
    return out


def bed_layers(fld: CompositionField, pal: RolePalette, cat: Catalog,
               style: str, level: float,
               edge_shapes: tuple[str, ...] | None = None,
               torn: bool = False, rocker: bool = False) -> list[Layer]:
    """베드 레이어 (그룹 맨 아래) — 계열의 꼴 × 크기 `level`(0~1)."""
    if style == "none":
        return []
    d = slab_axis(fld)
    ang = math.degrees(math.atan2(d[1], d[0]))
    vcx, vcy = fld.visual_center
    c0, half_len, half_wid = _support_extent(fld, d)
    ch = fld.char_h
    out: list[Layer] = []
    k = 0.6 + 0.6 * level                         # 0.6 ~ 1.2
    fs = 1.0 if fld.flow[0] >= 0 else -1.0       # 흐름 쪽 부호 (판은 그쪽으로 뻗는다)
    band_h = fld.frame_box[3] - fld.frame_box[1]
    if style == "slab":
        # 낮은 슬래브 — 인물 허리 높이를 지나는 얇은 사선 띠. 여백을 남기는 계열.
        # 흐름 쪽으로는 프레임 끝까지 달린다 (이음새 너머로 이어지는 띠).
        cx0 = vcx + d[0] * c0
        cy0 = vcy + d[1] * c0 - 0.08 * ch
        back = half_len * (0.9 + 0.5 * level)
        fwd = _flow_reach(fld, d, cx0, cy0) * 0.98
        w = back + fwd
        ang, d = _tilt_for(ang, w, band_h)
        cx = cx0 + fs * d[0] * (fwd - back) / 2
        cy = cy0 + fs * d[1] * (fwd - back) / 2
        h = ch * (0.22 + 0.16 * level)
        main = _rect(cx, cy, w, h, ang, pal.bed, cat)
        out.append(main)
        out.append(_chamfer(main, fs, d))
        # 얇은 자매 띠 — 판 위 가장자리에 붙는 액센트 선 (모터스포츠 문법)
        off = 0.5 * h + 0.06 * ch
        out.append(_rect(cx - d[1] * off, cy + d[0] * off, w * 0.92, 0.035 * ch, ang,
                         pal.primary, cat))
    elif style == "plate":
        # 판은 인물을 품고 **흐름 쪽으로 길게** 뻗는다 — 인물 크기의 사각은
        # 기울인 사진틀로 읽힌다 (레퍼런스의 판은 인물 뒤에서 리어 쿼터까지 간다)
        cx0 = vcx + d[0] * c0
        cy0 = vcy + d[1] * c0
        back = half_len * (0.85 + 0.35 * level)
        # 판은 **인물 크기로** 뻗는다 — 패널 끝까지 늘이면 기운 판의 세로 뻗음
        # (길이 x sin)이 밴드를 다 먹어 `_lift_above_rocker`가 높이를 깎는다.
        # 788x45로 눌린 판은 인물을 못 받치고 지나가는 띠가 된다 (실측 B0/A2:
        # 인물 뒤 받침 .50 → .68 · 판 45 → 84유닛 · 판-인물 거리 1.19 → 0.28
        # 인물폭). 이음새 너머로 잇는 몫은 쐐기의 좁은 띠·로커·슬래브가 진다.
        fwd = half_len * k
        w = back + fwd
        ang, d = _tilt_for(ang, w, band_h)
        cx = cx0 + fs * d[0] * (fwd - back) / 2
        cy = cy0 + fs * d[1] * (fwd - back) / 2
        h = min(BED_H_MAX * band_h,
                max(2 * half_wid * (0.85 + 0.35 * level), ch * (0.55 + 0.35 * level)))
        main = _rect(cx, cy, w, h, ang, pal.bed, cat)
        # 판 뒤의 그림자판 — 흐름 반대쪽 뒤끝을 조금 더 내밀어 깊이를 낸다
        sh = 0.06 * ch
        out.append(_rect(cx - fs * d[0] * sh * 2, cy - sh, w, h, ang,
                         pal.bed_alt, cat, alpha=88.0))
        out.append(main)
        out.append(_chamfer(main, fs, d))
    elif style == "wedge":
        # 사선판 둘 — 인물을 지나는 넓은 판 + 그 위를 흐름 쪽으로 더 길게 달리는 좁은 판
        cx0 = vcx + d[0] * c0
        cy0 = vcy + d[1] * c0
        # 넓은 판은 짧고 가파르게 (쐐기), 좁은 띠는 길고 얕게 (흐름) — 둘의
        # 기울기가 달라야 사선이 겹치며 흐름이 난다
        back = half_len * (0.8 + 0.3 * level)
        fwd = half_len * k
        w = back + fwd
        ang1, d1 = _tilt_for(ang, w, band_h)
        cx = cx0 + fs * d1[0] * (fwd - back) / 2
        cy = cy0 + fs * d1[1] * (fwd - back) / 2
        h = min(0.6 * band_h, ch * (0.45 + 0.30 * level))
        main = _rect(cx, cy, w, h, ang1, pal.bed, cat)
        out.append(main)
        out.append(_chamfer(main, fs, d1, frac=1.2))
        w2 = back * 0.6 + _flow_reach(fld, d, cx0, cy0) * 0.98
        ang2, d2 = _tilt_for(ang, w2, band_h)
        h2 = ch * (0.16 + 0.10 * level)
        off = 0.5 * h + 0.5 * h2 + 0.05 * ch
        up = 1.0 if fld.axis[1] >= 0 else -1.0
        cx2 = cx0 + fs * d2[0] * (w2 / 2 - back * 0.6) - d2[1] * off * up
        cy2 = cy0 + fs * d2[1] * (w2 / 2 - back * 0.6) + d2[0] * off * up
        out.append(_rect(cx2, cy2, w2, h2, ang2, pal.bed_alt, cat))
    elif style == "blob":
        # 덩어리 — 실루엣 후광을 품는 타원 + 흐름 쪽 작은 타원 (스플래시)
        w = max(fld.char_w, 2 * half_len * 0.9) * (0.9 + 0.3 * level)
        h = ch * (0.95 + 0.25 * level)
        out.append(_ellipse(vcx + d[0] * c0 * 0.5, vcy + d[1] * c0 * 0.5, w, h, ang,
                            pal.bed, cat))
        fx, fy = fld.flow
        out.append(_ellipse(vcx + fx * 0.45 * w, vcy + fy * 0.45 * w - 0.05 * ch,
                            0.55 * w, 0.6 * h, ang, pal.bed_alt, cat, alpha=90.0))
    out = _lift_above_rocker(out, fld, rocker)
    if torn and out and edge_shapes:
        fx, fy = fld.flow
        # 흐름 끝을 뜯는다 — 판의 흐름 쪽 끝에 큰 조각 몇을 겹쳐 곧은 끝을 지운다
        base = out[-1] if style != "plate" else out[1]
        w = base.sx * 2 * UNITS_PER_SCALE
        h = base.sy * 2 * UNITS_PER_SCALE
        ex = base.x + fx * 0.5 * w * abs(d[0]) * 0.9
        ey = base.y + fy * 0.5 * w * abs(d[0]) * 0.9
        out += _teeth(edge_shapes, cat, span=h * 1.1, x0=ex - 0.55 * h, top=ey,
                      band=h * 0.5, n=3, color=base.color, label=LABEL)
    return out


def keyline_layers(fld: CompositionField, color: tuple[int, int, int], cat: Catalog,
                   width: float | None = None) -> list[Layer]:
    """실루엣 **키라인** — 인물을 도려낸 스티커의 흰 테 (아웃라인성 보조 배경).

    임의 폴리곤은 못 쓰므로 (게임 도형은 카탈로그뿐) 실루엣을 `width`만큼 넓힌
    마스크를 **내접 원으로 탐욕스럽게 덮는다** — 거리 변환의 최대점에 그 반지름
    의 원을 놓고 지우기를 되풀이한다. 원의 대부분은 인물 밑에 숨고 테만 남는다.
    인물이 짙은 판 위에서 읽히게 하는 가장 값싼 길이다 (수십 장).
    """
    g = fld.grid
    ch = fld.char_h
    w = width if width is not None else KEYLINE_FRAC * ch
    k = g.px(w)
    sil = (fld.char > 0.5).astype(np.uint8)
    if not sil.any():
        return []
    grown = cv2.dilate(sil, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
    grown = grown & (fld.drawable > 0.5).astype(np.uint8)
    rest = grown.copy()
    out: list[Layer] = []
    reach = (cat.shapes[cat.circle].reach if cat.circle in cat.shapes else 1.0)
    while len(out) < KEYLINE_MAX:
        dist = cv2.distanceTransform(rest, cv2.DIST_L2, 3)
        r = float(dist.max())
        if r < 1.0:
            break
        yy, xx = np.unravel_index(int(np.argmax(dist)), dist.shape)
        x = g.x0 + (xx + 0.5) * g.cell
        y = g.y_top - (yy + 0.5) * g.cell
        rad = (r + 0.6) * g.cell
        out.append(Layer(shape=cat.circle, x=x, y=y,
                         sx=rad / UNITS_PER_SCALE / reach, sy=rad / UNITS_PER_SCALE / reach,
                         color=color, label="itasha_keyline"))
        cv2.circle(rest, (int(xx), int(yy)), max(1, int(round(r * 0.92))), 0, -1)
    return out
