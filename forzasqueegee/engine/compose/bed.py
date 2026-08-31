"""캐릭터 베드 — 인물 뒤에 받쳐 주는 **큰 색면**.

사람이 만든 이타샤의 인물은 맨 도색 위에 홀로 서지 않는다 — 린 시부야의
남색 판, EVELYNE의 검은 바탕 위 흰 백합 판, 코토네의 자홍 판이 인물 뒤를
받친다. 실루엣 전체를 두껍게 두르는 아웃라인이 아니라 **차체 디자인의
일부처럼** 보이는 판이어야 한다: 차 길이 방향으로 달리는 띠, 흐름을 가로지르는
사선 스트라이프, 실루엣을 품는 덩어리.

## 끝은 도형이 아니라 **차가** 낸다

레퍼런스의 판·띠는 예외 없이 차 실루엣이 잘라 끝난다 — 범퍼 끝에서, 벨트라인
에서, 로커에서. 도형이 패널 한가운데서 제 끝을 보이면 붙여 놓은 스티커로
읽힌다 (사용자 판정 2026-08-31: "선은 면 끝에서 끝까지, 박스도 멈추지 말고 면
끝까지"). 그래서 여기서 내는 사각은 전부 **프레임 밖까지** 뻗는다: 얕은 띠는
앞뒤 끝을 넘겨 범퍼가 자르고, 가파른 스트라이프는 위아래를 넘겨 벨트라인·
로커가 자른다 (`_run`). 넘긴 몫은 면 마스크가 잘라 그려지지 않으므로 공짜다.
그 위에 로커 띠가 덮인다 (`design._parts` — 로커는 판 **뒤에** 그린다).

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


# 얕은 띠가 수평에서 기울 수 있는 상한 (도). 포즈 축을 그대로 따르면 세운
# 인물에서 판이 세로로 서고 눕힌 인물에서는 "기울인 사진틀"이 된다 — 차체
# 그래픽은 차 길이 방향으로 달리되 살짝 기운다 (레퍼런스의 판·띠 실측 8~25°).
BED_TILT_MAX = 22.0


# 키라인(실루엣 후광)의 두께 — 인물 높이 대비. 레퍼런스의 스티커 컷 흰 테는
# 인물 높이의 2~4%다.
KEYLINE_FRAC = 0.035


# 키라인을 원으로 덮을 때의 상한 장수 — 잔 원이 더 있어 봐야 테가 안 달라진다.
KEYLINE_MAX = 90


# 판 높이의 상한 (차체 밴드 대비) — 밴드를 다 덮으면 판이 아니라 두 번째 베이스다.
BED_H_MAX = 0.82


# 사각이 프레임 밖으로 **얼마나 더** 나가나 (프레임 폭 대비). 끝이 프레임 선에
# 딱 걸리면 반올림 한 칸의 본색 실선이 남는다 — 넉넉히 넘긴다.
OVERSHOOT = 0.04


# 기운 띠의 세로 뻗음(길이 × sin)이 밴드의 이 몫을 넘지 않게 기울기를 눕힌다.
# 900유닛 띠는 5°가 한계다 — 레퍼런스의 긴 띠도 거의 수평이고, 가파른 것은
# 짧은 스트라이프다 (그쪽은 `wedge`가 따로 낸다).
BED_RISE_MAX = 0.55


# 사선 스트라이프(`wedge`)의 기울기 하한·상한 (도). 이 아래면 띠와 갈리지 않고,
# 이 위면 밴드를 가로지르는 폭이 좁아 막대가 된다.
WEDGE_TILT = (18.0, 30.0)


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


def _tilt_for(ang: float, length: float, band: float) -> tuple[float, tuple[float, float]]:
    """길이 `length`의 띠가 가질 수 있는 기울기 (도) — 세로 뻗음을 밴드 몫으로 묶는다."""
    lim = math.degrees(math.asin(max(0.0, min(1.0, BED_RISE_MAX * band / max(1e-6, length)))))
    a = max(-lim, min(lim, ang))
    r = math.radians(a)
    return a, (math.cos(r), math.sin(r))


def _support_extent(fld: CompositionField, d: tuple[float, float]) -> tuple[float, float, float]:
    """지지 구역을 축 `d`에 투영한 (중심, 반길이, 반폭) — 판이 인물 둘레에서 덮을 범위."""
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


def _run(box: tuple[float, float, float, float], cx: float, cy: float,
         d: tuple[float, float], h: float, sign: float) -> float:
    """`(cx, cy)`에서 축 `sign·d`로 **단면 전체가 상자를 벗어나는** 거리.

    높이 `h`의 띠는 두 모서리(중심선 ± 법선·h/2)가 다 밖으로 나가야 끝이 안
    보인다. 모서리마다 x 경계와 y 경계 중 먼저 닿는 것이 그 모서리의 탈출
    거리고, 둘 중 늦은 것이 띠의 끝이다. 상자는 `OVERSHOOT`만큼 넓혀 재므로
    되돌린 거리를 그대로 쓰면 끝이 프레임 밖에 선다.
    """
    x0, y0, x1, y1 = box
    pad = OVERSHOOT * (x1 - x0)
    x0, y0, x1, y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
    dx, dy = sign * d[0], sign * d[1]
    nx, ny = -d[1] * h / 2, d[0] * h / 2
    worst = 0.0
    for ox, oy in ((nx, ny), (-nx, -ny)):
        px, py = cx + ox, cy + oy
        t = math.inf
        if abs(dx) > 1e-9:
            t = min(t, ((x1 if dx > 0 else x0) - px) / dx)
        if abs(dy) > 1e-9:
            t = min(t, ((y1 if dy > 0 else y0) - py) / dy)
        worst = max(worst, max(0.0, t))
    return worst


def _through(box: tuple[float, float, float, float], cx: float, cy: float,
             d: tuple[float, float], h: float, ang: float,
             color: tuple[int, int, int], cat: Catalog, alpha: float = 100.0) -> Layer:
    """`(cx, cy)`를 지나 축 `d`로 **프레임을 관통하는** 높이 `h`의 사각.

    앞뒤 끝이 다 프레임 밖이라 그 끝을 자르는 것은 도형이 아니라 차다.
    """
    fwd = _run(box, cx, cy, d, h, 1.0)
    back = _run(box, cx, cy, d, h, -1.0)
    w = fwd + back
    return _rect(cx + d[0] * (fwd - back) / 2, cy + d[1] * (fwd - back) / 2,
                 w, h, ang, color, cat, alpha)


def _band_y(cy: float, h: float, fld: CompositionField, rocker: bool) -> float:
    """인물 자리에서 띠의 중심 높이 — 로커 띠 위 ~ 벨트라인 안에 든다 (들 수 있으면).

    높이는 안 깎는다: 기운 띠의 양 끝이 밴드를 넘는 것은 차가 자르는 몫이다.
    인물 자리에서만 띠가 로커 속으로 꺼지거나 벨트라인 위로 뜨지 않게 한다.
    """
    fx0, fy0, fx1, fy1 = fld.frame_box
    floor = fy0 + (ROCKER_FRAC * (fy1 - fy0) if rocker else 0.0)
    if fy1 - floor <= h:
        return (floor + fy1) / 2
    return max(floor + h / 2, min(fy1 - h / 2, cy))


def bed_layers(fld: CompositionField, pal: RolePalette, cat: Catalog,
               style: str, level: float,
               edge_shapes: tuple[str, ...] | None = None,
               torn: bool = False, rocker: bool = False,
               d_rot: float = 0.0, d_y: float = 0.0) -> list[Layer]:
    """베드 레이어 (그룹 맨 아래) — 계열의 꼴 × 크기 `level`(0~1).

    `d_rot`(도)·`d_y`(인물 높이 몫)는 **미세 조정 손잡이**다 (`design._refine` —
    이긴 후보의 좌표하강). 기본값이면 없던 때와 같다.
    """
    if style == "none":
        return []
    d = slab_axis(fld)
    ang = math.degrees(math.atan2(d[1], d[0])) + d_rot
    if d_rot:
        r0 = math.radians(ang)
        d = (math.cos(r0), math.sin(r0))
    vcx, vcy = fld.visual_center
    vcy += d_y * fld.char_h
    c0, _half_len, half_wid = _support_extent(fld, d)
    ch = fld.char_h
    out: list[Layer] = []
    fs = 1.0 if fld.flow[0] >= 0 else -1.0       # 흐름 쪽 부호
    box = fld.frame_box
    band_h = box[3] - box[1]
    span = box[2] - box[0]
    if style == "slab":
        # 낮은 슬래브 — 인물 허리 높이를 지나 **차 앞뒤를 관통하는** 얇은 띠.
        # 여백을 남기는 계열 (모터스포츠 문법: 띠 + 그 위 가는 자매 선).
        ang, d = _tilt_for(ang, span, band_h)
        h = ch * (0.22 + 0.16 * level)
        cx0 = vcx + d[0] * c0
        cy0 = _band_y(vcy + d[1] * c0 - 0.08 * ch, h, fld, rocker)
        out.append(_through(box, cx0, cy0, d, h, ang, pal.bed, cat))
        # 자매 선 — 띠 위 가장자리에 붙는 액센트 선. 띠와 같이 끝에서 끝까지.
        off = 0.5 * h + 0.06 * ch
        h2 = 0.035 * ch
        out.append(_through(box, cx0 - d[1] * off, cy0 + d[0] * off, d, h2, ang,
                            pal.primary, cat))
    elif style == "plate":
        # 판은 인물을 품고 **차 앞뒤를 관통한다** — 인물 크기의 사각은 기울인
        # 사진틀로 읽힌다 (레퍼런스의 판은 인물 뒤에서 리어 쿼터를 지나 범퍼에서
        # 잘린다). 기울기는 길이가 눕힌다 (`_tilt_for`) — 살짝 기운 넓은 띠.
        ang, d = _tilt_for(ang, span, band_h)
        h = min(BED_H_MAX * band_h,
                max(2 * half_wid * (0.85 + 0.35 * level), ch * (0.55 + 0.35 * level)))
        cx0 = vcx + d[0] * c0
        cy0 = _band_y(vcy + d[1] * c0, h, fld, rocker)
        # 판 아래의 그림자판 — 조금 내려 깔아 판 아랫선을 두 겹으로 (깊이)
        sh = 0.06 * ch
        out.append(_through(box, cx0 - fs * d[0] * sh * 2, cy0 - sh, d, h, ang,
                            pal.bed_alt, cat, alpha=88.0))
        out.append(_through(box, cx0, cy0, d, h, ang, pal.bed, cat))
    elif style == "wedge":
        # 사선 둘 — 인물을 지나 **벨트라인과 로커가 자르는** 가파른 스트라이프 +
        # 그 위를 차 앞뒤로 달리는 얕은 좁은 띠. 둘의 기울기가 달라야 사선이
        # 겹치며 흐름이 난다.
        lo_t, hi_t = WEDGE_TILT
        a1 = math.copysign(max(lo_t, min(hi_t, abs(ang))), ang if ang else fs)
        r1 = math.radians(a1)
        d1 = (math.cos(r1), math.sin(r1))
        h = min(0.6 * band_h, ch * (0.45 + 0.30 * level))
        cx0 = vcx + d1[0] * c0
        cy0 = vcy + d1[1] * c0
        out.append(_through(box, cx0, cy0, d1, h, a1, pal.bed, cat))
        ang2, d2 = _tilt_for(ang, span, band_h)
        h2 = ch * (0.16 + 0.10 * level)
        off = 0.5 * h + 0.5 * h2 + 0.05 * ch
        up = 1.0 if fld.axis[1] >= 0 else -1.0
        cy2 = _band_y(vcy + d2[1] * c0 + d2[0] * off * up, h2, fld, rocker)
        out.append(_through(box, vcx + d2[0] * c0 - d2[1] * off * up, cy2, d2, h2, ang2,
                            pal.bed_alt, cat))
    elif style == "blob":
        # 덩어리 — 실루엣 후광을 품는 타원 + 흐름 쪽 작은 타원 (스플래시). 뜯긴
        # 끝이 문법이라 관통하지 않는다 (`torn`이 흐름 끝을 조각으로 뜯는다).
        w = max(fld.char_w, 2 * _half_len * 0.9) * (0.9 + 0.3 * level)
        h = ch * (0.95 + 0.25 * level)
        out.append(_ellipse(vcx + d[0] * c0 * 0.5, vcy + d[1] * c0 * 0.5, w, h, ang,
                            pal.bed, cat))
        fx, fy = fld.flow
        out.append(_ellipse(vcx + fx * 0.45 * w, vcy + fy * 0.45 * w - 0.05 * ch,
                            0.55 * w, 0.6 * h, ang, pal.bed_alt, cat, alpha=90.0))
    if torn and out and edge_shapes:
        fx, fy = fld.flow
        # 흐름 끝을 뜯는다 — 판의 흐름 쪽 끝에 큰 조각 몇을 겹쳐 곧은 끝을 지운다
        base = out[-1]
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
