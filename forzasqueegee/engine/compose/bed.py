"""인물을 따르는 **축**과 실루엣 **키라인**.

큰 색면을 짓는 일은 `macro`로 옮겼다 (어휘 아홉과 그 연속 매개변수). 여기 남은
것은 그 색면이 쓰던 두 조각인데, 둘 다 색면 말고도 쓰는 데가 있어 제자리에
남는다:

- `slab_axis` — 포즈 장축과 흐름을 섞어 수평에서 `BED_TILT_MAX` 안으로 눕힌
  축. 큰 색면(`macro.plan`)·에코 조각(`echo`)·글자 자리(`textlayout`)가 나눠
  쓴다. 포즈 축을 그대로 쓰면 세운 인물에서 그래픽이 세로로 서고 눕힌 인물
  에서는 "기울인 사진틀"이 된다 — 차체 그래픽은 차 길이 방향으로 달리되
  살짝 기운다 (레퍼런스의 판·띠 실측 8~25°).
- `keyline_layers` — 인물을 도려낸 스티커의 흰 테. 짙은 색면 위에서 실루엣이
  읽히게 하는 가장 값싼 길이고, 후보의 한 축이다 (`design` — 점수가 켤지 끌지
  가른다).

레이어 라벨은 `itasha_bed`(색면)와 `itasha_keyline`이다 — 예산 사다리
(`design.TRIM_ORDER`)·바닥 요소 판정(`score.GROUND`)·구성 그래프(`graph.derive`)
가 그 이름을 읽는다.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from ..catalog import Catalog
from ..model import UNITS_PER_SCALE, Layer
from .field import CompositionField


# 축이 수평에서 기울 수 있는 상한 (도).
BED_TILT_MAX = 22.0


# 키라인(실루엣 후광)의 두께 — 인물 높이 대비. 레퍼런스의 스티커 컷 흰 테는
# 인물 높이의 2~4%다.
KEYLINE_FRAC = 0.035


# 키라인을 원으로 덮을 때의 상한 장수 — 잔 원이 더 있어 봐야 테가 안 달라진다.
KEYLINE_MAX = 90


def slab_axis(fld: CompositionField) -> tuple[float, float]:
    """그래픽이 따르는 축 — 포즈 장축과 흐름의 섞임, 수평에서 `BED_TILT_MAX` 안.

    흐름 쪽이 +다. 포즈 축이 세로(세운 인물)면 섞임이 가팔라지는데 그것을 그대로
    쓰면 그래픽이 세로로 선다 — 상한으로 눕힌다.
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


def keyline_layers(fld: CompositionField, color: tuple[int, int, int], cat: Catalog,
                   width: float | None = None) -> list[Layer]:
    """실루엣 **키라인** — 인물을 도려낸 스티커의 흰 테 (아웃라인성 보조 배경).

    임의 폴리곤은 못 쓰므로 (게임 도형은 카탈로그뿐) 실루엣을 `width`만큼 넓힌
    마스크를 **내접 원으로 탐욕스럽게 덮는다** — 거리 변환의 최대점에 그 반지름
    의 원을 놓고 지우기를 되풀이한다. 원의 대부분은 인물 밑에 숨고 테만 남는다.
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
