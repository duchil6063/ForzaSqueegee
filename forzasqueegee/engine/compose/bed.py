"""인물을 따르는 **축**.

큰 색면을 짓는 일은 `macro`로 옮겼다 (어휘 아홉과 그 연속 매개변수). 여기 남은
것은 그 색면이 쓰던 두 조각인데, 둘 다 색면 말고도 쓰는 데가 있어 제자리에
남는다:

- `slab_axis` — 포즈 장축과 흐름을 섞어 수평에서 `BED_TILT_MAX` 안으로 눕힌
  축. 큰 색면(`macro.plan`)·에코 조각(`echo`)·글자 자리(`textlayout`)가 나눠
  쓴다. 포즈 축을 그대로 쓰면 세운 인물에서 그래픽이 세로로 서고 눕힌 인물
  에서는 "기울인 사진틀"이 된다 — 차체 그래픽은 차 길이 방향으로 달리되
  살짝 기운다 (레퍼런스의 판·띠 실측 8~25°).

레이어 라벨은 `itasha_bed`(색면)다 — 예산 사다리
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

