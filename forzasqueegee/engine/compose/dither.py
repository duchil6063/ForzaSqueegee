"""디더 페이드 — 색면이 도색으로 **흩어지며 사라지는** 가장자리.

곧게 끝나는 색면은 붙여 놓은 스티커로 읽힌다. 사람이 만든 이타샤는 그 경계를
반드시 무언가로 흐린다: ARIS의 픽셀 디더, KOTONE의 물감 튐, EVELYNE의 꽃.
게임 어휘로 가장 싸게 흉내 나는 것이 **디더**다 — 사각 조각을 격자에 놓되
멀어질수록 듬성듬성하게 (하프톤). 조각 하나가 한 장이라 20~40장이면 된다.

그라데이션 도형(`catalog/gradient_catalog.json`의 G_BAR·G_DOT)은 **못 쓴다**:
그 넷은 그라데이션 탭 도형이라 주입 id 표(`fh6_layout.json`의 `shape_ids`
520종)에 없다 — 꾸밈 그룹은 주입으로 서므로 표 밖 도형은 템플릿 도형으로
그려진다. 그래서 알파가 아니라 **성김**으로 페이드를 만든다.

## 문법

- 격자는 판의 축을 따른다 (판이 기울면 디더도 같이 기운다).
- 열이 멀어질수록 채움 확률이 준다 (`1 - t`), 조각 크기도 준다.
- 어느 칸을 채우나는 **베이어 4x4 문턱 행렬**이 정한다 — 난수가 아니고,
  같은 입력이면 같은 그림이다. 규칙적인 격자에 규칙적인 구멍이라 무늬로 읽힌다.
"""

from __future__ import annotations

import math

from ..catalog import Catalog
from ..model import UNITS_PER_SCALE, Layer
from .field import CompositionField


LABEL = "itasha_fade"


# 베이어 4x4 (0~15) — 값이 작은 칸부터 찬다.
_BAYER = ((0, 8, 2, 10), (12, 4, 14, 6), (3, 11, 1, 9), (15, 7, 13, 5))


# 페이드 띠의 길이 (판 길이 대비). 이보다 길면 디더가 판보다 눈에 띈다.
FADE_SPAN = 0.55


# 세로 칸 수 — 칸은 이것으로 **정사각**이 된다 (판 높이 ÷ 행). 3행이면 조각이
# 커서 체커 깃발로 읽히고, 6행을 넘으면 멀리서 안 보인다.
FADE_ROWS = 5


# 조각 크기 (칸 대비) — 첫 열은 칸을 거의 채우고 끝 열은 절반이다.
FADE_FILL = (0.92, 0.46)


def fade_layers(fld: CompositionField, plate: Layer, cat: Catalog,
                flow_sign: float, *, rows: int = FADE_ROWS,
                span: float = FADE_SPAN, color: tuple[int, int, int] | None = None,
                budget: int = 40) -> list[Layer]:
    """`plate`의 **흐름 쪽 끝**에서 도색으로 흩어지는 디더 조각들.

    `plate`는 색면 사각(`bed._rect`가 낸 것)이고, 나오는 조각은 그 색·그 각을
    따른다. `budget`은 장수 상한 — 넘치면 먼 열부터 버린다.
    """
    if budget < 6 or rows < 2:
        return []
    w = abs(plate.sx) * 2 * UNITS_PER_SCALE
    h = abs(plate.sy) * 2 * UNITS_PER_SCALE
    r = math.radians(plate.rot)
    ax = (math.cos(r), math.sin(r))               # 판의 긴 축
    ay = (-ax[1], ax[0])
    fs = 1.0 if flow_sign >= 0 else -1.0
    # 칸은 정사각 — 판 높이를 행으로 나눈 것이 한 변이고, 열 수는 띠 길이가 낸다
    cell_y = h / rows
    cell_x = cell_y
    cols = max(2, int(round(span * w / cell_x)))
    # 띠는 판 끝에서 시작한다 (판 안으로 한 칸 물려 이가 맞물리게)
    x0 = plate.x + fs * ax[0] * (w / 2 - 0.5 * cell_x)
    y0 = plate.y + fs * ax[1] * (w / 2 - 0.5 * cell_x)
    col = color if color is not None else plate.color
    out: list[Layer] = []
    for j in range(cols):
        t = (j + 0.5) / cols
        keep = 1.0 - t                            # 이 열의 채움 확률
        size = cell_x * (FADE_FILL[0] + (FADE_FILL[1] - FADE_FILL[0]) * t)
        n_col = 0
        for i in range(rows):
            if _BAYER[i % 4][j % 4] / 16.0 >= keep:
                continue
            off = (i - (rows - 1) / 2) * cell_y
            cx = x0 + fs * ax[0] * (j + 0.5) * cell_x + ay[0] * off
            cy = y0 + fs * ax[1] * (j + 0.5) * cell_x + ay[1] * off
            if fld.grid.at(fld.drawable, cx, cy) < 0.5:
                continue                          # 안 그려지는 자리에 장을 안 쓴다
            out.append(Layer(shape=cat.square, x=cx, y=cy,
                             sx=size / 2 / UNITS_PER_SCALE,
                             sy=size / 2 / UNITS_PER_SCALE,
                             rot=plate.rot, color=col, label=LABEL))
            n_col += 1
            if len(out) >= budget:
                return out
        # **끊기면 거기서 끝**이다 — 판 끝이 휠아치에 걸리면 아치 너머 도색면에
        # 조각이 되살아나 판과 뚝 떨어진 체커 판이 뜬다 (giulia-07 실측: 앞펜더에
        # 흰 격자 한 덩이). 페이드는 판에서 이어져야 페이드다.
        if n_col == 0:
            break
    return out
