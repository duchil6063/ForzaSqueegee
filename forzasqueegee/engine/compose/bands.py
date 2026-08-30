"""로커 밴드 — 차체 하부를 채우는 투톤 면과 그 찢긴 윗선."""

from __future__ import annotations

from ..catalog import Catalog
from ..model import UNITS_PER_SCALE, Layer, rgb_to_hsb
from .look import Look


# 하부 투톤을 깔지 않는 베이스 명도 하한 — 어두운 차에는 검정 로커가 안 보인다
# (EVELYNE의 검은 차도 하부 투톤이 없다).
ROCKER_BASE_MIN = 0.30


# 로커 밴드가 차체 밴드의 몇 몫을 먹나. 레퍼런스의 하부 투톤은 차체 밴드의
# 2~4할이다 (Evo IX의 검정 하부 · EVELYNE의 검정 하부 · ARIS의 흰 하부).
ROCKER_FRAC = 0.26


# 찢긴 윗선을 만드는 조각 수 — 면 길이를 이만큼으로 나눠 얹는다.
ROCKER_TEETH = 18


# ---- 찢긴 윗선의 꼴 (2026-08-22 미리보기 판정) ----
# 톱니는 **납작해야** 뜯긴 가장자리로 읽힌다. 등방으로 놓으면 밴드보다 큰
# 조각이 서서 가장자리가 아니라 **가시**가 된다 — 옛 자(`1.5 × 밴드 높이`)로는
# 톱니 하나가 밴드의 1.5배였고, 앞·뒤 범퍼에서는 면 높이의 0.39배짜리 검은
# 가시가 범퍼 위로 솟았다 (frag0-03 미리보기).
#
# 두 축을 따로 잰다: 가로는 **이웃과 겹칠 만큼** (곧은 선이 지워지는 조건),
# 세로는 **밴드의 몇 할만** (레퍼런스의 찢긴 경계 진폭 — Evo IX의 빨강↔검정,
# Cygames 86의 흰 물감 가장자리가 밴드의 2~3할이다).
TEETH_OVERLAP = 1.45       # 가로 지름 ÷ 톱니 간격 — 1보다 커야 밑동이 이어진다


TEETH_AMP = 0.62           # 세로 지름 ÷ 밴드 높이 (반지름이 그 절반만큼 솟는다)


def stripe_layers(lk: Look, color: tuple[int, int, int], cat: Catalog,
                  shapes: tuple[str, ...] | None = None,
                  length: float | None = None,
                  frac: float = ROCKER_FRAC,
                  car: tuple[int, int, int] | None = None) -> list[Layer]:
    """**로커 밴드** — 차체 하부를 채우는 투톤 면 + 찢긴 윗선.

    ## 왜 납작한 막대가 아닌가

    레퍼런스의 사이드실 언저리에 실제로 있는 것은 **띠가 아니다**: 스폰서 로고
    행(어휘 밖이라 안 넣는다) 아니면 **하부 투톤 면**이다 — Evo IX·EVELYNE·
    수이세이가 검정 하부, ARIS·Cygames 86이 흰 하부다. 꽉 찬 사각 두 줄은 그
    어느 쪽도 아니고, 렌더에서 **빨간 파이프 두 개**로 읽혔다 (2026-08-21
    미리보기 판정).

    그래서 사각 하나로 하부를 채우고 **윗선을 모티프로 뜯는다**. 곧은 선을
    그대로 두면 도색 견본이 되고, 뜯으면 KOTONE의 스플래시 가장자리·Cygames
    86의 흰 물감 가장자리가 된다. 색은 **테두리와 같은 무채**다 — 하부까지
    액센트로 칠하면 차 전체가 액센트 판이 된다.

    `length`는 캔버스 유닛 — `design`이 면 실측에서 역산한 차 길이를 준다.
    """
    sq = cat.square
    vocab = shapes or (cat.circle,)
    cx, _cy = lk.center
    span = length if length is not None else lk.w
    if car is not None and rgb_to_hsb(*car)[2] < ROCKER_BASE_MIN:
        return []                    # 이미 어두운 차 — 검정 로커가 안 보인다
    band = frac * lk.h
    y0 = lk.box[1]                               # 사이드실
    # 아래로 넉넉히 뺀다 — 면 마스크가 로커 아래를 알아서 자른다 (모서리에서
    # 끊기는 것이 자연스럽다). 위는 밴드 높이까지만.
    lo = y0 - 0.5 * lk.h
    out = [Layer(shape=sq, x=cx, y=(lo + y0 + band) / 2,
                 sx=span / UNITS_PER_SCALE / 2,
                 sy=(y0 + band - lo) / UNITS_PER_SCALE / 2,
                 color=color, label="itasha_stripe")]
    # 톱니는 **불규칙**해야 뜯긴 것으로 읽힌다 — 같은 크기를 같은 높이에 고르게
    # 얹으면 울타리 말뚝이 된다 (2026-08-21 미리보기 판정). 크기·높이·가로
    # 자리를 셋 다 서로소 주기로 흔들고, 이웃과 넉넉히 겹쳐 밑동이 이어지게 한다.
    # 두 축의 자가 다르다 (`TEETH_OVERLAP`·`TEETH_AMP`) — 가로는 간격, 세로는
    # 밴드 높이다. 등방으로 재면 가시가 된다.
    out += _teeth(vocab, cat, span=span, x0=cx - span / 2, top=y0 + band,
                  band=band, n=ROCKER_TEETH, color=color,
                  label="itasha_stripe")
    return out


def _teeth(vocab: tuple[str, ...], cat: Catalog, *, span: float, x0: float,
           top: float, band: float, n: int,
           color: tuple[int, int, int],
           label: str = "") -> list[Layer]:
    """투톤 밴드의 **찢긴 윗선** — 납작한 조각을 겹쳐 얹어 곧은 선을 지운다.

    `top`이 밴드 윗선, `band`가 밴드 높이다 (둘 다 캔버스/면 유닛). 조각은
    윗선 근처에 걸터앉아 절반쯤 위로 솟는다. 크기는 도형의 뻗음으로 나눈다
    (`CatShape.reach`) — 물감 계열은 ±1.65라 안 나누면 진폭이 그만큼 커진다.
    """
    step = span / n
    rx = TEETH_OVERLAP * step / 2               # 이웃과 겹치는 가로 반지름
    ry = TEETH_AMP * band / 2                   # 밴드의 몇 할만 솟는 세로 반지름
    out: list[Layer] = []
    for i in range(n):
        jx = 0.30 * ((i * 7 % n) / max(1, n - 1) - 0.5)
        x = x0 + step * (i + 0.5 + jx)
        yy = top - 0.38 * band * ((i * 3 % n) / max(1, n - 1))
        k = 0.55 + 0.45 * ((i * 5 % n) / max(1, n - 1))
        name = vocab[i % len(vocab)]
        reach = (cat.shapes[name].reach if name in cat.shapes else 1.0)
        out.append(Layer(shape=name, x=x, y=yy,
                         sx=rx * k / UNITS_PER_SCALE / reach,
                         sy=ry * k / UNITS_PER_SCALE / reach,
                         # 납작한 조각은 **조금만** 돌린다 — 크게 돌리면 긴 축이
                         # 세로로 서서 다시 가시가 된다
                         rot=(17.0 * i) % 24.0 - 12.0,
                         color=color, label=label))
    return out
