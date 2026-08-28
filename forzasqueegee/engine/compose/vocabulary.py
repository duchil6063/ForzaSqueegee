"""모티프 어휘 — 어느 계열의 어느 도형을 쓰나."""

from __future__ import annotations

import math

from ..catalog import Catalog
from ..model import UNITS_PER_SCALE, rgb_to_hsb
from .look import Look
from .palette import MOTIF_THEME, theme_color


# 모티프가 실제로 덮는 몫 (외접 사각 대비). 별·꽃·스플래터는 사각 모서리를
# 안 채우므로 자리 검사는 이 내접 원으로 한다.
MOTIF_INSCRIBE = 0.78


_RING8 = tuple(i * math.pi / 4 for i in range(8))


def shape_half(cat: Catalog, name: str, size: float) -> float:
    """지름 `size` 유닛으로 서게 하는 **스케일 값** — 도형의 실제 뻗음을 나눈다.

    카탈로그 도형은 ±1이 아니다 (`CatShape.reach`): 꽃·물감·결정은 ±1.65,
    스월은 세로가 ±0.17이다. 나누지 않으면 같은 `size`가 계열마다 다른 크기로
    서고, 꽃 한 송이가 인물만 해진다.
    """
    sh = cat.shapes.get(name)
    return size / (2.0 * UNITS_PER_SCALE * (sh.reach if sh is not None else 1.0))


# ---- 모티프 어휘 (게임 도형 카탈로그 1,480종에서 발굴) ----
# 원과 마름모만 흩으면 **색종이로 읽힌다** (12호차 캡처 판정). 레퍼런스의 배경
# 모티프는 별·꽃·스플래터·스월이고, 게임에는 그 기성 도형이 다 있다. 여기 적은
# 것은 전부 `catalog/fh6_layout.json`의 주입 id 표(520종)에 있는 것만 골랐다 —
# 꾸밈 그룹은 주입으로 서기 때문이다.
#
# **한 벌은 두 종이다.** 레퍼런스의 모티프는 예외 없이 **한 가지를 되풀이**한다:
# EVELYNE의 백합 열두 송이가 전부 같은 꽃이고, 수이세이·Fate의 별무리는 찬 별과
# 빈 별 **두 종**뿐이며, 마린은 벚꽃 하나다. 종을 늘리면 무리가 아니라 클립아트
# 모음이 된다 (2026-08-22 판정: 별 8종 + 꽃 6종을 돌려 쓰니 한 면에 여덟 가지
# 도형이 서서 "도형을 남발해 도배한" 꼴이 됐다). 되풀이가 문양을 만들고, 변화는
# 종이 아니라 **크기·각도**가 낸다.
# 사람이 고를 수 있는 계열 이름 (CLI `--motif` · 편집기 [Motif Family]).
# `crystal`은 색조 표(`MOTIF_THEME`)가 안 뽑는다 — **사람만** 고르는 계열이다.
MOTIF_FAMILIES = ("star", "flower", "splat", "swirl", "crystal")


MOTIF_SETS: dict[str, tuple[str, str]] = {
    # 찬 별 + 빈 별 (수이세이·Fate의 별무리가 정확히 이 둘이다)
    "star": ("A_08", "A_18"),
    # 꽃잎이 갈라진 꽃 두 종 — 원형 꽃술(U_48)보다 꽃으로 읽힌다
    "flower": ("I_12", "I_15"),
    # 물감 튐 — 흘러내림·비말이 붙은 진짜 스플래터 (Cygames 86의 흰 튐)
    "splat": ("G_01", "G_05"),
    # 스월·붓결 — 흐름을 만드는 곡선 (ARIS의 리본)
    "swirl": ("U_77", "U_79"),
    # 결정·눈 — 차가운 단색 테마
    "crystal": ("I_31", "I_34"),
}


# **투톤 경계를 뜯는** 어휘 — 모티프 계열마다 짝이 있다 (로커 밴드·범퍼 밴드·
# 지붕 블랙아웃의 윗선). 곧은 선을 겹쳐 지우는 것이 일이므로 **큰 덩어리 도형**
# 만 쓴다 — 산포 모티프(`MOTIF_SETS`)와는 다른 벌이다.
EDGE_SETS: dict[str, tuple[str, ...]] = {
    "star": ("B_33", "A_21"),          # 별 폭발 (Ai의 자홍 별무리 · Yotsugi의 흰 톱니)
    "flower": ("B_26", "U_27"),        # 구름 로브 (Ishtar의 구름 프레임)
    "splat": ("G_02", "G_01"),         # 물감 필드 (EVELYNE의 흰 물감)
    "swirl": ("U_31", "B_26"),
    "crystal": ("A_23", "B_32"),
}


# 테마색이 없는 도안의 계열 — 물감 튐. 도상이 없어서(별=천체, 꽃=식물과 달리)
# 어느 캐릭터에나 얹히는 유일한 계열이다.
MOTIF_NEUTRAL = "splat"


def motif_family(lk: Look) -> str:
    """이 도안의 **모티프 계열** — 테마색의 색조가 고른다 (손튜닝 없음).

    테마색이 없는 도안(파스텔·무채)은 도상이 없는 계열로 간다 —
    꽃·별은 색과 뜻이 있어야 읽히고, 없는 그림에 얹으면 남의 스티커가 된다.

    **이 판정이 닿지 않는 것이 있다.** 레퍼런스의 계열은 캐릭터 의미에서 온다
    (수이세이가 별인 것은 이름이 '별마을 혜성'이라서고, ARIS가 픽셀인 것은
    게이머라서다) — 팔레트에서 유도할 수 있는 것이 아니다. 그래서 사람이
    덮어쓸 수 있어야 한다 (`build(motif=…)` · 편집기 [Motif Family]).
    """
    t = theme_color(lk)
    if t is None:
        return MOTIF_NEUTRAL
    h = rgb_to_hsb(*t)[0]
    for lo, hi, name in MOTIF_THEME:
        if lo <= h < hi:
            return name
    return "swirl"


def motif_shapes(lk: Look, cat: Catalog,
                 family: str | None = None) -> tuple[str, ...]:
    """이 도안의 **모티프 어휘** — 한 계열의 두 종뿐이다.

    별을 늘 섞던 옛 규약은 레퍼런스와 어긋난다: 꽃을 쓰는 레퍼런스(마린의 벚꽃 ·
    EVELYNE의 백합 · RIN의 아네모네)에는 별이 없고, 별을 쓰는 레퍼런스에는 꽃이
    없다. 한 대에 한 계열이다.

    `family`를 주면 그 계열로 못 박는다 (사람이 고른 것 — `build(motif=…)`).
    """
    got = tuple(n for n in MOTIF_SETS[family or motif_family(lk)]
                if n in cat.shapes)
    return got or (cat.circle,)


def edge_shapes(lk: Look, cat: Catalog,
                family: str | None = None) -> tuple[str, ...]:
    """투톤 경계를 뜯는 어휘 — 모티프 계열의 짝 (`EDGE_SETS`)."""
    got = tuple(n for n in EDGE_SETS[family or motif_family(lk)]
                if n in cat.shapes)
    return got or (cat.circle,)
