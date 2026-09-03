"""역할 팔레트 — 색을 **역할**로 짠다 (베이스 · 베드 · 주/부 액센트 · 그림자 · 하이라이트 · 무채).

옛 세 벌(`accent_color`·`accent_tint`·`accent_third`)은 "산포에 쓸 유채 세 색"
이었다. 사람이 만든 이타샤의 색은 그보다 역할이 많다 — 인물 뒤에 받치는
큰 색면(베드)은 인물의 **그림자색**이거나 무채고, 하이라이트색은 잔 조각·
선에만 쓰며, 베이스는 캐릭터와 녹지 않게 중립으로 물러난다.

여기서 하는 일은 도안에서 읽은 씨앗(`DesignIntent`)과 옛 액센트 규칙을 받아
역할표 한 벌을 짜는 것이고, **가독성 규칙**(실루엣 테두리와 베드의 명도차 ·
베드와 베이스의 명도차)을 여기서 보장한다. 후보마다 변종을 낸다
(`variant`) — 어느 변종이 이기는지는 점수가 정한다.
"""

from __future__ import annotations

import colorsys
from dataclasses import dataclass

from ..model import rgb_to_hsb
from .intent import DesignIntent
from .look import Look
from .palette import (
    INK_DARK, INK_LIGHT, accent_color, accent_third, accent_tint,
    achromatic_accent, readable_on)


# 베드와 실루엣 테두리 사이에 벌려야 할 명도차 — 이 아래면 실루엣이 베드에 묻힌다.
# (테두리는 후광·선이 한 번 더 지키므로 베이스 쪽 간격보다 작다)
BED_EDGE_GAP = 0.18


# 베드와 **인물 덩어리** 사이의 명도차 — 멀리서 인물이 판에서 떨어져 나오나.
#
# 테두리 간격만으로는 못 지킨다: 파스텔 인물은 속이 밝고 윤곽선이 검어서
# 테두리 자로는 "밝은 판"이 정답으로 나오는데, 그러면 멀리서 인물이 판에
# 녹는다 (실측 silvia-01: 테두리 명도차 0.46 · far 배율 인물 끌림 0.029 대
# 꾸밈 0.199 — 주역이 뒤집혔다). 멀리서 읽히는 것은 덩어리의 밝기다.
BED_BODY_GAP = 0.26


# 베드와 베이스 사이의 명도차 — 이 아래면 베드가 판으로 안 읽히고 얼룩이 된다.
# 레퍼런스의 판은 예외 없이 베이스와 확실히 갈린다 (흰 차 위 남색·검정 판).
BED_BASE_GAP = 0.32


# 인물 테두리가 이보다 어두우면 짙은 판을 못 쓴다 — 중간 명도로 간다.
BED_DARK_EDGE = 0.30


# 베드 채도 상한 — 배경은 인물보다 눌러야 한다 (테마 베이스와 같은 규칙).
BED_SAT_MAX = 0.62


ROLE_VARIANTS = ("shadow", "primary", "neutral", "inverse", "pastel", "neon")


# 파스텔 변종 — 판의 채도 상한 · 명도 하한 (무늬·꽃 프리셋의 "파스텔 바탕").
PASTEL_SAT = 0.30
PASTEL_VAL = 0.90


# 형광 변종 — 액센트의 채도 하한 · 명도 (다크 프리셋의 "형광 액센트").
NEON_SAT = 0.88
NEON_VAL = 1.0


@dataclass(frozen=True)
class RolePalette:
    base: tuple[int, int, int]           # 자동차 도색 (비닐 아님)
    bed: tuple[int, int, int]            # 인물 뒤 큰 색면
    bed_alt: tuple[int, int, int]        # 베드의 둘째 판 (사선판·그림자판)
    primary: tuple[int, int, int]        # 주 액센트 (큰 모티프·띠)
    secondary: tuple[int, int, int]      # 부 액센트 (밝은 자매)
    shadow: tuple[int, int, int]         # 그림자 액센트 (어두운 짝)
    highlight: tuple[int, int, int]      # 하이라이트 (잔 조각·선)
    dark: tuple[int, int, int]           # 그래픽 무채 (로커·후광·지붕)
    variant: str = "shadow"

    @property
    def motif_trio(self) -> tuple[tuple[int, int, int], ...]:
        """산포 모티프가 도는 세 색 — 옛 `(main, tint, third)` 자리다."""
        return (self.primary, self.secondary, self.highlight)


def _lum(c: tuple[int, int, int]) -> float:
    return (0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]) / 255.0


def _hsb_rgb(h: float, s: float, b: float) -> tuple[int, int, int]:
    return tuple(int(round(float(v) * 255)) for v in colorsys.hsv_to_rgb(
        float(h) % 1.0, max(0.0, min(1.0, float(s))), max(0.0, min(1.0, float(b)))))


def _shift(c: tuple[int, int, int], ds: float = 0.0, db: float = 0.0) -> tuple[int, int, int]:
    h, s, b = rgb_to_hsb(*c)
    return _hsb_rgb(h, s + ds, b + db)


def _pastel(c: tuple[int, int, int]) -> tuple[int, int, int]:
    """색조는 두고 채도를 눌러 밝힌다 — 무채면 근백."""
    h, s, b = rgb_to_hsb(*c)
    return _hsb_rgb(h, min(s, PASTEL_SAT), max(b, PASTEL_VAL))


def _neon(c: tuple[int, int, int]) -> tuple[int, int, int]:
    """색조는 두고 채도·명도를 끝까지 — 무채(살색·회색)는 형광이 못 되니 그대로."""
    h, s, b = rgb_to_hsb(*c)
    if s < 0.12:
        return c
    return _hsb_rgb(h, max(s, NEON_SAT), NEON_VAL)


def _separate(c: tuple[int, int, int], edge_lum: float, base_lum: float,
              prefer_dark: bool, body_lum: float | None = None
              ) -> tuple[int, int, int]:
    """색을 **테두리 · 인물 덩어리 · 베이스 셋 다에서** 명도로 뗀다 (색조는 지킨다)."""
    h, s, b = rgb_to_hsb(*c)
    s = min(s, BED_SAT_MAX)
    # 세 제약(테두리·덩어리·베이스와의 명도차)의 **여유**가 큰 명도를 고른다 —
    # 셋 다 만족하는 값이 없을 때도 (밝은 인물 + 밝은 차) 가장 덜 나쁜 명도가
    # 나온다. 같은 여유면 원래 명도에 가깝고 선호 방향(짙게/연하게)인 쪽이다.
    best, bs = b, -1e9
    for k in range(0, 101, 2):
        v = k / 100.0
        margin = min(abs(v - edge_lum) / BED_EDGE_GAP, abs(v - base_lum) / BED_BASE_GAP)
        if body_lum is not None:
            margin = min(margin, abs(v - body_lum) / BED_BODY_GAP)
        margin = min(margin, 1.0)
        sc = margin * 2.0 - 0.3 * abs(v - b) + (0.25 if (v < 0.5) == prefer_dark else 0.0)
        if sc > bs:
            best, bs = v, sc
    return _hsb_rgb(h, s, max(0.06, min(0.97, best)))


def role_palette(it: DesignIntent, lk: Look, base: tuple[int, int, int],
                 variant: str = "shadow") -> RolePalette:
    """역할 팔레트 한 벌.

    변종:

    - `shadow`  — 베드는 인물의 **그림자색**(어두운 짝) 계열. 레퍼런스의 대다수
      (린의 남색 판 · EVELYNE의 검정 위 흰 백합).
    - `primary` — 베드가 주 액센트의 눌린 판. 테마색이 확실한 도안.
    - `neutral` — 베드가 무채(근검정/근백) 판. 파스텔·저대비 도안.
    - `inverse` — 베이스가 무채로 물러나고 베드가 액센트를 쥔다 (밝은 도안 위 짙은 판).
    - `pastel`  — 베드가 주색·부색의 **파스텔**(옅고 밝은 판). 무늬·꽃 프리셋.
    - `neon`    — 액센트 셋이 **형광**으로 서고 베드는 그 형광의 눌린 판.
      검정 바탕에서만 뜻이 있다 (다크 프리셋 · `families` dark).
    """
    base_lum = _lum(base)
    prim = accent_color(lk, base)
    sec = accent_tint(prim, base)
    third = accent_third(prim, lk, base)
    edge = it.edge_lum
    dark = INK_DARK if base_lum > 0.55 else INK_LIGHT
    # 그림자·하이라이트 — 도안 씨앗이 있으면 그것, 없으면 주색에서 유도
    shadow = it.shadow_rgb or _shift(prim, ds=0.10, db=-0.38)
    hl = it.highlight_rgb or _shift(sec, ds=-0.15, db=0.12)
    if achromatic_accent(prim):
        # 무채 액센트 도안 — 그림자는 근검정, 하이라이트는 근백 (Cygames 86·EVELYNE)
        shadow = it.dark_neutral_rgb or (28, 28, 34)
        hl = it.light_neutral_rgb or (250, 250, 250)
    # 판은 **베이스의 반대**다 (흰 차엔 짙은 판, 검은 차엔 연한 판). 인물이
    # 아주 어두우면 짙은 판이 실루엣을 삼키므로 `_separate`가 중간 명도로 민다.
    body = it.body_lum
    # 판은 **인물 덩어리의 반대쪽**으로 간다 — 밝은 인물이면 짙은 판, 짙은
    # 인물이면 연한 판. 베이스의 반대라는 옛 규칙 위에 이것이 얹힌다.
    prefer_dark = body > 0.5 if abs(body - 0.5) > 0.08 else base_lum > 0.5
    if prefer_dark and edge < BED_DARK_EDGE and body < 0.5:
        prefer_dark = False
    if variant == "shadow":
        bed = _separate(shadow, edge, base_lum, prefer_dark, body)
        bed_alt = _separate(prim, edge, base_lum, prefer_dark, body)
    elif variant == "primary":
        bed = _separate(prim, edge, base_lum, prefer_dark, body)
        bed_alt = _separate(shadow, edge, base_lum, prefer_dark, body)
    elif variant == "neutral":
        bed = _separate((30, 30, 36) if prefer_dark else (246, 246, 246), edge,
                        base_lum, prefer_dark, body)
        bed_alt = _separate(prim, edge, base_lum, prefer_dark, body)
    elif variant == "pastel":
        # 판은 밝은 파스텔 — `_separate`는 명도만 고르니 채도를 먼저 눌러 두고
        # 연한 쪽을 선호시킨다 (인물이 아주 밝으면 자가 중간 명도로 민다)
        bed = _separate(_pastel(prim), edge, base_lum, False, body)
        bed_alt = _separate(_pastel(sec), edge, base_lum, False, body)
    elif variant == "neon":
        prim, sec, third = _neon(prim), _neon(sec), _neon(third)
        hl = _neon(hl)
        bed = _separate(prim, edge, base_lum, prefer_dark, body)
        bed_alt = _separate(third, edge, base_lum, prefer_dark, body)
    else:                                # inverse — 액센트가 판, 하이라이트가 잔것
        bed = _separate(prim, edge, base_lum, not prefer_dark, body)
        bed_alt = _separate(third, edge, base_lum, not prefer_dark, body)
        prim, sec, hl = third, prim, sec
    return RolePalette(base=base, bed=bed, bed_alt=bed_alt,
                       primary=readable_on(prim, base), secondary=readable_on(sec, base),
                       shadow=shadow, highlight=readable_on(hl, base), dark=dark,
                       variant=variant)


def bed_readability(pal: RolePalette, it: DesignIntent) -> float:
    """베드 위에서 실루엣이 읽히는 정도 (0~1) — 테두리 명도와 베드 명도의 차."""
    return max(0.0, min(1.0, abs(_lum(pal.bed) - it.edge_lum) / 0.5))
