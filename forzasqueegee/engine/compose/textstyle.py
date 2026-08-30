"""텍스트 스타일 — 계열·도안 분위기가 글꼴을 고르고, 역할 팔레트가 색을 준다.

`auto`는 사람이 고르지 않은 자리다. 레퍼런스의 타이포는 리버리 문법과 짝이다
(RIN SHIBUYA의 필기체는 부드러운 판 위에, ARIS의 픽셀 글자는 테크 무드에,
EVELYNE의 그래피티는 스플래시 위에) — 그래서 계열이 첫 근거고 도안 인상이
둘째다.
"""

from __future__ import annotations

from .. import textglyph as tg
from .families import Family
from .intent import DesignIntent
from .roles import RolePalette


# 계열 → 스타일 후보 (앞이 우선)
FAMILY_STYLES: dict[str, tuple[str, ...]] = {
    "minimal": ("minimal", "script"),
    "graphic_bed": ("script", "minimal", "brush"),
    "diagonal_flow": ("racing", "techno"),
    "motorsport": ("racing", "techno", "minimal"),
    "splash": ("brush", "graffiti"),
}


# 게임 글꼴 폴백 짝 (`textvinyl.FONTS`)
GAME_FONT: dict[str, str] = {
    "script": "script", "brush": "italic", "graffiti": "gothic",
    "racing": "condensed", "techno": "wide", "minimal": "sans2",
}


def choose_style(requested: str, fam: Family, it: DesignIntent) -> str:
    """`auto`면 계열 × 인상으로 고른다. 커스텀 6종 중 하나를 되돌린다 (`game`은 층 D의 일)."""
    if requested in tg.STYLES:
        return requested
    cands = FAMILY_STYLES.get(fam.name, ("minimal",))
    if it.impression == "sharp" and "techno" in cands:
        return "techno"
    if it.impression == "soft" and "script" in cands:
        return "script"
    if it.airy and "minimal" in cands:
        return "minimal"
    return cands[0]


def text_colors(pal: RolePalette, on_bed: bool, sub: bool = False
                ) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
    """(본색, 테두리, 그림자) — 판 위와 맨 도색 위가 다르다.

    판 위: 본색은 판의 반대 명도 무채(키라인과 같은 자) 또는 하이라이트, 테두리는 판색.
    도색 위: 본색은 주 액센트, 테두리는 무채 대비색, 그림자는 그림자 액센트.
    서브는 한 색으로 눌러 위계를 낸다.
    """
    def lum(c):
        return (0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]) / 255.0

    if on_bed:
        light = lum(pal.bed) < 0.5
        fill = (250, 250, 250) if light else (22, 22, 26)
        if not sub:
            # 하이라이트가 판과 충분히 갈리면 그것이 더 캐릭터답다
            if abs(lum(pal.highlight) - lum(pal.bed)) >= 0.45:
                fill = pal.highlight
        return fill, pal.bed, pal.shadow
    fill = pal.secondary if sub else pal.primary
    edge = pal.dark
    # 테두리는 본색과 갈려야 테다 — 짙은 액센트에 근검정 테는 한 덩이로 읽힌다
    # (실측: 남색 본색 + 검정 테 = 뭉개진 실루엣)
    if abs(lum(fill) - lum(edge)) < 0.3:
        edge = (250, 250, 250) if lum(fill) < 0.5 else (22, 22, 26)
    return fill, edge, pal.shadow
