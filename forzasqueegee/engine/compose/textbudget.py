"""텍스트 레이어 예산 — 남은 장수·엔진으로 **층(tier)**을 정한다.

| 층 | 무엇 |
|---|---|
| D | 게임 글꼴 글리프 (`textvinyl`) — 한 글자 한 장 (테두리 4·그림자 1은 벌 수). **기본 엔진** |
| A·B | 도형 맞춤 커스텀 도안 (`engine == "shapes"`) — 정책 사다리(`textfit.LADDER`) 네 칸을 A(고움)·B로 묶은 이름 |
| E | 글자 생략 |

게임 글꼴은 실제 타이포이고 값싸다 — 레퍼런스의 이름 글자가 이것이다. 도형
맞춤은 사람이 고른 엔진(`TextSpec.engine == "shapes"`)일 때만이고, 그때도
**고운 칸이 예산에 들 때만**이다: 거친 칸은 글자를 상자 덩어리로 만든다
(사용자 판정 2026-08-31). 안 들면 게임 글꼴로 물러난다.

어느 칸이냐는 **면에 남은 장수**(면 상한 − 도안 − 꾸밈)와 `priority`가 정한다.
`high`면 다른 꾸밈(산포·에코)이 먼저 물러난다 (`design`이 예산을 그쪽에서 뺀다).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ...i18n import msg
from .. import textglyph as tg
from .textspec import TextSpec
from .textstyle import GAME_FONT


TIERS = ("A", "B", "D", "E")


# 글자에 쓸 장수의 상한 (옆면 상한 3,000 대비) — 보통 / 우선순위 high
VALUE_FRAC = 0.20


VALUE_FRAC_HIGH = 0.32


# 게임 글꼴 글리프의 장수 — 공백 아닌 글자 수 × 벌 수 (본색 1 · 테두리 4 · 그림자 1)
def game_layers(text: str, outline: bool, shadow: bool) -> int:
    n = sum(1 for c in text if not c.isspace())
    return n * (1 + (4 if outline else 0) + (1 if shadow else 0))


def resolve_tri(v: str, default: bool) -> bool:
    return default if v == "auto" else v == "on"


@dataclass
class TextPlan:
    """예산이 정한 층 — 메인과 서브 따로."""

    tier_main: str
    tier_sub: str                 # 서브가 없으면 "E"
    outline: bool
    shadow: bool
    font: str                     # 게임 글꼴 (층 D가 쓴다 — 스타일의 짝)
    engine: str = "font"
    n_main: int = 0
    n_sub: int = 0
    ix_main: int = 1              # 사다리 칸 (커스텀일 때)
    ix_sub: int = 3
    free: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def custom_main(self) -> bool:
        return self.tier_main in ("A", "B")

    @property
    def custom_sub(self) -> bool:
        return self.tier_sub in ("A", "B")


def _game(text: str, outline: bool, shadow: bool, cap: int) -> tuple[int, bool, bool] | None:
    """게임 글꼴로 예산에 드는 (장수, 테두리, 그림자) — 그림자 → 테두리 순으로 뺀다."""
    for ol, sh in ((outline, shadow), (outline, False), (False, False)):
        n = game_layers(text, ol, sh)
        if n <= cap:
            return n, ol, sh
    return None


def plan_tiers(spec: TextSpec, style: str, free: int) -> TextPlan:
    """남은 장수 `free`에 맞는 층. 스타일은 이미 정해진 것(여섯 중 하나).

    기본 엔진(`font`)은 층 D다 — 예산에 안 들면 그림자 → 테두리를 빼고, 본색만으로도
    안 들면 생략(E). `shapes` 엔진은 메인을 예산에 드는 가장 고운 칸(A·B)으로,
    서브는 남는 몫에서 메인보다 두 칸 아래부터 (위계 — 서브는 테두리만) 짓고,
    어느 칸도 안 들면 게임 글꼴로 물러난다.
    """
    rule = tg.STYLE_RULES.get(style, tg.StyleRule())
    outline = resolve_tri(spec.outline, rule.outline_default)
    shadow = resolve_tri(spec.shadow, rule.shadow_default)
    font = GAME_FONT.get(style, "arial")
    # 글자가 제 값을 하는 상한 — 면 상한의 몫. 이름 하나에 도안의 절반을 쓰는 것은
    # 사람 문법이 아니다 (레퍼런스의 워드마크는 수십~백여 장 규모다). `high`만 더 준다.
    value = int(VALUE_FRAC_HIGH * 3000) if spec.priority == "high" else int(VALUE_FRAC * 3000)
    cap = min(free, value) if spec.max_layers is None else min(free, spec.max_layers)
    main = spec.main or ""
    sub = spec.sub
    notes: list[str] = []
    tier_main, n_main = "E", 0
    ix_main, ix_sub = 1, 3
    shapes = spec.engine == "shapes"
    if shapes:
        choice = tg.plan_for_budget(main, style, cap, outline, shadow)
        if choice is not None:
            tier_main, n_main, ix_main = choice.tier, choice.n, choice.ix
            outline, shadow = choice.outline, choice.shadow
        elif spec.allow_fallback_to_game_text:
            notes.append(msg("도형 맞춤 글자가 예산 {free:,}장에 안 든다 — 게임 글꼴로 간다",
                             free=cap))
    if tier_main == "E" and (not shapes or spec.allow_fallback_to_game_text):
        g = _game(main, outline, shadow, cap)
        if g is not None:
            n_main, outline, shadow = g
            tier_main = "D"
    tier_sub, n_sub = "E", 0
    if sub and tier_main != "E":
        left = cap - n_main
        if tier_main != "D":
            c = tg.plan_for_budget(sub, style, left, outline, False, ix_min=ix_main + 2)
            if c is not None:
                tier_sub, n_sub, ix_sub = c.tier, c.n, c.ix
        if tier_sub == "E" and (not shapes or spec.allow_fallback_to_game_text):
            g = _game(sub, outline, False, left)
            if g is not None:
                n_sub, _ol, _sh = g
                tier_sub = "D"
    if tier_main == "E":
        notes.append(msg("텍스트를 뺀다 — 면에 남은 장수 {free:,}장으로는 어느 층도 못 선다",
                         free=cap))
    else:
        notes.append(msg("텍스트 층 {tier} ({n:,}장, 남은 예산 {free:,}장)"
                         + (msg(" · 서브 층 {sub_tier} ({sub_n:,}장)") if sub else ""),
                         tier=tier_main, n=n_main, free=cap, sub_tier=tier_sub, sub_n=n_sub))
    return TextPlan(tier_main=tier_main, tier_sub=tier_sub, outline=outline,
                    shadow=shadow, font=font, engine=spec.engine, n_main=n_main, n_sub=n_sub,
                    ix_main=ix_main, ix_sub=ix_sub, free=cap, notes=notes)
