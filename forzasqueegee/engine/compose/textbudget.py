"""텍스트 레이어 예산 — 남은 장수·우선순위로 **층(tier)**을 정한다.

커스텀 텍스트 도안은 품질이 좋지만 비싸다 (필기체 한 줄이 수백 장). 그래서
층이 있다:

| 층 | 무엇 |
|---|---|
| A·B·C | 커스텀 도안 — 정책 사다리(`textfit.LADDER`) 일곱 칸을 A(고움)·B·C(거침)로 묶은 이름 |
| D | 게임 글꼴 비닐 (`textvinyl`) — 한 글자 한 장 (테두리·그림자는 벌 수) |
| E | 글자 생략 |

커스텀은 **칸** 단위로 움직인다: 예산에 드는 가장 고운 칸을 고르고, 어느 칸도
안 들면 그림자 → 테두리 순으로 벌을 뺀다 (`textglyph.plan_for_budget`). 층
이름은 기록·크기 하강(`textbuild.tier_for_size`)이 쓴다.

어느 칸이냐는 **면에 남은 장수**(면 상한 − 도안 − 꾸밈)와 `priority`가 정한다.
`high`면 다른 꾸밈(산포·에코)이 먼저 물러난다 (`design`이 예산을 그쪽에서 뺀다).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ...i18n import msg
from .. import textglyph as tg
from .textspec import TextSpec


TIERS = ("A", "B", "C", "D", "E")


# 글자에 쓸 장수의 상한 (옆면 상한 3,000 대비) — 보통 / 우선순위 high
VALUE_FRAC = 0.20


VALUE_FRAC_HIGH = 0.32


# 게임 글꼴 비닐의 장수 — 공백 아닌 글자 수 × 벌 수 (`gametext.TextJob.n_layers`)
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
    n_main: int = 0
    n_sub: int = 0
    ix_main: int = 1              # 사다리 칸 (커스텀일 때)
    ix_sub: int = 3
    free: int = 0
    allow_game: bool = True
    notes: list[str] = field(default_factory=list)

    @property
    def custom_main(self) -> bool:
        return self.tier_main in ("A", "B", "C")

    @property
    def custom_sub(self) -> bool:
        return self.tier_sub in ("A", "B", "C")


def plan_tiers(spec: TextSpec, style: str, free: int) -> TextPlan:
    """남은 장수 `free`에 맞는 층. 스타일은 이미 정해진 것(커스텀 6종 중 하나).

    메인은 예산에 드는 가장 고운 칸, 서브는 남는 몫에서 메인보다 두 칸 아래부터
    (위계 — 서브는 테두리만, 그림자 없음). 커스텀이 하나도 안 들면 게임 글꼴(D,
    허락할 때) 아니면 생략(E).
    """
    rule = tg.STYLE_RULES.get(style, tg.StyleRule())
    outline = resolve_tri(spec.outline, rule.outline_default)
    shadow = resolve_tri(spec.shadow, rule.shadow_default)
    # 글자가 제 값을 하는 상한 — 면 상한의 몫. 이름 하나에 도안의 절반을 쓰는 것은
    # 사람 문법이 아니다 (레퍼런스의 워드마크는 수십~백여 장 규모다). `high`만 더 준다.
    value = int(VALUE_FRAC_HIGH * 3000) if spec.priority == "high" else int(VALUE_FRAC * 3000)
    cap = min(free, value) if spec.max_layers is None else min(free, spec.max_layers)
    main = spec.main or ""
    sub = spec.sub
    notes: list[str] = []
    tier_main, n_main = "E", 0
    ix_main, ix_sub = 1, 3
    choice = tg.plan_for_budget(main, style, cap, outline, shadow)
    if choice is not None:
        tier_main, n_main, ix_main = choice.tier, choice.n, choice.ix
        outline, shadow = choice.outline, choice.shadow
    if tier_main == "E":
        g = game_layers(main, outline, shadow)
        if spec.allow_fallback_to_game_text and g <= cap:
            tier_main, n_main = "D", g
        elif spec.allow_fallback_to_game_text and game_layers(main, False, False) <= cap:
            tier_main, n_main = "D", game_layers(main, False, False)
            outline = shadow = False
    tier_sub, n_sub = "E", 0
    if sub and tier_main != "E":
        left = cap - n_main
        if tier_main != "D":
            c = tg.plan_for_budget(sub, style, left, outline, False, ix_min=ix_main + 2)
            if c is not None:
                tier_sub, n_sub, ix_sub = c.tier, c.n, c.ix
        if tier_sub == "E" and spec.allow_fallback_to_game_text:
            g = game_layers(sub, False, False)
            if g <= left:
                tier_sub, n_sub = "D", g
    if tier_main == "E":
        notes.append(msg("텍스트를 뺀다 — 면에 남은 장수 {free:,}장으로는 어느 층도 못 선다",
                         free=cap))
    else:
        notes.append(msg("텍스트 층 {tier} ({n:,}장, 남은 예산 {free:,}장)"
                         + (msg(" · 서브 층 {sub_tier} ({sub_n:,}장)") if sub else ""),
                         tier=tier_main, n=n_main, free=cap, sub_tier=tier_sub, sub_n=n_sub))
    return TextPlan(tier_main=tier_main, tier_sub=tier_sub, outline=outline,
                    shadow=shadow, n_main=n_main, n_sub=n_sub, ix_main=ix_main, ix_sub=ix_sub,
                    free=cap, allow_game=spec.allow_fallback_to_game_text, notes=notes)
