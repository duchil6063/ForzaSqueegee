"""역할표 — 차에 실린 **덩어리마다 무엇인가**를 읽는다 (주역 · 보조 · 로고 · 글자 · 그대로).

사람이 만든 리버리 30벌은 그림 한 장으로 짜이지 않는다 — 주역 캐릭터 한 덩어리
(≈2,440장·살색) + 후드·유리의 둘째 그림·치비(120~1,200장) + 스폰서 로고 무리
(≤120장·2~4색) + 글자 단어(글리프 연속)가 한 벌이다 (`work/lab/humanref`,
2026-09-02). 편집기에 실린 것이 재료이므로(사용자 결정 ①) 구성기는 덩어리마다
역할을 읽어야 어느 것을 앵커로 삼고 어느 것을 그대로 둘지 안다.

문턱은 사람 판 실측이다 (`work/lab/whole/ruler.material_stats`와 같은 자):

| 역할 | 자 |
|---|---|
| `text`    | 글꼴 글리프 페이지 비 ≥ 0.60 |
| `hero`    | ≥ 1,200장 — 살색이 없어도 후보다 (로봇·동물 주역) |
| `support` | 120 ~ 1,200장, 또는 작은데 로고 자에 안 드는 것 |
| `logo`    | ≤ 120장 · ≤ 4색 · 살색 0 |
| `pinned`  | 사람이 고른 것만 — "그대로" (꾸밈이 안 건드린다). 자동으로는 안 낸다 |

오판은 사람이 고친다 — 편집기 [Auto Decoration] 창의 실린 그림 표
(`engine.fls.studio`). 그래서 여기서는 **왜 그렇게 읽었나**(`why`)를 같이 낸다.
"""

from __future__ import annotations

import colorsys
from dataclasses import dataclass

from ...i18n import msg
from ..model import LayerPlan
from ..textvinyl import FONTS

ROLES = ("hero", "support", "logo", "text", "pinned")
ROLE_LABELS = {
    "hero": msg("주역"), "support": msg("보조"), "logo": msg("로고"),
    "text": msg("글자"), "pinned": msg("그대로"),
}

# 게임 글꼴 글리프 페이지 — 열한 글꼴의 대·소문자 그룹 (`textvinyl.FONTS`).
FONT_PAGES = frozenset(p for pair in FONTS.values() for p in pair)

HERO_MIN = 1200          # 이 위면 주역 후보 (사람 주역 중앙 2,440장 · p10 ≈ 1,200)
SUPPORT_MIN = 120        # 이 위면 그림 (사람 둘째 그림·치비의 아랫단)
LOGO_COLORS_MAX = 4      # 로고는 색이 적다 (사람 로고 중앙 2색)
SKIN_MIN = 0.03          # 살색 레이어 몫 — 이 위면 인물이 있다
GLYPH_MIN = 0.60         # 글리프 몫 — 이 위면 글자 덩어리다

# 로고·글자는 **절대 미러하지 않는다** (사용자 결정 ③) — 반대편에는 읽는
# 방향 그대로 다시 앉힌다. 그림은 미러한다.
NO_MIRROR_ROLES = frozenset({"logo", "text"})


@dataclass(frozen=True)
class CastEntry:
    role: str
    why: str
    layers: int
    colors: int
    skin: float          # 살색 레이어 몫 (0~1)
    glyph: float         # 글리프 레이어 몫 (0~1)

    @property
    def no_mirror(self) -> bool:
        return self.role in NO_MIRROR_ROLES

    @property
    def pinned(self) -> bool:
        return self.role == "pinned"


def _page(shape: str) -> str:
    return str(shape).split("_")[0]


def _skin(rgb) -> bool:
    """살색인가 — 머리 찾기(`intent._head_box`)와 같은 자. 사람 판의 자
    (`ruler._skin`, 채도 ≥ 0.12)보다 옅은 쪽을 더 받는다: 셀 도안의 살색은
    팔레트 스냅으로 채도가 0.1 아래로 내려앉는다 (cel-01 실측 0.35% → 이 자로 잡힌다)."""
    r, g, b = (int(v) / 255.0 for v in rgb)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return h <= 0.11 and 0.03 < s < 0.55 and v > 0.55


def estimate(plan: LayerPlan) -> CastEntry:
    """도안 하나의 역할 — 장수·색 수·살색 비·글리프 비로 읽는다."""
    layers = plan.layers
    n = len(layers)
    if n == 0:
        return CastEntry("support", msg("빈 도안"), 0, 0, 0.0, 0.0)
    colors = len({tuple(l.color) for l in layers if not l.mask})
    # 살색은 **넓이**로 잰다 (장수가 아니라) — 셀 도안은 획 한 마디가 한 장이라
    # 장수로 재면 얼굴 하나가 1,700장 중 여섯 장(0.35%)이다. 사람 판의 문턱
    # (`SKIN_MIN`)은 큰 채움 도형이 많은 판에서 나왔으니 넓이 몫이 그 뜻에 가깝다.
    area = [abs(float(l.sx) * float(l.sy)) for l in layers]
    tot = sum(a for l, a in zip(layers, area) if not l.mask) or 1e-9
    skin = sum(a for l, a in zip(layers, area) if not l.mask and _skin(l.color)) / tot
    glyph = sum(1 for l in layers if _page(l.shape) in FONT_PAGES) / n
    if glyph >= GLYPH_MIN:
        role, why = "text", msg("글리프 {g:.0%}", g=glyph)
    elif n >= HERO_MIN:
        role = "hero"
        why = (msg("{n:,}장 · 살색 {s:.1%}", n=n, s=skin) if skin > 0.0
               else msg("{n:,}장 · 살색 없음 (배경 그림이면 보조로 고칠 것)", n=n))
    elif n >= SUPPORT_MIN:
        role, why = "support", msg("{n:,}장 · 살색 {s:.1%}", n=n, s=skin)
    elif colors <= LOGO_COLORS_MAX and skin < 1e-9:
        role, why = "logo", msg("{n}장 · {c}색 · 살색 없음", n=n, c=colors)
    else:
        role = "support"
        why = msg("{n}장 · {c}색 · 살색 {s:.1%} — 작은 그림", n=n, c=colors, s=skin)
    return CastEntry(role, why, n, colors, round(skin, 4), round(glyph, 4))


def is_role(name: str) -> bool:
    return name in ROLES
