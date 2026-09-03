"""텍스트 스펙 — 사람이 넣은 캐릭터 이름·작품명과 그 옵션.

글자는 **선택**이다 (기본 꺼짐 — 이타샤 어휘의 "글자는 안 넣는다"를 옵션으로
푼 것이다). 켜면 **게임 글꼴 글리프**(`engine.textvinyl` — 한 글자 한 장, 실제
타이포)가 기본 엔진이고, `engine: shapes`를 고르면 동봉 OFL 글꼴을 도형으로
되짓는 커스텀 도안(`engine.textglyph`)을 쓴다 — 그쪽은 고운 층(A·B)이 예산에
들 때만이고 안 들면 게임 글꼴로 물러난다 (`textbudget`).

조리법(`<이름>.fsitasha.json`)의 `text` 키와 CLI 인자가 같은 꼴로 여기 온다.
문자열은 **그대로** 둔다 — 띄어쓰기·대소문자·구두점·줄바꿈을 안 건드린다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from ...i18n import msg


# `game`은 옛 이름이다 — 엔진이 갈리기 전에는 "게임 글꼴을 써라"가 스타일이었다.
# 이제 그것이 기본 엔진이라 `auto`와 같은 뜻이고, `from_dict`가 그렇게 푼다.
STYLES = ("auto", "script", "brush", "graffiti", "racing", "techno", "minimal", "game")


PLACEMENTS = ("auto", "side", "rear", "hood", "roof", "window")


ENGINES = ("font", "shapes")


PRIORITIES = ("high", "normal", "low")


TRI = ("auto", "on", "off")


@dataclass
class TextSpec:
    enabled: bool = False
    main: str | None = None          # 캐릭터 이름 (character tag / wordmark)
    sub: str | None = None           # 작품명 · 별칭 · 팀명 (보조)
    number: str | None = None        # 레이싱 번호 — 레이싱 프리셋만 앉힌다 (`presets`)
    style: str = "auto"
    engine: str = "font"             # font = 게임 글꼴 글리프 · shapes = 도형 맞춤 (OFL 글꼴)
    placement: str = "auto"
    priority: str = "normal"
    # `engine == "shapes"`인데 도형 맞춤이 예산에 안 들 때 게임 글꼴로 물러나도 되나.
    # 끄면 그 판에서는 글자를 뺀다 (`engine == "font"`에는 뜻이 없다 — 그것이 곧 게임 글꼴이다).
    allow_fallback_to_game_text: bool = True
    max_layers: int | None = None
    outline: str = "auto"
    shadow: str = "auto"

    @property
    def active(self) -> bool:
        return bool(self.enabled and self.main and self.main.strip())

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "TextSpec":
        d = dict(d or {})
        spec = cls(
            enabled=bool(d.get("enabled", False)),
            main=_clean(d.get("main")), sub=_clean(d.get("sub")),
            number=_clean(d.get("number")),
            style=("auto" if str(d.get("style") or "auto") == "game"
                   else str(d.get("style") or "auto")),
            engine=("font" if str(d.get("style") or "") == "game"
                    else str(d.get("engine") or "font")),
            placement=str(d.get("placement") or "auto"),
            priority=str(d.get("priority") or "normal"),
            allow_fallback_to_game_text=bool(d.get("allow_fallback_to_game_text", True)),
            max_layers=(int(d["max_layers"]) if d.get("max_layers") not in (None, "") else None),
            outline=str(d.get("outline") or "auto"),
            shadow=str(d.get("shadow") or "auto"))
        spec.validate()
        return spec

    def validate(self) -> None:
        if self.style not in STYLES:
            raise ValueError(msg("모르는 텍스트 스타일: {style} (있는 것: {styles})",
                                 style=self.style, styles=", ".join(STYLES)))
        if self.engine not in ENGINES:
            raise ValueError(msg("모르는 텍스트 엔진: {engine} (있는 것: {engines})",
                                 engine=self.engine, engines=", ".join(ENGINES)))
        if self.placement not in PLACEMENTS:
            raise ValueError(msg("모르는 텍스트 자리: {placement} (있는 것: {places})",
                                 placement=self.placement, places=", ".join(PLACEMENTS)))
        if self.priority not in PRIORITIES:
            raise ValueError(msg("모르는 텍스트 우선순위: {priority} (있는 것: {values})",
                                 priority=self.priority, values=", ".join(PRIORITIES)))
        for name, v in (("outline", self.outline), ("shadow", self.shadow)):
            if v not in TRI:
                raise ValueError(msg("텍스트 {name}은 auto · on · off 중 하나다 (받은 것: {value})",
                                     name=name, value=v))
        if self.max_layers is not None and self.max_layers < 0:
            raise ValueError(msg("텍스트 장수 상한은 0 이상이다"))


def _clean(v) -> str | None:
    """문자열은 그대로 — 다만 CLI 따옴표 안의 `\\n`은 줄바꿈으로 푼다."""
    if v is None:
        return None
    s = str(v).replace("\\n", "\n")
    return s if s.strip() else None


def text_from_args(args) -> TextSpec | None:
    """CLI 인자(`--text` …) → 스펙. `--text`가 없으면 None (기존 그대로)."""
    main = getattr(args, "text", None)
    if not main:
        return None
    tri = lambda v: "auto" if v in (None, "auto") else ("on" if v in ("on", True, "1") else "off")
    return TextSpec.from_dict({
        "enabled": True, "main": main, "sub": getattr(args, "subtext", None),
        "number": getattr(args, "text_number", None),
        "style": getattr(args, "text_style", None) or "auto",
        "engine": getattr(args, "text_engine", None) or "font",
        "placement": getattr(args, "text_placement", None) or "auto",
        "priority": getattr(args, "text_priority", None) or "normal",
        "allow_fallback_to_game_text": (getattr(args, "game_text_fallback", "on") or "on") != "off",
        "max_layers": getattr(args, "text_max_layers", None),
        "outline": tri(getattr(args, "text_outline", None)),
        "shadow": tri(getattr(args, "text_shadow", None)),
    })
