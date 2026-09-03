"""스타일 프리셋 — 사람 판에서 읽은 **리버리 한 벌의 문법**을 이름 하나로.

구성 계열(`families`)은 옆면 색면의 뼈대일 뿐이다. 사람이 "레이싱 스폰서"라고
부르는 것은 그 뼈대에 팔레트 · 바탕 도색 · 글자 스타일과 크기 · 로고 줄 · 다른
면의 배정 · 예산 사다리가 **한 벌로** 묶인 것이다 (계획 A단계, 사용자 결정 ④
2026-09-02: 스타일은 프리셋 드롭다운). 여기서는 그 묶음만 적는다 — 실제
매개변수는 인물이 정하고(`macro.plan`) 어느 후보가 이기는지는 점수가 정한다.

`auto`(프리셋 없음)는 종전 그대로다 — 계열 넷을 후보로 지어 점수로 고르고, 바탕·
글자·로고는 각자의 기본을 따른다. 프리셋을 고르면 **그 계열 안에서만** 고르고
나머지 묶음이 함께 온다. 사람이 따로 정한 것(바탕 색 · 글자 스타일 · 로고 자리 ·
면 배정)은 프리셋보다 앞이다 — 프리셋은 기본값의 묶음이지 명령이 아니다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ...i18n import msg
from .families import FAMILIES


# 예산이 모자랄 때 버리는 순서의 기본 (`design.TRIM_ORDER`와 같다). 프리셋은
# 제 정체가 아닌 것부터 뺀다 — 무늬·꽃은 산포가 정체라 스택을 먼저 내준다.
_TRIM = ("itasha_echo", "itasha_deco", "itasha_stack", "itasha_stripe")


@dataclass(frozen=True)
class StylePreset:
    name: str
    family: str                       # 구성 계열 (`families.FAMILIES`)
    variants: tuple[str, ...]         # 도는 팔레트 변종 (`roles.ROLE_VARIANTS`)
    base: str = "auto"                # 바탕 도색 기본 — auto · black · pastel
    text_style: str = "auto"          # 글자 스타일 기본 (사람이 auto로 두었을 때)
    text_scale: float = 1.0           # 워드마크 높이 배율 (큰 이름 = 1.3~1.4)
    text_priority: str | None = None  # 글자 우선순위 기본 (None이면 사람 값)
    number: bool = False              # 레이싱 번호를 받는다 (`TextSpec.number`)
    logo_row: float = 1.0             # 옆면 로커 로고 줄의 크기 배율 (0이면 줄 없음)
    rear_poster: bool = False         # 로고·글자가 없는 리어에 전신 축소를 앉힌다
    motif_k: float = 1.0              # 산포 크기·수 배율
    bed_level: float | None = None    # 베드 크기 (계열 기본을 덮는다)
    macro: tuple[tuple[str, str], ...] | None = None   # 계열의 어휘 짝을 좁힌다
    trim: tuple[str, ...] = _TRIM     # 면 상한 사다리

    def fam(self):
        return FAMILIES[self.family]


STYLE_PRESETS: dict[str, StylePreset] = {
    # 레이싱 스폰서 — 벨트 띠 + 핀 둘 + 홈(띠 3~4) 위에 로고 줄과 번호
    "racing": StylePreset("racing", family="motorsport",
                          variants=("primary", "neutral", "shadow"),
                          text_style="racing", number=True, logo_row=1.3, motif_k=0.85),
    # 무늬·꽃 — 파스텔 바탕 + 테마 산포 (치비는 사용자가 유리에 올린 보조 그림)
    "floral": StylePreset("floral", family="graphic_bed",
                          variants=("pastel", "neutral", "shadow"), base="pastel",
                          text_style="script", logo_row=0.8, motif_k=1.3, bed_level=0.5,
                          trim=("itasha_echo", "itasha_stack", "itasha_deco", "itasha_stripe")),
    # 스플래시·찢김 — 덩어리 베드와 무늬 도형 가장자리
    "splash": StylePreset("splash", family="splash",
                          variants=("shadow", "neutral", "primary"),
                          text_style="brush", logo_row=0.8, motif_k=1.1),
    # 미니멀 — 색면 하나 + 워드마크, 옆면 로고 줄은 없고 리어는 전신 축소
    "minimal": StylePreset("minimal", family="minimal",
                           variants=("neutral", "shadow"),
                           text_style="minimal", text_scale=1.2, logo_row=0.0,
                           rear_poster=True, macro=(("ribbon", "none"),)),
    # 다크 그래피티 — 검정 바탕 + 형광 액센트 + 큰 워드마크
    "dark": StylePreset("dark", family="dark", variants=("neon", "primary"),
                        base="black", text_style="graffiti", text_scale=1.4,
                        text_priority="high", logo_row=0.8,
                        trim=("itasha_echo", "itasha_deco", "itasha_stripe", "itasha_stack")),
}


PRESET_NAMES = tuple(STYLE_PRESETS)


# 옛 조리법의 `family`(구성 계열 드롭다운) → 프리셋. 사선 흐름은 다크가 뼈대만
# 물려받았고 바탕까지 검게 하니 자동으로 돌린다.
LEGACY_FAMILY = {"minimal": "minimal", "graphic_bed": "floral",
                 "motorsport": "racing", "splash": "splash"}


def label(name: str | None) -> str:
    """드롭다운 이름 (현재 언어)."""
    return {
        None: msg("자동"), "auto": msg("자동"),
        "racing": msg("레이싱 스폰서"), "floral": msg("무늬·꽃"),
        "splash": msg("스플래시·찢김"), "minimal": msg("미니멀"),
        "dark": msg("다크 그래피티"),
    }.get(name, str(name))


def tip(name: str | None) -> str:
    """드롭다운 툴팁 — 무엇이 한 벌로 오는가."""
    return {
        None: msg("후보를 다 지어 점수로 고른다 — 바탕·글자·로고는 각자의 기본"),
        "auto": msg("후보를 다 지어 점수로 고른다 — 바탕·글자·로고는 각자의 기본"),
        "racing": msg("벨트 띠 3~4 + 로커 위 로고 줄 + 레이싱 번호, 레이싱 글꼴"),
        "floral": msg("파스텔 바탕 + 테마 무늬 산포, 흘림 글꼴 — 치비는 유리에 올린 보조 그림"),
        "splash": msg("덩어리 색면에 튄 물감·찢김 가장자리, 붓 글꼴"),
        "minimal": msg("색면 하나 + 워드마크, 옆면 로고 줄 없음, 리어는 전신 축소"),
        "dark": msg("검정 바탕 + 형광 사선 액센트 + 큰 그래피티 워드마크"),
    }.get(name, "")


# 드롭다운 썸네일 — 프리셋마다 한 장 (`tools/preset_thumbs.py`가 굽는다).
THUMB_DIR = Path(__file__).with_name("thumbs")


def thumb(name: str) -> Path | None:
    p = THUMB_DIR / f"{name}.png"
    return p if p.is_file() else None


def listing() -> list[dict]:
    """편집기 드롭다운이 읽는 목록 (`flsedit state`의 `style_presets`) — 자동이 첫 줄.
    `thumb`은 절대 경로 (없으면 빈 문자열) — 창이 그림으로 보여 준다."""
    def row(key: str, name: str | None) -> dict:
        t = thumb(key)
        return {"key": key, "label": label(name), "tip": tip(name),
                "thumb": str(t) if t is not None else ""}
    return [row("auto", None)] + [row(n, n) for n in PRESET_NAMES]


def resolve(name: str | None) -> StylePreset | None:
    """이름 → 프리셋. None·"auto"면 None, 모르는 이름이면 ValueError."""
    if name is None or name == "auto":
        return None
    if name not in STYLE_PRESETS:
        raise ValueError(msg("모르는 스타일 프리셋: {name!r} (있는 것: {names})",
                             name=name, names=", ".join(PRESET_NAMES)))
    return STYLE_PRESETS[name]
