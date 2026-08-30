r"""텍스트 비닐 — 문자열을 **게임의 글꼴 비닐**로 조판해 도안 하나로 만든다.

이타샤에는 글자가 들어간다 (캐릭터 이름·서클명·"痛車"·차 번호). 그림으로
글자를 흉내내면 획마다 도형이 붙어 수십~수백 장이 드는데, FH6은 **글꼴 자체를
비닐로** 갖고 있어 한 글자가 한 장이다. 그래서 이 모듈은 AI도 래스터화도 쓰지
않는다 — 카탈로그에서 글리프를 꺼내 **한 글자 한 레이어**로 앉힌다.

## 글꼴은 카탈로그에 이미 있다

`catalog/vinyl_catalog.json`의 1,480종 가운데 **960종이 글꼴**이다 (24그룹 × 40).
한 그룹이 한 글꼴이고 대문자 그룹과 소문자 그룹이 짝을 이룬다. 그룹 안 순서는
실측으로 확정했다 (2026-08-17, `tools/../scratchpad/font_check.py`로 라스터를
라벨과 나란히 놓고 육안 대조):

    대문자 그룹: 01~26 = A~Z · 27~35 = 1~9 · 36 = 0 · 37~40 = ! ? @ &
    소문자 그룹: 01~26 = a~z · 27~ = $ £ ¥ € æ ^ ß @ # + % ; : /

**숫자는 대문자 그룹에만 있다** — 소문자 그룹은 그 자리에 기호가 들어 있다.

## 글자 크기·자리는 native 상자에서 나온다

카탈로그 loops는 축별 ±1 정규화라 원래 크기를 잃었지만, `Catalog`가 로드할 때
게임 에셋의 BBox로 **되돌려 놓는다** (`catalog.py`의 native 보정). 그래서 글리프
상자가 실제 설계 크기·자리이고, 그걸로 조판할 수 있다:

- 글자 폭 = 상자 폭 × 64 × 스케일 (`model.UNITS_PER_SCALE`)
- **글자 자리는 상자 왼쪽 끝을 펜에 맞춘다** — 설계 원점이 글리프 가운데인지
  베이스라인인지 가정하지 않으므로 글꼴이 바뀌어도 조판이 안 흔들린다.
- 세로는 원점을 그대로 둔다 (같은 그룹 글리프는 베이스라인을 공유한다).

## 테두리는 앞뒤 배치다

DC 가이드의 눈동자 기법과 같다: 같은 글리프를 조금 크게 뒤에 깔면 테두리가 된다.
글자당 2장이 되지만 이타샤 타이포는 테두리가 있어야 차체 색에 안 묻힌다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..i18n import msg
from .catalog import Catalog, default_catalog_path
from .model import UNITS_PER_SCALE, Layer, LayerPlan

# 그룹 안 인덱스(1-기반) → 글자
UPPER_ORDER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890!?@&"
LOWER_ORDER = "abcdefghijklmnopqrstuvwxyz$£¥€æ^ß@#+%;:/"

# 글꼴 이름 → (대문자 그룹, 소문자 그룹). 이름은 라스터를 본 눈대중 이름표다.
FONTS: dict[str, tuple[str, str]] = {
    "sans": ("S", "T"),            # 굵은 산세리프 — 기본
    "sans2": ("SS", "TT"),
    "condensed": ("OO", "PP"),     # 좁고 굵다 (레이싱 넘버 느낌)
    "gothic": ("KK", "LL"),        # 블랙레터
    "script": ("MM", "NN"),        # 손글씨 흘림
    "italic": ("K", "L"),          # 두꺼운 이탤릭
    "italic_light": ("M", "N"),
    "serif": ("QQ", "RR"),
    "round": ("O", "P"),
    "slab": ("Q", "R"),
    "narrow": ("UU", "VV"),
    "wide": ("WW", "XX"),
}
DEFAULT_FONT = "sans"
# 글자 사이 간격 (대문자 높이의 몫). **게임이 쓰는 값을 실측으로 맞췄다** —
# 이 값이 맞아야 구성 설계가 "이 이름이 문짝에 들어가나"를 옳게 계산하고,
# 중심 배치의 앵커(첫 글리프 원점) 역산이 안 밀린다. 글꼴마다 다르다:
#
# - `sans` 0.23 (2026-08-17, 'HIH' 스케일 1.0 = 315유닛 · 2026-08-18 재검 0.2271)
# - `condensed` 0.298 (2026-08-18, 'HIH' 스케일 1.5,
#   전체 문구 검증에서 스케일 0.5까지 선형이 확인됐다). 0.23으로 두면 예측 폭이
#   **과소**해 배너·리어 이름이 상자를 넘쳐 A필러·패널 경계에서 잘린다.
# - `script` 0.122 (2026-08-20, 'HIH' 스케일 1.4 — 같은 판의 sans 검산 0.2271로
#   방법이 서 있음을 확인했다). **기본값의 절반**이다: 필기체는 글자가 서로
#   기울어 파고들어 전진폭이 짧다. 0.23으로 두면 예측 폭이 **과대**해 백드롭
#   타이포가 실제보다 크게 계산돼 필요 없이 줄어들고, 중심 앵커 역산이 오른쪽으로
#   밀려 이름이 인물 뒤에서 치우친다.
#
# 공백 전진폭도 실측이다: condensed 'HATSUNE MIKU'의 틈 합이 정확히
# 10×tracking이었다 — **단어 틈이 글자 틈과 같다.** 우리 펜은 글자 뒤에 이미
# tracking을 더하므로 space 몫은 0이다 (sans의 0.34는 실측 전 값 — condensed만
# 쓰는 이타샤 경로에는 안 들어온다).
#
# **script의 공백은 못 쟀다** (측정판이 'H H'에서 죽었다 — 2026-08-20). 기본값
# 0.34를 쓴다: 0을 주면 tracking이 0.122로 좁아 단어가 붙어 버린다 (뒷유리
# 사인이 'HatsuneMiku'로 찍혔다 — 미리보기 판정). 백드롭은 단어를 **줄로 쪼개**
# 앉히므로 이 값을 안 쓴다 — 영향은 한 줄짜리 사인·배너뿐.
GAME_TRACKING = 0.23
FONT_TRACKING: dict[str, float] = {"sans": 0.23, "condensed": 0.298,
                                   "script": 0.122}
FONT_SPACE: dict[str, float] = {"sans": 0.34, "condensed": 0.0}

# 글자 테두리 한 벌 = **같은 크기 4벌을 대각 오프셋**으로 깐 것 (대문자 높이의
# 몫). 블록 확대 1벌은 이동량이 블록 폭에 비례해 긴 문구의 끝 글자가 분리된
# 조각으로 찍힌다 (2026-08-18 캡처). 여기 두는 이유는 **쓰는 쪽이 둘**이라서다 —
# 실전 경로(`auto.gametext`)와 미리보기(`engine.preview`)가 같은 값으로 그려야
# 테두리를 넣을지 뺄지를 미리보기로 판단할 수 있다 (2026-08-20: 미리보기만
# 1.12배 확대 사본이라 백드롭 테두리가 흰 덩어리로 보였다).
OUTLINE_SHIFT = 0.06


def font_tracking(font: str) -> float:
    return FONT_TRACKING.get(font, GAME_TRACKING)


def font_space(font: str) -> float:
    return FONT_SPACE.get(font, 0.34)


# 글꼴 그룹 이름 전부 (대·소문자 짝) — 렌더가 "이건 글꼴이다"를 물을 때 쓴다
ALL_FONT_GROUPS: frozenset[str] = frozenset(
    g for pair in FONTS.values() for g in pair)


def is_font(shape: str) -> bool:
    """그 도형이 글꼴 글리프인가 (이름 앞머리로 판단)."""
    return shape.split("_")[0] in ALL_FONT_GROUPS


def font_groups(font: str) -> tuple[str, str]:
    """글꼴 이름 → (대문자 그룹, 소문자 그룹). `"S/T"`처럼 직접 줄 수도 있다."""
    if font in FONTS:
        return FONTS[font]
    if "/" in font:
        a, b = font.split("/", 1)
        return a.strip(), b.strip()
    raise ValueError(msg("모르는 글꼴: {font} (아는 것: {known})",
                         font=font, known=", ".join(sorted(FONTS))))


def has_em_box(cat: Catalog, group: str) -> bool:
    """그 그룹이 em 상자 표식을 갖고 있나 = **글리프들이 기준선을 공유하나**.

    표식이 있으면 40자가 같은 상자에 정규화돼 있어 y를 0으로 두면 기준선이 맞는다.
    표식이 없는 그룹은 글리프마다 제 잉크 상자로 정규화돼 있어(실측: `L`만
    그렇다 — native 36종) 기준선을 알 길이 없다. 그런 그룹은 **쓰지 않는다** —
    소문자를 대문자로 올려 짝의 대문자 그룹으로 조판한다.
    """
    return bool(_group_marks(cat, group))


def glyph_name(ch: str, font: str = DEFAULT_FONT, cat: Catalog | None = None) -> str | None:
    """글자 → 카탈로그 도형 이름. 글꼴에 없는 글자면 None.

    소문자 그룹이 기준선을 공유하지 않으면(`has_em_box`) **대문자로 올려** 준다.
    """
    up, low = font_groups(font)
    i = UPPER_ORDER.find(ch)
    if i >= 0:
        return f"{up}_{i + 1:02d}"
    i = LOWER_ORDER.find(ch)
    if i >= 0 and ch.isalpha():
        if cat is not None and not has_em_box(cat, low):
            j = UPPER_ORDER.find(ch.upper())
            return f"{up}_{j + 1:02d}" if j >= 0 else None
        return f"{low}_{i + 1:02d}"
    return None


def supported(text: str, font: str = DEFAULT_FONT,
              cat: Catalog | None = None) -> list[str]:
    """조판할 수 없는 글자 목록 (공백·개행은 뺀다)."""
    return sorted({c for c in text
                   if not c.isspace() and glyph_name(c, font, cat) is None})


@dataclass
class Box:
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def w(self) -> float:
        return self.x1 - self.x0

    @property
    def h(self) -> float:
        return self.y1 - self.y0


# em 상자 표식 컷: 글리프마다 **알파 0인 삼각형 4개**가 ±1 네 귀퉁이에 박혀 있다.
# 게임은 안 그리지만(그래서 글자가 깨끗하게 나온다) 기하에는 있어서, 상자를
# 그대로 쓰면 **모든 글자의 상자가 같은 em 상자**가 되어 조판이 성냥갑처럼 벌어진다.
# 근거: `alpha_area`가 글자면적/(글자+귀퉁이)와 맞는다 — 'I' 0.324/0.368 = 0.88
# (카탈로그 0.8851), 'H' 0.755/0.799 = 0.945 (0.9472). 2026-08-17 확인.
# 표식 판정은 **글꼴 전체에서 되풀이되는 고리**를 찾는 것이다. 한 글꼴의 40자에
# 똑같은 상자로 들어 있는 작은 고리는 글자가 아니라 표식이다 — 자리·점 수로
# 가르려 했더니 글꼴마다 샜다 (획이 em 상자 밖으로 나가는 굵은 이탤릭, 표식보다
# 작은 글자인 ':' 같은 것). 되풀이 판정은 그 둘 다에 안 흔들린다.
MARK_MIN_FRAC = 0.8     # 그룹의 이 몫 이상에서 같은 상자로 나오면 표식
MARK_AREA_FRAC = 0.25   # 그룹 최대 고리 면적의 이 몫보다 작아야 표식
_MARKS: dict[tuple[int, str], set] = {}


def _bbox_key(loop: np.ndarray) -> tuple:
    lo, hi = loop.min(axis=0), loop.max(axis=0)
    return tuple(np.round(np.concatenate([lo, hi]), 2))


def _group_marks(cat: Catalog, group: str) -> set:
    """그 글꼴에서 표식으로 판정된 고리 상자들 (그룹 단위로 한 번 계산)."""
    key = (id(cat), group)
    if key in _MARKS:
        return _MARKS[key]
    names = [f"{group}_{i:02d}" for i in range(1, 41)]
    names = [n for n in names if n in cat.shapes]
    seen: dict[tuple, int] = {}
    area: dict[tuple, float] = {}
    big = 0.0
    for n in names:
        for l in cat[n].loops:
            k = _bbox_key(l)
            seen[k] = seen.get(k, 0) + 1
            a = abs(_loop_area(l))
            area[k] = a
            big = max(big, a)
    marks = {k for k, c in seen.items()
             if c >= MARK_MIN_FRAC * max(1, len(names))
             and area[k] < MARK_AREA_FRAC * max(1e-9, big)}
    _MARKS[key] = marks
    return marks


def ink_loops(cat: Catalog, name: str) -> list[np.ndarray]:
    """실제로 그려지는 고리만 (em 상자 표식을 뺀다)."""
    loops = list(cat[name].loops)
    if len(loops) <= 1:
        return loops
    marks = _group_marks(cat, name.split("_")[0])
    keep = [l for l in loops if _bbox_key(l) not in marks]
    return keep or loops


def _loop_area(loop: np.ndarray) -> float:
    x, y = loop[:, 0], loop[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def glyph_box(cat: Catalog, name: str) -> Box:
    """글리프의 **잉크** 상자 (정규화 좌표, y-up) — em 상자가 아니다."""
    pts = np.concatenate(ink_loops(cat, name), axis=0)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    return Box(float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1]))


def cap_height(cat: Catalog, font: str = DEFAULT_FONT) -> float:
    """그 글꼴의 대문자 높이 (정규화). 조판 스케일의 기준자 — 글자열이 바뀌어도
    같은 크기가 나오게 하려고 **문자열이 아니라 글꼴**에서 기준을 잡는다."""
    up, _ = font_groups(font)
    return glyph_box(cat, f"{up}_08").h        # 'H'


def text_layers(text: str, *, font: str = DEFAULT_FONT, height: float = 180.0,
                color: tuple[int, int, int] = (255, 255, 255),
                outline: tuple[int, int, int] | None = None,
                outline_grow: float = 0.14, tracking: float | None = None,
                space: float | None = None, cat: Catalog | None = None,
                label: str = "text") -> tuple[list[Layer], Box]:
    """문자열 → (레이어 목록, 캔버스 유닛 상자). 원점은 글자 블록의 가운데다.

    `height`는 **대문자 높이**(캔버스 유닛)다. `tracking`·`space`는 그 높이의 몫 —
    안 주면 글꼴별 실측값이다. `outline`을 주면 글자마다 뒤에 조금 큰 사본을
    깔아 테두리를 만든다.
    """
    cat = cat or Catalog(default_catalog_path())
    tracking = font_tracking(font) if tracking is None else tracking
    space = font_space(font) if space is None else space
    bad = supported(text, font, cat)
    if bad:
        raise ValueError(msg("이 글꼴에 없는 글자: {chars} "
                             "(글꼴 {font} — A~Z a~z 0~9 ! ? @ & 가 든다)",
                             chars="".join(bad), font=font))
    s = height / (cap_height(cat, font) * UNITS_PER_SCALE)
    layers: list[Layer] = []
    pen = 0.0
    x0 = y0 = 1e9
    x1 = y1 = -1e9
    for ch in text:
        if ch.isspace():
            pen += space * height
            continue
        name = glyph_name(ch, font, cat)
        assert name is not None
        gb = glyph_box(cat, name)
        gx = pen - gb.x0 * UNITS_PER_SCALE * s          # 상자 왼쪽 끝을 펜에 맞춘다
        if outline is not None:
            # 테두리 사본: 같은 **글리프 가운데**를 중심으로 키운다
            og = s * (1.0 + outline_grow)
            cx = gx + (gb.x0 + gb.w / 2) * UNITS_PER_SCALE * s
            cy = (gb.y0 + gb.h / 2) * UNITS_PER_SCALE * s
            layers.append(Layer(
                shape=name, x=cx - (gb.x0 + gb.w / 2) * UNITS_PER_SCALE * og,
                y=cy - (gb.y0 + gb.h / 2) * UNITS_PER_SCALE * og,
                sx=og, sy=og, color=outline, label=label + "_edge"))
        layers.append(Layer(shape=name, x=gx, y=0.0, sx=s, sy=s,
                            color=color, label=label))
        gx0 = pen
        gx1 = pen + gb.w * UNITS_PER_SCALE * s
        x0, x1 = min(x0, gx0), max(x1, gx1)
        y0 = min(y0, gb.y0 * UNITS_PER_SCALE * s)
        y1 = max(y1, gb.y1 * UNITS_PER_SCALE * s)
        pen = gx1 + tracking * height
    if not layers:
        raise ValueError(msg("빈 문자열"))
    if outline is not None:                         # 테두리가 상자를 넓힌다
        pad = outline_grow / 2 * height
        x0, y0, x1, y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    for l in layers:                                # 블록 가운데를 원점으로
        l.x -= cx
        l.y -= cy
    return layers, Box(x0 - cx, y0 - cy, x1 - cx, y1 - cy)


def preview(plan: LayerPlan, cat: Catalog | None = None, bg: int = 90,
            scale: float = 1.0) -> np.ndarray:
    """텍스트 플랜을 그림으로 (RGB) — **em 상자 표식을 빼고** 그린다.

    공용 렌더러(`engine.render`)는 고리마다 알파를 모르므로 표식 삼각형까지
    칠한다. 글자만 보려는 미리보기라 여기서 잉크 고리만 그린다.
    """
    import cv2

    cat = cat or Catalog(default_catalog_path())
    w, h = plan.image_size
    W, H = int(w * scale), int(h * scale)
    img = np.full((H, W, 3), bg, np.uint8)
    for l in plan.layers:
        rot = np.radians(l.rot)
        c, s = np.cos(rot), np.sin(rot)
        for loop in ink_loops(cat, l.shape):
            p = loop * np.array([l.sx, l.sy], np.float32) * UNITS_PER_SCALE
            p = p @ np.array([[c, s], [-s, c]], np.float32)
            p = p + np.array([l.x, l.y], np.float32)
            px = (p[:, 0] / plan.units_per_px + w / 2) * scale
            py = (h / 2 - p[:, 1] / plan.units_per_px) * scale
            cv2.fillPoly(img, [np.stack([px, py], 1).astype(np.int32)], l.rgb())
    return img


def text_metrics(text: str, *, font: str = DEFAULT_FONT, height: float = 180.0,
                 tracking: float | None = None, space: float | None = None,
                 cat: Catalog | None = None) -> dict[str, float]:
    """조판했을 때의 **크기와 기준점 오프셋** (캔버스 유닛). 게임 텍스트 배치용.

    반환: `w`·`h` = 잉크 상자 크기, `cx`·`cy` = **펜 원점에서 잉크 중심까지**.
    게임의 텍스트 덩어리는 **첫 글리프의 설계 원점**을 붙잡으므로(2026-08-17 실측:
    스케일 1.0→1.5에서 상자가 왼쪽 22%·오른쪽 78%로 자랐다), 중심을 (X, Y)에
    두려면 게임 좌표를 `X - cx`, `Y - cy`로 준다.
    """
    cat = cat or Catalog(default_catalog_path())
    tracking = font_tracking(font) if tracking is None else tracking
    space = font_space(font) if space is None else space
    s = height / (cap_height(cat, font) * UNITS_PER_SCALE)
    pen = 0.0
    x0 = y0 = 1e9
    x1 = y1 = -1e9
    for ch in text:
        if ch.isspace():
            pen += space * height
            continue
        name = glyph_name(ch, font, cat)
        if name is None:
            continue
        gb = glyph_box(cat, name)
        x0 = min(x0, pen)
        x1 = max(x1, pen + gb.w * UNITS_PER_SCALE * s)
        y0 = min(y0, gb.y0 * UNITS_PER_SCALE * s)
        y1 = max(y1, gb.y1 * UNITS_PER_SCALE * s)
        pen += gb.w * UNITS_PER_SCALE * s + tracking * height
    if x1 < x0:
        raise ValueError(msg("빈 문자열"))
    # 펜 원점 기준: 첫 글리프의 왼쪽 잉크는 그 글리프 상자의 x0만큼 밀려 있다
    first = next((glyph_name(c, font, cat) for c in text if not c.isspace()), None)
    off = glyph_box(cat, first).x0 * UNITS_PER_SCALE * s if first else 0.0
    return {"w": x1 - x0, "h": y1 - y0, "scale": s,
            "cx": off + (x0 + x1) / 2, "cy": (y0 + y1) / 2}


def text_plan(text: str, **kw) -> LayerPlan:
    """문자열 → `LayerPlan` (캔버스 유닛 = px, 상자에 여백 8%).

    이 플랜은 다른 도안과 똑같이 쓰인다 — 주입·창 조작·이타샤 배치가 전부
    그대로 돈다 (`units_per_px=1.0`이라 캔버스 유닛이 그림 px와 같다).
    """
    layers, box = text_layers(text, **kw)
    pad = 0.08 * max(box.w, box.h)
    w = int(round(box.w + 2 * pad))
    h = int(round(box.h + 2 * pad))
    plan = LayerPlan(source_image=f"text:{text}", image_size=(max(8, w), max(8, h)),
                     units_per_px=1.0, layers=layers)
    return plan
