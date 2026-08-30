r"""커스텀 텍스트 도안 — 문자열을 **동봉 글꼴로 래스터해 게임 도형으로 되짓는다**.

`textvinyl`은 게임의 글꼴 비닐(한 글자 한 장)을 조판한다 — 값싸지만 글꼴이 게임
것뿐이라 필기체·붓·그래피티·레이싱 워드마크 같은 **사람이 만든 이타샤의
타이포**가 안 나온다. 이 모듈은 반대쪽이다: 오픈 라이선스 글꼴(OFL,
`catalog/fonts/`)로 문자열을 그려 마스크를 얻고, 그 마스크를 잉크 안에 내접하는
막대·원·삼각형으로 덮는다 (`textfit`). 글자 한 자가 수 장~수십 장이 되므로
정책 사다리(`textfit.LADDER`)가 장수를 조인다 — 예산에 드는 것 중 가장 고운 판.

## 벌 (본색 · 테두리 · 그림자)

본색은 고운 정책으로 잉크를 그대로 덮는다. 테두리는 **본색과 같은 도형을
두께만큼 키운 사본**을 뒤에 깐다 — 테가 본색을 그대로 따라가 한 몸으로
읽힌다. 다만 사본 전부를 깔지 않고, 보이는 테(부풀린 실루엣 − 본색)에 제
몫이 있는 것만 남긴다 (`textfit.cover`). 그림자는 본색 사본을 밀어 놓고 위
벌(테두리·본색)에 안 가려지는 자리에 몫이 있는 것만 남긴다.

막대·원·삼각형은 좌우 대칭이라 미러 면에서도 같은 도형이 서지만 **글자는
뒤집히면 안 되므로** 텍스트 그룹은 면마다 따로 짓는다 (`compose.textlayout`).

## 스타일 = 글꼴 + 처리

| 스타일 | 글꼴 (OFL) | 처리 |
|---|---|---|
| script | Great Vibes | 가는 테두리 기본 |
| brush | Kaushan Script | 그림자 기본 |
| graffiti | Sedgwick Ave | 굵은 테두리·그림자 기본 |
| racing | Racing Sans One | 테두리 (이탤릭은 글꼴 자체) |
| techno | Audiowide | 그림자 기본 |
| minimal | Poppins Light | 본색만 |

글꼴은 전부 SIL Open Font License 1.1이고 각 폴더에 `OFL.txt`가 같이 있다.
시스템 글꼴은 안 쓴다 (재배포 권리가 없다).

글리프 윤곽(TrueType 곡선)을 직접 읽는 길은 안 택했다 — fontTools가 고정 의존
목록에 없고, 커닝·합성 글리프까지 스스로 풀어야 하는데 정작 도형을 앉히는
자는 픽셀 정답(내접 반지름·카운터)이라 래스터가 그 일을 더 잘한다. 래스터
높이를 240px로 올려도 장수는 안 줄고 시간만 2~3배였다 (실측, `RASTER_CAP`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..i18n import msg
from .catalog import Catalog
from .model import Layer
from . import textfit as tf


STYLES = ("script", "brush", "graffiti", "racing", "techno", "minimal")


STYLE_FONTS: dict[str, str] = {
    "script": "greatvibes/GreatVibes-Regular.ttf",
    "brush": "kaushanscript/KaushanScript-Regular.ttf",
    "graffiti": "sedgwickave/SedgwickAve-Regular.ttf",
    "racing": "racingsansone/RacingSansOne-Regular.ttf",
    "techno": "audiowide/Audiowide-Regular.ttf",
    "minimal": "poppins/Poppins-Light.ttf",
}


@dataclass(frozen=True)
class StyleRule:
    outline_default: bool = True     # `auto` 테두리
    shadow_default: bool = False     # `auto` 그림자
    outline_frac: float = 0.16       # 테두리 두께 (대문자 높이 대비)
    shadow_frac: float = 0.06        # 그림자 오프셋 (대문자 높이 대비)
    letter_case: str | None = None   # 강제 대소문자 (None = 그대로)


STYLE_RULES: dict[str, StyleRule] = {
    "script": StyleRule(outline_default=True, outline_frac=0.12),
    "brush": StyleRule(outline_default=False, shadow_default=True, shadow_frac=0.05),
    "graffiti": StyleRule(outline_default=True, shadow_default=True, outline_frac=0.22,
                          shadow_frac=0.08),
    "racing": StyleRule(outline_default=True, outline_frac=0.14),
    "techno": StyleRule(outline_default=False, shadow_default=True, shadow_frac=0.07),
    "minimal": StyleRule(outline_default=False, shadow_default=False),
}


# 래스터 대문자 높이 (px). 획 폭이 이 몫으로 잡히므로 너무 작으면 뼈대가 깨진다.
RASTER_CAP = 160


# 층 이름 → 사다리 칸 (`textfit.LADDER`). 층은 바깥(예산·기록)이 쓰는 이름이고
# 실제 판은 칸이 정한다 — 예산 탐색은 칸 단위로 움직인다.
TIER_INDEX = {"A": 1, "B": 3, "C": 5}


# 밑벌 도형이 보이는 자리에서 새로 덮어야 하는 최소 픽셀 (대문자 높이 대비 제곱)
UNDER_GAIN = 0.012


def tier_of(ix: int) -> str:
    return "A" if ix <= 2 else ("B" if ix <= 4 else "C")


def fonts_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "catalog" / "fonts"


def font_path(style: str) -> Path:
    if style not in STYLE_FONTS:
        raise ValueError(msg("모르는 텍스트 스타일: {style} (있는 것: {styles})",
                             style=style, styles=", ".join(STYLES)))
    return fonts_dir() / STYLE_FONTS[style]


def fonts_available() -> bool:
    return all((fonts_dir() / rel).is_file() for rel in STYLE_FONTS.values())


@dataclass
class TextRaster:
    mask: np.ndarray = field(repr=False)     # (H,W) bool
    cap_px: float                            # 대문자 높이 (px)
    lines: list[str]
    # 잉크 상자 (px, 원점은 마스크 왼쪽 위)
    box: tuple[int, int, int, int]

    @property
    def aspect(self) -> float:
        x0, y0, x1, y1 = self.box
        return (x1 - x0) / max(1, y1 - y0)

    @property
    def hratio(self) -> float:
        """잉크 상자 높이 / 대문자 높이 — 두 줄·디센더가 있으면 1을 넘는다. 배치는
        상자로 재고 크기는 대문자 높이로 말하므로 둘을 잇는 수다."""
        _x0, y0, _x1, y1 = self.box
        return max(1.0, (y1 - y0) / max(1.0, self.cap_px))


_MASK_CACHE: dict[tuple, "TextRaster"] = {}


def render_mask(text: str, style: str, cap_px: int = RASTER_CAP,
                line_gap: float = 0.25) -> TextRaster:
    """`_render_mask`의 캐시 — 후보 루프가 같은 문자열을 되풀이 묻는다."""
    key = (text, style, cap_px, line_gap)
    if key not in _MASK_CACHE:
        if len(_MASK_CACHE) > 64:
            _MASK_CACHE.clear()
        _MASK_CACHE[key] = _render_mask(text, style, cap_px, line_gap)
    return _MASK_CACHE[key]


def _render_mask(text: str, style: str, cap_px: int = RASTER_CAP,
                 line_gap: float = 0.25) -> TextRaster:
    """문자열(줄바꿈 포함) → 마스크. 공백·대소문자·구두점은 그대로다.

    줄은 `\\n`으로 갈린다. 줄 정렬은 가운데다 (워드마크의 기본). 여백은 테두리·
    그림자가 부풀어도 안 잘리게 대문자 높이의 0.3배다.
    """
    from PIL import Image, ImageDraw, ImageFont

    lines = [ln if ln.strip() else " " for ln in text.split("\n")] or [" "]
    font = ImageFont.truetype(str(font_path(style)), cap_px * 2)
    # 대문자 높이를 재서 요청한 cap_px에 맞춘다 (글꼴마다 em 대비 대문자 높이가 다르다)
    hb = font.getbbox("H")
    real_cap = max(1, hb[3] - hb[1])
    size = max(8, int(round(cap_px * 2 * cap_px / real_cap)))
    font = ImageFont.truetype(str(font_path(style)), size)
    hb = font.getbbox("H")
    cap = hb[3] - hb[1]
    gap = int(round(line_gap * cap))
    boxes = [font.getbbox(ln) for ln in lines]
    widths = [b[2] - b[0] for b in boxes]
    heights = [b[3] - b[1] for b in boxes]
    pad = int(0.30 * cap) + 4
    W = max(widths) + 2 * pad
    H = sum(heights) + gap * (len(lines) - 1) + 2 * pad
    im = Image.new("L", (max(8, W), max(8, H)), 0)
    dr = ImageDraw.Draw(im)
    y = pad
    for ln, b, w, h in zip(lines, boxes, widths, heights):
        x = pad + (max(widths) - w) // 2
        dr.text((x - b[0], y - b[1]), ln, fill=255, font=font)
        y += h + gap
    m = np.array(im) > 127
    ys, xs = np.where(m)
    if len(xs) == 0:
        raise ValueError(msg("텍스트에 잉크가 없다: {text!r}", text=text))
    return TextRaster(mask=m, cap_px=float(cap), lines=lines,
                      box=(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))


# ---------------------------------------------------------------- 맞춤 (캐시)

_FIT_CACHE: dict[tuple, tf.Fit] = {}
_CAT: dict[str, Catalog] = {}


def _catalog() -> Catalog:
    """맞춤이 쓰는 카탈로그 — 도형은 A_01·A_04·원뿐이라 기본 카탈로그면 된다."""
    if "cat" not in _CAT:
        from .catalog import default_catalog_path
        _CAT["cat"] = Catalog(default_catalog_path())
    return _CAT["cat"]


def _cached(key: tuple, make) -> tf.Fit:
    if key not in _FIT_CACHE:
        if len(_FIT_CACHE) > 256:
            _FIT_CACHE.clear()
        _FIT_CACHE[key] = make()
    return _FIT_CACHE[key]


def fit_fill(text: str, style: str, ix: int) -> tf.Fit:
    """본색 벌 — 사다리 `ix` 칸의 정책으로 잉크를 덮은 판 (캐시)."""
    ix = max(0, min(len(tf.LADDER) - 1, ix))

    def make():
        ras = render_mask(text, style)
        return tf.fit_mask(ras.mask, ras.cap_px, _catalog(), tf.LADDER[ix])
    return _cached((text, style, ix, "fill"), make)


def _grow_px(style: str, cap_px: float) -> float:
    return STYLE_RULES.get(style, StyleRule()).outline_frac * cap_px / 2


def _shift_px(style: str, cap_px: float, shadow_dir: tuple[float, float]) -> tuple[int, int]:
    """그림자 오프셋 (px, y-down). `shadow_dir`은 캔버스(y-up) 방향."""
    off = STYLE_RULES.get(style, StyleRule()).shadow_frac * cap_px
    dx, dy = shadow_dir
    n = math.hypot(dx, dy) or 1.0
    return int(round(off * dx / n)), int(round(-off * dy / n))


def _shifted(m: np.ndarray, sx: int, sy: int) -> np.ndarray:
    out = np.zeros_like(m)
    H, W = m.shape
    ys0, ys1 = max(0, sy), min(H, H + sy)
    xs0, xs1 = max(0, sx), min(W, W + sx)
    out[ys0:ys1, xs0:xs1] = m[ys0 - sy:ys1 - sy, xs0 - sx:xs1 - sx]
    return out


def fit_outline(text: str, style: str, ix: int) -> tf.Fit:
    """테두리 벌 — 본색 도형을 두께만큼 키운 사본 중 **보이는 테에 몫이 있는** 것."""
    def make():
        ras = render_mask(text, style)
        g = _grow_px(style, ras.cap_px)
        cat = _catalog()
        fill = fit_fill(text, style, ix)
        big = [tf.grown(p, g) for p in fill.prims]
        cov = tf.raster(fill.prims, ras.mask.shape, cat)
        ring = tf.raster(big, ras.mask.shape, cat) & ~cov
        keep = tf.cover(big, ring, cat, UNDER_GAIN * ras.cap_px ** 2)
        return tf.Fit(keep, fill.w, policy=fill.policy)
    return _cached((text, style, ix, "outline"), make)


def fit_shadow(text: str, style: str, ix: int, outline: bool,
               shadow_dir: tuple[float, float]) -> tf.Fit:
    """그림자 벌 — 본색 사본을 밀어 놓고, 위 벌(테두리·본색)에 안 가려지는 자리에
    몫이 있는 것만. 도형은 그림자 좌표계(밀기 전)로 둔다 — 오프셋은 레이어가 진다."""
    def make():
        ras = render_mask(text, style)
        sx, sy = _shift_px(style, ras.cap_px, shadow_dir)
        cat = _catalog()
        fill = fit_fill(text, style, ix)
        above = tf.raster(fill.prims, ras.mask.shape, cat)
        if outline:
            above |= tf.raster(fit_outline(text, style, ix).prims, ras.mask.shape, cat)
        visible = above & ~_shifted(above, -sx, -sy)   # 그림자 좌표계에서 본, 위 벌 밖
        keep = tf.cover(fill.prims, visible, cat, UNDER_GAIN * ras.cap_px ** 2)
        return tf.Fit(keep, fill.w, policy=fill.policy)
    return _cached((text, style, ix, outline, shadow_dir, "shadow"), make)


def count_layers(text: str, style: str, ix: int, outline: bool, shadow: bool,
                 shadow_dir: tuple[float, float] = (1.0, -1.0)) -> int:
    """칸 `ix`에서 서는 장수 — 본색 + (켜면) 테두리 + 그림자."""
    n = fit_fill(text, style, ix).n
    if outline:
        n += fit_outline(text, style, ix).n
    if shadow:
        n += fit_shadow(text, style, ix, outline, shadow_dir).n
    return n


def estimate_layers(text: str, style: str, tier: str = "A") -> int:
    """벌 하나(본색)의 장수 — 층 이름으로 (캐시)."""
    return fit_fill(text, style, TIER_INDEX.get(tier, 1)).n


@dataclass
class TextChoice:
    """예산이 고른 판 — 사다리 칸과 벌."""

    ix: int
    outline: bool
    shadow: bool
    n: int

    @property
    def tier(self) -> str:
        return tier_of(self.ix)


def plan_for_budget(text: str, style: str, budget: int, outline: bool, shadow: bool,
                    ix_min: int = 0) -> TextChoice | None:
    """예산 `budget`장에 드는 **가장 고운** 판. 벌은 그림자 → 테두리 순으로 뺀다.

    사다리 칸이 거칠어질수록 장수는 대체로 줄지만 단조롭지는 않다 (잔여 패스가
    거친 칸에서 더 많이 붙기도 한다) — 그래서 칸을 다 재고 드는 것 중 고운
    것을 고른다. 벌을 빼는 것은 칸을 거칠게 하는 것보다 나중이다 (테두리 없는
    고운 글자보다 테두리 있는 거친 글자가 읽힌다는 레퍼런스 문법).
    """
    for ol, sh in ((outline, shadow), (outline, False), (False, False)):
        if (ol, sh) != (outline, shadow) and not (outline or shadow):
            break
        for ix in range(ix_min, len(tf.LADDER)):
            n = count_layers(text, style, ix, ol, sh)
            if n <= budget:
                return TextChoice(ix=ix, outline=ol, shadow=sh, n=n)
    return None


@dataclass
class TextBlock:
    """지은 텍스트 도안 한 덩이 — 원점은 잉크 상자 가운데, 캔버스 유닛."""

    layers: list[Layer]
    w: float
    h: float
    cap: float                       # 대문자 높이 (유닛)
    tier: str
    style: str
    n_fill: int
    n_outline: int
    n_shadow: int
    ix: int = 1
    iou: float = 0.0                 # 본색 벌의 잉크 IoU (품질 기록)

    @property
    def n(self) -> int:
        return len(self.layers)


def build_text(text: str, style: str, height: float, cat: Catalog, *,
               tier: str = "A", ix: int | None = None,
               fill: tuple[int, int, int] = (255, 255, 255),
               outline: tuple[int, int, int] | None = None,
               shadow: tuple[int, int, int] | None = None,
               shadow_dir: tuple[float, float] = (1.0, -1.0),
               label: str = "text") -> TextBlock:
    """문자열 → `TextBlock`. `height`는 대문자 높이(캔버스 유닛).

    벌 순서는 그림자 → 테두리 → 본색이다 (뒤가 위). 색은 주면 쓰고 안 주면 그
    벌을 안 짓는다 — 어느 벌을 켤지는 예산이 정한다 (`compose.textbudget`).
    `ix`(사다리 칸)를 주면 층 이름보다 우선한다.
    """
    ix = TIER_INDEX.get(tier, 1) if ix is None else ix
    ras = render_mask(text, style)
    ff = fit_fill(text, style, ix)
    if not ff.prims:
        raise ValueError(msg("텍스트를 획으로 못 읽었다: {text!r}", text=text))
    upp = height / ras.cap_px
    x0, y0, x1, y1 = ras.box
    origin = ((x0 + x1) / 2, (y0 + y1) / 2)
    w, h = (x1 - x0) * upp, (y1 - y0) * upp
    layers: list[Layer] = []
    n_sh = n_ol = 0
    if shadow is not None:
        sx, sy = _shift_px(style, ras.cap_px, shadow_dir)
        fs = fit_shadow(text, style, ix, outline is not None, shadow_dir)
        sh = tf.to_layers(fs.prims, upp=upp, origin=origin, color=shadow,
                          label=label + "_shadow", dx=sx * upp, dy=-sy * upp)
        layers += sh
        n_sh = len(sh)
    if outline is not None:
        fo = fit_outline(text, style, ix)
        ol = tf.to_layers(fo.prims, upp=upp, origin=origin, color=outline,
                          label=label + "_edge")
        layers += ol
        n_ol = len(ol)
    fl = tf.to_layers(ff.prims, upp=upp, origin=origin, color=fill, label=label)
    layers += fl
    return TextBlock(layers=layers, w=w, h=h, cap=height, tier=tier_of(ix), style=style,
                     n_fill=len(fl), n_outline=n_ol, n_shadow=n_sh, ix=ix, iou=ff.iou)
