r"""커스텀 텍스트 도안 — 문자열을 **동봉 글꼴로 래스터해 게임 도형으로 되짓는다**.

`textvinyl`은 게임의 글꼴 비닐(한 글자 한 장)을 조판한다 — 값싸지만 글꼴이 게임
것뿐이라 필기체·붓·그래피티·레이싱 워드마크 같은 **사람이 만든 이타샤의
타이포**가 안 나온다. 이 모듈은 반대쪽이다: 오픈 라이선스 글꼴(OFL,
`catalog/fonts/`)로 문자열을 그려 마스크를 얻고, 그 마스크를 **뼈대 획**으로
읽어 막대(A_01)와 원(B_26)으로 다시 짓는다. 글자 한 자가 수 장~수십 장이 되므로
층(tier)이 장수를 조인다 (`compose.textbudget`).

## 되짓는 법

    마스크 → 세선화(`celfit.skeleton._thin`) → 곁가지 제거 → 경로 → RDP 단순화
    → 마디 사이 막대(폭 = 거리 변환) + 굽은 마디에 원(둥근 이음)

막대·원은 좌우 대칭이라 미러 면에서도 같은 도형이 서지만 **글자는 뒤집히면
안 되므로** 텍스트 그룹은 면마다 따로 짓는다 (`compose.textlayout`).

## 스타일 = 글꼴 + 처리

| 스타일 | 글꼴 (OFL) | 처리 |
|---|---|---|
| script | Great Vibes | 둥근 이음, 고른 폭 |
| brush | Kaushan Script | 폭이 살아 있는 획 (마디마다 폭을 잰다) |
| graffiti | Sedgwick Ave | 굵은 테두리·그림자 기본 |
| racing | Racing Sans One | 이탤릭 전단 |
| techno | Audiowide | 각진 이음 (원 없음) |
| minimal | Poppins Light | 가는 고른 폭 |

글꼴은 전부 SIL Open Font License 1.1이고 각 폴더에 `OFL.txt`가 같이 있다.
시스템 글꼴은 안 쓴다 (재배포 권리가 없다).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ..i18n import msg
from .catalog import Catalog
from .celfit.skeleton import _paths, _prune_spurs, _rdp, _thin
from .model import UNITS_PER_SCALE, Layer


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
    round_joins: bool = True         # 굽은 마디에 원을 놓나 (각진 글꼴은 안 놓는다)
    live_width: bool = False         # 마디마다 폭을 재나 (붓) / 획 하나에 한 폭
    skew: float = 0.0                # 전단 (레이싱 이탤릭)
    outline_default: bool = True     # `auto` 테두리
    shadow_default: bool = False     # `auto` 그림자
    outline_frac: float = 0.16       # 테두리 두께 (대문자 높이 대비)
    shadow_frac: float = 0.06        # 그림자 오프셋 (대문자 높이 대비)
    letter_case: str | None = None   # 강제 대소문자 (None = 그대로)


STYLE_RULES: dict[str, StyleRule] = {
    "script": StyleRule(round_joins=True, live_width=False, outline_default=True,
                        outline_frac=0.12),
    "brush": StyleRule(round_joins=True, live_width=True, outline_default=False,
                       shadow_default=True, shadow_frac=0.05),
    "graffiti": StyleRule(round_joins=True, live_width=True, outline_default=True,
                          shadow_default=True, outline_frac=0.22, shadow_frac=0.08),
    "racing": StyleRule(round_joins=False, live_width=False, skew=0.28,
                        outline_default=True, outline_frac=0.14),
    "techno": StyleRule(round_joins=False, live_width=False, outline_default=False,
                        shadow_default=True, shadow_frac=0.07),
    "minimal": StyleRule(round_joins=True, live_width=False, outline_default=False,
                         shadow_default=False),
}


# 래스터 대문자 높이 (px). 획 폭이 이 몫으로 잡히므로 너무 작으면 뼈대가 깨진다.
RASTER_CAP = 160


# 층별 단순화 — RDP 허용 오차(획 폭 대비)와 이음 원의 최소 꺾임각 (도).
# A: 곡선을 살린다 · B: 마디를 줄인다 · C: 굵직한 골격만.
TIER_EPS = {"A": 0.30, "B": 0.55, "C": 0.95}


TIER_JOIN_DEG = {"A": 18.0, "B": 28.0, "C": 45.0}


# 테두리·그림자 벌은 본색보다 **한 층 거친** 골격을 쓴다 — 밑에 깔리는 벌이라
# 마디가 성겨도 테는 매끈하게 읽히고, 장수는 반으로 준다.
UNDER_TIER = {"A": "B", "B": "C", "C": "C"}


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

    줄은 `\\n`으로 갈린다. 줄 정렬은 가운데다 (워드마크의 기본).
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
    pad = int(0.15 * cap) + 4
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


@dataclass
class Stroke:
    """뼈대 마디 하나 — 막대 한 장 (px 좌표, y-down)."""

    p0: tuple[float, float]
    p1: tuple[float, float]
    w: float


def _turn_deg(a, b, c) -> float:
    v1 = (b[0] - a[0], b[1] - a[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    n1, n2 = math.hypot(*v1), math.hypot(*v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    return math.degrees(math.acos(cos))


_STROKE_CACHE: dict[tuple, tuple] = {}


def text_strokes(text: str, style: str, tier: str
                 ) -> tuple["TextRaster", list["Stroke"], list[tuple[float, float, float]]]:
    """(래스터, 막대, 이음) — **캐시**. 후보 루프에서 같은 글자를 수백 번 묻는다
    (실측: 캐시 없이 후보 168벌에 212초, 캐시로 수 초)."""
    key = (text, style, tier)
    if key not in _STROKE_CACHE:
        if len(_STROKE_CACHE) > 96:
            _STROKE_CACHE.clear()
        ras = render_mask(text, style)
        strokes, joins, _w = mask_strokes(ras.mask, tier, STYLE_RULES.get(style, StyleRule()),
                                          ras.cap_px)
        _STROKE_CACHE[key] = (ras, strokes, joins)
    return _STROKE_CACHE[key]


def mask_strokes(mask: np.ndarray, tier: str = "A", rule: StyleRule | None = None,
                 cap_px: float = RASTER_CAP
                 ) -> tuple[list[Stroke], list[tuple[float, float, float]], float]:
    """마스크 → (막대들, 이음 원들 (x, y, r), 중앙 획 폭). 전부 px.

    곁가지는 **대문자 높이의 몫**으로 자른다 — 획 폭으로 자르면 가는 필기체에서
    세리프·계단 잉여가 전부 살아남아 장수가 세 배가 된다 (실측: Great Vibes
    1,530장 → 그 아래).
    """
    rule = rule or StyleRule()
    dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    sk = _thin(mask)
    if not sk.any():
        return [], [], 1.0
    wmed = float(np.median(dt[sk])) * 2.0
    sk = _prune_spurs(sk, max(2.0, wmed * 0.8, 0.07 * cap_px))
    if not sk.any():
        return [], [], wmed
    eps = TIER_EPS.get(tier, 0.55) * max(1.0, wmed)
    join_deg = TIER_JOIN_DEG.get(tier, 24.0)
    strokes: list[Stroke] = []
    joins: list[tuple[float, float, float]] = []
    seen_join: set[tuple[int, int]] = set()

    def _w_at(p) -> float:
        r, c = int(round(p[0])), int(round(p[1]))
        r = min(max(r, 0), dt.shape[0] - 1)
        c = min(max(c, 0), dt.shape[1] - 1)
        return max(1.0, 2.0 * float(dt[r, c]))

    for path, hj, tj in _paths(sk):
        pts = _rdp(path.astype(np.float64), eps)
        if len(pts) < 2:
            continue
        # 폭 — 획마다 한 값 (고른 폭) 또는 마디마다 (붓)
        if not rule.live_width:
            ws = [float(np.median([_w_at(q) for q in path[::max(1, len(path) // 12)]]))] * (len(pts) - 1)
        else:
            ws = [0.5 * (_w_at(pts[i]) + _w_at(pts[i + 1])) for i in range(len(pts) - 1)]
        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i + 1]
            strokes.append(Stroke(p0=(float(a[1]), float(a[0])), p1=(float(b[1]), float(b[0])),
                                  w=ws[i]))
        if rule.round_joins:
            for i in range(1, len(pts) - 1):
                if _turn_deg(pts[i - 1], pts[i], pts[i + 1]) >= join_deg:
                    key = (int(pts[i][0]), int(pts[i][1]))
                    if key not in seen_join:
                        seen_join.add(key)
                        joins.append((float(pts[i][1]), float(pts[i][0]),
                                      0.5 * (ws[i - 1] + ws[i]) / 2))
            # 분기점(다른 경로와 만나는 마디)에도 원 — 이음새 틈을 메운다
            for j, p in ((hj, pts[0]), (tj, pts[-1])):
                if j >= 0:
                    key = (int(p[0]), int(p[1]))
                    if key not in seen_join:
                        seen_join.add(key)
                        joins.append((float(p[1]), float(p[0]), _w_at(p) / 2))
    return strokes, joins, wmed


def strokes_to_layers(strokes: list[Stroke], joins: list[tuple[float, float, float]],
                      *, upp: float, origin: tuple[float, float], color: tuple[int, int, int],
                      cat: Catalog, grow: float = 0.0, skew: float = 0.0,
                      label: str = "text", alpha: float = 100.0,
                      extend: float = 0.5) -> list[Layer]:
    """px 획 → 캔버스 유닛 레이어. `origin`은 px 마스크에서 캔버스 (0,0)이 오는 자리.

    `grow`(유닛)는 폭을 양쪽으로 키운다 — 테두리 사본이 쓴다. `extend`는 막대
    양 끝을 폭의 몫만큼 늘려 이웃 막대와 겹치게 한다 (각진 이음의 틈 메움).
    `skew`는 x += skew·y 전단 (이탤릭) — 원점 기준이라 블록 가운데를 원점에 둔다.
    """
    ox, oy = origin
    out: list[Layer] = []
    reach = (cat.shapes[cat.circle].reach if cat.circle in cat.shapes else 1.0)

    def _xy(px: float, py: float) -> tuple[float, float]:
        x = (px - ox) * upp
        y = -(py - oy) * upp
        return x + skew * y, y

    for s in strokes:
        x0, y0 = _xy(*s.p0)
        x1, y1 = _xy(*s.p1)
        w = s.w * upp + 2 * grow
        dx, dy = x1 - x0, y1 - y0
        L = math.hypot(dx, dy)
        if L < 1e-6:
            continue
        ext = extend * w
        L2 = L + 2 * ext
        out.append(Layer(shape=cat.square, x=(x0 + x1) / 2, y=(y0 + y1) / 2,
                         sx=L2 / 2 / UNITS_PER_SCALE, sy=w / 2 / UNITS_PER_SCALE,
                         rot=math.degrees(math.atan2(dy, dx)) % 360.0,
                         color=color, alpha=alpha, label=label))
    for jx, jy, r in joins:
        x, y = _xy(jx, jy)
        rad = r * upp + grow
        out.append(Layer(shape=cat.circle, x=x, y=y,
                         sx=rad / UNITS_PER_SCALE / reach, sy=rad / UNITS_PER_SCALE / reach,
                         color=color, alpha=alpha, label=label))
    return out


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

    @property
    def n(self) -> int:
        return len(self.layers)


def build_text(text: str, style: str, height: float, cat: Catalog, *,
               tier: str = "A", fill: tuple[int, int, int] = (255, 255, 255),
               outline: tuple[int, int, int] | None = None,
               shadow: tuple[int, int, int] | None = None,
               shadow_dir: tuple[float, float] = (1.0, -1.0),
               label: str = "text") -> TextBlock:
    """문자열 → `TextBlock`. `height`는 대문자 높이(캔버스 유닛).

    벌 순서는 그림자 → 테두리 → 본색이다 (뒤가 위). 층 `A`는 셋 다, `B`는
    테두리까지, `C`는 본색만 — 색은 주면 쓰고 안 주면 그 벌을 안 짓는다 (층이
    이미 색을 걸러 준다, `compose.textbudget`).
    """
    rule = STYLE_RULES.get(style, StyleRule())
    ras, strokes, joins = text_strokes(text, style, tier)
    if not strokes:
        raise ValueError(msg("텍스트를 획으로 못 읽었다: {text!r}", text=text))
    _r, u_strokes, u_joins = text_strokes(text, style, UNDER_TIER.get(tier, "C"))
    upp = height / ras.cap_px
    x0, y0, x1, y1 = ras.box
    origin = ((x0 + x1) / 2, (y0 + y1) / 2)
    w, h = (x1 - x0) * upp, (y1 - y0) * upp
    layers: list[Layer] = []
    n_sh = n_ol = 0
    if shadow is not None and tier == "A":
        off = rule.shadow_frac * height
        dx, dy = shadow_dir
        n = math.hypot(dx, dy) or 1.0
        sh = strokes_to_layers(u_strokes, u_joins, upp=upp, origin=origin, color=shadow, cat=cat,
                               skew=rule.skew, label=label + "_shadow")
        for l in sh:
            l.x += off * dx / n
            l.y += off * dy / n
        layers += sh
        n_sh = len(sh)
    if outline is not None and tier in ("A", "B"):
        ol = strokes_to_layers(u_strokes, u_joins, upp=upp, origin=origin, color=outline, cat=cat,
                               grow=rule.outline_frac * height / 2, skew=rule.skew,
                               label=label + "_edge")
        layers += ol
        n_ol = len(ol)
    fl = strokes_to_layers(strokes, joins, upp=upp, origin=origin, color=fill, cat=cat,
                           skew=rule.skew, label=label)
    layers += fl
    if rule.skew:
        w += abs(rule.skew) * h
    return TextBlock(layers=layers, w=w, h=h, cap=height, tier=tier, style=style,
                     n_fill=len(fl), n_outline=n_ol, n_shadow=n_sh)


def estimate_layers(text: str, style: str, tier: str = "A") -> int:
    """벌 하나(본색)의 장수 어림 — 래스터·뼈대를 한 번 돌린다 (캐시 없음, 수십 ms)."""
    _r, strokes, joins = text_strokes(text, style, tier)
    return len(strokes) + len(joins)
