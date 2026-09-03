"""텍스트 조립 — 포즈 묶음 + 층 → 프레임 좌표의 레이어 (후보 하나의 글자 몫).

`design`이 후보마다 부른다: 필드·계열·팔레트가 정해진 뒤, 배치 후보
(`textlayout.layout_sets`)마다 글자 블록을 짓고 포즈로 돌려 앉힌다. 블록은
층이 정한다 — 층 D는 **게임 글꼴 글리프**(`textvinyl`, 한 글자 한 장 — 기본
엔진), 층 A·B는 도형 맞춤 커스텀 도안(`textglyph`). 둘 다 그냥 레이어라 글자
그룹 한 장(`text-<면>.json`)에 같이 실린다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..catalog import Catalog
from ..model import Layer
from .. import textglyph as tg
from .. import textvinyl as tv
from .field import CompositionField
from .roles import RolePalette
from .textbudget import TextPlan
from .textlayout import ROLES, TextPose, layout_sets, lockups, pose_mask
from .textstyle import text_colors


@dataclass
class TextSet:
    """후보 하나의 글자 몫 — 포즈들과 그 레이어 (프레임 좌표)."""

    poses: list[TextPose]
    layers: list[Layer]
    tier_main: str
    tier_sub: str
    style: str
    # 포즈마다의 **원형 블록** (원점 중심 · 기준 높이). 자리·각·크기만 바꿔
    # 다시 앉힐 때 글리프를 새로 안 지어도 되게 쥐고 있다 (`reposed`).
    protos: list = None

    @property
    def n(self) -> int:
        return len(self.layers)


# 원형 블록을 짓는 **기준 대문자 높이** (프레임 유닛). 실제 높이는 여기에
# 비례해 늘린다 — 글리프는 도형을 스케일한 것뿐이라 그 환산이 정확하다.
PROTO_H = 100.0


def _scaled(layers: list[Layer], k: float) -> list[Layer]:
    """블록을 `k`배로 (원점 중심 — 자리·크기가 다 비례한다)."""
    if abs(k - 1.0) < 1e-9:
        return list(layers)
    return [Layer(shape=l.shape, x=l.x * k, y=l.y * k, sx=l.sx * k, sy=l.sy * k,
                  rot=l.rot, skew=l.skew, color=l.color, alpha=l.alpha,
                  label=l.label, mask=l.mask) for l in layers]


def reposed(ts: "TextSet", *, dx: float = 0.0, dy: float = 0.0,
            k: float = 1.0) -> "TextSet":
    """글자 몫을 **다시 앉힌다** — 자리 `(dx, dy)` · 크기 `k`배.

    글리프를 새로 짓지 않는다: 원형 블록(`protos`)을 크기만 늘려 포즈로 돌린다.
    그래서 좌표하강이 글자 판도 다듬을 수 있다 — 옛 판은 손잡이마다 글자 수백
    장을 다시 지어야 해서 **글자가 있는 후보를 아예 안 다듬었다**.

    층 문턱(`tier_for_size`)을 넘는 크기 변화는 여기서 안 다룬다: 원형이 그
    층으로 지어졌으므로 부르는 쪽이 `k`를 문턱 안에 둔다 (`REFINE_CLAMP`).
    """
    if not ts.protos:
        return ts
    poses: list[TextPose] = []
    layers: list[Layer] = []
    for p, proto in zip(ts.poses, ts.protos):
        q = TextPose(role=p.role, text=p.text, x=p.x + dx, y=p.y + dy, rot=p.rot,
                     height=p.height * k, aspect=p.aspect, hratio=p.hratio,
                     on_bed=p.on_bed)
        poses.append(q)
        layers += _posed(_scaled(proto, q.height / PROTO_H), q)
    return TextSet(poses=poses, layers=layers, tier_main=ts.tier_main,
                   tier_sub=ts.tier_sub, style=ts.style, protos=ts.protos)


def _posed(layers: list[Layer], p: TextPose) -> list[Layer]:
    """원점 중심 블록 → 포즈 자리로 (돌리고 옮긴다)."""
    r = math.radians(p.rot)
    c, s = math.cos(r), math.sin(r)
    out = []
    for l in layers:
        x = p.x + l.x * c - l.y * s
        y = p.y + l.x * s + l.y * c
        out.append(Layer(shape=l.shape, x=x, y=y, sx=l.sx, sy=l.sy,
                         rot=(l.rot + p.rot) % 360.0, skew=l.skew, color=l.color,
                         alpha=l.alpha, label=l.label, mask=l.mask))
    return out


def _on_bed(fld: CompositionField, bed_alpha, p: TextPose) -> bool:
    if bed_alpha is None:
        return False
    m = pose_mask(fld, p)
    return bool(m.any() and float((bed_alpha[m] > 0.5).mean()) >= 0.5)


def game_text(text: str, font: str, cat: Catalog) -> str:
    """게임 글꼴이 그릴 수 있는 문자열 (없는 글자만 뺀다 — 띄어쓰기·줄바꿈은 지킨다)."""
    return "".join(ch for ch in text if ch.isspace() or tv.glyph_name(ch, font, cat))


# 두 줄 락업의 줄 간격 (대문자 높이 대비) — `textglyph._render_mask`와 같은 값
LINE_GAP = 0.25


# 그림자 오프셋 (대문자 높이 대비) — 게임 글자 도구의 그림자와 같은 자
SHADOW_FRAC = 0.06


# 단어 사이 **덧간격** (대문자 높이 대비) — 글자 틈에 더해진다.
#
# `textvinyl.FONT_SPACE`는 **게임 글자 도구가 내는 값**을 실측한 것이라 impact가
# 0이다 (그 도구는 단어 틈을 글자 틈과 같게 낸다). 우리는 글리프를 직접 앉히므로
# 조판이 우리 몫이고, 0을 그대로 쓰면 단어가 붙어 버린다 (실측: impact
# 'RIN SHIBUYA'의 N|S 틈이 다른 글자 틈과 같은 29.8유닛 — 'RINSHIBUYA'로 읽힌다).
# 0.28이면 단어 틈이 글자 틈의 두 배 남짓이 된다 (arial 23 → 51 · impact 29.8 → 57.8).
WORD_SPACE = 0.28


def font_metrics(text: str, font: str, cat: Catalog) -> tuple[float, float]:
    """게임 글꼴로 조판한 블록의 (잉크 상자 w/h, 상자 높이/대문자 높이).

    배치(`textlayout`)가 상자로 재므로 글꼴의 실제 폭이라야 한다 — 커스텀 글꼴의
    비율을 대면 게임 글자가 자리보다 넓거나 좁게 선다."""
    h0 = 100.0
    ws: list[float] = []
    hs: list[float] = []
    for ln in text.split("\n"):
        ok = game_text(ln, font, cat)
        if not ok.strip():
            continue
        m = tv.text_metrics(ok, font=font, height=h0, space=WORD_SPACE, kern=True, cat=cat)
        ws.append(m["w"])
        hs.append(m["h"])
    if not ws:
        return 1.0, 1.0
    h = sum(hs) + LINE_GAP * h0 * (len(hs) - 1)
    return max(ws) / max(1e-6, h), max(1.0, h / h0)


def font_block(text: str, font: str, height: float, cat: Catalog, *,
               fill: tuple[int, int, int], outline: tuple[int, int, int] | None = None,
               shadow: tuple[int, int, int] | None = None,
               shadow_dir: tuple[float, float] = (1.0, -1.0),
               label: str = "text") -> list[Layer]:
    """문자열 → 게임 글꼴 글리프 레이어 (원점 = 잉크 상자 가운데, 캔버스 유닛).

    줄은 `\\n`으로 갈리고 가운데 정렬로 쌓인다. 벌 순서는 그림자 → 테두리 → 본색
    (뒤가 위). 테두리는 **같은 글자를 원 위 여덟 자리에** 깐다 — 확대 사본 1벌은
    넓은 글자에서 테가 두꺼워지고 좁은 글자에서 얇아지며, 대각 넷만으로는 가는
    글꼴에서 테가 유령 사본 넷으로 갈린다
    (`textvinyl.outline_offsets`, 미리보기·실전 경로와 같은 자).
    """
    lines: list[tuple[list[Layer], float]] = []
    for ln in text.split("\n"):
        ok = game_text(ln, font, cat)
        if not ok.strip():
            continue
        ls, box = tv.text_layers(ok, font=font, height=height, color=fill, cat=cat,
                                 space=WORD_SPACE, kern=True, label=label)
        lines.append((ls, box.h))
    if not lines:
        return []
    total = sum(h for _l, h in lines) + LINE_GAP * height * (len(lines) - 1)
    body: list[Layer] = []
    y = total / 2
    for ls, h in lines:
        cy = y - h / 2
        for l in ls:
            l.y += cy
        body += ls
        y -= h + LINE_GAP * height
    out: list[Layer] = []
    if shadow is not None:
        dx, dy = shadow_dir
        n = math.hypot(dx, dy) or 1.0
        off = SHADOW_FRAC * height
        out += [Layer(shape=l.shape, x=l.x + off * dx / n, y=l.y + off * dy / n, sx=l.sx,
                      sy=l.sy, rot=l.rot, color=shadow, label=label + "_shadow")
                for l in body]
    if outline is not None:
        for ox, oy in tv.outline_offsets(tv.OUTLINE_SHIFT * height):
            out += [Layer(shape=l.shape, x=l.x + ox, y=l.y + oy, sx=l.sx, sy=l.sy,
                          rot=l.rot, color=outline, label=label + "_edge")
                    for l in body]
    return out + body


# 대문자 높이(프레임 유닛)가 이 아래면 도형 맞춤을 한 층 낮춘다 — 작은 글자에서
# 곡선 마디는 안 보이고 장수만 먹는다 (실측: 로커 위 25유닛 글자에 층 A 836장).
SMALL_B = 46.0


# 이 아래면 도형 맞춤이 아예 값을 못 한다 (수백 장이 점으로 뭉친다) — 게임 글꼴로.
SMALL_D = 28.0


def tier_for_size(tier: str, height: float) -> str:
    if tier in ("D", "E"):
        return tier
    if height < SMALL_D:
        return "D"
    if height < SMALL_B and tier == "A":
        return "B"
    return tier


def ix_for_size(ix: int, height: float) -> int:
    """사다리 칸의 크기 하강 — 층 한 단 = 두 칸 (`tier_for_size`와 같은 문턱)."""
    if height < SMALL_B:
        return max(ix, tg.TIER_INDEX["B"])
    return ix


def pose_proto(p: TextPose, pal: RolePalette, cat: Catalog, *, style: str,
               plan: TextPlan) -> list[Layer]:
    """포즈 하나의 **원형 블록** — 원점 중심 · 기준 높이 `PROTO_H` (캐시된다).

    같은 (글자·스타일·층·색·벌)이면 한 번만 짓는다. 후보 루프는 팔레트 변종과
    배치 후보를 돌며 같은 이름을 수십 번 조판하므로 이 캐시가 곧 속도이고,
    캐시가 있어야 좌표하강이 글자를 흔들 수 있다 (`reposed`).

    캐시는 **카탈로그에 붙여** 둔다 — 한 판에 카탈로그가 하나라 수명이 맞고,
    모듈 전역에 두면 카탈로그를 갈아 끼운 도구에서 남의 글리프가 나온다.
    """
    is_sub = p.role == "sub"
    tier = tier_for_size(plan.tier_sub if is_sub else plan.tier_main, p.height)
    if tier == "E":
        return []
    fill, edge, shadow = text_colors(pal, p.on_bed, sub=is_sub)
    outline = edge if plan.outline else None
    shad = shadow if (plan.shadow and not is_sub) else None
    ix = ix_for_size(plan.ix_sub if is_sub else plan.ix_main, p.height)
    label = "text_sub" if is_sub else ("text_number" if p.role == "number" else "text")
    key = (p.text, style, tier, ix, plan.font, fill, outline, shad, label)
    cache = cat.__dict__.setdefault("_text_proto", {})
    got = cache.get(key)
    if got is None:
        if tier == "D":
            got = font_block(p.text, plan.font, PROTO_H, cat, fill=fill,
                             outline=outline, shadow=shad, label=label)
        else:
            got = tg.build_text(p.text, style, PROTO_H, cat, tier=tier, ix=ix,
                                fill=fill, outline=outline, shadow=shad,
                                label=label).layers
        cache[key] = got
    return got


def pose_layers(p: TextPose, pal: RolePalette, cat: Catalog, *, style: str,
                plan: TextPlan) -> list[Layer]:
    """포즈 하나 → 레이어. 색·벌은 층과 `on_bed`가 정한다."""
    proto = pose_proto(p, pal, cat, style=style, plan=plan)
    return _posed(_scaled(proto, p.height / PROTO_H), p) if proto else []


def mirrored_set(ts: TextSet, pal: RolePalette, cat: Catalog, plan: TextPlan) -> TextSet:
    """반대편 옆면의 글자 몫 — 자리는 거울, 글자는 바로 읽힌다."""
    poses = [p.mirrored() for p in ts.poses]
    layers: list[Layer] = []
    protos: list[list[Layer]] = []
    for p in poses:
        proto = pose_proto(p, pal, cat, style=ts.style, plan=plan)
        protos.append(proto)
        if proto:
            layers += _posed(_scaled(proto, p.height / PROTO_H), p)
    return TextSet(poses=poses, layers=layers, protos=protos,
                   tier_main=ts.tier_main, tier_sub=ts.tier_sub, style=ts.style)


def text_box(text: str, style: str, plan: TextPlan, cat: Catalog, sub: bool = False
             ) -> tuple[float, float]:
    """배치가 재는 블록 비율 — 그 층이 실제로 그릴 글꼴의 것."""
    tier = plan.tier_sub if sub else plan.tier_main
    if tier == "D":
        return font_metrics(text, plan.font, cat)
    r = tg.render_mask(text, style)
    return r.aspect, r.hratio


def build_text_sets(fld: CompositionField, pal: RolePalette, cat: Catalog, *,
                    main: str, sub: str | None, style: str, plan: TextPlan,
                    rocker: bool, bed_alpha=None,
                    roles: tuple[str, ...] = ROLES, scale: float = 1.0) -> list[TextSet]:
    """배치 후보마다 `TextSet` 하나. 층이 E면 빈 목록. 이름의 줄 나눔(`lockups`)
    마다 배치를 따로 낸다 — 두 줄 락업은 포즈만 다른 게 아니라 글자 블록이 다르다.
    `scale`은 워드마크 상한 배율 (프리셋 — `presets`)."""
    if plan.tier_main == "E":
        return []
    box_sub = text_box(sub, style, plan, cat, sub=True) if sub else (1.0, 1.0)
    sets: list[TextSet] = []
    cands: list[list[TextPose]] = []
    for text in lockups(main):
        cands += layout_sets(fld, text, sub, text_box(text, style, plan, cat), box_sub,
                             rocker, roles, scale)
    for poses in cands:
        layers: list[Layer] = []
        protos: list[list[Layer]] = []
        keep: list[TextPose] = []
        used_sub = False
        for p in poses:
            p.on_bed = _on_bed(fld, bed_alpha, p)
            proto = pose_proto(p, pal, cat, style=style, plan=plan)
            if not proto:
                continue
            layers += _posed(_scaled(proto, p.height / PROTO_H), p)
            protos.append(proto)
            keep.append(p)
            used_sub = used_sub or p.role == "sub"
        if layers:
            sets.append(TextSet(poses=keep, layers=layers, protos=protos,
                                tier_main=tier_for_size(plan.tier_main, poses[0].height),
                                tier_sub=(tier_for_size(plan.tier_sub, poses[-1].height)
                                          if used_sub else "E"),
                                style=style))
    return sets
