"""텍스트 조립 — 포즈 묶음 + 층 → 프레임 좌표의 레이어 (후보 하나의 글자 몫).

`design`이 후보마다 부른다: 필드·계열·팔레트가 정해진 뒤, 배치 후보
(`textlayout.layout_sets`)마다 글자 블록을 짓고(`textglyph.build_text`) 포즈로
돌려 앉힌다. 층 D(게임 글꼴)는 레이어 대신 **면 글자 명세**가 되지만 점수를
매기려면 그림이 필요하므로 `textvinyl` 글리프로 같은 자리에 흉내를 낸다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..catalog import Catalog
from ..model import Layer
from .. import textglyph as tg
from .. import textvinyl as tv
from .field import CompositionField
from .roles import RolePalette
from .textbudget import TextPlan
from .textlayout import TextPose, layout_sets, pose_mask
from .textstyle import GAME_FONT, text_colors


@dataclass
class TextSet:
    """후보 하나의 글자 몫 — 포즈들과 그 레이어 (프레임 좌표)."""

    poses: list[TextPose]
    layers: list[Layer]                 # 커스텀 층의 레이어 (층 D면 흉내 그림)
    tier_main: str
    tier_sub: str
    style: str
    game_jobs: list[dict] = field(default_factory=list)   # 층 D의 면 글자 명세 (프레임 좌표)

    @property
    def n(self) -> int:
        """커스텀 층의 장수 (층 D의 흉내 그림은 안 센다)."""
        if self.tier_main == "D":
            return 0
        return len([l for l in self.layers if not l.label.startswith("game")])


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


def _game_layers(text: str, font: str, height: float, fill, outline, cat: Catalog
                 ) -> list[Layer]:
    """층 D의 흉내 그림 — 게임 글꼴에 없는 글자는 뺀다 (실행도 그렇게 넣는다)."""
    ok = "".join(ch for ch in text if ch.isspace() or tv.glyph_name(ch, font, cat))
    if not ok.strip():
        return []
    layers, _box = tv.text_layers(ok, font=font, height=height, color=fill,
                                  outline=outline, cat=cat, label="game")
    return layers


def game_text(text: str, font: str, cat: Catalog) -> str:
    """게임 글꼴이 그릴 수 있는 문자열 (없는 글자만 뺀다 — 띄어쓰기는 지킨다)."""
    return "".join(ch for ch in text if ch.isspace() or tv.glyph_name(ch, font, cat))


# 대문자 높이(프레임 유닛)가 이 아래면 층을 한 단 낮춘다 — 작은 글자에서 곡선
# 마디는 안 보이고 장수만 먹는다 (실측: 로커 위 25유닛 글자에 층 A 836장).
SMALL_B = 46.0


SMALL_C = 28.0


# 이 아래면 커스텀 도안이 아예 값을 못 한다 (수백 장이 점으로 뭉친다) — 게임
# 글꼴(허락할 때) 아니면 뺀다.
SMALL_D = 18.0


def tier_for_size(tier: str, height: float, allow_game: bool = True) -> str:
    if tier in ("D", "E"):
        return tier
    if height < SMALL_D:
        return "D" if allow_game else "E"
    if height < SMALL_C:
        return "C"
    if height < SMALL_B and tier == "A":
        return "B"
    return tier


def pose_layers(p: TextPose, pal: RolePalette, cat: Catalog, *, style: str,
                plan: TextPlan) -> tuple[list[Layer], dict | None]:
    """포즈 하나 → (레이어, 층 D면 면 글자 명세). 색·벌은 층과 `on_bed`가 정한다."""
    is_sub = p.role == "sub"
    tier = tier_for_size(plan.tier_sub if is_sub else plan.tier_main, p.height, plan.allow_game)
    if tier == "E":
        return [], None
    fill, edge, shadow = text_colors(pal, p.on_bed, sub=is_sub)
    outline = edge if (plan.outline and tier in ("A", "B", "D")) else None
    shad = shadow if (plan.shadow and tier in ("A", "D") and not is_sub) else None
    font = GAME_FONT.get(style, "sans")
    if tier == "D":
        job = {"text": game_text(p.text, font, cat), "font": font, "x": p.x, "y": p.y,
               "rot": p.rot, "height": p.height, "color": list(fill),
               **({"outline": list(outline)} if outline else {}),
               **({"shadow": list(shad)} if shad else {})}
        return _posed(_game_layers(p.text, font, p.height, fill, outline, cat), p), job
    blk = tg.build_text(p.text, style, p.height, cat, tier=tier, fill=fill,
                        outline=outline, shadow=shad,
                        label="text_sub" if is_sub else "text")
    return _posed(blk.layers, p), None


def mirrored_set(ts: TextSet, pal: RolePalette, cat: Catalog, plan: TextPlan) -> TextSet:
    """반대편 옆면의 글자 몫 — 자리는 거울, 글자는 바로 읽힌다."""
    poses = [p.mirrored() for p in ts.poses]
    layers: list[Layer] = []
    jobs: list[dict] = []
    for p in poses:
        ls, job = pose_layers(p, pal, cat, style=ts.style, plan=plan)
        layers += ls
        if job:
            jobs.append(job)
    return TextSet(poses=poses, layers=layers, tier_main=ts.tier_main,
                   tier_sub=ts.tier_sub, style=ts.style, game_jobs=jobs)


def build_text_sets(fld: CompositionField, pal: RolePalette, cat: Catalog, *,
                    main: str, sub: str | None, style: str, plan: TextPlan,
                    rocker: bool, bed_alpha=None,
                    roles: tuple[str, ...] = ("wordmark", "rocker", "signature")
                    ) -> list[TextSet]:
    """배치 후보마다 `TextSet` 하나. 층이 E면 빈 목록."""
    if plan.tier_main == "E":
        return []
    aspect_main = tg.render_mask(main, style).aspect
    aspect_sub = tg.render_mask(sub, style).aspect if sub else 1.0
    sets: list[TextSet] = []
    for poses in layout_sets(fld, main, sub, aspect_main, aspect_sub, rocker, roles):
        layers: list[Layer] = []
        jobs: list[dict] = []
        used_sub = False
        for p in poses:
            p.on_bed = _on_bed(fld, bed_alpha, p)
            ls, job = pose_layers(p, pal, cat, style=style, plan=plan)
            if not ls and job is None:
                continue
            layers += ls
            if job:
                jobs.append(job)
            used_sub = used_sub or p.role == "sub"
        if layers or jobs:
            sets.append(TextSet(poses=[p for p in poses if (p.role != "sub" or used_sub)],
                                layers=layers,
                                tier_main=tier_for_size(plan.tier_main, poses[0].height,
                                                        plan.allow_game),
                                tier_sub=(tier_for_size(plan.tier_sub, poses[-1].height,
                                                        plan.allow_game)
                                          if used_sub else "E"),
                                style=style, game_jobs=jobs))
    return sets
