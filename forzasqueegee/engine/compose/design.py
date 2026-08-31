"""구성 설계 — 사람 배치를 읽고 → 후보를 만들고 → 재서 → 고른다.

옛 고정 문법 한 벌을 대신하는 옆면 꾸밈의 머리다. 들어오는 것은
`build`가 이미 갖고 있던 것과 같다 (도안·프레임·인물 상자·그리기 판정)이고,
나가는 것은 **꾸밈 그룹 두 장**(배경·전경)에 더해 다른 면이 따라 쓸 설계
(역할 팔레트·흐름·계열)다.

## 결정성

후보 생성도 점수도 전부 결정적이다 — 같은 도안·같은 배치·같은 차면 같은
후보 목록이 같은 순서로 서고 같은 것이 이긴다. 난수는 없다 (황금각 산포·
이름 위상은 `scatter`와 같은 자다).

## 후보

계열(`families`) × 흐름(자동·뒤·앞) × 팔레트 변종(`roles`) × 베드 크기 두 단
— 도안 뜻이 계열 순서를 정하고(`rank_families`) 앞의 넷을 쓴다. 후보 하나가
꾸밈 캔버스 레이어 수십 장이라 만들고 재는 데 후보당 수십 ms다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from ...i18n import msg
from ..catalog import Catalog
from ..model import Layer, LayerPlan
from .bands import stripe_layers
from .bed import bed_layers, keyline_layers
from .echo import echo_layers
from .families import FAMILIES, Family, rank_families
from .field import CompositionField, build_field
from .intent import DesignIntent
from .look import Look
from .place import _refit_canvas
from dataclasses import replace as _replace
from .roles import RolePalette, role_palette
from .scatter import (
    DECO_FRONT_SIZE, DECO_GAP_MAX, DECO_TIER_SIZE, HALO_GROW, scatter_motifs)
from .score import ScoreCard, composite, raster_layers, score_design
from .textbudget import TextPlan, plan_tiers
from .textbuild import TextSet, build_text_sets
from .textscore import TEXT_WEIGHTS, text_parts
from .textspec import TextSpec
from .textstyle import choose_style
from .vocabulary import _RING8, edge_shapes, motif_shapes
from .roof import ROOF_DARK


# 모티프가 캔버스 끝에 딱 붙으면 기울일 때 모서리가 밖으로 나간다 — 조금 줄인다
DECO_FRAME_FILL = 0.98


# **로커 띠는 캔버스 끝까지 간다.** 캔버스가 곧 옆면 도색 폭이라(`build`의 `ds`)
# 여기서 물러난 몫이 그대로 패널 안의 곧은 단면이 된다 — 0.98로는 띠가 앞뒤로
# 8.7·4.7유닛 모자라 사이드실 끝에 사각 단면이 남았다 (Evo VIII 실측).
# 산포와 갈리는 이유: 띠는 안 기울고(rot 0), 넘친 몫은 면 마스크가 자른다.
DECO_BAND_FILL = 1.0


# 후보에 쓰는 계열 수 (순위 앞에서부터). 다섯 다 써도 되지만 뒤의 것은 거의 안 이긴다.
FAMILY_TOP = 4


# 팔레트 변종 — 계열마다 이 둘을 돈다 (넷을 다 돌면 후보가 배로 늘고 이기는 것은 같다)
VARIANTS_TRIED = ("shadow", "neutral", "primary")


@dataclass
class Design:
    family: Family
    pal: RolePalette
    fld: CompositionField
    back: list[Layer]                   # 옆면 배경 꾸밈 (프레임 좌표)
    front: list[Layer]                  # 옆면 전경 꾸밈 (인물 위)
    score: ScoreCard
    flow_rear: bool                     # 흐름이 차 뒤쪽인가
    level: float
    keyline: bool = False
    text: TextSet | None = None         # 글자 몫 (없으면 None)
    text_plan: TextPlan | None = None
    text_style: str | None = None
    trimmed: int = 0                    # 면 상한 때문에 뺀 산포·에코 장수
    notes: list[str] = field(default_factory=list)
    ranking: list[tuple[str, float]] = field(default_factory=list)

    @property
    def motif_colors(self) -> tuple[tuple[int, int, int], ...]:
        return self.pal.motif_trio

    @property
    def text_layers(self) -> list[Layer]:
        """옆면 텍스트 그룹의 레이어 (커스텀 층만 — 층 D 흉내 그림은 뺀다)."""
        if self.text is None:
            return []
        return [l for l in self.text.layers if not l.label.startswith("game")]

    def plan(self, src: LayerPlan, cat: Catalog, front: bool = False) -> LayerPlan | None:
        layers = self.front if front else self.back
        if not layers:
            return None
        # 수치는 넷째 자리에서 끊는다 — 게임 입력 스텝(이동 0.5 · 스케일 0.01 ·
        # 회전 0.1°)보다 훨씬 잘고, 부동소수 꼬리가 파일에 실려 같은 판이
        # 다른 파일로 보이는 일을 막는다
        rounded = [replace(l, x=round(l.x, 4), y=round(l.y, 4), sx=round(l.sx, 4),
                           sy=round(l.sy, 4), rot=round(l.rot % 360.0, 4))
                   for l in layers]
        return _refit_canvas(LayerPlan(source_image=src.source_image,
                                       image_size=src.image_size,
                                       units_per_px=src.units_per_px,
                                       layers=rounded), cat)


def _scatter(fld: CompositionField, fam: Family, pal: RolePalette, cat: Catalog,
             vocab: tuple[str, ...], halo: tuple[int, int, int] | None,
             over: bool, phase: float
             ) -> tuple[list[Layer], list[tuple[float, float, float, int]]]:
    """산포 — 자리는 필드가, 크기·층·간격은 `scatter_motifs`가 정한다."""
    fx0, fy0, fx1, fy1 = fld.frame_box
    ch = fld.char_h
    cx, cy = (fx0 + fx1) / 2, (fy0 + fy1) / 2
    rx = 0.5 * DECO_FRAME_FILL * (fx1 - fx0)
    ry = 0.55 * (fy1 - fy0)
    fl = fld.flow
    edge = fld.person_box[2] if fl[0] >= 0 else fld.person_box[0]
    ax = edge + fl[0] * 0.30 * fld.char_w
    ay = fld.visual_center[1] + fl[1] * 0.35 * ch
    hero = DECO_TIER_SIZE[0] * ch * fam.tier_scale / 2
    lim = max(0.0, rx - hero)
    ax = max(-lim, min(lim, ax))
    ay = max(fy0 + 0.15 * (fy1 - fy0), min(fy1 - 0.15 * (fy1 - fy0), ay))
    g = fld.grid
    # 뭉치는 자리를 **장식 구역 안으로** 옮긴다 — 인물 상자 모서리 너머가 뒷휠
    # 아치면 (레퍼런스의 인물은 대개 뒷휠 바로 앞이다) 무리의 핵이 구멍에
    # 앉아 후보가 전멸한다. 구역 안에서 그 자리에 가장 가까운 칸이 핵이다.
    if not over:
        ys_, xs_ = np.where(fld.decoration >= 0.5)
        if len(xs_):
            px_ = g.x0 + (xs_ + 0.5) * g.cell
            py_ = g.y_top - (ys_ + 0.5) * g.cell
            k_ = int(np.argmin((px_ - ax) ** 2 + (py_ - ay) ** 2))
            ax, ay = float(px_[k_]), float(py_[k_])
    ry = 0.42 * (fy1 - fy0)                       # 후보는 밴드 안에만 뜬다

    def _ok(x: float, y: float, rq: float) -> bool:
        if over:
            if g.at(fld.protected, x, y) > 0.5 or g.at(fld.drawable, x, y) < 0.5:
                return False
            return all(g.at(fld.protected, x + math.cos(t) * rq, y + math.sin(t) * rq) < 0.5
                       for t in _RING8)
        if g.at(fld.decoration, x, y) < 0.05:
            return False
        return all(g.at(fld.drawable, x + math.cos(t) * rq, y + math.sin(t) * rq) > 0.5
                   for t in _RING8)

    n = fam.front_n if over else fam.motif_n
    ref = ch * fam.tier_scale * (DECO_FRONT_SIZE if over else 1.0)
    out: list[Layer] = []
    stats: list[tuple[float, float, float, int]] = []
    if n <= 0:
        return out, stats
    for mo in scatter_motifs(center=(cx, cy), radii=(rx, ry), ref=ref, n=n,
                             vocab=vocab, cat=cat, colors=pal.motif_trio,
                             anchor_at=(ax, ay), avoid=fld.person_box, over=over,
                             place_ok=_ok, phase=phase,
                             gap=None if over else DECO_GAP_MAX,
                             pool=max(n * 6, 160)):
        if halo is not None and mo.tier <= 1 and not over:
            out.append(Layer(shape=mo.shape, x=mo.x, y=mo.y, sx=mo.half * HALO_GROW,
                             sy=mo.half * HALO_GROW, rot=mo.rot, color=halo,
                             alpha=mo.alpha, label="itasha_deco"))
        out.append(Layer(shape=mo.shape, x=mo.x, y=mo.y, sx=mo.half, sy=mo.half,
                         rot=mo.rot, color=mo.color, alpha=mo.alpha, label="itasha_deco"))
        stats.append((mo.x, mo.y, mo.size, mo.tier))
    return out, stats


def _keyline_color(pal: RolePalette) -> tuple[int, int, int]:
    """키라인 색 — 베드의 반대 명도 (짙은 판엔 밝은 테, 연한 판엔 짙은 테)."""
    b = (0.299 * pal.bed[0] + 0.587 * pal.bed[1] + 0.114 * pal.bed[2]) / 255.0
    return (250, 250, 250) if b < 0.5 else (24, 24, 28)


def _rocker(fld: CompositionField, lk: Look, cat: Catalog, car_rgb, vocab) -> list[Layer]:
    frame = _replace(lk, box=fld.frame_box, hull=None)
    return stripe_layers(frame, ROOF_DARK, cat, shapes=vocab, car=car_rgb,
                         length=DECO_BAND_FILL * (fld.frame_box[2] - fld.frame_box[0]))


def compose_design(plan: LayerPlan, lk: Look, it: DesignIntent, cat: Catalog,
                   car_rgb: tuple[int, int, int], *,
                   frame_box: tuple[float, float, float, float],
                   person_box: tuple[float, float, float, float],
                   L: np.ndarray, t: np.ndarray, frame_center: tuple[float, float],
                   u: float, rear_sign: float, drawable_at=None,
                   motif: str | None = None, halo: tuple[int, int, int] | None = None,
                   family: str | None = None, phase: float = 0.0,
                   text: TextSpec | None = None, cap: int | None = None,
                   n_person: int = 0, log=None) -> Design:
    """후보 생성 + 평가 + 선택. 되돌림은 이긴 설계다 (순위표는 `ranking`).

    `text`가 켜져 있으면 글자도 후보의 한 축이다 — 배치 후보(워드마크·로커·
    사인)마다, 그리고 "글자 없음"까지 같은 표에서 겨룬다. 층은 면에 남는 장수
    (`cap` − `n_person` − 꾸밈)와 우선순위가 정한다 (`textbudget`).
    """
    text_on = text is not None and text.active
    cap = cap or 3000
    vocab = motif_shapes(lk, cat, motif)
    edge_v = edge_shapes(lk, cat, motif)
    fx0, _f, fx1, _g = frame_box
    person_frac = (person_box[2] - person_box[0]) / max(1e-6, fx1 - fx0)
    fams = rank_families(it, lk, person_frac)
    fams = [family] if family is not None else fams[:FAMILY_TOP]
    # 흐름별 필드 — 후보가 나눠 쓴다 (필드 짓기가 후보보다 비싸다)
    fields: dict[str, CompositionField] = {}

    def _field(mode: str) -> CompositionField:
        if mode not in fields:
            flow = None if mode == "auto" else (rear_sign if mode == "rear" else -rear_sign, 0.0)
            fields[mode] = build_field(it, L, t, frame_center, u, frame_box, person_box,
                                       rear_sign, drawable_at=drawable_at, flow=flow)
        return fields[mode]

    cands: list[Design] = []
    for fname in fams:
        fam = FAMILIES[fname]
        for mode in fam.flows:
            fld = _field(mode)
            for variant in VARIANTS_TRIED:
                pal = role_palette(it, lk, car_rgb, variant)
                for level in (fam.bed_level, max(0.0, fam.bed_level - 0.25)):
                    base: list[Layer] = []
                    if fam.rocker:
                        base += _rocker(fld, lk, cat, car_rgb, edge_v)
                    base += bed_layers(fld, pal, cat, fam.bed, level,
                                       edge_shapes=edge_v, torn=fam.torn,
                                       rocker=fam.rocker)
                    sc, stats = _scatter(fld, fam, pal, cat, vocab, halo, False, phase)
                    tail: list[Layer] = list(sc)
                    if fam.echo:
                        tail += echo_layers(fld, it, pal, cat, n=max(3, fam.motif_n // 3),
                                            phase=phase)
                    front, _fs = _scatter(fld, fam, pal, cat, vocab, None, True, phase)
                    front_ras = raster_layers(front, fld, cat)
                    # 키라인은 후보의 한 축이다 — 짙은 판 위에서 실루엣이 읽히나를
                    # 점수(readability)가 가르고, 안 필요한 판에서는 장수만 먹는다
                    key_col = _keyline_color(pal)
                    keyl = keyline_layers(fld, key_col, cat)
                    # ---- 글자 — 배치 후보마다 한 벌, 그리고 "없음" ----
                    text_sets: list[TextSet | None] = [None]
                    tplan = None
                    tstyle = None
                    if text_on:
                        tstyle = choose_style(text.style, fam, it)
                        # 남는 장수: 우선순위 high면 산포·에코를 안 세고(글자가
                        # 먼저다 — 넘치면 그쪽을 뺀다), low면 절반만 준다
                        fixed = n_person + len(base) + len(keyl) + len(front) + 12
                        free = cap - fixed - (0 if text.priority == "high" else len(tail))
                        if text.priority == "low":
                            free = int(free * 0.5)
                        free = int(free * fam.text_budget)
                        tplan = plan_tiers(text, tstyle, max(0, free))
                        _b, bed_a = raster_layers([l for l in base if l.label == "itasha_bed"],
                                                  fld, cat)
                        text_sets += build_text_sets(
                            fld, pal, cat, main=text.main or "", sub=text.sub, style=tstyle,
                            plan=tplan, rocker=fam.rocker, bed_alpha=bed_a)
                    # 글자 래스터는 (배치 후보 × 이 팔레트)마다 한 번만 — 키라인·
                    # 베드 크기 변종이 같은 것을 나눠 쓴다
                    text_ras = {id(ts): raster_layers(ts.layers, fld, cat)
                                for ts in text_sets if ts is not None}
                    behind = composite(fld, pal, cat, base, [], front_raster=front_ras)["behind"] \
                        if text_on else None
                    for keyline in (False, True):
                        for ts in text_sets:
                            back = list(base)
                            tl = ts.layers if ts is not None else []
                            n_text = ts.n if ts is not None else 0
                            if keyline:
                                back += keyl
                            back_tail = list(tail)
                            # 면 상한 — 넘치면 산포·에코를 뒤에서부터 뺀다 (도안·
                            # 판·글자가 먼저다). 글자는 제 그룹이라 따로 센다.
                            trimmed = 0
                            while (n_person + len(back) + len(back_tail) + len(front) + n_text
                                   > cap - 4) and back_tail:
                                back_tail.pop()
                                trimmed += 1
                            back += back_tail
                            extra = None
                            if text_on:
                                extra = text_parts(fld, cat, ts.poses if ts else [],
                                                   tl, behind, front_alpha=front_ras[1])
                            # 점수용 합성: 글자 그룹은 꾸밈 그룹 **위**·도안 아래에 선다
                            card = score_design(fld, pal, cat, back, front,
                                                clutter_target=fam.clutter,
                                                empty_target=fam.empty_target,
                                                motifs=stats, rocker=fam.rocker,
                                                extra=extra, extra_weights=TEXT_WEIGHTS,
                                                text=text_ras.get(id(ts)),
                                                front_raster=front_ras)
                            flow_rear = (fld.flow[0] * rear_sign) > 0
                            cands.append(Design(family=fam, pal=pal, fld=fld, back=back,
                                                front=front, score=card,
                                                flow_rear=flow_rear, level=level,
                                                keyline=keyline, text=ts, text_plan=tplan,
                                                text_style=tstyle, trimmed=trimmed))
                    if fam.bed == "none":
                        break
    cands.sort(key=lambda d: -d.score.total)
    best = cands[0]
    best.ranking = [(f"{d.family.name}/{d.pal.variant}/{'rear' if d.flow_rear else 'front'}"
                     f"/{d.level:.2f}{'/key' if d.keyline else ''}"
                     + (f"/txt-{d.text.poses[0].role}-{d.text.tier_main}" if d.text else
                        ("/txt-none" if text_on else "")),
                     round(d.score.total, 3)) for d in cands[:8]]
    if text_on:
        if best.text is not None:
            p0 = best.text.poses[0]
            best.notes.append(msg(
                "텍스트: {style} 스타일 · 층 {tier} · {role} 자리 (높이 {h:.0f}유닛 · "
                "각 {rot:.0f}° · {where}) · {n:,}장{sub}",
                style=best.text.style, tier=best.text.tier_main, role=p0.role,
                h=p0.height, rot=p0.rot, where=msg("판 위") if p0.on_bed else msg("도색 위"),
                n=best.text.n,
                sub=(msg(" · 서브 층 {tier}", tier=best.text.tier_sub)
                     if best.text.tier_sub != "E" else "")))
        else:
            best.notes.append(msg("텍스트: 이긴 후보에 글자가 없다 — 글자 있는 후보가 "
                                  "가독성·어수선에서 밀렸다 (우선순위 {prio})",
                                  prio=text.priority))
        if best.text_plan is not None:
            best.notes += best.text_plan.notes
    if best.trimmed:
        best.notes.append(msg("면 상한 때문에 산포·에코 {n}장을 뺐다", n=best.trimmed))
    best.notes.append(msg(
        "구성 설계: 후보 {n}벌 중 {family} 계열 ({variant} 팔레트 · 흐름 {flow} · "
        "베드 {level:.2f}) 점수 {score:.3f} — {parts}",
        n=len(cands), family=best.family.name, variant=best.pal.variant,
        flow=msg("뒤") if best.flow_rear else msg("앞"), level=best.level,
        score=best.score.total, parts=best.score.text()))
    if len(cands) > 1:
        best.notes.append(msg("차점: {runner} ({score:.3f})",
                              runner=best.ranking[1][0], score=best.ranking[1][1]))
    best.notes.append(msg(
        "도안 뜻: 머리 {head} · 축 ({ax:.2f},{ay:.2f}) · 인상 {impr} (각 {ang:.2f}) · "
        "밀도 {dens:.2f} · 결 일관성 {coh:.2f} · 얼굴 방향 {face:+.2f}",
        head=msg("확신") if it.head_confident else (msg("어림") if it.head else msg("없음")),
        ax=best.fld.axis[0], ay=best.fld.axis[1], impr=it.impression,
        ang=it.angularity, dens=it.density, coh=it.flow_coherence, face=best.fld.face_dir))
    if log is not None:
        for name, s in best.ranking:
            log(f"    {s:.3f}  {name}")
    return best
