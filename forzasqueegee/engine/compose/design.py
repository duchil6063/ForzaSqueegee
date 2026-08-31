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
from ..model import Layer, LayerPlan, rnd
from .bands import _teeth, stripe_layers
from .bed import bed_layers, keyline_layers
from .echo import echo_layers
from .families import FAMILIES, Family, rank_families
from .field import CompositionField, build_field
from .graph import Rel, derive
from .macro import macro_layers, plan as macro_plan
from .intent import DesignIntent
from .look import Look
from .place import _refit_canvas
from dataclasses import replace as _replace
from .roles import RolePalette, role_palette
from .scatter import (
    DECO_FRONT_SIZE, DECO_GAP_MAX, DECO_TIER_SIZE, HALO_GROW, rhythm_motifs,
    scatter_motifs)
from .score import ScoreCard, _de, composite, raster_layers, score_design
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


# 예산이 모자랄 때 **버리는 순서** — 디자인 역할의 역순이다. 판(`itasha_bed`)은
# 여기 없다: 큰 구도는 마지막까지 남아야 장수를 줄여도 구성이 유지된다.
# 옛 자는 산포·에코만 뺐고, 그러고도 넘치면 `build`가 꾸밈 그룹을 **통째로**
# 버렸다 — 도안이 2,983장인 판 셋이 그래서 꾸밈 0장이 됐다.
TRIM_ORDER = ("itasha_echo", "itasha_deco", "itasha_keyline", "itasha_stripe")


# 모티프가 **제 뒤 배경에서 읽히는** 최소 색차 (Lab). 이 아래면 조각이 판에
# 묻어 없는 것과 같다 — 흰 판 위 흰 물감(07번: 아홉 중 다섯) · 흰 띠 위 흰 별
# (11번: 열여섯 중 넷)이 실측이다.
MOTIF_DE_MIN = 18.0


# 리듬의 최대형 조각이 **차체 밴드 높이**에서 차지할 수 있는 몫. 크기 자는
# 인물이지만(`DECO_TIER_SIZE`) 그 자는 넓은 옆면에서 잰 것이라 좁은 밴드에
# 그대로 쓰면 첫 조각이 밴드를 가로막는다 (`scatter.DECO_HERO_CAP`과 같은 사정).
RHYTHM_HERO_CAP = 0.62


# 리듬이 자리 검사를 이만큼도 못 통과하면 **옛 산포로 물러난다** — 곡선 하나는
# 휠아치 구멍이나 좁은 필드에 걸리면 통째로 죽을 수 있고, 그때 무리가 아예
# 없는 것보다는 뿌린 것이라도 있는 편이 낫다.
RHYTHM_MIN_KEEP = 0.34


# 후보에 쓰는 계열 수 (순위 앞에서부터). 다섯 다 써도 되지만 뒤의 것은 거의 안 이긴다.
FAMILY_TOP = 4


# 팔레트 변종 — 계열마다 이 둘을 돈다 (넷을 다 돌면 후보가 배로 늘고 이기는 것은 같다)
VARIANTS_TRIED = ("shadow", "neutral", "primary")


@dataclass(frozen=True)
class Tweak:
    """미세 조정 손잡이 — 이긴 후보를 좌표하강으로 다듬는다 (`_refine`).

    전부 0/1이면 손대기 전과 같다. 단위: 각은 도, `bed_dy`는 인물 높이 몫,
    `anchor_dx`는 인물 폭 몫, `motif_k`는 배수, `bed_w`는 색면 폭의 배수 편차다.
    색면의 **길이는 손잡이가 아니다** — 큰 색면은 늘 프레임 밖까지 나간다
    (`macro` — 끝을 자르는 것은 도형이 아니라 차다).
    """

    bed_rot: float = 0.0
    bed_dy: float = 0.0
    bed_w: float = 0.0
    anchor_dx: float = 0.0
    motif_k: float = 1.0


# 좌표하강이 도는 축과 걸음 (순서가 곧 결정성이다 — 같은 점수면 원래 값이 이긴다).
# 축마다 4걸음 × 4축 × 2패스 = 32벌.
REFINE_STEPS: tuple[tuple[str, tuple[float, ...]], ...] = (
    ("bed_rot", (-6.0, -3.0, 3.0, 6.0)),
    ("bed_dy", (-0.10, -0.05, 0.05, 0.10)),
    ("bed_w", (-0.22, -0.11, 0.11, 0.22)),
    ("anchor_dx", (-0.24, -0.12, 0.12, 0.24)),
    ("motif_k", (-0.16, -0.08, 0.08, 0.16)),
)


# 배수 손잡이의 상·하한 (여러 패스가 같은 쪽으로 걸어도 판이 뒤집히지 않게).
REFINE_CLAMP = {"motif_k": (0.70, 1.30),
                "bed_rot": (-12.0, 12.0), "bed_dy": (-0.20, 0.20),
                "bed_w": (-0.44, 0.44),
                "anchor_dx": (-0.48, 0.48)}


# 다듬을 상위 후보 수 · 좌표하강 패스 수.
REFINE_TOP = 2
REFINE_PASSES = 2


# **단계 A에서 살려 보내는 매크로 기하 수** (`compose_design`의 빔).
# 계열 × 흐름 × 어휘 짝 × 크기를 전수로 돌면 후보가 사백을 넘어 한 판에 이십
# 초가 넘는다. 큰 색면이 정해지기 전에는 산포·에코·글자가 순위를 거의 안 바꾸므로
# (둘 다 색면 위에 얹히는 잔 요소다) 기하를 먼저 여섯으로 추린다 — 여섯이면
# 계열 넷이 적어도 하나씩은 살고 이긴 계열은 변종 둘을 데려간다.
BEAM_MACRO = 6


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
    tweak: "Tweak" = field(default_factory=lambda: Tweak())
    macro: tuple[str, str] = ("ribbon", "ribbon")   # 이긴 매크로 어휘 짝
    notes: list[str] = field(default_factory=list)
    ranking: list[tuple[str, float]] = field(default_factory=list)

    @property
    def motif_colors(self) -> tuple[tuple[int, int, int], ...]:
        return self.pal.motif_trio

    @property
    def text_layers(self) -> list[Layer]:
        """옆면 텍스트 그룹의 레이어 (없으면 빈 목록)."""
        return list(self.text.layers) if self.text is not None else []

    def plan(self, src: LayerPlan, cat: Catalog, front: bool = False) -> LayerPlan | None:
        layers = self.front if front else self.back
        if not layers:
            return None
        # 수치는 넷째 자리에서 끊는다 — 게임 입력 스텝(이동 0.5 · 스케일 0.01 ·
        # 회전 0.1°)보다 훨씬 잘고, 부동소수 꼬리가 파일에 실려 같은 판이
        # 다른 파일로 보이는 일을 막는다
        rounded = [replace(l, x=rnd(l.x, 4), y=rnd(l.y, 4), sx=rnd(l.sx, 4),
                           sy=rnd(l.sy, 4), rot=rnd(l.rot % 360.0, 4))
                   for l in layers]
        return _refit_canvas(LayerPlan(source_image=src.source_image,
                                       image_size=src.image_size,
                                       units_per_px=src.units_per_px,
                                       layers=rounded), cat)


def _scatter(fld: CompositionField, fam: Family, pal: RolePalette, cat: Catalog,
             vocab: tuple[str, ...], halo: tuple[int, int, int] | None,
             over: bool, phase: float, anchor_dx: float = 0.0,
             angularity: float = 0.5
             ) -> tuple[list[Layer], list[tuple[float, float, float, int]]]:
    """무리 — 자리는 필드가, 크기·간격·꺾임은 리듬 곡선이 정한다 (`rhythm`)."""
    fx0, fy0, fx1, fy1 = fld.frame_box
    ch = fld.char_h
    cx, cy = (fx0 + fx1) / 2, (fy0 + fy1) / 2
    rx = 0.5 * DECO_FRAME_FILL * (fx1 - fx0)
    ry = 0.55 * (fy1 - fy0)
    fl = fld.flow
    edge = fld.person_box[2] if fl[0] >= 0 else fld.person_box[0]
    ax = edge + fl[0] * (0.30 + anchor_dx) * fld.char_w
    ay = fld.visual_center[1] + fl[1] * (0.35 + anchor_dx) * ch
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
    # ---- 배경 무리는 **리듬**이 놓는다 (`rhythm`) ----
    # 뭉치는 자리에서 흐름 쪽으로 곡선 하나를 걸으며 큰 것 → 잔것으로 잦아든다.
    # 황금각 산포는 자리가 크기와 무관해 "뿌린 것"으로 읽혔다. 전경 벌은 몇 장이
    # 인물을 스치는 것이 전부라 리듬이 없고, 옛 산포가 그 자를 그대로 쥔다.
    mos: list = []
    if not over:
        fdir = fld.flow
        # 갈 수 있는 거리 — 뭉치는 자리에서 흐름 쪽 프레임 변까지
        reach = abs((fx1 if fdir[0] >= 0 else fx0) - ax) / max(0.35, abs(fdir[0]))
        mos = rhythm_motifs(
            origin=(ax, ay), direction=fdir, reach=max(1.0, reach), ref=ref, n=n,
            vocab=vocab, cat=cat, colors=pal.motif_trio, avoid=fld.person_box,
            place_ok=_ok, phase=phase, angularity=angularity,
            strands=2 if n >= 12 else 1,
            size_max=RHYTHM_HERO_CAP * (fy1 - fy0))
    # 리듬이 자리 검사에 다 걸리는 판(휠아치 위에 핵이 앉은 옆면·좁은 필드)에서는
    # 옛 산포로 물러난다 — 황금각은 결정적 폴백으로 남는다
    if over or len(mos) < max(2, int(RHYTHM_MIN_KEEP * n)):
        mos = scatter_motifs(center=(cx, cy), radii=(rx, ry), ref=ref, n=n,
                             vocab=vocab, cat=cat, colors=pal.motif_trio,
                             anchor_at=(ax, ay), avoid=fld.person_box, over=over,
                             place_ok=_ok, phase=phase,
                             gap=None if over else DECO_GAP_MAX,
                             pool=max(n * 6, 160))
    for mo in mos:
        if halo is not None and mo.tier <= 1 and not over:
            out.append(Layer(shape=mo.shape, x=mo.x, y=mo.y, sx=mo.half * HALO_GROW,
                             sy=mo.half * HALO_GROW, rot=mo.rot, color=halo,
                             alpha=mo.alpha, label="itasha_deco"))
        out.append(Layer(shape=mo.shape, x=mo.x, y=mo.y, sx=mo.half, sy=mo.half,
                         rot=mo.rot, color=mo.color, alpha=mo.alpha, label="itasha_deco"))
        stats.append((mo.x, mo.y, mo.size, mo.tier))
    return out, stats


def _fit_cap(layers: list[Layer], room: int) -> tuple[list[Layer], int]:
    """`room`장에 맞게 **역할이 낮은 것부터** 뺀 목록과 뺀 장수.

    같은 역할 안에서는 뒤에서부터 뺀다 (산포·에코는 뒤가 잔것이다). 판만 남는
    자리까지 가면 그 이상은 안 뺀다 — 넘치는 판은 `build`가 잡는다.
    """
    n = len(layers) - max(0, room)
    if n <= 0:
        return layers, 0
    drop: set[int] = set()
    for label in TRIM_ORDER:
        if len(drop) >= n:
            break
        for i in range(len(layers) - 1, -1, -1):
            if len(drop) >= n:
                break
            if i not in drop and layers[i].label == label:
                drop.add(i)
    return [l for i, l in enumerate(layers) if i not in drop], len(drop)


def _readable_motifs(layers: list[Layer], fld: CompositionField, pal: RolePalette,
                     cat: Catalog, back: list[Layer]) -> list[Layer]:
    """조각의 색을 **제 뒤에 실제로 깔린 것**에서 읽히게 고른다.

    색은 베이스 도색을 보고 골라 놓는데(`palette.readable_on`), 조각이 판 위에
    앉으면 그 자가 안 맞는다 — 흰 판 위의 흰 물감은 없는 것과 같다 (실측: 옆면
    11장 146조각 중 열하나가 ΔE 18 아래, 그 아홉이 어두운 차 둘에 몰렸다).
    역할 팔레트의 세 색을 **원래 순서대로** 보고, 지금 색이 묻으면 그 자리에서
    가장 잘 읽히는 색으로 바꾼다 (순서를 지키므로 무리의 색 리듬은 남는다).
    """
    trio = pal.motif_trio
    # 바꿔 낄 색에는 **그림자·무채**도 넣는다 — 어두운 차의 무채 팔레트는 셋이
    # 전부 근백이라(07·11) 셋 안에서 고르면 흰 판 위의 흰 조각을 못 구한다.
    pool = tuple(dict.fromkeys(trio + (pal.shadow, pal.dark)))
    brgb, balpha = raster_layers(back, fld, cat)
    g = fld.grid
    out: list[Layer] = []
    for l in layers:
        if tuple(l.color) not in {tuple(c) for c in trio}:
            out.append(l)                        # 후광·무채 조각은 제 색이 있다
            continue
        c, r = g.to_cell(l.x, l.y)
        bg = pal.base
        if 0 <= c < g.cols and 0 <= r < g.rows and balpha[r, c] > 0.5:
            bg = tuple(int(v) for v in brgb[r, c])
        if _de(l.color, bg) >= MOTIF_DE_MIN:
            out.append(l)
            continue
        alt = max(pool, key=lambda t: _de(t, bg))
        out.append(_replace(l, color=alt) if _de(alt, bg) > _de(l.color, bg) else l)
    return out


def _keyline_color(pal: RolePalette) -> tuple[int, int, int]:
    """키라인 색 — 베드의 반대 명도 (짙은 판엔 밝은 테, 연한 판엔 짙은 테)."""
    b = (0.299 * pal.bed[0] + 0.587 * pal.bed[1] + 0.114 * pal.bed[2]) / 255.0
    return (250, 250, 250) if b < 0.5 else (24, 24, 28)


def _macro_colors(pal: RolePalette) -> dict[str, tuple[int, int, int]]:
    """매크로 명세의 색 역할 이름 → 실제 색 (`macro.MacroSpec.role`)."""
    return {"bed": pal.bed, "bed_alt": pal.bed_alt, "primary": pal.primary,
            "secondary": pal.secondary, "shadow": pal.shadow, "dark": pal.dark}


def _tear(base: list[Layer], fld: CompositionField, edge_v: tuple[str, ...],
          cat: Catalog) -> list[Layer]:
    """큰 색면의 **흐름 쪽 끝을 뜯는다** (스플래시 계열 — `Family.torn`).

    옛 `bed`가 판의 마지막 장에 톱니를 물리던 자리다. 매크로 어휘는 색면이
    프레임 밖까지 나가므로 뜯을 자리는 **프레임 안쪽 가장자리**다 — 흐름 쪽
    프레임 변에서 조금 안으로 들어온 선을 조각으로 문다.
    """
    first = next((l for l in base if l.label == "itasha_bed"), None)
    if first is None:
        return []
    fx0, fy0, fx1, fy1 = fld.frame_box
    fs = 1.0 if fld.flow[0] >= 0 else -1.0
    h = fld.char_h
    ex = (fx1 if fs > 0 else fx0) - fs * 0.18 * (fx1 - fx0)
    return _teeth(edge_v, cat, span=(fy1 - fy0) * 1.1, x0=ex - 0.5 * h,
                  top=(fy0 + fy1) / 2 + (fy1 - fy0) * 0.55, band=h * 0.45, n=3,
                  color=first.color, label="itasha_bed")


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

    def _base(fam: Family, fld: CompositionField, pal: RolePalette, level: float,
              kinds: tuple[str, str], tw: "Tweak") -> list[Layer]:
        """큰 색면 + 로커 — 후보의 **바탕**.

        판이 먼저, 로커가 그 위다 — 색면은 프레임을 관통해 로커 속으로도 내려가고
        로커 띠가 그 아랫단을 덮어 하부 투톤이 색면을 자른다 (레퍼런스의 검은
        하부는 판 위에 얹힌 층이다 — Evo IX·EVELYNE).
        """
        specs = macro_plan(fld, kinds, level, rocker=fam.rocker,
                           d_rot=tw.bed_rot, d_y=tw.bed_dy, d_w=tw.bed_w)
        base = macro_layers(specs, fld.frame_box, _macro_colors(pal), cat)
        if fam.torn and edge_v and base:
            base += _tear(base, fld, edge_v, cat)
        if fam.rocker:
            base += _rocker(fld, lk, cat, car_rgb, edge_v)
        return base

    def _parts(fam: Family, fld: CompositionField, pal: RolePalette, level: float,
               kinds: tuple[str, str], tw: "Tweak"
               ) -> tuple[list[Layer], list[Layer], list[Layer], list]:
        """후보 한 벌의 (바탕, 꼬리, 전경, 산포 통계) — 미세 조정도 이 자를 쓴다."""
        base = _base(fam, fld, pal, level, kinds, tw)
        fam_m = _replace(fam, tier_scale=fam.tier_scale * tw.motif_k)
        sc, stats = _scatter(fld, fam_m, pal, cat, vocab, halo, False, phase,
                             anchor_dx=tw.anchor_dx, angularity=it.angularity)
        tail: list[Layer] = list(sc)
        if fam.echo:
            tail += echo_layers(fld, it, pal, cat, n=max(3, fam.motif_n // 3),
                                phase=phase)
        tail = _readable_motifs(tail, fld, pal, cat, base)
        front, _fs = _scatter(fld, fam_m, pal, cat, vocab, None, True, phase,
                              anchor_dx=tw.anchor_dx, angularity=it.angularity)
        return base, tail, front, stats

    def _card(fam: Family, fld: CompositionField, pal: RolePalette, level: float,
              base, tail, front, front_ras, keyl, keyline: bool, ts, stats,
              text_ras, behind, tplan, tstyle, tw: "Tweak",
              kinds: tuple[str, str] = ("ribbon", "ribbon")) -> Design:
        """부품 한 벌 → 재고 담은 후보. 후보 루프와 미세 조정이 같은 자를 쓴다."""
        back = list(base)
        tl = ts.layers if ts is not None else []
        n_text = ts.n if ts is not None else 0
        if keyline:
            back += keyl
        back += tail
        # 면 상한 — 역할이 낮은 것부터 뺀다 (도안·판·글자가 먼저다).
        # 글자는 제 그룹이라 따로 센다.
        room = cap - 4 - n_person - len(front) - n_text
        back, trimmed = _fit_cap(back, room)
        extra = None
        if text_on:
            extra = text_parts(fld, cat, ts.poses if ts else [], tl, behind,
                               front_alpha=front_ras[1])
        # **구성 그래프** — 지어 놓은 부품에서 역할 노드를 되읽고 계열의 문법을
        # 건다 (`graph`). 점수가 "무엇이 있나"가 아니라 "무엇과 무엇이 어떤
        # 사이인가"를 잴 수 있게 하는 자리다. 여백 노드는 점수기가 붙인다.
        gr = derive(fld, back, front, stats,
                    text_poses=(ts.poses if ts is not None else None))
        gr.rels = tuple(Rel(k, a, b, w) for k, a, b, w in fam.rels())
        # 점수용 합성: 글자 그룹은 꾸밈 그룹 **위**·도안 아래에 선다
        card = score_design(fld, pal, cat, back, front,
                            clutter_target=fam.clutter, empty_target=fam.empty_target,
                            motifs=stats, rocker=fam.rocker,
                            extra=extra, extra_weights=TEXT_WEIGHTS,
                            text=text_ras, front_raster=front_ras, graph=gr)
        return Design(family=fam, pal=pal, fld=fld, back=back, front=front, score=card,
                      flow_rear=(fld.flow[0] * rear_sign) > 0, level=level,
                      keyline=keyline, text=ts, text_plan=tplan, text_style=tstyle,
                      trimmed=trimmed, tweak=tw, macro=kinds)

    # ---- 단계 A: **매크로 기하**만 겨룬다 -------------------------------------
    # 계열 × 흐름 × 어휘 짝 × 크기를 전수로 돌면 후보가 사백을 넘어 한 판에
    # 이십 초가 넘는다. 큰 색면이 정해지기 전에는 산포·에코·글자가 순위를 거의
    # 안 바꾸므로(둘 다 색면 위에 얹히는 잔 요소다) **기하를 먼저 추린다**:
    # 대표 팔레트 하나로 색면만 지어 재고, 살아남은 것에만 팔레트·키라인·글자를
    # 붙인다. 대표 팔레트는 `VARIANTS_TRIED[0]`이다 — 어느 하나를 골라야 하고,
    # 순위는 기하끼리의 견줌이라 팔레트가 같으면 공정하다.
    pal0 = role_palette(it, lk, car_rgb, VARIANTS_TRIED[0])
    seeds: list[tuple[tuple[int, float], Family, str, CompositionField,
                      tuple[str, str], float]] = []
    for fname in fams:
        fam = FAMILIES[fname]
        for mode in fam.flows:
            fld = _field(mode)
            for kinds in fam.macro:
                for level in (fam.bed_level, max(0.0, fam.bed_level - 0.25)):
                    base = _base(fam, fld, pal0, level, kinds, Tweak())
                    gr = derive(fld, base, [], [])
                    gr.rels = tuple(Rel(k, a, b, w) for k, a, b, w in fam.rels())
                    sc = score_design(fld, pal0, cat, base, [],
                                      clutter_target=fam.clutter,
                                      empty_target=fam.empty_target, motifs=[],
                                      rocker=fam.rocker, graph=gr)
                    seeds.append(((len(sc.fails), -round(sc.total, 6)), fam, mode,
                                  fld, kinds, level))
                    if fam.bed == "none":
                        break
    seeds.sort(key=lambda s: s[0])
    # **계열마다 하나는 살려 보낸다.** 점수순으로만 자르면 자를 느슨하게 잡은
    # 계열(minimal의 `clutter` 0.03~0.12)이 빔을 통째로 차지한다 — 실측 P4에서
    # 33판 중 스물일곱이 minimal·motorsport였고 graphic_bed는 한 판도 못 이겼다.
    # 큰 색면이 다른 구도는 조각까지 얹어 봐야 견줄 수 있으므로, 계열마다 가장
    # 좋은 기하 하나를 먼저 넣고 남는 자리를 점수순으로 채운다.
    keep: list = []
    seen: set[str] = set()
    for sd in seeds:
        if sd[1].name not in seen:
            seen.add(sd[1].name)
            keep.append(sd)
    for sd in seeds:
        if len(keep) >= BEAM_MACRO:
            break
        if sd not in keep:
            keep.append(sd)
    seeds = keep[:max(BEAM_MACRO, len(seen))]

    # ---- 단계 B: 살아남은 기하에 팔레트·키라인·글자를 붙인다 -------------------
    cands: list[Design] = []
    for _k, fam, _mode, fld, kinds, level in seeds:
        for variant in VARIANTS_TRIED:
            pal = role_palette(it, lk, car_rgb, variant)
            base, tail, front, stats = _parts(fam, fld, pal, level, kinds, Tweak())
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
            # 게임 글꼴 글리프는 후보당 수십 장이라 도형 맞춤(수백 장)과 달리
            # 다듬기를 막을 이유가 없다 — 다만 판을 흔들면 글자 자리도 흔들려
            # 다시 지어야 하므로 `_refine`은 여전히 글자 없는 판만 돈다.
            # 글자 래스터는 (배치 후보 × 이 팔레트)마다 한 번만 — 키라인·
            # 베드 크기 변종이 같은 것을 나눠 쓴다
            text_ras = {id(ts): raster_layers(ts.layers, fld, cat)
                        for ts in text_sets if ts is not None}
            behind = composite(fld, pal, cat, base, [], front_raster=front_ras)["behind"] \
                if text_on else None
            for keyline in (False, True):
                for ts in text_sets:
                    cands.append(_card(fam, fld, pal, level, base, tail, front,
                                       front_ras, keyl, keyline, ts, stats,
                                       text_ras.get(id(ts)), behind,
                                       tplan, tstyle, Tweak(), kinds))
    # 탈락 조건에 걸린 후보는 점수와 무관하게 뒤로 간다 (전멸하면 위반 수로 고른다).
    #
    # 총점은 **여섯째 자리에서 끊어** 견준다. 점수 항목 몇은 LAPACK을 거치는데
    # (`np.linalg.eigh` — 판·장식의 장축), 거기서 나오는 마지막 비트가 스레드 수에
    # 따라 흔들린다. 상위 후보가 0.001 안에 몰리는 것이 흔한 판이라(베드 크기
    # 변종 둘은 점수가 같기 일쑤다) 그 흔들림이 순위를 뒤집어 **같은 입력이 다른
    # 파일**을 냈다 (실측: deco.json 해시가 판마다 갈렸다). 끊으면 진짜로 같은
    # 후보들이 동점이 되고, 파이썬 정렬이 안정적이라 먼저 지은 것이 이긴다 —
    # 후보를 짓는 순서는 못 박혀 있으므로 그것으로 결정성이 선다.
    def _rank(d: Design) -> tuple[int, float]:
        return (len(d.score.fails), -round(d.score.total, 6))

    def _refine(d: Design) -> Design:
        """이긴 후보를 **좌표하강**으로 다듬는다 — 이산 후보 격자의 사이를 메운다.

        축·걸음·순서가 못 박혀 있고(`REFINE_STEPS`) 같은 점수면 원래 값이 이긴다
        — 난수도 전수 조합도 없다 (`REFINE_STEPS`).
        """
        keyl = keyline_layers(d.fld, _keyline_color(d.pal), cat)
        best_d, tw = d, d.tweak
        for _p in range(REFINE_PASSES):
            moved = False
            for name, steps in REFINE_STEPS:
                cur = getattr(tw, name)
                lo, hi = REFINE_CLAMP[name]
                for st in steps:
                    v = cur + st
                    if not (lo <= v <= hi):
                        continue
                    cand_tw = _replace(tw, **{name: v})
                    b2, t2, f2, s2 = _parts(d.family, d.fld, d.pal, d.level,
                                            d.macro, cand_tw)
                    cd = _card(d.family, d.fld, d.pal, d.level, b2, t2, f2,
                               raster_layers(f2, d.fld, cat), keyl, d.keyline,
                               None, s2, None, None, None, None, cand_tw, d.macro)
                    if _rank(cd) < _rank(best_d):
                        best_d, tw, moved = cd, cand_tw, True
            if not moved:
                break
        return best_d

    cands.sort(key=_rank)
    # 글자가 있는 판은 안 다듬는다 — 글자 벌은 판 알파에서 나오므로 손잡이마다
    # 글자를 다시 지어야 하고, 그러면 후보 하나가 수백 ms가 된다.
    if not text_on:
        for i in range(min(REFINE_TOP, len(cands))):
            cands[i] = _refine(cands[i])
        cands.sort(key=_rank)
    best = cands[0]
    best.ranking = [(f"{'!' * len(d.score.fails)}{d.family.name}/{d.pal.variant}"
                     f"/{'rear' if d.flow_rear else 'front'}"
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
