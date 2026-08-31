"""구도 점수 — 후보 꾸밈 한 벌을 **옆면 한 장으로 합성해** 재는 자.

후보를 만들고 → 렌더하고 → 재고 → 고르는 루프의 자다. 자는 가벼운
휴리스틱이지만 구조는 하나다: 필드 격자(`CompositionField`) 위에서 베이스 →
꾸밈 → 인물 → 전경을 합성한 그림과 격자 래스터를 겹쳐 항목마다 0~1을 내고
가중합한다. 항목:

| 항목 | 재는 것 |
|---|---|
| readability | 실루엣 테두리 안팎의 명도차 (베드·베이스 위에서 인물이 읽히나) |
| face | 전경 조각이 보호 구역(얼굴)을 덮는 몫의 벌점 |
| balance | 전체 잉크 무게중심의 가로 치우침 |
| clutter | 장식 커버리지가 계열 목표 구간 안인가 · 조각 수 |
| negative | 여백 구역이 비어 있나 |
| flow | 장식 무게중심이 흐름 쪽으로 갔나 · 장식 장축이 흐름과 나란한가 |
| cohesion | 모티프끼리 이어져 있나 (최근접 거리) |
| bed | 베드가 지지 구역을 덮되 밖으로 안 나가나 · 베이스와 갈리나 |
| continuity | 로커·베드가 프레임 끝(이음새)까지 닿나 |
| orphan | 무리에서 떨어진 중대형 조각 벌점 |

가중치는 `WEIGHTS`다. 더 똑똑한 평가기로 바꾸려면 `score_design`만 갈면 된다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..catalog import Catalog
from ..model import Layer, LayerPlan
from ..render import render_plan
from .field import CompositionField
from .roles import RolePalette


WEIGHTS = {
    "readability": 2.0, "face": 2.0, "balance": 0.8, "clutter": 1.0,
    "negative": 0.8, "flow": 1.0, "cohesion": 0.6, "bed": 1.2,
    "continuity": 0.5, "orphan": 0.6,
}


@dataclass
class ScoreCard:
    total: float = 0.0
    parts: dict[str, float] = field(default_factory=dict)
    info: dict[str, float] = field(default_factory=dict)

    def text(self) -> str:
        return " ".join(f"{k}={v:.2f}" for k, v in self.parts.items())


def _lum(img: np.ndarray) -> np.ndarray:
    f = img.astype(np.float32)
    return (0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]) / 255.0


def raster_layers(layers: list[Layer], fld: CompositionField, cat: Catalog
                  ) -> tuple[np.ndarray, np.ndarray]:
    """레이어들을 필드 격자에 렌더 → (rgb, alpha)."""
    g = fld.grid
    if not layers:
        return (np.zeros((g.rows, g.cols, 3), np.uint8),
                np.zeros((g.rows, g.cols), np.float32))
    cx = g.x0 + g.cols * g.cell / 2
    cy = g.y_top - g.rows * g.cell / 2
    plan = LayerPlan(source_image="score", image_size=(g.cols, g.rows),
                     units_per_px=g.cell,
                     layers=[Layer(shape=l.shape, x=l.x - cx, y=l.y - cy, sx=l.sx,
                                   sy=l.sy, rot=l.rot, skew=l.skew, color=l.color,
                                   alpha=l.alpha, label=l.label, mask=l.mask)
                             for l in layers])
    a = render_plan(plan, cat, bg=0).astype(np.int16)
    b = render_plan(plan, cat, bg=255).astype(np.int16)
    alpha = np.clip(1.0 - np.abs(b - a).mean(axis=2) / 255.0, 0.0, 1.0).astype(np.float32)
    rgb = np.clip(b, 0, 255).astype(np.uint8)
    return rgb, alpha


def composite(fld: CompositionField, pal: RolePalette, cat: Catalog,
              back: list[Layer], front: list[Layer],
              text: tuple[np.ndarray, np.ndarray] | None = None,
              front_raster: tuple[np.ndarray, np.ndarray] | None = None) -> dict:
    """베이스 → 꾸밈 → (글자) → 인물 → 전경 합성과 중간 래스터.

    `text`·`front_raster`는 미리 렌더한 (rgb, alpha) — 후보 루프가 같은 글자·전경을
    수백 번 렌더하지 않게 부르는 쪽이 캐시해 준다 (실측: 글자 335장을 후보마다
    렌더하니 97초).
    """
    g = fld.grid
    base = np.zeros((g.rows, g.cols, 3), np.float32)
    base[:] = pal.base
    brgb, balpha = raster_layers(back, fld, cat)
    img = base * (1 - balpha[..., None]) + brgb.astype(np.float32) * balpha[..., None]
    behind = img.copy()                          # 인물 뒤에 실제로 보이는 것
    if text is not None:
        trgb, talpha = text
        img = img * (1 - talpha[..., None]) + trgb.astype(np.float32) * talpha[..., None]
        balpha = np.maximum(balpha, talpha)
    ca = fld.char[..., None]
    img = img * (1 - ca) + fld.char_rgb.astype(np.float32) * ca
    frgb, falpha = front_raster if front_raster is not None else raster_layers(front, fld, cat)
    img = img * (1 - falpha[..., None]) + frgb.astype(np.float32) * falpha[..., None]
    return {"img": img, "behind": behind, "back_alpha": balpha, "front_alpha": falpha}


def _cluster_stats(motifs: list[tuple[float, float, float, int]], gap: float
                   ) -> tuple[float, float]:
    """(이어짐 몫, 고아 벌점) — (x, y, 크기, 층) 목록에서."""
    if len(motifs) < 2:
        return 1.0, 0.0
    linked = 0
    orphan = 0.0
    for i, (x, y, s, tier) in enumerate(motifs):
        nn = min(math.hypot(x - a, y - b) / max(1e-6, (s + c) / 2)
                 for k, (a, b, c, _t) in enumerate(motifs) if k != i)
        if nn <= gap:
            linked += 1
        elif tier <= 1:
            orphan += 1.0
    return linked / len(motifs), orphan / max(1.0, len(motifs) * 0.25)


def score_design(fld: CompositionField, pal: RolePalette, cat: Catalog,
                 back: list[Layer], front: list[Layer], *,
                 clutter_target: tuple[float, float], empty_target: float,
                 motifs: list[tuple[float, float, float, int]],
                 rocker: bool, gap: float = 3.0,
                 extra: dict[str, float] | None = None,
                 extra_weights: dict[str, float] | None = None,
                 text: tuple[np.ndarray, np.ndarray] | None = None,
                 front_raster: tuple[np.ndarray, np.ndarray] | None = None) -> ScoreCard:
    """`extra`는 다른 자(텍스트 — `textscore`)가 낸 항목들 — 같은 표에 가중합한다."""
    g = fld.grid
    comp = composite(fld, pal, cat, back, front, text=text, front_raster=front_raster)
    behind = comp["behind"]
    balpha, falpha = comp["back_alpha"], comp["front_alpha"]
    sil = fld.char > 0.5
    draw = fld.drawable > 0.5
    parts: dict[str, float] = {}
    info: dict[str, float] = {}

    # 1) 가독성 — 실루엣 테두리 안쪽 색과 바로 바깥의 배경 명도차
    k = np.ones((3, 3), np.uint8)
    inner = sil & ~cv2.erode(sil.astype(np.uint8), k).astype(bool)
    outer = cv2.dilate(sil.astype(np.uint8), k).astype(bool) & ~sil & draw
    if inner.any() and outer.any():
        lin = _lum(fld.char_rgb)[inner].mean()
        lout = _lum(behind)[outer]
        dl = float(np.abs(lout - lin).mean())
        # 테두리 바깥 명도의 **분산**도 본다 — 얼룩덜룩한 배경은 실루엣을 갉는다
        var = float(lout.std())
        parts["readability"] = max(0.0, min(1.0, dl / 0.42)) * (1.0 - 0.5 * min(1.0, var / 0.35))
        info["edge_dl"] = dl
    else:
        parts["readability"] = 0.5
    # 2) 얼굴 가림 — 전경 조각이 보호 구역을 덮는 몫
    prot = fld.protected > 0.5
    cover = float((falpha[prot] > 0.5).mean()) if prot.any() else 0.0
    parts["face"] = max(0.0, 1.0 - cover / 0.06)
    info["face_cover"] = cover
    # 3) 균형 — 전체 잉크(인물 + 장식) 가로 무게중심
    ink = np.maximum(fld.char, np.maximum(balpha, falpha)) * draw
    X, Y = g.centers()
    tot = float(ink.sum())
    if tot > 1e-6:
        mx = float((ink * X).sum() / tot)
        half = 0.5 * (fld.frame_box[2] - fld.frame_box[0])
        off = abs(mx - (fld.frame_box[0] + fld.frame_box[2]) / 2) / half
        parts["balance"] = max(0.0, 1.0 - max(0.0, off - 0.10) / 0.35)
        info["balance_off"] = off
    else:
        parts["balance"] = 0.5
    # 4) 어수선함 — 장식 커버리지 (밴드의 인물 밖 도색면 대비). 로커 띠와 디더
    #    페이드는 바닥 요소라 안 센다 — 세면 로커 계열이 전부 "어수선"으로
    #    떨어지고, 페이드는 판의 가장자리 처리인데 커버리지를 밀어 계열 고르기를
    #    바꿔 버린다 (실측: 페이드를 세니 33벌 중 28벌이 graphic_bed로 쏠렸다).
    room = draw & ~sil
    _mrgb, malpha = raster_layers(
        [l for l in back if l.label not in ("itasha_stripe", "itasha_fade")], fld, cat)
    cov = float((malpha[room] > 0.5).mean()) if room.any() else 0.0
    lo, hi = clutter_target
    if cov < lo:
        parts["clutter"] = max(0.0, 1.0 - (lo - cov) / max(0.05, lo))
    elif cov > hi:
        parts["clutter"] = max(0.0, 1.0 - (cov - hi) / 0.25)
    else:
        parts["clutter"] = 1.0
    info["coverage"] = cov
    # 5) 여백 — 여백 구역이 비었나
    neg = fld.negative > 0.5
    if neg.any():
        empty = 1.0 - float((malpha[neg] > 0.5).mean())
        parts["negative"] = max(0.0, min(1.0, 1.0 - max(0.0, empty_target - empty) / 0.5))
        info["empty"] = empty
    else:
        parts["negative"] = 1.0
    # 6) 흐름 — 모티프·에코 무게중심이 흐름 쪽에 있고 그 장축이 흐름과 나란한가
    #    (베드·로커는 뺀다 — 프레임을 가로지르는 판은 어느 쪽도 아니다)
    _srgb, salpha = raster_layers(
        [l for l in back if l.label in ("itasha_deco", "itasha_echo")], fld, cat)
    dm = salpha * draw * (~sil)
    dt = float(dm.sum())
    if dt > 1e-6:
        dx = float((dm * X).sum() / dt) - fld.visual_center[0]
        dy = float((dm * Y).sum() / dt) - fld.visual_center[1]
        proj = (dx * fld.flow[0] + dy * fld.flow[1]) / max(1.0, 0.5 * fld.char_w)
        side = max(0.0, min(1.0, 0.5 + proj))
        ys, xs = np.where(dm > 0.5)
        if len(xs) > 8:
            x = xs.astype(np.float64) - xs.mean()
            y = ys.astype(np.float64) - ys.mean()
            cov_ = np.array([[float((x * x).sum()), float((x * y).sum())],
                             [float((x * y).sum()), float((y * y).sum())]]) / len(x)
            vals, vecs = np.linalg.eigh(cov_)
            mj = vecs[:, int(np.argmax(vals))]
            fl = np.array([fld.flow[0], -fld.flow[1]])
            par = abs(float(mj @ fl))
        else:
            par = 0.5
        parts["flow"] = 0.6 * side + 0.4 * par
    else:
        parts["flow"] = 0.3
    # 7·10) 이어짐과 고아
    linked, orphan = _cluster_stats(motifs, gap)
    parts["cohesion"] = linked
    parts["orphan"] = max(0.0, 1.0 - orphan)
    # 8) 베드 — 지지 구역 덮음 · 밖으로 나감 · 베이스와 갈림
    bed = [l for l in back if l.label == "itasha_bed"]
    if bed:
        _brgb, ba = raster_layers(bed, fld, cat)
        supp = fld.support > 0.5
        inside = float((ba[supp] > 0.5).mean()) if supp.any() else 0.0
        spill = float((ba[draw & ~supp] > 0.5).mean()) if (draw & ~supp).any() else 0.0
        bl = _lum(np.array([[pal.bed]], np.uint8))[0, 0]
        base_l = _lum(np.array([[pal.base]], np.uint8))[0, 0]
        sep = min(1.0, abs(bl - base_l) / 0.25)
        parts["bed"] = 0.5 * min(1.0, inside / 0.55) + 0.3 * max(0.0, 1.0 - spill / 0.30) + 0.2 * sep
        info["bed_inside"] = inside
        info["bed_spill"] = spill
    else:
        parts["bed"] = 0.45                      # 베드 없음 — 중립 (계열이 그렇다)
    # 9) 이어짐 — 로커나 베드가 프레임 양 끝에 닿나
    edge_cols = np.zeros(g.cols, bool)
    edge_cols[:3] = True
    edge_cols[-3:] = True
    touch = float((balpha[:, edge_cols] > 0.5).any(axis=0).mean())
    parts["continuity"] = min(1.0, 0.5 * touch + (0.5 if rocker else 0.2))
    parts = {k: float(v) for k, v in parts.items()}
    info = {k: float(v) for k, v in info.items()}
    weights = dict(WEIGHTS)
    if extra:
        parts.update({k: float(v) for k, v in extra.items()})
        weights.update(extra_weights or {})
    total = sum(weights.get(k, 0.5) * v for k, v in parts.items()) / sum(
        weights.get(k, 0.5) for k in parts)
    return ScoreCard(total=float(total), parts=parts, info=info)
