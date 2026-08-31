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
| clutter | 모티프 커버리지가 계열 목표 구간 안인가 (판·로커는 바닥 요소라 안 센다) |
| negative | 여백 구역에 모티프가 없나 |
| flow | 장식 무게중심이 흐름 쪽으로 갔나 · 장식 장축이 흐름과 나란한가 |
| cohesion | 모티프끼리 이어져 있나 (최근접 거리) |
| integration | 판이 인물 뒤에 깔리고 포즈 축을 따르나 · 바탕과 갈리나 |
| hierarchy | 모티프 덩어리의 무게가 주역/조연/잔것으로 갈리나 |
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
from .boxes import major_axis
from .critic import critique
from .field import CompositionField
from .graph import CompositionGraph, Node, relation_score
from .roles import RolePalette


WEIGHTS = {
    # `readability`는 **제약이지 목표가 아니다.** 가중치 2.0으로 두면 그 자를
    # 최대로 만드는 길이 "판을 아예 안 깔고 인물을 맨 도색 위에 두는 것"이라
    # 큰 색면을 쓰는 계열이 한 판도 못 이긴다 (33판 중 graphic_bed 1 ·
    # diagonal_flow 0). 대신 탈락 문턱을 0.09 → 0.16으로 올려 **정말 안 읽히는
    # 판**을 거르고, 그 위에서는 구도가 정하게 둔다.
    "readability": 1.3, "face": 2.0, "balance": 1.8, "clutter": 1.0,
    "negative": 0.8, "flow": 1.0, "cohesion": 0.6, "integration": 1.4,
    "continuity": 0.5, "orphan": 0.6, "hierarchy": 1.4,
    # ---- 배율별 자 (`critic`) ----
    # 위 열한 항목은 33판 실측에서 넷이 **전부 1.000**이고 값이 갈리는 것은
    # `readability` 하나뿐이었다 (1위·2위 점수 차 중앙값 0.0000 — 후보를 고르는
    # 것이 점수가 아니라 후보를 짓는 순서였다). 아래 다섯이 배율마다 다른 것을
    # 물어 그 구멍을 메운다. 가중치가 큰 둘(`focal`·`macro`)이 "멀리서 인물이
    # 먼저 읽히나"와 "큰 덩어리가 무게를 나눠 쥐나"다 — 자동 생성 티의 두 뿌리다.
    "focal": 2.0, "macro": 2.2, "rhythm": 1.2, "negative_shape": 1.0,
    "gesture": 0.8, "presence": 1.4, "lengthwise": 1.2,
    # 요소 사이 — 구성 그래프의 문법이 지켜졌나 (`graph.DEFAULT_GRAMMAR`)
    "relations": 1.6,
}


# 실루엣이 **읽히는** 최소 테두리 명도차 — 이 아래면 인물이 배경에 묻는다.
# 0.09는 너무 낮아 아무 후보도 안 걸렸고(실측 33판 최저 0.31), 그래서 가독성이
# 탈락 조건이 아니라 사실상 순위 자로만 돌았다.
READ_FLOOR = 0.16


# 로커 띠의 잉크 — 판이 이것과 같은 색이면 하부와 한 덩이가 된다 (`roof.ROOF_DARK`).
ROCKER_INK = (16, 17, 20)


# **바닥 요소** — 프레임을 관통하는 판·띠와 로커. 구도의 뼈대라 어수선·여백·위계
# 자로는 안 잰다 (그 셋은 **모티프**의 자다). 세면 판이 서는 계열이 전부
# "어수선"으로 떨어지고 여백이 늘 차 있다.
GROUND = ("itasha_bed", "itasha_stripe")


@dataclass
class ScoreCard:
    total: float = 0.0
    parts: dict[str, float] = field(default_factory=dict)
    info: dict[str, float] = field(default_factory=dict)
    fails: tuple[str, ...] = ()          # 탈락 조건 — 하나라도 걸리면 후보가 아니다

    def text(self) -> str:
        t = " ".join(f"{k}={v:.2f}" for k, v in self.parts.items())
        return t + (f" [!{'/'.join(self.fails)}]" if self.fails else "")


def _band(v: float, lo: float, hi: float, soft: float) -> float:
    """구간 안이면 1, 밖이면 `soft`만큼 멀어지는 동안 0으로 내려간다."""
    if v < lo:
        return max(0.0, 1.0 - (lo - v) / max(1e-6, soft))
    if v > hi:
        return max(0.0, 1.0 - (v - hi) / max(1e-6, soft))
    return 1.0


def _de(a, b) -> float:
    """두 RGB의 Lab 거리 — 색이 갈리나를 사람 눈에 가깝게 잰다."""
    la, lb = (cv2.cvtColor(np.array([[list(c)]], np.uint8), cv2.COLOR_RGB2LAB)[0, 0]
              .astype(np.float32) for c in (a, b))
    return float(np.linalg.norm(la - lb))


def _blob_weights(alpha: np.ndarray, rgb: np.ndarray, base, room: np.ndarray
                  ) -> list[float]:
    """꾸밈 덩어리의 **시각 무게** 목록 (큰 것부터) — 면적 x 바탕 대비.

    사람이 만든 구도는 요소의 무게가 고르지 않다: 주역 하나, 조연 몇, 잔 것
    여럿이다. 면적만으로는 흰 바탕의 흰 판과 검은 판이 같아지므로 대비를 곱한다.
    """
    m = ((alpha > 0.5) & room).astype(np.uint8)
    if not m.any():
        return []
    n, lbl, st, _c = cv2.connectedComponentsWithStats(m, 8)
    tot = float(room.sum()) or 1.0
    out = []
    for i in range(1, n):
        a = float(st[i, cv2.CC_STAT_AREA])
        if a < 0.0008 * tot:
            continue
        col = rgb[lbl == i].mean(axis=0).round().astype(np.uint8)
        out.append(a / tot * min(1.0, _de(col, base) / 40.0))
    out.sort(reverse=True)
    return out


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
                 front_raster: tuple[np.ndarray, np.ndarray] | None = None,
                 graph: CompositionGraph | None = None) -> ScoreCard:
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
    # 4) 어수선함 — **모티프** 커버리지 (밴드의 인물 밖 도색면 대비). 판·로커는
    #    바닥 요소라 안 센다 (`GROUND`) — 관통하는 판은 여백의 절반을 덮는 것이
    #    정상이고, 어수선은 그 위에 흩은 조각이 낸다.
    room = draw & ~sil
    mrgb, malpha = raster_layers([l for l in back if l.label not in GROUND], fld, cat)
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
            mj, _e = major_axis(xs, ys)
            par = abs(mj[0] * fld.flow[0] + mj[1] * -fld.flow[1])
        else:
            par = 0.5
        parts["flow"] = 0.6 * side + 0.4 * par
    else:
        parts["flow"] = 0.3
    # 7·10) 이어짐과 고아
    linked, orphan = _cluster_stats(motifs, gap)
    parts["cohesion"] = linked
    parts["orphan"] = max(0.0, 1.0 - orphan)
    # 8) 배경 통합 — 판이 **인물 뒤에** 있고 포즈 축을 따르나 (+ 바탕과 갈리나)
    #    옛 `bed` 항목(지지 구역 덮음·삐져나감)을 대신한다: 그쪽은 판이 인물을
    #    지나가기만 해도 만점이라 "인물 뒤에 아무것도 없는" 구도를 못 걸렀다.
    #    판이 지지 구역 밖으로 나가는 몫은 안 잰다 — 판은 프레임을 관통하는
    #    것이 문법이다 (`bed`).
    bed = [l for l in back if l.label == "itasha_bed"]
    if bed:
        _brgb, ba = raster_layers(bed, fld, cat)
        backing = float((ba[sil] > 0.5).mean()) if sil.any() else 0.0
        sep = min(1.0, _de(pal.bed, pal.base) / 26.0)
        # 판의 축이 포즈 축과 나란한가 — 인물이 기울면 판도 기울어야 한 덩어리다
        ys_, xs_ = np.where(ba > 0.5)
        align = 0.5
        if len(xs_) > 40:
            mj_, _e = major_axis(xs_, ys_)
            # 포즈 축과 흐름 중 **가까운 쪽**을 따르면 된다 (누운 인물은 둘이 같다)
            align = max(abs(mj_[0] * fld.axis[0] + mj_[1] * -fld.axis[1]),
                        abs(mj_[0] * fld.flow[0] + mj_[1] * -fld.flow[1]))
        parts["integration"] = (0.55 * min(1.0, backing / 0.70)
                                + 0.20 * sep + 0.25 * align)
        info["backing"] = backing
    else:
        parts["integration"] = 0.40              # 판 없음 — 인물이 맨 도색 위에 뜬다
    # 9) 이어짐 — 로커나 베드가 프레임 양 끝에 닿나
    edge_cols = np.zeros(g.cols, bool)
    edge_cols[:3] = True
    edge_cols[-3:] = True
    touch = float((balpha[:, edge_cols] > 0.5).any(axis=0).mean())
    parts["continuity"] = min(1.0, 0.5 * touch + (0.5 if rocker else 0.2))
    # 11) 위계 — 모티프 덩어리의 무게가 주역/조연/잔것으로 갈리나
    #     (전부 비슷한 무게로 흩어진 판은 기계가 뿌린 것으로 읽힌다). 바닥 요소는
    #     안 센다 — 관통하는 판은 모든 조각과 한 덩이로 이어져 위계를 지운다.
    ws = _blob_weights(malpha, mrgb, pal.base, room)
    if len(ws) >= 2:
        tot_w = sum(ws) or 1e-9
        h1 = ws[0] / tot_w
        h2 = ws[1] / ws[0]
        tail = sum(1 for w in ws if w < 0.25 * ws[0])
        parts["hierarchy"] = (0.45 * _band(h1, 0.32, 0.66, 0.30)
                              + 0.35 * _band(h2, 0.22, 0.66, 0.28)
                              + 0.20 * min(1.0, tail / 3.0))
        info["h1"] = h1
        info["h2"] = h2
    else:
        parts["hierarchy"] = 0.35                # 덩어리가 하나뿐 — 위계가 없다
    # 12~16) **배율별 자** — 멀리/중간/가까이에서 다른 것을 묻는다 (`critic`).
    #     여기 넘기는 꾸밈 알파는 **판·로커까지 넣은 것**이다: 위 항목들이
    #     모티프만 보는 것과 갈리는 자리다 — 큰 판이야말로 멀리서 읽히는
    #     덩어리라 위계를 재려면 그것을 세야 한다.
    all_deco = np.maximum(balpha, falpha)
    cr = critique(
        img=comp["img"], sil=sil, room=room, ink=ink, deco_alpha=all_deco,
        base_lum=float(_lum(np.array([[list(pal.base)]], np.uint8))[0, 0]),
        motifs=motifs, cols=g.cols, cell=g.cell, x0=g.x0, y_top=g.y_top,
        visual_center=fld.visual_center, head_c=fld.head_center,
        face_dir=fld.face_dir, char_w=fld.char_w,
        gestures=fld.gestures or ((fld.texture[0], fld.texture[1],
                                   fld.texture_coherence),))
    parts.update(cr.parts)
    info.update(cr.info)
    # 17) **관계** — 구성 그래프의 문법이 이 기하에서 지켜졌나 (`graph`).
    #     여백 노드는 여기서 붙인다 (가장 큰 빈 덩이는 비평이 찾는다).
    if graph is not None:
        if cr.neg_box is not None:
            fa = max(1e-6, (fld.frame_box[2] - fld.frame_box[0])
                     * (fld.frame_box[3] - fld.frame_box[1]))
            nb = cr.neg_box
            graph.add(Node(id="neg", role="negative",
                           at=((nb[0] + nb[2]) / 2, (nb[1] + nb[3]) / 2),
                           axis=(1.0, 0.0),
                           extent=(nb[2] - nb[0], nb[3] - nb[1]), box=nb,
                           weight=(nb[2] - nb[0]) * (nb[3] - nb[1]) / fa,
                           kind="void", z=0))
        rv, rinfo = relation_score(graph)
        parts["relations"] = rv
        info.update(rinfo)
    parts = {k: float(v) for k, v in parts.items()}
    info = {k: float(v) for k, v in info.items()}
    # ---- 탈락 조건 (가중합에 안 섞는다) ----
    fails: list[str] = list(cr.fails)
    if info.get("face_cover", 0.0) > 0.06:
        fails.append("face")
    if info.get("edge_dl", 1.0) < READ_FLOOR:
        fails.append("readability")
    if cov > hi + 0.25:
        fails.append("clutter")
    # 판이 로커와 같은 색이면 한 덩이가 된다 — 관통하는 판은 늘 로커에 닿으므로
    # 색만 본다 (하부가 통째로 치솟는 꼴)
    if rocker and bed and _de(pal.bed, ROCKER_INK) < 12.0:
        fails.append("merge")
    weights = dict(WEIGHTS)
    if extra:
        parts.update({k: float(v) for k, v in extra.items()})
        weights.update(extra_weights or {})
    total = sum(weights.get(k, 0.5) * v for k, v in parts.items()) / sum(
        weights.get(k, 0.5) for k in parts)
    return ScoreCard(total=float(total), parts=parts, info=info, fails=tuple(fails))
