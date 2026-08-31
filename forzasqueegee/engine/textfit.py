r"""글자 마스크 → 게임 원시 도형 — **잉크 안에 내접하는** 막대·원·삼각형의 덮음.

`textglyph`의 옛 골격(뼈대 → RDP → 막대 + 이음 원)은 글꼴의 외곽 곡선·폭 변화·
카운터를 먼저 잃고 막대로 되짓는다 — 굵은 글꼴(racing·techno·graffiti)에서
막대 끝이 잉크 밖으로 0.5폭씩 뻗고 이음 원이 카운터를 메운다 (실측: racing
IoU 0.54 · 카운터 침범 68%). 이 모듈은 **래스터를 정답으로 두고** 도형을
잉크 안에 앉힌다:

1. 중심선(세선화)을 따라 폭을 잰다 (거리 변환 = 내접 반지름).
2. 곧은 조각은 막대 — 양 끝은 **잉크가 이어지는 데까지만** 늘린다 (단면 검사).
   폭이 변하면 여러 막대로 가른다 (붓 끝).
3. 굽은 조각은 내접 원의 사슬 — 간격이 곡선 허용 오차를 정한다 (원은 정의상
   카운터를 못 건드린다).
4. 남은 잉크(모서리·세리프·비스듬한 끝)는 잔여 패스가 덮는다 — 성분마다
   막대·직각삼각형·원 후보를 `덮음 − 밖으로 샘 − 카운터 침범`으로 재서
   이득이 있는 것만 받는다.

정책(`FitPolicy`)이 곡선 허용 오차·최소 조각 크기·잔여 패스 수를 정하고,
사다리(`LADDER`)가 고운 것부터 거친 것까지 네 칸을 준다 — 예산에 드는 가장
고운 칸을 고른다 (`textglyph.plan_for_budget`).

전부 결정적이다 (난수 없음, 후보 순서는 좌표 정렬).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import cv2
import numpy as np

from .catalog import Catalog
from .celfit.skeleton import _paths, _prune_spurs, _rdp_idx, _thin
from .model import UNITS_PER_SCALE, Layer


# 도형 이름 — 카탈로그에서 뜻이 고정된 것들 (`catalog.square`·`circle`은 IoU 탐색)
SHAPE_BAR = "A_01"        # 정사각 ±1
SHAPE_TRI_R = "A_04"      # 직각삼각형 (−1,−1)(−1,1)(1,−1) — 직각이 왼쪽 아래
SHAPE_TRI_I = "A_03"      # 이등변삼각형 꼭짓점 (0,0.866)


@dataclass(frozen=True)
class FitPolicy:
    """맞춤 정책 — 전부 **획 폭 대비** 비율이라 글꼴·크기와 무관하다."""

    eps_bar: float = 0.12        # 막대로 볼 곡선 허용 오차 (중심선 이탈, 폭 대비)
    eps_abs: float = 0.006       # 같은 오차의 절대 바닥 (대문자 높이 대비) — 가는 획용
    disc_gap: float = 0.55       # 원 사슬 간격 상한 (반지름 대비) — 실제 간격은 새그로 정한다
    bar_spill: float = 0.05      # 막대 하나가 잉크 밖(1px 여유)·카운터(×6)로 새도 되는 넓이 몫
    min_piece: float = 0.6       # 이보다 짧은 조각은 원으로 (폭 대비)
    join_deg: float = 22.0       # 막대끼리 이 각 넘게 꺾이면 마디에 원
    residual_area: float = 0.06  # 잔여 성분 최소 넓이 (폭² 대비)
    residual_passes: int = 3
    spill_w: float = 1.6         # 잔여 후보 채점 — 밖으로 샌 픽셀 벌점 (덮은 픽셀 1 기준)
    counter_w: float = 6.0       # 카운터 침범 픽셀 벌점
    round_ends: bool = True      # 자유 끝이 둥글면(잉크가 원을 받으면) 원으로 마감


# 층 이름 → 정책. 층 B가 사다리의 끝이다 — 그보다 거친 정책은 곡선을 막대 몇
# 개로 깎아 글자가 상자 덩어리로 읽힌다 (사용자 판정 2026-08-31). 예산이 층 B에도
# 안 들면 게임 글꼴 글리프가 맡는다 (`compose.textbudget`).
TIER_POLICY: dict[str, FitPolicy] = {
    "A": FitPolicy(),
    "B": FitPolicy(eps_bar=0.22, disc_gap=0.8, bar_spill=0.10, residual_area=0.12,
                   residual_passes=2, join_deg=30.0),
}


# 예산 사다리 — 고운 것부터 층 B까지 네 칸. 예산에 드는 것 중 가장 고운 칸을 쓴다.
LADDER: tuple[FitPolicy, ...] = (
    FitPolicy(eps_bar=0.08, disc_gap=0.45, bar_spill=0.03, residual_area=0.04, residual_passes=3),
    TIER_POLICY["A"],
    FitPolicy(eps_bar=0.16, disc_gap=0.65, bar_spill=0.07, residual_area=0.08, residual_passes=3,
              join_deg=26.0),
    TIER_POLICY["B"],
)


@dataclass
class Prim:
    """원시 도형 하나 — **px 좌표(y-down)**. `rot`은 px 평면에서 x축 기준 반시계(도)."""

    shape: str
    x: float
    y: float
    hx: float          # 반폭 (px, 도형 로컬 x의 ±1 → ±hx)
    hy: float
    rot: float = 0.0

    def poly(self, cat: Catalog) -> np.ndarray:
        """px 다각형 (N,2) — `render._draw_layer`와 같은 변환 (로컬 y-up → px y-down)."""
        loop = cat.shapes[self.shape].loops[0]
        r = math.radians(self.rot)
        c, s = math.cos(r), math.sin(r)
        px = loop[:, 0] * self.hx
        py = loop[:, 1] * self.hy
        # 캔버스(y-up) 회전 반시계 → px(y-down)에서는 y를 뒤집는다
        xs = px * c - py * s
        ys = px * s + py * c
        return np.stack([self.x + xs, self.y - ys], axis=1)


def _paint(acc: np.ndarray, prims: list[Prim], cat: Catalog) -> None:
    for p in prims:
        cv2.fillPoly(acc, [np.round(p.poly(cat) * 4).astype(np.int32)], 1,
                     lineType=cv2.LINE_8, shift=2)


def raster(prims: list[Prim], shape: tuple[int, int], cat: Catalog) -> np.ndarray:
    acc = np.zeros(shape, np.uint8)
    _paint(acc, prims, cat)
    return acc.astype(bool)


def _bad_frac(p: Prim, outside: np.ndarray, ctr: np.ndarray, cat: Catalog,
              ctr_w: float = 3.0) -> float:
    """도형 넓이 중 `outside`(잉크 밖)와 `ctr`(카운터, 가중)에 떨어진 몫 — ROI만 그린다."""
    poly = p.poly(cat)
    H, W = outside.shape
    x0, y0 = int(max(0, math.floor(poly[:, 0].min()) - 1)), int(max(0, math.floor(poly[:, 1].min()) - 1))
    x1, y1 = int(min(W, math.ceil(poly[:, 0].max()) + 2)), int(min(H, math.ceil(poly[:, 1].max()) + 2))
    if x1 <= x0 or y1 <= y0:
        return 1.0
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(m, [np.round((poly - (x0, y0)) * 4).astype(np.int32)], 1, shift=2)
    area = max(1, int(m.sum()))
    mb = m.astype(bool)
    return (int((mb & outside[y0:y1, x0:x1]).sum())
            + ctr_w * int((mb & ctr[y0:y1, x0:x1]).sum())) / area


def counters(mask: np.ndarray) -> np.ndarray:
    """잉크에 둘러싸인 빈 자리 (O·P·R·B·A의 속)."""
    inv = (~mask).astype(np.uint8)
    n, cc = cv2.connectedComponents(inv, connectivity=4)
    edge = np.unique(np.concatenate([cc[0], cc[-1], cc[:, 0], cc[:, -1]]))
    out = np.zeros(mask.shape, bool)
    for k in range(1, n):
        if k not in edge:
            out |= cc == k
    return out


# ---------------------------------------------------------------- 중심선 조각

def _section_inside(mask: np.ndarray, x: float, y: float, nx: float, ny: float,
                    r: float, k: int = 7) -> float:
    """점 (x,y)에서 법선 (nx,ny)으로 ±0.8r 단면을 떠서 잉크 안인 몫."""
    H, W = mask.shape
    hit = 0
    for i in range(k):
        t = -0.8 * r + 1.6 * r * i / (k - 1)
        px, py = int(round(x + nx * t)), int(round(y + ny * t))
        if 0 <= px < W and 0 <= py < H and mask[py, px]:
            hit += 1
    return hit / k


def _extend(mask: np.ndarray, p: np.ndarray, d: np.ndarray, r: float, limit: float,
            need: float = 0.85) -> float:
    """끝점 p에서 방향 d로 잉크가 이어지는 길이 (px). 단면의 `need` 이상이 잉크여야 한다."""
    n = (-d[1], d[0])
    best = 0.0
    step = 1.0
    t = step
    while t <= limit:
        q = p + d * t
        if _section_inside(mask, q[0], q[1], n[0], n[1], r) < need:
            break
        best = t
        t += step
    return best


def _bar(p0: np.ndarray, p1: np.ndarray, w: float) -> Prim:
    d = p1 - p0
    L = float(np.hypot(*d))
    ang = math.degrees(math.atan2(-d[1], d[0]))       # px y-down → 캔버스 각
    return Prim(SHAPE_BAR, float((p0[0] + p1[0]) / 2), float((p0[1] + p1[1]) / 2),
                L / 2, w / 2, ang % 360.0)


def _disc(x: float, y: float, r: float, cat: Catalog) -> Prim:
    reach = cat.shapes[cat.circle].reach
    return Prim(cat.circle, float(x), float(y), r / reach, r / reach, 0.0)


def _turn(a, b, c) -> float:
    v1 = b - a
    v2 = c - b
    n1, n2 = np.hypot(*v1), np.hypot(*v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos = max(-1.0, min(1.0, float(np.dot(v1, v2)) / (n1 * n2)))
    return math.degrees(math.acos(cos))


def _skeleton_prims(mask: np.ndarray, dt: np.ndarray, ctr: np.ndarray, cap_px: float,
                    pol: FitPolicy, cat: Catalog) -> tuple[list[Prim], float]:
    """중심선 → 막대·원. 되돌림 (도형들, 중앙 획 폭)."""
    sk = _thin(mask)
    if not sk.any():
        return [], 1.0
    wmed = float(np.median(dt[sk])) * 2.0
    sk = _prune_spurs(sk, max(2.0, wmed * 0.8, 0.07 * cap_px))
    if not sk.any():
        return [], wmed
    H, W = mask.shape
    k3 = np.ones((3, 3), np.uint8)
    outside = ~(cv2.dilate(mask.astype(np.uint8), k3) > 0)

    def r_at(p) -> float:
        rr = min(max(int(round(p[0])), 0), H - 1)
        cc = min(max(int(round(p[1])), 0), W - 1)
        return max(0.75, float(dt[rr, cc]) - 0.25)

    prims: list[Prim] = []
    seen_disc: set[tuple[int, int]] = set()

    def disc_at(p, r):
        key = (int(round(p[0])), int(round(p[1])))
        if key in seen_disc:
            return
        seen_disc.add(key)
        prims.append(_disc(p[1], p[0], r, cat))

    for path, hj, tj in _paths(sk):
        path = path.astype(np.float64)
        rs = np.array([r_at(q) for q in path])
        w = float(np.median(rs)) * 2.0
        eps = pol.eps_bar * max(1.0, w) + pol.eps_abs * cap_px
        idx = _rdp_idx(path, eps)
        # 마디 (y,x) → 조각. 조각마다 막대를 세워 **잉크에 대고 검사한다** — 밖으로
        # 새거나 카운터를 밟으면 폭을 줄여 보고, 그래도 안 되면 반으로 갈라
        # 다시 (깊이 한계 뒤에는 내접 원 사슬). 굵은 글꼴의 굽은 획은 중심선
        # 이탈이 작아도 막대 귀퉁이가 크게 새서, 중심선만 보고는 못 가른다.
        def discs_along(seg, rseg):
            r0 = max(1.0, float(np.median(rseg)))
            gap = max(2, int(round(min(pol.disc_gap * 2.0 * r0,
                                       max(0.4 * r0, math.sqrt(8.0 * r0 * eps))))))
            for q, r in zip(seg[::gap], rseg[::gap]):
                disc_at(q, r)
            disc_at(seg[-1], rseg[-1])

        def piece(seg, rseg, free_a: bool, free_b: bool, depth: int):
            L = float(np.hypot(*(seg[-1] - seg[0])))
            wl = float(np.median(rseg)) * 2.0
            if L < pol.min_piece * wl or len(seg) < 3 or depth >= 4:
                discs_along(seg, rseg)
                return
            a, b = seg[0], seg[-1]
            d = (b - a) / max(1e-6, L)
            pa, pb, dd = np.array([a[1], a[0]]), np.array([b[1], b[0]]), np.array([d[1], d[0]])
            for ws in (wl, 0.85 * wl, 0.7 * wl):
                if ws < 1.5:
                    break
                ea = _extend(mask, pa, -dd, ws / 2, (1.2 if free_a else 0.6) * ws)
                eb = _extend(mask, pb, dd, ws / 2, (1.2 if free_b else 0.6) * ws)
                bar = _bar(pa - dd * ea, pb + dd * eb, ws)
                if _bad_frac(bar, outside, ctr, cat, ctr_w=6.0) <= pol.bar_spill:
                    prims.append(bar)
                    return
                # 폭이 고른 조각은 줄여도 안 낫는다 — 바로 가른다
                if float(rseg.min()) * 2.0 > 0.8 * wl:
                    break
            h = len(seg) // 2
            piece(seg[:h + 1], rseg[:h + 1], free_a, False, depth + 1)
            piece(seg[h:], rseg[h:], False, free_b, depth + 1)

        for k in range(len(idx) - 1):
            i, j = idx[k], idx[k + 1]
            piece(path[i:j + 1], rs[i:j + 1], k == 0 and hj < 0, k == len(idx) - 2 and tj < 0, 0)
        # 꺾이는 마디에 원 (잉크가 둥글면 정확, 각지면 잔여 패스가 모서리를 채운다)
        for k in range(1, len(idx) - 1):
            if _turn(path[idx[k - 1]], path[idx[k]], path[idx[k + 1]]) >= pol.join_deg:
                disc_at(path[idx[k]], rs[idx[k]])
        # 분기점 원 — 가닥이 만나는 틈을 메운다
        for jn, q in ((hj, path[0]), (tj, path[-1])):
            if jn >= 0:
                disc_at(q, r_at(q))
        # 둥근 자유 끝
        if pol.round_ends:
            for jn, q in ((hj, path[0]), (tj, path[-1])):
                if jn < 0 and r_at(q) >= 0.35 * w:
                    disc_at(q, r_at(q))
    return prims, wmed


# ---------------------------------------------------------------- 잔여 패스

def _score(cand: Prim, need: np.ndarray, outside: np.ndarray, ctr: np.ndarray,
           cat: Catalog, pol: FitPolicy) -> float:
    m = raster([cand], need.shape, cat)
    gain = int((m & need).sum())
    spill = int((m & outside).sum())
    intr = int((m & ctr).sum())
    return gain - pol.spill_w * spill - pol.counter_w * intr


def _rect_cands(pts_xy: np.ndarray) -> list[Prim]:
    """성분 픽셀의 최소 외접 사각형 → 막대 후보와 그 반쪽 직각삼각형 넷."""
    (cx, cy), (w, h), ang = cv2.minAreaRect(pts_xy.astype(np.float32))
    w, h = max(w, 1.0) + 0.5, max(h, 1.0) + 0.5
    out = [Prim(SHAPE_BAR, cx, cy, w / 2, h / 2, (-ang) % 360.0),
           Prim(SHAPE_BAR, cx, cy, 0.85 * w / 2, 0.85 * h / 2, (-ang) % 360.0)]
    # 직각삼각형 — A_04의 직각은 로컬 (−1,−1). 사각형의 네 귀퉁이 각각에 직각을 둔다
    for k in range(4):
        out.append(Prim(SHAPE_TRI_R, cx, cy, w / 2, h / 2, (-ang + 90.0 * k) % 360.0))
    return out


def _residual_pass(mask: np.ndarray, dt: np.ndarray, cov: np.ndarray, ctr: np.ndarray,
                   visible: np.ndarray, w: float, cap_px: float, pol: FitPolicy, cat: Catalog
                   ) -> list[Prim]:
    need = mask & ~cov & visible
    if not need.any():
        return []
    outside = ~mask
    k = max(1, int(round(0.12 * w)))
    ker = np.ones((k, k), np.uint8)
    opened = cv2.morphologyEx(need.astype(np.uint8), cv2.MORPH_OPEN, ker).astype(bool)
    n, cc = cv2.connectedComponents(opened.astype(np.uint8), connectivity=8)
    # 가는 글꼴에서는 폭² 기준이 픽셀 몇 개가 된다 — 대문자 높이 몫의 바닥을 둔다
    min_area = pol.residual_area * w * w + (0.03 * cap_px) ** 2
    comps = []
    for c in range(1, n):
        ys, xs = np.nonzero(cc == c)
        if len(xs) < min_area:
            continue
        comps.append((int(ys.min()), int(xs.min()), xs, ys))
    comps.sort()                                     # 결정적 순서 (위 → 왼쪽)
    out: list[Prim] = []
    for _y, _x, xs, ys in comps:
        pts = np.stack([xs, ys], axis=1)
        cands = _rect_cands(pts)
        # 성분 안 가장 깊은 점의 내접 원
        d_in = dt[ys, xs]
        i = int(np.argmax(d_in))
        if d_in[i] >= 0.5:
            cands.append(_disc(float(xs[i]), float(ys[i]), float(d_in[i]) - 0.25, cat))
        best, bs = None, 0.0
        for cnd in cands:
            s = _score(cnd, need, outside, ctr, cat, pol)
            if s > bs:
                best, bs = cnd, s
        if best is not None and bs >= 0.5 * min_area:
            out.append(best)
            need_u = need.astype(np.uint8)
            cv2.fillPoly(need_u, [np.round(best.poly(cat) * 4).astype(np.int32)], 0, shift=2)
            need = need_u.astype(bool)
    return out


# ---------------------------------------------------------------- 맞춤 + 품질

@dataclass
class Fit:
    prims: list[Prim]
    w: float                         # 중앙 획 폭 (px)
    iou: float = 0.0
    spill: float = 0.0               # 도형 픽셀 중 잉크 밖 몫
    counter: float = 0.0             # 카운터 픽셀 중 덮인 몫
    bnd: float = 0.0                 # 경계 오차 (px, 대칭 평균)
    policy: FitPolicy = field(default_factory=FitPolicy)

    @property
    def n(self) -> int:
        return len(self.prims)

    @property
    def quality(self) -> float:
        """한 수로 접은 품질 — IoU에서 경계·카운터·샘을 뺀다 (지배 판정용)."""
        return self.iou - 0.5 * self.counter - 0.3 * self.spill - 0.02 * self.bnd


def _boundary_err(mask: np.ndarray, cov: np.ndarray) -> float:
    ker = np.ones((3, 3), np.uint8)
    bm = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, ker) > 0
    bc = cv2.morphologyEx(cov.astype(np.uint8), cv2.MORPH_GRADIENT, ker) > 0
    if not bm.any() or not bc.any():
        return 99.0
    dm = cv2.distanceTransform((~bm).astype(np.uint8), cv2.DIST_L2, 5)
    dc = cv2.distanceTransform((~bc).astype(np.uint8), cv2.DIST_L2, 5)
    return 0.5 * float(dm[bc].mean()) + 0.5 * float(dc[bm].mean())


def measure(fit: Fit, mask: np.ndarray, cat: Catalog, ctr: np.ndarray | None = None) -> Fit:
    cov = raster(fit.prims, mask.shape, cat)
    ctr = counters(mask) if ctr is None else ctr
    inter = int((cov & mask).sum())
    fit.iou = inter / max(1, int((cov | mask).sum()))
    fit.spill = int((cov & ~mask).sum()) / max(1, int(cov.sum()))
    fit.counter = (int((cov & ctr).sum()) / max(1, int(ctr.sum()))) if ctr.any() else 0.0
    fit.bnd = _boundary_err(mask, cov)
    return fit


def fit_mask(mask: np.ndarray, cap_px: float, cat: Catalog, pol: FitPolicy = FitPolicy(),
             visible: np.ndarray | None = None, measure_it: bool = True) -> Fit:
    """마스크 → 도형. `visible`을 주면 잔여 패스는 그 안만 본다 (테두리·그림자의
    밑에 깔려 안 보이는 속은 성기게 둔다)."""
    if not mask.any():
        return Fit([], 1.0, policy=pol)
    dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    ctr = counters(mask)
    prims, w = _skeleton_prims(mask, dt, ctr, cap_px, pol, cat)
    vis = np.ones(mask.shape, bool) if visible is None else visible
    for _ in range(pol.residual_passes):
        cov = raster(prims, mask.shape, cat)
        more = _residual_pass(mask, dt, cov, ctr, vis, max(2.0, w), cap_px, pol, cat)
        if not more:
            break
        prims += more
    fit = Fit(prims, w, policy=pol)
    return measure(fit, mask, cat, ctr) if measure_it else fit


# ---------------------------------------------------------------- 밑벌 (테두리·그림자)

def grown(p: Prim, g: float) -> Prim:
    """같은 도형을 `g`px 부풀린 사본 — 테두리는 **본색과 같은 도형을 조금 크게 뒤에**
    까는 것이다 (DC 가이드의 눈동자 기법). 삼각형은 중심에서 키우므로 빗변이
    g/√2만 나간다 — 잔여 조각이라 테에서는 안 보인다."""
    return replace(p, hx=p.hx + g, hy=p.hy + g)


def cover(prims: list[Prim], need: np.ndarray, cat: Catalog, min_gain: float
          ) -> list[Prim]:
    """`need`(보이는 자리)를 덮는 데 **제 몫이 있는** 도형만 — 큰 것부터 받고, 새로
    덮는 픽셀이 `min_gain` 아래면 버린다. 밑벌은 대부분 위 벌에 가려지므로
    본색 사본 그대로 깔면 장수만 먹는다 (실측: 테두리가 본색의 0.6배)."""
    order = sorted(range(len(prims)),
                   key=lambda i: (-prims[i].hx * prims[i].hy, prims[i].x, prims[i].y))
    left = need.copy()
    keep: list[int] = []
    for i in order:
        p = prims[i]
        poly = p.poly(cat)
        H, W = need.shape
        x0, y0 = int(max(0, math.floor(poly[:, 0].min()) - 1)), int(max(0, math.floor(poly[:, 1].min()) - 1))
        x1, y1 = int(min(W, math.ceil(poly[:, 0].max()) + 2)), int(min(H, math.ceil(poly[:, 1].max()) + 2))
        if x1 <= x0 or y1 <= y0:
            continue
        m = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillPoly(m, [np.round((poly - (x0, y0)) * 4).astype(np.int32)], 1, shift=2)
        roi = left[y0:y1, x0:x1]
        hit = m.astype(bool) & roi
        if int(hit.sum()) < min_gain:
            continue
        roi[hit] = False
        keep.append(i)
    keep.sort()                                      # 그리기 순서는 본색과 같게
    return [prims[i] for i in keep]


# ---------------------------------------------------------------- 레이어 변환

def to_layers(prims: list[Prim], *, upp: float, origin: tuple[float, float],
              color: tuple[int, int, int], label: str = "text",
              alpha: float = 100.0, dx: float = 0.0, dy: float = 0.0) -> list[Layer]:
    """px 도형 → 캔버스 유닛 레이어. `origin`은 px에서 캔버스 (0,0)이 오는 자리.
    `dx·dy`(유닛)는 그림자 오프셋. 레이어 기울기 축은 안 쓴다 (게임 주입이 못
    쓴다, `celfit.stroke` 문서) — 이탤릭은 글꼴이 갖고 온다."""
    ox, oy = origin
    out: list[Layer] = []
    for p in prims:
        x = (p.x - ox) * upp
        y = -(p.y - oy) * upp
        out.append(Layer(shape=p.shape, x=x + dx, y=y + dy,
                         sx=p.hx * upp / UNITS_PER_SCALE, sy=p.hy * upp / UNITS_PER_SCALE,
                         rot=p.rot % 360.0, color=color, alpha=alpha, label=label))
    return out
