"""§13 회귀 지표 — **도형 수도 품질이고, 어디서 줄었는지가 답이다.**

커버리지·구멍·RMSE만으로는 이 노선의 물음에 답이 안 된다. "같은 품질이면 더
적은 도형"이 목표인데 평균 오차는 도형을 더 쓰면 언제나 내려가기 때문이다.
여기서 내는 것은 **구조**다:

    총 도형 · 채움 · 선 · 보정        어디에 몇 장을 썼나
    영역당 도형 · 1장 · ≤2장 비율     한 영역을 몇 장으로 끝냈나
    보이는 넓이 · 부스러기 도형 몫    한 장이 실제로 **보이는** 넓이 (덮인 몫 제외)
    의미 영역 수 · 병합 수            분해가 무엇을 한 덩어리로 봤나
    작은 중요 영역 보존               눈·코 그림자·하이라이트가 살아남았나
    경계 넘김률                       색이 선을 넘어간 자리의 비율
    선/색면 틈 · 반대색 슬리버        맞물림이 어긋난 자리
    보정 도형 · 잔차 수리             마지막 수단을 몇 번 썼나
    중요도 가중 재현 오차             눈에 띄는 자리에 가중한 색 오차

이 표를 판끼리 나란히 세워 본다. 고정된 감축 목표를
먼저 정하지 않는다 — 기준판을 재고, 회귀 전체에서 **품질과 위상을 지키면서**
중앙값 도형 수가 실제로 줄었는지를 본다.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..catalog import Catalog
from ..celart import CelArt
from ..celart.marks import _MARK_DE, _MARK_RATIO
from ..model import LayerPlan
from . import residual

# "색이 틀렸다"의 문턱 — `residual._THR`과 같은 자
_THR = residual._THR
# 슬리버 판정 — 최대 내접 반경이 이보다 얇으면 실오라기다 (최소 도형 반폭 근방)
_SLIVER_R = 2.0


def _region_colors(cel: CelArt) -> tuple[np.ndarray, np.ndarray]:
    """(영역 id → RGB lut, 살아 있는 id 배열)."""
    n = int(cel.labels.max()) + 1 if cel.labels.max() >= 0 else 0
    lut = np.zeros((max(n, 1), 3), np.float32)
    ids = []
    for r in cel.regions:
        lut[r.rid] = r.color
        ids.append(r.rid)
    return lut, np.asarray(ids, np.int64)


def _crossing(cel: CelArt, render: np.ndarray, de: np.ndarray,
              lut: np.ndarray, ids: np.ndarray) -> tuple[int, int]:
    """(색이 남의 영역 색으로 칠해진 px, 그 판정을 잰 띠의 px).

    "색이 선을 넘지 않는다"를 그대로 잰다: 제 영역 색과 ΔE가 문턱을 넘는데
    **다른 영역의 색과는 뚜렷이 가까운** 픽셀을 센다. 자리는 영역 경계 띠로
    한정하되 **선 밑은 뺀다** — 그 자리의 목표는 원화의 선 색이지 어느 면의
    색도 아니라서, 넣으면 획의 색 오차가 통째로 "넘김"으로 잡힌다. 면
    한가운데의 얼룩도 넘김이 아니라 잔차이므로 띠 밖은 안 센다.
    """
    lb = cel.labels
    h, w = lb.shape
    bnd = np.zeros((h, w), bool)
    bnd[:, :-1] |= lb[:, :-1] != lb[:, 1:]
    bnd[:, 1:] |= lb[:, :-1] != lb[:, 1:]
    bnd[:-1] |= lb[:-1] != lb[1:]
    bnd[1:] |= lb[:-1] != lb[1:]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    band = cv2.dilate(bnd.astype(np.uint8), k).astype(bool) & (lb >= 0)
    if cel.line_mask is not None:
        band &= ~cel.line_mask
    cand = band & (de > _THR)
    total = int(band.sum())
    if not cand.any() or not len(ids):
        return 0, total
    pal = np.unique(lut[ids].astype(np.float32), axis=0)
    px = render[cand].astype(np.float32)
    own = lut[lb[cand]].astype(np.float32)
    d_own = np.linalg.norm(px - own, axis=1)
    best = np.full(len(px), np.inf, np.float32)
    step = 1 << 18
    for i in range(0, len(px), step):
        d = px[i:i + step, None, :] - pal[None, :, :]
        best[i:i + step] = np.sqrt((d * d).sum(2)).min(1)
    # 여유 6 = 문턱 ΔE의 절반 — "가깝다"가 팔레트가 촘촘해서 생긴 우연이
    # 아니라 실제로 남의 색을 칠한 자리여야 한다
    return int((best < d_own - 0.5 * _THR).sum()), total


def _mark_regions(cel: CelArt) -> set[int]:
    """무늬 보호 조각의 영역 id — `celart.marks`와 같은 판정."""
    lb = cel.labels
    n = int(lb.max()) + 1 if lb.max() >= 0 else 0
    if n <= 0 or n > 8192:
        return set()
    area = np.zeros(n, np.int64)
    col = np.zeros((n, 3), np.float32)
    for r in cel.regions:
        area[r.rid] = r.area
        col[r.rid] = r.color
    lab_col = cv2.cvtColor(col.reshape(-1, 1, 3).astype(np.uint8),
                           cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    border = {}
    peri = np.zeros(n, np.int64)
    for a, b in ((lb[:, :-1], lb[:, 1:]), (lb[:-1], lb[1:])):
        sel = (a >= 0) & (b >= 0) & (a != b)
        if not sel.any():
            continue
        key = a[sel].astype(np.int64) * n + b[sel]
        key = np.concatenate([key, b[sel].astype(np.int64) * n + a[sel]])
        uk, kc = np.unique(key, return_counts=True)
        for k_, c_ in zip(uk.tolist(), kc.tolist()):
            border[k_] = border.get(k_, 0) + c_
            peri[k_ // n] += c_
    best: dict[int, tuple[int, int]] = {}
    for k_, c_ in border.items():
        a_, b_ = divmod(k_, n)
        if c_ > best.get(a_, (0, -1))[0]:
            best[a_] = (c_, b_)
    out = set()
    for a_, (_, b_) in best.items():
        if area[a_] <= 0 or peri[a_] > 0.45 * area[a_]:
            continue
        de = float(np.linalg.norm(lab_col[a_] - lab_col[b_]))
        if de >= _MARK_DE and area[b_] >= _MARK_RATIO * area[a_]:
            out.add(int(a_))
    return out


def plan_metrics(plan: LayerPlan, cel: CelArt, cat: Catalog, *,
                 value: np.ndarray | None = None, price: float = 0.0,
                 min_px: int = 4, extra: dict | None = None) -> dict:
    """§13 지표 한 벌 — report의 `structure` 칸에 그대로 실린다."""
    owner, reg_of = residual.owner_map(plan, cel, cat, with_regions=True)
    res = residual.analyze(plan, cel, cat, value=value, price=price,
                           min_px=min_px, owner=owner)
    de = res["de"]
    lut = np.zeros((len(plan.layers) + 1, 3), np.uint8)
    lut[0] = 255
    for i, l in enumerate(plan.layers):
        lut[i + 1] = l.rgb()
    render = lut[owner + 1]

    labels = [l.label for l in plan.layers]
    n_ink = sum(1 for x in labels if x == "ink")
    n_hole = sum(1 for x in labels if x == "hole")
    n_fix = sum(1 for x in labels if x == "fix")
    out: dict = {
        "total_shapes": len(plan.layers),
        "line_shapes": n_ink,
        "fill_shapes": len(plan.layers) - n_ink - n_hole - n_fix,
        "correction_shapes": n_hole + n_fix,
    }

    # ── 영역당 도형 수 — **면 도형만** 센다 (획은 영역의 몫이 아니다)
    per: dict[int, int] = {}
    for i, rid in enumerate(reg_of):
        if labels[i] == "ink" or rid < 0:
            continue
        per[rid] = per.get(rid, 0) + 1
    drawn = [v for v in per.values()]
    nreg = len(cel.regions)
    out.update({
        "semantic_regions": nreg,
        "regions_drawn": len(drawn),
        "shapes_per_region": round(float(np.mean(drawn)), 3) if drawn else 0.0,
        "one_shape_ratio": round(sum(1 for v in drawn if v == 1)
                                 / max(1, len(drawn)), 4),
        "two_or_less_ratio": round(sum(1 for v in drawn if v <= 2)
                                   / max(1, len(drawn)), 4),
    })

    # ── **보이는 넓이** — 한 장이 최종 화면에서 실제로 차지하는 px.
    # 장수만 세면 "큰 한 장"과 "덮여 사라진 한 장"이 같아 보인다. 사람 도안은
    # 한 장 한 장이 제 몫의 면을 맡는데(가려질 것은 아예 안 그린다), 기계는
    # 잔차를 줍다 보면 **거의 안 보이는 조각**을 사게 된다 — 그 몫이 곧
    # "작은 patch를 여러 장 이어 붙인 느낌"의 자다. `owner`가 이미 그 답을
    # 들고 있어(픽셀마다 최종 소유 레이어) 새 라스터가 없다
    vis = np.bincount(owner[owner >= 0].ravel(),
                      minlength=len(plan.layers)).astype(np.int64)
    fill_i = [i for i, x in enumerate(labels) if x != "ink"]
    if fill_i:
        fv = vis[fill_i]
        out["visible_area_med"] = float(np.median(fv))
        out["tiny_visible_ratio"] = round(float((fv < 40).mean()), 4)
        out["dead_shape_ratio"] = round(float((fv == 0).mean()), 4)
    ink_i = [i for i, x in enumerate(labels) if x == "ink"]
    if ink_i:
        out["line_visible_med"] = float(np.median(vis[ink_i]))

    # ── 작은 중요 영역 보존 — 무늬 보호 조각 중 도형을 실제로 받은 몫
    marks = _mark_regions(cel)
    out["small_important_regions"] = len(marks)
    out["small_important_kept"] = round(
        sum(1 for r in marks if per.get(r, 0) > 0) / max(1, len(marks)), 4)

    # ── 맞물림 — 경계 넘김 · 선/색면 틈 · 반대색 슬리버
    rlut, ids = _region_colors(cel)
    cross, band = _crossing(cel, render, de, rlut, ids)
    out["boundary_crossing_rate"] = round(cross / max(1, band), 5)
    out["boundary_crossing_px"] = cross
    out["line_gap_px"] = res["res_gap_px"]
    wrongm = (owner >= 0) & (cel.labels >= 0) & (de > _THR)
    sliver = 0
    if wrongm.any():
        u = wrongm.astype(np.uint8)
        dt = cv2.distanceTransform(u, cv2.DIST_L2, 3)
        n, cc, stt, _ = cv2.connectedComponentsWithStats(u, connectivity=8)
        mx = np.zeros(n, np.float32)          # 성분별 최대 내접 반경 (한 패스)
        np.maximum.at(mx, cc.ravel(), dt.ravel())
        keep = (stt[:, cv2.CC_STAT_AREA] >= min_px) & (mx <= _SLIVER_R)
        keep[0] = False
        sliver = int(keep.sum())
    out["wrong_color_slivers"] = sliver

    # ── 중요도 가중 재현 오차 — 값 맵으로 가중한 ΔE (눈에 띄는 자리 우선)
    insil = cel.labels >= 0
    if value is not None:
        v = value[insil].astype(np.float64)
        out["imp_error"] = round(float((v * de[insil]).sum()
                                       / max(v.sum(), 1e-9)), 3)
    out["mean_de"] = round(float(de[insil].mean()), 3)
    for k in ("res_hole_px", "res_gap_px", "res_boundary_px", "res_wrong_px",
              "res_leak_px", "res_clusters", "res_tiny"):
        out[k] = res[k]
    if extra:
        out.update(extra)
    return out
