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
    보정 도형 · 잔차 수리 · **수리 몫**   마지막 수단을 몇 번 썼나
    **경계 충실도 p95**               목표 색 경계가 도안에서 몇 px 밀렸나
    **위상** (조각남 · 구멍 늘음)     면이 부스러기로 흩어지지 않았나
    중요도 가중 재현 오차             눈에 띄는 자리에 가중한 색 오차
    바탕 덮음 · 보이는 장/영역        한 형태를 **한 장**이 맡고 있나
    도형 갈아타기 · 쓴 종수           한 면을 같은 종류로 끝냈나
    가려진 짝                         한 형태가 조각으로 끊긴 정도
    **보이는 오차** · 면 안 틀림      경계 양자화를 뺀 색 오차 (아래)

**색 오차에는 자가 둘이다.** 평균·중요도 가중 ΔE는 경계 양자화까지 센다 —
실측(기준판 01·06·08)에서 문턱을 넘는 픽셀의 **96%가 셀 목표의 색 경계에서
1.5px 안**이고, 그 자리는 도형 가장자리가 게임 격자(이동 0.5·스케일 0.01)에
걸려 반 픽셀 어긋난 것이라 인게임 벡터 렌더에는 그렇게 안 나온다. 그것까지
넣으면 "장수를 줄이면 언제나 오차가 는다"로만 읽혀 **구조 개선과 화질 회귀를
못 가른다.** 그래서 경계 띠와 JND 하한을 뺀 `imp_error_seen`·`wrong_far_rate`를
함께 낸다 — 장수를 줄이는 변경은 이 둘로 판정한다.

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
from .geometry import _poly_px

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


def _border_table(cel: CelArt) -> tuple[dict, np.ndarray, np.ndarray]:
    """(접경 길이 dict {(a,b): px} · 영역별 둘레 · 영역 면적) — 4이웃 두 방향.

    `_mark_regions`와 `_latent`가 같은 표를 쓴다 (한 번만 센다).
    """
    lb = cel.labels
    n = int(lb.max()) + 1 if lb.max() >= 0 else 0
    border: dict = {}
    peri = np.zeros(max(n, 1), np.int64)
    area = np.zeros(max(n, 1), np.int64)
    for r in cel.regions:
        area[r.rid] = r.area
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
    return border, peri, area


def _latent(cel: CelArt, border: dict, area: np.ndarray) -> tuple[int, int]:
    """**가려져 갈라진 같은 색 짝**의 수와 그 px — 한 형태가 조각으로
    끊긴 정도를 재는 자.

    한 덩어리였을 것이 앞에 놓인 것에 가려 라벨 지도에서 두 조각으로 끊긴
    자리다 (머리칼이 얼굴에 가려 좌우로 갈린 꼴). 판정은 셋이다:

        ① 두 영역이 서로 **안 닿는다** (접경 0 — 그래프 병합이 못 본다)
        ② 색이 사실상 같다 (ΔE < `_MARK_DE` — 합쳐도 재현 손해가 없다)
        ③ **둘 다 닿는 이웃**이 있고 그 이웃이 둘의 합보다 작다
           (= 넓이 내림차순 그리기 순서에서 **나중에** 그려져 위를 덮는다)

    셋이 서면 그 짝은 한 장으로 그릴 수 있는 자리인데 지금은 조각마다
    따로 근사된다. 여기서는 세기만 한다.
    """
    lb = cel.labels
    n = int(lb.max()) + 1 if lb.max() >= 0 else 0
    if n <= 0:
        return 0, 0
    col = np.zeros((max(n, 1), 3), np.float32)
    live = np.zeros(max(n, 1), bool)
    for r in cel.regions:
        col[r.rid] = r.color
        live[r.rid] = True
    lab_col = cv2.cvtColor(col.reshape(-1, 1, 3).astype(np.uint8),
                           cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    nb: dict = {}
    for k_ in border:
        a_, b_ = divmod(k_, n)
        nb.setdefault(a_, set()).add(b_)
    pairs = set()
    for c_, ns in nb.items():
        if not live[c_]:
            continue
        ns = sorted(x for x in ns if live[x])
        for i in range(len(ns)):
            for j in range(i + 1, len(ns)):
                a_, b_ = ns[i], ns[j]
                if (a_ * n + b_) in border:          # ① 서로 닿으면 아니다
                    continue
                if area[c_] >= area[a_] + area[b_]:  # ③ 이웃이 먼저 그려진다
                    continue
                if float(np.linalg.norm(lab_col[a_] - lab_col[b_])) >= _MARK_DE:
                    continue                          # ②
                pairs.add((a_, b_))
    px = int(sum(min(int(area[a_]), int(area[b_])) for a_, b_ in pairs))
    return len(pairs), px


def _mark_regions(cel: CelArt, border: dict, peri: np.ndarray,
                  area: np.ndarray) -> set[int]:
    """무늬 보호 조각의 영역 id — `celart.marks`와 같은 판정."""
    lb = cel.labels
    n = int(lb.max()) + 1 if lb.max() >= 0 else 0
    if n <= 0 or n > 8192:
        return set()
    col = np.zeros((n, 3), np.float32)
    for r in cel.regions:
        col[r.rid] = r.color
    lab_col = cv2.cvtColor(col.reshape(-1, 1, 3).astype(np.uint8),
                           cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
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


def _topo_one(m: np.ndarray) -> tuple[int, int]:
    """(성분 수, 구멍 수) — 한 겹 여백을 두르고 배경 성분으로 구멍을 센다."""
    u = np.pad(m.astype(np.uint8), 1)
    n_c, _ = cv2.connectedComponents(u, connectivity=8)
    n_b, _ = cv2.connectedComponents((1 - u).astype(np.uint8), connectivity=4)
    return int(n_c - 1), int(max(0, n_b - 2))


def _topology(cel: CelArt, de: np.ndarray, min_area: int = 40) -> dict:
    """영역마다 **제 색으로 칠해진 자리**의 위상을 목표와 견준다.

    성분 수가 늘면 그 면이 조각났다는 뜻이고, 구멍 수가 늘면 면 한가운데가
    뚫렸다는 뜻이다. 둘 다 평균 색 오차로는 안 보이는 결함이다. 셈은 영역
    bbox 안에서만 돌아 값이 싸다.
    """
    lb = cel.labels
    ok = de <= _THR
    split = []
    hole_add = []
    for r in cel.regions:
        if r.area < min_area:
            continue
        x0, y0, x1, y1 = r.bbox
        mt = lb[y0:y1, x0:x1] == r.rid
        mr = mt & ok[y0:y1, x0:x1]
        ct, ht = _topo_one(mt)
        cr, hr = _topo_one(mr)
        if ct <= 0:
            continue
        split.append(cr / ct)
        hole_add.append(hr - ht)
    if not split:
        return {}
    return {"topo_split_med": round(float(np.median(split)), 3),
            "topo_split_mean": round(float(np.mean(split)), 3),
            "topo_hole_add_mean": round(float(np.mean(hole_add)), 3),
            "topo_regions": len(split)}


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
    n_seal = sum(1 for x in labels if x == "seal")
    ntot = max(1, len(plan.layers))
    out: dict = {
        "total_shapes": len(plan.layers),
        "line_shapes": n_ink,
        "fill_shapes": len(plan.layers) - n_ink - n_hole - n_fix - n_seal,
        "correction_shapes": n_hole + n_fix + n_seal,
        "seal_shapes": n_seal,
        # **사후 수리에 얼마를 쓰나** — 메움·수리·봉인이 총장수에서 차지하는 몫.
        # 이 노선의 목표는 "빈 자리를 나중에 때우지 않는다"이므로, 상류 배치가
        # 나아지면 여기가 내려가야 한다 (내려가지 않으면 상류가 아니라 하류를
        # 고친 것이다)
        "repair_layer_ratio": round((n_hole + n_fix + n_seal) / ntot, 4),
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
        # 한 장이 평균 몇 px를 **보이게** 맡나 — 장수당 이득의 자.
        # 중앙값은 부스러기 쪽 꼬리를 못 보고, 평균은 큰 바탕 한 장이 얼마를
        # 맡고 있는지를 함께 담는다 (둘이 벌어지면 조각붙임이다)
        out["visible_area_mean"] = round(float(fv.mean()), 1)
        out["tiny_visible_ratio"] = round(float((fv < 40).mean()), 4)
        out["dead_shape_ratio"] = round(float((fv == 0).mean()), 4)
    ink_i = [i for i, x in enumerate(labels) if x == "ink"]
    if ink_i:
        out["line_visible_med"] = float(np.median(vis[ink_i]))

    # ── **한 영역을 한 장이 얼마나 덮나** (원인 4·5를 재는 자).
    # 영역당 장수(위)는 "몇 장을 샀나"를 세지만, 사람 도안과 갈리는 것은
    # **큰 형태 한 장이 그 면의 몸통을 통째로 맡는가**이다. 조각붙임은
    # 장수가 같아도 이 값이 낮다 — 어느 한 장도 몸통을 못 맡고 저마다
    # 부스러기를 맡기 때문이다. `owner`가 픽셀마다 최종 소유자를 들고
    # 있으므로 (레이어, 영역) 칸마다 보이는 px를 세면 바로 나온다.
    insil0 = cel.labels >= 0
    seen = (owner >= 0) & insil0
    base_cov: list[float] = []
    vis_per_reg: dict[int, int] = {}
    if seen.any():
        nreg_k = int(cel.labels.max()) + 1
        key = owner[seen].astype(np.int64) * nreg_k + cel.labels[seen]
        uk, kc = np.unique(key, return_counts=True)
        top: dict[int, int] = {}
        for k_, c_ in zip(uk.tolist(), kc.tolist()):
            rid_ = k_ % nreg_k
            if labels[k_ // nreg_k] == "ink":
                continue                       # 획은 영역의 몫이 아니다
            vis_per_reg[rid_] = vis_per_reg.get(rid_, 0) + 1
            if c_ > top.get(rid_, 0):
                top[rid_] = c_
        for r in cel.regions:
            if r.rid in top:
                base_cov.append(min(1.0, top[r.rid] / max(1.0, float(r.area))))
    if base_cov:
        bc = np.asarray(base_cov, np.float64)
        out["base_cover_med"] = round(float(np.median(bc)), 4)
        out["base_cover_80"] = round(float((bc >= 0.8).mean()), 4)
    if vis_per_reg:
        out["visible_shapes_per_region"] = round(
            float(np.mean(list(vis_per_reg.values()))), 3)

    # ── **도형 갈아타기** (원인 4) — 한 영역 안에서 이어 놓은 채움 도형이
    # 종류를 바꾸는 비율. 사람은 한 면을 같은 종류 몇 장으로 끝내고, 기계는
    # 잔차를 줍느라 장마다 다른 도형을 고른다 (획 쪽 `family_switch`의 면 판)
    seq: dict[int, list[str]] = {}
    for i, rid in enumerate(reg_of):
        if labels[i] == "ink" or rid < 0:
            continue
        seq.setdefault(rid, []).append(plan.layers[i].shape)
    sw = tot_pair = 0
    for v in seq.values():
        for a_, b_ in zip(v, v[1:]):
            tot_pair += 1
            sw += a_ != b_
    if tot_pair:
        out["fill_family_churn"] = round(sw / tot_pair, 4)
    fam = {plan.layers[i].shape for i, x in enumerate(labels) if x != "ink"}
    out["fill_family_n"] = len(fam)

    # ── **덮여 들어간 몫** — 그린 넓이 대비 보이는 넓이. 1보다 크면 그 장이
    # 뒤 레이어 **밑까지** 뻗어 있다는 뜻이다 (사람이 큰 면을 뒤로 보내고
    # 나중 파츠가 모서리를 덮게 두는 그 자유를 실제로 쓰고 있나)
    if fill_i:
        drawn = np.zeros(len(plan.layers), np.int64)
        for i in fill_i:
            polys = [np.round(q).astype(np.int32)
                     for q in _poly_px(cat, plan.layers[i],
                                       plan.units_per_px, *cel.size)]
            if not polys:
                continue
            xs = np.concatenate([q[:, 0] for q in polys])
            ys = np.concatenate([q[:, 1] for q in polys])
            drawn[i] = max(0, int(xs.max() - xs.min())) *                 max(0, int(ys.max() - ys.min()))
        ok = (vis[fill_i] > 0) & (drawn[fill_i] > 0)
        if ok.any():
            out["fill_bbox_over_visible_med"] = round(float(np.median(
                drawn[fill_i][ok] / vis[fill_i][ok])), 3)

    # ── 작은 중요 영역 보존 — 무늬 보호 조각 중 도형을 실제로 받은 몫
    bord, peri_t, area_t = _border_table(cel)
    out["latent_pairs"], out["latent_px"] = _latent(cel, bord, area_t)
    marks = _mark_regions(cel, bord, peri_t, area_t)
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

    # ── **보이는 오차** — 경계 양자화를 빼고 남는 색 오차.
    # 실측(B0 01·06·08): ΔE가 문턱을 넘는 픽셀의 **96%가 셀 목표의 색 경계에서
    # 1.5px 안**이다. 그 자리는 도형의 가장자리가 게임 격자(이동 0.5·스케일
    # 0.01)에 걸려 반 픽셀 어긋난 것이고, 인게임은 벡터라 화면에 그렇게 안
    # 나온다. 평균 ΔE에 그것을 넣으면 "장수를 줄이면 언제나 오차가 는다"로만
    # 읽혀 **구조 개선과 화질 회귀를 못 가른다.** 여기서는 경계 띠(±1px)를
    # 빼고, JND 하한(`marks._MARK_DE`) 아래도 빼고 남는 **면 안의 오차**를 낸다.
    flat = cel.flat_render()
    fl = cv2.cvtColor(flat, cv2.COLOR_RGB2LAB).astype(np.float32)
    ce = np.zeros(de.shape, bool)
    dh = np.linalg.norm(fl[:, :-1] - fl[:, 1:], axis=2) > _MARK_DE
    ce[:, :-1] |= dh
    ce[:, 1:] |= dh
    dv_ = np.linalg.norm(fl[:-1] - fl[1:], axis=2) > _MARK_DE
    ce[:-1] |= dv_
    ce[1:] |= dv_
    seen = (cel.labels >= 0) & ~cv2.dilate(
        ce.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    de_seen = np.where(de >= _MARK_DE, de, 0.0)
    out["wrong_far_px"] = int(((de > _THR) & seen).sum())
    out["wrong_far_rate"] = round(float(out["wrong_far_px"]
                                        / max(int(seen.sum()), 1)), 5)
    if value is not None and seen.any():
        vs = value[seen].astype(np.float64)
        out["imp_error_seen"] = round(float((vs * de_seen[seen]).sum()
                                            / max(vs.sum(), 1e-9)), 4)

    # ── **경계 충실도** — 셀 목표의 색 경계가 도안에서 몇 px 밀렸나 (p95).
    # 평균 ΔE는 경계 양자화에 먹혀 이 축을 못 본다 (위 "보이는 오차" 문서).
    # 여기서는 자리를 직접 잰다: 목표 경계 픽셀마다 **가장 가까운 도안 경계**
    # 까지의 거리를 재고 그 분포의 95%를 낸다. 밀림이 한 겹(격자)이면 1 근처,
    # 면이 통째로 어긋났으면 크게 뜬다
    rb = np.zeros(de.shape, bool)
    rd = np.linalg.norm(render[:, :-1].astype(np.float32)
                        - render[:, 1:].astype(np.float32), axis=2) > 0
    rb[:, :-1] |= rd
    rb[:, 1:] |= rd
    rv = np.linalg.norm(render[:-1].astype(np.float32)
                        - render[1:].astype(np.float32), axis=2) > 0
    rb[:-1] |= rv
    rb[1:] |= rv
    tb = ce & (cel.labels >= 0)            # 목표 색 경계 (위에서 이미 쟀다)
    if tb.any() and rb.any():
        dist = cv2.distanceTransform((~rb).astype(np.uint8), cv2.DIST_L2, 5)
        out["boundary_d95"] = round(float(np.percentile(dist[tb], 95)), 3)
        out["boundary_d_med"] = round(float(np.median(dist[tb])), 3)

    # ── **위상** — 면이 조각나거나 구멍이 뚫렸나. 색 오차는 "얼마나 틀렸나"만
    # 재고 "몇 조각으로 갈렸나"는 못 잰다. 영역마다 제 색으로 제대로 칠해진
    # 자리(`de <= _THR`)를 놓고 성분 수와 구멍 수를 목표와 견준다 — 사람 도안이
    # 한 덩이로 유지하는 형태를 기계가 부스러기로 흩는지가 여기서 보인다
    out.update(_topology(cel, de, min_area=40))

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
