"""진입점 — CelArt 하나를 LayerPlan 하나로.

`fit_plan`은 셀 영역을 면으로 채우고(선 지도가 붙어 있으면 획도 함께 놓는다),
`fit_line_plan`은 획만 놓는다. 두 노선이 같은 기계를 쓰고 갈리는 곳은 그
문서에 적혀 있다. 레이어 수는 **가격이 정한다** — 예산을 채우지 않는다.
"""

from __future__ import annotations

import itertools
import os

import cv2
import numpy as np

from ...i18n import msg
from ..catalog import Catalog
from ..celart import CelArt, mark_mask
from ..model import Layer, LayerPlan
from ..price import _PRICE_INK
from . import census as _census
from .fill import _fit_bars, region_shape
from .geometry import _ink_cover, _min_span
from .layered import fill_region, grow_fill, mop_up
from .lines import _fit_lines, recolor_strokes
from .scoring import (_COVER_STOP, _MAX_PER_REGION, _PEN_FORBID_FILL,
                      _PEN_WASTE_FILL, _Scorer)
from .stroke import _CURVE_STATS, _stroke_forms
from .vocabulary import _FILL_SHAPES, _FILL_WIN, _check_vocab

# 아직 한 장도 못 받은 영역에서 마무리가 줍는 덩어리의 하한 (px).
# "보이는 구멍"(route_cel.HOLE_MIN_PX = 4px 군집)의 넓이 급이다 —
# 그보다 작은 조각은 봉인이 군집당 한 장으로 더 싸게 처리한다.
_MOP_MIN_FIRST = int(os.environ.get("FS_MOP_MIN_FIRST", 12))
# 획형 판정의 색·경계 자 (`_inklike`). 둘레의 이만큼이 잉크에 붙어 있으면
# 획 사이의 잔여로 보고, 색이 그 자리 획 색과 이 ΔE 안이면 그 선의 잔여로 본다.
# ΔE 문턱은 무늬 보호 조각의 JND(`celart.marks._MARK_DE`)의 세 배 — "같은
# 색"이 아니라 "그 선 색"을 묻는 자리라 넉넉해야 한다.
_INK_EDGE = float(os.environ.get("FS_THIN_INKEDGE", 0.6))
_INK_DE = float(os.environ.get("FS_THIN_INKDE", 12.0))


def _inklike(mask: np.ndarray, color, src_rgb, ink, roi) -> bool:
    """이 가는 영역이 **그 자리 획의 잔여**인가 — 색과 경계로 묻는다.

    선화가 안 지운 선 조각은 색이 그 자리 획의 색이고 (선화가 본 그 선이니까)
    둘레가 놓인 잉크에 붙어 있다. 눈 흰자·하이라이트·리본은 색이 획과 또렷이
    다르고 둘레가 **색 경계**다. 둘 중 하나라도 획 쪽이면 획으로 본다 —
    둘 다 아닐 때만 면으로 돌린다 (보수적인 쪽이 잘못 그려도 색만 한 겹
    틀리고, 반대로 틀리면 그 색이 통째로 사라진다).

    지도가 없으면(고전 폴백·색 표본 없음) True — 종전 판정 그대로다.
    """
    if ink is None or src_rgb is None:
        return True
    x0, y0, x1, y1 = roi
    ink_roi = ink[y0:y1, x0:x1]
    k3 = np.ones((3, 3), np.uint8)
    m8 = mask.astype(np.uint8)
    edge = mask & ~cv2.erode(m8, k3).astype(bool)
    near_ink = cv2.dilate(ink_roi.astype(np.uint8), k3).astype(bool)
    # ③ 경계 지지 — 둘레가 잉크에 붙어 있나 (붙어 있으면 획 사이의 잔여다)
    if edge.any() and float(near_ink[edge].mean()) >= _INK_EDGE:
        return True
    # ④ 제 색 — 이웃한 잉크 밑의 원화 색과 견준다. 가까우면 그 선의 잔여다
    nb = cv2.dilate(m8, k3, iterations=2).astype(bool) & ink_roi & ~mask
    if int(nb.sum()) < 16:
        return True                        # 견줄 잉크가 없다 — 종전 판정대로
    src = src_rgb[y0:y1, x0:x1]
    a = np.array([[color]], np.uint8)
    b = np.median(src[nb], axis=0).round().astype(np.uint8).reshape(1, 1, 3)
    la = cv2.cvtColor(a, cv2.COLOR_RGB2LAB).astype(np.float32).ravel()
    lb = cv2.cvtColor(b, cv2.COLOR_RGB2LAB).astype(np.float32).ravel()
    return float(np.linalg.norm(la - lb)) < _INK_DE


def fit_plan(cel: CelArt, cat: Catalog, *, budget: int = 3000,
             line_budget: int | None = None,
             source_image: str = "", log=print, progress=None,
             value: np.ndarray | None = None,
             price: float = 0.0,
             ink_free: np.ndarray | None = None,
             sid_start: int = 0) -> tuple[LayerPlan, dict]:
    """CelArt → LayerPlan + 통계. 결정적.

    `value`·`price`를 주면 **가격 설계**이다 — 한 장이 새로 맞히는 값이 λ에
    못 미치면 안 산다 (`price._PRICE_REL` 문서). 안 주면 값을 안 묻는다.

    `ink_free`는 **밖에서 이미 배치된 획**의 커버 지도다 (cel 노선 — 선
    도안이 먼저 서고 이 함수는 면만 채울 때. `cel.line_mask`는 None으로
    온다). 의미는 안에서 계산하는 지도와 같다 — 획이 덮는 자리는 면 채점의
    공짜 (`_Scorer`의 ink 지도).

    `sid_start`는 **획 그룹 id의 시작 번호**다. cel 노선은 선 도안을 먼저
    딴 판(`fit_line_plan`)과 이 함수를 **따로** 부르고 두 판의 레이어를 한
    플랜에 합치는데, 둘이 각자 0부터 번호를 매기면 **가는 잔여 막대 사슬
    (`_fit_bars`)과 획이 같은 그룹 번호를 쓴다.** `Layer.stroke`는 프루닝의
    원자 단위라(같은 값이면 통째로 살거나 죽는다) 겹치면 상관없는 두 무리가
    한 덩이로 잘리고, 구조 지표(`linemetrics`)도 남의 도형을 그 획의 마디로
    읽는다 (실측 M1-01: 획 539개 중 46개가 막대 사슬과 번호가 겹쳤다).
    """
    _check_vocab(cat, log)
    w, h = cel.size
    upp = 900.0 / h                      # painter와 같은 캔버스 배율
    plan = LayerPlan(source_image=source_image, image_size=(w, h),
                     units_per_px=upp)

    # 그리기 순서 지도: 픽셀 → 영역 순번 (배경 = -1)
    order_of = {r.rid: i for i, r in enumerate(cel.regions)}
    order_img = np.full(cel.labels.shape, -1, np.int32)
    pos = cel.labels >= 0
    lut = np.full(int(cel.labels.max()) + 1, -1, np.int32)
    for rid, o in order_of.items():
        lut[rid] = o
    order_img[pos] = lut[cel.labels[pos]]

    _grown: set = set()   # 계측 — 늘어난 장의 인덱스 (호출부가 pop)
    stats = {"regions": len(cel.regions), "skipped": 0, "uncovered_px": 0,
             "grown_fill": 0,
             "fill_layers": 0, "bar_layers": 0, "mop_layers": 0, "line_layers": 0,
             "big10_layers": 0, "cap_hit": 0, "share_hit": 0,
             # 획형으로 간 영역 · 도형을 한 장도 못 받은 영역 (P0 계측)
             "stroke_regions": 0, "stroke_px": 0,
             "empty_regions": 0, "empty_px": 0}
    # 유예 덮개 — 순이득이 λ×_FIX_DE_REPAIR~λ×_FIX_DE 구간인 획 덮개.
    # 지금 사면 포화 장에서 재컷이 채움을 밀어내므로(2단 수리와 같은 실측),
    # 배치·메움·수리가 끝나 예산 잔여가 확정된 뒤 파이프라인이 남는 만큼만
    # 산다. stats에 실어 나가되 report에는 안 남는다 (파이프라인이 pop)
    carve_defer: list = []
    stats["_carve_defer"] = carve_defer
    for k in _CURVE_STATS:
        _CURVE_STATS[k] = 0
    _FILL_WIN.clear()
    forms = _stroke_forms(cat)            # 획 어휘의 중심선 (프로세스 1회 계측)
    sids = itertools.count(sid_start)     # 획 그룹 id 발급기 (한 경로 = 한 획)

    # 신경망 선화가 있으면 **선 예산을 먼저 확보**한다 (선이 시각 품질의 상한).
    # 획 레이어는 그리기 순서상 맨 뒤(모든 면 위)로 가야 하므로 따로 모았다가
    # 영역 채움이 끝난 뒤 붙인다 — 사람의 마지막 선따기 순서와 같다
    line_layers: list[Layer] = []
    ink_cov: np.ndarray | None = ink_free
    if cel.line_mask is not None:
        lp = LayerPlan(image_size=(w, h), units_per_px=upp)
        # 선 예산은 **최종 상한 기준**으로 받는다 (fit 여유 배수에 비례시키면
        # 프루닝이 채움을 학살한다 — 실측 RMSE 74)
        lb = line_budget if line_budget is not None else budget // 3
        # 선화 모델이 없을 때의 폴백 — 선·면 동시 배치라 덮개가 산다.
        # 선 재구성 자체는 **같은 엔진**이고 정책만 갈린다
        from .policy import CEL_FALLBACK

        line_st: dict = {}
        n_line = _fit_lines(lp, cel, cat, upp, lb, forms, log,
                            sids=sids, pol=CEL_FALLBACK, stats=line_st,
                            value=value, price=price * _PRICE_INK,
                            carve_defer=carve_defer if price else None)
        stats["_rec"] = line_st.pop("_rec", None)
        stats.update({k: v for k, v in line_st.items() if k not in stats})
        line_layers = lp.layers
        # 폴백도 같은 손 — 획 색은 발자국 아래 원화 평균 (`recolor_strokes`)
        stats["recolored_strokes"] = recolor_strokes(lp, cel, cat, upp, log=log)
        stats["line_layers"] = n_line
        budget = budget - n_line
        # 획이 덮는 자리 — 면 배치에서 공짜다 (`_Scorer` 문서). 획은 이미
        # 전부 놓였으므로 지도가 확정이다
        if line_layers:
            ink_cov = _ink_cover(line_layers, cat, upp, w, h)
            stats["ink_free_px"] = int(ink_cov.sum())
    # §9 이음 당김의 예외 — 무늬 보호 조각 (`celart.marks`). 큰 면이 눈
    # 흰자·코 그림자를 1px씩 먹는 것이 이 당김의 유일한 해악이라 그 자리만 뺀다
    protect = mark_mask(cel)
    total = len(cel.regions)
    # 남은 영역 면적 합 (뒤부터 누적) — 영역별 예산 = 남은 예산 × 면적 비중.
    # 큰 영역이 예산을 독식해 후순위 영역이 통째로 빠지는 것을 막는다
    # (실측: 비례 배분 없이는 영역의 3/4이 통째로 못 그려졌다)
    suffix_area = np.cumsum([r.area for r in cel.regions][::-1])[::-1].astype(float)

    for oi, reg in enumerate(cel.regions):
        if progress:
            progress(oi / total, msg("영역 {cur}/{total}", cur=oi + 1, total=total))
        left = budget - len(plan.layers)
        if left <= 0:
            stats["skipped"] = total - oi
            log(msg("  경고: 예산 소진 — 영역 {n}개 못 그림", n=total - oi))
            break
        share = int(1.6 * left * reg.area / suffix_area[oi]) + 2

        x0, y0, x1, y1 = reg.bbox
        # ROI = bbox + 가드 여유 전체 — 도형이 닿을 수 있는 픽셀은 전부 채점판
        # 안에 둔다. bbox+8이던 시절 가드 여유(24+0.25×변)가 무벌점 구간이라
        # 채움이 그리로 뻗었다 (목을 가로지르는 타원이 그렇게 나왔다).
        # 가드는 8px로 좁힌다
        m = 24 + int(0.25 * max(x1 - x0, y1 - y0))
        x0 = max(0, x0 - m); y0 = max(0, y0 - m)
        x1 = min(w, x1 + m); y1 = min(h, y1 + m)
        roi = (x0, y0, x1, y1)
        omap = order_img[y0:y1, x0:x1]
        mask = cel.labels[y0:y1, x0:x1] == reg.rid
        # 획형 판정 (`fill.region_shape`) — 가늘고 · 길고 · 안 닫혔고 ·
        # **획 색이어야** 획이다. 가늘기만 보면 눈 흰자·하이라이트 같은 가는
        # **면**이 획 쪽으로 넘어가, 경로 길이 문턱에 걸려 통째로 안 그려진다
        dt0 = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
        _ilk = _inklike(mask, reg.color, cel.src_rgb, ink_cov, roi)
        strokelike, _wmed, _elong = region_shape(mask, dt0, _ilk)
        if strokelike:
            stats["stroke_regions"] += 1
            stats["stroke_px"] += int(reg.area)
        if reg.area >= 100 and not strokelike:
            # 닫기→열기로 1px대 요철·목을 정리 — 사람이 안 그리는 미세 위글에
            # 도형을 쓰지 않는다 (선·작은 영역은 뭉개질 수 있어 건너뛴다)
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            m8 = mask.astype(np.uint8)
            m8 = cv2.morphologyEx(cv2.morphologyEx(m8, cv2.MORPH_CLOSE, k),
                                  cv2.MORPH_OPEN, k)
            if m8.any():
                mask = m8.astype(bool)
        forbid = omap < oi                 # 먼저 그린 면 + 배경(-1)
        sc = _Scorer(cat, upp, w, h, roi, mask, forbid, omap < 0, guard=8.0,
                     pen_waste=_PEN_WASTE_FILL,
                     pen_forbid=_PEN_FORBID_FILL,
                     ink=ink_cov[y0:y1, x0:x1] if ink_cov is not None else None,
                     val=value[y0:y1, x0:x1] if value is not None else None,
                     seam=True,
                     protect=protect[y0:y1, x0:x1] if protect is not None else None,
                     silcap=True)

        n_reg = 0
        lo = len(plan.layers)              # 이 영역이 놓기 시작하는 자리
        cap = min(_MAX_PER_REGION, left, share)
        area = float(reg.area)
        # 계측 (`census`) — 영역 하나의 계보 칸을 연다. 끄면 아무 일도 안 한다
        if _census.ON:
            m8c = mask.astype(np.uint8)
            peri = int((m8c & ~cv2.erode(m8c, np.ones((3, 3), np.uint8))).sum())
            cr = _census.begin_region(
                rid=int(reg.rid), order=int(oi), area=int(reg.area),
                bbox=[int(v) for v in reg.bbox], roi=[int(v) for v in roi],
                color=[int(v) for v in reg.color], strokelike=bool(strokelike),
                wmed=round(float(_wmed), 3), elong=round(float(_elong), 3),
                peri=peri, mask_px=int(m8c.sum()), cap=int(cap),
                share=int(share), lo=int(lo))
            cr["compact"] = round(4.0 * np.pi * int(m8c.sum())
                                  / max(peri * peri, 1.0), 4)
            # 위상·오목함·최대 내접 반경 — "작아서 못 그리나, 모양 때문인가"를
            # 가르는 자 (§14·§15). 게임 최소 도형 반폭도 같이 실어 둔다
            cr.update(_census.region_geom(mask, dt0))
            cr["min_span"] = round(float(_min_span(upp)), 3)
            cr["inklike"] = bool(_ilk)
            if ink_cov is not None:
                near = cv2.dilate(ink_cov[y0:y1, x0:x1].astype(np.uint8),
                                  np.ones((3, 3), np.uint8)).astype(bool)
                cr["ink_near"] = round(float(near[mask].mean()), 4)
            cr["_dedge"] = cv2.distanceTransform(m8c, cv2.DIST_L2, 3)
            if ink_cov is not None:
                cr["_ink"] = ink_cov[y0:y1, x0:x1]
            if value is not None:
                cr["val_mean"] = round(float(value[y0:y1, x0:x1][mask].mean()), 3)
            _census.res_mark("start", int(np.count_nonzero(sc.residual)))
        if not strokelike:
            # 면 채움 — 바탕 한 장 먼저, 남은 것은 후보 경쟁 (§7·§8·§11)
            n_reg = fill_region(plan, sc, cat, reg.color, cap, area, price,
                                _COVER_STOP)
            stats["fill_layers"] += n_reg
        if _census.ON:
            _census.res_mark("after_fill", int(np.count_nonzero(sc.residual)))
            cr["fill_layers"] = int(n_reg)
        # **사기 전에 늘린다** — 경계 부스러기를 이미 놓은 도형의 한 스텝
        # 확장으로 먼저 먹는다 (레이어 0장). 그러고도 남는 것만 막대·마무리가
        # 산다 (`layered.grow_fill` 문서)
        if n_reg:
            stats["grown_fill"] += grow_fill(sc, plan.layers, lo, grown=_grown)
        if _census.ON:
            _census.res_mark("after_grow", int(np.count_nonzero(sc.residual)))
        # 가는 잔여 → 획 사슬
        if n_reg < cap and np.count_nonzero(sc.residual) > (1.0 - _COVER_STOP) * area:
            dt = cv2.distanceTransform(sc.residual.astype(np.uint8), cv2.DIST_L2, 3)
            n_bar = _fit_bars(plan, sc, dt, reg.color, cap - n_reg, forms,
                              ink=strokelike, sids=sids, price=price,
                              free_first=n_reg == 0)
            stats["bar_layers"] += n_bar
            n_reg += n_bar
        if _census.ON:
            _census.res_mark("after_bar", int(np.count_nonzero(sc.residual)))
        # 막대가 놓인 뒤 한 번 더 — 새로 놓인 막대도 늘릴 자리가 생긴다
        if n_reg:
            stats["grown_fill"] += grow_fill(sc, plan.layers, lo, grown=_grown)
        if _census.ON:
            _census.res_mark("after_grow2", int(np.count_nonzero(sc.residual)))
        # 마무리 — 남은 큰 덩어리만 줍는다. 작은 조각은 구멍 메움(컷 뒤,
        # 군집당 1장)이 더 싸게 처리하므로 여기서 예산을 쓰지 않는다
        # (실측: min_blob 12일 때 mop 549장 — 채움 예산을 갉아먹었다)
        if n_reg < cap:
            # 아직 한 장도 못 받은 영역에서는 덩어리 하한을 **보이는 구멍**
            # 크기까지 내린다 — 안 그러면 40px 미만의 작은 면이 통째로 안
            # 칠해진 채로 봉인까지 내려간다 (그 자리는 구멍 하나가 아니라
            # 색이 통째로 빠진 자리다). 첫 장 λ 면제도 그때라야 뜻이 산다
            n_mop = mop_up(plan, sc, cat, reg.color, cap - n_reg,
                           40 if n_reg else _MOP_MIN_FIRST, price,
                           n_reg == 0)
            stats["mop_layers"] += n_mop
            n_reg += n_mop
        if _census.ON:
            _census.res_mark("end", int(np.count_nonzero(sc.residual)))
            _census.end_region(layers=int(n_reg), hi=len(plan.layers))
            cr.pop("_dedge", None)
            cr.pop("_ink", None)
        if n_reg == 0:
            stats["empty_regions"] += 1
            stats["empty_px"] += int(reg.area)
        stats["uncovered_px"] += int(np.count_nonzero(sc.residual))
        if oi < 10:                        # 큰 면이 얼마를 먹나 (계획 3 계측)
            stats["big10_layers"] += n_reg
        if n_reg >= cap:                   # 영역 예산이 실제로 물린 횟수
            stats["cap_hit"] += 1
            if cap == share:
                stats["share_hit"] += 1

    plan.layers.extend(line_layers)       # 선화는 모든 면 위 (마지막 선따기)
    log(msg("  큰 면 10개에 {big10}장 (채움 {fill}·막대 {bar}·마무리 {mop} 중)"
            " · 영역 예산이 물린 곳 {cap_hit}/{total} (그중 배분 몫 {share_hit})",
            big10=stats["big10_layers"], fill=stats["fill_layers"],
            bar=stats["bar_layers"], mop=stats["mop_layers"],
            cap_hit=stats["cap_hit"], total=total,
            share_hit=stats["share_hit"]))
    if len(_FILL_SHAPES) > 1:             # 채움 어휘 튜닝용 계측
        tot = max(1, sum(_FILL_WIN.values()))
        top = sorted(_FILL_WIN.items(), key=lambda kv: -kv[1])
        stats["fill_win"] = dict(top)
        log(msg("  채움 도형: {items}",
                items=" · ".join(f"{k} {v}({100 * v / tot:.0f}%)"
                                 for k, v in top[:12])))
    stats["skew_cand"] = _CURVE_STATS["skew_cand"]   # §14 — 지어 본 전 아핀 후보
    if _CURVE_STATS["paths"]:             # 획 어휘 튜닝용 계측
        log(msg("  곡선 획 {ok}/{paths} (직선 {flat}·짧음 {short}·부적합 {nofit}"
                "·저득점 {lowgain}·획아님 {notline})",
                ok=_CURVE_STATS["ok"], paths=_CURVE_STATS["paths"],
                flat=_CURVE_STATS["flat"], short=_CURVE_STATS["short"],
                nofit=_CURVE_STATS["nofit"], lowgain=_CURVE_STATS["lowgain"],
                notline=_CURVE_STATS["notline"]))
    if progress:
        progress(1.0, msg("배치 완료"))
    stats["_grown_idx"] = sorted(_grown)
    return plan, stats


def fit_line_plan(cel: CelArt, cat: Catalog, *, budget: int = 3000,
                  source_image: str = "", log=print, progress=None,
                  value: np.ndarray | None = None,
                  price: float = 0.0, maps=None,
                  pol=None) -> tuple[LayerPlan, dict]:
    """선 도안 — 선화 획**만** 배치한다 (공통 엔진 단독, 면 채움 없음).

    사람이 원화를 반투명 오버레이로 깔고 선만 따라 긋는 방식의 자동화다
    (`references/사람작업/오버레이-선*.png`). **두 노선이 이 함수를 함께
    쓴다** — `cel.labels`는 실루엣 한 영역(0/-1)이면 되고, 갈리는 것은
    `pol`(노선 정책)과 `price`(잉크 가격) 둘뿐이다:

    - **가격** — line 노선은 안 준다 (선이 곧 도안 전부라 나눌 예산이 없다;
      파편 필터와 예산 우선순위만 거른다). cel 노선은 면이 같은 λ 자를 나눠
      쓰므로 잉크 몫(λ×`_PRICE_INK`)을 물린다 — 무가격이면 조밀한 그림에서
      선이 예산을 다 먹어 채움이 굶는다 (홍채·입 채움 소실). **어느 역할이
      가격을 무는가**는 정책이 정한다 (실루엣·고립 특징은 면제 — 길이로 값을
      매기면 구조적으로 지는데, 빠지면 그 자리 경계가 통째로 없어진다).
    - **정책** — 허용 스필·덮임·끊김, 밴드 여유, 덮어 그리기 여부, 한 획의
      도형 상한 (`policy` 문서). 두 노선의 **논리 획 그래프는 같다**: 정책은
      그 위에서 무엇을 그릴지와 어느 후보를 쓸지만 고른다. 그래서 노선별
      결과 차이가 나면 그 이유가 정책 한 칸으로 적힌다 (`report`의 `policy`·
      `dropped`).

    stats 자가 지표는 렌더와 같은 폴리곤 식(`_ink_cover`)으로 잰다:
    ink_cover(선 지도 중 획이 덮는 몫) · ink_stray(획 잉크 중 선 지도에서
    최소 획 폭보다 먼 몫 — 최소 폭 강제 스필은 안 센다) · outline_cover
    (실루엣 테에서 최소 획 폭 안에 잉크가 있는 몫 — 원화가 실루엣에 선을
    안 그린 그림이 여기서 드러난다. 알파 없는 입력은 None).
    """
    _check_vocab(cat, log)
    w, h = cel.size
    upp = 900.0 / h                      # cel 노선과 같은 캔버스 배율
    plan = LayerPlan(source_image=source_image, image_size=(w, h),
                     units_per_px=upp)
    for k in _CURVE_STATS:
        _CURVE_STATS[k] = 0
    forms = _stroke_forms(cat)
    sids = itertools.count()
    stats: dict = {}
    n = _fit_lines(plan, cel, cat, upp, budget, forms, log, sids=sids,
                   carve=False, progress=progress, stats=stats,
                   value=value, price=price * _PRICE_INK, maps=maps, pol=pol)
    # 획 색을 발자국 아래 원화 평균으로 — 강제 굵기의 단색 최적 (`recolor_strokes`)
    stats["recolored_strokes"] = recolor_strokes(plan, cel, cat, upp, log=log)
    stats["line_layers"] = n
    stats["strokes"] = len({l.stroke for l in plan.layers if l.stroke >= 0})
    if _CURVE_STATS["paths"]:             # 획 어휘 튜닝용 계측 (fit_plan과 동일)
        log(msg("  곡선 획 {ok}/{paths} (직선 {flat}·짧음 {short}·부적합 {nofit}"
                "·저득점 {lowgain}·획아님 {notline})",
                ok=_CURVE_STATS["ok"], paths=_CURVE_STATS["paths"],
                flat=_CURVE_STATS["flat"], short=_CURVE_STATS["short"],
                nofit=_CURVE_STATS["nofit"], lowgain=_CURVE_STATS["lowgain"],
                notline=_CURVE_STATS["notline"]))
    ink = _ink_cover(plan.layers, cat, upp, w, h)
    lm = cel.line_mask
    # "가깝다"의 자는 상수가 아니라 게임 격자 — 최소 도형(스케일 0.01)의 폭
    r = max(1, int(round(2.0 * _min_span(upp))))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    near_line = cv2.dilate(lm.astype(np.uint8), k5).astype(bool)
    stats["ink_cover"] = round(float((ink & lm).sum()) / max(1, int(lm.sum())), 4)
    # ±1px 완화판 — 획이 띠보다 반 픽셀 가늘거나 밀려 남는 **옆구리 실오라기**
    # (선은 이어져 보인다)를 커버리지 미달로 세지 않는다. 실측(01) 미커버의
    # 85%가 이 실오라기였다 — 진짜 결함(획 통째 누락·점선)은 이 완화로도 남는다
    near_ink = cv2.dilate(ink.astype(np.uint8),
                          np.ones((3, 3), np.uint8)).astype(bool)
    stats["ink_near"] = round(float((near_ink & lm).sum())
                              / max(1, int(lm.sum())), 4)
    # **면이 맡기로 한 경계는 "못 그린 선"이 아니다.** `ink_near`는 선 지도
    # **전체**를 분모로 써서 "선화 모델을 얼마나 따랐나"를 잰다 — 표현 결정
    # (`engine._fill_owns`)이 일부러 뺀 선이 그대로 감점이 된다. 병리 감지용
    # 검사(`route_cel`의 `ink`)는 그 자로 물으면 안 되므로, **긋기로 한 선**만
    # 분모에 넣은 자를 따로 낸다. 억제가 없으면 두 값은 같다.
    own_m = None
    rec = stats.get("_rec")
    for st in getattr(rec, "strokes", ()) if rec is not None else ():
        if st.dropped != "fill_owns" or len(st.path) < 2:
            continue
        if own_m is None:
            own_m = np.zeros((h, w), np.uint8)
        rx0, ry0 = st.roi[0], st.roi[1]
        pp = np.stack([st.path[:, 1] + rx0, st.path[:, 0] + ry0],
                      axis=1).round().astype(np.int32)
        cv2.polylines(own_m, [pp], False, 1,
                      max(1, int(round(max(st.width, 1.0)))))
    if own_m is not None:
        keep = lm & ~cv2.dilate(own_m, k5).astype(bool)
        stats["ink_near_drawn"] = round(float((near_ink & keep).sum())
                                        / max(1, int(keep.sum())), 4)
    else:
        stats["ink_near_drawn"] = stats["ink_near"]
    stats["ink_stray"] = round(float((ink & ~near_line).sum())
                               / max(1, int(ink.sum())), 4)
    stats["outline_cover"] = stats["outline_src"] = None
    if bool((cel.labels < 0).any()):      # 알파 없는 입력은 테가 액자 테두리다
        sel = cel.labels >= 0
        rim = sel & ~cv2.erode(sel.astype(np.uint8),
                               np.ones((3, 3), np.uint8)).astype(bool)
        near_ink = cv2.dilate(ink.astype(np.uint8), k5).astype(bool)
        stats["outline_cover"] = round(float((rim & near_ink).sum())
                                       / max(1, int(rim.sum())), 4)
        # 원화 쪽 상한 — 실루엣 테에 **선 지도**가 있는 몫. 잉크 몫과의 차가
        # 곧 "선은 있는데 획이 안 선" 배치 손실이다 (실측 01: 95% vs 80% —
        # 옅은 선의 대시 조각이 안 그려진 자리). 판정 문구가 이 둘로 원화
        # 탓과 배치 탓을 가른다
        stats["outline_src"] = round(float((rim & near_line).sum())
                                     / max(1, int(rim.sum())), 4)
    if progress:
        progress(1.0, msg("배치 완료"))
    return plan, stats
