"""사람 선따기 문법 — 폭 정책·덩어리 채움·이음 보수.

레퍼런스(`references/사람작업/오버레이-선1·선2·선3-인게임`)의 확대·실측 결론이다:
① 모든 획이 **짧은 변의 0.34%** 안의 균일한 가는 선이다 (`_LINE_W_REL`)
② 획은 길고 매끈하게 이어지며 교차점에 부스러기가 없다 ③ 작은 검정
덩어리(눈동자·장식)는 획이 아니라 채운 도형이다.
①의 폭 상한과 ③의 덩어리 채움은 엔진이 여기 것을 쓰고(`engine.build_strokes`),
②는 `_patch_seams`가 배치 뒤에 보증한다. 무엇을 획으로 볼지의 분류는
`graph.classify`가 한다.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

from ...i18n import msg
from ..catalog import Catalog
from ..celart import CelArt
from ..model import Layer, LayerPlan
from .geometry import _ink_cover, _layer, _mask_px, _min_span, _poly_px
from .scoring import _MIN_GAIN, _Scorer, _descend
from .skeleton import _rdp
from .vocabulary import _FILL_SHAPE, bar_for


# ── line 노선의 사람 선따기 문법 (`references/사람작업/오버레이-선*` 실측) ──
# 사람의 선따기(오버레이-선1·선2·선3-인게임)를 확대 대조·실측한 결론:
# ① 모든 획이 짧은 변의 0.30%로 균일하다 ② 획은 길고 매끈하게 이어지며
# 교차점에 부스러기·튀어나옴이 없다 ③ 작은 검정 덩어리(눈동자·장식)는 획이
# 아니라 채운 도형이다. 아래 상수는 그 문법의 자다 — 전부 line 노선(relax)
# 에서만 쓴다.
#
# **획 폭 상한 = 짧은 변의 0.34%** (작업 해상도 1,200에서 4.1px). 사람이 긋는
# 폭의 실측값이다 — 레퍼런스 셋의 잉크 질량 /
# (뼈대 길이 × 그 획의 농도)를 각 그림의 잉크 bbox **짧은 변 1,200** 정규화로
# 재면 폭 중앙이 3.96 · 4.19 · 4.66px (= 짧은 변의 0.33 · 0.35 · 0.39%)이고,
# 이진 문턱(거리변환 능선 ×2)이라는 다른 자로도 앞 둘이 4.09 · 4.52px로 맞는다.
# 신경망 선화가 가까운 두 선을 한 띠로 붙여 준 자리(04 실측: 폭 4px+ 띠의
# 다수가 중간해상에서 능선 2개)를 굵기 그대로 그리면 검은 덩어리가 되므로,
# 그보다 굵은 띠는 이 폭으로 눌러 긋는다. 예외: 폭이 경로 내내 일관(변동계수
# < 0.22)하고 길이가 폭의 4배 이상이면 의도된 굵은 선(굵은 테두리 그림체·
# 머리핀)으로 보고 그대로 둔다.
# 기준 변이 짧은 변인 것은 다른 길이 상수와 같은 이유다 — 작업 해상도가 짧은
# 변 고정이라 국소 구조의 px 크기가 그것을 따른다.
# (최소 도형 폭의 배수로 쓰면 안 된다: 최소 폭은 upp = 900/h라 **높이**를 따라
#  가로 구도 1.7px ↔ 세로 구도 3.5px로 갈리고, 같은 레퍼런스가 1.1~3.6배로
#  흩어진다.)
_LINE_W_REL = float(os.environ.get("FS_LINE_WREL", 0.0034))
_WCAP_CV = 0.22
# 뚱뚱 덩어리 채움 — 반폭이 최소 도형 폭을 넘는(= 선으로 못 그리는) 컴팩트
# 덩어리는 뼈대를 긋지 않고 **모멘트 타원으로 채운다** (사람의 눈동자·장식
# 문법 — 레퍼런스 선3_인게임의 나비 장식·판다 눈이 채운 도형이다). 존 안
# 덩어리는 무늬라 안 채운다. 상한을 넘는 대형 덩어리는 채움 대신 획으로
# 두는다 (선 노선이 면을 그리기 시작하면 안 된다).
# 굵기만으로 고르면 안 된다 — **컴팩트한 것만** 덩어리다 (`lines._fit_lines`의
# 후보 수집에서 `stroke._STROKE_SLIM`으로 거른다). 병합 띠·교차 뭉치도 굵어서,
# 그쪽에 타원이 얹히면 선화에 없던 덩어리가 선 위에 보인다
_FAT_MIN_AREA = 12.0      # px — 이보다 작으면 그냥 획
_FAT_MAX_MUL = 10.0       # 상한 = (이 배수 × 최소 도형 폭)²
# 이음 보수 — 획 하나를 다 놓은 뒤 경로를 걸어 안 덮인 틈(선폭 = 최소 도형
# 폭 이상)을 최소 폭 막대로 정확히 메운다. 채점을 안 거친다 — 이음은
# "이어져 보인다"가 목표지 점수가 아니다 (사용자 요구: 이어진 부분은
# 확실하게 하나의 선처럼). 0.7배는 양자화 지터까지 메워 획당 6장의 보수
# 폭주가 났다 (실측 1,935장) — 시각적 끊김은 선폭부터다.
_SEAM_GAP_MUL = float(os.environ.get("FS_SEAM_GAP", 1.0))
# 덮임 판정 반경 (px) — **1px = 렌더에서 맞닿음**. 최소 도형 폭의 배수로 두면
# 세로로 긴 구도(최소 폭 3.5px)에서 반경이 2px가 되어 7px 끊김이 숨는다
# (05·06·07에서 실제로 그랬다). 이어져 보이는가는 잉크가 붙어 있는가다.
_SEAM_COV_MUL = float(os.environ.get("FS_SEAM_COV", 1.0))


def _fill_fat(plan: LayerPlan, sc: _Scorer, cel: CelArt, upp: float,
              ys: np.ndarray, xs: np.ndarray, sid: int, left: int) -> int:
    """뚱뚱 덩어리 하나를 **모멘트 타원 1~2장**으로 채운다 (ROI-로컬 좌표).

    사람의 검정 덩어리 문법 — 눈동자·속눈썹 뭉치·장식은 획이 아니라 채운
    도형이다 (레퍼런스 선3_인게임의 나비 장식·판다 눈). 뼈대를 긋으면 덩어리
    안 그물 뼈대가 낙서가 된다 (01 preview의 눈가 낙서 실측). 2차 모멘트가
    같은 타원을 놓고 하강으로 다듬는다 — 잔여가 40% 넘게 남으면 가장 큰
    잔여 조각에 한 장 더 (상한 2장 — 선 노선이 면을 그리기 시작하면 안 된다).
    """
    if left <= 0 or not len(ys):
        return 0
    rx0, ry0 = sc.roi[0], sc.roi[1]
    band = np.zeros(sc.residual.shape, bool)
    band[ys, xs] = True
    band_d = cv2.dilate(band.astype(np.uint8),
                        np.ones((3, 3), np.uint8)).astype(bool)
    src = cel.src_rgb[ry0:sc.roi[3], rx0:sc.roi[2]]
    color = tuple(int(v) for v in np.median(src[ys, xs], axis=0))
    n = 0
    cur = band.copy()
    area0 = float(len(ys))
    for _ in range(2):
        yy, xx = np.nonzero(cur)
        if len(yy) < _FAT_MIN_AREA or n >= left:
            break
        cy, cx = float(yy.mean()), float(xx.mean())
        dy, dx = yy - cy, xx - cx
        c20 = float((dx * dx).mean())
        c02 = float((dy * dy).mean())
        c11 = float((dx * dy).mean())
        tr, det = c20 + c02, c20 * c02 - c11 * c11
        disc = max(0.0, tr * tr / 4.0 - det) ** 0.5
        l1, l2 = tr / 2.0 + disc, max(tr / 2.0 - disc, 1e-6)
        theta = 0.5 * np.arctan2(2.0 * c11, c20 - c02)
        lay = _layer(_FILL_SHAPE, rx0 + cx, ry0 + cy,
                     max(2.0 * np.sqrt(l1), 1.0), max(2.0 * np.sqrt(l2), 1.0),
                     theta, 0.0, color, sc.upp, sc.w, sc.h,
                     label="ink", stroke=sid)
        sc.set_band(band_d)
        gain, q = _descend(sc, lay, color, passes=2)
        _, mfin = sc.score(q)
        sc.set_band(None)
        if gain <= _MIN_GAIN * 0.5 or not mfin.any():
            break
        sc.commit(mfin)
        plan.layers.append(q)
        n += 1
        cur &= ~mfin
        if float(np.count_nonzero(cur)) < 0.4 * area0:
            break
    return n


# 이웃 도형을 늘려 틈을 메울 때 시도하는 스케일 스텝 (게임 스케일 0.01 단위).
# 틈의 96%가 길이 3~12px·현 이탈 0.3px인 **양자화 잔틈**이라(01 실측 196개),
# 새 막대 한 장을 놓느니 이미 있는 도형을 한두 칸 늘리는 것이 맞다 — 레이어가
# 예산이다. 세 칸까지 보는 것은 최소 스텝이 세로 구도에서 1.4px이라 12px 틈이
# 그 안에 들기 때문이다. 늘리기가 원래 자리를 잃으면(오목 도형) 기각한다.
_STRETCH = ((0.01, 0.0), (0.02, 0.0), (0.0, 0.01), (0.01, 0.01), (0.03, 0.0))
# 이웃 도형 격자 칸 (px) — 틈 표본 하나가 보는 이웃 후보의 범위.
_GRID = 16


def _seg_bars(seg: np.ndarray, wpx: float) -> list:
    """곧은 틈을 따라갈 막대들 — [(중점, 길이, 각)]. RDP로 마디를 끊는다."""
    pl = _rdp(seg, max(1.0, 0.5 * wpx))
    out = []
    for k in range(len(pl) - 1):
        p0, p1 = pl[k], pl[k + 1]
        ln = float(np.hypot(*(p1 - p0)))
        if ln >= 1.0:
            out.append(((p0 + p1) / 2.0, ln,
                        float(np.arctan2(p1[0] - p0[0], p1[1] - p0[1]))))
    return out


def _lay_box(cat: Catalog, lay: Layer, upp: float, w: int, h: int, pad: int):
    x0 = y0 = 10 ** 9
    x1 = y1 = -10 ** 9
    for p in _poly_px(cat, lay, upp, w, h):
        x0 = min(x0, int(np.floor(p[:, 0].min())))
        y0 = min(y0, int(np.floor(p[:, 1].min())))
        x1 = max(x1, int(np.ceil(p[:, 0].max())))
        y1 = max(y1, int(np.ceil(p[:, 1].max())))
    return (max(0, x0 - pad), max(0, y0 - pad),
            min(w, x1 + pad), min(h, y1 + pad))


def _stretch_cover(plan: LayerPlan, cat: Catalog, upp: float, w: int, h: int,
                   own, gy: np.ndarray, gx: np.ndarray,
                   cov: np.ndarray, allow: np.ndarray, ink: np.ndarray,
                   ker) -> int:
    """도형들을 **한 칸씩 늘려** 경로를 더 덮게 한다 — 늘린 장수.

    새 도형 대신 있는 도형을 키우므로 장수가 안 는다 (`_STRETCH` 문서).
    받는 조건은 이음 보수가 지켜 온 둘 그대로다: 덮던 자리를 잃지 않고
    (오목 도형 기각), 새로 나가는 잉크는 선 밴드 안이어야 한다. `cov`(경로
    표본의 덮임)와 `ink`(전체 잉크)는 그 자리에서 갱신한다.

    `own`은 볼 레이어 인덱스다 — 제 획의 도형이 먼저고, 그것으로 안 되면
    **이웃 획의 도형**이 이어진다 (틈은 대개 교차점이라 그 자리를 이미 지나는
    다른 획의 도형이 한 칸이면 닿는다).
    """
    n = 0
    for k in own:
        if cov.all():
            break
        lay = plan.layers[k]
        for dsx, dsy in _STRETCH:
            q = Layer(**{**lay.__dict__})
            q.sx = round(q.sx + dsx * (1.0 if q.sx >= 0 else -1.0), 4)
            q.sy = round(q.sy + dsy * (1.0 if q.sy >= 0 else -1.0), 4)
            q = q.quantized()
            bx0, by0, bx1, by1 = _lay_box(cat, q, upp, w, h, 2)
            box = (bx0, by0, bx1, by1)
            m1 = _mask_px(cat, q, upp, w, h, box)
            m0 = _mask_px(cat, lay, upp, w, h, box)
            # 덮던 자리를 잃으면 안 되고(오목 도형), 새 잉크는 선 밴드 안이어야
            # 한다. 여유 2%는 라스터 반올림 한 겹이다 (`merge._MERGE_LOSS`와
            # 같은 근거) — 0으로 두면 양자화 지터만으로 전부 기각된다
            n0 = max(1.0, float(m0.sum()))
            if float((m0 & ~m1).sum()) > 0.02 * n0:
                continue
            if float(((m1 & ~m0) & ~allow[by0:by1, bx0:bx1]).sum()) \
                    > 0.02 * max(1.0, float(m1.sum())):
                continue
            grown = cv2.dilate(m1.astype(np.uint8), ker).astype(bool)
            inb = ((gy >= by0) & (gy < by1) & (gx >= bx0) & (gx < bx1))
            hit = np.zeros_like(cov)
            hit[inb] = grown[gy[inb] - by0, gx[inb] - bx0]
            if int((hit & ~cov).sum()) < 1:   # 더 덮는 게 없으면 안 늘린다
                continue
            plan.layers[k] = q
            cov |= hit
            ink[by0:by1, bx0:bx1] |= grown
            n += 1
            break
    return n


def _patch_seams(plan: LayerPlan, cat: Catalog, upp: float,
                 size: tuple[int, int], placed: list, left: int,
                 log, st: dict, allow: np.ndarray,
                 forms: tuple | None = None,
                 owners: list | None = None) -> int:
    """획마다 경로를 걸어 **안 덮인 틈을 메운다** (line 노선 전용).

    사람 문법의 마지막 보증 — "이어진 부분은 확실하게 하나의 선처럼"
    (사용자 요구 2026-08-25). 곡선 한 장 근사의 처짐·쪼갠 지점·양자화 수축이
    남긴 틈은 채점으로는 안 잡힌다 (retrace 중립 지대). 그래서 여기는 채점이
    아니라 **기하**다: 경로 표본이 잉크에서 반폭 밖에 있는 연속 구간(선폭
    이상)마다 도형을 정확히 놓는다. 같은 stroke id를 물려받아 프루닝이 획과
    함께 다룬다.

    두 가지가 "낭비 없이, 각지지 않게"를 만든다:

    - **덮임은 도안 전체의 잉크로 본다.** 제 획의 마디만 보면 교차하는 다른
      획이 이미 그어 놓은 자리를 또 긋는다 — 그것이 같은 방향 짧은 도형이
      겹쳐 쌓이던 축이다 (실측: 보수 도형 넓이의 30%가 이미 잉크였고, 새
      잉크가 30%도 안 되는 보수가 14%였다). 판정은 렌더 결과를 묻는 것이라
      "이어져 보인다"의 정의에도 이쪽이 맞다.
    - **굽은 틈은 곡선으로 메운다.** 틈은 획을 따라 나므로 획이 굽으면 틈도
      굽는다. 곧은 틈만 막대다 (`stroke._is_straight`).
    - **잔틈은 새 도형이 아니라 이웃 도형을 늘려 메운다** (`_stretch_cover`).
      실측(01) 틈 196개의 중앙이 길이 5.7px·현 이탈 0.3px인 양자화 잔틈이라,
      한 장을 새로 놓는 것보다 있는 도형을 한 칸 키우는 것이 싸다.
    """
    from .stroke import _is_straight, _try_curve

    w, h = size
    min_w = 2.0 * _min_span(upp)
    # **덮임 판정은 잉크가 맞닿는가로 본다** (`_SEAM_COV_MUL` × 최소 도형 폭).
    # 획 폭(사람 폭)으로 재 보았더니 그 반폭 안의 끊김이 전부 숨어, 선화에서
    # 한 선인 자리가 도안에서 끊겼다 (사용자 지적 · 01 실측 끊김 0 → 30곳,
    # 05 22 → 37곳). 이어져 보이는가는 **렌더에서 붙어 있는가**지 반폭 안에
    # 있는가가 아니다. 틈은 최소 도형 폭부터 — 그보다 짧으면 메울 도형이 없다
    gap_min = max(2.0, _SEAM_GAP_MUL * min_w)
    rr = max(1, int(round(_SEAM_COV_MUL)))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rr + 1, 2 * rr + 1))
    ink = cv2.dilate(_ink_cover(plan.layers, cat, upp, w, h).astype(np.uint8),
                     ker).astype(bool)
    # 이웃 도형 찾기용 격자 — 레이어 bbox가 걸치는 칸에 그 인덱스를 담는다.
    # 한 번만 짓는다 (늘린 도형은 커지기만 하므로 격자가 낡아도 후보를 놓칠 뿐)
    grid: dict = {}
    lo_all = min((p[5] for p in placed), default=len(plan.layers))
    for i in range(lo_all, len(plan.layers)):
        if plan.layers[i].label != "ink":
            continue
        bx0, by0, bx1, by1 = _lay_box(cat, plan.layers[i], upp, w, h, 1)
        for cy in range(by0 // _GRID, by1 // _GRID + 1):
            for cx in range(bx0 // _GRID, bx1 // _GRID + 1):
                grid.setdefault((cy, cx), []).append(i)
    n = n_curve = n_grow = 0
    for si, (sid, path, wmed, color, sc, lo, hi) in enumerate(placed):
        own = owners[si] if owners is not None else None
        if n >= left:
            break
        if hi <= lo or len(path) < 2:
            continue
        rx0, ry0 = sc.roi[0], sc.roi[1]
        p = path.round().astype(int)
        gy = np.clip(p[:, 0] + ry0, 0, h - 1)
        gx = np.clip(p[:, 1] + rx0, 0, w - 1)
        cov = ink[gy, gx].copy()
        wpx = max(wmed, min_w)
        if not cov.all():
            # 잔틈은 새 도형이 아니라 있는 도형을 늘려 메운다 (장수 0장).
            # 제 획 → 그 자리를 지나는 이웃 획 순 (§ seam 전에 확장)
            cand = list(range(lo, hi))
            if grid is not None:
                near: dict = {}
                for i2 in np.nonzero(~cov)[0]:
                    for j2 in grid.get((int(gy[i2]) // _GRID,
                                        int(gx[i2]) // _GRID), ()):
                        if not (lo <= j2 < hi):
                            near[j2] = None
                cand += list(near)
            g = _stretch_cover(plan, cat, upp, w, h, cand,
                               gy, gx, cov, allow, ink, ker)
            n_grow += g
            if own is not None:
                own.grown += g
        i = 0
        while i < len(cov) and n < left:
            if cov[i]:
                i += 1
                continue
            j = i
            while j < len(cov) and not cov[j]:
                j += 1
            seg = path[max(0, i - 1):min(len(path), j + 1)]
            L = (float(np.hypot(*np.diff(seg, axis=0).T).sum())
                 if len(seg) >= 2 else 0.0)
            if L >= gap_min:
                lays = []
                if (forms is not None and len(seg) >= 10
                        and not _is_straight(seg, wpx)):
                    # 굽은 틈 — 곡선 한 장. 밴드를 걸어 이 틈 밖으로 안 자란다
                    bm = np.zeros(sc.residual.shape, np.uint8)
                    pp = seg.round().astype(np.int32)
                    cv2.polylines(bm, [np.stack([pp[:, 1], pp[:, 0]], axis=1)],
                                  False, 1, max(1, int(round(wpx)) + 3))
                    sc.set_band(bm.astype(bool))
                    got = _try_curve(sc, forms, seg, wpx, color, True, sid,
                                     race=True, line=True)
                    sc.set_band(None)
                    if got is not None:
                        lays = [got[1]]
                        n_curve += 1
                if not lays:
                    bname, bext, brot = bar_for(cat, upp, wpx)
                    lays = [_layer(bname, rx0 + m_[1], ry0 + m_[0],
                                   ln / 2.0 + wpx * 0.5, wpx / 2.0, th_, 0.0,
                                   color, sc.upp, sc.w, sc.h,
                                   label="ink", stroke=sid,
                                   ext=bext, rot_off=brot)
                            for m_, ln, th_ in _seg_bars(seg, wpx)]
                for lay in lays:
                    if n >= left:
                        break
                    q = lay.quantized()
                    _, mfin = sc.score(q)
                    sc.commit(mfin)
                    plan.layers.append(q)
                    ink |= cv2.dilate(
                        _mask_px(cat, q, upp, w, h, (0, 0, w, h)
                                 ).astype(np.uint8), ker).astype(bool)
                    n += 1
                    if own is not None:
                        own.seams += 1
            i = j
    st["seam_patches"] = n
    st["seam_curves"] = n_curve
    st["seam_grown"] = n_grow
    if n or n_grow:
        log(msg("  이음 보수 {n}장 (획 틈 메움 — 하나의 선으로, 곡선 {n_curve}장 · "
                "이웃 도형 늘려 메운 틈 {n_grow}개는 0장)",
                n=n, n_curve=n_curve, n_grow=n_grow))
    return n
