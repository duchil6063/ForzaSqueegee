"""면 채움 — 봉우리마다 씨앗을 잡고 하강으로 다듬어 영역을 덮는다.

닫힌 해로 씨앗을 잡는다: 2차 모멘트가 같은 타원·사각을 경쟁시켜(`_seed_moment`)
이긴 쪽을 쓰고, 확장 어휘 여섯은 배수 `_FILL_MARGIN`을 넘겨야 바탕을 이긴다.
남은 얇은 껍질은 막대 사슬(`_fit_bars`)이, 남은 큰 조각은 마무리(`_mop_up`)가
줍는다 — 그보다 작은 조각은 구멍 메움이 군집당 한 장으로 더 싸게 처리한다.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

from ..catalog import Catalog
from ..model import Layer, LayerPlan
from .geometry import _layer, _min_span
from .scoring import _Scorer, _descend
from .skeleton import _dt_along, _join_paths, _paths, _prune_spurs, _thin
from .stroke import _FORM_RASTER, _fit_path, _path_worth
from .vocabulary import (_FILL_BASE, _FILL_MARGIN, _FILL_MIN_R0, _FILL_MOMENT,
                         _FILL_SHAPE, _FILL_SHAPES, _FILL_TMPL, _FILL_TOP)


# **영역 껍질 컷** — 채움 도형이 못 맞춘 경계의 얇은 껍질을 막대로 쫓을 것인가
# (`_fit_bars`, 선화가 아닌 영역 잔여에만 건다). 막대가 플랜의 45%인데 그 까닭이
# 경로당 마디 수가 아니라 **경로 개수**임이 실측으로 갈렸으므로, 줄일 자리는
# 여기다. 이보다 가늘거나 짧은 껍질은 안 쫓는다 — 그 자리는 이웃 면과 선화가
# 덮고, 남으면 구멍 메움이 군집당 한 장으로 줍는다 (군집당 한 장이 마디마다
# 한 장보다 싸다).
#
# 폭 문턱은 px 상수가 아니라 **게임이 낼 수 있는 최소 도형 폭의 배수**다.
# 그래야 수요 적응으로 작업 해상도가 바뀌어도 같은 뜻이고("도형 한 장 값을
# 못 하는 굵기"), 상수가 캔버스 격자에서 나온다. 1.5배는 1200 기준 2.6px로,
# 스윕(1.8/5 · 2.6/8 · 3.6/12)의 무릎이다 — 3.6/12까지 올려도 막대가 83→60장
# 더 줄 뿐인데 마무리·메움이 그만큼 늘어 총장수가 안 준다.
_RIND_W_MUL = float(os.environ.get("FS_RIND_W_MUL", 1.5))
_RIND_LEN = float(os.environ.get("FS_RIND_LEN", 8))


def _win_pts(sc: _Scorer, px: int, py: int, r0: float) -> tuple[np.ndarray, float]:
    """봉우리 주변 창(4r0)의 잔여 점 구름 — 씨앗 둘이 같은 증거를 쓴다."""
    ys, xs = np.nonzero(sc.residual)
    win = 4.0 * r0
    selw = (np.abs(ys - py) <= win) & (np.abs(xs - px) <= win)
    return np.stack([xs[selw], ys[selw]], axis=1).astype(np.float64), win


def _seed_fat(sc: _Scorer, dt: np.ndarray, px: int, py: int, r0: float,
              color, pw: np.ndarray | None = None) -> Layer:
    """봉우리 씨앗 — 국소 PCA 방향으로 r0×(연장 추정) 타원부터 시작."""
    win = 4.0 * r0
    if pw is None:
        pw, win = _win_pts(sc, px, py, r0)
    ctr = pw.mean(axis=0)
    cov = np.cov((pw - ctr).T) if len(pw) > 8 else np.eye(2)
    evals, evecs = np.linalg.eigh(cov)
    d = evecs[:, int(np.argmax(evals))]
    theta = float(np.arctan2(d[1], d[0]))
    proj = (pw - [px, py]) @ d
    a = float(np.clip(np.percentile(np.abs(proj), 90), r0, win))
    x0, y0, _, _ = sc.roi
    # 씨앗은 보수적으로 안쪽(0.9배) — 하강이 늘리는 쪽으로 찾게 한다
    return _layer(_FILL_SHAPE, x0 + px, y0 + py, a * 0.9, r0 * 0.9, theta, 0.0,
                  color, sc.upp, sc.w, sc.h)


def _fill_tmpl(cat: Catalog, name: str) -> tuple[float, float, float, float]:
    """도형 단위 마스크의 **이미지 프레임 2차 모멘트** — (su, sv, tu, tv).

    su·sv = 로컬 x·y 표준편차, (tu, tv) = 로컬 중심(이미지 프레임이라 y 부호
    반전). `_layer(a, b)`는 로컬 좌표에 곱하는 배수라 a = σ/su 로 바로 풀린다.
    프로세스 1회 계측 (도형은 8종뿐).
    """
    t = _FILL_TMPL.get(name)
    if t is None:
        sh = cat[name]
        m = sh.rasterize(_FORM_RASTER)
        pts_all = np.concatenate(sh.loops, axis=0)
        lo = pts_all.min(axis=0)
        span = np.maximum(pts_all.max(axis=0) - lo, 1e-6)
        ys, xs = np.nonzero(m)
        lx = xs / (_FORM_RASTER - 1) * span[0] + lo[0]
        ly = (_FORM_RASTER - 1 - ys) / (_FORM_RASTER - 1) * span[1] + lo[1]
        t = (max(float(lx.std()), 1e-6), max(float(ly.std()), 1e-6),
             float(lx.mean()), -float(ly.mean()))
        _FILL_TMPL[name] = t
    return t


def _seed_moment(sc: _Scorer, pw: np.ndarray, name: str, color,
                 cat: Catalog) -> list[Layer]:
    """2차 모멘트 정합 씨앗 — 잔여 덩어리와 도형 마스크의 모멘트를 맞춘다.

    획이 쓰는 닫힌 해(`_affine_fit`)는 "열린 중심선" 전용이라 면에는 못 쓴다.
    면의 닫힌 해가 이것이다: 양쪽을 백색화하면 아핀은 직교변환만 남으므로,
    우리 자유도(회전 + 축별 스케일, 전단 없음)에서 회전은 **잔여의 주축**으로
    환원되고 스케일은 축별 표준편차 비로 바로 풀린다. 남는 자유도는 주축을
    어느 쪽에 붙이느냐(90° 배수 넷)뿐이라 후보가 유한하다 — 채움 어휘 조사에
    쓴 것과 같은 방법이다.

    씨앗 기하가 **도형마다** 갈리는 것이 요점이다. 타원 기준 상자 하나를 전
    어휘가 나눠 쓰면, 같은 상자를 덜 채우는 도형(삼각·초승달)이 잔여를 실제보다
    적게 덮은 자세에서 하강을 시작한다.
    """
    if len(pw) <= 8:
        return []
    mu = pw.mean(axis=0)
    cov = np.cov((pw - mu).T)
    evals, evecs = np.linalg.eigh(cov)
    lam = np.clip(evals, 1e-6, None)
    d = evecs[:, int(np.argmax(evals))]
    psi = float(np.arctan2(d[1], d[0]))
    sig = (float(np.sqrt(lam.max())), float(np.sqrt(lam.min())))
    su, sv, tu, tv = _fill_tmpl(cat, name)
    x0, y0, _, _ = sc.roi
    out = []
    for k in range(4):                    # 주축을 붙이는 네 자세
        th = psi + k * (np.pi / 2)
        a = sig[k % 2] / su               # 90° 돌면 장·단축이 바뀐다
        b = sig[1 - k % 2] / sv
        if a < 0.2 or b < 0.2:
            continue
        c, s = np.cos(th), np.sin(th)
        # 도형 중심이 원점이 아니면 그만큼 밀어야 마스크 중심이 덩어리에 온다
        cx = mu[0] - (a * tu * c - b * tv * s)
        cy = mu[1] - (a * tu * s + b * tv * c)
        out.append(_layer(name, x0 + cx, y0 + cy, a, b, th, 0.0, color,
                          sc.upp, sc.w, sc.h))
    return out


def _grow_step(sc: _Scorer, gain: float, q: Layer) -> tuple[float, Layer]:
    """양자화 내림 보정 — 스케일 한 스텝(≈1.7px 지름) 확장이 크게 손해가 아니면
    키운 쪽을 쓴다. 내림된 도형은 영역 경계에 1px대 슬리버를 남기고, 그 부스러기가
    구멍 메움 수요의 74%였다 (실측: 잔여 2.8k px 중 경계 인접 74%). 확장분은
    경계 밴드(1px 물림)가 흡수해 벌점이 거의 없다."""
    for dx, dy in ((0.01, 0.01), (0.01, 0.0), (0.0, 0.01)):   # 큰 확장 우선
        c = Layer(**{**q.__dict__})
        c.sx = round(c.sx + (dx if c.sx >= 0 else -dx), 4)
        c.sy = round(c.sy + (dy if c.sy >= 0 else -dy), 4)
        s = sc.score_val(c)
        if s >= gain - 2.0:
            return s, c
    return gain, q


def _place_fat(sc: _Scorer, dt: np.ndarray, px: int, py: int, r0: float,
               color, vocab: tuple[str, ...] | None = None) -> tuple[float, Layer]:
    """봉우리 하나에 최선 도형 배치 — 어휘를 씨앗 자리에서 채점해 상위
    `_FILL_TOP`개만 짧게 하강시키고, 이긴 쪽을 정밀 하강한다.

    씨앗 채점은 대리 지표가 아니라 목적함수 그대로다(내 잔여 − 침범 벌점).
    씨앗 상자가 타원 기준이라 상자를 덜 채우는 도형이 불리하긴 하지만, 잔여가
    그 모양이면 타원이 이웃 면을 침범해 벌점을 물어 순위가 뒤집힌다. 하강
    비용이 후보 수에 비례하므로 전수 하강은 안 한다.

    바탕(타원·사각)은 상위권 밖이어도 항상 하강시키고, 확장 어휘는 바탕보다
    `_FILL_MARGIN`배 이상 벌어야 이긴다 — 한 끗 차로 갈아타면 셀 경계와 덜
    맞는 모양이 이겨 특징이 흐려진다."""
    pw, _ = _win_pts(sc, px, py, r0)
    seed = _seed_fat(sc, dt, px, py, r0, color, pw=pw)
    # 어휘는 밖에서 온다 (§6 서술자 순위). 안 주면 손으로 고른 여덟이다
    if vocab is None:
        vocab = _FILL_SHAPES if r0 >= _FILL_MIN_R0 else _FILL_BASE
    scored = []
    for i, name in enumerate(vocab):
        cand = Layer(**{**seed.__dict__})
        cand.shape = name
        best_s, best_c = sc.score_val(cand), cand
        # 모멘트 정합 씨앗 — 같은 증거(창 점 구름)에서 도형마다 따로 푼 자세.
        # 같은 목적함수로 재어 좋은 쪽에서 하강을 시작한다
        if _FILL_MOMENT:
            for alt in _seed_moment(sc, pw, name, color, sc.cat):
                s = sc.score_val(alt)
                if s > best_s:
                    best_s, best_c = s, alt
        scored.append((-best_s, i, best_c))
    scored.sort(key=lambda t: t[:2])          # 동점은 어휘 순서 — 결정적
    # 하강 순서는 다시 **어휘 순서**로 되돌린다 — 하강 뒤 동점이면 앞 어휘가
    # 이기게 해, 어휘를 넓혀도 기존 어휘의 판정이 안 흔들리게 한다
    keep = {t[1] for t in scored[:max(1, _FILL_TOP)]}
    keep |= {i for i, n in enumerate(vocab) if n in _FILL_BASE}
    best = base = None
    for _, i, cand in sorted((t for t in scored if t[1] in keep), key=lambda t: t[1]):
        g, q = _descend(sc, cand, color, passes=2)
        if vocab[i] in _FILL_BASE and (base is None or g > base[0]):
            base = (g, q)
        if best is None or g > best[0]:
            best = (g, q)
    if base is not None and best[1].shape not in _FILL_BASE \
            and best[0] < _FILL_MARGIN * base[0]:
        best = base
    return _grow_step(sc, *_descend(sc, best[1], color, passes=3))


def _mop_up(plan: LayerPlan, sc: _Scorer, color, left: int,
            min_blob: int = 12, price: float = 0.0,
            free_first: bool = False) -> int:
    """잔여 덩어리 줍기 — 연결 성분마다 작은 타원 하나씩 (탐색 생략, 하강만).

    획·채움이 놓친 조각이 흰 반점으로 남는 것을 막는다. min_blob 미만 조각은
    화면에서 안 보이는 크기라 버린다.
    """
    n = 0
    x0, y0, _, _ = sc.roi
    while n < left:
        res = sc.residual.astype(np.uint8)
        cnt, cc, cstats, cent = cv2.connectedComponentsWithStats(res, connectivity=8)
        # 가장 큰 덩어리부터
        order = np.argsort(-cstats[1:, cv2.CC_STAT_AREA]) + 1
        placed = False
        for ci in order[:1]:
            if cstats[ci, cv2.CC_STAT_AREA] < min_blob:
                return n
            cm = cc == ci
            # 가격 — 덩어리가 통째로 λ에 못 미치면 어떤 도형으로도 값을 못 한다
            # (영역의 첫 장은 면제 — `fit_plan`의 같은 근거)
            if price and not (free_first and n == 0) and sc.worth_of(cm) < price:
                sc.commit(cm)              # 포기 — 잔여에서 지워 다음 덩어리로
                placed = True
                break
            dt = cv2.distanceTransform(cm.astype(np.uint8), cv2.DIST_L2, 3)
            py, px = np.unravel_index(int(dt.argmax()), dt.shape)
            r0 = max(1.0, float(dt.max()))
            lay = _seed_fat(sc, dt, px, py, r0, color) if cstats[ci, cv2.CC_STAT_AREA] > 40 \
                else _layer(_FILL_SHAPE, x0 + px, y0 + py, r0, r0, 0.0, 0.0,
                            color, sc.upp, sc.w, sc.h)
            gain, q = _descend(sc, lay, color, passes=3)
            if gain < 3.0:
                # 이 덩어리는 포기 — 잔여에서 지워 무한루프 방지
                sc.commit(cm)
                placed = True   # 루프 계속 (다음 덩어리)
                break
            gain, q = _grow_step(sc, gain, q)
            _, mfin = sc.score(q)
            sc.commit(mfin)
            plan.layers.append(q)
            n += 1
            placed = True
            break
        if not placed:
            break
    return n


def _fit_bars(plan: LayerPlan, sc: _Scorer, dt: np.ndarray, color,
              left: int, forms: tuple | None = None, ink: bool = False,
              sids=None, price: float = 0.0, free_first: bool = False) -> int:
    """가는 잔여를 획으로 — 굽은 경로는 곡선 한 장, 나머지는 A_22 막대 사슬.

    ink=True(획형 영역 = 선화·가닥)일 때만 "ink" 라벨 — 프루닝 보호 대상.
    굵은 영역의 경계 부스러기 막대는 "cel"로 남아 예산 초과 때 정리된다.
    경로 하나가 획 하나이므로 경로마다 새 그룹 id(`sids`)를 받는다.
    """
    if left <= 0:
        return 0
    x0, y0, _, _ = sc.roi
    res_w = 2.0 * float(np.median(dt[sc.residual & (dt > 0)])) if (sc.residual & (dt > 0)).any() else 2.0
    skel = _thin(sc.residual)
    skel = _prune_spurs(skel, max(3.0, 1.2 * res_w))
    n = 0
    rind_w = _RIND_W_MUL * 2.0 * _min_span(sc.upp)   # 껍질 컷 문턱 (게임 격자)
    paths = [p for p, _, _ in _join_paths(_paths(skel))]
    kept = []
    for path in paths:
        wmed = 2.0 * float(np.median(_dt_along(dt, path)))
        if len(path) < max(2.0, wmed):     # 폭보다 짧은 부스러기 — 마무리가 줍는다
            continue
        if not ink and (wmed < rind_w or len(path) < _RIND_LEN):
            # 굵은 영역 경계의 얇은 껍질 — 채움이 경계 밴드까지 닿고 이웃면·획이
            # 덮으므로 막대로 쫓지 않는다 (실측: 여기서 ~1,600장 샜다)
            continue
        kept.append((path, wmed))
    for path, wmed in kept:
        if n >= left:
            return n
        # 가격 (영역의 첫 장은 면제 — `fit_plan`의 같은 근거)
        if price and not (free_first and n == 0) \
                and _path_worth(sc, path, wmed) < price:
            continue
        # 경로 이동평균 — 뼈대의 계단 요철을 펴 긴 획이 서게 한다
        if len(path) >= 7:
            ker = np.array([1, 2, 3, 2, 1], np.float64) / 9.0
            mid = np.stack([np.convolve(path[:, 0], ker, "valid"),
                            np.convolve(path[:, 1], ker, "valid")], axis=1)
            path = np.concatenate([path[:1], mid, path[-1:]], axis=0)
        # 여기에는 획당 도형 상한(`policy.max_shapes`)을 안 건다 — 영역 안의 가는
        # 잔여 경로는 이미 3장 이하라 상한이 안 물린다 (실측 9장: 막대
        # 456→456·826→826처럼 그대로고 한 장은 +69로 되레 나빠졌다).
        # 상한이 실제로 무는 곳은 선화 쪽이다 (`_fit_lines`)
        n += _fit_path(plan, sc, dt, path, wmed, color, ink, left - n,
                       forms or ([], None), next(sids) if sids else -1)
    return n
