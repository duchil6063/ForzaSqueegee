"""면 채움 — 봉우리마다 씨앗을 잡고 하강으로 다듬어 영역을 덮는다.

닫힌 해로 씨앗을 잡는다: 2차 모멘트가 같은 타원·사각을 경쟁시켜(`_seed_moment`)
이긴 쪽을 쓰고, 확장 어휘 여섯은 배수 `_FILL_MARGIN`을 넘겨야 바탕을 이긴다.
남은 얇은 껍질은 막대 사슬(`_fit_bars`)이, 남은 큰 조각은 마무리
(`layered.mop_up`)가 줍는다 — 그보다 작은 조각은 구멍 메움이 군집당 한 장으로
더 싸게 처리한다.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

from ..catalog import Catalog
from ..model import UNITS_PER_SCALE, Layer, LayerPlan
from . import affine as A
from . import census as _census
from . import policy as _policy
from .geometry import _layer, _min_span
from .scoring import _Scorer, _descend
from .skeleton import _dt_along, _join_paths, _paths, _prune_spurs, _thin
from .stroke import _FORM_RASTER, _fit_path, _path_worth
from .vocabulary import (_FILL_BASE, _FILL_MARGIN, _FILL_MIN_R0,
                         _FILL_SHAPE, _FILL_SHAPES, _FILL_TMPL, _FILL_TOP)

# 도형별 **전** 2차 모멘트 (중심, 공분산) — `_FILL_TMPL`의 짝 (프로세스 1회)
_FILL_TMPL_FULL: dict = {}


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


# ── 획형 판정 — **이 영역은 획인가, 가는 면인가.**
#
# 가늘다는 것만으로는 못 가른다. 셀 분해는 선화가 안 지운 선 조각도, 눈
# 흰자·머리칼 하이라이트·리본·가는 음영도 똑같이 가는 영역으로 낸다. 앞엣것은
# 뼈대를 따라 획으로 그어야 하고 뒤엣것은 **면으로 채워야** 한다 — 획으로
# 보내면 짧은 조각은 경로 길이 문턱에 걸려 통째로 안 그려지고(그 자리가
# 그대로 미커버가 된다), 그려져도 획 라벨을 받아 프루닝에서 선 대접을 받는다.
#
# 자는 넷이고 **전부 이미 재고 있는 것**이다. 앞 셋은 기하·위상이라 여기서
# 재고, 넷째(색)는 지도를 봐야 해서 호출부가 재어 넘긴다:
#
# ① **가늘다** — 픽셀 85%가 경계에서 `_THIN_R`px 안. "최대 내접 반경"만 보면
#    선의 교차점(굵다) 때문에 타원 채움으로 넘어가 얼룩이 된다.
# ② **길다** — 면적 / 폭². 폭 w·길이 L인 띠는 면적이 wL이므로 이 값이 곧
#    L/w(가로세로비)다. 둥근 조각은 1 근처, 획은 대개 다섯을 넘는다. 뼈대를
#    다시 안 뽑고 거리변환 하나로 나오는 닫힌 셈이라 값이 싸다.
# ③ **닫혔다** — 구멍이 하나라도 있으면 그 영역은 **고리**다 (눈테·리본
#    매듭·안경). 획은 열린 곡선이라 구멍이 없다. 고리를 획으로 보내면 뼈대가
#    순환이라 경로가 임의로 끊기고 그 자리가 통째로 빈다.
# ④ **제 색이 있다** (`inklike`, 호출부) — 잔여 선 조각은 색이 **그 자리
#    획의 색**이다 (선화가 안 지운 그 선이니까). 눈 흰자·하이라이트는 획
#    색과 또렷이 다르다. 색이 다르고 경계가 획이 아니라 **색 경계**면 그것은
#    면이다 — 획으로 보내면 그 색이 통째로 사라진다.
#
# 문턱 3은 **면 채움이 한 장으로 끝낼 수 있는 가로세로비**다: 채움 어휘의
# 씨앗이 2차 모멘트 정합이라(`_seed_moment`) 3:1까지는 도형 한 장이 그대로
# 맞고, 그보다 길어지면 한 장으로는 못 덮어 마디로 쪼개는 쪽이 싸다.
_THIN_R = float(os.environ.get("FS_THIN_R", 3.2))
_THIN_ELONG = float(os.environ.get("FS_THIN_ELONG", 3.0))


def region_shape(mask: np.ndarray, dt: np.ndarray,
                 inklike: bool = True) -> tuple[bool, float, float]:
    """(획형인가, 폭 중앙값 px, 가로세로비) — `dt`는 mask의 거리변환.

    `inklike`는 호출부가 재는 넷째 자다 (위 ④): 이 영역의 색이 그 자리
    획의 색이거나 경계가 획에 붙어 있으면 True. False면 — 색도 다르고
    경계도 색 경계면 — 가늘고 길어도 **면**으로 본다.
    """
    d = dt[mask]
    if not d.size:
        return False, 0.0, 0.0
    thin = float(np.percentile(d, 85)) <= _THIN_R
    wmed = 2.0 * float(np.median(d))
    elong = float(np.count_nonzero(mask)) / max(wmed * wmed, 1.0)
    if not (thin and elong >= _THIN_ELONG and inklike):
        return False, wmed, elong
    return (not _has_hole(mask)), wmed, elong


def _has_hole(mask: np.ndarray) -> bool:
    """이 영역이 무언가를 **둘러싸고 있나** (구멍 ≥ 1) — 고리는 획이 아니다."""
    u = np.pad(mask.astype(np.uint8), 1)
    n_bg, _ = cv2.connectedComponents((1 - u).astype(np.uint8), connectivity=4)
    return n_bg > 2


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


def _fill_tmpl_full(cat: Catalog, name: str):
    """도형 단위 마스크의 **전 2차 모멘트** — (중심 (2,), 공분산 (2,2)).

    `_fill_tmpl`은 로컬 x·y의 표준편차만 낸다 (교차 모멘트를 안 본다). 그것이
    현행 씨앗이 대각 성분만 맞추는 까닭이고, 교차 모멘트가 0이 아닌 도형
    (초승달·삼각·쐐기)에서 씨앗이 **애초에 모멘트가 안 맞는 자세**로 서는
    까닭이다. 여기서는 세 성분을 다 들고 있는다.

    단위는 `affine.linear`가 기대하는 그대로 — 로컬 × `UNITS_PER_SCALE`이라
    여기서 나온 `sx`가 `Layer.sx`에 바로 들어간다. 좌표계는 도형 로컬(y-up)이고
    캔버스도 y-up이라 뒤집을 것이 없다. 프로세스 1회 계측.
    """
    t = _FILL_TMPL_FULL.get(name)
    if t is None:
        sh = cat[name]
        m = sh.rasterize(_FORM_RASTER)
        pts_all = np.concatenate(sh.loops, axis=0)
        lo = pts_all.min(axis=0)
        span = np.maximum(pts_all.max(axis=0) - lo, 1e-6)
        ys, xs = np.nonzero(m)
        if len(ys) < 8:
            t = (np.zeros(2), np.eye(2))
        else:
            lx = xs / (_FORM_RASTER - 1) * span[0] + lo[0]
            ly = (_FORM_RASTER - 1 - ys) / (_FORM_RASTER - 1) * span[1] + lo[1]
            p = np.stack([lx, ly], axis=1).astype(np.float64) * UNITS_PER_SCALE
            t = (p.mean(axis=0), np.cov(p.T))
        _FILL_TMPL_FULL[name] = t
    return t


def _seed_affine(sc: _Scorer, pw: np.ndarray, name: str, color,
                 cat: Catalog) -> list[Layer]:
    """**전 아핀** 모멘트 씨앗 — 2차 모멘트 세 성분을 정확히 맞춘다 (§8).

    현행 씨앗(`_seed_moment`)은 우리 자유도를 `회전 + 축별 스케일`로 보고
    양쪽의 **주축**만 맞춘다. 전단을 넣으면 상이 2×2 전체라, 모멘트를
    맞추는 해가 닫힌 식으로 바로 나온다: `M·Cₜ·Mᵀ = Cₓ`의 해가
    `M = Cₓ^½·Q·Cₜ^-½`이고 `Q`는 직교행렬이다 (`affine.fit_moment`).
    후보는 현행과 같은 네 자세(90° 배수)라 개수가 안 는다.

    이것이 §8이 말하는 자리다 — 사선 머리칼 덩어리·옷자락·비스듬한 그림자
    처럼 **평행사변형에 가까운 면**은 회전 + 비등방 스케일 한 장으로는
    몸통이 안 덮여 잔차 보정이 여러 장 붙는데, 전 아핀 한 장이면 통째로 덮는다.

    좌표는 캔버스 유닛에서 푼다 (레이어가 사는 계) — 이미지 px에서 풀고
    나중에 뒤집으면 전단의 부호가 프레임마다 갈린다.
    """
    if len(pw) <= 8:
        return []
    tc, ct = _fill_tmpl_full(cat, name)
    x0, y0, _, _ = sc.roi
    upp, w, h = sc.upp, sc.w, sc.h
    # 이미지 px 점 구름 → 캔버스 유닛 (`geometry._layer`와 같은 환산)
    ux = (x0 + pw[:, 0] - w / 2.0) * upp
    uy = (h / 2.0 - (y0 + pw[:, 1])) * upp
    q = np.stack([ux, uy], axis=1)
    mu = q.mean(axis=0)
    cx = np.cov(q.T)
    if not np.all(np.isfinite(cx)) or not np.all(np.isfinite(ct)):
        return []
    out = []
    for k in range(4):
        M = A.fit_moment(ct, cx, k)
        if not np.all(np.isfinite(M)):
            continue
        rot, sxv, syv, skv = A.decompose_linear(M)
        skv = A.q_skew(skv)
        # 전단이 0으로 접히면 현행 씨앗과 같은 자세다 — 중복은 안 넣는다
        if skv == 0.0 or not A.representable(skv):
            continue
        if abs(sxv) < 0.2 or abs(syv) < 0.2:
            continue
        c_ = np.array([[M[0, 0], M[0, 1]], [M[1, 0], M[1, 1]]]) @ tc
        out.append(Layer(shape=name,
                         x=float(mu[0] - c_[0]), y=float(mu[1] - c_[1]),
                         sx=float(sxv), sy=float(syv),
                         rot=float(rot % 360.0), skew=skv,
                         color=tuple(int(v) for v in color), alpha=100.0,
                         label="cel"))
    return out


def _seed_moment(sc: _Scorer, pw: np.ndarray, name: str, color,
                 cat: Catalog, skew: bool = False) -> list[Layer]:
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
    # **전 아핀 씨앗을 옆에 세운다** (§8) — 기존 넷은 하나도 안 지운다.
    # 어느 쪽이 설지는 같은 목적함수가 정한다 (`_place_fat`·`_place_whole`이
    # 씨앗마다 `score_val`을 물어 최선에서 하강을 시작한다)
    if skew and A.skew_useful(cat, name):
        out.extend(_seed_affine(sc, pw, name, color, cat))
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
    plain: dict[int, Layer] = {}           # 어휘별 **전단 없는** 최선 씨앗
    for i, name in enumerate(vocab):
        cand = Layer(**{**seed.__dict__})
        cand.shape = name
        best_s, best_c = sc.score_val(cand), cand
        plain[i] = cand
        # 모멘트 정합 씨앗 — 같은 증거(창 점 구름)에서 도형마다 따로 푼 자세.
        # 같은 목적함수로 재어 좋은 쪽에서 하강을 시작한다 (채움 장수
        # 5~32% 감소·커버리지 상승. 확장 어휘가 서는 것도 이 씨앗 덕이다)
        for alt in _seed_moment(sc, pw, name, color, sc.cat,
                                skew=_policy.skew_fill_default()):
            s = sc.score_val(alt)
            if s > best_s:
                best_s, best_c = s, alt
            if not alt.skew and s > sc.score_val(plain[i]):
                plain[i] = alt
        scored.append((-best_s, i, best_c))
    scored.sort(key=lambda t: t[:2])          # 동점은 어휘 순서 — 결정적
    # 하강 순서는 다시 **어휘 순서**로 되돌린다 — 하강 뒤 동점이면 앞 어휘가
    # 이기게 해, 어휘를 넓혀도 기존 어휘의 판정이 안 흔들리게 한다
    keep = {t[1] for t in scored[:max(1, _FILL_TOP)]}
    keep |= {i for i, n in enumerate(vocab) if n in _FILL_BASE}
    best = base = None
    for _, i, cand in sorted((t for t in scored if t[1] in keep), key=lambda t: t[1]):
        g, q = _descend(sc, cand, color, passes=2, skew=cand.skew != 0.0)
        # **전단 없는 안을 하강까지 데려간다** (§5). 씨앗 점수만으로 고르면
        # 전단 씨앗이 한 끗 앞선 자리에서 전단 없는 안이 **하강도 못 해 보고**
        # 탈락한다 — 그런데 하강은 두 안을 서로 다른 국소 최적으로 데려가므로
        # 씨앗 순위가 결과 순위가 아니다. 실측(표준 4장, 이 손 없이):
        # 레이어는 3~23장 줄었는데 보이는 오차가 4장 중 3장에서 나빠졌고
        # 04는 0.162 → 0.246이었다. 동률이면 전단 없는 쪽이 이긴다
        if q.skew:
            g0, q0 = _descend(sc, plain[i], color, passes=2)
            if g0 >= g:
                g, q = g0, q0
        if vocab[i] in _FILL_BASE and (base is None or g > base[0]):
            base = (g, q)
        if best is None or g > best[0]:
            best = (g, q)
    if base is not None and best[1].shape not in _FILL_BASE \
            and best[0] < _FILL_MARGIN * base[0]:
        best = base
    return _grow_step(sc, *_descend(sc, best[1], color, passes=3,
                                    skew=best[1].skew != 0.0))


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
    skel_raw = _thin(sc.residual)
    skel = _prune_spurs(skel_raw, max(3.0, 1.2 * res_w))
    n = 0
    rind_w = _RIND_W_MUL * 2.0 * _min_span(sc.upp)   # 껍질 컷 문턱 (게임 격자)
    # 계측 (`census`) — 깔때기 칸을 열고 접합 판정을 받아 적는다. 끄면 없다
    cen = _census.bar_open(
        res_px=int(np.count_nonzero(sc.residual)), res_w=round(res_w, 3),
        skel_raw_px=int(skel_raw.sum()), skel_px=int(skel.sum()),
        ink=bool(ink), price=round(float(price), 3), left=int(left),
        rind_w=round(rind_w, 3), rind_len=_RIND_LEN) if _census.ON else None
    raw = _paths(skel)
    joined = _join_paths(raw, rec=_census.join_rec(cen) if cen else None)
    if cen is not None:
        cen["raw_paths"] = len(raw)
        cen["joined_paths"] = len(joined)
        cen["free_ends"] = sum(1 for _p, hj, tj in joined if hj < 0 and tj < 0)
        cen["one_junc"] = sum(1 for _p, hj, tj in joined
                              if (hj < 0) != (tj < 0))
        cen["two_junc"] = sum(1 for _p, hj, tj in joined if hj >= 0 and tj >= 0)
        cen["raw_len"] = [int(len(p)) for p, _a, _b in raw]
    paths = [p for p, _, _ in joined]
    kept = []
    for path in paths:
        wmed = 2.0 * float(np.median(_dt_along(dt, path)))
        if len(path) < max(2.0, wmed):     # 폭보다 짧은 부스러기 — 마무리가 줍는다
            if cen is not None:
                cen["drop"]["short"] = cen["drop"].get("short", 0) + 1
            continue
        if not ink and (wmed < rind_w or len(path) < _RIND_LEN):
            # 굵은 영역 경계의 얇은 껍질 — 채움이 경계 밴드까지 닿고 이웃면·획이
            # 덮으므로 막대로 쫓지 않는다 (실측: 여기서 ~1,600장 샜다)
            if cen is not None:
                k = "rind_w" if wmed < rind_w else "rind_len"
                cen["drop"][k] = cen["drop"].get(k, 0) + 1
            continue
        kept.append((path, wmed))
    if cen is not None:
        cen["kept_paths"] = len(kept)
    for path, wmed in kept:
        if n >= left:
            if cen is not None:
                cen["drop"]["budget"] = cen["drop"].get("budget", 0) + 1
            return n
        # 가격 (영역의 첫 장은 면제 — `fit_plan`의 같은 근거)
        if price and not (free_first and n == 0) \
                and _path_worth(sc, path, wmed) < price:
            if cen is not None:
                cen["drop"]["price"] = cen["drop"].get("price", 0) + 1
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
        if cen is None:
            n += _fit_path(plan, sc, dt, path, wmed, color, ink, left - n,
                           forms or ([], None), next(sids) if sids else -1)
            continue
        # ── 계측 경로 — 단위 하나의 계보를 그대로 남긴다 (판정은 같다)
        u = _census.bar_unit(
            cen, samples=int(len(path)),
            arc=round(float(np.hypot(*np.diff(path, axis=0).T).sum()), 2),
            wmed=round(wmed, 3), lo=len(plan.layers))
        band = np.zeros(sc.residual.shape, np.uint8)
        cv2.polylines(band, [np.stack([path[:, 1], path[:, 0]], axis=1)
                             .round().astype(np.int32)], False, 1,
                      max(1, int(round(wmed))))
        u.update(_census.where(band.astype(bool), cen.get("_dedge"),
                               cen.get("_ink")))
        u["worth"] = round(_path_worth(sc, path, wmed), 2)
        got = _fit_path(plan, sc, dt, path, wmed, color, ink, left - n,
                        forms or ([], None), next(sids) if sids else -1,
                        rec=u)
        u["layers"] = int(got)
        n += got
    if cen is not None:
        cen.pop("_dedge", None)
        cen.pop("_ink", None)
        cen["bar_layers"] = int(n)
        cen["bar_units"] = len(cen["units"])
    return n
