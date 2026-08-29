"""후보 경쟁 — 획 하나를 어떻게 그릴지 **여러 안을 지어 겨루게** 한다.

한 경로를 즉시 확정하지 않는다. 곡선 한 장·두 장·곡선+막대·막대 사슬·더 세게
단순화한 안을 함께 만들고, 전부 **게임 변환 격자로 양자화한 실제 렌더 상태**에서
재 비교한다. 채점 기하와 렌더 기하가 같은 폴리곤 식(`geometry._poly_px`)이라
"플랜 렌더 = 비교 결과"가 성립한다.

비교는 사전식이다 (`pick`):

1. 중요한 획의 위상 보존 — 끊김 수가 정책 상한 안인가
2. 허용 가능한 덮임 / 밴드 밖 스필
3. 도형 수 최소 — **그 끊김이 부를 이음 보수까지 합쳐서** (`_SEAM_PER_BREAK`)
4. 기하 오차 최소

도형 수를 줄이려고 획 자체를 지우지 않는다 — **같은 획을 더 적은 도형으로**가
먼저다. 그래서 ①②를 못 넘긴 안은 도형이 적어도 진다.

탐색은 획 단위로 갇혀 있고 결정적이다: 마디 후보는 RDP 마디 몇 단계의 합집합
이고, 그 위에서 **아핀 잔차**(닫힌 해, 라스터 없음)로 작은 동적계획을 돌려
분절을 고른 뒤, 이긴 분절만 실제로 놓아 본다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..catalog import Catalog
from ..model import Layer, LayerPlan
from . import intent as I
from .geometry import _mask_px, _min_span
from .scoring import _Scorer
from .skeleton import _cross2, _rdp_idx, _resample

# DP가 볼 마디 후보 수 상한 — 획 단위 탐색 공간을 여기서 가둔다.
_DP_NODES = int(os.environ.get("FS_DP_NODES", 10))
# 분절 하나를 한 장에 담을 자격 — 아핀 잔차(대응점당 RMS, 캔버스 유닛)를
# 그 분절의 폭으로 나눈 값. 폭 안이면 렌더에서 획 몸통에 묻힌다
_DP_RES = float(os.environ.get("FS_DP_RES", 0.75))
# 도형 한 장의 값 — DP 비용에서 잔차 대비 몇 배로 치나 (도형 수 우선)
_DP_SHAPE = float(os.environ.get("FS_DP_SHAPE", 3.0))
# 실제로 라스터까지 재 볼 후보 수 상한 (겨루기 비용의 뚜껑)
_MAX_CANDS = int(os.environ.get("FS_MAX_CANDS", 6))
# **끊김 하나가 부르는 이음 보수 장수** — 후보 비용에서 끊김을 몇 장으로 치나.
# 상수가 아니라 실측이다 (표준 10장 두 판, 그은 획 11,491개): 끊김이 0인 획은
# 이음 보수를 0.06장 쓰고, 1·2·3개인 획은 1.18·2.57·4.55장을 쓴다 —
# 최소제곱 기울기 1.48이다. 1로 세면 "적게 놓고 뒤에서 많이 깁는" 안이 그만큼
# 싸 보인다.
#
# **개수만으로는 부족하다** — 그 실측의 중앙 틈이 5.7px이라 이 값은 "짧은
# 틈 하나"의 값이다. 긴 획에서는 틈 하나가 수백 px인데 그것도 1.5장으로 세면
# **경로를 거의 안 그린 안이 이긴다**. 도형 상한을 길이에 비례시키자
# (`policy.shapes_for`) 그 편향이 바로 드러났다: 실측(01) 1,039px 실루엣에서
# 촘촘한 안(16장·덮임 0.72)을 제치고 도형 **1장**·덮임 0.105인 안이 비용 2.5로
# 뽑혔고, 뒤에서 이음 보수를 35장 물었다. 상한과 이 값은 한 짝이다.
#
# 그래서 틈 하나의 값을 **이음 보수가 실제로 그것을 어떻게 메우는지**로 잰다:
# `grammar._seg_bars`가 틈을 폭 절반 허용오차의 RDP로 끊어 마디마다 한 장을
# 놓으므로, 그 마디 수가 곧 장수다 (`_seam_shapes`). 새 상수를 세우지 않는
# 것이 요점이다 — 자가 보수 규칙 그 자체라 둘이 어긋날 수 없다. 짧은 틈은
# 마디가 하나라 `_SEAM_PER_BREAK`가 그대로 바닥이 된다.
_SEAM_PER_BREAK = float(os.environ.get("FS_SEAM_PER_BREAK", 1.5))


@dataclass
class Candidate:
    """획 하나를 그리는 한 가지 안 — 전부 양자화된 레이어다."""

    kind: str
    layers: list[Layer] = field(default_factory=list)
    cover: float = 0.0        # 경로 표본 중 잉크에 닿은 몫
    stray: float = 0.0        # 제 잉크 중 허용 밴드 밖 몫
    breaks: int = 0           # 선폭 이상 끊긴 자리 수
    seam_est: float = 0.0     # 그 끊김을 메우는 데 들 도형 수 (길이로 잰 추정)
    err: float = 0.0          # 기하 오차 (이상 띠와의 대칭차 / 띠 넓이)
    gain: float = 0.0         # 채점판 순이득 합 (참고)

    @property
    def n(self) -> int:
        return len(self.layers)

    def summary(self) -> dict:
        return {"kind": self.kind, "n": self.n, "cover": round(self.cover, 4),
                "stray": round(self.stray, 4), "breaks": self.breaks,
                "seam_est": round(self.seam_est, 2),
                "err": round(self.err, 4)}


# ── 평가 — 양자화된 최종 라스터에서 잰다 ──────────────────────────────
def _seam_shapes(seg: np.ndarray, wpx: float) -> float:
    """이 틈을 이음 보수가 메우는 데 들 도형 수 — **보수 규칙 그 자체로** 잰다.

    `grammar._patch_seams`는 틈을 `_seg_bars`로 끊어 마디마다 한 장을 놓는다
    (굽은 틈은 곡선 한 장으로 갈 때도 있으나, 실측상 긴 틈은 막대 사슬이다 —
    01의 1,039px 획이 35장이었고 그 자리 마디 수와 맞는다). 여기서는 같은
    RDP 허용오차(`0.5 × 폭`)로 그 마디 수를 세기만 한다: 라스터를 안 뜨고
    상수도 새로 안 세운다.
    """
    from .skeleton import _rdp_idx

    if len(seg) < 2:
        return _SEAM_PER_BREAK
    # `_rdp_idx`는 `_rdp`와 같은 마디를 스택으로 낸다 — 긴 틈에서 재귀가
    # 깊어질 일이 없다 (여기는 후보마다 불리는 자리다)
    pl = _rdp_idx(seg.astype(np.float64), max(1.0, 0.5 * max(wpx, 1.0)))
    return max(_SEAM_PER_BREAK, float(max(1, len(pl) - 1)))


def evaluate(cand: Candidate, cat: Catalog, upp: float, w: int, h: int,
             path_g: np.ndarray, wpx: float, allow: np.ndarray,
             min_w: float, ink_so_far: np.ndarray | None = None) -> Candidate:
    """후보를 **실제 렌더 상태**에서 재 채운다 (제자리 수정 후 반환).

    `path_g`는 전장 좌표 경로 표본, `allow`는 잉크가 나가도 되는 자리(전장)다.

    **덮임은 도안 전체의 잉크로 본다** (`ink_so_far`). 제 후보의 잉크만 보면
    교차하는 다른 획이 이미 그어 놓은 자리를 "안 덮였다"로 세고, 그것을 메우려
    도형이 한 장 더 붙는다 — 겹쳐 쌓이던 축이 이것이다. 판정은 렌더 결과를
    묻는 것이라 "이어져 보인다"의 정의에도 이쪽이 맞다. 스필은 반대로 **제
    잉크만** 본다 (남이 나간 몫을 이 후보에 물릴 수 없다).
    """
    if not cand.layers and (cand.kind != "skip" or ink_so_far is None):
        cand.cover, cand.stray, cand.breaks, cand.err = 0.0, 0.0, 1, 1.0
        cand.seam_est = _SEAM_PER_BREAK
        return cand
    pad = int(max(4.0, 2.0 * wpx)) + 3
    x0 = max(0, int(path_g[:, 1].min()) - pad)
    y0 = max(0, int(path_g[:, 0].min()) - pad)
    x1 = min(w, int(path_g[:, 1].max()) + pad + 1)
    y1 = min(h, int(path_g[:, 0].max()) + pad + 1)
    # 후보 잉크는 제 상자 밖으로도 나갈 수 있다 — 상자를 도형 bbox까지 넓힌다
    for lay in cand.layers:
        for p in _lay_pts(cat, lay, upp, w, h):
            x0 = max(0, min(x0, int(np.floor(p[:, 0].min())) - 1))
            y0 = max(0, min(y0, int(np.floor(p[:, 1].min())) - 1))
            x1 = min(w, max(x1, int(np.ceil(p[:, 0].max())) + 2))
            y1 = min(h, max(y1, int(np.ceil(p[:, 1].max())) + 2))
    if x0 >= x1 or y0 >= y1:
        cand.cover, cand.stray, cand.breaks, cand.err = 0.0, 1.0, 1, 1.0
        cand.seam_est = _SEAM_PER_BREAK
        return cand
    box = (x0, y0, x1, y1)
    ink = np.zeros((y1 - y0, x1 - x0), bool)
    for lay in cand.layers:
        ink |= _mask_px(cat, lay, upp, w, h, box)
    n_ink = float(ink.sum())
    if n_ink <= 0 and cand.kind != "skip":
        cand.cover, cand.stray, cand.breaks, cand.err = 0.0, 1.0, 1, 1.0
        cand.seam_est = _SEAM_PER_BREAK
        return cand
    # 덮임 — 경로 표본이 잉크에 **닿는가** (1px 팽창 = 렌더에서 맞닿음)
    seen = ink if ink_so_far is None else (ink | ink_so_far[y0:y1, x0:x1])
    grown = cv2.dilate(seen.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    py = np.clip(path_g[:, 0] - y0, 0, y1 - y0 - 1)
    px = np.clip(path_g[:, 1] - x0, 0, x1 - x0 - 1)
    cov = grown[py, px]
    cand.cover = float(cov.mean())
    # 끊김 — 안 덮인 연속 구간의 호길이가 선폭 이상이면 눈에 보이는 틈이다
    gap_min = max(2.0, min_w)
    seg = np.hypot(*np.diff(path_g.astype(np.float64), axis=0).T) \
        if len(path_g) > 1 else np.zeros(0)
    breaks, i = 0, 0
    seam_est = 0.0
    while i < len(cov):
        if cov[i]:
            i += 1
            continue
        j = i
        while j < len(cov) and not cov[j]:
            j += 1
        a, b = max(0, i - 1), min(len(path_g), j + 1)
        gap = float(seg[a:min(len(seg), j)].sum())
        if gap >= gap_min:
            breaks += 1
            # 이 틈이 부를 이음 보수 장수 — 보수가 쓰는 그 규칙으로 센다
            seam_est += _seam_shapes(path_g[a:b], max(wpx, min_w))
        i = j
    cand.breaks = breaks
    cand.seam_est = seam_est
    # 스필 — 허용 밴드 밖으로 나간 **제** 잉크 (남이 나간 몫은 안 문다)
    cand.stray = (float((ink & ~allow[y0:y1, x0:x1]).sum()) / n_ink
                  if n_ink > 0 else 0.0)
    # 기하 오차 — 경로를 제 폭으로 그은 **이상 띠**와의 대칭차
    ideal = np.zeros_like(ink, np.uint8)
    pts = np.stack([px, py], axis=1).round().astype(np.int32)
    cv2.polylines(ideal, [pts], False, 1, max(1, int(round(max(wpx, min_w)))))
    idl = ideal.astype(bool)
    cand.err = (float((ink ^ idl).sum()) / max(1.0, float(idl.sum()))
                if n_ink > 0 else 0.0)
    return cand


def _lay_pts(cat: Catalog, lay: Layer, upp: float, w: int, h: int):
    from .geometry import _poly_px
    return _poly_px(cat, lay, upp, w, h)


# ── 비교 — 사전식 ─────────────────────────────────────────────────────
def _tier(c: Candidate, pol) -> tuple:
    """정책이 요구하는 조건을 몇 단 어겼나 (작을수록 좋다)."""
    return (int(c.breaks > pol.breaks_max),
            int(c.cover < pol.cover_min) + int(c.stray > pol.stray_max))


def pick(cands: list[Candidate], pol) -> Candidate | None:
    """사전식 비교로 하나를 고른다 — 위상 → 덮임/스필 → 도형 수 → 기하 오차.

    같은 단이면 도형 수가 적은 쪽이 이기고, 그것도 같으면 오차가 작은 쪽이
    이긴다. 정책의 `err_weight`가 0이 아니면 도형 한 장을 오차 그만큼으로 쳐
    "한 장 더 써서 훨씬 정확한" 안이 이길 수 있다.

    `skip`(도형 0장)도 후보다 — 교차하는 다른 획이 이미 그 자리를 다 그었으면
    한 장도 안 쓰는 것이 사전식 최적이다.
    """
    live = [c for c in cands if c.layers or c.kind == "skip"]
    if not live:
        return None
    def key(c: Candidate):
        t = _tier(c, pol)
        # **끊김은 나중에 이음 보수가 도형으로 문다** — 그래서 도형 수만 보면
        # "적게 놓고 뒤에서 많이 깁는" 안이 이긴다 (실측: 끊김이 남은 획 291개가
        # 배치 798장·이음 848장을 썼다). 후보의 진짜 값은 제 도형 + 그 끊김을
        # 메울 도형이라, 이음 보수를 돌리는 정책에서는 끊김을 함께 센다 —
        # 장수는 그 틈을 보수가 실제로 어떻게 메우는지가 정하고(`seam_est`),
        # 짧은 틈에서는 실측 상수 `_SEAM_PER_BREAK`(1.48)가 바닥이다
        cost = c.n + (max(c.seam_est, _SEAM_PER_BREAK * c.breaks)
                      if pol.seam_repair else 0)
        return (t[0], t[1], cost + pol.err_weight * c.err, c.err, c.kind)
    return min(live, key=key)


# ── 후보 생성 ─────────────────────────────────────────────────────────
def _seg_residual(U: np.ndarray, path: np.ndarray, sc: _Scorer,
                  n_form: int) -> float:
    """분절 하나를 **한 장에 담았을 때**의 최소 아핀 잔차 (대응점당 RMS, 유닛).

    라스터를 안 뜬다 — DP가 마디를 고르는 동안 수만 번 묻는 자리라 값이 싸야
    한다. 실제 채점은 이긴 분절만 받는다.
    """
    from .stroke import _affine_fit

    rx0, ry0 = sc.roi[0], sc.roi[1]
    X = _resample(np.stack([(rx0 + path[:, 1] - sc.w / 2) * sc.upp,
                            (sc.h / 2 - (ry0 + path[:, 0])) * sc.upp],
                           axis=1), n_form)
    _, sx, sy, res = _affine_fit(U, X)
    ok = (np.abs(sx) >= 0.01) & (np.abs(sy) >= 0.01) & np.isfinite(res)
    if not ok.any():
        return float("inf")
    return float(np.sqrt(np.min(res[ok]) / n_form))


def _dp_nodes(path: np.ndarray, wmed: float, it=None,
              cap: int = _DP_NODES) -> list[int]:
    """마디 후보 — RDP 세 단계 + **의도된 각** (상한 `cap`). 결정적.

    각을 후보에 **더한다**: RDP는 현에서 먼 자리를 끊으므로 다리가 짧은 꺾임
    (눈꼬리·옷 주름)이 마디가 안 될 수 있고, 그러면 DP는 그 각을 도형 하나에
    뭉갤 수밖에 없다. 상한을 넘으면 각이 먼저 남는다 (`intent` 문서).
    """
    got = {0, len(path) - 1}
    for mul in (0.7, 1.4, 2.8):
        got.update(_rdp_idx(path, max(1.0, mul * max(wmed, 1.0))))
    corners = [i for i in I.corner_nodes(it, len(path)) if 0 < i < len(path) - 1]
    got.update(corners)
    idx = sorted(got)
    if len(idx) <= cap:
        return idx
    # 너무 많으면 **각이 먼저**, 그다음 현에서 먼 순 — 굽음이 큰 자리가 마디다
    p0, p2 = path[0], path[-1]
    chord = max(float(np.hypot(*(p2 - p0))), 1e-6)
    dev = np.abs(_cross2(p2 - p0, path - p0)) / chord
    room = cap - 2
    keep = corners[:room]
    rest = sorted((i for i in idx[1:-1] if i not in set(keep)),
                  key=lambda i: -dev[i])[:max(0, room - len(keep))]
    return sorted({0, len(path) - 1} | set(keep) | set(rest))


def dp_segments(path: np.ndarray, wmed: float, sc: _Scorer,
                forms: tuple, it=None, max_shapes: int = 0) -> list[int] | None:
    """도형 수와 잔차를 함께 최소화하는 마디 분할 (작은 동적계획, 결정적).

    비용 = 도형 수 × `_DP_SHAPE` + Σ(정규화 잔차) + **각 아닌 자리에서 끊는
    값**(`intent.cut_penalty`). 잔차가 폭의 `_DP_RES`를 넘는 분절은 한 장에 못
    담으므로 그 전이를 막는다. 마디 후보가 둘뿐이면 (= 통짜 한 장) 굳이 DP를
    안 돈다.

    끊는 값이 도형 수와 **같은 저울**에 서는 것이 요점이다 — "끊을 이유가
    분명하면 여전히 끊고, 같은 값이면 각에서 끊는다"가 한 부등식이 된다.
    """
    _, U = forms
    if U is None or len(path) < 4:
        return None
    # 마디 후보 수는 이 획에 허용된 도형 수를 따라간다 — 상한이 길이에
    # 비례하는데 후보가 상수면 긴 획에서 DP가 애초에 그 안을 못 지어 본다
    nodes = _dp_nodes(path, wmed, it,
                      max(_DP_NODES, min(int(max_shapes) + 2, 2 * _DP_NODES)))
    if len(nodes) < 3:
        return None
    n_form = U.shape[1]
    K = len(nodes)
    INF = float("inf")
    cost = [INF] * K
    prev = [-1] * K
    cost[0] = 0.0
    wlim = _DP_RES * max(wmed, 1.0) * sc.upp     # px 폭 → 캔버스 유닛
    for j in range(1, K):
        for i in range(j):
            if cost[i] == INF:
                continue
            seg = path[nodes[i]:nodes[j] + 1]
            if len(seg) < 3:
                continue
            r = _seg_residual(U, seg, sc, n_form)
            if not np.isfinite(r) or r > wlim:
                continue
            c = cost[i] + _DP_SHAPE + r / max(wlim, 1e-9)
            if i > 0:                      # 시작점은 끊는 자리가 아니다
                c += I.cut_penalty(it, nodes[i])
            if c < cost[j] - 1e-9:
                cost[j], prev[j] = c, i
    if cost[K - 1] == INF:
        return None
    out, j = [], K - 1
    while j >= 0:
        out.append(nodes[j])
        j = prev[j]
        if j < 0:
            break
    out.reverse()
    return out if len(out) >= 2 else None


def _place_chain(plan: LayerPlan, sc: _Scorer, dt: np.ndarray,
                 path: np.ndarray, idx: list[int], wmed: float, color,
                 sid: int, forms: tuple, wcap: float, strict: bool,
                 wprof: np.ndarray | None = None,
                 grammar: bool = True, it=None) -> int:
    """마디 분할대로 놓는다 — 마디마다 곡선과 막대를 같은 채점판에서 겨룬다."""
    from .stroke import _fit_path

    n = 0
    for k in range(len(idx) - 1):
        seg = path[idx[k]:idx[k + 1] + 1]
        if len(seg) < 2:
            continue
        n += _fit_path(plan, sc, dt, seg, wmed, color, True, 4, forms, sid,
                       depth=3, strict=strict, wcap=wcap,
                       wprof=None if wprof is None
                       else wprof[idx[k]:idx[k + 1] + 1], grammar=grammar,
                       it=it.sub(idx[k], idx[k + 1] + 1) if it else None)
    return n


def build(sc: _Scorer, dt: np.ndarray, path: np.ndarray, wmed: float, color,
          sid: int, forms: tuple, cat: Catalog, upp: float, w: int, h: int,
          allow: np.ndarray, pol, band: np.ndarray | None,
          max_shapes: int,
          ink_so_far: np.ndarray | None = None,
          core: np.ndarray | None = None,
          wprof: np.ndarray | None = None, it=None) -> list[Candidate]:
    """획 하나의 후보 집합 — 전부 같은 상태에서 지어 되돌린다.

    지어 보는 안: ⓪ 안 그리기(이미 덮였다) ① 곡선 한 장 ② DP 분절
    ③ 재귀 분할 ④ 단순화 두 단(절반 장수 · 두 장). ⓪이나 ①이 정책 조건을 다
    넘기면 사전식 최적이므로 거기서 멈춘다 — 그 자리가 전체의 94%라
    겨루기 비용이 현행에 얇게 붙는다 (실측 10장 +51% 시간).
    """
    from .stroke import _fit_path, _fit_segments, _try_curve

    min_w = 2.0 * _min_span(upp)
    # 획 도형 문법(테이퍼·가늘기·폭·끝 뭉툭함)은 **두 노선 공통**이다 — 선은
    # 모든 면 위에 마지막으로 얹히므로 셀에서도 잎사귀·쐐기가 그대로 보인다
    # (실측 01: 셀 노선의 앞머리 선이 검은 쐐기 덩어리였다)
    rx0, ry0 = sc.roi[0], sc.roi[1]
    p = path.round().astype(int)
    path_g = np.stack([np.clip(p[:, 0] + ry0, 0, h - 1),
                       np.clip(p[:, 1] + rx0, 0, w - 1)], axis=1)
    wcap = max(wmed, 0.0)
    out: list[Candidate] = []
    journal = sc.begin()

    def run(kind: str, fn) -> Candidate:
        mark = len(journal)
        scratch = LayerPlan(image_size=(w, h), units_per_px=upp)
        fn(scratch)
        c = Candidate(kind=kind, layers=list(scratch.layers))
        sc.rollback(journal, mark)
        return evaluate(c, cat, upp, w, h, path_g, wmed, allow, min_w,
                        ink_so_far)

    if band is not None:
        sc.set_band(band, core)
    try:
        # ⓪ 아무것도 안 그리기 — 교차하는 다른 획이 이미 다 그었을 수 있다.
        #    도형 0장이라 사전식으로는 무엇도 못 이긴다
        if ink_so_far is not None:
            skip = evaluate(Candidate(kind="skip"), cat, upp, w, h, path_g,
                            wmed, allow, min_w, ink_so_far)
            if _tier(skip, pol) == (0, 0):
                return [skip]
        # ① 곡선 한 장 — **자격 게이트 없이** 뽑는다. 한 장으로 그을 자격을
        #    고정 문턱으로 미리 묻지 않고, 아래 사전식 비교가 실제 렌더에서
        #    가린다 (§ curve gate를 후보 경쟁으로)
        def _one(sp: LayerPlan):
            got = _try_curve(sc, forms, path, wmed, color, True, sid,
                             race=True, line=True, gate=False, wprof=wprof)
            if got is not None:
                _, mfin = sc.score(got[1])
                sc.commit(mfin)
                sp.layers.append(got[1])

        c1 = run("curve1", _one)
        if c1.layers:
            out.append(c1)
            if _tier(c1, pol) == (0, 0):
                return out                       # 사전식 최적 — 더 볼 것이 없다

        # ② DP 분절 — 도형 수와 잔차를 함께 최소화한 마디
        idx = dp_segments(path, wmed, sc, forms, it, max_shapes)
        if idx is not None and len(idx) - 1 <= max_shapes:
            out.append(run("dp%d" % (len(idx) - 1),
                           lambda sp: _place_chain(sp, sc, dt, path, idx, wmed,
                                                   color, sid, forms, wcap,
                                                   pol.name == "line", wprof,
                                                   it=it)))
        # ③ 현행 재귀 분할 — 곡선이 안 되면 쪼개서 다시 곡선
        out.append(run("split", lambda sp: _fit_path(
            sp, sc, dt, path, wmed, color, True, max_shapes, forms, sid,
            strict=pol.name == "line", wcap=wcap, wprof=wprof,
            grammar=True, it=it)))
        # ④ 단순화 두 단 — 같은 획을 더 적은 도형으로. `_fit_segments`는 마디가
        #    허용 장수를 넘으면 **허용오차를 키워** 마디를 줄이므로, 장수를
        #    조여 주는 것이 곧 "더 굵게 긋고 덜 쪼갠다"다. 두 단을 다 지어
        #    비교에 넘긴다 — 세게 줄인 안이 조건을 넘기면 그쪽이 이긴다
        if max_shapes > 2:
            out.append(run("coarse", lambda sp: _fit_segments(
                sp, sc, dt, path, wmed, color, True,
                max(2, max_shapes // 2), sid, strict=pol.name == "line",
                wcap=wcap, forms=forms, wprof=wprof, grammar=True, it=it)))
        out.append(run("simple", lambda sp: _fit_segments(
            sp, sc, dt, path, wmed, color, True, 2, sid,
            strict=pol.name == "line", wcap=wcap, forms=forms, wprof=wprof,
            grammar=True, it=it)))
    finally:
        if band is not None:
            sc.set_band(None)
        sc.end()
    return out[:_MAX_CANDS]


def commit(plan: LayerPlan, sc: _Scorer, cand: Candidate) -> int:
    """이긴 후보를 실제로 굳힌다 — 채점판 잔여도 그때 함께 지운다."""
    for lay in cand.layers:
        _, mfin = sc.score(lay)
        sc.commit(mfin)
        plan.layers.append(lay)
    return len(cand.layers)
