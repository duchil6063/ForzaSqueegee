"""도형 서술자 — 어휘를 **이름이 아니라 놓인 뒤의 모습**으로 고른다.

카탈로그의 도형 이름·계열은 게임 UI의 분류일 뿐 기하가 아니다. 여기서는
창 조작으로 도달하는 도형 전부를 한 번 계측해 서술자를 만들고, 획 어휘는
그 서술자로 판정한다 — 같은 도형도 비등방 스케일에 따라 획이 되기도 잎사귀가
되기도 하기 때문이다.

계측은 프로세스 1회이고 `work/cache/`에 캐시한다 (카탈로그가 바뀌면 다시
잰다). 캐시가 없어도 결과는 같다 — 값이 아니라 시간만 든다.

**가는 획의 자**가 여기 있다: 게임 최소 스케일(0.01)에서 도형이 실제로 칠하는
폭은 짧은 축의 반길이가 정한다. 상자가 정사각인 막대(A_22)는 그 폭이 최소 도형
폭 그대로지만, 짧은 축이 더 짧은 도형은 훨씬 가늘게 서고 **폭의 눈금도 그만큼
촘촘하다** (`min_width_px`). 실측(h=1961): 막대 2.79px ↔ U_45 0.19px — 사람 폭
4.1px 목표가 막대에서는 2.79px로 반올림되던 자리다.

배치가 지금 읽는 칸은 `stroke_ok`·`closed`·`length`·`hw_max`(어휘 자격),
`curv`(곧음), `ext_x`·`ext_y`(폭 눈금)다. 나머지(`sdf`·`pyr`·`sym`·`ecc`·
`convex`·`taper`·`slim`)는 어휘를 다시 고를 때 쓰는 **오프라인 분석 표면**이고
`tools/`가 읽는다 — 놓을 때의 테이퍼·가늘기 판정은 라스터가 아니라 닫힌 식
(`stroke._placed_form`)이 한다.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..catalog import Catalog, default_catalog_path
from ..model import UNITS_PER_SCALE
from .skeleton import _paths, _prune_spurs, _resample, _thin

_RASTER = 160          # 중심선·폭을 재는 등방 래스터 해상도
_N = 16                # 중심선 대응점 수 (아핀 맞춤과 같은 수)
_SDF = 32              # 부호 거리장 해상도
_PYR = (32, 16, 8)     # 저해상 래스터 피라미드
_SDF_N = 32            # 모멘트 정규화 라스터 한 변 (§6 후보 순위)
_CURVB = 32            # 외곽 곡률 프로파일 표본 수
_SCHEMA = 3            # 캐시 스키마 판 (칸이 늘면 올린다 — 옛 캐시는 무시된다)


@dataclass(frozen=True)
class ShapeDesc:
    """도형 하나의 형태 서술자 — 전부 로컬 좌표 × `UNITS_PER_SCALE` 단위."""

    name: str
    stroke_ok: bool          # 중심선이 한 가닥 — 획 후보 자격
    closed: bool             # 그 중심선이 **고리**인가 (반지·테두리 도형)
    center: np.ndarray       # (N,2) 중심선
    halfw: np.ndarray        # (N,) 중심선 위 반폭
    curv: np.ndarray         # (N-2,) 중심선 곡률 프로파일
    tan0: np.ndarray         # (2,) 머리 접선 (단위)
    tan1: np.ndarray         # (2,) 꼬리 접선
    hw_min: float            # 능선 최소 반폭 (10분위)
    hw_max: float            # 능선 최대 반폭
    ext_x: float             # bbox 로컬 x 반길이 (×UNITS_PER_SCALE)
    ext_y: float             # bbox 로컬 y 반길이
    length: float            # 중심선 길이
    taper: float             # hw_max / hw_min (등방)
    slim: float              # 2×중앙 반폭 / 길이
    ecc: float               # 이심률 (2차 모멘트 √(λ1/λ2))
    sym: float               # 주축 반전 IoU (1 = 완전 대칭)
    convex: float            # 면적 / 볼록 껍질 면적
    sdf: np.ndarray          # (_SDF,_SDF) float32 — 안쪽 +, 바깥 −, 반경 정규화
    pyr: tuple               # 저해상 래스터 피라미드 (bool)
    # ── §6 형태 서술 — 어느 primitive가 이 영역을 닮았나
    area: float              # 면적 (로컬 유닛²)
    peri: float              # 외곽 둘레 (로컬 유닛)
    circ: float              # 원형도 4πA/P² (1 = 원)
    concave: float           # 최대 오목 깊이 / √면적 (0 = 볼록)
    branches: int            # 세선화 가지 수 (끝점 + 분기점)
    mom: tuple               # 주 모멘트 (λ1, λ2) — 면적으로 정규화
    curvb: np.ndarray        # (_CURVB,) 외곽 곡률 프로파일 (둘레 등간격)
    npyr: np.ndarray         # (_SDF_N,_SDF_N) bool — **모멘트 정규화** 라스터

    @property
    def ext_min(self) -> float:
        return min(self.ext_x, self.ext_y)

    @property
    def ext_max(self) -> float:
        return max(self.ext_x, self.ext_y)

    @property
    def aspect(self) -> float:
        """가로세로비 — 긴 반길이 / 짧은 반길이 (상자 기준, ≥1)."""
        return float(self.ext_max / max(self.ext_min, 1e-9))

    @property
    def long_is_x(self) -> bool:
        """긴 축이 로컬 x인가 — 막대로 쓸 때 회전에 90°를 더할지 정한다."""
        return self.ext_x >= self.ext_y

    def min_width_px(self, upp: float) -> float:
        """최소 스케일(0.01)에서 이 도형이 칠하는 **가장 가는 폭** px.

        게임은 `loop × (sx,sy) × 64`를 그리므로, 짧은 축을 스케일 0.01로 누르면
        폭이 `2 × ext_min × 0.01 / upp`가 된다. 막대(A_22)는 상자가 정사각이라
        이 값이 최소 도형 폭 그 자체지만, 짧은 축이 더 짧은 도형은 그보다 가늘게
        서고 **폭의 눈금(스케일 0.01 한 칸)도 그만큼 촘촘하다**.
        """
        return 2.0 * self.ext_min * 0.01 / upp


def _iso_raster(sh, size: int) -> tuple[np.ndarray, float, np.ndarray]:
    """도형을 **등방**으로 라스터 — (마스크, 로컬→px 배율, bbox 최소 모서리).

    `CatShape.rasterize`는 bbox를 정사각으로 늘려 축마다 배율이 달라진다 —
    대조(IoU)에는 써도 **폭을 재는 데는 못 쓴다**.
    """
    if not sh.loops:
        return np.zeros((size, size), bool), 1.0, np.zeros(2, np.float32)
    pts = np.concatenate(sh.loops, axis=0)
    lo = pts.min(axis=0)
    span = np.maximum(pts.max(axis=0) - lo, 1e-6)
    s = (size - 3) / float(span.max())
    acc = np.zeros((size, size), np.uint8)
    for loop in sh.loops:
        q = (loop - lo) * s + 1.0
        q[:, 1] = size - 1 - q[:, 1]
        one = np.zeros_like(acc)
        cv2.fillPoly(one, [np.round(q).astype(np.int32)], 1)
        acc ^= one
    return acc.astype(bool), s, lo


def _moments(m: np.ndarray) -> tuple[float, float]:
    """(이심률, 주축 각) — 2차 모멘트."""
    ys, xs = np.nonzero(m)
    if len(ys) < 3:
        return 1.0, 0.0
    cy, cx = ys.mean(), xs.mean()
    dy, dx = ys - cy, xs - cx
    c20, c02, c11 = (dx * dx).mean(), (dy * dy).mean(), (dx * dy).mean()
    tr = c20 + c02
    disc = max(0.0, tr * tr / 4.0 - (c20 * c02 - c11 * c11)) ** 0.5
    l1, l2 = tr / 2.0 + disc, max(tr / 2.0 - disc, 1e-9)
    return float((l1 / l2) ** 0.5), float(0.5 * np.arctan2(2.0 * c11, c20 - c02))


def _symmetry(m: np.ndarray, theta: float) -> float:
    """주축에 대한 반전 IoU — 1이면 완전 대칭."""
    h, w = m.shape
    c = ((w - 1) / 2.0, (h - 1) / 2.0)
    deg = float(np.degrees(theta))
    rot = cv2.getRotationMatrix2D(c, deg, 1.0)
    a = cv2.warpAffine(m.astype(np.uint8), rot, (w, h), flags=cv2.INTER_NEAREST)
    b = a[::-1]
    u = int(np.logical_or(a, b).sum())
    return float(np.logical_and(a, b).sum()) / max(1, u)


def _convexity(m: np.ndarray) -> float:
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 1.0
    c = max(cnts, key=cv2.contourArea)
    hull = cv2.contourArea(cv2.convexHull(c))
    return float(cv2.contourArea(c) / hull) if hull > 1e-6 else 1.0


def _sdf(m: np.ndarray) -> np.ndarray:
    """부호 거리장 (안쪽 +, 바깥 −) — 최대 거리로 정규화한 (_SDF,_SDF)."""
    u = m.astype(np.uint8)
    din = cv2.distanceTransform(u, cv2.DIST_L2, 3)
    dout = cv2.distanceTransform(1 - u, cv2.DIST_L2, 3)
    f = din - dout
    r = max(float(np.abs(f).max()), 1e-6)
    return cv2.resize(f / r, (_SDF, _SDF), interpolation=cv2.INTER_AREA).astype(np.float32)


def norm_raster(m: np.ndarray, n: int = _SDF_N) -> np.ndarray:
    """마스크 → **모멘트 정규화** 라스터 (n,n) bool — 형태 대조의 공통 자.

    우리 배치 자유도는 회전 + 축별 스케일(전단 없음)이다. 그러니 두 형태를
    "같은 모양인가"로 견주려면 **그 자유도만큼 미리 정규화**해 놓아야 한다:
    무게중심으로 옮기고, 주축으로 돌리고, 축마다 표준편차로 나눈다. 남는
    자유도는 축 부호 넷뿐이라 대조가 유한하다 (미러는 음수 스케일로 낸다).

    창은 ±2.5σ — 이보다 좁으면 뾰족한 도형의 끝이 잘리고, 넓으면 몸통이
    가운데 몇 칸으로 뭉쳐 구분이 안 선다.
    """
    ys, xs = np.nonzero(m)
    if len(ys) < 8:
        return np.zeros((n, n), bool)
    if len(ys) > 20000:                    # 큰 마스크는 균일 솎기 (결정적)
        step = len(ys) // 20000 + 1
        ys, xs = ys[::step], xs[::step]
    p = np.stack([xs, ys], axis=1).astype(np.float64)
    mu = p.mean(0)
    cov = np.cov((p - mu).T)
    ev, evec = np.linalg.eigh(cov)
    idx = np.argsort(-ev)
    sig = np.sqrt(np.maximum(ev[idx], 1e-9))
    q = (p - mu) @ evec[:, idx] / sig
    g = np.clip(((q + 2.5) / 5.0 * n).astype(np.int64), 0, n - 1)
    out = np.zeros((n, n), bool)
    out[g[:, 1], g[:, 0]] = True
    return out


def _boundary(m: np.ndarray):
    """(외곽 컨투어, 면적 px, 둘레 px) — 가장 큰 성분 하나."""
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None, 0.0, 0.0
    c = max(cnts, key=cv2.contourArea)
    return c, float(cv2.contourArea(c)), float(cv2.arcLength(c, True))


def _concavity(c) -> float:
    """최대 오목 깊이 / √면적 — 볼록이면 0."""
    if c is None or len(c) < 4:
        return 0.0
    hull = cv2.convexHull(c, returnPoints=True).reshape(-1, 2).astype(np.float32)
    if len(hull) < 3:
        return 0.0
    a = max(cv2.contourArea(c), 1.0)
    pts = c.reshape(-1, 2).astype(np.float32)
    d = np.array([cv2.pointPolygonTest(hull.reshape(-1, 1, 2), (float(x), float(y)), True)
                  for x, y in pts[::max(1, len(pts) // 256)]], np.float32)
    return float(max(0.0, d.max()) / np.sqrt(a))


def _curv_profile(c, n: int = _CURVB) -> np.ndarray:
    """외곽 곡률 프로파일 — 둘레 등간격 n표본 (부호 있는 회전각/길이)."""
    if c is None or len(c) < n + 2:
        return np.zeros(n, np.float32)
    p = c.reshape(-1, 2).astype(np.float64)
    d = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))])
    L = float(d[-1])
    if L < 1e-6:
        return np.zeros(n, np.float32)
    t = np.linspace(0.0, L, n, endpoint=False)
    q = np.stack([np.interp(t, d, p[:, 0]), np.interp(t, d, p[:, 1])], axis=1)
    v1 = q - np.roll(q, 1, axis=0)
    v2 = np.roll(q, -1, axis=0) - q
    cr = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    den = np.hypot(*v1.T) * np.hypot(*v2.T)
    ang = np.arcsin(np.clip(cr / np.maximum(den, 1e-9), -1.0, 1.0))
    return (ang / max(L / n, 1e-9)).astype(np.float32)


def _branch_count(skel: np.ndarray) -> int:
    """세선화 가지 수 — 끝점 + 분기점 (8이웃 이웃 수로 센다)."""
    if not skel.any():
        return 0
    u = skel.astype(np.uint8)
    nb = cv2.filter2D(u, cv2.CV_8U, np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]],
                                             np.uint8), borderType=cv2.BORDER_CONSTANT)
    return int(((u > 0) & ((nb == 1) | (nb >= 3))).sum())


def _describe(sh) -> ShapeDesc | None:
    """도형 하나 → 서술자. 빈 도형은 None."""
    m, s, lo = _iso_raster(sh, _RASTER)
    if not m.any():
        return None
    pts_all = np.concatenate(sh.loops, axis=0)
    half = (pts_all.max(axis=0) - pts_all.min(axis=0)) / 2.0 * UNITS_PER_SCALE
    ext_x = float(half[0]) if len(half) == 2 else 0.0
    ext_y = float(half[1]) if len(half) == 2 else 0.0
    dt = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
    ecc, theta = _moments(m)
    sym = _symmetry(m, theta)
    conv = _convexity(m)
    sdf = _sdf(m)
    pyr = tuple(cv2.resize(m.astype(np.uint8), (n, n),
                           interpolation=cv2.INTER_AREA) > 0.4 for n in _PYR)
    skel = _thin(m)
    # §6 형태 서술 — 등방 라스터에서 재고 로컬 유닛으로 환산한다
    cnt, apx, ppx = _boundary(m)
    u = UNITS_PER_SCALE / max(s, 1e-9)
    area_u = apx * u * u
    peri_u = ppx * u
    circ = float(4.0 * np.pi * apx / max(ppx * ppx, 1e-9))
    concave = _concavity(cnt)
    curvb = _curv_profile(cnt) / max(u, 1e-9)
    branches = _branch_count(skel)
    npyr = norm_raster(m)
    ys_m, xs_m = np.nonzero(m)
    cm = np.cov(np.stack([xs_m, ys_m]).astype(np.float64)) if len(ys_m) > 2         else np.eye(2)
    ev_m = np.sort(np.maximum(np.linalg.eigvalsh(cm), 1e-9))[::-1]
    mom = (float(ev_m[0] / max(apx, 1.0)), float(ev_m[1] / max(apx, 1.0)))
    center = halfw = curv = None
    stroke_ok = closed = False
    tan0 = tan1 = np.zeros(2, np.float64)
    hw_min = hw_max = length = 0.0
    taper, slim = 1.0, 0.0
    if skel.any():
        # 능선 반폭 — 중심선을 못 뽑는 도형(막대·타원)도 **가장 가는 폭**은
        # 여기서 나온다. 최소 스케일이 낼 수 있는 폭의 자다
        ridge = dt[skel] / s * UNITS_PER_SCALE
        hw_min = float(np.percentile(ridge, 10))
        hw_max = float(ridge.max())
        wmed = 2.0 * float(np.median(dt[skel]))
        paths = _paths(_prune_spurs(skel, max(3.0, 1.2 * wmed)))
        if len(paths) == 1 and len(paths[0][0]) >= 10:
            p = np.asarray(paths[0][0], np.float64)
            # 래스터 px → 로컬 좌표 (등방 배율의 역, y 뒤집힘 복원)
            lx = (p[:, 1] - 1.0) / s + lo[0]
            ly = (_RASTER - 1 - p[:, 0] - 1.0) / s + lo[1]
            pts = np.stack([lx, ly], axis=1)
            pi = np.round(p).astype(int)
            hw = dt[pi[:, 0].clip(0, _RASTER - 1),
                    pi[:, 1].clip(0, _RASTER - 1)] / s
            d = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])
            L = float(d[-1])
            if L > 1e-6:
                t = np.linspace(0.0, L, _N)
                center = _resample(pts, _N) * UNITS_PER_SCALE
                halfw = np.interp(t, d, hw) * UNITS_PER_SCALE
                length = L * UNITS_PER_SCALE
                # 코어 80% — 도형 끝은 어느 획 도형이든 뾰족해서, 끝까지 넣으면
                # 가는 획과 쐐기가 똑같이 무한 테이퍼로 나온다
                core = halfw[max(1, int(0.1 * _N)):max(2, int(0.9 * _N))]
                if len(core) < 2:
                    core = halfw[1:-1] if _N > 2 else halfw
                hw_min = float(core.min())
                hw_max = float(core.max())
                taper = hw_max / max(hw_min, 1e-9)
                slim = 2.0 * float(np.median(core)) / max(length, 1e-9)
                v1 = center[1:-1] - center[:-2]
                v2 = center[2:] - center[1:-1]
                cr = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
                den = (np.hypot(*v1.T) * np.hypot(*v2.T)
                       * np.hypot(*(center[2:] - center[:-2]).T))
                curv = np.where(den > 1e-9, 2.0 * cr / np.maximum(den, 1e-9), 0.0)
                t0 = center[0] - center[1]
                t1 = center[-1] - center[-2]
                tan0 = t0 / max(float(np.hypot(*t0)), 1e-9)
                tan1 = t1 / max(float(np.hypot(*t1)), 1e-9)
                stroke_ok = True
                # 고리 판정 — 양끝이 폭 안에서 만나면 닫힌 중심선이다 (반지·
                # 테두리 도형). 열린 경로에 아핀으로 맞출 수 없으므로 획
                # 어휘에서 뺀다
                closed = bool(np.hypot(*(center[0] - center[-1]))
                              <= 2.0 * max(hw_max, 1e-9))
    z = np.zeros(0, np.float64)
    return ShapeDesc(
        name=sh.name, stroke_ok=stroke_ok, closed=closed,
        center=center if center is not None else z.reshape(0, 2),
        halfw=halfw if halfw is not None else z,
        curv=curv if curv is not None else z,
        tan0=tan0, tan1=tan1, hw_min=hw_min, hw_max=hw_max,
        ext_x=ext_x, ext_y=ext_y, length=length,
        taper=taper, slim=slim, ecc=ecc, sym=sym, convex=conv,
        sdf=sdf, pyr=pyr,
        area=area_u, peri=peri_u, circ=circ, concave=concave,
        branches=branches, mom=mom, curvb=curvb, npyr=npyr)


def reachable(cat: Catalog) -> tuple[str, ...]:
    """창 조작으로 **도달하는** 도형 이름 (cell_map). 표가 없으면 카탈로그 전체."""
    p = default_catalog_path().parent / "cell_map.json"
    if not p.is_file():
        return tuple(sorted(cat.shapes))
    cm = json.loads(p.read_text(encoding="utf-8"))
    names = {v for cells in cm.get("cells", {}).values() for v in cells.values()}
    names = {n for n in names if isinstance(n, str) and n in cat.shapes}
    return tuple(sorted(names)) or tuple(sorted(cat.shapes))


_CACHE: dict[str, ShapeDesc] = {}
_CACHE_KEY = ""
# 같은 카탈로그로 다시 물으면 **키를 다시 안 짓는다.** 키를 지으려면
# `cell_map.json`을 읽고 도형 폴리곤 전부를 해시해야 하는데, 이 함수는
# 좌표하강 안에서(도형 하나를 놓는 동안 수십 번) 불린다 — 그 자리에서는 이
# 한 번이 하강 전체보다 비싸다 (실측: 한 장이 2분에서 10분 넘게로).
# `cat`을 함께 들고 있어야 id가 재활용되지 않는다.
_BY_ID: dict[int, tuple] = {}


def _cache_path(key: str) -> Path:
    from ...paths import work_root

    return work_root() / "cache" / f"shapedesc-{key}.npz"


def _key(cat: Catalog, names: tuple[str, ...]) -> str:
    h = hashlib.sha1()
    h.update(f"{_RASTER}/{_SCHEMA}".encode())
    for n in names:
        h.update(n.encode())
        for l in cat[n].loops:
            h.update(np.ascontiguousarray(l, np.float32).tobytes())
    return h.hexdigest()[:16]


def _load_cache(path: Path, names: tuple[str, ...]) -> dict | None:
    if not path.is_file():
        return None
    try:
        z = np.load(path, allow_pickle=False)
        out = {}
        for n in names:
            if f"{n}/sdf" not in z:
                return None
            sc = z[f"{n}/scal"]
            out[n] = ShapeDesc(
                name=n, stroke_ok=bool(sc[0]), closed=bool(sc[9]),
                center=z[f"{n}/c"],
                halfw=z[f"{n}/w"], curv=z[f"{n}/k"],
                tan0=z[f"{n}/t0"], tan1=z[f"{n}/t1"],
                hw_min=float(sc[1]), hw_max=float(sc[2]), length=float(sc[3]),
                taper=float(sc[4]), slim=float(sc[5]), ecc=float(sc[6]),
                sym=float(sc[7]), convex=float(sc[8]),
                ext_x=float(sc[10]), ext_y=float(sc[11]),
                sdf=z[f"{n}/sdf"],
                pyr=tuple(z[f"{n}/p{i}"] for i in range(len(_PYR))),
                area=float(sc[12]), peri=float(sc[13]), circ=float(sc[14]),
                concave=float(sc[15]), branches=int(sc[16]),
                mom=(float(sc[17]), float(sc[18])),
                curvb=z[f"{n}/kb"], npyr=z[f"{n}/np"])
        return out
    except Exception:                          # noqa: BLE001 — 캐시는 재생성 가능
        return None


def _save_cache(path: Path, descs: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        d = {}
        for n, s in descs.items():
            d[f"{n}/c"] = s.center.astype(np.float32)
            d[f"{n}/w"] = s.halfw.astype(np.float32)
            d[f"{n}/k"] = s.curv.astype(np.float32)
            d[f"{n}/t0"] = s.tan0.astype(np.float32)
            d[f"{n}/t1"] = s.tan1.astype(np.float32)
            d[f"{n}/sdf"] = s.sdf
            d[f"{n}/kb"] = s.curvb.astype(np.float32)
            d[f"{n}/np"] = s.npyr
            for i, p in enumerate(s.pyr):
                d[f"{n}/p{i}"] = p
            d[f"{n}/scal"] = np.array(
                [float(s.stroke_ok), s.hw_min, s.hw_max, s.length, s.taper,
                 s.slim, s.ecc, s.sym, s.convex, float(s.closed),
                 s.ext_x, s.ext_y, s.area, s.peri, s.circ, s.concave,
                 float(s.branches), s.mom[0], s.mom[1]], np.float64)
        np.savez_compressed(path, **d)
    except Exception:                          # noqa: BLE001 — 캐시 실패는 무해
        pass


def descriptors(cat: Catalog, log=None) -> dict[str, ShapeDesc]:
    """도달 도형 전부의 서술자 (프로세스 1회 + 디스크 캐시)."""
    global _CACHE_KEY
    hit = _BY_ID.get(id(cat))
    if hit is not None:
        return hit[1]
    names = reachable(cat)
    key = _key(cat, names)
    if _CACHE and _CACHE_KEY == key:
        return _CACHE
    path = _cache_path(key)
    got = _load_cache(path, names)
    if got is None:
        got = {}
        for n in names:
            d = _describe(cat[n])
            if d is not None:
                got[n] = d
        _save_cache(path, got)
        if log:
            log(f"  도형 서술자 {len(got)}종 계측 (캐시 {path.name})")
    _CACHE.clear()
    _CACHE.update(got)
    _CACHE_KEY = key
    _BY_ID[id(cat)] = (cat, _CACHE)
    return _CACHE


# ── 놓인 뒤의 폭 — **닫힌 식** (라스터를 다시 안 뜬다) ────────────────
def placed_profile(center: np.ndarray, halfw: np.ndarray,
                   sx: float, sy: float) -> tuple[float, float, float, float]:
    """도형을 (sx, sy)로 놓았을 때의 (테이퍼, 폭/길이, 폭 중앙값, 길이).

    선형사상 M = diag(sx, sy)에서 중심선의 단위 접선 t와 반폭 r을 가진 띠
    조각은 폭이 `r·|det M| / |M t|`로 간다 (법선 성분 중 M t에 수직인 몫).
    그래서 라스터를 다시 뜨지 않고도 **놓인 뒤의 폭 프로파일**을 얻는다 —
    후보마다 묻는 자리라 값이 싸야 한다.

    `center`·`halfw`는 로컬 좌표 × `UNITS_PER_SCALE` (`ShapeDesc`와 `stroke.
    _stroke_forms`가 같은 단위로 낸다). 폭·길이도 그 단위로 나오므로 px는
    `/upp`이다.

    **가운데 80%만 본다.** 도형 끝은 어느 획 도형이든 뾰족해서, 끝까지 넣으면
    가는 획(초승달)과 쐐기가 똑같이 무한 테이퍼로 나온다. 레퍼런스 쪽 자도
    같은 이유로 양끝 표본을 뺐다.
    """
    d = np.diff(center, axis=0)
    seg = np.hypot(d[:, 0], d[:, 1])
    t = d / np.maximum(seg, 1e-12)[:, None]           # 마디별 단위 접선
    g = np.hypot(sx * t[:, 0], sy * t[:, 1])
    w = halfw[:-1] * abs(sx * sy) / np.maximum(g, 1e-12)
    if len(w) < 5:
        return 1.0, 0.0, 0.0, 0.0
    ml = seg * g
    cum = np.concatenate([[0.0], np.cumsum(ml)])
    length = float(cum[-1])
    if length <= 1e-9:
        return 1.0, 0.0, 0.0, 0.0
    core = w[(cum[:-1] >= 0.1 * length) & (cum[1:] <= 0.9 * length)]
    if len(core) < 3:
        core = w[1:-1]
    taper = float(core.max()) / max(float(core.min()), 1e-9)
    wmed = 2.0 * float(np.median(core))
    return taper, wmed / length, wmed, length


def placed_widths(center: np.ndarray, halfw: np.ndarray,
                  sx: float, sy: float) -> tuple[np.ndarray, np.ndarray, float]:
    """놓인 뒤의 **폭 프로파일 전체** — (마디별 폭, 마디 중점의 호길이 비율, 길이).

    `placed_profile`은 이 프로파일을 세 수로 접는다 (테이퍼·폭/길이·중앙값).
    프로파일 자체가 필요한 자리 — 원화 띠의 폭 변화를 따라가는가를 묻는
    `stroke._prof_pen` — 는 여기서 받는다. 단위는 `placed_profile`과 같다
    (로컬 × `UNITS_PER_SCALE`; px는 `/upp`).

    **끝을 안 자른다.** `placed_profile`이 가운데 80%만 보는 것은 "획처럼
    생겼나"를 묻는 자리라 도형 공통의 뾰족한 끝을 빼야 했기 때문이다. 여기서
    묻는 것은 "이 획의 폭 변화를 따라가나"라서 끝이 곧 답의 절반이다 —
    원화가 고른 띠인 자리에 뾰족한 물방울을 놓으면 그것이 잎사귀로 읽힌다.
    """
    d = np.diff(center, axis=0)
    seg = np.hypot(d[:, 0], d[:, 1])
    t = d / np.maximum(seg, 1e-12)[:, None]
    g = np.hypot(sx * t[:, 0], sy * t[:, 1])
    w = 2.0 * halfw[:-1] * abs(sx * sy) / np.maximum(g, 1e-12)
    ml = seg * g
    cum = np.concatenate([[0.0], np.cumsum(ml)])
    length = float(cum[-1])
    if length <= 1e-9 or len(w) < 2:
        return w, np.zeros(len(w)), 0.0
    mid = 0.5 * (cum[:-1] + cum[1:]) / length
    return w, mid, length


def layer_width_px(cat: Catalog, lay, upp: float) -> float:
    """이 레이어가 **실제로 칠하는 폭** px — 없으면 0 (중심선 없는 도형).

    계측(`linemetrics`)이 읽는 자리다. 배치 쪽 판정(`stroke._placed_form`)과
    같은 식이라 "재는 폭 = 고르는 폭"이 성립한다.
    """
    d = descriptors(cat).get(lay.shape)
    if d is None or not d.stroke_ok or len(d.center) < 3:
        return 0.0
    return placed_profile(d.center, d.halfw, lay.sx, lay.sy)[2] / max(upp, 1e-9)


# ── 획 어휘 판정 — 손 목록이 아니라 서술자가 고른다 ──────────────────
# 자격은 **기하 하나**다: 불투명하고, 중심선이 열린 한 가닥이며, 그 중심선이
# 폭에 비해 길다. 굽음·테이퍼·굵기로는 여기서 안 거른다 — 같은 도형도 비등방
# 스케일에 따라 획이 되기도 잎사귀가 되기도 하므로, 그 판정은 **놓은 뒤**의
# 폭 프로파일이 한다 (`stroke._placed_form`). 여기서 미리 거르면 어휘가
# 이유 없이 좁아진다 (실측: 등방 테이퍼로 거르면 현행 28종 중 26종이 탈락).
_MIN_ASPECT = float(os.environ.get("FS_DESC_ASPECT", 2.0))   # 길이 / 폭


def stroke_shapes(cat: Catalog, log=None) -> tuple[str, ...]:
    """획 어휘 — 서술자로 고른 이름 목록 (불투명·열린 한 가닥 중심선·길쭉함).

    `FS_STROKE_VOCAB`이 있으면 그쪽이 이긴다 (스윕용 수동 지정).
    """
    sv = os.environ.get("FS_STROKE_VOCAB", "").strip().upper()
    if sv:
        return tuple(s for s in sv.split(",") if s in cat.shapes)
    out = []
    for n, d in descriptors(cat, log).items():
        if not d.stroke_ok or d.closed or not cat[n].opaque:
            continue
        if d.length < _MIN_ASPECT * 2.0 * max(d.hw_max, 1e-9):
            continue
        out.append(n)
    return tuple(sorted(out))


def thin_shapes(cat: Catalog, upp: float, target_px: float,
                log=None) -> tuple[str, ...]:
    """최소 스케일에서 `target_px`보다 **가늘게** 설 수 있는 획 도형 (가는 순).

    막대(A_22)는 정규화 반폭이 1.0이라 최소 폭이 `2×_min_span(upp)`으로 고정
    이다 — 선 지도의 띠가 그보다 가늘면 그 폭으로는 못 긋고 넘치는 잉크가
    그대로 스필이 된다 (미덮 잉크의 90%가 이 자리다). 중심선 위 반폭이 1보다
    작은 도형은 같은 최소 스케일에서 훨씬 가늘게 서므로 그 자리를 맡을 수 있다.
    """
    des = descriptors(cat, log)
    out = [(des[n].min_width_px(upp), n) for n in stroke_shapes(cat, log)]
    return tuple(n for wpx, n in sorted(out) if wpx <= target_px)


# 직선 판정 — 중심선 곡률의 최대가 이보다 작으면 곧은 도형이다 (1/유닛).
_STRAIGHT_K = float(os.environ.get("FS_DESC_STRAIGHT_K", 0.004))


def straight_thin(cat: Catalog, upp: float, target_px: float,
                  log=None) -> tuple[str, ...]:
    """곧고 **가는** 도형 — 막대가 못 내는 폭의 직선 마디를 맡는다 (가는 순).

    막대(A_22)는 상자가 정사각이라 폭 눈금이 최소 도형 폭 그대로다. 선 지도의
    띠가 그보다 가늘면 넘치는 잉크가 전부 스필이 되므로, 같은 자리를 곧은
    가는 도형이 대신 긋는다. 곧음은 중심선 곡률로 잰다 — 이름·계열이 아니다.
    """
    des = descriptors(cat, log)
    out = []
    for n in stroke_shapes(cat, log):
        d = des[n]
        if len(d.curv) and float(np.abs(d.curv).max()) > _STRAIGHT_K:
            continue
        w = d.min_width_px(upp)
        if w <= target_px:
            out.append((w, n))
    return tuple(n for _, n in sorted(out))


# ── §6 채움 후보 순위 — **영역의 모습이 어휘를 고른다** ────────────────
# 종전에는 타원·사각을 늘 먼저 놓고 나머지 여섯을 뒤에 붙였다. 그것은 어휘가
# 여덟일 때의 타협이고, 어휘를 넓히면(도달 520종 중 불투명·뚱뚱한 76종) 전부
# 씨앗 채점을 돌릴 수 없다. 여기서는 **모멘트 정규화 라스터의 IoU**로 먼저
# 줄인다 — 회전·비등방 스케일·미러를 미리 정규화했으므로, 남는 것은 "우리
# 자유도로 옮겨 놓았을 때 이 영역과 얼마나 겹치나"뿐이다. 잔차를 잘 줍는
# 도형이 아니라 **닮은 도형**이 앞에 서는 것이 요점이다 (전자는 불꽃·폭발처럼
# 가장자리가 너덜한 도형을 뽑아 지각 지표를 나쁘게 했다 — `vocabulary` 실측).
_RANK_POSES = 4                       # 축 부호 넷 (미러는 음수 스케일이 낸다)
_RANK_CACHE: dict = {}


def _rank_bank(cat: Catalog, vocab: tuple[str, ...]):
    """어휘의 정규화 라스터 은행 — (이름 index, (4K, n²) float32)."""
    key = (id(cat), vocab)
    got = _RANK_CACHE.get(key)
    if got is not None:
        return got
    des = descriptors(cat)
    names, rows = [], []
    for n in vocab:
        d = des.get(n)
        if d is None or not d.npyr.any():
            continue
        for k in range(_RANK_POSES):
            m = d.npyr
            if k & 1:
                m = m[:, ::-1]
            if k & 2:
                m = m[::-1]
            rows.append(m.reshape(-1).astype(np.float32))
        names.append(n)
    bank = (np.stack(rows) if rows else np.zeros((0, _SDF_N * _SDF_N), np.float32))
    got = (tuple(names), bank, bank.sum(1))
    _RANK_CACHE[key] = got
    return got


def fill_rank(cat: Catalog, mask: np.ndarray, vocab: tuple[str, ...],
              top: int = 8) -> tuple[str, ...]:
    """이 잔여 덩어리를 닮은 순으로 어휘를 정렬해 앞 `top`개를 준다.

    닮음은 정규화 라스터 IoU다. 동점은 **어휘 순서**로 갈린다 — 어휘를 넓혀도
    기존 어휘의 판정이 안 흔들리게 하는 것이 이 노선의 규칙이다.
    """
    names, bank, bsum = _rank_bank(cat, vocab)
    if not names:
        return vocab[:top]
    r = norm_raster(mask).reshape(-1).astype(np.float32)
    rs = float(r.sum())
    if rs <= 0:
        return vocab[:top]
    inter = bank @ r
    iou = inter / np.maximum(bsum + rs - inter, 1e-6)
    best = iou.reshape(len(names), _RANK_POSES).max(1)
    order = sorted(range(len(names)), key=lambda i: (-float(best[i]), i))
    return tuple(names[i] for i in order[:max(1, top)])
