r"""차 한 대의 **얕은 3D 모형** — 옆면·윗면 실루엣 둘로 지은 시각 껍질.

## 왜 이 자가 서는가

면 유닛은 한 차 안에서 같은 자다 (`game.fold`) — 면마다 세계 좌표 **두 축**을
쥔다. 쥐지 않은 셋째 축(**깊이**)은 그 면 마스크에 없다. 그런데 이음선을
되짚으려면 깊이가 있어야 한다: 앞면이 그리기를 멈추는 자리(마스크 u 끝)가 옆면 유닛의
어디인지는 "그 자리에서 차가 얼마나 앞으로 나와 있나"이기 때문이다.

깊이 없이 하면 이음선이 **이웃 면 마스크의 끝**이 된다 — 앞면 → 옆면 넘침을
옆면의 코끝(u 최소)에 붙이는 것이다. 코끝은 차 한가운데(x≈0)의 자리고 앞면
마스크가 끝나는 자리는 x=±154다. 실비아(NIS_SilviaSpecR_02) 설치 마스크 실측:

| 이음새 | 마스크 끝으로 잰 자리 | 껍질이 되짚은 자리 | 어긋남 |
|---|---|---|---|
| front → top | top u −446.5 | −392.0 | 54.5 (차 길이의 6%) |
| front → side_left | side u −363.0 | −400.0 | 37.0 |

앞면 → 윗면이 55유닛 **앞으로** 밀려 있었다 — 그만큼 도안이 겹쳐 **같은 부분이
두 번** 그려진다. 사람이 본 증상이 그것이다.

전체 대조 (표본 24대 189짝): 마스크 끝선으로 잡은
이음선은 두 쪽이 차 위에서 **차 크기의 4.7%**(90분위 30%)만큼 딴 자리를
가리켰다. 그 어긋남이 곧 도안이 겹쳐 두 번 그려지는 양이다.

## 껍질

실루엣 둘의 교집합(일반 원기둥 둘)이면 깊이가 선다:

    H(x, y, z) = 옆면(z, y) ∧ 윗면(z, x)

앞면의 깊이 지도는 `zf(x,y) = max{z : H}`이고 이것이 앞·뒤 면 이음선의 자다.
옆면·윗면의 깊이는 이 껍질에서 다른 축과 무관해진다 (`xl(z)`·`yt(z)`) — 그래서
옆↔윗 이음선은 지금까지 쓰던 마스크 끝과 **같은 값**이고 안 바뀐다. 바뀌는
것은 깊이가 실제로 두 실루엣에 걸린 짝, 곧 앞·뒤 ↔ 옆·윗뿐이다.

**앞·뒤 마스크는 실루엣이 아니다** — 게임이 그 면에서 그리는 자리(범퍼 띠)라
차 앞모습보다 한참 작다 (실비아: 앞면 마스크 308×105, 차 앞모습 371×236).
그래서 껍질을 짓는 데 안 쓴다.

## 정면도

껍질이 있으면 "이 면이 그 자리를 얼마나 정면으로 보나"도 잰다 — 깊이 지도의
기울기다 (`cos = 1/√(1+|∇깊이|²)`). 0에 가까운 자리는 게임이 도안을 모서리에
문질러 바른다: 그려지기는 하지만 사람 눈에는 안 그려진 것과 같다 (실측 지도가
야코비 행렬식으로 떨어뜨리던 것과 같은 자다 — `surface.DET_MIN_FRAC`. 설치
마스크는 화면 warp가 없어 그 자를 못 쓴다). 배치판과 미리보기가 이 값으로
그 자리를 흐리게 깔아, 사람이 "여기 놓으면 안 보인다"를 놓기 전에 본다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from . import fold as gfold
from .surface import SurfaceMap

# 껍질 격자 간격 (면 유닛). 차 길이가 900유닛쯤이라 2면 450칸 — 이음선 중앙값에
# 넉넉하고 한 대 짓는 데 0.2초가 안 든다.
STEP = 2.0
# 한 축의 칸 수 상한 — 유닛이 큰 차에서 간격을 늘려 받는다.
MAX_CELLS = 640
# 깊이 지도를 미분하기 전에 미는 창 (격자 칸).
SMOOTH = 5
# 기울기를 재는 **폭** — 축 길이의 몫. 4%면 차 길이 900유닛에서 ±36유닛이라
# 코·꽁무니가 달아나는 구간(수십 유닛)은 잡고 계단 한 칸은 안 문다.
GRAD_WIN = 0.04
# 이보다 정면도가 낮으면 "모서리에 문질린 자리"로 본다. 도안이 그 자리에서
# 절반 이하로 눌린다는 뜻이다 (cos 0.5 = 60° 기울기).
HEAD_ON_MIN = 0.5
# 이음선이 껍질 밖일 때 안쪽으로 들이는 걸음 (면 크기의 몫 × 횟수).
_SEAM_STEP, _SEAM_STEPS = 0.01, 12
# 깊이가 **두 실루엣에 함께 걸리는** 면들 — 이 면과 옆·윗면 사이에서만 껍질이
# 이음선을 고쳐 준다 (`Hull.seam` 참조).
_FASCIA = ("front", "rear")


def _uv_axes(name: str) -> tuple[tuple[str, float], tuple[str, float]]:
    """면의 u·v가 쥔 (세계 축 글자, 부호)."""
    a, b = gfold.AXES[name]
    return gfold._axis(a), gfold._axis(b)


def _depth_axis(name: str) -> tuple[str, float]:
    """면이 **밖으로 보는** 축 — 그 면의 깊이 축이다."""
    return gfold._axis(gfold.NORMAL[name])


def _axis_grid(lo: float, hi: float) -> np.ndarray:
    step = max(STEP, (hi - lo) / MAX_CELLS)
    n = max(8, int(round((hi - lo) / step)) + 1)
    return np.linspace(lo, hi, n)


def _sample(smap: SurfaceMap, U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """면 마스크를 (면 유닛) 격자에서 뽑는다. 상자 밖은 False."""
    m = smap.mask
    mh, mw = m.shape
    u0, v0, u1, v1 = smap.paint
    xi = np.round((U - u0) / max(1e-6, u1 - u0) * (mw - 1)).astype(int)
    yi = np.round((v1 - V) / max(1e-6, v1 - v0) * (mh - 1)).astype(int)
    ok = (xi >= 0) & (xi < mw) & (yi >= 0) & (yi < mh)
    out = np.zeros(U.shape, bool)
    out[ok] = m[np.clip(yi, 0, mh - 1), np.clip(xi, 0, mw - 1)][ok]
    return out


def _smooth(d: np.ndarray, k: int = SMOOTH) -> np.ndarray:
    """NaN을 건너뛰는 상자 밀기 — 원래 NaN이던 자리는 NaN으로 남는다."""
    f = np.nan_to_num(d, nan=0.0).astype(np.float32)
    ok = np.isfinite(d).astype(np.float32)
    if f.ndim == 1:
        f2, ok2 = f[None, :], ok[None, :]
        kk = (min(k, max(1, f.size // 2 * 2 - 1)), 1)
        fs = cv2.blur(f2, kk)[0]
        os_ = cv2.blur(ok2, kk)[0]
    else:
        fs = cv2.blur(f, (k, k))
        os_ = cv2.blur(ok, (k, k))
    out = np.where(os_ > 0.15, fs / np.maximum(os_, 1e-6), np.nan)
    out[~np.isfinite(d)] = np.nan
    return out


@dataclass
class Hull:
    """옆·윗 실루엣으로 지은 껍질 + 면마다의 깊이 지도. 좌표는 **면 유닛**이다.

    세계 축은 `game.fold`의 것과 같다 — x = 차의 오른쪽 · y = 위 · z = 차의 앞.
    `off`는 면 v(높이)를 세계 y로 옮기는 보정이다 (기준은 `side_left`; 바닥선을
    맞추는 `fold.shared_offset`과 같은 근거다). x·z는 면끼리 원점이 같아 0이다.
    """

    zs: np.ndarray
    xs: np.ndarray
    ys: np.ndarray
    S_side: np.ndarray                 # (z, y)
    S_top: np.ndarray                  # (z, x)
    zf: np.ndarray                     # (x, y) — 앞쪽 껍질면의 z
    zr: np.ndarray                     # (x, y) — 뒤쪽
    xl: np.ndarray                     # (z,)  — 왼쪽 (x 최소)
    xr: np.ndarray                     # (z,)  — 오른쪽
    yt: np.ndarray                     # (z,)  — 위 (y 최대)
    off: dict[str, float] = field(default_factory=dict)   # 면 → 세계 y 보정
    boxes: dict[str, tuple] = field(default_factory=dict)  # 면 → 유닛 상자
    note: str = ""

    # ---------- 좌표 ----------
    def world(self, name: str, u, v) -> dict[str, np.ndarray]:
        """면 유닛 (u, v) → 그 면이 쥔 **세계 좌표 두 개**."""
        (ua, us_), (va, vs_) = _uv_axes(name)
        oy = self.off.get(name, 0.0)
        got = {}
        for ax, sg, val in ((ua, us_, u), (va, vs_, v)):
            got[ax] = sg * np.atleast_1d(np.asarray(val, float)) \
                + (oy if ax == "y" else 0.0)
        return got

    def to_face(self, name: str, w: dict[str, np.ndarray]
                ) -> tuple[np.ndarray, np.ndarray]:
        """세계 좌표 → 면 유닛 (u, v)."""
        (ua, us_), (va, vs_) = _uv_axes(name)
        oy = self.off.get(name, 0.0)
        return (us_ * (w[ua] - (oy if ua == "y" else 0.0)),
                vs_ * (w[va] - (oy if va == "y" else 0.0)))

    def depth(self, name: str, w: dict[str, np.ndarray]) -> np.ndarray:
        """그 면이 그 자리에서 만나는 차 표면의 **깊이 세계 좌표** (없으면 NaN)."""
        da, _ds = _depth_axis(name)
        if da == "z":
            grid = self.zf if name == "front" else self.zr
            return _bilerp(grid, self.xs, self.ys, w["x"], w["y"])
        if da == "x":
            got = _lerp1(self.xl if name == "side_left" else self.xr,
                         self.zs, w["z"])
            live = _sample_grid(self.S_side, self.zs, self.ys, w["z"], w["y"])
            return np.where(live, got, np.nan)
        got = _lerp1(self.yt, self.zs, w["z"])
        live = _sample_grid(self.S_top, self.zs, self.xs, w["z"], w["x"])
        return np.where(live, got, np.nan)

    def point(self, name: str, u, v) -> dict[str, np.ndarray]:
        """면 유닛 → **차 표면의 세계 좌표 셋** (깊이가 안 서면 그 자리는 NaN)."""
        w = self.world(name, u, v)
        da, _ = _depth_axis(name)
        w[da] = self.depth(name, w)
        return w

    # ---------- 이음선 ----------
    def seam(self, src: str, dst: str, edge: float,
             span: tuple[float, float] | None = None,
             n: int = 65) -> float | None:
        """src 유닛의 이음선(`edge`)이 **dst 유닛의 어디인가** — 중앙값. 못 세우면 None.

        `edge`는 src의 넘침 축(=dst 쪽) 좌표이고, `span`은 **공유 축**의 구간
        (src 유닛)이다 — 넘치는 그림이 걸친 자리에서만 재라는 뜻이다.

        하는 일은 한 줄이다: 이음선 위의 점들을 차 표면으로 되짚고
        (`point` — 여기서 깊이가 들어온다) 그 점을 dst 유닛으로 옮긴다.
        """
        if src not in gfold.BODY or dst not in gfold.BODY:
            return None
        if (src in _FASCIA) == (dst in _FASCIA):
            # **옆 ↔ 윗은 이 껍질이 보태 줄 것이 없다.** 실루엣 둘로 지은 껍질에서
            # 옆면의 깊이는 윗면 실루엣의 끝이고 윗면의 깊이는 옆면 실루엣의
            # 끝이라, 되짚은 자리가 곧 마스크 끝선이다. 실측 24대 282짝: 앞뒤↔옆윗은
            # 어긋남 중앙 0.047 → 0.010 (155/189 개선)인데 옆↔윗은 0.051 → 0.050
            # (35/93)이고 윗→옆은 되레 0.020 → 0.028이다. 안 건드린다.
            return None
        s_ax = gfold._find(src, _depth_axis(dst)[0])       # 넘침 축
        d_ax = gfold._find(dst, _depth_axis(src)[0])       # dst의 들어오는 축
        sh = gfold._shared(src, dst)
        if s_ax is None or d_ax is None or sh is None:
            return None
        w_ax = gfold._find(src, sh)
        if w_ax is None or w_ax[0] == s_ax[0]:
            return None
        lo, hi = span if span is not None else self._mask_span(src, w_ax[0])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return None
        ws = np.linspace(min(lo, hi), max(lo, hi), n)
        # 이음선이 껍질 밖으로 나가 있을 수 있다 — 면 상자가 차 실루엣보다 넓게
        # 저장된 차가 있다 (인테그라: 앞면 상자 폭 398 ↔ 차 폭 372). 그럴 때는
        # **안쪽으로 한 발씩 들여** 껍질에 닿는 첫 자리에서 잰다.
        d_s = _depth_axis(dst)[1] * s_ax[2]                # +면 좌표가 커질 때 넘친다
        box = self.boxes.get(src)
        i = 0 if s_ax[0] == "u" else 1
        ext = abs(box[i + 2] - box[i]) if box else abs(edge) or 1.0
        for k in range(_SEAM_STEPS + 1):
            e = float(edge) - d_s * k * _SEAM_STEP * ext
            es = np.full(n, e)
            u, v = (es, ws) if s_ax[0] == "u" else (ws, es)
            p = self.point(src, u, v)
            du, dv = self.to_face(dst, p)
            got = du if d_ax[0] == "u" else dv
            got = got[np.isfinite(got)]
            if len(got) >= max(3, n // 8):
                return float(np.median(got))
        return None

    def _mask_span(self, name: str, ax: str) -> tuple[float, float]:
        """그 면 유닛 상자의 한 축 구간 (span을 안 준 부름의 폴백)."""
        box = self.boxes.get(name)
        if box is None:
            return (np.nan, np.nan)
        i = 0 if ax == "u" else 1
        return (float(box[i]), float(box[i + 2]))

    # ---------- 정면도 ----------
    def head_on(self, name: str, smap: SurfaceMap) -> np.ndarray | None:
        """`smap` 마스크 격자 위의 **정면도** (0~1). 못 재면 None.

        깊이 지도의 기울기다: 1이면 그 자리를 정면으로 보고, 0에 가까우면 차가
        그 자리에서 옆으로 달아나 도안이 모서리에 문질린다.
        """
        if name not in gfold.BODY or smap.mask.size <= 1:
            return None
        mh, mw = smap.mask.shape
        u0, v0, u1, v1 = smap.paint
        us = u0 + (np.arange(mw) + 0.5) / mw * (u1 - u0)
        vs = v1 - (np.arange(mh) + 0.5) / mh * (v1 - v0)
        U, V = np.meshgrid(us, vs)
        w = self.world(name, U, V)
        da, _ = _depth_axis(name)
        if da == "z":
            g = self._slope_z(name)
            if g is None:
                return None
            gx = _bilerp(g[0], self.xs, self.ys, w["x"], w["y"])
            gy = _bilerp(g[1], self.xs, self.ys, w["x"], w["y"])
            s2 = np.nan_to_num(gx) ** 2 + np.nan_to_num(gy) ** 2
            live = np.isfinite(gx) & np.isfinite(gy)
        else:
            g = self._slope_1d(name)
            if g is None:
                return None
            gz = _lerp1(g, self.zs, w["z"])
            s2 = np.nan_to_num(gz) ** 2
            live = np.isfinite(gz)
        out = np.where(live, 1.0 / np.sqrt(1.0 + s2), 0.0)
        return out.astype(np.float32)

    def _slope_z(self, name: str) -> tuple[np.ndarray, np.ndarray] | None:
        """깊이 지도 ∂z/∂x·∂z/∂y (앞·뒤 면)."""
        key = "_g" + name
        got = getattr(self, key, None)
        if got is None:
            d = _smooth(self.zf if name == "front" else self.zr)
            if not np.isfinite(d).any():
                return None
            got = (_wide_grad(d, float(self.xs[1] - self.xs[0]), 0),
                   _wide_grad(d, float(self.ys[1] - self.ys[0]), 1))
            setattr(self, key, got)
        return got

    def _slope_1d(self, name: str) -> np.ndarray | None:
        """깊이 지도의 z 기울기 (옆·윗면). 이 껍질에서 그 깊이는 z만 탄다."""
        key = "_g" + name
        got = getattr(self, key, None)
        if got is None:
            src = {"side_left": self.xl, "side_right": self.xr}.get(name, self.yt)
            d = _smooth(src)
            if not np.isfinite(d).any():
                return None
            got = _wide_grad(d[None, :], float(self.zs[1] - self.zs[0]), 1)[0]
            setattr(self, key, got)
        return got


def _shift(a: np.ndarray, k: int, axis: int, fill=np.nan) -> np.ndarray:
    out = np.full_like(a, fill, dtype=float)
    sl_dst = [slice(None)] * a.ndim
    sl_src = [slice(None)] * a.ndim
    if k > 0:
        sl_dst[axis], sl_src[axis] = slice(k, None), slice(None, -k)
    elif k < 0:
        sl_dst[axis], sl_src[axis] = slice(None, k), slice(-k, None)
    out[tuple(sl_dst)] = a[tuple(sl_src)]
    return out


def _wide_grad(d: np.ndarray, step: float, axis: int) -> np.ndarray:
    """**폭 넓은 중앙차분** — 계단 한 칸에 기울기가 딸려 가지 않게.

    실루엣은 복셀이라 한 칸 계단(문 손잡이·미러·마스크 톱니) 하나가 그대로
    기울기 1이 된다 — 한 칸 차분으로 재면 문짝 한가운데에 "안 보이는 자리"
    세로줄이 선다 (실비아 옆면 실측). 축 길이의 `GRAD_WIN`만큼 떨어진 두 점을
    잇는 기울기는 그런 계단에 안 속고, 우리가 잡으려는 것(코·꽁무니가 몇십
    유닛에 걸쳐 달아나는 것)은 그대로 잡는다.

    한쪽 이웃이 껍질 밖(NaN)이면 0이다 — 껍질 밖으로 떨어지는 절벽은 표면의
    기울기가 아니다.
    """
    n = d.shape[axis]
    w = max(2, int(round(GRAD_WIN * n)))
    g = ((_shift(d, -w, axis) - _shift(d, w, axis)) / (2.0 * w * step))
    return np.where(np.isfinite(g), g, np.where(np.isfinite(d), 0.0, np.nan))


def _idx(axis: np.ndarray, v: np.ndarray) -> np.ndarray:
    a0, a1 = float(axis[0]), float(axis[-1])
    n = len(axis)
    return (np.asarray(v, float) - a0) / max(1e-9, a1 - a0) * (n - 1)


def _lerp1(vals: np.ndarray, axis: np.ndarray, at) -> np.ndarray:
    """1차원 지도를 축 위에서 최근접으로 뽑는다 (밖은 NaN)."""
    t = _idx(axis, at)
    i = np.round(t).astype(int)
    ok = (i >= 0) & (i < len(axis))
    out = np.full(np.shape(t), np.nan)
    out[ok] = vals[np.clip(i, 0, len(axis) - 1)][ok]
    return out


def _bilerp(grid: np.ndarray, ax0: np.ndarray, ax1: np.ndarray, a, b) -> np.ndarray:
    """2차원 지도를 최근접으로 뽑는다 (밖은 NaN)."""
    i = np.round(_idx(ax0, a)).astype(int)
    j = np.round(_idx(ax1, b)).astype(int)
    ok = (i >= 0) & (i < len(ax0)) & (j >= 0) & (j < len(ax1))
    out = np.full(np.shape(i), np.nan, float)
    out[ok] = grid[np.clip(i, 0, len(ax0) - 1), np.clip(j, 0, len(ax1) - 1)][ok]
    return out


def _sample_grid(grid: np.ndarray, ax0: np.ndarray, ax1: np.ndarray, a, b
                 ) -> np.ndarray:
    got = _bilerp(grid.astype(float), ax0, ax1, a, b)
    return np.nan_to_num(got) > 0.5


def build(maps: dict[str, SurfaceMap]) -> Hull | None:
    """면 지도 → 껍질. 옆면·윗면 마스크가 둘 다 없으면 None (부르는 쪽이 물러난다)."""
    side = maps.get("side_left") or maps.get("side_right")
    top = maps.get("top")
    if side is None or top is None:
        return None
    if side.mask.size <= 1 or top.mask.size <= 1 or not side.mask.any() \
            or not top.mask.any():
        return None
    # 세계 y 기준선 — 옆면이다. 다른 면의 보정은 **넘침 변환이 쓰는 바로 그 자**로
    # 잰다 (`fold.shared_offset` — 바닥선 맞춤 + 세로 구간 물림). 둘이 갈리면
    # 이음선과 공유 축 정렬이 서로 다른 높이를 가리켜 조각이 위아래로 어긋난다.
    off: dict[str, float] = {side.name: 0.0}
    for name in gfold.BODY:
        sm = maps.get(name)
        if sm is None or name == side.name or gfold._find(name, "y") is None:
            continue
        got, _why = gfold.shared_offset(sm, side, "y", 1.0)
        if got is not None:
            off[name] = float(got)

    zsp = _world_span(side, "z", off) or (-1.0, 1.0)
    zsp2 = _world_span(top, "z", off) or zsp
    zs = _axis_grid(min(zsp[0], zsp2[0]), max(zsp[1], zsp2[1]))
    xsp = _world_span(top, "x", off) or (-1.0, 1.0)
    xs = _axis_grid(*xsp)
    ysp = _world_span(side, "y", off) or (-1.0, 1.0)
    ys = _axis_grid(*ysp)

    S_side = _silhouette(side, "z", "y", zs, ys, off)
    S_top = _silhouette(top, "z", "x", zs, xs, off)
    if maps.get("side_right") is not None and maps.get("side_left") is not None:
        # 좌우 실루엣은 같은 차를 잰 것이다 — 합집합이 톱니를 메운다
        S_side |= _silhouette(maps["side_right"], "z", "y", zs, ys, off)

    nx, ny = len(xs), len(ys)
    zf = np.full((nx, ny), np.nan)
    zr = np.full((nx, ny), np.nan)
    for k in range(len(zs)):
        hit = S_top[k][:, None] & S_side[k][None, :]
        if not hit.any():
            continue
        zf[hit] = zs[k]
        first = hit & ~np.isfinite(zr)
        zr[first] = zs[k]
    xl = np.where(S_top.any(1), xs[np.argmax(S_top, axis=1)], np.nan)
    xr = np.where(S_top.any(1),
                  xs[len(xs) - 1 - np.argmax(S_top[:, ::-1], axis=1)], np.nan)
    yt = np.where(S_side.any(1),
                  ys[len(ys) - 1 - np.argmax(S_side[:, ::-1], axis=1)], np.nan)
    boxes = {n: maps[n].paint for n in gfold.BODY if n in maps}
    return Hull(zs=zs, xs=xs, ys=ys, S_side=S_side, S_top=S_top, zf=zf, zr=zr,
                xl=xl, xr=xr, yt=yt, off=off, boxes=boxes,
                note=f"격자 z{len(zs)}·x{nx}·y{ny}")


def _world_span(smap: SurfaceMap, letter: str, off: dict[str, float]
                ) -> tuple[float, float] | None:
    """그 면 상자가 세계 축 `letter`에서 덮는 구간."""
    got = gfold._find(smap.name, letter)
    if got is None:
        return None
    i = got[1]
    a, b = smap.paint[i], smap.paint[i + 2]
    o = off.get(smap.name, 0.0) if letter == "y" else 0.0
    a, b = got[2] * a + o, got[2] * b + o
    return (min(a, b), max(a, b))


def _silhouette(smap: SurfaceMap, ax0: str, ax1: str, g0: np.ndarray,
                g1: np.ndarray, off: dict[str, float]) -> np.ndarray:
    """면 마스크를 **세계 축 격자**로 다시 뜬다 (ax0 × ax1)."""
    A, B = np.meshgrid(g0, g1, indexing="ij")
    o = off.get(smap.name, 0.0)
    w = {ax0: A, ax1: B}
    (ua, us_), (va, vs_) = _uv_axes(smap.name)
    U = us_ * (w[ua] - (o if ua == "y" else 0.0))
    V = vs_ * (w[va] - (o if va == "y" else 0.0))
    return _sample(smap, U, V)


# ---------- 캐시 ----------
_CACHE: dict[tuple, "Hull | None"] = {}
_CACHE_MAX = 4


def _sig(maps: dict[str, SurfaceMap]) -> tuple:
    out = []
    for n in gfold.BODY:
        sm = maps.get(n)
        if sm is None:
            continue
        out.append((n, sm.paint, sm.mask.shape, int(sm.mask.sum())))
    return tuple(out)


def of(maps: dict[str, SurfaceMap]) -> Hull | None:
    """이 면 지도의 껍질 (한 번 지으면 캐시). 못 지으면 None."""
    key = _sig(maps)
    if key in _CACHE:
        return _CACHE[key]
    try:
        got = build(maps)
    except Exception:                              # noqa: BLE001 — 껍질 없이도 산다
        got = None
    if len(_CACHE) >= _CACHE_MAX:
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[key] = got
    return got
