r"""차체 면의 **도색 가능 영역**을 면 좌표로 잰다 — 차종마다 다른 것을 재는 자.

면 탭 이름·상한은 `catalog/body_tabs.json`이 쥐지만, 그것만으로는 도안을 앉힐
수 없다. **면이 얼마나 크고 어디가 칠해지는가**는 차종이 정하기 때문이다 (같은
`side_left`라도 세단과 픽업의 폭·높이·휠아치 자리가 다르다). 그래서 이 모듈이
면마다 다음 셋을 잰다:

1. **원점** — 변형 (0,0)이 화면 어디에 찍히나
2. **배율** — 면 유닛 1이 화면 몇 px인가 (x·y 따로)
3. **도색 마스크** — 그 면에서 실제로 칠해지는 영역 (면 유닛 격자)

재는 법은 **차분**이다: 면을 건드리기 전 화면과 프로브 도형을 얹은 화면을 빼면
칠해진 자리만 남는다. 차 색·조명·배경을 안 가정하므로 어떤 차에서도 선다.

## 좌표계

면 유닛 = **변형 박스에 그대로 들어가는 수치**다. 이동 x가 +면 오른쪽, y가 +면
위쪽이고, 스케일 1.0인 도형은 ±1 정규화 좌표에서 128유닛을 차지한다
(`engine.model.UNITS_PER_SCALE` × 2). 화면 px는 그 위에 얹힌 게임의 투영이라
차종·면마다 다르고, 우리는 **면 유닛으로만** 값을 남긴다.

## 아핀 가정

투영은 3D 곡면이라 화면 px ↔ 면 유닛이 엄밀히 선형이 아니다. 하지만 게임은
면마다 **그 면을 정면으로 보는 카메라**를 세우므로 판 중앙에서는 선형에 가깝고,
우리가 쓰는 것은 "이 도안이 이 면 안에 들어가나"라는 상자 대조뿐이다. 그래서
배율을 두 점으로 재고 그 오차는 실측 검증(배치 후 발자국 재측정)으로 잡는다.
곡면 위 렌더 품질은 여전히 우리 축이 아니다 (게임의 투영이다).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import cv2
import numpy as np

# 차분에서 **빼 놓을 UI 자리** (클라 비율 x0,y0,x1,y1). 변형 편집의 값 박스와
# 아래 키 안내는 프로브를 옮기면 같이 바뀌므로 도색으로 오인된다.
UI_RECTS = (
    (0.00, 0.00, 0.28, 0.28),      # 좌상단 도구·값 패널
    (0.00, 0.86, 1.00, 1.00),      # 하단 키 안내 띠
    (0.00, 0.80, 0.28, 1.00),      # 좌하단 레이어 카운터
    (0.93, 0.00, 1.00, 0.04),      # 우상단 FPS 카운터
)
MIN_BLOB = 60           # 이보다 작은 덩어리는 잡티 (px)
KEEP_FRAC = 0.10        # 가장 큰 덩어리의 이 몫보다 작은 덩어리는 버린다
MASK_W = 256            # 도색 마스크 저장 폭 (면 유닛 격자)


# ---------- 차분 ----------
def viewport(shape: tuple[int, ...]) -> np.ndarray:
    """UI를 뺀 화면 영역 마스크."""
    h, w = shape[:2]
    m = np.ones((h, w), bool)
    for x0, y0, x1, y1 in UI_RECTS:
        m[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = False
    return m


def _blobs(m: np.ndarray) -> np.ndarray:
    """잡티와 **떨어져 있는 작은 덩어리**를 버린다 (바닥 반사·유리 반사).

    가장 큰 덩어리가 그 면의 도색 판이다. 범퍼처럼 판이 갈라지는 면이 있어
    통째로 버리지 않고 **가장 큰 것의 10% 이상**만 남긴다.
    """
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    if n <= 1:
        return np.zeros_like(m)
    areas = stats[1:, cv2.CC_STAT_AREA]
    thr = max(MIN_BLOB, KEEP_FRAC * areas.max())
    keep = np.zeros(n, bool)
    keep[1:] = areas >= thr
    return keep[lab]


def footprint_mask(bg: np.ndarray, img: np.ndarray, thr: int = 34) -> np.ndarray:
    """**올린 그룹의 발자국** — 색을 모르는 도안이라 차분만으로 잰다.

    프로브와 달리 도안의 색을 모르므로 색조 자를 못 쓴다. 대신 차분을 쓰기 전에
    **노출을 맞춘다**: 화면 대부분은 안 바뀌었으므로 채널별 비율의 중앙값이 곧
    게임이 다시 잡은 노출이다. 그걸 되돌리고 나면 차분이 도안만 문다.
    """
    a = bg.astype(np.float32) + 1.0
    b = img.astype(np.float32) + 1.0
    vp = viewport(img.shape)
    gain = np.array([float(np.median(b[:, :, c][vp] / a[:, :, c][vp]))
                     for c in range(3)], np.float32)
    d = np.abs(b - a * gain).max(axis=2)
    return _blobs((d > thr) & vp)


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """(x0, y0, x1, y1) — 포함 경계. 빈 마스크면 None."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


# ---------- 면 매핑 (2차) ----------
def _basis(u, v):
    """[1, u, v, u², uv, v²] — 곡면 투영을 담을 최소한의 차수."""
    one = np.ones_like(np.asarray(u, np.float64))
    u = np.asarray(u, np.float64)
    v = np.asarray(v, np.float64)
    return np.stack([one, u, v, u * u, u * v, v * v], axis=-1)


@dataclass
class Warp:
    """면 유닛 → 화면 px 매핑 (2차 다항식). 계수 (2, 6).

    **아핀으로는 안 된다**는 것이 실측의 결론이다 (2026-08-17, side_left):
    프로브를 위로 55유닛 밀었더니 발자국 높이가 162px → 79px로 눌렸다. 옆면이
    어깨를 넘어 루프까지 이어지는 매핑이라, 같은 1유닛이 문짝에서는 1.27px이고
    루프 근처에서는 0.5px도 안 된다. 배율 하나로 적으면 도안이 어디에 앉을지
    2배씩 어긋난다. 그래서 **격자로 재서 2차식을 맞춘다** — 곡률이 담기고,
    역변환도 뉴턴 세 걸음이면 된다.
    """

    a: np.ndarray = field(default_factory=lambda: np.zeros((2, 6)))
    res: float = 0.0            # 맞춘 잔차 RMS (px)

    @classmethod
    def fit(cls, us, vs, xs, ys) -> "Warp":
        B = _basis(np.asarray(us), np.asarray(vs))
        n = B.shape[0]
        deg = 6 if n >= 7 else (3 if n >= 3 else 0)
        if deg == 0:
            raise ValueError("표본이 모자란다 (3점 이상)")
        Bd = B[:, :deg]
        cx, *_ = np.linalg.lstsq(Bd, np.asarray(xs, np.float64), rcond=None)
        cy, *_ = np.linalg.lstsq(Bd, np.asarray(ys, np.float64), rcond=None)
        a = np.zeros((2, 6))
        a[0, :deg], a[1, :deg] = cx, cy
        w = cls(a=a)
        pred = np.stack(w.to_px(np.asarray(us), np.asarray(vs)), axis=-1)
        got = np.stack([np.asarray(xs, float), np.asarray(ys, float)], axis=-1)
        w.res = float(np.sqrt(np.mean((pred - got) ** 2)))
        return w

    @classmethod
    def affine(cls, origin_px, px_per_unit) -> "Warp":
        ox, oy = origin_px
        kx, ky = px_per_unit
        a = np.zeros((2, 6))
        a[0, 0], a[0, 1] = ox, kx
        a[1, 0], a[1, 2] = oy, -ky
        return cls(a=a)

    def to_px(self, u, v):
        B = _basis(u, v)
        return B @ self.a[0], B @ self.a[1]

    def jac(self, u: float, v: float) -> np.ndarray:
        du = np.array([0.0, 1.0, 0.0, 2 * u, v, 0.0])
        dv = np.array([0.0, 0.0, 1.0, 0.0, u, 2 * v])
        return np.array([[self.a[0] @ du, self.a[0] @ dv],
                         [self.a[1] @ du, self.a[1] @ dv]])

    def sdet(self, u, v):
        """**부호 있는** 야코비 행렬식. 부호가 뒤집힌 자리는 매핑이 접힌 자리다 —
        2차식이 표본 밖에서 되돌아 같은 화면 띠를 두 번 덮는다 (실측: 옆면 유닛
        상자가 685유닛으로 부풀고 판이 위아래로 두 번 나왔다)."""
        u = np.asarray(u, float)
        v = np.asarray(v, float)
        a, b = self.a[0], self.a[1]
        dxdu = a[1] + 2 * a[3] * u + a[4] * v
        dxdv = a[2] + a[4] * u + 2 * a[5] * v
        dydu = b[1] + 2 * b[3] * u + b[4] * v
        dydv = b[2] + b[4] * u + 2 * b[5] * v
        return dxdu * dydv - dxdv * dydu

    def det(self, u, v):
        """국소 면적 배율 |J| — 이 자리에서 1유닛²이 화면 몇 px²인가."""
        u = np.asarray(u, float)
        v = np.asarray(v, float)
        a, b = self.a[0], self.a[1]
        dxdu = a[1] + 2 * a[3] * u + a[4] * v
        dxdv = a[2] + a[4] * u + 2 * a[5] * v
        dydu = b[1] + 2 * b[3] * u + b[4] * v
        dydv = b[2] + b[4] * u + 2 * b[5] * v
        return np.abs(dxdu * dydv - dxdv * dydu)

    def to_unit(self, x: float, y: float, seed: tuple[float, float] = (0.0, 0.0),
                iters: int = 8) -> tuple[float, float]:
        """px → 유닛 (뉴턴). 씨앗은 아핀 어림이면 충분하다."""
        u, v = float(seed[0]), float(seed[1])
        for _ in range(iters):
            px, py = self.to_px(np.array([u]), np.array([v]))
            e = np.array([float(px[0]) - x, float(py[0]) - y])
            if np.hypot(*e) < 0.05:
                break
            J = self.jac(u, v)
            try:
                d = np.linalg.solve(J, e)
            except np.linalg.LinAlgError:
                break
            u -= float(d[0])
            v -= float(d[1])
        return u, v


# ---------- 면 지도 ----------
@dataclass
class SurfaceMap:
    """면 하나의 실측 지도. 좌표는 전부 **면 유닛**이다."""

    name: str
    index: int
    origin_px: tuple[float, float]
    px_per_unit: tuple[float, float]      # 원점에서의 국소 배율 (뉴턴 씨앗·표시용)
    paint: tuple[float, float, float, float]   # (u0, v0, u1, v1) 유닛
    fill: float                            # paint 상자 안에서 실제로 칠해진 비율
    mask: np.ndarray = field(repr=False, default_factory=lambda: np.zeros((1, 1), bool))
    cap: int | None = None
    # **믿을 수 있나** — 실측이 어긋난 면은 자동 배치가 쓰지 않고 프리셋으로
    # 물러난다 (틀린 수치로 앉히면 도안이 면 밖으로 나가는데, 그건 사람이
    # 화면에서 보고서야 안다).
    uncertain: bool = False
    note: str = ""
    warp: Warp | None = field(repr=False, default=None)
    # 게임이 이 면에서 **실제로 그리는** 지도 (마스크가 더 좁을 수 있다). None이면
    # 자기 자신이다. 도색 마스크는 "게임이 여기에 칠할 수 있다"까지만 말한다 —
    # 옆면 그린하우스와 윗면 앞·뒷유리는 마스크에 있는데 안 칠해진다
    # (`game.seam`). 배치·판정·표시가 이걸 본다 (`engine.compose.drawable`).
    drawn: "SurfaceMap | None" = field(repr=False, default=None)

    def __post_init__(self):
        if self.warp is None:
            self.warp = Warp.affine(self.origin_px, self.px_per_unit)

    @property
    def width(self) -> float:
        return self.paint[2] - self.paint[0]

    @property
    def height(self) -> float:
        return self.paint[3] - self.paint[1]

    @property
    def aspect(self) -> float:
        return self.width / max(1e-6, self.height)

    def px_to_unit(self, x: float, y: float) -> tuple[float, float]:
        ox, oy = self.origin_px
        kx, ky = self.px_per_unit
        seed = ((x - ox) / kx, (oy - y) / ky)
        assert self.warp is not None
        return self.warp.to_unit(x, y, seed=seed)

    def masked_at(self, u: float, v: float) -> bool:
        """이 면 유닛 자리가 도색 마스크 안인가 — 산포 모티프·띠가 유리 구멍·
        휠아치에 떨어지지 않게 거르는 자다."""
        m = self.mask
        if m.size <= 1:
            return False
        u0, v0, u1, v1 = self.paint
        mh, mw = m.shape
        xi = int((u - u0) / max(1e-6, u1 - u0) * (mw - 1))
        yi = int((v1 - v) / max(1e-6, v1 - v0) * (mh - 1))
        return bool(0 <= xi < mw and 0 <= yi < mh and m[yi, xi])

    def blob_box(self) -> tuple[float, float, float, float] | None:
        """**가장 큰 도색 덩어리**의 유닛 상자. `top` 면에서 후드를 찾는 자다 —
        윗면 판은 유리(앞유리·선루프·뒷유리)가 셋으로 가르는데 그중 제일 큰
        덩어리가 후드다 (루프·리어데크는 유리 사이 좁은 띠로 남는다)."""
        m = self.mask
        if m.size <= 1 or not m.any():
            return None
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
        if n <= 1:
            return None
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        x, y = int(stats[k, cv2.CC_STAT_LEFT]), int(stats[k, cv2.CC_STAT_TOP])
        w, h = int(stats[k, cv2.CC_STAT_WIDTH]), int(stats[k, cv2.CC_STAT_HEIGHT])
        u0, v0, u1, v1 = self.paint
        mh, mw = m.shape
        upp = (u1 - u0) / mw
        vpp = (v1 - v0) / mh
        return (u0 + x * upp, v1 - (y + h) * vpp, u0 + (x + w) * upp, v1 - y * vpp)

    # ---------- 배치 계산 ----------
    def fit(self, aspect: float, coverage: float = 0.90,
            anchor: str = "center", bias_x: float = 0.5,
            margin: float = 0.02) -> tuple[float, float, float, float] | None:
        """이 면에 **비율 `aspect`(w/h)의 가장 큰 상자**를 앉힌다 (면 유닛).

        도색 마스크로 재므로 휠아치·범퍼 틈처럼 안 칠해지는 자리를 피한다 —
        `coverage`는 상자 안에서 칠해지는 몫의 하한이다 (1.0을 요구하면 곡면
        경계의 톱니 때문에 상자가 실용 크기보다 작아진다).

        `anchor`가 상자를 어디에 붙일지 정한다: 인물은 `bottom`(발이 사이드실),
        로고·타이포는 `center`다. `bias_x`는 0=앞, 1=뒤가 아니라 **마스크 상자
        기준 좌우 비율**이다 (면마다 앞뒤 방향이 다르므로 방향은 부르는 쪽이 안다).
        """
        m = self.mask
        if m.size <= 1 or not m.any():
            return None
        mh, mw = m.shape
        u0, v0, u1, v1 = self.paint
        upp = (u1 - u0) / mw                      # 마스크 픽셀당 유닛 (x)
        vpp = (v1 - v0) / mh
        integ = cv2.integral(m.astype(np.uint8))
        best = None
        # 마스크 좌표(px)에서 높이를 크게부터 줄여 가며 첫 합격을 고른다
        for bh in range(mh, 8, -max(1, mh // 60)):
            bw = int(round(bh * aspect * vpp / upp))
            if bw < 4 or bw > mw:
                continue
            ok = _coverage(integ, bh, bw) >= coverage
            if not ok.any():
                continue
            ys, xs = np.where(ok)
            score = _anchor_score(xs, ys, mw, mh, bw, bh, anchor, bias_x)
            k = int(np.argmax(score))
            x, y = int(xs[k]), int(ys[k])
            best = (u0 + x * upp, v1 - (y + bh) * vpp,
                    u0 + (x + bw) * upp, v1 - y * vpp)
            break
        if best is None:
            return None
        if margin:
            cx, cy = (best[0] + best[2]) / 2, (best[1] + best[3]) / 2
            k = 1.0 - margin
            best = (cx + (best[0] - cx) * k, cy + (best[1] - cy) * k,
                    cx + (best[2] - cx) * k, cy + (best[3] - cy) * k)
        return best


def _coverage(integ: np.ndarray, bh: int, bw: int) -> np.ndarray:
    """모든 자리에서 (bh×bw) 상자의 칠해진 비율 — 적분영상 한 방."""
    s = (integ[bh:, bw:] - integ[:-bh, bw:]
         - integ[bh:, :-bw] + integ[:-bh, :-bw])
    return s / float(bh * bw)


def _anchor_score(xs, ys, mw, mh, bw, bh, anchor: str, bias_x: float) -> np.ndarray:
    """합격한 자리들 중 어디를 고를지 — 앵커·좌우 비율에서 점수를 낸다."""
    cx_want = bias_x * (mw - bw)
    sx = -np.abs(xs - cx_want) / max(1.0, mw)
    if anchor == "bottom":
        sy = (ys + bh) / max(1.0, mh)             # 아래로 붙을수록 좋다
    elif anchor == "top":
        sy = -ys / max(1.0, mh)
    else:
        sy = -np.abs((ys + bh / 2) - mh / 2) / max(1.0, mh)
    return 2.0 * sy + sx


# ---------- 만들기 (실측값 → 지도) ----------
# 국소 면적 배율이 **중앙의 이 몫보다 작으면** 그 자리는 안 쓴다. 옆면이 어깨를
# 넘어 루프로 이어지는 자리가 여기서 걸러진다 — 칠해지기는 하지만 도안이 그
# 자리에서 몇 배로 눌려 사람 눈에 "안 그려진 것"과 같다 (실측: 유닛당 1.27px →
# 0.4px). 배치가 이 마스크로 크기를 정하므로 인물이 문짝에 남는다.
DET_MIN_FRAC = 0.35


def build(name: str, index: int, full: np.ndarray,
          origin_px: tuple[float, float], px_per_unit: tuple[float, float],
          cap: int | None = None, mask_w: int = MASK_W,
          warp: Warp | None = None) -> SurfaceMap:
    """차분 마스크(화면 px) + 매핑 → `SurfaceMap` (마스크를 면 유닛으로 편다).

    유닛 격자를 **매핑으로 화면에 보내** 표본을 뜬다 (역변환을 격자마다 풀지
    않는다 — 순변환은 다항식 한 방이다). 곡률로 눌리는 자리는 |J|로 떨어뜨린다.
    """
    bb = bbox(full)
    if bb is None:
        raise ValueError(f"{name}: 도색 마스크가 비었다 (프로브가 안 보였다)")
    x0, y0, x1, y1 = bb
    ox, oy = origin_px
    kx, ky = px_per_unit
    w = warp or Warp.affine(origin_px, px_per_unit)
    # 도색 상자 네 귀퉁이를 유닛으로 되짚어 유닛 상자를 잡는다
    corners = [w.to_unit(x, y, seed=((x - ox) / kx, (oy - y) / ky))
               for x, y in ((x0, y0), (x1, y0), (x0, y1), (x1, y1))]
    us = [c[0] for c in corners]
    vs = [c[1] for c in corners]
    u0, u1 = min(us), max(us)
    v0, v1 = min(vs), max(vs)
    # 마스크 격자는 **유닛 비율**로 짠다 (화면 비율이 아니다) — 유닛 공간에서
    # 등방이라 내접 상자 탐색이 한 축으로 치우치지 않는다.
    mh = max(8, int(round(mask_w * abs(v1 - v0) / max(1e-6, abs(u1 - u0)))))
    gu = np.linspace(u0, u1, mask_w)
    gv = np.linspace(v1, v0, mh)                  # 위에서 아래로 (이미지 순서)
    U, V = np.meshgrid(gu, gv)
    PX, PY = w.to_px(U, V)
    px = np.clip(np.round(PX), 0, full.shape[1] - 1).astype(int)
    py = np.clip(np.round(PY), 0, full.shape[0] - 1).astype(int)
    mask = full[py, px]
    mask, box = defold(mask, (u0, v0, u1, v1), w)
    return SurfaceMap(name=name, index=index, origin_px=(float(ox), float(oy)),
                      px_per_unit=(float(kx), float(ky)),
                      paint=tuple(round(v, 2) for v in box),
                      fill=round(float(mask.mean()), 4), mask=mask, cap=cap, warp=w)


def defold(mask: np.ndarray, box: tuple[float, float, float, float],
           w: Warp) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """접힌 자리와 눌린 자리를 떨어뜨리고 **유닛 상자를 남은 것에 맞춘다**.

    셋을 한자리에서 한다:
    1. 야코비 **부호**가 가운데와 다른 자리 = 매핑이 접힌 자리 (표본 밖 외삽).
    2. |J|가 가운데의 35% 미만 = 도안이 몇 배로 눌리는 자리 (어깨·루프 넘어감).
    3. 남은 마스크의 상자로 유닛 상자를 조인다 — 부풀린 상자를 그대로 두면
       배치가 없는 자리에 도안을 앉힌다.
    """
    if mask.size <= 1 or not mask.any():
        return mask, box
    mh, mw = mask.shape
    u0, v0, u1, v1 = box
    gu = np.linspace(u0, u1, mw)
    gv = np.linspace(v1, v0, mh)
    U, V = np.meshgrid(gu, gv)
    d = w.sdet(U, V)
    ref = float(np.median(d[mask]))
    if ref == 0.0:
        return mask, box
    keep = mask & (np.sign(d) == np.sign(ref)) & (np.abs(d) >= DET_MIN_FRAC * abs(ref))
    if not keep.any():
        return mask, box
    ys, xs = np.where(keep)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    sub = keep[y0:y1 + 1, x0:x1 + 1]
    return sub, (float(gu[x0]), float(gv[y1]), float(gu[x1]), float(gv[y0]))


# ---------- 저장·불러오기 ----------
def car_slug(car: str) -> str:
    keep = [c if (c.isalnum() or c in " -_") else "" for c in car]
    return ("".join(keep).strip().replace(" ", "-") or "car")[:60]


def map_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "catalog" / "surfaces"


def save(car: str, maps: dict[str, SurfaceMap], client: tuple[int, int],
         note: str = "", car_color: tuple[int, int, int] | None = None) -> Path:
    """`catalog/surfaces/<차>-<면>.png` + `<차>.json`. 마스크는 그림으로 남긴다."""
    d = map_dir()
    d.mkdir(parents=True, exist_ok=True)
    slug = car_slug(car)
    out = {"car": car, "slug": slug, "measured": date.today().isoformat(),
           "client": list(client), "note": note, "surfaces": {}}
    if car_color is not None:
        out["car_color"] = list(car_color)
    else:                                  # 이어서 잴 때 앞서 잰 색을 잃지 않는다
        old = map_dir() / f"{slug}.json"
        if old.exists():
            try:
                prev = json.loads(old.read_text(encoding="utf-8")).get("car_color")
            except (OSError, ValueError):
                prev = None
            if prev:
                out["car_color"] = prev
    for name, s in maps.items():
        png = f"{slug}-{name}.png"
        cv2.imencode(".png", (s.mask.astype(np.uint8) * 255))[1].tofile(str(d / png))
        out["surfaces"][name] = {
            "index": s.index, "cap": s.cap,
            "origin_px": [round(v, 1) for v in s.origin_px],
            "px_per_unit": [round(v, 4) for v in s.px_per_unit],
            "paint": list(s.paint), "fill": s.fill,
            "uncertain": s.uncertain, "note": s.note,
            "warp": [[round(float(v), 6) for v in row] for row in s.warp.a],
            "warp_res": round(float(s.warp.res), 2),
            "mask": png, "mask_size": [s.mask.shape[1], s.mask.shape[0]]}
    p = d / f"{slug}.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return p


def load(car: str) -> dict[str, SurfaceMap]:
    """그 차의 면 지도. 없으면 빈 dict (부르는 쪽이 프리셋으로 물러난다)."""
    p = map_dir() / f"{car_slug(car)}.json"
    if not p.exists():
        return {}
    raw = json.loads(p.read_text(encoding="utf-8"))
    got: dict[str, SurfaceMap] = {}
    for name, s in raw.get("surfaces", {}).items():
        mp = map_dir() / s["mask"]
        mask = np.zeros((1, 1), bool)
        if mp.exists():
            img = cv2.imdecode(np.fromfile(str(mp), np.uint8), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                mask = img > 127
        wp = None
        if s.get("warp"):
            wp = Warp(a=np.asarray(s["warp"], float),
                      res=float(s.get("warp_res") or 0.0))
        # 옛 지도(접힘을 안 떨어뜨린 것)도 여기서 고쳐 읽는다 — 저장본을 다시
        # 잴 필요가 없다 (매핑 계수와 마스크만 있으면 판정이 된다).
        if wp is not None:
            mask, pbox = defold(mask, tuple(s["paint"]), wp)
        else:
            pbox = tuple(s["paint"])
        got[name] = SurfaceMap(
            name=name, index=int(s["index"]), origin_px=tuple(s["origin_px"]),
            px_per_unit=tuple(s["px_per_unit"]), paint=pbox,
            fill=round(float(mask.mean()), 4), mask=mask, cap=s.get("cap"),
            uncertain=bool(s.get("uncertain", False)), note=s.get("note", ""),
            warp=wp)
    return got


def car_color(car: str) -> tuple[int, int, int] | None:
    """면 지도에 적힌 차 도색 색 (없으면 None)."""
    p = map_dir() / f"{car_slug(car)}.json"
    if not p.exists():
        return None
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    c = raw.get("car_color")
    return (int(c[0]), int(c[1]), int(c[2])) if c else None


def probe_side_width(car: str | None) -> float | None:
    """실측 지도가 잰 **옆면 가로** (유닛). 안 잰 차면 None. 마스크는 안 읽는다.

    `--media` 후보를 크기로 거를 때 쓰는 **유일한 실측 기준**이다
    (`carfiles.match_media`). 잰 면 중 옆면 가로만 쓰는 것은 설치 상자와의 비가
    0.82~0.96으로 좁은 유일한 자이기 때문이다 (11대 대조) — 앞면 세로는
    0.16~24배까지 튀는 병리가 있어 거르는 데 쓰면 맞는 차를 떨어뜨린다.
    """
    if not car:
        return None
    p = map_dir() / f"{car_slug(car)}.json"
    if not p.exists():
        return None
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    for name in ("side_left", "side_right"):
        s = (raw.get("surfaces") or {}).get(name) or {}
        box = s.get("paint")
        if box and not s.get("uncertain") and float(box[2]) - float(box[0]) > 0:
            return float(box[2]) - float(box[0])
    return None
