r"""면 모서리 — 한 면을 **넘친** 그림이 이웃 면 어디로 이어지나.

## 왜 이 자가 서는가

면 하나에 앉힌 도안이 그 면보다 크면 게임은 넘은 만큼을 **그냥 안 그린다**
(면마다 제 도색 마스크로 자른다). 사람이 보기에는 그림이 모서리에서 잘린다.
잘리지 않게 하려면 넘은 조각을 **이웃 면에 제 그룹으로 다시 올려야** 한다 —
도어 유리로 머리를 잇는 길(`seam`)과 같은 일을 차체 면 전부로 넓힌 것이다.

## 면 유닛은 차 한 대 안에서 한 자다

설치 마스크의 `Masks.xml`이 면마다 **에디터 축이 3D 어느 축인가**를 적어 둔다
(`xAxis`·`yAxis`). 설치본 **636대 전부** 예외 없이 같은 표이고 `xScale`·`yScale`도
전부 1이다. 회전 라벨까지 먹인 **에디터** 축:

| 면 | u (화면 오른쪽) | v (화면 위쪽) |
|---|---|---|
| `front` | −x (차의 왼쪽) | +y (위) |
| `side_left` | −z (차 뒤) | +y |
| `top` | −z (차 뒤) | +x (차의 오른쪽) |
| `side_right` | +z (차 뒤) | +y |
| `rear` | +x (차의 오른쪽) | +y |

축의 뜻은 실측 하나에 걸려 있다: **옆면 왼쪽 탭에서 +u가 차 뒤**다 (2026-08-17
캡처). 옆면 u가 −z이므로 차 앞이 +z이고, 윗면 탭은 위에서 내려다본 그림이라
차 뒤가 화면 오른쪽일 때 화면 위쪽은 차의 **오른쪽**이다 (지도와 같은 배치).
윗면 v가 +x이므로 +x = 차의 오른쪽 — 앞면 u(−x)가 차의 왼쪽인 것과 맞물린다
(앞에서 마주 보면 차의 왼쪽이 내 오른쪽이다).

차체 면의 유닛은 그래서 **한 차 안에서 같은 자**다. 확인 (설치 마스크 실측):
줄리아 GTAm은 길이 896유닛·높이 271·지붕 폭 387이 193유닛/m 하나로 맞고,
상자꼴 차는 면끼리 직접 견줄 수 있어 더 분명하다 — VW Type2는 윗면 폭 227 ↔
앞면 폭 231(1.7%), 옆면 높이 229 ↔ 앞면 높이 226(1.3%).

유리 면은 다르다 — 제 아틀라스 슬롯을 채우려고 따로 늘려 저장한다 (VW Type2의
도어 유리는 929유닛으로 **차 길이(609)보다 길다**). 그래서 유리 이음새는 배율을
마스크 **형상**에서 따로 잰다: 도어 유리는 옆면 그린하우스 실측(`seam.Seam`)이고,
윈드실드·뒷유리·선루프는 **유리 띠**다 (`glass_folds` — 윗면 마스크의 유리
구멍이 있으면 그것, 없으면(설치본 다수) 옆면 필러 프로필 `pillar_bands`.
유리 쪽 z 방향은 유리 마스크의 **사다리꼴**이 정한다 — 지붕 쪽이 좁다).
## 접기 변환은 등거리다

배율이 1이므로 모서리를 넘는 변환에 남는 것은 **부호 순열(회전·반사)과 이동**
뿐이다. 미지수는 둘:

- **이음선** — 넘치는 쪽 마스크 끝(옆면의 지붕선, 윗면의 옆 모서리)이다. **넘치는
  그림이 걸친 구간에서만** 잰다: 지붕선은 캐빈과 후드에서 수십 유닛 다르므로 면
  전체 중앙값으로 재면 그 자리가 아니다.
- **공유 축 어긋남** — 두 면이 같은 3D 축을 쥔 쪽이다. 셋으로 갈린다:
  - **차 폭(x)**: 0이다. 두 마스크가 다 차 중심선에 대칭이라 상자 중심이 x=0이다.
  - **차 높이(y)**: 모서리에서 **바닥선을 맞춘다**. 앞범퍼 아래와 사이드실 아래는
    같은 지상고에서 끝난다.
  - **차 길이(z)**: 0이다(원점 맞춤). 두 상자가 다 "그 면이 가진 차 길이 전부"라
    중심이 가깝고, 마스크에 이보다 나은 표지가 없다 — 남는 어긋남은 차 길이의
    몇 %다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .surface import SurfaceMap

# 에디터 축 (u, v) — `Masks.xml`의 xAxis·yAxis에 회전 라벨을 먹인 것.
AXES: dict[str, tuple[str, str]] = {
    "front": ("-x", "+y"),
    "side_left": ("-z", "+y"),
    "top": ("-z", "+x"),
    "side_right": ("+z", "+y"),
    "rear": ("+x", "+y"),
    # 유리 — **차 폭(x) 축만 믿는다.** 유리 면은 저장 회전이 차마다 달라 z축
    # 부호가 안 선다 (실측: 같은 코드로 인테그라·줄리아 윈드실드 조각이 카울
    # 반대편(지붕)에 붙었다). z 방향은 유리 마스크의 사다리꼴이 정한다
    # (`glass_folds` — 지붕 쪽이 좁고 카울/데크 쪽이 넓다).
    "window_left": ("-z", "+y"),
    "window_right": ("+z", "+y"),
    "windshield": ("-x", "+z"),
    "rear_window": ("+x", "-z"),
    "sunroof": ("-z", "+x"),
    "spoiler": ("+x", "-z"),
}

# 면이 **밖으로 보는** 방향 (차의 앞이 +z · 오른쪽이 +x · 위가 +y).
NORMAL: dict[str, str] = {
    "front": "+z", "rear": "-z", "side_left": "-x", "side_right": "+x",
    "top": "+y",
}

BODY: tuple[str, ...] = ("front", "side_left", "top", "side_right", "rear")

# 3D 축마다 **차 전체를 재는** 면 (다른 면은 그 일부만 가진다).
FULL_SPAN: dict[str, tuple[str, ...]] = {
    "x": ("top",),
    "y": ("side_left", "side_right"),
    "z": ("side_left", "side_right", "top"),
}
# 그 면보다 이만큼 넘게 크면 **배율이 다른 것**이다 — 설치본 619대 실측: 앞·뒤
# 면 높이는 옆면의 1.10배까지, 폭은 윗면의 1.21배까지가 정상이고, 스물 남짓한
# 차만 그 밖이다 (작은 면을 슬롯 채우려고 늘려 저장한 것 — 골프 R의 뒤 면이
# 옆면 높이의 2.12배). 그런 짝은 등거리 가정이 안 서므로 안 잇는다.
SPAN_LIMIT: dict[str, float] = {"x": 1.25, "y": 1.15, "z": 1.40}

# 끝선 분위수 — 중앙값이다. 안테나·윙·미러 같은 돌기 한 줄에 이음선이 딸려
# 가지 않게 한다.
EDGE_Q = 0.5

# 껍질이 되짚은 이음선을 **받아 주는 울타리** — 마스크 끝선에서 목적 면 크기의
# 이 몫보다 멀면 껍질이 표면을 놓친 것으로 보고 마스크 끝선을 쓴다.
#
# 근거는 뜻이다: 이 보정은 **모서리가 둥근 만큼**이라 차 크기의 몇 %~십몇 %다
# (실비아 앞→윗 6%). 면 크기의 3분의 1을 넘는 값은 둥근 모서리가 아니라 껍질이
# 엉뚱한 표면을 문 것이다 — 줄리아 GTAm 뒤 면은 마스크가 ±168.5인데 윗면 실루엣의
# 그 자리 반폭이 132라, 껍질이 차 **앞쪽** 표면을 물어 이음선을 차 반대편에 뒀다.
# 표본 24대 278짝 분포: 울타리 안 231짝은 중앙 5% · 90분위 16%이고, 물린 47짝은
# 중앙 58% · 최대 105%다.
SEAM_TOL = 0.30


def _axis(spec: str) -> tuple[str, float]:
    """`'-z'` → `('z', -1.0)`."""
    return spec[1], (-1.0 if spec[0] == "-" else 1.0)


def _grids(smap: SurfaceMap) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(마스크, 열 중심 u, 행 중심 v). 행 0 = 위(v1)."""
    m = smap.mask
    mh, mw = m.shape
    u0, v0, u1, v1 = smap.paint
    us = u0 + (np.arange(mw) + 0.5) / mw * (u1 - u0)
    vs = v1 - (np.arange(mh) + 0.5) / mh * (v1 - v0)
    return m, us, vs


def edge_line(smap: SurfaceMap, axis: str, sign: float,
              lo: float | None = None, hi: float | None = None) -> float | None:
    """마스크의 **끝선** — `axis`('u'|'v')를 `sign` 방향으로 갔을 때의 마지막 자리.

    다른 축의 구간 `[lo, hi]`(유닛)에 든 줄만 본다. 넘치는 그림이 걸친 구간에서
    재라는 것이 이 인자다. 마스크가 그 구간에 없으면 None.
    """
    m, us, vs = _grids(smap)
    if m.size <= 1 or not m.any():
        return None
    axes = (us, vs) if axis == "v" else (vs, us)   # (훑는 축, 재는 축)
    keep, val = axes
    sel = np.ones(len(keep), bool)
    if lo is not None:
        sel &= keep >= lo
    if hi is not None:
        sel &= keep <= hi
    sub = m[:, sel] if axis == "v" else m[sel].T   # 열 = 훑는 줄, 행 = 재는 축
    if not sub.size or not sub.any():
        return None
    # v는 행 0이 위(v1), u는 열 0이 왼쪽(u0) — 재는 축의 배열 순서에 맞춰 끝을 고른다
    first = (sign > 0) if axis == "v" else (sign < 0)
    idx = (np.argmax(sub, 0) if first
           else sub.shape[0] - 1 - np.argmax(sub[::-1], 0))
    got = val[idx[sub.any(0)]]
    return float(np.quantile(got, EDGE_Q)) if len(got) else None


@dataclass
class Fold:
    """면 A → 면 B **넘침 변환** (등거리) + 넘침 판정에 필요한 이음선.

    `A`·`b`는 면 유닛 아핀이다: `(u', v') = A·(u, v) + b`. 차체 면끼리는 A가 부호
    순열이라 그룹 배치(이동·균등 스케일·회전·미러)로 **그대로** 낼 수 있다 —
    캔버스에 구울 것이 없다.
    """

    src: str
    dst: str
    axis: str                      # src에서 넘치는 축 ('u'|'v')
    sign: float                    # +1이면 좌표가 커질 때 넘친다
    edge: float                    # 이음선 (src 유닛)
    A: np.ndarray = field(default_factory=lambda: np.eye(2))
    b: np.ndarray = field(default_factory=lambda: np.zeros(2))
    # 이음새 띠의 **건너편 끝** (src 유닛) — 유리 구멍처럼 목적 면이 src의 한
    # 구간에만 사는 짝이 쓴다. None이면 이음선 너머 전부다. 넘침 조각을 자르는
    # 자가 이걸 본다: 구멍 너머(지붕 뒤쪽)의 레이어는 유리가 아니라 제 면이
    # 그린다 — 조각에 넣으면 안 그려질 장수만 는다.
    far: float | None = None
    why: str = ""

    def to(self, u, v) -> tuple[np.ndarray, np.ndarray]:
        """src 유닛 → dst 유닛 (배열도 받는다)."""
        u = np.asarray(u, float)
        v = np.asarray(v, float)
        return (self.A[0, 0] * u + self.A[0, 1] * v + self.b[0],
                self.A[1, 0] * u + self.A[1, 1] * v + self.b[1])

    def over(self, box: tuple[float, float, float, float]) -> float:
        """이 상자가 이음선을 **넘은 양** (면 유닛). 안 넘으면 0."""
        if self.axis == "u":
            far = box[2] if self.sign > 0 else box[0]
        else:
            far = box[3] if self.sign > 0 else box[1]
        return max(0.0, self.sign * (far - self.edge))

    def box(self, box: tuple[float, float, float, float]
            ) -> tuple[float, float, float, float]:
        """상자를 dst 유닛으로 옮긴 축정렬 상자."""
        xs, ys = self.to(np.array([box[0], box[2], box[0], box[2]]),
                         np.array([box[1], box[1], box[3], box[3]]))
        return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))


def _shared(src: str, dst: str) -> str | None:
    """두 면이 함께 쥔 3D 축 (문자). 이웃이 아니면 None."""
    ns, nd = NORMAL.get(src), NORMAL.get(dst)
    if ns is None or nd is None:
        return None
    ls, ld = ns[1], nd[1]
    if ls == ld:                                  # 마주 보는 면 (앞↔뒤) — 안 닿는다
        return None
    return next(c for c in "xyz" if c not in (ls, ld))


def fold(maps: dict[str, SurfaceMap], src: str, dst: str,
         span: tuple[float, float] | None = None, hull=None) -> Fold | None:
    """차체 면 A → B 넘침 변환. 이웃이 아니거나 마스크가 비면 None.

    `span`은 **넘치는 그림이 공유 축에서 걸친 구간** (src 유닛)이다. 주면 이음선을
    그 구간에서만 재므로 캐빈 위로 넘긴 그림은 지붕선에, 후드 위로 넘긴 그림은
    펜더선에 붙는다.

    `hull`(`game.hull.Hull`)을 주면 이음선의 **건너편 자리**를 그것이 정한다 —
    src의 이음선을 차 표면으로 되짚고 dst 유닛으로 옮긴다. 없으면 dst 마스크의
    끝선으로 물러나는데, 그 끝선은 두 면이 겹치는 짝(앞↔옆·앞↔윗)에서 수십
    유닛 어긋난다 (`game.hull` 표). 부르는 쪽이 껍질을 들고 다닌다 —
    이 모듈은 껍질을 안 읽는다 (import 고리를 안 만든다).
    """
    sm, dm = maps.get(src), maps.get(dst)
    if sm is None or dm is None or sm.mask.size <= 1 or dm.mask.size <= 1:
        return None
    shared = _shared(src, dst)
    if shared is None:
        return None
    # 넘치는 방향 = 이웃 면이 밖으로 보는 방향. 들어가는 방향 = 내 면의 반대편.
    out_l, out_s = _axis(NORMAL[dst])
    in_l, in_s = _axis(NORMAL[src])
    in_s = -in_s
    si = _find(src, out_l)
    di = _find(dst, in_l)
    if si is None or di is None:
        return None
    s_ax, s_i, s_sgn = si
    d_ax, d_i, d_sgn = di
    d_s = out_s * s_sgn                           # src 좌표가 커질 때 넘치나
    d_d = in_s * d_sgn                            # dst 좌표가 커질 때 들어가나
    ss = _find(src, shared)
    ds = _find(dst, shared)
    if ss is None or ds is None:
        return None
    _sh_ax, sh_i, sh_sgn = ss
    _dh_ax, dh_i, dh_sgn = ds
    if not _same_metric(sm, dm, shared, sh_i, dh_i):
        return None
    sgn = sh_sgn * dh_sgn                         # 공유 축 부호 (src → dst)
    off, why = shared_offset(sm, dm, shared, sgn)
    if off is None:
        return None
    # 이음선 — src는 걸친 구간에서, dst는 그 구간을 옮긴 자리에서 잰다
    lo = hi = None
    if span is not None:
        lo, hi = min(span), max(span)
    e_s = edge_line(sm, s_ax, d_s, lo=lo, hi=hi)
    if e_s is None:
        e_s = edge_line(sm, s_ax, d_s)
    if e_s is None:
        return None
    dlo = dhi = None
    if lo is not None:
        a, b_ = sgn * lo + off, sgn * hi + off
        dlo, dhi = min(a, b_), max(a, b_)
    e_d = edge_line(dm, d_ax, -d_d, lo=dlo, hi=dhi)
    if e_d is None:
        e_d = edge_line(dm, d_ax, -d_d)
    if e_d is None:
        return None
    how = "마스크 끝"
    e_h = hull.seam(src, dst, e_s, span=(lo, hi) if lo is not None else None) \
        if hull is not None else None
    if e_h is not None:
        ext = abs(dm.paint[d_i + 2] - dm.paint[d_i])
        if abs(e_h - e_d) <= SEAM_TOL * max(1e-6, ext):
            e_d, how = float(e_h), "깊이"
    A = np.zeros((2, 2))
    b = np.zeros(2)
    A[d_i, s_i] = d_s * d_d
    b[d_i] = e_d - d_s * d_d * e_s
    A[dh_i, sh_i] = sgn
    b[dh_i] = off
    return Fold(src=src, dst=dst, axis=s_ax, sign=d_s, edge=float(e_s), A=A, b=b,
                why=f"이음선 {src} {s_ax}={e_s:.0f} → {dst} {d_ax}={e_d:.0f}"
                    f"({how}) · 공유 {shared}축 {why}")


def _same_metric(sm: SurfaceMap, dm: SurfaceMap, shared: str,
                 si: int, di: int) -> bool:
    """두 면이 **같은 배율**로 저장돼 있나 — 공유 축 길이로 본다 (`SPAN_LIMIT`)."""
    a = sm.paint[si + 2] - sm.paint[si]
    b = dm.paint[di + 2] - dm.paint[di]
    if a <= 0 or b <= 0:
        return False
    lim = SPAN_LIMIT[shared]
    full = FULL_SPAN[shared]
    if sm.name in full and dm.name in full:
        return 1.0 / lim <= b / a <= lim
    if sm.name in full:
        return b / a <= lim
    if dm.name in full:
        return a / b <= lim
    return True


def _find(name: str, letter: str) -> tuple[str, int, float] | None:
    """그 면에서 3D 축 `letter`를 쥔 에디터 축 → ('u'|'v', 0|1, 부호)."""
    for i, ax in enumerate(AXES.get(name, ())):
        lt, sg = _axis(ax)
        if lt == letter:
            return ("u" if i == 0 else "v", i, sg)
    return None


def shared_offset(sm: SurfaceMap, dm: SurfaceMap, shared: str, sgn: float
                  ) -> tuple[float | None, str]:
    """공유 축 어긋남 — 축마다 근거가 다르다 (모듈 설명 참조).

    `sgn`은 src → dst의 공유 축 부호다. 되돌리는 값 `off`로 `dst = sgn·src + off`.
    """
    if shared in ("x", "z"):
        return 0.0, ("중심선 대칭" if shared == "x" else "원점 맞춤")
    # 높이(y) — **면마다 제일 낮은 도색**을 맞춘다. 사이드실 아래와 범퍼 아래는
    # 같은 지상고에서 끝난다 (차 밑은 어느 방향에서 봐도 바닥이다).
    #
    # 모서리 띠에서 재면 안 된다: 앞범퍼가 모서리를 감아 도는 차는 옆면의 앞쪽
    # 아랫부분이 통째로 `front`의 것이라 옆면 띠의 바닥이 사이드실보다 한참
    # 위다 (줄리아 실측: 띠 바닥 −27 대 사이드실 −135). 그 값을 맞추면 앞면이
    # 지붕까지 올라간다.
    s_v = _find(sm.name, "y")
    d_v = _find(dm.name, "y")
    if s_v is None or d_v is None:
        return None, ""
    b_s = edge_line(sm, s_v[0], -s_v[2])
    b_d = edge_line(dm, d_v[0], -d_v[2])
    if b_s is None or b_d is None:
        return None, ""
    off = float(b_d - sgn * b_s)
    # **짧은 쪽이 긴 쪽 밖으로 못 나간다** — 두 면의 세로 구간은 같은 차의 같은
    # 높이를 잰 것이다. 뒷문이 땅에 안 닿는 차(픽업 테일게이트·해치 아래 범퍼가
    # 비도색)는 바닥 맞춤이 면을 통째로 띄우므로(골프 R 실측: 뒤 면 윗선이
    # 루프라인의 1.98배) 여기서 물린다.
    s0, s1 = sm.paint[s_v[1]], sm.paint[s_v[1] + 2]
    g0, g1 = dm.paint[d_v[1]], dm.paint[d_v[1] + 2]
    if sgn < 0:                                   # 공유 축 부호가 뒤집힌 짝
        g0, g1 = -g1, -g0
    # 이웃 상자를 내 좌표로 되돌리면 [g0 − sgn·off, g1 − sgn·off]다. **짧은 쪽이
    # 긴 쪽 안**에 들어야 한다 — 방향을 뒤집어도 같은 조건이라 왕복이 안 깨진다.
    tol = 0.05 * min(s1 - s0, g1 - g0)
    t = sgn * off
    lo, hi = ((g1 - s1 - tol, g0 - s0 + tol) if (g1 - g0) <= (s1 - s0)
              else (g0 - s0 - tol, g1 - s1 + tol))
    note = ""
    if not (lo <= t <= hi):
        off, note = sgn * min(max(t, lo), hi), " (세로 구간에 물림)"
    return off, f"바닥선 {b_s:.0f}→{b_d:.0f}{note}"


def neighbors(name: str) -> tuple[str, ...]:
    """이 차체 면이 모서리를 맞댄 면들."""
    return tuple(n for n in BODY if n != name and _shared(name, n) is not None)


# ---------- 유리 이음새 — 윗면 마스크의 유리 구멍에서 (2026-08-21) ----------
# 유리 면은 제 배율로 저장돼 등거리 가정이 안 선다. 배율의 근거는 차량 형상이다:
# 윗면 마스크는 유리 자리에서 도색이 끊긴다 (중앙 밴드의 구멍). 구멍의 차 길이
# 구간 ↔ 유리 잉크 상자를 축 표(`AXES`)대로 맞추면 축마다 배율·이음선이 선다.
# 구멍이 없는 마스크(유리 위까지 통짜로 칠한 차종 — 인테그라 실측)는 유리 자리를
# 읽을 수 없으므로 안 잇는다.
#
# 배율 비등방 상한 — 이보다 늘어난 이음새는 캔버스에 구워도 그림이 뭉개진다
# (도어 유리 실측: ×1.6에서 실루엣 IoU 0.978 · ×2.0에서 0.961. 설치 636대의
# 도어 유리 세로 늘림은 1.0~3.2였다 — 같은 울타리를 쓴다).
GLASS_ANISO = 3.6
# 중앙 밴드에서 도색으로 치는 행 비율 (compose.top_segments와 같은 문턱)
_GAP_SOLID = 0.55
# 유리 구멍으로 인정하는 최소 폭 (면 u폭의 몫) — 몰딩 한 줄에 안 속는다
_GAP_MIN = 0.02


def top_gaps(smap: SurfaceMap) -> list[tuple[float, float]]:
    """윗면 마스크 중앙 밴드의 **도색 끊김 구간들** (u 오름차순 = 앞→뒤).

    끊김이 곧 유리다: 첫 구간 = 윈드실드, 마지막 = 뒷유리 (둘 이상일 때),
    가운데 = 선루프 (셋 이상일 때). 실루엣 허리로 가른 차(구멍 없는 마스크)는
    끊김이 없으므로 빈 목록이다 — 유리 자리를 못 읽는 차다.
    """
    m = smap.mask
    if m.size <= 1 or not m.any():
        return []
    mh, mw = m.shape
    u0, _v0, u1, _v1 = smap.paint
    band = m[int(mh * 0.325):int(mh * 0.675), :]
    solid = band.mean(axis=0) > _GAP_SOLID
    runs: list[tuple[int, int]] = []
    start = None
    for i, s in enumerate(solid):
        if s and start is None:
            start = i
        elif not s and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, mw))
    min_w = max(3, int(0.05 * mw))
    runs = [(a, b) for a, b in runs if b - a >= min_w]
    upp = (u1 - u0) / mw
    out: list[tuple[float, float]] = []
    for (_, a), (b, _) in zip(runs, runs[1:]):
        if (b - a) * upp >= _GAP_MIN * (u1 - u0):
            out.append((u0 + a * upp, u0 + b * upp))
    return out


def _ink_box(smap: SurfaceMap) -> tuple[float, float, float, float] | None:
    """마스크 잉크 상자 (면 유닛)."""
    m = smap.mask
    if m.size <= 1 or not m.any():
        return None
    r = np.where(m.any(1))[0]
    c = np.where(m.any(0))[0]
    u0, v0, u1, v1 = smap.paint
    h, w = m.shape
    return (u0 + c[0] / max(1, w - 1) * (u1 - u0),
            v1 - r[-1] / max(1, h - 1) * (v1 - v0),
            u0 + c[-1] / max(1, w - 1) * (u1 - u0),
            v1 - r[0] / max(1, h - 1) * (v1 - v0))


def _edge_width(smap: SurfaceMap, u: float, side: float
                ) -> tuple[float, float] | None:
    """u 언저리(`side` 방향으로 면 폭의 2%)에서 마스크의 **v 구간** (중심, 폭)."""
    m, us, vs = _grids(smap)
    u0, _v0, u1, _v1 = smap.paint
    d = 0.02 * (u1 - u0)
    lo, hi = (u, u + d) if side > 0 else (u - d, u)
    sel = (us >= lo) & (us <= hi)
    if not sel.any():
        return None
    sub = m[:, sel]
    rows = np.where(sub.any(1))[0]
    if len(rows) < 2:
        return None
    a, b = vs[rows[-1]], vs[rows[0]]
    return ((a + b) / 2, b - a)


def _cross_width(smap: SurfaceMap, zi: int, at: float) -> float:
    """`zi`축(0=u·1=v)의 `at` 언저리(잉크의 6% 띠)에서 **다른 축의 마스크 폭**."""
    m, us, vs = _grids(smap)
    axes = (us, vs)
    scan = axes[zi]
    box = _ink_box(smap)
    if box is None:
        return 0.0
    span = (box[2] - box[0]) if zi == 0 else (box[3] - box[1])
    sel = np.abs(scan - at) <= max(1.0, 0.06 * span)
    if not sel.any():
        return 0.0
    sub = m[:, sel] if zi == 0 else m[sel, :]
    other = vs if zi == 0 else us
    has = np.where(sub.any(1) if zi == 0 else sub.any(0))[0]
    if len(has) < 2:
        return 0.0
    return abs(float(other[has[-1]] - other[has[0]]))


def _z_edge(name: str) -> tuple[str, int, float] | None:
    """그 면에서 차 길이(z)를 쥔 에디터 축."""
    return _find(name, "z")


# 지붕이 "섰다"고 보는 그린하우스 높이의 몫 — 필러 프로필에서 유리 띠를 자르는
# 문턱이다. 루프라인의 92%에 닿으면 필러가 끝나고 지붕이다.
_PILLAR_DONE = 0.92


def pillar_bands(side: SurfaceMap, belt: float, roof: float,
                 cabin: tuple[float, float], rear_dir: float
                 ) -> tuple[tuple[float, float] | None,
                            tuple[float, float] | None]:
    """옆면 실루엣의 **필러 프로필**에서 윈드실드·뒷유리의 차 길이 띠를 읽는다.

    되돌리는 것: (윈드실드 띠, 뒷유리 띠) — **윗면 u 유닛** (u = −z · 차체 면의
    z 원점 맞춤은 0이라 옆면 u와 한 자다).

    근거: 설치 옆면 마스크는 그린하우스를 실루엣 통째로 갖고 있고, 그 **윗
    경계가 곧 필러·지붕 프로필**이다 — A필러 구간에서 벨트라인부터 루프라인
    까지 올라가고, 그 구간의 유리가 앞에서 보면 윈드실드다 (뒤쪽 C필러가
    뒷유리). 윗면 마스크의 유리 구멍은 설치본 소수에만 있으므로(실측 40대 중
    4대) 이 프로필이 유리 자리의 기본 자다.
    """
    m, us, vs = _grids(side)
    if m.size <= 1 or not m.any():
        return None, None
    c0, c1 = min(cabin), max(cabin)
    if c1 - c0 <= 8.0 or roof - belt <= 4.0:
        return None, None
    sel = (us >= c0) & (us <= c1)
    if not sel.any():
        return None, None
    sub = m[:, sel]
    uu = us[sel]
    # 열마다 마스크 윗 경계 (없는 열은 벨트라인으로)
    has = sub.any(0)
    top_i = np.argmax(sub, 0)                     # 행 0 = 위(v1)
    v_top = np.maximum(np.where(has, vs[top_i], belt), belt)
    # 문턱은 **프로필 자신의** 지붕 높이 기준이다 (95백분위) — 바깥에서 잰
    # `roof`를 그대로 쓰면 아치 지붕·안테나 스파이크에서 문턱이 프로필 위로
    # 떠서 필러가 안 잡힌다 (실측: 미니EV의 윈드실드 띠가 캐빈 통째가 됐다).
    peak = float(np.quantile(v_top, 0.95))
    if peak - belt <= 4.0:
        return None, None
    thr = belt + _PILLAR_DONE * (peak - belt)
    up = np.where(v_top >= thr)[0]
    if not len(up):
        return None, None
    # +u가 차 뒤(rear_dir>0)면 앞 필러는 낮은 u 쪽이다
    if rear_dir > 0:
        ws_side = (float(uu[0]), float(uu[up[0]]))
        rw_side = (float(uu[up[-1]]), float(uu[-1]))
    else:
        ws_side = (float(uu[up[-1]]), float(uu[-1]))
        rw_side = (float(uu[0]), float(uu[up[0]]))
    # 옆면 u → 윗면 u (둘 다 z를 쥔다): side u 부호가 -z(왼쪽)면 그대로,
    # +z(오른쪽)면 뒤집는다. 윗면 u는 늘 -z다.
    sgn = -_find(side.name, "z")[2] if _find(side.name, "z") else 1.0
    def _flip(band):
        a, b = sgn * band[0], sgn * band[1]
        return (min(a, b), max(a, b))
    ws, rw = _flip(ws_side), _flip(rw_side)
    # 캐빈의 6할을 넘는 띠는 프로필이 뭉개진 것이다 (원박스 밴·아치 지붕) —
    # 그런 자로 이으면 유리가 지붕 절반까지 늘어난다. 차라리 안 잇는다.
    lim = 0.60 * (c1 - c0)
    if ws[1] - ws[0] <= 4.0 or ws[1] - ws[0] > lim:
        ws = None
    if rw[1] - rw[0] <= 4.0 or rw[1] - rw[0] > lim:
        rw = None
    return ws, rw


def glass_bands(maps: dict[str, SurfaceMap],
                hints: dict[str, tuple[float, float]] | None = None
                ) -> dict[str, tuple[float, float]]:
    """윗면에서 **유리가 차지하는 차 길이 띠들** — {유리 면 이름: (u0, u1)} 윗면 유닛.

    근거는 둘이고 마스크 구멍이 이긴다: ① 윗면 마스크의 **유리 구멍**
    (`top_gaps` — 앞 = 윈드실드 · 뒤 = 뒷유리 · 가운데 = 선루프) ② 구멍이 없는
    차(설치본 다수 — 실측 40대 중 36대)는 옆면 필러 프로필에서 읽은 `hints`
    (`pillar_bands` — 부르는 쪽이 넘긴다. 선루프는 실루엣에 안 보이므로 구멍이
    있을 때만 잡는다).

    **실측이 있으면 그것이 곧 구멍이다** — 인게임 프로브로 유리를 잰 차는 그리는
    지도(`SurfaceMap.drawn`)에 유리가 구멍으로 파여 있다 (`game.seam.top_glass`).
    """
    sm = maps.get("top")
    if sm is None:
        return {}
    # 실측이 있으면 **그것만** 본다 — 프로브가 이미 "여기는 유리다"를 말했으므로
    # 필러 어림으로 다시 가릴 것이 없다 (어림은 A필러·카울까지 문다).
    measured = sm.drawn is not None
    gaps = top_gaps(sm.drawn or sm)
    hints = {} if measured else (hints or {})
    which: dict[str, tuple[float, float]] = {}
    if gaps and hints:
        # 구멍의 정체는 **필러 띠와의 겹침**으로 가른다 — 순서로 가르면 윙
        # 장착부·콕핏 구멍이 윈드실드로 둔갑한다 (실측: 드리프트 코롤라의 뒤쪽
        # 구멍 하나가 윈드실드로 붙어 어긋남 57%). 어느 띠와도 안 겹치고 두 띠
        # 사이(지붕)에 든 구멍만 선루프다.
        ws_b, rw_b = hints.get("windshield"), hints.get("rear_window")

        def _ov(g, band):
            if band is None:
                return 0.0
            o = min(g[1], band[1]) - max(g[0], band[0])
            return max(0.0, o) / max(1e-6, g[1] - g[0])

        for g in gaps:
            o_ws, o_rw = _ov(g, ws_b), _ov(g, rw_b)
            if max(o_ws, o_rw) >= 0.3:
                lab = "windshield" if o_ws >= o_rw else "rear_window"
            elif (ws_b is not None and rw_b is not None
                    and g[0] >= ws_b[1] and g[1] <= rw_b[0]):
                lab = "sunroof"
            else:
                continue                          # 유리가 아닌 구멍 (윙 장착부 등)
            if lab not in which:
                which[lab] = g
    elif gaps:
        which["windshield"] = gaps[0]
        if len(gaps) >= 2:
            which["rear_window"] = gaps[-1]
        if len(gaps) >= 3:
            which["sunroof"] = gaps[len(gaps) // 2]
    for k, v in hints.items():
        if k != "sunroof":                        # 실루엣은 선루프를 못 본다
            which.setdefault(k, v)
    return which


def glass_folds(maps: dict[str, SurfaceMap], src: str,
                dsts: tuple[str, ...] = ("windshield", "rear_window", "sunroof"),
                hints: dict[str, tuple[float, float]] | None = None
                ) -> list[Fold]:
    """윗면 → 유리 면 넘침 변환들 — 유리 띠(`glass_bands`)가 이음새다.

    띠의 z 구간 ↔ 유리 잉크의 z 구간을 끝끼리 맞춘다 (양 끝이 한 번에 맞으므로
    앞뒤 어느 쪽에서 넘어와도 같은 변환이다). 차 폭(x)은 둘 다 중심선 대칭이라
    띠 곁 마스크의 v 중심 ↔ 유리 잉크 중심을 맞춘다 — 재는 자리는 **지붕 쪽
    모서리**다 (그린하우스 폭 ≈ 유리 폭. 카울 쪽은 펜더까지 물어 넓게 잰다).

    한 띠에 넘침 방향이 둘이다 (앞 구간에서 뒤로 · 뒤 구간에서 앞으로) —
    A·b가 같고 이음선만 다른 Fold 두 개를 낸다.
    """
    sm = maps.get(src)
    if src != "top" or sm is None:
        return []
    which = glass_bands(maps, hints)
    out: list[Fold] = []
    for dst, (g0, g1) in which.items():
        if dst not in dsts:
            continue
        dm = maps.get(dst)
        if dm is None or dm.mask.size <= 1:
            continue
        ink = _ink_box(dm)
        if ink is None:
            continue
        sz = _z_edge(src)
        dz = _z_edge(dst)
        sx_ = _find(src, "x")
        dx_ = _find(dst, "x")
        if None in (sz, dz, sx_, dx_):
            continue
        # 띠 곁 마스크의 v 구간 — 유리의 차 폭 자리다 (지붕 쪽 모서리 기준)
        if dst == "windshield":
            got = [_edge_width(sm, g1, +1.0)]
        elif dst == "rear_window":
            got = [_edge_width(sm, g0, -1.0)]
        else:
            got = [_edge_width(sm, g0, -1.0), _edge_width(sm, g1, +1.0)]
        got = [e for e in got if e is not None and e[1] > 1.0]
        if not got:
            continue
        w_top = sum(e[1] for e in got) / len(got)
        c_top = sum(e[0] for e in got) / len(got)
        # 배율 — 유리 잉크 구간 ÷ 윗면 구간 (축마다)
        z_span = (ink[3] - ink[1]) if dz[0] == "v" else (ink[2] - ink[0])
        x_span = (ink[2] - ink[0]) if dz[0] == "v" else (ink[3] - ink[1])
        if g1 - g0 <= 1.0 or w_top <= 1.0 or z_span <= 1.0 or x_span <= 1.0:
            continue
        s_z = z_span / (g1 - g0)
        s_x = x_span / w_top
        if max(s_z, s_x) / max(1e-6, min(s_z, s_x)) > GLASS_ANISO:
            continue
        A = np.zeros((2, 2))
        b = np.zeros(2)
        # z 방향은 **유리 마스크의 사다리꼴**이 정한다: 유리는 지붕 쪽이 좁고
        # 카울/데크 쪽이 넓다. 축 표의 유리 행은 못 쓴다 — 유리 면은 저장 회전이
        # 차마다 달라 부호가 안 선다 (실측: 같은 코드로 인테그라 윈드실드 조각은
        # 카울에 붙고 줄리아 조각은 지붕 끝에 붙었다). 지붕에 붙은 띠 끝(윈드실드
        # 는 g1·뒷유리는 g0)을 좁은 끝에, 반대 끝을 넓은 끝에 맞춘다 — 선형이라
        # 가운데는 저절로 선다.
        zi = dz[1]
        z_lo = ink[1] if dz[0] == "v" else ink[0]
        z_hi = ink[3] if dz[0] == "v" else ink[2]
        w_lo = _cross_width(dm, zi, z_lo)
        w_hi = _cross_width(dm, zi, z_hi)
        if w_lo <= 0 or w_hi <= 0:
            continue
        wide, narrow = (z_lo, z_hi) if w_lo >= w_hi else (z_hi, z_lo)
        if dst == "windshield":
            roof_src, far_src = g1, g0
        elif dst == "rear_window":
            roof_src, far_src = g0, g1
        else:                                     # 선루프 — 사다리꼴이 없다.
            # 축 표대로: 윗면과 같은 u=-z 저장이 다수라 부호 곱을 그대로 쓴다.
            roof_src = far_src = None
        if roof_src is not None:
            slope = (narrow - wide) / (roof_src - far_src)
            A[zi, sz[1]] = slope
            b[zi] = wide - slope * far_src
        else:
            sgn_z = sz[2] * dz[2]
            A[zi, sz[1]] = sgn_z * s_z
            d_hi = z_lo if dz[2] < 0 else z_hi
            b[zi] = d_hi - sgn_z * s_z * g0
        # x: 중심 맞춤 (둘 다 차 중심선 대칭)
        sgn_x = sx_[2] * dx_[2]
        A[dx_[1], sx_[1]] = sgn_x * s_x
        d_cx = ((ink[0] + ink[2]) / 2) if dx_[0] == "u" else ((ink[1] + ink[3]) / 2)
        b[dx_[1]] = d_cx - sgn_x * s_x * c_top
        why = (f"유리 띠 {g0:.0f}~{g1:.0f} ↔ {dst} 잉크 · 배율 z {s_z:.2f} · "
               f"x {s_x:.2f}")
        out.append(Fold(src=src, dst=dst, axis=sz[0], sign=+1.0, edge=float(g0),
                        far=float(g1), A=A, b=b, why=why))
        out.append(Fold(src=src, dst=dst, axis=sz[0], sign=-1.0, edge=float(g1),
                        far=float(g0), A=A, b=b, why=why))
    return out


def invert(f: Fold) -> Fold | None:
    """넘침 변환의 **역** — 유리 면에 직접 올린 그림을 차체 쪽으로 되잇는 자.

    이음선은 dst 쪽 좌표로 옮기고, 넘침 방향은 A의 부호를 따라 뒤집는다.
    """
    try:
        Ai = np.linalg.inv(f.A)
    except np.linalg.LinAlgError:
        return None
    # src의 (axis, sign) 넘침축이 dst의 어느 축으로 갔나
    si = 0 if f.axis == "u" else 1
    col = f.A[:, si]
    di = int(np.argmax(np.abs(col)))
    d_ax = "u" if di == 0 else "v"
    d_sgn = f.sign * (1.0 if col[di] > 0 else -1.0)

    p = np.zeros(2)
    p[si] = f.edge
    edge = float((f.A @ p + f.b)[di])
    # 역방향의 띠는 **열려 있다** — 갈 곳(차체 면)은 이음선 너머 전부다. 원래
    # `far`는 "목적 면(유리)이 src의 이 구간에만 산다"였고, 역에서는 그 구간이
    # src(유리) 자신이라 자를 것이 아니다.
    return Fold(src=f.dst, dst=f.src, axis=d_ax, sign=-d_sgn, edge=edge,
                A=Ai, b=-(Ai @ f.b), why=f"{f.why} (역)")


def span_of(src: str, dst: str, box: tuple[float, float, float, float]
            ) -> tuple[float, float] | None:
    """상자에서 이 짝의 **공유 축 구간**을 뽑는다 (src 유닛) — 이음선을 잴 자리다."""
    shared = _shared(src, dst)
    if shared is None:
        return None
    got = _find(src, shared)
    if got is None:
        return None
    return (box[0], box[2]) if got[0] == "u" else (box[1], box[3])


def folds_for(maps: dict[str, SurfaceMap], src: str,
              box: tuple[float, float, float, float] | None = None,
              span: tuple[float, float] | None = None, hull=None) -> list[Fold]:
    """이 면에서 나가는 넘침 변환 전부 (설 수 있는 것만).

    `box`(넘치는 그림이 이 면에서 덮는 상자)를 주면 이음선을 **그 상자가 걸친
    구간에서** 잰다 — 짝마다 공유 축이 다르므로 구간도 짝마다 다르다.
    """
    out = []
    for dst in neighbors(src):
        sp = span if span is not None else (
            span_of(src, dst, box) if box is not None else None)
        f = fold(maps, src, dst, span=sp, hull=hull)
        if f is not None:
            out.append(f)
    return out
