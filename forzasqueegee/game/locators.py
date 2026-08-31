r"""차량 **부품 자리** — 설치 파일의 `Locators.xml`을 면 유닛으로 옮긴다.

차량 zip에 `carLocator_*` 38~51개가 **미터 좌표**로 들어 있다: 도어 노브 넷 ·
주유구 · 휠 중심 넷 · 사이드스커트 · 후드 · 윙 · 범퍼 · 헤드/테일라이트.
도색 마스크에는 이 구멍이 없어서 지금까지 "**안 잰다**"로 비워 둔 자리였다
(업계 지침 — "도어 노브·주유구가 얼굴에 안 겹치게").

## 좌표계

`SceneTransform`의 `_41·_42·_43`이 이동(미터)이다. 3D 축은 `game/fold.py`의
축 표와 같은 규약이다: **+x = 차의 오른쪽 · +y = 위 · +z = 차 앞**. 확인:
`doorHandleLF`가 x 음수(왼쪽), `bumperF`가 z 양수(앞), 줄리아의 휠 z 간격
2.819 m가 실차 휠베이스 2.820 m다.

XML이지만 `ElementTree`로는 못 읽는다 — `BoneName="<root>"` 속성값에 홑화살괄호가
날것으로 들어 있어 파서가 죽는다. 그래서 정규식으로 이름·이동만 뜬다.

## 차 메시가 있으면 등록을 안 적합한다 (2026-08-31)

기하 덤프가 떠 있으면 `Registration`이 **면마다의 닫힌 식**을 그대로 쓴다
(`game.fsgeom` — `for_car`가 붙여 준다). 아래 적합은 덤프가 없을 때의 길이다.
갈리는 자리 둘:

- **배율이 한 자가 아니다.** 아래 등방 가정은 앞·뒤 면의 **세로**에서 안 선다 —
  덤프 14대에서 같은 물리 높이를 두 면으로 잰 값이 56% 벌어진다 (실비아 옆면
  204 유닛/m ↔ front 87 ↔ rear 144). 메시는 면마다 제 투영을 갖는다.
- **아치를 안 찾아도 된다.** 아래 적합은 휠아치 두 개에 직선을 맞춰 K와 u0를
  같이 뽑는데, 아치가 얕은 차는 범퍼 어림으로 물러나며 `uncertain`이 되고
  정밀이 필요한 자가 통째로 물러난다 (덤프 16대 중 셋). 메시는 그 셋을 세운다.

다만 **x축은 여전히 적합 쪽이다** — 덤프의 x 방향과 `fold.AXES`의 x 방향이
갈려 있다 (`game.hull.MeshHull`). 지금 이 자를 부르는 쪽이 묻는 것은 전부
z·y축이라(옆면 부품 자리 · 리어 범퍼 높이 · 후드 u) 다툼에 안 걸린다.

## 미터 → 면 유닛 등록 (`register`) — 덤프가 없을 때

면 유닛은 **한 차 안에서 등방**이고(`fold.AXES` — 636대 전부 xScale=yScale=1)
축 배치도 표가 준다. 그래서 남는 미지수는 **배율 K(유닛/m)와 면별 원점**뿐이다.

- **K와 u 원점은 휠아치가 준다** — 옆면 마스크 아래 경계의 파임 두 개가 휠 중심
  둘(`carLocator_wheel*`, z 간격 = 휠베이스)에 대응한다. 두 점이 직선 하나를
  못 박으므로 K와 u0가 같이 나온다.
- **아치를 못 찾으면 범퍼로 물러난다** (`_bumper_fit`) — 마스크에 파임이 없거나
  얕은 차가 있다. 그때는 옆면 상자의 u 폭을 앞·뒤 범퍼 z 간격에 대어 K를 낸다
  (보정 `BUMPER_K_CORR`). 아치만 못한 자라 그렇게 선 등록은 **늘 `uncertain`**
  이고, 정밀이 필요한 자(부품 회피·밴드 높이)는 그 표시를 보고 물러난다.
  없는 것보다 거친 것이 나은 이유는 등록이 아예 없으면 그 차의 자리가 FLS의
  면별 맞춤으로 넘어가는데 앞·뒤 면에서 세로로 크게 어긋나기 때문이다.
- **좌우가 서로의 검증자다.** 좌우 마스크는 별개 텍스처인데 같은 차를 잰다:
  실측 117대에서 K 어긋남 중앙 0.00% · p95 1.16%이고, u0는 **부호만 반대로
  같은 값**이다 (줄리아 -25.0/+25.0 · M5 CS -37.4/+41.4 — 옆면 u축이 좌우
  반대 방향이므로 같은 물리 오프셋이 이렇게 보인다). 이 둘이 어긋나면 아치
  검출이 샌 것이라 **스스로 물러난다**.
- **u0는 0이 아니다** — 마스크 상자는 차 길이의 한가운데를 원점에 두는데
  z=0은 휠베이스 한가운데라, 앞뒤 오버행이 다른 만큼 어긋난다 (줄리아 -0.12 m ·
  RAM TRX -0.24 m — 뒤 오버행이 긴 쪽으로, 방향이 실차와 맞는다).
- **v 원점은 마스크 최하단**이다. 로케이터 y=0이 사이드실 평면이기 때문이다:
  휠 중심 y가 0.12~0.19 m로 타이어 반지름(0.30~0.35)보다 훨씬 작아 지면이
  아니고, 아치 꼭대기에서 되짚은 허브 높이가 최하단과 중앙 6.7유닛(3 cm)에서
  맞는다 (113대). `fold.py`의 높이 등록("면마다 제일 낮은 도색을 맞춘다")과
  같은 자다.
- **윗면·앞뒤 면은 원점을 나눠 쓴다** — 윗면 u는 옆면과 같은 −z축이라 u0가
  같고(`fold.py`의 원점 맞춤), 차 폭(x) 축은 마스크가 중심선 대칭이라 0이다.

**유리 면은 안 한다** — 제 배율로 늘려 저장돼 있어(VW Type2 도어 유리 929유닛 >
차 길이 609) 이 등방 가정이 안 선다.

검증은 설치본 전수 대조로 했다 (등록률·잔차).
"""

from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import fold as gfold
from .surface import SurfaceMap

# `<Name value="…"/>` 바로 뒤의 `<SceneTransform …/>` 한 쌍만 뜬다
_PAIR = re.compile(r'<Name value="([^"]+)"\s*/>\s*<SceneTransform([^/>]*)/>')
_VAL = re.compile(r'value\._(\d\d)="([-0-9.eE]+)"')

# 등록을 못 믿을 자리들 (실측 117대 분포에서 잡은 문턱)
K_SPLIT_MAX = 0.03        # 좌우 K 어긋남 상한 (p95 = 1.16%)
ARCH_R_MIN = 0.20         # 아치 반지름(m) 하한 — 아래는 검출 실패 (p5 = 0.275)
ARCH_R_MAX = 0.70         # 상한 — 위는 아치 둘이 하나로 붙은 것 (p95 = 0.558)
U0_SPLIT_MAX = 0.06       # |u0_L + u0_R| 상한 (미터)
# 윗면이 옆면과 u 원점을 나눠 쓰나 — 범퍼 로케이터를 윗면 u로 옮겨 마스크 u 끝과
# 견준다. 범퍼 로케이터는 **부품 장착점**이라 차 끝과 몇 cm 다르므로(실측 중앙
# 11 cm) 넉넉히 잡되, 통째로 어긋난 차는 윗면만 버린다 (옆면은 그대로 쓴다).
TOP_END_MAX = 0.50        # 미터


def _axis_of(name: str, which: int) -> tuple[str, float] | None:
    """면의 u(0)·v(1) 축 → (3D 축 이름, 부호). 표에 없으면 None."""
    ax = gfold.AXES.get(name)
    if not ax:
        return None
    s = ax[which]
    return s[1], (1.0 if s[0] == "+" else -1.0)


# ---------- 읽기 ----------
@lru_cache(maxsize=16)
def read(media: str, root: Path | None = None) -> dict[str, tuple[float, float, float]]:
    """차 하나의 로케이터 — {이름: (x, y, z)} 미터. 없으면 빈 dict."""
    from . import carfiles

    base = root or carfiles.install_dir()
    if base is None:
        return {}
    zp = base / "media" / "Cars" / f"{media}.zip"
    try:
        with zipfile.ZipFile(zp) as z:
            raw = z.read("Locators.xml").decode("utf-8", "replace")
    except (OSError, KeyError, zipfile.BadZipFile):
        return {}
    out: dict[str, tuple[float, float, float]] = {}
    for m in _PAIR.finditer(raw):
        vals = {k: float(v) for k, v in _VAL.findall(m.group(2))}
        if not {"41", "42", "43"} <= vals.keys():
            continue
        # 같은 이름이 두 번 나오는 차가 있다 (인테그라 Exhaust_001) — 첫 것을 쓴다
        out.setdefault(m.group(1), (vals["41"], vals["42"], vals["43"]))
    return out


# ---------- 등록 ----------
@dataclass
class Registration:
    """미터 ↔ 면 유닛. `k`는 유닛/m, `u0`·`v0`는 면별 원점."""

    media: str
    k: float
    u0: dict[str, float] = field(default_factory=dict)
    v0: dict[str, float] = field(default_factory=dict)
    uncertain: bool = False
    note: str = ""
    # 차 메시 (`game.fsgeom.CarGeom`) — 있으면 **이것이 이긴다**. 적합이 아니라
    # 닫힌 식이고, 면마다 제 배율이라 등방 가정도 안 쓴다.
    geom: object = field(default=None, repr=False, compare=False)

    def has(self, surface: str) -> bool:
        """이 면의 자리를 낼 수 있나 — 메시든 적합이든."""
        au, av = _axis_of(surface, 0), _axis_of(surface, 1)
        if au is None or av is None:
            return False
        if self.geom is not None and "x" not in (au[0], av[0]) \
                and self.geom.get(surface) is not None:
            return True
        return surface in self.u0 and surface in self.v0

    def unit(self, surface: str, xyz: tuple[float, float, float]
             ) -> tuple[float, float] | None:
        """3D 미터 점 → 그 면의 (u, v) 유닛. 등록이 없는 면이면 None.

        **메시가 있으면 메시가 낸다.** 적합(`register`)은 배율 하나가 차 전체에
        통한다고 보는데, 앞·뒤 면의 세로 배율은 옆면의 절반 이하다 (덤프 14대:
        높이축이 면끼리 56% 벌어진다). 메시는 면마다 제 투영을 갖는다.

        **단 x축은 안 쓴다** — 덤프의 x 방향과 `fold.AXES`의 x 방향이 갈려 있다
        (`hull.MeshHull`). 지금 이 자를 부르는 쪽이 묻는 것은 전부 z·y축이라
        (옆면 부품 자리 · 리어 범퍼 높이 · 후드 u) 다툼에 안 걸린다.
        """
        au = _axis_of(surface, 0)
        av = _axis_of(surface, 1)
        if au is None or av is None:
            return None
        side = self.geom.get(surface) if self.geom is not None else None
        if side is not None and "x" not in (au[0], av[0]):
            u, v = side.to_face({"x": xyz[0], "y": xyz[1], "z": xyz[2]})
            return (float(u), float(v))
        if surface not in self.u0 or surface not in self.v0:
            return None
        p = {"x": xyz[0], "y": xyz[1], "z": xyz[2]}
        return (au[1] * self.k * p[au[0]] + self.u0[surface],
                av[1] * self.k * p[av[0]] + self.v0[surface])


def _lowest_paint(smap: SurfaceMap) -> float:
    """마스크에서 **실제로 칠해지는 가장 낮은 v** (짧은 돌기는 버린다).

    `seam.side_geom`의 사이드실과 같은 자다 — 옆면 말고도 쓰려고 따로 둔다.
    """
    m = smap.mask
    if m.size <= 1 or not m.any():
        return float(smap.paint[1])
    v0, v1 = smap.paint[1], smap.paint[3]
    vs = np.linspace(v1, v0, m.shape[0])
    solid = np.where(m.sum(1) >= 0.06 * m.shape[1])[0]
    return float(vs[solid[-1]]) if len(solid) else float(v0)


def _arch_sep(smap: SurfaceMap, locs: dict[str, tuple[float, float, float]],
              zf: float, zr: float) -> float | None:
    """**예상 아치 간격** (면 유닛) — 아치 검출의 씨앗이다.

    닭이 먼저냐의 자리다: 배율 K는 아치가 주는데 아치를 찾으려면 간격을 알아야
    한다. 끊는 것은 **범퍼 로케이터**다 — 앞뒤 범퍼 z 간격이 차 길이고 그것이
    곧 옆면 상자의 u 폭이므로, 아치를 안 쓰고 K를 어림할 수 있다. 이 어림은
    범퍼 장착점이 차 끝보다 조금 안쪽이라 몇 %가 틀리지만(전수 실측: 끝과
    중앙 9.7 cm), 씨앗은 **자리만 가리키면 되고** 최종 K는 찾은 아치가 정한다.
    """
    bf, br = locs.get("carLocator_bumperF"), locs.get("carLocator_bumperR")
    if bf is None or br is None:
        return None
    length = abs(bf[2] - br[2])
    if length < 1.0 or abs(zf - zr) < 0.5:
        return None
    k0 = (smap.paint[2] - smap.paint[0]) / length
    return k0 * abs(zf - zr) if k0 > 1e-6 else None


def _bumper_fit(smap: SurfaceMap, surface: str,
                locs: dict[str, tuple[float, float, float]]
                ) -> tuple[float, float, float, None] | None:
    """옆면 하나 → (K, u0, v0, None) — **아치 없이** 범퍼 간격으로.

    아치 검출이 통째로 실패하는 차가 있다 (마스크에 파임이 없거나 얕다 — 표본
    120대 중 21대). 그때 아무 말도 안 하면 그 차는 등록이 통째로 없어 자리를
    FLS 맞춤에 맡기게 되는데, 앞·뒤 면에서 그쪽이 세로로 1.3~1.7배 어긋난다.
    **거친 자라도 있는 편이 낫다**는 자리가 여기다.

    쓰는 자는 둘 다 게임 것이다: 도색 상자의 u 폭(= 게임의 면 좌표 상자 —
    2026-08-26 이동 클램프 실측으로 오차 0 확인)과 `Locators.xml`의 앞·뒤 범퍼
    z 간격(미터). `BUMPER_K_CORR`는 범퍼 장착점이 차 끝보다 안쪽인 몫이다.

    정확도는 아치만 못하다 — 아치가 서는 99대에서 범퍼K/아치K가 중앙 1.002지만
    |어긋남| 중앙 3.0% · p90 18.7%다. 그래서 이 길로 선 등록은 **늘 `uncertain`**
    이고, 정밀이 필요한 자(부품 회피·밴드 높이)는 이 등록을 안 쓰고 물러난다.
    """
    bf, br = locs.get("carLocator_bumperF"), locs.get("carLocator_bumperR")
    if bf is None or br is None:
        return None
    span = abs(bf[2] - br[2])
    ax = _axis_of(surface, 0)
    if span < 1.0 or ax is None or ax[0] != "z":
        return None
    k = (smap.paint[2] - smap.paint[0]) / span * BUMPER_K_CORR
    if k <= 1e-6:
        return None
    # 상자는 차 길이에 맞춰져 있으므로 **범퍼 한가운데가 u=0**이다
    zmid = (bf[2] + br[2]) / 2.0
    return (k, -ax[1] * k * zmid, _lowest_paint(smap), None)


def _side_fit(smap: SurfaceMap, sign: float, zf: float, zr: float,
              sep: float | None = None, seeded: bool = False
              ) -> tuple[float, float, float, float] | None:
    """옆면 하나 → (K, u0, v0, 아치 반지름 m). 아치가 둘이 아니면 None.

    `sign`은 그 면의 u축 부호다 (side_left −z, side_right +z). 앞바퀴(z 큼)는
    그래서 왼쪽 면에서 u가 작고 오른쪽 면에서 u가 크다.

    `sep`은 씨앗(예상 아치 간격)이고 `seeded`면 그 길로만 찾는다.
    """
    from . import seam as gseam

    g = gseam.side_geom(smap, arch_sep=sep, seed_arches=seeded)
    w = sorted(g.wheels, key=lambda t: t[0])
    if len(w) < 2 or abs(zf - zr) < 0.5:
        return None
    k = (w[-1][0] - w[0][0]) / abs(zf - zr)
    if k <= 1e-6:
        return None
    u_front = w[0][0] if sign < 0 else w[-1][0]
    return k, u_front - sign * k * zf, g.sill, (w[0][1] + w[-1][1]) / 2 / k


def _split_rank(fits: dict) -> tuple[int, float]:
    """좌우 일치 순위 — **작을수록 좋다**. (검사 통과 여부, 점수).

    이 자는 아치를 찾는 데 **안 쓴 것**이라 어느 답을 골라도 심판이 된다:
    좌우 옆면은 별개 텍스처인데 같은 차를 재므로 배율이 같아야 하고 u 원점은
    부호만 반대여야 한다.

    **점수만으로 고르면 안 된다.** 점수는 정규화한 어긋남의 합이라 한 축이
    문턱을 넘어도 다른 축이 작으면 합이 이길 수 있다 — 그러면 검사를 통과하던
    답이 통과 못 하는 답에 져서 차가 물러난다. 그래서 통과 여부가 **먼저**고
    점수는 그다음이다.
    """
    (kl, u0l, _vl, rl), (kr, u0r, _vr, rr) = fits["side_left"], fits["side_right"]
    k = (kl + kr) / 2.0
    ks = abs(kl - kr) / max(kl, kr)
    us = abs(u0l + u0r) / max(1e-6, k)
    rad = (rl + rr) / 2.0
    ok = (ks <= K_SPLIT_MAX and us <= U0_SPLIT_MAX
          and ARCH_R_MIN <= rad <= ARCH_R_MAX)
    return (0 if ok else 1, ks / K_SPLIT_MAX + us / U0_SPLIT_MAX)


def register(maps: dict[str, SurfaceMap], locs: dict[str, tuple[float, float, float]],
             media: str = "") -> Registration | None:
    """면 지도 + 로케이터 → 등록. 근거가 모자라면 None, 미심쩍으면 `uncertain`.

    **없는 것보다 틀린 것이 나쁘다** — 이 자는 "인물이 도어 노브를 피한다" 같은
    배치 결정에 쓰이므로, 못 믿을 때는 아무 말도 안 하는 쪽이 맞다.
    """
    wf, wr = locs.get("carLocator_wheelLF"), locs.get("carLocator_wheelLR")
    if wf is None or wr is None:
        return None
    # **아치를 두 길로 찾고 좌우 일치가 고른다.** 문턱 길은 파임이 얕은 차에서
    # 통째로 실패하고(전수 101대), 씨앗 길은 얕은 봉우리를 아치로 오인할 수 있다.
    # 어느 쪽도 혼자서는 못 믿으므로 **등록에 안 쓴 자**(좌우 대조)로 고른다.
    cands: list[dict] = []
    for seeded in (False, True):
        fits = {}
        for name, sign in (("side_left", -1.0), ("side_right", 1.0)):
            sm = maps.get(name)
            if sm is None or sm.mask.size <= 1:
                continue
            got = _side_fit(sm, sign, wf[2], wr[2], seeded=seeded,
                            sep=_arch_sep(sm, locs, wf[2], wr[2]))
            if got is not None:
                fits[name] = got
        if len(fits) == 2:
            cands.append(fits)
    notes: list[str] = []
    if cands:
        fits = min(cands, key=_split_rank)
    else:
        # **아치가 없으면 범퍼로 물러난다** — 없는 것보다 거친 것이 낫다
        # (`_bumper_fit`의 문서에 그 값을 잰 수가 있다).
        fits = {}
        for name in ("side_left", "side_right"):
            sm = maps.get(name)
            if sm is None or sm.mask.size <= 1:
                continue
            got = _bumper_fit(sm, name, locs)
            if got is not None:
                fits[name] = got
        if len(fits) != 2:
            return None
        notes.append("휠아치를 못 찾아 범퍼 간격으로 잰 배율이다 "
                     "(|어긋남| 중앙 3% · p90 19%)")
    (kl, u0l, vl, rl), (kr, u0r, vr, rr) = fits["side_left"], fits["side_right"]
    k = (kl + kr) / 2.0
    if abs(kl - kr) / max(kl, kr) > K_SPLIT_MAX:
        notes.append(f"좌우 배율이 {abs(kl - kr) / max(kl, kr) * 100:.1f}% 어긋난다 "
                     f"({kl:.0f}·{kr:.0f} 유닛/m)")
    # u0는 좌우가 **부호만 반대**여야 한다 (같은 물리 오프셋을 반대 축으로 본다)
    if abs(u0l + u0r) / k > U0_SPLIT_MAX:
        notes.append(f"좌우 u 원점이 {abs(u0l + u0r) / k * 100:.0f} cm 어긋난다")
    if rl is not None and rr is not None:          # 범퍼 길에는 아치가 없다
        rad = (rl + rr) / 2.0
        if not (ARCH_R_MIN <= rad <= ARCH_R_MAX):
            notes.append(f"휠아치 반지름이 {rad:.2f} m다 (아치 검출 실패로 본다)")
    u_side = (u0l - u0r) / 2.0            # 좌우 평균 (오른쪽은 부호가 반대다)
    # **옆면 판정은 여기서 끝난다** — 아래 윗면 검사는 윗면만 버리지 이 판정을
    # 안 뒤집는다 (옆면 등록은 성한데 윗면 하나 때문에 전부 버릴 이유가 없다).
    reg = Registration(media=media, k=round(k, 2), uncertain=bool(notes))
    reg.u0["side_left"] = round(u_side, 1)
    reg.u0["side_right"] = round(-u_side, 1)
    reg.v0["side_left"] = round(vl, 1)
    reg.v0["side_right"] = round(vr, 1)
    # 윗면은 옆면과 **같은 −z축**이라 u 원점을 나눠 쓴다 (`fold.py` 원점 맞춤).
    # 폭(x) 축은 마스크가 중심선 대칭이라 0이다 — 앞·뒤 면의 u도 같은 이유로 0.
    tm = maps.get("top")
    if tm is not None and tm.mask.size > 1:
        reg.u0["top"] = round(u_side, 1)
        reg.v0["top"] = 0.0
        # 등록에 **안 쓴** 로케이터로 건다 (독립 검증): 앞·뒤 범퍼가 윗면 마스크의
        # u 양 끝에 앉아야 한다. 어긋나면 윗면만 버린다 — 옆면 등록은 성하다.
        worst = 0.0
        for key, end in (("carLocator_bumperF", tm.paint[0]),
                         ("carLocator_bumperR", tm.paint[2])):
            p = locs.get(key)
            uv = reg.unit("top", p) if p is not None else None
            if uv is not None:
                worst = max(worst, abs(uv[0] - end) / k)
        if worst > TOP_END_MAX:
            reg.u0.pop("top")
            reg.v0.pop("top")
            notes.append(f"윗면 u 원점이 범퍼와 {worst * 100:.0f} cm 어긋난다 — "
                         f"윗면은 안 쓴다")
    for name in ("front", "rear"):
        sm = maps.get(name)
        if sm is None or sm.mask.size <= 1:
            continue
        reg.u0[name] = 0.0
        reg.v0[name] = round(_lowest_paint(sm), 1)
    reg.note = " · ".join(notes)
    return reg


# ---------- 쓰는 쪽이 묻는 것 ----------
# 얼굴에 겹치면 안 되는 부품 (업계 지침) — **차 밖에 보이는 것만**.
# `doorHandleInt*`는 실내 손잡이라 뺀다 (CRX 실측: 바깥 노브 x -0.815 옆에
# 실내 -0.670이 같이 있다 — 안 빼면 안 보이는 점을 피하느라 인물이 밀린다).
AVOID_SIDE = ("doorHandleLF", "doorHandleLR", "doorHandleRF", "doorHandleRR",
              "Fuel", "fuel")
# 지침이 부르는 이름 (사람이 읽는 노트용)
AVOID_KO = {"doorHandle": "도어 노브", "Fuel": "주유구", "fuel": "주유구"}


def avoid_points(reg: Registration | None,
                 locs: dict[str, tuple[float, float, float]],
                 surface: str) -> list[tuple[float, float, str]]:
    """그 면에서 **얼굴이 피해야 할 자리** — (u, v, 이름). 등록이 없으면 빈 목록.

    좌우 면은 제 쪽 부품만 본다 (반대편 노브는 그 면에 안 보인다). 주유구는
    한쪽에만 있고 (`carLocator_Fuel`의 x 부호가 그쪽이다) 중앙(x≈0)이면 뒤에
    있는 차라 옆면과 무관하다.
    """
    if reg is None or reg.uncertain or not reg.has(surface):
        return []
    want = "L" if surface == "side_left" else "R"
    out: list[tuple[float, float, str]] = []
    for key in AVOID_SIDE:
        p = locs.get("carLocator_" + key)
        if p is None:
            continue
        if abs(p[0]) < 0.15:               # 차 한가운데 = 이 면에 없다
            continue
        if key.startswith("doorHandle"):
            # 이름에 좌우가 박혀 있다 (…LF·…RR) — 제 쪽만
            if ("L" if p[0] < 0 else "R") != want:
                continue
        elif (("L" if p[0] < 0 else "R") != want):
            continue
        uv = reg.unit(surface, p)
        if uv is None:
            continue
        label = next((v for k, v in AVOID_KO.items() if key.startswith(k)), key)
        out.append((round(uv[0], 1), round(uv[1], 1), label))
    return out


def bumper_v(reg: Registration | None,
             locs: dict[str, tuple[float, float, float]],
             surface: str) -> float | None:
    """앞·뒤 면에서 **범퍼 높이** v — 하부 밴드를 앉힐 자리다.

    `hood_u`와 같은 성격의 자다: 경계가 아니라 **씨앗**이고, 마스크가 못 주는
    것을 설치 파일이 준다.

    왜 필요한가 (2026-08-21 머스탱 다크호스 실측):
    **리어는 도색 상자의 몫으로 밴드를 앉힐 수 없다.** 리어 아틀라스 구역은
    범퍼 밑면(밑으로 감기는 로어 밸런스)과 데크리드를 같이 담아 상자가 실제
    보이는 면보다 위아래로 넓다 — 옆면 로커와 **같은 물리 높이**(상자 −7~27%)에
    앉힌 밴드가 프로브 사각으로 재도 잉크 0px였고, 상자 몫을 42%까지 올려도
    범퍼 아래 모서리에만 걸렸다. 반면 `carLocator_bumperR`이 가리키는 v(상자
    60%)는 캡처에서 **번호판 위 범퍼 면 한가운데**였다.

    프론트는 이 자를 안 쓴다 — 그 구역은 페시아뿐이라(후드는 `top`이다) 상자
    아래끝이 곧 보이는 아래끝이고, 같은 실측에서 지금 규칙이 스플리터에 정확히
    앉았다. `carLocator_bumperF`는 상자 94%(범퍼 **윗**부분 장착점)라 하부
    밴드의 자가 아니다.

    등록이 없거나 미심쩍으면 None이다 (부르는 쪽이 상자 몫으로 물러난다).
    """
    # **리어에만 답한다.** 프론트에 같은 자를 대면 밴드가 그릴 위로 올라간다
    # (실측: 앞범퍼 로케이터는 상자 94% — 면 위에서 8%였다).
    if surface != "rear":
        return None
    p = locs.get("carLocator_bumperR")
    if reg is None or reg.uncertain or p is None or not reg.has(surface):
        return None
    uv = reg.unit(surface, p)
    return None if uv is None else round(uv[1], 1)


def hood_u(reg: Registration | None,
           locs: dict[str, tuple[float, float, float]]) -> float | None:
    """윗면에서 **후드 안의 한 점** u — `carLocator_hood`.

    이 자는 **경계가 아니라 씨앗**이다. 로케이터로 후드↔그린하우스 경계를 재
    보려고 넷을 견줬는데(대시·후드·시트캠·엔진, 이미 구간이 갈린 106대 대조)
    가장 나은 대시보드도 |오차| 중앙 28 cm · p90 115 cm라 경계로 못 쓴다.
    반면 **후드 로케이터가 후드 구간 안에 드는 비율은 93%**다 (전수 347면) —
    그래서 구간을
    가르는 일은 마스크에 맡기고, "**어느 구간이 후드인가**"만 이 점이 고른다
    (`compose.hood_index`).

    고치는 것: 앞 스플리터·노즈 벤트가 낸 짧은 구간이 첫 구간이 되는 차들이다
    (F50 70유닛 · AZ-1 66 · 네베라 64). 그 조각을 후드로 보면 너무 좁아
    (`HOOD_MIN_FRAC`) 후드 인물이 통째로 버려진다.
    """
    p = locs.get("carLocator_hood")
    if reg is None or reg.uncertain or p is None or not reg.has("top"):
        return None
    uv = reg.unit("top", p)
    return None if uv is None else round(uv[0], 1)


# ---------- 휠아치 어림 (마스크에 아치 구멍이 없는 차) ----------
# 설치본 636대 중 99대는 옆면 마스크 아래 경계에 아치 파임이 아예 없거나 하나뿐
# 이라 (CRX Mugen은 0) 아치 검출·로케이터 등록이 통째로 실패한다. 그 차들은
# 배치판·미리보기에 휠 자리가 안 보여서 인물이 뒷휠 위에 앉는 사고가 난다
# (2026-08-23 CRX 실차: 우측면 인물의 2/3가 휠 구멍에 삼켜졌다).
#
# 휠 **중심**은 로케이터가 정확히 안다 — 모르는 것은 배율 K뿐이다. 범퍼 로케이터
# z 간격을 옆면 상자 폭에 대면 K가 나오는데, 범퍼 장착점이 차 끝보다 안쪽이라
# 계통 오차가 있다. 아치가 검출되는 536대에서 잰 보정: K_아치/K_범퍼 중앙값
# 1.048, 보정 후 자리 오차 중앙 0.11 m · p90 0.36 m. 어림이므로 **자리 표시와
# 배치 회피에만** 쓰고 등록(부품 자리·밴드 높이)에는 안 쓴다.
BUMPER_K_CORR = 1.048
ARCH_R_EST = 0.37          # 아치 반지름 어림 (m) — 등록 536대 중앙값 0.359


def arch_fallback(media: str, maps: dict[str, SurfaceMap]
                  ) -> dict[str, dict]:
    """마스크가 아치를 모르는 차의 휠 자리 어림 — {면: {wheels, vc, k}}.

    `wheels`는 `seam.SideGeom.wheels` 꼴 ((u, 반지름) 앞뒤)이고 `vc`는 각 아치
    원 중심의 v (사이드실 + K·휠중심높이). 로케이터가 없으면 빈 dict.
    """
    locs = read(media)
    wf, wr = locs.get("carLocator_wheelLF"), locs.get("carLocator_wheelLR")
    bf, br = locs.get("carLocator_bumperF"), locs.get("carLocator_bumperR")
    if not (wf and wr and bf and br):
        return {}
    span = abs(bf[2] - br[2])
    if span < 1.0 or abs(wf[2] - wr[2]) < 0.5:
        return {}
    zmid = (bf[2] + br[2]) / 2
    out: dict[str, dict] = {}
    for name, sign in (("side_left", -1.0), ("side_right", 1.0)):
        sm = maps.get(name)
        if sm is None or sm.mask.size <= 1:
            continue
        k = (sm.paint[2] - sm.paint[0]) / span * BUMPER_K_CORR
        sill = _lowest_paint(sm)
        r = k * ARCH_R_EST
        got = sorted(((sign * k * (w[2] - zmid), sill + k * w[1]) for w in (wf, wr)),
                     key=lambda t: t[0])
        out[name] = {"wheels": tuple((round(u, 1), round(r, 1)) for u, _vc in got),
                     "vc": tuple(round(vc, 1) for _u, vc in got),
                     "k": round(k, 2)}
    return out


@lru_cache(maxsize=8)
def for_car(media: str) -> tuple[Registration | None, tuple]:
    """차 하나의 (등록, 로케이터 항목들) — 캐시. 로케이터가 없으면 (None, ()).

    **차 메시가 있으면 등록이 그 위에 선다** (`game.fsgeom`): 배율·원점을 적합
    대신 닫힌 식으로 받으므로 휠아치를 못 찾는 차(설치본 99대)도 자리가 선다.
    """
    from . import carfiles, fsgeom

    locs = read(media)
    if not locs:
        return None, ()
    try:
        maps = carfiles.surface_maps(media)
    except (OSError, KeyError, ValueError):
        return None, ()
    reg = register(maps, locs, media=media)
    geom = fsgeom.for_car(media)
    if geom is not None:
        side = geom.get("side_left") or geom.get("side_right")
        k = side.units_per_m[0] if side is not None else None
        if reg is None and k:
            # 아치를 못 찾아 등록이 아예 안 서던 차 — 메시가 자리를 준다
            reg = Registration(media=media, k=round(k, 2),
                               note="차 메시 (아치 적합 없음)")
        if reg is not None:
            reg.geom = geom
            if k:
                reg.k = round(k, 2)
            # 아치 적합의 미심쩍음은 이제 `unit`에 안 실린다 — 메시가 낸다
            reg.uncertain = False
            reg.note = (reg.note + " · " if reg.note else "") + "배율·자리는 차 메시"
    return reg, tuple(sorted(locs.items()))
