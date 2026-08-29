r"""차체 **옆면의 뼈대** — 벨트라인·사이드실·루프라인·휠아치와 유리 이음새.

## 왜 필요한가

설치 마스크의 `side_left`는 **차 옆면 실루엣 통째**다 — 그린하우스(유리 자리)가
구멍이 아니라 흰 판으로 들어 있다. 그런데 게임이 그 면에 실제로 그리는 것은
**벨트라인 아래뿐**이다. 유리는 `window_left`라는 **별도 면**이 쥔다.

근거 (2026-08-20 실측): 인게임 프로브(색 차분) 마스크와 설치 마스크를 같은
에디터 유닛 격자에 겹치면, 프로브가 칠한 영역의 윗선이 정확히 벨트라인이고
그린하우스 전체가 "설치에는 있는데 안 칠해지는" 영역으로 남는다 (인테그라·
시빅·실비아·RX-7·에보·RS6·챌린저·NSX — 15개 면).

그래서 옆면에 도안을 앉힐 때 설치 마스크를 그대로 쓰면 **인물이 벨트라인에서
뚝 잘린다** (줄리아 12호차 실차 캡처의 증상). 이 모듈이 벨트라인을 찾아 옆면을
**차체 부분만**으로 자르고, 그 위(머리)는 유리 면으로 넘길 좌표 변환을 준다.

## 벨트라인은 폭 프로필의 무릎이다

옆면 실루엣을 가로로 훑으면 `width(v)`는 지붕에서 0으로 시작해 그린하우스를
따라 완만히 늘다가, **벨트라인에서 후드·트렁크가 한꺼번에 들어와 급히 뛴다**.
그래서 위에서 내려오며 `width(v)`가 몸통폭(95백분위)의 `BELT_FRAC`를 처음 넘는
자리가 벨트라인이다. 프로브 정답 15면 대조: 평균오차 2.2%, 최대 6.7%
(상자 높이 대비). 임계 0.75~0.85 어디를 써도 3% 안이라 평평한 최적이다.

폭 프로필이 무너지는 차(오픈톱·픽업 등 그린하우스가 없거나 지붕이 곧 몸통인
차)를 위해 상수 비율(`BELT_CONST` = 정답 중앙값)을 하한·상한으로 물린다.

## 유리 이음새

`window_left`는 제 유닛계를 쓴다 (아틀라스 할당이 면마다 달라 배율이 다르다 —
같은 차에서 옆면 191유닛/m, 도어 유리는 그 1.4배쯤). 두 면을 잇는 변환은
**그린하우스 상자 ↔ 유리 잉크 상자**로 맞춘다: 세로는 벨트라인이 유리 아랫선,
가로는 캐빈 폭이 유리 폭이다. 이음새가 벨트라인이므로 세로 기준점은 상자
가운데가 아니라 **아랫선**이다.

(화면 px를 거쳐 잇는 길은 막혔다 — 탭마다 카메라가 다르다. 프로브 warp를
합성해 보면 유리가 문짝 한가운데로 떨어진다. 2026-08-20 실측.)
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .surface import SurfaceMap

# 벨트라인 = 위에서 내려오며 width(v)가 몸통폭의 이 몫을 처음 넘는 자리.
# 프로브 15면 대조에서 0.75~0.85가 전부 평균 3% 안이고 0.80이 최소였다.
BELT_FRAC = 0.80
# 그린하우스가 안 읽히는 차를 위한 상수 비율 (도색 상자 아래에서부터의 몫).
# 정답 15면의 중앙값 0.626, 범위 0.597~0.702.
BELT_CONST = 0.626
BELT_LO, BELT_HI = 0.50, 0.80          # 검출값을 이 범위로 물린다
# 그린하우스 덩어리로 인정할 최소 크기 (u는 차 길이의 몫, v는 상자 높이의 몫).
CABIN_MIN_U, CABIN_MIN_V = 0.18, 0.06
# 휠아치로 인정할 최소 파임 (상자 높이의 몫)
ARCH_MIN_DEPTH = 0.10
# ---- 씨앗 재훑기 (`wheel_arches(sep=…)`) ----
# 휠 로케이터가 **아치 사이 간격**을 알려 줄 때 쓰는 값들. 문턱 하나로 훑는
# 길은 아치가 얕은 차에서 통째로 실패한다 (실측 101대) — 간격을 알면 "이 자리에
# 아치가 있어야 한다"를 물을 수 있으므로 문턱을 훨씬 낮춰도 헛것을 안 문다.
ARCH_SEED_DEPTH = 0.030   # 씨앗이 있을 때의 파임 하한 (상자 높이의 몫)
ARCH_SEED_WIN = 0.05      # 예상 자리 둘레 이 몫(차 길이) 안에서 봉우리를 찾는다


def _grid(smap: SurfaceMap, step: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """면 마스크를 **유닛 등간격 격자**로 다시 뜬다 (행 0 = 위). (mask, us, vs)."""
    u0, v0, u1, v1 = smap.paint
    w = max(8, int(round((u1 - u0) / step)))
    h = max(8, int(round((v1 - v0) / step)))
    us = np.linspace(u0, u1, w)
    vs = np.linspace(v1, v0, h)
    m = smap.mask
    mh, mw = m.shape
    xi = np.clip(np.round((us - u0) / max(1e-6, u1 - u0) * (mw - 1)).astype(int), 0, mw - 1)
    yi = np.clip(np.round((v1 - vs) / max(1e-6, v1 - v0) * (mh - 1)).astype(int), 0, mh - 1)
    return m[np.ix_(yi, xi)], us, vs


@dataclass
class SideGeom:
    """옆면 하나의 뼈대 — 전부 **면 유닛**이다."""

    belt: float                     # 벨트라인 v (유리와의 이음새)
    sill: float                     # 사이드실 v (도색 마스크 최하단)
    roof: float                     # 루프라인 v (그린하우스 꼭대기)
    cabin: tuple[float, float]      # 그린하우스 u 범위
    wheels: tuple[tuple[float, float], ...] = ()   # (중심 u, 반지름) 앞→뒤 아님, u순
    note: str = ""

    @property
    def body_height(self) -> float:
        """차체(유리 제외) 높이 — 로커에서 벨트라인까지."""
        return self.belt - self.sill

    @property
    def glass_height(self) -> float:
        return self.roof - self.belt


def side_geom(smap: SurfaceMap, arch_sep: float | None = None,
              seed_arches: bool = False) -> SideGeom:
    """옆면 설치 마스크 → 뼈대. 프로브 없이 기하만으로 선다.

    `arch_sep`(면 유닛)을 주면 휠아치를 **그 간격의 짝**으로 찾는다 — 설치
    파일의 휠 로케이터가 아는 것이다 (`game.locators`). `seed_arches`면 문턱
    길을 아예 안 보고 씨앗 길로만 찾는다 (부르는 쪽이 두 답을 견줄 때).
    """
    m, us, vs = _grid(smap)
    u0, v0, u1, v1 = smap.paint
    hv = v1 - v0
    wid = m.sum(1).astype(float)
    wmax = float(np.percentile(wid, 95)) or 1.0
    belt = float(vs[-1])
    for i in range(len(vs)):                       # 행 0 = 위
        if wid[i] >= BELT_FRAC * wmax:
            belt = float(vs[i])
            break
    # 상수 비율로 물린다 — 폭 프로필이 무너지는 차 (오픈톱·픽업)
    lo, hi = v0 + BELT_LO * hv, v0 + BELT_HI * hv
    note = ""
    if not (lo <= belt <= hi):
        note = f"폭 무릎 {belt:.0f}이 범위 밖 — 상수 {BELT_CONST}"
        belt = v0 + BELT_CONST * hv
    # 사이드실 = 마스크 최하단 (짧은 돌기는 버린다)
    run = m.sum(1)
    solid = np.where(run >= 0.06 * m.shape[1])[0]
    sill = float(vs[solid[-1]]) if len(solid) else float(v0)
    # 그린하우스 = 벨트라인 위 가장 큰 덩어리 (윙·안테나를 뺀다)
    above = (vs >= belt)
    cabin = (u0, u1)
    roof = float(v1)
    if above.any():
        sub = m[above].astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(sub, 8)
        best, ba = 0, 0
        for i in range(1, n):
            if st[i, cv2.CC_STAT_AREA] > ba:
                ba, best = st[i, cv2.CC_STAT_AREA], i
        if best:
            x, y = st[best, cv2.CC_STAT_LEFT], st[best, cv2.CC_STAT_TOP]
            w_, h_ = st[best, cv2.CC_STAT_WIDTH], st[best, cv2.CC_STAT_HEIGHT]
            vsa = vs[above]
            if (w_ / len(us) >= CABIN_MIN_U) and (h_ * abs(vsa[0] - vsa[-1])
                                                  / max(1, len(vsa) - 1) / hv >= CABIN_MIN_V):
                cabin = (float(us[x]), float(us[min(len(us) - 1, x + w_ - 1)]))
                roof = float(vsa[y])
            else:
                note = (note + " · " if note else "") + "그린하우스 덩어리가 작다"
                roof = float(vsa[y]) if best else roof
    return SideGeom(belt=round(belt, 1), sill=round(sill, 1), roof=round(roof, 1),
                    cabin=(round(cabin[0], 1), round(cabin[1], 1)),
                    wheels=wheel_arches(smap, sill, sep=arch_sep,
                                        seeded=seed_arches), note=note)


def arch_depth(smap: SurfaceMap, sill: float | None = None
               ) -> tuple[np.ndarray, np.ndarray, float]:
    """아래 경계의 **파임 프로파일** — (us, depth, 상자 높이). 유닛.

    `depth[j]`는 그 열의 도색 최하단이 사이드실보다 얼마나 위인가다. 휠아치는
    이 프로파일의 봉우리로 나타난다 (마스크에 구멍이 뚫려 있으므로).
    """
    m, us, vs = _grid(smap)
    _h, w = m.shape
    bot = np.full(w, np.nan)
    for j in range(w):
        col = np.where(m[:, j])[0]
        if len(col):
            bot[j] = vs[col[-1]]
    hv = float(smap.paint[3] - smap.paint[1])
    if np.all(np.isnan(bot)):
        return us, np.zeros(w), hv
    base = sill if sill is not None else float(np.nanmin(bot))
    return us, np.nan_to_num(bot - base, nan=0.0), hv


def _runs(deep: np.ndarray, us: np.ndarray, min_frac: float
          ) -> list[tuple[float, float]]:
    """참인 구간들 → (중심 u, 반폭). 너무 짧은 구간은 버린다."""
    w = len(deep)
    out: list[tuple[float, float]] = []
    j = 0
    while j < w:
        if not deep[j]:
            j += 1
            continue
        k = j
        while k < w and deep[k]:
            k += 1
        if (k - j) >= min_frac * w:
            out.append((float((us[j] + us[k - 1]) / 2),
                        float((us[k - 1] - us[j]) / 2)))
        j = k
    return out


def _seeded_pair(us: np.ndarray, d: np.ndarray, hv: float, sep: float
                 ) -> tuple[tuple[float, float], ...]:
    """**간격을 아는 채로** 아치 짝을 고른다 — 문턱이 아니라 자리로 찾는다.

    프로파일을 `sep`만큼 밀어 겹치면(둘 다 파여 있어야 하므로 **작은 쪽**을
    쓴다) 두 아치가 나란히 서는 자리에서 봉우리가 선다. 그 자리가 앞 아치이고
    반대쪽이 뒷 아치다. 문턱은 씨앗이 있으니 훨씬 낮게(3%) 둘 수 있다 — 자리가
    이미 못 박혀 있어 얕은 파임 하나를 아치로 오인할 여지가 없다.

    반지름은 그 봉우리의 **반높이 폭**이다 (원형 아치의 현). 못 찾으면 빈 짝.
    """
    w = len(us)
    if w < 8 or sep <= 0:
        return ()
    upp = (float(us[-1]) - float(us[0])) / max(1, w - 1)     # 열당 유닛
    shift = int(round(sep / max(1e-6, upp)))
    if not (2 <= shift < w - 2):
        return ()
    pair = np.minimum(d[:w - shift], d[shift:])              # 둘 다 파인 정도
    if not len(pair):
        return ()
    # 봉우리 자리 — 예상 간격이 몇 % 어긋날 수 있으므로 창을 두고 고른다
    i0 = int(np.argmax(pair))
    if pair[i0] < ARCH_SEED_DEPTH * hv:
        return ()
    win = max(2, int(ARCH_SEED_WIN * w))
    out: list[tuple[float, float]] = []
    for i in (i0, i0 + shift):
        lo, hi = max(0, i - win), min(w, i + win + 1)
        j = lo + int(np.argmax(d[lo:hi]))
        half = 0.5 * float(d[j])
        a = j
        while a > 0 and d[a - 1] >= half:
            a -= 1
        b = j
        while b < w - 1 and d[b + 1] >= half:
            b += 1
        out.append((float(us[j]), max(upp, float(us[b] - us[a]) / 2)))
    return tuple(sorted(out, key=lambda t: t[0]))


def wheel_arches(smap: SurfaceMap, sill: float | None = None,
                 sep: float | None = None, seeded: bool = False
                 ) -> tuple[tuple[float, float], ...]:
    """휠아치 (중심 u, 반지름) — 아래 경계가 크게 파인 구간.

    다리가 휠아치를 가로질러 잘리는 것은 문법상 허용이지만, 인물의 좌우 자리를
    '뒷문~뒷펜더'로 잡으려면 뒷바퀴가 어디인지 알아야 한다.

    **`sep`(예상 아치 간격, 면 유닛)이 있으면 문턱 대신 자리로 찾는다** —
    설치 파일의 휠 로케이터가 아는 것이다 (`game.locators`). 문턱 하나로 훑는
    길은 부품 등록의 병목이었다 (실측: 못 하는 101대가 전부 아치를 0~1개만
    찾은 차다 — DB11의 파임은 상자 높이의 9%로 문턱 10% 바로 아래이고,
    Vantage는 앞 아치만 파여 있다).

    **어느 길이 맞는지는 여기서 안 정한다.** 씨앗 길은 얕은 봉우리도 아치로
    보므로 잘 잡히던 차에서 오히려 나쁜 답을 낼 수 있다 (실측: 그냥 갈아
    끼웠더니 등록은 535→561대로 늘었는데 믿을 수 있는 것이 484→476대로
    줄었다 — R8·리갈 GNX가 새로 물러났다). 그래서 부르는 쪽(`locators.register`)이
    **두 답을 다 받아 좌우 일치로 고른다** — 좌우 대조는 어느 답을 짓는 데도
    안 쓴 자라 심판이 될 수 있다.
    """
    us, d, hv = arch_depth(smap, sill)
    if not len(d) or d.max() <= 0:
        return ()
    if seeded and sep:
        return _seeded_pair(us, d, hv, sep)
    got = _runs(d > ARCH_MIN_DEPTH * hv, us, 0.04)
    got.sort(key=lambda t: -t[1])
    return tuple(sorted(got[:2], key=lambda t: t[0]))


def body_map(smap: SurfaceMap, geom: SideGeom | None = None) -> SurfaceMap:
    """옆면 지도를 **차체(벨트라인 아래)만**으로 자른 사본.

    이걸 배치 계산에 주면 내접 상자·마스크 판정이 전부 유리를 안 넘본다 —
    꾸밈 그룹·산포 모티프가 유리에 떨어지는 사고도 여기서 같이 막힌다.
    """
    geom = geom or side_geom(smap)
    m = smap.mask.copy()
    u0, v0, u1, v1 = smap.paint
    mh = m.shape[0]
    cut = int(round((v1 - geom.belt) / max(1e-6, v1 - v0) * (mh - 1)))
    m[:max(0, cut)] = False
    if not m.any():
        return smap
    rows = np.where(m.any(1))[0]
    top_v = v1 - rows[0] / max(1, mh - 1) * (v1 - v0)
    out = SurfaceMap(name=smap.name, index=smap.index, origin_px=smap.origin_px,
                     px_per_unit=smap.px_per_unit,
                     paint=(u0, v0, u1, round(top_v, 1)),
                     fill=round(float(m[rows[0]:].mean()), 4),
                     mask=m[rows[0]:], cap=smap.cap, uncertain=smap.uncertain,
                     note=(smap.note + "+body") if smap.note else "body",
                     warp=smap.warp)
    return out


def punch_arches(smap: SurfaceMap, wheels: tuple[tuple[float, float], ...],
                 vcs: tuple[float, ...]) -> SurfaceMap:
    """지도에 **휠아치 원반 구멍**을 뚫은 사본 — 마스크가 아치를 모르는 차용.

    설치 마스크에 아치 파임이 있는 차는 구멍이 이미 마스크에 있어 배치판·
    미리보기·배치 계산이 전부 자연히 피한다. 이 자는 그 구멍을 로케이터
    어림(`game.locators.arch_fallback`)으로 같은 꼴로 만들어 준다 — 어림이므로
    반지름은 보수적(중앙값)이고, 자리 오차는 중앙 0.11 m다.
    """
    if not wheels or smap.mask.size <= 1:
        return smap
    m = smap.mask.copy()
    mh, mw = m.shape
    u0, v0, u1, v1 = smap.paint
    us = u0 + (np.arange(mw) + 0.5) / mw * (u1 - u0)
    vs = v1 - (np.arange(mh) + 0.5) / mh * (v1 - v0)
    U, V = np.meshgrid(us, vs)
    for (uc, r), vc in zip(wheels, vcs):
        m[(U - uc) ** 2 + (V - vc) ** 2 <= r * r] = False
    if not m.any():
        return smap
    return SurfaceMap(name=smap.name, index=smap.index, origin_px=smap.origin_px,
                      px_per_unit=smap.px_per_unit, paint=smap.paint,
                      fill=round(float(m.mean()), 4), mask=m, cap=smap.cap,
                      uncertain=smap.uncertain,
                      note=(smap.note + "+arch") if smap.note else "arch",
                      warp=smap.warp)


# ---------- 윗면 유리 (2026-08-21 인게임 프로브 실측) ----------
# 옆면 그린하우스와 같은 이야기가 윗면에도 있다: **설치 마스크는 앞유리·뒷유리
# 위까지 통째로 칠해져 있는데 게임은 거기에 차체 비닐을 안 그린다.** 프로브
# 마스크(색 차분)와 설치 마스크를 같은 에디터 유닛 격자에 겹치면 프로브가 칠한
# 자리가 **후드 · 지붕 · 데크 셋으로 갈리고** 그 사이 두 구간이 유리다
# (실비아·시빅·에보·데몬·두랑고·911 여섯 대. 프로브가 설치 마스크 안에 든 몫은
# 0.97~1.00이라 두 지도의 정렬은 믿을 수 있다).
#
# **가운데 구간만 유리로 친다.** 프로브는 코끝·꽁무니에서도 아무것도 못 잡는데
# (실비아 앞 84유닛·911 뒤 68유닛), 그 자리는 유리가 아니라 면이 카메라에서
# 달아난 자리다 — 게임이 안 그린다는 증거가 아니라 그 카메라에서 안 보인다는
# 뜻이고, 사람은 차를 다른 각도에서도 본다. 그쪽은 껍질의 정면도가 부드럽게
# 알린다 (`game.hull`). 유리는 **재질**이라 어느 각도에서도 안 그려진다.
TOP_BAND = (0.325, 0.675)   # 중앙 밴드 (compose.top_segments와 같은 띠)
TOP_SOLID = 0.55            # 설치가 "칠한다"고 보는 행 비율
TOP_PROBE_MAX = 0.25        # 프로브가 "안 칠했다"고 보는 행 비율
TOP_RUN_MIN = 0.06          # 후드·지붕·데크로 인정할 최소 폭 (면 길이의 몫)
TOP_GLASS_MIN = 0.03        # 유리로 인정할 최소 폭 (면 길이의 몫)
# 프로브 마스크가 설치 마스크 안에 이만큼 안 들어오면 두 지도가 안 맞는 것이다
# (실측: 소프트탑 미아타 0.77 — 프로브 상자가 어긋났다. 나머지는 0.97 이상).
TOP_FIT_MIN = 0.90


def _cols(sm: SurfaceMap, us: np.ndarray, vs: np.ndarray) -> np.ndarray:
    """면 마스크를 (us × vs) 격자에서 뽑는다 — 행 0 = vs[0]."""
    m = sm.mask
    mh, mw = m.shape
    u0, v0, u1, v1 = sm.paint
    xi = np.clip(np.round((us - u0) / max(1e-6, u1 - u0) * (mw - 1)), 0, mw - 1)
    yi = np.clip(np.round((v1 - vs) / max(1e-6, v1 - v0) * (mh - 1)), 0, mh - 1)
    out = m[np.ix_(yi.astype(int), xi.astype(int))]
    return (out & (us[None, :] >= u0) & (us[None, :] <= u1)
            & (vs[:, None] >= v0) & (vs[:, None] <= v1))


def top_glass(install: SurfaceMap, probe: SurfaceMap | None
              ) -> list[tuple[float, float]]:
    """윗면에서 **게임이 안 그리는 유리 띠** (윗면 u 구간). 실측이 없으면 빈 목록.

    프로브가 칠한 덩어리들 **사이**만 유리로 본다 (앞뒤 끝의 여백은 유리가 아니다
    — 위 상수 설명). 프로브 지도가 없거나, 의심스럽거나(`uncertain`), 설치
    마스크와 안 겹치면 빈 목록이고 부르는 쪽은 지금까지처럼 윗면을 통째로 쓴다.
    """
    if probe is None or probe.uncertain or probe.mask.size <= 1 \
            or install.mask.size <= 1:
        return []
    u0, v0, u1, v1 = install.paint
    if u1 - u0 <= 1.0 or v1 - v0 <= 1.0:
        return []
    W = 480
    us = np.linspace(u0, u1, W)
    H = max(60, int(round(W * (v1 - v0) / (u1 - u0))))
    vs = np.linspace(v1, v0, H)
    A, B = _cols(install, us, vs), _cols(probe, us, vs)
    if not B.any() or float((A & B).sum()) / max(1, B.sum()) < TOP_FIT_MIN:
        return []
    band = slice(int(H * TOP_BAND[0]), int(H * TOP_BAND[1]))
    a_c, b_c = A[band].mean(0), B[band].mean(0)
    upp = (u1 - u0) / (W - 1)
    solid = _runs_of(b_c >= TOP_PROBE_MAX, TOP_RUN_MIN * (u1 - u0) / upp)
    out: list[tuple[float, float]] = []
    for (_, a), (b, _) in zip(solid, solid[1:]):
        if (b - a) * upp < TOP_GLASS_MIN * (u1 - u0):
            continue
        if float((a_c[a:b] > TOP_SOLID).mean()) < 0.5:
            continue                              # 설치도 안 칠하는 구간 (구멍)
        out.append((round(float(us[a]), 1), round(float(us[b - 1]), 1)))
    return out


def _runs_of(flag: np.ndarray, min_len: float) -> list[tuple[int, int]]:
    """참인 구간 [시작, 끝) 색인 — 짧은 것은 버린다."""
    out: list[tuple[int, int]] = []
    start = None
    for i, s in enumerate(flag):
        if s and start is None:
            start = i
        elif not s and start is not None:
            if i - start >= min_len:
                out.append((start, i))
            start = None
    if start is not None and len(flag) - start >= min_len:
        out.append((start, len(flag)))
    return out


def top_body(smap: SurfaceMap, bands: list[tuple[float, float]]) -> SurfaceMap:
    """윗면 지도에서 **유리 띠를 지운 사본** — 옆면의 `body_map`과 같은 자.

    배치 계산·마스크 판정이 유리를 안 넘보게 한다. 상자는 그대로다 (유리는 면
    안쪽의 구멍이라 끝선이 안 바뀐다).
    """
    if not bands or smap.mask.size <= 1:
        return smap
    m = smap.mask.copy()
    mh, mw = m.shape
    u0, _v0, u1, _v1 = smap.paint
    us = u0 + (np.arange(mw) + 0.5) / mw * (u1 - u0)
    for a, b in bands:
        m[:, (us >= min(a, b)) & (us <= max(a, b))] = False
    if not m.any():
        return smap
    return SurfaceMap(name=smap.name, index=smap.index, origin_px=smap.origin_px,
                      px_per_unit=smap.px_per_unit, paint=smap.paint,
                      fill=round(float(m.mean()), 4), mask=m, cap=smap.cap,
                      uncertain=smap.uncertain,
                      note=(smap.note + "+glass") if smap.note else "glass",
                      warp=smap.warp)


@dataclass
class Seam:
    """옆면 유닛 → 도어 유리 유닛. 이음새(벨트라인)를 기준선으로 잇는다."""

    su: float                # 가로 배율 (유리 유닛 / 옆면 유닛)
    sv: float                # 세로 배율
    cu: float                # 옆면 캐빈 중심 u
    gu: float                # 유리 잉크 중심 u
    belt: float              # 옆면 벨트라인 v
    gv0: float               # 유리 잉크 아랫선 v
    iou: float = 0.0

    def to_window(self, u, v):
        """옆면 유닛 (u, v) → 유리 유닛. 배열도 받는다."""
        return ((np.asarray(u, float) - self.cu) * self.su + self.gu,
                (np.asarray(v, float) - self.belt) * self.sv + self.gv0)

    @property
    def scale(self) -> float:
        """그룹 배치용 **등방 배율** — 그룹은 균등 스케일만 되기 때문이다.

        가로를 따른다: 이음새에서 머리 폭이 어긋나면 바로 보이지만, 세로는
        유리 위쪽 여백으로 흡수된다 (모자라면 지붕에 안 닿을 뿐이다).
        """
        return self.su


def _ink_box(mask: np.ndarray, box: tuple[float, float, float, float]):
    r = np.where(mask.any(1))[0]
    c = np.where(mask.any(0))[0]
    if not len(r) or not len(c):
        return None
    u0, v0, u1, v1 = box
    h, w = mask.shape
    return (u0 + c[0] / max(1, w - 1) * (u1 - u0),
            v1 - r[-1] / max(1, h - 1) * (v1 - v0),
            u0 + c[-1] / max(1, w - 1) * (u1 - u0),
            v1 - r[0] / max(1, h - 1) * (v1 - v0))


# 다듬기 — 상자 맞춤 씨앗 둘레를 훑는다. 배율은 곱, 이동은 캐빈 크기의 몫.
REFINE_S = (0.80, 0.90, 1.00, 1.10, 1.22)
REFINE_D = (-0.12, -0.06, 0.0, 0.06, 0.12)
# 그린하우스 **밖으로 나간** 유리에 물리는 벌점. 그린하우스는 유리의 상위집합이라
# 겹침만 최대화하면 유리가 판금(패스트백 엔진 뚜껑)까지 늘어난다.
REFINE_PENALTY = 1.6


def seam(side: SurfaceMap, win: SurfaceMap, geom: SideGeom | None = None,
         refine: bool = True) -> Seam | None:
    """옆면 ↔ 도어 유리 이음새 변환. 그린하우스가 안 읽히면 None.

    씨앗은 상자 맞춤이다 — 그린하우스 상자(벨트라인~루프라인 × 캐빈 폭)에 유리
    잉크 상자를 얹는다. 세로 기준은 **아랫선끼리** (이음새가 벨트라인이다).

    그 다음 `refine`이 국소 탐색으로 다듬는다. 상자 맞춤만 쓰면 패스트백에서
    유리가 뒤로 늘어난다 (911 카레라 RS: 그린하우스 뒤쪽 절반이 유리가 아니라
    엔진 뚜껑 판금인데 상자는 그걸 구분 못 한다).
    """
    geom = geom or side_geom(side)
    if geom.glass_height <= 1.0 or geom.cabin[1] - geom.cabin[0] <= 1.0:
        return None
    gb = _ink_box(win.mask, win.paint)
    if gb is None:
        return None
    cw = geom.cabin[1] - geom.cabin[0]
    s = Seam(su=round((gb[2] - gb[0]) / cw, 4),
             sv=round((gb[3] - gb[1]) / geom.glass_height, 4),
             cu=round((geom.cabin[0] + geom.cabin[1]) / 2, 1),
             gu=round((gb[0] + gb[2]) / 2, 1),
             belt=geom.belt, gv0=round(gb[1], 1))
    if refine:
        s = _refine(side, win, geom, s, cw)
    s.iou = round(_seam_iou(side, win, geom, s), 3)
    return s


def _refine(side: SurfaceMap, win: SurfaceMap, geom: SideGeom, seed: Seam,
            cw: float) -> Seam:
    """씨앗 둘레 국소 탐색 — 그린하우스 안에 들어가면서 가장 큰 유리를 고른다."""
    m, us, vs = _grid(side, step=3.0)
    above = vs >= geom.belt - 2.0
    if not above.any() or not m[above].any():
        return seed
    G = m[above]
    U, V = np.meshgrid(us, vs[above])
    g, a0, b0, a1, b1 = win.mask, *win.paint
    gh_, gw_ = g.shape
    best, bj = seed, -1e18
    for fu in REFINE_S:
        for fv in REFINE_S:
            for du in REFINE_D:
                for dv in REFINE_D:
                    c = Seam(su=seed.su * fu, sv=seed.sv * fv,
                             cu=seed.cu + du * cw, gu=seed.gu,
                             belt=seed.belt + dv * geom.glass_height, gv0=seed.gv0)
                    wu, wv = c.to_window(U, V)
                    xi = np.round((wu - a0) / max(1e-6, a1 - a0) * (gw_ - 1)).astype(int)
                    yi = np.round((b1 - wv) / max(1e-6, b1 - b0) * (gh_ - 1)).astype(int)
                    ok = (xi >= 0) & (xi < gw_) & (yi >= 0) & (yi < gh_)
                    gm = np.zeros_like(G)
                    gm[ok] = g[np.clip(yi, 0, gh_ - 1), np.clip(xi, 0, gw_ - 1)][ok]
                    j = float((gm & G).sum()) - REFINE_PENALTY * float((gm & ~G).sum())
                    if j > bj:
                        bj, best = j, c
    return Seam(su=round(best.su, 4), sv=round(best.sv, 4),
                cu=round(best.cu, 1), gu=round(best.gu, 1),
                belt=round(best.belt, 1), gv0=round(best.gv0, 1))


def _seam_iou(side: SurfaceMap, win: SurfaceMap, geom: SideGeom, s: Seam) -> float:
    """변환의 자기 점검 — 옮긴 유리가 그린하우스와 얼마나 겹치나.

    그린하우스는 필러·루프레일까지 품으므로 1.0이 나올 수 없다 (유리는 그
    안쪽이다). 0.55쯤이 정상이고, 그보다 낮으면 캐빈 검출이 샌 것이다.
    """
    m, us, vs = _grid(side)
    above = vs >= geom.belt
    if not above.any():
        return 0.0
    gh = m[above]
    U, V = np.meshgrid(us, vs[above])
    wu, wv = s.to_window(U, V)
    g = win.mask
    a0, b0, a1, b1 = win.paint
    gh_, gw_ = g.shape
    xi = np.round((wu - a0) / max(1e-6, a1 - a0) * (gw_ - 1)).astype(int)
    yi = np.round((b1 - wv) / max(1e-6, b1 - b0) * (gh_ - 1)).astype(int)
    ok = (xi >= 0) & (xi < gw_) & (yi >= 0) & (yi < gh_)
    gm = np.zeros_like(gh)
    gm[ok] = g[np.clip(yi, 0, gh_ - 1), np.clip(xi, 0, gw_ - 1)][ok]
    inter = float((gm & gh).sum())
    union = float((gm | gh).sum())
    return inter / max(1.0, union)


# ---------- 인물 자리 (발=사이드실, 몸=차체 밴드를 가로로 채운다, 자리=문짝) ----------
# 인물 중심을 뒷바퀴 중심에서 **앞으로** 이만큼 민다 (차 길이의 몫). 좁은 인물
# (버스트)이 문짝 안에 넉넉히 들어갈 때만 실제로 쓰인다 — 넓은 인물은 아래 문짝
# 클램프가 도로 앞으로 끌어온다.
PERSON_AHEAD_OF_REAR = 0.07
# 휠아치가 안 읽힐 때 쓰는 좌우 자리 (앞=0, 뒤=1)
PERSON_BIAS = 0.62

# ---- 인물이 써도 되는 자리의 상한 (2026-08-20 레퍼런스 픽셀 실측) ----
# 레퍼런스 옆면 넉 장을 격자 오버레이로 재서 **인물 잉크 상자 ÷ 휠아치 사이
# 구간(문짝)**을 뽑았다: RIN SHIBUYA 0.79 · ARIS 0.70 · KOTONE 0.84 ·
# EVELYNE 0.62 → 중앙값 0.745. 이것이 "측면을 최대한 쓴다"의 실측 값이다.
PERSON_DOOR_FILL = 0.75
# 휠아치를 못 읽는 면의 폴백 — 도색 상자 폭의 몫 (같은 넉 장의 차 길이 대비
# 실측 0.20~0.42에서, 문짝이 차 길이의 절반쯤이므로 0.75×0.55에 해당한다)
PERSON_SPAN_MAX = 0.42
# **인물 잉크 상자 ÷ 차체 밴드(로커~벨트라인)**. 레퍼런스 실측은 중앙값 1.10
# (RIN 0.76 · ARIS 1.40 · KOTONE 1.12 · EVELYNE 1.08)이지만 그것은 벨트라인 위
# 13%가 **도어 유리 면으로 이어져 있을 때**의 값이다. 우리는 이제 그림을 이웃
# 면에 자동으로 안 잇는다 (사용자 지시 2026-08-27) — 넘긴 몫은 그 자리에서
# 잘리므로 벨트라인을 안 넘는 값으로 앉힌다 (잘린 머리를 내느니 작게 선다).
# 유리까지 쓰고 싶으면 편집기에서 도안을 벨트라인으로 **가르고** 위쪽 반을
# 유리 면에 올린다.
PERSON_BAND_FILL = 0.96


def door_span(geom: SideGeom) -> tuple[float, float] | None:
    """**휠아치 사이** 유닛 구간 (문짝). 아치를 못 찾으면 None.

    아치는 도색 마스크의 구멍이라 그 위의 획은 사라진다. 레퍼런스 넉 장 중
    셋은 인물 상자가 이 구간 **안에** 온전히 들어간다 (EVELYNE만 뒤 아치를
    조금 문다) — 인물의 좌우 예산이 곧 이 구간이다.
    """
    if len(geom.wheels) < 2:
        return None
    (u0, r0), (u1, r1) = geom.wheels[0], geom.wheels[-1]
    lo, hi = u0 + 0.85 * r0, u1 - 0.85 * r1
    return (lo, hi) if hi - lo > 20.0 else None


def person_budget(smap: SurfaceMap, geom: SideGeom) -> tuple[float, float]:
    """인물이 써도 되는 **(폭, 높이)** — 면 유닛. 크기 결정의 유일한 예산이다.

    폭은 문짝(휠아치 사이)의 몫이고 높이는 차체 밴드의 몫이다. 둘 다 위 상수의
    레퍼런스 실측이고, 이 예산 안에서 **가장 커지는 각도**를 부르는 쪽이 푼다
    (`engine.compose.person_pose`).
    """
    u0, _v0, u1, _v1 = smap.paint
    door = door_span(geom)
    w = ((door[1] - door[0]) * PERSON_DOOR_FILL if door is not None
         else (u1 - u0) * PERSON_SPAN_MAX)
    return max(1e-6, w), max(1e-6, geom.body_height * PERSON_BAND_FILL)


def person_span(smap: SurfaceMap, geom: SideGeom, size: tuple[float, float],
                rear_dir: float) -> tuple[float, float, float, float]:
    """정해진 크기 `size`(폭, 높이)의 인물을 옆면 어디에 앉힐지 — 면 유닛 상자.

    **아래는 언제나 사이드실**이다 (레퍼런스 문법: 발·몸 아랫단이 로커에 붙고
    다리가 휠아치를 가로지르며 잘리는 것은 허용). 좌우는 뒷바퀴 조금 앞을
    바라되 **문짝 구간 안으로 클램프**한다 — 눕힌 인물은 폭이 문짝의 3/4이라
    뒤로 치우치면 리어 범퍼에서 잘린다. 레퍼런스의 인물 중심은 차 길이의
    0.41~0.68(중앙값 0.49)로 사실상 문짝 한가운데다.

    `rear_dir`는 이 면에서 +1이면 +u가 차 뒤다.
    """
    u0, _v0, u1, _v1 = smap.paint
    w, h = size
    if geom.wheels:
        rear = max(geom.wheels, key=lambda t: t[0] * rear_dir)
        cx = rear[0] - rear_dir * PERSON_AHEAD_OF_REAR * (u1 - u0)
    else:
        f = PERSON_BIAS if rear_dir > 0 else 1.0 - PERSON_BIAS
        cx = u0 + f * (u1 - u0)
    door = door_span(geom)
    if door is not None:
        lo, hi = door[0] + w / 2, door[1] - w / 2
        cx = (door[0] + door[1]) / 2 if lo > hi else min(max(cx, lo), hi)
    cx = min(max(cx, u0 + w / 2), u1 - w / 2)
    return (cx - w / 2, geom.sill, cx + w / 2, geom.sill + h)
