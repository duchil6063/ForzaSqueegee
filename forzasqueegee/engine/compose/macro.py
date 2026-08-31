"""매크로 기하 — 인물 뒤에 서는 **큰 색면의 어휘**.

## 왜 다시 짓나

옛 어휘는 넷이었다 (`bed`의 slab · plate · wedge · blob). 33판을 구워 보니
그중 셋이 **같은 그림**을 낸다: 전부 `_through`로 프레임을 관통하는 사각이고,
기울기는 길이가 눕혀서(`_tilt_for` — 900유닛 띠는 5°가 한계) 결국 "패널을
가로지르는 거의 수평인 띠" 하나가 된다. 33판의 그림을 늘어놓으면 매크로가
사실상 한 가지다.

그리고 그 한 가지는 **너무 크다**. 관통하는 띠가 인물을 품으려면 폭이 인물
높이만 해야 하고, 그러면 900유닛 × 인물 높이라 판 넓이가 인물의 5.8배(중앙값)·
최대 17.2배가 된다 — 판이 아니라 두 번째 베이스 도색이다.

## 푸는 법 — **어느 변으로 나가나**를 어휘에 넣는다

"판과 띠는 차가 자르는 데까지 간다"는 규칙(2026-08-31 사용자 판정)은 지킨다:
색면은 패널 한가운데서 제 끝을 보이면 안 된다. 그런데 옛 어휘는 그 규칙을
**앞뒤로 나가는 것**으로만 풀었다. 위아래로 나가도 된다 — 벨트라인과 로커가
자르는 가파른 색면은 길이가 밴드 높이/sin θ뿐이라 (60°면 173유닛) 폭을 인물
높이보다 넓게 줘도 인물의 1.6배에 머문다. 인물을 품으면서 패널을 안 덮는
색면이 거기서 나온다.

그래서 여기 어휘는 전부 **뻗어 나가는 방향**으로 정의된다: 어느 각으로 서든
단면 전체가 프레임 밖까지 간다 (`_run`).

## 어휘

    ribbon    관통하는 띠 — 폭·기울기·끝 기울기(전단)·가늘어짐
    stack     나란한 띠 여러 겹 (레이싱 스트라이프 문법)
    blade     한쪽으로 가늘어지는 띠
    chevron   두 띠가 한 점에서 꺾인다 (화살)
    split     선 하나로 가른 반쪽 색면 (투톤)
    corner    프레임 모서리에 앉는 직각 삼각형
    bracket   두 띠가 직각으로 만나는 ㄱ자
    burst     한 점에서 퍼지는 날 여럿 (인물 뒤 광선)
    sweep     휘는 띠 (조각 여럿으로 낸 호)

각 원시형은 **연속 매개변수**를 받는다 (`MacroSpec`) — 축·폭·가늘어짐·끝
기울기·겹 수·벌어짐·곡률. 템플릿이 아니라 매개변수 공간이라 후보가 그 사이를
메울 수 있다 (`design._refine`의 좌표하강이 그 위를 걷는다).

좌표는 **프레임 좌표**(꾸밈 캔버스), y-up이다. 라벨은 옛 판과 같은
`itasha_bed`다 — 예산 사다리·바닥 요소 판정·구성 그래프가 그 이름을 읽는다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from ..catalog import Catalog
from ..model import UNITS_PER_SCALE, Layer


LABEL = "itasha_bed"


# 어휘 이름 — `families.Family.macro`가 이 중에서 고른다. `none`은 **안 세운다**
# (짝 자리에만 쓴다): 큰 색면 하나로 끝나는 구도가 정답인 판이 있고, 그것을
# 후보에 두지 않으면 짝이 없어도 억지로 하나를 세운다.
KINDS = ("ribbon", "stack", "blade", "chevron", "split", "corner", "bracket",
         "burst", "sweep", "none")


# 사각이 프레임 밖으로 얼마나 더 나가나 (프레임 폭 대비) — `bed.OVERSHOOT`과
# 같은 사정이다: 끝이 프레임 선에 딱 걸리면 반올림 한 칸의 본색 실선이 남는다.
OVERSHOOT = 0.04


# 가늘어짐을 사각 하나로 못 낸다 (전단은 평행을 지킨다). 이 값 아래면 사각
# 한 장, 위면 **사각 + 삼각형의 합집합**이다 (같은 색이라 사다리꼴로 읽힌다).
TAPER_MIN = 0.12


# 삼각형 어휘 — 밑변이 로컬 −x, 꼭짓점이 +x다 (`B_21` 실측: (−1,−1)(−1,+1)(+1,0)).
TRI_POINT_X = "B_21"


# 직각 삼각형 — 직각이 (−1,−1)이고 빗변이 그 대각이다 (`A_04`).
TRI_RIGHT = "A_04"


@dataclass(frozen=True)
class MacroSpec:
    """큰 색면 하나의 **매개변수** — 원시형 이름과 연속 값들.

    단위: 각은 도(프레임 x축 기준 반시계), 길이·폭은 프레임 유닛,
    `taper`·`cut`·`curve`는 0~1(또는 ±1) 무차원.
    """

    kind: str
    at: tuple[float, float] = (0.0, 0.0)
    ang: float = 0.0
    width: float = 100.0            # 축 법선 방향 폭
    taper: float = 0.0              # 0 = 평행, 1 = 끝이 뾰족
    cut: float = 0.0                # 끝 기울기 (전단) — ±1이 45°
    count: int = 1                  # 겹·날 수 (stack·burst·sweep)
    gap: float = 0.35               # 겹 사이 (폭 대비)
    spread: float = 40.0            # 벌어짐 (chevron·burst, 도)
    curve: float = 0.0              # 휨 (sweep) — 전체 호의 도
    side: float = 1.0               # 채우는 쪽 (split — 축 법선의 부호)
    role: str = "bed"               # 색 역할 (`roles.RolePalette`의 칸 이름)
    alpha: float = 100.0
    z: int = 0                      # 낮을수록 먼저 (아래) 그린다


def _rect(x: float, y: float, w: float, h: float, rot: float,
          color: tuple[int, int, int], cat: Catalog, alpha: float = 100.0,
          skew: float = 0.0) -> Layer:
    """길이 `w`(로컬 x) × 폭 `h`(로컬 y)의 사각 — `skew`면 평행사변형."""
    return Layer(shape=cat.square, x=x, y=y, sx=w / 2 / UNITS_PER_SCALE,
                 sy=h / 2 / UNITS_PER_SCALE, rot=rot % 360.0, skew=skew,
                 color=color, alpha=alpha, label=LABEL)


def _tri(x: float, y: float, w: float, h: float, rot: float, name: str,
         color: tuple[int, int, int], cat: Catalog, alpha: float = 100.0) -> Layer:
    return Layer(shape=name, x=x, y=y, sx=w / 2 / UNITS_PER_SCALE,
                 sy=h / 2 / UNITS_PER_SCALE, rot=rot % 360.0, color=color,
                 alpha=alpha, label=LABEL)


def _run(frame: tuple[float, float, float, float], cx: float, cy: float,
         d: tuple[float, float], half: float, sign: float) -> float:
    """`(cx, cy)`에서 축 `sign·d`로 **단면 전체가 프레임을 벗어나는** 거리.

    `half`는 축 법선 방향 반폭(전단 몫까지 더한 것)이다. 두 모서리가 다 나가야
    끝이 안 보이므로 늦은 쪽을 쓴다 (`bed._run`과 같은 자다 — 그쪽은 판만
    쓰고 이쪽은 어휘 전부가 쓴다).
    """
    x0, y0, x1, y1 = frame
    pad = OVERSHOOT * (x1 - x0)
    x0, y0, x1, y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
    dx, dy = sign * d[0], sign * d[1]
    nx, ny = -d[1] * half, d[0] * half
    worst = 0.0
    for ox, oy in ((nx, ny), (-nx, -ny)):
        px, py = cx + ox, cy + oy
        t = math.inf
        if abs(dx) > 1e-9:
            t = min(t, ((x1 if dx > 0 else x0) - px) / dx)
        if abs(dy) > 1e-9:
            t = min(t, ((y1 if dy > 0 else y0) - py) / dy)
        worst = max(worst, max(0.0, t))
    return worst


def _dir(ang: float) -> tuple[float, float]:
    r = math.radians(ang)
    return math.cos(r), math.sin(r)


def _band(frame, at, ang, width, cut, taper, color, cat, alpha) -> list[Layer]:
    """**관통하는 띠 한 장** (또는 가늘어질 때 두 장).

    끝이 프레임 밖이라 그 끝을 자르는 것은 도형이 아니라 차다. `cut`(전단)은
    양 끝을 비스듬히 썰고, `taper`는 앞쪽을 좁힌다.
    """
    d = _dir(ang)
    # 전단은 로컬 x를 y에 비례해 민다 — 축 법선 반폭이 그만큼 늘어난 것처럼
    # 프레임을 벗어나야 끝이 안 보인다
    half = width / 2
    fwd = _run(frame, at[0], at[1], d, half, 1.0) + abs(cut) * half
    back = _run(frame, at[0], at[1], d, half, -1.0) + abs(cut) * half
    ln = fwd + back
    cx = at[0] + d[0] * (fwd - back) / 2
    cy = at[1] + d[1] * (fwd - back) / 2
    if taper < TAPER_MIN:
        return [_rect(cx, cy, ln, width, ang, color, cat, alpha, skew=cut)]
    # 가늘어짐 — 좁은 폭의 사각 + 넓은 끝에서 좁은 끝으로 가는 삼각형의 **합집합**.
    # 전단은 평행을 지키므로 사각 하나로는 사다리꼴이 안 나온다.
    narrow = width * (1.0 - taper)
    out = [_rect(cx, cy, ln, narrow, ang, color, cat, alpha, skew=cut)]
    # 삼각형은 밑변이 로컬 −x(뒤쪽)이고 꼭짓점이 +x(앞쪽)다
    out.append(_tri(cx, cy, ln, width, ang, TRI_POINT_X, color, cat, alpha))
    return out


def build(spec: MacroSpec, frame: tuple[float, float, float, float],
          color: tuple[int, int, int], cat: Catalog) -> list[Layer]:
    """명세 하나 → 레이어들 (프레임 좌표). 어느 어휘든 프레임 밖까지 나간다."""
    k = spec.kind
    if k == "none":
        return []
    at, ang, w = spec.at, spec.ang, max(4.0, spec.width)
    a = spec.alpha
    if k == "ribbon":
        return _band(frame, at, ang, w, spec.cut, spec.taper, color, cat, a)
    if k == "blade":
        return _band(frame, at, ang, w, spec.cut, max(0.35, spec.taper), color, cat, a)
    if k == "stack":
        # 나란한 띠 여러 겹 — 폭이 겹마다 줄고 사이가 벌어진다 (레이싱 스트라이프).
        # **`width`는 겹 하나가 아니라 묶음 전체의 폭이다** — 겹마다 폭을 주면
        # 세 겹이 밴드를 통째로 덮는다 (실측: 그렇게 해서 판 넓이가 인물의
        # 7.5배가 됐다).
        out: list[Layer] = []
        n = max(1, spec.count)
        wts = [0.55 ** i for i in range(n)]
        gaps = spec.gap * (n - 1)
        unit = w / max(1e-6, sum(wts) + gaps)
        nx, ny = -_dir(ang)[1], _dir(ang)[0]
        off = -w / 2
        for i, wt in enumerate(wts):
            wi = unit * wt
            c = (at[0] + nx * (off + wi / 2), at[1] + ny * (off + wi / 2))
            out += _band(frame, c, ang, wi, spec.cut, spec.taper, color, cat, a)
            off += wi + unit * spec.gap
        return out
    if k == "chevron":
        # 두 띠가 한 점에서 꺾인다 — 각각 제 쪽으로 프레임을 나간다
        out = []
        for s in (-1.0, 1.0):
            aa = ang + s * spec.spread / 2
            d = _dir(aa)
            half = w / 2
            fwd = _run(frame, at[0], at[1], d, half, 1.0)
            cx = at[0] + d[0] * fwd / 2
            cy = at[1] + d[1] * fwd / 2
            out.append(_rect(cx, cy, fwd, w, aa, color, cat, a, skew=spec.cut))
        return out
    if k == "split":
        # 선 하나로 가른 색면 — 선의 `side` 쪽을 **프레임 끝까지 전부** 채운다.
        # 위·아래·한쪽 끝, 세 변을 다 차가 자르므로 보이는 끝이 하나도 없다.
        #
        # 깊이를 매개변수로 두면 안 된다: 기운 선에서 끝까지의 구역은 띠가
        # 아니라 사다리꼴이라, 폭을 정해 놓으면 먼 쪽 변이 패널 **안**에 선다
        # (실측: 45°에서 절반만 채워 세로 선이 남았다). 덮어야 할 깊이는
        # **프레임 모서리에서 되읽는다** — 채우는 쪽 모서리 중 가장 먼 것까지.
        # 넓이를 정하는 것은 `width`가 아니라 선의 자리(`at`)다 (`plan`).
        d = _dir(ang)
        nx, ny = -d[1] * spec.side, d[0] * spec.side
        x0, y0, x1, y1 = frame
        pad = OVERSHOOT * (x1 - x0)
        depth = 0.0
        for px, py in ((x0 - pad, y0 - pad), (x1 + pad, y0 - pad),
                       (x1 + pad, y1 + pad), (x0 - pad, y1 + pad)):
            depth = max(depth, (px - at[0]) * nx + (py - at[1]) * ny)
        if depth <= 1e-6:
            return []
        c = (at[0] + nx * depth / 2, at[1] + ny * depth / 2)
        fwd = _run(frame, c[0], c[1], d, depth / 2, 1.0)
        ln = fwd + _run(frame, c[0], c[1], d, depth / 2, -1.0)
        cc = (c[0] + d[0] * (fwd - ln / 2), c[1] + d[1] * (fwd - ln / 2))
        return [_rect(cc[0], cc[1], ln, depth, ang, color, cat, a)]
    if k == "corner":
        # 프레임 **끝을 비스듬히 자른** 직각 삼각형 — 세로 다리가 밴드를 온전히
        # 건너고(위아래를 차가 자른다) 가로 다리가 프레임 끝에 붙는다. 빗변
        # 하나만 패널 안에 있다.
        #
        # 다리 둘을 같게 두면 안 된다: 밴드 높이의 몇 배짜리 삼각형이 되어
        # 밴드 안에서는 **모서리가 안 보이는 큰 덩어리**로 읽힌다 (실측: 판
        # 위쪽에 네모난 검은 덩이가 떴다).
        x0, y0, x1, y1 = frame
        pad = OVERSHOOT * (x1 - x0)
        sx = 1.0 if at[0] >= (x0 + x1) / 2 else -1.0
        sy = 1.0 if at[1] >= (y0 + y1) / 2 else -1.0
        H = (y1 - y0) + 2 * pad
        Lx = min(max(w, 1e-6), 2.0 * MACRO_AREA_MAX * (x1 - x0))
        ex = (x1 + pad if sx > 0 else x0 - pad)
        ey = (y1 + pad if sy > 0 else y0 - pad)
        # A_04의 직각은 (−1,−1)이고 다리가 로컬 +x·+y다 — 회전 0/180은 로컬 x가
        # 세계 x, 90/270은 세계 y다. 다리를 그에 맞춰 배분한다.
        rot = {(-1.0, -1.0): 0.0, (1.0, -1.0): 90.0,
               (1.0, 1.0): 180.0, (-1.0, 1.0): 270.0}[(sx, sy)]
        lw, lh = (Lx, H) if rot in (0.0, 180.0) else (H, Lx)
        return [_tri(ex - sx * Lx / 2, ey - sy * H / 2, lw, lh, rot,
                     TRI_RIGHT, color, cat, a)]
    if k == "bracket":
        # ㄱ자 — 두 띠가 직각으로 만나 각자 프레임을 나간다
        out = []
        for aa in (ang, ang + 90.0):
            d = _dir(aa)
            half = w / 2
            fwd = _run(frame, at[0], at[1], d, half, 1.0)
            out.append(_rect(at[0] + d[0] * fwd / 2, at[1] + d[1] * fwd / 2,
                             fwd, w, aa, color, cat, a))
        return out
    if k == "burst":
        # 한 점에서 퍼지는 날 — 인물 뒤 광선 (바깥으로 넓어진다)
        out = []
        n = max(2, spec.count)
        for i in range(n):
            t = (i / (n - 1) - 0.5) if n > 1 else 0.0
            aa = ang + t * spec.spread
            d = _dir(aa)
            half = w / 2
            fwd = _run(frame, at[0], at[1], d, half, 1.0)
            if fwd <= 1e-6:
                continue
            # 밑변이 바깥(넓다), 꼭짓점이 안쪽(점) — 삼각형을 뒤집어 놓는다
            out.append(_tri(at[0] + d[0] * fwd / 2, at[1] + d[1] * fwd / 2,
                            fwd, w, aa + 180.0, TRI_POINT_X, color, cat, a))
        return out
    if k == "sweep":
        # 휘는 띠 — 각을 조금씩 돌린 조각들을 겹쳐 호를 낸다. 조각 하나하나가
        # 프레임을 나가지는 않지만 **양 끝 조각**은 나간다 (호의 끝이 안 보인다).
        out = []
        n = max(3, spec.count)
        step = spec.curve / max(1, n - 1)
        d0 = _dir(ang - spec.curve / 2)
        # 호의 반지름 — 주어진 휨으로 프레임을 건너는 길이
        chord = math.hypot(frame[2] - frame[0], frame[3] - frame[1])
        r = chord / max(1e-6, 2 * math.sin(math.radians(max(1.0, abs(spec.curve)) / 2)))
        seg = chord / n * 1.35
        px = at[0] - d0[0] * chord / 2
        py = at[1] - d0[1] * chord / 2
        aa = ang - spec.curve / 2
        for _i in range(n):
            d = _dir(aa)
            out.append(_rect(px + d[0] * seg / 2, py + d[1] * seg / 2,
                             seg, w, aa, color, cat, a))
            px += d[0] * seg * 0.86
            py += d[1] * seg * 0.86
            aa += step
        _ = r
        return out
    return _band(frame, at, ang, w, spec.cut, spec.taper, color, cat, a)


def macro_layers(specs: tuple[MacroSpec, ...],
                 frame: tuple[float, float, float, float],
                 colors: dict[str, tuple[int, int, int]],
                 cat: Catalog) -> list[Layer]:
    """명세 여럿 → 레이어 (z가 낮은 것부터 = 아래부터)."""
    out: list[Layer] = []
    for sp in sorted(specs, key=lambda s: (s.z, s.kind, s.ang)):
        out += build(sp, frame, colors.get(sp.role, colors["bed"]), cat)
    return out


def n_layers(spec: MacroSpec) -> int:
    """이 명세가 몇 장인가 — 예산을 미리 재는 자리."""
    k, t = spec.kind, spec.taper
    per = 2 if t >= TAPER_MIN else 1
    if k in ("ribbon",):
        return per
    if k == "blade":
        return 2
    if k == "stack":
        return max(1, spec.count) * per
    if k == "chevron":
        return 2
    if k in ("split", "corner"):
        return 1
    if k == "bracket":
        return 2
    if k == "burst":
        return max(2, spec.count)
    if k == "sweep":
        return max(3, spec.count)
    return per


def counter_angle(ang: float, spread: float = 62.0, sign: float = 1.0) -> float:
    """가로지르는 짝의 각 — `graph.deliberate`가 바라는 40~90° 밖에 안 둔다."""
    return ang + sign * spread


def specs_from(spec: MacroSpec, **kw) -> MacroSpec:
    return replace(spec, **kw)


# ---- 계획 — 인물에서 매개변수를 뽑는다 -------------------------------------

# 얕은 띠(앞뒤로 나가는 것)가 수평에서 기울 수 있는 상한 (도). `bed.BED_TILT_MAX`와
# 같은 사정이다 — 900유닛 띠를 포즈 축대로 세우면 판이 세로로 선다.
FLAT_MAX = 22.0


# 얕은 띠의 세로 뻗음(길이 × sin)이 밴드의 이 몫을 넘지 않게 눕힌다 (`bed.BED_RISE_MAX`).
FLAT_RISE = 0.55


# 가파른 색면(위아래로 나가는 것)의 각 구간 (도). 이 아래면 얕은 띠와 안 갈리고,
# 90°면 밴드를 곧게 가른다.
STEEP_RANGE = (44.0, 90.0)


# 가파른 색면의 폭 — 인물 폭의 몫 (level이 그 사이를 걷는다).
STEEP_W = (1.25, 2.35)


# 얕은 띠의 폭 — 인물 높이의 몫.
FLAT_W = (0.24, 0.78)


# 가로지르는 짝의 폭 — 인물 크기의 몫. 짝은 **판이 아니라 선**이다.
#
# 슬래브로 두면 인물 옆에 제2의 판이 떠서 "떨어져 나온 덩어리"로 읽힌다 (실측
# P3d~P3e: 33판 중 여덟에 인물 오른쪽 위로 검은 사변형이 떴고, 겹치게 옮겨도
# 그대로였다 — 문제는 자리가 아니라 **굵기와 색**이었다). 레퍼런스의 가로지르는
# 요소는 예외 없이 가는 액센트 선이거나 확실한 투톤 면이고, 그 사이의 중간
# 굵기 판은 없다.
COUNTER_W = (0.055, 0.13)


# 어느 어휘가 **가파른가** (위아래로 나간다). 나머지는 앞뒤로 나가는 얕은 것이다.
STEEP_KINDS = ("split", "corner", "bracket", "burst", "sweep", "chevron")


# 큰 색면 **하나**가 프레임에서 차지할 수 있는 넓이의 상한 (프레임 대비).
#
# 이 상한이 없으면 어휘를 늘린 것이 곧 "더 크게 덮는 법을 늘린 것"이 된다:
# 반평면 `split`과 겹마다 폭을 받던 `stack`이 정확히 그랬다 (판 넓이가 인물의
# 5.1배 → 7.5배 · 꾸밈 잉크 0.26 → 0.43 · 여백 0.22 → 0.15). 색면은 구도지
# 두 번째 베이스 도색이 아니다.
MACRO_AREA_MAX = 0.34


def _strip_area(frame, ang: float, w: float) -> float:
    """각 `ang`·폭 `w`의 띠가 프레임 안에서 덮는 넓이 (어림).

    띠는 먼저 닿는 변으로 나간다 — 가로로 나가면 길이가 `W/|cos|`, 세로면
    `H/|sin|`이고 둘 중 짧은 쪽이 실제 길이다.
    """
    W = frame[2] - frame[0]
    H = frame[3] - frame[1]
    r = math.radians(ang)
    ln = min(W / max(1e-3, abs(math.cos(r))), H / max(1e-3, abs(math.sin(r))))
    return min(w * ln, W * H)


def _cap_width(frame, ang: float, w: float, share: float = MACRO_AREA_MAX) -> float:
    """넓이 상한을 지키는 폭 — `_strip_area`가 `share`를 넘지 않게 줄인다."""
    W = frame[2] - frame[0]
    H = frame[3] - frame[1]
    a = _strip_area(frame, ang, w)
    if a <= share * W * H or a <= 1e-9:
        return w
    return w * share * W * H / a


def _fold(ang: float, lo: float, hi: float) -> float:
    """각(도)을 180° 주기로 접어 `[lo, hi]` 안의 **가장 가까운** 값으로.

    포즈 축은 부호가 없다 (장축은 양방향이 같다) — 그래서 접어도 뜻이 안 변한다.
    """
    a = ang % 180.0
    if a > 90.0:
        a -= 180.0                                # (−90, 90]
    s = 1.0 if a >= 0 else -1.0
    m = min(max(abs(a), lo), hi)
    return s * m


def _flat_tilt(ang: float, length: float, band: float) -> float:
    """얕은 띠가 가질 수 있는 기울기 — 세로 뻗음을 밴드 몫으로 묶는다."""
    lim = math.degrees(math.asin(max(0.0, min(1.0, FLAT_RISE * band
                                              / max(1e-6, length)))))
    lim = min(lim, FLAT_MAX)
    return max(-lim, min(lim, ang))


def _lerp(rng: tuple[float, float], t: float) -> float:
    return rng[0] + (rng[1] - rng[0]) * max(0.0, min(1.0, t))


def plan(fld, kinds: tuple[str, str], level: float, *, rocker: bool = False,
         d_rot: float = 0.0, d_y: float = 0.0, d_w: float = 0.0
         ) -> tuple[MacroSpec, ...]:
    """구도 필드 + 어휘 짝 → 큰 색면 명세 (주 색면 · 가로지르는 짝).

    매개변수는 전부 **인물에서** 나온다: 축은 포즈 장축과 흐름의 섞임, 폭은
    인물 크기의 몫, 자리는 시각 중심이다. `d_*`는 좌표하강의 손잡이다
    (`design._refine`) — 0이면 손대기 전과 같다.

    짝의 각은 주 색면에서 `counter_angle`만큼 떨어진다 (`graph.deliberate`가
    바라는 40~90°) — 그래서 판 둘의 관계가 "실수처럼 읽히는 사이"에 안 든다.
    """
    fx0, fy0, fx1, fy1 = fld.frame_box
    band = fy1 - fy0
    span = fx1 - fx0
    ch, cw = fld.char_h, fld.char_w
    vcx, vcy = fld.visual_center
    vcy += d_y * ch
    ax, ay = fld.axis
    fx, fy = fld.flow
    if ax * fx + ay * fy < 0:                     # 축은 부호가 없다 — 흐름 쪽으로
        ax, ay = -ax, -ay
    pose = math.degrees(math.atan2(ay, ax))
    mix = math.degrees(math.atan2(0.45 * ay + 0.55 * fy, 0.45 * ax + 0.55 * fx))
    primary, counter = kinds
    out: list[MacroSpec] = []

    if primary in STEEP_KINDS:
        # 가파른 색면 — **벨트라인과 로커가 자른다.** 길이가 밴드/ sin θ 뿐이라
        # 폭을 인물보다 넓게 줘도 패널을 안 덮는다 (옛 관통 띠의 병목이 이것이다).
        ang = _fold(pose + d_rot, *STEEP_RANGE)
        w = _cap_width(fld.frame_box, ang,
                       _lerp(STEEP_W, level) * cw * (1.0 + d_w))
        at = (vcx + fx * 0.10 * cw, vcy)
        side = 1.0
        if primary == "split":
            # **한쪽 끝에 붙는다** — 흐름 쪽 프레임 변에서 `depth_x`만큼 들어온
            # 자리를 선이 지난다. 그 선의 끝 쪽을 전부 채우므로(빌더가 모서리에서
            # 깊이를 되읽는다) 색면의 세 변을 다 차가 자른다.
            #
            # 넓이는 **선의 자리**가 정한다: 기운 선과 세로 변 사이의 구역은
            # 사다리꼴이고 그 넓이가 정확히 `밴드 × depth_x`다 (평행한 두 변이
            # 가로라 가운데 높이의 깊이가 평균이다). 그래서 depth_x에 상한을
            # 걸면 넓이에 상한을 건 것과 같다.
            end = fx1 if fx >= 0 else fx0
            fsign = 1.0 if fx >= 0 else -1.0
            depth_x = min(abs(end - (vcx - fsign * 0.45 * cw)),
                          MACRO_AREA_MAX * span)
            w = depth_x
            nx = -math.sin(math.radians(ang))
            side = 1.0 if nx * fsign > 0 else -1.0
            at = (end - fsign * depth_x, vcy)
        out.append(MacroSpec(kind=primary, at=at, ang=ang, width=w,
                             taper=0.18 if primary in ("blade", "burst") else 0.0,
                             count=5 if primary == "burst" else 4,
                             spread=52.0 if primary == "burst" else 44.0,
                             curve=34.0 if primary == "sweep" else 0.0,
                             side=side, role="bed", z=0))
        c_ang = _flat_tilt(mix, span, band)
        c_w = _cap_width(fld.frame_box, c_ang,
                         _lerp(COUNTER_W, level) * max(w, ch), share=0.16)
        c_y = _band_y(vcy - 0.16 * ch, c_w, fld, rocker)
        out.append(MacroSpec(kind=counter, at=(vcx, c_y), ang=c_ang, width=c_w,
                             taper=0.30 if counter == "blade" else 0.0,
                             count=3 if counter == "stack" else 1,
                             gap=0.5, role="primary", z=1))
    else:
        # 얕은 띠 — 앞뒤로 나간다 (레퍼런스의 긴 띠). 짝은 **가로지른다**.
        ang = _flat_tilt(mix + d_rot, span, band)
        w = _cap_width(fld.frame_box, ang, _lerp(FLAT_W, level) * ch * (1.0 + d_w))
        cy = _band_y(vcy - 0.06 * ch, w, fld, rocker)
        out.append(MacroSpec(kind=primary, at=(vcx, cy), ang=ang, width=w,
                             taper=0.34 if primary == "blade" else 0.0,
                             cut=0.34 if primary == "ribbon" else 0.0,
                             count=3 if primary == "stack" else 1,
                             gap=0.42, role="bed", z=0))
        sign = 1.0 if fld.flow[0] >= 0 else -1.0
        c_ang = _fold(counter_angle(ang, 62.0, sign), *STEEP_RANGE)
        c_w = _cap_width(fld.frame_box, c_ang,
                         _lerp(COUNTER_W, level) * cw * 2.2, share=0.10)
        # 짝은 주 색면을 **가로지른다** — 띠의 중심선 위에 세워야 둘이 만나
        # 한 그래픽으로 읽힌다. 인물 높이에 두면 띠와 안 만나고 따로 뜬다.
        out.append(MacroSpec(kind=counter, at=(vcx + sign * 0.55 * cw, cy),
                             ang=c_ang, width=c_w,
                             taper=0.30 if counter in ("blade", "burst") else 0.0,
                             count=4 if counter in ("burst", "sweep") else 1,
                             spread=46.0, curve=30.0 if counter == "sweep" else 0.0,
                             role="primary", z=1))
    return tuple(out)


def _band_y(cy: float, h: float, fld, rocker: bool) -> float:
    """얕은 띠의 중심 높이 — 로커 위 ~ 벨트라인 안 (`bed._band_y`와 같은 자)."""
    from .bands import ROCKER_FRAC
    fx0, fy0, fx1, fy1 = fld.frame_box
    floor = fy0 + (ROCKER_FRAC * (fy1 - fy0) if rocker else 0.0)
    if fy1 - floor <= h:
        return (floor + fy1) / 2
    return max(floor + h / 2, min(fy1 - h / 2, cy))
