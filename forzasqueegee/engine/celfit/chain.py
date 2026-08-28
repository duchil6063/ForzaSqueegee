"""획 사슬의 **이음매** — 한 획을 여러 장으로 그을 때 마디끼리 어떻게 만나나.

한 획이 도형 한 장에 안 담기면 사슬로 간다 (`stroke._fit_segments`·
`candidates.dp_segments`). 그런데 마디는 **각자 제 채점만 보고** 자리를 잡는다
(`scoring._descend`) — 이웃 마디가 어디서 끝났는지를 아무도 안 본다. 그래서
같은 획인데 마디 경계에서 옆으로 어긋나고(층계) 방향이 꺾인다. 실측(표준 10장)
한 획 안 이웃 마디의 접선 꺾임이 중앙 28°·p90 69°였다 — 사람이 한 획을 나눠
그으면 그 자리가 안 보이는데, 여기서는 그 자리가 곧 "자동 벡터화 티"다.

이 모듈은 사슬을 **하나로 본다**: 마디를 경로 순으로 세우고, 이음마다

    틈  = 앞 마디 끝점과 뒤 마디 시작점의 거리 (px)
    꺾임 = 앞 마디 끝 접선과 뒤 마디 시작 접선의 각 (도)

를 재 그 둘을 줄이는 방향으로 마디를 민다. 미는 축·스텝은 배치의 하강과 같은
게임 격자이고(`Layer.quantized`), 채점 점수가 허용치보다 더 떨어지는 이동은
기각한다 — 이음을 맞추려고 획이 제 선에서 벗어나면 안 된다.

**끝점·접선은 서술자의 중심선을 실제 변환으로 옮겨 읽는다** (`descriptor`) —
계측(`linemetrics._placed_axis`)과 같은 식이라 "재는 꺾임 = 고치는 꺾임"이다.
중심선이 없는 도형(채움 어휘·덩어리)은 사슬에서 빠진다.
"""

from __future__ import annotations

import os

import numpy as np

from ..catalog import Catalog
from ..model import UNITS_PER_SCALE, Layer
from .descriptor import descriptors, placed_widths

# 틈 1px의 값 (px 점수) — 틈은 이음 보수가 도형 한 장으로 문다. 한 장이
# `_MIN_GAIN`(6px)이므로 몇 px 틈이 그만한 값이면 충분하다.
_W_GAP = float(os.environ.get("FS_CHAIN_GAP", 2.0))
# 꺾임 1도의 값 — 획 폭에 비례한다 (굵은 획일수록 같은 각이 크게 보인다).
_W_KINK = float(os.environ.get("FS_CHAIN_KINK", 0.18))
# 이 안의 틈은 안 문다 — 양자화 한 칸(0.5유닛)은 어차피 못 없앤다.
_GAP_TOL = 1.0
# 점수를 이만큼까지는 내줘도 된다 (px, 마디당). 이음이 값이 있는 만큼만 —
# 0이면 하강이 이미 국소 최적이라 아무것도 안 움직인다.
_TOL = float(os.environ.get("FS_CHAIN_TOL", 3.0))
# 획이 제 경로 끝까지 못 간 몫의 값 — 이음 벌점과 같은 저울이되 조금 약하다
# (끝은 다른 획이 이미 그었을 수도 있다)
_W_REACH = float(os.environ.get("FS_CHAIN_REACH", 1.0))
_PASSES = int(os.environ.get("FS_CHAIN_PASSES", 1))
# **놓는 동안** 이음을 맞출까 (`anchor_pen`) — 끄면 다 놓은 뒤 `polish`만 돈다.
_ON_DESCEND = os.environ.get("FS_JOINT_DESCEND", "1") != "0"


def on() -> bool:
    """사슬 이음 정리를 쓸까 (`FS_CHAIN_JOINT=0`이면 종전 동작)."""
    from . import ablation

    return ablation.joint()


def placed_line(cat: Catalog, lay: Layer, upp: float, w: int, h: int):
    """레이어의 **놓인 중심선** — (끝점 2개 px, 그 자리 접선 2개) 또는 None.

    식은 `geometry._poly_px`와 같다 (같은 회전·비등방 스케일). 계측
    (`linemetrics._placed_axis`)이 방향만 읽는 자리를 여기서는 끝점까지 읽는다.
    """
    return line_of(descriptors(cat).get(lay.shape), lay, upp, w, h)


def line_of(d, lay: Layer, upp: float, w: int, h: int):
    """서술자를 **이미 찾아 둔** 자리용 (`placed_line`의 기하 절반).

    좌표하강 안에서는 도형이 안 바뀌고 변환만 움직인다. 그런데
    `descriptors(cat)`는 부를 때마다 카탈로그의 도달 도형을 훑어 키를 짓는다 —
    하강 스텝마다 부르면 한 장이 분 단위로 늘어진다 (실측). 찾기는 한 번,
    기하는 스텝마다.
    """
    if d is None:
        return None
    if not d.stroke_ok or len(d.center) < 3:
        return _bar_line(d, lay, upp, w, h)
    th = np.radians(lay.rot)
    c, s = np.cos(th), np.sin(th)
    p = (d.center * np.array([lay.sx, lay.sy], np.float64)) \
        @ np.array([[c, s], [-s, c]], np.float64)
    p += np.array([lay.x, lay.y], np.float64)
    px = np.stack([p[:, 0] / upp + w / 2.0, h / 2.0 - p[:, 1] / upp], axis=1)
    if len(px) < 3:
        return None
    k = max(1, len(px) // 8)
    t0 = px[k] - px[0]
    t1 = px[-1] - px[-1 - k]
    n0, n1 = float(np.hypot(*t0)), float(np.hypot(*t1))
    if n0 < 1e-9 or n1 < 1e-9:
        return None
    return px[0], px[-1], t0 / n0, t1 / n1


def _bar_line(d, lay: Layer, upp: float, w: int, h: int):
    """중심선이 없는 도형(막대·둥근사각)의 **놓인 장축** — 끝점 2개와 접선 2개.

    `descriptor`가 중심선을 내는 것은 세선화가 한 가닥으로 떨어지는 도형뿐이다
    (`stroke_ok`). 둥근사각(A_22)은 상자가 정사각이라 거기서 빠지는데, 획
    사슬의 마디로는 가장 많이 쓰인다 — 그 자리를 비워 두면 사슬 태반이
    이음 정리·계측 밖으로 나간다. 상자가 아는 것으로 대신한다: **놓인 뒤
    긴 쪽**이 획 방향이고 그 반길이가 끝점이다 (`geometry._layer`가 폭을
    짧은 축에 싣는 것과 같은 규약).
    """
    ex = abs(lay.sx) * d.ext_x
    ey = abs(lay.sy) * d.ext_y
    if max(ex, ey) < 1e-9:
        return None
    th = np.radians(lay.rot)
    c, s_ = np.cos(th), np.sin(th)
    v = np.array([ex, 0.0]) if ex >= ey else np.array([0.0, ey])
    v = np.array([c * v[0] - s_ * v[1], s_ * v[0] + c * v[1]])
    o = np.array([lay.x, lay.y], np.float64)

    def px(q):
        return np.array([q[0] / upp + w / 2.0, h / 2.0 - q[1] / upp])

    a, b = px(o - v), px(o + v)
    t = b - a
    n = float(np.hypot(*t))
    if n < 1e-9:
        return None
    t = t / n
    return a, b, t, t


def _order(cat: Catalog, layers: list[Layer], path_g: np.ndarray,
           upp: float, w: int, h: int):
    """마디를 **경로 순**으로 세운다 — [(레이어 인덱스, 시작점, 끝점, 시작 접선,
    끝 접선)]. 레이어 순서가 곧 경로 순서는 아니다 (재귀 분할·겨루기).

    경로 위 호길이로 정렬하고, 각 마디의 두 끝 중 호길이가 작은 쪽을 시작으로
    삼는다 — 도형의 로컬 방향은 아핀 맞춤이 뒤집어 놓을 수 있다.
    """
    if len(path_g) < 2:
        return []
    d = np.concatenate([[0.0], np.cumsum(np.hypot(
        *np.diff(path_g.astype(np.float64), axis=0).T))])

    def arc(pt: np.ndarray) -> float:
        i = int(np.argmin((path_g[:, 1] - pt[0]) ** 2
                          + (path_g[:, 0] - pt[1]) ** 2))
        return float(d[i])

    out = []
    for i, lay in enumerate(layers):
        got = placed_line(cat, lay, upp, w, h)
        if got is None:
            return []                      # 중심선 없는 도형이 섞였다 — 손 뗀다
        a, b, ta, tb = got
        sa, sb = arc(a), arc(b)
        if sa <= sb:
            out.append((i, 0.5 * (sa + sb), a, b, ta, tb))
        else:
            out.append((i, 0.5 * (sa + sb), b, a, -tb, -ta))
    out.sort(key=lambda e: e[1])
    return [(e[0], e[2], e[3], e[4], e[5]) for e in out]


def _joint_pen(geo, wpx: float, reach=None) -> float:
    """사슬 전체의 이음 벌점 (px 점수) — 틈 + 꺾임 (+ 획 양끝 못 미침).

    `geo`는 마디마다 (시작점, 끝점, 시작 접선, 끝 접선)이고 경로 순이다.
    `reach`를 주면 (경로 시작점, 경로 끝점)이다 — 사슬이 제 경로의 끝까지 안
    가면 그만큼 문다. 그 못 미친 자리가 곧 이음 보수가 도형으로 메우던 틈이다
    (실측 표준 10장, 이음 보수가 획 도형의 7~19%).
    """
    kw = _W_KINK * max(wpx, 1.0)
    pen = 0.0
    for (_, e0, _, t0), (s1, _, t1, _) in zip(geo, geo[1:]):
        gap = float(np.hypot(*(s1 - e0)))
        pen += _W_GAP * max(0.0, gap - _GAP_TOL)
        cos = float(np.clip(np.dot(t0, t1), -1.0, 1.0))
        pen += kw * float(np.degrees(np.arccos(cos)))
    if reach is not None and geo:
        for pt, end in ((geo[0][0], reach[0]), (geo[-1][1], reach[1])):
            pen += _W_REACH * max(0.0, float(np.hypot(*(pt - end))) - _GAP_TOL)
    return pen


def anchor(path_g: np.ndarray, k: int = 3):
    """경로 조각의 **양끝과 그 접선** — (P0, P1, T0, T1), 전부 px (x, y).

    `path_g`는 전장 좌표 (y, x)다. T0는 조각 안쪽을 향하고 T1은 바깥을 향한다
    (`placed_line`이 내는 ta·tb와 같은 규약).
    """
    if len(path_g) < 2:
        return None
    xy = np.stack([path_g[:, 1], path_g[:, 0]], axis=1).astype(np.float64)
    k = max(1, min(k, len(xy) - 1))
    t0, t1 = xy[k] - xy[0], xy[-1] - xy[-1 - k]
    n0, n1 = float(np.hypot(*t0)), float(np.hypot(*t1))
    if n0 < 1e-9 or n1 < 1e-9:
        return None
    return xy[0], xy[-1], t0 / n0, t1 / n1


def anchor_pen(desc, lay: Layer, upp: float, w: int, h: int,
               anc, wpx: float, bmax: float = 0.0) -> float:
    """이 마디가 제 경로 조각의 **끝과 방향**에서 벗어난 값 (px 점수).

    저울은 사슬 이음 정리와 **같다** (`_W_REACH`·`_W_KINK`·`_GAP_TOL`) — 재는
    양이 같기 때문이다. 다른 것은 언제 묻느냐뿐이다: `polish`는 다 놓은 뒤
    밀어서 고치고, 이쪽은 **놓는 동안** 조종한다. 놓을 때 자리를 제대로
    잡으면 밀어서 고칠 것이 그만큼 준다.

    이웃 마디를 안 본다. 두 마디가 각각 제 경로 접선에 맞으면 이음은 저절로
    매끈하므로(`stroke._tang_pen` 문서), 이웃 대신 **경로**를 보는 것이
    재귀 분할·겨루기가 경로를 어떻게 쪼개도 흔들리지 않는 자다.

    `bmax`가 0보다 크면 **배부름 상한**이다 — 넘는 자리는 `inf`를 돌려 하강이
    아예 못 가게 한다 (`steer` 문서).
    """
    if anc is None:
        return 0.0
    # 배부름 상한 — 넘으면 그 자리는 아예 못 선다 (`steer` 문서). 벌점이
    # 아니라 거부라 저울이 안 늘어난다
    if bmax > 0.0 and bulge_of(desc, lay) > bmax:
        return float("inf")
    got = line_of(desc, lay, upp, w, h)
    if got is None:
        return 0.0
    a, b, ta, tb = got
    P0, P1, T0, T1 = anc
    # 도형의 로컬 방향은 아핀 맞춤이 뒤집어 놓을 수 있다 — 가까운 쪽으로 맞춘다
    if (np.hypot(*(a - P0)) + np.hypot(*(b - P1))
            > np.hypot(*(b - P0)) + np.hypot(*(a - P1))):
        a, b, ta, tb = b, a, -tb, -ta
    # 허용 오차는 **배치가 이미 쓰는 그 여유**다 — 마디는 이웃과 겹치라고
    # 반폭만큼 길게 놓인다 (`stroke._fit_segments`). 그 겹침을 벌하면 조종 항이
    # 배치 규칙과 싸운다. 양자 한 칸(`_GAP_TOL`)이 바닥이다
    tol = max(_GAP_TOL, 0.5 * max(wpx, 1.0))
    pen = _W_REACH * (max(0.0, float(np.hypot(*(a - P0))) - tol)
                      + max(0.0, float(np.hypot(*(b - P1))) - tol))
    kw = _W_KINK * max(wpx, 1.0)
    for u, v in ((ta, T0), (tb, T1)):
        cos = float(np.clip(np.dot(u, v), -1.0, 1.0))
        pen += kw * float(np.degrees(np.arccos(cos)))
    return pen


def bulge_of(desc, lay: Layer) -> float:
    """놓인 뒤의 **최대 폭 / 제 중앙 폭** — `stroke._bulge_ratio`와 같은 닫힌 식.

    거기서는 어휘 표(`_FORMS`)의 색인으로 묻고 여기서는 서술자로 묻는다 —
    같은 `placed_widths`라 두 수가 같다. 중심선이 없는 도형(둥근사각 계열)은
    폭이 상수라 1.0이다.
    """
    if desc is None or not desc.stroke_ok or len(desc.center) < 5:
        return 1.0
    w, _mid, length = placed_widths(desc.center, desc.halfw, lay.sx, lay.sy)
    if length <= 1e-9 or len(w) < 5:
        return 1.0
    med = float(np.median(w))
    return float(w.max()) / med if med > 1e-9 else 1.0


def steer(cat: Catalog, sc, path_g: np.ndarray, wpx: float, w: int, h: int,
          lay: Layer):
    """`scoring._descend`에 넘길 조종 항 — 축이 꺼져 있으면 None.

    경로 조각 하나에 한 번 지어 마디 하나의 하강 내내 쓴다: 앵커도 서술자도
    고정이고 움직이는 것은 변환뿐이다 (`line_of` 문서).

    **배부름 상한을 함께 들고 간다.** 조종 항은 도형의 양끝을 경로 끝에
    맞추려 하는데, 그 수단에 비등방 스케일이 들어 있다 — 늘리다 보면 몸통이
    부푼다 (실측 표준 10장: 배부름 p90 1.53 → 1.64). 어휘 선택은 이미 그
    자를 갖고 있으므로(`stroke._STROKE_BULGE`) 새 벌점을 세우지 않고 **그
    게이트를 하강까지 민다**: 하강은 시작 도형보다 더 부풀 수 없다. 시작이
    이미 상한을 넘어 있으면 그 값이 상한이라 하강이 막히지 않고 **더 나빠지지만
    않는다** — 게이트가 이미 통과시킨 판단을 하강이 뒤집지 못하게 하는 것이
    이 규칙의 뜻이다.
    """
    if not _ON_DESCEND:
        return None
    anc = anchor(path_g)
    if anc is None:
        return None
    desc = descriptors(cat).get(lay.shape)
    if desc is None:
        return None
    from .stroke import _STROKE_BULGE

    bmax = (max(_STROKE_BULGE, bulge_of(desc, lay))
            if _STROKE_BULGE > 0.0 else 0.0)
    return lambda q: anchor_pen(desc, q, sc.upp, w, h, anc, wpx, bmax)


def polish(layers: list[Layer], sc, cat: Catalog, upp: float,
           path_g: np.ndarray, wpx: float, w: int, h: int) -> int:
    """사슬의 이음을 맞춘다 (제자리 수정) — 움직인 마디 수.

    좌표하강이되 목적함수가 **사슬 전체**다: Σ점수 − 이음 벌점. 축·스텝은
    배치의 하강과 같고(x·y·rot에 길이·폭 축을 더한다 — 틈은 미는 것보다
    늘리는 것으로 닫힌다), 결과는 늘 양자화된 레이어다. 점수만 보는 하강은
    이미 국소 최적이라 이 함수가 움직이는 것은 **이음이 그 손해보다 값이
    있는 자리뿐이다** (`_TOL`).
    """
    if len(layers) < 2 or not on():
        return 0
    ch = _order(cat, layers, path_g, upp, w, h)
    if len(ch) < 2:
        return 0
    dmap = descriptors(cat)                # 하강 안에서 다시 찾지 않는다
    idx = [e[0] for e in ch]
    cur = [layers[i] for i in idx]
    base = [sc.score_val(l) for l in cur]
    orig = list(base)      # 점수 허용치의 기준 — 옮길수록 느슨해지면 안 된다
    geo = [(e[1], e[2], e[3], e[4]) for e in ch]

    reach = (np.array([path_g[0][1], path_g[0][0]], np.float64),
             np.array([path_g[-1][1], path_g[-1][0]], np.float64))

    def obj(sv, gm) -> float:
        return sum(sv) - _joint_pen(gm, wpx, reach)

    best = obj(base, geo)
    if best >= sum(base) - 1e-9:
        return 0                # 이음이 이미 흠 없다 — 밀어 볼 것이 없다
    moved = 0
    ds = 1.0 / UNITS_PER_SCALE
    steps = ((2.0 * upp, 4.0, 2.0 * upp * ds), (1.0 * upp, 2.0, upp * ds),
             (0.5 * upp, 1.0, 0.5 * upp * ds))
    for p in range(_PASSES):
        dxy, drot, dsc = steps[min(p, len(steps) - 1)]
        improved = False
        for k in range(len(cur)):
            for axis, st in (("x", dxy), ("y", dxy), ("rot", drot),
                             ("sx", dsc), ("sy", dsc)):
                hit = False
                for sign in (1.0, -1.0):
                    q = Layer(**{**cur[k].__dict__})
                    v = getattr(q, axis)
                    # 스케일은 부호가 미러라 크기는 절댓값이 는다
                    setattr(q, axis, v + sign * st * (-1.0 if axis in ("sx", "sy")
                                                      and v < 0 else 1.0))
                    if axis in ("sx", "sy") and abs(getattr(q, axis)) < 0.01:
                        continue
                    q = q.quantized()
                    got = line_of(dmap.get(q.shape), q, upp, w, h)
                    if got is None:
                        continue
                    sv = sc.score_val(q)
                    if sv < orig[k] - _TOL:
                        continue
                    a, b, ta, tb = got
                    # 방향은 원래 마디의 시작 접선과 같은 쪽으로 맞춘다
                    if float(np.dot(ta, geo[k][2])) < 0:
                        a, b, ta, tb = b, a, -tb, -ta
                    sv2 = list(base)
                    gm = list(geo)
                    sv2[k] = sv
                    gm[k] = (a, b, ta, tb)
                    val = obj(sv2, gm)
                    if val > best + 1e-6:
                        best, base, geo, cur[k] = val, sv2, gm, q
                        improved = hit = True
                        moved += 1
                        break
                if hit:
                    break
        if not improved:
            break
    for k, i in enumerate(idx):
        layers[i] = cur[k]
    return moved
