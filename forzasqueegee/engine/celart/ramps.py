"""팔레트 **접기** — 다 그린 판의 색을 역할색 한 벌로 되돌린다 (goal §19~§22).

## 왜

셀 노선은 분해에서 32색을 뽑는다. 그런데 **다 그린 판에는 850색이 있다**
(실측 cel-01: 1,701장에 854색). 새는 자리가 둘이다:

    label=ink   858장 · 557색   ← 획 색이 제 발자국의 **평균**이라 획마다 다르다
    label=cel   803장 · 289색   ← 면 색이 소유 px 평균이라 조각마다 흩어진다

사람 판은 옆면 하나가 96색이고 상위 8색이 69%를 덮는다 (작가 17인 중앙값).
그 차이는 "사람이 색을 아껴 쓴다"가 아니라 **같은 역할에 같은 색을 다시
쓴다**는 뜻이다 — 머리 그늘은 어디서나 같은 머리 그늘색이다.

## 무엇을 하나

레이어 색을 **묶는다**. 레이어를 지우지도, 자리를 옮기지도, 도형을 바꾸지도
않는다 — 장수와 기하는 그대로고 `color`만 바뀐다. 그래서 이 패스는 커버리지·
봉인·이음새 같은 기하 불변식을 원리적으로 못 건드린다.

묶는 자는 셋이다.

1. **재현** — 색을 옮긴 값을 그 색이 칠한 넓이로 가중한 Oklab 거리로 센다.
2. **경계 보존** (§21) — 원화에서 눈에 보이는 경계를 이루던 두 색은 **못
   묶는다**. 인접은 잉크 상자가 겹치는 레이어 쌍이고, "보이는"의 자는
   이미 있는 저대비 경계 자와 같은 단위다.
3. **반경** — 어느 색도 제 무리 중심에서 `MAX_MOVE_DE` 넘게 못 멀어진다.
   색 수는 목표가 아니라 그 반경의 **결과**다. 반경 자체는 유도가 아니라
   파레토로 정한다 (`MAX_MOVE_DE`의 표).

## 실측 (cel-01 전 구간 · 저장소의 제 자로)

| 접는 대상 | 색 | top8 | mean_de | imp_error_seen | wrong_far_rate | rmse_src |
|---|---|---|---|---|---|---|
| 안 접음   | 854 | 0.09 | 3.351 | 0.104 | 0.00095 | 11.6 |
| **획만**  | 517 | 0.12 | 3.367 | 0.106 | 0.00095 | 11.6 |
| 획+면     | 280 | 0.19 | 4.753 | 0.464 | 0.00104 | 11.7 |

**획만 접는 것이 기본이다** — 색이 39% 줄면서 저장소의 보이는-오차 자가
안 움직인다 (0.104 → 0.106 · 틀린 픽셀 913 → 913). 면까지 접으면 색은 280까지
가지만 그 자가 4.5배가 된다: 채택 안 한다 (`FS_RAMP_LABELS=ink,cel`로 켤 수는
있다 — 값과 대가를 위 표가 적어 둔다).

## 결정성

병합 순서는 (거리, 색 바이트) 안정 정렬이라 같은 입력이면 같은 표가 나온다.
난수 없음.
"""

from __future__ import annotations

import os

import numpy as np

from ...i18n import msg
from ..catalog import Catalog
from ..model import LayerPlan

# **색 수를 목표로 삼지 않는다** (§22·§34.1). 처음 판은 64·96·128·160 네 예산을
# 짓고 재현 오차가 문턱 안에 드는 가장 작은 것을 골랐다 — 64색까지 접혔고
# 지각 자로도 곱게 보였지만(ΔE00 중앙 1.32 · 경계 소실 0.0%), **저장소의 제
# 자로 재니 회귀였다**: `imp_error_seen` 0.10 → 3.61 · `wrong_far_rate`
# 0.00095 → 0.01639 · `mean_de` 3.35 → 7.42 (cel-01 전 구간 실측).
#
# 그래서 자를 바꿨다. 접는 것은 **가까운 것끼리만**이고, 색 수는 그 결과다:
# 어떤 색도 제자리에서 이만큼 넘게 안 움직인다. 850색의 태반은 획 색이 제
# 발자국 평균이라 생긴 **거의 같은 색**이라 이 반경 안에서도 크게 접힌다.
#
# 반경을 처음에는 **유도**로 잡았다: `celart.marks._MARK_DE = 4.0`(Lab ΔE)
# 아래를 저장소가 이미 "안 보이는 차이"로 치므로 그 4분의 3(3.0)이면 옮긴 색이
# 새로 문턱을 넘지 않는다는 것이었다. **그 유도값은 재 본 수가 아니었다.**
# 재 보니 그 반경이 접기를 신호가 아니라 **자기 자신으로** 막고 있었다:
# 다 접은 판에서 이웃 색까지의
# 최근접 거리 중앙이 판마다 3.60~4.59로 **정확히 반경 언저리에 눌려 있다**
# (표준 11판). 더 묶을 것이 없어서 멈춘 것이 아니라 반경이 막아서 멈춘다.
#
# 그래서 반경을 유도에서 **파레토**로 옮긴다 (표준 11판 · 저장소의 제 자로,
# Δ는 판별 상대변화의 평균):
#
#     반경   획색  총색  top8   top16  Δ보이는오차  Δp90   나빠진 판
#     3.0    131   391   0.168  0.271     +0.0%    +0.0%    0/11
#     4.5     83   349   0.232  0.349     +3.2%    +5.4%   10/11
#     6.0     57   319   0.275  0.408     +8.0%   +13.9%   11/11
#
# 4.5가 무릎이다 — 획 색이 **37% 줄고** 상위 8색 몫이 0.168 → 0.232로 느는데
# 대가는 3%대다. 6.0으로 더 가면 색을 31% 더 줄이는 값으로 오차가 2.5배가
# 된다. `rmse_src`·커버리지·선커버·경계 p95는 세 반경 모두에서 안 움직인다
# (이 패스는 색만 바꾸므로 원리적으로 그렇다).
#
# **문맥은 이 자를 못 이긴다.** 획마다 제 양옆 색면 짝을 읽어 "같은 일을 하는
# 획"의 가족을 짓고 가족 안에서만 크게 묶는 안(InkRoleGraph)을 지어 같은
# 자로 재 봤다: 획색 89 · 총색 367 · top8 0.223에 **Δ보이는오차 +9.1%**로,
# 반경 4.5(획색 83 · 총색 349 · top8 0.232 · +3.2%)에 **모든 축에서 진다**.
# 색을 옮겨도 되는 폭을 정하는 것은 "그 획이 무슨 일을 하나"가 아니라
# **그 색이 얼마나 움직이나**였다.
MAX_MOVE_DE = float(os.environ.get("FS_RAMP_MOVE_DE", 4.5))

# **무엇을 접나** — 기본은 획(`ink`)뿐이다.
#
# 판의 850색 중 557색이 획 색이다. 획 색은 원화에서 잰 값이 아니라 **제 발자국
# 아래 픽셀의 평균**이라(획마다 다른 값이 나오는 이유가 그것이다) 지켜야 할
# 정본이 없다 — 사람 판도 선은 한 잉크색으로 긋는다. 면(`cel`) 색은 다르다:
# 그 영역의 측정값이라 옮기면 그대로 재현 오차다.
#
# 실측이 그 구분을 뒷받침한다 (cel-01 전 구간). 전부 접으면 854→280색이지만
# 저장소의 보이는-오차 자가 `imp_error_seen` 0.104 → 0.464로 4.5배가 된다.
# 획만 접으면 색은 크게 줄면서 면 안 오차가 안 움직인다 — 획은 대부분 경계
# 띠라 그 자가 애초에 안 세는 자리다.
FOLD_LABELS = tuple(x for x in os.environ.get("FS_RAMP_LABELS", "ink").split(",") if x)
# 원화에서 **보이는 경계**의 자 (Lab). 이보다 벌어진 두 색이 맞닿아 있으면
# 못 묶는다 — 묶으면 그 경계가 그림에서 사라진다.
EDGE_DE = float(os.environ.get("FS_RAMP_EDGE_DE", 9.0))
# 인접 판정에 쓰는 상자 여유 (유닛) — 획은 면 위에 얹히므로 딱 붙어 있다.
ADJ_PAD = 0.5


def _lab(rgb: np.ndarray) -> np.ndarray:
    c = np.asarray(rgb, np.float64) / 255.0
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124, 0.3576, 0.1805],
                  [0.2126, 0.7152, 0.0722],
                  [0.0193, 0.1192, 0.9505]])
    xyz = c @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]),
                     200 * (f[:, 1] - f[:, 2])], axis=1)


def de00(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """CIEDE2000 색차 (Lab 입력, (N,3) 둘). 채택 판정은 이 자로 한다 (§20)."""
    a = np.atleast_2d(np.asarray(a, np.float64))
    b = np.atleast_2d(np.asarray(b, np.float64))
    L1, a1, b1 = a[:, 0], a[:, 1], a[:, 2]
    L2, a2, b2 = b[:, 0], b[:, 1], b[:, 2]
    C1, C2 = np.hypot(a1, b1), np.hypot(a2, b2)
    Cb = 0.5 * (C1 + C2)
    G = 0.5 * (1 - np.sqrt(Cb ** 7 / (Cb ** 7 + 25.0 ** 7 + 1e-30)))
    ap1, ap2 = (1 + G) * a1, (1 + G) * a2
    Cp1, Cp2 = np.hypot(ap1, b1), np.hypot(ap2, b2)
    hp1 = np.degrees(np.arctan2(b1, ap1)) % 360.0
    hp2 = np.degrees(np.arctan2(b2, ap2)) % 360.0
    dLp = L2 - L1
    dCp = Cp2 - Cp1
    dh = hp2 - hp1
    dh = np.where(dh > 180, dh - 360, np.where(dh < -180, dh + 360, dh))
    dh = np.where((Cp1 * Cp2) == 0, 0.0, dh)
    dHp = 2 * np.sqrt(Cp1 * Cp2) * np.sin(np.radians(dh) / 2)
    Lb = 0.5 * (L1 + L2)
    Cpb = 0.5 * (Cp1 + Cp2)
    hsum = hp1 + hp2
    hdiff = np.abs(hp1 - hp2)
    hpb = np.where(Cp1 * Cp2 == 0, hsum,
                   np.where(hdiff <= 180, 0.5 * hsum,
                            np.where(hsum < 360, 0.5 * (hsum + 360),
                                     0.5 * (hsum - 360))))
    T = (1 - 0.17 * np.cos(np.radians(hpb - 30))
         + 0.24 * np.cos(np.radians(2 * hpb))
         + 0.32 * np.cos(np.radians(3 * hpb + 6))
         - 0.20 * np.cos(np.radians(4 * hpb - 63)))
    dth = 30 * np.exp(-(((hpb - 275) / 25) ** 2))
    Rc = 2 * np.sqrt(Cpb ** 7 / (Cpb ** 7 + 25.0 ** 7 + 1e-30))
    Sl = 1 + 0.015 * (Lb - 50) ** 2 / np.sqrt(20 + (Lb - 50) ** 2)
    Sc = 1 + 0.045 * Cpb
    Sh = 1 + 0.015 * Cpb * T
    Rt = -np.sin(np.radians(2 * dth)) * Rc
    return np.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2
                   + Rt * (dCp / Sc) * (dHp / Sh))


def _layer_boxes(plan: LayerPlan, cat: Catalog) -> np.ndarray:
    from ..compose.look import layer_points

    out = np.zeros((len(plan.layers), 4), np.float64)
    for i, l in enumerate(plan.layers):
        pts = layer_points(l, cat)
        if len(pts):
            out[i] = (pts[:, 0].min(), pts[:, 1].min(),
                      pts[:, 0].max(), pts[:, 1].max())
        else:
            out[i] = (l.x, l.y, l.x, l.y)
    return out


def _forbidden(colors: list, idx: np.ndarray, boxes: np.ndarray,
               lab: np.ndarray) -> set:
    """**못 묶는 색 쌍** — 맞닿아 있고 원화에서 눈에 보이는 경계를 이루는 쌍.

    인접은 잉크 상자 겹침으로 잰다. 레이어 수가 커도 상자 겹침은 정렬 한 번에
    나온다 — x로 훑으며 겹치는 구간만 본다 (전수 N²을 피한다).
    """
    n = len(boxes)
    if n < 2:
        return set()
    order = np.argsort(boxes[:, 0], kind="stable")
    bad: set = set()
    for oi in range(n):
        i = int(order[oi])
        xi1 = boxes[i, 2] + ADJ_PAD
        for oj in range(oi + 1, n):
            j = int(order[oj])
            if boxes[j, 0] > xi1:
                break                              # x로 이미 벌어졌다
            if (boxes[j, 1] > boxes[i, 3] + ADJ_PAD
                    or boxes[i, 1] > boxes[j, 3] + ADJ_PAD):
                continue
            ci, cj = int(idx[i]), int(idx[j])
            if ci == cj or (ci, cj) in bad:
                continue
            if float(np.linalg.norm(lab[ci] - lab[cj])) >= EDGE_DE:
                bad.add((min(ci, cj), max(ci, cj)))
    return bad


def _merge(lab: np.ndarray, weight: np.ndarray, bad: set,
           move: float = MAX_MOVE_DE):
    """제약 병합 — 더 못 묶을 때까지 가장 가까운 쌍을 합친다.

    가중 평균 연결(Ward와 같은 결)이라 무게가 큰 색이 대표를 잡는다. 두 제약이
    있다: 못 묶는 쌍(경계, §21)은 무리가 합쳐질 때 함께 물려 다니고
    (a-b가 금지면 a가 든 무리와 b가 든 무리도 금지),
    **어느 원색도 제 무리 중심에서 `move` Lab ΔE 넘게 멀어질 수 없다**
    (자는 `celart.marks._MARK_DE`와 같은 계다).
    """
    n = len(lab)
    cen = lab.copy()
    w = weight.astype(np.float64).copy()
    alive = np.ones(n, bool)
    member = [{i} for i in range(n)]
    forb = [set() for _ in range(n)]
    for i, j in bad:
        forb[i].add(j)
        forb[j].add(i)
    parent = np.arange(n)
    d = np.linalg.norm(cen[:, None, :] - cen[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    for i in range(n):
        for j in forb[i]:
            d[i, j] = d[j, i] = np.inf
    # **가장 가까운 쌍은 힙이 낸다** — 행렬 전체 `argmin`을 바퀴마다 돌리면
    # 기각(반경 밖)이 잦은 판에서 그 하나가 이 단의 태반이다 (실측 11번 판:
    # 1,354색에 바퀴 59,321번 · 9초). 행렬은 종전 그대로 쓰고(값·기각·행
    # 갱신이 전부 같은 자리에 같은 값으로 적힌다) 힙은 그 위의 색인일 뿐이다:
    # 꺼낸 항목의 값이 행렬의 지금 값과 다르면 낡은 것이라 버린다. 동점은
    # `(값, i, j)` 순이라 `argmin`의 행우선 첫 자리와 같은 쌍이 나온다.
    import heapq

    iu, ju = np.triu_indices(n, 1)
    vals = d[iu, ju]
    fin = np.isfinite(vals)
    heap = list(zip(vals[fin].tolist(), iu[fin].tolist(), ju[fin].tolist()))
    heapq.heapify(heap)
    live = n
    while live > 1 and heap:
        val, i, j = heapq.heappop(heap)
        if d[i, j] != val:
            continue                               # 낡은 항목 (갱신·기각·사망)
        wt = w[i] + w[j]
        cn = (cen[i] * w[i] + cen[j] * w[j]) / max(wt, 1e-12)
        # **어느 원색도 반경 밖으로 안 나간다** — 나가면 그 쌍만 막고 계속한다
        mem = member[i] | member[j]
        far = float(np.linalg.norm(lab[sorted(mem)] - cn, axis=1).max())
        if far > move:
            d[i, j] = d[j, i] = np.inf
            continue
        cen[i] = cn
        w[i] = wt
        member[i] = mem
        forb[i] |= forb[j]
        alive[j] = False
        parent[list(member[j])] = i
        for m in member[j]:
            parent[m] = i
        d[j, :] = np.inf
        d[:, j] = np.inf
        nd = np.linalg.norm(cen - cen[i], axis=1)
        nd[~alive] = np.inf
        nd[i] = np.inf
        for f in forb[i]:
            if alive[f]:
                nd[f] = np.inf
        # 금지 관계는 대칭이라 상대 쪽 줄도 막는다
        d[i, :] = nd
        d[:, i] = nd
        for k in np.flatnonzero(np.isfinite(nd)).tolist():
            heapq.heappush(heap, (float(nd[k]), min(i, k), max(i, k)))
        live -= 1
    # 무리 대표 색인
    root = np.arange(n)
    for i in range(n):
        if not alive[i]:
            continue
        for m in member[i]:
            root[m] = i
    return root, cen


def fold_colors(plan: LayerPlan, cat: Catalog, *, move: float = MAX_MOVE_DE,
                labels: tuple = FOLD_LABELS,
                log=None) -> tuple[LayerPlan, dict]:
    """판 하나의 색을 역할색으로 접는다. 장수·기하는 안 건드린다.

    `labels`가 접을 대상을 정한다 — 기본은 획(`ink`)뿐이다 (모듈 머리말).
    """
    if not plan.layers:
        return plan, {}
    pick = [i for i, l in enumerate(plan.layers)
            if not labels or l.label in labels]
    if len(pick) < 4:
        return plan, {"colors": len({tuple(l.color) for l in plan.layers}),
                      "after": len({tuple(l.color) for l in plan.layers})}
    cols = [tuple(plan.layers[i].color) for i in pick]
    uniq = sorted(set(cols))
    if len(uniq) < 4:
        return plan, {"colors": len(uniq), "after": len(uniq)}
    cmap = {c: i for i, c in enumerate(uniq)}
    idx = np.array([cmap[c] for c in cols], np.int64)
    lab = _lab(np.array(uniq, np.float64))
    boxes = _layer_boxes(plan, cat)[np.asarray(pick, int)]
    area = np.maximum((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
                      1e-9)
    weight = np.zeros(len(uniq))
    np.add.at(weight, idx, area)
    bad = _forbidden(uniq, idx, boxes, lab)
    stats = {"colors": len(uniq), "edges_locked": len(bad)}
    root, cen = _merge(lab, weight, bad, move)
    new = cen[root]
    de = de00(lab, new)
    wsum = weight.sum() or 1.0
    mean = float((de * weight).sum() / wsum)
    p95 = float(np.percentile(np.repeat(de, np.maximum(
        (weight / wsum * 10000).astype(int), 1)), 95))
    rgb = np.clip(_lab_to_rgb(cen), 0, 255).astype(int)
    table = {uniq[i]: tuple(int(v) for v in rgb[int(root[i])])
             for i in range(len(uniq))}
    # **사본이라야 한다** — 같은 Layer 객체를 물면 색을 바꾸는 순간 원본 판도
    # 같이 바뀐다 (A/B가 제 뒤를 재는 꼴이 된다).
    from dataclasses import replace as _replace

    out = LayerPlan(source_image=plan.source_image, image_size=plan.image_size,
                    units_per_px=plan.units_per_px,
                    layers=[_replace(l, color=table.get(tuple(l.color), l.color))
                            if i in set(pick) else _replace(l)
                            for i, l in enumerate(plan.layers)])
    stats.update({"move_de": move, "folded": len(pick),
                  "before_all": len({tuple(l.color) for l in plan.layers}),
                  "after": len({tuple(l.color) for l in out.layers}),
                  "mean_de00": round(mean, 3), "p95_de00": round(p95, 3)})
    if log:
        log(msg("  팔레트 접기 {a}색 → {b}색 (반경 ΔE {mv:g}) · "
                "평균 ΔE00 {m:.2f} · p95 {p:.2f} · 경계 잠금 {e}쌍",
                a=stats["before_all"], b=stats["after"], mv=move, m=mean, p=p95,
                e=len(bad)))
    return out, stats


def _lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]
    fy = (L + 16) / 116
    fx = fy + a / 500
    fz = fy - b / 200
    def _inv(t):
        return np.where(t ** 3 > 216 / 24389, t ** 3, (116 * t - 16) * 27 / 24389)
    xyz = np.stack([_inv(fx) * 0.95047, _inv(fy), _inv(fz) * 1.08883], axis=1)
    m = np.array([[3.2406, -1.5372, -0.4986],
                  [-0.9689, 1.8758, 0.0415],
                  [0.0557, -0.2040, 1.0570]])
    c = xyz @ m.T
    c = np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.abs(c) ** (1 / 2.4) - 0.055)
    return np.round(np.clip(c, 0, 1) * 255)
