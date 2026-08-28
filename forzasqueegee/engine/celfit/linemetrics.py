"""선 도안의 **구조 지표** — 사람이 그은 것처럼 보이나를 재는 자리.

커버리지(`ink_near`)·스필(`ink_stray`)·RMSE만으로는 이 노선의 물음에 답이
안 된다. 셋 다 "선이 있는 자리에 잉크가 있나"를 묻는 자라서, **어떻게 그었나**
가 통째로 안 보인다. 실제로 그랬다: 잉크 능선 폭 중앙이 3.83px인데 원화 선
지도는 2.11px이라 머리칼 가닥이 서로 붙어 덩어리로 읽히는데, `ink_stray`는
0.0017이었다 — 스필의 자가 "선 지도에서 최소 도형 폭 밖"이라 두 배로 굵어진
획도 전부 그 안이기 때문이다.

여기서 내는 것은 **획의 모습**이다:

    width_ratio      놓인 폭 / 원화 띠 폭 (중앙·p90) — 굵기 충실도
    width_over       원화 띠의 배가 넘게 굵은 획의 몫
    width_bulge      한 도형 안의 **최대 폭 / 제 중앙 폭** (중앙·p90) — 배부름.
                     `width_ratio`는 도형마다 한 수(중앙)로 접어 묻기 때문에
                     "가운데가 부푼" 잎사귀를 원리적으로 못 본다. 부푸는 까닭은
                     도형이 아니라 **비등방 스케일**이라 등방 서술자로도 못
                     본다 (`stroke._STROKE_BULGE` 문서). 레퍼런스 쪽 자는
                     최대/**최소**를 라스터로 재므로 눈금이 달라 직접
                     견줄 수 없다
    end_pointy       끝 폭이 제 중앙의 절반도 안 되는 도형의 몫 (잎사귀 티)
    tang_err         도형 끝이 경로 접선과 어긋난 각 (중앙·p90) — 흐름 충실도
    joint_kink       한 획 안 이웃 마디가 **이음에서** 이루는 각 (중앙·p90).
                     경로가 실제로 꺾인 자리에서는 커야 맞으므로 `tang_err`와
                     함께 읽는다 — 이쪽이 "따라 그었나"의 자다
    joint_gap        그 이음의 중심선 끝 사이 거리 px (중앙)
    joint_kink_flat  **각이 아닌 자리**의 이음각 (중앙·p90) — 곡률 튐 그 자체.
                     `joint_kink`는 "경로가 실제로 꺾인 자리"와 "매끈한 호를
                     한가운데서 끊어 생긴 각"을 함께 세므로 개선을 못 읽는다.
                     사람 도안에 없는 것은 뒤쪽뿐이라 그것만 따로 낸다
    corner_at_joint  **의도된 각** 중 실제로 이음이 선 몫 (`intent` 문서).
                     각이 도형 하나 안에 뭉개지면 그 각이 둥글어진다 —
                     `joint_kink_flat`과 짝이다 (한쪽만 보면 "안 끊으면
                     좋다"·"많이 끊으면 좋다"로 각각 기운다)
    reach_miss       사슬이 제 경로 끝까지 못 간 거리 px (중앙)
    seam_ratio       이음 보수가 획 도형에서 차지하는 몫
    shapes/stroke    한 획을 몇 장으로 그었나 · 1장 · ≤2장 비율
    stroke_len       논리 획 길이 (중앙) · 길이 1,000px당 획 수 (파편화)
    kink             한 획 안 이웃 도형의 접선 꺾임 (도) — 곡률 튐
    family_switch    한 획이 두 계열 이상의 도형을 섞어 쓴 몫
    long_break       긴 획(중앙 길이 이상) 중 끊김이 남은 몫

**게이트가 아니다.** 전부 report의 `structure`에 실어 판 사이를 대 보는
자리다 — 어느 축이 무엇을 바꿨는지는 그 표가 답한다.

폭은 라스터를 다시 안 뜬다 — 배치가 도형을 고를 때 쓰는 그 닫힌 식
(`descriptor.placed_profile`)을 그대로 부른다. "재는 폭 = 고르는 폭"이라야
"폭을 맞췄다"가 검증 가능한 말이 된다.
"""

from __future__ import annotations

import numpy as np

from ..catalog import Catalog
from ..model import Layer, LayerPlan
from . import chain
from . import intent as I
from .descriptor import descriptors, layer_width_px, placed_widths


def _placed_axis(cat: Catalog, lay: Layer, upp: float):
    """레이어의 **놓인 중심선** 방향과 중점 px — 중심선이 없으면 None.

    도형마다 로컬 프레임이 달라서 `Layer.rot`을 그대로 견주면 뜻이 없다
    (같은 방향으로 놓인 두 도형의 rot이 90° 어긋난다). 그래서 서술자의
    중심선을 실제 변환(회전 × 비등방 스케일)으로 옮겨 방향을 읽는다.
    """
    d = descriptors(cat).get(lay.shape)
    if d is None or not d.stroke_ok or len(d.center) < 3:
        return None
    th = np.radians(lay.rot)
    c, s = np.cos(th), np.sin(th)
    p = d.center * np.array([lay.sx, lay.sy], np.float64)
    p = p @ np.array([[c, s], [-s, c]], np.float64)
    v = p[-1] - p[0]
    if float(np.hypot(*v)) < 1e-9:
        return None
    mid = np.array([lay.x, lay.y], np.float64) + p.mean(axis=0)
    # 위치는 **이미지 방향**으로 돌려 준다 (캔버스는 y-up) — 경로를 따라
    # 마디를 줄 세우는 데만 쓰므로 원점 이동은 필요 없다
    return v / float(np.hypot(*v)), np.array(
        [mid[0], -mid[1]], np.float64) / max(upp, 1e-9)


def _fam(name: str) -> str:
    """도형 계열 — 카탈로그 이름의 앞 글자 (게임 탭). 계열 섞임의 자."""
    return name.split("_")[0] if "_" in name else name


def _pct(v: np.ndarray, q: float) -> float:
    return float(np.percentile(v, q)) if len(v) else 0.0


def _tang_err(ch, path_g: np.ndarray) -> list[float]:
    """마디마다 **양끝의 접선이 경로와 어긋난 각** (도) — 흐름 충실도의 자.

    이음각(`joint_kink`)만으로는 "경로가 실제로 꺾인 자리"와 "도형이 엉뚱한
    방향으로 나간 자리"가 안 갈린다 (RDP가 마디를 굽음이 큰 자리에서 끊으므로
    이음각이 큰 것이 맞는 자리가 있다). 여기서는 도형 끝의 방향을 **그 자리
    경로의 방향**과 직접 견준다 — 배치가 최소화하는 양(`stroke._tang_pen`)과
    같은 자다.
    """
    if len(path_g) < 3:
        return []
    xy = np.stack([path_g[:, 1], path_g[:, 0]], axis=1).astype(np.float64)
    out = []
    for _i, a, b, ta, tb in ch:
        for pt, t in ((a, ta), (b, -tb)):
            k = int(np.argmin(((xy - pt) ** 2).sum(axis=1)))
            j0, j1 = max(0, k - 2), min(len(xy) - 1, k + 2)
            v = xy[j1] - xy[j0]
            n = float(np.hypot(*v))
            if n < 1e-9:
                continue
            cos = abs(float(np.clip(np.dot(t, v / n), -1.0, 1.0)))
            out.append(float(np.degrees(np.arccos(cos))))
    return out


def _joint_at(path_g: np.ndarray, pt: np.ndarray) -> int:
    """이음 중점에 가장 가까운 경로 표본 인덱스 (path_g는 (y,x), pt는 (x,y))."""
    xy = np.stack([path_g[:, 1], path_g[:, 0]], axis=1).astype(np.float64)
    return int(np.argmin(((xy - pt) ** 2).sum(axis=1)))


def stroke_metrics(plan: LayerPlan, rec, cat: Catalog, upp: float) -> dict:
    """배치된 플랜 + 재구성 자취 → 구조 지표 (rec가 없으면 빈 dict)."""
    if rec is None or not rec.strokes:
        return {}
    by: dict[int, list] = {}
    for lay in plan.layers:
        if lay.stroke >= 0:
            by.setdefault(lay.stroke, []).append(lay)

    wr: list[float] = []          # 폭 비 (놓인 / 원화)
    kink: list[float] = []
    jkink: list[float] = []       # 이음에서의 각 (끝 접선끼리)
    terr: list[float] = []        # 도형 끝 접선 대 경로 접선
    jflat: list[float] = []       # 각이 **아닌** 자리의 이음각
    jgap: list[float] = []
    reach: list[float] = []
    corner_n = corner_hit = 0
    bulge: list[float] = []       # 한 도형 안의 최대 폭 / 제 중앙 폭
    pointy = n_shape = 0
    n_seam = 0
    nsh: list[int] = []
    slen: list[float] = []
    mixed = multi = 0
    long_brk = long_n = 0
    lens = [float(s.ev.length) for s in rec.strokes if s.shapes]
    med_len = float(np.median(lens)) if lens else 0.0
    for s in rec.strokes:
        if not s.shapes:
            continue
        lays = by.get(s.sid) or []
        if not lays:
            continue
        # 이음·꺾임은 **배치 도형만** 본다 — 이음 보수 장은 그 자리를 깁는
        # 도형이라 "어떻게 그었나"의 답이 아니다 (플랜 순서상 뒤에 붙는다)
        core = lays[:max(1, s.shapes)] if s.shapes else lays
        n_seam += max(0, len(lays) - len(core))
        for lay in lays:
            d = descriptors(cat).get(lay.shape)
            if d is None or not d.stroke_ok or len(d.center) < 5:
                continue
            pw, _m, L = placed_widths(d.center, d.halfw, lay.sx, lay.sy)
            if L <= 0 or len(pw) < 5:
                continue
            med = float(np.median(pw))
            n_shape += 1
            if med > 1e-9 and float(min(pw[0], pw[-1])) < 0.5 * med:
                pointy += 1
            if med > 1e-9:
                bulge.append(float(pw.max()) / med)
        if len(core) >= 1:
            pg = np.stack([s.path[:, 0] + s.roi[1], s.path[:, 1] + s.roi[0]],
                          axis=1)
            ch = chain._order(cat, core, pg, upp, plan.image_size[0],
                              plan.image_size[1])
            if ch:
                terr.extend(_tang_err(ch, pg))
                # 이음이 **각 자리**에 섰나 — `intent`가 지목한 각과 대 본다
                cs = (s.intent.corner if s.intent is not None
                      and len(s.intent.corner) == len(pg) else None)
                win = max(3, int(round(1.5 * max(s.width, 1.0))))
                jn: list[int] = []
                for (_, _, e0, _, t0), (_, s1, _, t1, _) in zip(ch, ch[1:]):
                    jgap.append(float(np.hypot(*(s1 - e0))))
                    ang = float(np.degrees(np.arccos(float(
                        np.clip(np.dot(t0, t1), -1.0, 1.0)))))
                    jkink.append(ang)
                    if cs is None:
                        continue
                    k = _joint_at(pg, 0.5 * (e0 + s1))
                    jn.append(k)
                    lo, hi = max(0, k - win), min(len(cs), k + win + 1)
                    near = float(cs[lo:hi].max()) if hi > lo else 0.0
                    if near <= 0.0:
                        jflat.append(ang)      # 매끈한 자리를 끊어 생긴 각
                if cs is not None:
                    got = I.nodes_of(I.StrokeIntent(cs), len(cs))
                    corner_n += len(got)
                    corner_hit += sum(1 for c in got.tolist()
                                      if any(abs(c - k) <= win for k in jn))
                reach.append(float(np.hypot(*(ch[0][1] - pg[0][::-1]))))
                reach.append(float(np.hypot(*(ch[-1][2] - pg[-1][::-1]))))
        nsh.append(len(lays))
        slen.append(float(s.ev.length))
        # 폭 — 도형마다 닫힌 식으로 재고 획 안에서 길이 가중 없이 중앙을 쓴다
        got = [w for w in (layer_width_px(cat, l, upp) for l in lays) if w > 0]
        if got:
            wr.append(float(np.median(got)) / max(float(s.width), 1e-6))
        fams = {_fam(l.shape) for l in lays}
        if len(lays) > 1:
            multi += 1
            if len(fams) > 1:
                mixed += 1
            # 꺾임 — **놓인 중심선 방향**의 이웃 간 차 (한 획 안에서 얼마나
            # 튀나). 사람이 한 획을 여러 장으로 나눠 그으면 마디끼리 방향이
            # 매끄럽게 이어지고, 기계가 나누면 그 자리가 각으로 읽힌다.
            # 순서는 획 경로를 따라 — 레이어 순서가 곧 경로 순서는 아니다
            ax = [(a, m) for a, m in
                  (_placed_axis(cat, l, upp) or (None, None) for l in lays)
                  if a is not None]
            if len(ax) > 1:
                t0 = s.path[0][::-1] + np.array([s.roi[0], s.roi[1]])
                t1 = s.path[-1][::-1] + np.array([s.roi[0], s.roi[1]])
                u = t1 - t0
                if float(np.hypot(*u)) > 1e-9:
                    u = u / float(np.hypot(*u))
                    ax.sort(key=lambda am: float(np.dot(am[1], u)))
                for (a, _), (b, _) in zip(ax, ax[1:]):
                    d = abs(float(np.degrees(np.arctan2(
                        a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1]))))
                    kink.append(min(d, 180.0 - d))
        if med_len and s.ev.length >= med_len:
            long_n += 1
            if s.cand.get("breaks", 0):
                long_brk += 1

    ink_len = float(sum(slen))
    wr_a = np.asarray(wr, np.float64)
    out = {
        "width_ratio_med": round(float(np.median(wr_a)), 3) if len(wr_a) else None,
        "width_ratio_p90": round(_pct(wr_a, 90), 3) if len(wr_a) else None,
        "width_over": (round(float((wr_a > 1.5).mean()), 4)
                       if len(wr_a) else None),
        "stroke_len_med": round(med_len, 1),
        "strokes_per_kpx": (round(1000.0 * len(nsh) / ink_len, 2)
                            if ink_len > 0 else None),
        "kink_med_deg": round(float(np.median(kink)), 1) if kink else 0.0,
        "kink_p90_deg": round(_pct(np.asarray(kink), 90), 1) if kink else 0.0,
        "family_switch": round(mixed / multi, 4) if multi else 0.0,
        "long_break_ratio": round(long_brk / long_n, 4) if long_n else 0.0,
        "width_bulge_med": (round(float(np.median(bulge)), 3)
                            if bulge else None),
        "width_bulge_p90": round(_pct(np.asarray(bulge), 90), 3) if bulge else None,
        "end_pointy": round(pointy / n_shape, 4) if n_shape else 0.0,
        "tang_err_med": round(float(np.median(terr)), 1) if terr else 0.0,
        "tang_err_p90": round(_pct(np.asarray(terr), 90), 1) if terr else 0.0,
        "joint_kink_med": round(float(np.median(jkink)), 1) if jkink else 0.0,
        "joint_kink_p90": round(_pct(np.asarray(jkink), 90), 1) if jkink else 0.0,
        "joint_kink_flat_med": (round(float(np.median(jflat)), 1)
                                if jflat else 0.0),
        "joint_kink_flat_p90": (round(_pct(np.asarray(jflat), 90), 1)
                                if jflat else 0.0),
        "joint_flat_ratio": (round(len(jflat) / len(jkink), 4)
                             if jkink else 0.0),
        "corner_n": corner_n,
        "corner_at_joint": (round(corner_hit / corner_n, 4)
                            if corner_n else 0.0),
        "joint_gap_med": round(float(np.median(jgap)), 2) if jgap else 0.0,
        "reach_miss_med": round(float(np.median(reach)), 2) if reach else 0.0,
        "seam_ratio": (round(n_seam / max(1, sum(nsh)), 4) if nsh else 0.0),
    }
    return out
