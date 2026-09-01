"""겹침 병합 — **같은 방향으로 겹쳐 놓인 짧은 막대를 하나로** (배치 뒤).

한 획을 마디로 끊고, 그 마디를 이웃 마디와 겹치게 늘리고, 남은 틈을 이음
보수로 메우는 세 기계가 각자 옳게 굴어도 결과에는 같은 방향 짧은 막대가
겹쳐 쌓인다 — 레이어가 예산이므로 그것이 곧 낭비다 (사용자 지적 ②).

여기서는 그것을 **결과물에서** 걷는다: 방향이 같고 축이 겹치는 막대 둘을
감싸는 막대 한 장으로 바꾼다. 받는 조건은 사용자가 준 그대로다 —
**합쳐서 그림이 나빠지면 안 합친다.** 세 가지를 라스터로 직접 대조한다:
덮이던 자리가 드러나지 않고, 선 밴드 밖으로 잉크가 새지 않고, 원화와의 색
오차가 안 는다. 셋 다 문턱이 아니라 대조라 새 상수가 거의 없다.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

from ...i18n import msg
from ..catalog import Catalog
from ..model import UNITS_PER_SCALE, Layer, LayerPlan
from .geometry import _poly_px
from .skeleton import _cross2
from .vocabulary import _BAR_SHAPE


# 방향 문턱 — 이보다 어긋난 둘은 "같은 방향"이 아니다. 게임 회전 스텝이
# 0.1°이므로 자유도 쪽 하한은 없고, 위쪽은 합친 막대가 원래 둘을 덮느냐가
# 스스로 막는다 (어긋날수록 감싸는 막대가 뚱뚱해져 아래 게이트에 걸린다).
# 15°는 그 게이트가 실제로 물리기 시작하는 자리라 값이 아니라 여유다.
_MERGE_ANG = float(os.environ.get("FS_MERGE_ANG", 15.0))
# 받아 주는 손실·군살. 둘 다 합집합·합본 넓이 대비다.
# - 손실(덮이던 자리가 드러남): 2%는 라스터 반올림 한 겹의 크기다 (막대
#   테두리 한 줄 ≈ 길이×1px / 넓이). 0으로 두면 양자화 지터만으로 전부 기각된다.
# - 군살(새로 나가는 잉크): 선 밴드(`allow`) **밖**으로 나가는 몫만 센다 —
#   밴드 안 군살은 어차피 그을 자리라 낭비가 아니다.
_MERGE_LOSS = float(os.environ.get("FS_MERGE_LOSS", 0.02))
_MERGE_FAT = float(os.environ.get("FS_MERGE_FAT", 0.02))
_MERGE_PASSES = int(os.environ.get("FS_MERGE_PASSES", 3))


def _bar_geom(lay: Layer, upp: float, w: int, h: int):
    """막대 레이어 → (중심 px, 축 단위벡터, 반길이 px, 반폭 px).

    `geometry._layer`의 역이다 — 같은 식이라 왕복이 정확하다.

    **전단이 들면 반길이가 는다.** 로컬 ±1 상자가 `x → sx·p + skew·sy·q`로
    가므로 x의 최대가 `|sx| + |skew·sy|`다 (y 쪽은 안 바뀐다 — 전단은 y를
    안 건드린다). 이 값이 짝을 고르는 자라, 안 고치면 기울어진 막대의
    "축으로 이어지나" 판정이 짧게 잡혀 이을 짝을 놓친다.
    """
    a = (abs(lay.sx) + abs(lay.skew * lay.sy)) * UNITS_PER_SCALE / upp
    b = abs(lay.sy) * UNITS_PER_SCALE / upp
    th = -np.radians(lay.rot)              # 이미지 y-down 각
    cx = lay.x / upp + w / 2.0
    cy = h / 2.0 - lay.y / upp
    return (np.array([cy, cx]), np.array([np.sin(th), np.cos(th)]), a, b)


def _mask_of(cat: Catalog, lays, upp: float, w: int, h: int, box):
    """레이어 몇 장을 한 창(box)에 라스터한다 (짝홀 규칙)."""
    x0, y0, x1, y1 = box
    out = np.zeros((y1 - y0, x1 - x0), np.uint8)
    for lay in lays:
        one = np.zeros_like(out)
        for p in _poly_px(cat, lay, upp, w, h, x0, y0):
            mm = np.zeros_like(out)
            cv2.fillPoly(mm, [np.round(p).astype(np.int32)], 1)
            one ^= mm
        out |= one
    return out.astype(bool)


def _box_of(cat: Catalog, lays, upp: float, w: int, h: int, pad: int = 2):
    xs0 = ys0 = 10 ** 9
    xs1 = ys1 = -10 ** 9
    for lay in lays:
        for p in _poly_px(cat, lay, upp, w, h):
            xs0 = min(xs0, int(np.floor(p[:, 0].min())))
            ys0 = min(ys0, int(np.floor(p[:, 1].min())))
            xs1 = max(xs1, int(np.ceil(p[:, 0].max())))
            ys1 = max(ys1, int(np.ceil(p[:, 1].max())))
    return (max(0, xs0 - pad), max(0, ys0 - pad),
            min(w, xs1 + pad), min(h, ys1 + pad))


def _fuse(a: Layer, b: Layer, upp: float, w: int, h: int,
          color=None) -> Layer | None:
    """막대 둘을 감싸는 막대 한 장 — 방향이 너무 어긋나면 None."""
    ca, ua, la, wa = _bar_geom(a, upp, w, h)
    cb, ub, lb, wb = _bar_geom(b, upp, w, h)
    if float(np.dot(ua, ub)) < 0:          # 반대로 누운 같은 축
        ub = -ub
    d = abs(np.degrees(np.arctan2(_cross2(ua, ub), float(np.dot(ua, ub)))))
    if d > _MERGE_ANG:
        return None
    # 합본 축 = 반길이로 가중한 평균 방향 (긴 쪽이 축을 정한다)
    u = ua * la + ub * lb
    nu = float(np.hypot(*u))
    if nu < 1e-9:
        return None
    u = u / nu
    nvec = np.array([-u[1], u[0]])
    # 감싸는 상자는 **두 막대의 모서리 여덟 점**으로 잡는다 — 중심선만 보면
    # 비스듬한 짝에서 모서리가 밖으로 삐져나가 "덮이던 자리"가 드러난다
    corn = np.array([c_ + sl * l_ * u_ + sw * w_ * np.array([-u_[1], u_[0]])
                     for c_, u_, l_, w_ in ((ca, ua, la, wa), (cb, ub, lb, wb))
                     for sl in (1, -1) for sw in (1, -1)])
    t, s = corn @ u, corn @ nvec
    c = 0.5 * (t.min() + t.max()) * u + 0.5 * (s.min() + s.max()) * nvec
    half = 0.5 * float(t.max() - t.min())
    halfw = 0.5 * float(s.max() - s.min())
    # **합치기는 잇는 것이지 굵히는 것이 아니다** — 합본이 굵은 쪽보다 더
    # 굵어지면 그것은 같은 방향으로 겹친 짝이 아니다 (여유 1px = 라스터 한 겹)
    if halfw > max(wa, wb) + 1.0:
        return None
    th = float(np.arctan2(u[0], u[1]))
    return Layer(shape=_BAR_SHAPE,
                 x=float((c[1] - w / 2.0) * upp),
                 y=float((h / 2.0 - c[0]) * upp),
                 sx=max(0.01, half * upp / UNITS_PER_SCALE),
                 sy=max(0.01, halfw * upp / UNITS_PER_SCALE),
                 rot=(-np.degrees(th)) % 360.0, skew=0.0,
                 color=a.color if color is None else color,
                 alpha=100.0, label=a.label,
                 stroke=a.stroke if abs(a.sx) >= abs(b.sx) else b.stroke
                 ).quantized()


def merge_costrokes(plan: LayerPlan, cat: Catalog, upp: float,
                    size: tuple[int, int], allow: np.ndarray,
                    src: np.ndarray, lo: int, log, st: dict) -> int:
    """같은 방향으로 겹친 막대들을 합친다 — 반환 = 줄인 장수.

    `allow`는 잉크가 나가도 되는 자리(선 밴드), `src`는 원화 색(합본 색의
    자). `lo`부터의 레이어만 본다. 수렴할 때까지(최대 `_MERGE_PASSES`)
    되풀이한다 — 한 번 합친 것이 다음 이웃과 또 합쳐지면서 짧은 막대 사슬이
    긴 한 장으로 걷힌다.
    """
    w, h = size
    srcf = src.astype(np.float32)
    gone = 0
    for _ in range(max(1, _MERGE_PASSES)):
        idx = [i for i in range(lo, len(plan.layers))
               if plan.layers[i].label == "ink"
               and plan.layers[i].shape == _BAR_SHAPE]
        if len(idx) < 2:
            break
        geo = {i: _bar_geom(plan.layers[i], upp, w, h) for i in idx}
        cell = {}
        # 격자 칸은 **긴 막대의 반길이**만큼 — 3×3 이웃이 중심 거리 2칸을
        # 덮으므로 이래야 "축으로 이어지는" 짝(중심 거리 ≤ 반길이 합)을 다 본다
        step = max(8.0, float(np.percentile([g[2] for g in geo.values()], 90)))
        for i in idx:
            c = geo[i][0]
            cell.setdefault((int(c[0] // step), int(c[1] // step)), []).append(i)
        dead: set[int] = set()
        n_pass = 0
        for i in idx:
            if i in dead:
                continue
            ci, ui, li = geo[i][:3]
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    for j in cell.get((int(ci[0] // step) + dy,
                                       int(ci[1] // step) + dx), ()):
                        if j <= i or j in dead or i in dead:
                            continue
                        a, b = plan.layers[i], plan.layers[j]
                        cj, lj = geo[j][0], geo[j][2]
                        # 축 방향으로 이어지거나 겹치는 것만 (나란한 이웃 가닥은
                        # 합치면 그 사이 흰 틈이 메워져 그림이 뭉갠다)
                        v = cj - ci
                        if abs(float(v @ np.array([-ui[1], ui[0]]))) \
                                > geo[i][3] + geo[j][3]:
                            continue
                        if abs(float(v @ ui)) > li + lj:
                            continue
                        box = _box_of(cat, (a, b), upp, w, h)
                        if box[0] >= box[2] or box[1] >= box[3]:
                            continue
                        ma = _mask_of(cat, (a,), upp, w, h, box)
                        mb = _mask_of(cat, (b,), upp, w, h, box)
                        un = ma | mb
                        nu = float(un.sum())
                        if nu < 1:
                            continue
                        # 합본 색 = 넓이 가중 평균. 색이 달라도 후보로 두고,
                        # 아래 게이트 ③이 "그림이 나빠지나"로 가린다
                        na_, nb_ = float(ma.sum()), float(mb.sum())
                        col = tuple(int(round((na_ * ca_ + nb_ * cb_)
                                              / max(na_ + nb_, 1.0)))
                                    for ca_, cb_ in zip(a.color, b.color))
                        m = _fuse(a, b, upp, w, h, color=col)
                        if m is None:
                            continue
                        box = _box_of(cat, (a, b, m), upp, w, h)
                        ma = _mask_of(cat, (a,), upp, w, h, box)
                        mb = _mask_of(cat, (b,), upp, w, h, box)
                        un = ma | mb
                        mm = _mask_of(cat, (m,), upp, w, h, box)
                        nu, nm = float(un.sum()), float(mm.sum())
                        if nu < 1 or nm < 1:
                            continue
                        # ① 덮이던 자리가 드러나면 안 합친다 (사용자 조건)
                        if float((un & ~mm).sum()) > _MERGE_LOSS * nu:
                            continue
                        # ② 선 밴드 밖으로 나가는 군살도 안 된다
                        out = (mm & ~un) & ~allow[box[1]:box[3], box[0]:box[2]]
                        if float(out.sum()) > _MERGE_FAT * nm:
                            continue
                        # ③ **색이 나빠지면 안 합친다** — 합집합 자리에서
                        # 원화와의 오차를 잰다. 지금은 나중에 그린 쪽 색이
                        # 보이므로 그것이 비교 대상이다 (문턱이 아니라 대조라
                        # 새 상수가 없다). 같은 색끼리는 오차가 그대로다
                        sb = srcf[box[1]:box[3], box[0]:box[2]]
                        top = np.where(mb[..., None], np.float32(b.color),
                                       np.float32(a.color))
                        e0 = float(np.abs(top - sb)[un].sum())
                        e1 = float(np.abs(np.float32(col) - sb)[un].sum())
                        if e1 > e0:
                            continue
                        plan.layers[i] = m
                        geo[i] = _bar_geom(m, upp, w, h)
                        ci, ui, li = geo[i][:3]
                        dead.add(j)
                        n_pass += 1
        if not n_pass:
            break
        plan.layers[:] = [l for k, l in enumerate(plan.layers) if k not in dead]
        gone += n_pass
    st["merged_bars"] = gone
    if gone:
        log(msg("  겹침 병합 {gone}장 감소 (같은 방향 막대 합침 — 덮임 손실 없음)",
                gone=gone))
    return gone
