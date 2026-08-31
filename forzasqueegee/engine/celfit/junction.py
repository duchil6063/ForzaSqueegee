"""§29 **교차점** — 획**끼리** 만나는 자리를 라스터에서 직접 맞춘다.

한 획 **안**의 이음은 `chain`이 맡는다 (마디를 경로 순으로 세워 틈·꺾임을
줄인다). 여기가 맡는 것은 그 짝이다: T·Y·X 교차처럼 **다른 획끼리** 만나는
자리. 그 자리는 지금까지 아무도 안 봤다 — 획마다 제 경로만 보고 서므로,
두 획의 끝이 반 픽셀씩 물러서면 교차점 한복판에 흰 점이 남고(끊김), 반대로
서로 지나쳐 겹치면 검은 덩어리가 된다(뭉침). 사람이 그은 선에서 그 자리는
그냥 지나가는 자리다. 실측(기준판 11장) 접합점의 **33%가 끊겨 있었다**.

**남는 것을 깁지 않고 만나게 한다.** 이 단은 레이어를 한 장도 안 산다 —
이미 놓인 끝 마디를 게임 격자 한 칸씩 밀거나 한 스텝 키워 창 안의 잉크가
한 덩이가 되게 할 뿐이다.

자는 창 하나에서 읽는다 (라스터 = 인게임이 그리는 그 폴리곤):

    끊김   창 안 잉크의 8이웃 성분 수 (1이 아니면 안 만난 것이다)
    틈     창 안 선 지도 중 잉크가 안 덮은 px
    스필   창 안 잉크 중 허용 밴드 **밖** px (`scoring._PEN_LINE`과 같은 저울)
    뭉침   창 안 최대 두께가 제 획 폭의 `_BULGE`배를 넘은 몫

**창 밖의 손해도 같은 저울에 얹는다.** 창은 작고 마디는 길다 — 창 안에서
좋아지는 수가 제 길이 전체로는 선을 벗어나는 수일 수 있다. 그래서 후보의
비용에 **그 마디가 잃는 제 선 지도 px**을 더한다. 단위가 창 안의 틈과 같아
(선 지도 px) 상수가 늘지 않는다. 이 항 없이 돌린 판은 접합점 틈을
.335 → .309로 줄이는 대신 선 커버리지가 .969 → .965로 내려가고 보이는
오차가 11% 늘었다. 반대로 **손해를 아예 금지**하면(자격으로 걸면) 전역 미세
조정 뒤에는 어느 수도 통과하지 못해 이 단이 무동작이 된다 (실측 11: 이동 0회).

움직이는 축·스텝은 배치의 하강과 같은 게임 격자다 (이동 0.5유닛·스케일 0.01).

**도는 자리는 미세 조정 다음, 봉인 앞이다.** 앞에 두면 전역 미세 조정이 그
자리를 다시 제 목표로 되돌리고, 뒤에 두면 옮긴 잉크가 드러낸 미커버 표본을
봉인이 못 막는다.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

from ..catalog import Catalog
from ..model import Layer
from . import chain
from .descriptor import descriptors
from .geometry import _min_span, _poly_px
from .scoring import _PEN_LINE

# 창 반경 = 이 배 × 그 자리 획 폭 (계측 `linemetrics._J_R`과 같은 자)
_R = float(os.environ.get("FS_JUNC_R", 2.5))
# 이 배 넘게 굵으면 뭉친 것이다 (계측 `linemetrics._J_BULGE`와 같은 자)
_BULGE = float(os.environ.get("FS_JUNC_BULGE", 1.6))
# 끊김 한 갈래의 값 (px 점수) — 틈 px과 같은 눈금. 끊김 하나가 이음 보수
# 도형 1.5장을 부르므로(`candidates._SEAM_PER_BREAK`) 틈 몇 px보다 비싸다
_W_BREAK = float(os.environ.get("FS_JUNC_BREAK", 8.0))
_W_BULGE = float(os.environ.get("FS_JUNC_BULGE_W", 2.0))
_PASSES = int(os.environ.get("FS_JUNC_PASSES", 2))
# 0이면 이 단을 안 돈다 (스윕 스위치)
_ON = float(os.environ.get("FS_JUNC", 1.0))


def _win_of(cat: Catalog, lay: Layer, upp: float, w: int, h: int, pad: int = 2):
    """레이어 폴리곤의 창 (x0, y0, x1, y1) — 화면 밖은 자른다. 빈 창은 None."""
    polys = _poly_px(cat, lay, upp, w, h)
    xs = np.concatenate([p[:, 0] for p in polys])
    ys = np.concatenate([p[:, 1] for p in polys])
    x0 = max(0, int(np.floor(xs.min())) - pad)
    y0 = max(0, int(np.floor(ys.min())) - pad)
    x1 = min(w, int(np.ceil(xs.max())) + pad)
    y1 = min(h, int(np.ceil(ys.max())) + pad)
    return None if x0 >= x1 or y0 >= y1 else (x0, y0, x1, y1)


def _draw(cat: Catalog, lay: Layer, upp: float, w: int, h: int,
          win: tuple[int, int, int, int], m: np.ndarray) -> None:
    """창 안에 이 레이어를 그린다 (렌더와 같은 반올림 폴리곤)."""
    x0, y0 = win[0], win[1]
    polys = [np.round(p).astype(np.int32) - np.array([x0, y0], np.int32)
             for p in _poly_px(cat, lay, upp, w, h)]
    if len(polys) == 1:
        cv2.fillPoly(m, polys, 1)
        return
    one = np.zeros_like(m)
    acc = np.zeros_like(m)
    for p in polys:
        one[:] = 0
        cv2.fillPoly(one, [p], 1)
        acc ^= one
    m |= acc


def _on_line(cat: Catalog, lay: Layer, upp: float, w: int, h: int,
             line_mask: np.ndarray) -> float:
    """이 마디가 덮는 **선 지도 px** — 이동의 자격 (선 충실도)."""
    win = _win_of(cat, lay, upp, w, h)
    if win is None:
        return 0.0
    x0, y0, x1, y1 = win
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    _draw(cat, lay, upp, w, h, win, m)
    return float(np.count_nonzero(m.astype(bool) & line_mask[y0:y1, x0:x1]))


def _cost(full: np.ndarray, part: np.ndarray, need: np.ndarray,
          allow: np.ndarray, wmed: float) -> float:
    """창 하나의 비용 — 끊김 + 틈 + 스필 + 뭉침 (px 점수).

    끊김·뭉침은 **참여한 끝 마디**(`part`)만 본다 — 창 안 잉크 전부를 세면
    지나가는 남의 획이 늘 한 성분 더 잡혀 "이 접합점이 끊겼나"를 못 묻는다
    (계측 `linemetrics._junction_metrics`와 같은 자). 틈·스필은 창 안 잉크
    전부(`full`)로 본다 — 그 자리를 누가 덮든 화면은 같다.
    """
    u = part.astype(np.uint8)
    nc = cv2.connectedComponents(u, connectivity=8)[0] - 1
    c = _W_BREAK * max(0, nc - 1)
    c += float(np.count_nonzero(need & ~full))
    c += _PEN_LINE * float(np.count_nonzero(full & ~allow))
    if wmed > 1e-6 and u.any():
        thick = 2.0 * float(cv2.distanceTransform(u, cv2.DIST_L2, 3).max())
        c += _W_BULGE * max(0.0, thick - _BULGE * wmed)
    return c


def close_gaps(plan, cat: Catalog, upp: float, size: tuple[int, int],
               strokes, line_mask: np.ndarray, bg: np.ndarray | None,
               stats: dict) -> int:
    """접합점마다 끝 마디를 밀어 **만나게 한다** (레이어 0장). 움직인 수."""
    if _ON <= 0.0 or not strokes or line_mask is None:
        return 0
    w, h = size
    # 잉크가 나가도 되는 자리 — 선 밴드 안 (배치가 쓰는 그 자). 면 위는
    # 어차피 선이 모든 면 위에 얹히므로 실루엣 안도 허용이다
    rr = max(1, int(round(2.0 * _min_span(upp))))
    allow = cv2.dilate(line_mask.astype(np.uint8),
                       cv2.getStructuringElement(
                           cv2.MORPH_ELLIPSE, (2 * rr + 1, 2 * rr + 1))
                       ).astype(bool)
    if bg is not None:
        allow = allow | ~bg
    by: dict[int, list[int]] = {}
    for i, l in enumerate(plan.layers):
        if l.label == "ink" and l.stroke >= 0:
            by.setdefault(l.stroke, []).append(i)
    if not by:
        return 0
    dmap = descriptors(cat)
    # 접합점 → [(획, 끝 마디의 레이어 index, 그 끝점, 폭)]
    ends: dict[tuple[int, int], list] = {}
    for s in strokes:
        lays = by.get(s.sid) or []
        if not lays or len(s.path) < 2:
            continue
        for jid, i in ((s.head_j, 0), (s.tail_j, -1)):
            if jid < 0:
                continue
            pt = np.array([s.path[i][1] + s.roi[0], s.path[i][0] + s.roi[1]],
                          np.float64)
            best = None
            for li in lays:
                got = chain.line_of(dmap.get(plan.layers[li].shape),
                                    plan.layers[li], upp, w, h)
                if got is None:
                    continue
                d = min(float(np.hypot(*(got[0] - pt))),
                        float(np.hypot(*(got[1] - pt))))
                if best is None or d < best[0]:
                    best = (d, li)
            if best is not None:
                ends.setdefault((s.comp, int(jid)), []).append(
                    (s.sid, best[1], pt, float(s.width)))
    junc = [(k, v) for k, v in sorted(ends.items())
            if len({e[0] for e in v}) >= 2]
    if not junc:
        return 0
    box = {}
    for i, lay in enumerate(plan.layers):
        got = _win_of(cat, lay, upp, w, h, pad=0)
        if got is not None:
            box[i] = got
    st = 0.5                               # 게임 이동 스텝 (유닛)
    ds = 0.01                              # 게임 스케일 스텝
    moved = 0
    n_open = 0                             # 비용이 남아 있는 접합점
    gains: list[float] = []                # 실제로 받은 개선폭 (계측)
    for _key, v in junc:
        pt = np.mean([e[2] for e in v], axis=0)
        wmed = float(np.median([e[3] for e in v]))
        r = max(4, int(round(_R * max(wmed, 1.0))))
        x0 = max(0, int(pt[0]) - r); x1 = min(w, int(pt[0]) + r + 1)
        y0 = max(0, int(pt[1]) - r); y1 = min(h, int(pt[1]) + r + 1)
        if x1 - x0 < 3 or y1 - y0 < 3:
            continue
        win = (x0, y0, x1, y1)
        hit = [i for i, b in box.items()
               if b[0] <= x1 and b[2] >= x0 and b[1] <= y1 and b[3] >= y0]
        if not hit:
            continue
        need = line_mask[y0:y1, x0:x1]
        ok = allow[y0:y1, x0:x1]
        movable = sorted({e[1] for e in v})
        # 안 움직이는 이웃 잉크는 창마다 **한 번만** 그린다 — 후보마다 다시
        # 그리면 그 비용이 이 단의 전부가 된다
        fixed = np.zeros((y1 - y0, x1 - x0), np.uint8)
        for i in hit:
            if i not in movable:
                _draw(cat, plan.layers[i], upp, w, h, win, fixed)

        def ink_of(over: dict, _fx=fixed, _win=win, _mv=movable):
            """(창 안 잉크 전부, 참여 마디의 잉크)."""
            part = np.zeros_like(_fx)
            for i in _mv:
                _draw(cat, over.get(i, plan.layers[i]), upp, w, h, _win, part)
            return (_fx | part).astype(bool), part.astype(bool)

        cur = _cost(*ink_of({}), need, ok, wmed)
        if cur <= 0.0:
            continue
        n_open += 1
        for _p in range(_PASSES):
            best = None
            for li in movable:
                lay = plan.layers[li]
                keep = _on_line(cat, lay, upp, w, h, line_mask)
                toward = pt - np.array([lay.x / upp + w / 2.0,
                                        h / 2.0 - lay.y / upp])
                nrm = float(np.hypot(*toward))
                tw = toward / nrm if nrm > 1e-9 else np.zeros(2)
                # **긴 축만 키운다** — 짧은 축은 곧 선 굵기다 (`holes._growable`
                # 이 획에 거는 그 규칙). 스케일은 중심 대칭이라 반대쪽도 자라
                # 므로, 키우기에 **접합점 쪽 한 칸 이동**을 짝지어 한쪽으로만
                # 자라는 수도 함께 물어본다 (`layered.grow_fill`의 그 짝)
                gx, gy = (ds, 0.0) if abs(lay.sx) >= abs(lay.sy) else (0.0, ds)
                mx, my = st * tw[0], -st * tw[1]
                cands = [(st, 0.0, 0.0, 0.0), (-st, 0.0, 0.0, 0.0),
                         (0.0, st, 0.0, 0.0), (0.0, -st, 0.0, 0.0),
                         (0.0, 0.0, gx, gy), (0.0, 0.0, ds, ds),
                         (mx, my, 0.0, 0.0), (mx, my, gx, gy)]
                for dx, dy, dsx, dsy in cands:
                    q = Layer(**{**lay.__dict__})
                    q.x = lay.x + dx
                    q.y = lay.y + dy
                    q.sx = lay.sx + (dsx if lay.sx >= 0 else -dsx)
                    q.sy = lay.sy + (dsy if lay.sy >= 0 else -dsy)
                    if abs(q.sx) < 0.01 or abs(q.sy) < 0.01:
                        continue
                    q = q.quantized()
                    # **제 선 지도 손해를 같은 저울에 얹는다.** 창 안의 틈과
                    # 같은 단위(선 지도 px)라 상수가 안 늘어난다 — 창이 못 보는
                    # 마디 나머지의 손해를 여기서 본다
                    c = _cost(*ink_of({li: q}), need, ok, wmed)
                    c += max(0.0, keep - _on_line(cat, q, upp, w, h, line_mask))
                    if c < cur - 1e-9 and (best is None or c < best[0]):
                        best = (c, li, q)
            if best is None:
                break
            gains.append(cur - best[0])
            cur, li, q = best
            plan.layers[li] = q
            got = _win_of(cat, q, upp, w, h, pad=0)
            if got is not None:
                box[li] = got
            moved += 1
            if cur <= 0.0:
                break
    stats["junction_moves"] = moved
    stats["junctions"] = len(junc)
    stats["junction_open"] = n_open
    if gains:
        stats["junction_gain_med"] = round(float(np.median(gains)), 2)
    return moved
