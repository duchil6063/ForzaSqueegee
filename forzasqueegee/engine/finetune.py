"""cel 노선 마무리 — 전역 미세 조정 (DiffCompositing 개념의 이산 이식).

celfit 배치는 영역별 그리디다: 채점판이 자기 영역 ROI만 보고, 나중 면이
실제로 어디를 덮는지는 벌점 근사(_PEN_WASTE)로만 안다. 이 패스는 **완성된
스택 전체**를 놓고 레이어 하나씩 게임 양자화 스텝(이동 0.5·스케일 0.01·회전
0.1°) 이웃으로 좌표하강시킨다. 전 레이어 불투명이라 픽셀 색 = 그 픽셀을
덮는 최상위 레이어 색 — "레이어 i를 움직이면 정확히 어떤 px가 어떤 색으로
바뀌는지"를 증분으로 계산할 수 있고, 목표 대비 제곱 오차가 줄어드는 이동만
받는다. DiffCompositing(SIGGRAPH Asia 2020)의 "합성 스택 전체를 미분해
요소를 미세 조정"을, 고정 도형·양자화 파라미터 어휘에 맞게 기울기 대신
양자화 이웃 탐색으로 옮긴 것이다.

목표 이미지 = 셀 평면 렌더 (선화 px는 원화 색 — flat_render가 이미 그렇다).
배치(celfit)·선화(_fit_lines)가 각자 겨눴던 목표의 합성이라, 채움 경계와
선화 획이 같은 잣대 위에서 함께 미세 조정된다.

불변 보장:
- 레이어 수·순서·색·도형 불변 — 기하(x·y·sx·sy·rot)만 움직인다 (skew는
  주입이 안 쓰므로 건드리지 않는다).
- 실루엣 px를 새로 노출하는 이동은 **무조건 기각** — holes 게이트(4px+
  군집 0)가 패스 후에도 성립한다.
- 캔버스 rect 밖 돌출을 늘리는 이동도 기각 — 밖은 채점이 못 보는 무벌점
  구간이라 열어 두면 경계 레이어가 그리로 샌다 (celfit 가드와 같은 논리).
- 결정적 (난수 없음, 순서 고정).
"""

from __future__ import annotations

import cv2
import numpy as np

from .catalog import Catalog
from .celart import CelArt
from .celfit import _poly_px
from .model import Layer, LayerPlan

# 비용 상수 (제곱 RGB 오차 단위, 채널당 최대 255² × 3 = 195,075)
# 실루엣 밖(캔버스) 침범 px당 상수 — 인게임 배경은 흰 프리뷰가 아니라 차체
# 도색이라, 밝은 색 스필이 "흰 목표와 가깝다"고 싸지면 안 된다 (pruneplan
# _BG_PEN과 같은 논리의 제곱 오차판)
_P_BG = 4000.0
# 실루엣 미커버(구멍) px당 상수 바닥 — 기존 1~3px 반점을 덮는 성장에 상을
# 주고, 밝은 셀 색 위 핀홀도 색 불문 비용이 되게 한다. 신규 노출은 어차피
# 기각이라 이 값은 "기존 구멍을 메우는 이득"으로만 작동한다
_P_HOLE = 8000.0
_EPS = 1.0            # 이보다 못 버는 이동은 무시 (부동소수 채터 방지)
_MAX_WALK = 8         # 같은 방향 연속 스텝 상한

# (속성, 게임 양자화 스텝) — 짧은 스텝 먼저 (대부분의 개선은 반 스텝이다)
_AXES = (("x", 0.5), ("y", 0.5), ("x", 1.0), ("y", 1.0),
         ("sx", 0.01), ("sy", 0.01), ("rot", 0.1), ("rot", 0.4))


def _win_mask(cat: Catalog, lay: Layer, upp: float, w: int, h: int
              ) -> tuple[np.ndarray, np.ndarray, float]:
    """레이어 → (bbox [x0,y0,x1,y1], 창 마스크(bool), 캔버스 밖 돌출 px).

    래스터는 count_hole_clusters·celfit과 같은 식(round → fillPoly, 짝홀 XOR)
    — 게이트가 보는 커버리지와 이 패스의 소유자 모델이 같은 픽셀을 본다.
    """
    polys = _poly_px(cat, lay, upp, w, h)
    xs = np.concatenate([p[:, 0] for p in polys])
    ys = np.concatenate([p[:, 1] for p in polys])
    ext = float(max(0.0, -xs.min(), xs.max() - w, -ys.min(), ys.max() - h))
    x0 = max(0, int(np.floor(xs.min())) - 1)
    y0 = max(0, int(np.floor(ys.min())) - 1)
    x1 = min(w, int(np.ceil(xs.max())) + 2)
    y1 = min(h, int(np.ceil(ys.max())) + 2)
    if x0 >= x1 or y0 >= y1:
        return np.array([0, 0, 0, 0], np.int32), np.zeros((0, 0), bool), ext
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    off = np.array([x0, y0], np.int32)
    rp = [np.round(p).astype(np.int32) - off for p in polys]
    if len(rp) == 1:
        cv2.fillPoly(m, rp, 1)
    else:
        for p in rp:
            mm = np.zeros_like(m)
            cv2.fillPoly(mm, [p], 1)
            m ^= mm
    return np.array([x0, y0, x1, y1], np.int32), m.astype(bool), ext


def refine_plan(plan: LayerPlan, cel: CelArt, cat: Catalog, *,
                log=print, progress=None, max_passes: int = 3,
                only: list[int] | None = None, tag: str = "전역") -> dict:
    """플랜을 제자리에서 미세 조정. 반환: 통계 딕셔너리 (report용).

    `only`를 주면 **그 레이어들만** 민다 (§12 잔차 초점 패스). 소유자 모델은
    여전히 스택 전체를 보므로 판정은 같고, 훑는 대상만 좁아진다 — 잔차가
    남은 자리를 **보정 도형을 사기 전에** 기존 도형의 이동·스케일·회전으로
    먼저 고치는 자리다.
    """
    layers = plan.layers
    n = len(layers)
    w, h = cel.size
    upp = plan.units_per_px
    for l in layers:
        if l.mask or l.alpha < 99.5 or cat[l.shape].gradient is not None:
            log("  경고: 전역 미세 조정 생략 — 불투명 소유자 모델 밖 레이어 포함")
            return {"moved_layers": 0, "accepts": 0, "skipped": True}

    tgt = cel.flat_render().astype(np.int32)          # 목표 (선화 px = 원화 색)
    sil = cel.labels >= 0
    lut = np.full((n + 1, 3), 255, np.int32)          # [0] = 배경 흰색
    for i, l in enumerate(layers):
        lut[i + 1] = l.rgb()
    # 미커버 px 비용: 흰 노출 오차 + 실루엣 안이면 색 불문 바닥
    ucost = ((255 - tgt) ** 2).sum(2).astype(np.float64)
    ucost[sil] += _P_HOLE

    boxes = np.zeros((n, 4), np.int32)
    masks: list[np.ndarray] = [None] * n              # type: ignore[list-item]
    exts = np.zeros(n, np.float64)
    owner = np.full((h, w), -1, np.int32)             # 최상위 레이어 (-1 = 배경)
    for i, l in enumerate(layers):
        boxes[i], masks[i], exts[i] = _win_mask(cat, l, upp, w, h)
        x0, y0, x1, y1 = boxes[i]
        owner[y0:y1, x0:x1][masks[i]] = i

    def cost_at(ys: np.ndarray, xs: np.ndarray, o: np.ndarray) -> np.ndarray:
        """픽셀들의 비용 — 소유자 o(-1 = 미커버) 기준."""
        d = ((tgt[ys, xs] - lut[o + 1]) ** 2).sum(1).astype(np.float64)
        cov = o >= 0
        d[cov & ~sil[ys, xs]] += _P_BG
        unc = ~cov
        if unc.any():
            d[unc] = ucost[ys[unc], xs[unc]]
        return d

    def unders(i: int, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
        """i가 비켜난 픽셀들이 드러내는 소유자 (i 아래 최상위, 없으면 -1)."""
        u = np.full(len(ys), -1, np.int32)
        dy0, dy1 = int(ys.min()), int(ys.max())
        dx0, dx1 = int(xs.min()), int(xs.max())
        js = np.flatnonzero((boxes[:i, 0] <= dx1) & (boxes[:i, 2] > dx0)
                            & (boxes[:i, 1] <= dy1) & (boxes[:i, 3] > dy0))
        unres = np.ones(len(ys), bool)
        for j in js[::-1]:
            if not unres.any():
                break
            bx0, by0, bx1, by1 = boxes[j]
            sel = (unres & (ys >= by0) & (ys < by1)
                   & (xs >= bx0) & (xs < bx1))
            if not sel.any():
                continue
            idx = np.flatnonzero(sel)
            hit = masks[j][ys[idx] - by0, xs[idx] - bx0]
            idx = idx[hit]
            u[idx] = j
            unres[idx] = False
        return u

    def try_move(i: int, cand: Layer):
        """이동 평가 — (Δ비용, 커밋 데이터) 또는 None(기각)."""
        box_n, m_n, ext_n = _win_mask(cat, cand, upp, w, h)
        if ext_n > exts[i] + 0.5:          # 캔버스 밖 돌출 증가 — 무벌점 구간
            return None
        box_o, m_o = boxes[i], masks[i]
        ux0 = int(min(box_o[0], box_n[0])); uy0 = int(min(box_o[1], box_n[1]))
        ux1 = int(max(box_o[2], box_n[2])); uy1 = int(max(box_o[3], box_n[3]))
        if ux0 >= ux1 or uy0 >= uy1:
            return None
        mo = np.zeros((uy1 - uy0, ux1 - ux0), bool)
        if m_o.size:
            mo[box_o[1] - uy0:box_o[3] - uy0, box_o[0] - ux0:box_o[2] - ux0] = m_o
        mn = np.zeros_like(mo)
        if m_n.size:
            mn[box_n[1] - uy0:box_n[3] - uy0, box_n[0] - ux0:box_n[2] - ux0] = m_n
        ow = owner[uy0:uy1, ux0:ux1]
        add = mn & ~mo & (ow < i)          # 새로 덮고, 위에 아무도 없는 px
        rem = mo & ~mn & (ow == i)         # 비켜나 드러나는 px
        delta = 0.0
        ysA, xsA = np.nonzero(add)
        ysD, xsD = np.nonzero(rem)
        if not len(ysA) and not len(ysD):
            return None
        u = np.zeros(0, np.int32)
        if len(ysD):
            ysD = ysD + uy0; xsD = xsD + ux0
            u = unders(i, ysD, xsD)
            if bool(((u < 0) & sil[ysD, xsD]).any()):
                return None                # 실루엣 신규 노출 = 새 구멍 — 기각
            cur = cost_at(ysD, xsD, np.full(len(ysD), i, np.int32))
            delta += float(cost_at(ysD, xsD, u).sum() - cur.sum())
        if len(ysA):
            ysA = ysA + uy0; xsA = xsA + ux0
            cur = cost_at(ysA, xsA, ow[add])
            delta += float(cost_at(ysA, xsA,
                                   np.full(len(ysA), i, np.int32)).sum()
                           - cur.sum())
        return delta, (ysA, xsA, ysD, xsD, u, box_n, m_n, ext_n)

    def commit(i: int, cand: Layer, data) -> None:
        ysA, xsA, ysD, xsD, u, box_n, m_n, ext_n = data
        if len(ysD):
            owner[ysD, xsD] = u
        if len(ysA):
            owner[ysA, xsA] = i
        boxes[i], masks[i], exts[i] = box_n, m_n, ext_n
        layers[i] = cand

    accepts = 0
    moved = np.zeros(n, bool)
    gain = 0.0
    todo = list(range(n)) if only is None else [i for i in only if 0 <= i < n]
    if not todo:
        return {"moved_layers": 0, "accepts": 0, "cost_gain": 0.0}
    for p in range(max_passes):
        pass_accepts = 0
        for si, i in enumerate(todo):
            if progress and si % 250 == 0:
                progress((p + si / len(todo)) / max_passes,
                         f"미세 조정 {p + 1}/{max_passes}")
            for name, step in _AXES:
                for sign in (1.0, -1.0):
                    for _ in range(_MAX_WALK):
                        lay = layers[i]
                        v = getattr(lay, name) + sign * step * (
                            1.0 if getattr(lay, name) >= 0
                            or name not in ("sx", "sy") else -1.0)
                        if name in ("sx", "sy") and abs(v) < 0.01:
                            break
                        cand = Layer(**{**lay.__dict__})
                        setattr(cand, name, v)
                        cand = cand.quantized()
                        res = try_move(i, cand)
                        if res is None or res[0] > -_EPS:
                            break
                        commit(i, cand, res[1])
                        gain -= res[0]
                        accepts += 1
                        pass_accepts += 1
                        moved[i] = True
        log(f"  {tag} 미세 조정 패스 {p + 1}: 이동 {pass_accepts}회 수락")
        if pass_accepts < max(8, len(todo) // 100):
            break
    if progress:
        progress(1.0, "미세 조정 완료")
    stats = {"moved_layers": int(moved.sum()), "accepts": accepts,
             "cost_gain": round(gain, 0)}
    log(f"  {tag} 미세 조정: 레이어 {stats['moved_layers']}/{len(todo)}개 이동 "
        f"(수락 {accepts}회)")
    return stats
