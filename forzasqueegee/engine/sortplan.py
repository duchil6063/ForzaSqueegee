"""플랜 안전 정렬 — 같은 도형·색 연속 구간(Y 스탬프 그룹)을 최대화.

원리: 최종 이미지는 "겹치면서 색이 다른" 레이어 쌍의 상대 순서에만 의존한다.
그 쌍들의 원래 순서를 보존하는 임의의 재배열은 렌더 결과가 동일하다 (같은 색
겹침·비겹침 쌍은 순서 무관). 비가환 쌍을 DAG 간선으로 두고, 위상 순서 안에서
직전 레이어와 (도형,색)이 같은 것 → 색이 같은 것 순으로 탐욕 스케줄링한다.

겹침 판정은 저해상 마스크 래스터(긴 변 256px, 1px 팽창 마진) AND.
"""

from __future__ import annotations

import numpy as np

from .catalog import Catalog
from .model import UNITS_PER_SCALE, Layer, LayerPlan

MASK_LONG_SIDE = 256


def _color_key(l: Layer) -> tuple[float, float, float]:
    # 창 조작 색 세션 = 게임 색 입력 UI에 넣는 HSB가 같은 무리 (적용 시점 변환)
    return l.hsb()


def _group_key(l: Layer) -> tuple:
    return (l.shape, _color_key(l), l.mask)


def _polys(l: Layer, plan: LayerPlan, catalog: Catalog) -> list[np.ndarray]:
    """레이어의 변환된 폴리곤들 (이미지 px 좌표, 패딩 전) — 렌더와 같은 변환."""
    sh = catalog[l.shape]
    w, h = plan.image_size
    upp = plan.units_per_px
    rot = np.radians(l.rot)
    c, s = np.cos(rot), np.sin(rot)
    out = []
    for loop in sh.loops:
        pts = loop * np.array([l.sx, l.sy], np.float32) * UNITS_PER_SCALE
        pts = pts @ np.array([[c, s], [-s, c]], np.float32)
        pts += np.array([l.x, l.y], np.float32)
        px = pts[:, 0] / upp + w / 2
        py = h / 2 - pts[:, 1] / upp
        out.append(np.stack([px, py], axis=1))
    return out


def plan_pad_px(plan: LayerPlan, catalog: Catalog) -> int:
    """모든 레이어를 포함하는 이미지 rect 밖 패딩 px.

    경계 밴드 마스크·돌출 도형은 rect 밖에서도 겹침·컷이 유효하므로, 겹침
    래스터와 렌더 검증은 이 패딩을 포함한 전체 범위에서 해야 한다.
    """
    w, h = plan.image_size
    pad = 0.0
    for l in plan.layers:
        for p in _polys(l, plan, catalog):
            pad = max(pad, float(-p[:, 0].min()), float(p[:, 0].max() - w),
                      float(-p[:, 1].min()), float(p[:, 1].max() - h))
    return int(np.ceil(pad)) + 1 if pad > 0 else 0


def _layer_mask(l: Layer, plan: LayerPlan, catalog: Catalog,
                mw: int, mh: int, sc: float, pad: int) -> np.ndarray:
    """레이어 발자국 마스크 (짝홀 구멍 무시 = 외곽 채움, 안전 방향 과대평가)."""
    import cv2
    m = np.zeros((mh, mw), np.uint8)
    for p in _polys(l, plan, catalog):
        poly = np.round((p + pad) * sc).astype(np.int32)
        cv2.fillPoly(m, [poly], 1)
    if m.any():
        m = cv2.dilate(m, np.ones((3, 3), np.uint8))  # 1px 마진
    return m.astype(bool)


def sort_plan(plan: LayerPlan, catalog: Catalog) -> tuple[LayerPlan, dict]:
    """렌더 동일성이 보장되는 범위에서 그룹 연속성을 최대화한 새 플랜 반환."""
    layers = plan.layers
    n = len(layers)
    w, h = plan.image_size
    pad = plan_pad_px(plan, catalog)
    pw, ph = w + 2 * pad, h + 2 * pad
    sc = MASK_LONG_SIDE / max(pw, ph)
    mw, mh = max(1, round(pw * sc)), max(1, round(ph * sc))

    masks = [_layer_mask(l, plan, catalog, mw, mh, sc, pad) for l in layers]
    boxes = []
    for m in masks:
        ys, xs = np.nonzero(m)
        boxes.append(None if len(ys) == 0 else (xs.min(), ys.min(), xs.max(), ys.max()))
    colors = [_color_key(l) for l in layers]

    def overlaps(i: int, j: int) -> bool:
        bi, bj = boxes[i], boxes[j]
        if bi is None or bj is None:
            return False
        if bi[0] > bj[2] or bj[0] > bi[2] or bi[1] > bj[3] or bj[1] > bi[3]:
            return False
        return bool(np.logical_and(masks[i], masks[j]).any())

    # 비가환 쌍 → DAG (i<j, 색 다르고 겹침 = i 먼저).
    # 마스크는 "먼저 그려진 것 전부"를 잘라내므로 색 무관하게 겹치는 모든 쌍과 비가환.
    # 그라데이션 도형(반투명 텍스처)은 같은 색이라도 순서 교환 시 소프트 에지가
    # 하드 채움에 덮이므로 겹치는 모든 쌍과 비가환.
    grad = [catalog[l.shape].gradient is not None for l in layers]
    succ: list[list[int]] = [[] for _ in range(n)]
    indeg = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if (layers[i].mask or layers[j].mask or grad[i] or grad[j]
                    or colors[i] != colors[j]) and overlaps(i, j):
                succ[i].append(j)
                indeg[j] += 1

    # 탐욕 위상 스케줄: 같은 (도형,색) > 같은 색 > 원래 순서
    import heapq
    ready: list[int] = []  # 힙 (원래 인덱스)
    for i in range(n):
        if indeg[i] == 0:
            heapq.heappush(ready, i)
    order: list[int] = []
    last: Layer | None = None
    while ready:
        pick = None
        if last is not None:
            lk, lc = _group_key(last), _color_key(last)
            same_g = [i for i in ready if _group_key(layers[i]) == lk]
            if same_g:
                pick = min(same_g)
            else:
                same_c = [i for i in ready if colors[i] == lc]
                if same_c:
                    pick = min(same_c)
        if pick is None:
            pick = ready[0]
        ready.remove(pick)
        heapq.heapify(ready)
        order.append(pick)
        last = layers[pick]
        for j in succ[pick]:
            indeg[j] -= 1
            if indeg[j] == 0:
                heapq.heappush(ready, j)

    assert len(order) == n, "위상 스케줄 불완전 (사이클 불가인데 발생)"
    new_layers = [layers[i] for i in order]

    def run_stats(ls: list[Layer]) -> tuple[int, int]:
        groups = 1 if ls else 0
        hsb_sessions = 0
        for k, l in enumerate(ls):
            if k > 0 and _group_key(l) != _group_key(ls[k - 1]):
                groups += 1
        for k, l in enumerate(ls):
            first_of_group = k == 0 or _group_key(l) != _group_key(ls[k - 1])
            if first_of_group and (k == 0 or _color_key(l) != _color_key(ls[k - 1])):
                if not l.mask and _color_key(l) != (0.0, 0.0, 1.0):  # 흰색·마스크는 HSB 불필요
                    hsb_sessions += 1
        return groups, hsb_sessions

    g0, h0 = run_stats(layers)
    g1, h1 = run_stats(new_layers)
    stats = {"layers": n, "groups_before": g0, "groups_after": g1,
             "hsb_before": h0, "hsb_after": h1}
    return LayerPlan(source_image=plan.source_image, image_size=plan.image_size,
                     units_per_px=plan.units_per_px, layers=new_layers), stats


def render_equal(a: LayerPlan, b: LayerPlan, catalog: Catalog) -> bool:
    """정렬 전후 렌더 픽셀 동일성 검증 (AA 오차 허용 임계, rect 밖 돌출 포함)."""
    from .render import render_plan
    pad = plan_pad_px(a, catalog)
    ra = render_plan(a, catalog, pad=pad)
    rb = render_plan(b, catalog, pad=pad)
    diff = np.abs(ra.astype(np.int16) - rb.astype(np.int16))
    return bool((diff > 8).mean() < 0.001)  # AA 경계 픽셀만 허용
