"""덮어서 그리기 — 굵게 긋고 면 색 도형으로 도로 덮어 **가는** 선을 만든다.

캔버스가 긴 변 900유닛 고정이라 최소 도형(스케일 0.01)의 폭이 1.71~4.34px다.
1~2px 선은 안 그리면 틈이고 그리면 얼룩이라 굵기로는 못 푼다 — 남는 자리는
**이미 그려진 픽셀 위**다. 덮개는 제 획과 같은 도형·같은 크기를 제 얇은 축으로
이동 스텝만큼 민 사본이라 새 자유도도 새 어휘도 안 쓴다 (`_carve_lines` 문서).
"""

from __future__ import annotations

import numpy as np

from ...i18n import msg
from ..catalog import Catalog
from ..celart import CelArt
from ..model import UNITS_PER_SCALE, Layer, LayerPlan
from .geometry import _mask_px, _poly_px


def _carve_lines(plan: LayerPlan, cel: CelArt, cat: Catalog, upp: float,
                 left: int, log=print, floor: float = 0.0,
                 floor_lo: float = 0.0, defer: list | None = None) -> int:
    """**덮어서 그리기** — 획을 굵게 긋고 면 색 도형으로 도로 덮어, 양자화
    최소 도형보다 **가는** 선을 만든다 (사람이 쓰는 문법).

    캔버스는 긴 변 900유닛 고정이라 최소 도형(스케일 0.01)의 폭이 px로는
    1.71~4.34px이다 (증거 3). 1~2px 선을 그으면 나머지가 이웃 면으로
    삐져나가 얼룩이 된다 — 안 그리면 틈이고 그리면 얼룩이라 굵기로는 못
    푼다. 남은 자리는 **이미 그려진 픽셀 위**다: 삐져나간 자리를 **면 색 도형 한 장으로 도로 덮으면**
    남는 초승달이 곧 선이고, 그 폭은 최소 도형이 아니라 **이동 스텝(0.5유닛)**
    으로 정해진다. 덮개의 경계가 도형 경계라 선이 매끈하게 이어지는 것은 덤이다.

    덮개는 제 획과 **같은 도형·같은 크기를 제 얇은 축으로 밀어 놓은 사본**이다.
    새 자유도도 새 어휘도 안 쓴다 — 획이 [-t/2, t/2]를 덮을 때 d만큼 민 사본은
    [d-t/2, d+t/2]를 덮으므로 남는 선폭이 정확히 **d**가 된다. d는 이동
    스텝의 배수라 상수가 아니라 게임 입력에서 나온다.

    채점은 굵기가 아니라 **삐져나갈 자리의 색차**다 (지난 판정이 남긴 잣대):

    ```
    이득 = 덮개가 덮는 px 중 이 획이 지금 칠하고 있는 자리의 오차
    비용 = 덮개가 칠한 뒤의 오차                      (둘 다 셀(flat)과의 채널 거리)
    덮는다 ⟺ 이득 − 비용 > 0                          (상수를 안 늘린다)
    ```

    `flat_render()`는 **선 픽셀에 원화 색을 도로 얹는다** — 그래서 덮개가 선
    자리를 먹으면 비용이 저절로 붙고, 이웃 면 위로 나간 만큼은 그 면 색을
    되돌리므로 공짜다. 배경으로 나가면 흰색과의 거리라 역시 저절로 막힌다.
    "같은 색이면 공짜"라는 지렛대가 여기서는 **성립한다** — 셀 경계와 달리
    덮는 자리가 이미 그려진 픽셀이기 때문이다.

    덮개는 제 획 **바로 뒤**에 끼운다: 뒤에 오는 획은 덮개 위에 그려지므로
    다른 획을 먹지 않는다. `stroke` 그룹 id와 `ink` 라벨을 제 획에서 물려받아
    프루닝이 **획과 덮개를 함께** 살리거나 버린다 (덮개만 잘리면 선이 도로
    굵어진다).
    """
    if left <= 0 or cel.line_mask is None:
        return 0
    flat = cel.flat_render().astype(np.int16)
    n = _carve_range(plan, cel.line_mask, flat, cat, upp, cel.size, left, 0,
                     floor, floor_lo=floor_lo, defer=defer)
    if n:
        log(msg("  획 덮개 {n}장 (덮어서 그리기 — 최소 도형보다 가는 선)", n=n))
    return n


def _carve_range(plan: LayerPlan, lm: np.ndarray, flat: np.ndarray,
                 cat: Catalog, upp: float, size: tuple[int, int],
                 left: int, lo: int, floor: float = 0.0,
                 floor_lo: float = 0.0, defer: list | None = None) -> int:
    """`plan.layers[lo:]`의 획마다 덮개를 시도해 **제 획 바로 뒤**에 끼운다.

    `defer`를 주면 **2단 가격**이다 (2단 잔차 수리와 같은 무늬): 순이득이
    `floor`를 넘는 덮개는 지금 사고, `floor_lo`~`floor` 구간은 (순이득, 앵커,
    덮개)로 `defer`에 미룬다 — 배치가 끝나 예산 잔여가 확정된 뒤(파이프라인)
    남는 만큼만 순이득 순으로 산다. 덮개의 채점(획 색↔셀 목표)은 다른
    레이어와 무관해 늦게 사도 값이 그대로다. 앵커는 덮개가 **바로 뒤에**
    끼어야 하는 레이어다 — 첫째 덮개가 유예되면 둘째는 버린다 (둘째의 채점은
    첫째가 이미 그려진 상태를 전제한다).
    """
    if left <= 0 or lo >= len(plan.layers):
        return 0
    w, h = size
    out: list[Layer] = []
    n = 0
    ask = floor_lo if defer is not None and floor_lo < floor else floor
    for lay in plan.layers[lo:]:
        out.append(lay)
        if n >= left or lay.label != "ink" or lay.alpha < 100.0:
            continue
        anchor = lay
        for j, (net, cov) in enumerate(
                _carve_one(cat, lay, lm, flat, upp, w, h, ask, both=True)):
            if net > floor:
                if n >= left:
                    break
                out.append(cov)
                n += 1
                anchor = cov
            elif defer is not None:
                defer.append((net, anchor, cov))
                break                     # 첫째가 유예면 둘째는 성립 안 한다
    if n:
        plan.layers[lo:] = out
    return n


def _carve_one(cat: Catalog, lay: Layer, keep: np.ndarray, flat: np.ndarray,
               upp: float, w: int, h: int, floor: float = 0.0,
               both: bool = False) -> list[tuple[float, Layer]]:
    """레이어 하나의 덮개 — (순이득, 레이어) 목록 (없으면 빈 목록).

    `keep` = 이 레이어가 **계속 칠해야 하는** px (획이면 선 지도, 메움이면 구멍
    px). 두 자리에만 쓴다: ① 덮개가 keep 밖으로 안 나가면 덮을 것이 없다
    ② 덮개 색은 keep 밖에서 표본한다. **자르는 것은 keep이 아니라 순이득**이다
    — keep을 먹으면 `flat`이 거기에 원래 색을 들고 있으므로 비용이 저절로 붙는다.

    `both=True`면 첫 덮개의 **반대쪽**에 둘째 덮개를 시도한다 (양쪽으로 삐져
    나간 획은 한쪽 덮개로는 반이 남는다). 둘째는 첫째를 모르고 채점하면 안
    된다 — 겹치는 자리는 이미 첫 덮개 색이라, 획 색 기준 이득은 이중 계상이고
    두 덮개가 획 중심을 다 먹으면 선이 통째로 지워진다. 그래서
    ① 채점은 **첫 덮개가 반영된 상태**로 한다 (겹침 px의 현재 색 = 첫 덮개 색)
    ② 두 밀림의 합이 (굵기 + 이동 스텝)보다 작으면 후보에서 뺀다 — 남는
       선폭이 스텝 미만이면 선이 아니라 지우개다.
    """
    step = 0.5                            # 게임 이동 스텝(유닛) — 상수가 아니다
    thick = 2.0 * abs(lay.sy) * UNITS_PER_SCALE      # 유닛
    kmax = min(8, int(thick / step))
    if kmax < 1:
        return []
    pad = int(thick / upp) + 3
    polys = _poly_px(cat, lay, upp, w, h)
    xs = np.concatenate([p[:, 0] for p in polys])
    ys = np.concatenate([p[:, 1] for p in polys])
    x0 = max(0, int(xs.min()) - pad); y0 = max(0, int(ys.min()) - pad)
    x1 = min(w, int(xs.max()) + pad + 1); y1 = min(h, int(ys.max()) + pad + 1)
    if x0 >= x1 or y0 >= y1:
        return []
    roi = (x0, y0, x1, y1)
    m0 = _mask_px(cat, lay, upp, w, h, roi)
    lmw = keep[y0:y1, x0:x1]
    if not (m0 & ~lmw).any():             # keep 밖으로 안 나갔다 — 덮을 것이 없다
        return []
    fw = flat[y0:y1, x0:x1]
    err0 = np.abs(np.asarray(lay.rgb(), np.int16) - fw).sum(2)
    rot = np.radians(lay.rot)
    ux, uy = -np.sin(rot), np.cos(rot)               # 도형의 얇은 축 (캔버스)

    def _score(sgn: float, k: int, err_cur: np.ndarray, gainable: np.ndarray,
               taken: np.ndarray | None):
        """밀림 sgn·k의 덮개 후보 채점 — (순이득, 레이어, 마스크) 또는 None.

        `err_cur` = 지금 화면의 px별 오차 (첫 덮개가 있으면 그 색 반영),
        `gainable` = 덮으면 이득이 나는 자리 (획 + 이미 선 덮개), `taken` =
        색 표본에서 뺄 자리 (첫 덮개 — 그 색을 다시 표본하면 제자리걸음이다).
        """
        d = sgn * k * step
        cand = Layer(**{**lay.__dict__})
        cand.x = lay.x + d * ux
        cand.y = lay.y + d * uy
        cand = cand.quantized()
        cm = _mask_px(cat, cand, upp, w, h, roi)
        if not cm.any():
            return None
        # 덮개 색은 **덮을 자리의 셀 목표**에서 표본한다 — 획 밖·선
        # 밖(= 되돌릴 면)이 1순위, 없으면 선 밖 전체
        sel = cm & ~lmw & ~m0
        if taken is not None:
            sel &= ~taken
        if not sel.any():
            sel = cm & ~lmw
            if not sel.any():
                return None
        col = np.median(fw[sel], axis=0)
        cand.color = tuple(int(v) for v in col)
        cand = cand.quantized()
        # 채점은 양자화 **뒤**의 값으로 (색은 바이트 정본이라 그대로다)
        gain = float(err_cur[cm & gainable].sum())
        cost = float(np.abs(np.asarray(cand.rgb(), np.int16)
                            - fw)[cm].sum(-1).sum())
        return gain - cost, cand, cm

    # `floor`(가격 설계) = 덮개도 한 장이므로 λ만큼은 벌어야 산다. 0이면
    # 게이트는 순이득 > 0이다
    best_net, best = floor, None
    best_sgn, best_k, best_cm = 0.0, 0, None
    for sgn in (1.0, -1.0):
        for k in range(1, kmax + 1):
            got = _score(sgn, k, err0, m0, None)
            if got is not None and got[0] > best_net:
                best_net, best = got[0], got[1]
                best_sgn, best_k, best_cm = sgn, k, got[2]
    if best is None:
        return []
    out = [(best_net, best)]
    if not both:
        return out
    # 둘째 덮개 — 반대쪽만, 첫째가 반영된 상태로 (docstring ①②)
    err1 = np.where(best_cm,
                    np.abs(np.asarray(best.rgb(), np.int16) - fw).sum(2), err0)
    best2_net, best2 = floor, None
    for k in range(1, kmax + 1):
        if (best_k + k) * step < thick + step:
            continue                      # 남는 선폭 < 이동 스텝 — 지우개다
        got = _score(-best_sgn, k, err1, m0 | best_cm, best_cm)
        if got is not None and got[0] > best2_net:
            best2_net, best2 = got[0], got[1]
    if best2 is not None:
        out.append((best2_net, best2))
    return out
