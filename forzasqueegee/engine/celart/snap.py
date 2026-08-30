"""§10 획 라스터에 스냅 — 색 경계를 **그은 선 밑**에 앉힌다.

배치된 획(`ink`)을 r만큼 팽창한 밴드 안의 픽셀은 밴드 **밖에서 걸어서 가장
가까운** 픽셀의 라벨을 받는다. 이것이 사용자 제약(2026-08-26)을 기하로
만든다:

- 다른 색 영역의 경계가 획 아래(밴드 중앙선)에 선다 — 채움이 제 몫 밴드까지
  덮으므로 색면과 획 사이에 틈이 없고, 획 너머로는 다른 색이 안 나간다
  (**빈 공간 금지 · 선 침범 금지**).
- 분해 경계가 획에서 1~2px 어긋난 자리의 반대색 슬리버 — "색이 선을 넘은"
  것으로 보이는 실체 — 가 지워진다.
- 같은 영역이 획을 가로지르면 같은 라벨이 밴드 양쪽에서 이어진다.

**자가 유클리드가 아니라 측지다.** 직선 자는 가는 구조를 뛰어넘어, 밴드
안쪽 픽셀이 걸어서는 닿을 수 없는 건너편 영역의 라벨을 받는다 (손가락 사이·
머리칼 가닥 틈). 그러면 획 밑에서 만나는 두 라벨이 그래프상 이웃이 아니게
되고, 그 자리가 곧 "색이 선을 넘어간" 자국이다. 걸어서 재면 밴드 밑에서
만나는 라벨은 **정의상 인접한 두 면**이고, 같은 영역이 선을 가로지르면
하나의 정체성이 밴드 밑에서 이어진다.

동률은 **넓은 영역이 가져간다** (`geodesic.propagate`에 넓이 내림차순 순위를
준다) — 결정적이다.

r는 상수가 아니라 게임 격자다 (호출부: 최소 도형 반폭 ≈ 1px@1200).

**색 가드 (X7 검수).** 밴드를 지우고 밖에서 걸어 들어오면, 밴드 안에**만**
살던 구조가 통째로 이웃에게 넘어간다 — 선의 **주제**(어두운 끈·옷단)가
그렇다: 선 지도보다 획이 가늘거나 어긋난 자리마다 양쪽의 밝은 면이 이어져
어두운 선 위에 흰 점선이 찍히고(X7-01 #2), 옷 경계 밴드가 살색으로 넘어간다
(X7-06 #3·#4, 실측: 결함 px의 대부분이 이 스냅에서 생겼다). 그래서 **선
지도가 덮지 않아 눈에 보일 픽셀**은 원화색을 물어 본다 — 새 라벨의 영역색이
옛 라벨보다 Lab로 `guard` 이상 나쁘면 안 바꾼다. 선 지도 밑(어차피 선이
덮는다)은 그대로 스냅해 경계가 획 중앙선에 서는 본기능을 지킨다.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

from .geodesic import propagate
from .model import CelArt, Region

# 보이는 픽셀의 라벨 교체 허용 한계 (Lab ΔE, 새 라벨 적합도 − 옛 라벨 적합도).
# 15는 X7 실측: 표시 22곳의 "어두운 곳→밝은 라벨" 생성 0, 슬리버 지우기는
# 선 지도 밑과 적합도 비슷한 교체로 살아남는다
_SNAP_GUARD = float(os.environ.get("FS_SNAP_GUARD", 15.0))


def snap_labels_to_ink(labels: np.ndarray, sel: np.ndarray, ink: np.ndarray,
                       r: int, *, src: np.ndarray | None = None,
                       colors: np.ndarray | None = None,
                       visible: np.ndarray | None = None,
                       guard: float = _SNAP_GUARD) -> np.ndarray:
    """영역 라벨을 **배치된 획 라스터**에 스냅한다 — 채움 목표를 만든다.

    `src`(RGB 원화)·`colors`(rid→영역색 LUT)·`visible`(선 지도가 안 덮는
    자리)이 다 오면 색 가드가 켜진다 (모듈 문서). 하나라도 없으면 종전 그대로.
    """
    if not ink.any():
        return labels
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    zone = cv2.dilate(ink.astype(np.uint8), k).astype(bool) & sel
    keep = sel & ~zone
    if not zone.any() or not keep.any():
        return labels
    n = int(labels.max()) + 1 if labels.max() >= 0 else 0
    seed = np.where(keep, labels, -1).astype(np.int32)
    area = np.bincount(labels[keep].ravel(), minlength=max(n, 1)).astype(np.int64)
    order = np.empty(max(n, 1), np.int64)
    order[np.argsort(-area, kind="stable")] = np.arange(max(n, 1))
    got, _ = propagate(seed, zone, order=order)
    out = labels.copy()
    fill = zone & (got >= 0)
    if src is not None and colors is not None and visible is not None:
        chg = fill & visible & (got != labels) & (labels >= 0)
        if chg.any():
            ys, xs = np.nonzero(chg)

            def _lab(rgb):
                return cv2.cvtColor(rgb.reshape(-1, 1, 3).astype(np.uint8),
                                    cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)

            clab = _lab(np.ascontiguousarray(colors))
            pxl = _lab(src[ys, xs])
            d_new = np.linalg.norm(pxl - clab[got[ys, xs]], axis=1)
            d_old = np.linalg.norm(pxl - clab[labels[ys, xs]], axis=1)
            bad = (d_new - d_old) > guard
            fill[ys[bad], xs[bad]] = False
    out[fill] = got[fill]
    return out


def rebuild_regions(labels: np.ndarray, regions: list[Region]) -> list[Region]:
    """스냅 뒤의 영역 표 재계산 — 색은 유지, 면적·bbox만 다시 잰다.

    스냅으로 빈 영역(획 밴드 안에만 살던 슬리버)은 표에서 빠진다 — 그 자리는
    이웃 라벨이 받았고 위를 획이 덮는다. 순서는 넓이 내림차순(그리기 순서).
    """
    sel = labels >= 0
    color_of = {r.rid: r.color for r in regions}
    ids, inv, counts = np.unique(labels[sel], return_inverse=True,
                                 return_counts=True)
    ys, xs = np.nonzero(sel)
    h, w = labels.shape
    x0 = np.full(len(ids), w, np.int64)
    x1 = np.zeros(len(ids), np.int64)
    y0 = np.full(len(ids), h, np.int64)
    y1 = np.zeros(len(ids), np.int64)
    np.minimum.at(x0, inv, xs)
    np.maximum.at(x1, inv, xs)
    np.minimum.at(y0, inv, ys)
    np.maximum.at(y1, inv, ys)
    out = [Region(rid=int(ids[i]), color=color_of[int(ids[i])],
                  area=int(counts[i]),
                  bbox=(int(x0[i]), int(y0[i]), int(x1[i]) + 1, int(y1[i]) + 1))
           for i in range(len(ids))]
    out.sort(key=lambda r: -r.area)
    return out


def region_table(labels: np.ndarray, sm: np.ndarray, sel: np.ndarray,
                 w: int, h: int) -> list[Region]:
    """영역 표 (넓이 내림차순 = 그리기 순서). 대표색은 평활 이미지의 영역 평균."""
    ids, inv, counts = np.unique(labels[sel], return_inverse=True,
                                 return_counts=True)
    sums = np.zeros((len(ids), 3), np.float64)
    np.add.at(sums, inv, sm[sel].astype(np.float64))
    ys, xs = np.nonzero(sel)
    x0 = np.full(len(ids), w, np.int64)
    x1 = np.zeros(len(ids), np.int64)
    y0 = np.full(len(ids), h, np.int64)
    y1 = np.zeros(len(ids), np.int64)
    np.minimum.at(x0, inv, xs)
    np.maximum.at(x1, inv, xs)
    np.minimum.at(y0, inv, ys)
    np.maximum.at(y1, inv, ys)
    regions = [
        Region(rid=int(ids[i]),
               color=tuple(int(round(v)) for v in sums[i] / counts[i]),
               area=int(counts[i]),
               bbox=(int(x0[i]), int(y0[i]), int(x1[i]) + 1, int(y1[i]) + 1))
        for i in range(len(ids))
    ]
    regions.sort(key=lambda r: -r.area)   # 큰 면 먼저 = 사람 순서
    return regions


def with_regions(labels: np.ndarray, sm: np.ndarray, sel: np.ndarray,
                 **kw) -> CelArt:
    """라벨 지도 하나로 CelArt 짓기 (표까지)."""
    h, w = labels.shape
    return CelArt(size=(w, h), labels=labels,
                  regions=region_table(labels, sm, sel, w, h), **kw)
