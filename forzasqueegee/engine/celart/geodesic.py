"""측지 전파 — **장벽을 넘지 않는** 최근접 이웃 찾기 한 벌.

유클리드 최근접(`cv2.distanceTransformWithLabels`)은 자를 직선으로 대므로 가는
구조를 **뛰어넘는다**: 손가락 사이·머리칼 가닥 틈에서 건너편 색을 끌어온다.
선을 경계로 쓰는 이 노선에서는 그 한 번의 도약이 곧 "색이 선을 넘었다"이다.

여기서는 자를 **화면 위를 걸어서** 댄다. 씨앗에서 8이웃 물결로 번지므로 도달
거리는 체스판 측지 거리이고, 같은 거리에서 갈리면 **우선순위(rank)가 낮은
쪽**이 가져간다 — 호출부가 넓이 내림차순 순위를 주면 "동률은 넓은 면이"가 되고
결과는 시프트 순서와 무관하게 결정적이다.

번지면서 **씨앗 픽셀의 자리**도 함께 나른다 (`src`). 라벨만 나르면 "어느 면에
속하나"는 알아도 "그 면의 어느 색인가"는 모른다 — 선 밑을 그 면 **안쪽의**
색으로 메우는 것이 §1의 요구다.
"""

from __future__ import annotations

import numpy as np

_SHIFTS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


def _min_into(best: np.ndarray, key: np.ndarray) -> None:
    """8이웃 시프트 최소 — `best`에 제자리로 접는다 (경계 감김 없음)."""
    h, w = key.shape
    for dy, dx in _SHIFTS:
        ys0, ys1 = max(0, dy), h + min(0, dy)
        xs0, xs1 = max(0, dx), w + min(0, dx)
        yd0, yd1 = max(0, -dy), h + min(0, -dy)
        xd0, xd1 = max(0, -dx), w + min(0, -dx)
        np.minimum(best[yd0:yd1, xd0:xd1], key[ys0:ys1, xs0:xs1],
                   out=best[yd0:yd1, xd0:xd1])


def propagate(lab: np.ndarray, zone: np.ndarray,
              order: np.ndarray | None = None,
              max_rounds: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """씨앗 라벨을 `zone` 안으로 측지 전파한다.

    `lab`은 int32 (h,w) — 음수가 아직 안 정해진 자리다. 씨앗(≥0)에서 물결이
    번져 `zone` 안의 미정 픽셀을 채운다. `zone` 밖은 절대 안 건드린다(장벽).

    `order`는 라벨 → 우선순위 배열(작을수록 이긴다). 안 주면 라벨 값이 곧
    우선순위다.

    반환 = (채운 라벨 지도, 씨앗 픽셀의 flat index 지도) — 미정 자리는 각각
    -1이다 (씨앗에서 걸어 닿을 수 없는 섬).
    """
    h, w = lab.shape
    out = lab.astype(np.int32, copy=True)
    n = int(out.max()) + 1 if out.size and out.max() >= 0 else 0
    rank = (np.arange(max(n, 1), dtype=np.int64) if order is None
            else np.asarray(order, np.int64))
    of_rank = np.zeros(max(int(rank.max()) + 1, 1), np.int32) if n else np.zeros(1, np.int32)
    if n:
        of_rank[rank[:n]] = np.arange(n, dtype=np.int32)
    big = np.int64(h) * np.int64(w)
    inf = np.int64(np.iinfo(np.int64).max // 4)
    flat = np.arange(h * w, dtype=np.int64).reshape(h, w)
    src = np.full((h, w), -1, np.int64)
    have = out >= 0
    src[have] = flat[have]
    todo = zone & ~have
    for _ in range(max_rounds):
        if not todo.any():
            break
        have = out >= 0
        key = np.where(have, rank[np.maximum(out, 0)] * big + src, inf)
        best = np.full((h, w), inf, np.int64)
        _min_into(best, key)
        new = todo & (best < inf)
        if not new.any():
            break
        k = best[new]
        out[new] = of_rank[(k // big).astype(np.int64)]
        src[new] = k % big
        todo &= ~new
    return out, src
