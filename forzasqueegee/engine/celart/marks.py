"""무늬 보호 조각 — **작지만 없어지면 특징이 통째로 사라지는** 영역.

흰자·코 그림자·눈 하이라이트·볼터치가 그것이다. 판정은 기하 하나다:

    접경이 가장 긴 이웃이 나보다 `_MARK_RATIO`배 넓고,
    그 이웃과 ΔE ≥ `_MARK_DE`이며,
    둘레가 면적의 45% 이하다 (= 가늘지 않고 콤팩트하다).

색차가 작아도(평활 뒤 ΔE 한 자리) 평평한 큰 면 위라 마스킹이 없어 또렷이
보인다 — 면적×색차² 단일 잣대로는 상한이 좁아지면 "넓지만 흐릿"에 다시 밀려
코·점 같은 작은 특징이 사라진다. `_MARK_DE`는 진짜 안 보이는 조각(JND 미만)
까지 지키지 않기 위한 하한이다.

같은 판정을 두 곳이 쓴다 — 그래프 병합(`rag._mark`) · 면 채움
(`celfit.fit_plan`이 `_Scorer`에 넘기는 protect). 상수를 늘리지 않으려고
한 자리에 둔다.
"""

from __future__ import annotations

import cv2
import numpy as np

_MARK_RATIO = 10.0
_MARK_DE = 4.0


def mark_mask(cel) -> np.ndarray:
    """최종 영역 위에서 다시 계산한 **무늬 보호 조각의 픽셀 마스크**.

    `celfit.fit_plan`이 `_Scorer`의 protect로 넘긴다 — 면을 채울 때 이 조각
    위로는 스필을 안 봐준다.
    큰 면을 키우는 쪽이 순이득으로는 이기지만 그 자리가 하필 눈 흰자면 눈이
    어두운 덩어리가 된다 (얼굴 지각차가 눈에 띄게 나빠진다).
    """
    lab = cel.labels
    n = int(lab.max()) + 1 if lab.size and lab.max() >= 0 else 0
    if n <= 0:
        return np.zeros(lab.shape, bool)
    area = np.zeros(n, np.int64)
    col = np.zeros((n, 3), np.float32)
    for r in cel.regions:
        area[r.rid] = r.area
        col[r.rid] = r.color
    lab_col = cv2.cvtColor(col.reshape(-1, 1, 3).astype(np.uint8),
                           cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    # 접경 길이 — 4이웃 두 방향이면 충분하다 (양방향으로 더한다)
    if n > 4096:                           # 영역이 많으면 이 가드를 안 건다
        return np.zeros(lab.shape, bool)
    border = np.zeros((n, n), np.int32)
    for a, b in ((lab[:, :-1], lab[:, 1:]), (lab[:-1], lab[1:])):
        sel = (a >= 0) & (b >= 0) & (a != b)
        if sel.any():
            np.add.at(border, (a[sel], b[sel]), 1)
            np.add.at(border, (b[sel], a[sel]), 1)
    keep = np.zeros(n, bool)
    peri = border.sum(1)
    for rid in range(n):
        if area[rid] <= 0 or peri[rid] <= 0:
            continue
        other = int(np.argmax(border[rid]))
        if border[rid, other] <= 0:
            continue
        de = float(np.linalg.norm(lab_col[rid] - lab_col[other]))
        keep[rid] = (de >= _MARK_DE
                     and area[other] >= _MARK_RATIO * area[rid]
                     and peri[rid] <= 0.45 * area[rid])
    if not keep.any():
        return np.zeros(lab.shape, bool)
    out = np.zeros(lab.shape, bool)
    pos = lab >= 0
    out[pos] = keep[lab[pos]]
    return out
