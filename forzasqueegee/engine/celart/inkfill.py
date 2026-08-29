"""§1 선 제거 — 선을 지우지 않고 **양쪽 면에 나눠 준다**.

선 픽셀을 "가장 가까운 비선 픽셀 색"으로 메우면(종전) 자가 직선이라 가는
구조를 뛰어넘는다: 손가락 사이 틈에서 건너편 살색이, 앞머리 가닥 사이에서
얼굴색이 선 자리로 넘어온다. 그 색은 곧 팔레트·watershed의 입력이 되므로
**분해가 시작되기 전에** 색이 선을 넘는다.

여기서는 선 graph를 색 영역의 **장벽**으로 쓴다:

    면 A │ 선 │ 면 B      →   각 선 픽셀은 A·B 중 걸어서 가까운 쪽에 귀속되고,
                              그 면 **안쪽의** 색으로 메워진다

1. 선 마스크를 닫아(3×3) 1px 끊김을 잇는다 — 물이 새면 A와 B가 한 면이 된다.
2. 그 장벽을 뺀 실루엣의 연결 성분 = **면**(face). 이것이 색과 독립인 위상이다.
3. 면들에서 동시에 측지 전파(`geodesic.propagate`)해 선 픽셀을 귀속시킨다 —
   교차점은 씨앗이 여럿이라 여러 면이 함께 밀려오고, 거리로 갈린다.
4. 귀속된 면 **안쪽**의 색(전파가 나른 씨앗 픽셀)으로 메운다.

**같은 면 안을 지나는 선**(옷 주름·머리칼 속 결)은 양쪽이 같은 면이라 자연히
한 색으로 복원된다 — 판정이 따로 없다.

씨앗에서 걸어 닿지 못하는 선 섬(사방이 선으로 막힌 조각)만 Telea inpaint로
메운다. 그 자리는 정의상 "같은 영역 안의 고립 구멍"이라 요청의 단서와 맞고,
색 영역 사이의 주 선 제거에는 안 쓴다.
"""

from __future__ import annotations

import cv2
import numpy as np

from .geodesic import propagate

# 장벽 닫기 커널 — 선 지도의 1px 끊김을 잇는다. **장벽에만** 쓴다: 렌더 선을
# 닫으면 획이 굵어지고, 틈 픽셀의 원화 색은 그대로라 옅은 획의 어두움이
# watershed 능선으로 살아난다.
_CLOSE = 3


def faces_of(sel: np.ndarray, line_mask: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """(면 지도, 장벽) — 선을 장벽으로 본 실루엣의 연결 성분.

    면 id는 **넓이 내림차순**으로 매긴다. 그래야 측지 전파의 동률이 "넓은
    면이 가져간다"가 되고(`geodesic.propagate`의 order 없이도), 이후 어느
    단계가 면 id를 순서로 써도 뜻이 같다.
    """
    if line_mask is None or not line_mask.any():
        out = np.where(sel, 0, -1).astype(np.int32)
        return out, np.zeros(sel.shape, bool)
    barrier = cv2.morphologyEx(
        line_mask.astype(np.uint8), cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_CLOSE, _CLOSE))).astype(bool) & sel
    free = sel & ~barrier
    if not free.any():                     # 온통 선 — 면을 못 가른다
        return np.where(sel, 0, -1).astype(np.int32), barrier
    n, cc = cv2.connectedComponents(free.astype(np.uint8), connectivity=4)
    area = np.bincount(cc[free].ravel(), minlength=n)
    area[0] = 0
    order = np.argsort(-area[1:], kind="stable") + 1     # 넓이 내림차순
    lut = np.zeros(n, np.int32)
    lut[order] = np.arange(len(order), dtype=np.int32)
    out = np.full(sel.shape, -1, np.int32)
    out[free] = lut[cc[free]]
    return out, barrier


def complete(src: np.ndarray, sel: np.ndarray, line_mask: np.ndarray | None,
             log=print) -> tuple[np.ndarray, np.ndarray]:
    """(선을 면 색으로 메운 RGB, 면 지도) — 면 지도는 선 밑까지 채워져 있다.

    선이 없으면 원본과 실루엣 한 면을 그대로 돌려준다.
    """
    if line_mask is None or not line_mask.any():
        return src, np.where(sel, 0, -1).astype(np.int32)
    faces, barrier = faces_of(sel, line_mask)
    zone = sel & (faces < 0)
    full, from_px = propagate(faces, zone)
    out = src.copy()
    got = zone & (from_px >= 0)
    if got.any():
        flat = src.reshape(-1, 3)
        out[got] = flat[from_px[got]]
    lost = zone & (from_px < 0)
    if lost.any():
        # 걸어 닿지 못한 선 섬 — 같은 영역 안의 고립 구멍이다 (§1 단서)
        out = cv2.inpaint(np.ascontiguousarray(out[..., ::-1]),
                          lost.astype(np.uint8), 3, cv2.INPAINT_TELEA)[..., ::-1]
        out = np.ascontiguousarray(out)
    nf = int(full.max()) + 1 if full.max() >= 0 else 0
    log(f"  선 제거: 면 {nf}개로 갈라 귀속 "
        f"(선 {int(zone.sum()):,}px · 고립 {int(lost.sum()):,}px)")
    return out, full
