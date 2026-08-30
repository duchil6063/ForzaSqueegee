"""§2 분해의 원자 — **최종 영역이 아니라 재료**를 만든다.

    segmentation atom != Forza shape

이 단계는 "이 그림의 의미 영역이 무엇인가"를 아직 안 답한다. 나중에 그래프
병합(`rag`)이 답할 수 있도록 **조금 과분할된** 조각을 낼 뿐이다. 여기서 최종
영역을 결정하려 들면 되돌릴 수 없는 실수를 한다 — 한 번 합쳐진 눈동자와
속눈썹은 어느 뒷단계도 못 가른다.

원자는 두 갈래로 난다:

- **선이 있으면 watershed** — 팔레트 라벨의 확신 코어에서 물을 채운다. 안내
  이미지에서 선을 다시 어둡게 눌러 놓았으므로 물이 선 안에서 만난다
  (DanbooRegion의 문법 — 가중치 대신 우리 선 지도를 장벽으로 쓴다).
- **선이 없으면 연결 성분** — 폴백(모델 미설치)이다.

그 위에 **그리드 재분할**(SNIC 계열의 값싼 대역)이 한 겹 더 간다: 넓으면서
속이 고르지 않은 원자만 콤팩트한 조각으로 쪼갠다. 그라데이션이 한 원자로
흘러 버린 자리(watershed는 씨앗이 하나면 통째로 채운다)를 나중 병합이
다시 가를 수 있게 재료로 남기는 일이다.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

from ...i18n import msg

# 그리드 재분할 — 이 둘을 **함께** 넘긴 원자만 쪼갠다. 넓기만 하고 속이 고른
# 면(흰 셔츠)은 쪼개 봐야 병합이 도로 붙이므로 재료가 안 된다.
_SPLIT_AREA = float(os.environ.get("FS_ATOM_SPLIT_AREA", 0.015))   # 실루엣 비
_SPLIT_STD = float(os.environ.get("FS_ATOM_SPLIT_STD", 6.0))       # 내부 Lab 편차
_SPLIT_N = int(os.environ.get("FS_ATOM_SPLIT_N", 6))               # 한 원자의 조각 상한


def watershed_atoms(lbl: np.ndarray, K: int, sel: np.ndarray,
                    guide: np.ndarray, line_mask: np.ndarray | None,
                    log) -> np.ndarray:
    """팔레트 라벨의 **확신 코어**에서 watershed로 넓혀 원자를 만든다.

    씨앗은 성분을 1회 침식한 코어. 침식으로 사라지는 작은 성분(코 그림자·
    하이라이트)은 성분 전체를 씨앗으로 쓴다 — 이걸 빼면 12차 판정의 "코
    소실"이 여기서 재발한다.
    """
    h, w = sel.shape
    markers = np.zeros((h, w), np.int32)
    core_ok = sel if line_mask is None else (sel & ~line_mask)
    ker = np.ones((3, 3), np.uint8)
    nid = 0
    for k in range(K):
        m = ((lbl == k) & sel).astype(np.uint8)
        if not m.any():
            continue
        n, cc = cv2.connectedComponents(m, connectivity=4)
        core = (cv2.erode(m, ker) > 0) & core_ok
        has = np.bincount(cc[core].ravel(), minlength=n) > 0
        has[0] = True                                   # cc 0 = 이 라벨 밖
        seed = np.where(has[cc], core, cc > 0) & (cc > 0)
        markers[seed] = cc[seed] + nid
        nid += n - 1
    bg = nid + 1
    markers[~sel] = bg                                  # 실루엣 밖 = 한 덩어리
    cv2.watershed(np.ascontiguousarray(guide[..., ::-1]), markers)
    labels = np.where((markers > 0) & (markers != bg), markers - 1, -1)
    labels[~sel] = -1
    labels = _close_gaps(labels, sel)
    log(msg("  watershed 원자 {n}개 (씨앗 {seeds})",
            n=int(labels.max()) + 1, seeds=nid))
    return labels


def cc_atoms(lbl: np.ndarray, K: int, sel: np.ndarray) -> np.ndarray:
    """선화가 없을 때의 원자 — 팔레트 라벨의 연결 성분."""
    labels = np.full(sel.shape, -1, np.int32)
    nid = 0
    for k in range(K):
        m = (lbl == k).astype(np.uint8)
        if not m.any():
            continue
        n, cc = cv2.connectedComponents(m, connectivity=4)
        add = cc > 0
        labels[add] = cc[add] + nid - 1
        nid += n - 1
    return labels


def _close_gaps(labels: np.ndarray, sel: np.ndarray) -> np.ndarray:
    """watershed가 남긴 1px 경계선(-1)을 가장 가까운 원자에 붙인다."""
    unk = labels < 0
    if not (unk & sel).any() or not (~unk).any():
        return labels
    h, w = labels.shape
    _, nl = cv2.distanceTransformWithLabels(
        unk.astype(np.uint8), cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(~unk)
    lut = np.zeros(int(nl.max()) + 1, np.int64)
    lut[nl[ys, xs]] = ys.astype(np.int64) * w + xs
    near = labels.ravel()[lut[nl].ravel()].reshape(h, w)
    return np.where(sel & unk, near, labels).astype(np.int32)


def oversegment(labels: np.ndarray, lab: np.ndarray, sel: np.ndarray,
                guide: np.ndarray, log) -> np.ndarray:
    """넓고 속이 고르지 않은 원자를 콤팩트한 조각으로 쪼갠다 (그리드 재분할).

    씨앗을 정사각 격자로 놓고 그 원자 **안에서만** watershed를 다시 돌린다 —
    경계는 여전히 색이 가장 급하게 변하는 자리에 선다. 조각 수는 면적 비례로
    정하고 상한(`_SPLIT_N`)을 둔다: 재료를 만드는 일이지 잘게 부수는 일이
    아니다.
    """
    if _SPLIT_AREA <= 0:
        return labels
    n = int(labels.max()) + 1 if labels.max() >= 0 else 0
    if n <= 0:
        return labels
    area_cap = _SPLIT_AREA * float(sel.sum())
    counts = np.bincount(labels[sel].ravel(), minlength=n)
    nxt = n
    out = labels.copy()
    split = 0
    for rid in np.argsort(-counts, kind="stable").tolist():
        if counts[rid] < area_cap:
            break
        m = labels == rid
        vals = lab[m]
        if float(vals.std(axis=0).sum()) < _SPLIT_STD:
            continue
        k = min(_SPLIT_N, int(counts[rid] // area_cap) + 1)
        if k < 2:
            continue
        ys, xs = np.nonzero(m)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        sub = m[y0:y1, x0:x1]
        step = max(4, int(np.sqrt(counts[rid] / k)))
        mk = np.zeros(sub.shape, np.int32)
        sid = 0
        for gy in range(step // 2, sub.shape[0], step):
            for gx in range(step // 2, sub.shape[1], step):
                if sub[gy, gx]:
                    sid += 1
                    mk[gy, gx] = sid
        if sid < 2:
            continue
        mk[~sub] = sid + 1
        cv2.watershed(np.ascontiguousarray(guide[y0:y1, x0:x1][..., ::-1]), mk)
        piece = np.where(sub & (mk > 0) & (mk != sid + 1), mk - 1, -1)
        alive = [v for v in np.unique(piece[piece >= 0]).tolist()]
        if len(alive) < 2:
            continue
        lut = {v: (rid if i == 0 else nxt + i - 1) for i, v in enumerate(alive)}
        nxt += len(alive) - 1
        split += 1
        blk = out[y0:y1, x0:x1]
        for v, dst in lut.items():
            blk[piece == v] = dst
        # watershed 능선(-1)은 원자 주인에게 그대로 남긴다 (아래에서 붙는다)
    if split:
        log(msg("  그리드 재분할: 원자 {split}개 → {pieces}조각",
                split=split, pieces=nxt - n + split))
    return out
