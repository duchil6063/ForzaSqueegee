"""§5 팔레트 — **색 표현**만 맡는다. 어디가 한 영역인가는 안 정한다.

팔레트와 영역을 가르는 것이 이 노선의 전제다:

- 같은 팔레트 색이라고 한 영역이 아니다 — 눈동자와 신발이 같은 검정일 수 있다.
  공간·위상은 면 지도(`inkfill.faces_of`)와 영역 그래프(`rag`)가 맡는다.
- 다른 팔레트 색이라고 다른 의미 영역도 아니다 — 머리칼의 톤 두 단은 사람이
  한 덩어리로 본다. 그 판단은 `rag`의 병합 비용이 한다.

여기서 하는 일은 **색 사전을 짓는 것**뿐이고, 그 사전이 지켜야 할 것은 평균
오차가 아니라 **꼬리**다. 평균은 넓은 평면이 지배하므로 작은 고채도 특징
(노란 홍채 1,100px = 캔버스의 0.09%)이 씻겨도 평균은 내려간다. 그래서 두
단으로 지킨다:

1. K 후보를 작은 것부터 훑되 **평균 오차와 꼬리 비율 둘 다** 통과해야 채택.
2. 그러고도 남은 꼬리 중 **뭉쳐 있는 것**(= 작은 색면이지 잡음이 아닌 것)에는
   센터를 하나씩 더 준다. K를 통째로 올리는 것과 다르다 — K를 올리면 넓은
   면의 톤 단이 함께 늘어 영역 수요가 폭증하는데, 여기서는 그 자리에만 준다.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

# 팔레트 후보 (작은 것부터, 오차 기준 채택). 상한 48 — 24에서 멈추면 색이
# 조밀한 그림(레이스·홍채·볼터치)에서 평균 오차가 5.9까지 남아 작은 고채도
# 특징이 이웃 톤에 씻긴다 (실측 04: 홍채 적색 파편화·볼터치 소실. 48이면 4.1로
# 내려오고 눈·볼이 육안 복원, 단순한 그림은 앞 후보에서 멈춰 바이트 불변).
_K_CANDIDATES = tuple(int(v) for v in os.environ.get(
    "FS_K_CANDS", "12,16,24,32,48").split(","))
_K_MEAN_DE = 3.0                   # 채택 기준: 평균 Lab 오차 (JND ≈ 2.3보다 약간 위)
# 채택 기준 2 — **꼬리 보호.** "명백히 틀린 색"(ΔE > 15, 수리 문턱 12와 3분류
# 34 사이) px 비율이 이보다 크면 다음 후보로 간다. 0.2%는 실측 갈림에서
# 온다: 전 장 종점(K=48)의 꼬리가 0.03~0.22%, 07의 조기 정지 꼬리가 0.26%다.
_K_TAIL_DE = float(os.environ.get("FS_K_TAIL_DE", 15.0))
_K_TAIL = float(os.environ.get("FS_K_TAIL", 0.002))
# 꼬리 센터 보강 (2단) — 잔차가 큰 픽셀이 **뭉쳐 있는** 자리에만 센터를 준다.
# 크기 문턱은 캔버스 비례다: 홍채 한 짝이 실루엣의 0.09%였고, 그보다 한 자리
# 작은 0.03%를 바닥으로 둔다 (그 아래는 AA 테두리·잡티라 센터 값을 못 한다).
_K_TAIL_MIN = float(os.environ.get("FS_K_TAIL_MIN", 3e-4))
_K_TAIL_EXTRA = int(os.environ.get("FS_K_TAIL_EXTRA", 8))


def _assign(data: np.ndarray, ctr: np.ndarray) -> np.ndarray:
    """최근접 센터 라벨 (덩어리로 잘라 센다 — K×N 거리 행렬을 안 만든다)."""
    out = np.empty(len(data), np.int32)
    step = max(1, 1 << 20)
    for i in range(0, len(data), step):
        d = data[i:i + step, None, :] - ctr[None, :, :]
        out[i:i + step] = np.argmin((d * d).sum(2), axis=1).astype(np.int32)
    return out


def _tail_centers(lab: np.ndarray, sel: np.ndarray, resid: np.ndarray,
                  ctr: np.ndarray) -> np.ndarray:
    """꼬리 뭉치마다 센터 하나 — 넓은 것부터 `_K_TAIL_EXTRA`개까지."""
    bad = np.zeros(sel.shape, np.uint8)
    bad[sel] = (resid > _K_TAIL_DE).astype(np.uint8)
    if not bad.any():
        return ctr
    n, cc, st, _ = cv2.connectedComponentsWithStats(bad, connectivity=8)
    if n <= 1:
        return ctr
    areas = st[1:, cv2.CC_STAT_AREA]
    floor = max(16.0, _K_TAIL_MIN * float(sel.sum()))
    order = [i + 1 for i in np.argsort(-areas, kind="stable").tolist()
             if areas[i] >= floor][:_K_TAIL_EXTRA]
    if not order:
        return ctr
    add = np.stack([lab[cc == ci].mean(axis=0) for ci in order]).astype(np.float32)
    return np.concatenate([ctr, add], axis=0)


def quantize(lab: np.ndarray, sel: np.ndarray, log=print):
    """평활 Lab → (센터 수, 라벨 지도 int32(-1=배경), 센터, 계측 dict)."""
    data = lab[sel].reshape(-1, 3)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    ctr = None
    for K in _K_CANDIDATES:
        cv2.setRNGSeed(1)
        err, lbl_flat, ctr = cv2.kmeans(data, K, None, crit, 3,
                                        cv2.KMEANS_PP_CENTERS)
        resid = np.linalg.norm(data - ctr[lbl_flat.ravel()], axis=1)
        mean_de = float(np.sqrt(err / len(data)))
        tail = float((resid > _K_TAIL_DE).mean())
        if mean_de <= _K_MEAN_DE and tail <= _K_TAIL:
            break
    extra = 0
    if _K_TAIL_EXTRA > 0:
        ctr2 = _tail_centers(lab, sel, resid, ctr)
        if len(ctr2) > len(ctr):
            lbl2 = _assign(data, ctr2)
            resid2 = np.linalg.norm(data - ctr2[lbl2], axis=1)
            tail2 = float((resid2 > _K_TAIL_DE).mean())
            if tail2 < tail:               # 꼬리를 실제로 줄일 때만 산다
                extra = len(ctr2) - len(ctr)
                ctr, lbl_flat = ctr2, lbl2
                mean_de = float(np.sqrt((resid2 ** 2).mean()))
                tail = tail2
    K = len(ctr)
    out = np.full(sel.shape, -1, np.int32)
    out[sel] = np.asarray(lbl_flat).ravel().astype(np.int32)
    log(f"  팔레트 {K}색"
        + (f" (꼬리 보강 +{extra})" if extra else "")
        + f" · 평균 Lab 오차 {mean_de:.2f} · ΔE>{_K_TAIL_DE:g} 꼬리 {tail * 100:.2f}%")
    return K, out, ctr, {"palette_k": K, "palette_extra": extra,
                         "palette_mean_de": round(mean_de, 2),
                         "palette_tail": round(tail, 5)}
