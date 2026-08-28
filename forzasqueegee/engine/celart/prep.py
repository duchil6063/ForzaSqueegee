"""전처리 — 배경 채움과 평활, 그리고 라벨 다수결.

여기까지는 "무엇을 그릴 것인가"를 아직 안 묻는다. 원화를 셀 재해석이 볼 수
있는 상태로 눕히는 일만 한다.
"""

from __future__ import annotations

import cv2
import numpy as np

_MS_SP, _MS_SR = 13, 30            # mean-shift 공간·색 창 (셀 톤 뭉침 수위)


def _fill_bg_nearest(rgb: np.ndarray, sel: np.ndarray) -> np.ndarray:
    """투명 픽셀을 가장 가까운 불투명 픽셀 색으로 채운다 — 평활 필터가 배경
    쓰레기 색(투명 영역의 임의 RGB)을 실루엣 가장자리로 끌고 오는 것을 막는다."""
    if sel.all():
        return rgb
    _, lbl = cv2.distanceTransformWithLabels(
        (~sel).astype(np.uint8), cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(sel)
    lut = np.zeros(int(lbl.max()) + 1, np.int64)
    lut[lbl[ys, xs]] = ys.astype(np.int64) * rgb.shape[1] + xs
    src = lut[lbl]
    return rgb.reshape(-1, 3)[src.ravel()].reshape(rgb.shape)


def smooth(src: np.ndarray) -> np.ndarray:
    """mean-shift + bilateral — 부드러운 음영을 톤 면으로 뭉친다 (셀 재해석의 실체).

    bilateral만으로는 그라데이션이 수천 조각으로 갈라진다 (실측: 4,012 영역).
    sp 13·sr 30 — 톤 띠를 크게 뭉쳐 영역 수요를 줄인다. 선화가 이미 빠져
    있어 세게 뭉쳐도 윤곽은 안 뭉개진다 (눈·하이라이트는 ΔE가 커서 생존).
    """
    ms = cv2.pyrMeanShiftFiltering(cv2.cvtColor(src, cv2.COLOR_RGB2BGR),
                                   _MS_SP, _MS_SR, maxLevel=1)
    return cv2.bilateralFilter(cv2.cvtColor(ms, cv2.COLOR_BGR2RGB), 9, 40, 7)


def _smooth_labels(lbl: np.ndarray, K: int, sel: np.ndarray,
                   kernels: tuple[int, ...] = (3, 3, 5)) -> np.ndarray:
    """다수결로 라벨 소금후추·경계 위글 제거.

    마지막 5×5 라운드가 경계를 편다 — 저해상 원본을 1200으로 올린 AA 위글이
    도형 소비를 부풀리는 것(실측: 머리칼 영역 둘레/√면적 30)을 눌러 준다.
    사람도 미세 요철을 그대로 그리지 않는다.

    K장을 쌓아 놓고 argmax를 치지 않고 **누적**으로 이긴다 — H×W×K float32는
    세로로 긴 캔버스에서 수백 MB다(2Mpx·K24면 200MB). 결과는 완전히
    같다: `argmax`는 최대가 여럿이면 가장 앞 k를 고르는데, 등호 없는 `>`로
    누적해도 앞 k가 남는다. 합은 0/1의 정수라 uint8·int16 어느 쪽에서도 정확하다.
    """
    for ks in kernels:
        best = None
        arg = np.zeros(lbl.shape, np.int32)
        for k in range(K):
            v = cv2.boxFilter((lbl == k).astype(np.uint8), cv2.CV_16S,
                              (ks, ks), normalize=False)
            if best is None:
                best = v
                continue
            win = v > best
            best[win] = v[win]
            arg[win] = k
        lbl = np.where(sel, arg, lbl)
    return lbl
