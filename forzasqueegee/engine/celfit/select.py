"""획 선별 — 이 획이 **무엇을 가르고 있나**를 양옆에서 읽는다.

획 위가 아니라 양옆(폭/2+2.5px 법선)에서 읽는 것이 요점이다: 선 위는 무엇을
재든 "대비가 세다"로 나온다. 그 한 표본에서 경계성·실루엣성이 나오고,
역할 판정(`graph.classify`)과 계측 도구가 그 자를 쓴다.
"""

from __future__ import annotations

import cv2
import numpy as np


# 방향 응집 상수 — 나란한 반복 가닥 판정 (전 이미지 공통)
_THIN_R = 5.0          # 근접 반경 = 폭 × 이 배수 (하한 3px)
_THIN_COS = 0.6428     # 배각 내적 하한 = cos 50° → 접선 각차 25° 이내
_THIN_DE = 34.0        # 색 근접 (OpenCV Lab 노름)
_THIN_COVER = 0.50     # 경로 표본의 이 비율 이상이 덮이면 반복 가닥
_THIN_BND = 0.30       # 경계성(양옆 영역이 다른 표본 비율)이 이 이상이면 윤곽 — 보호
_THIN_SIL = 0.10       # 한쪽이 배경(-1)인 표본이 이 비율만 넘어도 실루엣 윤곽 — 보호


def _side_pts(path: np.ndarray, wmed: float, shape: tuple[int, int],
              rx0: int, ry0: int) -> tuple[np.ndarray, ...]:
    """경로 표본마다 **양옆**(폭/2+2.5px 법선) 픽셀 좌표 (idx, ay, ax, by, bx).

    "이 획이 무엇을 가르고 있나"를 묻는 판정이 전부 이 한 표본을 쓴다 —
    경계성·실루엣성(`_bnd_frac`)이 이 한 표본을 쓴다. 획 **위**가 아니라
    양옆에서 읽는 것이 요점이다: 선 위는 무엇을 재든 "대비가 세다"로 나온다.
    """
    h, w = shape
    idx = np.arange(0, len(path), 3)
    if not len(idx):
        return idx, idx, idx, idx, idx
    j0 = np.maximum(idx - 2, 0)
    j1 = np.minimum(idx + 2, len(path) - 1)
    tan = path[j1] - path[j0]
    norm = np.hypot(tan[:, 0], tan[:, 1])
    norm[norm < 1e-9] = 1.0
    off = wmed / 2.0 + 2.5
    oy = -tan[:, 1] / norm * off
    ox = tan[:, 0] / norm * off
    ys = path[idx, 0] + ry0
    xs = path[idx, 1] + rx0
    return (idx,
            np.clip(np.round(ys + oy), 0, h - 1).astype(np.int64),
            np.clip(np.round(xs + ox), 0, w - 1).astype(np.int64),
            np.clip(np.round(ys - oy), 0, h - 1).astype(np.int64),
            np.clip(np.round(xs - ox), 0, w - 1).astype(np.int64))


def _side_labels(path: np.ndarray, wmed: float, labels: np.ndarray,
                 rx0: int, ry0: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """경로 표본마다 양옆 픽셀의 영역 라벨 (표본 index, a, b)."""
    idx, ay, ax, by, bx = _side_pts(path, wmed, labels.shape, rx0, ry0)
    if not len(idx):
        return idx, idx, idx
    return idx, labels[ay, ax], labels[by, bx]


def _bnd_frac(path: np.ndarray, wmed: float, labels: np.ndarray,
              rx0: int, ry0: int) -> tuple[float, float]:
    """경로 양옆의 (경계성, 실루엣성) 표본 비율.

    경계성 = 양옆 영역 라벨이 다른 비율, 실루엣성 = 한쪽이 배경(-1)인 비율.
    역할 판정(`graph.classify`)의 실루엣·경계 획이 같은 잣대를 쓴다.
    """
    idx, la, lb = _side_labels(path, wmed, labels, rx0, ry0)
    if not len(idx):
        return 0.0, 0.0
    return (float(np.mean(la != lb)), float(np.mean((la < 0) | (lb < 0))))
