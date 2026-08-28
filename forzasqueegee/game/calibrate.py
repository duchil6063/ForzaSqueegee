"""P 격자 뷰 기반 px/유닛 자동 보정.

실측 사실 (2026-08-02, 1776×999 클라이언트 / 게임 해상도 2560×1440):
- 캔버스 뷰는 직교·등방. 1 캔버스 유닛 = 게임 렌더 해상도 1px (실측 ppu 0.6934 ≈ 999/1440)
- 미세 격자 1칸 = 128/3 유닛 (스케일 1.0 도형 폭 128유닛 = 정확히 3칸)
- 보정식: px/유닛 = 격자주기px × 3 / 128 (WASD+OCR 이동 실측과 0.1% 내 일치)
"""

from __future__ import annotations

import numpy as np

UNITS_PER_GRID_CELL = 128.0 / 3.0


def grid_period_px(img: np.ndarray) -> float | None:
    """밝은 격자 캔버스 캡처에서 미세 격자 주기(px). 격자 미검출 시 None.

    UI를 피해 우중앙 ROI의 행/열 평균 밝기 자기상관으로 주기 추정,
    다수 피크로 서브픽셀 정밀화. 가로/세로 불일치(>2%)면 None.
    """
    h, w = img.shape[:2]
    roi = img[int(0.06 * h):int(0.87 * h), int(0.28 * w):int(0.98 * w)]
    roi = roi.astype(np.float32).mean(axis=2)

    def period(profile: np.ndarray) -> float | None:
        p = profile - profile.mean()
        ac = np.correlate(p, p, mode="full")[len(p) - 1:]
        if ac[0] <= 0:
            return None
        ac /= ac[0]
        hi = min(len(ac) - 1, 400)
        peaks = [i for i in range(15, hi)
                 if ac[i] > ac[i - 1] and ac[i] >= ac[i + 1] and ac[i] > 0.25]
        if not peaks:
            return None
        base = peaks[0]
        # 관측 가능한 가장 먼 n번째(≥2배수) 피크로 정밀화. 2배수 피크만 있어도
        # 써야 한다 — 세로 프로파일은 피크가 2개뿐인 창 크기가 있어(1350·900 실측)
        # 정수 양자화된 주기가 가로 서브픽셀 주기와 2% 넘게 어긋나 기각됐다.
        far = [l for l in peaks if round(l / base) >= 2]
        if far:
            l = far[-1]
            return l / round(l / base)
        return float(base)

    px = period(roi.mean(axis=0))
    py = period(roi.mean(axis=1))
    if px is None or py is None or abs(px - py) / max(px, py) > 0.02:
        return None
    return (px + py) / 2.0


def px_per_unit(img: np.ndarray) -> float | None:
    """캡처에서 px/유닛. 격자 미검출(배경 꺼짐/차량 뷰) 시 None."""
    p = grid_period_px(img)
    return None if p is None else p / UNITS_PER_GRID_CELL
