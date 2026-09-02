"""셀 재해석의 자료형 — 영역 하나와 그 지도.

`CelArt` 하나가 "이 그림을 무엇으로 그릴 것인가"의 답이다: 평면 색 영역 지도
(`labels`)와 그리기 순서(`regions`, 넓이 내림차순)와 그 위에 마지막으로 얹는
선화(`line_mask`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_ALPHA_OPAQUE = 128                # 이 미만 알파는 배경으로 본다


@dataclass
class Region:
    """평면 색 영역 하나 — 그리기 순서 = regions 리스트 순서."""

    rid: int                      # labels 값
    color: tuple[int, int, int]   # sRGB 대표색 (영역 평균)
    area: int                     # px
    bbox: tuple[int, int, int, int]  # x0, y0, x1, y1 (반개구간)


@dataclass
class CelArt:
    """셀 재해석 결과: 영역 지도 + 그리기 순서 + 선화."""

    size: tuple[int, int]                 # (w, h)
    labels: np.ndarray                    # int32 (h,w), 영역 id, -1 = 투명 배경
    regions: list[Region] = field(default_factory=list)  # 넓이 내림차순
    # 신경망 선화 (있을 때): 선 마스크와 선 색 표본용 원화. 획은 celfit이
    # 이 마스크에서 뽑아 **모든 면 위에** 얹는다 (사람의 마지막 선따기 순서)
    line_mask: np.ndarray | None = None   # bool (h,w)
    src_rgb: np.ndarray | None = None     # uint8 (h,w,3) — 선 색 표본용
    # 분해 자취 (`decompose`가 채운다) — 계측·디버그 겹판이 읽는다.
    trace: dict = field(default_factory=dict)

    def flat_render(self, bg: int = 255) -> np.ndarray:
        """영역 대표색 + 선화를 얹은 셀화 (RGB) — cel.png 미리보기·검증 기준."""
        h, w = self.labels.shape
        out = np.full((h, w, 3), bg, np.uint8)
        lut = np.zeros((max((r.rid for r in self.regions), default=0) + 1, 3), np.uint8)
        for r in self.regions:
            lut[r.rid] = r.color
        sel = self.labels >= 0
        out[sel] = lut[self.labels[sel]]
        if self.line_mask is not None and self.src_rgb is not None:
            out[self.line_mask] = self.src_rgb[self.line_mask]
        return out

    def flat_render_rgba(self) -> np.ndarray:
        """`flat_render` + 알파 — 영역이나 선이 있는 픽셀만 불투명 (RGBA).
        배경은 흰색 그대로라 알파를 버리고 읽어도 같은 그림이다."""
        opaque = self.labels >= 0
        if self.line_mask is not None and self.src_rgb is not None:
            opaque = opaque | self.line_mask
        return np.dstack([self.flat_render(),
                          np.where(opaque, 255, 0).astype(np.uint8)])
