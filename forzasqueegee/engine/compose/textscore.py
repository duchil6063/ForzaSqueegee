"""텍스트 점수 — 워드마크가 **캐릭터를 받치나, 해치나**.

`score.score_design`이 이 항목들을 제 표에 합친다 (`TEXT_WEIGHTS`). 글자가
있는 후보와 없는 후보가 같은 표에서 겨루므로 `present` 항목이 "글자를 넣기로
했으면 넣어라"를 말하고, 나머지가 "넣되 해치지 마라"를 말한다.

| 항목 | 재는 것 |
|---|---|
| text_present | 스펙이 켜졌는데 글자가 섰나 |
| text_read | 글자 본색과 그 뒤 배경(베드·베이스)의 명도차 — 상자를 넷으로 갈라 **가장 나쁜 칸** |
| text_busy | 글자 뒤 배경의 가장자리 밀도 (로커 띠·베드 경계가 글자를 가로지르면 안 읽힌다) |
| text_occlude | 인물이 글자를 덮는 몫 (≤ 0.35면 만점 · 인물 뒤 워드마크는 0.5) |
| text_flow | 글자 각이 흐름과 나란하거나 직교하나 |
| text_clutter | 글자가 장식 구역에서 차지하는 몫 |
| text_hier | 서브가 메인보다 확실히 작나 |
| text_negative | 여백 구역을 조금 쓰나 (0~0.5 사이가 좋다) |
| text_cut | 글자 상자 중 안 그려지는(휠아치·마스크 밖) 몫 — 잘린 글자는 안 읽힌다 |
| text_size | 대문자 높이가 밴드 대비 충분한가 — 이름은 한눈에 읽혀야 한다 (레퍼런스: 밴드의 1/4~1/3) |
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from ..catalog import Catalog
from ..model import Layer
from .field import CompositionField
from .textlayout import TextPose, occlude_max, pose_fit, pose_mask


TEXT_WEIGHTS = {
    "text_present": 2.0, "text_read": 1.0, "text_busy": 0.8, "text_occlude": 1.0,
    "text_flow": 0.5, "text_clutter": 0.5, "text_hier": 0.3, "text_negative": 0.3,
    "text_cut": 1.5, "text_size": 1.0,
}


# 대문자 높이 / 밴드 높이 — 이 위면 크기 만점, 아래로 갈수록 깎인다 (0.08에서 0)
SIZE_FULL = 0.26


# 테두리 벌이 있을 때 가독성·어수선 항목의 바닥 — 테두리가 바탕에서 글자를 떼어 낸다
OUTLINE_READ_FLOOR = 0.85
OUTLINE_BUSY_FLOOR = 0.80



def absent_parts() -> dict[str, float]:
    """글자 없는 후보의 항목 — `present`만 0이고 나머지는 "해친 것이 없다"(1.0).

    항목이 아예 없으면 가중 평균의 분모가 줄어 글자 없는 후보가 **거저** 이긴다
    (실측: motorsport 계열에서 자리가 셋이나 있는데 글자 없음이 이겼다).
    """
    return {k: (0.0 if k == "text_present" else 1.0) for k in TEXT_WEIGHTS}


def _lum(c) -> float:
    return (0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]) / 255.0


def text_parts(fld: CompositionField, cat: Catalog, poses: list[TextPose],
               text_layers: list[Layer], behind: np.ndarray,
               front_alpha: np.ndarray | None = None,
               outline: bool = False) -> dict[str, float]:
    """텍스트 항목들 (0~1). `behind`는 글자가 깔리기 **직전**의 합성(베이스+베드).

    `front_alpha`(전경 모티프 래스터)가 글자를 덮는 몫도 인물이 덮는 몫에 더한다.
    `outline`이면 테두리 벌이 있다 — 테두리는 바탕 명도·어수선과 무관하게 글자를
    읽히게 하는 물건이라 `text_read`·`text_busy`의 바닥을 올린다 (실측 giulia-01
    레이싱: 띠 위 글자가 read .18·busy 0으로 져서 글자 없음이 이겼다)."""
    parts: dict[str, float] = {}
    if not poses:
        return absent_parts()
    parts["text_present"] = 1.0
    main = poses[0]
    m = pose_mask(fld, main)
    # 가독성 — 본색 명도 vs 상자 안 배경 명도. 상자를 가로·세로로 갈라 넷 중 **가장
    # 나쁜 칸**을 쓴다 — 평균은 절반이 검은 띠 위에 얹힌 짙은 글자도 통과시킨다
    fills = [l.color for l in text_layers if l.label == "text"]
    if fills and m.any():
        tl = _lum(fills[0])
        bl_img = (0.299 * behind[..., 0] + 0.587 * behind[..., 1] + 0.114 * behind[..., 2]) / 255.0
        ys, xs = np.nonzero(m)
        ym, xm = float(np.median(ys)), float(np.median(xs))
        worst = 1.0
        for q in ((ys <= ym) & (xs <= xm), (ys <= ym) & (xs > xm),
                  (ys > ym) & (xs <= xm), (ys > ym) & (xs > xm)):
            if q.sum() < 4:
                continue
            bl = float(bl_img[ys[q], xs[q]].mean())
            worst = min(worst, abs(tl - bl) / 0.45)
        parts["text_read"] = max(0.0, min(1.0, worst))
        # 배경의 가장자리 밀도 — 띠·베드 경계가 상자를 가로지르면 글자 획과 섞인다
        gx = cv2.Sobel(bl_img.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(bl_img.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        busy = float((np.hypot(gx, gy)[m] > 0.6).mean())
        parts["text_busy"] = 1.0 if busy <= 0.015 else max(0.0, 1.0 - (busy - 0.015) / 0.08)
    if outline:
        parts["text_read"] = max(parts.get("text_read", 0.0), OUTLINE_READ_FLOOR)
        parts["text_busy"] = max(parts.get("text_busy", 0.0), OUTLINE_BUSY_FLOOR)
    else:
        parts["text_read"] = 0.5
        parts["text_busy"] = 1.0
    # 인물(+전경 모티프)이 덮는 몫
    cover = fld.char > 0.5
    if front_alpha is not None:
        cover = cover | (front_alpha > 0.5)
    occ = float(cover[m].mean()) if m.any() else 1.0
    lim = occlude_max(main.role)
    if main.role == "behind":
        # 인물 뒤는 상한 안에서도 가린 몫만큼 깎는다 — 같은 크기의 빈 자리가 있으면
        # 그쪽이 이기고, 없을 때만 가려진 큰 글자가 작은 글자를 이긴다
        parts["text_occlude"] = (max(0.0, 1.0 - 0.3 * occ / max(1e-6, lim)) if occ <= lim
                                 else max(0.0, 0.7 - (occ - lim) / 0.4))
    else:
        parts["text_occlude"] = 1.0 if occ <= lim else max(0.0, 1.0 - (occ - lim) / 0.4)
    # 흐름 정렬 — 나란하거나 직교
    fa = math.atan2(fld.flow[1], fld.flow[0])
    d = math.radians(main.rot) - fa
    parts["text_flow"] = max(abs(math.cos(d)), 0.8 * abs(math.sin(d)))
    # 어수선 — 글자 상자가 장식 구역(인물 밖 도색면)에서 차지하는 몫
    room = (fld.drawable > 0.5) & (fld.char < 0.5)
    frac = float(m[room].mean()) if room.any() else 0.0
    parts["text_clutter"] = 1.0 if frac <= 0.22 else max(0.0, 1.0 - (frac - 0.22) / 0.3)
    # 위계
    subs = [p for p in poses if p.role == "sub"]
    if subs:
        r = subs[0].height / max(1e-6, main.height)
        parts["text_hier"] = 1.0 if 0.3 <= r <= 0.6 else max(0.0, 1.0 - abs(r - 0.45) / 0.5)
    else:
        parts["text_hier"] = 1.0
    # 잘림 — 상자 중 그려지지 않는 몫 (휠아치·벨트라인 밖). 10%까지는 배치가
    # 허락하지만 점수는 0부터 깎는다 — 덜 잘리는 자리가 이긴다
    draw, _o, _p = pose_fit(fld, main)
    parts["text_cut"] = max(0.0, min(1.0, 1.0 - (1.0 - draw) / 0.12))
    # 크기 — 작은 이름은 장식이지 이름이 아니다 (실측: 18유닛 한 줄이 34유닛 두 줄과 비겼다)
    band = fld.frame_box[3] - fld.frame_box[1]
    r = main.height / max(1e-6, band)
    parts["text_size"] = 1.0 if r >= SIZE_FULL else max(0.0, (r - 0.08) / (SIZE_FULL - 0.08))
    # 여백 활용 — 상자의 일부(0~50%)가 여백 구역에 있으면 좋다
    neg = fld.negative > 0.5
    nf = float(neg[m].mean()) if m.any() else 0.0
    parts["text_negative"] = 1.0 if nf <= 0.5 else max(0.0, 1.0 - (nf - 0.5) / 0.5)
    return parts
