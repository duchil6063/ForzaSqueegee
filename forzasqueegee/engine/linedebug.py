"""선 재구성 자취 — `strokes.json`과 겹그림 한 장.

엔진이 무엇을 왜 그렇게 그렸는지는 최종 도안만 봐서는 안 보인다. 획마다 역할·
증거·채택 후보를 적어 두면 회귀 계측과 육안 대조가
같은 근거를 본다.

겹그림 색은 **역할**이다 (`_COLORS`) — 실루엣·구조선·특징선·내부 윤곽·색 경계·
무늬·부스러기. 안 그은 획은 **같은 색을 어둡게** 그린다: 무늬로 걸러진 것과
부스러기로 걸러진 것이 색으로 갈려야 "무엇을 왜 뺐나"가 보인다.

획 위 점의 크기는 **쓴 도형 수**다 — 한 장으로 끝난 획과 여러 장으로 쪼개진
획이 한눈에 갈린다. 이음 보수가 붙은 획은 흰 테를 두르고, detail 판에만 있던
선은 노란 고리를 두른다.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from ..paths import run_file

# BGR — 겹그림은 cv2로 쓴다
_COLORS = {
    "SILHOUETTE": (60, 220, 60),
    "STRUCTURE": (220, 200, 60),
    "FEATURE": (0, 190, 255),
    "INTERNAL_CONTOUR": (220, 120, 220),
    "COLOR_BOUNDARY": (200, 160, 90),
    "TEXTURE": (90, 60, 220),
    "NOISE": (110, 110, 110),
}
_DIM = 0.40                    # 안 그은 획은 제 역할 색을 이만큼 어둡게
_DETAIL = (255, 200, 0)        # detail 판에만 있던 선


def save(out: Path, rec, pol, size: tuple[int, int],
         line_mask: np.ndarray | None = None,
         struct: dict | None = None) -> dict:
    """`<이름>.strokes.json`을 쓰고 요약을 돌려준다 (rec가 없으면 빈 dict).

    `struct`(`celfit.stroke_metrics`)를 주면 요약에 함께 실린다 — 폭 충실도·
    파편화·꺾임처럼 커버리지가 못 보는 구조를 판끼리 대 보는 자리다.
    """
    if rec is None:
        return {}
    data = {"policy": {"name": pol.name, "notes": pol.notes,
                       **{k: getattr(pol, k) for k in
                          ("cover_min", "stray_max", "breaks_max",
                           "band_slack", "carve", "fill_below",
                           "texture_simplify", "max_shapes")}},
            "summary": {**rec.report(pol), **(struct or {})},
            "strokes": rec.per_stroke()}
    run_file(out, "strokes.json").write_text(
        json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data["summary"]


def overlay(out: Path, rec, size: tuple[int, int],
            line_mask: np.ndarray | None = None) -> None:
    """`<이름>.lines_debug.png` — 무엇을 어떻게 그렸는지 한 장으로."""
    if rec is None:
        return
    w, h = size
    img = np.full((h, w, 3), 24, np.uint8)
    if line_mask is not None:
        img[line_mask] = (58, 58, 58)      # 선 지도를 바탕에 옅게
    def draw(strokes, drawn: bool):
        for s in strokes:
            pts = np.stack([s.path[:, 1] + s.roi[0],
                            s.path[:, 0] + s.roi[1]], axis=1)
            pts = np.round(pts).astype(np.int32)
            col = _COLORS.get(s.role, (128, 128, 128))
            if not drawn:                     # 안 그은 획 — 같은 색을 어둡게
                col = tuple(int(v * _DIM) for v in col)
            cv2.polylines(img, [pts], False, col, 1, cv2.LINE_AA)
            if not drawn:
                continue
            mid = pts[len(pts) // 2]
            if s.ev.detail_only >= 0.5:
                cv2.circle(img, tuple(int(v) for v in mid), 4, _DETAIL, 1)
            # 점 크기 = 쓴 도형 수 (1장이면 점 하나)
            cv2.circle(img, tuple(int(v) for v in mid),
                       max(1, min(6, s.shapes)), col, -1)
            if s.seams:
                cv2.circle(img, tuple(int(v) for v in mid),
                           max(2, min(7, s.shapes) + 2), (255, 255, 255), 1)
    draw(rec.dropped, False)
    draw(rec.strokes, True)
    y = 18
    for name, col in list(_COLORS.items()) + [("detail-only", _DETAIL)]:
        cv2.putText(img, name, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1,
                    cv2.LINE_AA)
        y += 16
    cv2.putText(img, "dim = not drawn", (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (150, 150, 150), 1, cv2.LINE_AA)
    ok, buf = cv2.imencode(".png", img)
    if ok:
        run_file(out, "lines_debug.png").write_bytes(buf.tobytes())
