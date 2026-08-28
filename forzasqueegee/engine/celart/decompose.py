"""셀 재해석 — 원화 한 장을 **평면 색 영역 목록**으로 다시 읽는다.

    RGBA → 선 제거(면 귀속 복원, §1) → 평활 → 팔레트(§5)
    → watershed 원자 + 그리드 재분할(§2) → RAG·MDL 병합(§3·§4)
    → 영역 표(넓이 내림차순 = 그리기 순서)

레퍼런스 분석(`references/영상/` 5편 + `사람작업/벤티-사람·벤티-페인터.png`
대조)에서 확정한 사람 방식의 뼈대가 이 순서의 근거다:

- 도형 하나 = **의미 있는 영역 하나** (머리카락 덩어리·피부·그림자 면·획).
  painter의 "타원 = 픽셀 뭉치"와 정반대이고, 모자이크 티가 없는 이유다.
- 색면은 평면(단색)이고 경계가 또렷하다. 부드러운 음영은 톤 단계 몇 개로
  재해석된다(셀 애니 문법).
- 그리는 순서는 큰 면 → 작은 면 → 디테일. 뒤에 그리는 면이 앞 면의 삐져나온
  가장자리를 덮어 주므로, 앞서 그린 면은 **나중 면 밑으로는 대충 삐져나가도
  된다** — 사람이 빨리 그리는 핵심 요령이고 celfit의 배치 자유도가 된다.

여기서는 그 재해석만 한다 (도형 배치는 celfit 패키지).

**분해 단계는 예산에 안 묶는다.** 무엇을 그릴지(분해)와 몇 장을 쓸지(가격)는
다른 물음이다. 묶어 두면 예산을 내릴 때 눈·코·입이 분해 단계에서 병합돼
사라진다 (실측 700장: 영역 120개, 입이 통째로 없어졌다). 상한
(`_MAX_REGIONS`)은 그래프가 폭주하지 않게 두는 뚜껑일 뿐이고, 그 안에서
몇 개가 남을지는 그림이 정한다.

선화는 신경망(`lineart.extract` = AniLines)이 먼저 뽑아 `line_mask`로 빠지고,
그 자리는 §1이 **양쪽 면에 나눠** 메운다. 모델이 없으면 폴백: 선을 안 빼고
그대로 분해해도 검은 선이 팔레트의 어두운 클러스터로 살아남아 "가늘고 긴
영역"이 되고, 가늘냐 넓냐는 celfit이 기하로 가른다 — 품질은 떨어지지만
한 버튼은 끝까지 돈다.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

from .. import celaxes
from ..model import UNITS_PER_SCALE
from ..price import price_of
from . import atoms, dense, inkfill, legacy, palette, prep
from .marks import mark_mask
from .model import _ALPHA_OPAQUE, CelArt
from .prep import _fill_bg_nearest
from .rag import RegionGraph
from .snap import region_table, regularize

# 영역 수 상한 — **뚜껑이지 목표가 아니다.** 가격 설계(값이 안 되는 영역은
# 도형을 안 받는다)에서 잰 무릎이 650이고, 그래프 병합이 이 수까지 못 내려온
# 그림에서만 강제 병합이 돈다. 위로는 셀만 좋아지고 배치가 못 따라가는
# 구간(예산 포화 장에서 못 그린 영역이 늘어난다), 아래로는 강제 병합이 뛰어
# 무늬가 색 진창이 된다. 선화는 line_mask 별도 경로라 무관.
_MAX_REGIONS = int(os.environ.get("FS_MAX_REGIONS", 650))


def decompose(rgba: np.ndarray, *, max_regions: int = _MAX_REGIONS,
              line_mask: np.ndarray | None = None,
              log=print, value: np.ndarray | None = None,
              merge_gain: float = 0.0,
              price: float = 0.0, debug: bool = False) -> CelArt:
    """RGBA(작업 해상도) → CelArt. 결정적(시드 고정).

    `line_mask`를 주면 선화를 여기서 안 뽑는다 — 호출부가 작업 해상도보다 큰
    중간본에서 뽑아 줄여 온 것을 그대로 쓴다 (가는 선이 축소로 씻기는 것을
    막는 경로). 안 주면 지금 이미지에서 뽑는다.

    `value`는 값 맵(`importance.place_weight`)이고 `price`는 가격 λ다. 둘이
    있으면 **그래프 병합이 배치와 같은 자를 쓴다** — "도형 한 장을 아끼려고
    이만큼의 색 오차를 감수할 것인가"가 한 부등식이 된다 (`rag` 문서).
    안 주면 값 맵 없이 λ만 캔버스 크기에서 유도한다.
    """
    from .. import lineart

    h, w = rgba.shape[:2]
    sel = rgba[..., 3] >= _ALPHA_OPAQUE
    if not sel.any():
        raise SystemExit("불투명 픽셀이 없다 — 입력을 확인할 것")
    trace: dict = {}

    # 0) 선화 — 선 픽셀을 **양쪽 면에 나눠** 복원한다 (§1).
    src = _fill_bg_nearest(rgba[..., :3], sel)
    if line_mask is not None:
        line_mask = line_mask & sel
    else:
        lm = lineart.extract(src, log=log)
        line_mask = lineart.hysteresis(lm) & sel if lm is not None else None
    faces = None
    if line_mask is not None:
        log(f"  선화: 선 픽셀 {int(line_mask.sum()):,}개")
        if celaxes.on("INKFILL"):
            src, faces = inkfill.complete(src, sel, line_mask, log)
        else:
            # 대조군 — 종전의 유클리드 최근접 채움 (자가 직선이라 가는 구조를
            # 뛰어넘는다: 그 도약이 곧 "색이 선을 넘었다"이다)
            src = _fill_bg_nearest(src, sel & ~line_mask)
            faces = inkfill.faces_of(sel, line_mask)[0]

    # 1) 평활 — mean-shift가 부드러운 음영을 톤 면으로 뭉친다
    sm = prep.smooth(src)
    lab = cv2.cvtColor(sm, cv2.COLOR_RGB2LAB).astype(np.float32)

    # 2) 팔레트 — 색 표현만 맡는다 (§5)
    K, lbl, _ctr, pstats = palette.quantize(lab, sel, log)
    trace.update(pstats)
    lbl = prep._smooth_labels(lbl, K, sel)

    # 3) 원자 — 선화가 있으면 watershed가 경계를 정한다 (§2)
    guide = sm
    if line_mask is not None:
        barrier = cv2.morphologyEx(
            line_mask.astype(np.uint8), cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))).astype(bool) & sel
        guide = sm.copy()
        guide[barrier] = rgba[..., :3][barrier]     # 선을 다시 어둡게 = 능선
        labels = atoms.watershed_atoms(lbl, K, sel, guide, barrier, log)
    else:
        labels = atoms.cc_atoms(lbl, K, sel)
    if celaxes.on("OVERSEG"):
        labels = atoms.oversegment(labels, lab, sel, guide, log)
    if debug:
        trace["atom_labels"] = labels.copy()

    # 4) 병합 — 무엇이 한 덩어리인가 (§3·§4)
    lam = price if price > 0.0 else price_of(
        value if value is not None else np.ones((h, w), np.float32))
    if celaxes.on("RAG"):
        n = int(labels.max()) + 1 if labels.max() >= 0 else 0
        feat = None
        if celaxes.on("DENSE") and n:
            feat = dense.region_features(src, labels, sel, n, log=log)
        g = RegionGraph(labels, lab, sel, ink=line_mask, imp=value, feat=feat)
        if debug:
            trace["marks_before"] = g.mark_ids()
        labels = g.merge(lam, max_regions, log)
        trace.update(g.stats)
        trace["dense"] = feat is not None
    else:
        labels = legacy.merge_regions(labels, lab, sel, max_regions, log,
                                      value=value, merge_gain=merge_gain)

    # 5) 영역 표 — 대표색은 평활 이미지의 영역 평균 (팔레트 중심보다 국소 충실)
    regions = region_table(labels, sm, sel, w, h)

    # 5b) 경계 펴기 (기본 꺼짐) — `snap.regularize` 문서. 무늬 보호 조각을
    #     지키려면 판정에 영역 표가 있어야 하므로 표를 짓고 나서 편다
    mult = os.environ.get("FS_CEL_SMOOTH", "")
    if mult and float(mult) > 0:
        upp = 900.0 / h                       # celfit과 같은 캔버스 배율
        r = int(round(float(mult) * 0.01 * UNITS_PER_SCALE / upp))
        if r >= 1:
            keep = mark_mask(CelArt(size=(w, h), labels=labels, regions=regions))
            labels = regularize(labels, sel, r, keep)
            regions = region_table(labels, sm, sel, w, h)
            log(f"  경계 펴기 r={r}px (×{float(mult):g} 최소 반폭) · "
                f"무늬 보호 {int(keep.sum()):,}px")

    log(f"  영역 {len(regions)}개 (병합 후)")
    trace["regions_decomposed"] = len(regions)
    return CelArt(size=(w, h), labels=labels, regions=regions,
                  line_mask=line_mask,
                  src_rgb=_fill_bg_nearest(rgba[..., :3], sel),
                  faces=faces, trace=trace)
