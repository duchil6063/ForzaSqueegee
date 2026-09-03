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

from ...i18n import msg
from ..price import price_of
from ..stop import stop_here
from . import atoms, dense, inkfill, palette, prep
from .model import _ALPHA_OPAQUE, CelArt
from .prep import _fill_bg_nearest
from .rag import _AUDIT as _RAG_AUDIT
from .rag import RegionGraph
from .snap import region_table

# 영역 수 상한 — **뚜껑이지 목표가 아니다.** 가격 설계(값이 안 되는 영역은
# 도형을 안 받는다)에서 잰 무릎이 650이고, 그래프 병합이 이 수까지 못 내려온
# 그림에서만 강제 병합이 돈다. 위로는 셀만 좋아지고 배치가 못 따라가는
# 구간(예산 포화 장에서 못 그린 영역이 늘어난다), 아래로는 강제 병합이 뛰어
# 무늬가 색 진창이 된다. 선화는 line_mask 별도 경로라 무관.
_MAX_REGIONS = int(os.environ.get("FS_MAX_REGIONS", 650))
# **영역 색 상한** (사람 리버리 계획 4단계 · 색). 다 그린 판의 색은 접기 뒤에도
# 면 색이 태반이다 (표준 11장: 채움 291색 ↔ 획 131색) — 면 색은 영역마다 제
# 평균이라 얼굴 살색과 손 살색이 1/255씩 다르다. 사람 판은 같은 역할에 같은
# 색을 다시 쓴다(옆면 중앙 94색·p90 242). 다 그린 뒤 면 색을 옮기면 그대로
# 재현 오차라 기각됐으므로(`ramps` 머리말) **여기서**, 목표(cel.png)가 함께
# 움직이는 자리에서 접는다 — 자는 원화 기준(`rmse_src`)이다.
#
# 128 채택 (표준 11장, 획 상한 128과 함께): 판 색 420 → 188(−55%, 채움 291 →
# 101) · 장수 −0.8% · `rmse_src` +0.8%(6/11 판이 0.1 나빠짐) · 보이는 오차
# 중앙 .162 → .139(02만 +30%, 꽃 톤 — 크롭으로는 안 갈린다) · 커버리지·선
# 근접 불변. 차 자의 색 벌점 .45 → .03, 재료 무리 .722 → .729(= 사람 판).
# 0이면 안 접는다. 반경은 어느 영역색도 그 너머로는 안 옮긴다는 상한이다.
_REGION_K = int(os.environ.get("FS_CEL_REGION_K", 128))
_REGION_DE = float(os.environ.get("FS_CEL_REGION_DE", 12.0))


def _adjacent_pairs(labels: np.ndarray, sel: np.ndarray) -> np.ndarray:
    """맞닿은 영역 id 쌍 (a < b), 4-이웃. 배경(-1)은 뺀다."""
    out = []
    for a, b in ((labels[:, :-1], labels[:, 1:]), (labels[:-1, :], labels[1:, :])):
        m = (a != b) & (a >= 0) & (b >= 0)
        if m.any():
            lo = np.minimum(a[m], b[m]).astype(np.int64)
            hi = np.maximum(a[m], b[m]).astype(np.int64)
            out.append(np.unique(lo * (int(labels.max()) + 1) + hi))
    if not out:
        return np.zeros((0, 2), np.int64)
    key = np.unique(np.concatenate(out))
    n = int(labels.max()) + 1
    return np.stack([key // n, key % n], axis=1)


def _fold_region_colors(regions: list, labels: np.ndarray, sel: np.ndarray,
                        k_max: int, move: float) -> tuple[list, dict]:
    """영역 대표색을 상한 `k_max` 안으로 접는다 (`ramps.fold_palette`).

    **맞닿은 두 영역이 눈에 보이게 갈려 있으면(ΔE ≥ JND) 못 묶는다** — 병합이
    λ를 치르고 남긴 경계다. 그래서 접히는 것은 떨어져 있는 같은 역할의 색
    (얼굴 살색 ↔ 손 살색)과 맞닿았어도 안 보이는 차이뿐이다. 넓이 가중이라
    큰 면이 대표를 잡는다.
    """
    from dataclasses import replace

    from .marks import _MARK_DE
    from .ramps import fold_palette

    pos = {r.rid: i for i, r in enumerate(regions)}
    cols = [tuple(int(v) for v in r.color) for r in regions]
    w = np.array([float(r.area) for r in regions])
    arr = np.asarray(cols, np.uint8).reshape(-1, 1, 3)
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    pairs = _adjacent_pairs(labels, sel)
    bad = []
    for a, b in pairs.tolist():
        i, j = pos.get(int(a)), pos.get(int(b))
        if i is None or j is None:
            continue
        if float(np.linalg.norm(lab[i] - lab[j])) >= _MARK_DE:
            bad.append((i, j))
    table, st = fold_palette(cols, w, bad, k_max=k_max, move=move)
    out = [replace(r, color=table.get(c, c)) for r, c in zip(regions, cols)]
    return out, st


def decompose(rgba: np.ndarray, *, max_regions: int = _MAX_REGIONS,
              line_mask: np.ndarray | None = None,
              log=print, value: np.ndarray | None = None,
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
        raise SystemExit(msg("불투명 픽셀이 없다 — 입력을 확인할 것"))
    trace: dict = {}

    # 0) 선화 — 선 픽셀을 **양쪽 면에 나눠** 복원한다 (§1).
    src = _fill_bg_nearest(rgba[..., :3], sel)
    if line_mask is not None:
        line_mask = line_mask & sel
    else:
        lm = lineart.extract(src, log=log)
        line_mask = lineart.hysteresis(lm) & sel if lm is not None else None
    src0 = src
    if line_mask is not None:
        log(msg("  선화: 선 픽셀 {n:,}개", n=int(line_mask.sum())))
        src = inkfill.complete(src, sel, line_mask, log)[0]

    stop_here()
    # 1) 평활 — mean-shift가 부드러운 음영을 톤 면으로 뭉친다
    sm = prep.smooth(src)
    lab = cv2.cvtColor(sm, cv2.COLOR_RGB2LAB).astype(np.float32)

    stop_here()
    # 2) 팔레트 — 색 표현만 맡는다 (§5)
    K, lbl, _ctr, pstats = palette.quantize(lab, sel, log)
    trace.update(pstats)
    lbl = prep._smooth_labels(lbl, K, sel)

    stop_here()
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
    stop_here()
    n_ws = int(labels.max()) + 1 if _RAG_AUDIT else -1   # 감사 — 원자 출신 경계
    labels = atoms.oversegment(labels, lab, sel, guide, log)
    if debug:
        trace["atom_labels"] = labels.copy()

    stop_here()
    # 4) 병합 — 무엇이 한 덩어리인가 (§3·§4)
    lam = price if price > 0.0 else price_of(
        value if value is not None else np.ones((h, w), np.float32))
    n = int(labels.max()) + 1 if labels.max() >= 0 else 0
    feat = dense.region_features(src, labels, sel, n, log=log) if n else None
    g = RegionGraph(labels, lab, sel, ink=line_mask, imp=value, feat=feat)
    if debug:
        trace["marks_before"] = g.mark_ids()
    stop_here()
    labels = g.merge(lam, max_regions, log)
    if _RAG_AUDIT:                        # 계보 감사 (`rag.lineage`) — 판정 무관
        lin = g.lineage(ws_atoms=n_ws)
        am = g.labels
        ok = sel & (am >= 0)
        na = max(int(am.max()) + 1, 1)
        cnt = np.bincount(am[ok].astype(np.int64) * K + lbl[ok].astype(np.int64),
                          minlength=na * K).reshape(na, K)
        apal = cnt.argmax(1)
        for a in range(na):
            r = str(g.find(a))
            if r in lin and cnt[a].sum():
                d = lin[r].setdefault("pal", {})
                d[int(apal[a])] = d.get(int(apal[a]), 0) + int(cnt[a].sum())
        for rec in lin.values():
            d = rec.get("pal") or {}
            rec["pal_n"] = len(d)
            rec["pal"] = int(max(d.items(), key=lambda kv: kv[1])[0]) if d else -1
        for rec in lin.values():          # 이웃과 팔레트가 갈리나 (§29)
            nb = lin.get(str(rec.get("best_nbr")))
            rec["nbr_pal"] = nb["pal"] if nb else -1
        trace["lineage"] = lin
        trace["lam"] = float(g.lam)
        trace["atoms_ws"] = int(n_ws)
    trace.update(g.stats)
    trace["dense"] = feat is not None

    # 5) 영역 표 — 대표색은 평활 이미지의 영역 평균 (팔레트 중심보다 국소 충실)
    regions = region_table(labels, sm, sel, w, h)
    if _REGION_K > 0 and regions:
        regions, fst = _fold_region_colors(regions, labels, sel, _REGION_K,
                                           _REGION_DE)
        trace["region_fold"] = fst
        log(msg("  영역 색 접기 {a}색 → {b}색 (상한 {k}·반경 ΔE {mv:g}) · "
                "평균 ΔE00 {m:.2f} · 최대 {x:.2f} · 경계 잠금 {e}쌍",
                a=fst["before"], b=fst["after"], k=_REGION_K, mv=_REGION_DE,
                m=fst.get("mean_de00", 0.0), x=fst.get("max_de00", 0.0),
                e=fst.get("locked", 0)))

    # **삼킨 선 (X7 검수).** 어두운 잔선·끈이 선 지도에서 빠지면 세 손이
    # 차례로 지운다 — 귀속 배리어(CLOSE)가 밝은 면 색으로 덧칠하고, 평활이
    # 이웃 면으로 뭉개고, 병합이 밝은 영역에 흡수한다. 결과는 같다: **영역색이
    # 원화보다 훨씬 밝은 어두운 px** (X7-01 #1·#2류 — 어두운 끈이 살색·중간톤
    # 판이 된다). 그 px를 여기서 선으로 기록한다 — 호출부가 선 얹기 마스크에
    # 합치면 flat_render(목표)가 원화색으로 되칠하고, 수리·메움 기계도 정직한
    # 목표를 본다. 라벨은 안 만진다. 문턱 40은 "명백히 딴 색"(팔레트 꼬리
    # 15의 갑절 이상)이고, 4px 미만 군집은 화면에서 안 보이는 크기라
    # (`route_cel.HOLE_MIN_PX`와 같은 논리) 안 센다.
    if line_mask is not None and regions:
        lut_ = np.zeros((int(labels.max()) + 1, 3), np.uint8)
        for _r in regions:
            lut_[_r.rid] = _r.color
        reg_lab = cv2.cvtColor(lut_.reshape(-1, 1, 3),
                               cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.int16)
        src_lab = cv2.cvtColor(src0, cv2.COLOR_RGB2LAB).astype(np.int16)
        px_reg = reg_lab[np.maximum(labels, 0)]
        d_l = px_reg[..., 0].astype(np.int32) - src_lab[..., 0].astype(np.int32)
        d_e = np.linalg.norm(px_reg.astype(np.float32)
                             - src_lab.astype(np.float32), axis=-1)
        cand = sel & (labels >= 0) & ~line_mask & (d_l > 40) & (d_e > 40)
        if cand.any():
            n_cc, cc = cv2.connectedComponents(cand.astype(np.uint8))
            sizes = np.bincount(cc.ravel(), minlength=n_cc)
            swallowed = cand & (sizes[cc] >= 4)
        else:
            swallowed = cand
        trace["line_swallowed"] = swallowed
        if swallowed.any():
            log(msg("  삼킨 선 {px:,}px — 밝은 면에 먹힌 어두운 잔선, "
                    "선 얹기로 되칠한다", px=int(swallowed.sum())))

    log(msg("  영역 {n}개 (병합 후)", n=len(regions)))
    trace["regions_decomposed"] = len(regions)
    return CelArt(size=(w, h), labels=labels, regions=regions,
                  line_mask=line_mask,
                  src_rgb=_fill_bg_nearest(rgba[..., :3], sel),
                  trace=trace)
