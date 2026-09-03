"""획 배치 구동 — 선 지도 하나를 **공통 선 재구성 엔진**에 태운다.

여기에는 정책이 없다. 증거 지도를 짓고(`evidence`), 엔진에 논리 획을 짓게 하고
(`engine.build_strokes`), 정책이 고른 후보로 놓게 한 뒤(`engine.place_strokes`)
덮어 그리기만 노선 정책대로 붙인다. 두 노선(`line`·`cel`)이 이 한 함수를 쓴다 —
갈리는 것은 넘겨받는 `pol` 하나뿐이다.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

from ...i18n import msg
from ..catalog import Catalog
from ..celart import CelArt
from ..model import Layer, LayerPlan
from ..price import fix_min_gain, repair_min_gain
from . import engine as E
from . import policy as P
from .carve import _carve_lines
from .evidence import build_maps
from .geometry import _min_span, _poly_px

# §26-b 선↔면 보정의 라운드 수와 걸음 (게임 이동 스텝의 배수).
# 0이면 안 돈다. 두 바퀴면 최대 한 칸 반 — 그 이상은 획이 제 선 지도를 떠난다.
_ALIGN_ROUNDS = int(os.environ.get("FS_ALIGN_ROUNDS", 2))
# 획 색을 무엇으로 정하나 (`recolor_strokes`) — `core`(기본): 발자국 중 씨앗
# 색에 가까운 절반의 평균(심 색) · `mean`: 발자국 아래 원화 평균(최소제곱
# 최적, 종전 기본) · `seed`: 뼈대 중앙값 그대로(보정 없음). 근거는 함수 문서.
_INK_COLOR = os.environ.get("FS_INK_COLOR", "core").strip().lower()
_INK_CORE_Q = float(os.environ.get("FS_INK_CORE_Q", 50.0))
# **획 색 상한** (사람 리버리 계획 4단계 · 색). 획 색은 그룹마다 제 발자국에서
# 잰 심 색이라 그룹 수만큼 색이 난다(표준 11장 577색). 맨 끝 접기(`celart.ramps`,
# 반경 4.5)가 그것을 131색으로 묶지만 큰 판(02·11)은 경계 잠금에 막혀 300색이
# 남는다. 여기서 — 채움·미세 조정·수리가 그 색을 보기 **전에** — 상한 안으로
# 접는다. 잠금은 맨 끝 접기와 같은 자(맞닿은 획의 ΔE ≥ `ramps.EDGE_DE`)다.
# 128 채택: 획 색 131 → 98(맨 끝 접기 뒤 94) · 장수·`rmse_src`·보이는 오차 불변
# (11장 ±0.3%). 대가는 거의 영역 상한(`celart.decompose._REGION_K`) 몫이다.
# 0이면 안 접는다.
_INK_K = int(os.environ.get("FS_INK_K", 128))
_INK_K_DE = float(os.environ.get("FS_INK_K_DE", 12.0))


def _fit_lines(plan: LayerPlan, cel: CelArt, cat: Catalog, upp: float,
               budget: int, forms: tuple, log, sids=None,
               value: np.ndarray | None = None, price: float = 0.0,
               carve_defer: list | None = None, carve: bool = True,
               progress=None, stats: dict | None = None,
               pol=None, maps=None) -> int:
    """선 지도 → 획 레이어 (곡선·막대, 색은 원화 표본, 전부 ink).

    금지는 투명 배경뿐이다 — 선은 모든 면 위에 얹히므로 면 침범 개념이 없다.
    경로 하나가 획 하나라 `sids`에서 새 그룹 id를 받는다.

    `pol`은 노선 정책(`policy.LINE`·`policy.CEL`)이다. 안 주면 line 정책 —
    가장 빡빡한 쪽이 기본이다.

    `carve=False`는 덮어 그리기(획 덮개)를 통째로 끈다 — 덮개는 **면 색으로
    도로 덮는** 문법이라 면 채움이 없는 자리에는 덮을 색이 없다. 정책도 같은
    칸을 들고 있어 둘 다 참일 때만 돈다.
    """
    st = stats if stats is not None else {}
    pol = pol or P.LINE
    w, h = cel.size
    if maps is None:
        # 증거가 따로 안 왔다 — 선 지도 자체가 증거다 (고전 폴백)
        maps = build_maps(None, None, cel.line_mask, cel.line_mask,
                          cel.src_rgb, cel.labels >= 0,
                          value if value is not None
                          else (cel.labels >= 0).astype(np.float32))
    rec = E.build_strokes(plan, cel, maps, cat, upp, sids, log, pol)
    n = rec.fat_fills + E.place_strokes(
        plan, rec, cel, cat, upp, max(0, budget - rec.fat_fills), forms, pol,
        log, price=price, progress=progress)
    st.update(rec.report(pol))
    st["_rec"] = rec                       # 노선이 debug·per-stroke에 쓴다
    if carve and pol.carve:
        # 덮개도 한 장이다 — 가격 설계에서는 λ만큼 벌어야 산다
        n += _carve_lines(plan, cel, cat, upp, budget - n, log,
                          floor=fix_min_gain(price) if price else 0.0,
                          floor_lo=repair_min_gain(price) if price else 0.0,
                          defer=carve_defer)
    return n


def _ink_of(cat: Catalog, lay: Layer, upp: float, w: int, h: int):
    """레이어 하나의 1x 라스터 (마스크, x0, y0) — 렌더와 같은 폴리곤 식."""
    polys = _poly_px(cat, lay, upp, w, h)
    if not polys:
        return None
    rp = [np.round(q).astype(np.int32) for q in polys]
    x0 = max(0, min(int(q[:, 0].min()) for q in rp) - 1)
    y0 = max(0, min(int(q[:, 1].min()) for q in rp) - 1)
    x1 = min(w, max(int(q[:, 0].max()) for q in rp) + 2)
    y1 = min(h, max(int(q[:, 1].max()) for q in rp) + 2)
    if x0 >= x1 or y0 >= y1:
        return None
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    off = np.array([x0, y0], np.int32)
    if len(rp) == 1:
        cv2.fillPoly(m, [rp[0] - off], 1)
    else:
        for q in rp:
            mm = np.zeros_like(m)
            cv2.fillPoly(mm, [q - off], 1)
            m ^= mm
    return m.astype(bool), x0, y0


def recolor_strokes(plan: LayerPlan, cel: CelArt, cat: Catalog, upp: float,
                    log=print) -> int:
    """획 그룹 색 = 발자국 아래 원화 픽셀 중 **심에 가까운 절반의 평균** (`_INK_COLOR`).

    획 색은 뼈대 중심선 표본의 중앙값으로 태어난다 (`engine.build_strokes`) —
    선의 **심** 색이다. 배치가 끝난 뒤 sid 그룹마다 발자국을 다시 떠서 그 아래
    원화 픽셀로 갈아 끼운다. 한 획 = 단색 규약은 그대로다 — 그룹 하나에 색
    하나. 실루엣 밖 px는 평균에서 뺀다 (배경은 목표가 아니다).

    **어느 평균인가.** 발자국 전체의 평균(`mean`)은 단색 한 장의 최소제곱
    최적이라 픽셀 자(`rmse_src`·`imp_error_seen`)가 가장 좋다. 그런데 원화
    선은 안티에일리어스 심 1~2px에 옅은 어깨가 붙은 띠이고, 놓인 획은 그 띠
    폭(중앙 ~2px, `descriptor.placed_widths` 실측)을 **한 색으로** 칠한다 —
    전체 평균을 쓰면 획이 심보다 밝기 +18~30(중앙, p90 +50~90) 밝아져 같은
    폭에서 "굵고 옅은 띠"로 읽히고, 셀 재해석(cel.png)의 선보다 옅어 면과의
    구분이 죽는다 (2026-09-02 사용자 관찰, 표준 9장). 사람 눈은 선을 어두운
    심으로 읽으므로 발자국 픽셀 중 **씨앗 색(심)에 가까운 절반**의 평균을
    쓴다(`core`) — 밝기 극단이 아니라 씨앗과의 거리로 고르므로 밝은 선(하이
    라이트)도 같은 규칙에 든다. 실측(01·03·08·09): 심 대비 밝기 차 중앙
    +20 → +6, 보이는 오차는 +12% 나빠지고 도형은 +2~5%(수리 `fix`가 는다).
    `seed`(뼈대 중앙값 그대로)는 그보다 조금 더 어둡고(+4) 수리가 더 는다.
    """
    src = cel.src_rgb
    if src is None or _INK_COLOR == "seed":
        return 0
    w, h = cel.size
    sil = cel.labels >= 0
    groups: dict[int, list[Layer]] = {}
    for lay in plan.layers:
        if lay.label == "ink" and not lay.mask and lay.stroke >= 0:
            groups.setdefault(lay.stroke, []).append(lay)
    n = 0
    picked: list[tuple[list, tuple, int, tuple]] = []   # (그룹, 색, px, 상자)
    for lays in groups.values():
        px: list[np.ndarray] = []
        box = [w, h, 0, 0]
        for lay in lays:
            got = _ink_of(cat, lay, upp, w, h)
            if got is None:
                continue
            m, x0, y0 = got
            mb = m & sil[y0:y0 + m.shape[0], x0:x0 + m.shape[1]]
            if not mb.any():
                continue
            px.append(src[y0:y0 + m.shape[0], x0:x0 + m.shape[1]][mb]
                      .reshape(-1, 3))
            box = [min(box[0], x0), min(box[1], y0),
                   max(box[2], x0 + m.shape[1]), max(box[3], y0 + m.shape[0])]
        if not px:
            continue
        col = np.concatenate(px, axis=0).astype(np.float64)
        if _INK_COLOR == "core" and len(col) >= 4:
            # **심 색** — 발자국 픽셀 중 태어난 색(뼈대 중앙값)에 가까운 절반의
            # 평균. 발자국 평균은 최소제곱 최적이지만 사람 눈은 선의 어두운
            # 심으로 선을 읽는다 — 평균색 한 장은 같은 폭에서 "굵고 옅은 띠"로
            # 보인다 (2026-09-02 사용자 관찰). 어두운 선·밝은 선 어느 쪽이든
            # 성립하도록 밝기 극단이 아니라 씨앗 색과의 거리로 고른다
            seed = np.asarray(lays[0].color, np.float64)
            dist = np.linalg.norm(col - seed[None], axis=1)
            keep = dist <= np.percentile(dist, _INK_CORE_Q)
            col = col[keep] if keep.any() else col
        c = tuple(int(round(v)) for v in col.mean(axis=0))
        picked.append((lays, c, len(col), tuple(box)))
    if _INK_K > 0 and len(picked) > 1:
        table, fst = _fold_ink(picked, _INK_K, _INK_K_DE)
        picked = [(lays, table.get(c, c), k, b) for lays, c, k, b in picked]
        log(msg("  획 색 접기 {a}색 → {b}색 (상한 {k}·반경 ΔE {mv:g}) · "
                "평균 ΔE00 {m:.2f} · 최대 {x:.2f} · 경계 잠금 {e}쌍",
                a=fst["before"], b=fst["after"], k=_INK_K, mv=_INK_K_DE,
                m=fst.get("mean_de00", 0.0), x=fst.get("max_de00", 0.0),
                e=fst.get("locked", 0)))
    tol = 0 if _INK_K > 0 else 2           # 접은 색은 표 그대로 — ±2 잔색이 색 수로 남는다
    for lays, c, _k, _b in picked:
        if max(abs(a - b) for a, b in zip(c, lays[0].color)) <= tol:
            continue                       # 이미 그 색이다 — 손 안 댄다
        for lay in lays:
            lay.color = c
        n += 1
    if n:
        log(msg("  획 색 보정 {n}그룹 — 발자국 아래 원화 평균색", n=n))
    return n


def _fold_ink(picked: list, k_max: int, move: float) -> tuple[dict, dict]:
    """획 그룹 색을 상한 안으로 접는다 (`ramps.fold_palette`).

    잠금은 맨 끝 접기와 같다 — 발자국 상자가 맞닿고(여유 1px) 원화에서 보이는
    경계(ΔE ≥ `ramps.EDGE_DE`)를 이루는 두 획은 못 묶는다. 무게는 발자국 px다.
    """
    from ..celart.ramps import EDGE_DE, fold_palette, _lab

    cols = [c for _l, c, _k, _b in picked]
    wts = np.array([float(k) for _l, _c, k, _b in picked])
    boxes = np.array([b for _l, _c, _k, b in picked], np.float64)
    lab = _lab(np.array(cols, np.float64))
    order = np.argsort(boxes[:, 0], kind="stable")
    bad = []
    for oi in range(len(order)):
        i = int(order[oi])
        for oj in range(oi + 1, len(order)):
            j = int(order[oj])
            if boxes[j, 0] > boxes[i, 2] + 1.0:
                break
            if boxes[j, 1] > boxes[i, 3] + 1.0 or boxes[i, 1] > boxes[j, 3] + 1.0:
                continue
            if cols[i] != cols[j] and float(np.linalg.norm(lab[i] - lab[j])) >= EDGE_DE:
                bad.append((i, j))
    return fold_palette(cols, wts, bad, k_max=k_max, move=move)


def align_to_regions(layers: list, cat: Catalog, upp: float, w: int, h: int,
                     line_mask: np.ndarray, labels: np.ndarray,
                     log=print) -> int:
    """§26-b **놓인 획을 잠정 색 경계에 맞춰 한 칸씩 민다** — 되돌린 장수 반환.

    §26이 "면 지도가 어느 선을 그릴지 고른다"였다면 이쪽은 **어디에 그을지**다.
    선 도안이 서고 나면 그 다음 단(`celart.snap`)이 색 경계를 획 밑으로
    끌어다 앉히는데, 그때까지 획 자리는 신경망 선 지도 하나로만 정해져 있다.
    선 지도가 색 경계에서 한 칸 밀려 있으면 그 밀림이 그대로 굳고, 스냅이
    색면을 거기에 맞춰 끌어와 **두 면의 경계가 통째로 한 칸 밀린다**.

    그래서 스냅 **전에** 반대 방향을 한 번 준다: 획마다 게임 이동 스텝
    (0.5유닛) 여덟 방향을 놓아 보고, **선 커버리지를 하나도 안 잃으면서**
    잠정 색 경계 띠를 더 많이 덮는 수가 있으면 그리로 민다. 선 충실도가
    자격이고 경계 정렬이 점수다 — 순서가 뒤바뀌면 획이 제 선을 떠난다.

    자격을 "제 몫"으로 세면 안 된다: 제 잉크가 선 지도를 덮는 픽셀 수만 보면,
    **다른 획이 이미 덮은 자리**로 옮겨 가면서 수가 그대로일 수 있다 — 그
    사이 저 혼자 덮던 자리가 열린다 (실측 09: 선 커버리지 .975 → .958).
    그래서 획마다 **혼자 덮고 있는 선 픽셀**을 먼저 세고, 그것을 하나도
    안 놓치는 수만 받는다 (같은 자를 `coverage.unique_layers`가 면 쪽에서 쓴다).

    커버 수는 **옮길 때마다 그 자리에서 갱신한다.** 라운드 머리에서 한 번만
    세면 둘이 나눠 덮던 픽셀이 어느 쪽 보호도 못 받아, 그 둘이 같은 라운드에
    함께 옮기면 그 자리가 열린다. 옮긴 도형의 옛 자리를 빼고 새 자리를 더하면
    다음 획이 보는 "혼자 덮는 자리"가 언제나 참이다 — 면 쪽
    `coverage.repair_cut`이 컷에서 하는 일과 같은 이치다.

    셈은 **정확 커버**로 한다. 지표(`ink_near`)가 ±1px 팽창으로 재길래 자도
    팽창으로 맞춰 봤으나 그것은 되레 나빴다 (실측 05: 이동 81 → 566장인데
    선 커버리지는 .9311로 같고 봉인 전 미커버가 1,500 → 4,418표본). 팽창하면
    이웃 획끼리 겹쳐 "혼자 덮는 자리"가 거의 사라져 자격이 헐거워지고, 그만큼
    많이 밀린 획이 스냅을 통해 색면 배치까지 흔든다. **적게, 확실할 때만 민다.**

    라스터는 렌더와 같은 폴리곤 식(`_poly_px`)이라 채점과 결과가 안 갈리고,
    좌표는 놓을 때와 같은 격자에 선다(`Layer.quantized`). 결정적이다.

    색 경계가 없거나(실루엣 한 장 — line 노선) 라운드가 0이면 아무 일도 안 한다.
    """
    if _ALIGN_ROUNDS <= 0 or not len(layers):
        return 0
    sel = labels >= 0
    bnd = np.zeros(labels.shape, bool)
    dh = ((labels[:, :-1] != labels[:, 1:])
          & (labels[:, :-1] >= 0) & (labels[:, 1:] >= 0))
    bnd[:, :-1] |= dh
    bnd[:, 1:] |= dh
    dv = ((labels[:-1] != labels[1:])
          & (labels[:-1] >= 0) & (labels[1:] >= 0))
    bnd[:-1] |= dv
    bnd[1:] |= dv
    if not bnd.any():
        return 0
    # 띠는 게임 격자 — 최소 도형 반폭. 획이 "그 경계 위에 있다"의 자다
    r = max(1, int(round(_min_span(upp))))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    band = cv2.dilate(bnd.astype(np.uint8), k).astype(bool) & sel
    step = 0.5                             # 게임 이동 스텝 (유닛)
    # 커버 수 — 어느 선 픽셀을 **혼자** 덮고 있나의 자. 옮길 때마다 갱신한다
    cnt = np.zeros((h, w), np.uint8)
    for lay in layers:
        if lay.label != "ink" or lay.mask:
            continue
        g = _ink_of(cat, lay, upp, w, h)
        if g is None:
            continue
        gm, gx, gy = g
        box = cnt[gy:gy + gm.shape[0], gx:gx + gm.shape[1]]
        box[gm & (box < 255)] += 1
    moved = 0
    for _ in range(_ALIGN_ROUNDS):
        hit = 0
        for i, lay in enumerate(layers):
            if lay.label != "ink" or lay.mask:
                continue
            got = _ink_of(cat, lay, upp, w, h)
            if got is None:
                continue
            m, x0, y0 = got
            hh, ww = m.shape
            base_al = int((m & band[y0:y0 + hh, x0:x0 + ww]).sum())
            # 이 획이 **혼자** 덮고 있는 선 픽셀 (제 상자 안 좌표)
            keep = (m & line_mask[y0:y0 + hh, x0:x0 + ww]
                    & (cnt[y0:y0 + hh, x0:x0 + ww] == 1))
            n_keep = int(keep.sum())
            best = None
            for dx, dy in ((step, 0.0), (-step, 0.0), (0.0, step), (0.0, -step),
                           (step, step), (step, -step), (-step, step),
                           (-step, -step)):
                c = Layer(**{**lay.__dict__})
                c.x = round(c.x + dx, 4)
                c.y = round(c.y + dy, 4)
                c = c.quantized()
                g2 = _ink_of(cat, c, upp, w, h)
                if g2 is None:
                    continue
                m2, ax, ay = g2
                al2 = int((m2 & band[ay:ay + m2.shape[0],
                                     ax:ax + m2.shape[1]]).sum())
                if al2 <= base_al:
                    continue
                # 자격 — 혼자 덮던 선 픽셀을 하나라도 놓으면 진다.
                # 두 상자의 겹침 안에서만 셈한다 (전장 배열을 안 만든다)
                if n_keep:
                    h2, w2 = m2.shape
                    ox, oy = x0 - ax, y0 - ay
                    sx0, sy0 = max(0, -ox), max(0, -oy)
                    sx1, sy1 = min(ww, w2 - ox), min(hh, h2 - oy)
                    if sx1 <= sx0 or sy1 <= sy0:
                        continue           # 겹치는 자리가 없다 — 전부 잃는다
                    sub = keep[sy0:sy1, sx0:sx1]
                    if int(sub.sum()) != n_keep:
                        continue           # 겹침 밖에 남은 keep 픽셀이 있다
                    if (sub & ~m2[oy + sy0:oy + sy1, ox + sx0:ox + sx1]).any():
                        continue
                if best is None or al2 > best[0]:
                    best = (al2, c)
            if best is not None:
                g3 = _ink_of(cat, best[1], upp, w, h)
                if g3 is not None:
                    m3, cx, cy = g3
                    old = cnt[y0:y0 + hh, x0:x0 + ww]
                    old[m & (old > 0)] -= 1
                    new = cnt[cy:cy + m3.shape[0], cx:cx + m3.shape[1]]
                    new[m3 & (new < 255)] += 1
                    layers[i] = best[1]
                    hit += 1
        moved += hit
        if not hit:
            break
    if moved:
        log(msg("  선↔면 보정: 획 {moved}장을 색 경계로 한 칸씩 밀었다",
                moved=moved))
    return moved
