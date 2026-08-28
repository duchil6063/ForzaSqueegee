"""구멍과 커버리지 — **흰 바탕이 드러나나**를 재고 메운다.

사용자가 보는 결함은 "누가 칠했나"가 아니라 흰 바탕이 드러나는가이므로 그것을
그대로 잰다 (`silhouette_cover`·`count_hole_clusters`). 메우는 손은 둘이다:
기존 레이어를 한 스텝 키우는 성장(레이어 0장)과 군집당 한 장을 놓는 메움.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..catalog import Catalog
from ..celart import CelArt
from ..model import Layer, LayerPlan
from ..price import _HOLE_PRICE
from .geometry import _layer, _mask_px, _min_span, _poly_px
from .vocabulary import _FILL_SHAPE


def _sil_rim(labels: np.ndarray, upp: float) -> np.ndarray:
    """실루엣 **최외곽 테** — 도형이 경계에 못 맞추는 폭. 구멍으로 안 센다.

    폭은 게임 **이동 스텝**(0.5유닛)이 정한다. 하강이 그보다 잘게 못 움직이므로
    이 테 안쪽으로는 어떤 도형도 경계에 딱 맞출 수 없다 — 넘치거나 물러서거나
    둘 중 하나이고, 물러선 자리를 구멍으로 세면 그것을 메우는 도형이 다시
    넘쳐 순환이 된다. 상수가 아니라 격자 유도값이다 (0.67~1.69px = 1~2겹).

    겹수는 정수라야 하는데 이동 스텝은 그렇지 않다(1~2px 사이의 소수다).
    **내림하고 최소 한 겹**으로 잡는다 — 올림은 안 메우기로 한 테가 실제 폭보다
    넓어져 흰 테가 그대로 드러난다.
    """
    k = max(1, int(0.5 / max(upp, 1e-6)))
    inside = (labels >= 0).astype(np.uint8)
    core = cv2.erode(inside, np.ones((3, 3), np.uint8), iterations=k)
    return inside.astype(bool) & ~core.astype(bool)


def silhouette_cover(plan: LayerPlan, cel: CelArt, cat: Catalog) -> float:
    """플랜이 **실제로** 칠한 실루엣 비율 — 어느 레이어가 칠했는지는 안 묻는다.

    영역별 자가 커버리지(`stats["uncovered_px"]`)와 다른 자다. 그쪽은 "내 면을
    내 도형이 덮었나"라서, **값이 안 되는 영역을 이웃 색에 맡기는 것**이 설계인
    가격 설계에서는 설계대로 돌수록 수치가 나빠진다 (사람도 작은 조각을 따로
    안 칠하고 옆 색이 흐르게 둔다). 사용자가 보는 결함은 "누가 칠했나"가 아니라
    **흰 바탕이 드러나나**이므로 그것을 그대로 잰다.
    """
    w, h = cel.size
    cov = np.zeros((h, w), np.uint8)
    for lay in plan.layers:
        for p in _poly_px(cat, lay, plan.units_per_px, w, h):
            cv2.fillPoly(cov, [np.round(p).astype(np.int32)], 1)
    sil = cel.labels >= 0
    n = int(sil.sum())
    return 1.0 if not n else float((cov.astype(bool) & sil).sum()) / n


def count_hole_clusters(plan: LayerPlan, cel: CelArt, cat: Catalog,
                        min_px: int = 1,
                        value: np.ndarray | None = None,
                        price: float = 0.0) -> int:
    """현재 플랜 커버리지 기준 실루엣 구멍 군집 수 — report `holes` 검사용."""
    w, h = cel.size
    cov = np.zeros((h, w), np.uint8)
    for lay in plan.layers:
        for p in _poly_px(cat, lay, plan.units_per_px, w, h):
            cv2.fillPoly(cov, [np.round(p).astype(np.int32)], 1)
    holes = ((cel.labels >= 0) & ~cov.astype(bool)
             & ~_sil_rim(cel.labels, plan.units_per_px))
    ncc, lab, cstats, _ = cv2.connectedComponentsWithStats(
        holes.astype(np.uint8), connectivity=8)
    if ncc <= 1:
        return 0
    keep = cstats[1:, cv2.CC_STAT_AREA] >= min_px
    if price and value is not None:
        # 가격 설계의 "보이는 구멍" — 메움과 **같은 자**로 센다. 게이트가
        # 메움보다 엄하면 못 고칠 것을 요구하는 게이트가 된다
        wsum = np.zeros(ncc, np.float64)
        np.add.at(wsum, lab[holes], value[holes])
        keep &= wsum[1:] >= price * _HOLE_PRICE
    return int(keep.sum())


def grow_covers(plan: LayerPlan, cel: CelArt, cat: Catalog,
                log=print, passes: int = 2) -> int:
    """기존 레이어를 양자화 한 스텝(스케일 +0.01 ≈ 지름 1.7px) 키워 인접 구멍을
    흡수한다 — **새 레이어 0장**. 잔여 구멍의 74%가 경계 부스러기라(실측),
    구멍마다 메움 타원을 쓰면 ~900장이 들지만 성장은 공짜다.

    가시 변화는 "이 레이어 위에 아무도 없는 px"뿐이다: 구멍 px는 이득,
    먼저 그린 레이어 소유 px는 색 단차 해악, 실루엣 밖은 오염.
    해악이 작은 성장만 적용한다 — 색 단차 합 `harm > 90.0`(ΔE×px)이거나
    실루엣 밖 `spill > 2`px이면 안 키운다. 대상은 단일 고리 도형(타원·사각·
    막대)의 비선화 레이어 — 선화 획을 키우면 선이 눈에 띄게 굵어진다.
    """
    w, h = cel.size
    upp = plan.units_per_px
    sil = cel.labels >= 0
    lab_of = np.array([cv2.cvtColor(np.array([[l.rgb()]], np.uint8),
                                    cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32)
                       for l in plan.layers])
    owner = np.full((h, w), -1, np.int32)
    for i, lay in enumerate(plan.layers):
        for p in _poly_px(cat, lay, upp, w, h):
            cv2.fillPoly(owner, [np.round(p).astype(np.int32)], i)
    grown = 0
    for _ in range(passes):
        changed = 0
        for i, lay in enumerate(plan.layers):
            if lay.label == "ink" or lay.shape not in ("A_01", "A_02", "A_22"):
                continue
            big = Layer(**{**lay.__dict__})
            big.sx = round(big.sx + (0.01 if big.sx >= 0 else -0.01), 4)
            big.sy = round(big.sy + (0.01 if big.sy >= 0 else -0.01), 4)
            polys = _poly_px(cat, big, upp, w, h)
            xs = np.concatenate([p[:, 0] for p in polys])
            ys = np.concatenate([p[:, 1] for p in polys])
            x0 = max(0, int(xs.min()) - 2); y0 = max(0, int(ys.min()) - 2)
            x1 = min(w, int(xs.max()) + 3); y1 = min(h, int(ys.max()) + 3)
            if x0 >= x1 or y0 >= y1:
                continue
            roi = (x0, y0, x1, y1)
            g = _mask_px(cat, big, upp, w, h, roi) & ~_mask_px(cat, lay, upp, w, h, roi)
            if not g.any():
                continue
            own = owner[y0:y1, x0:x1]
            vis = g & (own < i)               # 위에 아무도 없는 성장분
            ben = int(np.count_nonzero(vis & (own == -1) & sil[y0:y1, x0:x1]))
            if ben == 0:
                continue
            spill = int(np.count_nonzero(vis & ~sil[y0:y1, x0:x1]))
            oth = vis & (own >= 0)
            harm = 0.0
            if oth.any():
                harm = float(np.linalg.norm(
                    lab_of[own[oth]] - lab_of[i][None], axis=1).sum())
            if spill > 2 or harm > 90.0:
                continue
            plan.layers[i] = big
            own[vis] = i
            grown += 1
            changed += 1
        if not changed:
            break
    if grown:
        log(f"  레이어 성장 {grown}회 (구멍 흡수, 추가 레이어 0장)")
    return grown


def fill_holes(plan: LayerPlan, cel: CelArt, cat: Catalog,
               log=print, min_px: int = 1, max_layers: int = 600,
               value: np.ndarray | None = None, price: float = 0.0) -> int:
    """플랜 전체 커버리지 대비 실루엣 구멍(흰 핀홀)을 셀 색 타원으로 메꾼다.

    핀홀은 영역 경계의 양자화 슬리버·포기 잔여·프루닝 컷에서 오므로 **프루닝
    뒤에** 불러야 한다 — 앞에서 메꾸면 "작은 px×큰 ΔE < 큰 px×작은 ΔE"라
    프루닝이 도로 걷어낸다 (실측: 600장 메꿔도 핀홀 그대로). 레이어는 "hole"
    라벨 — 재컷 보호 대상이다 (보호 없이는 재컷이 작은 메움부터 걷어 컷→메움
    반복이 잔여 ~800군집 평형에 갇혔다. 실측: _BG_PEN 상수 벌점으로도 부족).

    군집이 **굽은 사슬**이면 곧게 편 타원 한 장으로 안 덮인다 — 실루엣 밖
    초과가 나면 주축 투영 중앙에서 반으로 갈라 조각마다 제 타원을 세운다
    (아래 out_bg 분기). 이 쪼개기가 없으면 대체 도형(내접원)이 양자화 최소
    도형으로 뭉개져 1px 테를 한 장에 3px씩밖에 못 줍고, 컷→메움이 평형에
    갇힌다. 메움 레이어는 **맨 뒤에 얹는다** — 맨 앞에 넣으면 소유 px가
    구멍 px뿐이라 컷이 도로 걷어 구멍이 라운드마다 다시 열린다 (실측).
    """
    upp = plan.units_per_px
    w, h = cel.size
    cov = np.zeros((h, w), np.uint8)
    for lay in plan.layers:
        for p in _poly_px(cat, lay, upp, w, h):
            cv2.fillPoly(cov, [np.round(p).astype(np.int32)], 1)
    holes = ((cel.labels >= 0) & ~cov.astype(bool)
             & ~_sil_rim(cel.labels, plan.units_per_px))
    flat = cel.flat_render()
    # 근접 군집(2px 이내)을 그룹으로 묶는다 — 잔여의 74%가 경계 인접 부스러기라
    # 같은 경계를 따라 줄지은 조각들을 가는 회전 타원 **하나**로 덮는 편이
    # 군집당 한 장보다 훨씬 싸다 (레이어 수 = 예산이다)
    grp = cv2.dilate(holes.astype(np.uint8),
                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    ncc, cc, cstats, _ = cv2.connectedComponentsWithStats(grp, connectivity=8)
    gsz = np.bincount(cc[holes].ravel(), minlength=ncc)   # 그룹별 실제 구멍 px
    n = 0
    order = np.argsort(-gsz[1:]) + 1
    todo = int((gsz[1:] >= min_px).sum())
    done = 0
    for ci in order:
        if n >= max_layers:
            log(f"  경고: 구멍 메움 상한 — 군집 {todo - done}개 남음")
            break
        if gsz[ci] < min_px:
            break
        x0 = cstats[ci, cv2.CC_STAT_LEFT]; y0 = cstats[ci, cv2.CC_STAT_TOP]
        x1 = x0 + cstats[ci, cv2.CC_STAT_WIDTH]; y1 = y0 + cstats[ci, cv2.CC_STAT_HEIGHT]
        rem = (cc[y0:y1, x0:x1] == ci) & holes[y0:y1, x0:x1]
        # 가격 — 흰 자국이 λ의 `_HOLE_PRICE`몫에 못 미치면 안 메운다
        if price and value is not None and \
                float(value[y0:y1, x0:x1][rem].sum()) < price * _HOLE_PRICE:
            continue
        # 그룹당 ≤4장 — 타원 한 장이 못 덮은 잔여(오목·꺾인 사슬)에 겹쳐 찍는다.
        # 한 장만 찍고 넘어가면 잔여가 다음 라운드에 군집으로 재등장해 컷→메움
        # 반복이 수렴하지 않는다 (실측: 라운드당 잔여 465→403→391 정체)
        stack = [rem.copy()]
        used = 0
        while stack and used < 4 and n < max_layers:
            piece = stack.pop() & rem
            if not piece.any():
                continue
            area = int(np.count_nonzero(piece))
            ys2, xs2 = np.nonzero(piece)
            pw = np.stack([xs2, ys2], axis=1).astype(np.float64)
            ctr = pw.mean(axis=0)
            dtc = cv2.distanceTransform(np.pad(piece, 1).astype(np.uint8),
                                        cv2.DIST_L2, 3)[1:-1, 1:-1]
            r0 = float(dtc.max())
            # 색 표본은 군집 **안**에서 — 오목 군집은 무게중심이 군집 밖(이웃
            # 면·배경)일 수 있어 그대로 읽으면 엉뚱한 색 타원이 된다
            spy, spx = np.unravel_index(int(dtc.argmax()), dtc.shape)
            color = tuple(int(v) for v in flat[y0 + spy, x0 + spx])
            if len(pw) > 4:
                cov_ = np.cov((pw - ctr).T)
                evals, evecs = np.linalg.eigh(cov_)
                d = evecs[:, int(np.argmax(evals))]
                theta = float(np.arctan2(d[1], d[0]))
                proj = (pw - ctr) @ d
                perp = (pw - ctr) @ np.array([-d[1], d[0]])
                # 넉넉히 키운다(+1.2px): 양자화 최소 스케일을 뚫고 1px 구멍도
                # 확실히 덮으며, 스필은 구멍 색 ≈ 주변 셀 색이라 흰 점보다 낫다
                a = float(np.abs(proj).max()) + 1.2
                # 굽은 가는 군집(잘린 획 자리)의 수직 퍼짐을 타원 반폭으로 쓰면
                # 대형 얼룩이 된다 — 실제 두께(내접 반경) 기준으로 캡
                b = min(float(np.abs(perp).max()) + 1.2, r0 * 1.5 + 1.5)
            else:
                theta, a, b = 0.0, 1.6, 1.6
            lay = _layer(_FILL_SHAPE, x0 + ctr[0], y0 + ctr[1], a, b, theta, 0.0,
                         color, upp, w, h, label="hole").quantized()
            # 실루엣 밖 초과 검사 — 메움 타원이 배경으로 삐져나오면(경계 구멍에서
            # 실측: 어깨 밖 분홍 타원) 내접원으로 대체한다
            gx0 = max(0, int(x0 + ctr[0] - a - 4)); gy0 = max(0, int(y0 + ctr[1] - a - 4))
            gx1 = min(w, int(x0 + ctr[0] + a + 5)); gy1 = min(h, int(y0 + ctr[1] + a + 5))
            mm = _mask_px(cat, lay, upp, w, h, (gx0, gy0, gx1, gy1))
            out_bg = int(np.count_nonzero(mm & (cel.labels[gy0:gy1, gx0:gx1] < 0)))
            if out_bg > max(40, int(1.5 * area)):
                # **굽은 사슬은 곧게 편 타원이 배경을 문다** — 주축 투영 중앙에서
                # 반으로 갈라 조각마다 제 타원을 세운다 (획 배치가 최대 이탈점에서
                # 쪼개는 것과 같은 결). 안 갈랐을 때의 대체(내접원)는 양자화 최소
                # 도형으로 뭉개져 1px 초승달을 3px씩밖에 못 줍는다 — 가는 머리칼
                # 가닥의 바깥 테 같은 긴 초승달에 여러 장을 쓰고도 대부분이 남아
                # 라운드를 다 쓰고 게이트가 실패한다.
                # 멈추는 자리는 **양자화 최소 도형**이다 — 그보다 짧게 갈라도
                # 찍을 수 있는 것이 없다. 상수가 아니라 게임 입력 스텝에서 나온다.
                # 점 수 조건도 유도값이다: 반씩 갈랐을 때 양쪽이 각각 제 타원을
                # 세울 수 있어야 하므로 위 `len(pw) > 4`의 두 배가 필요하다
                if (len(pw) > 8
                        and float(np.abs(proj).max()) > 2 * _min_span(upp)):
                    med = float(np.median(proj))
                    lo = np.zeros_like(piece); hi = np.zeros_like(piece)
                    lo[ys2[proj <= med], xs2[proj <= med]] = True
                    hi[ys2[proj > med], xs2[proj > med]] = True
                    if lo.any() and hi.any():
                        stack.append(lo)
                        stack.append(hi)
                        continue
                lay = _layer(_FILL_SHAPE, x0 + spx, y0 + spy, r0 + 1.2, r0 + 1.2,
                             0.0, 0.0, color, upp, w, h, label="hole").quantized()
            covered = piece & _mask_px(cat, lay, upp, w, h, (x0, y0, x1, y1))
            if not covered.any():          # 양자화로 0px 도형 — 더 찍어도 헛돈다
                continue
            plan.layers.append(lay)
            n += 1
            used += 1
            rem &= ~covered
            left = piece & rem
            if left.any():
                stack.append(left)
        if not rem.any():
            done += 1
    if n:
        log(f"  구멍 메움 {n}장 (군집 {done}/{todo})")
    return n
