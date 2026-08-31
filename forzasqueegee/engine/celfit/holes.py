"""구멍과 커버리지 — **흰 바탕이 드러나나**를 재고 메운다.

사용자가 보는 결함은 "누가 칠했나"가 아니라 흰 바탕이 드러나는가이므로 그것을
그대로 잰다 (`silhouette_cover`·`count_hole_clusters`). 메우는 손은 둘이다:
기존 레이어를 한 스텝 키우는 성장(레이어 0장)과 군집당 한 장을 놓는 메움.
"""

from __future__ import annotations

import cv2
import numpy as np

from ...i18n import msg
from ..catalog import Catalog
from ..celart import CelArt
from ..model import Layer, LayerPlan
from ..celart.marks import _MARK_DE
from ..price import _HOLE_PRICE
from ..stop import stop_here
from .geometry import (_SUB_BITS, _layer, _mask_px, _min_span,
                       _poly_px, poly_bbox, poly_mask)
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


# 성장이 감수하는 색 단차의 상한 (ΔE×px) — **값의 자**다. §18 봉인은 이
# 자를 안 쓴다 (`harm_per_px` 참조).
_GROW_HARM = 90.0


def _growable(cat: Catalog, lay: Layer, ink_long: bool = False) -> bool:
    """이 레이어를 한 스텝 키워도 되나 — **어휘가 답한다** (목록이 아니라).

    자격은 "단일 고리·불투명 도형"이고, 채움 어휘 76종이 **전부** 그렇다
    (`vocabulary._FILL_ALL` — 단일 루프·불투명이 그 어휘의 선발 조건이다).
    그래서 판정을 이름 목록이 아니라 카탈로그에 묻는다: 목록으로 몇 종만
    열어 두면 방패·삼각·초승달로 그린 면이 경계 부스러기를 공짜로 못 먹고
    메움 도형을 부른다 (실측 S0-09: 봉인이 성장 13회에 그치고 268장을 샀다).

    선화 획(`ink`)은 기본으로 뺀다 — 키우면 선이 눈에 띄게 굵어진다. 다만
    `ink_long`이면 **긴 축만** 키우는 조건으로 넣는다 (아래 `grow_covers`의
    후보 축 선택): 획의 긴 축은 길이라, 키워도 굵어지지 않고 **끝이 조금
    길어질 뿐**이다. 두 획이 만나는 모서리에 남는 반 픽셀짜리 쐐기가 잔여
    미커버의 태반이라(실측 W4-01: 1px 이하 군집 227개 중 79.5%가 잉크에서
    1px 안), 그 자리를 도형을 사서 메우는 대신 획 끝을 잇는 것이 사람 손에도
    가깝다. 뺄셈 마스크와 그라디언트 도형은 늘 뺀다.
    """
    if lay.mask:
        return False
    if lay.label == "ink" and not ink_long:
        return False
    sh = cat[lay.shape]
    return len(sh.loops) == 1 and sh.gradient is None


def _owner_map(plan: LayerPlan, cat: Catalog, upp: float, w: int, h: int,
               ss: int) -> np.ndarray:
    """표본마다 **맨 위 레이어 index** (배경 -1) — 성장의 가시성 판정용."""
    own = np.full((h * ss, w * ss), -1, np.int32)
    for i, lay in enumerate(plan.layers):
        for p in _poly_px(cat, lay, upp, w, h):
            cv2.fillPoly(own, [np.round(p * ss * (1 << _SUB_BITS)).astype(np.int32)],
                         i, shift=_SUB_BITS)
    return own


def grow_covers(plan: LayerPlan, cel: CelArt, cat: Catalog,
                log=print, passes: int = 2,
                need: np.ndarray | None = None, ss: int = 1,
                harm_per_px: float | None = None,
                ink_long: bool = False) -> int:
    """기존 레이어를 양자화 한 스텝(스케일 +0.01 ≈ 지름 1.7px) 키워 인접 구멍을
    흡수한다 — **새 레이어 0장**. 잔여 구멍의 74%가 경계 부스러기라(실측),
    구멍마다 메움 타원을 쓰면 ~900장이 들지만 성장은 공짜다.

    가시 변화는 "이 레이어 위에 아무도 없는 px"뿐이다: 구멍 px는 이득,
    먼저 그린 레이어 소유 px는 해악, 실루엣 밖은 오염.

    **해악은 목표 대비 오차의 증가로 잰다** — 아래 색과의 단차가 아니라.
    두 자는 다르다: 아래 색이 그 자리의 **셀 목표와 이미 다르면** 그 위를
    덮는 것은 해악이 아니라 수리다. 단차로 재던 자는 그것을 못 가려서, 경계
    부스러기를 먹으러 가는 성장이 "반대편에서 남의 색을 문다"는 이유로 늘
    막혔다 (실측 W4-01: 미커버 8,432표본에 성장 317회). 새 자는 이미 있는
    목표(`cel.flat_render`)만 쓴다 — 배치·수리·메움이 전부 그 목표를 본다.

    해악이 작은 성장만 적용한다 — 오차 증가 합이 상한(`_GROW_HARM`)을 넘거나
    실루엣 밖 `spill > 2`px이면 안 키운다. 대상 판정은 `_growable`.

    `need`(bool, 아래 `ss` 격자)를 주면 이득을 **그 자리로 한정한다** —
    §18 봉인이 쓰는 자리다.

    `ss > 1`이면 판정 전체를 **슈퍼샘플 격자**에서 한다. 1px 격자에서는
    "이미 덮은 픽셀"이라 성장분(`g`)에 안 들어오지만 그 픽셀 **안**에 미커버
    표본이 있는 자리가 있고(벡터가 격자보다 잘다), 1x로는 그 자리를 겨눌
    길이 자체가 없다. 이득·해악·스필은 전부 1x px 눈금으로 환산해 견준다 —
    상한이 격자 배율에 안 흔들리게.

    `harm_per_px`를 주면 상한이 상수가 아니라 **얻는 구멍 px당** 그만큼이
    된다 (봉인 전용 — 봉인은 λ와도 색과도 거래하지 않지만, 성장이 건드리는
    자리가 얻는 자리보다 훨씬 넓으면 그 자체가 새 얼룩이라 비례 상한은 둔다).

    `ink_long`이면 선화 획도 **긴 축으로만** 키운다 (`_growable` 문서).
    """
    w, h = cel.size
    upp = plan.units_per_px
    sil = cel.labels >= 0
    if ss > 1:
        sil = np.repeat(np.repeat(sil, ss, axis=0), ss, axis=1)
    lab_of = np.array([cv2.cvtColor(np.array([[l.rgb()]], np.uint8),
                                    cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32)
                       for l in plan.layers])
    # 목표 Lab — 해악의 자 (위 문서). 한 번만 만든다
    tgt = cv2.cvtColor(cel.flat_render(), cv2.COLOR_RGB2LAB).astype(np.float32)
    if ss > 1:
        tgt = np.repeat(np.repeat(tgt, ss, axis=0), ss, axis=1)
    owner = _owner_map(plan, cat, upp, w, h, ss)
    area = float(ss * ss)                  # 표본 → 1x px 환산
    grown = 0
    for _ in range(passes):
        changed = 0
        for i, lay in enumerate(plan.layers):
            stop_here()
            if not _growable(cat, lay, ink_long):
                continue
            cur = _poly_px(cat, lay, upp, w, h)   # 후보 셋이 함께 쓴다
            best = None
            # 획은 **긴 축 하나만** — 짧은 축은 곧 선 굵기다
            axes = ((1, 1), (1, 0), (0, 1))
            if lay.label == "ink":
                axes = ((1, 0),) if abs(lay.sx) >= abs(lay.sy) else ((0, 1),)
            # **축을 따로 키워 본다.** 두 축을 함께 키우면 도형 둘레 전체가
            # 한 스텝 나가므로, 구멍 하나를 먹자고 반대편에서 남의 색을 그만큼
            # 문다 — 큰 바탕 도형은 그 해악 합이 늘 상한을 넘어 **영영 못
            # 큰다** (실측 W1-01: 미커버 8,859표본에 성장 58회). 한 축만
            # 키우면 문는 둘레가 절반이라 같은 상한에서 훨씬 자주 통과한다.
            # 순서가 곧 동점 우선순위다 (둘 다 → 가로 → 세로).
            for gx, gy in axes:
                big = Layer(**{**lay.__dict__})
                if gx:
                    big.sx = round(big.sx + (0.01 if big.sx >= 0 else -0.01), 4)
                if gy:
                    big.sy = round(big.sy + (0.01 if big.sy >= 0 else -0.01), 4)
                polys = _poly_px(cat, big, upp, w, h)
                box = poly_bbox(polys, w, h, ss, pad=2)
                if box is None:
                    continue
                x0, y0, x1, y1 = box
                shape = (y1 - y0, x1 - x0)
                g = (poly_mask(polys, shape, x0 / ss, y0 / ss, ss)
                     & ~poly_mask(cur, shape, x0 / ss, y0 / ss, ss))
                if not g.any():
                    continue
                own = owner[y0:y1, x0:x1]
                vis = g & (own < i)           # 위에 아무도 없는 성장분
                ben = int(np.count_nonzero(
                    vis & (need[y0:y1, x0:x1] if need is not None
                           else (own == -1) & sil[y0:y1, x0:x1])))
                if ben == 0:
                    continue
                spill = np.count_nonzero(vis & ~sil[y0:y1, x0:x1]) / area
                oth = vis & (own >= 0)
                harm = 0.0
                if oth.any():
                    # 목표 대비 오차가 **얼마나 늘어나나** (위 문서). 줄어드는
                    # 자리는 0으로 두고 세지 않는다 — 이득으로 치면 성장이
                    # 수리를 흉내 내며 커지고, 그것은 이 손의 일이 아니다.
                    # JND 아래 증가는 화면이 안 바뀐다 (§15와 같은 자)
                    t = tgt[y0:y1, x0:x1][oth]
                    d_new = np.linalg.norm(lab_of[i][None] - t, axis=1)
                    d_old = np.linalg.norm(lab_of[own[oth]] - t, axis=1)
                    harm = float(np.maximum(d_new - d_old - _MARK_DE,
                                            0.0).sum()) / area
                cap = _GROW_HARM if harm_per_px is None                     else max(_GROW_HARM, harm_per_px * ben / area)
                if spill > 2 or harm > cap:
                    continue
                if best is None or ben > best[0]:
                    best = (ben, big, vis, x0, y0, x1, y1)
            if best is None:
                continue
            _ben, big, vis, x0, y0, x1, y1 = best
            plan.layers[i] = big
            owner[y0:y1, x0:x1][vis] = i
            grown += 1
            changed += 1
        if not changed:
            break
    if grown:
        log(msg("  레이어 성장 {grown}회 (구멍 흡수, 추가 레이어 0장)",
                grown=grown))
    return grown


def fill_holes(plan: LayerPlan, cel: CelArt, cat: Catalog,
               log=print, min_px: int = 1, max_layers: int = 600,
               value: np.ndarray | None = None, price: float = 0.0,
               holes: np.ndarray | None = None,
               label: str = "hole", group_r: int = 2,
               at: int | None = None) -> int:
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
    갇힌다.

    `group_r`은 조각을 묶는 반경(px)이다 — 클수록 한 장이 더 멀리까지 줄지은
    조각을 함께 덮는다. 지나치게 뻗으면 아래 실루엣 초과 판정이 그룹을 반씩
    가르므로 넉넉히 잡아도 된다.

    `holes`(1x bool)를 주면 그 자리를 그대로 메운다 — 1px 격자의 미커버가
    아니라 §18 슈퍼샘플 불변이 겨눈 자리를 받는 통로다. `label`은 그렇게 놓인
    장의 라벨이고(seal은 "seal"), 프루닝 보호·구조 지표가 그것으로 가른다.

    `at`을 주면 그 index에 **끼운다** (기본은 맨 뒤). 구멍 px는 정의상 아무
    레이어도 안 덮는 자리라 z가 어디든 그 px에서는 보인다 — 스택 바닥(0)에
    끼우면 타원의 스필이 이웃 면과 선화 **밑**으로 들어가 경계 잔티가 안
    보인다 (스필을 색 경계 안쪽·선 아래로 유도하는 자리). 컷이 도로 걷는
    문제는 라벨 보호("hole"/"seal")가 막는다.
    """
    upp = plan.units_per_px
    w, h = cel.size
    if holes is None:
        cov = np.zeros((h, w), np.uint8)
        for lay in plan.layers:
            for p in _poly_px(cat, lay, upp, w, h):
                cv2.fillPoly(cov, [np.round(p).astype(np.int32)], 1)
        holes = ((cel.labels >= 0) & ~cov.astype(bool)
                 & ~_sil_rim(cel.labels, plan.units_per_px))
    flat = cel.flat_render()
    # 근접 군집을 그룹으로 묶는다 — 잔여의 74%가 경계 인접 부스러기라 같은
    # 경계를 따라 줄지은 조각들을 가는 회전 타원 **하나**로 덮는 편이 군집당
    # 한 장보다 훨씬 싸다 (레이어 수 = 예산이다).
    #
    # **색으로는 안 가른다.** 한 장은 색이 하나뿐이라 색까지 맞춰 묶는 편이
    # 옳아 보이지만, 실측이 그 반대다 (W3-11: 그룹 158 → 408개, 봉인 도형
    # 265 → 438장): 조각이 하필 색 경계를 따라 흩어져 있어 색으로 가르면
    # 한 덩이가 색 수만큼 쪼개진다. 그러면서 얻는 것은 조각 몇 px의 색이고,
    # 잃는 것은 그만큼의 레이어다 — 덮이기만 하면 차 도색은 안 비친다.
    ys, xs = np.nonzero(holes)
    if not len(ys):
        return 0
    grp = cv2.dilate(holes.astype(np.uint8), cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * group_r + 1, 2 * group_r + 1)))
    _n, cc = cv2.connectedComponents(grp, connectivity=8)
    gid = cc[ys, xs].astype(np.int64)
    gids, inv, gsz = np.unique(gid, return_inverse=True, return_counts=True)
    n = 0
    order = np.argsort(-gsz, kind="stable")
    todo = int((gsz >= min_px).sum())
    done = 0
    for gi in order:
        stop_here()
        if n >= max_layers:
            log(msg("  경고: 구멍 메움 상한 — 군집 {n}개 남음", n=todo - done))
            break
        if gsz[gi] < min_px:
            break
        sel = inv == gi
        gy = ys[sel]; gx = xs[sel]
        x0 = int(gx.min()); y0 = int(gy.min())
        x1 = int(gx.max()) + 1; y1 = int(gy.max()) + 1
        rem = np.zeros((y1 - y0, x1 - x0), bool)
        rem[gy - y0, gx - x0] = True
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
                         color, upp, w, h, label=label).quantized()
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
                             0.0, 0.0, color, upp, w, h, label=label).quantized()
            covered = piece & _mask_px(cat, lay, upp, w, h, (x0, y0, x1, y1))
            if not covered.any():          # 양자화로 0px 도형 — 더 찍어도 헛돈다
                continue
            if at is None:
                plan.layers.append(lay)
            else:
                plan.layers.insert(at, lay)
                at += 1
            n += 1
            used += 1
            rem &= ~covered
            left = piece & rem
            if left.any():
                stack.append(left)
        if not rem.any():
            done += 1
    if n:
        log(msg("  구멍 메움 {n}장 (군집 {done}/{todo})",
                n=n, done=done, todo=todo))
    return n
