"""잔차 수리 — 배치가 끝난 뒤 "덮였지만 색이 틀린" 자국을 고친다.

`repair_mismatch`가 응집 자국(얼룩 음영·빗나간 획·끊긴 선)을 보정 도형으로
덮는다.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..catalog import Catalog
from ..celart import CelArt
from ..model import LayerPlan
from .geometry import _layer, _mask_px, _poly_px
from .vocabulary import _FILL_SHAPE


# 덧칠 수리가 선 지도 자리에서 세는 **이득의 가중** (1.0 = 다 센다,
# 0.0 = 안 센다). 근거는 `repair_mismatch` 문서
_FIX_LINEGAIN = 0.5


def repair_mismatch(plan: LayerPlan, cel: CelArt, cat: Catalog,
                    log=print, thr: float = 12.0, thr_line: float = 25.0,
                    min_px: int = 16, min_gain: float = 900.0,
                    max_layers: int = 400, protect_lines: bool = False) -> int:
    """"덮였지만 색이 틀린" 응집 잔차를 보정 도형으로 수리한다.

    구멍 메움은 미커버 px만 보고, 채움은 영역당 도형 몇 장 근사에서 멈춘다 —
    그 사이의 "커버됐지만 셀 목표와 색이 다른" 자리(얼룩진 음영, 빗나간 획
    자국, 끊긴 선)는 어느 단계도 안 고쳤다 (9차 판정 "완성되지 못한 부분"의
    실체. 컷 전 렌더에도 같은 오차라 프루닝 몫이 아니고, 예산 6,000을 줘도
    fit은 3,100~3,300장에서 수요를 소진했다 — 실측).

    렌더 vs flat_render의 ΔE 응집 군집을 fill_holes와 같은 PCA 타원 ≤4장으로
    덮는다. 세 갈래로 나뉘고 **삽입 위치가 다르다**:

    - 면 수리(선화 아닌 px, 소유자가 선화 블록 아래): 영역 색 타원을 선화
      블록 바로 앞에 끼운다 — 선은 위에 남는다.
    - 덧칠 수리(선화 아닌 px, 소유자가 선화·메움 블록): 빗나간 획 자국을
      영역 색으로 덮는 수정액 — 맨 뒤에 얹는다.
    - 선 수리(line_mask px): 끊긴 획 틈을 원화 색 도형으로 잇는다 — 맨 뒤,
      label "ink" (그리기 순서·컷 보호 모두 선화와 같다). 문턱은 따로 높게
      (thr_line): 선망 전체에는 "한 획 = 단색 vs 원화의 변하는 선 색" 저강도
      편차가 깔려 있어(실측 두 타깃 모두 ~100k px, 육안 무해) thr로 자르면
      선망이 통째로 거대 그룹이 된다 — 획 **누락**(밑의 면 색이 드러나
      ΔE 30~50)만 조준한다.

    그룹을 영역 id로 가르지 않는다 — 빗나간 획·경계 얼룩은 여러 영역을
    가로질러 id 분할이 min_px 미만 파편을 만든다 (실측: 덧칠 수요의 절반이
    파편으로 걸러졌다). 대신 패치 색을 **패치마다** 잔여의 내접점에서 목표
    (flat)를 표본한다 — fill_holes와 같은 수법이라 경계에 걸쳐도 색이 맞다.

    채택 게이트는 순이득(고친 ΔE − 새로 만든 ΔE − 실루엣 밖 스필 80/px)
    ≥ min_gain — 증류된 플랜의 컷 바닥(시각 영향 ~800)보다 높게 잡아,
    재컷이 수리를 도로 걷어내는 진동을 막는다. label "fix"는 컷 보호를
    **안 받는다** — 영향이 바닥을 밑돌면 잘려도 맞다.

    덧칠만은 순이득에서 **선 지도 자리의 이득을 절반만 센다**(`_FIX_LINEGAIN`).
    덧칠은 선화 블록 **위**에 얹히므로 그 자리를 덮으면 획이 가려지는데,
    획이 지금 비어 보이면(err 큼) 순이득은 덮을수록 오른다고 채점한다 — 이득이
    구조적으로 뒤집혀 있다. 덮인 획은 소유 px가 0이 되어 컷 보호(`ink`)까지
    잃고 잘린다. 비용(새로 만드는 ΔE)은 그대로 세므로 "선 자리를 통째로
    빼기"(이득도 비용도 빠져 덧칠이 오히려 늘던 갈래)와 다르다.

    면 수리는 면제한다 — 선화 블록 **아래**에 끼우므로 획을 못 가린다.
    """
    line_pen = 40.0 if protect_lines else 0.0
    upp = plan.units_per_px
    w, h = cel.size
    flat = cel.flat_render()
    flat_lab = cv2.cvtColor(flat, cv2.COLOR_RGB2LAB).astype(np.float32)
    insil = cel.labels >= 0
    # 현재 렌더의 px 소유자 (최상위 레이어 index, -1 = 미커버)
    owner = np.full((h, w), -1, np.int32)
    for i, lay in enumerate(plan.layers):
        polys = [np.round(p).astype(np.int32)
                 for p in _poly_px(cat, lay, upp, w, h)]
        if len(polys) == 1:
            cv2.fillPoly(owner, polys, i)
        else:                               # 짝홀 규칙 (구멍 있는 도형)
            mm = np.zeros((h, w), np.uint8)
            for p in polys:
                m2 = np.zeros_like(mm)
                cv2.fillPoly(m2, [p], 1)
                mm ^= m2
            owner[mm.astype(bool)] = i
    lut = np.vstack([np.array([l.rgb() for l in plan.layers], np.uint8)
                     .reshape(-1, 3), [[255, 255, 255]]]).astype(np.uint8)
    rend = lut[owner]                       # -1 → 마지막 항(흰 배경)
    err = np.linalg.norm(cv2.cvtColor(rend, cv2.COLOR_RGB2LAB)
                         .astype(np.float32) - flat_lab, axis=-1)
    ink0 = next((i for i, l in enumerate(plan.layers) if l.label == "ink"),
                len(plan.layers))
    lm = (cel.line_mask if cel.line_mask is not None
          else np.zeros((h, w), bool))
    base = insil & (owner >= 0)
    cats = (("fill", base & ~lm & (owner < ink0) & (err > thr)),
            ("over", base & ~lm & (owner >= ink0) & (err > thr)),
            ("line", base & lm & (err > thr_line)))

    # 군집 추출 — 근접 묶음(5×5 팽창)은 fill_holes와 같다
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    groups = []
    for kind, m in cats:
        if not m.any():
            continue
        grp = cv2.dilate(m.astype(np.uint8), k5)
        ncc, cc, cst, _ = cv2.connectedComponentsWithStats(grp, connectivity=8)
        for ci in range(1, ncc):
            x0 = int(cst[ci, cv2.CC_STAT_LEFT]); y0 = int(cst[ci, cv2.CC_STAT_TOP])
            x1 = x0 + int(cst[ci, cv2.CC_STAT_WIDTH])
            y1 = y0 + int(cst[ci, cv2.CC_STAT_HEIGHT])
            sub = (cc[y0:y1, x0:x1] == ci) & m[y0:y1, x0:x1]
            if np.count_nonzero(sub) < min_px:
                continue
            groups.append((float(err[y0:y1, x0:x1][sub].sum()),
                           y0, x0, kind, sub))
    groups.sort(key=lambda g: (-g[0], g[1], g[2]))

    added = {"fill": [], "over": [], "line": []}
    n = 0
    for _score, y0, x0, kind, rem0 in groups:
        if n >= max_layers:
            break
        rem = rem0.copy()
        gx1, gy1 = x0 + rem.shape[1], y0 + rem.shape[0]
        # 그룹당 장수는 면적 비례 — 대형 군집(수천 px 얼룩·긴 획 자국)은 4장
        # 으로 부족하고, 반복마다 잔여를 다시 PCA하므로 굽은 자국도 마디로
        # 쪼개져 잡힌다
        n_patch = min(12, max(4, int(np.count_nonzero(rem)) // 150))
        placed = 0
        while placed < n_patch and n < max_layers and rem.any():
            # 패치는 잔여의 **최심점이 속한 연결 성분** 하나에 맞춘다 — 팽창
            # 묶음 그룹은 색이 다른 여러 자국을 품을 수 있어, 전체 PCA는
            # 헐렁한 타원이 되어 기각만 남는다. 기각해도 그 성분만 접고 다음
            # 성분으로 간다 — 그룹째 포기하면 덧칠 수요의 절반이 안 잡힌다
            dtc = cv2.distanceTransform(np.pad(rem, 1).astype(np.uint8),
                                        cv2.DIST_L2, 3)[1:-1, 1:-1]
            r0 = float(dtc.max())
            spy, spx = np.unravel_index(int(dtc.argmax()), dtc.shape)
            color = tuple(int(v) for v in flat[y0 + spy, x0 + spx])
            _, cc2 = cv2.connectedComponents(rem.astype(np.uint8), connectivity=8)
            comp = cc2 == cc2[spy, spx]
            ys2, xs2 = np.nonzero(comp)
            pw = np.stack([xs2, ys2], axis=1).astype(np.float64)
            ctr = pw.mean(axis=0)
            if len(pw) > 4:
                cov_ = np.cov((pw - ctr).T)
                evals, evecs = np.linalg.eigh(cov_)
                d = evecs[:, int(np.argmax(evals))]
                theta = float(np.arctan2(d[1], d[0]))
                proj = (pw - ctr) @ d
                perp = (pw - ctr) @ np.array([-d[1], d[0]])
                a = float(np.abs(proj).max()) + 1.2
                b = min(float(np.abs(perp).max()) + 1.2, r0 * 1.5 + 1.5)
            else:
                theta, a, b = 0.0, 1.6, 1.6

            def _try(cx, cy, aa, bb, th):
                """패치 후보 채점 — (순이득, 레이어, 갱신 클로저 인자들)."""
                lay = _layer(_FILL_SHAPE, cx, cy, aa, bb, th, 0.0, color,
                             upp, w, h,
                             label="ink" if kind == "line" else "fix").quantized()
                wx0 = max(0, int(cx - aa - 4)); wy0 = max(0, int(cy - aa - 4))
                wx1 = min(w, int(cx + aa + 5)); wy1 = min(h, int(cy + aa + 5))
                mm = _mask_px(cat, lay, upp, w, h, (wx0, wy0, wx1, wy1))
                # 순이득 채점은 **가시 px**만 — 면 수리는 선화 블록 아래
                # 삽입이라 위(선화·메움) 소유 px는 안 바뀐다
                vis = (mm & (owner[wy0:wy1, wx0:wx1] < ink0)
                       if kind == "fill" else mm)
                ins = insil[wy0:wy1, wx0:wx1]
                c_lab = cv2.cvtColor(np.array([[color]], np.uint8),
                                     cv2.COLOR_RGB2LAB
                                     ).reshape(3).astype(np.float32)
                de_new = np.linalg.norm(flat_lab[wy0:wy1, wx0:wx1] - c_lab,
                                        axis=-1)
                vin = vis & ins
                gain = float(err[wy0:wy1, wx0:wx1][vin].sum())
                # 덧칠은 선화 위에 얹히므로 선 지도 자리의 이득을 깎는다
                # (비용은 그대로 — 위 docstring)
                if kind == "over" and _FIX_LINEGAIN < 1.0:
                    vlm = vin & lm[wy0:wy1, wx0:wx1]
                    gain -= (1.0 - _FIX_LINEGAIN) * float(
                        err[wy0:wy1, wx0:wx1][vlm].sum())
                net = (gain - float(de_new[vin].sum())
                       - 80.0 * float(np.count_nonzero(vis & ~ins)))
                # 선화 자리 침범 벌점 (감축 모드) — 면·덧칠 패치가 선화 목표
                # px를 덮으면 그 자리 획이 지금 비어 있어도(err 큼 = 이득처럼
                # 보임) 수정액이 선을 막는다. 순이득으로는 "빗나간 자국 수리
                # 이득 > 선 몇 px 손해"가 성립해 버린다 (턱 선 수십 px가
                # 살색 덧칠에 덮여 "선 끊김"으로 보였다)
                if line_pen and kind != "line":
                    net -= line_pen * float(np.count_nonzero(
                        vin & lm[wy0:wy1, wx0:wx1]))
                return net, lay, (wy0, wy1, wx0, wx1, vin, de_new)

            net, lay, upd = _try(x0 + ctr[0], y0 + ctr[1], a, b, theta)
            if net < min_gain:
                # 내접원 폴백 (fill_holes와 같은 수법) — 색이 다른 갈래를 품은
                # 성분은 전체 PCA 타원이 헐렁해 기각되지만, 최심점의 원은
                # 국소적으로 순이득이 난다
                net, lay, upd = _try(x0 + spx, y0 + spy, r0 + 1.2, r0 + 1.2, 0.0)
            covered = rem & _mask_px(cat, lay, upp, w, h, (x0, y0, gx1, gy1))
            if net < min_gain or not (covered & comp).any():
                rem &= ~comp                # 이 성분은 포기 — 다음 성분 진행
                continue
            added[kind].append(lay)
            n += 1
            placed += 1
            wy0, wy1, wx0, wx1, vin, de_new = upd
            err[wy0:wy1, wx0:wx1][vin] = de_new[vin]   # 이중 수리 방지
            rem &= ~covered
    if n:
        plan.layers[ink0:ink0] = added["fill"]
        plan.layers.extend(added["over"])
        plan.layers.extend(added["line"])
        log(f"  잔차 수리 {n}장 (면 {len(added['fill'])}"
            f"·덧칠 {len(added['over'])}·선 {len(added['line'])})")
    return n
