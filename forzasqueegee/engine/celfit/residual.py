"""§12 전역 잔차 — **무엇이 남았는지 이름을 붙이고, 새 도형은 마지막에 산다.**

배치가 끝난 뒤의 잔차를 한 덩어리로 보지 않는다. 종류마다 고칠 방법이 다르고,
그중 대부분은 **도형을 더 사지 않고** 고쳐진다:

    hole        실루엣 안인데 아무도 안 덮었다 (흰 배경이 드러난다)
    gap         그 구멍이 하필 **선 옆**이다 — 선과 색면 사이의 빈 띠
    boundary    덮였지만 색이 틀리고, 자리가 **영역 경계**다 (한쪽이 못 미쳤다)
    wrong       덮였지만 색이 틀리고, 경계도 아니다 (얼룩·빗나간 획)
    leak        실루엣 **밖**을 덮었다 (배경 위 스필)
    tiny        위 어느 것이든 **값이 λ에 못 미친다** — 안 고치는 게 맞다

고치는 순서는 요청의 사다리 그대로다:

    ① 기존 도형 이동 → ② 스케일 → ③ 회전 → ④ 이웃 도형 조정
    → ⑤ 영역 쌍 조정 → ⑥ 그래도 값이 크면 **그때** 보정 도형 추가

①~⑤는 이미 우리 손에 있는 기계가 한다 — `finetune.refine_plan`이 레이어
기하를 게임 양자화 스텝 이웃으로 좌표하강시키고, 실루엣 신규 노출은
무조건 기각한다. 그 패스가 맨 마지막에 돌면 구멍 메움과 잔차 수리가 먼저
도형을 사 버린 뒤다. 여기서는 같은 기계를 **잔차 자리에 초점을 맞춰 먼저**
돌린다 — 그 순서가 ⑥을 최후 수단으로 만든다.

이 모듈이 직접 도형을 사지는 않는다 — 진단(`analyze`)과 초점(`focus_layers`)만
낸다. 사는 일은 `holes.fill_holes`·`repair.repair_mismatch`가 하고, 이 단을
지난 뒤라 살 것이 줄어 있다.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..catalog import Catalog
from ..celart import CelArt
from ..model import LayerPlan
from .geometry import _poly_px

# "색이 틀렸다"의 문턱 ΔE — 잔차 수리(`repair_mismatch`)와 같은 자를 쓴다.
_THR = 12.0
# 경계·선 옆으로 치는 거리 (px) — 최소 도형 폭 근방. 이 안이면 "한쪽이 못
# 미친 자리"이지 얼룩이 아니다
_NEAR = 3


def owner_map(plan: LayerPlan, cel: CelArt, cat: Catalog,
              with_regions: bool = False):
    """픽셀 → 최상위 레이어 index (-1 = 미커버). 렌더와 같은 폴리곤 식.

    `with_regions`면 (소유자 지도, 레이어별 소속 영역 id) 짝을 낸다 — 도형이
    **어느 영역을 그리려고** 놓였나다 (나중 레이어에 덮여 안 보이더라도 그
    영역이 쓴 장수로 센다. §13 "영역당 도형 수"의 뜻).
    """
    w, h = cel.size
    upp = plan.units_per_px
    owner = np.full((h, w), -1, np.int32)
    reg_of: list[int] = []
    for i, lay in enumerate(plan.layers):
        if with_regions:
            reg_of.append(-1)              # 자리를 먼저 잡는다 (index 정렬 보장)
        polys = [np.round(p).astype(np.int32)
                 for p in _poly_px(cat, lay, upp, w, h)]
        if not polys:
            continue
        xs = np.concatenate([p[:, 0] for p in polys])
        ys = np.concatenate([p[:, 1] for p in polys])
        x0 = max(0, int(xs.min()) - 1)
        y0 = max(0, int(ys.min()) - 1)
        x1 = min(w, int(xs.max()) + 2)
        y1 = min(h, int(ys.max()) + 2)
        if x0 >= x1 or y0 >= y1:
            continue
        off = np.array([x0, y0], np.int32)
        m = np.zeros((y1 - y0, x1 - x0), np.uint8)
        if len(polys) == 1:
            cv2.fillPoly(m, [polys[0] - off], 1)
        else:
            for p in polys:
                one = np.zeros_like(m)
                cv2.fillPoly(one, [p - off], 1)
                m ^= one
        owner[y0:y1, x0:x1][m > 0] = i
        if with_regions:
            sub = cel.labels[y0:y1, x0:x1][m > 0]
            sub = sub[sub >= 0]
            reg_of[i] = int(np.bincount(sub).argmax()) if sub.size else -1
    return (owner, reg_of) if with_regions else owner


def analyze(plan: LayerPlan, cel: CelArt, cat: Catalog, *,
            value: np.ndarray | None = None, price: float = 0.0,
            min_px: int = 4, owner: np.ndarray | None = None) -> dict:
    """잔차를 종류별로 세고, **고칠 값이 있는 자리**의 마스크를 낸다.

    반환에는 클래스별 px 수·군집 수와 `actionable`(값이 λ를 넘는 군집의 합집합)
    이 들어간다. `actionable`이 §12의 초점이다.
    """
    w, h = cel.size
    if owner is None:
        owner = owner_map(plan, cel, cat)
    flat = cel.flat_render()
    lut = np.zeros((len(plan.layers) + 1, 3), np.uint8)
    lut[0] = 255                              # [0] = 배경 흰색
    for i, l in enumerate(plan.layers):
        lut[i + 1] = l.rgb()
    render = lut[owner + 1]
    a = cv2.cvtColor(render, cv2.COLOR_RGB2LAB).astype(np.float32)
    b = cv2.cvtColor(flat, cv2.COLOR_RGB2LAB).astype(np.float32)
    de = np.linalg.norm(a - b, axis=2)

    insil = cel.labels >= 0
    # 구멍은 **실루엣 내부**에서만 센다 — 최외곽 테는 게임 이동 스텝 격자상
    # 어떤 도형도 못 맞추는 폭이라 구멍 자(`holes._sil_rim`)·봉인
    # 자(`coverage.sil_core`)가 이미 빼는 자리다. 여기만 세면 `gap`의 99.8%가
    # 실루엣 선 밑의 그 테였다 (실측 G1-01 629/630 · 07 575/577, 안쪽 4px↑
    # 군집 0) — 자가 거짓말을 하고, 초점 미세 조정이 못 고치는 군집을 쫓는다
    from .holes import _sil_rim
    core = insil & ~_sil_rim(cel.labels, plan.units_per_px)
    covered = owner >= 0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * _NEAR + 1,) * 2)
    # 영역 경계 띠 — 라벨이 바뀌는 자리
    bnd = np.zeros((h, w), bool)
    lb = cel.labels
    bnd[:, :-1] |= (lb[:, :-1] != lb[:, 1:])
    bnd[:, 1:] |= (lb[:, :-1] != lb[:, 1:])
    bnd[:-1] |= (lb[:-1] != lb[1:])
    bnd[1:] |= (lb[:-1] != lb[1:])
    bnd = cv2.dilate(bnd.astype(np.uint8), k).astype(bool) & insil
    near_ink = (cv2.dilate(cel.line_mask.astype(np.uint8), k).astype(bool)
                if cel.line_mask is not None else np.zeros((h, w), bool))

    hole = core & ~covered
    leak = ~insil & covered
    wrong = covered & insil & (de > _THR)
    cls = np.zeros((h, w), np.uint8)          # 0 = 정상
    cls[wrong & ~bnd] = 4                     # wrong
    cls[wrong & bnd] = 3                      # boundary
    cls[hole] = 1                             # hole
    cls[hole & near_ink] = 2                  # gap
    cls[leak] = 5                             # leak

    names = {1: "hole", 2: "gap", 3: "boundary", 4: "wrong", 5: "leak"}
    out: dict = {f"res_{n}_px": int((cls == i).sum()) for i, n in names.items()}
    bad = cls > 0
    act = np.zeros((h, w), bool)
    tiny = 0
    clusters = 0
    if bad.any():
        n, cc, st, _ = cv2.connectedComponentsWithStats(
            bad.astype(np.uint8), connectivity=8)
        keep = np.zeros(n, bool)
        vmap = value if value is not None else np.ones((h, w), np.float32)
        # 군집의 값 = Σ(값 × ΔE) — 수리 문턱(`price.repair_min_gain`)과 같은 단위
        wsum = np.bincount(cc.ravel(),
                           weights=(vmap * np.maximum(de, 1.0)).ravel(),
                           minlength=n)
        for i in range(1, n):
            if st[i, cv2.CC_STAT_AREA] < min_px:
                continue
            clusters += 1
            if price > 0.0 and wsum[i] < price:
                tiny += 1
                continue
            keep[i] = True
        act = keep[cc]
    out.update({"res_clusters": clusters, "res_tiny": tiny,
                "res_actionable_px": int(act.sum())})
    out["actionable"] = act
    out["classes"] = cls
    out["de"] = de
    out["owner"] = owner
    return out


def focus_layers(plan: LayerPlan, cat: Catalog, cel: CelArt,
                 act: np.ndarray, owner: np.ndarray) -> list[int]:
    """초점 자리를 **덮고 있거나 맞닿은** 레이어 index — ①~⑤가 만질 후보.

    잔차 자리의 소유자(위에서 덮은 레이어)와, 그 자리를 놓친 이웃(초점을
    넓혀 잡은 소유자)이 함께 들어간다 — 구멍은 정의상 소유자가 없으므로
    **넓힌 자리**에서만 후보가 나온다 (§12 ④ "adjacent shape 조정").
    """
    if not act.any():
        return []
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    grown = cv2.dilate(act.astype(np.uint8), k).astype(bool)
    ids = np.unique(owner[grown])
    return sorted(int(i) for i in ids.tolist() if i >= 0)
