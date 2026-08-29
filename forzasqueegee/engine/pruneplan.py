"""가시 기여 0 레이어 제거 — 렌더 동일 보장 레이어 절감 (39차 "품질 유지 최대 절감").

원리: 전 레이어 불투명(alpha 100) + 마스크(먼저 그려진 것 컷) 의미론에서
최종 픽셀 = 그 픽셀을 덮는 최상위 비마스크 레이어 색 (그 위에 마스크가
있으면 배경). 순서대로 덮어쓰며 픽셀별 소유자를 구하면, 어느 픽셀의
소유자도 아닌 비마스크 레이어는 제거해도 렌더가 변하지 않는다. 비소유
레이어 제거는 다른 레이어의 소유권을 바꾸지 않으므로 1패스로 충분.

반투명 레이어가 하나라도 있으면 소유자 모델이 성립하지 않아 중단한다
(불투명 플랜 전용 — `painter`의 알파 플랜에는 못 쓴다). 마스크 레이어는 항상 유지.

같은 소유자 모형 위에 손이 셋이다: `prune_plan`(기여 0 제거) ·
`prune_impact`(예산에 맞춰 영향 하위부터 컷) · `prune_price`(§16 사후 가격 —
기여가 λ 문턱에 못 미치는 장을 되판다). 영향 계산은 `_layer_impact` 한 벌이다.

소유자 래스터는 2배 슈퍼샘플 + 전 레이어 포함 패딩(rect 밖 돌출 유효) —
서브픽셀 슬리버가 0픽셀로 뭉개져 가시 레이어가 제거되는 것을 방지.
"""

from __future__ import annotations

import cv2
import numpy as np

from .catalog import Catalog
from .model import LayerPlan
from .sortplan import _polys, plan_pad_px

_SS = 2  # 슈퍼샘플 배율
# 컷으로 **진짜 배경(아무도 안 그린 캔버스)**이 드러나는 px의 영향 가중 —
# 인게임 배경은 흰 프리뷰가 아니라 차체 도색이라, 미커버는 **색과 무관하게**
# 실핀홀이다. ΔE 곱셈만으로는 밝은 레이어(흰 배경과 ΔE 작음)를 보호하지 못해
# 재컷이 라운드마다 새 핀홀을 열었다 (메움을 천 장 넘게 부어도 수백 군집이
# 남았다). px당 상수 바닥(_BG_PEN)이 색 불문 보호를 보장한다 — Lab 색차의
# 최대 노름이 ~207이므로 400은 어떤 단일 px 색 단차보다 위다
_BG_MULT = 8.0
_BG_PEN = 400.0


def prune_plan(plan: LayerPlan, catalog: Catalog, min_vis: float = 0.0,
               strict_labels: tuple[str, ...] = ("ink",)
               ) -> tuple[LayerPlan, dict]:
    """가시 기여가 min_vis px(이미지 픽셀, 서브픽셀 포함) 이하인 비마스크
    레이어를 제거한 새 플랜과 통계 반환. min_vis 0 = 기여 0만 (렌더 동일).

    `strict_labels` 레이어에는 **ε-프루닝을 적용하지 않는다**(기여 0만 제거).
    선을 최소 도형으로 정확히 품는 플랜에서는 세그먼트 하나하나가 다른 도형이
    못 덮는 자리를 맡으므로, 기여가 몇 px 이하라고 지우면 그 자리가 그대로
    얇아진다."""
    layers = plan.layers
    grad = [catalog[l.shape].gradient is not None for l in layers]
    for i, l in enumerate(layers):
        if not l.mask and not grad[i] and l.alpha < 99.5:
            raise ValueError(f"반투명 레이어(alpha {l.alpha}) 포함 — 프루닝 불가")
    w, h = plan.image_size
    pad = plan_pad_px(plan, catalog)
    ow, oh = (w + 2 * pad) * _SS, (h + 2 * pad) * _SS
    owner = np.full((oh, ow), -1, np.int32)  # -1 = 배경
    poly_buf = np.zeros((oh, ow), np.uint8)
    for i, l in enumerate(layers):
        if grad[i]:
            continue  # 그라데이션(반투명 텍스처)은 아무것도 가리지 않는 것으로 취급
        polys = [np.round((p + pad) * _SS).astype(np.int32)
                 for p in _polys(l, plan, catalog)]
        val = -1 if l.mask else i  # 마스크 = 먼저 그려진 것 컷 → 배경 소유
        if len(polys) == 1:
            cv2.fillPoly(owner, polys, int(val))
        else:  # 짝홀 규칙 (구멍)
            poly_buf[:] = 0
            for p in polys:
                m = np.zeros_like(poly_buf)
                cv2.fillPoly(m, [p], 1)
                poly_buf ^= m
            owner[poly_buf.astype(bool)] = val
    vis = np.bincount(owner[owner >= 0].ravel(), minlength=len(layers))
    vis_px = vis / (_SS * _SS)  # 이미지 픽셀 단위
    keep = [l for i, l in enumerate(layers)
            if l.mask or grad[i]
            or vis_px[i] > (0.0 if l.label in strict_labels else min_vis)]
    stats = {"before": len(layers), "after": len(keep),
             "removed": len(layers) - len(keep)}
    return LayerPlan(source_image=plan.source_image, image_size=plan.image_size,
                     units_per_px=plan.units_per_px, layers=keep), stats


def _layer_impact(plan: LayerPlan, catalog: Catalog, bg: int,
                  weight: np.ndarray | None,
                  sil: np.ndarray | None = None):
    """레이어별 (시각 영향, 소유 px, 그라디언트 여부) — `prune_impact` 문서의 그 자.

    영향 = Σ(소유 px의 ΔE_Lab(내 색, 제거 시 드러나는 색) × 중요도). 배경이
    드러나는 px에는 색 불문 바닥(`_BG_PEN`)을 문다. 예산에 맞추는 컷
    (`prune_impact`)과 사후 가격(`prune_price`)이 같은 자를 쓴다.

    `sil`(실루엣 마스크, 이미지 좌표계)을 주면 **실루엣 밖을 칠한 몫은 기여가
    아니라 해악**으로 센다 (부호를 뒤집는다). 그 자리에는 원화가 없으므로
    덮을수록 나빠지는데, 바닥 벌점(`_BG_PEN`)은 "밑이 안 칠해졌다"만 보고
    그런 스필을 **가장 값진 장**으로 읽는다 — 컷에서는 그 오독이 무해했지만
    (실루엣 밖을 지켜도 장수만 쓴다) 사후 가격에서는 정반대로 작동한다.
    안 주면 종전 그대로다.
    """
    layers = plan.layers
    grad = [catalog[l.shape].gradient is not None for l in layers]
    for i, l in enumerate(layers):
        if not l.mask and not grad[i] and l.alpha < 99.5:
            raise ValueError(f"반투명 레이어(alpha {l.alpha}) 포함 — 프루닝 불가")
    w, h = plan.image_size
    pad = plan_pad_px(plan, catalog)
    ow, oh = (w + 2 * pad) * _SS, (h + 2 * pad) * _SS
    owner = np.full((oh, ow), -1, np.int32)
    poly_buf = np.zeros((oh, ow), np.uint8)
    polys_px: list[list[np.ndarray]] = []
    for i, l in enumerate(layers):
        polys = [np.round((p + pad) * _SS).astype(np.int32)
                 for p in _polys(l, plan, catalog)]
        polys_px.append(polys)
        if grad[i]:
            continue
        val = -1 if l.mask else i
        if len(polys) == 1:
            cv2.fillPoly(owner, polys, int(val))
        else:
            poly_buf[:] = 0
            for p in polys:
                m = np.zeros_like(poly_buf)
                cv2.fillPoly(m, [p], 1)
                poly_buf ^= m
            owner[poly_buf.astype(bool)] = val

    # 증분 렌더 — 레이어 i를 찍기 전에, i가 소유한 px의 현재 색을 읽는다.
    # painted = 아래에 뭐라도 그려진 px — 소유 px가 painted 밖이면 컷 시 진짜
    # 배경(핀홀)이 드러나므로 _BG_MULT 가중을 문다
    canvas = np.full((oh, ow, 3), bg, np.uint8)
    painted = np.zeros((oh, ow), np.uint8)
    impact = np.zeros(len(layers), np.float64)
    # 소유 px를 레이어별로 미리 그룹화 — 레이어마다 `owner == i` 전장 스캔
    # (레이어 수 × 캔버스 px)이 이 함수 비용의 대부분이었다 (호출당 수십 초).
    # 안정 정렬이라 그룹 안 픽셀은 행우선 순서 그대로 —
    # 부동소수 합산 순서까지 기존과 동일하다
    flat_own = owner.ravel()
    lin = np.flatnonzero(flat_own >= 0)
    vals = flat_own[lin]
    vis = np.bincount(vals, minlength=len(layers))
    own_at = lin[np.argsort(vals, kind="stable")]
    starts = np.zeros(len(layers) + 1, np.int64)
    starts[1:] = np.cumsum(vis)
    canvas_f = canvas.reshape(-1, 3)
    painted_f = painted.ravel()
    # 중요도 맵을 소유자 래스터와 같은 격자(패딩 + 슈퍼샘플)로 올린다
    sf = None
    if sil is not None:
        spad = np.zeros((h + 2 * pad, w + 2 * pad), np.uint8)
        spad[pad:pad + h, pad:pad + w] = sil.astype(np.uint8)
        sf = cv2.resize(spad, (ow, oh),
                        interpolation=cv2.INTER_NEAREST).ravel().astype(bool)
    wf = None
    if weight is not None:
        wpad = np.ones((h + 2 * pad, w + 2 * pad), np.float32)
        wpad[pad:pad + h, pad:pad + w] = weight
        wf = cv2.resize(wpad, (ow, oh), interpolation=cv2.INTER_NEAREST).ravel()
    for i, l in enumerate(layers):
        if grad[i]:
            impact[i] = 1e18            # 그라데이션은 소유자 모델 밖 — 지킨다
            continue
        if vis[i]:
            sel_own = own_at[starts[i]:starts[i + 1]]
            below = canvas_f[sel_own]
            mine = np.array([[l.rgb()]], np.uint8)
            b_lab = cv2.cvtColor(below.reshape(-1, 1, 3),
                                 cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
            m_lab = cv2.cvtColor(mine, cv2.COLOR_RGB2LAB
                                 ).reshape(3).astype(np.float32)
            d = np.linalg.norm(b_lab - m_lab[None], axis=1)
            if wf is not None:
                d *= wf[sel_own]
            bgpx = painted_f[sel_own] == 0
            if sf is None:
                d[bgpx] = np.maximum(d[bgpx] * _BG_MULT, _BG_PEN)
            else:
                out = ~sf[sel_own]
                d[bgpx & ~out] = np.maximum(d[bgpx & ~out] * _BG_MULT, _BG_PEN)
                d[out] = -d[out]           # 실루엣 밖은 기여가 아니라 해악
            impact[i] = float(d.sum()) / (_SS * _SS)
        color = (bg, bg, bg) if l.mask else l.rgb()
        pval = 0 if l.mask else 1          # 마스크는 배경을 되살린다
        if len(polys_px[i]) == 1:
            cv2.fillPoly(canvas, polys_px[i], color)
            cv2.fillPoly(painted, polys_px[i], pval)
        else:
            poly_buf[:] = 0
            for p in polys_px[i]:
                mm = np.zeros_like(poly_buf)
                cv2.fillPoly(mm, [p], 1)
                poly_buf ^= mm
            canvas[poly_buf.astype(bool)] = color
            painted[poly_buf.astype(bool)] = pval
    return impact, vis, grad


def prune_price(plan: LayerPlan, catalog: Catalog, min_impact: float,
                protect_labels: tuple[str, ...] = ("ink", "hole"),
                bg: int = 255, weight: np.ndarray | None = None,
                sil: np.ndarray | None = None,
                rounds: int = 3,
                protect_idx: frozenset | set = frozenset()
                ) -> tuple[LayerPlan, dict]:
    """§16 **사후 가격** — 다 그려 놓고 값을 다시 묻는다.

    가격 설계는 살 때 "이 한 장이 λ만큼 버는가"를 묻는다. 그런데 그 답은
    **산 시점의 잔여**에 대한 것이다: 뒤에 그린 면이 그 자리를 덮어 버리면
    실제 기여는 그보다 훨씬 작아진다. 배치가 끝난 판에서 다시 물으면
    "샀는데 안 보이는 장"이 그대로 나온다 — 그것이 곧 부스러기 도형
    (`metrics.tiny_visible_ratio`)이고, 조각붙임의 한 축이다.

    자는 새로 안 만든다. 기여는 `prune_impact`가 예산 컷에 쓰는 그 영향
    (Σ 소유 px의 값×ΔE, 배경 노출은 바닥 벌점)이고, 문턱은 같은 단위의
    수리 문턱 `price.repair_min_gain(λ)`이다 — "지금 이 자리에 이 장을
    **새로 사겠는가**"를 묻는 것과 같은 부등식이다. 못 미치면 되판다.

    `protect_labels`는 손대지 않는다 — 획(`ink`)은 선 노선과 공유하는
    산출물이고, 구멍 메움(`hole`)은 게이트가 지키는 자리다. 배경이 드러나는
    장은 영향의 바닥 벌점 때문에 문턱을 한참 넘어 저절로 남는다.

    `sil`(실루엣)을 주면 **실루엣 밖으로 샌 몫은 기여에서 뺀다** (`_layer_impact`
    문서) — 큰 바탕 도형이 실루엣을 넘어가면 그 자리는 인게임에서 차 도색 위에
    남는 자국이다. 안쪽에서 버는 것보다 밖에서 흘리는 것이 크면 되판다.

    `protect_idx`는 §18 커버리지 불변이 주는 **자격 보호**다 (첫 바퀴 기준
    index) — 혼자 덮고 있는 장은 값과 무관하게 못 판다. 되팔기는 커버리지를
    줄이는 손이므로 라벨 보호만으로는 부족하다 (`celfit.coverage` 문서).

    한 장을 되팔면 그 밑이 드러나 이웃 장의 기여가 바뀌므로 **수렴까지**
    돈다 (`rounds`는 폭주 방지 뚜껑).
    """
    guard = {id(plan.layers[i]) for i in protect_idx
             if 0 <= i < len(plan.layers)}
    stats = {"before": len(plan.layers), "removed": 0, "rounds": 0}
    for _ in range(max(1, rounds)):
        layers = plan.layers
        impact, _vis, grad = _layer_impact(plan, catalog, bg, weight, sil)
        groups: dict[int, list[int]] = {}
        for i, l in enumerate(layers):
            if l.mask or grad[i] or l.label in protect_labels or id(l) in guard:
                continue
            groups.setdefault(l.stroke if l.stroke >= 0 else ~i, []).append(i)
        cut: set[int] = set()
        for _k, g in sorted(groups.items()):        # 결정적 순회
            if float(impact[g].mean()) < min_impact:
                cut.update(g)
        if not cut:
            break
        plan = LayerPlan(source_image=plan.source_image,
                         image_size=plan.image_size,
                         units_per_px=plan.units_per_px,
                         layers=[l for i, l in enumerate(layers) if i not in cut])
        stats["removed"] += len(cut)
        stats["rounds"] += 1
    stats["after"] = len(plan.layers)
    return plan, stats


def prune_impact(plan: LayerPlan, catalog: Catalog, budget: int,
                 protect_labels: tuple[str, ...] = ("ink",),
                 bg: int = 255,
                 weight: np.ndarray | None = None,
                 protect_idx: frozenset | set = frozenset()
                 ) -> tuple[LayerPlan, dict]:
    """시각 영향이 작은 레이어부터 잘라 예산에 맞춘 새 플랜 반환 (cel 노선용).

    영향 = Σ(소유 px의 ΔE_Lab(내 색, 제거 시 드러나는 색)). 소유 px는 위에서
    아무도 안 덮으므로 "드러나는 색" = 그 레이어를 찍기 직전의 캔버스 색이다
    — 소유자 1패스 + 증분 렌더 1패스로 정확히 구한다 (단일 제거 기준; 동시
    다중 제거의 연쇄 노출은 근사). 같은 색 겹침은 영향 0이라 먼저 잘리고,
    흰 배경이 뚫리는 자리는 영향이 커서 남는다 — vis px 기준(min_vis)이
    만들던 반점이 없다. `protect_labels`(선화 획)와 마스크·그라데이션은 지키고,
    소유 px 0인 레이어는 무조건 정리한다.

    `weight`는 이미지 좌표계(h, w) 곱 배수 맵(`importance.masking_weight`) —
    같은 색차라도 평평한 면 위가 더 눈에 띈다는 것을 반영한다. 배경 노출의
    상수 바닥(_BG_PEN)에는 안 곱한다: 그건 "색 불문 핀홀 보호"라 중요도와
    무관하다.

    컷의 단위는 레이어가 아니라 **획 그룹**(`Layer.stroke`)이다 — 한 획에서
    나온 마디는 전부 살리거나 전부 버린다. 아래 원자화 주석 참조.

    `protect_idx`(§18 커버리지 불변)는 **라벨 보호와 급이 다르다** — 라벨
    보호는 예산이 모자라면 양보하지만(아래 "보호 양보"), 이쪽은 자격이라
    양보가 없다. 혼자 덮고 있는 장을 자르면 그 자리가 그대로 구멍이다.
    그룹에 한 장이라도 걸리면 그룹째 지킨다 (컷의 원자 단위가 그룹이다).
    """
    layers = plan.layers
    if len(layers) <= budget:
        return plan, {"before": len(layers), "after": len(layers), "removed": 0}
    impact, vis, grad = _layer_impact(plan, catalog, bg, weight)
    # 획 그룹 원자화 — 한 획에서 나온 마디(`stroke` 같은 값)는 전부 살리거나
    # 전부 버린다. 중간 마디만 잘리면 그 자리가 빈틈이 되어 획이 점선으로
    # 읽힌다. 그룹의 순위는 **마디당 평균 영향** — 합으로 매기면 긴 획은 마디
    # 수만으로 상위에 붙어 컷이 짧은 획에만 떨어지고, 단일 레이어(그룹 크기 1)
    # 와 잣대가 달라진다.
    # 상한 3,000에서는 사실상 무동작이다 (`ink`가 이미 보호라 그룹이 되는 것은
    # 영역 경계 막대뿐 — 실측 4장 전부 도안↔셀 지각차 ±0.0001). 값은 **감축
    # 모드**에서 나온다: 보호 양보가 걸려 ink까지 잘릴 때 획이 중간에서
    # 끊기는 대신 통째로 빠진다
    groups: dict[int, list[int]] = {}
    for i, l in enumerate(layers):
        if l.mask or grad[i]:
            continue
        if l.label in protect_labels and vis[i] > 0:
            continue
        groups.setdefault(l.stroke if l.stroke >= 0 else ~i, []).append(i)
    if protect_idx:                       # 자격 보호 — 그룹째 뺀다 (양보 없음)
        groups = {k: g for k, g in groups.items()
                  if not any(i in protect_idx for i in g)}
    gorder = sorted(groups.values(),
                    key=lambda g: (float(impact[g].mean()), g[0]))
    n_cut = len(layers) - budget
    cut: set[int] = set()
    # 남은 필요분보다 큰 그룹은 건너뛴다 — 더 작은 그룹으로 예산에 딱 맞춘다.
    # 딱 맞는 조합이 없으면 마지막에 영향 최소 그룹을 통째로 넘긴다(예산 미만)
    for g in gorder:
        if len(cut) >= n_cut:
            break
        if len(g) > n_cut - len(cut):
            continue
        cut.update(g)
    if len(cut) < n_cut:
        for g in gorder:
            if g[0] in cut:
                continue
            cut.update(g)
            if len(cut) >= n_cut:
                break
    if len(cut) < n_cut:
        # 보호 양보 — 보호 레이어 합이 예산을 넘으면 비보호만 자르는 것이
        # 오히려 학살이다 (실측 상한 2000: ink+hole 1,959장 보호 → 채움(cel)
        # 1,089→41장 전멸, RMSE 37). 모자란 만큼은 보호도 영향 하위부터 잘라
        # 라벨 불문 "덜 보이는 것부터"로 되돌린다. 보호가 예산 안에 들면
        # (기존 전 케이스) 이 분기는 안 탄다
        order = np.argsort(impact, kind="stable")
        extra = [int(i) for i in order
                 if int(i) not in cut and int(i) not in protect_idx
                 and not layers[i].mask and not grad[i]]
        cut |= set(extra[:n_cut - len(cut)])
    keep = [l for i, l in enumerate(layers) if i not in cut]
    return (LayerPlan(source_image=plan.source_image, image_size=plan.image_size,
                      units_per_px=plan.units_per_px, layers=keep),
            {"before": len(layers), "after": len(keep), "removed": len(cut)})
