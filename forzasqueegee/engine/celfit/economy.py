"""레이어 경제 — **한 장 한 장이 무엇 때문에 팔렸나**를 갈래별로 센다.

`metrics`가 "몇 장을 어디에 썼나"를 재는 자리라면 여기는 그 **까닭**을
묻는다. 물음은 하나다: 사람이 3,000장 안에서 그리는 그림과 우리 판이
갈리는 자리가 색 채움인가 선인가.

    ramp_*        **색 변화 재현에 팔린 장수** — 원화에 경계가 없는데
                  팔레트가 그은 등고선(비탈, `celart.rag._ramp`)으로 갈린
                  이웃 영역들을 한 무리로 묶고, 그 무리가 쓴 채움 장수를 센다.
                  무리 하나를 한 덩어리로 그릴 수 있다면 `ramp_excess`가
                  그때 도로 받는 장수의 상한이다
    tiny_fill_layers   화면에서 40px도 안 보이는 채움 장 (조각붙임의 자)
    layers_per_kpx_*   중요도 상위 1/4과 나머지의 **넓이당 장수** — 중요하지
                  않은 자리에 장수가 몰려 있나
    layers_per_region_p90  한 영역이 최악에 몇 장을 먹나

전부 계측이다 — 배치를 바꾸지 않는다.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..celart import CelArt
from ..celart.rag import _ramp


def _region_lab(cel: CelArt) -> np.ndarray:
    """영역 id → 대표색 Lab (없는 id는 0)."""
    n = max((r.rid for r in cel.regions), default=0) + 1
    lut = np.zeros((n, 3), np.uint8)
    for r in cel.regions:
        lut[r.rid] = r.color
    return cv2.cvtColor(lut.reshape(-1, 1, 3),
                        cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float64)


def ramp_groups(cel: CelArt) -> tuple[dict[int, int], int]:
    """**비탈로 갈린 영역 무리** — (영역 id → 무리 id, 무리 수).

    이웃 두 영역의 접경에서 실제로 일어나는 색 단차(`step`)가 두 대표색의
    거리(`de`)보다 훨씬 작으면 그 경계는 계단이 아니라 **비탈을 자른 등고선**
    이다 (`celart.rag._ramp` — 분해 쪽 병합이 이미 쓰는 그 자). 그런 간선으로
    이어진 영역들을 union-find로 묶는다. 무리 크기가 2 이상이면 원화에서는
    한 덩어리로 부드럽게 변하는 자리를 우리가 여러 평면으로 쪼갠 것이다.
    """
    lb = cel.labels
    if lb.max() < 0 or cel.src_rgb is None:
        return {}, 0
    lab = cv2.cvtColor(cel.src_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    n = int(lb.max()) + 1
    cnt: dict[tuple[int, int], float] = {}
    ssum: dict[tuple[int, int], float] = {}
    for (a, b), (la, lbb) in (((lb[:, :-1], lb[:, 1:]), (lab[:, :-1], lab[:, 1:])),
                              ((lb[:-1], lb[1:]), (lab[:-1], lab[1:]))):
        d = (a >= 0) & (b >= 0) & (a != b)
        if not d.any():
            continue
        aa = np.minimum(a[d], b[d]).astype(np.int64)
        bb = np.maximum(a[d], b[d]).astype(np.int64)
        st = np.linalg.norm(la[d].astype(np.float32)
                            - lbb[d].astype(np.float32), axis=1)
        key = aa * n + bb
        uk, inv = np.unique(key, return_inverse=True)
        c = np.bincount(inv).astype(np.float64)
        s = np.bincount(inv, weights=st.astype(np.float64))
        for i, k in enumerate(uk.tolist()):
            cnt[k] = cnt.get(k, 0.0) + c[i]
            ssum[k] = ssum.get(k, 0.0) + s[i]
    if not cnt:
        return {}, 0
    rlab = _region_lab(cel)
    parent = {r.rid: r.rid for r in cel.regions}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for k, c in cnt.items():
        a, b = k // n, k % n
        if a not in parent or b not in parent:
            continue
        de = float(np.linalg.norm(rlab[a] - rlab[b]))
        if _ramp(ssum[k] / max(c, 1.0), de) < 0.5:
            continue                       # 또렷한 계단 — 무리로 안 묶는다
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[ra] = rb
    grp = {rid: find(rid) for rid in parent}
    return grp, len({v for v in grp.values()})


def layer_economy(cel: CelArt, cat, plan_layers: list, labels: list[str],
                  reg_of: list[int], vis: np.ndarray,
                  value: np.ndarray | None,
                  owner: np.ndarray | None = None) -> dict:
    """§10 레이어 경제 진단 한 벌 (report의 `structure`에 실린다)."""
    n_grad = sum(1 for l in plan_layers
                 if l.shape in cat.shapes and cat[l.shape].gradient is not None)
    fill_i = [i for i, x in enumerate(labels) if x != "ink"]
    out: dict = {
        "layers_fill": len(fill_i),
        "layers_ink": len(plan_layers) - len(fill_i),
        "layers_gradient": n_grad,
        "tiny_fill_layers": int((vis[fill_i] < 40).sum()) if fill_i else 0,
    }
    # 영역당 채움 장수 — 최악(상위 10%)이 몇 장인가
    per: dict[int, int] = {}
    for i, rid in enumerate(reg_of):
        if labels[i] == "ink" or rid < 0:
            continue
        per[rid] = per.get(rid, 0) + 1
    if per:
        out["layers_per_region_p90"] = round(
            float(np.percentile(np.asarray(list(per.values()), np.float64), 90)), 2)

    # **중요도별 넓이당 장수** — 값 맵 상위 1/4과 나머지에서 각각 10k px당
    # 몇 장이 보이나. 사람은 얼굴에 장수를 몰고 배경 옷자락은 한 장으로
    # 끝낸다 — 두 수가 벌어지지 않으면 장수가 값이 없는 자리에 깔린 것이다
    if value is not None and owner is not None and fill_i:
        sel = cel.labels >= 0
        if sel.any():
            thr = float(np.percentile(value[sel], 75))
            hi = sel & (value >= thr)
            n_hi = int(hi.sum())
            n_lo = int(sel.sum()) - n_hi
            own = np.where(hi, owner, -1)
            hit = np.bincount(own[own >= 0].ravel(),
                              minlength=len(plan_layers)).astype(np.int64)
            fi = np.asarray(fill_i, np.int64)
            in_hi = hit[fi] > 0.5 * np.maximum(vis[fi], 1)
            if n_hi:
                out["layers_per_kpx_hi"] = round(
                    1e4 * float(in_hi.sum()) / n_hi, 2)
            if n_lo > 0:
                out["layers_per_kpx_lo"] = round(
                    1e4 * float((~in_hi).sum()) / n_lo, 2)

    # **비탈 무리** — 색 변화 재현에 팔린 장수
    grp, ngrp = ramp_groups(cel)
    if grp:
        size: dict[int, int] = {}
        for rid, g in grp.items():
            size[g] = size.get(g, 0) + 1
        multi = {g for g, k in size.items() if k >= 2}
        regs = [rid for rid, g in grp.items() if g in multi]
        lay = sum(per.get(rid, 0) for rid in regs)
        out.update({
            "ramp_groups": len(multi),
            "ramp_regions": len(regs),
            "ramp_fill_layers": lay,
            # 무리마다 한 덩어리로 그릴 수 있다면 그때 도로 받는 장수의 상한
            "ramp_excess_layers": max(0, lay - len(multi)),
        })
    return out
