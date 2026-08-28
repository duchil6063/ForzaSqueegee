"""RAG 이전의 병합 — **ablation 대조군**이다 (`FS_CEL_RAG=0`).

문턱 사다리 하나로 정하던 방식: "접경 충분한 이웃 중 색이 가장 가까운 것"을
목표로 잡고, 면적이 작을수록 큰 색차까지 병합한다. 상한 초과분은 시각
영향(면적 × 색차²) 하위부터 강제로 흡수시킨다.

현행은 `rag`다 — 이 파일은 "그래프 병합이 무엇을 바꿨나"를 재는 자리로만
남는다 (§14의 a0 대조판). 지금 돌아가는 경로가 아니므로 여기 수치는 새로
튜닝하지 않는다.
"""

from __future__ import annotations

import numpy as np

from .marks import _MARK_DE, _MARK_RATIO

_MIN_AREA = 40                     # px² — 무조건 병합되는 조각 크기 (1200 기준)
_MERGE_DE = 2.0                    # 이 미만 색차 이웃은 크기 무관 병합
# 단계적 병합: 면적이 작을수록 큰 색차까지 허용 (AA 혼색·그라데이션 부스러기)
_GRAD_MERGE = ((120, 8.0), (500, 4.0))
_SLIVER_BOOST = 1.6                # 가늘지만 선 아닌 미세 가닥의 병합 문턱 배수


def _region_stats(labels: np.ndarray, lab: np.ndarray, sel: np.ndarray):
    """영역 id → (면적, 평균 Lab) + 인접 쌍 접경 길이."""
    ids, inv, counts = np.unique(labels[sel], return_inverse=True,
                                 return_counts=True)
    sums = np.zeros((len(ids), 3), np.float64)
    np.add.at(sums, inv, lab[sel].astype(np.float64))
    means = sums / counts[:, None]
    idx = {int(i): j for j, i in enumerate(ids)}

    shifts = ((labels[:, :-1], labels[:, 1:]), (labels[:-1, :], labels[1:, :]))
    pairs = []
    for a, b in shifts:
        d = (a != b) & (a >= 0) & (b >= 0)
        pairs.append(np.stack([a[d], b[d]], axis=1))
    p = np.concatenate(pairs, axis=0)
    p = np.concatenate([p, p[:, ::-1]], axis=0)          # 대칭
    key = p[:, 0].astype(np.int64) * (int(labels.max()) + 1) + p[:, 1]
    uk, kc = np.unique(key, return_counts=True)
    return ids, idx, counts, means, uk, kc


def merge_regions(labels: np.ndarray, lab: np.ndarray, sel: np.ndarray,
                  max_regions: int, log,
                  value: np.ndarray | None = None,
                  merge_gain: float = 0.0) -> np.ndarray:
    """단계적 병합 → 상한 맞춤 (대조군)."""
    forced = 0
    priced = 0
    for _pass in range(12):
        ids, idx, counts, means, uk, kc = _region_stats(labels, lab, sel)
        n = len(ids)
        base = int(labels.max()) + 1
        vsum = None
        if value is not None and merge_gain > 0.0:
            vsum = np.bincount(labels[sel].ravel(),
                               weights=value[sel].astype(np.float64),
                               minlength=base)

        nbr: dict[int, list[tuple[int, int]]] = {}      # rid → [(접경, 이웃)]
        border_tot: dict[int, int] = {}
        for k, c in zip(uk.tolist(), kc.tolist()):
            a, b = divmod(k, base)
            nbr.setdefault(a, []).append((int(c), int(b)))
            border_tot[a] = border_tot.get(a, 0) + int(c)

        # 병합 목표: 접경이 최대 접경의 25% 이상인 이웃 중 ΔE 최소
        target: dict[int, tuple[int, float]] = {}
        for rid, cands in nbr.items():
            bmax = max(c for c, _ in cands)
            best, bde = -1, 1e9
            for c, other in cands:
                if c * 4 < bmax:
                    continue
                de = float(np.linalg.norm(means[idx[rid]] - means[idx[other]]))
                if de < bde:
                    best, bde = other, de
            if best >= 0:
                target[rid] = (best, bde)

        def _mark(rid: int) -> bool:
            j = idx[rid]
            return (target[rid][1] >= _MARK_DE
                    and counts[idx[target[rid][0]]] >= _MARK_RATIO * counts[j]
                    and border_tot.get(rid, 0) <= 0.45 * counts[j])

        def _thresh(area: int) -> float:
            if area < _MIN_AREA:
                return 1e9
            for amax, de in _GRAD_MERGE:
                if area < amax:
                    return de
            return _MERGE_DE

        remap: dict[int, int] = {}
        for j, rid in enumerate(ids.tolist()):
            if rid not in target:
                continue
            t = _thresh(int(counts[j]))
            other = target[rid][0]
            thin = border_tot.get(rid, 0) > 0.45 * counts[j]
            darker = means[idx[rid]][0] < means[idx[other]][0] - 6.0
            if t < 1e9 and thin:
                t *= 0.35 if darker else _SLIVER_BOOST
            if target[rid][1] < t:
                remap[rid] = target[rid][0]

        over = n - len(set(remap)) - max_regions

        def _imp(j: int, rid: int) -> float:
            w_ = 3.0 if (border_tot.get(rid, 0) > 0.45 * counts[j]
                         and means[idx[rid]][0]
                         < means[idx[target[rid][0]]][0] - 6.0) else 1.0
            return float(counts[j]) * target[rid][1] ** 2 * w_

        if merge_gain > 0.0 and vsum is not None:
            for j, rid in enumerate(ids.tolist()):
                if rid not in target or rid in remap or _mark(rid):
                    continue
                w_ = 3.0 if (border_tot.get(rid, 0) > 0.45 * counts[j]
                             and means[idx[rid]][0]
                             < means[idx[target[rid][0]]][0] - 6.0) else 1.0
                if float(vsum[rid]) * target[rid][1] * w_ < merge_gain:
                    remap[rid] = target[rid][0]
                    priced += 1
            over = n - len(set(remap)) - max_regions
        if over > 0:
            impact = sorted(
                (_imp(j, rid), rid)
                for j, rid in enumerate(ids.tolist())
                if rid in target and rid not in remap)
            impact = ([c for c in impact if not _mark(c[1])]
                      + [c for c in impact if _mark(c[1])])
            for _, rid in impact[:over]:
                remap[rid] = target[rid][0]
                forced += 1

        if not remap:
            break

        # union-find — a↔b 상호 지목이 사이클로 병합을 지우는 것을 막는다
        parent: dict[int, int] = {}

        def _find(r: int) -> int:
            while parent.get(r, r) != r:
                parent[r] = parent.get(parent[r], parent[r])
                r = parent[r]
            return r

        for a, b in remap.items():
            ra, rb = _find(a), _find(b)
            if ra == rb:
                continue
            if counts[idx[ra]] < counts[idx[rb]]:
                ra, rb = rb, ra
            parent[rb] = ra

        lut = np.arange(base, dtype=np.int32)
        for a in set(remap) | set(remap.values()):
            lut[a] = _find(a)
        pos = labels >= 0
        labels[pos] = lut[labels[pos]]
    log(f"  병합: 상한 강제 {forced}장"
        + (f" · 가격 {priced}장" if priced else ""))
    return labels
