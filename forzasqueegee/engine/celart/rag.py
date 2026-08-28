"""§2·§3 영역 인접 그래프와 **MDL 병합** — 무엇이 한 덩어리인가를 여기서 정한다.

원자(`atoms`)를 노드로, 접경을 간선으로 놓는다. 노드가 들고 있는 것:

    면적 · 평균 Lab · 색 분산 · 중요도 · 콤팩트함 · 둘레 · 실루엣 접경
    · 선이 받치는 경계 비율 · 이웃 목록 · 경계 색 계단 · (선택) 밀집 시각 특징

병합은 ΔE 문턱 하나로 안 정한다. **합치면 총비용이 줄어드는가**를 묻는다:

    merge_cost = 색 재현 손해
               + 중요한 경계를 지우는 값
               + 선이 받치는 경계를 지우는 값
               + 특징 불연속
               − 줄어드는 도형 수 × λ

단위를 "값 픽셀"로 통일한 것이 요점이다 — 마지막 항의 λ가 배치·가격이 쓰는
그 λ이므로, **"도형 한 장을 아끼려고 이만큼의 색 오차를 감수할 것인가"**가
그대로 한 부등식이 된다. 색이 조금 달라도 한 장으로 끝나는 큰 덩어리가 되면
합치고, 면적이 작아도 평평한 면 위에서 또렷하면(중요도·계단이 크다) 안 합친다.

도형 수 추정은 라스터를 안 본다. 등주 초과량

    complexity(A, P) = 1 + κ·max(0, P²/(4πA) − 1)

이면 **면적과 둘레가 더해지고 접경이 두 번 빠지는** 것만으로 병합 후 값이
닫힌 식으로 나온다 (원판 둘이 붙으면 1+κ, 따로면 2 — 뱀처럼 길어지는 병합은
되레 늘어난다). 라스터 추정을 매 간선마다 돌리면 그래프가 못 돈다.

**작은 것을 무조건 큰 것에 흡수시키지 않는다.** 순서는 비용순이고, 상한
(`max_regions`)에 걸려 억지로 합쳐야 할 때만 비용이 양수인 간선을 산다 —
그때도 무늬 보호 조각(`_mark`: 훨씬 큰 면 위에 앉은 콤팩트한 조각)은
일반 후보가 남아 있는 한 뒤로 미룬다.
"""

from __future__ import annotations

import heapq
import os

import numpy as np

from .marks import _MARK_DE, _MARK_RATIO

# ── 비용의 저울 (전 이미지 공통 — 타깃별 손튜닝 금지) ────────────────────
# 색 재현 손해의 기준 색차 — Ward 항 `A1A2/(A1+A2)·(ΔE/DE_REF)²`을 λ와 견줄
# 크기로 맞추는 자다. 6.0은 **종전 문턱 사다리와 눈금이 맞물리게** 고른 값이다:
# 이 항만 놓고 λ(1200×1961에서 565)와 견주면 5,000px짜리 둘은 ΔE 2 근처에서,
# 100px 조각은 ΔE 8 근처에서 부호가 바뀐다 — 각각 종전의 `_MERGE_DE` 2.0과
# `_GRAD_MERGE` 사다리 (120, 8.0)이 손으로 놓아 두던 자리다. 사다리를 면적이
# 대신 푸는 셈이고, 실제 채택은 경계·특징 항까지 더한 총비용이 정한다.
_DE_REF = float(os.environ.get("FS_RAG_DE_REF", 6.0))
# 경계를 지우는 값 — 접경 px당. 색 계단(`gnorm`, 이미지 중앙값 정규화)과
# **선이 받치는 몫**을 따로 문다. 선 밑 경계는 사람이 실제로 그은 자리이므로
# 세 배다. 금지가 아니라 값이다 — 같은 색이 선을 가로질러 이어지는 일은
# 있어야 하고(§10), 그때는 색 재현 손해가 0이라 이 값을 넘긴다.
_BND_G = float(os.environ.get("FS_RAG_BND_G", 1.0))
_BND_LINE = float(os.environ.get("FS_RAG_BND_LINE", 3.0))
# 밀집 시각 특징의 불연속 (모델이 있을 때만) — 접경 px당 코사인 거리 배수
_FEAT_W = float(os.environ.get("FS_RAG_FEAT", 2.0))
# 등주 초과량을 도형 수로 옮기는 배수 (`complexity`)
_KAPPA = float(os.environ.get("FS_RAG_KAPPA", 0.5))
# 중요도를 색 재현 손해에 얹는 지수 — 1.0이면 그대로 곱한다
_IMP_P = float(os.environ.get("FS_RAG_IMP", 1.0))
# 무늬 보호 조각의 색 재현 손해 배수 — 지우면 특징이 통째로 사라지는 조각
_MARK_MUL = float(os.environ.get("FS_RAG_MARK", 4.0))


def complexity(area, peri):
    """이 형상을 덮는 데 드는 도형 수의 닫힌 추정 (`_KAPPA` 문서)."""
    a = np.maximum(area, 1.0)
    return 1.0 + _KAPPA * np.maximum(0.0, peri * peri / (4.0 * np.pi * a) - 1.0)


class RegionGraph:
    """원자 그래프 — 노드 특징과 접경, 그리고 병합."""

    def __init__(self, labels: np.ndarray, lab: np.ndarray, sel: np.ndarray,
                 ink: np.ndarray | None = None,
                 imp: np.ndarray | None = None,
                 feat: np.ndarray | None = None):
        n = int(labels.max()) + 1 if labels.max() >= 0 else 0
        self.n0 = n
        self.labels = labels
        base = max(n, 1)
        flat = labels[sel]
        self.area = np.bincount(flat.ravel(), minlength=n).astype(np.float64)
        col = np.zeros((n, 3), np.float64)
        np.add.at(col, flat.ravel(), lab[sel].astype(np.float64))
        self.col = col / np.maximum(self.area, 1)[:, None]
        sq = np.zeros((n, 3), np.float64)
        np.add.at(sq, flat.ravel(), lab[sel].astype(np.float64) ** 2)
        self.var = np.maximum(sq / np.maximum(self.area, 1)[:, None]
                              - self.col ** 2, 0).sum(1)
        if imp is not None:
            iv = np.zeros(n, np.float64)
            np.add.at(iv, flat.ravel(), imp[sel].astype(np.float64))
            self.imp = iv / np.maximum(self.area, 1)
        else:
            self.imp = np.ones(n, np.float64)
        self.feat = None
        if feat is not None and feat.shape[0] == n:
            self.feat = feat.astype(np.float32)

        # ── 접경 — 4이웃 두 방향 (양방향으로 더한다)
        pa, pb, pde, pln = [], [], [], []
        pairs = [((labels[:, :-1], labels[:, 1:]), (lab[:, :-1], lab[:, 1:]),
                  (ink[:, :-1], ink[:, 1:]) if ink is not None else (None, None)),
                 ((labels[:-1], labels[1:]), (lab[:-1], lab[1:]),
                  (ink[:-1], ink[1:]) if ink is not None else (None, None))]
        for (a, b), (la, lb), (ia, ib) in pairs:
            d = (a >= 0) & (b >= 0) & (a != b)
            if not d.any():
                continue
            pa.append(a[d])
            pb.append(b[d])
            pde.append(np.linalg.norm(la[d].astype(np.float32)
                                      - lb[d].astype(np.float32), axis=1))
            pln.append((ia[d] | ib[d]).astype(np.float64) if ia is not None
                       else np.zeros(int(d.sum()), np.float64))
        # 실루엣 접경 — 이웃이 배경이거나 화면 밖
        sil = np.zeros(n, np.float64)
        for a, b in ((labels[:, :-1], labels[:, 1:]), (labels[:-1], labels[1:])):
            for x, y in ((a, b), (b, a)):
                d = (x >= 0) & (y < 0)
                if d.any():
                    np.add.at(sil, x[d].ravel(), 1.0)
        for edge in (labels[0], labels[-1], labels[:, 0], labels[:, -1]):
            d = edge >= 0
            if d.any():
                np.add.at(sil, edge[d].ravel(), 1.0)
        self.sil = sil

        self.adj: list[dict] = [dict() for _ in range(n)]
        self.peri = sil.copy()
        self.g_ref = 1.0
        if pa:
            aa = np.concatenate(pa)
            bb = np.concatenate(pb)
            de = np.concatenate(pde)
            ln = np.concatenate(pln)
            aa2 = np.concatenate([aa, bb]).astype(np.int64)
            bb2 = np.concatenate([bb, aa]).astype(np.int64)
            de2 = np.concatenate([de, de]).astype(np.float64)
            ln2 = np.concatenate([ln, ln])
            key = aa2 * base + bb2
            uk, inv = np.unique(key, return_inverse=True)
            cnt = np.bincount(inv).astype(np.float64)
            gsum = np.bincount(inv, weights=de2)
            lsum = np.bincount(inv, weights=ln2)
            ua = (uk // base).astype(np.int64)
            ub = (uk % base).astype(np.int64)
            for i in range(len(uk)):
                self.adj[int(ua[i])][int(ub[i])] = [float(cnt[i]), float(gsum[i]),
                                                    float(lsum[i])]
            np.add.at(self.peri, ua, cnt)
            self.g_ref = max(1e-3, float(np.median(gsum / np.maximum(cnt, 1))))
        self.alive = self.area > 0
        self.parent = np.arange(max(n, 1), dtype=np.int64)
        self.version = np.zeros(max(n, 1), np.int64)
        self.merges = 0
        self.stats: dict = {}

    # ── 노드 특징 — 비용식이 직접 안 읽는 칸도 이름을 준다 (§2의 목록.
    #    분석 도구·디버그가 같은 뜻으로 읽어야 하므로 여기 한 자리에 둔다)
    def compact(self, a: int) -> float:
        """콤팩트함 — 둘레 / (2√(π·면적)). 1이면 원판, 클수록 너덜하다."""
        return float(self.peri[a] / max(2.0 * np.sqrt(np.pi * max(self.area[a], 1.0)),
                                        1e-9))

    def line_frac(self, a: int) -> float:
        """이 영역의 둘레 중 **선이 받치는** 몫."""
        tot = lsum = 0.0
        for e in self.adj[a].values():
            tot += e[0]
            lsum += e[2]
        return float(lsum / max(tot, 1e-9))

    def sil_frac(self, a: int) -> float:
        """이 영역의 둘레 중 **실루엣 테**인 몫."""
        return float(self.sil[a] / max(self.peri[a], 1e-9))

    # ── union-find
    def find(self, a: int) -> int:
        p = self.parent
        while p[a] != a:
            p[a] = p[p[a]]
            a = p[a]
        return int(a)

    # ── 판정
    def _mark(self, a: int) -> bool:
        """무늬 보호 조각인가 — 훨씬 큰 면 위에 앉은 콤팩트한 조각."""
        nb = self.adj[a]
        if not nb or self.peri[a] > 0.45 * self.area[a]:
            return False
        other = max(nb.items(), key=lambda kv: (kv[1][0], -kv[0]))[0]
        de = float(np.linalg.norm(self.col[a] - self.col[other]))
        return bool(de >= _MARK_DE and self.area[other] >= _MARK_RATIO * self.area[a])

    def cost(self, a: int, b: int, lam: float) -> float:
        """이 간선을 합치는 총비용 (음수면 합치는 쪽이 싸다)."""
        e = self.adj[a].get(b)
        if e is None:
            return float("inf")
        bnd, gsum, lsum = e
        aa, ab = self.area[a], self.area[b]
        de = float(np.linalg.norm(self.col[a] - self.col[b]))
        imp = (aa * self.imp[a] + ab * self.imp[b]) / max(aa + ab, 1.0)
        recon = (aa * ab / max(aa + ab, 1.0)) * (de / _DE_REF) ** 2 * imp ** _IMP_P
        if self._mark(a) or self._mark(b):
            recon *= _MARK_MUL
        gnorm = min(3.0, (gsum / max(bnd, 1.0)) / self.g_ref)
        # **경계를 지키는 값은 그 경계가 보일 때만 든다.** 선 밑 경계라고 무조건
        # 비싸게 매기면 머리칼처럼 같은 색 면을 선이 여러 조각으로 가른 자리가
        # 영영 안 합쳐진다 — 그런데 선은 어차피 **모든 면 위에** 따로 그어진다
        # (그리기 순서). 두 면의 색이 같으면 합쳐도 선은 그대로 보이고 색이
        # 넘어갈 것도 없다. 지켜야 하는 것은 "다른 색이 선을 넘는 것"이므로
        # 값을 색차로 재어 건다 (§3의 line-supported boundary removal cost가
        # 뜻하는 바 그대로다).
        vis = min(1.0, de / _DE_REF)
        keep = bnd * (_BND_G * gnorm + _BND_LINE * (lsum / max(bnd, 1.0))) * vis
        if self.feat is not None:
            fd = 1.0 - float(np.dot(self.feat[a], self.feat[b]))
            keep += _FEAT_W * bnd * max(0.0, fd)
        saved = (complexity(aa, self.peri[a]) + complexity(ab, self.peri[b])
                 - complexity(aa + ab, self.peri[a] + self.peri[b] - 2.0 * bnd))
        return float(recon + keep - saved * lam)

    # ── 병합
    def _union(self, a: int, b: int) -> int:
        """면적이 큰 쪽을 남긴다 (그리기 순서·id 안정)."""
        if self.area[a] < self.area[b]:
            a, b = b, a
        e = self.adj[a].pop(b)
        self.adj[b].pop(a, None)
        bnd = e[0]
        aa, ab = self.area[a], self.area[b]
        self.col[a] = (self.col[a] * aa + self.col[b] * ab) / max(aa + ab, 1.0)
        self.imp[a] = (self.imp[a] * aa + self.imp[b] * ab) / max(aa + ab, 1.0)
        if self.feat is not None:
            f = self.feat[a] * aa + self.feat[b] * ab
            nrm = float(np.linalg.norm(f))
            if nrm > 1e-9:
                self.feat[a] = (f / nrm).astype(np.float32)
        self.area[a] = aa + ab
        self.peri[a] = self.peri[a] + self.peri[b] - 2.0 * bnd
        self.sil[a] = self.sil[a] + self.sil[b]
        for c, ec in self.adj[b].items():
            self.adj[c].pop(b, None)
            if c == a:
                continue
            cur = self.adj[a].get(c)
            if cur is None:
                self.adj[a][c] = list(ec)
                self.adj[c][a] = list(ec)
            else:
                for i in range(3):
                    cur[i] += ec[i]
                self.adj[c][a] = cur
        self.adj[b] = {}
        self.alive[b] = False
        self.parent[b] = a
        self.version[a] += 1
        self.merges += 1
        return a

    def _run(self, lam: float, live: int, stop: int, protect: bool,
             sign_only: bool) -> tuple[int, int]:
        """비용 낮은 간선부터 합친다 — 반환 (남은 영역 수, 합친 수).

        `sign_only`면 비용이 음수인 간선만 산다 (MDL 단). 아니면 상한
        (`stop`)에 닿을 때까지 양수도 산다 (상한 강제 단). `protect`면 무늬
        보호 조각이 낀 간선은 **일반 후보가 마른 뒤**로 미룬다 — 상한은
        마스킹돼 안 보이는 쪽에서 채운다는 종전 2단계와 같은 규칙이다.
        """
        heap: list = []
        held: list = []

        def push(a: int) -> None:
            for b in self.adj[a]:
                lo, hi = (a, b) if a < b else (b, a)
                item = (self.cost(a, b, lam), lo, hi,
                        int(self.version[lo]), int(self.version[hi]))
                small = lo if self.area[lo] < self.area[hi] else hi
                heapq.heappush(held if (protect and self._mark(small)) else heap,
                               item)

        for a in range(self.n0):
            if self.alive[a]:
                push(a)
        done = 0
        while live > max(stop, 1):
            if not heap:
                if held:
                    heap, held = held, []
                    continue
                break
            c, a, b, va, vb = heapq.heappop(heap)
            if not (self.alive[a] and self.alive[b]):
                continue
            if va != self.version[a] or vb != self.version[b]:
                continue
            if b not in self.adj[a]:
                continue
            if sign_only and c >= 0.0:
                break
            r = self._union(a, b)
            live -= 1
            done += 1
            push(r)
        return live, done

    def merge(self, lam: float, max_regions: int, log=print) -> np.ndarray:
        """MDL 병합 → 상한 맞춤. 반환 = 새 라벨 지도.

        두 단이다. **① MDL** — 총비용이 줄어드는 간선만, 상한과 무관하게
        수렴까지. 여기가 "무엇이 한 덩어리인가"를 답하는 단이고 그림마다
        다른 수의 영역이 남는다. **② 상한 강제** — 그러고도 상한을 넘으면
        비용이 낮은 것부터 산다 (무늬 보호 조각은 뒤로).
        """
        live = int(self.alive.sum())
        live, gained = self._run(lam, live, 1, protect=False, sign_only=True)
        forced = 0
        if live > max_regions:
            live, forced = self._run(lam, live, max_regions, protect=True,
                                     sign_only=False)
        lut = np.arange(max(self.n0, 1), dtype=np.int32)
        for a in range(self.n0):
            lut[a] = self.find(a)
        out = self.labels.copy()
        pos = out >= 0
        out[pos] = lut[out[pos]]
        log(f"  RAG 병합: 원자 {self.n0} → 영역 {live}개 "
            f"(이득 {gained}장 · 상한 강제 {forced}장)")
        self.stats = {"atoms": self.n0, "region_merges": self.merges,
                      "merge_gain": gained, "merge_forced": forced}
        return out

    def mark_ids(self) -> set:
        """살아 있는 노드 중 무늬 보호 조각 — 계측(작은 중요 영역 보존)용."""
        return {a for a in range(self.n0) if self.alive[a] and self._mark(a)}
