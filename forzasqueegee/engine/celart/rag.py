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

from ...i18n import msg
from .marks import _MARK_DE, _MARK_RATIO

# 계보 감사 — `celfit.census`와 **같은 스위치**를 쓴다 (한 번 구우면 분해
# 계보와 배치 계보가 같은 판에서 나온다). 끄면 이 파일의 어느 손도 안 불리고
# 산출물은 바이트 동일이다.
_AUDIT = os.environ.get("FS_UNIT_CENSUS", "").strip() not in ("", "0", "false")

# ── 비용의 저울 (전 이미지 공통 — 타깃별 손튜닝 금지) ────────────────────
# 색 재현 손해의 기준 색차 — Ward 항 `A1A2/(A1+A2)·(ΔE/DE_REF)²`을 λ와 견줄
# 크기로 맞추는 자다. 6.0은 **면적이 문턱을 대신 풀도록** 고른 값이다: 이 항만
# 놓고 λ(1200×1961에서 565)와 견주면 5,000px짜리 둘은 ΔE 2 근처에서, 100px
# 조각은 ΔE 8 근처에서 부호가 바뀐다 — 큰 면일수록 작은 색차에도 안 합쳐지고
# 작은 조각일수록 크게 합쳐진다. 실제 채택은 경계·특징 항까지 더한 총비용이
# 정한다.
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
# §27 위상 손해 — 둘러싸임이 이 몫을 넘으면 그 초과분에 비례해 경계를 지킨다.
# 0.8은 "제 둘레의 다섯 중 넷을 이 상대와 나눈다" = 사실상 그 면 안이다.
# 무게는 경계 또렷함(`_BND_G`)과 같은 급으로 두되 선 밑 경계(`_BND_LINE`)
# 보다는 낮다 — 위상은 지킬 값이지만 선이 이미 그린 경계보다 세지는 않다
_TOPO_MIN = float(os.environ.get("FS_RAG_TOPO_MIN", 0.8))
_TOPO_W = float(os.environ.get("FS_RAG_TOPO", 2.0))
# §19 **비탈 할인** — 그라디언트를 자른 경계의 색 재현 손해를 이만큼 깎는다.
# 0이면 안 깎는다. 근거는 `_ramp` 문서.
_RAMP = float(os.environ.get("FS_RAG_RAMP", 0.75))
# **복잡도 값 배수** (goal §15의 λ_regions) — 병합에서만 λ에 곱한다. 배치·가격이
# 쓰는 λ는 안 건드린다: 이 손잡이가 묻는 것은 "도형 한 장을 아끼려고 색 오차를
# 얼마나 살 것인가"이고, 그 답이 곧 분해 단계의 복잡도-충실도 파레토다.
# 1.0이면 종전 그대로다 (바이트 동일).
_LAM_MUL = float(os.environ.get("FS_RAG_LAM_MUL", 1.0))


def _de_seen(de: float | np.ndarray):
    """§15 — **안 보이는 색차는 색차가 아니다.** ΔE에서 JND 하한을 뺀 나머지.

    비용식의 색 재현 손해는 ΔE²에 비례한다 — 그런데 그 ΔE에는 바닥이 없어서,
    **아무도 못 보는 차이**에도 값을 매긴다. 넓은 두 면이 ΔE 3으로 갈려 있으면
    면적이 커서 손해 항이 λ를 넘고, 그 경계가 그대로 남아 도형을 따로 산다.
    화면은 한 픽셀도 안 달라지는데 장수만 든다 — 요청 §1의 "가짜 경계"가
    분해 단계에 남는 자리가 여기다.

    하한은 새 상수가 아니라 `marks._MARK_DE`다 — 같은 파일이 "진짜 안 보이는
    조각(JND 미만)까지 지키지 않기 위한 하한"으로 이미 쓰고 있고, 무늬 보호
    조각 판정이 그 값으로 "보이나"를 가른다. 같은 물음이므로 같은 자를 쓴다.
    """
    return np.maximum(0.0, np.asarray(de, np.float64) - _MARK_DE)


def _ramp(step: float, de: float) -> float:
    """§19 — 이 경계는 **계단인가 비탈인가**. 0 = 또렷한 계단, 1 = 순수 비탈.

    두 면의 색차 `de` 중 **접경 한 겹에서 실제로 일어나는 몫**(`step`)을 본다.
    사람이 그은 색 경계는 한 픽셀 안에서 다 일어나므로 `step ≈ de`이고,
    부드러운 음영을 팔레트가 자른 자리는 같은 차이가 여러 픽셀에 퍼져 있어
    `step ≪ de`다. 비율이라 그림 밝기·팔레트 수에 안 흔들린다 — 새 지도도
    새 스캔도 없이 이미 있는 두 값(`gsum/bnd`·중심색 거리)만으로 선다.

    왜 깎나: 비용식의 색 재현 손해는 ΔE²에 비례하는데, **비탈을 자른 두 칸을
    합칠 때의 손해는 그 ΔE가 말하는 것보다 작다.** 원화에 애초에 그 자리에
    경계가 없었기 때문이다 — 합치면 사라지는 것은 "경계"가 아니라 팔레트가
    임의로 그은 등고선 하나다. 셀 그림체가 음영을 두세 단으로 추상하는 것도
    같은 이유이고, 사람이 리버리를 만들 때 그라디언트를 여러 장으로 안 쪼개는
    것도 같은 이유다.

    또렷한 경계는 이 할인을 못 받는다 (`step ≈ de` → 비탈 0). 중요도·무늬
    보호는 위 항에 그대로 남아 있어, 평평한 면 위의 작고 또렷한 조각은
    비탈로 오인돼도 그쪽 항이 지킨다.
    """
    if de <= _MARK_DE:                     # JND 아래는 비탈을 물을 것도 없다
        return 0.0
    return float(max(0.0, 1.0 - min(1.0, step / de)))


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
        # 감사 전용 — 상한 강제 단에서 **비용이 양수인데도 산** 간선 (§27).
        # 켜지 않으면 늘 비어 있다
        self.forced_log: list = []

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

    def cost(self, a: int, b: int, lam: float, parts: dict | None = None) -> float:
        """이 간선을 합치는 총비용 (음수면 합치는 쪽이 싸다).

        `parts`를 주면 갈래를 그 dict에 적는다 (감사 전용 — `lineage`). 판정에는
        안 쓰이고 안 주면 아무 일도 안 한다.
        """
        e = self.adj[a].get(b)
        if e is None:
            return float("inf")
        bnd, gsum, lsum = e
        aa, ab = self.area[a], self.area[b]
        de = de_raw = float(np.linalg.norm(self.col[a] - self.col[b]))
        step = gsum / max(bnd, 1.0)        # 접경 한 겹의 평균 색 단차
        mark = self._mark(a) or self._mark(b)
        # §15 — 안 보이는 차이는 값이 없다. **무늬 보호 조각은 뺀다**: 그
        # 판정의 뜻이 "훨씬 큰 평평한 면 위라 마스킹이 없어 작은 색차도 또렷이
        # 보인다"이므로(`marks` 문서), 하필 그 자리에 "안 보인다"의 하한을
        # 걸면 눈 흰자·코 그림자가 통째로 병합된다
        if not mark:
            de = float(_de_seen(de))
        imp = (aa * self.imp[a] + ab * self.imp[b]) / max(aa + ab, 1.0)
        recon = (aa * ab / max(aa + ab, 1.0)) * (de / _DE_REF) ** 2 * imp ** _IMP_P
        if mark:
            recon *= _MARK_MUL
        elif _RAMP:
            # §19 — 비탈을 자른 경계는 색 재현 손해를 깎는다 (`_ramp` 문서).
            # 무늬 보호 조각은 뺀다: 그 판정의 뜻이 "평평한 면 위라 작은 색차도
            # 또렷이 보인다"라, 하필 그 자리에 할인을 걸면 특징이 사라진다
            recon *= 1.0 - _RAMP * _ramp(step, de_raw)
        gnorm = min(3.0, step / self.g_ref)
        # **경계를 지키는 값은 그 경계가 보일 때만 든다.** 선 밑 경계라고 무조건
        # 비싸게 매기면 머리칼처럼 같은 색 면을 선이 여러 조각으로 가른 자리가
        # 영영 안 합쳐진다 — 그런데 선은 어차피 **모든 면 위에** 따로 그어진다
        # (그리기 순서). 두 면의 색이 같으면 합쳐도 선은 그대로 보이고 색이
        # 넘어갈 것도 없다. 지켜야 하는 것은 "다른 색이 선을 넘는 것"이므로
        # 값을 색차로 재어 건다 (§3의 line-supported boundary removal cost가
        # 뜻하는 바 그대로다).
        vis = min(1.0, de / _DE_REF)
        k_bnd = bnd * (_BND_G * gnorm + _BND_LINE * (lsum / max(bnd, 1.0))) * vis
        keep = k_bnd
        k_feat = 0.0
        if self.feat is not None:
            fd = 1.0 - float(np.dot(self.feat[a], self.feat[b]))
            k_feat = _FEAT_W * bnd * max(0.0, fd)
            keep += k_feat
        # §27 **위상 손해 — 둘러싸인 면을 삼키면 구멍 하나가 사라진다.**
        # 지금까지 병합이 지키는 것은 색(재현 손해)과 경계의 또렷함뿐이라,
        # 한 면이 다른 면에 **완전히 둘러싸여 있다**는 사실은 값이 없었다.
        # 그런데 그것은 색 오차가 아니라 **형태**의 문제다: 눈·홍채·단추·
        # 무늬 구멍은 둘러싸여 있고, 삼켜지면 그 자리에 색차가 아니라 형태가
        # 통째로 없어진다 (평균 ΔE로는 거의 안 보이는데 그림은 딴것이 된다).
        #
        # 자는 새 라스터가 아니라 이미 세고 있는 둘레다: 제 둘레의 거의
        # 전부를 상대와 나누고 있으면 그 면은 상대 **안**에 있다. 무늬 보호
        # 조각(`_mark`)이 같은 일을 하지만 그쪽은 "작고 콤팩트한" 조각만
        # 본다 — 고리·리본·큰 구멍은 그 판정에서 빠진다. 여기가 그 짝이다.
        #
        # 색차로 가리는 것(`vis`)은 그대로다: 두 면의 색이 같으면 삼켜도
        # 화면이 안 바뀌므로 지킬 위상이 없다.
        encl = max(bnd / max(self.peri[a], 1.0), bnd / max(self.peri[b], 1.0))
        k_topo = 0.0
        if encl > _TOPO_MIN:
            k_topo = (_TOPO_W * bnd * vis
                      * (encl - _TOPO_MIN) / (1.0 - _TOPO_MIN))
            keep += k_topo
        saved = (complexity(aa, self.peri[a]) + complexity(ab, self.peri[b])
                 - complexity(aa + ab, self.peri[a] + self.peri[b] - 2.0 * bnd))
        if parts is not None:
            parts.update(recon=float(recon), keep_bnd=float(k_bnd),
                         keep_feat=float(k_feat), keep_topo=float(k_topo),
                         saved=float(saved * lam), de=de_raw, de_seen=float(de),
                         step=float(step), bnd=float(bnd),
                         line_frac=float(lsum / max(bnd, 1.0)),
                         encl=float(encl), mark=bool(mark), vis=float(vis),
                         ramp=_ramp(step, de_raw), area_a=float(aa),
                         area_b=float(ab))
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
        마스킹돼 안 보이는 쪽에서 상한을 채운다.
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
            if _AUDIT and c >= 0.0:
                self.forced_log.append((int(a), int(b), float(c),
                                        float(self.area[a]), float(self.area[b])))
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
        lam = float(lam) * _LAM_MUL
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
        log(msg("  RAG 병합: 원자 {atoms} → 영역 {live}개 "
                "(이득 {gained}장 · 상한 강제 {forced}장)",
                atoms=self.n0, live=live, gained=gained, forced=forced))
        self.stats = {"atoms": self.n0, "region_merges": self.merges,
                      "merge_gain": gained, "merge_forced": forced}
        return out

    def lineage(self, lam: float, ws_atoms: int = -1) -> dict:
        """감사 전용 — 살아남은 영역마다 **원자·기하·못 산 최선 이웃**.

        `merge` 뒤에 부른다. 여기서 재는 것은 전부 그래프가 이미 들고 있는
        값이고(새 분류기도 새 문턱도 없다), 최선 이웃 비용은 **지금 이 상태의
        `cost`가 내는 그 값 그대로**다 — "이 영역이 왜 이웃과 안 합쳐졌나"를
        production 식으로 답한다.

        `ws_atoms`를 주면 원자 출신을 가른다 (그 수 미만 = watershed,
        이상 = 그리드 재분할 — `atoms.oversegment`가 뒤에 번호를 잇는다).
        """
        kids: dict[int, list] = {}
        for a in range(self.n0):
            if self.area[a] <= 0:
                continue
            kids.setdefault(self.find(a), []).append(a)
        # 강제 병합에 낀 뿌리 (양수 비용을 산 자리)
        forced_at: dict[int, list] = {}
        for a, b, c, _aa, _ab in self.forced_log:
            forced_at.setdefault(self.find(a), []).append(round(c, 2))
        out: dict = {}
        for r, ks in kids.items():
            best = None
            for b in self.adj[r]:
                pp: dict = {}
                c = self.cost(r, b, lam, parts=pp)
                if best is None or c < best[0]:
                    best = (c, b, pp)
            rec = {
                "atoms": len(ks),
                "atom_area": sorted((int(self.area[k]) for k in ks
                                     if k != r or len(ks) == 1), reverse=True)[:8],
                "grid_atoms": (sum(1 for k in ks if k >= ws_atoms)
                               if ws_atoms >= 0 else -1),
                "area": float(self.area[r]), "peri": float(self.peri[r]),
                "sil": float(self.sil[r]), "imp": round(float(self.imp[r]), 4),
                "var": round(float(self.var[r]), 2),
                "nbr": len(self.adj[r]), "mark": bool(self._mark(r)),
                "forced": forced_at.get(r, []),
            }
            if best is not None:
                c, b, pp = best
                rec["best_cost"] = round(float(c), 3)
                rec["best_lam"] = round(float(c) / max(lam, 1e-9), 4)
                rec["best_nbr"] = int(b)
                rec["best_nbr_area"] = float(self.area[b])
                rec["best"] = {k: (round(v, 4) if isinstance(v, float) else v)
                               for k, v in pp.items()}
            out[str(r)] = rec
        return out

    def mark_ids(self) -> set:
        """살아 있는 노드 중 무늬 보호 조각 — 계측(작은 중요 영역 보존)용."""
        return {a for a in range(self.n0) if self.alive[a] and self._mark(a)}
