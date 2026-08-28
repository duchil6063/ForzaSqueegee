"""§7·§8·§11 면 채움 — **큰 바탕 한 장부터, 그리고 장수를 값으로 센다.**

사람이 색면을 만드는 순서 그대로다:

1. 영역 **대부분을 덮는 큰 바탕 도형** 한 장
2. 남은 큰 잔차 보정
3. 중요한 오목·돌출 보정
4. 작은 잔차는 시각 영향에 따라 생략

종전 채움은 1번이 없었다 — 잔여의 **최대 내접 봉우리**에서 시작해 그 자리를
정확히 맞추고, 남은 것을 또 맞추고를 반복했다. 그러면 한 장이면 될 면이
"봉우리마다 한 장"으로 쪼개진다. 첫 장을 **영역 전체의
2차 모멘트**에서 잡으면 그 한 장이 몸통을 통째로 덮고, 나머지는 진짜 보정만
남는다.

첫 장을 지나치게 정확히 맞추지 않는다. 그 한 장이 이웃 영역으로 삐져나가도
**나중에 그릴 면이나 맨 위의 선이 가리는 자리**라면 값이 0이다 — 채점판이
이미 그렇게 세고 있고(`scoring._PEN_WASTE_FILL`·`ink`), 여기서는 그 자유를
실제로 쓰는 씨앗을 넣어 주는 것이다.

**장수 자체가 비용이다** (§11). 후보끼리 겨룰 때 "새로 맞힌 값"만 보면 늘
잘게 쪼개는 쪽이 이긴다 — 두 장이면 언제나 한 장보다 많이 덮기 때문이다.
그래서 후보의 값을

    J = 새로 맞힌 값 − λ × (1 + 남은 잔차가 더 부를 도형 수)

로 잰다. 뒤 항의 추정은 라스터를 다시 안 본다 — `rag.complexity`의 등주
초과량을 잔차 성분마다 쓴다 (원판이면 1, 너덜하면 그 이상). 같은 식을
분해 쪽 병합이 이미 쓰고 있으므로 두 단계가 같은 자로 도형 수를 센다.

영역의 **첫 장은 λ를 면제**한다 (`plan.fit_plan`의 같은 근거) — 스필이
공짜인 근거가 "나중 면이 덮는다"인데 그 면을 아예 안 사면 근거가 깨진다.

**빔 탐색 대신 한 수 앞 추정을 쓴다.** 후보마다 실제로 다음 장까지 놓아 보면
하강 비용이 후보 수만큼 곱해진다 (배치 시간이 판을 못 굽는 수준으로 뛴다).
`est_shapes`가 그 자리를 닫힌 셈으로 대신하므로 탐색 폭은 후보 수에 머물고,
비교는 여전히 **"이 안으로 갔을 때 총 몇 장이 드나"** 위에서 이뤄진다.
탐색은 영역 단위로 갇혀 있고 결정적이다 (난수 없음, 동점은 어휘 순서).
"""

from __future__ import annotations

import os

import cv2
import numpy as np

from .. import celaxes
from ..catalog import Catalog
from ..celart.rag import complexity
from ..model import Layer, LayerPlan
from .descriptor import fill_rank
from .fill import _grow_step, _place_fat, _seed_moment
from .scoring import _MIN_GAIN, _STROKE_R, _Scorer, _descend
from .vocabulary import _FILL_ALL, _FILL_BASE, _FILL_SHAPES, _FILL_WIN

# 바탕 후보로 세울 어휘 수 — 서술자 순위 상위 몇 종인가. 순위가 있으니 넓은
# 어휘(불투명·뚱뚱한 76종)를 열어도 실제로 채점하는 것은 이 수만큼이다.
_BASE_TOP = int(os.environ.get("FS_CEL_BASE_TOP", 6))
# 후보 집합 덮개를 거는 **처음 몇 장**인가 (§8). 구조가 갈리는 것은 앞 몇
# 장이고, 그 뒤는 잔차 보정이라 탐욕이 곧 최선이다 — 탐색 비용의 뚜껑이다.
_SC_STEPS = int(os.environ.get("FS_CEL_SC_STEPS", 2))
# 도형 수 추정에서 셀 잔차 성분의 최소 크기 (이보다 작으면 어차피 안 산다)
_EST_MIN = 40


def est_shapes(res: np.ndarray) -> float:
    """이 잔차가 앞으로 부를 **도형 수의 닫힌 추정** (`rag.complexity`).

    성분마다 면적과 경계 픽셀 수를 세어 등주 초과량을 더한다 — 원판이면 1,
    너덜한 껍질이면 그 이상이다. 라스터 탐색이 아니라 셈이라 후보마다 불러도
    채점 비용을 안 넘는다.
    """
    u = res.astype(np.uint8)
    if not u.any():
        return 0.0
    n, cc, st, _ = cv2.connectedComponentsWithStats(u, connectivity=8)
    if n <= 1:
        return 0.0
    edge = u & ~cv2.erode(u, np.ones((3, 3), np.uint8))
    peri = np.bincount(cc[edge > 0].ravel(), minlength=n).astype(np.float64)
    area = st[:, cv2.CC_STAT_AREA].astype(np.float64)
    keep = np.arange(n) > 0
    keep &= area >= _EST_MIN
    if not keep.any():
        return 0.0
    return float(complexity(area[keep], np.maximum(peri[keep], 1.0)).sum())


def _base_vocab(cat: Catalog, mask: np.ndarray) -> tuple[str, ...]:
    """이 덩어리를 닮은 순의 채움 어휘 (§6). 축이 꺼지면 손으로 고른 여덟."""
    if not celaxes.on("DESCFIT"):
        return _FILL_SHAPES
    ranked = fill_rank(cat, mask, _FILL_ALL, top=_BASE_TOP)
    # 바탕(타원·사각)은 순위와 무관하게 늘 후보다 — 확장 어휘가 바탕을 이기려면
    # `_FILL_MARGIN`배를 벌어야 한다는 규칙이 그 위에서 선다
    return tuple(dict.fromkeys(_FILL_BASE + ranked))


def _place_whole(sc: _Scorer, cat: Catalog, color, vocab: tuple[str, ...]
                 ) -> tuple[float, Layer] | None:
    """**영역 전체**의 모멘트에서 잡은 바탕 후보 — 몸통을 한 장으로 덮는다."""
    ys, xs = np.nonzero(sc.residual)
    if len(ys) < 16:
        return None
    if len(ys) > 40000:                   # 큰 면은 균일 솎기 (결정적, 모멘트 불변)
        step = len(ys) // 40000 + 1
        ys, xs = ys[::step], xs[::step]
    pw = np.stack([xs, ys], axis=1).astype(np.float64)
    best = None
    for name in vocab:
        for alt in _seed_moment(sc, pw, name, color, cat):
            s = sc.score_val(alt)
            if best is None or s > best[0]:
                best = (s, alt)
    if best is None:
        return None
    return _descend(sc, best[1], color, passes=3)


def _cand_value(sc: _Scorer, q: Layer, price: float, free: bool) -> tuple:
    """후보 하나의 (J, 새로 맞힌 값, 마스크, 창) — §8 회계를 한 수로 접는다.

        J = 새로 맞힌 값 − 잘못 덮은 값 − λ×(1 + 남은 잔차가 부를 도형 수)

    갈래는 `_Scorer.account`가 준다: 먼저 그린 면 침범·배경 침범·실루엣 밖
    물림은 벌점을 물고, **나중에 그릴 면 위 스필과 획이 덮는 자리는 무료**다
    (그리기 순서가 가려 준다 — §7이 쓰는 그 자유이고, 여기서는 그 자유가
    후보 비교에서도 유지된다는 뜻이다).

    **벌점을 빼지 않으면 안 된다**: 후보는 저마다 제자리에서 하강을 마친
    상태라 각자 국소 최적이지만, 서로 다른 안을 견줄 때 값만 보면 "많이 덮되
    배경으로 새는" 안이 이긴다. 실루엣 밖으로 샌 자국은 아무도 안 덮는다.

    단위는 λ가 이미 서 있는 그 가정 위에 있다 — 값 맵의 중앙값이 1이라
    "값 픽셀"과 "픽셀"이 같은 눈금이다 (`importance.place_weight`).
    """
    _, m, box = sc._score_impl(q)
    if m is None:
        return (-1e18, 0.0, None, None, {})
    worth = sc.worth(m, box)
    acc = sc.account(m, box)
    bx0, by0, bx1, by1 = box
    after = sc.residual.copy()
    after[by0:by1, bx0:bx1] &= ~m
    j = (worth - acc["pen"] - (0.0 if free else price)
         - price * est_shapes(after))
    return (j, worth, m, box, acc)


def fill_region(plan: LayerPlan, sc: _Scorer, cat: Catalog, color,
                cap: int, area: float, price: float, cover_stop: float) -> int:
    """영역 하나를 채운다 — 바탕 한 장 → 잔차 보정. 반환 = 쓴 도형 수."""
    n = 0
    blocked = np.zeros_like(sc.residual)   # 포기한 봉우리 (잔여는 남긴다)
    setcover = celaxes.on("SETCOVER") and price > 0.0
    layered = celaxes.on("LAYERED")
    while (n < cap
           and np.count_nonzero(sc.residual) > (1.0 - cover_stop) * area):
        open_res = sc.residual & ~blocked
        if not open_res.any():
            break
        dt = cv2.distanceTransform(open_res.astype(np.uint8), cv2.DIST_L2, 3)
        r0 = float(dt.max())
        if r0 <= _STROKE_R:
            break
        py, px = np.unravel_index(int(dt.argmax()), dt.shape)
        vocab = _base_vocab(cat, sc.residual)
        cands: list[tuple[float, Layer]] = []
        # ① 봉우리 후보 — 종전의 자리 (좁은 목·오목 주머니는 여기서만 잡힌다)
        got = _place_fat(sc, dt, px, py, r0, color, vocab=vocab)
        if got is not None:
            cands.append(got)
        # ② 바탕 후보 — **잔여 전체**의 모멘트 (§7). 첫 장에서 몸통을 통째로
        #    덮는 자리이고, 뒤 장에서도 남은 껍질을 한 장으로 두르는 안이 된다
        if layered and (n < _SC_STEPS or n == 0):
            whole = _place_whole(sc, cat, color, vocab)
            if whole is not None:
                cands.append(whole)
        if not cands:
            break
        free = n == 0
        if setcover and len(cands) > 1 and n < _SC_STEPS:
            # 동점은 **어휘가 아니라 후보 순서**로 갈린다 (봉우리 안이 먼저) —
            # 결정적이고, 종전 동작이 동점에서 그대로 남는다
            scored = [(_cand_value(sc, q, price, free), i, g, q)
                      for i, (g, q) in enumerate(cands)]
            scored.sort(key=lambda t: (-t[0][0], t[1]))
            (j, worth, mfin, fbox, _acc), _i, gain, q = scored[0]
        else:
            gain, q = max(cands, key=lambda t: t[0])
            _, mfin, fbox = sc._score_impl(q)
            worth = sc.worth(mfin, fbox)
        if gain < _MIN_GAIN or mfin is None:
            cv2.circle(blocked.view(np.uint8), (px, py),
                       max(2, int(r0 / 2)), 1, -1)
            continue
        # 가격 — 이 한 장이 새로 맞히는 값이 λ에 못 미치면 안 산다 (첫 장 면제)
        if price and not free and worth < price:
            cv2.circle(blocked.view(np.uint8), (px, py),
                       max(2, int(r0 / 2)), 1, -1)
            continue
        sc.commit_box(mfin, fbox)
        plan.layers.append(q)
        _FILL_WIN[q.shape] = _FILL_WIN.get(q.shape, 0) + 1
        n += 1
    return n


def grow_fill(sc: _Scorer, layers: list, lo: int, passes: int = 8) -> int:
    """**사기 전에 늘린다** — 이 영역이 이미 놓은 도형을 한 스텝 키워 잔여를
    먹는다 (레이어 0장). 반환은 늘린 도형 수.

    영역 채움이 끝나면 남는 잔여의 태반은 **경계 부스러기**다: 게임 격자가
    스케일을 내림해 도형이 제 면보다 조금 작게 서기 때문이다 (`_grow_step`
    문서 — 실측 잔여의 74%가 경계 인접). 종전에는 그 부스러기를 `_fit_bars`가
    막대로, `mop_up`이 타원으로 **사서** 메웠다 — 그것이 "작은 patch를 여러 장
    이어 붙인 느낌"의 자리다 (실측 01: 막대 249 + 마무리 158 = 채움 도형의 40%).

    같은 일을 하는 손이 이미 있다 (`holes.grow_covers`) — 다만 그것은 도안이
    다 선 **뒤**에 돌아, 그때는 이미 산 뒤다. 여기서는 사기 **전에** 같은
    한 스텝을 물어본다. 자는 `_grow_step`과 같다: 점수가 안 떨어지고 잔여를
    실제로 먹을 때만 늘린다 — 벌점이 그대로 걸려 있으므로 배경·먼저 그린 면을
    무는 확장은 스스로 진다.
    """
    if lo >= len(layers):
        return 0
    n = 0
    for _ in range(max(1, passes)):        # 수렴까지 (상한은 폭주 방지)
        moved = False
        for i in range(lo, len(layers)):
            q = layers[i]
            base = sc.score_val(q)
            for dx, dy in ((0.01, 0.01), (0.01, 0.0), (0.0, 0.01)):
                c = Layer(**{**q.__dict__})
                c.sx = round(c.sx + (dx if c.sx >= 0 else -dx), 4)
                c.sy = round(c.sy + (dy if c.sy >= 0 else -dy), 4)
                s, m, box = sc._score_impl(c)
                if m is None or s <= base:
                    continue
                bx0, by0, bx1, by1 = box
                if not (m & sc.residual[by0:by1, bx0:bx1]).any():
                    continue
                sc.commit_box(m, box)
                layers[i] = c
                n += 1
                moved = True
                break
        if not moved:
            break
    return n


def mop_up(plan: LayerPlan, sc: _Scorer, cat: Catalog, color, left: int,
           min_blob: int, price: float, free_first: bool) -> int:
    """남은 덩어리 줍기 — **덩어리마다 그 모양을 닮은 도형 한 장**.

    종전 마무리(`fill._mop_up`)는 덩어리마다 작은 타원을 놓았다. 바탕 한
    장(§7)으로 몸통을 덮고 나면 남는 것은 대개 **굽은 껍질**인데, 타원으로는
    한 껍질에 두세 장이 든다 — 바탕이 아낀 장수를 여기서 도로 쓴다.

    여기서는 덩어리의 2차 모멘트에서 씨앗을 잡고 **서술자 순위**(§6)로 고른
    어휘를 겨루게 한다. 굽은 껍질에는 초승달·부메랑이 한 장으로 선다.
    """
    n = 0
    while n < left:
        res = sc.residual.astype(np.uint8)
        cnt, cc, cstats, _ = cv2.connectedComponentsWithStats(res, connectivity=8)
        if cnt <= 1:
            return n
        ci = int(np.argmax(cstats[1:, cv2.CC_STAT_AREA])) + 1
        if cstats[ci, cv2.CC_STAT_AREA] < min_blob:
            return n
        cm = cc == ci
        free = free_first and n == 0
        # 가격 — 덩어리가 통째로 λ에 못 미치면 어떤 도형으로도 값을 못 한다
        if price and not free and sc.worth_of(cm) < price:
            sc.commit(cm)                  # 포기 — 잔여에서 지워 다음 덩어리로
            continue
        ys, xs = np.nonzero(cm)
        pw = np.stack([xs, ys], axis=1).astype(np.float64)
        vocab = _base_vocab(cat, cm)
        best = None
        for name in vocab:
            for alt in _seed_moment(sc, pw, name, color, cat):
                s = sc.score_val(alt)
                if best is None or s > best[0]:
                    best = (s, alt)
        if best is None:
            sc.commit(cm)
            continue
        gain, q = _descend(sc, best[1], color, passes=3)
        if gain < 3.0:
            sc.commit(cm)                  # 이 덩어리는 포기 (무한루프 방지)
            continue
        gain, q = _grow_step(sc, gain, q)
        _, mfin, fbox = sc._score_impl(q)
        if mfin is None:
            sc.commit(cm)
            continue
        sc.commit_box(mfin, fbox)
        plan.layers.append(q)
        _FILL_WIN[q.shape] = _FILL_WIN.get(q.shape, 0) + 1
        n += 1
    return n
