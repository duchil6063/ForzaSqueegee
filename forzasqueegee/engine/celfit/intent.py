"""획의 **의도** — 논리 획과 후보 생성 사이의 한 겹.

`LogicalStroke`는 "선 지도가 이 자리에 있다"는 관찰이고, `Candidate`는 "그것을
이 도형들로 그렸다"는 답이다. 그 사이의 물음이 **어디서 끊을 것인가**다.
이 층이 없으면 답이 세 자리에 흩어지고 셋 다 같은 것만 본다 —
**현에서의 최대 이탈**:

    stroke._fit_path      곡선 한 장이 안 되면 이탈 최대점에서 쪼갠다
    stroke._fit_segments  RDP 마디 (= 이탈이 허용오차를 넘는 자리)
    candidates._dp_nodes  RDP 세 단계의 합집합

이탈 최대점은 **매끈한 호의 한가운데**다. 거기서 끊으면 사람이 한 번에 긋는
굽음이 두 장으로 갈리고 그 자리에 각이 선다 — 실측(표준 10장) 한 획 안 이웃
도형의 방향 차가 중앙 30°였다. 반대로 눈꼬리·턱선·옷 주름처럼 **사람이 실제로
꺾은 자리**는 현에서 멀지 않을 수 있어 마디가 안 되고, 그러면 그 각이 도형
하나에 뭉개져 둥글어진다.

여기서는 그 답을 한 자리로 모으고 자를 바꾼다: **평활이 지킨 각이 곧 마디다.**
`skeleton.corner_strength`가 이미 "계단이냐 의도된 꺾임이냐"를 가르고 있고
(`smooth_path`가 그 세기만큼 평활을 물린다), 같은 판정을 분절이 그대로 쓴다.
새 문턱을 세우지 않는 것이 요점이다 — 평활이 지킨 각과 분절이 끊는 자리가
어긋나면 "각을 지켰다"가 두 뜻이 된다.

    평활: 각에서 원본을 남긴다      → 각이 경로에 남는다
    분절: 각에서 끊는다             → 각이 도형 **사이**에 선다 (한 장에 안 뭉갠다)
    분절: 각이 아닌 곳은 안 끊는다  → 매끈한 굽음이 한 장으로 간다
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from .skeleton import corner_strength

# 각 판정의 기선 — `smooth_path`가 5탭에서 쓰는 값과 같다 (`_smooth_kernel`의
# 반폭이 2라 `max(3, 2)`). 같은 자를 써야 "평활이 지킨 각"과 "분절이 끊는 각"이
# 같은 각이다.
_BASE = 3
# RDP 마디를 각으로 끌어당기는 창 (px 아닌 **표본 수**). 뼈대 표본은 1px
# 간격이라 사실상 px다. 획 폭에 비례시키되 바닥은 기선과 같다 — 그보다 멀리서
# 끌어오면 각이 아닌 자리를 각이라고 우기는 셈이다.
_SNAP_MUL = float(os.environ.get("FS_INTENT_SNAP", 1.5))
# 각이 아닌 자리에서 끊는 값 (DP 비용, 도형 한 장 `_DP_SHAPE` 대비). 0.33이면
# 각 없는 마디 셋이 도형 한 장 값이다 — 끊을 이유가 분명하면 여전히 끊는다.
_CUT_OFF = float(os.environ.get("FS_INTENT_CUT", 1.0))


@dataclass(frozen=True)
class StrokeIntent:
    """한 획의 **끊을 자리** — 표본마다 "의도된 각"의 세기 0~1.

    경로와 같은 길이라 경로를 자를 때 **함께 잘린다** (`sub`) — 재귀 분할이
    내려가도 각 자리가 어긋나지 않는다. 폭 프로파일은 여기 안 둔다: 그 사실의
    주인은 `LogicalStroke.widths`이고, 여기 사본을 두면 둘이 갈릴 수 있다.
    """

    corner: np.ndarray                # (N,) float 0~1

    def sub(self, lo: int, hi: int | None = None) -> "StrokeIntent":
        return StrokeIntent(self.corner[lo:hi])

    def idx(self, margin: int = 0) -> np.ndarray:
        """각 표본의 인덱스 (양끝 `margin` 안은 뺀다) — 세기 내림차순."""
        n = len(self.corner)
        lo, hi = margin, n - margin
        if hi <= lo:
            return np.zeros(0, np.int64)
        w = np.zeros(n, np.float64)
        w[lo:hi] = self.corner[lo:hi]
        got = np.nonzero(w > 0.0)[0]
        return got[np.argsort(-w[got], kind="stable")]


def build(path: np.ndarray) -> StrokeIntent:
    """경로 하나의 의도 — 평활이 쓰는 그 각 판정을 그대로 부른다."""
    return StrokeIntent(corner_strength(path, base=_BASE)
                        if len(path) >= 2 * _BASE + 1
                        else np.zeros(len(path), np.float64))


def snap_nodes(idx: list[int], it: StrokeIntent | None,
               wmed: float) -> list[int]:
    """마디 인덱스를 가까운 **각**으로 끌어당긴다 (마디 수는 그대로).

    RDP는 현에서 먼 자리를 끊으므로 매끈한 호에서는 호 한가운데가 마디가 되고,
    각은 몇 px 옆에 있어도 마디가 안 된다. 창(`_SNAP_MUL` × 폭) 안에 각이
    있으면 그리로 옮긴다 — **개수가 안 바뀌므로 도형 수도 안 바뀐다.**
    """
    if it is None or len(idx) < 3:
        return idx
    cs = it.corner
    if not cs.any():
        return idx
    win = max(_BASE, int(round(_SNAP_MUL * max(wmed, 1.0))))
    n = len(cs)
    out = [idx[0]]
    for k in idx[1:-1]:
        lo, hi = max(1, k - win), min(n - 1, k + win + 1)
        if hi <= lo:
            out.append(k)
            continue
        seg = cs[lo:hi]
        if seg.max() <= 0.0:
            out.append(k)
            continue
        # 같은 세기면 원래 마디에 가까운 쪽 (결정적)
        cand = np.nonzero(seg >= seg.max() - 1e-9)[0] + lo
        out.append(int(cand[np.argmin(np.abs(cand - k))]))
    out.append(idx[-1])
    return sorted(dict.fromkeys(out))


def split_index(dev: np.ndarray, it: StrokeIntent | None, lo: int,
                hi: int) -> int:
    """재귀 분할의 쪼갤 자리 — **각이 있으면 각에서**, 없으면 이탈 최대점.

    `dev`는 현에서의 이탈(경로와 같은 길이), `[lo, hi]`는 허용 구간이다.
    각을 고를 때도 이탈을 아주 안 보지는 않는다 — 이탈이 최대의 절반도 안 되는
    자리에서 끊으면 두 조각이 여전히 굽어 다음 단계에서 또 쪼개진다.
    """
    fallback = int(np.argmax(dev))
    if it is None:
        return fallback
    cs = it.corner
    if len(cs) != len(dev) or not cs.any():
        return fallback
    w = np.zeros(len(cs), np.float64)
    w[lo:hi + 1] = cs[lo:hi + 1]
    got = np.nonzero((w > 0.0) & (dev >= 0.5 * float(dev.max())))[0]
    if not len(got):
        return fallback
    return int(got[np.argmax(w[got])])


def cut_penalty(it: StrokeIntent | None, k: int) -> float:
    """DP가 표본 `k`에서 끊을 때 무는 여벌 값 (도형 한 장 대비 배수).

    각에서 끊으면 0, 매끈한 자리에서 끊으면 `_CUT_OFF`다. 도형 수 항과 같은
    저울에 서므로 "끊을 값이 있으면 끊는다"는 그대로고, **같은 값이면 각에서**
    끊는다.
    """
    if it is None:
        return 0.0
    cs = it.corner
    if not (0 <= k < len(cs)):
        return 0.0
    return _CUT_OFF * (1.0 - float(cs[k]))


def nodes_of(it: StrokeIntent | None, n: int) -> np.ndarray:
    """각 인덱스 (세기 내림차순, 양끝 제외).

    `_BASE`만큼은 양끝에서 떼 둔다 — 그 안의 "각"은 끝점 접선이 만드는 인공물
    이고, 마디로 세워도 도형이 설 자리가 없다. 계측(`linemetrics`)도 이쪽을
    쓴다.
    """
    if it is None or n <= 2 * _BASE + 1:
        return np.zeros(0, np.int64)
    return it.idx(margin=_BASE)


def corner_nodes(it: StrokeIntent | None, n: int) -> np.ndarray:
    """DP 마디 후보로 얹을 각 인덱스."""
    return nodes_of(it, n)

