"""획 그래프 — 뼈대 경로를 **의미를 가진 획**으로 승격한다.

뼈대는 픽셀 위상이지 사람의 획이 아니다. 접합점을 노드, 가닥을 간선으로 본
그래프 위에서 ① 한 획으로 이어지는 간선 사슬을 찾고(`continue_strokes`)
② 간선마다 역할을 판정한다(`classify`). 둘 다 **의미 라벨을 안 쓴다** —
위상·색·선 신뢰도·중요도만으로 판단하므로 인물이든 메카든 같은 규칙이다.

역할은 그리기 정책이 아니라 **관찰**이다: 무엇을 그릴지는 노선 정책이
정하고(`policy`), 여기서는 "이 획이 무엇인가"만 답한다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from .evidence import StrokeEvidence
from .select import _THIN_BND, _THIN_SIL
from .skeleton import _end_dir, _rdp_idx

# ── 역할 ──────────────────────────────────────────────────────────────
SILHOUETTE = "SILHOUETTE"            # 한쪽이 배경 — 가장 눈에 띄는 윤곽
STRUCTURE = "STRUCTURE"              # 길고 신뢰도 높은 주요 구조선
INTERNAL_CONTOUR = "INTERNAL_CONTOUR"  # 영역 경계인데 색으로는 안 갈리는 내부 윤곽
COLOR_BOUNDARY = "COLOR_BOUNDARY"    # 양옆 색이 실제로 다른 경계
FEATURE = "FEATURE"                  # 짧지만 고립·고중요 특징선 (콧선·입가)
TEXTURE = "TEXTURE"                  # 반복 무늬의 한 가닥
NOISE = "NOISE"                      # 부스러기

ROLES = (SILHOUETTE, STRUCTURE, INTERNAL_CONTOUR, COLOR_BOUNDARY,
         FEATURE, TEXTURE, NOISE)
# 보호 등급 — 예산 컷·단순화가 이 순서를 따른다 (작을수록 먼저 지킨다)
ROLE_RANK = {SILHOUETTE: 0, FEATURE: 1, STRUCTURE: 1, INTERNAL_CONTOUR: 2,
             COLOR_BOUNDARY: 2, TEXTURE: 3, NOISE: 4}


@dataclass
class LogicalStroke:
    """의미 있는 획 하나 — 후보 생성·정책 선택이 다루는 단위."""

    sid: int
    path: np.ndarray                  # (N,2) ROI-로컬 (y,x), 평활 후
    n_raw: int                        # 평활 전 표본 수 (파편 판정의 자)
    width: float                      # 폭 중앙값 px (정책 상한 적용 전)
    widths: np.ndarray                # 표본별 폭
    color: tuple[int, int, int]
    comp: int                         # 연결 성분 id
    roi: tuple[int, int, int, int]
    head_j: int = -1
    tail_j: int = -1
    ev: StrokeEvidence = field(default_factory=StrokeEvidence)
    intent: object = None             # 끊을 자리 (`intent.StrokeIntent`)
    role: str = STRUCTURE
    members: int = 1                  # 합쳐 들어온 원 간선 수 (이어긋기 자취)
    # 성분이 들고 다니는 배치 도구 (채점판·거리변환·접합점 차수 표)
    sc: object = None
    dt: object = None
    jdeg: dict = field(default_factory=dict)
    # 배치 결과 (정책 선택 뒤에 채워진다)
    kind: str = ""                    # 채택한 후보 종류
    shapes: int = 0                   # 쓴 도형 수
    seams: int = 0                    # 이음 보수 장수
    grown: int = 0                    # 이웃 도형 확장으로 메운 틈 수
    dropped: str = ""                 # 안 그었으면 그 이유
    cand: dict = field(default_factory=dict)   # 채택 후보의 계측

    @property
    def rank(self) -> int:
        """보호 등급 — 작을수록 먼저 지킨다.

        **detail 판에만 있던 선은 한 단 내린다** (요청 §1 "낮은 우선순위 detail
        후보"). 실루엣은 그 자체로 최상위라 안 내린다 — 내리면 detail이 처음
        찾아 준 윤곽이 도로 예산 컷에 먼저 걸린다.
        """
        r = ROLE_RANK.get(self.role, 3)
        if self.ev.detail_only >= 0.5 and r >= 1:
            r += 1
        return r


# ── 이어긋기 (§ stroke continuation) ───────────────────────────────────
# 접선 각차 상한 (도) — 이보다 벌어진 짝은 아예 안 본다. 35°는 옛 이음의 값이고
# 여기서는 **후보 문턱**일 뿐이다 (실제 선택은 아래 비용이 한다).
_JOIN_ANGLE = float(os.environ.get("FS_JOIN_ANGLE", 50.0))
# 비용 가중 — 전부 무차원으로 정규화한 뒤 더한다.
_W_TAN = 1.0        # 접선 불일치 (1 - cos)
_W_CURV = 0.6       # 곡률 불연속 (|Δκ| / (κ 합 + ε))
_W_WIDTH = 0.8      # 폭 불연속 (|Δw| / max w)
_W_CONF = 0.6       # 선 신뢰도 불연속
_W_SHAPE = 0.5      # 합친 뒤 늘어나는 마디 수 (음수면 이득)
_COST_MAX = float(os.environ.get("FS_JOIN_COST", 1.10))   # 이보다 비싸면 안 잇는다
# **주요 contour 우선** (§3 — 긴 선이 짧은 곁가지 때문에 끊기지 않게).
#
# 접합점 하나에서 이을 수 있는 짝이 여럿일 때 종전에는 **가장 싼 짝** 하나만
# 잇고 나머지 끝은 그 자리에서 버려졌다 (`_merge`가 접합점 id를 소비한다).
# 그래서 긴 A·긴 B·짧은 C가 만나는 T자에서 cost(A,C)가 조금이라도 싸면 A는
# 곁가지 C로 꺾여 나가고 **B는 영영 고아가 된다** — 사람이 한 획으로 긋는
# 주요 윤곽이 곁가지 하나 때문에 두 동강 나는 자리다.
#
# 새 비용 항을 더하지 않는다. 바꾸는 것은 **선택 규칙** 둘이다:
# ① 접합점을 **가장 긴 가닥이 붙은 순서로** 푼다 (종전은 접합점 id 순 = 임의).
#    주요 윤곽이 제 짝을 먼저 고른다.
# ② 그 접합점에서 최선 비용의 `_JOIN_SLACK` 안에 드는 짝들 중 **양쪽이 다
#    긴 짝**을 고른다 (자 = 짧은 쪽 길이). 후보가 하나뿐이면 아무 일도 안 한다.
# 비용 문턱(`_COST_MAX`)은 그대로다 — 안 이을 짝을 새로 잇지는 않는다.
_JOIN_SLACK = float(os.environ.get("FS_JOIN_SLACK", 0.25))
_JOIN_MAIN = os.environ.get("FS_JOIN_MAIN", "1") != "0"


def _tan(path: np.ndarray, head: bool) -> np.ndarray:
    return _end_dir(path, head)


def _end_curv(path: np.ndarray, head: bool) -> float:
    """끝 부근의 부호 있는 곡률 (진행 방향 기준, 1/px)."""
    k = min(9, len(path) - 1)
    if k < 4:
        return 0.0
    seg = path[:k + 1] if head else path[-k - 1:][::-1]
    v1 = seg[k // 2] - seg[0]
    v2 = seg[k] - seg[k // 2]
    cross = float(v1[0] * v2[1] - v1[1] * v2[0])
    den = float(np.hypot(*v1) * np.hypot(*v2) * np.hypot(*(seg[k] - seg[0])))
    return 2.0 * cross / den if den > 1e-9 else 0.0


def _nodes(path: np.ndarray, eps: float) -> int:
    """이 경로를 그릴 때 필요한 마디 수 어림 (`stroke._fit_segments`와 같은 자)."""
    return max(1, len(_rdp_idx(path, max(1.0, eps))) - 1)


def _join_cost(a: LogicalStroke, a_head: bool, b: LogicalStroke,
               b_head: bool) -> float | None:
    """두 획을 그 끝에서 이을 때의 비용 — 못 이으면 None.

    사람이 교차점에서 하나의 선을 계속 긋는 것과 닮은 조합을 고른다: 접선이
    이어지고, 굽음이 뒤집히지 않고, 폭·선 신뢰도가 연속이며, 합친 뒤 마디 수가
    안 늘어야 싸다. 가장 가까운 끝을 잇는 것이 아니다.
    """
    ta, tb = _tan(a.path, a_head), _tan(b.path, b_head)
    cos = -float(np.dot(ta, tb))              # 마주보면 +1
    if cos < np.cos(np.radians(_JOIN_ANGLE)):
        return None
    cost = _W_TAN * (1.0 - cos)
    ka, kb = _end_curv(a.path, a_head), _end_curv(b.path, b_head)
    # 진행 방향이 반대라 부호 규약을 맞춘다 — 한쪽을 뒤집어 이으므로 부호가 같아야
    # 같은 방향으로 휜 것이다
    cost += _W_CURV * abs(ka - kb) / (abs(ka) + abs(kb) + 0.02)
    wa, wb = max(a.width, 0.5), max(b.width, 0.5)
    cost += _W_WIDTH * abs(wa - wb) / max(wa, wb)
    cost += _W_CONF * abs(a.ev.basic - b.ev.basic)
    # 합친 뒤 마디 수 변화 — 이으면 RDP가 마디를 공유해 줄기도 한다
    eps = 0.7 * max(wa, wb)
    merged = np.concatenate([a.path[::-1] if a_head else a.path,
                             (b.path if b_head else b.path[::-1])[1:]], axis=0)
    d = _nodes(merged, eps) - _nodes(a.path, eps) - _nodes(b.path, eps)
    cost += _W_SHAPE * max(-1.0, min(2.0, float(d)))
    return cost


def continue_strokes(strokes: list[LogicalStroke]) -> list[LogicalStroke]:
    """접합점에서 이어지는 획 사슬을 하나로 합친다 (수렴까지 반복, 결정적).

    간선 단위로 각각 맞추면 하나의 자연스러운 획이 접합점마다 끊긴다. 접합점의
    끝 조합을 전부 재 이을 만한 짝을 고른다 — 한 번에 접합점 하나당 한 쌍씩,
    남은 끝은 다음 라운드에서 다시 겨룬다. 접합점을 푸는 순서와 그 안에서 짝을
    고르는 규칙은 **주요 윤곽이 먼저**다 (`_JOIN_MAIN` 문서).

    **잉크 뒷받침은 구조가 보증한다**: 여기서 잇는 두 끝은 같은 접합점 뭉치를
    나눠 쓰는 사이라 그 사이에 선 픽셀이 반드시 있다. 떨어진 조각을 잇는 일
    (선화의 점선)은 배치 전에 선 지도 위에서 따로 한다 (`bridge_line_gaps`).
    """
    items = [s for s in strokes if len(s.path) >= 2]
    changed = True
    while changed:
        changed = False
        ends: dict[int, list[tuple[int, bool]]] = {}
        for i, s in enumerate(items):
            if s.head_j >= 0:
                ends.setdefault(s.head_j, []).append((i, True))
            if s.tail_j >= 0:
                ends.setdefault(s.tail_j, []).append((i, False))
        merged: set[int] = set()
        new: list[LogicalStroke] = []
        plen = [float(len(t.path)) for t in items]
        # ① 가장 긴 가닥이 붙은 접합점부터 (`_JOIN_MAIN` 문서)
        order = (sorted(ends, key=lambda k: (-max(plen[a] for a, _ in ends[k]), k))
                 if _JOIN_MAIN else sorted(ends))
        for key in order:
            lst = ends[key]
            cands = []
            for a in range(len(lst)):
                for b in range(a + 1, len(lst)):
                    (i, hi), (j, hj) = lst[a], lst[b]
                    if i == j or i in merged or j in merged:
                        continue
                    c = _join_cost(items[i], hi, items[j], hj)
                    if c is not None and c <= _COST_MAX:
                        cands.append((c, min(plen[i], plen[j]), i, hi, j, hj))
            if not cands:
                continue
            # ② 최선 비용의 여유 안에서 **양쪽이 다 긴** 짝 (동점은 비용 순)
            lo = min(c[0] for c in cands)
            best = (min((c for c in cands if c[0] <= lo + _JOIN_SLACK),
                        key=lambda c: (-c[1], c[0], c[2], c[4]))
                    if _JOIN_MAIN else min(cands, key=lambda c: (c[0], c[2], c[4])))
            _, _, i, hi, j, hj = best
            new.append(_merge(items[i], hi, items[j], hj))
            merged.update((i, j))
            changed = True
        if changed:
            items = new + [s for k, s in enumerate(items) if k not in merged]
    return items


def _merge(a: LogicalStroke, a_head: bool, b: LogicalStroke,
           b_head: bool) -> LogicalStroke:
    """두 획을 한 획으로 — a의 접점 끝이 꼬리, b의 접점 끝이 머리가 되게 붙인다."""
    pa, ah, at = (a.path[::-1], a.tail_j, a.head_j) if a_head \
        else (a.path, a.head_j, a.tail_j)
    wa = a.widths[::-1] if a_head else a.widths
    pb, bh, bt = (b.path, b.head_j, b.tail_j) if b_head \
        else (b.path[::-1], b.tail_j, b.head_j)
    wb = b.widths if b_head else b.widths[::-1]
    path = np.concatenate([pa, pb[1:]], axis=0)
    widths = np.concatenate([wa, wb[1:]]) if len(wa) and len(wb) \
        else (wa if len(wa) else wb)
    # 색은 긴 쪽을 따른다 — 표본 median을 다시 뜨려면 원화 창이 필요하고,
    # 이어지는 두 조각의 색은 정의상 이미 가깝다
    big = a if len(a.path) >= len(b.path) else b
    return LogicalStroke(
        sid=min(a.sid, b.sid), path=path, n_raw=a.n_raw + b.n_raw,
        width=float(np.median(widths)) if len(widths) else big.width,
        widths=widths, color=big.color, comp=big.comp, roi=big.roi,
        head_j=ah, tail_j=bt, ev=big.ev, role=big.role,
        sc=big.sc, dt=big.dt, jdeg=big.jdeg,
        members=a.members + b.members)


# ── 역할 판정 (§ 구조선/특징선/texture 판별) ───────────────────────────
# 실루엣·경계 문턱은 양옆 표본을 재는 쪽(`select`)과 **같은 자**를 쓴다
_SIL, _BND = _THIN_SIL, _THIN_BND
# 파편 문턱 — cel의 6px는 면이 받쳐 줄 때의 값이다. 빈 바탕에서는 6~12px
# 부스러기가 그대로 낙서로 보인다 (01 preview 실측 — 눈가 X자 낙서). 고립
# 특징(콧선·입가)은 이보다 짧아도 긋는다: 사람도 그 둘은 짧은 획으로 남긴다.
# 단 고립이라도 0.6% 미만(7px@1200)은 AA 반점이라 무엇으로도 못 살린다
_LINE_FRAG_REL = 0.01     # 기본 파편 문턱 = 짧은 변의 1% (12px@1200)
_LINE_FRAG_MUL = 3.0      # × 획 폭과 큰 쪽
_ISO_LEN_REL = 0.006      # 고립 특징 생존 최소 길이 (7px@1200)
_DE = 12.0           # 양옆 Lab 색차 — 이상이면 색으로 설명되는 경계
_CONF = 0.55         # 선 신뢰도 — 이상이면 선화가 확실히 본 선
# detail 판의 신뢰도를 얼마로 쳐 줄까 — **낮은 우선순위 증거**다 (요청 §1).
# detail은 basic의 거의 상위집합이지만 해칭·잎사귀 노이즈를 함께 얹으므로
# (09 실측: 선 px +80%, 그중 48%가 detail 전용), 같은 값으로 세면 노이즈가
# 부스러기 보호를 통째로 뚫는다. 0.8이면 detail 0.69가 basic 0.55 자리다
_DETAIL_W = float(os.environ.get("FS_DETAIL_W", 0.8))
# 무늬 판정 — 넷을 **함께** 넘어야 무늬다 (하나만으로는 머리칼 다발과 안 갈린다)
_TEX_REPEAT = 0.70   # 나란한 이웃에 덮인 표본 비율
_TEX_PAR = 1.6       # 표본당 평행 이웃 수
_TEX_ENC = 0.55      # 폐쇄/갇힘 — 양옆이 둘 다 잉크
# **주변 획 대비** 중요도. 절대값을 쓰면 안 된다 — 획은 정의상 값이 가장 높은
# 자리라 전부 상한에 붙는다 (`evidence.StrokeEvidence.imp_rel` 문서)
_TEX_IMP = 1.0
# 무늬로 보려면 그 끝의 교차점이 몇 갈래여야 하나 — 3은 T자 하나뿐이라
# 머리칼이 갈라진 자리도 걸린다. 4부터가 선망 뭉치다
_TEX_DEG = int(os.environ.get("FS_TEX_DEG", 4))


def classify(strokes: list[LogicalStroke], frag_px: float, iso_px: float,
             min_w: float, simple: bool = False) -> None:
    """획마다 역할을 매긴다 (제자리 수정).

    순서가 뜻이다 — 보호 조건을 먼저 묻고 무늬·부스러기는 마지막에 남는 것에만
    묻는다. 그래서 레이스 무늬 한가운데의 콧선·입가도 무늬로 안 걸린다.
    **AA 반점(길이 < iso_px)은 무엇으로도 못 살린다** — 그 크기는 최소 도형
    으로도 못 그려서, 살려 봐야 놓을 도형이 없다.

    `simple=True`는 역할 판정을 통째로 끈 대조군이다 (ablation): 실루엣·고립·
    파편 셋만 옛 규칙 그대로 가른다 — 선 신뢰도 보호도, 무늬 판정도 없다.
    """
    for s in strokes:
        ev = s.ev
        short = s.n_raw < max(_LINE_FRAG_MUL * max(s.width, min_w), frag_px)
        speck = s.n_raw < iso_px
        # ① 실루엣 — 한쪽이 배경. 가장 눈에 띄는 윤곽이라 다른 조건을 안 본다
        if ev.sil >= _SIL and not speck:
            s.role = SILHOUETTE
            continue
        # ② 고립 특징 — **양끝이 자유고**(위상) 나란한 이웃이 하나도 없다
        #    (기하). 짧아도 빠지면 그 자리의 경계가 통째로 없어진다 (콧선·
        #    입가·손가락 사이). 위상 조건을 함께 묻는 것이 요점이다: 교차점에
        #    걸린 조각은 이웃이 없어 보여도 선망의 일부라 혼자 뜬 특징이 아니다
        if ev.repeat <= 1e-6 and ev.free_ends == 2 and not speck:
            s.role = FEATURE
            continue
        if simple:
            s.role = NOISE if short else STRUCTURE
            continue
        # ③ 부스러기 — 반점이거나, 짧으면서 옅고 아무 보호도 못 받는다.
        #    **명확한 선 신뢰도는 보호다** (짧은 콧선·눈꺼풀 주름) — 다만 반점
        #    아래로는 보호가 무의미하다
        conf = max(ev.basic, _DETAIL_W * ev.detail)
        if speck or (short and conf < _CONF and ev.imp_rel < _TEX_IMP):
            s.role = NOISE
            continue
        # ④ 무늬 — 나란한 이웃에 덮여 있고(반복성), 그 이웃이 촘촘하며(평행
        #    밀도), 주변 획들보다 안 띈다(상대 값). 세 조건이 "여럿이 같은
        #    리듬으로 놓여 있다"를 말한다.
        #
        #    종전에는 여기에 위상 조건(갇힘·자유 끝 0·접합점 차수 4+)이 더
        #    있었다. 그 셋은 **그물·레이스만** 잡는다 — 표준 11장에서 무늬
        #    판정이 한 개도 안 났고(report `texture_dropped` 0), 머리칼 다발
        #    처럼 한쪽 끝이 열린 반복은 전부 빠져나갔다. 위상은 "선망 안쪽인가"를
        #    묻지 "여럿이 같은 리듬인가"를 안 묻는다.
        #
        #    **무늬라고 지우는 것이 아니다.** 이 라벨은 다발마다 대표 몇 가닥만
        #    남기는 자리로 보낸다 (`texture_representatives`) — 사람이 머리칼
        #    열 가닥을 서너 가닥으로 줄여 긋는 그 손이다. 그래서 판정이 조금
        #    넉넉해도 특징이 통째로 사라지지 않는다.
        #
        #    실루엣·고립 특징은 위에서 이미 빠져나갔고, 상대 값 조건이 남아
        #    있어 다발 한가운데의 콧선·입가는 여기 안 걸린다.
        if (ev.repeat >= _TEX_REPEAT and ev.parallel >= _TEX_PAR
                and ev.imp_rel < _TEX_IMP and ev.sil < _SIL):
            s.role = TEXTURE
            continue
        # ⑤ 경계 획 — 양옆 색이 갈리거나 셀 영역이 갈린다. 색으로 설명되면
        #    COLOR_BOUNDARY, 아니면 색만으로는 안 생기는 내부 윤곽이라 더 세게
        #    지킨다 (선 노선의 `labels`는 실루엣 한 장이라 색차 쪽이 주된 자다)
        if ev.bnd >= _BND or ev.side_de >= _DE:
            s.role = COLOR_BOUNDARY if ev.side_de >= _DE else INTERNAL_CONTOUR
            continue
        # ⑥ 짧은 잔여는 부스러기, 나머지는 구조선
        s.role = NOISE if short else STRUCTURE


# 한 다발에서 남길 대표 가닥의 몫과 하한·상한 (사용자 지시: "머리카락 10가닥
# → 강도/길이/공간 분포를 대표하는 3~5가닥"). 몫이 먼저고 하한·상한이 뚜껑이다.
_TEX_KEEP_FRAC = float(os.environ.get("FS_TEX_KEEP", 0.33))
_TEX_KEEP_MIN, _TEX_KEEP_MAX = 3, 5


def _bundles(tex: list[LogicalStroke]) -> list[list[int]]:
    """무늬 획을 **다발로 묶는다** — 가깝고 나란한 것끼리 (union-find).

    다발이 단위인 것이 요점이다. 그림 전체의 무늬를 한 뭉치로 보면 머리칼과
    옷주름과 배경 빗금이 같은 저울에 올라, 한쪽이 다른 쪽을 통째로 밀어낸다.
    """
    n = len(tex)
    par = list(range(n))

    def find(a: int) -> int:
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a

    mid = []
    ang = []
    for s in tex:
        p = s.path
        m = p[len(p) // 2]
        mid.append((float(m[0] + s.roi[1]), float(m[1] + s.roi[0])))
        j0, j1 = max(0, len(p) // 2 - 2), min(len(p) - 1, len(p) // 2 + 2)
        t = p[j1] - p[j0]
        a2 = 2.0 * np.arctan2(t[0], t[1])
        ang.append((np.cos(a2), np.sin(a2)))
    for i in range(n):
        ri = max(6.0, 6.0 * max(tex[i].width, 1.0))
        for j in range(i + 1, n):
            if (mid[i][0] - mid[j][0]) ** 2 + (mid[i][1] - mid[j][1]) ** 2                     > ri * ri:
                continue
            if ang[i][0] * ang[j][0] + ang[i][1] * ang[j][1] < 0.6428:
                continue
            a, b = find(i), find(j)
            if a != b:
                par[a] = b
    out: dict[int, list[int]] = {}
    for i in range(n):
        out.setdefault(find(i), []).append(i)
    return [g for _k, g in sorted(out.items())]


def texture_representatives(strokes: list[LogicalStroke]) -> set[int]:
    """무늬 다발에서 **대표 가닥**으로 남길 획 (파이썬 id 집합).

    무늬를 통째로 지우면 그 자리가 빈다. 사람은 머리칼 열 가닥을 서너 가닥으로
    줄여 긋지 다 긋지도, 다 지우지도 않는다 (사용자 지시). 그래서 다발마다
    **리듬을 대표하는** 가닥을 남긴다:

    - 단위는 **다발**이다 (`_bundles`) — 머리칼과 옷주름이 서로를 안 밀어낸다.
    - 수는 다발 크기의 `_TEX_KEEP_FRAC`이고 3~5가 뚜껑이다.
    - 고르는 자리는 다발을 **가로지르는 축 위에 고르게** 편 자리다. 가닥의
      중점을 그 축에 투영해 등간격 눈금에 가장 가까운 가닥을 집으므로,
      남은 가닥들이 원래의 **간격**을 그대로 들고 있다. 같은 눈금을 두 가닥이
      다투면 오래 살아남는 쪽(§21 지속성)이 이기고, 그 다음이 긴 쪽이다
      (동점은 입력 순서 — 결정적이다).

    표준 11장에서는 이 손이 아무것도 안 버린다 — 다발이 안 잡히기 때문이다
    (`policy._TEXTURE_SIMPLIFY` 문서의 실측). 단일 인물화가 아니라 반복 해칭·
    레이스가 있는 그림에서 도는 자리다.
    """
    tex = [s for s in strokes if s.role == TEXTURE]
    if len(tex) < 4:
        return {id(s) for s in tex}          # 다발이 아니다 — 전부 남긴다
    keep: set[int] = set()
    for grp in _bundles(tex):
        if len(grp) <= _TEX_KEEP_MIN:
            keep |= {id(tex[i]) for i in grp}
            continue
        k = int(min(_TEX_KEEP_MAX,
                    max(_TEX_KEEP_MIN,
                        round(len(grp) * _TEX_KEEP_FRAC))))
        pts = np.array([[tex[i].path[len(tex[i].path) // 2][0] + tex[i].roi[1],
                         tex[i].path[len(tex[i].path) // 2][1] + tex[i].roi[0]]
                        for i in grp], np.float64)
        c = pts - pts.mean(0)
        # 다발을 **가로지르는** 축 = 2차 모멘트의 작은 축 (가닥은 나란하므로
        # 긴 축이 가닥 방향이고, 간격은 그 직교 방향에 실린다)
        ev, evec = np.linalg.eigh(np.cov(c.T) + np.eye(2) * 1e-9)
        proj = c @ evec[:, 0]
        lo, hi = float(proj.min()), float(proj.max())
        for t in range(k):
            want = lo + (hi - lo) * (t + 0.5) / k
            # 한 눈금을 두 가닥이 다투면 **오래 살아남는 쪽**이 이긴다 (§21
            # 지속성 — 배율을 낮춰도 남는 가닥이 그 다발의 대표다), 그 다음이
            # 긴 쪽, 마지막이 입력 순서다 (결정적)
            order = sorted(range(len(grp)),
                           key=lambda j: (abs(proj[j] - want),
                                          -tex[grp[j]].ev.persist,
                                          -tex[grp[j]].ev.length, j))
            for j in order:
                if id(tex[grp[j]]) not in keep:
                    keep.add(id(tex[grp[j]]))
                    break
    return keep
