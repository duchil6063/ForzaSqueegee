"""구성 그래프 — 요소가 아니라 **요소 사이**를 적는다.

사람이 리버리를 짤 때 정하는 것은 "여기에 사각 하나, 저기에 별 열둘"이 아니다.
"이 판은 인물을 **받치고**, 저 사선은 판을 **가로지르고**, 별무리는 판의 **반대쪽
으로 무게를 나눈다**"는 관계다. 요소는 그 관계를 이루려고 서는 것이지 그 반대가
아니다.

지금 코드는 요소만 있고 관계가 없다 — `bed`가 판을 내고 `scatter`가 조각을
내지만 그 둘이 어떤 사이인지는 아무 데도 안 적혀 있고, 그래서 아무도 못 잰다.
그 결과가 실측에 그대로 나온다: 33판 전부에서 판이 하나뿐이고(둘째 덩어리가
첫째의 19%), 조각은 예외 없이 판과 **같은 쪽**에 몰린다.

## 노드와 관계

노드는 구도에서의 **역할**이다 (`ROLES`) — 도형 종류가 아니다. 관계는 잴 수
있는 것만 둔다 (`RELATIONS`):

    parallel_to  축이 나란하다
    counter_to   축이 어긋난다 (40~90°) — 판 둘이 서로를 가로지른다
    continues    한쪽 끝에서 다른 쪽이 이어 나간다 (이음새 너머 포함)
    frames       상자가 상대를 품는다
    supports     상대 뒤에 깔려 상대를 받친다
    overlaps     겹친다
    avoids       안 겹친다 (틈이 있다)
    echoes       작은 되풀이다 (축이 같고 크기가 몫이다)
    balances     주역을 사이에 두고 반대쪽에서 무게를 맞춘다

문법은 계열이 쥔다 (`families.Family.grammar`) — "이 계열은 무엇과 무엇이 어떤
사이여야 하나". 점수는 그 문법이 **실제 기하에서 지켜졌나**를 잰다
(`relation_score`). 지키라고 강제하지 않는다: 후보 여럿이 서로 다른 정도로
지키고, 잘 지킨 것이 이긴다.

좌표는 전부 **프레임 좌표**다 (`design`의 꾸밈 캔버스).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..model import UNITS_PER_SCALE, Layer
from .boxes import major_axis


# 구도에서의 역할 — 도형이 아니라 자리다.
ROLES = ("hero", "macro", "counter", "motif", "echo", "text", "negative", "trim")


RELATIONS = ("parallel_to", "counter_to", "continues", "frames", "supports",
             "overlaps", "avoids", "echoes", "balances", "deliberate")


# `counter_to`가 만족스러운 각 구간 (도). 40° 아래면 "비뚤어진 나란함"으로
# 읽히고 (레퍼런스에 없는 꼴이다 — 나란하거나 확실히 가로지르거나 둘 중 하나다),
# 90°가 온전한 가로지름이다.
COUNTER_DEG = (40.0, 90.0)


# `echoes`의 크기 몫 구간 — 되풀이는 원본의 이만큼이라야 "작은 되풀이"로 읽힌다.
ECHO_RATIO = (0.10, 0.62)


# `balances`에서 두 무게가 서로 몇 배 안이어야 맞선 것으로 보나.
BALANCE_RATIO = 4.0


# `frames`에서 품는 쪽이 품기는 쪽의 몇 배까지인가.
FRAME_RATIO = (1.3, 4.5)


# **공통 문법** — 계열과 무관하게 언제나 재는 관계다. 여기 넣는 조건은 하나다:
# 후보 손잡이(팔레트·베드 크기·흐름·좌표하강)에 따라 **값이 실제로 갈릴 것**.
# 만들기 나름으로 늘 1.000이 되는 관계는(예: `avoids(motif, hero)`는 산포가
# 이미 보장한다) 자가 아니라 희석이다 — 그런 항목 넷이 열한 항목짜리 옛 점수를
# 못 쓰게 만들었다.
#
#   balances(motif, neg)   무리와 여백이 주역을 사이에 두고 맞선다
#   frames(macro0, hero)   큰 색면이 인물을 품되 삼키지 않는다
#   overlaps(macro1, m0)   둘째 색면이 첫째와 **만난다** (따로 뜬 판이 아니다)
#   echoes(echo, motif)    잔 조각이 무리의 작은 되풀이다
#
# `deliberate`(나란하거나 확실히 가로지르거나)는 여기 없다 — 짝의 각을
# `macro.counter_angle`이 늘 40° 밖으로 두므로 33판 전부 1.000이었다. 관계는
# 남겨 두되(계열이 쓸 수 있다) 공통 문법에서는 뺀다.
# 글자가 있는 판에만 걸리는 둘 — 없는 노드를 가리키는 관계는 안 세므로
# (`relation_score`) 글자 없는 판은 옛 판과 같은 자로 재진다.
#   balances(text, motif)   글자와 무리가 주역을 사이에 두고 맞선다
#   parallel_to(text, m0)   워드마크가 큰 색면과 나란히 달린다 (레이싱 문법)
DEFAULT_GRAMMAR = (
    ("balances", "motif", "neg", 1.0),
    ("frames", "macro0", "hero", 1.0),
    ("overlaps", "macro1", "macro0", 0.9),
    ("echoes", "echo", "motif", 0.6),
    ("balances", "text", "motif", 0.7),
    ("parallel_to", "text", "macro0", 0.5),
)


@dataclass(frozen=True)
class Node:
    """구도 요소 하나 — **무엇인가**가 아니라 **어떤 자리인가**."""

    id: str
    role: str
    at: tuple[float, float]                    # 중심 (프레임 좌표)
    axis: tuple[float, float] = (1.0, 0.0)     # 장축 단위벡터
    extent: tuple[float, float] = (0.0, 0.0)   # (길이, 폭) — 프레임 유닛
    box: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    weight: float = 0.0                        # 시각 무게 (프레임 넓이 대비)
    kind: str = ""                             # 원시형 이름 (ribbon·blade·cluster…)
    z: int = 0

    @property
    def length(self) -> float:
        return self.extent[0]

    @property
    def width(self) -> float:
        return self.extent[1]


@dataclass(frozen=True)
class Rel:
    kind: str
    a: str
    b: str
    weight: float = 1.0


@dataclass
class CompositionGraph:
    nodes: dict[str, Node] = field(default_factory=dict)
    rels: tuple[Rel, ...] = ()

    def add(self, n: Node) -> None:
        self.nodes[n.id] = n

    def get(self, i: str) -> Node | None:
        return self.nodes.get(i)

    def of_role(self, role: str) -> list[Node]:
        return [n for n in self.nodes.values() if n.role == role]


# ---- 기하 헬퍼 -------------------------------------------------------------

def _clip(box, frame):
    x0 = max(box[0], frame[0])
    y0 = max(box[1], frame[1])
    x1 = min(box[2], frame[2])
    y1 = min(box[3], frame[3])
    return (x0, y0, max(x0, x1), max(y0, y1))


def _area(box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _inter(a, b) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return w * h if w > 0 and h > 0 else 0.0


def _cos(a: tuple[float, float], b: tuple[float, float]) -> float:
    na, nb = math.hypot(*a), math.hypot(*b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return abs((a[0] * b[0] + a[1] * b[1]) / (na * nb))


def _obox(at, axis, extent) -> tuple[float, float, float, float]:
    """중심·축·(길이, 폭) → 축 정렬 상자 (회전 사각의 외접)."""
    hx, hy = extent[0] / 2, extent[1] / 2
    ax, ay = axis
    ex = abs(ax) * hx + abs(ay) * hy
    ey = abs(ay) * hx + abs(ax) * hy
    return (at[0] - ex, at[1] - ey, at[0] + ex, at[1] + ey)


def _soft(v: float, lo: float, hi: float, soft: float) -> float:
    if v < lo:
        return max(0.0, 1.0 - (lo - v) / max(1e-6, soft))
    if v > hi:
        return max(0.0, 1.0 - (v - hi) / max(1e-6, soft))
    return 1.0


# ---- 레이어 → 노드 --------------------------------------------------------

def _rect_node(l: Layer, i: str, role: str, frame, z: int) -> Node:
    """사각 레이어 하나 → 노드. 길이는 축 방향, 폭은 그 법선 방향이다.

    판·띠는 프레임 밖까지 뻗으므로(`bed._through` — 끝은 차가 낸다) 무게는
    **프레임 안으로 자른 넓이**로 잰다. 안 그러면 밖으로 길게 나간 띠가 판보다
    무거워진다.
    """
    r = math.radians(l.rot)
    axis = (math.cos(r), math.sin(r))
    ext = (2 * abs(l.sx) * UNITS_PER_SCALE, 2 * abs(l.sy) * UNITS_PER_SCALE)
    box = _clip(_obox((l.x, l.y), axis, ext), frame)
    fa = _area(frame) or 1.0
    return Node(id=i, role=role, at=(l.x, l.y), axis=axis, extent=ext, box=box,
                weight=_area(box) / fa, kind="rect", z=z)


def _cluster_node(pts: list[tuple[float, float, float]], i: str, role: str,
                  frame, z: int) -> Node | None:
    """조각 무리 하나 → 노드. 축은 자리들의 장축, 크기는 퍼진 범위다."""
    if not pts:
        return None
    xs = np.array([p[0] for p in pts], np.float64)
    ys = np.array([p[1] for p in pts], np.float64)
    cx, cy = float(xs.mean()), float(ys.mean())
    if len(pts) >= 3:
        d, _e = major_axis(xs, ys)
    else:
        dx, dy = float(xs[-1] - xs[0]), float(ys[-1] - ys[0])
        n = math.hypot(dx, dy) or 1.0
        d = (dx / n, dy / n)
    t = (xs - cx) * d[0] + (ys - cy) * d[1]
    s = -(xs - cx) * d[1] + (ys - cy) * d[0]
    sizes = [p[2] for p in pts]
    ln = float(t.max() - t.min()) + max(sizes)
    wd = float(s.max() - s.min()) + max(sizes)
    box = _clip((cx - ln / 2, cy - wd / 2, cx + ln / 2, cy + wd / 2), frame)
    # 조각 무리의 무게는 **덮은 넓이의 합**이다 (외접 상자가 아니다 — 무리는
    # 성기므로 상자로 재면 판보다 무거워진다)
    ink = sum(math.pi * (0.5 * sz) ** 2 for sz in sizes)
    return Node(id=i, role=role, at=(cx, cy), axis=d, extent=(ln, wd), box=box,
                weight=ink / (_area(frame) or 1.0), kind="cluster", z=z)


def derive(fld, layers_back: list[Layer], layers_front: list[Layer],
           motifs: list[tuple[float, float, float, int]],
           text_poses=None, negative_box=None) -> CompositionGraph:
    """지어 놓은 부품에서 그래프를 **되읽는다**.

    Phase 3~4가 그래프를 먼저 짓고 기하를 거기서 내기 전까지의 다리다 — 관계를
    지금 기하에서 잴 수 있게 해 준다 (같은 관계 자를 그 뒤에도 그대로 쓴다).
    """
    frame = fld.frame_box
    g = CompositionGraph()
    pb = fld.person_box
    fa = _area(frame) or 1.0
    g.add(Node(id="hero", role="hero", at=fld.visual_center,
               axis=fld.axis, extent=(fld.char_w, fld.char_h),
               box=_clip(pb, frame), weight=_area(_clip(pb, frame)) / fa, z=50))
    beds = [l for l in layers_back if l.label == "itasha_bed"]
    # 판은 무거운 것부터 macro · counter · trim 순이다 — 가장 무거운 것이 구도의
    # 큰 색면이고, 그 다음이 그것을 가로지르는 짝이다.
    nodes = [_rect_node(l, f"_b{k}", "macro", frame, z=10 + k)
             for k, l in enumerate(beds)]
    nodes.sort(key=lambda n: (-n.weight, n.at[0], n.at[1]))
    for k, n in enumerate(nodes):
        role = "macro" if k == 0 else ("counter" if k == 1 else "trim")
        g.add(Node(id=f"macro{k}", role=role, at=n.at, axis=n.axis,
                   extent=n.extent, box=n.box, weight=n.weight, kind=n.kind,
                   z=n.z))
    stripes = [l for l in layers_back if l.label == "itasha_stripe"]
    if stripes:
        w = sum(_area(_rect_node(l, "x", "trim", frame, 0).box) for l in stripes) / fa
        base = _rect_node(max(stripes, key=lambda l: abs(l.sx) * abs(l.sy)),
                          "rocker", "trim", frame, z=20)
        g.add(Node(id="rocker", role="trim", at=base.at, axis=base.axis,
                   extent=base.extent, box=base.box, weight=w, kind="rocker", z=20))
    mo = _cluster_node([(m[0], m[1], m[2]) for m in motifs], "motif", "motif",
                       frame, z=30)
    if mo is not None:
        g.add(mo)
    ech = [l for l in layers_back if l.label == "itasha_echo"]
    if ech:
        en = _cluster_node([(l.x, l.y, 2 * max(abs(l.sx), abs(l.sy)) * UNITS_PER_SCALE)
                            for l in ech], "echo", "echo", frame, z=32)
        if en is not None:
            g.add(en)
    if layers_front:
        fn = _cluster_node([(l.x, l.y, 2 * max(abs(l.sx), abs(l.sy)) * UNITS_PER_SCALE)
                            for l in layers_front], "front", "motif", frame, z=60)
        if fn is not None:
            g.add(fn)
    if text_poses:
        p = text_poses[0]
        r = math.radians(p.rot)
        w, h = p.w, p.h
        at = (p.x, p.y)
        axis = (math.cos(r), math.sin(r))
        box = _clip(_obox(at, axis, (w, h)), frame)
        g.add(Node(id="text", role="text", at=at, axis=axis, extent=(w, h),
                   box=box, weight=_area(box) / fa, kind="wordmark", z=40))
    if negative_box is not None:
        g.add(Node(id="neg", role="negative",
                   at=((negative_box[0] + negative_box[2]) / 2,
                       (negative_box[1] + negative_box[3]) / 2),
                   axis=(1.0, 0.0),
                   extent=(negative_box[2] - negative_box[0],
                           negative_box[3] - negative_box[1]),
                   box=_clip(negative_box, frame),
                   weight=_area(_clip(negative_box, frame)) / fa,
                   kind="void", z=0))
    return g


# ---- 관계 재기 ------------------------------------------------------------

def _rel_value(kind: str, a: Node, b: Node, hero: Node | None) -> float:
    """관계 하나가 **얼마나 지켜졌나** (0~1)."""
    if kind == "parallel_to":
        return _cos(a.axis, b.axis)
    if kind == "counter_to":
        deg = math.degrees(math.acos(max(0.0, min(1.0, _cos(a.axis, b.axis)))))
        return _soft(deg, COUNTER_DEG[0], COUNTER_DEG[1], 30.0)
    if kind == "continues":
        # b의 시작이 a의 끝에 닿고 축이 나란한가
        ax, ay = a.axis
        end = (a.at[0] + ax * a.length / 2, a.at[1] + ay * a.length / 2)
        d = math.hypot(b.at[0] - end[0], b.at[1] - end[1])
        near = max(0.0, 1.0 - d / max(1e-6, 0.5 * (a.length + b.length)))
        return 0.5 * near + 0.5 * _cos(a.axis, b.axis)
    if kind == "frames":
        # 품되 **삼키지 않는다** — 상대를 다 담으면서 제 크기가 상대의 1.3~4.5배
        # 안이라야 "품는 판"이다. 위로 열어 두면 밴드를 통째로 덮은 판이 만점을
        # 받는다 (실측: 판 넓이가 인물의 5.8배가 그렇게 나왔다).
        ab = _area(b.box)
        if ab <= 0:
            return 0.0
        inside = _inter(a.box, b.box) / ab
        return inside * _soft(_area(a.box) / ab, *FRAME_RATIO, soft=2.5)
    if kind == "supports":
        ab = _area(b.box)
        if ab <= 0:
            return 0.0
        return (_inter(a.box, b.box) / ab) * (1.0 if a.z < b.z else 0.4)
    if kind == "overlaps":
        m = min(_area(a.box), _area(b.box))
        return min(1.0, _inter(a.box, b.box) / m) if m > 0 else 0.0
    if kind == "avoids":
        m = min(_area(a.box), _area(b.box))
        if m <= 0:
            return 1.0
        return max(0.0, 1.0 - _inter(a.box, b.box) / m)
    if kind == "echoes":
        sa = max(a.extent) or 1e-6
        sb = max(b.extent) or 1e-6
        return 0.5 * _cos(a.axis, b.axis) + 0.5 * _soft(sa / sb, *ECHO_RATIO, soft=0.35)
    if kind == "deliberate":
        # 나란하거나 확실히 가로지르거나 — 그 **사이**(15~40°)가 실수처럼 읽힌다.
        # 레퍼런스의 판 둘은 예외 없이 둘 중 하나다.
        return max(_soft(_cos(a.axis, b.axis), 0.94, 1.0, 0.10),
                   _rel_value("counter_to", a, b, hero))
    if kind == "balances":
        if hero is None:
            return 0.5
        va = (a.at[0] - hero.at[0], a.at[1] - hero.at[1])
        vb = (b.at[0] - hero.at[0], b.at[1] - hero.at[1])
        na, nb = math.hypot(*va), math.hypot(*vb)
        if na < 1e-6 or nb < 1e-6:
            return 0.0
        # 주역을 사이에 두고 반대쪽인가 (내적이 음수) + 무게가 견줄 만한가
        opp = max(0.0, -(va[0] * vb[0] + va[1] * vb[1]) / (na * nb))
        wa, wb = max(a.weight, 1e-9), max(b.weight, 1e-9)
        ratio = max(wa, wb) / min(wa, wb)
        return 0.65 * opp + 0.35 * _soft(ratio, 1.0, BALANCE_RATIO, BALANCE_RATIO)
    return 0.5


def relation_score(g: CompositionGraph) -> tuple[float, dict[str, float]]:
    """문법이 기하에서 지켜진 정도 — 관계마다 재서 가중평균.

    문법에 적힌 노드가 이 후보에 없으면 그 관계는 **안 센다** (없는 것을 벌하지
    않는다 — 계열마다 요소가 다르다). 아무것도 못 재면 중립 0.5다.
    """
    hero = g.get("hero")
    tot = wsum = 0.0
    info: dict[str, float] = {}
    for r in g.rels:
        a, b = g.get(r.a), g.get(r.b)
        if a is None or b is None:
            continue
        v = _rel_value(r.kind, a, b, hero)
        tot += r.weight * v
        wsum += r.weight
        info[f"rel_{r.kind}"] = max(info.get(f"rel_{r.kind}", 0.0), v)
    if wsum <= 0:
        return 0.5, info
    return tot / wsum, info
