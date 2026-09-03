"""차 **한 대**의 구성 — 면마다 무엇을 맡고 몇 장을 쓰나.

옛 길은 면마다 따로였다: 옆면·윗면이 도안을 받고, 나머지 면은 남은 자리에
모티프를 몇 장 흩는 것이 전부였다 (실측 12판: 리어 10장 · 프론트 10장 ·
도어 유리 4장 · 뒷유리 0장). 사람 판 28벌을 같은 자로 재면 그 자리가
리어 335 · 프론트 73 · 도어 유리 740 · 뒷유리 589장이다 —
**장수가 모자란 것이 아니라 그 면에 맡은 일이 없었다.**

여기서 하는 일은 셋이다.

1. **역할을 준다** (`assign_roles`) — 면 이름이 아니라 기하로 정한다:
   쓸 수 있는 넓이 · 가로세로비 · 유리인가 · 좌우 짝이 있나.
2. **도안 하나에서 변주를 짓는다** (`variants`) — 새 그림을 지어내지 않는다.
   있는 도안을 결정적으로 다시 자르고(머리·상반신) 색을 줄인다(포스터·엠블럼).
3. **예산을 나눈다** (`allocate`) — 면마다 "한 장 더 주면 얼마나 좋아지나"를
   보고 한계효용 순으로 준다. 사람 평균을 목표로 삼지 않는다: 자산의 **품질
   곡선이 포화**하면 남은 장수가 저절로 다른 면으로 간다.

## 차 한 대를 보는 배분 (`allocate_hier`)

3의 한계효용에는 **차 한 대가 없다.** 면마다 제 곡선이 포화할 때까지 받으므로,
빈 면을 다 채우고 나면 이번엔 무게가 고르게 퍼진다 — 사람 리버리에 있는
주역·조연·받침의 위계가 사라진다 (실측 33판: 나머지 무게 0.145 · 사람 p95
0.092 · 위계 점수 0.198 ↔ 사람 0.607).

고친 자리는 둘이다.

**① 무게 자를 평가기와 맞춘다.** 옛 예상 무게는 `면 넓이 × 품질`이었는데
품질이 포화하는 자라 사실상 **면 넓이 그 자체**였다. 실측으로 대 보니 몫이
어긋났다 (예상 ↔ 실측: 옆면 0.323↔0.416 · 윗면 0.368↔0.231). 지금은
평가기와 같은 식(`ink_weight` = 칠한 넓이 × 대비)을 배치 배율로 어림한다
(`expected_mass`) — 도안만 올라간 면에서 실측/예상이 1.00~1.19다.

**② 면 위의 나머지는 앞 판에서 잰다** (`measure_mass`). 옆면 무게의 63~84%는
꾸밈 그룹·면 도형이고 그것은 배분 시점에 아직 없다 — 넓이로도 잉크로도 몫이
안 맞았다 (접은 단위 몫 L1 0.26~0.58). 그래서 `auto.itasha.compose_config`가
**두 판**으로 돈다: 첫 판을 지어 재고, 그 실측을 넣어 다시 짓는다.

배수는 **범위**다 (`wholeeval.HUMAN_PRIOR`). 사람 p10~p90 안이면 벌점이 0이라
배수가 정확히 1 — 옛 배분과 같은 답이 나온다. 사람 p50으로 끌어당기는
controller가 아니다.

결정성: 난수 없음. 정렬은 전부 안정 정렬이고 동점은 이름으로 가른다.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, replace

import numpy as np

from ...i18n import msg
from ..catalog import Catalog
from ..model import LayerPlan
from .look import Look, layer_points
from .place import _refit_canvas


# 면 역할 — 사람 판 28벌에서 읽은 것이지 이름표가 아니다 (`work/lab/whole`).
# hero      주역 대형 (옆면·윗면)
# portrait  이차 인물 — 얼굴 크롭 (도어 유리)
# bust      상반신 크롭 (뒷유리)
# poster    색을 줄인 전신 (리어)
# emblem    3색 실루엣 뱃지 (프론트)
ROLES = ("hero", "portrait", "bust", "poster", "emblem")

# 유리 면 — 게임이 반투명으로 덮는다
GLASS = frozenset(("windshield", "rear_window", "window_left", "window_right",
                   "sunroof"))

# 이 몫보다 작은 면에는 변주를 안 앉힌다 (제일 큰 면 대비 쓸 수 있는 넓이).
# 스포일러·선루프처럼 손바닥만 한 면에 인물을 구겨 넣으면 파편으로 읽힌다.
MIN_AREA_FRAC = 0.06

# 변주 한 벌이 가질 수 있는 장수의 상·하한. 상한은 면 상한(1000)이 아니라
# **그 변주가 뜻을 갖는 크기**다 — 얼굴 크롭에 900장을 주면 원화의 머리를
# 통째로 옮긴 것이고, 그 위는 곡선이 이미 포화라 값을 못 한다.
VARIANT_CAP = {"portrait": 900, "bust": 700, "poster": 420, "emblem": 160,
               "hero": 3000}
VARIANT_MIN = {"portrait": 120, "bust": 100, "poster": 80, "emblem": 24,
               "hero": 1}

# 포스터·엠블럼의 색 수 (사람 실측: 리어 중앙 7색 · 프론트 4색 · 뒷유리 18색).
# 목표가 아니라 **역할의 정의**다 — 색을 줄이는 것이 그 역할이 하는 일이다.
VARIANT_COLORS = {"poster": 7, "emblem": 3, "bust": 18,
                  "portrait": 0}                  # 0 = 안 줄인다


@dataclass
class Variant:
    """도안 하나에서 결정적으로 뽑은 **다른 해석** 한 벌."""

    kind: str
    plan: LayerPlan
    why: str
    # **자른 상자** (변주 로컬 좌표, 원점 중심). 배치가 이 상자를 면에 맞춘다 —
    # 남긴 레이어의 잉크 범위를 쓰면 상자에 걸친 큰 색면이 캔버스를 원본만큼
    # 넓혀, 얼굴 크롭을 앉혀도 얼굴이 면의 1/4로 작아진다 (실측). 넘친 몫은
    # 면이 자른다 — 사람 판의 유리 인물도 가장자리에서 그렇게 잘려 있다.
    box: tuple[float, float, float, float] = (-1.0, -1.0, 1.0, 1.0)
    value: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))
    # **시각 무게** 레이어별 — 평가기의 `ruler.visual_weight`와 같은 정의다
    # (칠한 넓이 × 대비). `value`(무엇을 먼저 버리나)와 다른 물음이다.
    weight: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))
    # **칠한 넓이** 레이어별 — 옅게 하기(`fade`)의 밑감이다. 색을 제 평균색으로
    # 전부 끌어당기면 무게가 `넓이 × 바닥(0.05)`만 남으므로(`flat`), 그 사이는
    # 두 누계의 선형 혼합으로 얻는다 (`mass` — 배분이 이 자를 수천 번 묻는다).
    area: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))

    @property
    def flat(self) -> np.ndarray:
        """대비를 다 뺐을 때의 레이어별 무게 (`fade=1`의 끝점)."""
        return self.area * _CONTRAST_FLOOR
    # 잉크 상자 (배율 1의 캔버스 유닛) — 배치가 면에 맞출 크기다
    ink: tuple[float, float, float, float] = (-1.0, -1.0, 1.0, 1.0)

    def budgeted(self, n: int, fade: float = 0.0) -> LayerPlan:
        """장수 `n`으로 줄인 사본 — **값이 큰 것부터** 남기고 순서는 지킨다.

        `fade`는 **옅게 하기**다 (§21의 받침 문법): 남긴 색을 이 변주의
        넓이가중 평균색 쪽으로 그만큼 끌어당긴다. 1이면 단색이 된다. 자리와
        크기는 그대로라 **넓이는 안 줄고 대비만 준다** — 시각 무게를 낮추는
        둘째 손잡이이고, 첫째(`SurfaceJob.fill`)와 달리 잉크가 차지하는 면적을
        안 깎는다.
        """
        keep = (list(range(len(self.plan.layers)))
                if n >= len(self.plan.layers)
                else sorted(int(i) for i in
                            np.argsort(-self.value, kind="stable")[:n]))
        if fade <= 0.0:
            if n >= len(self.plan.layers):
                return self.plan
            return LayerPlan(source_image=self.plan.source_image,
                             image_size=self.plan.image_size,
                             units_per_px=self.plan.units_per_px,
                             layers=[replace(self.plan.layers[i]) for i in keep])
        lays = [self.plan.layers[i] for i in keep]
        ar = (self.area[np.asarray(keep, int)] if len(self.area)
              else np.ones(len(lays)))
        cols = _faded([l.color for l in lays], ar, fade)
        return LayerPlan(source_image=self.plan.source_image,
                         image_size=self.plan.image_size,
                         units_per_px=self.plan.units_per_px,
                         layers=[replace(l, color=c)
                                 for l, c in zip(lays, cols)])

    def quality(self, n: int) -> float:
        """장수 `n`일 때의 품질 — 남긴 레이어가 덮는 **시각 값**의 몫 (0~1).

        오목한 곡선이라 한계효용이 단조 감소다 — 예산 배분이 탐욕적이어도
        최적이다 (물채우기와 같은 성질).
        """
        if not len(self.value):
            return 0.0
        n = max(0, min(int(n), len(self.value)))
        s = np.sort(self.value)[::-1]
        tot = float(s.sum())
        return float(s[:n].sum() / tot) if tot > 1e-12 else 0.0

    def _keep(self, n: int) -> np.ndarray:
        """예산 `n`이 남기는 레이어 색인 — `budgeted`와 **같은 규칙**이다."""
        if n >= len(self.value):
            return np.arange(len(self.value))
        return np.argsort(-self.value, kind="stable")[:max(0, int(n))]

    def _cum(self, key: str = "_cumw", src: str = "weight") -> np.ndarray:
        """예산 순서(값 큰 것부터)로 쌓은 무게 누계 — `mass`가 매번 정렬하지
        않게 한 번만 짓는다 (배분이 이 자를 수천 번 묻는다)."""
        w = getattr(self, src)
        c = self.__dict__.get(key)
        if c is None or len(c) != len(w) + 1:
            if not len(w):
                c = np.zeros(1)
            else:
                order = np.argsort(-self.value, kind="stable")
                c = np.concatenate(([0.0], np.cumsum(np.asarray(w)[order])))
            self.__dict__[key] = c
        return c

    def mass(self, n: int, fade: float = 0.0) -> float:
        """예산 `n`이 남긴 레이어의 **시각 무게 합** (배율 1의 캔버스 유닛²).

        `quality`와 갈라 두는 까닭은 실측이다: 품질은 예산이 조금만 붙어도
        1에 붙어 버려 (리어 poster 416장에서 0.70, 유리 portrait 696장에서
        0.97) 무게 대리로 쓰면 **면 넓이 그 자체**가 된다. 무게는 안 그렇다 —
        얇은 획 700장과 넓은 색면 400장의 무게는 두 배 넘게 갈린다.

        `fade`는 **옅게 하기**다 (`budgeted`) — 대비가 그만큼 빠지므로 무게도
        그만큼 준다. 무게가 `넓이 × 대비`의 합이고 옅게 하기가 대비를 선형으로
        깎으므로, 두 끝점(`weight`·`flat`)의 선형 혼합이 정확한 값이다.
        """
        if not len(self.weight):
            return 0.0
        i = max(0, min(int(n), len(self.weight)))
        v = float(self._cum()[i])
        if fade <= 0.0 or not len(self.flat):
            return v
        f = float(min(max(fade, 0.0), 1.0))
        return (1.0 - f) * v + f * float(self._cum("_cumf", "flat")[i])


@dataclass
class SurfaceJob:
    """면 하나가 맡은 일."""

    name: str
    role: str
    kind: str                                     # 변주 이름 (hero면 원 도안)
    budget: int
    area: float                                   # 쓸 수 있는 넓이 (면 유닛²)
    why: str = ""
    # **시각 위계** (§3) — PRIMARY / SECONDARY / SUPPORT / MICRO.
    # 면 이름이 아니라 **예상 무게**가 정한다 (`assign_tiers`).
    tier: str = "MICRO"
    mass: float = 0.0                             # 예상 무게 몫 (거울 짝 접은 값)
    # **투영 몫** — 이 변주를 면의 몇 할 크기로 앉히나 (`place.BODY_FILL` 기준).
    # 장수를 깎지 않고 시각 무게만 낮추는 손잡이다 (§7): 무게는 이 값의
    # 제곱에 붙는다 (넓이 자라서). 1.0이면 종전과 같다.
    fill: float = 1.0
    # **옅게 하기** — 받침 면의 무게를 크기 대신 **대비**로 낮추는 둘째
    # 손잡이 (`Variant.budgeted`의 `fade`). 0이면 종전과 같다.
    #
    # 왜 둘째 손잡이가 필요한가. 크기(`fill`)만으로 무게를 맞추면 받침 면의
    # 그림이 **빽빽한 축소판**이 된다. 사람 판은 그렇지 않다 (실측, 그 판의
    # 옆면 잉크 상자를 1로 놓고 잰 중앙값 — 사람 29벌 ↔ W3H3 33판):
    #
    #     면       무게 몫        잉크 상자/옆면    장당 넓이      면 안 대비
    #     리어    .022 ↔ .022     .283 ↔ .094      1216 ↔   42   0.128 ↔ 0.270
    #     프론트  .020 ↔ .012     .514 ↔ .074      6832 ↔   63   0.151 ↔ 0.235
    #     유리    .033 ↔ .035     .096 ↔ .051       100 ↔   25   0.374 ↔ 0.351
    #
    # 무게 몫은 이미 맞았다 (§ W3H3의 성공). 갈리는 것은 **같은 무게를 무엇으로
    # 만들었나**다: 사람은 크고 옅은 표시를 넓게 펴고, 우리는 작고 진한 표시를
    # 좁게 모은다. 대비를 깎으면 무게가 선형으로 주는데 넓이는 안 준다 —
    # 크기 손잡이(무게 ∝ 크기²)와 방향이 다른 축이다.
    fade: float = 0.0


# **이음새를 건너 이어진다** (§17) — 두 면의 그림이 한 그림이라는 관계.
# 여기서는 그 관계를 **세우기만** 한다. 실제로 건너 그리는 손은 이미 있는
# `compose.seams.carry`이고, 이 자리는 "어느 짝이 그럴 자격이 있나"의 답이다.
CONTINUE_ACROSS_SEAM = "continue_across_seam"


@dataclass
class SeamLink:
    """이어질 수 있는 면 짝 하나 — 방향은 **무게가 큰 쪽에서 작은 쪽으로**."""

    src: str
    dst: str
    axis: str                                     # src에서 넘치는 축 (u|v)
    edge: float                                   # 이음선 (src 유닛)
    confidence: float = 0.0                       # 이음새 토막 신뢰의 중앙값
    why: str = ""


@dataclass
class WholeCarPlan:
    jobs: dict[str, SurfaceJob] = field(default_factory=dict)
    variants: dict[str, Variant] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    # 접은 단위(거울 짝 한 벌) → 등급·무게 몫. 주역이 어디인지의 답이다.
    hierarchy: dict[str, str] = field(default_factory=dict)
    faults: tuple = ()                            # 어긴 위계 규칙 (§3)
    # §17 — 이어질 수 있는 면 짝 (`CONTINUE_ACROSS_SEAM`)
    links: list[SeamLink] = field(default_factory=list)


# ── 면 기하 ─────────────────────────────────────────────────────────


def surface_area(smap) -> float:
    """쓸 수 있는 넓이 — 도색 상자 × 채움 몫 (면 유닛²)."""
    p0, q0, p1, q1 = smap.paint
    return max(0.0, (p1 - p0)) * max(0.0, (q1 - q0)) * max(0.0, float(smap.fill))


# 면이 맡는 일 — **사람 판 28벌이 그 면에 실제로 무엇을 두었나**에서 왔다
# (`work/lab/whole/human.json`, 작가 17인 중앙값):
#
#   도어 유리   740장 · 29색  → 얼굴이 벨트라인 위로 이어진다     portrait
#   뒷유리      589장 · 18색  → 상반신                            bust
#   리어        335장 ·  7색  → 색을 줄인 전신 (넓은 판 + 되풀이) poster
#   프론트       73장 ·  4색 · 큰 도형 몇   → 실루엣 뱃지         emblem
#
# 차종 상수가 아니다 — 게임 면 슬롯은 모든 차가 같은 열한 칸이다. 여기 없는
# 면(스포일러·선루프)은 **근거가 모자라서** 비운다: 작가 17인 중 둘만 쓴다.
# **윈드실드는 여기 없다.** 사람 판에서 그 면은 글자 자리다 (글리프 8.8% —
# 열한 면 중 가장 높다. 다음이 프론트 2.5%). 글자를 켜면 `facetext`가 이미
# 그 자리를 맡으므로, 글자가 없을 때 인물 크롭을 얹는 것은 근거가 없다 —
# 실제로 얹어 보니 얼굴이 면 밖으로 잘리고 목·가슴만 남았다 (실측).
ROLE_BY_SURFACE = {
    "window_left": "portrait", "window_right": "portrait",
    "rear_window": "bust", "rear": "poster", "front": "emblem",
}

# 이보다 길쭉한 면에는 인물을 안 앉힌다 — 띠로 내려앉는다 (기하 게이트).
THIN_ASPECT = 4.5

# 역할을 못 세울 때 물러나는 차례 — 도안에 따라 얼굴 크롭이 너무 작게 나오는
# 판이 있다 (머리를 못 찾은 도안). 그 면을 비우는 대신 한 단 물러난다.
FALLBACK = {"portrait": ("bust",), "bust": ("portrait",),
            "poster": ("emblem",), "emblem": ("poster",)}


def assign_roles(maps: dict, taken: set) -> dict[str, tuple[str, float]]:
    """면 → (역할, 넓이). `taken`은 이미 주역이 앉은 면이다.

    역할은 표가 주고 **기하가 거른다**: 지도가 의심스럽거나, 제일 큰 면 대비
    너무 작거나, 인물을 앉히기엔 너무 길쭉하면 역할이 내려앉거나 빠진다.
    """
    got: dict[str, tuple[str, float]] = {}
    areas = {n: surface_area(m) for n, m in maps.items()
             if m is not None and not m.uncertain}
    if not areas:
        return got
    big = max(areas.values())
    for name, a in sorted(areas.items()):
        role = ROLE_BY_SURFACE.get(name)
        if role is None or name in taken or big <= 0 or a / big < MIN_AREA_FRAC:
            continue
        m = maps[name]
        ar = max(m.width, m.height) / max(1e-6, min(m.width, m.height))
        if ar >= THIN_ASPECT:
            continue                               # 인물이 설 수 없는 띠 자리
        got[name] = (role, a)
    return got


# ── 변주 짓기 ───────────────────────────────────────────────────────


def _ink_box(plan: LayerPlan, cat: Catalog):
    lo = np.array([1e9, 1e9]), np.array([-1e9, -1e9])
    a, b = lo
    for l in plan.layers:
        pts = layer_points(l, cat)
        if len(pts):
            a = np.minimum(a, pts.min(axis=0))
            b = np.maximum(b, pts.max(axis=0))
    if b[0] < a[0]:
        return None
    return float(a[0]), float(a[1]), float(b[0]), float(b[1])


def _layer_boxes(plan: LayerPlan, cat: Catalog) -> np.ndarray:
    """(N,4) 레이어별 잉크 상자 (x0,y0,x1,y1) — 캔버스 유닛."""
    out = np.zeros((len(plan.layers), 4), np.float64)
    for i, l in enumerate(plan.layers):
        pts = layer_points(l, cat)
        if len(pts):
            out[i] = (pts[:, 0].min(), pts[:, 1].min(),
                      pts[:, 0].max(), pts[:, 1].max())
        else:
            out[i] = (l.x, l.y, l.x, l.y)
    return out


# 상자에 **얼마나 걸쳐야** 남기나. 닿기만 하면 남기던 첫 판은 얼굴 크롭에
# 전신 색면이 딸려 와서, 상자를 면에 맞추면 그림이 상자의 두 배로 서고 머리가
# 잘려 나갔다 (실측: 뒷유리에 목 아래만 남았다).
CROP_KEEP_SELF = 0.35             # 제 넓이의 이만큼이 상자 안이면 남긴다
CROP_KEEP_BOX = 0.60              # 또는 상자를 이만큼 덮으면 남긴다 (바탕 색면)


def _crop(plan: LayerPlan, cat: Catalog, box, boxes: np.ndarray) -> LayerPlan:
    """상자에 **제대로 걸친** 레이어만 남기고 상자 중심을 원점으로 옮긴 사본.

    남은 몫이 상자 밖으로 조금 넘치는 것은 그대로 둔다 — 면이 알아서 자른다
    (`layers_on`과 같은 규약. 사람 판의 유리 인물도 가장자리에서 잘려 있다).
    """
    x0, y0, x1, y1 = box
    ox = np.maximum(0.0, np.minimum(boxes[:, 2], x1) - np.maximum(boxes[:, 0], x0))
    oy = np.maximum(0.0, np.minimum(boxes[:, 3], y1) - np.maximum(boxes[:, 1], y0))
    ov = ox * oy
    own = np.maximum((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), 1e-9)
    barea = max((x1 - x0) * (y1 - y0), 1e-9)
    keep = np.where((ov > 0) & ((ov / own >= CROP_KEEP_SELF)
                                | (ov / barea >= CROP_KEEP_BOX)))[0]
    if len(keep) < 4:                             # 너무 깐깐하면 닿는 것 전부
        keep = np.where(ov > 0)[0]
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    got = LayerPlan(source_image=plan.source_image, image_size=plan.image_size,
                    units_per_px=plan.units_per_px,
                    layers=[replace(plan.layers[int(i)],
                                    x=plan.layers[int(i)].x - cx,
                                    y=plan.layers[int(i)].y - cy)
                            for i in keep])
    return _refit_canvas(got, cat)


def _lab(rgb) -> np.ndarray:
    c = np.asarray(rgb, np.float64) / 255.0
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124, 0.3576, 0.1805],
                  [0.2126, 0.7152, 0.0722],
                  [0.0193, 0.1192, 0.9505]])
    xyz = c @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]),
                     200 * (f[:, 1] - f[:, 2])], axis=1)


def _quantize(plan: LayerPlan, cat: Catalog, k: int, boxes: np.ndarray
              ) -> LayerPlan:
    """색을 `k`개 역할색으로 줄인 사본 — **넓이가 큰 색이 대표를 잡는다**.

    전역 k-means를 안 쓴다 (goal §34.4): 씨앗을 넓이 순으로 잡고 가까운 것을
    빨아들이는 결정적 탐욕이라 같은 입력이면 같은 표가 나오고, **큰 면의 색이
    반드시 살아남는다** (작은 하이라이트가 큰 바탕을 끌고 가지 않는다).
    """
    if k <= 0 or not plan.layers:
        return plan
    area = np.maximum((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
                      1e-9)
    cols: dict[tuple, float] = {}
    for i, l in enumerate(plan.layers):
        cols[tuple(l.color)] = cols.get(tuple(l.color), 0.0) + float(area[i])
    uniq = sorted(cols, key=lambda c: (-cols[c], c))
    if len(uniq) <= k:
        return plan
    lab = _lab(list(uniq))
    seeds: list[int] = []
    for i in range(len(uniq)):
        if len(seeds) >= k:
            break
        if all(float(np.linalg.norm(lab[i] - lab[j])) > 12.0 for j in seeds):
            seeds.append(i)
    while len(seeds) < min(k, len(uniq)):         # 다 가까우면 넓이 순으로 채운다
        for i in range(len(uniq)):
            if i not in seeds:
                seeds.append(i)
                break
    sl = lab[np.asarray(seeds, int)]
    table = {uniq[i]: uniq[seeds[int(np.argmin(
        np.linalg.norm(sl - lab[i], axis=1)))]] for i in range(len(uniq))}
    return LayerPlan(source_image=plan.source_image, image_size=plan.image_size,
                     units_per_px=plan.units_per_px,
                     layers=[replace(l, color=table[tuple(l.color)])
                             for l in plan.layers])


# 값 = 넓이^AREA_POW × 대비 × 주목. 넓이를 그대로 쓰면 **선이 통째로 죽는다** —
# 셀 도안의 획은 면보다 두 자릿수 작아서, 예산을 깎으면 큰 색면만 남고 얼굴이
# 사라졌다 (실측: 1,701장을 240장으로 깎았더니 머리가 통째로 없어졌다).
AREA_POW = 0.55
# 주목 자리에서의 값 배수 (§20의 subject_weight) — 얼굴이 예산 경쟁에서 이긴다.
FOCUS_GAIN = 3.0


def _value(plan: LayerPlan, cat: Catalog, boxes: np.ndarray,
           focus=None) -> np.ndarray:
    """레이어마다 **시각 값** — 예산을 깎을 때 무엇을 먼저 버리나를 정한다.

    넓이만 보면 큰 바탕만 남아 얼굴이 사라지고, 대비만 보면 티끌이 살아남는다.
    `focus`(중심 x·y·반경)를 주면 그 둘레의 값을 올린다 — 사람이 얼굴부터
    그리는 것과 같은 순서다.
    """
    if not plan.layers:
        return np.zeros(0)
    ar = np.maximum((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
                    1e-9)
    lab = _lab([l.color for l in plan.layers])
    mean = np.average(lab, axis=0, weights=ar)
    de = np.linalg.norm(lab - mean, axis=1) / 100.0 + 0.05
    v = np.power(ar, AREA_POW) * de
    if focus is not None:
        fx, fy, fr = focus
        cx = 0.5 * (boxes[:, 0] + boxes[:, 2])
        cy = 0.5 * (boxes[:, 1] + boxes[:, 3])
        d2 = ((cx - fx) ** 2 + (cy - fy) ** 2) / max(fr * fr, 1e-9)
        v = v * (1.0 + FOCUS_GAIN * np.exp(-0.5 * d2))
    return v


# ── 시각 무게 (예상) ────────────────────────────────────────────────
#
# 평가기가 판을 다 세운 뒤 재는 자는 `ruler.visual_weight` = Σ(칠한 넓이 × 대비)
# 다. 발전기가 배분할 때 쓰던 자는 `면 넓이 × 품질`이었는데, 실측으로 그 둘이
# 갈렸다 (33판 중앙값 · `work/lab/whole/predaudit.py`):
#
#   단위       예상 몫   실측 몫   비
#   side        0.323    0.416   1.29
#   top         0.368    0.231   0.63
#   front       0.014    0.031   2.20
#
# 까닭은 `quality`가 포화하는 자다 (예산이 붙으면 1에 붙는다). 그래서 예상
# 무게가 **면 넓이 그 자체**로 무너지고, 윗면처럼 넓지만 그림이 다 안 채우는
# 면이 옆면과 같은 주역으로 읽혔다 (§6의 top1/top2 문제).
#
# 여기서는 **평가기와 같은 정의**를 예상 쪽에도 놓는다: 배치 배율을 먼저
# 어림하고(`fit_scale`) 그 배율에서 칠해질 넓이 × 대비를 더한다.


_CONTRAST_FLOOR = 0.05        # 단색 면이 무게 0이 되는 것을 막는 바닥


def _areas_of(layers, cat: Catalog | None = None) -> np.ndarray:
    """레이어마다 **칠한 넓이** (배율 1의 캔버스 유닛²) — `ruler.areas`와 같은 식."""
    n = len(layers)
    out = np.empty(n, np.float64)
    for i, l in enumerate(layers):
        a = 4.0
        if cat is not None:
            try:
                a = float(cat[l.shape].area)
            except KeyError:
                pass
        out[i] = a * abs(float(l.sx) * float(l.sy)) * 64.0 * 64.0
    return out


def _faded(colors, area: np.ndarray, fade: float) -> list:
    """색을 **넓이가중 평균색 쪽으로** `fade`만큼 끌어당긴 사본 (sRGB 바이트).

    이동은 Lab에서 한다 — 무게 자(`ink_weight`)가 재는 대비도 Lab 거리라
    "옅게 한 만큼 무게가 준다"가 정확히 성립한다 (`Variant.mass`의 선형 혼합).
    """
    from ..celart.ramps import _lab_to_rgb

    f = float(min(max(fade, 0.0), 1.0))
    if f <= 0.0 or not len(colors):
        return [tuple(int(v) for v in c) for c in colors]
    lab = _lab(list(colors))
    w = np.maximum(np.asarray(area, np.float64), 1e-9)
    mean = np.average(lab, axis=0, weights=w)
    return [tuple(int(v) for v in c)
            for c in _lab_to_rgb(mean + (1.0 - f) * (lab - mean))]


def ink_weight(layers, cat: Catalog) -> np.ndarray:
    """레이어마다 **칠한 넓이 × 대비** (배율 1의 캔버스 유닛²).

    `work/lab/whole/ruler.visual_weight`와 **같은 식**이다 — 자를 두 벌 두면
    발전기와 평가기가 서로 다른 그림을 본다. 대비는 넓이가중 평균색과의 Lab
    거리이고, 바닥 0.05는 단색 면이 무게 0이 되는 것을 막는다.
    """
    if not len(layers):
        return np.zeros(0)
    ar = _areas_of(layers, cat)
    lab = _lab([l.color for l in layers])
    wmean = np.average(lab, axis=0, weights=np.maximum(ar, 1e-9))
    contrast = np.linalg.norm(lab - wmean, axis=1) / 100.0 + _CONTRAST_FLOOR
    return ar * contrast


# 배치가 쓰는 채움 몫·마스크 덮힘 — `place.BODY_FILL` · `place.fit_on`의 기본값과
# 같아야 어림이 맞는다. 여기서 다시 적는 까닭은 순환 수입을 피하려는 것뿐이다.
_FIT_FILL = 0.94
_FIT_COVER = 0.88


def fit_scale(smap, ink: tuple, *, group_unit: float = 1.0) -> float:
    """이 면에 이 잉크 상자를 앉힐 때의 **배율 × 그룹유닛** (면유닛/캔버스유닛).

    `place.fit_on` → `place_in_rect`의 수와 같다: 마스크 안에 같은 비율의
    가장 큰 상자를 넣고 `fill`을 곱한다. 마스크가 그 비율을 못 받으면
    `autoplace`의 폴백(도색 상자 × 0.8)으로 물러난다 — 그쪽도 같은 자다.
    """
    w = max(1e-6, float(ink[2] - ink[0]))
    h = max(1e-6, float(ink[3] - ink[1]))
    rect = smap.fit(w / h, coverage=_FIT_COVER, anchor="center", bias_x=0.5)
    if rect is None:
        p0, q0, p1, q1 = smap.paint
        return min((p1 - p0) / w, (q1 - q0) / h) * 0.8
    return min((rect[2] - rect[0]) / w, (rect[3] - rect[1]) / h) * _FIT_FILL


def expected_mass(smap, var: "Variant", n: int, *, fill: float = 1.0,
                  fade: float = 0.0) -> float:
    """면 `smap`에 변주 `var`를 `n`장 · 투영 몫 `fill`로 앉혔을 때의 예상 무게.

    배율의 제곱이 곱해진다 — 넓이 자이기 때문이다. 그래서 `fill`을 절반으로
    줄이면 무게가 1/4이 된다: **장수를 안 깎고 무게만 낮추는 손잡이**다 (§7).

    실측으로 이 어림은 도안만 올라간 면에서 거의 정확하다 (33판 중앙값
    실측/예상 = 유리 1.02 · 뒷유리 1.02 · 리어 1.07 · 프론트 1.19). 옆면·윗면은
    그 위에 꾸밈 그룹·면 도형이 더 올라가므로 4.2배·2.8배로 갈린다 — 그 몫은
    예상할 수 없어서 `base_mass`(앞 판의 실측)로 받는다.
    """
    g = fit_scale(smap, var.ink) * max(0.0, float(fill))
    return float(g * g * var.mass(n, fade))


# 얼굴 자리를 찾는 자 — 디테일 밀도의 **무게중심**이다.
# `intent.head`(머리 상자)는 이 자리에서 못 믿는다: 표준 5장에 대 보니 얼굴을
# 맞힌 것이 1.5뿐이고(누운 그림에서 엉덩이·손을 짚었다), 디테일 무게중심은
# 5장 모두를 맞혔다. 애니 그림의 얼굴은 눈·입·머리끝이 몰려 있어 정의상
# 디테일이 가장 촘촘한 자리다. 모델은 안 쓴다.
FOCUS_PCT = 92                # 이 백분위 위의 디테일만 무게로 센다
FOCUS_SIGMA = 1.5             # 상자 반지름 = 표준편차 × 이 값
FOCUS_MIN = 0.13              # 최소 반지름 (짧은 변의 몫)


def focus_point(it) -> tuple[float, float, float] | None:
    """도안에서 **눈이 가는 자리** — (x, y, 반지름) 캔버스 유닛."""
    import cv2

    alpha = getattr(it, "alpha", None)
    if alpha is None or not alpha.size:
        return None
    h, w = alpha.shape
    sil = (alpha > 0.5).astype(np.float32)
    k = max(5, int(0.06 * min(h, w)) | 1)
    sc = cv2.blur(it.detail * sil, (k, k)) * cv2.blur(sil, (k, k))
    if not (sc > 0).any():
        return None
    m = sc >= float(np.percentile(sc[sc > 0], FOCUS_PCT))
    ys, xs = np.nonzero(m)
    if not len(ys):
        return None
    g = sc[m]
    tot = float(g.sum()) or 1.0
    cy = float((ys * g).sum() / tot)
    cx = float((xs * g).sum() / tot)
    sy = math.sqrt(float(((ys - cy) ** 2 * g).sum() / tot))
    sx = math.sqrt(float(((xs - cx) ** 2 * g).sum() / tot))
    r = max(max(sx, sy) * FOCUS_SIGMA, FOCUS_MIN * min(h, w))
    x, y = it.to_xy(cx, cy)
    return float(x), float(y), float(r * it.upp)


def variants(plan: LayerPlan, lk: Look, it, cat: Catalog,
             kinds: set) -> dict[str, Variant]:
    """도안 하나 → 필요한 종류의 변주만. 없는 것은 조용히 빠진다.

    새 포즈를 지어내지 않는다 (goal §9) — 있는 레이어를 다시 자르고 색을
    줄이는 것뿐이다.
    """
    out: dict[str, Variant] = {}
    boxes = _layer_boxes(plan, cat)
    ink = _ink_box(plan, cat)
    if ink is None:
        return out
    ix0, iy0, ix1, iy1 = ink
    ih = max(1e-6, iy1 - iy0)
    focus = focus_point(it)
    head = getattr(it, "head", None)
    if focus is not None and head is not None and getattr(it, "head_confident", False):
        # 머리 상자가 그 자리를 감싸면 상자 쪽을 쓴다 — 머리끝까지 든다
        hx0, hy0, hx1, hy1 = head
        if hx0 <= focus[0] <= hx1 and hy0 <= focus[1] <= hy1:
            focus = ((hx0 + hx1) / 2, (hy0 + hy1) / 2,
                     max(focus[2], 0.5 * max(hx1 - hx0, hy1 - hy0)))

    def _add(kind: str, p: LayerPlan, why: str, box) -> None:
        if len(p.layers) < VARIANT_MIN.get(kind, 1):
            return
        b = _layer_boxes(p, cat)
        q = _quantize(p, cat, VARIANT_COLORS.get(kind, 0), b)
        hw, hh = (box[2] - box[0]) / 2.0, (box[3] - box[1]) / 2.0
        ib = (float(b[:, 0].min()), float(b[:, 1].min()),
              float(b[:, 2].max()), float(b[:, 3].max())) if len(b) else (
                  -hw, -hh, hw, hh)
        out[kind] = Variant(kind=kind, plan=q, why=why,
                            box=(-hw, -hh, hw, hh), ink=ib,
                            value=_value(q, cat, b, _shift(focus, box)),
                            weight=ink_weight(q.layers, cat),
                            area=_areas_of(q.layers, cat))

    def _around(scale: float) -> tuple:
        """주목 자리를 중심으로 반지름의 `scale`배 상자 (잉크 상자에 물린다)."""
        fx, fy, fr = focus
        r = fr * scale
        return (max(ix0, fx - r), max(iy0, fy - r),
                min(ix1, fx + r), min(iy1, fy + r))

    if "portrait" in kinds:
        if focus is not None:
            box, why = _around(1.6), msg("얼굴 자리 ±{k:g}배", k=1.6)
        else:
            box = (ix0, iy1 - 0.34 * ih, ix1, iy1)
            why = msg("얼굴 자리를 못 찾았다 — 잉크 상자 윗 34%")
        _add("portrait", _crop(plan, cat, box, boxes), why, box)

    if "bust" in kinds:
        if focus is not None:
            box, why = _around(3.0), msg("얼굴 자리 ±{k:g}배", k=3.0)
        else:
            box = (ix0, iy1 - 0.52 * ih, ix1, iy1)
            why = msg("얼굴 자리를 못 찾았다 — 잉크 상자 윗 52%")
        _add("bust", _crop(plan, cat, box, boxes), why, box)

    if "poster" in kinds:
        _add("poster", _crop(plan, cat, ink, boxes),
             msg("전신 · 색 {k}개로 줄임", k=VARIANT_COLORS["poster"]), ink)

    if "emblem" in kinds:
        _add("emblem", _crop(plan, cat, ink, boxes),
             msg("전신 실루엣 · 색 {k}개로 줄임", k=VARIANT_COLORS["emblem"]), ink)

    return out


def _shift(focus, box):
    """주목 자리를 자른 상자의 로컬 좌표로 옮긴다 (`_crop`이 중심을 원점으로)."""
    if focus is None:
        return None
    cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
    return focus[0] - cx, focus[1] - cy, focus[2]


# ── 실측 무게 (판이 다 선 뒤) ───────────────────────────────────────


def measure_mass(cfg_path, cat: Catalog) -> tuple[dict, dict]:
    """다 선 구성 파일 → (면별 **도안 말고** 있는 것의 무게, 면별 장수).

    `whole.allocate_hier`의 `base_mass`가 먹는 꼴이다. 도안 그룹(`decal-*`)의
    무게만 빼는 까닭은 그것이 배분이 실제로 움직일 수 있는 유일한 몫이라서다 —
    꾸밈 그룹·면 도형·글자는 다른 손이 짓는다.

    자는 평가기와 같다 (`ink_weight` = `ruler.visual_weight`). 대비는 면 전체의
    평균색을 기준으로 재므로 덩어리별 무게가 완전히 더해지지는 않는다 (실측
    0.51~1.00) — 그래서 **전체에서 도안 몫을 뺀다**. 도안 몫만은 예상과 실측이
    같다는 것을 실측으로 확인했다 (33판 8면 전부 비 1.000).
    """
    import json
    from pathlib import Path

    from .. import preview as _preview

    cfg_path = Path(cfg_path)
    raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    mass: dict[str, float] = {}
    counts: dict[str, int] = {}
    for item in raw.get("placements", []):
        name = item.get("surface")
        if not name:
            continue
        try:
            chunks = _preview.surface_chunks(item, cfg_path.parent, cat)
        except Exception:                          # noqa: BLE001 — 어림은 보조다
            continue
        allx = [l for _n, c in chunks for l in c]
        tot = float(ink_weight(allx, cat).sum())
        art = sum(float(ink_weight(c, cat).sum())
                  for n, c in chunks if n.startswith("decal"))
        mass[name] = max(0.0, tot - art)
        counts[name] = len(allx)
    return mass, counts


# ── 예산 ────────────────────────────────────────────────────────────


STEP = 8                                          # 한 번에 주는 장수
# 여덟 장이 **제 값을 해야** 받는다 — 이득이 이 아래면 그 면은 거기서 멈춘다.
# 전체 예산 상한을 두고 나눠 갖는 대신 이 문턱을 쓴다: 상한을 두면 사람 합계
# (중앙 8,665장)를 목표로 삼는 꼴이 되고 (§34.6), 문턱은 "이 여덟 장이 그림을
# 얼마나 바꾸나"만 묻는다. 이득은 `면 넓이 몫 × Δ품질`이고 Δ품질의 총합이
# 1이므로, 값은 "제일 큰 면에서 전체 값의 0.1%"라는 뜻이다.
MARGIN_MIN = 0.001


# ── 위계를 아는 배분 (§5) ────────────────────────────────────────────
#
# 옛 배분은 `면 넓이 몫 × Δ품질`만 봤다. 그 자에는 **차 한 대**가 없다 — 면
# 하나하나가 제 곡선이 포화할 때까지 받으므로, 면을 다 채우고 나면 무게가
# 고르게 퍼진다 (W1/W2 실측: 나머지 무게 0.145 · 사람 p95 0.092).
#
# 여기서 더하는 것은 **범위**다. 사람 백분위 표(`wholeeval.HUMAN_PRIOR`)를
# 목표로 삼지 않는다 (§31 금지): p10~p90 안이면 벌점이 0이라 배수가 정확히
# 1이고 (= 종전과 같은 배분), 밖으로 나가야 비로소 배수가 는다. 사람 p50으로
# 끌어당기는 controller가 아니다.

# 벌점 1점이 이득을 깎는 세기. 지수라 배수가 절대 음수가 안 되고, 벌점이
# 0인 자리에서 기울기가 이어진다.
HIER_K = 6.0

# 투영 몫 눈금 (§7) — 장수를 안 깎고 무게만 낮추는 손잡이. 사람 판의 리어·
# 유리 그림이 면을 꽉 채우지 않는 것과 같은 자리다. 아래로만 간다.
FILL_STEPS = (1.0, 0.85, 0.72, 0.6, 0.5, 0.42)
# **옅게 하기 눈금** (`SurfaceJob.fade`) — 받침 면의 무게를 크기 대신 대비로
# 낮추는 축. 0.55에서 멈추는 것은 사람 실측이 그 언저리이기 때문이다: 사람의
# 리어·프론트 면 안 대비가 0.128·0.151이고 우리가 0.270·0.235라, 대비를 절반쯤
# 깎으면 그 자리에 선다. 더 깎으면 단색 판이라 그림이 아니다.
FADE_STEPS = tuple(float(x) for x in
                   os.environ.get("FS_FADE_STEPS",
                                  "0,0.15,0.3,0.45,0.55").split(",") if x)
# 옅게 해도 되는 등급 — **받침 아래만** (§27: 주역·조연의 충실도는 안 건드린다).
# 등급은 배분 도중의 무게 몫으로 그 자리에서 매긴다 (`wholeeval.tiers` — 배치
# 뒤에 다시 매기는 그 함수 그대로다).
FADE_TIERS = frozenset(
    x for x in os.environ.get("FS_FADE_TIERS", "SUPPORT,MICRO").split(",") if x)
# 옅게 해도 되는 **역할** — 등급만으로는 부족하다. 사람 판의 면 안 대비를
# 역할로 갈라 보면 둘이 완전히 갈린다 (그 판의 옆면을 1로 놓고 잰 중앙값):
#
#     역할                사람   우리(W3H3)
#     poster (리어)       0.128     0.270    ← 사람이 훨씬 옅다
#     emblem (프론트)     0.151     0.235    ← 사람이 훨씬 옅다
#     portrait (유리)     0.374     0.351    ← 이미 사람 쪽이 **더 진하다**
#     bust (뒷유리)       0.344     0.331    ← 마찬가지
#
# 그래서 인물 크롭(portrait·bust)을 옅게 하면 **사람에게서 멀어진다**. 실측
# (SUP1, 등급으로만 걸러 유리까지 옅게 한 판): 유리의 잉크 상자/옆면이
# .051 → .073(사람 .096)이고 덮임이 .589 → .814(사람 .958)로 좋아지는데,
# 면 안 대비가 .351 → **.202**로 사람(.374)에게서 7배 멀어진다 — 도어 유리의
# 얼굴이 그만큼 바래는 거래다. 넓이 셋을 얻자고 얼굴을 바래게 하지 않는다.
#
# 전신을 줄인 판(poster·emblem)만 옅게 한다. 거기서는 방향이 맞다
# (리어 .270 → .251 · 프론트 .235 → .203, 둘 다 사람 쪽).
FADE_ROLES = frozenset(
    x for x in os.environ.get("FS_FADE_ROLES", "poster,emblem").split(",") if x)


def _feat_penalty(mass: dict, counts: dict, prior: dict | None = None) -> float:
    """무게·장수 한 벌의 **사람 범위 벗어남** (0 = 전부 범위 안).

    평가기와 같은 특징·같은 무게를 쓴다 (`wholeeval`) — 자를 두 벌 두면
    발전기가 고른 것과 평가기가 재는 것이 갈린다.
    """
    from . import wholeeval as WE

    ft = WE.features(mass, counts)
    if ft is None:
        return 0.0
    pen = WE.penalties(ft, prior)
    if not pen:
        return 0.0
    wsum = sum(WE.FEATURE_W.get(k, 0.5) for k in pen)
    return float(sum(WE.FEATURE_W.get(k, 0.5) * v for k, v in pen.items())
                 / max(wsum, 1e-9))


def _cap_of(name: str, kind: str, vs: dict, caps: dict | None) -> int:
    return min(int((caps or {}).get(name, 1000)),
               VARIANT_CAP.get(kind, 1000),
               len(vs[kind].plan.layers) if kind in vs else 0)


def allocate(roles: dict, vs: dict, *, caps: dict | None = None,
             margin: float = MARGIN_MIN, maps: dict | None = None,
             base_mass: dict | None = None, prior: dict | None = None
             ) -> dict[str, int]:
    """한계효용 순 배분 (§11) — 면마다 한 장 더 줬을 때의 이득이 큰 쪽부터.

    이득 = `면 넓이 몫 × Δ품질`. 넓이는 그 면이 차에서 얼마나 보이나의
    대리값이고, Δ품질은 그 변주의 곡선이 준다 — 곡선이 포화하면 그 면은
    저절로 더 안 받는다. 사람 평균을 목표로 쓰지 않는다 (§34.6).

    `maps`와 `base_mass`를 함께 주면 **차 한 대의 위계**가 이득에 곱해진다
    (`allocate_hier`) — 지금 판의 무게 몫이 사람 범위 밖으로 나가는 쪽이면
    이득이 깎인다. 둘 중 하나라도 없으면 옛 자 그대로다.
    """
    if not roles:
        return {}
    if maps is not None and base_mass is not None:
        return {n: j[0] for n, j in
                allocate_hier(roles, vs, caps=caps, margin=margin, maps=maps,
                              base_mass=base_mass, prior=prior).items()}
    big = max(a for _, a in roles.values()) or 1.0
    got = {n: 0 for n in roles}
    cap = {n: _cap_of(n, roles[n][0], vs, caps) for n in roles}
    while True:
        best, gain = None, margin
        for name in sorted(roles):
            kind, area = roles[name]
            v = vs.get(kind)
            if v is None or got[name] + STEP > cap[name]:
                continue
            g = (area / big) * (v.quality(got[name] + STEP) - v.quality(got[name]))
            if g > gain + 1e-12:
                best, gain = name, g
        if best is None:
            break
        got[best] += STEP
    # 하한에 못 미치는 면은 아예 안 쓴다 — 조각으로 남기지 않는다
    return {n: v for n, v in got.items()
            if v >= VARIANT_MIN.get(roles[n][0], 1)}


def allocate_hier(roles: dict, vs: dict, *, caps: dict | None = None,
                  margin: float = MARGIN_MIN, maps: dict,
                  base_mass: dict, base_counts: dict | None = None,
                  prior: dict | None = None
                  ) -> dict[str, tuple[int, float, float]]:
    """면 → (장수, 투영 몫, 옅게 하기). 위계를 아는 배분 (§4·§5·§7).

    `base_mass`는 **이 배분이 안 건드리는 것**의 시각 무게다 (주역 도안 · 꾸밈
    그룹 · 면 도형). 앞 판의 실측에서 온다 — 배분 시점에는 꾸밈이 아직 없어서
    예상할 수 없고, 실측으로 대 보니 옆면 무게의 63~84%가 꾸밈이었다.
    `base_counts`는 그 면들이 이미 쥔 장수다 (`decorated` 특징을 세는 데만
    쓴다) — 없으면 "이미 꾸민 면"으로 친다.

    세 판으로 돈다.

    1. **장수** — 옛 한계효용에 위계 배수를 곱해 탐욕적으로 준다.
    2. **무게 낮추기** — 장수를 고정한 채 무게가 넘치는 면을 한 눈금씩 줄인다.
       레이어는 그대로 두고 시각 무게만 낮추는 자리다 (§7). 벌점이 더 안
       줄면 멈춘다. 손잡이가 **둘**이고 방향이 다르다:

           투영 몫(`fill`)   그림을 작게 앉힌다 — 무게 ∝ 크기², 넓이가 준다
           옅게 하기(`fade`) 대비를 뺀다      — 무게 ∝ 대비, 넓이는 그대로

       크기만 쓰면 받침 면이 **빽빽한 축소판**이 된다 (`SurfaceJob.fade`의
       실측표: 사람의 받침 면은 잉크가 옆면의 .283~.514를 덮고 장당 넓이가
       1,216~6,832인데, W3H3는 .074~.094에 42~63이었다 — 무게 몫은 이미
       같은데 문법이 다르다). 그래서 **받침 아래 등급에서는 옅게 하기를 먼저
       묻는다**: 같은 벌점을 지우면서 넓이를 안 깎는 쪽이 문법에 맞다.
       등급은 그 자리의 무게 몫으로 매긴다 (`wholeeval.tiers` — 배치 뒤에
       다시 매기는 그 함수 그대로다).
    3. **장수 되돌리기** — 줄어든 크기에서 값을 못 하는 장을 **위에서부터**
       덜어낸다. 이득이 `면 넓이 몫 × Δ품질`인데 그 넓이가 이제 `fill²`만큼
       작아졌으므로, 안 보일 장은 문턱(`margin`)에 걸린다. 이 판이 없으면
       작게 앉힌 면에 큰 면의 장수가 그대로 남는다 — 실측으로 W3H의 받침 면
       레이어 중 **15~27%가 1유닛² 미만**으로 투영됐다 (W2는 2~8%).

       0에서 다시 쌓지 **않는** 까닭도 실측이다: 그렇게 하면 판에 따라 면이
       통째로 빠져 꾸민 면이 6 → 3으로 줄었다 (요청 §8의 실패 조건). 어느
       면을 세울지는 ①이 제 크기에서 정한 것이고, 이 판은 **장수만** 깎는다 —
       변주 하한(`VARIANT_MIN`) 아래로는 안 내려간다.

    결정성: 동점은 이름으로 가른다. 난수 없음. 판 수가 고정이라 수렴을
    기다리지 않는다.
    """
    big = max(a for _, a in roles.values()) or 1.0
    got = {n: 0 for n in roles}
    fill = {n: 1.0 for n in roles}
    cap = {n: _cap_of(n, roles[n][0], vs, caps) for n in roles}

    # 배치 배율은 예산·투영 몫과 무관하다 (잉크 상자와 면만 본다) — 마스크
    # 내접 탐색이라 비싸므로 면마다 **한 번만** 잰다. 이걸 매 시도마다 다시
    # 부르면 배분이 판당 9초씩 더 걸린다 (실측).
    gs: dict[str, float] = {}
    for nm in roles:
        v, sm = vs.get(roles[nm][0]), maps.get(nm)
        gs[nm] = fit_scale(sm, v.ink) if (v is not None and sm is not None) else 0.0

    def _mass(cur: dict, cf: dict, fd: dict | None = None) -> dict:
        m = dict(base_mass)
        for nm, b in cur.items():
            v = vs.get(roles[nm][0])
            if v is None or not gs.get(nm) or b <= 0:
                continue
            g = gs[nm] * cf[nm]
            m[nm] = m.get(nm, 0.0) + g * g * v.mass(
                b, 0.0 if fd is None else fd.get(nm, 0.0))
        return m

    from . import wholeeval as WE

    bc = dict(base_counts) if base_counts is not None else {
        n: WE.DECORATED_MIN for n in base_mass}

    def _counts(cur: dict) -> dict:
        c = dict(bc)
        for nm, b in cur.items():
            c[nm] = c.get(nm, 0) + int(b)
        return c

    def _budget(cur: dict, gate: float = margin) -> dict:
        """① 장수 — 위계 배수를 곱한 한계효용 순 (제자리 아님, 새 사전)."""
        cur = dict(cur)
        while True:
            pen0 = _feat_penalty(_mass(cur, fill), _counts(cur), prior)
            best, gain = None, gate
            for name in sorted(roles):
                kind, area = roles[name]
                v = vs.get(kind)
                if v is None or cur[name] + STEP > cap[name]:
                    continue
                # 넓이는 **투영된** 넓이다 — 작게 앉힌 면의 한 장은 그만큼
                # 덜 보이므로 덜 번다 (판 ③).
                g = (area * fill[name] * fill[name] / big) * (
                    v.quality(cur[name] + STEP) - v.quality(cur[name]))
                if g <= 0.0:
                    continue
                trial = dict(cur)
                trial[name] += STEP
                dp = _feat_penalty(_mass(trial, fill), _counts(trial),
                                   prior) - pen0
                g *= math.exp(-HIER_K * dp)
                if g > gain + 1e-12:
                    best, gain = name, g
            if best is None:
                return cur
            cur[best] += STEP

    got = _budget(got)
    live = {n: b for n, b in got.items()
            if b >= VARIANT_MIN.get(roles[n][0], 1)}
    # **주역이 가벼우면** 제 크기의 받침은 첫 장부터 위계를 깨서 ①이 아무 면도
    # 못 세운다 (사전이 사람 판으로 좁혀진 뒤 — top1 p95 .83). 면이 통째로
    # 빠지는 것이 §8의 실패 조건이므로 투영 몫을 한 눈금씩 줄여 다시 묻는다 —
    # ②가 하는 일을 ① 앞으로 당긴 것뿐이라 손잡이는 같다.
    for f in FILL_STEPS[1:]:
        if live:
            break
        fill = {n: f for n in roles}
        # 문턱도 f²로 — 같은 여덟 장이 작게 앉으면 그만큼 덜 버는 것은 구조다
        got = _budget({n: 0 for n in roles}, margin * f * f)
        live = {n: b for n, b in got.items()
                if b >= VARIANT_MIN.get(roles[n][0], 1)}
    if not live:
        return {}

    # ② 무게 낮추기 — 크기(`fill`)와 옅게 하기(`fade`) 두 손잡이
    fade = {n: 0.0 for n in live}

    def _tier_now() -> dict:
        """이 자리의 등급 — 평가기와 같은 함수다 (`wholeeval.tiers`)."""
        sh = WE.shares(_mass(live, fill, fade))
        return WE.tiers(sh) if sh else {}

    def _step(seq: tuple, cur: float) -> float | None:
        i = seq.index(cur) if cur in seq else 0
        return seq[i + 1] if i + 1 < len(seq) else None

    while True:
        pen0 = _feat_penalty(_mass(live, fill, fade), _counts(live), prior)
        if pen0 <= 1e-9:
            break
        tier = _tier_now()
        # **옅게 하기가 먼저다** — 받침 아래 등급에서만. 크기 한 눈금이
        # 지우는 벌점이 언제나 더 크므로(무게 ∝ 크기²) "많이 지우는 쪽"으로
        # 고르면 대비 축은 한 번도 안 뽑힌다. 그래서 크기와 겨루게 하지 않고
        # **먼저 물어본다**: 대비로 지울 수 있는 만큼 지우고, 그래도 남으면
        # 그때 크기를 깎는다. 어느 쪽이든 벌점이 더 안 줄면 그 자리에서 멈추므로
        # 필요 이상으로 옅어지지 않는다.
        best, drop = None, 1e-9
        for axis in ("fade", "fill"):
            for name in sorted(live):
                if axis == "fade":
                    if roles[name][0] not in FADE_ROLES:
                        continue           # 인물 크롭은 안 바래게 한다
                    if tier.get(WE.MIRROR.get(name, name),
                                "MICRO") not in FADE_TIERS:
                        continue
                    nxt = _step(FADE_STEPS, fade[name])
                    if nxt is None:
                        continue
                    trial = dict(fade)
                    trial[name] = nxt
                    m = _mass(live, fill, trial)
                else:
                    nxt = _step(FILL_STEPS, fill[name])
                    if nxt is None:
                        continue
                    trial = dict(fill)
                    trial[name] = nxt
                    m = _mass(live, trial, fade)
                d = pen0 - _feat_penalty(m, _counts(live), prior)
                if d > drop + 1e-12:
                    best, drop = (axis, name), d
            if best is not None:
                break                      # 대비로 지워지면 크기는 안 묻는다
        if best is None:
            break
        axis, name = best
        if axis == "fade":
            fade[name] = _step(FADE_STEPS, fade[name])
        else:
            fill[name] = _step(FILL_STEPS, fill[name])

    # ③ 장수 되돌리기 — 값을 못 하게 된 장을 위에서부터 덜어낸다
    while any(fill[n] < 1.0 for n in live):
        pen0 = _feat_penalty(_mass(live, fill, fade), _counts(live), prior)
        best, loss = None, margin
        for name in sorted(live):
            kind, area = roles[name]
            v = vs.get(kind)
            floor = VARIANT_MIN.get(kind, 1)
            if v is None or live[name] - STEP < floor:
                continue
            g = (area * fill[name] * fill[name] / big) * (
                v.quality(live[name]) - v.quality(live[name] - STEP))
            trial = dict(live)
            trial[name] -= STEP
            dp = _feat_penalty(_mass(trial, fill, fade), _counts(trial),
                               prior) - pen0
            g *= math.exp(HIER_K * dp)        # 덜어내서 나빠지면 손해가 커진다
            if g < loss - 1e-12:
                best, loss = name, g
        if best is None:
            break
        live[best] -= STEP
    return {n: (b, fill[n], fade[n]) for n, b in live.items()}


def assign_tiers(wc: WholeCarPlan, taken_mass: dict | None = None,
                 maps: dict | None = None) -> None:
    """면마다 **시각 위계**를 매긴다 (§3) — 제자리 수정, 레이어는 안 건드린다.

    등급을 면 이름으로 고정하지 않는다: 윗면이 주역인 판도 옆면이 주역인 판도
    있다. 정하는 것은 **예상 무게 몫**뿐이고, 자와 문턱은 평가기와 한 벌이다
    (`wholeeval.tiers`) — 배치 뒤 실측으로 다시 매기면 같은 함수가 답한다.

    예상 무게는 `넓이 × 품질(예산)`이다. 넓이는 그 면이 차에서 얼마나 보이나의
    대리값이고, 품질은 그 변주의 시각 값 중 예산이 실제로 담아낸 몫이다
    (`Variant.quality`). 실측 무게(`ruler.visual_weight` = Σ 칠한 넓이 × 대비)의
    **대리**지 같은 값이 아니다 — 라스터를 안 뜨는 자리라 그렇고, 판이 다 선
    뒤의 판정은 언제나 실측 쪽이 한다.

    `taken_mass`는 이미 주역이 앉은 면(옆면·윗면)의 예상 무게다 — 그 면들이
    빠진 채로 등급을 매기면 남은 면끼리 주역을 나눠 갖는 꼴이 된다.
    """
    from . import wholeeval as WE

    mass = dict(taken_mass or {})
    for name, job in wc.jobs.items():
        v = wc.variants.get(job.kind)
        sm = (maps or {}).get(name)
        if maps is not None and v is not None and sm is not None:
            # 위계를 아는 길 — 배분이 쓴 것과 **같은** 무게 자다
            mass[name] = mass.get(name, 0.0) + expected_mass(
                sm, v, job.budget, fill=job.fill, fade=job.fade)
            continue
        q = v.quality(job.budget) if v is not None else 0.0
        mass[name] = mass.get(name, 0.0) + job.area * q
    sh = WE.shares(mass)
    if sh is None:
        return
    tier = WE.tiers(sh)
    wc.hierarchy = dict(tier)
    wc.faults = WE.tier_faults(sh)
    for name, job in wc.jobs.items():
        unit = WE.MIRROR.get(name, name)
        job.tier = tier.get(unit, "MICRO")
        job.mass = float(sh.get(unit, 0.0))


# 등급 사이의 **내리막**만 잇는다 — 무게가 작은 면에서 큰 면으로 그림이
# 흘러 들어오면 주역이 어디인지가 흐려진다 (§3의 위계).
_TIER_RANK = {"PRIMARY": 0, "SECONDARY": 1, "SUPPORT": 2, "MICRO": 3}


def seam_links(wc: WholeCarPlan, atlas, taken: set) -> list[SeamLink]:
    """이어질 수 있는 면 짝 (§17) — 관계만 세운다, 그리지는 않는다.

    자격은 넷이다.

    * 두 면이 **둘 다 일을 맡고 있다** (도안이 앉은 면이거나 변주를 받은 면).
    * `atlas`에 그 이음새가 서 있다 (`compose.atlas.Seam`).
    * 이음새 토막의 신뢰가 `seams.CONF_MIN` 위다 — 두 마스크가 그 자리에서
      서로 다른 것을 쥔 짝은 이어 붙이면 자리가 틀어진다.
    * 방향이 **내리막**이다 (주역 → 조연 → 받침).

    실제로 건너 그리는 것은 `compose.seams.carry`다 — 이 목록은 그 자에게
    "어느 짝을 물어볼 가치가 있나"를 주는 자리고, 못 잇기로 판정되면 거기서
    끊긴다 (`seams` 문서의 못 이으면 끊는다).
    """
    from . import seams as _seams

    out: list[SeamLink] = []
    if atlas is None:
        return out
    live = set(wc.jobs) | set(taken)
    seen: set = set()
    for src in sorted(live):
        for seam in atlas.seams.get(src, ()):
            if (src, seam.dst) in seen:
                continue                          # 같은 짝의 둘째 이음새
            seen.add((src, seam.dst))
            if seam.dst not in live:
                continue
            a = wc.hierarchy.get(_unit(src), "MICRO")
            b = wc.hierarchy.get(_unit(seam.dst), "MICRO")
            if _TIER_RANK.get(a, 3) > _TIER_RANK.get(b, 3):
                continue                          # 오르막 — 위계가 흐려진다
            segs = seam.fold.segments
            conf = (float(np.median([sg.confidence for sg in segs]))
                    if segs else 1.0)
            if conf < _seams.CONF_MIN:
                continue
            out.append(SeamLink(src=src, dst=seam.dst, axis=seam.fold.axis,
                                edge=float(seam.fold.edge), confidence=conf,
                                why=msg("{a} → {b} 이음새 신뢰 {c:.2f}",
                                        a=src, b=seam.dst, c=conf)))
    return out


def _unit(name: str) -> str:
    """면 이름 → 접은 단위 이름 (거울 짝은 한 벌, `wholeeval.MIRROR`)."""
    from .wholeeval import MIRROR

    return MIRROR.get(name, name)


def plan_car(plan: LayerPlan, lk: Look, it, cat: Catalog, maps: dict, *,
             taken: set, caps: dict | None = None,
             margin: float = MARGIN_MIN,
             taken_mass: dict | None = None,
             base_mass: dict | None = None,
             base_counts: dict | None = None) -> WholeCarPlan:
    """차 한 대의 구성 — 역할 → 변주 → 예산 → 위계 (§3).

    `taken_mass`는 이미 주역이 앉은 면의 예상 무게다 (`assign_tiers`).

    `base_mass`를 주면 **위계를 아는 배분**으로 간다 (`allocate_hier`) — 이
    배분이 안 건드리는 것(주역 도안 · 꾸밈 · 면 도형)의 실측 무게이고 앞 판에서
    잰다. `taken_mass`는 그 안에 이미 들어 있어야 한다. 없으면 옛 자 그대로다
    (바이트 동일).
    """
    wc = WholeCarPlan()
    roles = assign_roles(maps, taken)
    if not roles:
        return wc
    want = {k for k, _ in roles.values()}
    wc.variants = variants(plan, lk, it, cat,
                           want | {b for k in want for b in FALLBACK.get(k, ())})
    # 못 지은 역할은 **한 단 물러난다** — 머리를 못 찾은 도안에서 얼굴 크롭이
    # 너무 작게 나오면 그 면을 비우는 대신 상반신·띠가 대신 선다.
    picked: dict[str, tuple[str, float]] = {}
    for name, (kind, area) in roles.items():
        for k in (kind, *FALLBACK.get(kind, ())):
            if k in wc.variants:
                picked[name] = (k, area)
                break
    roles = picked
    if base_mass is not None:
        got = allocate_hier(roles, wc.variants, caps=caps, margin=margin,
                            maps=maps, base_mass=base_mass,
                            base_counts=base_counts)
    else:
        got = {n: (b, 1.0, 0.0) for n, b in
               allocate(roles, wc.variants, caps=caps, margin=margin).items()}
    for name, (n, fl, fd) in sorted(got.items()):
        kind, area = roles[name]
        why = wc.variants[kind].why
        if fl < 1.0:
            why += msg(" · 면의 {k:.0%} 크기로 (위계)", k=fl)
        if fd > 0.0:
            why += msg(" · 대비를 {k:.0%} 뺐다 (받침)", k=fd)
        wc.jobs[name] = SurfaceJob(name=name, role=kind, kind=kind, budget=n,
                                   area=area, why=why, fill=fl, fade=fd)
    if base_mass is not None:
        assign_tiers(wc, base_mass, maps)
    else:
        assign_tiers(wc, taken_mass)
    return wc
