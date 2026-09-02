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

결정성: 난수 없음. 정렬은 전부 안정 정렬이고 동점은 이름으로 가른다.
"""

from __future__ import annotations

import math
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

    def budgeted(self, n: int) -> LayerPlan:
        """장수 `n`으로 줄인 사본 — **값이 큰 것부터** 남기고 순서는 지킨다."""
        if n >= len(self.plan.layers):
            return self.plan
        keep = sorted(int(i) for i in np.argsort(-self.value, kind="stable")[:n])
        return LayerPlan(source_image=self.plan.source_image,
                         image_size=self.plan.image_size,
                         units_per_px=self.plan.units_per_px,
                         layers=[replace(self.plan.layers[i]) for i in keep])

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


@dataclass
class SurfaceJob:
    """면 하나가 맡은 일."""

    name: str
    role: str
    kind: str                                     # 변주 이름 (hero면 원 도안)
    budget: int
    area: float                                   # 쓸 수 있는 넓이 (면 유닛²)
    why: str = ""


@dataclass
class WholeCarPlan:
    jobs: dict[str, SurfaceJob] = field(default_factory=dict)
    variants: dict[str, Variant] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


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
        out[kind] = Variant(kind=kind, plan=q, why=why,
                            box=(-hw, -hh, hw, hh),
                            value=_value(q, cat, b, _shift(focus, box)))

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


# ── 예산 ────────────────────────────────────────────────────────────


STEP = 8                                          # 한 번에 주는 장수
# 여덟 장이 **제 값을 해야** 받는다 — 이득이 이 아래면 그 면은 거기서 멈춘다.
# 전체 예산 상한을 두고 나눠 갖는 대신 이 문턱을 쓴다: 상한을 두면 사람 합계
# (중앙 8,665장)를 목표로 삼는 꼴이 되고 (§34.6), 문턱은 "이 여덟 장이 그림을
# 얼마나 바꾸나"만 묻는다. 이득은 `면 넓이 몫 × Δ품질`이고 Δ품질의 총합이
# 1이므로, 값은 "제일 큰 면에서 전체 값의 0.1%"라는 뜻이다.
MARGIN_MIN = 0.001


def allocate(roles: dict, vs: dict, *, caps: dict | None = None,
             margin: float = MARGIN_MIN) -> dict[str, int]:
    """한계효용 순 배분 (§11) — 면마다 한 장 더 줬을 때의 이득이 큰 쪽부터.

    이득 = `면 넓이 몫 × Δ품질`. 넓이는 그 면이 차에서 얼마나 보이나의
    대리값이고, Δ품질은 그 변주의 곡선이 준다 — 곡선이 포화하면 그 면은
    저절로 더 안 받는다. 사람 평균을 목표로 쓰지 않는다 (§34.6).
    """
    if not roles:
        return {}
    big = max(a for _, a in roles.values()) or 1.0
    got = {n: 0 for n in roles}
    cap = {n: min(int((caps or {}).get(n, 1000)),
                  VARIANT_CAP.get(roles[n][0], 1000),
                  len(vs[roles[n][0]].plan.layers) if roles[n][0] in vs else 0)
           for n in roles}
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


def plan_car(plan: LayerPlan, lk: Look, it, cat: Catalog, maps: dict, *,
             taken: set, caps: dict | None = None,
             margin: float = MARGIN_MIN) -> WholeCarPlan:
    """차 한 대의 구성 — 역할 → 변주 → 예산."""
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
    budget = allocate(roles, wc.variants, caps=caps, margin=margin)
    for name, n in sorted(budget.items()):
        kind, area = roles[name]
        wc.jobs[name] = SurfaceJob(name=name, role=kind, kind=kind, budget=n,
                                   area=area, why=wc.variants[kind].why)
    return wc
