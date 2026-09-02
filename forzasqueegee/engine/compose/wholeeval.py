"""차 한 대의 **위계 평가** — 층을 갈라 묻는다 (goal §1~§4).

## 왜 층인가

옛 자는 가중합 하나였다 (`compose.score`). 가중합은 **큰 장점 하나가 치명적인
단점 하나를 상쇄한다** — 얼굴이 잘려 나간 판이 구도 점수 하나로 이길 수 있다.
그래서 묻는 순서를 셋으로 가른다.

    ① 성립하나 (`validate`)         — 하나라도 어기면 후보가 아니다.
    ② 도안이 살아 있나 (`fidelity`) — 꾸밈 점수로 못 갚는다. 바닥이 있다.
    ③ 위계가 사람 범위인가 (`hierarchy`) — 여기서만 연속 점수가 나온다.

## ③이 재는 것 — **고르게 퍼진 것이 목표가 아니다**

사람 리버리에는 주역·조연·받침의 위계가 있다. 그래서 `role_spread`(면별 장수의
엔트로피)를 최대로 미는 것은 목표가 아니라 **과분산**이다. 실측이 그것을 그대로
보여 준다 (사람 27판·작가 17인 ↔ 우리 33판, 거울 짝을 접은 자로 · 중앙값):

| | 사람 p10 | 사람 p50 | 사람 p90 | W0(전) | W1(후) |
|---|---|---|---|---|---|
| top1 | 0.537 | 0.628 | 0.850 | 0.599 | **0.448** |
| ent  | 0.281 | 0.553 | 0.649 | 0.473 | **0.765** |
| rem  | 0.000 | 0.028 | 0.081 | 0.005 | **0.145** |

W0은 한 덩이에 쏠려 있었고 W1은 반대편으로 지나갔다. 자는 그 **둘 다**를
벌해야 한다 — 그래서 목표값이 아니라 **범위**로 문다 (§2): 사람 p10~p90 안이면
벌점이 없고, p5~p95 밖으로 나가면 빠르게 는다.

## 거울 짝을 먼저 접는다

옆면 좌/우와 도어 유리 좌/우는 **한 벌**이다 (보는 사람은 한 번에 한쪽만
본다). 안 접으면 사람 판의 `p2/p1`이 1.000으로 나와 — 짝의 무게가 정의상 같다 —
위계 자가 통째로 죽는다 (실측 28판 중 13판이 정확히 1.000).

## 결정성

난수 없음. 정렬은 안정 정렬이고 동점은 이름으로 가른다. 백분위 표는 상수라
같은 입력이면 같은 점수가 나온다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# ── 거울 짝 — 한 벌로 접는다 ─────────────────────────────────────────
MIRROR = {"side_left": "side", "side_right": "side",
          "window_left": "window", "window_right": "window"}

# 이 장수 이상이면 "꾸민 면"으로 센다 (`ruler.car_stats`의 자와 같다)
DECORATED_MIN = 20

# 위계 특징 — 전부 **접은** 무게 몫에서 나온다 (`features`)
FEATURES = ("units", "decorated", "top1", "top2", "top3", "rem",
            "hhi", "ent", "ps", "ss")

# 백분위 자리 (`HUMAN_PRIOR` 값의 순서)
QS = (5, 10, 25, 50, 75, 90, 95)

# 사람 corpus의 백분위 표 — `work/lab/whole/prior.py`가 뽑는다.
# 작가 17인 단위로 먼저 접은 값이다 (§32: 한 사람이 같은 문법을 다섯 올려도
# 표가 다섯이 되지 않게). **목표값이 아니라 범위다.**
#
# 사람 판 26벌 · 작가 16인 (2026-09-02: 무게가 0인 도색 전용 판 하나와 **본인
# 판 하나**(옆면 하나만 있는 시험 판 — 만든이 태그가 프로필 id)를 뺐다. 그 판이
# 빠지면서 아래 끝이 올라갔다: units p5 2.6→3.75 · top1 p95 0.95→0.83 ·
# ent p5 0.15→0.30 · ps p5 0.04→0.16).
HUMAN_PRIOR: dict[str, tuple] = {
    "units": (3.75, 4.5, 5, 5.75, 6.25, 7, 7.25),
    "decorated": (2.75, 3, 4, 5, 6, 7, 7),
    "top1": (0.5112, 0.5353, 0.5818, 0.6256, 0.7241, 0.7852, 0.827),
    "top2": (0.7964, 0.8038, 0.8315, 0.8861, 0.9396, 0.9876, 0.9905),
    "top3": (0.9077, 0.9179, 0.9434, 0.9708, 0.9918, 0.9994, 0.9997),
    "rem": (0.0003, 0.0006, 0.0082, 0.0292, 0.0566, 0.0821, 0.0923),
    "hhi": (0.3932, 0.3951, 0.4338, 0.482, 0.5662, 0.6503, 0.7197),
    "ent": (0.3037, 0.364, 0.4982, 0.5558, 0.6031, 0.6521, 0.6756),
    "ps": (0.1571, 0.2024, 0.2816, 0.4312, 0.5348, 0.5713, 0.6519),
    "ss": (0.0384, 0.0751, 0.2929, 0.488, 0.7075, 1.134, 1.29),
}

# 특징마다의 무게 — 위계를 정하는 것이 무겁다. `units`·`decorated`는 "몇 면을
# 썼나"라 위계 자체는 아니고, 범위를 크게 벗어난 판(면 하나에 다 몰기)만 잡는다.
FEATURE_W = {
    "units": 0.4, "decorated": 0.6,
    "top1": 1.0, "top2": 0.8, "top3": 0.5, "rem": 0.5,
    "hhi": 1.0, "ent": 1.0, "ps": 0.8, "ss": 0.8,
}

# p5~p95 밖으로 나간 뒤의 기울기 — 범위 폭(p95−p5)만큼 더 나가면 벌점이
# 이만큼 는다. 안(p10~p90)은 0이고 사이 구간은 0→1로 이어진다.
OUT_SLOPE = 4.0

# ③ 안에서도 **다른 물음은 갈라 둔다** — 하나가 다른 하나를 못 갚게.
#
#   surface_use    몇 면이 실제로 일을 맡았나 (빈 면 문제)
#   concentration  그 무게가 주역·조연·받침으로 갈리나 (쏠림/과분산)
#
# 갈라야 하는 근거는 실측이다. W0(전)은 **쏠림이 아니라 빈 면**이 문제였고
# (`decorated` 2 · 나머지 여덟 특징이 전부 사람 범위 안), W1(후)은 반대로
# 면은 다 채웠는데 무게가 퍼졌다 (`rem` 0.196 · 사람 p95 0.092). 두 벌을 한
# 가중평균에 넣으면 W0의 빈 면이 나머지 여덟에 묻혀 사람 판과 같은 점수가
# 나온다 (실측 0.842 ↔ 사람 0.826). 그래서 **무리마다 재고 나쁜 쪽을 쓴다.**
GROUPS = {
    "surface_use": ("units", "decorated"),
    "concentration": ("top1", "top2", "top3", "rem", "hhi", "ent", "ps", "ss"),
}

# ── 재료 (2026-09-02 계획 §0) — 셋째 무리, **재면 묻고 안 재면 안 묻는다** ──
#
# 사람 판 30벌(작가 17인)과의 간극은 위계가 아니라 **재료 수**였다: 사람 옆면은
# 큰 색면 15장·6색을 바닥에 깔고 그 위에 주역 한 덩어리(레이어 상자의 절반),
# 그 위에 로고·글자 6단어를 얹는다. 우리는 색면 1~3장·주역 작게·산포였다.
# 자는 `work/lab/whole/ruler.material_stats`(면 하나) → 여기서 **옆면 좌우를
# 평균으로 접어** 특징 한 벌로 만든다. 발전기는 이 무리를 아직 안 본다 —
# `evaluate(material=None)`이면 무리가 서지 않아 옛 점수 그대로다.
MATERIAL_FEATURES = ("big_n", "big_cols", "hero_share", "objects", "words",
                     "glyph_h", "skew", "mask", "alpha_lt", "colors")

# 사람 corpus의 백분위 표 — `work/lab/whole/prior.py`가 뽑아 붙인다 (p5~p95).
# **목표값이 아니라 범위다.** 없는 특징은 안 묻는다. 옆면 좌우 평균, 작가
# 15~16인 단위로 접은 값 (2026-09-02, 사람 판 26벌).
#
# `colors`는 작가 하나(Tohru, 옆면 2,400색)가 p95를 800으로 끌어 올린다 —
# p90 242까지가 실제 범위고 우리 795는 그 밖이다.
MATERIAL_PRIOR: dict[str, tuple] = {
    "big_n": (5.1, 6.4, 9.25, 12.25, 15.5, 20.8, 22.3),
    "big_cols": (1.35, 1.7, 3, 4.75, 6, 7.6, 8.3),
    "hero_share": (0.2652, 0.2892, 0.3615, 0.4436, 0.5247, 0.6446, 0.6977),
    "objects": (0, 0, 0.375, 1.5, 2.25, 3.45, 3.975),
    "words": (0, 0.2, 1, 2, 3.875, 18.2, 26.1),
    "glyph_h": (0, 0.8192, 2.816, 7.04, 12.32, 21.63, 31.09),
    "skew": (0.0275, 0.0328, 0.1069, 0.2606, 0.4507, 0.6031, 0.7747),
    "mask": (0, 0, 0, 0.0164, 0.0389, 0.0875, 0.1397),
    "alpha_lt": (0, 0.0017, 0.0069, 0.0129, 0.0783, 0.1167, 0.2819),
    "colors": (15.25, 24.5, 37.25, 93.75, 175.4, 242.2, 799.9),
}

MATERIAL_W = {"big_n": 1.0, "big_cols": 0.8, "hero_share": 1.0, "objects": 0.6,
              "words": 0.8, "glyph_h": 0.4, "skew": 0.5, "mask": 0.4,
              "alpha_lt": 0.4, "colors": 0.8}

# 꼬리가 긴 특징은 **로그 자**로 문는다 — 색 수는 작가 하나가 2,400색이라 선형
# 자로는 그 작가를 뺀 사전에서 벌점이 3을 넘어 무리 점수를 혼자 정한다 (실측:
# 사람 판의 colors 평균 벌점 3.21 · alpha_lt 1.91). 개수·몫 자체는 사전에 그대로
# 두고 잴 때만 log1p를 씌운다. 벌점 상한도 둔다 — 특징 하나가 무리를 못 삼키게.
MATERIAL_LOG = frozenset(("colors", "words", "glyph_h", "big_n", "objects",
                          "alpha_lt", "mask"))
MATERIAL_PEN_CAP = 2.0


def material_features(sec: dict) -> dict:
    """면별 재료 통계 → 옆면 특징 한 벌 (좌우는 평균으로 접는다).

    `sec`는 면 이름 → `ruler.surface_stats`(또는 `material_stats`)의 dict다.
    옆면이 하나도 없으면 빈 dict — 그러면 이 무리는 안 선다.
    """
    sides = [sec[k] for k in ("side_left", "side_right")
             if k in sec and sec[k]]
    out: dict[str, float] = {}
    for k in MATERIAL_FEATURES:
        vals = [float(s[k]) for s in sides if k in s and s[k] is not None]
        if vals:
            out[k] = sum(vals) / len(vals)
    return out


def material_penalties(mat: dict, prior: dict | None = None) -> dict:
    pr = prior if prior is not None else MATERIAL_PRIOR
    out = {}
    for k in MATERIAL_FEATURES:
        if k not in mat or not pr.get(k):
            continue
        v, q = float(mat[k]), tuple(float(x) for x in pr[k])
        if k in MATERIAL_LOG:
            # 몫(0~1)은 100배 해서 개수처럼 — log1p(0.01)은 선형과 같아 뜻이 없다
            s = 100.0 if q[-1] <= 1.0 else 1.0
            v = math.log1p(s * max(v, 0.0))
            q = tuple(math.log1p(s * max(x, 0.0)) for x in q)
        out[k] = min(penalty(v, q), MATERIAL_PEN_CAP)
    return out


def material_score(mat: dict, prior: dict | None = None) -> float | None:
    """재료 무리 점수 0~1 — 잰 특징이 하나도 없으면 None (무리가 안 선다)."""
    pen = material_penalties(mat, prior)
    if not pen:
        return None
    wsum = sum(MATERIAL_W.get(k, 0.5) for k in pen)
    return float(math.exp(-sum(MATERIAL_W.get(k, 0.5) * v
                               for k, v in pen.items()) / max(wsum, 1e-9)))


# ── 무게 몫 → 위계 특징 ──────────────────────────────────────────────


def fold(weights: dict) -> dict:
    """거울 짝을 한 벌로 접은 무게 — {단위 이름: 무게}."""
    out: dict[str, float] = {}
    for name in sorted(weights):
        k = MIRROR.get(name, name)
        out[k] = out.get(k, 0.0) + float(weights[name])
    return out


def shares(weights: dict) -> dict | None:
    """접은 단위마다의 **무게 몫** (합 1). 무게가 0이면 None."""
    f = fold(weights)
    tot = sum(f.values())
    if tot <= 0.0:
        return None
    return {k: v / tot for k, v in f.items()}


def features(weights: dict, counts: dict | None = None) -> dict | None:
    """위계 특징 한 벌 — 무게가 하나도 없으면 None.

    `weights`는 면 이름 → 시각 무게(`ruler.visual_weight`)이고, `counts`는 면
    이름 → 레이어 수다 (`decorated`를 세는 데만 쓴다).
    """
    sh = shares(weights)
    if sh is None:
        return None
    p = sorted(sh.values(), reverse=True)
    n = len(p)
    top1 = p[0]
    top2 = sum(p[:2])
    top3 = sum(p[:3])
    hhi = sum(v * v for v in p)
    ent = (-sum(v * math.log(max(v, 1e-12)) for v in p) / math.log(n)
           if n > 1 else 0.0)
    p2 = p[1] if n > 1 else 0.0
    support = max(0.0, 1.0 - top2)
    dec = 0
    if counts:
        agg: dict[str, int] = {}
        for name in sorted(counts):
            k = MIRROR.get(name, name)
            agg[k] = agg.get(k, 0) + int(counts[name])
        dec = sum(1 for v in agg.values() if v >= DECORATED_MIN)
    return {"units": float(n), "decorated": float(dec),
            "top1": float(top1), "top2": float(top2), "top3": float(top3),
            "rem": float(max(0.0, 1.0 - top3)), "hhi": float(hhi),
            "ent": float(ent), "ps": float(p2 / max(top1, 1e-9)),
            "ss": float(support / max(p2, 1e-6))}


# ── 사람 범위 사전 ───────────────────────────────────────────────────


def penalty(value: float, q: tuple) -> float:
    """백분위 표 `q` 안에서 이 값이 얼마나 밖인가 (0 = 사람 범위 안).

    p10~p90은 0, p5~p10과 p90~p95는 0→1, 그 밖은 `OUT_SLOPE` 기울기로 는다.
    **범위 안에 든 것 자체는 상을 주지 않는다** (§2) — 밖으로 나간 것만 문다.
    """
    p5, p10, _p25, _p50, _p75, p90, p95 = q
    span = max(float(p95) - float(p5), 1e-9)
    if p10 <= value <= p90:
        return 0.0
    if value < p10:
        if value >= p5:
            return float((p10 - value) / max(p10 - p5, 1e-9))
        return float(1.0 + OUT_SLOPE * (p5 - value) / span)
    if value <= p95:
        return float((value - p90) / max(p95 - p90, 1e-9))
    return float(1.0 + OUT_SLOPE * (value - p95) / span)


def penalties(feats: dict, prior: dict | None = None) -> dict:
    """특징마다의 벌점."""
    pr = prior or HUMAN_PRIOR
    return {k: penalty(float(feats[k]), pr[k])
            for k in FEATURES if k in feats and k in pr}


def group_scores(feats: dict, prior: dict | None = None) -> dict:
    """무리마다의 점수 0~1 (`GROUPS`) — 1이면 그 무리가 전부 사람 범위 안."""
    pen = penalties(feats, prior)
    out = {}
    for g, keys in GROUPS.items():
        got = [(FEATURE_W.get(k, 0.5), pen[k]) for k in keys if k in pen]
        if not got:
            continue
        wsum = sum(w for w, _ in got)
        out[g] = float(math.exp(-sum(w * v for w, v in got) / max(wsum, 1e-9)))
    return out


def hierarchy_score(feats: dict, prior: dict | None = None) -> float:
    """위계 점수 0~1 — **나쁜 무리가 그대로 점수다** (`GROUPS` 문서).

    무리 안에서는 가중평균이고 무리 사이에서는 최솟값이라, 면을 다 채운 판이
    무게 쏠림을 못 갚고 무게가 고운 판이 빈 면을 못 갚는다 (§1의 상쇄 금지).
    """
    gs = group_scores(feats, prior)
    return float(min(gs.values())) if gs else 0.0


# ── 시각 위계 등급 (§3) ──────────────────────────────────────────────
#
# 면 **이름으로 고정하지 않는다** — 윗면이 주역일 수도 옆면이 주역일 수도
# 있다. 정하는 것은 접은 무게 몫뿐이고, 자는 제일 큰 단위 대비 비다.
TIERS = ("PRIMARY", "SECONDARY", "SUPPORT", "MICRO")
TIER_PRIMARY = 0.55        # 최대 몫의 이만큼 이상이면 주역
TIER_SECOND = 0.22         # 그 아래로 이만큼 이상이면 조연
TIER_SUPPORT = 0.04        # 그 아래로 이만큼 이상이면 받침, 아니면 잔것


def tiers(sh: dict) -> dict:
    """접은 단위 → 등급. `sh`는 `shares`가 낸 몫이다."""
    if not sh:
        return {}
    top = max(sh.values()) or 1e-9
    out = {}
    for k in sorted(sh):
        r = sh[k] / top
        out[k] = ("PRIMARY" if r >= TIER_PRIMARY else
                  "SECONDARY" if r >= TIER_SECOND else
                  "SUPPORT" if sh[k] >= TIER_SUPPORT else "MICRO")
    return out


def tier_mass(sh: dict) -> dict:
    """등급마다의 무게 합."""
    t = tiers(sh)
    out = dict.fromkeys(TIERS, 0.0)
    for k, v in sh.items():
        out[t[k]] += float(v)
    return out


def tier_faults(sh: dict) -> tuple:
    """§3의 위계 규칙을 어긴 자리 — 빈 튜플이면 지켜졌다.

    * 주역이 하나는 있어야 한다.
    * 받침이 주역보다 세면 안 된다.
    * 작은 것들이 **합쳐서** 주역을 압도하면 안 된다.
    """
    if not sh:
        return ("no_mass",)
    t = tiers(sh)
    m = tier_mass(sh)
    bad = []
    if not any(v == "PRIMARY" for v in t.values()):
        bad.append("no_primary")
    if m["SUPPORT"] > m["PRIMARY"]:
        bad.append("support_over_primary")
    if m["SUPPORT"] + m["MICRO"] > m["PRIMARY"]:
        bad.append("small_over_primary")
    return tuple(bad)


# ── ① 성립 판정 ─────────────────────────────────────────────────────


def validate(counts: dict, caps: dict, *, extra: tuple = ()) -> tuple:
    """하나라도 걸리면 후보가 아니다 — 걸린 이름들.

    `counts`는 면별 장수, `caps`는 면별 상한이다. `extra`는 호출부가 이미 잰
    치명 조건(직렬화 실패·심한 잘림·셀 구멍 등)을 그대로 얹는 자리다 —
    자를 두 벌 세우지 않으려고 여기서 다시 재지 않는다.
    """
    bad = list(extra)
    for name in sorted(counts):
        n = int(counts[name])
        if n < 0:
            bad.append("negative:%s" % name)
        cap = int(caps.get(name, 0) or 0)
        if cap and n > cap:
            bad.append("cap:%s" % name)
    return tuple(bad)


# ── ② 도안 충실도 ───────────────────────────────────────────────────
#
# **꾸밈 점수로 못 갚는다** — 바닥을 못 넘기면 위계가 아무리 고와도 후보가
# 아니다. 값은 호출부가 이미 재 둔 것을 받는다 (cel 노선 보고서·구도 자):
# 자를 여기서 새로 세우면 두 벌이 된다.
#
# 부호는 자마다 다르므로 방향을 함께 준다: `("min", 바닥)`이면 그 위여야 하고
# `("max", 천장)`이면 그 아래여야 한다.
FIDELITY_FLOOR = {
    "face": ("max", 0.06),            # 얼굴 보호 구역이 덮인 몫 (`score`의 자)
    "readability": ("min", 0.16),     # 실루엣 테두리 명도차 (`score.READ_FLOOR`)
    # 실루엣 커버리지 — 봉인이 1로 만드는 값이라 바닥은 그 바로 아래다
    # (실측 표준 11장: 0.9993~0.9999).
    "coverage": ("min", 0.995),
    "imp_error_seen": ("max", 0.40),  # 보이는 색 오차 (`celfit.metrics`)
    "wrong_far_rate": ("max", 0.01),  # 크게 틀린 픽셀 몫
    "hole_left": ("max", 0.0),        # 남은 구멍 군집
}


def fidelity(readings: dict, floors: dict | None = None) -> tuple:
    """바닥을 못 넘긴 자 이름들 — 빈 튜플이면 통과.

    `readings`에 없는 자는 **안 묻는다** (재지 않은 것을 벌하지 않는다).
    """
    fl = floors or FIDELITY_FLOOR
    bad = []
    for k in sorted(fl):
        if k not in readings or readings[k] is None:
            continue
        how, lim = fl[k]
        v = float(readings[k])
        if (how == "min" and v < lim) or (how == "max" and v > lim):
            bad.append(k)
    return tuple(bad)


# ── 한 벌로 ─────────────────────────────────────────────────────────


@dataclass
class CarScore:
    """차 한 대의 층별 판정."""

    valid: bool = True
    violations: tuple = ()            # ① 성립 (하나라도 있으면 후보가 아니다)
    fidelity_fails: tuple = ()        # ② 도안 충실도
    faults: tuple = ()                # ③ 위계 규칙 (§3)
    hierarchy: float = 0.0            # ③ 점수 0~1 (나쁜 무리)
    group: dict = field(default_factory=dict)   # ③ 무리별 점수 (`GROUPS`)
    feats: dict = field(default_factory=dict)
    pen: dict = field(default_factory=dict)
    tier: dict = field(default_factory=dict)
    material: dict = field(default_factory=dict)   # 재료 특징 (잰 판만)
    mpen: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.violations and not self.fidelity_fails

    def text(self) -> str:
        t = "H=%.3f" % self.hierarchy
        if self.group:
            t += " (" + " ".join("%s %.2f" % (k[:3], v)
                                 for k, v in sorted(self.group.items())) + ")"
        if self.violations:
            t += " [X %s]" % "/".join(self.violations)
        if self.fidelity_fails:
            t += " [F %s]" % "/".join(self.fidelity_fails)
        if self.faults:
            t += " [h %s]" % "/".join(self.faults)
        return t


def evaluate(weights: dict, counts: dict, *, caps: dict | None = None,
             readings: dict | None = None, prior: dict | None = None,
             extra: tuple = (), material: dict | None = None,
             material_prior: dict | None = None) -> CarScore:
    """세 층을 한 번에 — `weights`는 면별 시각 무게, `counts`는 면별 장수.

    `material`은 `material_features`가 낸 옆면 재료 특징이다. 주면 ③에
    "material" 무리가 하나 더 서고(나쁜 무리가 점수), 안 주면 종전과 같다.
    """
    sc = CarScore()
    sc.violations = validate(counts, caps or {}, extra=extra)
    sc.fidelity_fails = fidelity(readings or {})
    ft = features(weights, counts)
    if ft is None:
        sc.violations = sc.violations + ("no_mass",)
        sc.valid = False
        return sc
    sh = shares(weights) or {}
    sc.feats = ft
    sc.pen = penalties(ft, prior)
    sc.group = group_scores(ft, prior)
    if material:
        ms = material_score(material, material_prior)
        if ms is not None:
            sc.material = dict(material)
            sc.mpen = material_penalties(material, material_prior)
            sc.group["material"] = ms
    sc.hierarchy = float(min(sc.group.values())) if sc.group else 0.0
    sc.tier = tiers(sh)
    sc.faults = tier_faults(sh)
    sc.valid = sc.ok
    return sc
