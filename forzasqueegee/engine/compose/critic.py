"""구도 비평 — **여러 배율로** 재고 왜 그런지 남긴다.

`score`는 후보 하나를 한 배율(필드 격자)에서 재는 자였다. 그것으로 33판을
재 보니 열한 항목 중 넷이 **모든 판에서 정확히 1.000**이고(`face`·`cohesion`·
`negative`·`orphan`), 판마다 값이 갈리는 것은 `readability` 하나뿐이었다
(std 0.105, 나머지는 0.00~0.09). 1위와 2위의 점수 차 중앙값이 **0.0000**이다 —
후보를 실제로 고르는 것은 점수가 아니라 후보를 짓는 순서였다.

그 결과가 눈에 보이는 증상이다: 유일하게 값이 갈리는 자가 "실루엣 테두리
안팎의 명도차"라서, 그 자를 최대로 만드는 길은 **인물 뒤에 반대 명도의 큰
판을 통째로 까는 것**이다. 33판의 판 넓이는 인물 넓이의 5.8배(중앙값)·최대
17.2배였고, 가장 무거운 덩어리 하나가 시각 무게의 78%를 쥐었다 (둘째는 첫째의
19%). 사람이 만든 리버리는 큰 덩어리 둘 셋이 무게를 나눠 쥔다.

## 이 모듈이 하는 일

합성 그림 하나에서 **세 배율**을 만들고(멀리·중간·가까이) 배율마다 다른 것을
묻는다.

    far   누가 먼저 읽히나 — 인물인가 꾸밈인가 (`focal`)
    mid   큰 덩어리가 몇이고 무게가 어떻게 갈리나 (`macro`)
    near  조각의 리듬 · 여백의 꼴 (`rhythm` · `negative_shape`)

배율은 격자를 다시 렌더하지 않고 **흐림 반경**으로 낸다 — 같은 래스터를 세 번
흐리는 값은 다시 렌더하는 것과 같은 정보를 주고 (후보당 수백 번 부르는 자리라
렌더를 늘릴 수 없다) 결정적이다.

결정성: cv2의 흐림·연결성분과 `boxes.major_axis`의 닫힌 식만 쓴다 — LAPACK도
BLAS 행렬곱도 안 거친다 (`determinism-traps`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from .boxes import major_axis


# 배율의 흐림 반경 (프레임 폭의 몫). 멀리 = 차 한 대를 한눈에 보는 거리에서
# 남는 것, 중간 = 패널 하나를 보는 거리, 가까이 = 그대로.
FAR_FRAC = 0.055
MID_FRAC = 0.018


# 덩어리로 세는 최소 넓이 (그릴 수 있는 면 대비). 이보다 잔 것은 잔티끌이다.
BLOB_MIN = 0.0016


# 큰 덩어리의 **목표 구조** — 사람이 만든 리버리의 범위다.
#
# 근거 둘. 하나, 레퍼런스 사진에서 밴드만 잘라 잰 값 (`work/lab/deco/refstat.py`
# — 히나타 판이 인물·반사 오염이 가장 적다: hero 0.58 · h2 0.44 · 덩어리 20).
# 둘, 도안 33판 실측의 **반대쪽**: 지금은 hero 0.78 · h2 0.19 · 덩어리 3이다.
# 구간은 그 사이를 넉넉히 잡았다 — 자가 목표를 못 박으면 후보가 그 값 하나로
# 몰려 다시 한 가지 그림이 된다.
# 구간은 실측으로 두 번 조였다. 첫 판 (0.30, 0.62)/(0.30, 0.90)/(2, 5)는 매크로
# 어휘를 갈아 넣고 나니 33판 중 열일곱이 0.987 위로 몰려 자가 아니었고, 그것을
# (0.28, 0.52)로 좁혔더니 이번엔 주 색면이 너무 잘게 갈려 덩어리 무게가 0.42로
# 내려앉았다 (레퍼런스 0.58). 지금 값은 그 둘 사이다 — **주역 하나가 확실히
# 무겁고 조연이 그 절반쯤**인 사람 리버리의 꼴이다.
MACRO_W1 = (0.36, 0.60)          # 가장 무거운 덩어리가 쥐는 몫
MACRO_W2 = (0.38, 0.92)          # 둘째 / 첫째
MACRO_N = (3, 7)                 # 큰 덩어리 수 (잔티끌 뺀)


# 가장 큰 꾸밈 덩어리가 **인물 넓이의** 몇 배까지인가. 이 위면 판이 아니라
# 두 번째 베이스 도색이다 (실측 중앙값 5.8 · 최대 17.2가 지금 자리다).
MACRO_SPAN = (0.45, 3.2)


# **꾸밈이 실제로 있나** — 그릴 수 있는 면(인물 밖) 중 꾸밈이 덮은 몫의 목표.
#
# 계열마다 다른 `clutter` 목표(minimal은 0.03~0.12)는 계열을 **고르는** 데는
# 못 쓴다: 목표가 느슨한 계열이 언제나 만점을 받아 이긴다. 실제로 매크로 어휘를
# 갈아 넣은 뒤 minimal이 여덟 판을 가져가면서 꾸밈 잉크 중앙값이 0.44 → 0.18로
# 내려앉았고, 그중 셋은 큰 색면 하나에 조각 둘이 전부였다. 계열과 무관한
# 자가 하나 있어야 "디자인이 있긴 한가"를 물을 수 있다.
#
# 구간의 위 끝은 옛 판의 실측(0.44)이 이미 과했다는 판정에서, 아래 끝은
# 레퍼런스 밴드 크롭에서 인물을 뺀 어림(히나타 0.37)에서 왔다.
#
# 위 끝은 한 번 늘렸다. (0.18, 0.40)·부드러움 0.16으로 두니 0.56에서 0으로
# 떨어지는데, 큰 색면을 쓰는 계열은 여기 걸려 **한 판도 못 이겼다** (33판 중
# graphic_bed 1 · diagonal_flow 0). 가중치 1.4짜리 항목이 0이 되는 것은 총점
# 0.074고, 그것만으로 순위가 뒤집혔다. 막으려던 것은 "판을 통째로 덮은 것"
# (옛 실측 0.7~0.8)이지 큰 색면 자체가 아니다.
PRESENCE = (0.18, 0.46)


# **길이 방향 쏠림** — 다섯 토막 커버리지의 최대 − 최소.
#
# 사람 리버리와 견줄 수 있는 몇 안 되는 자다 (`work/lab/deco/bench.py`): 사진
# 에서 덩어리 구조는 음영 때문에 못 재지만 "어느 토막이 차 있나"는 재진다.
# 실측 넷 — ARIS [1.0 .68 .62 .48 .50] (앞이 무겁다) · 코토네 [.67 .69 .93 .99
# .94] (뒤가 무겁다) · 히나타 [.38 .27 .30 .57 .33] (넷째 토막에 봉우리) · 린
# [.78 .80 .76 .87 .85] (고르다). 기복이 0.14~0.52다.
#
# 우리 판은 0.028이었다 — **차 길이를 따라 완벽하게 고르다**. 관통하는 띠가
# 모든 토막을 같은 몫으로 덮고 무리가 그 위에 얇게 얹히기 때문이다. 자동
# 생성 티가 숫자 하나로 나온 자리이고, 이 자가 그것을 민다: 한쪽 끝에서
# 끝나는 색면(split·corner·가늘어지는 날)과 한쪽에 몰린 무리가 이긴다.
LENGTHWISE = (0.12, 0.55)


# 여백 한 덩이의 목표 — 그릴 수 있는 면 대비 넓이와 **꼴**(상자 채움).
# 얇고 긴 자투리는 여백이 아니라 남은 자리다.
# (구간은 실측으로 조인다: 첫 판의 (0.10, 0.42)·채움 0.42는 33판 전부가
# 1.000이라 자가 아니었다 — 지금 생성기가 닿는 범위의 **위쪽**을 목표로 둔다.)
NEG_AREA = (0.20, 0.45)
NEG_FILL = 0.62


# **경쟁자의 자격.** 인물을 받치는 덩어리는 무대지 경쟁자가 아니다 — 인물
# 실루엣의 이 몫 이상을 (조금 부풀려) 감싸면 무대로 본다. 크기로 가르려 했더니
# 안 갈렸다 (짙은 판이 인물의 2배라 크기 구간 안에 들어왔다): 무대와 경쟁자를
# 가르는 것은 크기가 아니라 **인물과 겹치나**다.
COMPETE_BACKING_MAX = 0.25


# 경쟁자로 세는 최소 크기 (인물 넓이 대비) — 이보다 잔 것은 잔 조각이다.
COMPETE_MIN = 0.08


# **프레임을 건너는 것은 바탕이다.** 로커 띠·관통 판처럼 상자가 프레임의 이
# 몫 이상을 가로지르는 덩어리는 차의 구조로 읽히지 주역과 겨루지 않는다
# (레퍼런스의 검은 로커가 인물을 이기지 않는다). 인물을 안 받치는 것만으로
# 경쟁자를 가르면 흰 차의 검은 로커가 끌림 0.77로 주역을 뺏은 것이 된다.
GROUND_SPAN = 0.85


# 인물이 멀리서 가져야 할 **절대 끌림** 구간 (명도차). 아래면 판에 녹은 것이고,
# 위(0.6 넘게)면 판이 인물만 남기고 아무것도 안 하는 것이다.
HERO_PULL = (0.14, 0.60)


# 주역을 **잃은** 것으로 보는 자 — 멀리서 인물이 아예 안 떨어져 나온다.
#
# 탈락 조건은 이 하나뿐이다. 경쟁 비(`focal_ratio`)로도 떨어뜨려 봤더니 33판 중
# 열아홉이 걸렸고, 전멸한 판에서는 순위가 다시 총점으로 정해져 자가 아무것도
# 안 하면서 이름만 남았다. 겨룸은 점수(`focal`)로 밀고, 탈락은 "안 보인다"에만
# 쓴다.
HERO_PULL_FLOOR = 0.065


@dataclass
class Critique:
    """비평 한 벌 — 항목 점수와 그 근거 수치."""

    parts: dict[str, float] = field(default_factory=dict)
    info: dict[str, float] = field(default_factory=dict)
    fails: tuple[str, ...] = ()
    # 가장 큰 빈 덩이의 상자 (프레임 좌표) — 구성 그래프의 `negative` 노드가
    # 이걸 쓴다 (`graph.derive`). 없으면 None.
    neg_box: tuple[float, float, float, float] | None = None


def _lum(img: np.ndarray) -> np.ndarray:
    f = img.astype(np.float32)
    return (0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]) / 255.0


def _odd(n: float) -> int:
    return max(1, int(n) | 1)


def scales(lum: np.ndarray, cols: int) -> tuple[np.ndarray, np.ndarray]:
    """(far, mid) — 같은 래스터를 두 반경으로 흐린 것. near는 원본이다."""
    kf, km = _odd(FAR_FRAC * cols), _odd(MID_FRAC * cols)
    return cv2.blur(lum, (kf, kf)), cv2.blur(lum, (km, km))


def _pull(lum: np.ndarray, mask: np.ndarray, k: int,
          valid: np.ndarray | None = None) -> float:
    """`mask` 안과 **떨어진 둘레**의 명도차 — "이 덩어리가 얼마나 끌리나".

    둘레는 `k`칸 **밖에서 시작하는** 고리다. 바로 붙은 한 칸을 쓰면 안 된다:
    `lum`이 이미 반경 `k`로 흐려진 그림이라 경계 양쪽이 서로 섞여 있어서 차가
    구조적으로 0에 가깝게 나온다 (실측 silvia-01: 짙은 판 위의 밝은 인물인데도
    끌림이 0.029로 나와 주역이 뒤집힌 것으로 오판했다). 흐림 반경 밖으로 나가야
    "멀리서 이 덩어리가 둘레와 다른 톤인가"를 재는 것이 된다.

    `valid`(그릴 수 있는 자리)를 주면 면 밖은 안 센다 — 패널 위쪽 비도색면이
    둘레에 섞이면 어느 덩어리든 대비가 세게 나온다.
    """
    m = mask.astype(np.uint8)
    if not m.any():
        return 0.0
    ker = np.ones((2 * k + 1, 2 * k + 1), np.uint8)
    near = cv2.dilate(m, ker).astype(bool)
    far_ = cv2.dilate(m, ker, iterations=3).astype(bool)
    ring = far_ & ~near
    if valid is not None:
        ring = ring & valid
        inner = mask & valid
    else:
        inner = mask
    if not ring.any() or not inner.any():
        return 0.0
    return float(abs(lum[inner].mean() - lum[ring].mean()))


def _band(v: float, lo: float, hi: float, soft: float) -> float:
    if v < lo:
        return max(0.0, 1.0 - (lo - v) / max(1e-6, soft))
    if v > hi:
        return max(0.0, 1.0 - (v - hi) / max(1e-6, soft))
    return 1.0


def blobs(alpha: np.ndarray, lum: np.ndarray, base_lum: float, room: np.ndarray
          ) -> list[dict]:
    """꾸밈 덩어리 목록 (무게 큰 것부터) — 넓이 × 바탕 명도차.

    `score._blob_weights`와 같은 자인데 **명도**로 재고 상자·중심까지 낸다
    (배치 관계를 물어야 하므로). Lab 거리로 재던 옛 자는 색이 갈리되 명도가
    같은 판을 무겁게 봤다 — 멀리서는 그런 판이 안 보인다.
    """
    m = ((alpha > 0.5) & room).astype(np.uint8)
    if not m.any():
        return []
    n, lbl, st, cen = cv2.connectedComponentsWithStats(m, 8)
    tot = float(room.sum()) or 1.0
    out: list[dict] = []
    for i in range(1, n):
        a = float(st[i, cv2.CC_STAT_AREA])
        if a < BLOB_MIN * tot:
            continue
        sel = lbl == i
        dl = abs(float(lum[sel].mean()) - base_lum)
        out.append({
            "area": a / tot, "weight": a / tot * min(1.0, dl / 0.34),
            "cx": float(cen[i][0]), "cy": float(cen[i][1]), "dl": dl,
            "x0": float(st[i, cv2.CC_STAT_LEFT]), "y0": float(st[i, cv2.CC_STAT_TOP]),
            "w": float(st[i, cv2.CC_STAT_WIDTH]), "h": float(st[i, cv2.CC_STAT_HEIGHT]),
            "sel": sel})
    out.sort(key=lambda b: (-b["weight"], b["cx"], b["cy"]))
    return out


def focal(far: np.ndarray, sil: np.ndarray, bl: list[dict], k: int,
          valid: np.ndarray | None = None) -> tuple[float, dict]:
    """**멀리서 누가 먼저 읽히나** — 인물의 끌림 대 가장 센 꾸밈 덩어리의 끌림.

    사람이 만든 이타샤는 차에서 멀어질수록 인물만 남는다. 자동 생성물은 인물
    뒤에 깐 큰 판이 먼저 읽혀서, 멀리서 보면 "판 위에 뭔가 있는" 그림이 된다.
    """
    hero = _pull(far, sil, k, valid)
    # **바탕은 경쟁자가 아니다.** 큰 색면이 세게 읽히는 것은 잘못이 아니라
    # 문법이다 (레퍼런스의 흰 차 위 검은 판). 주역을 뺏는 것은 인물과 **비슷한
    # 크기**의 요소다 — 판을 경쟁자로 세면 짙은 판을 깐 판이 전부 "주역 상실"로
    # 떨어진다 (실측 silvia-01: 흰 차 위 남색 판의 끌림 0.768 대 인물 0.140 —
    # 레퍼런스 그대로인 구도가 탈락했다).
    ha = float(sil.sum()) or 1.0
    ker = np.ones((2 * k + 1, 2 * k + 1), np.uint8)
    top = 0.0
    rows, cols = sil.shape
    for b in bl:
        if float(b["sel"].sum()) < COMPETE_MIN * ha:
            continue
        if max(b["w"] / max(1, cols), b["h"] / max(1, rows)) >= GROUND_SPAN:
            continue                              # 프레임을 건넌다 — 바탕이다
        grown = cv2.dilate(b["sel"].astype(np.uint8), ker).astype(bool)
        if float((grown & sil).sum()) / ha >= COMPETE_BACKING_MAX:
            continue                              # 인물을 받친다 — 무대다
        top = max(top, _pull(far, b["sel"], k, valid))
    ratio = hero / max(1e-6, hero + top) if (hero + top) > 1e-6 else 0.5
    # 두 가지를 같이 본다: 인물이 **얼마나** 읽히나(절대) · 비슷한 크기의 것에
    # 안 밀리나(상대).
    return (0.55 * _band(hero, *HERO_PULL, soft=0.16)
            + 0.45 * _band(ratio, 0.50, 1.0, 0.30)), {
        "hero_pull": hero, "deco_pull": top, "focal_ratio": ratio}


def macro(bl: list[dict], sil_area: float, room_area: float) -> tuple[float, dict]:
    """큰 덩어리의 **무게 배분** — 주역 하나 · 조연 하나 · 잔것 몇.

    지금 판의 병목이 여기다: 판 하나가 78%를 쥐고 둘째가 그 19%뿐이라 "큰 판
    위에 잔 조각"이라는 한 가지 구조밖에 안 나온다.
    """
    if not bl:
        return 0.25, {"m_w1": 0.0, "m_w2": 0.0, "m_n": 0, "m_span": 0.0}
    ws = [b["weight"] for b in bl]
    tot = sum(ws) or 1e-9
    w1 = ws[0] / tot
    w2 = (ws[1] / ws[0]) if len(ws) > 1 else 0.0
    span = bl[0]["area"] * room_area / max(1.0, sil_area)
    n = len(ws)
    sc = (0.34 * _band(w1, *MACRO_W1, soft=0.28)
          + 0.30 * _band(w2, *MACRO_W2, soft=0.30)
          + 0.20 * _band(span, *MACRO_SPAN, soft=2.4)
          + 0.16 * _band(float(n), *MACRO_N, soft=2.0))
    return sc, {"m_w1": w1, "m_w2": w2, "m_n": float(n), "m_span": span}


def _tau(a: list[float], b: list[float]) -> float:
    """켄달 타우 — 두 수열이 같은 방향으로 가나 (−1~1). 표본이 작아 O(n²)로 센다."""
    n = len(a)
    if n < 3:
        return 0.0
    con = dis = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = (a[i] - a[j]) * (b[i] - b[j])
            if s > 0:
                con += 1
            elif s < 0:
                dis += 1
    tot = con + dis
    return (con - dis) / tot if tot else 0.0


def rhythm(motifs: list[tuple[float, float, float, int]]) -> tuple[float, dict]:
    """조각이 **하나의 리듬**으로 읽히나 — 크기가 자리를 따라 자라거나 잦아드나.

    사람이 그린 무리는 큰 것 → 중간 → 잔것이 한 방향으로 흐른다. 황금각 산포는
    자리는 고르되 크기가 자리와 무관해서 "뿌린 것"으로 읽힌다.

    세 가지를 본다: 축을 따른 **크기 단조성**(켄달 타우) · 간격이 매끄럽게
    변하나(같은 간격이면 기계, 제멋대로면 뿌린 것) · 크기 층이 셋 이상인가.
    """
    if len(motifs) < 3:
        return 0.5, {"r_tau": 0.0, "r_gap_cv": 0.0, "r_tiers": float(len(motifs))}
    xs = np.array([m[0] for m in motifs], np.float64)
    ys = np.array([m[1] for m in motifs], np.float64)
    d, _e = major_axis(xs, ys)
    t = (xs - xs.mean()) * d[0] + (ys - ys.mean()) * d[1]
    order = np.argsort(t, kind="stable")
    pos = [float(t[i]) for i in order]
    size = [float(motifs[i][2]) for i in order]
    tau = abs(_tau(pos, size))
    gaps = [pos[i + 1] - pos[i] for i in range(len(pos) - 1)]
    gm = float(np.mean(gaps)) if gaps else 0.0
    if gm > 1e-6 and len(gaps) >= 3:
        cv = float(np.std(gaps) / gm)
        # 0.25~0.85가 목표: 0은 자로 잰 등간격(기계), 1.2 위는 뭉텅이와 빈 곳
        gap = _band(cv, 0.25, 0.85, 0.45)
    else:
        gap = 0.5
    tiers = len({m[3] for m in motifs})
    return (0.50 * tau + 0.30 * gap + 0.20 * min(1.0, (tiers - 1) / 2.0),
            {"r_tau": tau, "r_gap_cv": (float(np.std(gaps) / gm) if gm > 1e-6 else 0.0),
             "r_tiers": float(tiers)})


def presence(deco_alpha: np.ndarray, room: np.ndarray) -> tuple[float, dict]:
    """꾸밈이 인물 밖 도색면을 덮은 몫 — 너무 비었나 · 너무 덮었나."""
    if not room.any():
        return 0.5, {"p_ink": 0.0}
    v = float((deco_alpha[room] > 0.5).mean())
    return _band(v, *PRESENCE, soft=0.24), {"p_ink": v}


def lengthwise(deco_alpha: np.ndarray, room: np.ndarray) -> tuple[float, dict]:
    """길이를 다섯 토막으로 나눈 커버리지의 **기복** — 한쪽에 몰렸나."""
    if not room.any():
        return 0.5, {"l_span": 0.0}
    w = room.shape[1]
    f5 = []
    for i in range(5):
        sl = slice(i * w // 5, (i + 1) * w // 5)
        r5 = room[:, sl]
        f5.append(float((deco_alpha[:, sl][r5] > 0.5).mean()) if r5.any() else 0.0)
    span = max(f5) - min(f5)
    return _band(span, *LENGTHWISE, soft=0.14), {"l_span": span}


def negative_shape(ink: np.ndarray, room: np.ndarray, head_c, face_dir: float,
                   cell: float, char_w: float, x0: float = 0.0, y_top: float = 0.0
                   ) -> tuple[float, dict, tuple[float, float, float, float] | None]:
    """여백이 **꼴을 가졌나** — 가장 큰 빈 덩이의 넓이와 채움, 그리고 시선 앞.

    지금 자(`score`의 `negative`)는 "여백 구역에 모티프가 없나"만 물어서 33판이
    전부 1.000이었다 — 아무것도 안 놓으면 만점이라 자가 아니다. 여백은 **일부러
    비운 하나의 꼴**이라야 구도의 일부가 된다.
    """
    empty = (room & ~(ink > 0.5)).astype(np.uint8)
    if not empty.any():
        return 0.0, {"n_area": 0.0, "n_fill": 0.0, "n_gaze": 0.0}, None
    n, lbl, st, cen = cv2.connectedComponentsWithStats(empty, 8)
    tot = float(room.sum()) or 1.0
    best, ba = None, 0.0
    for i in range(1, n):
        a = float(st[i, cv2.CC_STAT_AREA])
        if a > ba:
            best, ba = i, a
    area = ba / tot
    w = float(st[best, cv2.CC_STAT_WIDTH])
    h = float(st[best, cv2.CC_STAT_HEIGHT])
    fill = ba / max(1.0, w * h)
    bx = float(st[best, cv2.CC_STAT_LEFT])
    by = float(st[best, cv2.CC_STAT_TOP])
    box = (x0 + bx * cell, y_top - (by + h) * cell,
           x0 + (bx + w) * cell, y_top - by * cell)
    gaze = 0.0
    if head_c is not None and abs(face_dir) > 0.15:
        # 시선 앞 한 칸 — 얼굴에서 얼굴이 보는 쪽으로 인물 폭의 0.6배 자리가
        # 비어 있나 (레퍼런스는 시선 앞을 비우거나 글자를 놓는다)
        gy = int(head_c[1])
        gx = int(head_c[0] + math.copysign(0.6 * char_w / cell, face_dir))
        H, W = empty.shape
        if 0 <= gx < W and 0 <= gy < H:
            gaze = float(empty[max(0, gy - 2):gy + 3, max(0, gx - 2):gx + 3].mean())
    sc = _band(area, *NEG_AREA, soft=0.22) * (0.55 + 0.45 * min(1.0, fill / NEG_FILL))
    return min(1.0, sc + 0.10 * gaze), {
        "n_area": area, "n_fill": fill, "n_gaze": gaze}, box


def gesture(bl: list[dict], vc: tuple[float, float], gestures,
            cell: float, x0: float, y_top: float) -> tuple[float, dict]:
    """꾸밈이 **인물의 몸짓에서 흘러나오나**.

    인물의 시각 중심에서 꾸밈 덩어리들의 무게중심으로 가는 벡터가 몸짓 방향
    (뻗은 팔·머리카락·무기)과 나란하면 장식이 인물과 한 몸으로 읽힌다. 아무
    방향도 아니면 옆에 놓인 별개의 물건이다.

    `gestures`는 (dx, dy, 세기) 목록 (프레임 좌표) — 없으면 중립 0.5.
    """
    if not bl or not gestures:
        return 0.5, {"g_cos": 0.0}
    tw = sum(b["weight"] for b in bl) or 1e-9
    cx = sum(b["weight"] * b["cx"] for b in bl) / tw
    cy = sum(b["weight"] * b["cy"] for b in bl) / tw
    px = x0 + (cx + 0.5) * cell
    py = y_top - (cy + 0.5) * cell
    dx, dy = px - vc[0], py - vc[1]
    n = math.hypot(dx, dy)
    if n < 1e-6:
        return 0.5, {"g_cos": 0.0}
    dx, dy = dx / n, dy / n
    best = 0.0
    for gx, gy, wgt in gestures:
        gn = math.hypot(gx, gy)
        if gn < 1e-6:
            continue
        best = max(best, wgt * (dx * gx + dy * gy) / gn)
    return max(0.0, min(1.0, 0.5 + 0.5 * best)), {"g_cos": best}


def heatmaps(*, img: np.ndarray, sil: np.ndarray, room: np.ndarray,
             ink: np.ndarray, deco_alpha: np.ndarray, base_lum: float,
             cols: int) -> dict[str, np.ndarray]:
    """비평이 **실제로 본 것**을 그림으로 (디버그 — `work/lab/deco/heat.py`).

    점수만 보고는 "왜 나쁜가"를 못 되짚는다. 여기서 내는 것은 판정에 쓴 바로
    그 배열이다: 멀리 본 명도 · 인물과 꾸밈의 끌림 · 큰 덩어리 · 빈 덩이.
    """
    lum = _lum(img)
    far, mid = scales(lum, cols)
    bl = blobs(deco_alpha, mid, base_lum, room)
    masses = np.zeros(lum.shape, np.float32)
    for i, b in enumerate(bl[:6]):
        masses[b["sel"]] = 1.0 - 0.14 * i
    empty = (room & ~(ink > 0.5)).astype(np.uint8)
    big = np.zeros(lum.shape, np.float32)
    if empty.any():
        n, lb, st, _c = cv2.connectedComponentsWithStats(empty, 8)
        k = max(range(1, n), key=lambda i: st[i, cv2.CC_STAT_AREA], default=0)
        if k:
            big[lb == k] = 1.0
    return {"far": far, "mid": mid, "near": lum, "masses": masses,
            "empty": big, "sil": sil.astype(np.float32),
            "room": room.astype(np.float32)}


def critique(*, img: np.ndarray, sil: np.ndarray, room: np.ndarray,
             ink: np.ndarray, deco_alpha: np.ndarray, base_lum: float,
             motifs: list[tuple[float, float, float, int]],
             cols: int, cell: float, x0: float, y_top: float,
             visual_center: tuple[float, float], head_c, face_dir: float,
             char_w: float, gestures=None) -> Critique:
    """합성 그림 한 장 → 배율별 항목 + 근거 수치 + 탈락 조건."""
    lum = _lum(img)
    far, mid = scales(lum, cols)
    k = max(1, _odd(FAR_FRAC * cols) // 2)
    bl = blobs(deco_alpha, mid, base_lum, room)
    ns, ni, neg_box = negative_shape(ink, room, head_c, face_dir, cell, char_w,
                                     x0=x0, y_top=y_top)
    parts: dict[str, float] = {}
    info: dict[str, float] = {}
    for name, (v, i) in (
            ("focal", focal(far, sil, bl, k, room | sil)),
            ("macro", macro(bl, float(sil.sum()), float(room.sum()))),
            ("rhythm", rhythm(motifs)),
            ("negative_shape", (ns, ni)),
            ("presence", presence(deco_alpha, room)),
            ("lengthwise", lengthwise(deco_alpha, room)),
            ("gesture", gesture(bl, visual_center, gestures, cell, x0, y_top))):
        parts[name] = float(v)
        info.update({k2: float(v2) for k2, v2 in i.items()})
    fails: list[str] = []
    # 주역 상실 — 꾸밈 덩어리가 인물보다 세게 읽히면 (비 = hero/(hero+deco)가
    # HERO_PULL_MIN 배에 해당하는 지점 아래) 그건 이타샤가 아니다
    if info.get("hero_pull", 1.0) < HERO_PULL_FLOOR:
        fails.append("hero")
    return Critique(parts=parts, info=info, fails=tuple(fails), neg_box=neg_box)
