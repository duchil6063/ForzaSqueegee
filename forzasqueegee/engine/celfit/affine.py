"""기울기(skew) — 아핀의 **잃어버린 한 자유도**를 한 자리에 모은다.

게임 레이어의 선형부는 `R(rot) · Sk(skew) · S(sx, sy)`다 (`geometry._poly_px`·
`render._draw_layer`·`fls.binfmt.transform_matrix`가 전부 같은 식). 네 수는
2×2 행렬의 자유도 넷과 정확히 맞물리므로 **임의의 가역 아핀을 한 장으로 낼 수
있다**. 그런데 배치·채점은 오래 `(rot, sx, sy)` 셋만 썼다 — 그 셋의 상은
`R·D` 꼴이라 국소 선형변환의 두 축이 **직교해야** 한다. 비스듬히 눌린 면,
사선으로 잘린 경계, 한쪽으로 흐른 머리칼은 그 상 밖이라 도형 두 장으로
쪼개 근사할 수밖에 없었다.

여기 있는 것은 그 한 자유도를 **안전하게** 다시 넣기 위한 다섯 가지다:

1. `linear`·`decompose_linear` — 같은 행렬이면 **언제나 같은** 네 수 (정본
   분해). 같은 변환의 다른 매개화가 섞이면 결정성이 깨진다.
2. `q_skew`·`representable` — 게임 격자(0.01 스텝)와 실측 도달 범위.
3. `fit_full` — 중심선을 **전 아핀**으로 맞추는 닫힌 해. 제한 맞춤
   (`stroke._affine_fit`)보다 잔차가 절대 크지 않다.
4. `fit_ribbon`·`ribbon_res_of` — **띠**(중심선 + 폭)를 맞추는 닫힌 해와, 아무
   자세나 같은 자로 재는 잔차. 전단이 뜻을 가지려면 목적함수가 폭까지 봐야
   한다 (아래 「띠 맞춤」 문단).
5. `skew_useful` — 이 도형에서 전단이 **표현력을 더하나**. 이름이 아니라
   기하가 답한다.

**기본은 여전히 꺼짐이다** (`FS_SKEW`). 아래는 **팔을 갈라 잰** 기각 근거다 —
통합 스위치 하나로 재면 세부가 뒤섞인다 (2026-09-01 독립 A/B, 표준 01·04·06.
면 확인 판은 11장 전수).

    B 획만    총 도형 +0.4~0.8% (3/3)   ← 아끼려던 그것이 안 준다
              봉인 전 미커버 +1.5~19.4% (3/3) → 봉인 +3/+5/+14장 · 보수 +3/+10/+9장
              좋아지는 것: 배부름 −4.8% · 스필 −16.7% · 접선 −2.5% (전부 3/3)
    C 면만    총 도형 −1.2% (10/11) · 채움 −0.8% (11/11) · 전단 29장
              그런데 **보이는 오차 +5.9% (8/11 퇴행, 최대 +24.9%)** ·
              면 안 틀림 +11.4% (7/11) · 경계 넘김률 (8/11 퇴행)
    D 미세조정만  **무동작** — 기준판과 레이어 바이트까지 같다 (`finetune._SKEW_AXIS`)
    E 셋 다   어느 쪽보다도 나쁘다 — 보이는 오차 +12.5% (3/3) ·
              접합 끊김률 +29.2% (3/3) · 총 도형 부호가 C와 뒤집힌다

**획 쪽의 기전은 띠가 아니라 끝이다.** 앞선 문서가 "띠를 덜 촘촘히 덮는다"로
적었는데, 띠 폭은 되레 좋아진다 (`width_ratio_med` 0.860 → 0.876). 나빠지는
것은 **끝 뾰족함**(+13.8% · +5.7% · +21.5%, 3/3)이고, 그 뾰족한 끝이 여는
쐐기 틈이 봉인 전 미커버를 늘린다. 채점판(`scoring._score_impl`)이 재는 것은
`새 잔여 − 스필 벌점`이라 스필을 줄이는 전단이 그 저울에서 이기고, 못 덮고
남긴 표본의 값은 그 자리에서 안 물린다. 끝비 게이트를 더 조이면 후보가 하나도
안 남아 기본 경로로 수렴한다 (`bc14541` 실측).

**"커버리지가 퇴행한다"는 재현되지 않는다.** 전 팔·전 판에서
`hard_hole_samples` 0 · `seal_after` 0 · `hole_left` 0이다. 움직인 것은
`metrics.coverage`(연성 `silhouette_cover`)의 소수 넷째 자리와
`outline_cover` −0.17%뿐이다 — **하드 게이트가 아니다.**

여기 있는 것은 "켰을 때 안 망가지게"와 그 기각 근거다.

## 왜 정본 분해가 필요한가

`(rot, sx, sy, skew)`는 같은 행렬을 여러 꼴로 쓸 수 있다 (rot+180°와 sx·sy
부호 반전이 같은 변환이다). 그대로 최적화하면 같은 입력이 판마다 다른 수로
적히고, 그 차이가 그대로 plan.json의 글자 차이가 된다. 그래서 분해는
**FLS의 `decomposeTransform2D` 하나만** 쓴다 (`fls.binfmt.decompose`) — 저장·
불러오기 왕복이 이미 그 규약 위에 서 있으므로, 여기서 다른 규약을 세우면
왕복이 제자리가 아니게 된다.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

from ..catalog import Catalog
from ..fls.binfmt import decompose as _fls_decompose
from ..model import UNITS_PER_SCALE, rnd

# 게임 입력 스텝 (`model.Layer.quantized`와 같은 수 — 여기서 새로 세우지 않는다)
SKEW_STEP = 0.01
# **실측으로 확인된 도달 범위.** 에디터 기울기 도구(도구 키 `4`)는 최소
# 14.4까지 클램프가 안 걸렸다 (2026-09-01 인게임 실측). 그 위는 안 재 봤으므로
# 도달 검사(`pipeline._reach_check`)는 잰 데까지만 보증한다 — "게임이 못 낸다"가
# 아니라 "우리가 확인한 범위 밖"이라는 뜻이다.
SKEW_MAX = float(os.environ.get("FS_SKEW_MAX", 14.4))

# **수치 랭크 가드** — 중심선의 가로 퍼짐이 이보다 얇으면 전 아핀 해가
# 부정정이다 (곧은 중심선에서는 전단 방향이 데이터로 안 정해진다). 품질
# 문턱이 아니라 float64 랭크 문턱이다: 2×2 정규방정식의 고윳값 비가 이 아래면
# 역행렬이 부동소수 잡음을 증폭해 **같은 입력이 판마다 다른 답을 낸다**.
_RANK_EPS = 1e-8


# ── 정본 표현 ─────────────────────────────────────────────────────────
def linear(rot: float, sx: float, sy: float, skew: float) -> np.ndarray:
    """`(rot, sx, sy, skew)` → 2×2 선형부 `R·Sk·S` (`X = M·U`).

    `U`는 **도형 로컬 × `UNITS_PER_SCALE`** 단위다 (`stroke._stroke_forms`·
    `descriptor.ShapeDesc.center`가 내는 그 단위) — 그래야 여기서 나온 `sx`가
    `Layer.sx`에 그대로 들어간다.
    """
    th = np.radians(float(rot))
    c, s = np.cos(th), np.sin(th)
    # Sk·S = [[sx, skew·sy], [0, sy]]
    return np.array([[c * sx, c * skew * sy - s * sy],
                     [s * sx, s * skew * sy + c * sy]], np.float64)


def decompose_linear(m: np.ndarray) -> tuple[float, float, float, float]:
    """2×2 선형부 → **정본** `(rot도, sx, sy, skew)`.

    분해는 FLS `decomposeTransform2D` 하나뿐이다 (`fls.binfmt.decompose`) —
    저장 왕복이 그 규약 위에 서 있어서, 여기서 다른 규약을 세우면 같은 도안이
    파일로 오갈 때 다른 수로 적힌다. 이동은 이 함수의 관심이 아니라 0을 넣는다.

    정본이 보장하는 것 셋 (`work/lab/tests/test_skew_geometry.py`):
    - 같은 행렬이면 같은 네 수 (등가 매개화가 한 답으로 접힌다)
    - 합성 → 분해가 제자리
    - **전단 없이 낼 수 있는 변환은 반드시 `skew=0`으로 돌아온다**
    """
    m = np.asarray(m, np.float64)
    _, _, sx, sy, rot, skew = _fls_decompose(
        ((float(m[0, 0]), float(m[0, 1]), 0.0),
         (float(m[1, 0]), float(m[1, 1]), 0.0)))
    return rot, sx, sy, skew


def q_skew(v: float) -> float:
    """게임 입력 스텝(0.01)으로 양자화 — **음의 0을 안 남긴다** (`model.rnd`)."""
    v = float(v)
    if not np.isfinite(v):
        return 0.0
    return rnd(round(v / SKEW_STEP) * SKEW_STEP, 4)


def representable(v: float) -> bool:
    """이 기울기를 게임이 **그대로** 낼 수 있나 (도달 검사의 자).

    "0인가"가 아니라 "유한하고 · 확인된 범위 안이고 · 입력 격자 위인가"다.
    """
    v = float(v)
    if not np.isfinite(v) or abs(v) > SKEW_MAX + 1e-9:
        return False
    return abs(v - q_skew(v)) <= 1e-9


def shear_visible(skew: float, sy: float, ext_y: float) -> bool:
    """이 전단이 이 도형·이 자세에서 **화면을 움직이나**.

    전단은 로컬 y에 비례해 x를 민다 — 미는 최대 거리가 `|skew| · |sy| · ext_y`
    (캔버스 유닛)다. 그것이 **이동 양자(0.5유닛)**에도 못 미치면 그 전단은
    이 자리에서 무동작이다. 새 상수가 아니라 이미 있는 두 격자의 비교다.

    묻는 크기가 자리마다 다르다는 것이 요점이다:

    - **해석적 씨앗**은 맞춤이 낸 `skew` 그대로 묻는다 — 씨앗의 값은 큰
      전단에 있으므로 한 스텝으로 물으면 될 것도 안 된다.
    - **이웃 탐색**(하강·미세 조정·접합점·이음 늘리기)은 한 스텝을 묻는다
      (`step_matters`) — 거기서 움직이는 양이 한 스텝이기 때문이다.
    """
    return abs(float(skew)) * abs(float(sy)) * max(float(ext_y), 0.0) >= 0.5


def step_matters(sy: float, ext_y: float, upp: float) -> bool:
    """기울기 **한 스텝**이 이 도형·이 배율에서 실제로 화면을 움직이나.

    `shear_visible`에 한 스텝(0.01)을 넣은 것 — 이웃 탐색 전용이다.

    **이 자를 안 걸면 전단이 도안에 흩뿌려진다.** 실측(표준 5장, 자를 씨앗
    경로에만 걸고 이웃 탐색에는 안 걸었을 때): 전단을 받은 1,968장 중
    **68.6%가 정확히 한 스텝(0.01)**이고 84.8%가 0.02 이하였다. 한 스텝이
    미는 거리가 이동 양자 아래라 그 이득은 폴리곤 경계 픽셀 몇 개가 뒤집히는
    라스터 잡음이고, 장수도 품질도 안 움직였다 (레이어 −29~+5장 · 보이는
    오차 ±0.02). 값을 하는 것은 **해석적 씨앗이 낸 큰 전단** 쪽이다.
    """
    return shear_visible(SKEW_STEP, sy, ext_y)


_EXT_Y: dict[tuple[int, str], float] = {}


def shape_ext_y(cat: Catalog, name: str) -> float:
    """도형 로컬 y의 반경 (캔버스 유닛) — 전단의 지렛대. 프로세스 1회."""
    key = (id(cat), name)
    got = _EXT_Y.get(key)
    if got is None:
        try:
            got = float(max(np.abs(np.asarray(l, np.float64)[:, 1]).max()
                            for l in cat[name].loops)) * UNITS_PER_SCALE
        except Exception:                  # noqa: BLE001 — 못 재면 안 막는다
            got = UNITS_PER_SCALE
        _EXT_Y[key] = got
    return got


def step_visible(cat: Catalog, lay) -> bool:
    """이 **레이어**에서 기울기 한 스텝이 화면을 움직이나 (`step_matters`의 껍데기).

    이웃 탐색(하강·미세 조정·접합점·이음 늘리기)이 전단 축에 들어가기 전에
    묻는 자리다 — 안 물으면 그 자리가 한 스텝짜리 전단을 대량으로 만든다.
    """
    return step_matters(lay.sy, shape_ext_y(cat, lay.shape), 0.0)


# ── 전 아핀 닫힌 해 ───────────────────────────────────────────────────
def fit_full(U: np.ndarray, X: np.ndarray):
    """중심선 `U(S,N,2)`를 목표 `X(N,2)`에 **전 아핀**으로 맞춘다.

    `stroke._affine_fit`의 짝이다 — 저쪽은 상을 `R·D`(직교 두 축)로 제한하고
    이쪽은 2×2 전체를 푼다. 그래서 잔차가 **절대 크지 않다**. 해는 평범한
    최소제곱이라 닫힌 형태다: 중심을 맞춘 뒤 `A = (Uᵀ U)⁻¹ Uᵀ X`이고
    선형부는 `M = Aᵀ`다.

    반환 `(rot, sx, sy, skew, 잔차제곱)` — 전부 `(S,)`.

    **역행렬은 2×2 닫힌 식으로 푼다** (LAPACK을 안 부른다) — 라이브러리
    행렬 연산은 스레드 수·정렬에 따라 마지막 자리가 흔들려 같은 입력이 판마다
    다른 답을 낸다 (`determinism-traps`). 정규방정식의 고윳값 비가
    `_RANK_EPS` 아래면 **부정정**이라 제한 맞춤으로 물러난다 (곧은 중심선에서는
    전단 방향이 데이터로 안 정해진다 — §7이 막대에 보수적이어야 하는 까닭이
    이름이 아니라 여기 있다).
    """
    from .stroke import _affine_fit

    U = np.asarray(U, np.float64)
    X = np.asarray(X, np.float64)
    Uc = U - U.mean(axis=1, keepdims=True)
    Xc = X - X.mean(axis=0, keepdims=True)
    G = np.einsum("sni,snj->sij", Uc, Uc)           # (S,2,2) 정규방정식
    B = np.einsum("sni,nj->sij", Uc, Xc)            # (S,2,2)
    det = G[:, 0, 0] * G[:, 1, 1] - G[:, 0, 1] * G[:, 1, 0]
    tr = G[:, 0, 0] + G[:, 1, 1]
    # 고윳값 비 = 4·det / (tr² + …) 의 대소 — 대칭 2×2라 λ₁λ₂ = det, λ₁+λ₂ = tr.
    # λmin/λmax ≥ eps ⟺ det ≥ eps/(1+eps)² · tr² 이므로 tr²에 대고 잰다
    good = det > _RANK_EPS * tr * tr
    inv = np.zeros_like(G)
    d = np.where(good, det, 1.0)
    inv[:, 0, 0] = G[:, 1, 1] / d
    inv[:, 1, 1] = G[:, 0, 0] / d
    inv[:, 0, 1] = -G[:, 0, 1] / d
    inv[:, 1, 0] = -G[:, 1, 0] / d
    A = np.einsum("sij,sjk->sik", inv, B)           # (S,2,2), X ≈ Uc·A
    res = float((Xc ** 2).sum()) - np.einsum("sij,sij->s", A, B)
    S = U.shape[0]
    rot = np.zeros(S)
    sx = np.zeros(S)
    sy = np.zeros(S)
    sk = np.zeros(S)
    for i in range(S):
        if not good[i]:
            continue
        rot[i], sx[i], sy[i], sk[i] = decompose_linear(A[i].T)
    if not good.all():
        # 부정정인 자리는 제한 맞춤(전단 0)을 그대로 쓴다 — 후보가 비지 않는다
        th0, sx0, sy0, res0 = _affine_fit(U, X)
        bad = ~good
        rot[bad] = np.degrees(th0[bad]) % 360.0
        sx[bad], sy[bad] = sx0[bad], sy0[bad]
        sk[bad] = 0.0
        res[bad] = res0[bad]
    return rot, sx, sy, sk, np.maximum(res, 0.0)


def fit_moment(cov_t: np.ndarray, cov_x: np.ndarray, k: int) -> np.ndarray:
    """2차 모멘트를 **정확히** 맞추는 선형부 (면 채움의 전 아핀 씨앗).

    `M · cov_t · Mᵀ = cov_x`의 해는 `M = cov_x^½ · Q · cov_t^-½`이고 `Q`는
    임의의 직교행렬이다. `k`(0~3)가 그 `Q`를 90° 배수 넷 중 하나로 고른다 —
    현행 씨앗(`fill._seed_moment`)이 주축을 붙이는 네 자세와 같은 후보 집합이다.

    갈리는 것은 **대각 성분만 보느냐**다. 현행은 도형 템플릿의 로컬 x·y
    표준편차만 쓰므로, 교차 모멘트가 0이 아닌 도형(초승달·삼각·쐐기)에서는
    모멘트가 애초에 안 맞는 자세에서 하강을 시작한다. 여기서는 세 성분을 다
    맞춘다 — 그 대가로 일반적으로 전단이 든다. 그것이 §8이 말하는
    "평행사변형에 가까운 면을 한 장이 통째로 설명한다"다.
    """
    q = k % 4
    Q = np.array([[np.cos(q * np.pi / 2), -np.sin(q * np.pi / 2)],
                  [np.sin(q * np.pi / 2), np.cos(q * np.pi / 2)]], np.float64)
    return _sqrtm2(cov_x) @ Q @ _sqrtm2(cov_t, inverse=True)


def _sqrtm2(c: np.ndarray, inverse: bool = False) -> np.ndarray:
    """대칭 양정치 2×2의 제곱근 (또는 그 역) — **닫힌 식**.

    `eigh`를 안 쓴다: LAPACK 경로는 같은 입력에서 마지막 자리가 흔들린다
    (`determinism-traps`). 2×2는 판별식으로 고윳값이 바로 나오고, 제곱근은
    `√C = (C + √det·I) / √(tr + 2√det)`라는 항등식으로 고유벡터 없이 풀린다.
    """
    c = np.asarray(c, np.float64)
    a, b, d = float(c[0, 0]), float(0.5 * (c[0, 1] + c[1, 0])), float(c[1, 1])
    det = max(a * d - b * b, 1e-18)
    t = max(a + d + 2.0 * np.sqrt(det), 1e-18)
    s = np.array([[a + np.sqrt(det), b], [b, d + np.sqrt(det)]],
                 np.float64) / np.sqrt(t)
    if not inverse:
        return s
    sd = s[0, 0] * s[1, 1] - s[0, 1] * s[1, 0]
    if abs(sd) < 1e-18:
        return np.eye(2)
    return np.array([[s[1, 1], -s[0, 1]], [-s[1, 0], s[0, 0]]], np.float64) / sd


# ── 어느 도형에서 전단이 표현력을 더하나 ──────────────────────────────
# **회전 대칭이 답이다** — 이름 표가 아니라 기하다.
#
# 전단 없는 상은 `{R·D·P}`이고, 전단을 넣으면 상이 `{A·P : A 가역}`으로 넓어
# 진다. 임의의 `A`를 특이값분해하면 `A = U·Σ·Vᵀ`이므로 `A·P = U·Σ·(VᵀP)`인데,
# **`P`가 `Vᵀ`에 대해 불변이면** 그것은 곧 `U·Σ·P` = 전단 없는 상 안이다.
# `V`는 임의이므로: 전단이 아무것도 안 더하는 도형 = **모든 회전에 불변한
# 도형**(원·고리)뿐이다. 유한 대칭(사각형의 4겹)은 그 유한한 각에서만
# 겹치므로 전단이 여전히 표현력을 더한다 — 눌린 사각형은 평행사변형이고,
# 그것은 회전·비등방 스케일로 못 낸다.
#
# 그래서 자는 하나다: **제 무게중심 둘레로 돌렸을 때 겹치나.**
#
# 문턱 0.97은 **실측 골짜기**다 (도달 도형 520종 전수, `_rot_iou`를 1°~89°
# 전 각으로 돌려 잰 최소 IoU):
#
#     1.0000  E_01~E_04 · B_09      — 빈 도형 (그릴 것이 없다)
#     0.9863  B_26~B_29             — 고리
#     0.9856  A_02 · V_54~V_84 일곱  — 원
#     ───────────────── 여기가 비어 있다 ─────────────────
#     0.9555  U_58 · U_41
#     0.9455  A_12   0.9224 A_06   …   0.8514 A_22(둥근사각)   …
#
# 원·고리 무리(0.9856~0.9863)와 그다음(0.9555) 사이가 통째로 비었다. 0.97은
# 그 빈 띠의 한가운데다 — 값이 아니라 골짜기라, 어휘를 넓혀도 판정이 안 흔들린다.
# (0.985로 두면 원 무리의 바닥에서 여유가 0.0006밖에 안 남아 래스터 해상도를
#  바꾸는 것만으로 판정이 뒤집힌다.)
_ROT_INVARIANT = float(os.environ.get("FS_SKEW_ROT_IOU", 0.97))
_ROT_RASTER = 128
_USEFUL: dict[tuple[int, str], bool] = {}


def _rot_iou(cat: Catalog, name: str) -> float:
    """이 도형을 제 무게중심 둘레로 돌렸을 때의 **최소** IoU (1 = 원).

    문턱 아래로 떨어지면 그 자리에서 멈춘다 — 답이 `< _ROT_INVARIANT` 하나로만
    쓰이므로, 아래로 갈린 뒤의 값은 판정을 안 바꾼다 (그때 반환값은 최소의
    상계다).
    """
    m = cat[name].rasterize(_ROT_RASTER).astype(np.uint8)
    if not m.any():
        return 1.0
    mo = cv2.moments(m, binaryImage=True)
    if mo["m00"] < 1:
        return 1.0
    cx, cy = mo["m10"] / mo["m00"], mo["m01"] / mo["m00"]
    base = m.astype(bool)
    n0 = float(base.sum())
    worst = 1.0
    # 1°~89° — 흔한 대칭(2·3·4·6·8·48겹)의 배수만 밟지 않게 촘촘히 훑는다
    for deg in range(1, 90):
        R = cv2.getRotationMatrix2D((cx, cy), float(deg), 1.0)
        r = cv2.warpAffine(m, R, (m.shape[1], m.shape[0]),
                           flags=cv2.INTER_NEAREST).astype(bool)
        inter = float((base & r).sum())
        union = float((base | r).sum())
        if union < 1.0:
            continue
        worst = min(worst, inter / union)
        if worst < _ROT_INVARIANT:         # 이미 갈렸다 — 더 볼 것 없다
            break
    return worst if n0 > 0 else 1.0


def report(layers) -> dict:
    """§14 계측 — 이 판이 전단을 **어디에 얼마나** 썼나 (report.json용).

    A/B로만 답할 수 있는 것(오차·스필·이음의 전후 차)은 여기 없다 — 그것은
    OFF/ON 두 판의 같은 지표를 견주는 자리다. 여기서는 한 판 안에서 셀 수
    있는 것만 낸다: 쓴 장수·몫·역할별 갈래·`|skew|` 분포·도형별 상위.
    """
    n = len(layers)
    sk = [l for l in layers if l.skew]
    out = {"skew_layers": len(sk),
           "skew_ratio": round(len(sk) / max(1, n), 4),
           "skew_ink": sum(1 for l in sk if l.label == "ink"),
           "skew_fill": sum(1 for l in sk if l.label != "ink")}
    if sk:
        a = np.abs(np.array([l.skew for l in sk], np.float64))
        out.update({"skew_abs_med": round(float(np.median(a)), 4),
                    "skew_abs_p90": round(float(np.percentile(a, 90)), 4),
                    "skew_abs_max": round(float(a.max()), 4),
                    # **흩뿌림의 자** — 두 스텝 이하짜리 전단의 몫. 이것이
                    # 크면 전단이 값을 하는 것이 아니라 라스터 잡음을 좇고
                    # 있다는 뜻이다 (`step_matters` 문단의 실측 68.6%)
                    "tiny_skew_ratio": round(
                        float((a <= 2.0 * SKEW_STEP + 1e-9).mean()), 4)})
        top: dict[str, int] = {}
        for l in sk:
            top[l.shape] = top.get(l.shape, 0) + 1
        out["skew_shapes"] = dict(sorted(top.items(),
                                         key=lambda t: (-t[1], t[0]))[:8])
    return out


def skew_useful(cat: Catalog, name: str) -> bool:
    """이 도형에서 전단이 **표현력을 더하나** (프로세스 1회 계측 후 기억).

    회전에 통째로 불변한 도형(원·고리)에서는 전단된 결과가 회전 + 비등방
    스케일로 그대로 나오므로, 후보를 지어 봐야 자유도 중복과 수치 불안정만
    남는다 (§4).
    """
    key = (id(cat), name)
    got = _USEFUL.get(key)
    if got is None:
        try:
            got = _rot_iou(cat, name) < _ROT_INVARIANT
        except Exception:                  # noqa: BLE001 — 못 재면 안 쓴다
            got = False
        _USEFUL[key] = got
    return got


# ── 띠 맞춤 — 전단이 **뜻을 갖는** 목적함수 ───────────────────────────
# 앞선 판(`b739ffb`·`2f07f7c`)이 전단을 켜고 진 까닭은 게이트가 아니라
# **목적함수**였다. `fit_full`이 최소화하는 것은 중심선 잔차뿐인데, 놓인 폭은
#
#     w(t) = 2·반폭(t)·|det M| / |M·t̂(t)|
#
# 이라 전단이 `det M`을 안 바꾸고 `|M·t̂|`만 키운다 — 즉 **폭을 공짜로 깎아
# 중심선을 싸게 맞추는 지름길**이 열려 있다. 실측(표준 11장, 획 도형 13,092장)
# 에서 |k|가 6을 넘는 자리의 놓인 폭 중앙이 0.57px(원화 띠 ~2px)까지 무너졌다.
# 폭 바닥(`stroke._STROKE_WMIN`)을 세우면 그 거래는 막히지만, 바닥을 어디에
# 두든 **전단 후보가 하나도 안 남는다** — 중심선만 보는 씨앗이 내놓는 전단은
# 애초에 폭을 희생한 것뿐이기 때문이다.
#
# 그래서 자를 조이는 대신 **맞추는 대상을 바꾼다**: 중심선 한 가닥이 아니라
# **띠**(중심선 + 폭)를 맞춘다. 도형의 반폭 벡터 `D = 반폭·n̂`와 원화 띠의
# 반폭 벡터 `Dx`를 대응쌍으로 함께 넣으면, 전단이 폭을 무너뜨리는 순간 그
# 대가를 **목적함수 안에서** 치른다. 게이트가 사후에 거르던 것을 씨앗이
# 애초에 안 만든다.
#
# 두 가지가 공짜로 딸려 온다:
#
# - **랭크가 산다.** 곧은 중심선에서 `Uᵀ U`는 랭크 1이라 `fit_full`이
#   부정정으로 물러났다 (`_RANK_EPS`). 폭 벡터는 접선에 **수직**이라 그 방향을
#   채워 준다 — 곧은 획에서도 전단이 데이터로 정해진다.
# - **반사가 제자리다.** `det M < 0`이면 +90°로 잡은 법선이 반대쪽으로 가므로
#   부호 두 벌을 다 풀어 잔차가 작은 쪽을 쓴다 (닫힌 해 두 번, 여전히 싸다).
#
# 순위는 여전히 **중심선 잔차**로 매긴다 (`fit_full`·`_affine_fit`과 같은 자) —
# 띠는 자세를 정하고, 그 자세가 기존 후보와 겨루는 저울은 그대로 둔다. 그래야
# 전단 후보가 "중심선을 더 잘 맞춰서"가 아니라 실제 이득으로만 이긴다.

# 폭 항의 무게. 1.0이면 "중심선 한 점과 폭 한 점이 같은 값"이다.
RIBBON_W = float(os.environ.get("FS_SKEW_RIBBON", 1.0))


def normals(p: np.ndarray) -> np.ndarray:
    """폴리라인의 **단위 법선** (접선을 +90° 돌린 것). 마지막 축이 (x, y).

    끝점은 이웃 마디의 접선을 그대로 쓴다 (중앙차분의 가장자리 처리) — 새
    규약이 아니라 `descriptor.placed_widths`가 마디 접선을 쓰는 것과 같은 자다.
    """
    p = np.asarray(p, np.float64)
    t = np.zeros_like(p)
    t[..., 1:-1, :] = p[..., 2:, :] - p[..., :-2, :]
    t[..., 0, :] = p[..., 1, :] - p[..., 0, :]
    t[..., -1, :] = p[..., -1, :] - p[..., -2, :]
    n = np.hypot(t[..., 0], t[..., 1])
    t = t / np.maximum(n, 1e-12)[..., None]
    return np.stack([-t[..., 1], t[..., 0]], axis=-1)


def _solve2(G: np.ndarray, B: np.ndarray):
    """`A = G⁻¹B` — 대칭 2×2 닫힌 식 (LAPACK을 안 부른다, `determinism-traps`).

    반환 `(A, 풀렸나)`. 부정정인 자리의 `A`는 0이라 뒤에서 물러난다.
    """
    det = G[:, 0, 0] * G[:, 1, 1] - G[:, 0, 1] * G[:, 1, 0]
    tr = G[:, 0, 0] + G[:, 1, 1]
    good = det > _RANK_EPS * tr * tr
    d = np.where(good, det, 1.0)
    inv = np.zeros_like(G)
    inv[:, 0, 0] = G[:, 1, 1] / d
    inv[:, 1, 1] = G[:, 0, 0] / d
    inv[:, 0, 1] = -G[:, 0, 1] / d
    inv[:, 1, 0] = -G[:, 1, 0] / d
    A = np.einsum("sij,sjk->sik", inv, B)
    return np.where(good[:, None, None], A, 0.0), good


def fit_ribbon(U: np.ndarray, D: np.ndarray,
               X: np.ndarray, Dx: np.ndarray, lam: float = RIBBON_W):
    """**띠**(중심선 `U` + 반폭 벡터 `D`)를 목표 띠(`X`, `Dx`)에 전 아핀으로.

    `U(S,N,2)`·`D(S,N,2)`는 도형 로컬 × `UNITS_PER_SCALE`, `X(N,2)`·`Dx(N,2)`는
    캔버스 유닛이다. 반폭 벡터는 **점이 아니라 차이**라 이동이 저절로 빠진다 —
    중심만 맞추면 되고 따로 뺄 것이 없다.

    정규방정식은 `fit_full`과 같은 꼴에 항이 하나 더 붙는다:

        G = Ucᵀ Uc + λ · Dᵀ D          B = Ucᵀ Xc + λ · Dᵀ Dx

    반환 `(rot, sx, sy, skew, 중심선 잔차)` — 전부 `(S,)`. 잔차가 띠가 아니라
    **중심선**인 것이 요점이다 (위 문단).
    """
    from .stroke import _affine_fit

    U = np.asarray(U, np.float64)
    D = np.asarray(D, np.float64)
    X = np.asarray(X, np.float64)
    Dx = np.asarray(Dx, np.float64)
    Uc = U - U.mean(axis=1, keepdims=True)
    Xc = X - X.mean(axis=0, keepdims=True)
    Gc = np.einsum("sni,snj->sij", Uc, Uc)
    Bc = np.einsum("sni,nj->sij", Uc, Xc)
    Gd = np.einsum("sni,snj->sij", D, D) * lam
    xx = float((Xc ** 2).sum())
    # **부호 두 벌** — `det M < 0`이면 법선이 반대쪽으로 간다. 닫힌 해라 둘 다
    # 풀고 (띠) 잔차가 작은 쪽을 쓴다. 동점은 `+`가 이긴다 (결정적).
    best = None
    for sgn in (1.0, -1.0):
        Bd = np.einsum("sni,nj->sij", D, Dx * sgn) * lam
        A, good = _solve2(Gc + Gd, Bc + Bd)
        # 띠 잔차 — 어느 부호를 쓸지 고르는 자 (순위에는 안 쓴다)
        rb = (xx + lam * float((Dx ** 2).sum())
              - np.einsum("sij,sij->s", A, Bc + Bd))
        if best is None:
            best = (A, good, rb)
        else:
            take = rb < best[2] - 1e-12
            best = (np.where(take[:, None, None], A, best[0]),
                    good & best[1], np.where(take, rb, best[2]))
    A, good, _ = best
    # 순위용 **중심선** 잔차 — 이 자세를 기존 후보와 같은 저울에 세운다
    res = xx - 2.0 * np.einsum("sij,sij->s", A, Bc) + \
        np.einsum("sij,sij->s", A, np.einsum("sij,sjk->sik", Gc, A))
    S = U.shape[0]
    rot = np.zeros(S)
    sx = np.zeros(S)
    sy = np.zeros(S)
    sk = np.zeros(S)
    for i in range(S):
        if not good[i]:
            continue
        rot[i], sx[i], sy[i], sk[i] = decompose_linear(A[i].T)
    if not good.all():
        th0, sx0, sy0, res0 = _affine_fit(U, X)
        bad = ~good
        rot[bad] = np.degrees(th0[bad]) % 360.0
        sx[bad], sy[bad] = sx0[bad], sy0[bad]
        sk[bad] = 0.0
        res[bad] = res0[bad]
    return rot, sx, sy, sk, np.maximum(res, 0.0)


def linear_batch(rot: np.ndarray, sx: np.ndarray, sy: np.ndarray,
                 skew: np.ndarray) -> np.ndarray:
    """`linear`의 벡터판 — `(S,)` 넷 → `(S,2,2)`. `rot`은 **라디안**."""
    c, s = np.cos(rot), np.sin(rot)
    m = np.empty((len(rot), 2, 2), np.float64)
    m[:, 0, 0] = c * sx
    m[:, 0, 1] = c * skew * sy - s * sy
    m[:, 1, 0] = s * sx
    m[:, 1, 1] = s * skew * sy + c * sy
    return m


def ribbon_res_of(U: np.ndarray, D: np.ndarray, X: np.ndarray,
                  Dx: np.ndarray, M: np.ndarray,
                  lam: float = RIBBON_W) -> np.ndarray:
    """**주어진 자세**의 띠 잔차 — 후보를 한 저울에 세우는 자.

    `fit_ribbon`이 제 해의 잔차를 내는 것과 달리 이쪽은 **아무 자세나** 잰다.
    전단 후보와 `skew=0` 후보를 같은 자로 견주는 데 쓴다: 씨앗이 최소화한 것이
    서로 다르므로(한쪽은 띠, 한쪽은 중심선) 중심선 잔차로 줄을 세우면 띠
    후보가 제 강점을 못 보여 준 채 상위 `_CURVE_TOP` 밖으로 밀린다.

    `Dx`의 부호는 반사에서 뒤집히므로 두 벌 중 작은 쪽을 쓴다 (`fit_ribbon`과
    같은 규약).
    """
    Uc = U - U.mean(axis=1, keepdims=True)
    Xc = X - X.mean(axis=0, keepdims=True)
    pu = np.einsum("sni,sji->snj", Uc, M)
    rc = ((pu - Xc[None]) ** 2).sum(axis=(1, 2))
    pd = np.einsum("sni,sji->snj", D, M)
    rp = ((pd - Dx[None]) ** 2).sum(axis=(1, 2))
    rm = ((pd + Dx[None]) ** 2).sum(axis=(1, 2))
    return rc + lam * np.minimum(rp, rm)
