"""획 맞춤 — 경로 하나를 **곡선 도형 한 장**으로, 안 되면 곡선 여러 장으로.

사람의 획 문법이다: FH5 튜토리얼은 클립보드의 곡선 한 종을 회전·크기만 바꿔
반복해 선화 전체를 230장으로 끝낸다. 아핀 최소제곱(`_affine_fit`)이 어휘
28종의 중심선을 경로에 닫힌 해로 맞추고, 자격 게이트를 못 넘으면 최대
이탈점에서 쪼개 다시 곡선을 묻는다 — 그래서 한 획이 쓸 장수에 상한이 걸린다.

**막대(A_22)는 확실히 직선인 자리에만 쓴다** (`_is_straight` — 사용자 요구
2026-08-26). 마디까지 내려가도 굽은 마디는 곡선과 막대를 같은 채점판에서
겨루게 하고 비기면 곡선이 이긴다.

**무엇이 획 도형인가는 놓인 뒤의 폭 프로파일이 답한다** — 테이퍼·가늘기·
원화 대비 폭·끝 뭉툭함(`_STROKE_*`)이 자격을 묻고, 폭 프로파일과 끝
접선(`_W_PROF`·`_W_TANG`)이 그중 무엇을 쓸지 순위를 매긴다. 넷 다 닫힌
식이라(`descriptor.placed_widths`) 후보마다 물어도 라스터를 안 뜬다.
"""

from __future__ import annotations

import os
from dataclasses import replace

import cv2
import numpy as np

from ..catalog import Catalog
from ..model import UNITS_PER_SCALE, Layer, LayerPlan
from .geometry import _layer, _min_span
from .scoring import _MIN_GAIN, _Scorer, _descend
from .skeleton import (_cross2, _paths, _prune_spurs, _rdp_idx, _resample,
                       _thin)
from . import chain
from . import intent as I
from . import affine as A
from . import policy as _policy
from .descriptor import placed_profile, placed_widths
from .vocabulary import bar_for, min_stroke_width_px, stroke_vocab


_FORM_N = 16           # 중심선 대응점 수 (아핀 맞춤)
_FORM_RASTER = 160     # 중심선을 뽑는 래스터 해상도
_CURVE_TOP = 8         # 아핀 잔차 상위 몇 개를 실제로 채점하나
# 곡선 한 장 게이트 셋 (`_try_curve` 꼬리) — 스윕용 스위치. 셋은 "한 장으로
# 그을 자격"을 묻는다. 빡빡하면 획이 막대 사슬로 떨어지는데, 막대는 마디마다
# 한 장이라 **선화가 플랜의 절반**이 된다. 더 열면(0.28/0.50/5.0) 장수가 몇
# 장 더 줄 뿐이라 여기가 무릎이다.
_CURVE_GAIN = float(os.environ.get("FS_CURVE_GAIN", 0.40))   # 순이득 / 획 질량
_CURVE_ON = float(os.environ.get("FS_CURVE_ON", 0.55))       # 잉크 중 내 면 위 몫
_CURVE_AREA = float(os.environ.get("FS_CURVE_AREA", 3.5))    # 면적 / 획 질량 상한
# 막대 사슬의 단순화 허용오차 = 이 배수 × 획 폭 (`_fit_segments`). 선화 마디의
# 69~89%가 이 막대라 **마디 수를 실제로 정하는 상수는 여기다**. 스윕용.
_BAR_EPS = float(os.environ.get("FS_BAR_EPS", 0.7))


# 막대 폭 배수 — 1.0이면 획을 선 띠의 폭에 **맞춘다**. 실측상 선 띠의 8~30%가
# 안 덮이고 그 자리가 남은 오차의 최대 덩어리라, 띠를 **덮게** 하는 쪽을
# 재려면 이 배수가 필요하다. 스윕용.
_BAR_W = float(os.environ.get("FS_BAR_W", 1.0))


# **직선 판정** — 막대·사각 계열을 쓸 자격 (사용자 요구 2026-08-26: "확실히
# 직선이라고 판단되는 자리가 아니면 직선 계열 도형을 쓰지 않는다"). 자는
# 현에서의 최대 이탈이고 단위는 획 폭이다: 반폭(0.5)을 못 넘으면 굽음이 획
# 몸통 안에 묻혀 테두리가 직선에서 안 벗어난다 — 렌더에서 굽음이 읽히기
# 시작하는 경계다. px 바닥 0.5는 게임의 이동 양자라 그 아래 굽음은 그릴
# 수단 자체가 없다.
_STRAIGHT_DEV = float(os.environ.get("FS_STRAIGHT_DEV", 0.5))
# 쪼개기 — 곡선 한 장이 안 되면 **곡선 여러 장으로 쪼개서라도 곡선을 쓴다**.
# 깊이 4단이면 16조각까지 나눌 수 있어 획당 상한(`policy.max_shapes`)이
# 실제 제약이 되고, 최소 표본은 `_try_curve`가 요구하는 10표본의 두 배라야
# 짧은 굽은 조각도 곡선 자격을 얻는다.
_SPLIT_DEPTH = int(os.environ.get("FS_SPLIT_DEPTH", 4))
_SPLIT_MIN = int(os.environ.get("FS_SPLIT_MIN", 20))
# **선에 적합한 도형만** (line 노선, 사용자 요구 2026-08-26 "타원 같은 게 딱
# 보여서 선화처럼 안 느껴진다"). 어휘 목록을 손으로 고르지 않는다 — **놓인 뒤의
# 모습**을 재서 거른다. 같은 도형도 비등방 스케일을 얼마나 받느냐에 따라 가는
# 획이 되기도 하고 잎사귀·쐐기가 되기도 하기 때문이다 (곡선 띠를 한 축으로만
# 누르면 폭이 접선 방향을 따라 갈린다).
#
# 자는 사람 획의 실측이다 (레퍼런스 획 284개):
# 사람 획은 폭이 거의 일정하고(테이퍼 중앙 1.0 · **p90 2.0**)
# 길이에 비해 아주 가늘다(폭/길이 중앙 0.041~0.056 · **p90 0.09**). 그쪽은
# 라스터(거리변환 능선)로 잰 값이고 여기 식은 닫힌 해라, 우리 배치 733장을
# 두 자로 나란히 재 대응을 확인했다: 획형 도형에서 테이퍼는 ±0.5 안에서 같고
# (U_31 1.24↔2.00 · U_13 2.12↔2.00 · A_36 1.70↔2.20 · C_22 1.86↔1.40),
# 폭/길이는 식이 라스터의 0.7배다(라스터 median 폭이 끝을 물어 커진다).
# 그래서 문턱을 식 기준 2.5 / 0.065로 둔다 — 라스터로 2.0 / 0.09 자리다.
# 걸리는 것은 쐐기·잎사귀·덩어리뿐이다: A_38 8.05 · F_07 8.28 · A_03 5.27 ·
# A_39 4.19 · A_04 3.80(테이퍼), U_03 0.188 · A_04 0.204 · A_27 0.149 ·
# A_29 0.129 · U_70 0.108 · U_46 0.104(폭/길이).
_STROKE_TAPER = float(os.environ.get("FS_STROKE_TAPER", 2.5))
_STROKE_SLIM = float(os.environ.get("FS_STROKE_SLIM", 0.065))
# **놓인 폭이 원화 획 폭의 몇 배까지인가** (line 노선의 폭 충실도).
# 위 둘은 "획처럼 생겼나"의 자이지 "이 획만큼 가는가"의 자가 아니다 —
# 길이에 비례한 상대 문턱이라, 100px 경로에 폭 6px 도형이 slim 0.06으로
# 그대로 통과한다. 실측(01, 배치된 획 도형의 능선 폭 대 그 획의 원화 띠 폭):
# 중앙 1.47배 · p90 2.09배이고, 도안 전체로도 잉크 능선 폭 중앙이 4.0px ↔
# 원화 2.0px였다 (01·04·07 잉크 px가 선 지도의 1.27·1.16·1.44배). 머리칼
# 가닥이 서로 붙어 덩어리로 읽히던 자리가 이것이다.
# 사람 획은 폭이 원화를 따라간다 — 그래서 자를 하나 더 세운다. 1.0은 못
# 쓴다: 게임 격자(스케일 0.01 눈금)가 목표 폭을 반올림하므로 그만큼은
# 열어야 한다 (`vocabulary.bar_for`의 `_BAR_WERR`와 같은 사정).
_STROKE_WMAX = float(os.environ.get("FS_STROKE_WMAX", 1.15))
# 폭의 **바닥** — 기본은 천장의 거울이다 (`1 / _STROKE_WMAX`). 스윕용 스위치.
_STROKE_WMIN = float(os.environ.get("FS_STROKE_WMIN", 1.0 / _STROKE_WMAX))
# 전단 씨앗이 **무엇을 맞추나** — `ribbon`(중심선 + 폭) 또는 `center`(중심선만,
# 옛 경로). A/B용 스위치다 (`affine.fit_ribbon` 문단이 왜 띠인지 적는다).
_SKEW_FIT = os.environ.get("FS_SKEW_FIT", "ribbon").strip().lower()
# **폭 프로파일**을 후보 순위에 얼마로 넣나 (`_prof_pen`, 0 = 안 본다).
# 위 셋은 전부 폭을 **한 수**로 줄여 묻는다 (중앙·테이퍼·폭/길이). 그래서
# "가운데는 원화 폭인데 끝이 뾰족한" 물방울이 다 통과한다 — 실측(표준 10장)
# 획 도형의 16.1%가 U_35(등방 테이퍼 3.36 · 끝 반폭이 가운데의 0.20)였고,
# 그것이 머리칼 가닥이 잎사귀로 읽히던 자리다. 가운데 80%만 재는 자
# (`placed_profile` 문서)로는 원리적으로 못 본다.
#
# 그래서 폭을 **프로파일로** 묻는다: 놓인 폭과 원화 띠 폭의 차를 길이로
# 적분하면 단위가 px²라 채점 점수(px)와 그대로 더해진다. 1.0이면 "폭이 1px
# 어긋난 자리 1px는 잘못 칠한 1px과 같다"는 뜻이고, 그 저울은 이미
# 채점판에 있다 (`scoring._PEN_LINE`). 새 상수를 세우는 것이 아니라 이미
# 있는 저울에 폭 축을 얹는 것이다.
_W_PROF = float(os.environ.get("FS_WIDTH_PROFILE", 1.0))
# 목표 프로파일의 상한 배수 — 원화 띠가 이 획의 폭보다 이만큼 넘게 굵어지는
# 자리(다른 선과 붙은 병합 띠)는 이 획의 폭이 아니다.
_PROF_CAP = float(os.environ.get("FS_PROF_CAP", 1.5))
# **끝 접선 어긋남**의 값 — 각 1도가 획 폭 1px당 얼마인가 (`_tang_pen`).
# 0.2면 30° 어긋난 끝이 폭 3px 획에서 18px² — 도형 한 장 값(`_MIN_GAIN` 6px)의
# 세 배다. 이음이 각으로 읽히는 것이 그만큼 비싸다는 뜻이고, 그 저울은 폭
# 프로파일 항과 같다.
_W_TANG = float(os.environ.get("FS_TANGENT", 0.2))
# **끝이 뾰족한 도형은 획이 아니다** — 놓인 폭 프로파일의 양끝이 중앙의 이만큼은
# 돼야 한다 (0 = 안 본다). 위 자들이 전부 폭을 한 수로 줄여 묻는 탓에 생긴
# 구멍이다: `placed_profile`이 가운데 80%만 재므로 "가운데는 원화 폭인데 끝이
# 물방울처럼 뾰족한" 도형이 테이퍼 게이트를 그대로 통과한다. 실측(표준 10장,
# `placed_widths`로 끝까지 재면) 획 도형의 **35%가 끝 폭이 중앙의 절반 미만**
# 이었고, 그것이 머리칼 가닥·옷 주름이 잎사귀로 읽히던 자리다.
#
# 사람 도안에는 이 문제가 없다 — 레퍼런스는 둥근사각(양끝 캡)을 회전·크기만
# 바꿔 반복해 긋는다. 게다가 한 획을 여러 장으로 나눌 때 **마디 사이 이음이
# 뾰족한 끝끼리 만나면** 그 자리가 렌더에서 렌즈꼴로 벌어진다 (실측 이음 틈
# 중앙 3.3px).
#
# 0.45는 회귀가 고른 값이다. 어휘 38종의 등방 분포로는 0.49(I_39)와
# 0.62(U_13) 사이가 비어 있어 0.55가 자연스러워 보이지만, **놓인 뒤**의 비는
# 비등방 스케일에 따라 움직인다 — 실측(01·04)에서 0.55와 0.45의 차는
# 뾰족한 끝 0.2% ↔ 2.2%(문법을 안 걸면 40%다)인데 도형은 2.0%,
# 이음 보수는 5% 적다. 어중간한 테이퍼(U_31 0.47 · U_03 0.48 · I_39 0.49)를
# 획으로 인정하는 쪽이 사람 획에도 가깝다 — 사람도 획 끝을 조금은 흘린다.
_STROKE_END = float(os.environ.get("FS_STROKE_END", 0.45))
# **몸통이 부푸는 도형은 획이 아니다** — 놓인 폭의 최대가 제 중앙의 이 배를
# 넘으면 쐐기·잎사귀다 (0 = 안 본다). `_STROKE_END`의 짝이고 자도 같은 닫힌
# 식이다 (`_bulge_ratio`).
#
# **부푸는 까닭이 도형이 아니라 비등방 스케일이라는 것이 요점이다.** 어휘
# 38종의 **등방** 배부름은 태반이 1.1 아래인데(U_62 1.07 · U_46 1.07),
# 실제로 놓인 것은 중앙 1.42 · p90 2.20이었다 (표준 01, 획 도형 862장 —
# U_62가 2.00 · U_46이 2.68로 섰다). 놓인 폭이 `2·반폭·|sx·sy| / |M·t̂|`라
# 중심선의 접선이 눌린 축을 향하는 자리에서 폭이 `|sx/sy|`배로 부풀기
# 때문이다. 그래서 등방 서술자로는 원리적으로 못 보고, 폭 프로파일 **벌점**
# 으로도 안 잡힌다 — 실측(01) `_W_PROF`를 3배로 올려도 p90이 2.22 → 2.13인데
# 도형이 +1.9%·이음 보수가 +6.7%다. 게이트라야 듣는다: 1.8에서 p90이
# 2.23 → 1.54(−31%) · 긴 획 끊김 −10.5% · 이음 틈 −7.4%이고 값은 도형
# +1.4%다.
#
# 1.8이라는 수는 레퍼런스 쪽 자(최대/최소를
# 라스터로 잰다)와 **눈금이 다르다** — 이쪽은 최대/중앙을 닫힌 식으로 잰다.
# 두 자를 직접 견줄 수는 없고, 여기서는 우리 분포의 꼬리(p90 2.2 이상)만
# 잘라 내는 자리로 세운 것이다.
_STROKE_BULGE = float(os.environ.get("FS_STROKE_BULGE", 1.8))


def _straight_dev(path: np.ndarray) -> float:
    """경로가 제 현에서 벗어나는 최대 거리 px (직선 판정의 자)."""
    p0, p2 = path[0], path[-1]
    chord = float(np.hypot(*(p2 - p0)))
    if chord < 1e-6:                       # 고리 — 현이 없으니 직선일 수 없다
        return float(np.hypot(*(path - p0).T).max())
    return float(np.abs(_cross2(p2 - p0, path - p0)).max() / chord)


def _is_straight(path: np.ndarray, wpx: float) -> bool:
    """**확실히 직선인가** — 막대·사각 계열을 쓸 자격 (`_STRAIGHT_DEV` 문서)."""
    return _straight_dev(path) <= max(0.5, _STRAIGHT_DEV * max(wpx, 1.0))


def _fit_path(plan: LayerPlan, sc: _Scorer, dt: np.ndarray, path: np.ndarray,
              wmed: float, color, ink: bool, left: int, forms: tuple,
              sid: int = -1, depth: int = 0, strict: bool = False,
              wcap: float = 0.0, wprof: np.ndarray | None = None,
              grammar: bool = False, it=None, skew_ok=None) -> int:
    """경로 하나를 획으로 — **곡선이 기본이고 막대는 직선일 때만**.

    ① 확실히 직선이면(`_is_straight`) 막대가 맞는 도형이다 ② 아니면 곡선 한
    장을 시도하고 ③ 안 되면 최대 이탈점에서 쪼개 재귀하며 ④ 끝까지 안 되면
    마디마다 다시 ①을 묻는다 (`_fit_segments` — 굽은 마디는 곡선으로).
    쪼개도 `sid`는 그대로 물려준다 — 한 획에서 나온 마디는 전부 한 그룹이라
    프루닝이 중간에서 자르지 못한다.

    `grammar`: 획 도형 문법(테이퍼·가늘기·폭·끝 뭉툭함)을 걸까 — **두 노선
    공통**이다 (`_try_curve`의 `line` 인자). 선은 모든 면 위에 마지막으로
    얹히므로 셀 노선에서도 잎사귀·쐐기가 그대로 보인다. 면 영역의 가는 잔여
    경로(`fill._fit_bars`)만 이 문법 밖이다 — 그쪽은 획이 아니라 채움이다.

    `strict`(line 노선): 마디의 한 스텝 연장을 **순개선일 때만** 받는다.
    `wcap`(line 노선): 마디 폭 상한 px — 병합 띠·교차 뭉치에서 dt가 준
    굵기를 그대로 긋지 않는다 (0 = 무제한). `wprof`는 경로 표본마다의 원화 띠
    폭이다 (`_prof_pen` — 쪼갤 때 경로와 같이 잘라 물려준다).

    `it`(`intent.StrokeIntent`): 이 획을 **어디서 끊을 것인가**의 의도. 쪼갤
    자리를 이탈 최대점이 아니라 의도된 각에서 고른다 (`intent` 문서). 경로와
    같은 길이라 쪼갤 때 함께 잘려 내려간다."""
    if left <= 0 or len(path) < 2:
        return 0
    if not _is_straight(path, wmed):
        arc = _try_curve(sc, forms, path, wmed, color, ink, sid,
                         line=grammar, wprof=wprof, skew_ok=skew_ok)
        if arc is not None:
            _, mfin = sc.score(arc[1])
            sc.commit(mfin)
            plan.layers.append(arc[1])
            return 1
        if depth < _SPLIT_DEPTH and len(path) >= _SPLIT_MIN:
            # 어느 도형도 안 맞는 불규칙 곡선 — 현에서 가장 먼 점이 마디다
            p0, p2 = path[0], path[-1]
            chord = max(float(np.hypot(*(p2 - p0))), 1e-6)
            dev = np.abs(_cross2(p2 - p0, path - p0)) / chord
            e = _SPLIT_MIN // 2
            # **각이 있으면 각에서 끊는다** — 이탈 최대점은 매끈한 호의
            # 한가운데라 거기서 끊으면 사람이 한 번에 긋는 굽음이 두 장으로
            # 갈리고 그 자리에 각이 선다 (`intent.split_index`)
            i = I.split_index(dev, it, e, len(path) - 1 - e)
            if e <= i <= len(path) - 1 - e and dev[i] > max(0.5, 0.2 * wmed):
                a = _fit_path(plan, sc, dt, path[:i + 1], wmed, color, ink,
                              left, forms, sid, depth + 1, strict, wcap,
                              None if wprof is None else wprof[:i + 1],
                              grammar, it.sub(0, i + 1) if it else None,
                              skew_ok)
                b = _fit_path(plan, sc, dt, path[i:], wmed, color, ink,
                              left - a, forms, sid, depth + 1, strict, wcap,
                              None if wprof is None else wprof[i:], grammar,
                              it.sub(i) if it else None, skew_ok)
                return a + b
    return _fit_segments(plan, sc, dt, path, wmed, color, ink, left, sid,
                         strict=strict, wcap=wcap, forms=forms, wprof=wprof,
                         grammar=grammar, it=it, skew_ok=skew_ok)


def _fit_segments(plan: LayerPlan, sc: _Scorer, dt: np.ndarray,
                  path: np.ndarray, wmed: float, color, ink: bool,
                  left: int, sid: int = -1, strict: bool = False,
                  wcap: float = 0.0, forms: tuple | None = None,
                  wprof: np.ndarray | None = None,
                  grammar: bool = False, it=None, skew_ok=None) -> int:
    """마디 사슬 (최후 수단) — 폭 비례 허용오차로 마디를 끊고 **마디마다 도형을
    고른다**: 그 마디의 원래 호가 직선이면 막대(A_22), 굽었으면 곡선이다.

    허용오차 0.7×폭: 이탈이 획 폭 안이면 획 몸통이 경로를 거의 그대로 덮고
    빈틈은 1px대라 안 보인다 — 마디 수(= 레이어 수)가 준다.

    마디를 끊는 자와 도형을 고르는 자가 다른 것이 요점이다. RDP는 **현**에서의
    이탈로 끊으므로 마디 사이 호는 늘 허용오차 안이지만, 그 안에서도 획 폭의
    반을 넘는 굽음은 렌더에서 각으로 읽힌다 (사용자 지적 ①). 그래서 마디마다
    곡선과 막대를 **같은 채점판에서 겨루게** 하고 비기면 곡선이 이긴다 —
    새 문턱을 세우지 않고 "웬만하면 곡선"이 성립한다.
    """
    x0, y0, _, _ = sc.roi
    n = 0
    eps = float(np.clip(_BAR_EPS * wmed, 1.0, 3.5))
    # 마디를 **의도된 각으로 끌어당긴다** — 개수는 그대로라 도형 수가 안
    # 바뀌고, 이음이 각 자리에 서므로 매끈한 굽음 한가운데가 안 꺾인다
    idx = I.snap_nodes(_rdp_idx(path, eps), it, wmed)
    # 마디가 획 예산을 넘으면 **허용오차를 키워** 줄인다 — 중간에서 잘라 점선을
    # 만드는 대신 굵직하게 긋는다. 오차 상한(폭의 8배)은 획이 경로를 아예 못
    # 따라가 `_descend`에서 통째로 기각되는 것을 막는 바닥이다
    while 0 < left < len(idx) - 1 and eps < 8.0 * max(wmed, 1.0):
        eps *= 1.5
        idx = I.snap_nodes(_rdp_idx(path, eps), it, wmed)
    # 그래도 넘치면 호길이 등분으로 간다 — 아래 반복문은 예산이 차면 그 자리에서
    # 멈추므로 남은 마디가 통째로 빠져 **획 뒷동강이 점선으로 사라진다**.
    # 등분은 마디가 굵어질 뿐 획 전체를 긋는다 (점선이 굵기보다 눈에 띈다)
    if 0 < left < len(idx) - 1:
        d = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(path, axis=0).T))])
        idx = I.snap_nodes(
            sorted(set(int(np.searchsorted(d, t)) for t in
                       np.linspace(0.0, float(d[-1]), left + 1))
                   | {0, len(path) - 1}), it, wmed)
    for i in range(len(idx) - 1):
        if n >= left:
            return n
        a, b = idx[i], idx[i + 1]
        p0, p1 = path[a], path[b]
        L = float(np.hypot(*(p1 - p0)))
        if L < 1.5:
            continue
        mid = (p0 + p1) / 2
        # 마디 폭 = 경로 주변 거리변환 중앙값 ×2. 반길이는 폭의 절반만큼 연장 —
        # 이웃 마디와 겹쳐 점선 틈을 없앤다 (같은 색 겹침은 공짜)
        samp = np.linspace(0, 1, max(3, int(L / 2)))[:, None]
        pts = (p0[None] * (1 - samp) + p1[None] * samp).astype(int)
        wpx = 2.0 * float(np.median(dt[pts[:, 0].clip(0, dt.shape[0] - 1),
                                       pts[:, 1].clip(0, dt.shape[1] - 1)]))
        if wcap:
            wpx = min(wpx, wcap)          # line 노선 — 병합 띠 굵기를 안 따른다
        wpx = max(wpx, 1.2) * _BAR_W
        # 굽은 마디는 곡선과 겨루게 한다 (`_try_curve`의 gate 없는 겨루기 모드)
        arc = path[a:b + 1]
        best_c = None
        if forms is not None and not _is_straight(arc, wpx) and len(arc) >= 10:
            best_c = _try_curve(sc, forms, arc, wpx, color, ink, sid, race=True,
                                line=grammar, skew_ok=skew_ok,
                                wprof=None if wprof is None else wprof[a:b + 1])
        theta = float(np.arctan2(p1[0] - p0[0], p1[1] - p0[1]))
        # 막대 도형은 **목표 폭이 고른다** — 둥근사각의 폭 눈금이 최소 도형 폭
        # 그대로라 가는 획에서 목표를 못 맞힌다 (`vocabulary.bar_for`)
        bname, bext, brot = bar_for(sc.cat, sc.upp, wpx)
        lay = _layer(bname, x0 + mid[1], y0 + mid[0],
                     L / 2 + wpx * 0.5, wpx / 2, theta, 0.0,
                     color, sc.upp, sc.w, sc.h,
                     label="ink" if ink else "cel", stroke=sid,
                     ext=bext, rot_off=brot)
        # **조종 항** — 이 마디의 제 경로 조각. 하강이 픽셀 이득만 보고
        # 자리를 잡으면 이웃 마디와 만나는 자리에서 방향이 꺾인다
        # (`chain.anchor_pen`). 마디마다 제 조각의 양끝·접선에 맞추면 이음이
        # 저절로 맞는다 — 이웃을 안 보므로 쪼개기 순서와 무관하다.
        # **획의 문법 안에서만** 건다: 면 영역의 가는 잔여 경로
        # (`fill._fit_bars`)는 획이 아니라 채움이라 이음이 없다 — 거기까지
        # 걸면 채움이 잔여를 덮는 대신 경로 끝을 좇는다
        steer = (chain.steer(sc.cat, sc,
                             np.stack([arc[:, 0] + y0, arc[:, 1] + x0], axis=1),
                             wpx, sc.w, sc.h, lay) if grammar else None)
        gain, q = _descend(sc, lay, color, passes=3, steer=steer)
        if best_c is not None and best_c[0] >= gain:   # 비기면 곡선이 이긴다
            if best_c[0] < _MIN_GAIN * 0.5:
                continue
            _, mfin = sc.score(best_c[1])
            sc.commit(mfin)
            plan.layers.append(best_c[1])
            n += 1
            continue
        if gain < _MIN_GAIN * 0.5:
            continue
        # 양자화(스케일 스텝 0.01 = 1.7px)로 마디가 짧아지면 틈이 남는다 —
        # 한 스텝 연장이 크게 손해가 아니면 이어붙인다. line 노선(strict)은
        # **순개선일 때만** — "손해 2.0 안"의 공짜 연장이 획 끝을 교차선
        # 너머로 밀어내는 침범의 한 축이었다 (밴드 한정 아래에서 제 밴드 안
        # 연장은 retrace 이득이 있어 여전히 붙고, 밖 연장은 이득 0이라 죽는다)
        q2 = Layer(**{**q.__dict__})
        # 늘리는 축은 **길이 축**이다 — 가는 도형은 로컬 y가 길이라(rot_off 90)
        # sx를 밀면 획이 길어지는 게 아니라 굵어진다
        if brot:
            q2.sy = round(q2.sy + 0.01, 4)
        else:
            q2.sx = round(q2.sx + 0.01, 4)
        s2 = sc.score_val(q2)
        if (s2 > gain + 1e-6) if strict else (s2 >= gain - 2.0):
            q, gain = q2, s2
        _, mfin = sc.score(q)
        sc.commit(mfin)
        plan.layers.append(q)
        n += 1
    return n


def _path_worth(sc: _Scorer, path: np.ndarray, wmed: float) -> float:
    """이 획이 담고 있는 값 — 경로를 제 폭으로 그은 띠 ∩ 잔여 (배치 전 선별).

    실제로 놓아 본 뒤 무르는 대신 **놓기 전에** 묻는다: 획은 한 장이 아니라
    한 획(도형 1~3장)이 단위라, 놓고 나서 무르려면 잔여를 되돌려야 한다.
    """
    m = np.zeros(sc.residual.shape, np.uint8)
    pts = np.stack([path[:, 1], path[:, 0]], axis=1).round().astype(np.int32)
    cv2.polylines(m, [pts], False, 1, max(1, int(round(wmed))))
    return sc.worth_of(m.astype(bool) & sc.residual)


_FORMS: dict = {}      # 프로세스 1회 계측 (수요 적응 재생성이 다시 안 돌게)


def _placed_form(i: int, sx: float, sy: float,
                 skew: float = 0.0) -> tuple[float, float, float]:
    """도형 i를 (sx, sy, skew)로 놓았을 때의 (테이퍼, 폭/길이, 폭 [캔버스 유닛]).

    식은 `descriptor.placed_profile` 하나다 — 배치가 고르는 폭과 계측이 재는
    폭이 같은 자여야 "폭을 맞췄다"가 검증 가능한 말이 된다.

    **전단은 반드시 여기까지 와야 한다** (§6): 획 도형 문법 넷은 전부 놓인 뒤의
    폭을 묻는데, 전단은 접선이 눌린 축을 향하는 자리에서 폭을 바꾼다. 자에
    전단을 안 넣으면 "중심선은 잘 맞는데 폭이 찌그러진" 후보가 게이트를 그대로
    통과한다.
    """
    U, W = _FORMS.get("u"), _FORMS.get("w")
    if U is None or W is None:
        return 1.0, 0.0, 0.0
    taper, slim, wmed, _L = placed_profile(U[i], W[i], sx, sy, skew)
    return taper, slim, wmed


def _prof_pen(i: int, sx: float, sy: float, wtgt: np.ndarray,
              upp: float, skew: float = 0.0) -> float:
    """**폭 프로파일 어긋남**을 px² 넓이로 — 후보 순위의 항 (`_W_PROF` 문서).

    도형 i를 (sx, sy)로 놓았을 때의 마디별 폭(닫힌 식, `descriptor.
    placed_widths`)을 원화 띠의 폭 `wtgt`(경로 호길이 비율로 준 px 배열)와
    나란히 세워 차의 절대값을 길이로 적분한다. 단위가 px²라 채점 점수(px)와
    바로 더할 수 있다 — 새 저울을 세우지 않는다.

    `_FORMS`가 도형마다 뒤집은 사본을 함께 들고 있으므로(`_stroke_forms`)
    폭 배열도 같이 뒤집혀 있다 — 진행 방향이 맞춰진 채로 비교된다.
    """
    U, W = _FORMS.get("u"), _FORMS.get("w")
    if U is None or W is None or not len(wtgt):
        return 0.0
    w, mid, length = placed_widths(U[i], W[i], sx, sy, skew)
    if length <= 1e-9 or len(w) < 2:
        return 0.0
    tgt = np.interp(mid, np.linspace(0.0, 1.0, len(wtgt)), wtgt)
    return float(np.mean(np.abs(w / max(upp, 1e-9) - tgt))) * (length / max(upp, 1e-9))


def _tang_pen(i: int, sx: float, sy: float, th: float,
              X: np.ndarray, wpx: float, skew: float = 0.0) -> float:
    """**끝 접선 어긋남** — 도형의 양끝이 경로와 얼마나 다른 방향으로 나가나.

    아핀 맞춤(`_affine_fit`)은 대응점 **거리**만 줄인다 — 끝에서 어느 방향으로
    나가는지는 안 본다. 그런데 한 획을 여러 장으로 나눠 그으면 이웃 마디가
    만나는 자리에서 보이는 것이 바로 그 방향이다: 두 마디가 각각 경로 접선에
    맞으면 이음은 **저절로** 매끈하고, 하나라도 어긋나면 그 자리가 각으로
    읽힌다 (실측 표준 10장 이음각 중앙 24~28°).

    그래서 이음을 나중에 밀어서 맞추는 대신 **도형을 고를 때** 묻는다. 값은
    각(도) × 획 폭이라 단위가 px²에 가깝고, 폭 프로파일 항과 같은 저울에 선다
    (`_W_TANG`). 새 계측이 없다 — 중심선은 이미 `_FORMS`에 있다.
    """
    U = _FORMS.get("u")
    if U is None or len(X) < 3:
        return 0.0
    u = U[i]
    k = max(1, len(u) // 8)
    c, sn = np.cos(th), np.sin(th)
    R = np.array([[c, -sn], [sn, c]], np.float64)

    def d(v):
        v = v * np.array([sx, sy], np.float64)
        if skew:                           # 전단 — 회전 전 (`geometry._poly_px`)
            v = np.array([v[0] + skew * v[1], v[1]])
        v = R @ v
        n = float(np.hypot(*v))
        return v / n if n > 1e-12 else None

    kx = max(1, len(X) // 8)
    out = 0.0
    for a, b in ((d(u[k] - u[0]), X[kx] - X[0]),
                 (d(u[-1] - u[-1 - k]), X[-1] - X[-1 - kx])):
        nb = float(np.hypot(*b))
        if a is None or nb < 1e-12:
            continue
        cos = float(np.clip(np.dot(a, b / nb), -1.0, 1.0))
        out += float(np.degrees(np.arccos(cos)))
    return _W_TANG * max(wpx, 1.0) * out


def _end_ratio(i: int, sx: float, sy: float, skew: float = 0.0) -> float:
    """도형 i를 (sx, sy, skew)로 놓았을 때 **양끝 폭 중 가는 쪽 / 중앙 폭**
    (`_STROKE_END`).

    `placed_profile`이 일부러 빼는 양끝을 여기서는 그것만 본다 — 두 자가 같은
    닫힌 식(`descriptor.placed_widths`)을 쓰므로 새 계측이 붙지 않는다.
    """
    U, W = _FORMS.get("u"), _FORMS.get("w")
    if U is None or W is None:
        return 1.0
    w, _mid, length = placed_widths(U[i], W[i], sx, sy, skew)
    if length <= 1e-9 or len(w) < 5:
        return 1.0
    med = float(np.median(w))
    return float(min(w[0], w[-1])) / med if med > 1e-9 else 1.0


def _bulge_ratio(i: int, sx: float, sy: float, skew: float = 0.0) -> float:
    """도형 i를 (sx, sy, skew)로 놓았을 때 **최대 폭 / 중앙 폭** (`_STROKE_BULGE`).

    `_end_ratio`가 끝을 묻는다면 이쪽은 **몸통이 부푸는가**를 묻는다. 부푸는
    까닭이 도형 자체가 아니라 **비등방 스케일**이라는 것이 요점이다: 놓인 폭은
    `2·반폭·|sx·sy| / |M·t̂|`이라(`descriptor.placed_widths`) 중심선의 접선이
    눌린 축을 향하는 자리에서 폭이 `|sx/sy|` 배로 부푼다. 그래서 등방 서술자
    (`ShapeDesc.taper`)로는 못 본다 — 어휘 38종의 등방 배부름은 태반이 1.1
    아래인데 실제로 놓인 것은 중앙 1.42 · p90 2.2였다.
    """
    U, W = _FORMS.get("u"), _FORMS.get("w")
    if U is None or W is None:
        return 1.0
    w, _mid, length = placed_widths(U[i], W[i], sx, sy, skew)
    if length <= 1e-9 or len(w) < 5:
        return 1.0
    med = float(np.median(w))
    return float(w.max()) / med if med > 1e-9 else 1.0


def _width_shape_ok(i: int, lay) -> bool:
    """이 **놓인 레이어**의 폭 모양이 획의 자를 지키나 (끝 뭉툭함·몸통 배부름).

    후보 게이트가 씨앗에 거는 두 절대 자(`_STROKE_END`·`_STROKE_BULGE`)를
    같은 닫힌 식으로 다시 묻는다 — 새 문턱이 아니라 **같은 자를 늦게 한 번 더**
    거는 것이다. 하강이 자세를 민 뒤에도 성립해야 "폭을 맞췄다"가 참이 된다.
    """
    if _end_ratio(i, lay.sx, lay.sy, lay.skew) < _STROKE_END:
        return False
    return not (_STROKE_BULGE > 0.0
                and _bulge_ratio(i, lay.sx, lay.sy, lay.skew) > _STROKE_BULGE)


def _raster_iso(sh, size: int) -> tuple[np.ndarray, float]:
    """도형을 **등방**으로 라스터한다 — (마스크, 로컬→px 배율).

    `CatShape.rasterize`는 bbox를 정사각으로 늘려서 축마다 배율이 다르다 —
    도형 대조(IoU)에는 그래도 되지만 **폭을 재는 데는 못 쓴다** (긴 축이
    눌린 도형에서 폭이 축 방향에 따라 갈린다).
    """
    pts = np.concatenate(sh.loops, axis=0)
    lo = pts.min(axis=0)
    span = np.maximum(pts.max(axis=0) - lo, 1e-6)
    s = (size - 3) / float(span.max())
    acc = np.zeros((size, size), np.uint8)
    for loop in sh.loops:
        q = (loop - lo) * s + 1.0
        q[:, 1] = size - 1 - q[:, 1]
        one = np.zeros_like(acc)
        cv2.fillPoly(one, [np.round(q).astype(np.int32)], 1)
        acc ^= one
    return acc.astype(bool), s


def _stroke_forms(cat: Catalog) -> tuple[list[str], np.ndarray | None]:
    """획 도형의 **중심선** — (이름 목록, (S,N,2) 배열). 로컬 좌표 × 스케일 단위.

    래스터 뼈대를 호길이 등간격 `_FORM_N`점으로 뽑는다. 뒤집은 사본을 뒤에
    이어 붙여 경로 진행 방향 양쪽을 한 번에 푼다 (대응점 순서가 반대인 맞춤).
    중심선이 한 가닥이 아닌 도형은 어휘에서 빠진다.

    중심선과 함께 그 자리의 **반폭**도 재 `_FORMS["w"]`에 둔다 (등방 래스터의
    거리변환, 같은 단위). 놓인 뒤의 폭 프로파일을 닫힌 식으로 얻는 데 쓴다 —
    `_placed_form` 문서.
    """
    if _FORMS:
        return _FORMS["names"], _FORMS["u"]
    names, forms, wids = [], [], []
    for name in stroke_vocab(cat):
        try:
            sh = cat[name]
        except KeyError:
            continue
        m = sh.rasterize(_FORM_RASTER)
        if not m.any():
            continue
        dt = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 3)
        skel = _thin(m)
        if not skel.any():
            continue
        wmed = 2.0 * float(np.median(dt[skel]))
        paths = _paths(_prune_spurs(skel, max(3.0, 1.2 * wmed)))
        if len(paths) != 1 or len(paths[0][0]) < 10:
            continue
        p = paths[0][0]
        # 래스터 px → 로컬 좌표 (rasterize와 같은 매핑, y 뒤집힘 복원)
        pts_all = np.concatenate(sh.loops, axis=0)
        lo = pts_all.min(axis=0)
        span = np.maximum(pts_all.max(axis=0) - lo, 1e-6)
        lx = p[:, 1] / (_FORM_RASTER - 1) * span[0] + lo[0]
        ly = (_FORM_RASTER - 1 - p[:, 0]) / (_FORM_RASTER - 1) * span[1] + lo[1]
        # 반폭은 **등방** 래스터에서 — 같은 중심선 위 같은 자리를 되짚는다
        mi, si = _raster_iso(sh, _FORM_RASTER)
        dti = cv2.distanceTransform(mi.astype(np.uint8), cv2.DIST_L2, 5)
        ix = np.clip(((lx - lo[0]) * si + 1.0).round().astype(int), 0, _FORM_RASTER - 1)
        iy = np.clip((_FORM_RASTER - 1 - ((ly - lo[1]) * si + 1.0)).round().astype(int),
                     0, _FORM_RASTER - 1)
        hw = dti[iy, ix] / si                 # 로컬 단위 반폭
        d = np.concatenate([[0.0], np.cumsum(np.hypot(
            *np.diff(np.stack([lx, ly], axis=1), axis=0).T))])
        t = np.linspace(0.0, float(d[-1]), _FORM_N)
        names.append(name)
        forms.append(_resample(np.stack([lx, ly], axis=1), _FORM_N)
                     * UNITS_PER_SCALE)
        wids.append(np.interp(t, d, hw) * UNITS_PER_SCALE)
    if not forms:
        _FORMS.update(names=[], u=None, w=None)
        return [], None
    u = np.asarray(forms, np.float64)
    ww = np.asarray(wids, np.float64)
    _FORMS.update(names=names + names, u=np.concatenate([u, u[:, ::-1]], axis=0),
                  w=np.concatenate([ww, ww[:, ::-1]], axis=0))
    return _FORMS["names"], _FORMS["u"]


def _affine_fit(U: np.ndarray, X: np.ndarray):
    """중심선 U(S,N,2)를 목표 경로 X(N,2)에 **회전 + 비등방 스케일**로 맞춘다.

    게임 레이어의 변환은 `pts×(sx,sy) → 기울기 → 회전`이라 임의 아핀까지 낼 수
    있지만 **기울기는 안 쓴다** — 주입 경로가 레코드 오프셋을 못 찾아 skew를
    0으로 눌러 쓰기 때문이다. 남는 자유도 R(θ)·diag(sx,sy)의 최소제곱 해는
    닫힌 형태다: 최적 sx·sy를 대입한 잔차가 θ에 대해 K0+K1·cos2θ+K2·sin2θ 꼴이라
    2θ = atan2(K2, K1). 격자 탐색이 필요 없어 어휘를 수십 종으로 넓혀도 싸다.

    반환 (θ, sx, sy, 잔차제곱) — 전부 (S,).
    """
    Uc = U - U.mean(axis=1, keepdims=True)
    Xc = X - X.mean(axis=0, keepdims=True)
    P = np.maximum((Uc[..., 0] ** 2).sum(axis=1), 1e-12)
    Q = np.maximum((Uc[..., 1] ** 2).sum(axis=1), 1e-12)
    M = np.einsum("sni,nj->sij", Uc, Xc)
    A, B, C_, D = M[:, 0, 0], M[:, 0, 1], M[:, 1, 0], M[:, 1, 1]
    K0 = (A * A + B * B) / (2 * P) + (D * D + C_ * C_) / (2 * Q)
    K1 = (A * A - B * B) / (2 * P) + (D * D - C_ * C_) / (2 * Q)
    K2 = A * B / P - C_ * D / Q
    th = 0.5 * np.arctan2(K2, K1)
    c, s = np.cos(th), np.sin(th)
    sx = (c * A + s * B) / P
    sy = (c * D - s * C_) / Q
    res = float((Xc ** 2).sum()) - (K0 + K1 * np.cos(2 * th) + K2 * np.sin(2 * th))
    return th, sx, sy, np.maximum(res, 0.0)


# 튜닝 계측용 (동작에는 영향 없음). `skew_*`는 §14의 자다:
#   skew_cand  전 아핀 후보를 몇 벌 지어 봤나 (실제로 놓인 장수는 `affine.report`)
_CURVE_STATS = {"paths": 0, "tried": 0, "ok": 0, "short": 0, "flat": 0,
                "nofit": 0, "lowgain": 0, "notline": 0, "skew_cand": 0,
                #   skew_guard  하강 뒤 폭 모양 자에 걸려 전단을 되돌린 횟수
                "skew_guard": 0}


def _use_skew(flag) -> bool:
    """이 자리에서 전단 후보를 지을까 — `None`이면 노선 정책의 기본값."""
    return _policy.skew_stroke_default() if flag is None else bool(flag)


def _form_ext_y(i: int) -> float:
    """어휘 `i` 중심선의 로컬 y 반경 — `affine.step_matters`가 쓰는 지렛대.

    전단은 로컬 y에 비례해 x를 민다. 도형이 y로 안 뻗어 있으면(곧고 납작한
    막대꼴) 한 스텝이 미는 거리가 이동 양자에도 못 미쳐 이 축이 무동작이다.
    """
    U = _FORMS.get("u")
    if U is None or i >= len(U):
        return 0.0
    return float(np.abs(U[i][:, 1]).max())


def _affine_full(U: np.ndarray, X: np.ndarray):
    """전 아핀 맞춤 — `affine.fit_full`의 얇은 껍데기 (순환 임포트 회피)."""
    return A.fit_full(U, X)


def _form_offsets() -> np.ndarray | None:
    """어휘의 **반폭 벡터** `(S,N,2)` — 중심선 법선 × 반폭. 프로세스 1회.

    띠 맞춤(`affine.fit_ribbon`)의 원본 쪽 대응점이다. `_FORMS`의 중심선·반폭이
    이미 같은 단위(로컬 × `UNITS_PER_SCALE`)라 여기서 새로 재는 것이 없다 —
    법선만 얹는다. 뒤집은 사본도 함께 들어 있으므로(`_stroke_forms`) 법선도
    같이 뒤집혀, 진행 방향이 맞춰진 채로 짝이 선다.
    """
    d = _FORMS.get("d")
    if d is not None:
        return d
    U, W = _FORMS.get("u"), _FORMS.get("w")
    if U is None or W is None:
        return None
    d = A.normals(U) * np.asarray(W, np.float64)[..., None]
    _FORMS["d"] = d
    return d


def _affine_ribbon(U: np.ndarray, X: np.ndarray, hw: np.ndarray):
    """**띠** 맞춤 — `affine.fit_ribbon`의 껍데기 (순환 임포트 회피).

    `hw`는 목표 경로 각 점의 반폭(캔버스 유닛)이다. 어휘 쪽 반폭 벡터가 없으면
    (`_FORMS`가 안 섰으면) 중심선 맞춤으로 물러난다.
    """
    D = _form_offsets()
    if D is None:
        return A.fit_full(U, X)
    return A.fit_ribbon(U, D, X, A.normals(X) * np.asarray(hw, np.float64)[:, None])


def _try_curve(sc: _Scorer, forms: tuple, path: np.ndarray, wpx: float,
               color, ink: bool = False, sid: int = -1, race: bool = False,
               line: bool = False, gate: bool = True,
               wprof: np.ndarray | None = None,
               skew_ok=None) -> tuple[float, Layer] | None:
    """경로 전체를 곡선 도형 **한 장**으로 — 되면 (점수, 레이어), 아니면 None.

    `race=True`는 **겨루기 모드**다 (`_fit_segments`의 마디): 자격 게이트를
    묻지 않고 최선 후보를 그대로 돌려준다 — 그 자리의 대안은 "막대냐 곡선이냐"
    뿐이라 자격이 아니라 점수로 가리는 것이 맞다. 게이트는 "이 경로를 통째로
    한 장에 담아도 되나"를 묻는 자라서 마디에는 안 맞는다.

    `gate=False`는 그 자격 판정을 **통째로** 뺀다 — 후보 경쟁(`candidates`)이
    쓰는 길이다. 거기서는 곡선 한 장이 막대 사슬·DP 분절과 **양자화된 실제
    렌더**에서 겨루므로, 고정 문턱으로 미리 떨어뜨리면 "약간 더 스필이 나지만
    이음새가 줄고 도형 수도 주는" 곡선을 볼 기회 자체가 없어진다.

    `line=True`는 후보를 **선에 적합한 도형만**으로 좁힌다 (획 도형 문법,
    `_fit_path`의 `grammar` — **두 노선 공통**이다). 자는 넷이고 전부 놓인
    뒤의 모습을 본다: 테이퍼(`_STROKE_TAPER`) · 가늘기(`_STROKE_SLIM`) ·
    원화 띠 대비 폭(`_STROKE_WMAX`) · **끝 뭉툭함**(`_STROKE_END`). 그 위에
    폭 프로파일(`_W_PROF`)과 끝 접선(`_W_TANG`)이 순위를 매긴다.
    겨루기 모드에서도 그대로 건다: 막대와 겨루는 자리라도 잎사귀·쐐기는
    획이 아니고, 원화 띠보다 훨씬 굵은 도형은 그 자리의 획이 아니라 덩어리다.
    """
    _CURVE_STATS["paths"] += 1
    width_fit = prof_fit = line
    min_w = 2.0 * _min_span(sc.upp)
    wtgt = max(wpx, min_w)
    # 목표 폭 프로파일 — 원화 띠 폭을 **어휘가 실제로 낼 수 있는 바닥**과 이
    # 획의 폭 상한 사이로 눌러 둔다 (`vocabulary.min_stroke_width_px` 문서).
    # 바닥이 막대 폭(2.79px@h=1961)이던 동안에는 원화가 1.9px인 자리에서
    # 목표가 통째로 상수 2.79로 눌려, 이 항이 "얼마나 굵지 못한가"를 재고
    # 굵기 변화(리듬)를 아예 못 봤다. 상한 쪽도 같은 바닥을 쓴다 — 병합 띠·
    # 교차 뭉치에서 dt가 준 굵기를 목표로 삼으면 그 자리만 도형이 부푼다
    wfloor = min_stroke_width_px(sc.cat, sc.upp)
    wt = None
    if prof_fit and wprof is not None and len(wprof) >= 3:
        wt = np.clip(np.asarray(wprof, np.float64), wfloor,
                     _PROF_CAP * max(wpx, wfloor))
        if len(wt) >= 5:                   # dt 표본의 ±1px 톱니를 편다
            k = np.ones(5) / 5.0
            wt = np.convolve(np.pad(wt, 2, mode="edge"), k, "valid")
    names, U = forms
    if U is None or len(path) < 10:
        _CURVE_STATS["short"] += 1
        return None
    p0, p2 = path[0], path[-1]
    chord = np.hypot(*(p2 - p0))
    dev = np.abs(_cross2(p2 - p0, path - p0)) / max(chord, 1e-6)
    if gate and not race and (float(dev.max()) < max(1.2, 0.35 * wpx)
                              or chord < 2.5 * wpx):
        _CURVE_STATS["flat"] += 1
        return None                        # 거의 직선 — 막대가 낫다
    # 경로를 **캔버스 유닛**으로 옮겨 맞춘다 (레이어 좌표계와 같은 계).
    # ROI-로컬 px를 그대로 쓰면 ROI 원점만큼 어긋난다 (옛 호 배치의 버그)
    rx0, ry0, _, _ = sc.roi
    X = _resample(np.stack([(rx0 + path[:, 1] - sc.w / 2) * sc.upp,
                            (sc.h / 2 - (ry0 + path[:, 0])) * sc.upp],
                           axis=1), _FORM_N)
    th, sx, sy, res = _affine_fit(U, X)
    # ── **전 아핀 후보를 기존 후보 옆에 세운다** (§3·§6) ──────────────
    # 기존 `skew=0` 안은 하나도 안 지운다 (§5). 여기서 하는 일은 어휘마다
    # "전단까지 써서 맞춘 자세"를 **한 벌 더** 만들어 같은 줄에 세우는 것뿐이고,
    # 순위·게이트·하강·후보 경쟁이 전부 두 벌을 같은 자로 견준다.
    #
    # 더하는 자리를 좁게 잡는다 (§4):
    # - 회전 대칭이 큰 도형(원·고리)은 전단해 봐야 회전 + 비등방 스케일과
    #   같은 상이라 자유도만 겹친다 (`affine.skew_useful`)
    # - 양자화해서 0이 되는 전단은 기존 후보와 **같은 레이어**라 뺀다
    # - 맞춘 전단이 이동 양자(0.5유닛)도 못 미는 자리는 무동작이라 뺀다
    #   (`affine.shear_visible`. **한 스텝이 아니라 맞춘 크기로 묻는다** —
    #    씨앗의 값은 큰 전단에 있으므로 한 스텝으로 물으면 획 어휘가 통째로
    #    걸려 후보가 하나도 안 선다: 획 도형의 `sy`는 폭이라 0.1 언저리다)
    fidx = np.arange(len(names))           # 후보 → 어휘 색인
    skw = np.zeros(len(names), np.float64)
    if _use_skew(skew_ok):
        # **씨앗은 띠를 맞춘다** — 중심선만 맞추면 전단이 폭을 깎아 중심선을
        # 싸게 사는 지름길이 열린다 (`affine.fit_ribbon` 문단). 목표 반폭은
        # 순위 항이 쓰는 그 프로파일(`wt`)이고, 없으면 이 획의 폭 하나다 —
        # 새 자를 안 세운다.
        hw = (np.interp(np.linspace(0.0, 1.0, _FORM_N),
                        np.linspace(0.0, 1.0, len(wt)), wt)
              if wt is not None and len(wt) >= 2
              else np.full(_FORM_N, wtgt, np.float64))
        hw = 0.5 * hw * sc.upp                 # px 폭 → 캔버스 유닛 반폭
        rot2, sx2, sy2, sk2, res2 = (
            _affine_full(U, X) if _SKEW_FIT == "center"
            else _affine_ribbon(U, X, hw))
        keep = []
        for i in range(len(names)):
            k = A.q_skew(sk2[i])
            if k == 0.0 or not A.representable(k):
                continue
            if not np.isfinite(res2[i]) or abs(sx2[i]) < 0.01 or abs(sy2[i]) < 0.01:
                continue
            if not A.skew_useful(sc.cat, names[i]):
                continue
            if not A.shear_visible(k, sy2[i], _form_ext_y(i)):
                continue
            # **전단이 획의 폭 모양을 나쁘게 만들면 안 된다** (§6의 금지 조항:
            # "centerline 오차만 줄이고 획 폭이 찌그러지는 후보는 금지").
            # 새 문턱을 안 세운다 — **같은 도형을 전단 없이 놓았을 때**와
            # 견줘, 끝이 더 뾰족해지거나 몸통이 더 부풀면 뺀다. 절대 자
            # (`_STROKE_END`·`_STROKE_BULGE`)는 그대로 뒤에서 또 묻는다.
            #
            # 이 자가 없으면 전단이 폭을 무너뜨린다: 놓인 폭이
            # `2·반폭·|sx·sy| / |M·t̂|`인데 전단은 `det M`을 안 바꾸고 `|M·t̂|`만
            # 키우므로, 큰 전단은 획을 슬리버로 만든다. 실측(표준 11장, 이 자
            # 없이): 채택 전단의 중앙이 **1.44**·p90 5.6·최대 14.25까지 갔고
            # 뾰족한 끝이 0.029 → 0.178(+508%) · 밴드 밖 스필 +117%였다.
            if (_end_ratio(i, sx2[i], sy2[i], k)
                    < _end_ratio(i, sx[i], sy[i], 0.0) - 1e-9):
                continue
            if (_bulge_ratio(i, sx2[i], sy2[i], k)
                    > _bulge_ratio(i, sx[i], sy[i], 0.0) + 1e-9):
                continue
            keep.append(i)
        if keep:
            ki = np.asarray(keep, int)
            _CURVE_STATS["skew_cand"] += len(ki)
            fidx = np.concatenate([fidx, ki])
            skw = np.concatenate([skw, np.array(
                [A.q_skew(sk2[i]) for i in keep], np.float64)])
            th = np.concatenate([th, np.radians(rot2[ki])])
            sx = np.concatenate([sx, sx2[ki]])
            sy = np.concatenate([sy, sy2[ki]])
            res = np.concatenate([res, res2[ki]])
    ok = (np.abs(sx) >= 0.01) & (np.abs(sy) >= 0.01) & np.isfinite(res)
    if not ok.any():
        _CURVE_STATS["nofit"] += 1
        return None
    _CURVE_STATS["tried"] += 1
    # 잔차 순으로 보되, line 노선은 **선에 적합한 도형만** 후보에 남긴다 —
    # 놓인 뒤의 폭 프로파일로 거른다 (`_STROKE_TAPER`·`_STROKE_SLIM`). 걸러 낸
    # 만큼 뒤에서 채워야 후보 수가 안 줄어 어휘가 좁아지지 않는다
    if len(fidx) > len(names):
        # **줄을 띠 잔차로 세운다.** 씨앗이 최소화한 것이 서로 다르므로
        # (전단 후보는 띠, `skew=0` 후보는 중심선) 중심선 잔차로 세우면 띠
        # 후보가 제 강점을 못 보여 준 채 상위 `_CURVE_TOP` 밖으로 밀린다 —
        # 실측(표준 3장): 중심선으로 줄을 세우면 띠 후보를 569·536·898벌
        # 지어도 상위 `_CURVE_TOP`에 거의 못 든다 — 띠 후보는 정의상 중심선
        # 최적이 아니라 제 `fit_full` 짝보다 중심선 잔차가 크고, 어휘가
        # 도형마다 뒤집은 사본까지 들고 있어(`_stroke_forms`) 경쟁자가 76벌이다.
        #
        # 띠 잔차는 두 후보를 **같은 자**로 잰다: 중심선 어긋남 + λ×폭 어긋남.
        # 그래서 전단이 이기려면 "폭까지 맞추면서 중심선도 낫다"를 보여야 한다.
        # 동점은 여전히 `|skew|`가 작은 쪽이 이긴다 (§5).
        _D = _form_offsets()
        rr = (res if _D is None else A.ribbon_res_of(
            U[fidx], _D[fidx], X, hw[:, None] * A.normals(X),
            A.linear_batch(th, sx, sy, skw)))
        rank = np.lexsort((np.abs(skw), np.where(ok, rr, np.inf)))
    else:
        rank = np.argsort(np.where(ok, res, np.inf))
    order, n_shape = [], 0
    # 가늘기 자는 **이 호가 요구하는 비**보다 엄할 수 없다 — 폭 wtgt를 호
    # 길이만큼 긋는 도형의 폭/길이는 정확히 wtgt/L이라, 절대 자(0.065)를 그대로
    # 두면 31px보다 짧은 호(폭 2px)는 어떤 도형도 못 넘어 막대 사슬로
    # 떨어진다. 실측(G1~G5) 짧은 마디(<15px) 이음의 30°대 지그재그가 이
    # 자리다. 상수가 아니라 호에서 유도한다
    arc_len = float(np.hypot(*np.diff(path, axis=0).T).sum())
    slim_lim = max(_STROKE_SLIM, wtgt / max(arc_len, 1.0))
    for i in rank:
        if not ok[i] or len(order) >= _CURVE_TOP:
            break
        # 획 도형 문법은 **놓인 뒤의 폭**을 묻는다 — 전단이 그 폭을 바꾸므로
        # 자에도 전단이 들어가야 한다 (§6: "중심선 오차만 줄이고 획 폭이
        # 찌그러지는 후보는 금지"). 넷 다 같은 닫힌 식을 쓴다
        fi, kk = int(fidx[i]), float(skw[i])
        if line:
            tp, sl, wu = _placed_form(fi, float(sx[i]), float(sy[i]), kk)
            if tp > _STROKE_TAPER or sl > slim_lim:
                n_shape += 1
                continue
            # **폭 충실도** — 놓인 폭이 원화 띠보다 이만큼 넘게 굵으면 획이
            # 아니라 덩어리다 (`_STROKE_WMAX`). 바닥은 게임이 낼 수 있는
            # 최소 도형 폭이다 — 그보다 가는 띠를 그리는 데 드는 초과는
            # 강제된 것이라 벌할 수 없다 (`scoring._PEN_LINE`의 배수와 같은 수)
            if width_fit and wu / sc.upp > _STROKE_WMAX * wtgt:
                n_shape += 1
                continue
            # **폭에는 바닥도 있어야 한다** — 전단이 뚫는 새 길이다.
            #
            # `_STROKE_WMAX`는 천장만 본다 ("원화 띠보다 이만큼 넘게 굵으면
            # 덩어리다"). 전단이 없던 동안에는 바닥이 필요 없었다: 놓인 폭이
            # `2·반폭·|sx·sy| / |M·t̂|`인데 `M = diag(sx,sy)`면 어휘의 최소 폭이
            # 스스로 바닥이었다. 전단은 `det M`을 안 바꾸고 `|M·t̂|`만 키우므로
            # **폭을 얼마든지 가늘게 만든다** — 그런데 맞춤이 최소화하는 것은
            # 중심선 잔차뿐이라, 전단이 "중심선을 싸게 맞추고 몸통을 희생하는"
            # 지름길이 된다 (§6이 금지한 그 거래).
            #
            # 실측(표준 11장, 바닥 없이 구운 판의 획 도형 13,092장): 원화 띠가
            # ~2px인데 놓인 폭 중앙이 |k| 0.5~2에서 1.78px · 2~6에서 1.17px ·
            # 6 위에서 **0.57px**로 무너지고, 대신 마디 길이가 25.7 → 74px로
            # 늘었다. 한 장이 더 긴 경로를 맡은 것이 아니라 획이 실오라기가 된
            # 것이다.
            #
            # 바닥은 새 상수가 아니라 **천장의 거울**이다 (`1 / _STROKE_WMAX`) —
            # "원화 띠 폭을 게임 격자가 반올림하는 만큼"이 양쪽으로 같은 여유다.
            # 전단 후보에만 건다: 전단 없는 자리는 어휘가 이미 바닥이라 이 자를
            # 걸면 기존 판정이 바뀐다 (끈 판이 기존과 바이트가 같아야 한다).
            if width_fit and kk and wu / sc.upp < _STROKE_WMIN * wtgt:
                n_shape += 1
                continue
            # **끝 뭉툭함** — 뾰족한 끝은 획이 아니라 잎사귀다 (`_STROKE_END`)
            if (prof_fit and _end_ratio(fi, float(sx[i]),
                                        float(sy[i]), kk) < _STROKE_END):
                n_shape += 1
                continue
            # **몸통 배부름** — 비등방 스케일이 만든 쐐기다 (`_STROKE_BULGE`)
            if (prof_fit and _STROKE_BULGE > 0.0
                    and _bulge_ratio(fi, float(sx[i]), float(sy[i]),
                                     kk) > _STROKE_BULGE):
                n_shape += 1
                continue
        order.append(i)
    c, s = np.cos(th), np.sin(th)
    m = U.mean(axis=1)                     # 중심선 무게중심 (로컬)
    Xm = X.mean(axis=0)
    cands = []
    for i in order:
        fi, kk = int(fidx[i]), float(skw[i])
        # 중심선 무게중심을 목표 무게중심에 맞춘다 — 선형부가 `R·Sk·S`라
        # 전단이 들면 그 이동도 전단을 타야 한다. `kk == 0`이면 항이 통째로
        # 빠져 기존 식 그대로다 (`0.0`을 더해 `-0.0`을 뒤집지도 않는다)
        dx = c[i] * sx[i] * m[fi, 0] - s[i] * sy[i] * m[fi, 1]
        dy = s[i] * sx[i] * m[fi, 0] + c[i] * sy[i] * m[fi, 1]
        if kk:
            dx += c[i] * kk * sy[i] * m[fi, 1]
            dy += s[i] * kk * sy[i] * m[fi, 1]
        lay = Layer(shape=names[fi],
                    x=float(Xm[0] - dx), y=float(Xm[1] - dy),
                    sx=float(sx[i]), sy=float(sy[i]),
                    rot=float(np.degrees(th[i]) % 360.0), skew=kk,
                    color=tuple(int(v) for v in color), alpha=100.0,
                    label="ink" if ink else "cel", stroke=sid)
        # **폭 프로파일**을 점수와 같은 저울에 얹는다 (`_W_PROF`) — 가운데
        # 폭만 맞고 끝이 뾰족한 물방울이 여기서 진다
        pen = (_W_PROF * _prof_pen(fi, float(sx[i]), float(sy[i]), wt,
                                   sc.upp, kk) if wt is not None else 0.0)
        if prof_fit and _W_TANG > 0.0:
            pen += _tang_pen(fi, float(sx[i]), float(sy[i]), float(th[i]),
                             X, wtgt, kk)
        cands.append((sc.score_val(lay) - pen, lay, pen, fi))
    if not cands:
        if n_shape:
            _CURVE_STATS["notline"] += 1
        return None
    # 안정 정렬 — 동점은 잔차 순위 그대로, 그 안에서도 전단이 얕은 쪽이 먼저
    # (§5의 결정적 동점 규칙. 전단 후보가 없으면 둘째 키가 상수라 무동작이다)
    cands.sort(key=lambda t: (-t[0], abs(t[1].skew)))
    # 조종 항 — 놓인 도형의 양끝·접선을 **이 경로 조각**에 맞춘다. 순위
    # 항(`_tang_pen`)이 후보를 고르는 자라면 이쪽은 고른 뒤 **어디에 세울지**의
    # 자다: 하강이 픽셀 이득만 보면 끝이 경로에서 미끄러진다 (`chain.steer`)
    anc_g = (np.stack([path[:, 0] + ry0, path[:, 1] + rx0], axis=1)
             if line else None)          # 채움의 잔여 경로는 획이 아니다
    best = None
    for _, lay, pen, i in cands[:2]:                      # 상위 2개만 정밀 하강
        steer = (chain.steer(sc.cat, sc, anc_g, wtgt, sc.w, sc.h, lay)
                 if anc_g is not None else None)
        # 전단 축은 **씨앗 둘레의 작은 격자 이웃**만 본다 (§9) — 연속 범위를
        # 훑지 않는다. 씨앗이 이미 닫힌 해라 그 둘레 한두 칸이면 족하고,
        # 0 쪽은 언제나 함께 물어본다 (`scoring._descend`의 `skew` 문서)
        gain, q = _descend(sc, lay, color, passes=3, steer=steer,
                           skew=lay.skew != 0.0)
        # **하강도 전단 축에서 같은 지름길을 탄다** — 씨앗에 건 폭 모양 자를
        # 하강 뒤에도 건다. 씨앗만 보고 놓아 주면 자가 헐거워진다: 하강은
        # 픽셀 이득만 보므로 `sx·sy·전단`을 밀어 끝을 얼마든지 뾰족하게 만들 수
        # 있고, 전단은 놓인 폭의 분모(`|M·t̂|`)를 직접 키우는 축이라 **접선이
        # 도는 양끝에서 가장 세게 문다**.
        #
        # 같은 자를 전역 미세 조정도 건다 (`finetune`) — 실측에서 판에 놓인
        # 전단이 **전부 그 패스에서 나왔고** 그 무리의 끝비 p10이 절대 자
        # (0.45) 아래인 0.250까지 내려갔다. 두 자리가 같은 닫힌 식을 쓴다
        # (`descriptor.width_shape_ok`).
        #
        # 어기면 **그 자리에서 전단 축을 끄고 다시 내린다** — 후보를 버리지
        # 않는다. 버리면 막대 사슬로 떨어져 이 획이 통째로 나빠진다.
        if q.skew and prof_fit and not _width_shape_ok(i, q):
            _CURVE_STATS["skew_guard"] += 1
            gain, q = _descend(sc, replace(lay, skew=0.0), color, passes=3,
                               steer=steer, skew=False)
        # 하강 뒤에도 같은 저울로 견준다 — 하강은 sx·sy·rot·전단을 미므로
        # 폭도 끝 방향도 바뀐다
        adj = gain
        if wt is not None:
            adj -= _W_PROF * _prof_pen(i, q.sx, q.sy, wt, sc.upp, q.skew)
        if prof_fit and _W_TANG > 0.0:
            adj -= _tang_pen(i, q.sx, q.sy, np.radians(q.rot), X, wtgt, q.skew)
        if best is None or adj > best[0]:
            best = (adj, q, gain)
    if best is not None:
        best = (best[2], best[1])          # 게이트·반환은 늘 순이득으로 본다
    # 품질 게이트 — 느슨한 기준(순이득 6px·면적 5배)은 제대로 놓인 곡선을 마구
    # 통과시켜 "획 밖으로 튀어나오는 호·도형으로 보이는 초승달"이 났다 (7차
    # 판정 지적). 셋 다 만족해야 곡선 한 장, 아니면 막대 사슬 (상수는 위
    # `_CURVE_*` — 스윕용 스위치가 붙어 있다):
    # ① 순이득 ≥ 획 질량(길이×폭)의 55% — 경로 대부분을 실제로 덮는다
    # ② 잉크의 65% 이상이 내 면 위 — 밖으로 뻗은 호·통통한 초승달 기각
    #    (영역 획은 낭비 벌점이 0.12라 점수만으로는 못 거른다)
    # ③ 면적 ≤ 획 질량의 2.5배 — 획이지 면이 아니다
    if best and not gate:
        # 후보 경쟁이 재 가린다 — 여기서는 최선 후보를 그대로 넘긴다
        _CURVE_STATS["ok"] += 1
        return best
    if best:
        pot = len(path) * max(wpx, 1.5)
        mfin = sc.score(best[1])[1]
        area = float(np.count_nonzero(mfin))
        # 밴드가 걸려 있으면 "내 면 위" 판정도 밴드다 — 다른 선을 쓸어 담는
        # 곡선이 sc.mask(성분 전체) 기준으로는 게이트 ②를 통과해 버린다
        on = float(np.count_nonzero(mfin & (sc.limit if sc.limit is not None
                                            else sc.mask)))
        # 겨루기 모드는 면 판정만 남긴다 — 막대와 겨루더라도 **밖으로 뻗은
        # 호**는 안 된다 (그 하나는 점수로 안 걸린다). 나머지 둘은 "한 장에
        # 담을 자격"이라 마디에는 안 묻는다
        if race:
            if on >= _CURVE_ON * area and area <= _CURVE_AREA * pot:
                _CURVE_STATS["ok"] += 1
                return best
        elif (best[0] >= max(_MIN_GAIN, _CURVE_GAIN * pot)
                and on >= _CURVE_ON * area and area <= _CURVE_AREA * pot):
            _CURVE_STATS["ok"] += 1
            return best
    _CURVE_STATS["lowgain"] += 1
    return None
