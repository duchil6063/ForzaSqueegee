"""도형 어휘 — 이 노선이 낼 수 있는 도형과 그중 무엇을 고르나.

전부 cell_map 단색·기본 탭 도형(창 조작 가능)이고 **정점 알파가 전부 255인
것만** 담는다 (`_FILL_ALL` 문서). 어휘가 바뀌면 저장 템플릿에 심을 씨앗
레이어 종도 바뀐다 — 그 목록이 `shape_vocabulary()`다.
"""

from __future__ import annotations

import os

from ..catalog import Catalog
from ..model import UNITS_PER_SCALE


# 도형 어휘 — 전부 cell_map 단색·기본 탭 도형 (창 조작 가능).
#
# **어휘는 정점 알파가 전부 255인 도형만 담는다** (`CatShape.opaque`). 외곽선만
# 보면 테이퍼 붓·페이드 도형도 꽉 찬 단색 도형으로 보이는데, 게임은 그것을
# 정점 알파의 선형 보간으로 **옅게** 그린다. 도달 520종 중 97종이 그렇다.
# 이 칸을 안 보던 동안 어휘 49종에 14종이 섞여 있었고, 도안은 꽉 찬 그림을
# 그리는데 인게임은 비쳐 보이는 그림이 나왔다 (인게임 실측: U_19가 잉크의
# 63%·U_21이 16%만 올라간다. U_19는 채움 어휘라 7장에서 1,115장 쓰였다).
# 근거·자는 에셋 COLOR 속성과 인게임 스탬프 대조다 (예측과 상관 .97).
# 채움은 같은 색 겹침이 공짜라 회전·전단 덮개로 어떤 면이든 서고 이음새가 안
# 보인다. 봉우리마다 타원·사각(A_01) 두 씨앗을 경쟁시켜 이긴 쪽을 쓴다
# (`_place_fat`) — 직선 모서리 면(옷·폰)은 사각이, 둥근 면은 타원이 이긴다.
_FILL_SHAPE = "A_02"                      # 채움 씨앗 기본 (+비등방·회전·전단)
_BAR_SHAPE = "A_22"                       # 획 막대 = 둥근사각 (양끝 캡)
# 채움 어휘 — cell_map 도달 520종을 획과 같은 방식으로 전수 조사해 걸렀다:
# 단일 루프(구멍 없음)·불투명·뚱뚱함(최대 내접반경/최대변 ≥ 0.10) 164종에서,
# **우리 DOF(회전·비등방 스케일·전단·미러)로 서로 옮겨지는 것**을 2차 모멘트
# 백색화 후 회전 정렬 IoU ≥ 0.88로 묶어 91개 대표만 남긴 것이다. 타원·사각이
# 그중 두 군(각 21·37종)이라, 나머지 89군이 "타원으로는 못 맞추는 모양"이다.
# 씨앗 기하(중심·연장·방향)는 같고 도형만 갈아 끼워 점수로 고른다 — 점수가
# 곧 목적함수(내 잔여 − 침범 벌점)라 대리 지표가 아니다.
# (91개 대표 중 정점 알파가 성한 76종 — 뺀 열다섯은 B_04·B_05·B_23·B_29·
#  B_31·B_32·B_35·B_38·B_40·U_19·U_21·U_42·U_63·U_71·V_73이고 잉크의
#  7~75%만 올라간다)
_FILL_ALL = (
    "A_01", "A_08", "A_09", "A_21", "A_23", "A_25", "A_26", "A_27", "A_29",
    "A_31", "A_32", "A_36", "A_37", "A_38", "A_39", "C_23", "D_03", "D_04",
    "D_11", "D_13", "D_21", "D_22", "D_24", "D_26", "D_36", "D_37", "D_38",
    "D_39", "E_32", "F_06", "F_07", "F_08", "F_09", "F_10", "F_12", "F_13",
    "F_16", "F_19", "F_20", "F_26", "F_33", "F_36", "F_37", "G_01", "G_02",
    "G_04", "G_15", "G_20", "G_36", "H_11", "H_19", "H_31", "H_39", "I_05",
    "I_07", "I_14", "I_28", "I_38", "U_03", "U_04", "U_06", "U_12", "U_13",
    "U_14", "U_18", "U_23", "U_34", "U_35", "U_54", "U_74", "U_75", "U_76",
    "U_77", "U_78", "U_79", "U_80",
)
_FILL_BASE = (_FILL_SHAPE, "A_01")        # 타원·사각 — 어디서나 쓰는 바탕
# 채택 어휘 8종 — 91개 대표를 다 넣으면 불꽃·폭발처럼 가장자리가 너덜한 도형이
# 잔여를 잘 주워 점수로 이기지만 지각 지표가 나빠진다(전역·얼굴 둘 다).
# 원형도가 높은 매끈한 덩어리만 남긴 것이 이 여덟이다.
# 순서가 곧 동점 우선순위다 (앞이 이긴다) — 실측한 순서 그대로 둔다.
_FILL_SHAPES = (_FILL_SHAPE, "A_01",      # 타원 · 사각 (= 바탕)
                "A_09",                   # 방패 (위 평평·아래 둥근)
                "U_04",                   # 삼각
                "A_27",                   # 초승달 (굵은 굽은 면)
                "U_35",                   # 곡옥 (한쪽이 굵은 물방울)
                "U_23")                   # 부메랑 (완만한 굽은 면)
# 여덟째였던 U_19(팩맨형)는 뺐다 — 정점 알파가 페이드라 인게임 잉크가 53%다.
# 채움 어휘라 제일 많이 쓰이던 축이었다
_FILL_TOP = 3                             # 짧은 하강을 붙일 상위 후보 수
# 확장 어휘를 큰 봉우리에만 걸어 보는 스위치 (내접반경 px). 0 = 전부.
# 실측으로 값이 없었다 — 얼굴 악화는 작은 자리가 아니라 얼굴에 걸친 **큰 면**의
# 가장자리가 바뀌어서 생긴다 (r0≥8로 묶어도 얼굴이 다 나빴다).
_FILL_MIN_R0 = 0.0
# 확장 어휘가 바탕을 이기는 데 필요한 배수 — 한 끗 차 교체를 막는다. 이 장치가
# 없으면(1.0) 전역이 나빠지고, 1.30까지 올리면 도로 나빠진다 — 1.15가 실측 최선.
_FILL_MARGIN = 1.15

# 어휘 스위치 (스윕용) — `all`이면 91종 전수, `off`면 타원·사각만.
_v = os.environ.get("FS_FILL_VOCAB", "").strip().lower()
if _v == "all":
    _FILL_SHAPES = _FILL_ALL
elif _v == "off":
    _FILL_SHAPES = (_FILL_SHAPE, "A_01")
elif _v:
    _FILL_SHAPES = tuple(s for s in _v.upper().split(",") if s)
_FILL_TOP = int(os.environ.get("FS_FILL_TOP", _FILL_TOP))
_FILL_MIN_R0 = float(os.environ.get("FS_FILL_MIN_R0", _FILL_MIN_R0))
_FILL_MARGIN = float(os.environ.get("FS_FILL_MARGIN", _FILL_MARGIN))
_FILL_WIN: dict[str, int] = {}            # 도형별 채택 수 (어휘 튜닝용 계측)
_FILL_TMPL: dict[str, tuple[float, float, float, float]] = {}   # 도형 모멘트 캐시
# 2차 모멘트 정합 씨앗 (`_seed_moment`) — 상수를 안 늘리고 씨앗만 바꾼다.
# 채움 장수 5~32% 감소·커버리지 상승. 확장 어휘(8종)가 서는 것도 이 씨앗이
# 있어야 한다. 끄기: `FS_FILL_MOMENT=0`
_FILL_MOMENT = os.environ.get("FS_FILL_MOMENT", "1") != "0"


# 획 어휘 — 굽은 경로를 막대 사슬 대신 **한 장으로** 긋는다 (사람의 획 문법).
# 조건은 "중심선이 열린 한 가닥"뿐이다: 아핀 맞춤(회전 + 비등방 스케일)을 쓰므로
# 원호일 필요가 없고, 카탈로그의 곡선형 단색 도형을 전부 어휘로 삼을 수 있다.
# cell_map 도달(창 조작 가능) 520종을 전수 조사해 걸러낸 42종 —
# 굽음(볼록/현 0.10~1.03) · 굵기(0.02~0.35) · 테이퍼(1.1~20.6)를 두루 덮는다.
# 42종 중 **정점 알파가 성한 28종**만 쓴다. 뺀 열넷(B_04·B_05·B_21·B_24·
# U_19·U_21·U_42·U_43·U_51·U_52·U_63·U_67·U_68·V_63)은 하필 테이퍼가 예쁜
# 붓이라 아핀 맞춤이 자주 고르는데, 그 테이퍼가 기하가 아니라 알파다 —
# 인게임에서는 반투명으로 그려진다
_STROKE_SHAPES = (
    "A_03", "A_04", "A_27", "A_29", "A_31", "A_32", "A_33", "A_36", "A_38",
    "A_39", "C_22", "F_07", "I_39", "U_03", "U_09", "U_12", "U_13", "U_28",
    "U_31", "U_38", "U_46", "U_47", "U_54", "U_55", "U_62", "U_70", "U_80",
    "V_11",
)
# 획 어휘 스위치 (스윕용) — 쉼표 목록으로 통째로 갈아 끼운다 (`FS_FILL_VOCAB`과
# 같은 결). 어휘를 바꾸면 **템플릿 씨앗도 같이 바꿔야** 인게임이 그 도형을 그린다
_sv = os.environ.get("FS_STROKE_VOCAB", "").strip().upper()
if _sv:
    _STROKE_SHAPES = tuple(s for s in _sv.split(",") if s)
# 서술자로 고른 어휘를 쓸 것인가 (`descriptor.stroke_shapes`). 위 손 목록은
# 그 계측의 옛 결과라 카탈로그가 바뀌면 낡는다 — 서술자는 그때마다 다시 잰다
_DESC_VOCAB = os.environ.get("FS_DESC_VOCAB", "1") != "0" and not _sv
_STROKE_CACHE: dict = {}


def stroke_vocab(cat: Catalog | None = None) -> tuple[str, ...]:
    """획 어휘 — 카탈로그를 주면 **서술자가 고른 것**, 없으면 손 목록.

    자격은 기하 하나다: 불투명하고, 중심선이 열린 한 가닥이며, 폭에 비해
    길다 (`descriptor.stroke_shapes`). 굽음·테이퍼로는 여기서 안 거른다 —
    같은 도형도 비등방 스케일에 따라 획이 되기도 잎사귀가 되기도 하므로
    그 판정은 **놓은 뒤**의 폭 프로파일이 한다 (`stroke._placed_form`).
    """
    if cat is None or not _DESC_VOCAB:
        return _STROKE_SHAPES
    got = _STROKE_CACHE.get("v")
    if got is None:
        from .descriptor import stroke_shapes

        got = stroke_shapes(cat) or _STROKE_SHAPES
        _STROKE_CACHE["v"] = got
    return got


# 막대의 폭 눈금이 목표 폭을 이만큼보다 크게 빗나가면 **가는 도형**으로 바꾼다.
# 막대(A_22)는 상자가 정사각이라 폭 눈금이 최소 도형 폭 그 자체다: 세로로 긴
# 구도(h=1961)에서 2.79px 배수라 사람 폭 4.1px 목표가 2.79px로 반올림된다
# (= 목표의 68%). 짧은 축이 더 짧은 도형은 같은 스케일 한 칸이 훨씬 가늘어
# 목표 폭을 그대로 낸다 (`descriptor.straight_thin`).
_BAR_WERR = float(os.environ.get("FS_BAR_WERR", 0.10))
_BAR_CACHE: dict = {}


def bar_for(cat: Catalog, upp: float, wpx: float) -> tuple[str, tuple, float]:
    """이 폭을 낼 막대 도형 — (이름, 로컬 반길이 (ex,ey), 회전 보정 도).

    기본은 둥근사각(A_22)이다 — 양끝 캡이 있어 마디끼리 이을 때 노치가 없다.
    그 폭 눈금으로 목표를 못 맞히는 자리에서만 곧고 가는 도형으로 바꾼다.
    """
    from .descriptor import descriptors, straight_thin

    if not _DESC_VOCAB:                   # 대조군 — 둥근사각만 (§10 ablation)
        return _BAR_SHAPE, (UNITS_PER_SCALE, UNITS_PER_SCALE), 0.0
    des = descriptors(cat)
    d0 = des.get(_BAR_SHAPE)
    if d0 is None:
        return _BAR_SHAPE, (UNITS_PER_SCALE, UNITS_PER_SCALE), 0.0
    step = d0.min_width_px(upp)
    q = max(1.0, round(wpx / step)) * step
    if step <= 1e-9 or abs(q - wpx) <= _BAR_WERR * max(wpx, 1e-6):
        return _BAR_SHAPE, (d0.ext_x, d0.ext_y), 0.0
    key = round(upp, 6)
    thin = _BAR_CACHE.get(key)
    if thin is None:
        thin = straight_thin(cat, upp, step)
        _BAR_CACHE[key] = thin
    for name in thin:                     # 가는 순 — 첫째가 눈금이 제일 촘촘하다
        d = des[name]
        if d.min_width_px(upp) <= wpx:
            return name, (d.ext_x, d.ext_y), 0.0 if d.long_is_x else 90.0
    return _BAR_SHAPE, (d0.ext_x, d0.ext_y), 0.0


_WFLOOR_CACHE: dict = {}
# **어휘가 실제로 낼 수 있는 가장 가는 폭**을 쓸 것인가 (끄면 막대 폭이 바닥).
_WFLOOR = os.environ.get("FS_LINE_WFLOOR", "1") != "0"
# 그 바닥의 하한 px — 채점이 1px 격자라 그보다 가는 목표는 뜻이 없다.
_WFLOOR_MIN = 1.0


def min_stroke_width_px(cat: Catalog, upp: float) -> float:
    """획 어휘가 **실제로 낼 수 있는 가장 가는 폭** px (하한 `_WFLOOR_MIN`).

    엔진 곳곳이 "게임이 낼 수 있는 최소 폭"으로 `2 × geometry._min_span`을
    써 왔다. 그것은 **둥근사각(A_22)의** 최소 폭이다 — 상자가 정사각이라
    최소 스케일에서 최소 도형 폭 그대로 선다. 그런데 어휘 38종 중 태반은
    짧은 축이 훨씬 짧아 같은 스케일에서 훨씬 가늘게 선다 (h=1961 실측:
    막대 2.79px ↔ U_45 0.19px · U_31 0.40px). `bar_for`가 막대를 고를 때는
    이미 그 사실을 쓰는데, **폭의 목표와 이상 띠는 여전히 막대 기준**이었다.

    그 차이가 곧 "선이 원화보다 굵다"의 남은 절반이다: 목표 프로파일이
    `clip(원화 띠, 2.79, …)`이라 원화가 1.9px인 자리에서 **목표가 통째로
    2.79 상수로 눌린다** — 폭 프로파일 항(`stroke._prof_pen`)이 "얼마나
    2.79에 못 미치나"를 재게 되어, 가늘게 그은 획을 벌하고 굵기 변화(리듬)를
    아예 못 본다. 여기서 바닥을 어휘 쪽으로 내리면 목표가 원화 띠 그 자체가
    된다.

    `FS_LINE_WFLOOR=0`이면 종전대로 막대 폭이 바닥이다.
    """
    if not _WFLOOR:
        return 2.0 * 0.01 * UNITS_PER_SCALE / max(upp, 1e-9)
    key = round(upp, 6)
    got = _WFLOOR_CACHE.get(key)
    if got is None:
        from .descriptor import descriptors

        des = descriptors(cat)
        got = min((des[n].min_width_px(upp) for n in stroke_vocab(cat)
                   if n in des), default=_WFLOOR_MIN)
        got = max(_WFLOOR_MIN, float(got))
        _WFLOOR_CACHE[key] = got
    return got


def shape_vocabulary(cat: Catalog | None = None) -> tuple[str, ...]:
    """이 노선이 낼 수 있는 도형 이름 전부 (획 어휘 + 채움 어휘 + 바탕).

    **저장한 템플릿 그룹에 씨앗 레이어를 몇 종 심어야 하나**를 정하는 데 쓴다.
    다시 연 비닐 그룹은 **제 저장본이 참조한 도형 에셋만** 그릴 수 있어서,
    주입한 도형 id가 그 밖이면 조용히 템플릿 도형으로 그려진다
    (`auto/template.seed` 문서). 환경 스위치(`FS_FILL_VOCAB`)로 어휘가 바뀌면
    이 목록도 같이 바뀌므로 템플릿을 그때 다시 만들어야 한다."""
    return tuple(dict.fromkeys((_FILL_SHAPE, "A_01", _BAR_SHAPE)
                               + stroke_vocab(cat) + _FILL_SHAPES))


_VOCAB_CHECKED = False


def _check_vocab(cat: Catalog, log) -> None:
    """어휘에 **반투명 도형이 섞였는지** 굽기 전에 본다 (프로세스 1회).

    소스의 세 목록은 걸러 두었지만 `FS_FILL_VOCAB`이 열려 있고 카탈로그도
    다시 지을 수 있다. 여기서 안 잡으면 증상이 인게임에서만 보이고, 그
    자리에서는 "왜 옅지"가 배치 문제로 보인다.
    """
    global _VOCAB_CHECKED
    if _VOCAB_CHECKED:
        return
    _VOCAB_CHECKED = True
    bad = [(n, cat[n].alpha_area) for n in shape_vocabulary(cat)
           if n in cat.shapes and not cat[n].opaque]
    if bad:
        log("경고: 어휘에 반투명 도형이 있다 — 인게임이 도안보다 옅게 그린다: "
            + " · ".join(f"{n}(잉크 {a:.0%})" for n, a in bad))
