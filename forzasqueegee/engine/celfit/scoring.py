"""채점판과 좌표하강 — **어디에 놓을지**를 정하는 비용 모형.

내 면 위 겹침은 공짜(같은 색이라 안 보인다), 먼저 그린 면·배경 침범은 무겁게,
나중 면 위 스필은 가볍게 문다. 여기 상수가 곧 "이 한 장이 제 값을 하는가"의
자다 — 무엇을 **살지**는 가격(`price`)이 따로 묻는다.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

from ..catalog import Catalog
from ..model import UNITS_PER_SCALE, Layer
from .geometry import _grad_alpha, _min_span, _poly_px


# 획 채점의 낭비 벌점 — 선 띠 **밖** 픽셀 하나당. 1.0보다 싸면 "새로 먹는
# 픽셀의 일부만 띠 위면 이득"이 되어 `_descend`가 획을 스스로 부풀리고, 그렇게
# 샌 잉크가 경계 밴드를 어둡게 만든다. 1.0이 스윕의 안쪽 봉우리다.
_PEN_LINE = float(os.environ.get("FS_LINE_WASTE", 1.0))
# 벌점을 **강제된 스필만큼 깎는다**. 최소 도형이 선 띠보다 굵으면 넘치는 것은
# 불가피한데(띠보다 굵은 그림에서 100%를 넘는다), 그 몫까지 벌하면 획이 이득을
# 못 내고 죽는다 — 벌점 1.0을 그대로 걸면 그런 그림만 통째로 무너진다.
# 새 상수를 더하지 않는다: 배수를 **게임 격자에서 유도한다**
# (띠 폭 / 최소 도형 폭, 상한 1).
# 이 둘이 **큰 면의 테두리 값을 정한다.** 계측에서
# 비싼 면은 큰 도형 하나가 몸통을 덮고 나머지 열~서른 장이 너덜한 가장자리
# 주머니를 하나씩 줍는 모양이었다. 반지름이 이보다 큰 주머니가 남아 있고
# 커버리지가 목표에 못 미치는 동안 계속 줍는다.
_STROKE_R = float(os.environ.get("FS_STROKE_R", 2.6))    # px — 이 이하 주머니는 안 줍는다
_MIN_GAIN = 6.0        # px — 이보다 못 버는 도형은 안 찍는다
_COVER_STOP = float(os.environ.get("FS_COVER_STOP", 0.985))  # 영역 커버리지 목표
_PEN_FORBID = 4.0      # 먼저 그린 면 침범 벌점 (px당) — 나중 획·면이 일부 덮는다
# **채움의** 먼저 그린 면 침범 벌점 — 획(4.0)보다 네 배 무겁다.
# 유령 계단(목표에 없는 저대비 경계, `work/lab/lcb.py`)의 97%는 **침범**이다:
# 채움이 이웃 영역으로 2~8px 넘친 자리가 ΔE 4~12라 수리 문턱(12) 아래·
# JND(4) 위에 남는다. 그 78%는 밑에 목표색 도형이 깔려 있는데 넘친 조각이
# 위에서 가리는 꼴이고, 순서를 바꾸면 반대쪽이 드러나는 상호 스필이라
# (쌍마다 좋은 순서를 골라도 1,341 → 1,198px) 순서 재배열로는 안 닫힌다 —
# 넘침 자체를 배치에서 비싸게 하는 것이 손이다. 획에는 안 건다(공용 상수를
# 올리면 이음 보수 몫 +0.04·joint_gap 악화 — 획 침범은 선화가 덮는 자리다).
# 16은 무릎이다 (표준 3장, 보이는 오차 Σ / 도형): 8 −.087/+89 · **16
# −.196/+198** · 32 −.191/+264. 11장: 보이는 오차 med .366→.274(11/11) ·
# 면 안 틀림 10/11 · 유령 med 1,881→1,425 · 넘김률 11/11 · 도형 +3.5%
# (10번 2,973 ≤ 3,000) · 게이트 전부 통과 · 획 지표 불변.
_PEN_FORBID_FILL = 16.0
# 투명 배경 침범은 따로 더 무겁게 — 아무도 안 덮어 흰 자국이 그대로 남는다.
# 4×만으로는 큰 채움이 "새 픽셀 이득"으로 벌점을 눌러 실루엣 밖 사각형이
# 찍히는 일이 실제로 났다 (어깨 밖에 살색 사각형이 그대로 남았다)
_PEN_BG = 12.0
_PEN_WASTE = 0.12      # 나중 면 위 스필 벌점 (px당)
# 영역 채움에서는 이 벌점을 **면제**한다 — 그 자리는 나중에 그리는 면이 덮는
# 자리라 사람도 신경 쓰지 않는다("큰 면은 뒤로 보내면 기존 파츠가 모서리를 다
# 커버해준다"). 0.12를 물리면 하강이 이웃 면 앞에서 멈춰, 한 장이면 될 면을
# 여러 장으로 쪼갠다. 획 배치(_fit_lines)는 모든 면 위에 그려져 덮어 줄 것이
# 없으므로 면제 대상이 아니다
_PEN_WASTE_FILL = 0.0
_MAX_PER_REGION = 80   # 영역당 도형 상한 (예산 폭주 방지)


# 경계 1px 물림 밴드를 **배경 위에서도** 무료로 둘 것인가. 무료의 근거는
# "물림은 이웃 면·선화가 덮는다"인데 실루엣 **밖**에는 덮어 줄 이웃도 획도
# 없다 — 흰 자국이 그대로 남는다 (실측: 그 한 겹이 지각차의 5~14%).
# 사다리는 **이미 있는 비용 등급**이라 새 상수가 없다:
#   0 = 무료 · 1 = 낭비 급 · 2 = 침범 급(먼저 그린 면) · 3 = 배경 급
# **기본 2** (2026-08-14 육안 채택) — 실루엣 최외곽 테를 구멍으로 안 세는
# `holes._sil_rim`과 묶여서만 선다. 혼자 켜면 보호 라벨 되먹임으로 붕괴한다.
_SIL_BAND = int(os.environ.get("FS_SIL_BAND", "2"))


# 이어긋기 (line 노선 전용) — `_Scorer`의 이어긋기 축 문서 참조. retrace는
# 덮인 밴드 위 px의 값 배율, far는 ±최소 도형 폭 밖 이탈에 **더** 무는 벌점
_RETRACE = float(os.environ.get("FS_LINE_RETRACE", 0.6))
_PEN_FAR = float(os.environ.get("FS_LINE_FAR", 3.0))
# **이상 띠 밖의 선 픽셀은 내 이득이 아니다** — 그 값 배율 (1.0 = 안 깎는다).
#
# 탐색 밴드(`set_band`의 첫 인자)는 제 폭의 두 배 남짓이라(정책 `band_slack`)
# 조밀한 선망에서 **이웃 가닥이 그 안에 들어온다**. 그 가닥 픽셀도 잔여라
# 배율이 1.0이면 px당 1.0의 새 이득이 되고, 하강이 그 이득을 좇아 도형을
# 옆으로 부풀린다 — 이웃 가닥을 삼킨 자리가 곧 "머리칼이 덩어리로
# 읽히는" 자리다.
#
# 이상 띠(`core`) 밖 물림을 이미 낭비로 무는 자리가 바로 아래에 있는데
# (`free & ~core`), 그것은 **선이 아닌 자리**만 문다. 이웃 가닥은 선 픽셀이라
# 그 항에 안 걸린다 — 여기가 그 짝이다. 값을 0으로 두지 않는 것은 교차점
# 근처에서 두 획이 실제로 같은 픽셀을 나눠 갖기 때문이다.
#
# **혼자서는 거의 아무 일도 안 한다** (실측 01: 도형 ±0.3%, 지표 전부 노이즈
# 수준). 이상 띠가 막대 폭(2.79px@h=1961)이면 원화 띠(1.9px)보다 굵어 띠
# 밖으로 나갈 일 자체가 적기 때문이다. 폭 바닥이 어휘 실측이라
# (`vocabulary.min_stroke_width_px`) 띠가 원화 띠와 같고, 그래야 이 항이
# 문다 — 둘은 한 벌이다.
_CORE_GAIN = float(os.environ.get("FS_LINE_CORE_GAIN", 0.25))


# ── §24 **덧칠 띠 — 보이는 목표와 칠하는 목표를 가른다.**
# 영역마다 제 라벨까지만 정확히 칠하면 두 면이 만나는 자리에서 셋 중 하나가
# 난다: 틈(둘 다 못 미침) · 반대색 슬리버(먼저 그린 쪽이 넘어와 안 덮임) ·
# 그 자리를 줍는 보정 도형. 틈이 특히 나쁘다 — 그 자리는 인게임에서 차
# 도색이 그대로 비친다.
#
# 틈이 나는 까닭은 배치가 **1x 격자**에서 "덮었다"를 판정하는데 게임은
# 벡터로 그리기 때문이다. 픽셀 중심이 도형 안에 들면 배치는 그 픽셀을 덮은
# 것으로 치지만, 2배 표본에서는 그 픽셀의 넷 중 셋이 비어 있을 수 있다
# (`coverage` 문서). 양쪽 면이 같은 경계에서 각자 멈추면 그 반 픽셀이 그대로
# 남는다 — 실측(W7 11장) 봉인 전 미커버 표본의 **군집 90%가 크기 4px 이하로
# 색 경계에 붙어 있었다**.
#
# 그래서 목표를 둘로 가른다:
#
# - **보이는 목표** (`mask`·`residual`) — 이 면이 화면에서 차지해야 할 자리.
#   값(`worth`)·커버리지·막대·마무리가 보는 자는 여전히 이것 하나다.
# - **칠하는 목표** (`band_res`) — 그 위에 **덧칠해도 되는** 자리. 여기를
#   덮는 것은 값이 아니고(가격·커버리지는 안 본다) 배치 점수에서만 작은
#   무게로 센다. 하강이 경계에서 멈추지 않고 **한 겹 넘어가** 이웃과 겹치게
#   만드는 것이 목적이고, 겹친 자리는 아래 규칙으로 언제나 누군가가 덮는다.
#
# 덧칠이 허용되는 자리는 셋이고, 깊이는 상수가 아니라 **격자**가 정한다:
#
# ① **나중에 그릴 면 위** — 두 몫(`kd` = 이동 스텝 0.5유닛의 두 배)까지.
#    틈의 폭은 양쪽이 각자 물러선 만큼이고 물러섬의 한 몫이 이동 스텝이라,
#    둘이 마주 물러서면 최악이 두 몫이다. 그 면이 나중에 덮으므로 안 보인다.
# ② **먼저 그린 면 위** — 한 겹만. 경계의 주인은 나중 면이라는 그리기 순서의
#    규칙 그대로다. 그 위는 아무도 안 덮으니 한 겹을 넘으면 색이 그대로 보인다.
# ③ **놓인 획 밑** — 획 반폭 + 한 겹까지. 선화는 모든 면 위에 얹히므로 그
#    밑은 무엇을 칠하든 안 보인다. 스냅이 색 경계를 획 중앙선에 앉혀 두었으니
#    (`celart.snap`) 양쪽 면이 여기서 만나 서로를 덮는다.
#
# 실루엣 **밖**은 안 넣는다 — 거기는 덮어 줄 나중 면도 획도 없다.
# 무늬 보호 조각(`celart.marks`)도 뺀다 — 큰 면이 눈 흰자·코 그림자를
# 1px씩 먹는 것이 이 덧칠의 유일한 해악이다.
#
# **띠는 잔여처럼 닳는다.** 이미 덮은 띠 픽셀은 값을 잃는다 — 안 그러면 같은
# 자리를 다시 덮는 도형이 계속 이득을 받아 하강이 그리로 쏠린다.
#
# 값이 1(진짜 목표 px와 같은 무게)보다 작아야 한다 — **이 당김이 배치를 끌고
# 가면 안 되고**, 밀지 않으면 안 가는 자리라는 것만 알려 주면 된다. 실측
# (D1↔D2, 09)에서 1.0은 채움을 352 → 453장으로 불리고 봉인 전 미커버를
# 4,408 → 4,830표본으로 **되레 늘렸다**: 띠가 이득의 한 축이 되면 하강이 제
# 잔여 대신 띠를 좇아 한 장이 맡는 몫이 줄고, 그만큼 도형이 늘면서 도형끼리
# 만나는 자리(= 틈이 나는 자리)도 함께 는다.
_BAND_GAIN = float(os.environ.get("FS_CEL_BAND", 0.35))

# ── §24-b **늘리기 전용의 보수적인 자** — "가장자리는 반만 덮은 것이다."
# 채점 라스터는 1x다: 반올림한 폴리곤을 `fillPoly`로 그려 **중심이 안에 든
# 픽셀**을 덮었다고 센다. 게임은 벡터라 그 픽셀의 2배 표본 넷 중 셋이 비어
# 있을 수 있고, 이웃 도형이 같은 자리에서 똑같이 멈추면 그 반 픽셀이 그대로
# 미커버로 남는다 (실측 A0-09: 미커버 군집의 48%가 **채움 도형끼리 만나는
# 자리**의 노치였다 — 배경도 획도 아닌, 제 면 안이다).
#
# 자를 하나 더 두지 않고 **같은 라스터를 한 겹 깎아** 쓴다: 확정할 때
# 잔여에서는 도형 마스크를 그대로 빼되, **늘리기가 보는 잔여**에서는 한 겹
# 깎은 것만 뺀다. 그러면 도형 가장자리 한 겹이 "아직 안 덮은 자리"로 남고,
# 늘리기(`layered.grow_fill` — 레이어 0장)가 그 자리를 겨눠 도형을 한 스텝
# 키운다. 두 도형이 만나는 자리는 그렇게 **겹쳐서** 만난다.
#
# 이 자를 보는 것은 늘리기뿐이다. 사는 자리(채움·막대·마무리·가격)는 전부
# 원래 잔여를 보므로 **도형을 더 사지 않는다** — 공짜인 손에만 목표를 준다.
_EDGE_SLACK = int(os.environ.get("FS_EDGE_SLACK", 1))

# ── §17 **실루엣 밖은 값이 아니라 금지다.**
# 실루엣 밖 스필은 px당 `_PEN_BG`로 **값을 매겨** 왔다. 그 값(12)은 한 번
# 올린 값이다 — "4×만으로는 큰 채움이 새 픽셀 이득으로 벌점을 눌러 실루엣 밖
# 사각형이 찍혔다"(위 `_PEN_BG` 문서). 값으로 두는 한 그 실패는 **면이 커질
# 때마다 되돌아온다**: 이득은 면적에 비례해 자라는데 벌점은 px당 상수라,
# 충분히 큰 면에서는 언제나 눌린다 (실측: 병합으로 면이 커진 판에서 실루엣
# 밖 스필이 실루엣에서 최대 86px까지 뻗었다 — 작은 면에서는 38px이다).
#
# 그런데 그 자리는 **거래가 아니다.** 안쪽 스필이 공짜인 근거는 "나중 면·획이
# 덮는다"인데 밖에는 덮어 줄 것이 없고, 이 노선에는 뺄셈 마스크가 없어 나중에
# 지울 수도 없다. 이득이 0이고 되돌릴 수 없으면 값이 아니라 **제약**이다.
#
# 문턱은 새 상수가 아니라 게임 격자다 — 도형 가장자리는 최소 도형 폭
# (`geometry._min_span`) 눈금에만 설 수 있으므로 그 한 칸까지의 물림은
# **강제**다 (같은 자를 `snap_labels_to_ink` 반경과 껍질 컷이 쓴다).
# 그 밖으로 나가는 후보는 점수가 아니라 자격에서 진다.


def _shrink(m: np.ndarray) -> np.ndarray:
    """도형 마스크에서 가장자리 `_EDGE_SLACK` 겹을 깎는다 (§24-b)."""
    if _EDGE_SLACK <= 0:
        return m
    return cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8),
                     iterations=_EDGE_SLACK).astype(bool)


class _Scorer:
    """영역 하나의 채점판 — ROI 배열 셋 (내 잔여·내 면·금지).

    비용 모형: 내 면 위는 겹쳐도 공짜(같은 색이라 안 보인다), 먼저 그린 면·배경
    침범은 무겁게, 나중 면 위 스필은 가볍게 문다. 내 면의 1px 팽창 밴드는
    금지에서 뺀다 — 경계에 딱 붙여 찍게 해 이웃면 사이 흰 실틈을 없앤다
    (1px 물림은 나중 면·선화 획이 덮는다).
    """

    def __init__(self, cat: Catalog, upp: float, w: int, h: int,
                 roi: tuple[int, int, int, int],
                 mask: np.ndarray, forbid: np.ndarray, bg: np.ndarray,
                 pen_waste: float = _PEN_WASTE, guard: float | None = None,
                 ink: np.ndarray | None = None, val: np.ndarray | None = None,
                 soft: np.ndarray | None = None, retrace: float = 0.0,
                 pen_far: float = 0.0, seam: bool = False,
                 protect: np.ndarray | None = None,
                 silcap: bool = False, pen_forbid: float = _PEN_FORBID):
        self.cat, self.upp, self.w, self.h, self.roi = cat, upp, w, h, roi
        self.pen_forbid = pen_forbid      # 먼저 그린 면 침범 벌점 (px당)
        self.mask = mask                  # 내 면 전체 (bool, ROI)
        self.residual = mask.copy()       # 아직 안 덮인 내 면
        # §24-b 늘리기 전용 잔여 — 도형 가장자리 한 겹은 안 덮은 것으로 친다.
        # 면 채움(`seam`)에서만 쓴다 — 획 배치에는 늘리기 단이 없어서, 두면
        # 확정마다 침식 한 번이 순수한 비용이다
        self.res_edge = mask.copy() if (seam and _EDGE_SLACK > 0) else None
        self.val = val                    # 값 맵 ROI (가격 설계) — None이면 px 수
        self.pen_waste = pen_waste
        self.guard = guard                # ROI 밖 허용 여유 (None = 기본식)
        # **이어긋기 축** (line 노선의 획 채점 전용, 기본 0 = 안 켠다).
        # 사람은 오버레이의 선을 픽셀대로 따지 않는다 — 겹치든 벗어나든 획을
        # **이어** 긋고, 약간의 이탈은 오차로 친다
        # (`references/사람작업/오버레이-선*.png`). 잔여만 값으로 치면 그 문법이 안 선다: 교차점·마디 겹침에서
        # 이미 덮인 밴드 위 마디가 이득 0이라 죽고(중간 끊김), 하강이 겹침
        # 구간에서 마디를 움츠려 이음새마다 틈이 열린다.
        # - retrace: 이미 덮인 **내 밴드 위** px도 이 배율로 값을 친다 — 겹쳐
        #   그리기는 같은 색이라 공짜인데 채점만 그걸 모르고 있었다.
        # - soft·pen_far: 밴드에서 soft(±최소 도형 폭) 안의 이탈은 낭비
        #   벌점(= 오차 취급)이고, 그 **밖**의 이탈만 pen_far를 더 문다 —
        #   "약간 벗어나는 건 오차, 튀어나오는 건 벌점"의 두 단이다.
        self.retrace = retrace
        band = cv2.dilate(mask.astype(np.uint8),
                          np.ones((3, 3), np.uint8)).astype(bool)
        # **획이 덮는 자리는 1px 물림과 같은 등급의 공짜다.** 선화는 모든 면 위에
        # 마지막으로 얹히므로 그 밑의 면 경계는 안 보인다 — FH5 튜토리얼에서
        # 사람이 선화를 다 그린 뒤 색을 그 밑에 까는 것이 이것이다 (머리카락
        # 한 덩이가 타원 한 장이고, 삐져나온 가장자리는 선이 가린다). 자리를
        # 실제로 배치된 획 레이어에서 받으므로 안 그은 선은 안 세고, 선이
        # 없는 그림에서는 아무 일도 안 한다
        if ink is not None:
            band = band | (ink & ~bg)
        self.forbid = forbid & ~band & ~bg   # 먼저 그린 면 (경계 밴드 제외)
        # 경계 1px 물림은 **무비용** — 벌점 면제(낭비 0.12×)만으로는 "테두리
        # 부스러기 몇 px 이득 < 밴드 낭비"라 하강이 경계에 딱 붙지 않고, 그
        # 부스러기가 구멍 메움 수요의 74%가 됐다 (실측). 물림은 이웃 면·선화가
        # 덮으므로 공짜가 맞다 — 다만 그 근거는 실루엣 **안**에서만 성립한다
        # (`_SIL_BAND`). 밖에는 덮어 줄 이웃이 없어 흰 자국이 그대로 남는다
        self.bg = bg & ~band                 # 투명 배경 (밖 물림은 아래에서)
        self.free = band & ~mask
        # 밖 물림은 `_spill`에 넣기만 해서는 **채움에서 무효**다 — 채움의 낭비
        # 벌점이 `_PEN_WASTE_FILL = 0.0`이라 획에만 걸린다. 그래서 따로 세고
        # 따로 문다. 값은
        # `max(제 스필 값, 등급)` — 밖 물림이 안쪽 스필보다 싸질 수 없다
        self.outb = np.zeros_like(self.free)
        self.pen_out = 0.0
        if _SIL_BAND:
            self.free = self.free & ~bg
            self.outb = band & ~mask & bg
            grade = (_PEN_WASTE, _PEN_FORBID, _PEN_BG)[_SIL_BAND - 1]
            # 벌점을 **강제된 물림만큼 깎는다** — 획 낭비 벌점의 적응과 같은
            # 수다. 도형의 가장자리는 게임 **이동 스텝** 격자에만 설 수 있어,
            # 그 양자가 밴드(1px)보다 굵으면 무는 것이 불가피하다. 배수 =
            # min(1, 1px / 위치 양자) = min(1, upp/0.5) — 새 상수가 아니라
            # 격자 유도값이고, 세로로 긴 구도일수록 1보다 작아진다
            grade *= min(1.0, upp / 0.5)
            self.pen_out = max(self.pen_waste, grade)
        # §24 덧칠 띠 — **칠하는 목표**. 보이는 목표(mask)와 갈린다 (위 문서).
        # 세 자리를 격자에서 유도한 깊이만큼 연다: 나중 면 두 몫 · 먼저 그린
        # 면 한 겹 · 놓인 획 밑은 획 반폭까지. 무늬 보호 조각은 뺀다
        self.band_res = None
        if seam and _BAND_GAIN > 0.0:
            # ② 먼저 그린 면 위 한 겹 (`free`가 이미 1px 물림 밴드다)
            bd = self.free & forbid
            # ① 나중에 그릴 면 위 두 몫 — 틈의 폭은 양쪽이 각자 물러선
            # 만큼이고, 물러섬의 한 몫은 게임 **이동 스텝**(0.5유닛)이다.
            # 둘이 마주 물러서면 최악이 두 몫이므로 겹침도 두 몫이라야
            # 틈이 원리적으로 안 생긴다 (`1.0/upp` = 두 몫 px)
            kd = max(1, int(round(1.0 / max(upp, 1e-6))))
            deep = cv2.dilate(mask.astype(np.uint8),
                              cv2.getStructuringElement(
                                  cv2.MORPH_ELLIPSE,
                                  (2 * kd + 1, 2 * kd + 1))).astype(bool)
            bd = bd | (deep & ~mask & ~forbid & ~bg)
            # ③ 놓인 획 밑 — 스냅이 색 경계를 획 **중앙선**에 앉혀 두었으므로
            # (`celart.snap`) 건너편 가장자리까지 가려면 획 반폭이 더 든다.
            # 최소 도형 반폭(`_min_span`)이 그 눈금이다 — 새 상수가 아니다
            if ink is not None:
                rk = kd + max(1, int(round(_min_span(upp))))
                wide = cv2.dilate(mask.astype(np.uint8),
                                  cv2.getStructuringElement(
                                      cv2.MORPH_ELLIPSE,
                                      (2 * rk + 1, 2 * rk + 1))).astype(bool)
                bd = bd | (wide & ink & ~mask & ~bg)
            if protect is not None:
                bd = bd & ~protect
            # 띠도 잔여처럼 닳는다 — 이미 덮은 자리는 값을 잃는다
            self.band_res = bd if bd.any() else None
        # §17 실루엣에서 **격자 한 칸보다 멀리** 있는 배경 — 여기를 무는 후보는
        # 자격에서 진다 (점수로 안 다툰다). 획 배치는 안 건다: 획은 제 밴드
        # 안에서만 놓이고 실루엣 테를 따라 그어야 한다
        self.farbg = None
        if silcap and bg.any():
            r = max(1, int(round(_min_span(upp))))
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)
            far = cv2.erode(bg.astype(np.uint8), k).astype(bool)
            self.farbg = far if far.any() else None
        self._spill = ~mask & ~self.free & ~self.outb  # 낭비 후보 (forb·bg 포함)
        self.pen_far = pen_far
        self._farm = (self._spill & ~soft & ~bg
                      if soft is not None and pen_far > 0.0 else None)
        # **제 경로 밴드 한정** (line 노선) — `set_band` 문서. None = 무제한
        self.limit: np.ndarray | None = None
        # **제 획의 이상 띠** (line 노선, `set_band`의 둘째 인자). 밴드가
        # "어디까지 나가도 되나"라면 이쪽은 "이 획이 얼마나 굵은가"다
        self.core: np.ndarray | None = None
        # 점수 메모 — 하강이 같은 후보를 패스마다 다시 묻는다 (스텝 열이 스텝
        # 수보다 짧은 축·씨앗 경쟁 뒤 정밀 하강). 점수는 잔여에만 의존하므로
        # 배치 확정(commit)에서 비운다. 색은 점수에 안 들어가 키에서 뺀다
        self._memo: dict = {}
        # 되돌리기 일지 — 후보 경쟁은 **같은 상태에서** 여러 안을 지어 보고
        # 이긴 하나만 남긴다. 잔여 전장을 후보마다 복사하면 그 비용이 채점을
        # 넘으므로, 지운 픽셀만 창 단위로 적어 두고 되돌린다
        self._journal: list | None = None

    def set_band(self, m: np.ndarray | None,
                 core: np.ndarray | None = None) -> None:
        """이득·retrace를 **이 마스크 안으로 한정**한다 (ROI 크기 bool, None=해제).

        line 노선에서 획 하나를 놓는 동안 그 획의 경로 밴드를 건다. 채점판의
        mask·residual은 **성분 전체**(이어진 선망 전부)라, 한정이 없으면 하강이
        교차하는 **다른 선** 위로 마디를 늘려도 이득(fresh residual + retrace)을
        얻는다 — "다른 선을 침범할 정도로 길어지는" 실제 원인이다 (사용자 지적
        2026-08-25). cel 노선은 안 건다 (밑을 면이 받쳐 증상이 없다).

        밴드 **밖** 제 성분 px는 낭비로 문다 (`_PEN_LINE`). 중립으로 두었더니
        "득이 없다"만으로는 안 물러섰다 — 초기 크기·양자화가 도형을 밖으로
        밀어 놓으면 하강이 되돌릴 이유가 없어, 선화에서 안 가로지르는 자리를
        도안이 가로지르고 도형이 만나는 지점에서 한쪽이 튀어나왔다 (사용자
        지적 2026-08-26). 교차점을 **지나가는** 것은 여전히 공짜다 — 지나가는
        자리는 제 경로 위라 밴드 안이다.

        `core`는 그 획의 **이상 띠**다 (제 폭 + 양자화 여유). 있으면 1px 물림
        면제(`free`)와 이어긋기 값(`retrace`)을 그 안으로 좁힌다 — 없으면
        둘 다 **성분 전체의 선 픽셀** 둘레에 걸린다. 선망이 조밀한 자리
        (머리칼 다발)에서는 그것이 곧 "굵어져도 공짜"라서, 하강이 이웃 가닥까지
        먹는 폭까지 도형을 부풀린다: 실측(표준 10장) 놓인 폭 중앙 4.0px ↔
        원화 띠 2.0px. 사람 획은 원화 폭을 따라가므로 여기를 좁힌다.
        """
        self.limit = m
        self.core = core
        self._memo.clear()

    def commit(self, m: np.ndarray) -> None:
        """배치 확정 — 잔여(와 덧칠 띠)에서 m을 지운다. 메모도 비운다."""
        if self._journal is not None:
            self._journal.append(
                (None, (m & self.residual).copy(),
                 None if self.band_res is None else (m & self.band_res).copy(),
                 None if self.res_edge is None
                 else (_shrink(m) & self.res_edge).copy()))
        self.residual &= ~m
        if self.band_res is not None:
            self.band_res &= ~m
        if self.res_edge is not None:
            self.res_edge &= ~_shrink(m)
        self._memo.clear()

    def commit_box(self, m: np.ndarray, box: tuple[int, int, int, int]) -> None:
        """창 단위 배치 확정 — `_score_impl`이 준 작은 마스크를 그대로 쓴다.

        확정마다 ROI 전장 마스크를 다시 그리면 그 비용이 채점 비용을 넘는다.
        """
        bx0, by0, bx1, by1 = box
        if self._journal is not None:
            self._journal.append(
                (box, (m & self.residual[by0:by1, bx0:bx1]).copy(),
                 None if self.band_res is None
                 else (m & self.band_res[by0:by1, bx0:bx1]).copy(),
                 None if self.res_edge is None
                 else (_shrink(m) & self.res_edge[by0:by1, bx0:bx1]).copy()))
        self.residual[by0:by1, bx0:bx1] &= ~m
        if self.band_res is not None:
            self.band_res[by0:by1, bx0:bx1] &= ~m
        if self.res_edge is not None:
            self.res_edge[by0:by1, bx0:bx1] &= ~_shrink(m)
        self._memo.clear()

    def eats(self, m: np.ndarray, box: tuple[int, int, int, int] | None) -> bool:
        """이 도형이 **아직 덜 덮인 자리**에 걸쳐 있나 — 늘리기의 자격 판정.

        자가 셋이고 갈수록 무르다:

        - **보이는 잔여** — 아직 아무도 안 칠한 내 면.
        - **덧칠 띠** (§24) — 경계를 한 겹 넘어 이웃과 겹칠 자리.
        - **가장자리 잔여** (§24-b) — 1x에서는 덮였지만 **반만** 덮인 자리.
          도형의 테는 언제나 여기에 걸리므로, 이 자는 사실상 "제 면 안에
          아직 반 픽셀이라도 성긴 자리가 남아 있는 도형은 계속 늘린다"는
          뜻이다. 멈추는 것은 이 자가 아니라 **점수**다 (`grow_fill`은
          점수가 내려가면 안 늘린다) — 먼저 그린 면·배경·실루엣 자격이
          바깥 테두리를 그대로 지키므로, 도형은 제 면과 덧칠 띠를 다 채운
          자리에서 저절로 선다. 실측(09): 도형당 평균 1.8스텝.
        """
        if m is None or box is None:
            return False
        bx0, by0, bx1, by1 = box
        if (m & self.residual[by0:by1, bx0:bx1]).any():
            return True
        if (self.band_res is not None
                and (m & self.band_res[by0:by1, bx0:bx1]).any()):
            return True
        return (self.res_edge is not None
                and bool((m & self.res_edge[by0:by1, bx0:bx1]).any()))

    def begin(self) -> list:
        """되돌릴 수 있는 구간을 연다 — 반환한 일지를 `rollback`에 준다.

        후보 경쟁이 쓴다: 같은 상태에서 여러 안을 실제로 지어 보고 이긴 하나만
        남긴다. 중첩은 안 한다 (열려 있으면 그 일지를 이어 쓴다).
        """
        if self._journal is None:
            self._journal = []
        return self._journal

    def rollback(self, journal: list, mark: int = 0) -> None:
        """`mark` 이후에 지운 잔여·띠를 되돌린다 (일지는 그 자리에서 자른다)."""
        for box, removed, bnd, edg in reversed(journal[mark:]):
            if box is None:
                self.residual |= removed
                if bnd is not None and self.band_res is not None:
                    self.band_res |= bnd
                if edg is not None and self.res_edge is not None:
                    self.res_edge |= edg
            else:
                bx0, by0, bx1, by1 = box
                self.residual[by0:by1, bx0:bx1] |= removed
                if bnd is not None and self.band_res is not None:
                    self.band_res[by0:by1, bx0:bx1] |= bnd
                if edg is not None and self.res_edge is not None:
                    self.res_edge[by0:by1, bx0:bx1] |= edg
        del journal[mark:]
        self._memo.clear()

    def end(self) -> None:
        """되돌리기 구간을 닫는다 (남은 확정은 그대로 굳는다)."""
        self._journal = None

    def _score_impl(self, lay: Layer):
        """(점수, 도형 마스크(bbox 창), 창 (x0,y0,x1,y1)) — 마스크 없으면 None.

        래스터·카운트는 도형 bbox 창으로 한정한다 (정수 평행이동이라 전장
        래스터와 결과 동일). ROI 전장 스캔 5회가 채점 비용의 대부분이었다
        (실측 호출당 수십 초 → 크롭 후 수 초)."""
        x0, y0, x1, y1 = self.roi
        polys = _poly_px(self.cat, lay, self.upp, self.w, self.h, x0, y0)
        # ROI 밖 초과 가드 — 채점은 ROI 안만 보므로, ROI를 크게 벗어나는 도형
        # (예: 반지름 큰 호)은 안에서 좋아 보여도 전역에선 대형 오염이다.
        # 실측: 선 획 호가 화면 절반짜리 쐐기로 찍혔다. 이 여유 구간은 채점
        # 밖(무벌점)이므로, ROI를 미리 넓혀 준 호출자는 guard로 좁혀야 한다
        # (실측: 채움 타원이 여유 구간의 피부 위로 뻗어 목을 가로질렀다)
        margin = (self.guard if self.guard is not None
                  else 24.0 + 0.25 * max(x1 - x0, y1 - y0))
        for p in polys:
            if (p[:, 0].min() < -margin or p[:, 1].min() < -margin
                    or p[:, 0].max() > (x1 - x0) + margin
                    or p[:, 1].max() > (y1 - y0) + margin):
                return -1e9, None, None
        rp = [np.round(p).astype(np.int32) for p in polys]
        bx0 = max(0, min(int(p[:, 0].min()) for p in rp) - 1)
        by0 = max(0, min(int(p[:, 1].min()) for p in rp) - 1)
        bx1 = min(x1 - x0, max(int(p[:, 0].max()) for p in rp) + 2)
        by1 = min(y1 - y0, max(int(p[:, 1].max()) for p in rp) + 2)
        if bx0 >= bx1 or by0 >= by1:
            return 0.0, None, None
        off = np.array([bx0, by0], np.int32)
        m = np.zeros((by1 - by0, bx1 - bx0), np.uint8)
        if len(rp) == 1:
            cv2.fillPoly(m, [rp[0] - off], 1)
        else:
            for p in rp:
                mm = np.zeros_like(m)
                cv2.fillPoly(mm, [p - off], 1)
                m ^= mm
        m = m.astype(bool)
        # 알파 프로파일 도형은 **가중해서** 센다 — 반투명한 자리는 그만큼만
        # 덮은 것이고 그만큼만 침범한 것이다. 단색이면 wt가 None이라 정수
        # 카운트로 간다 (기본 어휘에는 그라디언트가 없다)
        wt = _grad_alpha(self.cat, lay, self.upp, self.w, self.h,
                         x0, y0, bx0, by0, m.shape)

        def cnt(sel: np.ndarray) -> float:
            hit = m & sel
            return float(np.count_nonzero(hit) if wt is None else wt[hit].sum())

        if self.farbg is not None:         # §17 — 자격 판정 (점수 아님)
            if np.count_nonzero(m & self.farbg[by0:by1, bx0:bx1]):
                return -1e9, None, None
        res_box = self.residual[by0:by1, bx0:bx1]
        gain_box = (res_box if self.limit is None
                    else res_box & self.limit[by0:by1, bx0:bx1])
        new = cnt(gain_box)
        forb = cnt(self.forbid[by0:by1, bx0:bx1])
        bg = cnt(self.bg[by0:by1, bx0:bx1])
        waste = cnt(self._spill[by0:by1, bx0:bx1]) - forb - bg
        s = (new - _PEN_BG * bg - self.pen_forbid * forb - self.pen_waste * waste)
        if self.pen_out:
            s -= self.pen_out * cnt(self.outb[by0:by1, bx0:bx1])
        if self.band_res is not None:      # §24 — 덧칠 띠는 잔여처럼 닳는다
            s += _BAND_GAIN * cnt(self.band_res[by0:by1, bx0:bx1])
        if self.limit is not None:         # 밴드 밖 제 성분 위 잉크도 낭비다
            s -= self.pen_waste * cnt(self.mask[by0:by1, bx0:bx1]
                                      & ~self.limit[by0:by1, bx0:bx1])
        if self.core is not None:          # 이상 띠 밖의 물림은 공짜가 아니다
            cb_ = self.core[by0:by1, bx0:bx1]
            s -= self.pen_waste * cnt(self.free[by0:by1, bx0:bx1] & ~cb_)
            if _CORE_GAIN < 1.0:           # 띠 밖 잔여 = **이웃 가닥** (위 문서)
                s -= (1.0 - _CORE_GAIN) * cnt(gain_box & ~cb_)
        if self.retrace:                   # 이어긋기 — 덮인 밴드 위도 값이다
            trace_box = self.mask[by0:by1, bx0:bx1]
            fresh = new
            if self.limit is not None:
                trace_box = trace_box & self.limit[by0:by1, bx0:bx1]
            if self.core is not None:      # 이웃 가닥 재추적은 이어긋기가 아니다
                cb = self.core[by0:by1, bx0:bx1]
                trace_box = trace_box & cb
                # 새 잉크 쪽도 같은 띠로 좁혀야 항이 음수로 안 돈다 (이미 문
                # 잔여를 여기서 또 빼는 셈이 된다) — 이 항은 "**덮인** 자리도
                # 값이다"만 말한다
                fresh = cnt(gain_box & cb)
            s += self.retrace * (cnt(trace_box) - fresh)
        if self._farm is not None:         # 먼 이탈만 더 문다 (near는 오차)
            s -= self.pen_far * cnt(self._farm[by0:by1, bx0:bx1])
        if wt is not None:
            # 확정용 마스크는 **실제로 덮은 자리**로 좁힌다 — 알파가 옅은
            # 가장자리를 잔여에서 지우면 그 자리를 아무도 다시 안 줍는다
            m = m & (wt >= 0.5)
        return s, m, (bx0, by0, bx1, by1)

    def worth(self, m: np.ndarray | None,
              box: tuple[int, int, int, int] | None) -> float:
        """이 도형이 **새로 덮는** 자리의 값 (지각 가중 픽셀).

        점수(`_score_impl`)와 다른 자다: 점수는 벌점까지 넣어 **어디에 놓을지**를
        고르고, 값은 벌점 없이 **살지 말지**를 묻는다. 둘을 한 수로 합치면
        벌점이 큰 자리가 싸 보여 가격이 벌점의 함수가 된다.
        """
        if m is None or box is None:
            return 0.0
        bx0, by0, bx1, by1 = box
        hit = m & self.residual[by0:by1, bx0:bx1]
        if self.val is None:
            return float(np.count_nonzero(hit))
        return float(self.val[by0:by1, bx0:bx1][hit].sum())

    def account(self, m: np.ndarray | None,
                box: tuple[int, int, int, int] | None) -> dict:
        """이 도형이 덮는 자리의 **갈래별 회계** (px) — §8 후보 비교의 재료.

        점수(`_score_impl`)는 이것들을 한 수로 접어 버려 "어디에 놓을지"만
        답한다. 후보끼리 **다른 안**을 견줄 때는 갈래가 보여야 한다:

            new    새로 덮는 내 면 (값은 `worth`가 따로 잰다)
            forb   먼저 그린 면 침범 — 그 면이 그만큼 가려진다
            bg     투명 배경 침범 — 아무도 안 덮어 흰 자국이 남는다
            outb   실루엣 **밖** 물림 (`_SIL_BAND` 등급)
            waste  **나중에 그릴 면** 위 스필 — 그 면이 덮어 준다 (채움에선 무료)
            free   획이 덮는 자리·1px 물림 — 가려지므로 무료
            pen    위 갈래에 각자의 벌점을 곱한 합

        어휘가 전부 불투명이라(`_check_vocab`) 셈은 정수 카운트다.
        """
        if m is None or box is None:
            return {"new": 0.0, "forb": 0.0, "bg": 0.0, "outb": 0.0,
                    "waste": 0.0, "free": 0.0, "pen": 0.0}
        bx0, by0, bx1, by1 = box

        def cnt(sel: np.ndarray) -> float:
            return float(np.count_nonzero(m & sel[by0:by1, bx0:bx1]))

        forb = cnt(self.forbid)
        bg = cnt(self.bg)
        outb = cnt(self.outb) if self.pen_out else 0.0
        waste = cnt(self._spill) - forb - bg
        pen = (_PEN_BG * bg + self.pen_forbid * forb + self.pen_waste * waste
               + self.pen_out * outb)
        return {"new": cnt(self.residual), "forb": forb, "bg": bg, "outb": outb,
                "waste": waste, "free": cnt(self.free), "pen": pen}

    def worth_of(self, sel: np.ndarray) -> float:
        """마스크 하나가 담고 있는 값 (ROI 전장 마스크) — 배치 전 선별용."""
        if self.val is None:
            return float(np.count_nonzero(sel))
        return float(self.val[sel].sum())

    def score_val(self, lay: Layer) -> float:
        """점수만 (마스크 불필요한 하강·후보 비교용) — 메모 적용."""
        key = (lay.shape, lay.x, lay.y, lay.sx, lay.sy, lay.rot, lay.skew)
        s = self._memo.get(key)
        if s is None:
            s = self._score_impl(lay)[0]
            self._memo[key] = s
        return s

    def score(self, lay: Layer) -> tuple[float, np.ndarray]:
        x0, y0, x1, y1 = self.roi
        s, m, box = self._score_impl(lay)
        full = np.zeros((y1 - y0, x1 - x0), bool)
        if m is not None:
            full[box[1]:box[3], box[0]:box[2]] = m
        return s, full


def _descend(sc: _Scorer, lay: Layer, color, passes: int = 4,
             steer=None) -> tuple[float, Layer]:
    """좌표하강 — 축마다 ± 스텝, 개선 시 유지. 스텝은 패스마다 절반.

    `steer(layer) -> 벌점`은 **하강을 조종만 하는 항**이다 (px 점수 단위).
    하강은 `점수 − steer`를 올리지만 **돌려주는 점수는 언제나 순수 점수**다 —
    그 값으로 후보끼리 겨루고 게이트를 묻는 자리가 여럿이라(`_MIN_GAIN`,
    막대↔곡선 겨루기) 저울을 바꾸면 그 전부의 뜻이 갈린다. 조종 항은
    "같은 점수면 어느 자리에 설까"만 고른다.

    쓰는 자리는 획 사슬이다: 마디가 각자 제 픽셀 이득만 보고 자리를 잡으면
    이웃과 만나는 자리에서 방향이 꺾인다 (`chain` 문서). 마디마다 **제 경로
    조각의 양끝과 그 접선**을 조종 항으로 주면 이음이 저절로 맞는다 —
    이웃을 안 봐도 되는 것이 요점이라 재귀 분할이 경로를 어떻게 쪼개도 같은
    자가 선다.

    **기울기(skew) 축은 없다.** 그릴 수 있는 축만 민다 — 두 출력 경로 어느
    쪽도 기울기를 못 낸다: 창 조작에는 그 도구가 없고(`auto/run_plan`이 멈춘다),
    주입은 레코드에서 그 자리를 못 찾아 **조용히 빼고 쓴다**(`game/inject`).
    여기서 기울기를 밀면 그 이득은 점수판에만 남고 게임에는 안 올라가므로,
    도안(렌더)과 인게임이 그만큼 갈린다. 실측(7장, 이 축을 넣고 구운 판을
    skew=0으로 다시 렌더): 픽셀 0.70~1.89%가 어긋나고 셀 일치도 lpips가
    +0.0026~+0.0185 나빠진다 — 상수 하나를 놓고 채택을 가르던 폭보다 크다.
    """
    upp = sc.upp

    def obj(q: Layer) -> float:
        return sc.score_val(q) - (steer(q) if steer is not None else 0.0)

    best = obj(lay)
    # (속성, 스텝 열) — px 감각의 스텝을 게임 단위로 환산
    axes = (
        ("x", (2.0 * upp, 1.0 * upp, 0.5 * upp)),
        ("y", (2.0 * upp, 1.0 * upp, 0.5 * upp)),
        ("sx", (2.0 * upp / UNITS_PER_SCALE, 1.0 * upp / UNITS_PER_SCALE,
                0.5 * upp / UNITS_PER_SCALE)),
        ("sy", (2.0 * upp / UNITS_PER_SCALE, 1.0 * upp / UNITS_PER_SCALE,
                0.5 * upp / UNITS_PER_SCALE)),
        ("rot", (8.0, 3.0, 1.0)),
    )
    for p in range(passes):
        improved = False
        for name, steps in axes:
            st = steps[min(p, len(steps) - 1)]
            for sign in (1.0, -1.0):
                cand = Layer(**{**lay.__dict__})
                v = getattr(cand, name) + sign * st * (1 if getattr(cand, name) >= 0
                                                       or name not in ("sx", "sy") else -1)
                if name in ("sx", "sy") and abs(v) < 0.01:
                    continue
                setattr(cand, name, v)
                s = obj(cand)
                if s > best + 1e-6:
                    best, lay, improved = s, cand, True
                    break
        if not improved and p >= 1:
            break
    q = lay.quantized()
    return sc.score_val(q), q
