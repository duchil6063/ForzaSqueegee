"""cel·line 노선 2단 — 셀 영역(celart)·선화를 인게임 도형 레이어로 옮긴다.

사람 방식의 재현이 목표다. 근거는 `references/영상/` 5편과 제작자가
쓴 가이드(dcinside 포호갤 217078)다:

- **레이어는 비싸다 — 사람은 흉상 한 장을 400~700장으로 끝낸다.** 영상에서
  읽은 완성 시점의 카운터: 호시노 아이 185 · 이슈타르 397 · 린도 치하야 639 ·
  프리렌 664 · FH6 타임랩스 724 (모두 /3000). 가이드의 얼굴 데칼은 147장,
  눈동자 하나가 18장이다. 상한 3,000을 채우는 도안은 사람의 4~15배를 쓰는
  것이고, 그 값은 그대로 사용자가 문다 — 주입은 그 장수만큼의 템플릿 그룹을
  인게임에서 먼저 만들어야 하고 창 조작은 장당 6초다. 그래서 이 패키지의
  게이트는 언제나 **이 한 장이 제 값을 하는가**를 묻는다.
- **면 = 같은 색 불투명 도형 몇 장의 덮개.** 셀 면은 단색이라 도형끼리 겹친
  이음새가 안 보인다 — painter의 모자이크 티가 원리적으로 없다. 문제는 외곽
  경계의 정확도뿐이다.
- **그리기 순서가 배치 자유도다.** 큰 면부터 그리므로, 지금 면이 나중(더 작은)
  면 밑으로 삐져나가는 것은 허용된다 — 나중 면이 덮는다. 앞서 그린 면과 투명
  배경 쪽 침범만 벌점을 문다. 가이드가 말하는 "이걸 뒤로 보내면 기존에
  만들어뒀던 파츠가 모서리를 전부 다 커버해줘서 채색이 한 방에 완성"이다.
- **가는 영역 = 획이고, 한 획은 도형 한 장이다.** 뼈대(세선화) 경로에 곡선
  도형을 통째로 맞춘다 — 사람이 곡선·부메랑 한 장으로 긋는 그 문법이다.
  FH5 튜토리얼은 클립보드에 곡선 한 종을 담아 놓고 회전·크기만 바꿔 반복해
  선화 전체를 230장으로 끝낸다. 한 장으로 안 되면 쪼개서 곡선 여러 장으로
  가고, **막대(A_22)는 확실히 직선인 자리에만** 쓴다 (`stroke._is_straight`).
  한 획이 쓸 수 있는 장수에는 상한(`policy.max_shapes`)이 걸린다.
- **선 도안은 선에 적합한 도형만 쓴다.** 어휘를 손으로 고르지 않고 **놓인 뒤의
  모습**을 재서 거른다 — 폭이 고르고(테이퍼) 길이에 비해 가늘어야(폭/길이)
  획이다 (`stroke._STROKE_TAPER`·`_STROKE_SLIM`, 레퍼런스 실측). 같은 도형도
  비등방 스케일에 따라 획이 되기도 잎사귀가 되기도 한다.
- **획 폭 상한은 사람이 그은 폭이다** — 짧은 변의 0.34% (`grammar._LINE_W_REL`,
  레퍼런스 실측). 신경망 선화가 두 선을 한 띠로 붙여 준 자리를 굵기 그대로
  그리면 검은 덩어리가 되므로 그 폭으로 눌러 긋는다.
- 도형 어휘는 창 조작 가능한 단색 도형(cell_map)만 쓴다 — run·inject 양 경로
  모두 그릴 수 있는 플랜만 낸다.

탐색은 결정적이다: **닫힌 해로 씨앗을 잡고**(면은 2차 모멘트 정합
`fill._seed_moment`, 획은 아핀 최소제곱 `stroke._affine_fit`) 그 자리에서
좌표하강으로 다듬는다. 하강은 축마다 스텝을 줄여 가며 점수를 올린다. 점수 =
새로 덮은 내 면 픽셀 − 4×금지(먼저 그린 면·배경) − 0.12×낭비(나중 면 위 스필).
채점 폴리곤 기하는 render._draw_layer와 같은 식이라 "플랜 렌더 = 채점 결과"가
보장된다.

## 모듈 구성 — 아래로 갈수록 위를 쓴다

    geometry    레이어 → 화면 픽셀 (폴리곤·마스크·잉크 지도). 밑판
    descriptor  도형 서술자 — 놓인 뒤의 모습으로 어휘를 고른다 (오프라인 계측)
    vocabulary  도형 어휘 — 무엇을 낼 수 있나 (막대는 목표 폭이 고른다)
    price       가격 λ — 한 장이 얼마를 벌어야 사나
    scoring     채점판 `_Scorer`·좌표하강 — 어디에 놓나
    skeleton    세선화·경로 — 마스크를 획의 중심선으로
    bridge      끊긴 획 잇기 (배치 전, 선 지도 위에서)
    evidence    선 증거 — 신경망 출력을 마스크가 아니라 증거원으로
    graph       획 그래프 — 이어긋기·역할 판정 (구조선/특징선/무늬)
    intent      획의 의도 — 어디서 끊나 (평활이 지킨 각이 곧 마디)
    stroke      획 맞춤 — 곡선 한 장, 안 되면 쪼개서 곡선 (막대는 직선일 때만)
    candidates  후보 경쟁 — 여러 안을 지어 양자화 렌더에서 겨루게 한다
    chain       획 사슬의 이음매 — 마디끼리 맞춘다 (한 획을 하나로 본다)
    policy      노선 정책 — 두 노선이 갈리는 칸을 한 자리에
    engine      **공통 선 재구성 엔진** — 두 노선이 함께 쓴다
    fill        면 채움 — 씨앗·성장·껍질 막대·마무리
    layered     면 채움의 층 쌓기 — 큰 바탕 한 장 먼저, 장수를 값으로 센다
    residual    잔차 진단 — 종류마다 이름을 붙이고 초점을 낸다
    metrics     회귀 지표 — 도형 수도 품질이다 (구조·맞물림·보존)
    select      획 선별 문턱 — 경계성·실루엣성을 역할로 가르는 자
    grammar     사람 선따기 문법 — 폭 정책·덩어리 채움·이음 보수
    linemetrics 선 도안 구조 지표 — 폭 충실도·파편화·꺾임 (게이트 아님)
    merge       겹침 병합 — 같은 방향으로 겹친 막대를 하나로 (배치 뒤)
    carve       덮어서 그리기 — 최소 도형보다 가는 선 (cel 폴백 전용)
    lines       엔진 구동 `_fit_lines`
    holes       구멍·커버리지 — 재고(게이트) 메운다(성장·메움). **가격의 자**
    coverage    §18 커버리지 불변 — 미커버 표본 0 봉인. **불변의 자** (λ 무관)
    repair      잔차 수리
    plan        진입점 `fit_plan` / `fit_line_plan`

밖에서 쓰는 것은 이 파일이 다시 내보내는 이름뿐이다 — 노선(`route_cel`·
`route_line`)은 `fit_plan`·`fit_line_plan`·`bridge_line_gaps`와 마무리 일체를,
미세 조정(`finetune`)과 계측 도구(`tools/`)는 그 아래 부품(`_poly_px`·`_Scorer`
등)을 그대로 부른다. 밑줄 이름을 다시 내보내는 것은 그래서다.

**모듈 전역을 밖에서 바꿔 스윕할 때는 그 상수가 실제로 사는 모듈을 밀어야
한다** — 여기 다시 내보낸 이름에 대입해 봐야 정작 읽는 함수는 안 본다
(예: `celfit.scoring._STROKE_R`).
"""

from __future__ import annotations

from ..price import fix_min_gain, price_of, repair_min_gain
from . import policy
from .bridge import bridge_line_gaps
from .candidates import Candidate
from . import coverage
from .descriptor import (descriptors, straight_thin, stroke_shapes,
                         thin_shapes)
from .engine import Reconstruction, build_strokes, place_strokes
from .evidence import EvidenceMaps, StrokeEvidence, build_maps
from .fill import _fit_bars, _place_fat
from .geometry import _ink_cover, _min_span, _poly_px
from .graph import ROLES, LogicalStroke, classify, continue_strokes
from .holes import (count_hole_clusters, fill_holes, grow_covers,
                    silhouette_cover)
from .layered import est_shapes, fill_region, mop_up
from .linemetrics import stroke_metrics
from .lines import _fit_lines
from .metrics import plan_metrics
from .plan import fit_line_plan, fit_plan
from .repair import repair_mismatch
from .residual import analyze as residual_analyze
from .residual import focus_layers, owner_map
from .scoring import (_COVER_STOP, _MAX_PER_REGION, _MIN_GAIN, _PEN_WASTE,
                      _PEN_WASTE_FILL, _STROKE_R, _Scorer)
from .select import _THIN_BND, _THIN_SIL
from .skeleton import (_dt_along, _end_dir, _join_paths, _paths, _prune_spurs,
                       _thin)
from .stroke import _stroke_forms
from .vocabulary import shape_vocabulary

__all__ = [
    # 노선 진입점
    "fit_plan", "fit_line_plan", "bridge_line_gaps",
    # 공통 선 재구성 엔진 — 두 노선이 함께 쓴다
    "policy", "build_maps", "EvidenceMaps", "StrokeEvidence",
    "LogicalStroke", "ROLES", "classify", "continue_strokes",
    "build_strokes", "place_strokes", "Reconstruction", "Candidate",
    "descriptors", "stroke_shapes", "thin_shapes", "straight_thin",
    # 마무리 — 파이프라인이 배치 뒤에 돌린다
    "silhouette_cover", "count_hole_clusters", "grow_covers",
    "fill_holes", "repair_mismatch", "coverage",
    # 가격 λ
    "price_of", "fix_min_gain", "repair_min_gain",
    # 면 채움의 층 쌓기 · 잔차 진단 · 회귀 지표
    "fill_region", "mop_up", "est_shapes",
    "residual_analyze", "focus_layers", "owner_map", "plan_metrics",
    "stroke_metrics",
    # 어휘 — 저장 템플릿에 심을 씨앗 도형 목록
    "shape_vocabulary",
]
