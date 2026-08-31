r"""이타샤 구성 설계 — 도안·면 실측에서 **이타샤 한 대**를 짠다 (게임 안 건드림).

`auto/itasha.py`는 "면에 그룹을 앉히는 손"이고, 이 모듈은 **무엇을 어디에 어떻게
앉힐지 정하는 머리**다. 둘을 받아 하나를 낸다:

    도안 플랜(들) + 면 실측 지도(`game/surface`)  →  이타샤 구성

자동이든 손 배치든 **한 길**이다 — 자동 경로는 도안 자리를 계산해 손 배치
목록으로 바꿔 놓고(`build`), 그 뒤(면 밖 자르기·꾸밈 그룹·관통 띠·
지붕 블랙아웃)는 같은 코드가 짓는다.

## 이타샤 문법 (근거: 레퍼런스 8장 + 제작 영상 + 업계 지침 조사)

- **베이스는 단색 자동차 도색**이다 (흰/검이 다수, 상징색이 확실한 도안만
  테마색 — `base_paint`) — 비닐로 덮지 않는다. 장수 0장이고 도료 질감(광택·
  메탈릭)은 비닐로 낼 수 없다. 편집기는 이 값을 초기값으로 주고 사람이
  바꾼다 (`build(base_rgb=…)`).
- 측면 = 주역 캐릭터 대형. 좌우 미러(또는 서로 다른 캐릭터).
- 인물은 **아래로 붙인다** — 발이 사이드실. 벨트라인은 안 넘는다.
- **면을 넘긴 그림은 그 자리에서 잘린다** (사용자 지시 2026-08-27) — 이웃 면으로
  자동으로 안 잇는다. 감아 돌리려면 편집기에서 도안을 이음선으로 가르고
  (KFPS·FLS의 [선으로 가르기]) 한쪽을 그 면에 따로 올린다.
- 꾸밈 모티프는 인물 **곁에** 선다 (`deco_layers`의 `avoid`) — 레퍼런스의 큰
  별·꽃은 예외 없이 인물 옆 맨 도색 위에 있고, 인물 위를 지나는 것은 잔것
  몇 송이뿐이다.
- **꾸밈은 면이 아니라 도안에서 자란다** (`DecoAnchor`) — 무리의 핵이 도안
  뒤쪽에 있고, 크기가 도안 크기의 몫이며(패널이 상한을 쥔다), 도안이 없는 면의
  무리는 이웃 면의 도안 상자를 이음새 너머로 투영해 그 자리에서 자란다
  (Fate R34의 리어 별무리는 리어 쿼터에서 들어온다). 자리를 정하는 문법은
  캔버스 꾸밈 그룹과 면 도형이 **한 벌을 나눠 쓴다** (`scatter_motifs`).
- 인물 뒤에는 **꾸밈 그룹**(로커 밴드·큰 색면·무리·에코)이 제 그룹으로 깔리고,
  로커·큰 색면·모티프가 이웃 면으로 이어져 차 전체를 접착한다
  (`design`·`atlas`·`flow_shapes`).
- **사람 배치를 읽고, 후보를 만들고, 재서 고른다** (`intent` → `field` → `design`).
  사람이 앉힌 도안은 어느 후보에서도 안 움직인다.

## 구도를 어떻게 짜나 — 요소가 아니라 **요소 사이**

옛 구조는 "무엇을 놓나"의 목록이었다: 계열이 판 하나와 조각 몇을 내고 점수가
그 결과를 쟀다. 33판을 구워 재 보니 그 길의 천장이 분명했다 — 점수 열한 항목
중 넷이 모든 판에서 정확히 1.000이고, 값이 갈리는 것은 "실루엣 테두리 명도차"
하나뿐이며, 1위와 2위의 차 중앙값이 0.0000이었다 (후보를 고른 것은 점수가
아니라 후보를 짓는 **순서**였다). 그 하나뿐인 자를 최대로 만드는 길이 인물 뒤에
반대 명도의 큰 판을 통째로 까는 것이라, 판 넓이가 인물의 5.8배였다.

지금 구조는 다섯 층이다.

1. **읽는다** (`intent`) — 실루엣·머리·포즈 축에 더해 **몸짓**을 읽는다: 윤곽
   거리 함수의 봉우리가 뻗은 팔·머리카락·무기이고, 그 돌출이 세기다. 열린 쪽·
   어두운 덩어리·시선도 같이 읽는다. 모델은 안 쓴다.
2. **자리를 편다** (`field`) — 보호·지지·장식·여백 구역과 흐름. 흐름은 빈 자리·
   몸짓·얼굴·열린 쪽이 함께 고른다.
3. **관계를 적는다** (`graph`) — 노드는 역할(hero·macro·counter·motif·echo·
   text·negative·trim)이고 관계는 잴 수 있는 것만 둔다. 계열은 "무엇과 무엇이
   어떤 사이여야 하나"만 말하는 **문법 프리셋**으로 내려앉았다 (`families`).
4. **기하를 낸다** (`macro`·`rhythm`) — 큰 색면은 아홉 어휘의 연속 매개변수고
   (`macro`), 조각은 곡선 하나를 걸어 나온다 (`rhythm` — 원점에서 잦아들며).
   둘 다 **어느 변으로 프레임을 나가나**가 정의의 일부다.
5. **잰다** (`critic`·`score`) — 같은 합성을 세 배율로 갈라 배율마다 다른 것을
   묻고(멀리 누가 읽히나 · 덩어리 무게가 갈리나 · 조각이 리듬인가), 그래프의
   문법이 기하에서 지켜졌나를 잰다. 자에 넣는 조건은 하나다: **후보에 따라
   값이 실제로 갈릴 것.** 만들기 나름으로 늘 1.000이 되는 항목은 자가 아니라
   희석이다.

탐색은 두 단이다 (`design`): 대표 팔레트 하나로 **기하만** 먼저 재 계열마다
하나씩 살려 여섯을 남기고, 거기에만 팔레트·키라인·글자를 붙여 다시 잰 뒤
이긴 둘을 좌표하강으로 다듬는다. 난수는 없다.
- 윗면 = 후드 인물(도안 재사용) + 지붕 블랙아웃.
- **글자는 기본으로 안 넣는다** — 스폰서 이름·넘버판 숫자 같은 텍스트는 어휘에
  없다. 사람이 캐릭터 이름을 **넣기로 하면**(`textspec.TextSpec`) 그 글자가 꾸밈의
  한 요소로 후보에 들어간다: 커스텀 텍스트 도안(동봉 OFL 글꼴 → 잉크에 내접하는
  막대·원·삼각형, `engine.textglyph`·`textfit`)이 기본이고 면 예산이 모자라면 정책
  사다리를 거친 칸으로 내리다(층 A→B→C) 게임 글꼴 비닐(D)로 물러나고 그래도 안
  되면 뺀다(E) (`textbudget`). 자리는 필드가 정한다 (`textlayout` — 워드마크 ·
  인물 뒤 · 로커 위 · 사인 · 두 줄 락업) 그리고 점수가 고른다 (`textscore`).

## 크기는 계산해서 나온다

프리셋 상수(측면 0.25)는 차 한 대·도안 한 장에서 잰 값이라 다른 차·다른 종횡비
에서 어긋난다. 실측 지도가 있으면 **도색 마스크에 도안 비율의 최대 내접 상자**를
앉혀(`SurfaceMap.fit`) 그 상자에서 스케일과 이동을 역산한다. 지도가 없는 면은
프리셋으로 물러난다 (`auto.itasha.PRESET`).


## 모듈 구성 — 아래로 갈수록 위를 쓴다

    boxes       상자 산술과 자잘한 자 — 이 패키지의 밑판 (아무것도 안 쓴다).
    look        도안 읽기 — 플랜 한 장이 면에서 **어떤 생김새인가**.
    palette     색 — 베이스 도색 · 액센트 · 테마색.
    vocabulary  모티프 어휘 — 어느 계열의 어느 도형을 쓰나.
    scatter     산포 문법 — 조각을 어디에 몇 개 흩나 (캔버스·면이 나눠 쓴다).
    bands       로커 밴드 — 차체 하부를 채우는 투톤 면과 그 찢긴 윗선.
    roof        지붕 블랙아웃 — 윗면의 후드 뒤 구간을 검정으로 덮는다.
    place       배치 — 도안을 면 어디에 얼마로 앉히나 (자리 수학과 그 밑감).
    folds       면 이음새 접기 그래프 — 꾸밈 뿌리를 이웃 면으로 투영하는 자.
    atlas       차 한 대의 면 지도 — 이음새·차체 선, 그리고 그것을 **건너는** 자.
    autoplace   자동 자리 — 편집기가 도안을 처음 앉히는 그 자리.
    surfshapes  면에 직접 놓는 꾸밈 — 관통 띠 · 산포 모티프의 도형 명세.
    intent      도안 읽기 2단계 — 실루엣 · 머리 · 포즈 축 · 디테일 · 색 역할 씨앗.
    graph       구성 그래프 — 요소가 아니라 요소 **사이**를 적는다 (관계 문법).
    roles       역할 팔레트 — 베이스 · 베드 · 주/부 액센트 · 그림자 · 하이라이트 · 무채.
    field       구성 필드 — 배치 둘레의 보호 · 지지 · 장식 · 여백 구역과 흐름.
    bed         캐릭터 베드 — 인물 뒤 큰 색면 (판 · 쐐기 · 슬래브 · 덩어리) + 키라인.
    echo        그래픽 에코 — 인물의 결 · 뾰족함 · 블록을 되풀이하는 잔 조각.
    rhythm      리듬 곡선 — 조각이 원점에서 잦아들며 흘러 나오는 자리.
    macro       매크로 기하 — 인물 뒤 큰 색면의 어휘 (띠·겹·날·화살·가름·모서리…).
    families    구성 계열 — minimal · graphic_bed · diagonal_flow · motorsport · splash.
    critic      구도 비평 — 같은 합성을 세 배율(멀리·중간·가까이)로 재는 자.
    score       구도 점수 — 후보 한 벌을 옆면 한 장으로 합성해 재는 자.
    textspec    텍스트 스펙 — 사람이 넣은 이름·작품명과 옵션 (기본 꺼짐).
    textstyle   텍스트 스타일 — 계열·인상이 글꼴을, 역할 팔레트가 색을 준다.
    textbudget  텍스트 예산 — 남은 장수·우선순위로 층(A~E)을 정한다.
    textlayout  텍스트 배치 — 워드마크 · 로커 위 · 사인 포즈를 필드에서 찾는다.
    textbuild   텍스트 조립 — 포즈 + 층 → 프레임 좌표 레이어 (또는 게임 글자 명세).
    textscore   텍스트 점수 — 가독성 · 가림 · 흐름 · 어수선 · 위계.
    facetext    다른 면의 글자 — 자리를 못 박았을 때 (rear · hood · roof · window).
    design      구성 설계 — 후보 생성 + 평가 + 선택 (옆면 꾸밈 그룹의 머리).
    rigs        차 한 대의 면 지도와 옆면 뼈대 — 실측이 프리셋보다 우선한다.
    groups      구성 파일의 그룹 항목 — 플랜 파일을 쓰고 그것을 가리킨다.
    build       구성 한 대 짜기 — 이 패키지의 진입점.

밖에서 쓰는 것은 이 파일이 다시 내보내는 이름뿐이다 — 갈라 놓기 전과
같이 `compose.<이름>`으로 전부 닿는다 (밑줄 이름도 도구가 부른다).
**모듈 전역을 밖에서 바꿔 스윕할 때는 그 상수가 실제로 사는 모듈을
밀어야 한다** — 여기 다시 내보낸 이름에 대입해 봐야 정작 읽는 함수는
안 본다.
"""

from __future__ import annotations

# 갈라 놓기 전 이 모듈이 들여와 쥐고 있던 이름 — `engine.preview`가 캔버스 유닛
# 환산을 `compose.UNITS_PER_SCALE`로 읽는다 (같은 자를 두 번 두지 않는다).
from ..model import UNITS_PER_SCALE
from .boxes import (
    CANVAS_UNITS, DEFAULT_GROUP_UNIT, _clamp_box, _face_phase, _gap, _group_unit,
    _overlap, _rel, _union)
from .look import (
    Look, PALE_B, PALE_S, _is_pale, layer_points, look, person_ink, rot_ink,
    rot_ink_box)
from .palette import (
    ACCENT_B_GAP, ACCENT_DARK_MAX, ACCENT_GREY_MAX, ACCENT_HUE_GAP, ACCENT_HUE_STEP,
    ACCENT_LIGHT_MIN, ACCENT_SRC_MIN, ACCENT_S_MIN, ACCENT_WARM, BASE_BLACK,
    BASE_HUE_NEAR, BASE_PALE_SHARE, BASE_SAT_MIN, BASE_THEME_SAT, BASE_THEME_SHARE,
    BASE_WHITE, INK_DARK, INK_LIGHT, MOTIF_THEME, PERSON_HUE_NEAR, PERSON_VAL_GAP,
    RETREAT_SAT, SKIN_HUE, SKIN_SAT_MAX, SKIN_VAL_MIN, THEME_AREA_MIN,
    _separate_from_person, accent_color, accent_third, accent_tint,
    achromatic_accent, base_paint, contrast_ink, dominant, is_skin, readable_on,
    theme_color)
from .vocabulary import (
    EDGE_SETS, MOTIF_FAMILIES, MOTIF_INSCRIBE, MOTIF_NEUTRAL, MOTIF_SETS, _RING8,
    edge_shapes, motif_family, motif_shapes, shape_half)
from .scatter import (
    DECO_ANCHOR_GAP, DECO_FALLOFF, DECO_FRONT_N, DECO_FRONT_SIZE,
    DECO_GAP_MAX, DECO_HERO_CAP, DECO_N, DECO_SEP, DECO_TIER, DECO_TIER_MAX,
    DECO_TIER_SIZE, HALO_GROW, Motif, deco_layers, scatter_motifs)
from .bands import (
    ROCKER_BASE_MIN, ROCKER_FRAC, ROCKER_TEETH, TEETH_AMP, TEETH_OVERLAP, _teeth,
    stripe_layers)
from .roof import (
    ROOF_DARK, ROOF_MIN_FRAC, ROOF_TEETH, ROOF_TEETH_AMP, hood_index, roof_blackout,
    top_segments)
from .place import (
    BODY_BIAS, BODY_FILL, EXPOSED_FLOOR, EXPOSED_FULL, FACE_FRAC_BUST,
    FACE_FRAC_TALL, LIE_GAIN_MIN, LIE_HEAD_REAR, LIE_MAX, LIE_TIE, ManualPlace,
    PART_PAD, Place, ROLE_EXTRA, ROLE_MAIN, ROLE_REAR, SideRig, TILT_ASPECT,
    TILT_FULL, TILT_MAX, _refit_canvas, dodge_parts, door_span,
    drawable, face_zone, fit_on, layers_on, manual_box, person_pose, person_tilt,
    place_in_rect, place_xf, surface_exposure, take_layers)
from .folds import _all_folds, _pillar_hints, seam_fold
from .atlas import BodyLines, Seam, VehicleAtlas, build_atlas
from .autoplace import _side_place, auto_place, mirror_place
from .surfshapes import (
    DECO_REACH, DecoAnchor, FACE_ROCKER_FRAC, FLOW_TEETH, GLASS,
    deco_anchor, flow_shapes, surface_deco_shapes)
from .intent import DesignIntent, read_intent
from .roles import RolePalette, role_palette
from .field import CompositionField, build_field
from . import critic
from .critic import Critique, critique, heatmaps
from .graph import (
    CompositionGraph, DEFAULT_GRAMMAR, Node, RELATIONS, ROLES, Rel, derive,
    relation_score)
from .bed import bed_layers, keyline_layers
from .echo import echo_layers
from .families import FAMILIES, FAMILY_NAMES, Family, rank_families
from .rhythm import Beat, RhythmCurve, beats, curve_for
from .macro import (
    KINDS as MACRO_KINDS, MACRO_AREA_MAX, MacroSpec, macro_layers,
    plan as macro_plan)
from .score import ScoreCard, composite, raster_layers, score_design
from .design import DECO_FRAME_FILL, Design, compose_design
from .textspec import PLACEMENTS as TEXT_PLACEMENTS, STYLES as TEXT_STYLES, TextSpec
from .textbudget import TextPlan, plan_tiers
from .textlayout import TextPose, layout_sets
from .textbuild import TextSet, build_text_sets
from .textscore import TEXT_WEIGHTS, text_parts
from .facetext import face_text
from .rigs import (
    _arch_fallback, _avoid_on, _bumper_seed, _hood_seed, _place_for, carfiles_pick,
    probe_ok, side_rigs, surfaces_for)
from .groups import (
    _hand_group_job, _hand_spread, _plan_sig, _unique_group_counts)
from .build import (
    HOOD_MIN_FRAC, HOOD_ROT_SIGN, HOOD_TILT, Recipe, _deco_usable, _hood_place, build)
