"""명령줄 문법 — 하위 명령과 그 인자 (`argparse` 정의 한 벌)."""

from __future__ import annotations

import argparse

from ..i18n import msg


def _text_args(p: argparse.ArgumentParser) -> None:
    """캐릭터 이름 글자 인자 — `itasha`와 `flsedit text`가 같은 벌을 쓴다.

    문자열은 **그대로** 간다 (띄어쓰기·대소문자·구두점; `\\n`은 줄바꿈)."""
    p.add_argument("--text", default=None, metavar=msg("이름"),
                   help=msg("캐릭터 이름 글자를 꾸밈에 넣는다 (예: \"Asuka Langley\"). "
                            "커스텀 텍스트 도안(동봉 OFL 글꼴)이 기본이고 면 예산이 "
                            "모자라면 층을 낮추다 게임 글꼴로 물러난다"))
    p.add_argument("--subtext", default=None, metavar=msg("작품명"),
                   help=msg("보조 글자 — 작품명·별칭·팀명 (메인 밑에 작게)"))
    p.add_argument("--text-number", default=None, dest="text_number", metavar=msg("번호"),
                   help=msg("레이싱 번호 — 리어 쿼터의 큰 숫자 (레이싱 스폰서 프리셋에서만 선다)"))
    p.add_argument("--text-style", default=None, dest="text_style",
                   help=msg("auto · script · brush · graffiti · racing · techno · minimal "
                            "(기본 auto: 구성 계열이 고른다. 스타일이 글꼴을 고른다)"))
    p.add_argument("--text-engine", default=None, dest="text_engine",
                   choices=["font", "shapes"],
                   help=msg("font(기본: 게임 글꼴 글리프, 한 글자 한 장) · "
                            "shapes(동봉 OFL 글꼴을 도형으로 되짓기 — 고운 층이 예산에 들 때만)"))
    p.add_argument("--text-placement", default=None, dest="text_placement",
                   help=msg("auto · side · rear · hood · roof · window (기본 auto = 옆면)"))
    p.add_argument("--text-priority", default=None, dest="text_priority",
                   help=msg("high · normal · low — high면 산포·에코보다 글자가 먼저다"))
    p.add_argument("--game-text-fallback", default=None, dest="game_text_fallback",
                   choices=("on", "off"),
                   help=msg("예산이 모자랄 때 게임 글꼴 비닐로 물러나나 (기본 on)"))
    p.add_argument("--text-max-layers", default=None, type=int, dest="text_max_layers",
                   help=msg("글자에 쓸 장수 상한 (층을 낮추는 시험용 레버)"))
    p.add_argument("--text-outline", default=None, dest="text_outline",
                   choices=("auto", "on", "off"), help=msg("테두리"))
    p.add_argument("--text-shadow", default=None, dest="text_shadow",
                   choices=("auto", "on", "off"), help=msg("그림자"))


def _logo_args(p: argparse.ArgumentParser) -> None:
    """로고 인자 — `itasha`와 `flsedit decorate`가 같은 벌을 쓴다."""
    p.add_argument("--logo", action="append", default=None, metavar=msg("이미지"),
                   help=msg("사용자 로고 이미지(또는 도안) — 여러 번 줄 수 있다. 셀 노선으로 "
                            "벡터화해(300장 상한) 스폰서 문법으로 앉힌다"))
    p.add_argument("--logo-placement", default=None, dest="logo_placement",
                   choices=("auto", "rear", "front", "windshield", "rocker"),
                   help=msg("로고 자리 — auto(워터마크는 리어, 로고는 옆면 로커 줄·리어·"
                            "프론트) · rear · front · windshield · rocker(옆면 줄만)"))


def _face_args(p: argparse.ArgumentParser) -> None:
    """면 배정 인자 — `itasha`와 `flsedit decorate`가 같은 벌을 쓴다 (`compose.facespec`)."""
    p.add_argument("--face", nargs="*", default=None, metavar=msg("면=모드"),
                   help=msg("면이 맡는 일 — `window=auto|support|continue|crop|empty` · "
                            "`rear_window|rear|front=auto|logos|crop|empty` · "
                            "`windshield=auto|logos|empty`. auto는 로고·글자가 있으면 "
                            "그것(리어 워드마크+로고 줄 · 윈드실드 글자 띠 · 유리 로고 열), "
                            "없으면 크롭으로 물러난다 (리어·뒷유리는 비운다)"))


def build_parser() -> argparse.ArgumentParser:
    """`python -m forzasqueegee`의 파서. 명령 실행은 `cli.<갈래>`가 한다."""
    parser = argparse.ArgumentParser(prog="forzasqueegee",
                                     description=msg("FH6 리버리 제작 보조 툴"))
    # 기본 None = 저장된 언어(`work/state/lang.json`, 없으면 ko)를 쓴다.
    # 값을 주면 **이번 실행만** 덮는다 — 아예 저장하려면 `lang` 명령.
    parser.add_argument("--lang", default=None, choices=("ko", "en"),
                        help=msg("UI 언어 — 이번 실행만 (저장은 `lang` 명령)"))
    parser.add_argument("--no-admin", action="store_true",
                        help=msg("관리자 권한을 묻지 않는다 (메모리 주입은 못 쓴다). "
                                 "권한 승격으로 다시 뜬 프로세스도 이 표시를 달고 온다"))
    parser.add_argument("--game-dir", default=None, metavar=msg("경로"),
                        help=msg("FH6 설치 폴더를 **이번 실행에만** 못 박는다 "
                                 "(media 폴더가 있는 곳). 자동 탐색은 Steam 규약만 "
                                 "알아서 Game Pass·옮겨 온 설치본을 못 찾는다. "
                                 "아예 저장하려면 `gamedir` 명령"))
    sub = parser.add_subparsers(dest="command", required=True)

    p_mk = sub.add_parser("make", help=msg("이미지 하나 → 도안 하나 (배치·자가 점검을 한 명령으로)"))
    p_mk.add_argument("image", help=msg("원본 이미지 (PNG/JPG, 투명 배경 가능)"))
    p_mk.add_argument("-o", "--out", required=True, help=msg("출력 폴더"))
    p_mk.add_argument("--route", choices=("cel", "painter", "line"),
                      default="cel",
                      help=msg("노선 — 사람이 고른다 (자동 판별은 만들지 않는다). "
                               "cel(기본)은 셀 재해석 + 가격 설계로 장수를 그림이 "
                               "정한다, painter는 KFPS 동일 로직 (GPU, --shapes가 "
                               "목표 장수), line은 원화의 선만 획으로 딴다 "
                               "(면 채움 없음 — 차 도색 위에 선화만 얹는 도안)"))
    p_mk.add_argument("--shapes", type=int, default=3000, help=msg("레이어 상한 (≤3000)"))
    p_mk.add_argument("--size", type=int, default=1200, help=msg("[디버그] 작업 해상도"))
    p_mk.add_argument("--cut-bg", action="store_true",
                      help=msg("배경 제거 전처리를 켠다 (알파 없는 입력에서 신경망으로 인물만 딴다)"))
    p_mk.add_argument("--no-crop", action="store_true",
                      help=msg("크롭 전처리를 끈다 (인물 bbox로 안 자르고 프레임 그대로 쓴다)"))

    p_gui = sub.add_parser("gui", help=msg("제품 창 — 이미지 끌어다 놓기 → 레이어 수 → 생성"))
    p_gui.add_argument("image", nargs="?", default=None,
                       help=msg("열자마자 물릴 이미지 (없으면 창에서 고른다)"))

    p_fp = sub.add_parser("painter",
                          help=msg("KFPS 동일 로직 도형 근사 — GPU (painter 산출물)"))
    p_fp.add_argument("image", help=msg("입력 이미지 (PNG/JPG)"))
    p_fp.add_argument("-o", "--out", required=True, help=msg("출력 폴더"))
    p_fp.add_argument("--shapes", type=int, default=0,
                      help=msg("목표 레이어 수 (0 = 프리셋 stopAt)"))
    p_fp.add_argument("--preset", choices=("shaded", "flat", "gradients"),
                      default="shaded",
                      help=msg("KFPS 프리셋 — shaded(기본, 애니/디지털 아트), "
                               "flat(스티커·로고·평면색), gradients(부드러운 그라데이션)"))
    p_fp.add_argument("--seed", type=int, default=0,
                      help=msg("원시 생성 씨드 (0 = 무작위, KFPS 기본)"))
    p_fp.add_argument("--repair", action="store_true",
                      help=msg("국소 수리(Edge Repair)를 켠다 (KFPS 기본 꺼짐)"))
    p_fp.add_argument("--luma", choices=("on", "off"), default=None,
                      help=msg("루마 밴딩 전처리 강제 (기본: 프리셋 값 — flat만 켬)"))
    p_fp.add_argument("--heatmap", action="store_true",
                      help=msg("세부 열지도 유도를 켠다 (KFPS 기본 꺼짐)"))
    p_fp.add_argument("--boost", action="store_true",
                      help=msg("표본 2배 (KFPS Vroom Boost)"))
    p_fp.add_argument("--finalize-only", action="store_true",
                      help=msg("원시 생성 없이 남아 있는 체크포인트로 마무리만 다시 돈다"))

    p_in = sub.add_parser("inject", help=msg("메모리 주입으로 플랜 적용 (FP와 같은 경로)"))
    p_in.add_argument("plan", nargs="?", help=msg("도안 경로 (--probe면 생략)"))
    p_in.add_argument("--probe", action="store_true",
                      help=msg("읽기 전용 조사 — 그룹·레이어 배치 확인 (쓰기 전 필수)"))
    p_in.add_argument("--count", type=int, default=None,
                      help=msg("찾을 그룹의 레이어 수 (probe 보조)"))
    p_in.add_argument("--force", action="store_true",
                      help=msg("배치 미검증 상태에서도 쓰기 강행"))
    p_in.add_argument("--template", action="store_true",
                      help=msg("캔버스 장수를 플랜에 **딱 맞춘다** (장당 0.44초 · "
                               "STOP 파일로 중단). 기본 준비 쪽이 대개 빠르다"))
    p_in.add_argument("--no-prepare", action="store_true",
                      help=msg("준비를 건너뛰고 값만 쓴다 (기본은 준비한다: 템플릿이 "
                               "없으면 만들고 · 맞으면 그대로 · 씨앗이 틀렸으면 다시 심는다)"))
    p_in.add_argument("--canvas", type=int, default=None,
                      help=msg("열어 둔 그룹의 레이어 수 (플랜보다 많아도 된다). "
                               "남는 레이어는 알파 0으로 덮어 안 보이게 한다 — "
                               "3,000장 템플릿 하나로 어떤 플랜이든 올릴 수 있다"))
    p_in.add_argument("--table", default=None,
                      help=msg("레이어 표 주소 (예: 0x2cebd6fa040). 후보가 여럿일 때 "
                               "지운 그룹의 잔재를 피한다"))

    p_it = sub.add_parser("itasha",
                          help=msg("이타샤 — 도안들을 현재 차량의 면마다 올린다 "
                                   "(그룹 준비 → 차체 배치 → 적용)"))
    p_it.add_argument("config", nargs="?", default=None,
                      help=msg("itasha.json 경로 (--plan을 주면 생략)"))
    p_it.add_argument("--plan", nargs="*", default=None,
                      help=msg("구성 파일 대신 플랜을 바로 준다 — 프리셋으로 구성을 "
                               "지어 그대로 실행한다 (하나면 좌우 측면에 미러로)"))
    p_it.add_argument("-o", "--out", default=None,
                      help=msg("--plan으로 지은 구성 파일을 쓸 자리 (기본: itasha.json)"))
    p_it.add_argument("--car", default=None,
                      help=msg("면 실측 지도를 고를 차 이름 (기본: catalog/body_tabs.json의 차)"))
    p_it.add_argument("--media", default=None, metavar="MAKE_Model_YY",
                      help=msg("설치 파일 차량을 **못 박는다** (media/Cars/<이것>.zip). "
                               "이름 매칭이 문턱을 못 넘으면 설치 면 지도를 통째로 "
                               "버리고 프리셋으로 물러나므로, 애매한 차는 이걸로 "
                               "고른다. 목록은 --list-cars"))
    p_it.add_argument("--list-cars", action="store_true",
                      help=msg("설치된 차량 미디어명을 찍고 끝낸다 "
                               "(--car/--media를 주면 그 이름의 후보만 점수와 함께)"))
    p_it.add_argument("--no-paint", action="store_true",
                      help=msg("베이스 도색(자동차 도색 메뉴)을 안 칠한다"))
    p_it.add_argument("--no-watermark", action="store_true", dest="no_watermark",
                      help=msg("내장 ForzaSqueegee 워터마크를 뺀다 (기본은 리어 범퍼에 선다)"))
    p_it.add_argument("--no-deco", action="store_true",
                      help=msg("꾸밈을 빼고 **도안만** 올린다 (꾸밈 그룹·관통 밴드·"
                               "지붕 블랙아웃·모티프 없음). 베이스 도색은 그대로다"))
    p_it.add_argument("--base", default=None, metavar="#RRGGBB",
                      help=msg("베이스 도색 색을 직접 준다 (기본: 도안에서 고른다 — "
                               "레퍼런스 분포 규칙, engine/compose.base_paint)"))
    # `choices=`를 안 쓰는 이유: 목록의 임자는 `engine.compose.MOTIF_FAMILIES`인데
    # 파서를 세우는 시점에 그걸 읽으면 `--help`에도 compose 임포트(0.16초)가
    # 붙는다 — 이 CLI는 무거운 것을 전부 늦게 들여온다. 값 검사는 `compose.build`가
    # 하고 (모르는 이름이면 있는 목록을 적어 ValueError), 부르는 자리가 그것을
    # "오류: …"로 찍는다.
    p_it.add_argument("--motif", default=None, metavar=msg("계열"),
                      help=msg("꾸밈 모티프 계열을 직접 고른다 — star · flower · "
                               "splat · swirl · crystal (기본: 도안의 테마색이 "
                               "고른다, engine/compose.motif_family). 계열은 원래 "
                               "캐릭터 의미에서 오는 것이라 팔레트로는 거기까지 못 "
                               "간다 — 수이세이에 star처럼 아는 사람이 짚는 자리다"))
    p_it.add_argument("--style", default=None, metavar=msg("프리셋"),
                      help=msg("스타일 프리셋 — racing(레이싱 스폰서) · floral(무늬·꽃) · "
                               "splash(스플래시·찢김) · minimal(미니멀) · dark(다크 그래피티) "
                               "(기본: 자동 — 계열 후보를 다 지어 점수로 고른다). 계열에 "
                               "바탕 도색·글자 스타일과 크기·로고 줄·리어 배정이 한 벌로 온다 "
                               "(engine/compose/presets)"))
    p_it.add_argument("--family", default=None, metavar=msg("계열"),
                      help=msg("옆면 꾸밈의 구성 계열만 못 박는다 (엔진 레버) — minimal · "
                               "graphic_bed · diagonal_flow · dark · motorsport · splash "
                               "(기본: 프리셋의 계열, 없으면 후보를 다 지어 점수로 고른다, "
                               "engine/compose/design)"))
    _text_args(p_it)
    _logo_args(p_it)
    _face_args(p_it)
    p_it.add_argument("--no-mirror", action="store_true",
                      help=msg("우측면을 미러하지 않는다"))
    p_it.add_argument("--flip", action="store_true",
                      help=msg("도안을 좌우반전한다 (= 오른쪽 면을 원본 그대로 쓴다). "
                               "인물을 눕히면 그림의 좌우축이 세로로 서므로 이 한 "
                               "비트가 인물이 어느 옆구리를 바닥에 두고 눕는지를 "
                               "가른다 — 얼굴이 땅을 보면 이걸 준다"))
    p_it.add_argument("--preset-only", action="store_true",
                      help=msg("실측 지도를 안 쓰고 프리셋 상수로만 앉힌다"))
    p_it.add_argument("--make-only", action="store_true",
                      help=msg("구성 파일만 짓고 끝낸다 (게임을 안 건드린다)"))
    p_it.add_argument("--dry-run", action="store_true",
                      help=msg("검증하고 계획만 보여 준다 (게임을 안 건드린다)"))
    p_it.add_argument("--restart", action="store_true",
                      help=msg("진행 파일을 버리고 처음부터"))
    p_it.add_argument("--no-prepare", action="store_true",
                      help=msg("그룹 준비를 건너뛴다 (이미 게임에 저장돼 있을 때)"))
    p_it.add_argument("--keep-existing", action="store_true",
                      help=msg("면에 이미 있는 레이어를 안 지운다 (기본은 비우고 "
                               "올린다 — 안 그러면 두 번째 실행이 예산 초과로 막힌다)"))
    p_it.add_argument("--no-autofit", action="store_true",
                      help=msg("올린 뒤 화면으로 재서 스스로 맞추는 단계를 끈다 "
                               "(실측 지도가 있는 면에서만 도는 단계다)"))
    p_it.add_argument("--yes", action="store_true",
                      help=msg("'현재 자동차에 적용'을 묻지 않고 진행한다 "
                               "(지금 차량의 디자인을 덮는다)"))

    p_lang = sub.add_parser("lang",
                            help=msg("UI 언어를 보여 주거나 못 박아 저장한다 "
                                     "(GUI·CLI·편집기 공통, 기본 ko)"))
    p_lang.add_argument("value", nargs="?", default=None, choices=("ko", "en"),
                        help=msg("저장할 언어 — 안 주면 지금 값만 보여 준다"))

    p_gd = sub.add_parser("gamedir",
                          help=msg("FH6 설치 폴더를 보여 주거나 못 박아 저장한다 "
                                   "(자동 탐색이 못 찾을 때)"))
    p_gd.add_argument("path", nargs="?", default=None,
                      help=msg("설치 폴더 — `media/Cars`가 있는 곳. 안 주면 지금 "
                               "쓰는 자리만 보여 준다"))
    p_gd.add_argument("--clear", action="store_true",
                      help=msg("저장해 둔 폴더를 지운다 (자동 탐색으로 돌아간다)"))

    p_cs = sub.add_parser("cars",
                          help=msg("설치 폴더의 차량 정보를 동기화하거나 보여 준다 "
                                   "(면 탭 구성·상한 — 게임은 안 건드린다)"))
    p_cs.add_argument("--sync", action="store_true",
                      help=msg("설치 폴더를 다시 훑어 차량 색인을 뜬다 (`work/state/cars.json`)"))
    p_cs.add_argument("--car", default=None, metavar="MAKE_Model_YY",
                      help=msg("그 차의 면 구성만 보여 준다"))

    p_ov = sub.add_parser("overlay", help=msg("수동 따라 그리기 오버레이 실행"))
    p_ov.add_argument("plan", help=msg("도안 또는 원본 이미지(PNG/JPG) 경로 — "
                                       "이미지를 주면 원본 참조 모드로만 실행"))

    p_run = sub.add_parser("run", help=msg("도안 자동 그리기 (진행 저장·재개, STOP 파일로 중단)"))
    p_run.add_argument("plan", help=msg("도안 경로"))
    p_run.add_argument("--start", type=int, default=None, help=msg("시작 레이어 인덱스 (기본: 진행 파일에서 이어서)"))
    p_run.add_argument("--limit", type=int, default=None, help=msg("이번 실행 최대 레이어 수"))

    p_sort = sub.add_parser("sortplan", help=msg("플랜을 도형·색 그룹 연속으로 안전 정렬 (렌더 동일성 검증)"))
    p_sort.add_argument("plan", help=msg("도안 경로"))
    p_sort.add_argument("-o", "--out", default=None, help=msg("출력 경로 (기본: 같은 폴더 plan_sorted.json)"))

    p_prune = sub.add_parser("pruneplan",
                             help=msg("가시 기여 0 레이어 제거 (렌더 동일 보장 절감, 검증 포함)"))
    p_prune.add_argument("plan", help=msg("도안 경로"))
    p_prune.add_argument("-o", "--out", default=None,
                         help=msg("출력 경로 (기본: 같은 폴더 plan_pruned.json)"))
    p_prune.add_argument("--min-vis", type=float, default=0.0,
                         help=msg("가시 기여 N px 이하 레이어도 제거 (0=기여 0만·렌더 동일, "
                                  ">0이면 렌더 diff 통계 출력)"))

    p_kfi = sub.add_parser("kfpsimport",
                           help=msg("KFPS JSON(편집기·게임 내보내기, 생성기 finals) "
                                    "→ 도안 (overlay·run·inject 전 경로에서 "
                                    "그대로 쓴다)"))
    p_kfi.add_argument("kfps", help=msg("KFPS 타입코드 JSON 또는 생성기 "
                                        "*.v2.json 경로 — 자동 판별"))
    p_kfi.add_argument("-o", "--out", required=True,
                       help=msg("출력 폴더 (<폴더이름>.plan.json·<폴더이름>.preview.png)"))
    p_kfi.add_argument("--image", default=None,
                       help=msg("원본 이미지 — 오버레이 원본 모드에 쓰인다"))

    p_kfe = sub.add_parser("kfpsexport",
                           help=msg("도안 → KFPS 타입코드 JSON (KFPS 편집기"
                                    "에서 열어 고치고 kfpsimport로 되가져온다 — "
                                    "도형 word·마스크까지 그대로)"))
    p_kfe.add_argument("plan", help=msg("도안(*.plan.json) 경로 (노선 무관)"))
    p_kfe.add_argument("-o", "--out", default=None,
                       help=msg("출력 경로 (기본: 같은 폴더 <폴더이름>.kfps.json)"))

    p_fe = sub.add_parser("flsexport",
                          help=msg("도안·이타샤 구성 → **게임이 읽는 파일** "
                                   "(LayerGroup_*/C_group 또는 Livery_*/C_livery + "
                                   "header) + FLS 편집기가 여는 .3so. 창 조작 없이 "
                                   "게임에 넣는 길이다"))
    p_fe.add_argument("plan", help=msg("도안(*.plan.json, 비닐 그룹) 또는 "
                                       "구성(*.itasha.json, 리버리)"))
    p_fe.add_argument("-o", "--out", default=None,
                      help=msg("컨테이너 폴더를 놓을 자리 (기본: 입력 파일 옆). "
                               "게임 저장 컨테이너 뿌리를 주면 바로 그리드에 뜬다"))
    p_fe.add_argument("--name", default=None,
                      help=msg("저장 이름 (기본: 폴더 이름)"))
    p_fe.add_argument("--open", action="store_true",
                      help=msg("쓰고 나서 FLS 편집기로 연다"))

    p_fi = sub.add_parser("flsimport",
                          help=msg("FLS·게임 파일(.3so · C_group · C_livery) → "
                                   "도안. 리버리는 면마다 도안 + *.itasha.json으로 "
                                   "펴서 [Itasha] 메뉴가 그대로 문다"))
    p_fi.add_argument("path", help=msg(".3so · C_group · C_livery 파일 또는 그 폴더"))
    p_fi.add_argument("-o", "--out", default="out/flsimport",
                      help=msg("출력 뿌리 (기본: out/flsimport)"))

    p_fx = sub.add_parser(
        "flsedit",
        help=msg("내장 FLS 편집기의 [Itasha] 메뉴가 부르는 엔진 — 리버리 "
                 "프로젝트(.3so)를 열어 고쳐 다시 쓴다. 사람이 직접 칠 일은 "
                 "거의 없다 (편집기가 QSettings `itasha/command`로 부른다)"))
    p_fx.add_argument("action",
                      choices=("load-design", "auto-place", "decoration",
                               "no-decoration", "decorate", "motif", "family", "style",
                               "mirror", "base-paint", "text", "no-text",
                               "export-group", "rebuild", "state"),
                      help=msg("load-design(도안 올리기) · auto-place(자동 자리) · "
                               "decoration/no-decoration(꾸밈) · motif(계열) · "
                               "decorate(자동 꾸밈 창 — 프리셋·모티프·도색·글자를 한 번에) · "
                               "style(스타일 프리셋) · family(구성 계열) · mirror(좌우 대칭) · "
                               "base-paint(베이스 도색) · text/no-text(캐릭터 이름 글자) · "
                               "export-group(비닐 그룹을 KFPS JSON·plan.json으로 — 게임 "
                               "컨테이너는 편집기 제 [File → Export]) · rebuild · state"))
    _text_args(p_fx)
    p_fx.add_argument("--project", required=True,
                      help=msg("편집기가 저장한 `.3so` 경로"))
    p_fx.add_argument("--geometry", default=None,
                      help=msg("그 차의 면 기하 덤프(`<미디어명>.fsgeom`) — 편집기에 "
                               "올라간 차 모델이 곧 이타샤의 차다"))
    p_fx.add_argument("--slot", type=int, default=None,
                      help=msg("지금 보고 있는 구획 번호 (0~10). 안 주면 면을 "
                               "안 가리는 명령으로 돈다"))
    p_fx.add_argument("--group", nargs="*", default=None, metavar=msg("이름"),
                      help=msg("편집기에서 **고른 그룹** 이름 (`FS:decal-1-fit` 따위) "
                               "— 안 주면 보고 있는 면의 도안이 대상이다"))
    p_fx.add_argument("--design", default=None,
                      help=msg("load-design이 올릴 도안 — *.plan.json · KFPS JSON · "
                               "`.3so`(비닐 그룹) · C_group 아무거나"))
    p_fx.add_argument("--format", choices=("kfps", "plan"), default="plan",
                      help=msg("export-group의 갈래"))
    p_fx.add_argument("--color", default=None, metavar="#RRGGBB",
                      help=msg("base-paint 색 (안 주면 도안에서 고른다)"))
    p_fx.add_argument("--auto-paint", action="store_true", dest="auto_paint",
                      help=msg("decorate: 바탕 도색을 도안에서 고른다"))
    p_fx.add_argument("--no-text", action="store_true", dest="no_text",
                      help=msg("decorate: 글자를 뺀다"))
    p_fx.add_argument("--family", default=None,
                      help=msg("motif 계열 — star · flower · splat · swirl · crystal "
                               "(안 주면 도안의 테마색이 고른다)"))
    p_fx.add_argument("--style", default=None,
                      help=msg("style·decorate 명령의 스타일 프리셋 — auto · racing · floral · "
                               "splash · minimal · dark (편집기 드롭다운의 값)"))
    p_fx.add_argument("--composition", default=None,
                      help=msg("family 명령의 구성 계열 — minimal · graphic_bed · "
                               "diagonal_flow · dark · motorsport · splash (안 주면 자동: "
                               "후보를 다 지어 점수로 고른다)"))
    _logo_args(p_fx)
    _face_args(p_fx)
    p_fx.add_argument("--no-logos", action="store_true", dest="no_logos",
                      help=msg("decorate: 사용자 로고를 다 뺀다 (워터마크는 별개)"))
    p_fx.add_argument("--watermark", default=None, choices=("on", "off"),
                      help=msg("decorate: 내장 ForzaSqueegee 워터마크 (기본 on)"))
    p_fx.add_argument("--symmetry", default=None, choices=("on", "off"),
                      help=msg("decorate: 한쪽 옆면에만 있으면 반대편에 세운다 (기본 on) — "
                               "그림은 거울, 로고·글자는 읽는 방향 그대로"))
    p_fx.add_argument("--role", nargs="*", default=None, metavar=msg("번호=역할"),
                      help=msg("decorate: 실린 덩어리의 역할을 사람이 정한다 — "
                               "`<번호>=<hero|support|logo|text|pinned|auto>` "
                               "(번호는 `state`의 designs 차례, auto면 추정으로 되돌린다)"))
    p_fx.add_argument("-o", "--out", default=None,
                      help=msg("export-group이 쓸 자리"))

    p_ed = sub.add_parser("edit",
                          help=msg("내장 KFPS 편집기를 연다 — 도안을 브라우저 "
                                   "편집기에서 고치고 [Export JSON]이 곧장 도안이 "
                                   "된다 (out/kfpsedit/<이름>/<이름>.plan.json). "
                                   "제품 창의 [편집기 열기] → KFPS와 같은 서버다"))
    p_ed.add_argument("plan", nargs="?", default=None,
                      help=msg("열자마자 물릴 도안 또는 KFPS JSON (없으면 빈 "
                               "캔버스 — 편집기 안 [Import JSON]에 out/의 도안이 "
                               "전부 나온다)"))
    p_ed.add_argument("--port", type=int, default=0,
                      help=msg("서버 포트 (기본: 47615 고정, 막혀 있으면 임시 — "
                               "편집기 즐겨찾기·단축키가 포트(origin)에 붙어 "
                               "있어서 고정이 기본이다)"))
    p_ed.add_argument("--no-browser", action="store_true",
                      help=msg("브라우저를 안 연다 (주소만 찍고 서버로 대기)"))
    p_ed.add_argument("--recover", action="store_true",
                      help=msg("지난 편집의 자동 복구본을 연다 (도안 대신) — 창이 "
                               "죽어 편집을 못 내보냈을 때. 복구본이 있으면 도안을 "
                               "열 때도 프로젝트로 굳혀 두니 잃지는 않는다"))

    p_md = sub.add_parser("models",
                          help=msg("신경망 모델을 미리 받아 두거나 확인한다 "
                                   "(안 받아도 쓸 때 저절로 받는다)"))
    p_md.add_argument("--check", action="store_true",
                      help=msg("받아 둔 것만 세고 끝낸다 (네트워크를 안 탄다)"))
    p_md.add_argument("--verify", action="store_true",
                      help=msg("받아 둔 파일의 SHA-256을 전수 대조한다"))

    p_bm = sub.add_parser("bordermask",
                          help=msg("기존 플랜의 경계 밖 돌출을 덮는 4밴드 마스크 미니 플랜 생성 (인게임 캔버스에 추가 적용용)"))
    p_bm.add_argument("plan", help=msg("도안(*.plan.json) 경로"))
    p_bm.add_argument("-o", "--out", default=None,
                      help=msg("출력 경로 (기본: <도안 폴더>/border/<도안 폴더 이름>.plan_border.json "
                               "— auto_progress 충돌 방지용 별도 폴더)"))
    return parser
