# tools/fls-patch — FLS에 얹는 우리 패치

[ForzaLiveryStudio](https://github.com/Arstz/ForzaLiveryStudio)(AGPL-3.0)의
**고정 커밋 위에 얹는 패치 묶음**이다. 업스트림 사본을 통째로 들이는 대신
이렇게 두는 이유는 셋이다 — 무엇을 바꿨는지가 그대로 보이고, 업스트림 갱신이
`git am` 한 번이며, AGPL이 요구하는 "대응 소스"가 **고정 커밋 + 이 패치들**로
완전하기 때문이다.

- 업스트림 고정: `tools/fls_build.py`의 `PIN` (현재 태그 1.2.1)
- 얹고 짓기: `python tools/fls_build.py`
- 손으로 고친 뒤 다시 뽑기: `python tools/fls_build.py --export-patches`
  (작업 트리는 `D:/dev/fls-src`, 브랜치 `itasha`)

## 라이선스 — **AGPL-3.0-or-later** (저장소 MIT가 아니다)

이 폴더의 `*.patch`는 AGPL-3.0-or-later 저작물인 ForzaLiveryStudio를 **고친
것**이다. 새로 쓴 줄은 우리 것이지만 문맥 줄은 업스트림(© Arstz)의 것이고,
무엇보다 이 패치의 쓸모는 AGPL 저작물에 얹혀 그 일부가 되는 데 있다. 그래서
**저장소 뿌리의 MIT는 이 폴더에 적용되지 않는다** — 여기는 AGPL-3.0-or-later다
(전문: 이 폴더의 `LICENSE`).

우리가 새로 쓴 파일(`main_window_itasha.cpp` · `main_window_split.cpp` ·
`project_canvas_split.cpp` · `split_line.h` · `editor_state_split.cpp` ·
`i18n.h` · `i18n.cpp`)의
저작권은 ForzaSqueegee 기여자들에게 있고, AGPL-3.0-or-later로 내놓는다.

반대 방향은 문제가 없다: ForzaSqueegee 본체(MIT)는 AGPL과 호환이라 이 패치가
본체를 물들이지 않는다. 본체와 편집기는 프로세스 경계로 갈려 있다
(`forzasqueegee/flseditor.py` · 편집기 쪽은 `QProcess`).

## 패치

| 번호 | 무엇 | 왜 |
|---|---|---|
| 0001 | Qt 플러그인 경로를 배치 무관으로 | 업스트림이 vcpkg 트리 모양을 못 박아 공식 Qt 설치본으로는 configure가 죽는다. 업스트림에 보낼 만한 순수 이식성 고침이다. |
| 0002 | 면 기하 덤프 · [Itasha] 메뉴 · 선으로 가르기 | 편집기 안에서 이타샤를 짓는 데 필요한 전부다. ① `dumpItashaGeometry` / `--itasha-dump` — 차 메시를 구획마다 투영해 깊이 래스터로 뜬다 (면 넘어 옮기기가 어림이 아니라 조회가 된다). `QApplication` 전에 답해서 화면 없이 돈다. ② [Itasha] 메뉴 — 도안 올리기(plan.json · KFPS 타입코드 JSON · 비닐 그룹) · 자동 배치 · 꾸밈 짜기/빼기 · 모티프 계열 · 좌우 대칭 · 베이스 도색 · 그룹 내보내기(KFPS JSON·plan.json — 게임 컨테이너는 편집기 제 [File → Export]가 쓴다, 사용자 결정 2026-09-03). 항목마다 프로젝트를 저장하고 `QSettings("itasha/command")`의 엔진에 넘긴 뒤 쓴 것을 다시 연다. 대상은 **고른 도안 하나**다 — `selectedItashaGroups`가 고른 것에서 가장 가까운 `FS:` 그룹까지 올라간다. 엔진 출력은 UTF-8로 읽는다 (cp949 기계에서 한국어가 깨진다). ③ [Edit → Split Selection at a Line] — 고른 레이어·그룹을 기준선 하나로 두 묶음으로 가른다. 게임 도형은 반으로 못 자르므로 선에 걸친 도형은 **사본을 만들어 양쪽에 다** 넣고(면이 제 몫만 그린다), 여유 폭이 그 겹침을 안쪽으로 넓힌다. 선은 캔버스에서 끌고(`SplitTool`) 수치는 옆 패널이 쥔다. |
| 0003 | 한국어 UI · 언어 설정 | 표시 문자열 전부가 `gui::i18n::t()` — 영→한 내장 대응표(`src/gui/i18n.cpp`) — 를 거친다. 표에 없는 문자열은 영어 그대로 통과해 글이 사라지지 않는다. 언어는 QSettings `ui/language`(기본 한국어)이고 설정 [일반] 페이지의 콤보로 고르며 적용은 재시작이다. Qt 표준 대화상자(확인/취소·색 고르기)는 실행 파일 옆 `translations/qtbase_ko.qm`이 옮긴다(`fls_build.py`의 deploy가 싣는다). 기능을 겸하는 이름 — 도구 id·설정 키·리버리 구획 이름 — 은 옮기지 않는다. |
| 0004 | [Auto Decoration...] 창 — 구성 계열 · 무늬 계열 · 바탕 도색 · 캐릭터 이름 글자 | 흩어져 있던 항목(꾸밈 넣기 · 무늬 계열 · 바탕 도색)을 **창 하나**로 모은다. 창을 열면 `flsedit state`로 조리법을 읽어 현재 값을 채우고, [짓기]를 누르면 `flsedit decorate --composition … --family … --color #…\|--auto-paint --text … \| --no-text`로 한 번에 넘겨 꾸밈을 켠다. 구성 계열(자동 + minimal·graphic_bed·diagonal_flow·motorsport·splash) · 무늬 계열 · 바탕 도색(도안에서/직접 색) · 글자 묶음(켜기 체크, 이름·작품명·스타일·자리·우선순위·테두리·그림자·레이어 상한·게임 글꼴 폴백). 마지막 글자 입력은 QSettings `itasha/text/*`에도 남는다. 문자열은 `QProcess` 인자 하나로 그대로 넘어가 띄어쓰기가 산다. 메뉴에는 [Auto Decoration...] · [Drop Decoration] · [Mirror] · [Export Group As](KFPS·plan)만 남는다 (예전 낱개 액션 `decoration`·`motif`·`family`·`base-paint`·`text`는 CLI에 그대로 있다). |
| 0005 | [Auto Decoration...] 창의 **실린 그림 표** — 덩어리마다 역할 | 차에 실린 덩어리 전부가 꾸밈의 재료다 (엔진 `engine/compose/cast.py`). 창이 `flsedit state`의 `designs`를 한 줄씩 보인다 — 면 · 그림 · 장수 · 역할 콤보(자동(추정) · 주역 · 보조 · 로고 · 글자 · 그대로). 추정 까닭은 툴팁이다. 사람이 고른 값은 `--role <번호>=<역할>`로 다른 옵션보다 앞에 넘어가고, 엔진이 조리법에 `role_user`로 못 박는다. |
| 0006 | [Auto Decoration...] 창의 **로고 · 좌우** 묶음 | 사용자 결정 ②·③ (2026-09-02). 로고 묶음: 내장 ForzaSqueegee 워터마크 체크(기본 켬) · 로고 이미지 목록([로고 이미지 추가...]·[빼기], 0~N장, 벡터화 상한 110장 안내) · 자리 콤보(자동 · 리어 범퍼 · 프론트 범퍼 · 윈드실드 · 옆면 로커 줄만). 좌우 묶음: "한쪽 옆면에만 그림이 있으면 반대편에도 세웁니다" 체크(기본 켬)와 로고·글자는 미러하지 않고 읽는 방향 그대로 다시 앉힌다는 안내. 창을 열면 `flsedit state`의 `logos`·`symmetry`로 채우고, [짓기]가 `--logo <경로>…\|--no-logos --watermark on\|off --logo-placement … --symmetry on\|off`로 넘긴다 (엔진 `engine/compose/logokit.py`·`sponsor.py`, `engine/fls/studio.act_logos`·`act_symmetry`). |
| 0007 | [Auto Decoration...] 창의 **면 배정** 묶음 | 사람 판은 유리·리어·프론트에 주역 크롭을 거의 안 돌린다 (계획 5단계, 2026-09-03). 면마다 콤보 하나 — 도어 유리(자동 · 보조 그림만 · 옆면 그림을 벨트라인 위로 이어 그리기 · 초상 크롭 · 비움) · 뒷유리 · 리어 · 프론트(자동 · 로고·글자 · 크롭 · 비움) · 윈드실드(자동 · 로고·글자 · 비움). 자동은 로고·글자가 있으면 그것(리어 워드마크 + 로고 줄 · 윈드실드 글자 띠 · 유리 로고 열 + 문구), 없으면 크롭으로 물러난다 (리어·뒷유리는 비운다). 창을 열면 `flsedit state`의 `faces`로 채우고(없으면 전부 자동), [짓기]가 `--face 면=모드 …`를 **맨 뒤에** 넘긴다 (목록 인자라 뒤에 다른 옵션이 오면 안 된다). 엔진 `engine/compose/facespec.py`·`facetext.assigned_text`·`build._continuations`, `engine/fls/studio.act_faces`. |
| 0008 | 로고 안내문의 벡터화 상한 300 | 창의 안내문이 110장이라 적혀 있었는데 엔진 `LOGO_LAYERS`는 300이다 — 숫자를 엔진과 맞춘다. |
| 0009 | [Auto Decoration...] 창의 **스타일 프리셋** 드롭다운 + 썸네일 · 레이싱 번호 칸 | 구성 계열 콤보(엔진 계열 이름을 C++ 표로 되풀이하던 것)를 사람 판에서 읽은 프리셋 — 자동 · 레이싱 스폰서 · 무늬·꽃 · 스플래시·찢김 · 미니멀 · 다크 그래피티 — 로 재편한다 (계획 A단계, 사용자 결정 ④ 2026-09-02). 목록·이름·설명·썸네일 경로는 `flsedit state`의 `style_presets`가 준다 (엔진 `engine/compose/presets.py` — 계열 + 바탕 도색·팔레트·글자 스타일과 크기·로고 줄·리어 배정·예산 사다리가 한 벌). 창은 그 목록으로 콤보를 짓고 고른 프리셋의 그림을 아래에 보이며, [짓기]가 `--style <키>`로 넘긴다. 글자 묶음은 기본 켬(이름이 비면 안 선다)이고 **레이싱 번호** 칸이 생겼다 (`--text-number`, 레이싱 프리셋만 앉힌다). 옛 조리법의 `family`는 엔진이 프리셋으로 옮긴다. |
| 0010 | [Itasha] 메뉴에서 **게임 컨테이너 내보내기 둘**을 뺀다 | [Export to Game Container...](Livery_/C_livery)와 [Export Group As → Game Container (C_group)]은 편집기 제 [File → Export]와 같은 파일을 쓰는데, 그쪽은 3D 프리뷰로 썸네일까지 찍는다 (사용자 결정 2026-09-03). [Export Group As]에는 KFPS JSON·plan.json만 남고, CLI `flsedit export`·`--format fls`도 함께 뺐다. |
