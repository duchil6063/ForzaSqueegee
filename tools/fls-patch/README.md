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
| 0002 | 면 기하 덤프 · [Itasha] 메뉴 · 선으로 가르기 | 편집기 안에서 이타샤를 짓는 데 필요한 전부다. ① `dumpItashaGeometry` / `--itasha-dump` — 차 메시를 구획마다 투영해 깊이 래스터로 뜬다 (면 넘어 옮기기가 어림이 아니라 조회가 된다). `QApplication` 전에 답해서 화면 없이 돈다. ② [Itasha] 메뉴 — 도안 올리기(plan.json · KFPS 타입코드 JSON · 비닐 그룹) · 자동 배치 · 꾸밈 짜기/빼기 · 모티프 계열 · 좌우 대칭 · 베이스 도색 · 컨테이너 내보내기 · 그룹 내보내기. 항목마다 프로젝트를 저장하고 `QSettings("itasha/command")`의 엔진에 넘긴 뒤 쓴 것을 다시 연다. 대상은 **고른 도안 하나**다 — `selectedItashaGroups`가 고른 것에서 가장 가까운 `FS:` 그룹까지 올라간다. 엔진 출력은 UTF-8로 읽는다 (cp949 기계에서 한국어가 깨진다). ③ [Edit → Split Selection at a Line] — 고른 레이어·그룹을 기준선 하나로 두 묶음으로 가른다. 게임 도형은 반으로 못 자르므로 선에 걸친 도형은 **사본을 만들어 양쪽에 다** 넣고(면이 제 몫만 그린다), 여유 폭이 그 겹침을 안쪽으로 넓힌다. 선은 캔버스에서 끌고(`SplitTool`) 수치는 옆 패널이 쥔다. |
| 0003 | 한국어 UI · 언어 설정 | 표시 문자열 전부가 `gui::i18n::t()` — 영→한 내장 대응표(`src/gui/i18n.cpp`) — 를 거친다. 표에 없는 문자열은 영어 그대로 통과해 글이 사라지지 않는다. 언어는 QSettings `ui/language`(기본 한국어)이고 설정 [일반] 페이지의 콤보로 고르며 적용은 재시작이다. Qt 표준 대화상자(확인/취소·색 고르기)는 실행 파일 옆 `translations/qtbase_ko.qm`이 옮긴다(`fls_build.py`의 deploy가 싣는다). 기능을 겸하는 이름 — 도구 id·설정 키·리버리 구획 이름 — 은 옮기지 않는다. |
