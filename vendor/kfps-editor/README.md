# vendor/kfps-editor — 내장 KFPS Fabric 편집기의 동봉본

제품이 띄우는 비닐 편집기(브라우저 워크스페이스)의 **무수정 사본**이다.
`forzasqueegee/kfpseditor.py`의 로컬 서버가 이 폴더를 KFPS와 같은 URL
(`/tools/fabric-editor/…`)로 서빙한다 — editor.js가 그 절대 경로를
하드코딩하고 있어서 폴더 배치가 아니라 URL을 맞춘다. 라이선스 전문은
`../../THIRD_PARTY_NOTICES.md`와 이 폴더의 LICENSE 파일들에 있다.

## 출처

- 저장소: https://github.com/heyitshestia/kloudys-forza-painter-suite (MIT)
- 가져온 시점: 2026-08-25, 커밋 `0af4f21f984ad42f33dcf570ad36ad8e704092b6`
  (KFPS 버전 3.1.40, 2026-08-24) — `vendor/galatea`·`engine/kfpsjson.py`가
  대조한 그 커밋이다 (타입코드 JSON 스키마가 같은 판이라는 뜻)

## 파일

| 파일 | 무엇 | SHA-256 앞 16자리 |
|---|---|---|
| `index.html` | 편집기 페이지 (`tools/fabric-editor/index.html` 원본 그대로) | `746ec4c95b7e64d1` |
| `editor.js` | 편집기 본체 (486KB) | `3e4fef31e5f00815` |
| `editor-core.js` · `editor-fabric-adapter.js` | 기하·어댑터 | `24daf01691a017f4` · `42514a22afcd1671` |
| `style.css` | 편집기 스타일 | `c730b397946b76fa` |
| `vendor/fabric.min.js` | Fabric.js (KFPS 동봉 빌드, `LICENSE.fabricjs`) | `48f8f0915beb512f` |
| `shape-names.json` · `shape-words.json` | 도형 가족·이름·게임 word 대응 | `3ec1ced864bb2fc7` · `0e4a8b90e78295fc` |
| `Resources/Vinyls/<가족>/<번호>` | 도형 메시 2,800파일 (35가족 × 40도형, JSON 메시 + PNG 미리보기) | **저장소에 없다 — 받는다** (아래) |
| `assets/kfps-logo.ico` | 파비콘 (index.html이 참조) | — |
| `EDITOR_MANUAL.md` | 편집기 사용 설명 (KFPS `tools/fabric-editor/README.md` 원본) | — |
| `LICENSE.kfps` | KFPS 저장소 LICENSE 사본 (MIT) | — |
| `LICENSE.fabricjs` | Fabric.js 라이선스 사본 (MIT) | — |

**안 가져온 것**: `start_fabric_editor.py`(KFPS의 로컬 서버 — 우리는
`forzasqueegee/kfpseditor.py`가 같은 API 표면을 우리 저장소 배선으로
제공한다), `tests/`(node 전용 개발 검사 — 원본 저장소에서 돌린다).

## 도형 리소스는 저장소에 없다 — 받는다

`Resources/`의 2,800파일은 **게임 비닐 도형의 메시 데이터**(정점·인덱스·알파)와
그 미리보기다. 게임 에셋에서 나온 것이라 **우리가 재배포하지 않는다** — KFPS의
MIT는 KFPS가 쓴 것에 대한 것이지 게임 자료를 재허락할 권한이 아니기 때문이다
(`../../LICENSE`의 "게임에서 뽑아낸 자료"). FLS 편집기·신경망 모델과 같은 길로,
쓸 때 위 고정 커밋에서 받는다.

```
python tools/get_kfps.py            # 받는다
python tools/get_kfps.py --check    # 있는지만 본다
python tools/get_kfps.py --verify   # 집계 SHA-256 대조
```

git이 있으면 부분 체크아웃으로 이 폴더의 36MB만 받고(실측 4초), 없으면 커밋
zip(250MB급)을 받아 이 폴더만 푼다. 편집기를 처음 열 때 제품이 물어보고 받아
주므로 손으로 돌릴 일은 대개 없다.

**검증**: 집계 SHA-256 `291821b134a74d55…9227ea07` (2,800파일). 집계는 정렬한
`<상대경로> <파일 sha256>` 줄들의 sha256이다 (`get_kfps.aggregate`) — 받은 것이
그 커밋 그대로가 아니면 **자리에 안 놓는다**. editor.js가 읽는 포맷과 우리
타입코드 대조가 같은 판 위에 서야 하기 때문이다.

## 갱신하는 법

KFPS 저장소의 `tools/fabric-editor/`에서 위 파일들을 그대로 덮고, 이 표의
커밋·해시를 다시 적는다. `Resources/`는 덮지 말고 `tools/get_kfps.py`의
`PIN`·`AGG`를 새 커밋 것으로 고친 뒤 다시 받는다 (`--verify`로 대조). **바이트 그대로**여야 한다 — Windows에서 보통
클론은 core.autocrlf가 개행을 CRLF로 바꿔 놓으므로
`git -c core.autocrlf=false archive HEAD tools/fabric-editor | tar -x`
꼴로 뜬다 (editor.js 485,821B가 맞으면 제대로 뜬 것). 그 뒤 `forzasqueegee/kfpseditor.py`의 서버 API가
`start_fabric_editor.py`의 새 계약과 갈리지 않는지 대조한다 — 특히
`/api/fabric-editor/*` 엔드포인트 목록과 요청/응답 필드, 그리고 세션 토큰
헤더(`X-KFPS-Editor-Session`) 규약. 도형 리소스 포맷
(`{Info, Vertices, Indices, VerticesAlpha}`)이 바뀌면 editor.js가 함께
바뀌므로 통째로 갱신하면 된다.
