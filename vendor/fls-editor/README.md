# vendor/fls-editor — 내장 FLS 편집기가 서는 자리

제품 창의 [FLS 편집기 열기]가 띄우는 네이티브 편집기다. **이타샤가 여기서
지어진다** — [Itasha] 메뉴의 항목들이 우리 엔진(`python -m forzasqueegee
flsedit`)을 불러 리버리 프로젝트를 고쳐 다시 연다.

바이너리는 저장소에 안 들어간다 (이 README만 남는다). 받거나 지어야 한다.

## 받기 (기본)

```
python tools/get_fls.py              # 우리 빌드 — [Itasha] 메뉴가 있다
python tools/get_fls.py --official   # 업스트림 공식 릴리스 — 메뉴 없음
python tools/get_fls.py --check      # 지금 있는 것만 확인
```

제품 창의 [FLS 편집기]가 없는 것을 보면 물어보고 이 길로 받아 준다. 우리
릴리스의 자리와 SHA-256은 `release.json`의 `fls`에 있고, 그것을 못 받으면
업스트림 공식 릴리스로 물러난다.

## 짓기 — 소스에서

```
python tools/fls_build.py --setup    # 툴체인 (Qt·MinGW·zlib) — 한 번만
python tools/fls_build.py            # 소스 동기화 → 패치 → 빌드 → 이 자리에 배포
python tools/fls_build.py --check    # 지금 상태만 본다
python tools/fls_build.py --package  # 릴리스에 올릴 두 벌을 dist/에
```

업스트림 고정 커밋 위에 `tools/fls-patch/*.patch`를 얹어 짓는다.

어느 쪽도 없어도 **내보내기는 다 된다** — 게임이 읽는 것은 파일
(`.3so`·`C_group`·`C_livery`)이지 편집기가 아니다.

## 라이선스 — AGPL-3.0과 대응 소스

FLS는 **AGPL-3.0**이다. 우리는 그것을 고쳐 지은 판을 **릴리스로 배포**하는데,
그러면 대응 소스를 같이 줄 의무가 따라붙는다. 그래서 `--package`가 두 벌을
짓고 **같은 릴리스에** 올린다 — 바이너리와, 고정 커밋에 패치를 얹은 **완전한
소스 트리**다 (남의 저장소가 살아 있기를 기대하는 것으로는 그 의무가 안 끝난다).
저장소에는 그 소스를 다시 만들 재료가 그대로 있다:

- **업스트림 고정 커밋** — https://github.com/Arstz/ForzaLiveryStudio
  태그 `1.2.1` = `5e890e1766eedd884cfa0d1234e135431bb7cdde`
  (`tools/fls_build.py`의 `PIN`이 임자다)
- **우리가 고친 것 전부** — `tools/fls-patch/*.patch`

고정 커밋을 받아 그 패치를 얹으면 같은 바이너리가 나온다
(`tools/fls_build.py`가 그 일을 한 명령으로 한다).

우리 파이썬 코드는 편집기를 **바깥 프로그램으로 실행만** 하므로
(`forzasqueegee/flseditor.py` · 편집기 쪽은 `QProcess`) 링크 경계가 갈려
파생물이 아니다.
