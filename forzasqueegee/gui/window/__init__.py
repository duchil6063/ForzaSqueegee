"""제품 창 — 이미지 넣기 → 노선 선택 → 생성 → 그 도안을 게임에 올리기.

    이미지 선택 → ○ 페인터 [레이어 수]  ○ 셀 → [도안 생성]
                → [내보내기] · [오버레이] · [인게임 적용]
                → [KFPS 편집기 열기] · [FLS 편집기 열기]

**로직은 창에 없다.** 만들기는 `engine.pipeline.make()`, 올리기는
`engine.fls` · `overlay.guide` · `game.inject`를 그대로 부르고 그것들이 주는
`log`·`progress`·`report.json`을 그리기만 한다 — 문턱도 유도값도 여기서 정하지
않는다. 노출 값은 **페인터의 레이어 수** · **셀의 레이어 상한** · **배경 자동
제거 체크** 셋이다. 페인터는 그 수를 **넣어야** 생성이 열리고, 셀은 기본
3,000(인게임 상한 = 채택 기준)에서 내리기만 한다 — 상한 미만이면 엔진이 감축
모드(영역·선 예산 비례 축소)로 돈다. 배경 제거 체크가 있는 이유: 알파 없는
입력에 자동 발동하는 전처리라, 배경까지 통째로 그리려던 사용자(전면 랩 도안
등)는 GUI에서 끌 방법이 이것뿐이다.

긴 작업은 **만들기와 주입뿐이다** (make 2.5분~57분 · 주입 몇 초). FLS 파일
저장은 몇 초라 스레드가 없다. 중단은 경로마다 다르다:

- `make`   `should_stop`(창의 플래그를 읽는 람다)을 엔진에 넘긴다 — 긴 단마다
           `engine.stop.stop_here()`가 그것을 물어 `Cancelled`를 올린다. 다 만든
           판은 안 버린다 (노선이 끝난 뒤로는 검사점이 없다)
- `inject` 쓰기 자체는 몇 초지만 **템플릿 채우기가 붙으면 길다** — 플랜 폴더에
           `STOP` 파일을 놓으면 레이어 경계에서 멈춘다

**이타샤는 여기 없다** — 내장 FLS 편집기의 [Itasha] 메뉴가 짓는다
(`engine.fls.studio`). 도안 만들기는 값을 넣고 기다리는 일이고 이타샤는 3D 차를
보면서 끌어 놓는 일이라, 한 창에 같이 둘 것이 아니다. 편집기는 차 모델을 제 손으로
읽고 그리므로 그쪽이 임자다 — 이 창에서 가는 길은 [FLS 편집기 열기]다.

**편집기 단추는 묻지 않는다** (사용자 지시 2026-08-26) — [KFPS 편집기 열기]와
[FLS 편집기 열기]가 각자 제 편집기를 곧장 띄운다 (도안이 없어도 열린다).

오버레이와 편집기 창은 스레드가 아니라 **이 프로세스의 창**이다 —
Qt 창은 주 스레드에서만 산다.

## 모듈 구성

창 하나가 하는 일이 넷이라 갈래마다 믹스인 하나다 — `MakeWindow`는 그 넷을
물려받은 껍데기이고, 화면을 짓는 것은 `shell`뿐이다.

    parts    창이 쓰는 부품 — 작업 스레드 한 벌과 작은 위젯들.
    make     도안 만들기 — 값을 세우고 `engine.pipeline.make()`를 스레드에 건다.
    plan     물고 있는 도안 — 고르기(plan·KFPS·FLS) · 바꿔 물기 · KFPS 내보내기.
    apply    게임에 올리기 — 파일 저장 · 오버레이 · 메모리 주입 세 갈래.
    flsops   FLS 갈래 — 게임 컨테이너로 쓰고(`C_group`+`header`) 편집기로 연다.
    sidewin  형제 창 — 오버레이 · 내장 KFPS 편집기.
    shell    제품 창 본체 — 화면을 짓고, 갈래들이 함께 쓰는 것을 쥔다.

밖에서 쓰는 것은 `run`(과 `MakeWindow`)뿐이다.
"""

from __future__ import annotations

from .parts import (
    IMAGE_SUFFIXES,
    ROOT,
    _ApplyJob,
    _Drop,
    _Job,
    _Pane,
    _plan_layers,
    _plan_source,
    _root_out,
    _Stream,
)
from .make import _MakeOps
from .plan import _PlanOps
from .apply import _ApplyOps
from .flsops import _FlsOps
from .sidewin import _SideWindows
from .shell import MakeWindow, run

__all__ = ["run"]
