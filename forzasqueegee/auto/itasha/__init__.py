"""이타샤 — 여러 도안을 **현재 타고 있는 차**의 면마다 올린다.

    python -m forzasqueegee itasha itasha.json
    python -m forzasqueegee itasha --plan out/시로코/plan.json    # 프리셋 한 방

세 단계다:

1. **그룹 준비** — 플랜마다 새 비닐 그룹을 만들어 주입하고 저장한다
   (`auto.template.ensure_ready` + `game.inject` + `Driver.save_group`).
   이미 같은 장수의 저장 그룹이 있으면 건너뛴다.
2. **차체 배치** — `비닐 & 데칼 적용`에서 면 탭마다 그룹을 불러와 앉힌다
   (`auto.bodyedit`).
3. **커밋** — Esc → "현재 자동차에 적용". **차 디자인을 덮는 행위라 기본은
   물어본다** (`--yes`로 건너뛴다).

진행은 설정 파일 옆 `<이름>.progress.json`에 적어 두고 이어서 한다.

## 그룹을 **장수로** 찾는다

게임의 '내 비닐 그룹' 그리드는 이름을 그림으로만 보여 준다 — 우리 OCR은 숫자
템플릿뿐이라 이름을 못 읽는다. 대신 정보 패널의 **레이어 수**를 읽는다. 플랜
장수를 아니까 그게 곧 이름표다. 그래서 **한 설정 안에서 장수가 겹치면 거부**한다
(`pruneplan`으로 한 쪽을 한 장이라도 줄이면 갈린다).

## 모듈 구성 — 아래로 갈수록 위를 쓴다

    config    구성 파일 — 규약·검증·짓기. 게임을 건드리기 전에 여기서 다 걸러 낸다.
    progress  진행과 시간 — 어디까지 했나, 무엇이 얼마나 걸렸나.
    cartabs   지금 타는 차가 구성의 차인가 — 차체 에디터의 **면 탭 수**로 본다.
    shapes    면에 **직접** 놓는 것 — 도형 위저드와 게임 텍스트 도구.
    groups    비닐 그룹 준비 — 빈 그룹을 만들어 플랜을 주입하고 슬롯에 저장한다.
    place     차체 배치 — `비닐 & 데칼 적용`에서 면 탭마다 그룹을 불러와 앉힌다.
    run       이타샤 전체 실행 — 준비 → 배치 → 커밋.

밖에서 쓰는 것은 이 파일이 다시 내보내는 이름뿐이다 — 갈라 놓기 전과 같이
`itasha.<이름>`으로 전부 닿는다. 게임을 건드리는 것은 cartabs·shapes·groups·
place·run이고, config·progress는 게임 없이 도는 순수부다.
"""

from __future__ import annotations

from .config import (
    DEFAULT_PLACE,
    NAME_OK,
    PRESET,
    PRESET_ITASHA,
    Config,
    GroupLoad,
    Placement,
    _as_list,
    _car_tabs,
    _check,
    _groups,
    _placement,
    _rel,
    _resolve_tabs,
    ascii_name,
    compose_config,
    load_config,
    make_config,
)
from .progress import Clock, load_progress, save_progress, timing_summary
from .cartabs import PART_TABS, check_car_tabs, verify_car
from .shapes import (
    HSB_TOL,
    _add_shape_job,
    _add_shape_run,
    _rgb_key,
    _shape_axes,
    _shape_batches,
    _soft_xy,
    add_shape_job,
    add_shape_jobs,
    add_text_job,
)
from .groups import (
    _dodge_count,
    prepare_group,
    prepare_group_reuse,
    prepare_groups,
    scan_saved_groups,
)
from .place import (
    TINY_SCALE,
    _shot,
    autofit,
    footprint_of,
    measure_placement,
    place_all,
    place_one,
)
from .run import describe, finish, run
