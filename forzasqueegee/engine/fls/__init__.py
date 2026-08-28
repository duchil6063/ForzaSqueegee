"""FLS(ForzaLiveryStudio) 판 입출력 — 게임이 쓰는 파일 그대로.

`ForzaLiveryStudio`(AGPL-3.0, https://github.com/Arstz/ForzaLiveryStudio)가
문서로 남긴 규격을 파이썬으로 다시 쓴 것이다. 세 판을 읽고 쓴다:

| 판 | 무엇 | 우리 쪽 |
|---|---|---|
| `.3so` | FLS 편집기 프로젝트 (gzip JSON) | `project` |
| `C_group` + `header` | 게임 비닐 그룹 컨테이너 폴더 | `cgroup` · `folder` |
| `C_livery` + `header` | 게임 리버리 컨테이너 폴더 (면 11칸) | `livery` · `folder` |

**이것이 창 조작을 대신한다.** 창 조작은 도안을 게임 창에 대고 장당 6초씩
그리지만(3,000장이면 5시간), 여기서는 같은 값을 파일로 적어 저장 폴더에 놓는다.
이타샤도 마찬가지다 — 면마다 그룹을 불러오고 옮기고 차를 칠하는 일이
`C_livery` 한 장이다.

리버리가 **어느 차의 것인가**는 설치 파일이 답한다: 차 zip 안 애니메이션 클립
이름(`carclips_<id>.clipd`)이 곧 게임 차 id다 (`game.carfiles.car_id` —
2026-08-26 전수 실측으로 596/596 일치).

바깥에서 쓰는 것은 대개 `bridge`의 넷이다:

    plan_project / plan_folder     도안 → 편집기 프로젝트 · 게임 폴더
    itasha_project / itasha_folder 이타샤 구성 → 리버리 프로젝트 · 게임 폴더
    import_any                     FLS 파일 → 도안(+구성)
"""

from __future__ import annotations

from . import bridge, cgroup, folder, header, ids, livery, materials, project
from .bridge import (
    import_any,
    itasha_folder,
    itasha_project,
    itasha_sections,
    plan_folder,
    plan_project,
)
from .folder import read_group, read_livery, sniff, write_group, write_livery

__all__ = ["bridge", "cgroup", "folder", "header", "ids", "livery", "materials",
           "project", "import_any", "itasha_folder", "itasha_project",
           "itasha_sections", "plan_folder", "plan_project", "read_group",
           "read_livery", "sniff", "write_group", "write_livery"]
