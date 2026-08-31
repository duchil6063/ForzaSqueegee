"""카탈로그 도형 이름 ↔ **게임 도형 id**.

표는 둘이다. `catalog/fh6_layout.json`의 `shape_ids`는 에디터 셀을 한 장씩
커밋해 실측한 520종(메모리 주입이 레코드 +0x7A에 쓰는 그 u16 — 규칙은 id =
100×페이지 + 페이지 안 번호)이고, `catalog/fls_shape_ids.json`은 FLS 편집기의
기하 표에서 옮긴 1,400종이다 (`tools/fls_shape_ids.py` — 실측 520종과 전 항목
일치를 확인했다). 둘째 표가 더한 것은 **글꼴 글리프**(11글꼴 × 80자)다 — 이것이
있어 게임 글꼴 글자를 파일 노선(C_group·C_livery)에 레이어 그대로 쓴다
(`engine.textvinyl`).

우리 표에 없는 id는 **들여올 때 버린다** — 카탈로그에 없는 도형은 렌더도 배치도
못 하므로 조용히 다른 도형으로 바꿔 놓는 것보다 세어서 말하는 쪽이 낫다
(`engine.kfpsjson`이 모르는 word를 다루는 방식과 같다).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path


def layout_path() -> Path:
    return Path(__file__).resolve().parents[3] / "catalog" / "fh6_layout.json"


def fls_table_path() -> Path:
    return Path(__file__).resolve().parents[3] / "catalog" / "fls_shape_ids.json"


@lru_cache(maxsize=1)
def maps() -> tuple[dict[str, int], dict[int, str]]:
    """(이름 → id, id → 이름). 실측 표가 먼저고 FLS 표가 나머지를 채운다.
    같은 id가 둘이면 먼저 나온 이름이 이긴다."""
    raw = json.loads(layout_path().read_text(encoding="utf-8"))
    name_to_id = {str(k): int(v) for k, v in (raw.get("shape_ids") or {}).items()}
    fp = fls_table_path()
    if fp.exists():
        more = json.loads(fp.read_text(encoding="utf-8")).get("shape_ids") or {}
        for k, v in more.items():
            name_to_id.setdefault(str(k), int(v))
    id_to_name: dict[int, str] = {}
    for name, sid in name_to_id.items():
        id_to_name.setdefault(sid, name)
    return name_to_id, id_to_name


def id_of(name: str) -> int | None:
    return maps()[0].get(name)


def name_of(shape_id: int) -> str | None:
    return maps()[1].get(int(shape_id))
