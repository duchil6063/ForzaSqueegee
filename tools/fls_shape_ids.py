"""FLS 편집기의 도형 기하 표 → **전체 도형 id 표** (`catalog/fls_shape_ids.json`).

    .venv/Scripts/python.exe tools/fls_shape_ids.py [--fls vendor/fls-editor]

우리 주입 표(`catalog/fh6_layout.json`의 `shape_ids`)는 에디터 셀을 한 장씩
커밋해 실측한 520종이다 — 글꼴 글리프(11글꼴 × 80자)는 셀 스윕에 안 들어
있어 파일 노선(C_group)이 쓰지 못했다. FLS는 게임 에셋의 도형 1,400개를
id마다 기하로 갖고 있고 (`assets/vector/shape_geometry.json.gz`), 각 항목에
**우리 카탈로그 이름**을 `source`로 달아 두었다 (정규화 알파 이미지 대조,
40종 묶음 단위). 그 표를 그대로 옮긴다.

검증: 실측 520종과 한 항목도 어긋나지 않고, 같은 이름이 두 id를 갖지 않는다
(둘 중 하나라도 깨지면 쓰지 않는다). 읽는 쪽은 `engine.fls.ids` — 실측 표가
먼저, 이 표는 그 밖의 이름을 채운다.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fls", default=str(ROOT / "vendor" / "fls-editor"))
    ap.add_argument("--out", default=str(ROOT / "catalog" / "fls_shape_ids.json"))
    a = ap.parse_args()
    geom_p = Path(a.fls) / "assets" / "vector" / "shape_geometry.json.gz"
    names_p = Path(a.fls) / "assets" / "vector" / "shape_names.json"
    geom = json.load(gzip.open(geom_p, "rt", encoding="utf-8"))
    names = json.load(open(names_p, encoding="utf-8"))
    measured = json.load(open(ROOT / "catalog" / "fh6_layout.json", encoding="utf-8"))["shape_ids"]
    table: dict[str, int] = {}
    for sid, rec in geom["shapes"].items():
        src = rec.get("source")
        if not src:
            continue
        if src in table:
            print(f"같은 이름이 두 id를 갖는다: {src} → {table[src]}·{sid}", file=sys.stderr)
            return 1
        table[src] = int(sid)
    bad = [(n, i, table.get(n)) for n, i in measured.items() if table.get(n) != i]
    if bad:
        print(f"실측 표와 어긋난다 ({len(bad)}): {bad[:8]}", file=sys.stderr)
        return 1
    pages: dict[int, str] = {}
    for sid in table.values():
        pages.setdefault(sid // 100, names.get(str(sid // 100 * 100 + 1), {}).get("name", ""))
    out = {
        "_": "게임 도형 id ↔ 카탈로그 이름 — FLS 편집기의 shape_geometry.json.gz(source 항목)에서 "
             "옮겼다 (tools/fls_shape_ids.py). 실측 표(fh6_layout.json shape_ids)와 전 항목 일치를 "
             "확인한 것만 쓴다. 페이지(id//100)의 첫 도형 이름은 pages에.",
        "mapping_method": geom.get("mapping_method"),
        "pages": {str(k): v for k, v in sorted(pages.items())},
        "shape_ids": dict(sorted(table.items(), key=lambda kv: kv[1])),
    }
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(table)}종 → {a.out} (실측 {len(measured)}종 전부 일치)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
