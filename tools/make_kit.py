"""내장 로고 키트를 굽는다 — `docs/logo.png` → `catalog/kit/logo.plan.json` (+ `logo-dark`).

    .venv/Scripts/python.exe tools/make_kit.py

한 번만 돈다 (결과가 저장소에 든다). 밝은 바탕용 `logo`는 셀 노선으로 굽고
시각 기여 하위 컷으로 `LOGO_LAYERS`장 아래로 깎는다 (`engine.compose.logokit`).
어두운 바탕용 `logo-dark`는 **같은 기하**에 잉크만 바꾼 것이다 — `docs/logo-dark.png`
가 `logo.png`의 검정을 흰색으로 바꾼 그림이라(픽셀 실측) 따로 구우면 기하만
갈린다. 그래서 검정 잉크(0,0,0)를 흰색(255,255,255)으로 되칠한다.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from forzasqueegee.engine.catalog import Catalog, default_catalog_path  # noqa: E402
from forzasqueegee.engine.compose import logokit  # noqa: E402
from forzasqueegee.engine.model import LayerPlan  # noqa: E402

LIGHT_INK = (0, 0, 0)
DARK_INK = (255, 255, 255)


def main() -> int:
    cat = Catalog(default_catalog_path())
    src = ROOT / "docs" / "logo.png"
    logokit.KIT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="fs-kit-"))
    try:
        plan_path = logokit.vectorize(src, tmp, cat=cat, log=print)
        plan = LayerPlan.load(plan_path)
        plan.source_image = "docs/logo.png"
        plan.save(logokit.WATERMARK["light"])
        for l in plan.layers:
            if tuple(l.color) == LIGHT_INK:
                l.color = DARK_INK
        plan.source_image = "docs/logo-dark.png"
        plan.save(logokit.WATERMARK["dark"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"→ {logokit.WATERMARK['light']} ({len(plan.layers)}장) · "
          f"{logokit.WATERMARK['dark'].name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
