"""스타일 프리셋 썸네일 — 편집기 드롭다운 아래에 보이는 그림 한 장씩.

    .venv/Scripts/python.exe tools/preset_thumbs.py --tag TH --car giulia --plan 08

`work/lab/deco/<TAG>-<프리셋>/<차>-<판>/face-side_left.png`(`deco/run.py --style`로 구운
옆면)를 차체 밴드 높이로 잘라 `forzasqueegee/engine/compose/thumbs/<프리셋>.png`
(폭 360)에 쓴다. 자동은 `<TAG>-auto`가 있으면 그것, 없으면 `--auto-tag`의 판이다.
같은 판을 프리셋마다 구워야 그림이 프리셋의 차이만 보인다.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(os.environ.get("FS_ROOT") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(ROOT))

from forzasqueegee.engine.compose.presets import PRESET_NAMES, THUMB_DIR   # noqa: E402

W = 360


def crop(img: np.ndarray) -> np.ndarray:
    """면 그림에서 **차체 밴드**(도색이 있는 세로 구간)를 잡아 폭 전체로 자른다."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 바탕(면 밖)은 한 색이라 분산이 0에 가깝다 — 행마다 분산으로 도색 띠를 찾는다
    var = g.astype(np.float32).var(axis=1)
    rows = np.where(var > var.max() * 0.05)[0]
    y0, y1 = (int(rows[0]), int(rows[-1]) + 1) if len(rows) else (0, img.shape[0])
    # 폭은 **면 전체**다 — 가운데만 잘라 내면 인물만 남고 프리셋의 차이(띠·로고
    # 줄·번호·산포)가 안 보인다. 높이는 밴드가 정한다 (창은 폭 360으로 맞춘다).
    out = img[y0:y1]
    h = max(1, int(round(W * out.shape[0] / out.shape[1])))
    return cv2.resize(out, (W, h), interpolation=cv2.INTER_AREA)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="TH")
    ap.add_argument("--auto-tag", default=None, help="자동 판의 태그 (기본 <TAG>-auto)")
    ap.add_argument("--car", default="giulia")
    ap.add_argument("--plan", default="08")
    a = ap.parse_args()
    lab = ROOT / "work" / "lab" / "deco"
    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    done = 0
    for name in ("auto",) + PRESET_NAMES:
        tag = (a.auto_tag or f"{a.tag}-auto") if name == "auto" else f"{a.tag}-{name}"
        src = lab / tag / f"{a.car}-{a.plan}" / "face-side_left.png"
        if not src.is_file():
            print(f"  {name}: 없음 — {src}")
            continue
        img = cv2.imdecode(np.fromfile(str(src), np.uint8), cv2.IMREAD_COLOR)
        out = THUMB_DIR / f"{name}.png"
        cv2.imencode(".png", crop(img))[1].tofile(str(out))
        print(f"  {name}: {out.relative_to(ROOT)} ({out.stat().st_size:,}바이트)")
        done += 1
    print(f"→ {done}장")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
