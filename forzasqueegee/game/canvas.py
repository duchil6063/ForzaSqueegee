r"""캔버스를 **화면으로** 묻는 자들 — 캡처·캔버스 사각형·씨앗 검사.

주입은 메모리에 값을 쓸 뿐이라 "그래서 화면에 뭐가 그려졌나"를 모른다. 그 물음은
캡처로만 답할 수 있고, 답이 필요한 자리가 둘이다:

- **씨앗 검사** — 다시 연 비닐 그룹은 제 저장본이 참조한 도형 에셋만 그린다.
  그 밖의 도형 id를 주입하면 **조용히 다른 도형으로** 그려져 도안은 멀쩡한데
  인게임만 갈린다. 화면을 안 보면 못 잡는다.
- **인게임 대조** — 같은 캡처 위에서 잰다.

둘이 같은 자를 써야 하므로 여기에 둔다. 문턱·상수의 근거는 각 함수 문서에 있다.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from ..engine.model import Layer, LayerPlan
from ..i18n import msg
from . import io as gio

# 레이어를 캔버스 **밖으로 미는** 값 — 안 보이게 치우는 유일한 수단이다
# (주입은 레이어를 못 지운다). `game/inject`의 남는 자리 덮개와 같은 값이다.
PARK = {"shape": "A_01", "x": 2000.0, "y": 2000.0, "sx": 0.01, "sy": 0.01}
SEED_IOU = 0.55                # 씨앗 판정 문턱 — `seed_missing` 문서


def capture() -> np.ndarray:
    return gio.capture(gio.find_hwnd())


def capture_settled(settle: float = 3.0, tries: int = 20) -> np.ndarray:
    """화면이 **다시 그려진 뒤** 찍는다.

    게임은 주입한 값을 바로 그리지 않는다 — 쓰기가 끝나고 **1.7초 뒤**에 그룹을
    다시 그린다(63차 실측). 그 전에 찍으면 앞 플랜의 그림이 잡혀 IoU가 0.75로
    떨어지는데, 실패가 아니라 자를 잘못 댄 것이다. 그래서 한 번 쉬고, 연속
    두 장이 같아질 때까지 기다린다."""
    time.sleep(settle)
    prev = capture()
    for _ in range(tries):
        time.sleep(0.3)
        cur = capture()
        if cur.shape == prev.shape and np.array_equal(cur, prev):
            return cur
        prev = cur
    return prev


def canvas_rect(plan: LayerPlan, cw: int, ch: int) -> tuple[int, int, int, int]:
    """플랜이 차지하는 클라이언트 사각 (x0, y0, x1, y1).

    1 캔버스 유닛 = 렌더 1px(2560×1440)이고 클라 배율 = 클라높이/1440이다.
    따라서 이미지 1px = units_per_px × (클라높이/1440) 클라 px, 원점은 클라 중심.
    """
    iw, ih = plan.image_size
    k = plan.units_per_px * ch / 1440.0
    w, h = iw * k, ih * k
    return (round(cw / 2 - w / 2), round(ch / 2 - h / 2),
            round(cw / 2 + w / 2), round(ch / 2 + h / 2))


def park_plan(n: int) -> LayerPlan:
    """전 레이어를 캔버스 밖으로 밀어 둔 판 — 빈 캔버스를 **아무 때나** 찍는 수단."""
    return LayerPlan(source_image="park", image_size=(1200, 1200),
                     units_per_px=0.75, layers=[Layer(**PARK) for _ in range(n)])


def _inject(plan: LayerPlan, tmp: Path) -> None:
    """준비 단계를 **건너뛰고** 값만 쓴다 (준비가 이 함수를 부르므로 재귀 금지)."""
    from .inject import apply_plan

    tmp.parent.mkdir(parents=True, exist_ok=True)
    plan.save(tmp)
    apply_plan(tmp, prepare=False)


def seed_missing(shapes: list[str], n: int, bg: np.ndarray | None = None,
                 out: Path | None = None, log=print) -> list[str]:
    """이 그룹이 **실제로 못 그리는** 도형 이름들.

    다시 연(또는 새로 생긴) 비닐 그룹은 제 저장본이 참조한 도형 에셋만 그리고,
    그 밖의 id를 주입하면 **조용히 다른 도형으로** 그린다 (`auto/template.seed`).
    그러면 도안은 멀쩡한데 인게임만 갈리는데, 그 자리에서는 **도안 탓처럼
    보인다** — 실측으로 겪었다 (씨앗이 통째로 날아간 그룹에서 ch0188이 IoU
    .9581 → .8678 · 내부 색오차 0.22 → 2.99).

    **대체되는 도형은 정해져 있지 않다.** "타원으로 그려진다"고 적혀 있었는데
    실측하면 배치마다 다르다 — 같은 그룹에서 A_03·A_31은 둥근 사각으로,
    A_30·U_10·U_79는 가는 호로 나왔다. 그래서 "대체 도형과 얼마나 닮았나"로는
    못 잰다. **제 도형과 닮았나** 하나만 본다.

    잣대는 도형별 실루엣 IoU이고 **크게 찍어서** 잰다 (스케일 1.2, 한 화면 12종).
    작게 찍으면 못 쓴다 — 얇은 도형은 폭 반 픽셀에 IoU가 30%씩 흔들린다.
    크게 찍은 실측 분포(어휘 35 + 씨앗 없는 4종):

        서는 것   0.658 ~ 1.000   (제일 낮은 둘이 얇은 U_28·U_31)
        대체된 것 0.051 ~ 0.452   (제일 높은 A_03은 둥근 사각으로 대체됐다)

    사이가 0.452~0.658로 비어 있어 **0.55**를 문턱으로 둔다.

    **판은 그룹 장수(`n`)로 채운다** — 주입은 그 장수짜리 그룹을 찾으므로 12장짜리
    판을 그냥 주입하면 3,000장 그룹을 못 찾고 메모리를 계속 훑는다. 남는 레이어는
    캔버스 밖으로 밀어 둔다. 렌더는 **보이는 12장만** 뜬다.

    `bg`를 안 주면 여기서 빈 캔버스를 한 번 찍는다 (전부 밀어 두고 캡처).
    """
    from ..engine.catalog import Catalog, default_catalog_path
    from ..engine.render import render_plan

    from ..paths import work_root

    tmp = (Path(out) if out else work_root() / "verify" / "seedcheck") / "probe.json"
    if bg is None:
        _inject(park_plan(n), tmp)
        bg = capture_settled()
    cat = Catalog(default_catalog_path())
    cols, rows, cw, chh, sc_shape = 4, 3, 220.0, 250.0, 1.2
    missing: list[str] = []
    for b0 in range(0, len(shapes), cols * rows):
        batch = shapes[b0:b0 + cols * rows]
        vis = []
        for i, s in enumerate(batch):
            cx = (i % cols - (cols - 1) / 2) * cw
            cy = ((rows - 1) / 2 - i // cols) * chh
            vis.append(Layer(shape=s, x=round(cx, 1), y=round(cy, 1),
                             sx=sc_shape, sy=sc_shape,
                             color=(255, 255, 255)))
        plan = LayerPlan(source_image="seed", image_size=(1200, 1200),
                         units_per_px=0.75,
                         layers=[Layer(**PARK) for _ in range(max(0, n - len(vis)))] + vis)
        seen = LayerPlan(source_image="seed", image_size=(1200, 1200),
                         units_per_px=0.75, layers=vis)
        _inject(plan, tmp)
        cap = capture_settled().astype(np.float32)
        x0, y0, x1, y1 = canvas_rect(seen, cap.shape[1], cap.shape[0])
        w = render_plan(seen, cat, bg=255).astype(np.float32)
        bk = render_plan(seen, cat, bg=0).astype(np.float32)
        ar = cv2.resize(1.0 - np.clip((w - bk).mean(axis=2), 0, 255) / 255.0,
                        (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
        B = bg[y0:y1, x0:x1].astype(np.float32)
        ag = np.clip(((cap[y0:y1, x0:x1] - B) / np.maximum(1.0, 255.0 - B)).mean(axis=2),
                     0, 1)
        sc = (x1 - x0) / 1200.0
        for i, s in enumerate(batch):
            cx = (i % cols - (cols - 1) / 2) * cw
            cy = ((rows - 1) / 2 - i // cols) * chh
            X, Y = (cx / 0.75 + 600) * sc, (600 - cy / 0.75) * sc
            r = int(105 / 0.75 * sc)
            sl = (slice(int(Y - r), int(Y + r)), slice(int(X - r), int(X + r)))
            a, g = ar[sl] > 0.5, ag[sl] > 0.5
            if float((a & g).sum()) / max(1.0, float((a | g).sum())) < SEED_IOU:
                missing.append(s)
    if missing:
        log(msg("  이 그룹이 못 그리는 도형 {n}/{total}종: ", n=len(missing),
                total=len(shapes))
            + " ".join(missing[:20]) + (" …" if len(missing) > 20 else ""))
    return missing
