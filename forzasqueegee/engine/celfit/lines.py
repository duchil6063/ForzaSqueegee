"""획 배치 구동 — 선 지도 하나를 **공통 선 재구성 엔진**에 태운다.

여기에는 정책이 없다. 증거 지도를 짓고(`evidence`), 엔진에 논리 획을 짓게 하고
(`engine.build_strokes`), 정책이 고른 후보로 놓게 한 뒤(`engine.place_strokes`)
덮어 그리기만 노선 정책대로 붙인다. 두 노선(`line`·`cel`)이 이 한 함수를 쓴다 —
갈리는 것은 넘겨받는 `pol` 하나뿐이다.
"""

from __future__ import annotations

import numpy as np

from ..catalog import Catalog
from ..celart import CelArt
from ..model import LayerPlan
from ..price import fix_min_gain, repair_min_gain
from . import engine as E
from . import policy as P
from .carve import _carve_lines
from .evidence import build_maps


def _fit_lines(plan: LayerPlan, cel: CelArt, cat: Catalog, upp: float,
               budget: int, forms: tuple, log, sids=None,
               value: np.ndarray | None = None, price: float = 0.0,
               carve_defer: list | None = None, carve: bool = True,
               progress=None, stats: dict | None = None,
               pol=None, maps=None) -> int:
    """선 지도 → 획 레이어 (곡선·막대, 색은 원화 표본, 전부 ink).

    금지는 투명 배경뿐이다 — 선은 모든 면 위에 얹히므로 면 침범 개념이 없다.
    경로 하나가 획 하나라 `sids`에서 새 그룹 id를 받는다.

    `pol`은 노선 정책(`policy.LINE`·`policy.CEL`)이다. 안 주면 line 정책 —
    가장 빡빡한 쪽이 기본이다.

    `carve=False`는 덮어 그리기(획 덮개)를 통째로 끈다 — 덮개는 **면 색으로
    도로 덮는** 문법이라 면 채움이 없는 자리에는 덮을 색이 없다. 정책도 같은
    칸을 들고 있어 둘 다 참일 때만 돈다.
    """
    st = stats if stats is not None else {}
    pol = pol or P.LINE
    w, h = cel.size
    if maps is None:
        # 증거가 따로 안 왔다 — 선 지도 자체가 증거다 (고전 폴백)
        maps = build_maps(None, None, cel.line_mask, cel.line_mask,
                          cel.src_rgb, cel.labels >= 0,
                          value if value is not None
                          else (cel.labels >= 0).astype(np.float32))
    rec = E.build_strokes(plan, cel, maps, cat, upp, sids, log, pol)
    n = rec.fat_fills + E.place_strokes(
        plan, rec, cel, cat, upp, max(0, budget - rec.fat_fills), forms, pol,
        log, price=price, progress=progress)
    st.update(rec.report(pol))
    st["_rec"] = rec                       # 노선이 debug·per-stroke에 쓴다
    if carve and pol.carve:
        # 덮개도 한 장이다 — 가격 설계에서는 λ만큼 벌어야 산다
        n += _carve_lines(plan, cel, cat, upp, budget - n, log,
                          floor=fix_min_gain(price) if price else 0.0,
                          floor_lo=repair_min_gain(price) if price else 0.0,
                          defer=carve_defer)
    return n
