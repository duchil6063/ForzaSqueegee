"""로고 키트·스폰서 문법 회귀 검사 — 불변식과 결정성.

    .venv/Scripts/python.exe tools/logo_regress.py

보는 것:
- 키트 두 벌이 있고 장수가 `LOGO_LAYERS` 아래다 · 두 벌은 기하가 같고 잉크만 다르다
- `LogoSpec`이 모르는 자리를 거부하고 문자열·dict 항목을 같은 꼴로 받는다
- `cap_layers`가 상한을 지킨다
- 반대편 앉히기(`Placed.mirrored` · `reseat_place`)가 자리만 거울이고 그림은 안 뒤집는다
- 옆면 줄(`side_row`)이 같은 입력에 같은 답을 낸다 · 로고끼리 안 겹친다 · 인물 상자를 안 덮는다
- 유리 덩이(`_pane`): 마스크가 B필러로 둘이면 줄은 큰 덩이 안에, 한 덩이면 답이 없다(그대로)
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from forzasqueegee.engine.catalog import Catalog, default_catalog_path  # noqa: E402
from forzasqueegee.engine.compose import logokit, sponsor  # noqa: E402
from forzasqueegee.engine.compose.autoplace import mirror_place, reseat_place  # noqa: E402
from forzasqueegee.engine.compose.field import CompositionField, FieldGrid  # noqa: E402
from forzasqueegee.engine.compose.place import ManualPlace  # noqa: E402
from forzasqueegee.engine.model import LayerPlan  # noqa: E402
from forzasqueegee.game import surface as gsurf  # noqa: E402

FAILS: list[str] = []


def check(cond: bool, what: str) -> None:
    print(("  ok   " if cond else "  FAIL ") + what)
    if not cond:
        FAILS.append(what)


def _field(frame=(-450.0, -90.0, 450.0, 90.0), person=(-120.0, -90.0, 130.0, 90.0)):
    """빈 옆면 필드 — 전부 그려지고, 인물 상자만 실루엣이다."""
    cell = 6.0
    cols = int((frame[2] - frame[0]) / cell)
    rows = int((frame[3] - frame[1]) / cell)
    g = FieldGrid(x0=frame[0], y_top=frame[3], cell=cell, cols=cols, rows=rows)
    zero = np.zeros((rows, cols), np.float32)
    char = zero.copy()
    c0 = int((person[0] - frame[0]) / cell)
    c1 = int((person[2] - frame[0]) / cell)
    r0 = int((frame[3] - person[3]) / cell)
    r1 = int((frame[3] - person[1]) / cell)
    char[r0:r1, c0:c1] = 1.0
    return CompositionField(grid=g, frame_box=frame, person_box=person, char=char,
                            char_rgb=np.zeros((rows, cols, 3), np.float32), detail=zero,
                            drawable=np.ones((rows, cols), np.float32), head=zero,
                            protected=char.copy(), support=zero, decoration=zero,
                            negative=zero, flow=(1.0, 0.0))


def main() -> int:
    cat = Catalog(default_catalog_path())
    print("[키트]")
    light, dark = logokit.WATERMARK["light"], logokit.WATERMARK["dark"]
    check(light.is_file() and dark.is_file(), "catalog/kit에 logo · logo-dark가 있다")
    pl, pd = LayerPlan.load(light), LayerPlan.load(dark)
    check(len(pl.layers) <= logokit.LOGO_LAYERS, f"logo {len(pl.layers)}장 ≤ {logokit.LOGO_LAYERS}")
    same_geo = len(pl.layers) == len(pd.layers) and all(
        (a.shape, a.x, a.y, a.sx, a.sy, a.rot) == (b.shape, b.x, b.y, b.sx, b.sy, b.rot)
        for a, b in zip(pl.layers, pd.layers))
    check(same_geo, "두 벌의 기하가 같다 (잉크만 다르다)")
    check(any(tuple(l.color) == (0, 0, 0) for l in pl.layers)
          and any(tuple(l.color) == (255, 255, 255) for l in pd.layers),
          "logo는 검정 잉크 · logo-dark는 흰 잉크")

    print("[스펙]")
    spec = logokit.LogoSpec.from_dict({"images": ["a.png", {"image": "b.png", "plan": None}]})
    check(spec.watermark and len(spec.images) == 2 and spec.images[0]["image"] == "a.png",
          "문자열·dict 항목을 같은 꼴로 받는다 · 워터마크 기본 켬")
    try:
        logokit.LogoSpec.from_dict({"placement": "roof"})
        check(False, "모르는 자리를 거부한다")
    except ValueError:
        check(True, "모르는 자리를 거부한다")
    check(not logokit.LogoSpec.from_dict({"watermark": False}).active, "둘 다 없으면 비활성")

    print("[상한]")
    big = LayerPlan(source_image="", image_size=pl.image_size, units_per_px=pl.units_per_px,
                    layers=[replace(l, x=l.x + 3.0 * (i % 7), y=l.y + 2.0 * (i % 5))
                            for i in range(3) for l in pl.layers])
    capped = logokit.cap_layers(big, cat, cap=80)
    check(len(capped.layers) <= 80 < len(big.layers), f"cap_layers {len(big.layers)} → {len(capped.layers)} ≤ 80")

    print("[반대편]")
    item = logokit.LogoItem(plan=light, kind="watermark", name="wm")
    pr = sponsor.load_proto(item, cat)
    a = sponsor.Placed(proto=pr, x=120.0, y=-40.0, w=80.0, rot=12.0)
    b = a.mirrored()
    check((b.x, b.y, b.w, b.rot) == (-120.0, -40.0, 80.0, 348.0), "자리는 거울 · 각은 반대 · 크기 같다")
    la, lb = a.layers(), b.layers()
    check(all(x.sx == y.sx and x.sy == y.sy and x.shape == y.shape for x, y in zip(la, lb)),
          "로고 도형은 안 뒤집힌다 (sx 부호 그대로)")
    smap = gsurf.SurfaceMap(name="side_left", index=3, origin_px=(0, 0), px_per_unit=(1, 1),
                            paint=(-450.0, -80.0, 450.0, 80.0), fill=1.0)
    dmap = replace(smap, name="side_right")
    mp = ManualPlace(plan=light, surface="side_left", x=300.0, y=-40.0, scale=0.03, rot=12.0,
                     role="logo", no_mirror=True)
    m, r = mirror_place(mp, smap, dmap, "side_right"), reseat_place(mp, smap, dmap, "side_right")
    check((m.x, m.y, m.rot) == (r.x, r.y, r.rot) and m.mirror != r.mirror and r.role == "logo",
          "reseat_place = mirror_place의 자리, 뒤집기만 없다, 역할을 물려받는다")

    print("[옆면 줄]")
    fld = _field()
    protos = [pr, replace(pr, item=logokit.LogoItem(plan=light, kind="user", name="u1")),
              replace(pr, item=logokit.LogoItem(plan=light, kind="user", name="u2"))]
    n1, n2 = [], []
    row1 = sponsor.side_row(protos, fld, None, n1)
    row2 = sponsor.side_row(protos, fld, None, n2)
    check([(p.x, p.y, p.w) for p in row1] == [(p.x, p.y, p.w) for p in row2], "같은 입력 → 같은 줄")
    check(len(row1) == 3, f"셋 다 앉는다 ({len(row1)}/3)")
    boxes = [p.box for p in row1]
    ok = True
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a_, b_ = boxes[i], boxes[j]
            if min(a_[2], b_[2]) - max(a_[0], b_[0]) > 0 and min(a_[3], b_[3]) - max(a_[1], b_[1]) > 0:
                ok = False
    check(ok, "로고끼리 안 겹친다")
    px = fld.person_box
    check(all(p.box[0] >= px[2] or p.box[2] <= px[0] for p in row1), "인물 상자를 안 덮는다")
    check(all(p.x > px[2] for p in row1), "흐름 쪽(+x)에 선다")
    wm = next(p for p in row1 if p.proto.item.kind == "watermark")
    us = next(p for p in row1 if p.proto.item.kind == "user")
    # 줄 높이 상한(`ROW_H`)이 큰 로고를 먼저 깎으므로 비는 정확히 절반이 아니다 —
    # 워터마크가 더 작고, 깎이지 않은 워터마크는 제 폭 그대로다
    check(wm.w < us.w and abs(wm.w - sponsor.WATERMARK_K * sponsor.SIDE_LOGO_W * 900.0) < 1e-6,
          "워터마크는 사용자 로고보다 작다 (옆면 폭의 4.75%)")

    print("[유리 덩이]")
    # 200×60 마스크 — 앞 창 80px · 필러 4px · 뒤 창 116px (px = u + 100)
    mask = np.zeros((60, 200), bool)
    mask[:, :80] = True
    mask[:, 84:] = True
    win = gsurf.SurfaceMap(name="window_left", index=8, origin_px=(100.0, 30.0),
                           px_per_unit=(1.0, 1.0), paint=(-100.0, -30.0, 100.0, 30.0),
                           fill=float(mask.mean()), mask=mask, cap=1000)
    pane = sponsor._pane(win)
    check(pane is not None and abs(pane[0] - (-16.0)) < 1e-6 and abs(pane[1] - 100.0) < 1e-6,
          f"두 덩이 → 큰 덩이(뒤 창)의 u 범위 {pane}")
    one = replace(win, mask=np.ones((60, 200), bool))
    check(sponsor._pane(one) is None, "한 덩이 → None (리어·프론트·윈드실드는 그대로)")
    small = mask.copy()
    small[:, :80] = False
    small[:, :6] = True                               # 5% 미만 부스러기
    check(sponsor._pane(replace(win, mask=small)) is None, "5% 미만 부스러기는 덩이가 아니다")
    nw: list[str] = []
    row = sponsor.face_row(protos[1:], win, [], side_w=900.0, floor_v=None, center=None, notes=nw)
    check(bool(row) and all(pl.box[0] >= -16.0 - 1e-6 for pl in row),
          f"유리 로고 줄이 큰 덩이 안에 선다 ({len(row)}개, x={[round(pl.x, 1) for pl in row]})")

    print("[요약]")
    if FAILS:
        print(f"  FAIL {len(FAILS)}: " + " · ".join(FAILS))
        return 1
    print("  전부 통과")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
