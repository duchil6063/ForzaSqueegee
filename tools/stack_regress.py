"""색면 스택 회귀 검사 — 불변식과 결정성 (`engine/compose/stack.py`).

    .venv/Scripts/python.exe tools/stack_regress.py

보는 것:
- 계열마다 적은 조각이 `stack.PIECES` 안의 이름이다 · 가장자리 도형은 게임 도형 id 표에 있다
- 관통 조각(벨트·핀·아치)은 단면 전체가 프레임 밖으로 나간다 (끝은 차가 낸다)
- 홈은 **마스크**이고 블록 바로 위(짝 아래)에 선다 · 마스크는 홈뿐이다
- 아치 날은 드로어블의 구멍에서 서고, 머리가 그 위면 안 선다
- 가파른 블록(burst)에서는 핀·가장자리가 얕은 짝을 숙주로 삼는다
- 같은 입력 → 같은 조각 · 같은 레이어 (결정성)
- `itasha_stack`이 바닥 요소(`score.GROUND`)·예산 사다리·이음새 정책에 들어 있다
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from forzasqueegee.engine.catalog import Catalog, default_catalog_path  # noqa: E402
from forzasqueegee.engine.compose import design, score, seams, stack  # noqa: E402
from forzasqueegee.engine.compose.families import FAMILIES  # noqa: E402
from forzasqueegee.engine.compose.field import CompositionField, FieldGrid  # noqa: E402
from forzasqueegee.engine.compose.macro import plan as macro_plan  # noqa: E402
from forzasqueegee.engine.model import UNITS_PER_SCALE  # noqa: E402

FAILS: list[str] = []


def check(cond: bool, what: str) -> None:
    print(("  ok   " if cond else "  FAIL ") + what)
    if not cond:
        FAILS.append(what)


def _field(frame=(-450.0, -90.0, 450.0, 90.0), person=(-120.0, -90.0, 130.0, 90.0),
           holes=((250.0, 330.0), (-360.0, -280.0)), head=(60.0, 60.0)):
    """옆면 필드 — 인물 상자만 실루엣, 밴드 아래쪽에 휠아치 구멍 둘."""
    cell = 6.0
    cols = int((frame[2] - frame[0]) / cell)
    rows = int((frame[3] - frame[1]) / cell)
    g = FieldGrid(x0=frame[0], y_top=frame[3], cell=cell, cols=cols, rows=rows)
    zero = np.zeros((rows, cols), np.float32)
    char = zero.copy()
    c0, c1 = int((person[0] - frame[0]) / cell), int((person[2] - frame[0]) / cell)
    r0, r1 = int((frame[3] - person[3]) / cell), int((frame[3] - person[1]) / cell)
    char[r0:r1, c0:c1] = 1.0
    draw = np.ones((rows, cols), np.float32)
    band = frame[3] - frame[1]
    rr = int((frame[3] - (frame[1] + 0.35 * band)) / cell)
    for a, b in holes:
        draw[rr:, int((a - frame[0]) / cell):int((b - frame[0]) / cell)] = 0.0
    return CompositionField(grid=g, frame_box=frame, person_box=person, char=char,
                            char_rgb=np.zeros((rows, cols, 3), np.float32), detail=zero,
                            drawable=draw, head=zero, protected=char.copy(), support=zero,
                            decoration=zero, negative=zero, flow=(1.0, 0.0),
                            head_center=head, visual_center=(5.0, 0.0))


def _corners(l, cat):
    """레이어 상자의 네 모서리 (프레임 좌표) — 전단·회전을 먹인 것."""
    import math
    hx, hy = stack._native_half(cat, l.shape)
    w, h = abs(l.sx) * UNITS_PER_SCALE * hx, abs(l.sy) * UNITS_PER_SCALE * hy
    r = math.radians(l.rot)
    c, s = math.cos(r), math.sin(r)
    out = []
    for dx, dy in ((-w, -h), (w, -h), (w, h), (-w, h)):
        dx2 = dx + dy * l.skew
        out.append((l.x + dx2 * c - dy * s, l.y + dx2 * s + dy * c))
    return out


def _outside(pt, frame) -> bool:
    x, y = pt
    return x < frame[0] or x > frame[2] or y < frame[1] or y > frame[3]


def main() -> int:
    cat = Catalog(default_catalog_path())
    ids = json.loads((ROOT / "catalog" / "fls_shape_ids.json").read_text(encoding="utf-8"))["shape_ids"]
    print("[계열]")
    for name, fam in FAMILIES.items():
        check(all(p in stack.PIECES for p in fam.stack), f"{name}: 조각 이름 {fam.stack}")
    for fam, shapes in stack.EDGE_SHAPES.items():
        check(all(s in cat.shapes and s in ids for s in shapes), f"{fam}: 가장자리 도형 {shapes}이 id 표에 있다")
    check("itasha_stack" in score.GROUND, "itasha_stack은 바닥 요소다 (어수선·위계에서 뺀다)")
    check("itasha_stack" in design.TRIM_ORDER
          and design.TRIM_ORDER.index("itasha_stack") > design.TRIM_ORDER.index("itasha_deco"),
          "예산 사다리에서 산포·에코 뒤에 뺀다")
    check(seams.policy_for("itasha_stack") == seams.CONTINUE, "이음새 정책: 이어 간다")

    colors = {"bed": (30, 30, 60), "bed_alt": (60, 60, 110), "primary": (200, 60, 80),
              "secondary": (240, 160, 170), "shadow": (20, 20, 40), "dark": (20, 22, 28),
              "highlight": (250, 240, 240)}
    fld = _field()
    frame = fld.frame_box

    print("[얕은 블록 — graphic_bed / ribbon+blade]")
    fam = FAMILIES["graphic_bed"]
    specs = macro_plan(fld, ("ribbon", "blade"), 0.75, rocker=True)
    pieces = stack.plan(fld, fam.name, fam.stack, specs, 0.75, colors=colors, rocker=True)
    kinds = [p.kind for p in pieces]
    check(kinds == ["belt", "arch", "edge", "edge", "gap"], f"조각 {kinds}")
    groups = stack.build(pieces, frame, colors, cat)
    layers = [l for _z, ls in groups for l in ls]
    check(all(l.label == stack.LABEL for l in layers), "라벨은 itasha_stack")
    gaps = [l for l in layers if l.mask]
    check(len(gaps) == 1 and all(p.kind == "gap" for p in pieces if p.z == sorted(g[0] for g in groups if g[1] and g[1][0].mask)[0]),
          "마스크는 홈 하나뿐")
    zs = {p.kind: p.z for p in pieces}
    block_z = min(s.z for s in specs)
    counter_z = max(s.z for s in specs)
    check(block_z < zs["edge"] < zs["gap"] < counter_z < zs["arch"] < zs["belt"],
          f"z 순서: 블록 < 가장자리 < 홈 < 짝 < 아치 < 벨트 ({zs})")
    for p, ls in ((p, ls) for p in pieces for z, ls in groups if z == p.z and p.kind in ("belt", "pin")):
        for l in ls:
            cs = _corners(l, cat)
            if p.side:
                # 한쪽으로만 나가는 조각 — 흐름 쪽 두 모서리는 밖, 시작은 인물 상자 안
                order = sorted(cs, key=lambda c: c[0] * p.side)
                fwd, back = order[2:], order[:2]
                check(len(fwd) == 2 and all(_outside(c, frame) for c in fwd)
                      and all(fld.person_box[0] <= c[0] <= fld.person_box[2] for c in back),
                      f"{p.kind}: 흐름 쪽 끝은 프레임 밖 · 시작은 인물 뒤 ({[round(c[0]) for c in cs]})")
            else:
                check(all(_outside(c, frame) for c in cs), f"{p.kind}: 네 모서리가 프레임 밖 ({[round(c[0]) for c in cs]})")
    arch = next(p for p in pieces if p.kind == "arch")
    check(250.0 < arch.at[0] < 330.0, f"아치 날은 흐름 쪽 구멍 위에 선다 (x={arch.at[0]:.0f})")
    check(arch.ang == stack.ARCH_ANG, f"아치 날의 각 {arch.ang}°")
    belt = next(p for p in pieces if p.kind == "belt")
    check(belt.at[1] + belt.width / 2 <= frame[3] and belt.at[1] - belt.width / 2 > frame[3] - 0.2 * (frame[3] - frame[1]),
          "벨트 띠는 벨트라인 바로 아래 5분의 1 안")
    check(belt.role == "primary", "블록이 무채와 안 갈리면 벨트는 주 액센트")
    light = dict(colors, bed=(230, 225, 210))
    pl_ = stack.plan(fld, fam.name, fam.stack, specs, 0.75, colors=light, rocker=True)
    check(next(p for p in pl_ if p.kind == "belt").role == "dark", "옅은 블록 위 벨트는 무채 잉크")
    edges = [p for p in pieces if p.kind == "edge"]
    check(all(p.shape in ("D_03", "D_02") and p.role == specs[0].role for p in edges),
          "가장자리는 찢김 도형 · 블록 색")
    gap = next(p for p in pieces if p.kind == "gap")
    check(gap.ang == specs[0].ang and gap.cut == specs[0].cut, "홈은 블록과 같은 각·전단")

    print("[머리가 아치 위]")
    fld2 = _field(head=(290.0, 60.0))
    p2 = stack.plan(fld2, fam.name, fam.stack, specs, 0.75, colors=colors, rocker=True)
    check("arch" not in [p.kind for p in p2], "머리 뒤에는 날을 안 세운다")

    print("[구멍 없는 지도]")
    fld3 = _field(holes=())
    check(stack.arches(fld3) == [], "구멍이 없으면 아치 없음")
    p3 = stack.plan(fld3, fam.name, fam.stack, specs, 0.75, colors=colors, rocker=True)
    a3 = next(p for p in p3 if p.kind == "arch")
    check(abs(a3.at[0] - (frame[2] - 0.22 * (frame[2] - frame[0]))) < 1e-6, "그때 날은 흐름 쪽 프레임 22% 자리")

    print("[가파른 블록 — splash / burst+ribbon]")
    fam = FAMILIES["splash"]
    specs = macro_plan(fld, ("burst", "ribbon"), 0.8, rocker=True)
    pieces = stack.plan(fld, fam.name, fam.stack, specs, 0.8, colors=colors, rocker=True)
    kinds = [p.kind for p in pieces]
    check(kinds == ["arch", "edge", "edge"], f"조각 {kinds}")
    host = next(s for s in specs if s.kind == "ribbon")
    check(all(abs(((p.ang - host.ang) % 180.0)) < 1e-6 and p.role == host.role and p.z > host.z
              for p in pieces if p.kind == "edge"), "가장자리는 얕은 짝(ribbon)을 숙주로 — 그 색·각·위")
    specs_n = macro_plan(fld, ("burst", "none"), 0.8, rocker=True)
    pn = stack.plan(fld, fam.name, fam.stack, specs_n, 0.8, colors=colors, rocker=True)
    check([p.kind for p in pn] == ["arch"], "숙주가 없으면 가장자리를 안 붙인다")

    print("[split 블록 — graphic_bed / split+ribbon]")
    fam = FAMILIES["graphic_bed"]
    specs = macro_plan(fld, ("split", "ribbon"), 0.75, rocker=True)
    pieces = stack.plan(fld, fam.name, fam.stack, specs, 0.75, colors=colors, rocker=True)
    kinds = [p.kind for p in pieces]
    check(kinds == ["belt", "arch", "edge", "edge", "gap"], f"조각 {kinds} (핀·가장자리는 짝 ribbon, 홈은 split 선을 따라)")
    gap = next(p for p in pieces if p.kind == "gap")
    sp = specs[0]
    check(abs(gap.ang - sp.ang) < 1e-6 and gap.cut == 0.0, "split의 홈은 가른 선과 나란하고 전단이 없다")

    print("[결정성]")
    a = stack.plan(fld, "motorsport", FAMILIES["motorsport"].stack,
                   macro_plan(fld, ("stack", "corner"), 0.55, rocker=True), 0.55, colors=colors, rocker=True)
    b = stack.plan(fld, "motorsport", FAMILIES["motorsport"].stack,
                   macro_plan(fld, ("stack", "corner"), 0.55, rocker=True), 0.55, colors=colors, rocker=True)
    check(a == b, "같은 입력 → 같은 조각")
    la = [(l.shape, l.x, l.y, l.sx, l.sy, l.rot, l.skew, l.color, l.mask) for _z, ls in stack.build(a, frame, colors, cat) for l in ls]
    lb = [(l.shape, l.x, l.y, l.sx, l.sy, l.rot, l.skew, l.color, l.mask) for _z, ls in stack.build(b, frame, colors, cat) for l in ls]
    check(la == lb, "같은 조각 → 같은 레이어")
    check([p.kind for p in a] == ["belt", "pin", "pin", "gap"], f"motorsport: {[p.kind for p in a]}")
    check(all(abs(l.skew) > 0 for l in [l for _z, ls in stack.build(a, frame, colors, cat) for l in ls if not l.mask][:1]),
          "벨트 띠는 전단이 있다 (사람 판의 눕힌 끝)")

    print("[요약]")
    print(f"  {len(FAILS)} FAIL")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
