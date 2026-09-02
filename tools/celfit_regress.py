"""셀 노선 회귀 검사 — 긴 스팬 · 획 역할색 · 받침 문법의 불변식.

    python tools/celfit_regress.py          # 판을 안 굽는다 (수 초)

보는 것:

- **긴 스팬** (`celfit.candidates`) — 잔차 문턱을 열면 마디가 줄고, 문턱이
  1.0이면 종전과 같은 마디가 나오며, 짧고 급한 굽음에서는 열어도 안 준다.
  그리고 **기하가 나쁜 긴 한 장은 사전식 비교에서 진다** (덮임·끊김이 먼저다).
- **팔레트 접기** (`celart.ramps`) — 반경이 색 수를 정하고(단조), 어느 색도
  제 무리 중심에서 반경 넘게 안 움직이며, 맞닿은 채 크게 벌어진 두 색은
  못 묶고, 레이어 수·기하는 안 바뀐다.
- **받침 문법** (`compose.whole`) — 옅게 하기가 무게를 선형으로 깎고, **예상
  무게와 실측 무게가 같으며**(발전기 = 평가기), `fade=0`이면 종전과 같은 판이
  나오고, 배분이 받침 등급에서 크기 대신 대비를 먼저 깎는다.

인게임 확인이 필요한 것은 여기서 못 잰다 — 이 자는 **기하와 색까지**다.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np                                                    # noqa: E402

from forzasqueegee.engine.catalog import (                            # noqa: E402
    Catalog, default_catalog_path)
from forzasqueegee.engine.celart import ramps as RA                    # noqa: E402
from forzasqueegee.engine.celfit import candidates as C               # noqa: E402
from forzasqueegee.engine.celfit.scoring import _Scorer               # noqa: E402
from forzasqueegee.engine.celfit.stroke import _stroke_forms          # noqa: E402
from forzasqueegee.engine.compose import whole as W                   # noqa: E402
from forzasqueegee.engine.compose import wholeeval as WE              # noqa: E402
from forzasqueegee.engine.model import Layer, LayerPlan               # noqa: E402

FAILS: list[str] = []
CAT = Catalog(default_catalog_path())
UPP = 900.0 / 1600.0
W_PX = H_PX = 1600


def check(cond: bool, what: str) -> None:
    print(("  ok   " if cond else "  FAIL ") + what)
    if not cond:
        FAILS.append(what)


# ── 긴 스팬 ──────────────────────────────────────────────────────────


def _scorer() -> _Scorer:
    """빈 판의 채점판 — `dp_segments`는 잔차만 쓰므로 내용이 필요 없다."""
    m = np.ones((H_PX, W_PX), bool)
    z = np.zeros((H_PX, W_PX), bool)
    return _Scorer(CAT, UPP, W_PX, H_PX, (0, 0, W_PX, H_PX), m, z, z)


def _arc(n: int, radius: float, sweep: float, cx=800.0, cy=800.0):
    """호 하나 — (y, x) 표본 `n`개."""
    t = np.linspace(-sweep / 2, sweep / 2, n)
    return np.stack([cy + radius * np.sin(t), cx + radius * np.cos(t)], axis=1)


def _sshape(n: int, amp: float, length: float):
    x = np.linspace(0.0, length, n)
    return np.stack([800.0 + amp * np.sin(2 * np.pi * x / length), 400.0 + x],
                    axis=1)


def _line(n: int, length: float):
    x = np.linspace(0.0, length, n)
    return np.stack([np.full(n, 800.0), 400.0 + x], axis=1)


def _segs(path, wmed, sc, forms, mul, max_shapes):
    idx = C.dp_segments(path, wmed, sc, forms, None, max_shapes, res_mul=mul)
    return None if idx is None else len(idx) - 1


def longspan_level() -> None:
    print("긴 스팬 — 잔차 문턱을 열면 무엇이 달라지나")
    sc = _scorer()
    forms = _stroke_forms(CAT)
    check(forms[1] is not None and len(forms[0]) > 20,
          f"획 어휘 {len(forms[0])}벌")
    cases = {
        "긴 완만한 곡선": (_arc(400, 900.0, 0.55), 2.0, 18),
        "긴 일정 곡률": (_arc(400, 300.0, 1.6), 2.0, 18),
        "짧은 급커브": (_arc(40, 18.0, 2.4), 2.0, 4),
        "S 곡선": (_sshape(400, 70.0, 520.0), 2.0, 18),
        "직선": (_line(300, 500.0), 2.0, 18),
        "긴 완만 (굵은 획)": (_arc(400, 900.0, 0.55), 6.0, 18),
    }
    for name, (path, wm, ms) in cases.items():
        a = _segs(path, wm, sc, forms, 1.0, ms)
        b = _segs(path, wm, sc, forms, 2.0, ms)
        c = _segs(path, wm, sc, forms, 4.0, ms)
        # 기본 인자가 곧 `res_mul=1.0`이라야 끈 판이 종전과 같다
        base = C.dp_segments(path, wm, sc, forms, None, ms)
        base_n = None if base is None else len(base) - 1
        check(base_n == a, f"{name}: 기본 인자 = res_mul 1.0 ({base_n} = {a})")
        if a is None:
            check(True, f"{name}: 문턱 1.0에서는 DP가 아예 안 선다 (완화 {c})")
            continue
        check((c or 99) <= (b or 99) <= a,
              f"{name}: 마디 수가 문턱과 단조 ({a} → {b} → {c})")
    # 긴 완만한 곡선에서는 실제로 줄어야 값이 있다
    path, wm, ms = cases["긴 완만한 곡선"]
    a, c = _segs(path, wm, sc, forms, 1.0, ms), _segs(path, wm, sc, forms, 4.0, ms)
    check(a is None or (c is not None and c < a),
          f"긴 완만한 곡선: 완화가 실제로 마디를 줄인다 ({a} → {c})")
    # 짧은 급커브는 열어도 안 준다 (얻을 것이 없다)
    path, wm, ms = cases["짧은 급커브"]
    a, c = _segs(path, wm, sc, forms, 1.0, ms), _segs(path, wm, sc, forms, 4.0, ms)
    check(a is None or c == a,
          f"짧은 급커브: 완화해도 그대로 ({a} → {c})")
    check(C._LONG_MIN >= 2 and all(m >= 1.0 for m in C._LONG_RES),
          f"긴 스팬 스위치 {C._LONG_RES} · 최소 장수 {C._LONG_MIN}")


def pick_level() -> None:
    print("긴 스팬 — 기하가 나쁘면 장수가 적어도 진다")
    from forzasqueegee.engine.celfit import policy as P

    pol = P.CEL

    def cand(kind, n, *, cover=1.0, breaks=0, stray=0.0, err=0.05, seam=0.0):
        c = C.Candidate(kind=kind, layers=[Layer(shape="A_01")] * n,
                        cover=cover, stray=stray, breaks=breaks, err=err,
                        seam_est=seam)
        return c

    good1 = cand("long1", 1)
    good3 = cand("dp3", 3)
    check(C.pick([good1, good3], pol) is good1,
          "기하가 같으면 한 장이 이긴다")
    gap1 = cand("long1", 1, cover=0.5, breaks=1, seam=3.0)
    check(C.pick([gap1, good3], pol) is good3,
          "끊긴 한 장은 흠 없는 세 장에 진다")
    spill1 = cand("long1", 1, stray=0.9)
    check(C.pick([spill1, good3], pol) is good3,
          "밴드 밖으로 넘친 한 장은 진다")
    # 같은 단이면 이음 보수까지 합친 장수가 가른다
    thin = cand("long2", 2, breaks=0, seam=0.0)
    fat = cand("dp5", 5)
    check(C.pick([thin, fat], pol) is thin, "같은 단이면 적은 장수가 이긴다")


# ── 팔레트 접기 ──────────────────────────────────────────────────────


def _ink_plate(colors: list[tuple]) -> LayerPlan:
    """획 색만 다른 판 하나 — 같은 자리에 겹치지 않게 눕힌 막대들."""
    lays = [Layer(shape="A_22", x=float(i * 40), y=0.0, sx=1.0, sy=0.2,
                  rot=0.0, color=c, label="ink", stroke=i)
            for i, c in enumerate(colors)]
    return LayerPlan(image_size=(512, 512), units_per_px=1.0, layers=lays)


def palette_level() -> None:
    print("팔레트 접기 — 반경이 색 수를 정하고 기하는 안 바뀐다")
    # 두 무리: 어두운 획 넷(서로 3 안쪽) + 밝은 획 넷
    cols = [(40, 34, 36), (43, 37, 39), (37, 31, 33), (46, 40, 42),
            (200, 190, 185), (204, 194, 189), (196, 186, 181), (208, 198, 193)]
    plan = _ink_plate(cols)
    got = []
    for r in (0.0, 1.0, 3.0, 4.5, 6.0, 20.0):
        out, st = RA.fold_colors(plan, CAT, move=r)
        n = len({tuple(l.color) for l in out.layers})
        got.append((r, n))
        check(len(out.layers) == len(plan.layers),
              f"반경 {r}: 레이어 수가 그대로 ({len(out.layers)})")
        check(all((a.shape, a.x, a.y, a.sx, a.sy) == (b.shape, b.x, b.y, b.sx, b.sy)
                  for a, b in zip(plan.layers, out.layers)),
              f"반경 {r}: 기하가 그대로다")
        # **어느 원색도 반경 밖으로 안 나간다**
        far = max(float(RA.de00(RA._lab(np.array([a.color], float)),
                                RA._lab(np.array([b.color], float)))[0])
                  for a, b in zip(plan.layers, out.layers))
        check(far <= max(r, 1e-9) + 1.5,
              f"반경 {r}: 가장 멀리 간 색 ΔE00 {far:.2f}")
    ns = [n for _, n in got]
    check(all(a >= b for a, b in zip(ns, ns[1:])),
          f"반경이 커지면 색 수가 단조 감소 {ns}")
    check(ns[0] == len({tuple(c) for c in cols}),
          f"반경 0이면 안 접는다 ({ns[0]}색)")
    check(ns[-1] >= 2, f"맞닿은 밝기 차이는 끝까지 안 묶인다 ({ns[-1]}색)")
    # 채택 지점 — 유도가 아니라 파레토가 정한 값이다
    check(RA.MAX_MOVE_DE > 0.75 * 4.0,
          f"반경 {RA.MAX_MOVE_DE} > 옛 유도값 3.0")
    check(RA.FOLD_LABELS == ("ink",),
          f"접는 대상은 획뿐 {RA.FOLD_LABELS}")


# ── 받침 문법 ────────────────────────────────────────────────────────


class _Map:
    """면 지도 흉내 — `fit_scale`·`surface_area`가 묻는 것만 있다."""

    def __init__(self, side: float, cap: int = 1000):
        self.paint = (0.0, 0.0, side, side)
        self.fill = 1.0
        self.uncertain = False
        self.cap = cap
        self.drawn = None

    def fit(self, aspect, coverage=0.88, anchor="center", bias_x=0.5):
        s = self.paint[2] * coverage
        w = s if aspect >= 1.0 else s * aspect
        h = s if aspect <= 1.0 else s / aspect
        return (0.0, 0.0, w, h)


def _variant(kind: str, n: int, spread: int = 5) -> W.Variant:
    """색이 `spread`가지인 변주 하나 — 넓이는 같고 대비만 갈린다."""
    cols = [(20 + 50 * (i % spread), 40, 200 - 30 * (i % spread))
            for i in range(n)]
    lays = [Layer(shape="A_01", x=float(i % 8) * 4, y=float(i // 8) * 4,
                  sx=0.4, sy=0.4, color=c, label="cel")
            for i, c in enumerate(cols)]
    p = LayerPlan(image_size=(256, 256), units_per_px=1.0, layers=lays)
    return W.Variant(kind=kind, plan=p, why="",
                     value=np.linspace(1.0, 0.2, n),
                     weight=W.ink_weight(p.layers, CAT),
                     area=W._areas_of(p.layers, CAT),
                     ink=(-10.0, -10.0, 10.0, 10.0),
                     box=(-10.0, -10.0, 10.0, 10.0))


def support_level() -> None:
    print("받침 문법 — 옅게 하기가 무게를 깎는다")
    v = _variant("poster", 64)
    n = 64
    m0, m1 = v.mass(n, 0.0), v.mass(n, 1.0)
    check(m1 < m0, f"다 옅게 하면 무게가 준다 ({m0:.0f} → {m1:.0f})")
    half = v.mass(n, 0.5)
    check(abs(half - 0.5 * (m0 + m1)) < 1e-6 * max(m0, 1.0),
          "중간 값은 두 끝점의 선형 혼합이다")
    check(all(v.mass(n, a) >= v.mass(n, b) - 1e-9
              for a, b in zip(W.FADE_STEPS, W.FADE_STEPS[1:])),
          "눈금을 따라 단조 감소")

    # **예상과 실측이 같아야 한다** — 발전기의 `mass`와 평가기의 `ink_weight`.
    # 정확히 같지는 않다: 예상은 Lab에서 선형으로 섞고 실제 색은 바이트로
    # 반올림된다 (그 오차가 0.2% 안이라야 배분의 눈금보다 훨씬 작다)
    for f in W.FADE_STEPS:
        got = float(W.ink_weight(v.budgeted(n, f).layers, CAT).sum())
        want = v.mass(n, f)
        check(abs(got - want) <= 1e-2 * max(want, 1.0),
              f"fade {f:.2f}: 예상 {want:.1f} ≈ 실측 {got:.1f} "
              f"({100 * abs(got - want) / max(want, 1.0):.2f}%)")

    a = v.budgeted(n, 0.0)
    check([tuple(l.color) for l in a.layers]
          == [tuple(l.color) for l in v.plan.layers],
          "fade 0이면 색을 안 건드린다")
    b = v.budgeted(n, 0.55)
    check(len({tuple(l.color) for l in b.layers})
          <= len({tuple(l.color) for l in a.layers}),
          "옅게 하면 색 수가 안 는다")
    # 자리·크기는 그대로 — 넓이를 안 깎는 것이 이 손잡이의 요점이다
    check(all((x.shape, x.x, x.y, x.sx, x.sy) == (y.shape, y.x, y.y, y.sx, y.sy)
              for x, y in zip(a.layers, b.layers)),
          "옅게 해도 자리·크기는 그대로다 (넓이 불변)")


def allocate_level() -> None:
    print("받침 문법 — 배분이 받침에서 대비를 먼저 깎는다")
    maps = {"side_left": _Map(800.0, 3000), "rear": _Map(200.0, 1000),
            "front": _Map(180.0, 1000)}
    roles = {"rear": ("poster", W.surface_area(maps["rear"])),
             "front": ("emblem", W.surface_area(maps["front"]))}
    vs = {"poster": _variant("poster", 400), "emblem": _variant("emblem", 160)}
    base = {"side_left": 3.0e8}
    # 사람 사전을 **이 물음만 남기고** 연다: 합성 판은 면이 셋뿐이라 면 수·
    # 쏠림 특징이 통째로 사람 범위 밖이고, 그 벌점은 어느 손잡이로도 못 고쳐
    # 상수로 깔린다 — 그러면 두 손잡이의 Δ가 다 0이라 판 ②가 첫 바퀴에 멈춘다.
    # 여기서 묻는 것은 "무게가 넘칠 때 어느 축을 먼저 쓰나"뿐이므로 주역 몫
    # 하나만 남긴다.
    prior = {k: (0.0, 0.0, 0.0, 0.0, 0.0, 1e9, 1e9) for k in WE.HUMAN_PRIOR}
    prior["top1"] = (0.95, 0.96, 0.97, 0.98, 0.99, 0.995, 0.999)
    got = W.allocate_hier(roles, vs, maps=maps, base_mass=base, prior=prior,
                          caps={n: 1000 for n in maps})
    check(bool(got), f"배분이 섰다 {({k: (v[0], v[1], v[2]) for k, v in got.items()})}")
    check(any(v[2] > 0.0 for v in got.values()),
          "받침 등급에서 옅게 하기가 실제로 걸린다")
    for name, (n, fl, fd) in got.items():
        check(0.0 <= fd <= max(W.FADE_STEPS), f"{name}: 옅게 {fd}가 눈금 안")
        check(fl in W.FILL_STEPS, f"{name}: 크기 {fl}가 눈금 안")
    # 옅게 하기를 끄면 같은 자리에서 크기가 대신 깎여야 한다 (손잡이 하나 판)
    keep = W.FADE_STEPS
    try:
        W.FADE_STEPS = (0.0,)
        off = W.allocate_hier(roles, vs, maps=maps, base_mass=base,
                              prior=prior, caps={n: 1000 for n in maps})
    finally:
        W.FADE_STEPS = keep
    check(all(v[2] == 0.0 for v in off.values()),
          "눈금이 하나면 아무것도 안 옅어진다")
    lit = sum(v[1] for v in got.values())
    dim = sum(v[1] for v in off.values())
    check(lit >= dim - 1e-9,
          f"옅게 하기를 쓰면 크기를 덜 깎는다 (Σfill {lit:.2f} ≥ {dim:.2f})")


def tier_level() -> None:
    print("받침 문법 — 등급 자는 평가기와 한 벌이다")
    sh = WE.shares({"a": 10.0, "b": 3.0, "c": 0.5, "d": 0.05})
    t = WE.tiers(sh)
    check(t["a"] == "PRIMARY", "제일 무거운 것이 주역")
    check(W.FADE_TIERS <= set(WE.TIERS), f"옅게 하는 등급 {sorted(W.FADE_TIERS)}")
    check("PRIMARY" not in W.FADE_TIERS and "SECONDARY" not in W.FADE_TIERS,
          "주역·조연은 안 옅게 한다 (§27)")
    check(not (W.FADE_ROLES & {"portrait", "bust", "hero"}),
          f"인물 크롭은 안 옅게 한다 ({sorted(W.FADE_ROLES)})")


def determinism_level() -> None:
    print("결정성 — 같은 입력이 같은 답")
    sc = _scorer()
    forms = _stroke_forms(CAT)
    path = _arc(400, 900.0, 0.55)
    a = C.dp_segments(path, 2.0, sc, forms, None, 18, res_mul=4.0)
    b = C.dp_segments(path, 2.0, sc, forms, None, 18, res_mul=4.0)
    check(a == b, "긴 스팬 DP 두 번이 같다")
    cols = [(40, 34, 36), (43, 37, 39), (37, 31, 33), (200, 190, 185)]
    o1 = RA.fold_colors(_ink_plate(cols), CAT)[0]
    o2 = RA.fold_colors(_ink_plate(cols), CAT)[0]
    check([tuple(l.color) for l in o1.layers]
          == [tuple(l.color) for l in o2.layers], "팔레트 접기 두 번이 같다")
    v = _variant("poster", 64)
    check([tuple(l.color) for l in v.budgeted(64, 0.3).layers]
          == [tuple(l.color) for l in _variant("poster", 64)
              .budgeted(64, 0.3).layers], "옅게 하기 두 번이 같다")


def main() -> int:
    longspan_level()
    pick_level()
    palette_level()
    support_level()
    allocate_level()
    tier_level()
    determinism_level()
    print(f"{'FAIL' if FAILS else 'PASS'} {len(FAILS)}")
    for f in FAILS:
        print("  - " + f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
