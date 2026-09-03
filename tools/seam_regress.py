"""이음새 회귀 검사 — 면 전환의 불변식.

    python tools/seam_regress.py            # 실측 설치 마스크가 있는 차로 (수 초)

보는 것:

- **조각별 이음새** (`game.fold.SeamSegment`) — 토막이 서고, 토막 위의 점이
  건너편 토막의 이음선으로 간다 (source seam point → target seam point).
- **띠 잇기** (`compose.seams.carry`) — 건너간 띠가 이음선 위에서 정확히 만나고
  (`pos` 0), 두께가 부당하게 안 깎이며, 못 이을 조건에서는 **끊는다**.
- **정책** — 얼굴·글자는 피하고, 모르는 역할은 끊는다.
- **놓는 자리** (`surfshapes.flow_shapes`) — `anchor_u`·`top_v`로 준 점을 띠가
  실제로 지난다.
- **결정성** — 같은 입력 두 번이 같은 토막·같은 결정.

인게임 확인이 필요한 것은 여기서 못 잰다 (곡면에서 실제로 이어져 보이나) —
이 자는 **면 유닛 기하까지**다.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np                                                    # noqa: E402

from forzasqueegee.engine.compose import seams as gseams              # noqa: E402
from forzasqueegee.engine.compose.surfshapes import flow_shapes       # noqa: E402
from forzasqueegee.engine.model import UNITS_PER_SCALE                # noqa: E402
from forzasqueegee.game import carfiles, fold as gfold                # noqa: E402


FAILS: list[str] = []
CARS = ("NIS_SilviaSpecR_02", "ALF_GiuliaGTAm_21", "MAZ_Miata_94")
PAIRS = (("side_left", "rear"), ("side_left", "front"), ("side_left", "top"),
         ("front", "side_left"), ("rear", "side_left"), ("top", "side_left"))


def check(cond: bool, what: str) -> None:
    print(("  ok   " if cond else "  FAIL ") + what)
    if not cond:
        FAILS.append(what)


def _maps(car: str):
    try:
        return carfiles.surface_maps(car)
    except Exception:                              # noqa: BLE001
        return {}


def segments_level() -> None:
    print("[조각별 이음새]")
    any_car = False
    for car in CARS:
        maps = _maps(car)
        if not maps:
            continue
        any_car = True
        for src, dst in PAIRS:
            if src not in maps or dst not in maps:
                continue
            f = gfold.fold(maps, src, dst)
            if f is None:
                continue
            check(len(f.segments) >= 2,
                  f"{car[:12]} {src}→{dst}: 토막 {len(f.segments)}개")
            if not f.segments:
                continue
            # **이음선 위의 점은 건너편 이음선으로 간다** — 토막마다.
            worst = 0.0
            for sg in f.segments:
                t = sg.mid
                u, v = (sg.src_edge, t) if f.axis == "u" else (t, sg.src_edge)
                du, dv = f.to_local(u, v)
                pi = f._perp_index()
                got = du if pi == 0 else dv
                worst = max(worst, abs(got - sg.dst_edge))
            check(worst < 1e-6,
                  f"{car[:12]} {src}→{dst}: 토막 위 점이 건너편 이음선으로 (최대 어긋남 {worst:.2e})")
            # 신뢰는 0~1
            check(all(0.0 <= sg.confidence <= 1.0 for sg in f.segments),
                  f"{car[:12]} {src}→{dst}: 신뢰가 0~1")
            # 기울기 — 유한하고, 아핀 하나로 물러난 자리에서는 0이다
            s_sl, d_sl = f.seam_tangent(f.segments[0].mid)
            check(np.isfinite(s_sl) and np.isfinite(d_sl),
                  f"{car[:12]} {src}→{dst}: 이음선 기울기 src {s_sl:+.3f} · dst {d_sl:+.3f}")
    check(any_car, "실측 설치 마스크가 있는 차가 하나 이상")


def bottom_line_level() -> None:
    """차체 아랫선은 이음새를 건너도 **아랫선**이다 (`shared_offset`의 근거)."""
    print("[바닥선]")
    from forzasqueegee.game import seam as gseam
    for car in CARS:
        maps = _maps(car)
        if "side_left" not in maps:
            continue
        g = gseam.side_geom(maps["side_left"])
        for dst in ("rear", "front"):
            if dst not in maps:
                continue
            f = gfold.fold(maps, "side_left", dst)
            if f is None:
                continue
            _u, v = f.to_local(f.edge, g.sill)
            q0, q1 = maps[dst].paint[1], maps[dst].paint[3]
            frac = (v - q0) / max(1e-6, q1 - q0)
            check(frac < 0.20,
                  f"{car[:12]} side→{dst}: 사이드실이 면 아래 {frac:.2f} 자리로 간다")


def carry_level() -> None:
    print("[띠 잇기]")
    for car in CARS:
        maps = _maps(car)
        if "side_left" not in maps or "rear" not in maps:
            continue
        f = gfold.fold(maps, "side_left", "rear")
        if f is None:
            continue
        sm = maps["rear"]
        ph = sm.paint[3] - sm.paint[1]
        src = gseams.Band(v=-40.0, angle=6.0, thickness=48.0, at_u=0.0)
        con = gseams.carry(f, src, "macro", dst_box=sm.paint, tilt_max=18.0,
                           thick_max=0.30 * ph)
        check(con.policy in (gseams.CONTINUE, gseams.TERMINATE),
              f"{car[:12]} 큰 색면: {con.policy} — {con.why}")
        if con.carried:
            err = gseams.seam_error(f, src, con.band)
            check(err["pos"] < 1e-6,
                  f"{car[:12]} 이음선 위 자리 어긋남 {err['pos']:.2e}유닛")
            check(err["width"] >= gseams.WIDTH_MIN,
                  f"{car[:12]} 두께 비 {err['width']:.3f} ≥ {gseams.WIDTH_MIN}")
            check(err["tilt"] <= gseams.TILT_CLAMP_MAX + 1e-9,
                  f"{car[:12]} 각 차 {err['tilt']:.2f}° ≤ {gseams.TILT_CLAMP_MAX}")
            # **기울인 띠는 이음선 위에서만 맞는 게 아니라 그 각으로 이어져야 한다**
            d = gseams._direction(f, src.angle)
            import math
            want = math.degrees(math.atan2(d[1], d[0]))
            want = want - 180.0 if want > 90.0 else want
            check(abs(con.band.angle - max(-18.0, min(18.0, want))) < 1e-9,
                  f"{car[:12]} 건너간 각 {con.band.angle:.2f}° = 클램프한 값")
        # 두께 상한이 아주 짜면 **끊는다** (억지로 안 깎는다)
        con2 = gseams.carry(f, src, "macro", dst_box=sm.paint, tilt_max=18.0,
                            thick_max=0.10 * src.thickness)
        check(con2.policy == gseams.TERMINATE,
              f"{car[:12]} 두께를 못 지키면 끊는다 — {con2.why}")
        # 면 밖으로 가는 높이도 끊는다
        far = gseams.Band(v=10_000.0, angle=0.0, thickness=10.0, at_u=0.0)
        con3 = gseams.carry(f, far, "macro", dst_box=sm.paint, tilt_max=18.0,
                            thick_max=ph)
        check(con3.policy == gseams.TERMINATE,
              f"{car[:12]} 면 밖 높이는 끊는다 — {con3.why}")


def policy_level() -> None:
    print("[정책]")
    check(gseams.policy_for("text") == gseams.AVOID, "글자는 이음새를 피한다")
    check(gseams.policy_for("face") == gseams.AVOID, "얼굴은 이음새를 피한다")
    check(gseams.policy_for("itasha_bed") == gseams.CONTINUE, "큰 색면은 이어 간다")
    check(gseams.policy_for("itasha_stripe") == gseams.CONTINUE, "하부 투톤은 이어 간다")
    check(gseams.policy_for("itasha_deco") == gseams.TERMINATE, "산포 모티프는 끊는다")
    check(gseams.policy_for("무엇이든") == gseams.TERMINATE, "모르는 역할은 끊는다")
    # 낮은 신뢰 — 손으로 지은 이음새로
    f = gfold.Fold(src="a", dst="b", axis="u", sign=1.0, edge=0.0,
                   A=np.array([[1.0, 0.0], [0.0, 1.0]]), b=np.zeros(2),
                   segments=(gfold.SeamSegment(-100, 0, 0.0, 0.0, 0.0, 0.10),
                             gfold.SeamSegment(0, 100, 0.0, 0.0, 0.0, 0.10)))
    con = gseams.carry(f, gseams.Band(v=0.0, angle=0.0, thickness=10.0, at_u=0.0),
                       "macro", dst_box=(-100, -100, 100, 100), tilt_max=18.0,
                       thick_max=100.0)
    check(con.policy == gseams.TERMINATE, f"신뢰가 낮으면 끊는다 — {con.why}")


def place_level() -> None:
    """`flow_shapes`가 준 점을 띠가 실제로 지나나."""
    print("[놓는 자리]")
    import math
    for car in CARS:
        maps = _maps(car)
        sm = maps.get("rear")
        if sm is None:
            continue
        au = sm.paint[0] + 0.8 * (sm.paint[2] - sm.paint[0])
        cv = (sm.paint[1] + sm.paint[3]) / 2
        got = flow_shapes((10, 20, 30), sm, mode="macro", center_v=cv, anchor_u=au,
                          rot=8.0, height=40.0)
        worst = max(abs(float(s["y"]) + (au - float(s["x"]))
                        * math.tan(math.radians(float(s["rot"]))) - cv) for s in got)
        check(worst <= 0.06, f"{car[:12]} macro: 조각 {len(got)}개가 이음선 위 점을 지난다 "
                             f"(최대 {worst:.3f}유닛 — 좌표는 소수 한 자리로 끊는다)")
        # **이음새에서 겹친다** — 띠는 면 끝을 넘겨서 끝난다 (넘긴 몫은 면
        # 마스크가 자른다). 겹침의 자는 캔버스 유닛이 아니라 **면 제 폭**이라
        # 차 크기와 무관하다 (`surfshapes.BAND_OVERSHOOT`).
        u0 = min(float(s["x"]) - abs(float(s["sx"])) * UNITS_PER_SCALE for s in got)
        u1 = max(float(s["x"]) + abs(float(s["sx"])) * UNITS_PER_SCALE for s in got)
        w = sm.paint[2] - sm.paint[0]
        over = min(sm.paint[0] - u0, u1 - sm.paint[2]) / w
        check(0.0 < over <= 0.15,
              f"{car[:12]} macro: 이음새 양쪽으로 면 폭의 {over:.3f}만큼 넘긴다 (0 초과 0.15 이하)")
        top = sm.paint[1] + 0.3 * (sm.paint[3] - sm.paint[1])
        got = flow_shapes((10, 20, 30), sm, mode="rocker", top_v=top)
        band = [s for s in got if s.get("shape") == "A_01"]
        worst = max(abs(float(s["y"]) + abs(float(s["sy"])) * UNITS_PER_SCALE - top)
                    for s in band)
        check(worst <= 0.06, f"{car[:12]} rocker: 윗선이 준 높이에 선다 (최대 {worst:.3f}유닛)")
        check(all(float(s["sy"]) > 0 for s in band),
              f"{car[:12]} rocker: 밴드 높이가 양수다 (건너온 윗선이 상자 밑에 떨어져도)")
        # 상자 아래끝보다 낮은 윗선을 줘도 밴드가 뒤집히지 않는다
        low = sm.paint[1] - 0.05 * (sm.paint[3] - sm.paint[1])
        band2 = [s for s in flow_shapes((10, 20, 30), sm, mode="rocker", top_v=low)
                 if s.get("shape") == "A_01"]
        check(all(float(s["sy"]) > 0 for s in band2),
              f"{car[:12]} rocker: 상자 밑 윗선에서도 높이가 양수다")


def determinism_level() -> None:
    print("[결정성]")
    for car in CARS[:1]:
        maps = _maps(car)
        if "side_left" not in maps:
            continue
        a = gfold.fold(maps, "side_left", "rear")
        b = gfold.fold(carfiles.surface_maps(car), "side_left", "rear")
        check(a is not None and b is not None
              and [tuple(vars(s).values()) for s in a.segments]
              == [tuple(vars(s).values()) for s in b.segments],
              "같은 입력 두 번 → 같은 토막")
        sm = maps["rear"]
        src = gseams.Band(v=-40.0, angle=6.0, thickness=48.0, at_u=0.0)
        k = [gseams.carry(f, src, "macro", dst_box=sm.paint, tilt_max=18.0,
                          thick_max=0.3 * (sm.paint[3] - sm.paint[1])) for f in (a, b)]
        check(k[0].policy == k[1].policy and k[0].metrics == k[1].metrics,
              "같은 입력 두 번 → 같은 결정·같은 원자료")


def text_margin_level() -> None:
    """글자의 이음새 여유 자 — 상자가 프레임을 넘으면 음수다."""
    print("[글자 여유]")
    from forzasqueegee.engine.compose.field import CompositionField, FieldGrid
    from forzasqueegee.engine.compose.textlayout import SEAM_PAD, TextPose, seam_margin

    z = np.zeros((10, 10), np.float32)
    fld = CompositionField(grid=FieldGrid(x0=-450.0, y_top=100.0, cols=180, rows=40),
                           frame_box=(-450.0, -100.0, 450.0, 100.0),
                           person_box=(-100.0, -80.0, 100.0, 80.0),
                           char=z, char_rgb=np.zeros((10, 10, 3), np.uint8), detail=z,
                           drawable=z, exposed=z, head=z, protected=z, support=z, decoration=z,
                           negative=z)
    mid = TextPose(role="wordmark", text="X", x=0.0, y=0.0, rot=0.0, height=50.0,
                   aspect=2.0)
    check(abs(seam_margin(fld, mid) - (450.0 - 50.0) / 900.0) < 1e-9,
          f"가운데 글자의 여유 {seam_margin(fld, mid):.4f}")
    edge = TextPose(role="wordmark", text="X", x=440.0, y=0.0, rot=0.0, height=50.0,
                    aspect=2.0)
    check(seam_margin(fld, edge) < 0, f"프레임을 넘은 글자의 여유 {seam_margin(fld, edge):.4f} < 0")
    check(0.0 < SEAM_PAD < 0.2, f"여유 문턱 {SEAM_PAD}")


def main() -> int:
    segments_level()
    bottom_line_level()
    carry_level()
    policy_level()
    place_level()
    determinism_level()
    text_margin_level()
    print(f"{'FAIL' if FAILS else 'PASS'} {len(FAILS)}")
    for f in FAILS:
        print("  - " + f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
