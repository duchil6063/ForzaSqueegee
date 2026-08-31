"""FLS 기하 덤프(`.fsgeom`)를 **설치 마스크와 대 본다** — 메시 노선의 첫 자.

    python tools/geom_check.py                   # work/geom에 떠 둔 덤프 전부
    python tools/geom_check.py --media MAZ_Miata_94
    python tools/geom_check.py --dump --limit 40  # 없는 덤프를 그 자리에서 뜬다

꾸밈 기하를 메시로 옮기기 전에 **덤프가 우리 좌표계에 맞는가**를 먼저 판정한다.
다섯을 잰다:

- `[box]`  면 유닛 상자·격자 모양이 설치 마스크와 같은가 (여기가 틀리면 나머지는
           볼 것도 없다).
- `[cover]` 깊이 래스터가 보는 몫과 도색 마스크의 겹침 — 둘은 **서로 다른 것**을
           재므로(칠하는 자리 ↔ 보이는 표면) 겹침이 낮은 것 자체는 흠이 아니다.
           빈 래스터(도어 유리)를 짚는 자리다.
- `[scale]` 면마다의 유닛/m, 그리고 **같은 물리 길이를 두 면으로 잰 값의 어긋남**.
           `game.fold`는 "면 유닛은 한 차 안에서 한 자"를 전제하는데, 그 전제가
           앞·뒤 면에서도 서는지 여기서 처음 잰다.
- `[k]`    `locators.register`가 휠아치에 적합한 배율 ↔ 덤프가 닫힌 식으로 주는
           배율. 심판은 **로케이터 휠베이스**다 (미터, 실차와 0.5% 안에서 맞는
           것이 따로 확인됐다): 마스크 아치 간격 ÷ (K × 휠베이스)가 1에서 얼마나
           떨어지나. 값이 한 자리에 뭉치면 아치 중심 검출의 계통 편향이고,
           흩어지면 둘 중 하나가 못 믿을 자다.
- `[cross]` **면끼리의 깊이 일관성** — 옆면 한 점을 세계로 풀어 윗면에 던지고,
           윗면이 그 자리에서 말하는 깊이와 옆면이 말한 높이를 견준다. 접기
           그래프를 메시로 갈아 끼울 때(`game.hull`) 이 값이 합격선이 된다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from forzasqueegee.game import carfiles, fsgeom, locators as gl, seam as gseam  # noqa: E402

BODY = ("front", "rear", "top", "side_left", "side_right")

FAILS: list[str] = []


def _fail(what: str) -> None:
    FAILS.append(what)
    print("    FAIL " + what)


# ---------- [box] ----------
def check_box(media: str, geom: fsgeom.CarGeom, maps: dict) -> None:
    bad = 0
    for tab, side in sorted(geom.sides.items()):
        sm = maps.get(tab)
        if sm is None:
            _fail(f"{media} {tab}: 설치 마스크에 없는 면이 덤프에 있다")
            bad += 1
            continue
        if max(abs(a - b) for a, b in zip(side.box, sm.paint)) > 0.51:
            _fail(f"{media} {tab}: 상자가 다르다 {side.box} ≠ {sm.paint}")
            bad += 1
        elif side.depth is not None and side.depth.shape != sm.mask.shape:
            _fail(f"{media} {tab}: 격자가 다르다 {side.depth.shape} ≠ {sm.mask.shape}")
            bad += 1
    print(f"    상자·격자 {len(geom.sides) - bad}/{len(geom.sides)} 일치")


# ---------- [cover] ----------
def check_cover(media: str, geom: fsgeom.CarGeom, maps: dict, rows: list) -> None:
    for tab in sorted(geom.sides):
        side = geom.sides[tab]
        sm = maps.get(tab)
        if sm is None or side.depth is None or side.depth.shape != sm.mask.shape:
            rows.append((media, tab, side.seen, float("nan"), float("nan")))
            continue
        live = np.isfinite(side.depth)
        m = sm.mask
        inter = float((live & m).sum())
        rows.append((media, tab, side.seen,
                     inter / max(1.0, float(m.sum())),          # 마스크 중 보이는 몫
                     inter / max(1.0, float((live | m).sum()))))  # IoU


# ---------- [scale] ----------
# 같은 물리 길이를 쥔 (면, 축) 짝 — `game.fold`의 축 표와 같은 뜻이다.
SHARED = {
    "길이(z)": (("side_left", 0), ("side_right", 0), ("top", 0)),
    "폭(x)": (("front", 0), ("rear", 0), ("top", 1)),
    "높이(y)": (("side_left", 1), ("side_right", 1), ("front", 1), ("rear", 1)),
}


def check_scale(media: str, geom: fsgeom.CarGeom, rows: list) -> None:
    for what, pairs in SHARED.items():
        vals = []
        for tab, axis in pairs:
            side = geom.sides.get(tab)
            if side is None:
                continue
            vals.append((tab, side.units_per_m[axis]))
        if len(vals) < 2:
            continue
        ks = [v for _t, v in vals]
        spread = (max(ks) - min(ks)) / max(1e-9, float(np.median(ks)))
        rows.append((media, what, spread, " ".join(f"{t}={v:.0f}" for t, v in vals)))


# ---------- [k] ----------
def check_k(media: str, geom: fsgeom.CarGeom, maps: dict, rows: list) -> None:
    side = geom.sides.get("side_left")
    sm = maps.get("side_left")
    if side is None or sm is None:
        return
    k_geom = side.units_per_m[0]
    try:
        reg, _items = gl.for_car(media)
    except Exception:                              # noqa: BLE001 — 없어도 판은 선다
        reg = None
    k_loc = reg.k if reg is not None else float("nan")
    locs = gl.read(media)
    zs = sorted(v[2] for k, v in locs.items() if "wheel" in k.lower())
    wb = (max(zs) - min(zs)) if len(zs) >= 2 else float("nan")
    try:
        arches = gseam.side_geom(sm).wheels
    except Exception:                              # noqa: BLE001
        arches = ()
    sep = abs(arches[-1][0] - arches[0][0]) if len(arches) >= 2 else float("nan")
    rows.append((media, k_geom, k_loc, wb, sep,
                 sep / max(1e-9, k_geom * wb),
                 sep / max(1e-9, k_loc * wb) if reg is not None else float("nan")))


# ---------- [cross] ----------
# 두 면이 **같은 표면**을 본다고 볼 문턱 (미터). 이보다 크게 벌어진 점은
# 어긋남이 아니라 가림이다 — 문짝 아래쪽은 윗면에서 안 보이므로 윗면은 그
# 자리에서 지붕을 본다 (실측: 어긋남 분포가 1 cm 아래와 30 cm 위로 갈린다).
SAME_SURFACE = 0.03


# 견줄 면 짝 — 이웃하는 차체 면 전부. **앞·뒤 ↔ 옆·윗이 요점이다**: 그 짝의
# 세로 배율이 서로 2배 넘게 다르므로(`[scale]`), 여기서도 맞으면 덤프의 투영이
# 옳고 `game.fold`의 등거리 가정이 틀린 것이다.
CROSS_PAIRS = (("side_left", "top"), ("front", "side_left"), ("front", "top"),
               ("rear", "side_left"), ("rear", "top"))


def check_cross(media: str, geom: fsgeom.CarGeom, rows: list,
                n: int = 4000) -> None:
    """한 면 → 세계 → 이웃 면. **둘 다 보는 표면**에서 좌표계가 얼마나 맞나."""
    for sname, dname in CROSS_PAIRS:
        src, dst = geom.sides.get(sname), geom.sides.get(dname)
        if src is None or dst is None or src.depth is None or dst.depth is None:
            continue
        h, w = src.depth.shape
        u0, v0, u1, v1 = src.box
        rs, cs = np.where(np.isfinite(src.depth))
        if len(rs) < 32:
            continue
        step = max(1, len(rs) // n)
        rs, cs = rs[::step], cs[::step]
        us = u0 + cs / max(1, w - 1) * (u1 - u0)
        vs = v1 - rs / max(1, h - 1) * (v1 - v0)
        world = src.world(us, vs)
        du, dv = dst.to_face(world)
        # 목적 면이 그 자리에서 말하는 깊이 ↔ 출발 면이 말한 그 축의 좌표
        axis = fsgeom.AXIS_LETTER[dst.depth_axis]
        d = np.abs(dst.at(du, dv) - world[axis])
        ok = np.isfinite(d)
        same = ok & (d <= SAME_SURFACE)
        pair = f"{sname}→{dname}"
        if same.sum() < 16:
            rows.append((media, pair, float(ok.mean()), 0.0,
                         float("nan"), float("nan")))
            continue
        upm = float(np.median(dst.units_per_m))
        med = float(np.median(d[same]))
        rows.append((media, pair, float(ok.mean()),
                     float(same.sum() / max(1, ok.sum())), med, med * upm))


# ---------- 판 ----------
def one(media: str, geom: fsgeom.CarGeom, acc: dict) -> None:
    try:
        maps = carfiles.surface_maps(media)
    except Exception as e:                         # noqa: BLE001
        _fail(f"{media}: 설치 마스크를 못 읽었다 ({e})")
        return
    check_box(media, geom, maps)
    check_cover(media, geom, maps, acc["cover"])
    check_scale(media, geom, acc["scale"])
    check_k(media, geom, maps, acc["k"])
    check_cross(media, geom, acc["cross"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--media", default=None, help="한 대만")
    ap.add_argument("--dump", action="store_true",
                    help="덤프가 없는 차는 그 자리에서 뜬다 (동봉 편집기)")
    ap.add_argument("--limit", type=int, default=0, help="차 수 상한")
    args = ap.parse_args()

    if args.media:
        names = [args.media]
    elif args.dump:
        names = carfiles.list_cars()
    else:
        names = sorted(p.stem for p in fsgeom.geom_dir().glob("*.fsgeom"))
    if args.limit:
        names = names[:args.limit]
    if not names:
        print("덤프가 없다 — `--dump`로 뜨거나 work/geom에 놓을 것")
        return 2

    acc = {"cover": [], "scale": [], "k": [], "cross": []}
    print("[box]")
    done = 0
    for media in names:
        geom = fsgeom.for_car(media)
        if geom is None and args.dump:
            try:
                fsgeom.dump(media)
            except Exception as e:                 # noqa: BLE001
                print(f"    skip {media} ({e})")
                continue
            geom = fsgeom.for_car(media)
        if geom is None:
            print(f"    skip {media} (덤프 없음)")
            continue
        one(media, geom, acc)
        done += 1
    print(f"    차 {done}대")

    print("[cover] 면마다 (보이는 칸 / 마스크 중 보이는 몫 / IoU)")
    per: dict[str, list] = {}
    for _m, tab, seen, cov, iou in acc["cover"]:
        per.setdefault(tab, []).append((seen, cov, iou))
    for tab in sorted(per, key=lambda t: (t not in BODY, t)):
        a = np.array(per[tab], float)
        with np.errstate(invalid="ignore"):
            print(f"    {tab:14s} seen {np.nanmedian(a[:, 0]):.2f}  "
                  f"cov {np.nanmedian(a[:, 1]):.2f}  IoU {np.nanmedian(a[:, 2]):.2f}"
                  + ("   ← 래스터 빈칸" if np.nanmedian(a[:, 0]) < 0.01 else ""))

    print("[scale] 같은 길이를 두 면으로 잰 값의 벌어짐 (0이면 한 자)")
    per2: dict[str, list] = {}
    for _m, what, spread, _txt in acc["scale"]:
        per2.setdefault(what, []).append(spread)
    for what, vals in per2.items():
        v = np.array(vals, float)
        print(f"    {what:10s} 중앙 {np.median(v) * 100:5.1f}%  "
              f"p90 {np.percentile(v, 90) * 100:5.1f}%  최대 {v.max() * 100:5.1f}%")
    worst = sorted(acc["scale"], key=lambda r: -r[2])[:3]
    for m, what, spread, txt in worst:
        print(f"      최악: {m} {what} {spread * 100:.1f}% — {txt}")

    print("[k] 유닛/m — 덤프(닫힌 식) ↔ locators(아치 적합), 심판은 휠베이스")
    print("    아치간격 ÷ (K × 휠베이스)가 1이면 그 K가 맞다")
    for m, kg, kl, wb, sep, rg, rl in acc["k"]:
        print(f"    {m:24s} K_geom {kg:6.1f}  K_loc {kl:6.1f}  "
              f"wb {wb:5.3f}m  아치 {sep:6.1f}  →  geom {rg:5.3f}  loc {rl:5.3f}")
    if acc["k"]:
        rg = np.array([r[5] for r in acc["k"]], float)
        rg = rg[np.isfinite(rg)]
        if len(rg):
            print(f"    geom 비: 중앙 {np.median(rg):.3f} · 벌어짐 "
                  f"{(rg.max() - rg.min()):.3f} (뭉치면 아치 검출의 계통 편향)")

    print("[cross] 면 → 이웃 면 (풀린 몫 / 둘 다 보는 몫 / 그 자리 어긋남)")
    print(f"    나머지는 가림이다 — 문짝 아래는 윗면에서 안 보인다 (문턱 "
          f"{SAME_SURFACE * 100:.0f} cm)")
    per3: dict[str, list] = {}
    for _m, pair, frac, share, dm, du in acc["cross"]:
        per3.setdefault(pair, []).append((frac, share, dm, du))
    for pair in [f"{a}→{b}" for a, b in CROSS_PAIRS if f"{a}→{b}" in per3]:
        a = np.array(per3[pair], float)
        with np.errstate(invalid="ignore"):
            print(f"    {pair:22s} 풀림 {np.nanmedian(a[:, 0]):.2f}  "
                  f"같은면 {np.nanmedian(a[:, 1]):.2f}  어긋남 중앙 "
                  f"{np.nanmedian(a[:, 3]):.2f} 유닛 · p90 "
                  f"{np.nanpercentile(a[:, 3], 90):.2f}")

    print("FAIL" if FAILS else "PASS", len(FAILS))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
