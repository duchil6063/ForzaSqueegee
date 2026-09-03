"""면 배정 회귀 검사 — 스펙·`auto` 풀이·이어 그리기 기하·결정성 (계획 5단계).

    .venv/Scripts/python.exe tools/face_regress.py

보는 것:
- `FaceSpec`이 모르는 면·모드를 거부하고, dict·CLI 꼴을 같은 스펙으로 받는다
- `auto`가 로고·글자의 유무로 갈리는 표 (도어 유리·프론트는 없으면 크롭으로 물러난다,
  리어·뒷유리는 안 물러난다, `empty`는 아무것도 안 준다, `continue`는 유리에만)
- 이어 그리기의 변환: 유리 이음새의 **아랫변**에서 옆면 점과 유리 점이 만난다
  (가로는 `su`, 세로는 이음선에서 정확히 0 어긋남)
- 윈드실드 띠의 지붕 끝 판정 (`facetext._roof_end`) — 좁은 끝이 지붕이다
- 뒷유리 날개 (`facetext._wings`) — 글자 양옆에 점대칭 둘, 자리가 없으면 0
- 스튜디오 `act_faces`가 준 면만 바꾸고 조리법에 남는다
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from forzasqueegee.engine.catalog import Catalog, default_catalog_path  # noqa: E402
from forzasqueegee.engine.compose import facetext, families  # noqa: E402
from forzasqueegee.engine.compose.facespec import (  # noqa: E402
    FACES, FACE_OF, MODES, FaceSpec)
from forzasqueegee.engine.compose.place import ManualPlace, place_xf  # noqa: E402
from forzasqueegee.engine.compose.roles import RolePalette  # noqa: E402
from forzasqueegee.game import fold as gfold, surface as gsurf  # noqa: E402

FAILS: list[str] = []


def check(cond: bool, what: str) -> None:
    print(("  ok   " if cond else "  FAIL ") + what)
    if not cond:
        FAILS.append(what)


def _spec_checks() -> None:
    print("[spec]")
    s = FaceSpec.from_args(["window=continue", "rear=crop", "window_left=crop"])
    check(s.window == "crop" and s.rear == "crop", "CLI 꼴 — 뒤에 준 값이 이기고 window_left는 window로 접힌다")
    check(FaceSpec.from_dict(s.to_dict()) == s, "dict 왕복")
    check(FaceSpec.from_args(None) is None and FaceSpec.from_args([]) is None, "빈 인자는 None")
    for bad in (["rear=continue"], ["windshield=crop"], ["hood=crop"], ["window"]):
        try:
            FaceSpec.from_args(bad)
            check(False, f"거부해야 한다: {bad}")
        except ValueError:
            check(True, f"거부: {bad}")
    check(FaceSpec().all_auto and not s.all_auto, "all_auto")
    check(all(m in MODES[f] for f in FACES for m in ("auto", "empty")), "모든 면에 auto·empty")


def _auto_table() -> None:
    print("[auto]")
    a = FaceSpec()
    # (면, 로고, 글자) → (crop, sponsor)
    table = {
        ("window_left", False, False): (True, False),
        ("window_left", True, False): (False, True),
        ("window_left", False, True): (False, True),
        ("front", False, False): (True, False),
        ("front", True, False): (False, True),
        ("rear", False, False): (False, True),
        ("rear_window", False, False): (False, True),
        ("windshield", False, False): (False, True),
    }
    for (face, lg, tx), (crop, sp) in table.items():
        got = a.resolve(face, logos=lg, text=tx)
        check((got.crop, got.sponsor) == (crop, sp) and not got.cont,
              f"auto {face} logos={lg} text={tx} → crop={crop} sponsor={sp}")
    e = FaceSpec(**{f: "empty" for f in FACES})
    check(all(not (g.crop or g.sponsor or g.cont)
              for g in (e.resolve(n, logos=True, text=True) for n in FACE_OF)),
          "empty는 아무것도 안 준다")
    c = FaceSpec(window="continue")
    g = c.resolve("window_right", logos=True, text=True)
    check(g.cont and not g.crop and not g.sponsor, "continue는 유리에 사본만")
    k = FaceSpec(rear="crop", front="crop", rear_window="crop", window="crop")
    check(all(k.resolve(n, logos=True, text=True).crop
              for n in ("rear", "front", "rear_window", "window_left")),
          "crop은 로고·글자가 있어도 크롭")
    check(not FaceSpec(front="logos").resolve("front", logos=False, text=False).crop,
          "front=logos는 로고가 없어도 크롭으로 안 물러난다")
    check(FaceSpec(window="support").resolve("window_left", logos=True, text=True)
          == FaceSpec(window="support").resolve("window_left", logos=False, text=False),
          "support는 로고·글자와 무관")


def _continuation_geometry() -> None:
    print("[continue]")
    # 옆면 → 유리 아핀 (실측 줄리아 꼴: su .926 · sv 1.102, 유리 아랫변 -67)
    su, sv = 0.926, 1.102
    gv0 = -67.0
    pivot = 14.5
    b = np.array([-81.1, gv0 - pivot * sv])
    f = gfold.Fold(src="side_left", dst="window_left", axis="v", sign=1.0, edge=27.6,
                   A=np.diag([su, sv]), b=b)
    mp = ManualPlace(plan=Path("x.json"), surface="side_left", x=126.5, y=-15.5,
                     scale=0.26, rot=0.0, mirror=False)
    k = su
    x2 = su * mp.x + float(f.b[0])
    y2 = k * mp.y + float(f.b[1]) + (sv - k) * ((gv0 - float(f.b[1])) / sv)
    cp = ManualPlace(plan=mp.plan, surface="window_left", x=x2, y=y2, scale=mp.scale * k,
                     rot=0.0, mirror=False)
    gu = 1.0
    L1, t1 = place_xf(mp, gu)
    L2, t2 = place_xf(cp, gu)
    # 이음선(옆면 v = pivot) 위의 점 — 캔버스 점 p로 옆면에서 그 높이가 되는 p를 잡는다
    for px in (-200.0, 0.0, 300.0):
        py = (pivot - t1[1]) / L1[1, 1]
        p = np.array([px, py])
        s_pt = L1 @ p + t1
        w_true = np.array(f.to(s_pt[0], s_pt[1]))
        w_got = L2 @ p + t2
        check(abs(w_got[1] - gv0) < 1e-6 and abs(w_true[1] - gv0) < 1e-6,
              f"이음선 점 u={px:g}: 유리 아랫변에서 만난다 (v {w_got[1]:.3f})")
        check(abs(w_got[0] - w_true[0]) < 1e-6, f"이음선 점 u={px:g}: 가로 자리가 아핀과 같다")
    # 이음선 위 100유닛에서 세로 어긋남은 정확히 (sv-su)×100
    py = (pivot + 100.0 - t1[1]) / L1[1, 1]
    p = np.array([0.0, py])
    s_pt = L1 @ p + t1
    w_true = np.array(f.to(s_pt[0], s_pt[1]))
    w_got = L2 @ p + t2
    check(abs((w_true[1] - w_got[1]) - (sv - su) * 100.0) < 1e-6,
          "이음선 위 100유닛의 세로 어긋남 = (sv−su)×100")


def _mask_map(mask: np.ndarray, paint=(-100.0, -50.0, 100.0, 50.0)) -> gsurf.SurfaceMap:
    return gsurf.SurfaceMap(name="windshield", index=0, origin_px=(0.0, 0.0),
                            px_per_unit=(1.0, 1.0), paint=paint, fill=1.0, mask=mask)


def _roof_end() -> None:
    print("[roof end]")
    h, w = 40, 80
    trap = np.zeros((h, w), bool)
    for r in range(h):                    # 행 0 = v1 (위) — 위가 좁은 사다리꼴
        half = int(10 + 30 * r / (h - 1))
        trap[r, w // 2 - half:w // 2 + half] = True
    v, width = facetext._roof_end(_mask_map(trap))
    check(v == 50.0 and width < 60.0, f"위가 좁으면 지붕은 v1 (폭 {width:.0f})")
    v, width = facetext._roof_end(_mask_map(trap[::-1].copy()))
    check(v == -50.0, "아래가 좁으면 지붕은 v0")


def _wings() -> None:
    print("[wings]")
    cat = Catalog(default_catalog_path())
    pal = RolePalette(base=(240, 240, 240), bed=(30, 30, 60), bed_alt=(60, 60, 90),
                      primary=(200, 40, 40), secondary=(40, 40, 200), shadow=(20, 20, 20),
                      highlight=(250, 250, 250), dark=(10, 10, 10), variant="primary")
    sm = _mask_map(np.ones((40, 80), bool), paint=(-200.0, -50.0, 200.0, 50.0))
    for fam, want in (("splash", 2), ("minimal", 0)):
        design = SimpleNamespace(family=families.FAMILIES[fam], pal=pal)
        got = facetext._wings(design, cat, sm, 0.0, 0.0, 120.0, 24.0)
        check(len(got) == want, f"{fam}: 날개 {want}개")
        if got:
            check(got[0]["rot"] == 0.0 and got[1]["rot"] == 180.0
                  and abs(got[0]["x"] + got[1]["x"]) < 1e-6, "점대칭 한 쌍")
            check(got[0]["x"] < -60.0 and got[1]["x"] > 60.0, "글자 밖에 선다")
    design = SimpleNamespace(family=families.FAMILIES["splash"], pal=pal)
    got = facetext._wings(design, cat, sm, 0.0, 0.0, 380.0, 24.0)
    check(len(got) == 0, "자리가 없으면 0")
    a = facetext._wings(design, cat, sm, 0.0, 0.0, 120.0, 24.0)
    b = facetext._wings(design, cat, sm, 0.0, 0.0, 120.0, 24.0)
    check(a == b, "결정성")


def _studio() -> None:
    print("[studio]")
    from forzasqueegee.engine.fls import studio

    st = SimpleNamespace(state=studio._blank_state(), notes=[])
    check(st.state["faces"] == {f: "auto" for f in FACES}, "빈 조리법은 전부 auto")
    studio.act_faces(st, ["window=continue", "rear=crop"])
    check(st.state["faces"]["window"] == "continue" and st.state["faces"]["rear"] == "crop"
          and st.state["faces"]["front"] == "auto", "준 면만 바뀐다")
    studio.act_faces(st, {"rear": "auto"})
    check(st.state["faces"]["rear"] == "auto" and st.state["faces"]["window"] == "continue",
          "dict로도 준 면만")
    try:
        studio.act_faces(st, ["rear=continue"])
        check(False, "모르는 모드는 거부")
    except ValueError:
        check(True, "모르는 모드는 거부")


def main() -> int:
    _spec_checks()
    _auto_table()
    _continuation_geometry()
    _roof_end()
    _wings()
    _studio()
    print("\n실패 %d" % len(FAILS))
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
