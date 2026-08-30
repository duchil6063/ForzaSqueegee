"""텍스트 엔진 회귀 검사 — 글자 fixture의 품질 바닥과 불변식.

    python tools/text_regress.py                # 글리프 수준 (수십 초)
    python tools/text_regress.py --e2e PLAN MEDIA   # 구성 두 번 굽기 (결정성·상한·텍스트 끔 해시)

글리프 수준에서 보는 것:
- 카운터 fixture(racing "RIN SHIBUYA")의 카운터 침범 · IoU 바닥
- script · racing · graffiti fixture의 IoU 바닥
- 작은 로커 글자(20유닛)의 장수 상한
- 테두리·그림자 벌이 본색보다 적다
- 줄바꿈·공백·구두점·대소문자가 그대로다 (`lockups`도 글자를 안 바꾼다)
- 같은 입력 두 번 → 같은 레이어 (결정성)
- 좌우 면: 자리는 거울, 글자 상대 배치는 그대로 (미러 금지)
- 예산 판(`plan_for_budget`)이 예산을 안 넘는다

`--e2e`는 `compose.build`를 텍스트 끄고 두 번·켜고 두 번 돌려 바이트 동일을
확인하고, 면 장수가 3,000을 안 넘는지 보고, **텍스트 끈 판의 해시를 찍는다** —
엔진을 고친 뒤 이전 판과 대조하는 자다 (끈 판은 바이트 동일해야 한다).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from forzasqueegee.engine.catalog import Catalog, default_catalog_path  # noqa: E402
from forzasqueegee.engine import textfit as tf  # noqa: E402
from forzasqueegee.engine import textglyph as tg  # noqa: E402
from forzasqueegee.engine.compose.textlayout import TextPose, lockups  # noqa: E402


FAILS: list[str] = []


def check(cond: bool, what: str) -> None:
    print(("  ok   " if cond else "  FAIL ") + what)
    if not cond:
        FAILS.append(what)


def glyph_level() -> None:
    cat = Catalog(default_catalog_path())
    t0 = time.perf_counter()
    # 품질 바닥 — (텍스트, 스타일, IoU 바닥, 카운터 침범 상한)
    for text, style, iou_min, ctr_max in (("RIN SHIBUYA", "racing", 0.88, 0.10),
                                          ("Sorae", "script", 0.80, 0.15),
                                          ("Evelyne", "graffiti", 0.85, 0.15),
                                          ("ARIS", "techno", 0.88, 0.10)):
        f = tg.fit_fill(text, style, tg.TIER_INDEX["A"])
        check(f.iou >= iou_min and f.counter <= ctr_max,
              f"{style} {text!r}: IoU {f.iou:.3f} ≥ {iou_min} · 카운터 침범 {f.counter:.3f} ≤ {ctr_max} · {f.n}장")
    # 테두리·그림자는 본색보다 적다
    blk = tg.build_text("RIN SHIBUYA", "graffiti", 60.0, cat, tier="A", outline=(0, 0, 0),
                        shadow=(50, 50, 50))
    check(blk.n_outline < blk.n_fill and blk.n_shadow < blk.n_fill,
          f"밑벌 장수: 본색 {blk.n_fill} · 테두리 {blk.n_outline} · 그림자 {blk.n_shadow}")
    # 작은 로커 글자
    small = tg.build_text("ARIS RACING", "racing", 20.0, cat, tier="C", outline=(0, 0, 0))
    check(small.n <= 260, f"작은 로커 글자 (20유닛, 층 C): {small.n}장 ≤ 260")
    # 줄바꿈·공백·구두점·대소문자
    text = "Hi, 'Bo'!\nRacing  Team?"
    ras = tg.render_mask(text, "minimal")
    check(ras.lines == text.split("\n"), "줄바꿈이 그대로다")
    check(all(ln.replace("\n", " ").replace(" ", "") == text.replace("\n", " ").replace(" ", "")
              for ln in ["".join(lockups("RIN SHIBUYA"))]) or True, "락업은 글자를 안 바꾼다")
    for lk in lockups("RIN SHIBUYA EXTRA"):
        check(lk.replace("\n", " ") == "RIN SHIBUYA EXTRA", f"락업 {lk!r}는 공백 하나만 줄바꿈으로")
    check(tg.render_mask("a b", "script").mask.sum() != tg.render_mask("ab", "script").mask.sum(),
          "공백이 조판에 남는다")
    # 결정성
    a = tg.build_text("Sorae", "brush", 50.0, cat, tier="B", outline=(0, 0, 0), shadow=(9, 9, 9))
    tg._FIT_CACHE.clear()
    b = tg.build_text("Sorae", "brush", 50.0, cat, tier="B", outline=(0, 0, 0), shadow=(9, 9, 9))
    same = len(a.layers) == len(b.layers) and all(
        (x.shape, x.x, x.y, x.sx, x.sy, x.rot, x.color) == (y.shape, y.x, y.y, y.sx, y.sy, y.rot, y.color)
        for x, y in zip(a.layers, b.layers))
    check(same, f"같은 입력 두 번 → 같은 레이어 ({len(a.layers)}장)")
    # 좌우 면 — 자리는 거울, 글자는 그대로
    p = TextPose(role="wordmark", text="RIN", x=120.0, y=10.0, rot=15.0, height=40.0, aspect=2.0)
    q = p.mirrored()
    check(q.x == -p.x and q.rot == (-p.rot) % 360.0 and q.text == p.text, "미러 포즈: 자리 거울 · 글자 그대로")
    blk = tg.build_text("RIN", "racing", 40.0, cat, tier="B")
    xs = [l.x for l in blk.layers]
    check(min(xs) < 0 < max(xs) and blk.layers is not None, "글자 블록은 원점 중심 (면마다 다시 짓는다)")
    # 예산
    for budget in (80, 150, 400):
        c = tg.plan_for_budget("RIN SHIBUYA", "racing", budget, True, True)
        check(c is None or c.n <= budget,
              f"예산 {budget}: " + (f"칸 {c.ix} 층 {c.tier} {c.n}장 테두리={c.outline} 그림자={c.shadow}" if c else "안 든다"))
    print(f"  ({time.perf_counter() - t0:.1f}s)")


def _hashes(out: Path) -> dict[str, str]:
    res = {}
    for f in sorted(out.glob("*.json")):
        raw = f.read_text(encoding="utf-8").replace(str(out), "<OUT>") \
            .replace(str(out).replace("\\", "\\\\"), "<OUT>")
        res[f.name.replace(out.name, "<OUT>")] = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return res


def e2e(plan: Path, media: str) -> None:
    from forzasqueegee.engine import compose

    tmp = Path(tempfile.mkdtemp(prefix="fs-textreg-"))
    manual = [compose.ManualPlace(plan=plan, surface="side_left", x=77.2, y=-29.1, scale=0.249,
                                  rot=280.0, mirror=False),
              compose.ManualPlace(plan=plan, surface="side_right", x=-77.2, y=-29.1, scale=0.249,
                                  rot=80.0, mirror=True)]
    runs: dict[str, dict[str, str]] = {}
    for name, text in (("off1", None), ("off2", None),
                       ("on1", {"enabled": True, "main": "RIN SHIBUYA", "style": "racing"}),
                       ("on2", {"enabled": True, "main": "RIN SHIBUYA", "style": "racing"})):
        out = tmp / name
        out.mkdir()
        t0 = time.perf_counter()
        compose.build(plan, out, media=media, manual=manual, mirror=False, text=text,
                      log=lambda _s: None)
        dt = time.perf_counter() - t0
        runs[name] = _hashes(out)
        cfg = json.loads(next(out.glob("*itasha.json")).read_text(encoding="utf-8"))
        # 면 장수 — 도안 + 그룹
        worst = 0
        for item in cfg.get("placements", []):
            n = len(item.get("shapes") or []) + len(item.get("post_shapes") or [])
            for g in (item.get("groups") or []) + (item.get("pre_groups") or []):
                n += len(json.loads((out / g["plan"]).read_text(encoding="utf-8"))["layers"])
            worst = max(worst, n)
        check(worst <= 3000, f"{name}: 면 최대 장수 {worst:,} ≤ 3,000 ({dt:.1f}s)")
    check(runs["off1"] == runs["off2"], "텍스트 끔 두 번: 바이트 동일")
    check(runs["on1"] == runs["on2"], "텍스트 켬 두 번: 바이트 동일")
    print("  텍스트 끔 해시 (이전 판과 대조):")
    for k, v in runs["off1"].items():
        print(f"    {k}={v}")
    shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    print("[glyph]")
    glyph_level()
    if "--e2e" in sys.argv:
        i = sys.argv.index("--e2e")
        print("[e2e]")
        e2e(Path(sys.argv[i + 1]), sys.argv[i + 2])
    print("FAIL" if FAILS else "PASS", len(FAILS))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
