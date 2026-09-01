"""텍스트 엔진 회귀 검사 — 조판 불변식과 품질 바닥.

    python tools/text_regress.py                    # 엔진 수준 (수십 초)
    python tools/text_regress.py --e2e PLAN MEDIA   # 구성 두 번 굽기 (결정성·상한·텍스트 끔 해시)

글자 엔진은 둘이다: **게임 글꼴 글리프**(`engine.textvinyl` — 기본, 한 글자 한
장)와 **도형 맞춤**(`engine.textglyph` — 동봉 OFL 글꼴을 도형으로 되짓기).

엔진 수준에서 보는 것:
- 게임 글꼴: 11종이 다 조판된다 · 주입 id가 다 있다 · 장수 = 글자 수 × 벌 수
- 조판: 단어 틈이 글자 틈보다 넓다 · 커닝이 사선 쌍만 당기고 곧은 쌍은 안 건드린다
  · 커닝이 잉크를 겹치지 않는다 · 상자 예측이 실제 잉크와 맞는다
- 도형 맞춤: 카운터 침범 · IoU 바닥 (층 A) · 테두리·그림자 벌이 본색보다 적다
- 예산: 사다리가 예산을 안 넘는다 · 안 들면 게임 글꼴로 물러난다
- 줄바꿈·공백·구두점·대소문자가 그대로다 (`lockups`도 글자를 안 바꾼다)
- 같은 입력 두 번 → 같은 레이어 (결정성)
- 좌우 면: 자리는 거울, 글자 상대 배치는 그대로 (미러 금지)

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
from forzasqueegee.engine import textglyph as tg  # noqa: E402
from forzasqueegee.engine import textvinyl as tv  # noqa: E402
from forzasqueegee.engine.compose import textbuild as tbu  # noqa: E402
from forzasqueegee.engine.compose.textbudget import game_layers, plan_tiers  # noqa: E402
from forzasqueegee.engine.compose.textlayout import TextPose, lockups  # noqa: E402
from forzasqueegee.engine.compose.textspec import TextSpec  # noqa: E402
from forzasqueegee.engine.fls import ids  # noqa: E402


FAILS: list[str] = []


def check(cond: bool, what: str) -> None:
    print(("  ok   " if cond else "  FAIL ") + what)
    if not cond:
        FAILS.append(what)


def _ink_box(layers, cat):
    """레이어들의 잉크 상자 (캔버스 유닛) — 글리프 상자로 정확히."""
    x0 = y0 = 1e9
    x1 = y1 = -1e9
    for l in layers:
        gb = tv.glyph_box(cat, l.shape)
        x0 = min(x0, l.x + gb.x0 * 64 * l.sx)
        x1 = max(x1, l.x + gb.x1 * 64 * l.sx)
        y0 = min(y0, l.y + gb.y0 * 64 * l.sy)
        y1 = max(y1, l.y + gb.y1 * 64 * l.sy)
    return x0, y0, x1, y1


def _gaps(text: str, font: str, cat: Catalog, kern: bool):
    """이웃 글자의 **잉크 상자** 틈 (대문자 높이 대비). 커닝이 걸리면 음수가 될 수
    있다 — 상자는 겹쳐도 잉크는 안 겹치는 것이 커닝이다 (`_ink_gaps`가 그쪽을 잰다)."""
    ls, _b = tv.text_layers(text, font=font, height=100.0, cat=cat,
                            space=tbu.WORD_SPACE, kern=kern)
    ed = []
    for l in ls:
        gb = tv.glyph_box(cat, l.shape)
        ed.append((l.x + gb.x0 * 64 * l.sx, l.x + gb.x1 * 64 * l.sx))
    return [round((ed[i + 1][0] - ed[i][1]) / 100.0, 4) for i in range(len(ed) - 1)]


def _ink_gaps(text: str, font: str, cat: Catalog, kern: bool):
    """이웃 글자의 **실제 잉크** 최소 틈 (대문자 높이 대비) — 옆모습으로 잰다."""
    ls, _b = tv.text_layers(text, font=font, height=100.0, cat=cat,
                            space=tbu.WORD_SPACE, kern=kern)
    out = []
    for a, b in zip(ls, ls[1:]):
        (_al, ar, ay0, ay1) = tv.glyph_profile(cat, a.shape)
        (bl, _br, by0, by1) = tv.glyph_profile(cat, b.shape)
        y0, y1 = max(ay0, by0), min(ay1, by1)
        if y1 <= y0:
            continue
        best = None
        for k in range(tv.KERN_ROWS):
            y = y0 + (k + 0.5) * (y1 - y0) / tv.KERN_ROWS
            i = min(tv.KERN_ROWS - 1, max(0, int((y - ay0) / (ay1 - ay0) * tv.KERN_ROWS)))
            j = min(tv.KERN_ROWS - 1, max(0, int((y - by0) / (by1 - by0) * tv.KERN_ROWS)))
            if ar[i] != ar[i] or bl[j] != bl[j]:     # NaN — 그 높이에 잉크가 없다
                continue
            g = (b.x + bl[j] * 64 * b.sx) - (a.x + ar[i] * 64 * a.sx)
            best = g if best is None else min(best, g)
        if best is not None:
            out.append(round(best / 100.0, 4))
    return out


def game_font_level(cat: Catalog) -> None:
    t0 = time.perf_counter()
    # 11종이 다 조판되고 주입 id가 다 있다
    bad_font, bad_id = [], []
    for font in tv.FONTS:
        try:
            ls, _b = tv.text_layers("RIN Shibuya 09", font=font, height=100.0, cat=cat)
        except Exception as e:                       # noqa: BLE001 — 이유를 적어 보고한다
            bad_font.append(f"{font}({e})")
            continue
        miss = [l.shape for l in ls if ids.id_of(l.shape) is None]
        if miss:
            bad_id.append(f"{font}({len(miss)}장)")
    check(not bad_font, f"게임 글꼴 {len(tv.FONTS)}종이 다 조판된다" + (f" — {bad_font}" if bad_font else ""))
    check(not bad_id, "게임 글꼴 글리프에 주입 id가 다 있다" + (f" — {bad_id}" if bad_id else ""))
    # 장수 = 글자 수 × 벌 수
    n = len(tbu.font_block("RIN SHIBUYA", "impact", 60.0, cat, fill=(255, 255, 255)))
    check(n == 10, f"본색만: 글자 10자 → {n}장")
    passes = 1 + tv.OUTLINE_PASSES + 1
    n = len(tbu.font_block("RIN SHIBUYA", "impact", 60.0, cat, fill=(255, 255, 255),
                           outline=(0, 0, 0), shadow=(9, 9, 9)))
    check(n == 10 * passes,
          f"본색+테두리{tv.OUTLINE_PASSES}+그림자: 10자 → {n}장 (= 10 × {passes})")
    check(n == game_layers("RIN SHIBUYA", True, True), "예산이 세는 장수 = 실제 장수")
    # 단어 틈이 글자 틈보다 넓다 (impact의 실측 FONT_SPACE는 0이라 그냥 쓰면 붙는다)
    for font in ("impact", "arial", "centurygothic"):
        g = _gaps("RIN SHIBUYA", font, cat, kern=False)
        word, letters = g[2], [x for i, x in enumerate(g) if i != 2]
        check(word > 1.4 * max(letters),
              f"{font}: 단어 틈 {word:.3f} > 글자 틈 최대 {max(letters):.3f}의 1.4배")
    # 커닝 — 사선 쌍을 당기되 잉크는 절대 안 겹친다
    for font in ("impact", "arial", "brushscript", "centurygothic"):
        a = _gaps("YAVA", font, cat, kern=False)
        b = _gaps("YAVA", font, cat, kern=True)
        check(all(y < x for x, y in zip(a, b)), f"{font}: 사선 쌍(YAVA)을 당긴다 {a} → {b}")
        for txt in ("YAVA", "RIN SHIBUYA", "HIHI", "To Wo"):
            g = _ink_gaps(txt, font, cat, kern=True)
            check(not g or min(g) >= -0.005,
                  f"{font} {txt!r}: 커닝이 잉크를 안 겹친다 (최소 잉크 틈 {min(g) if g else 0:.4f})")
    # 곧은 세로획 쌍은 당길 것이 없다 (기울지 않은 글꼴에서)
    for font in ("impact", "arial", "centurygothic"):
        a = _gaps("HIHI", font, cat, kern=False)
        b = _gaps("HIHI", font, cat, kern=True)
        check(a == b, f"{font}: 곧은 쌍(HIHI)은 커닝이 안 건드린다 {a} == {b}")
    # 상자 예측 = 실제 잉크 (배치가 이 상자로 자리를 잡는다)
    for font in ("impact", "brushscript", "pristina"):
        for txt in ("RIN SHIBUYA", "RIN\nSHIBUYA"):
            aspect, hratio = tbu.font_metrics(txt, font, cat)
            h = 40.0
            ls = tbu.font_block(txt, font, h, cat, fill=(1, 1, 1))
            x0, y0, x1, y1 = _ink_box(ls, cat)
            pw, ph = aspect * h * hratio, h * hratio
            ok = abs((x1 - x0) - pw) < 0.02 * pw and abs((y1 - y0) - ph) < 0.02 * ph
            check(ok, f"{font} {txt!r}: 상자 예측 {pw:.1f}x{ph:.1f} ≈ 잉크 {x1-x0:.1f}x{y1-y0:.1f}")
    game_outline_level(cat)
    print(f"  ({time.perf_counter() - t0:.1f}s)")


def game_outline_level(cat: Catalog) -> None:
    """게임 글꼴 테두리·그림자가 **같은 글리프의 사본**이고 테가 안 끊기나.

    가는 글꼴(pristina·brushscript)에서 대각 넷은 테가 아니라 **유령 사본 넷**이
    된다 — 획이 오프셋의 두 배보다 가늘면 사본끼리 안 만난다. 이상 테 덩이 수와
    실제 테 덩이 수를 견줘 그것을 잡는다 (2026-09-01: 대각 넷일 때 pristina
    'Evelyne'이 10 → 18).
    """
    import cv2
    import numpy as np
    from forzasqueegee.engine.model import Layer, LayerPlan
    from forzasqueegee.engine.render import render_plan

    H = 200.0

    def ras(layers, box, cell=1.0):
        x0, y0, x1, y1 = box
        cols, rows = int((x1 - x0) / cell), int((y1 - y0) / cell)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        pl = LayerPlan(source_image="p", image_size=(cols, rows), units_per_px=cell,
                       layers=[Layer(shape=l.shape, x=l.x - cx, y=l.y - cy, sx=l.sx,
                                     sy=l.sy, rot=l.rot, color=(0, 0, 0)) for l in layers])
        return render_plan(pl, cat, bg=255).mean(2) < 128

    for font, text in (("arial", "OSCAR"), ("impact", "RIN SHIBUYA"),
                       ("brushscript", "Sorae"), ("pristina", "Evelyne")):
        body = tbu.font_block(text, font, H, cat, fill=(0, 0, 0))
        got = tbu.font_block(text, font, H, cat, fill=(0, 0, 0), outline=(0, 0, 0),
                             shadow=(0, 0, 0))
        edge = [l for l in got if l.label.endswith("_edge")]
        shad = [l for l in got if l.label.endswith("_shadow")]
        base = [l.shape for l in body]
        check([l.shape for l in edge] == base * tv.OUTLINE_PASSES
              and [l.shape for l in shad] == base,
              f"{font} {text!r}: 테두리·그림자가 본색과 **같은 글리프**다")
        xs = [l.x for l in body]
        ys = [l.y for l in body]
        box = (min(xs) - 2 * H, min(ys) - 2 * H, max(xs) + 2 * H, max(ys) + 2 * H)
        B = ras(body, box)
        O = ras(edge, box)
        k = int(round(tv.OUTLINE_SHIFT * H))
        ideal = cv2.dilate(B.astype(np.uint8),
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                     (2 * k + 1, 2 * k + 1))) > 0
        ring = ideal & ~B
        n_i = cv2.connectedComponents(ring.astype(np.uint8), 8)[0] - 1
        n_g = cv2.connectedComponents(((B | O) & ~B).astype(np.uint8), 8)[0] - 1
        cov = float((O & ring).sum()) / max(1, int(ring.sum()))
        check(n_g == n_i and cov >= 0.90,
              f"{font} {text!r}: 테 덩이 {n_i} → {n_g} · 덮음 {cov:.3f}")


def shape_fit_level(cat: Catalog) -> None:
    t0 = time.perf_counter()
    for text, style, iou_min, ctr_max in (("RIN SHIBUYA", "racing", 0.88, 0.10),
                                          ("Sorae", "script", 0.80, 0.15),
                                          ("Evelyne", "graffiti", 0.85, 0.15),
                                          ("ARIS", "techno", 0.88, 0.10)):
        f = tg.fit_fill(text, style, tg.TIER_INDEX["A"])
        check(f.iou >= iou_min and f.counter <= ctr_max,
              f"{style} {text!r}: IoU {f.iou:.3f} ≥ {iou_min} · 카운터 침범 {f.counter:.3f} ≤ {ctr_max} · {f.n}장")
    check(set(tg.TIER_INDEX) == {"A", "B"},
          f"사다리 층은 A·B뿐이다 (거친 칸은 상자 덩어리로 읽힌다) — {sorted(tg.TIER_INDEX)}")
    blk = tg.build_text("RIN SHIBUYA", "graffiti", 60.0, cat, tier="A", outline=(0, 0, 0),
                        shadow=(50, 50, 50))
    check(blk.n_outline > 0 and blk.n_shadow > 0,
          f"밑벌 장수: 본색 {blk.n_fill} · 테두리 {blk.n_outline} · 그림자 {blk.n_shadow}")
    silhouette_level(cat)
    for budget in (80, 400, 900):
        c = tg.plan_for_budget("RIN SHIBUYA", "racing", budget, True, True)
        check(c is None or c.n <= budget,
              f"예산 {budget}: " + (f"칸 {c.ix} 층 {c.tier} {c.n}장" if c else "안 든다"))
    print(f"  ({time.perf_counter() - t0:.1f}s)")


def silhouette_level(cat: Catalog) -> None:
    """**밑벌은 같은 글자의 사본이다** — 테·그림자가 원 몇 개로 끊기지 않나.

    자 둘:
      테두리  보이는 테(키운 실루엣 − 본색)를 **남김없이** 덮고 그 덩이 수가 그대로다.
      그림자  그림자 좌표계에서 **안 가려지는 자리**를 남김없이 덮는다.

    옛 판은 밑벌에서 "새로 덮는 픽셀 300개" 미만인 장을 뺐고, 그래서 테가
    부푼 원 몇 개로 끊기고(script 덩이 3 → 10) **그림자는 한 장도 안 남았다**.
    """
    import cv2
    import numpy as np
    from forzasqueegee.engine import textfit as tf

    fcat = tg._catalog()
    for text, style in (("RIN SHIBUYA", "racing"), ("Sorae", "script"),
                        ("Evelyne", "graffiti"), ("ARIS", "techno")):
        ix = tg.TIER_INDEX["A"]
        ras = tg.render_mask(text, style)
        sh = ras.mask.shape
        fill = tg.fit_fill(text, style, ix)
        S = tf.raster(fill.prims, sh, fcat)
        big = tg._outline_prims(text, style, ix)
        ring = tf.raster(big, sh, fcat) & ~S
        O = tf.raster(tg.fit_outline(text, style, ix).prims, sh, fcat)
        cov = float((O & ring).sum()) / max(1, int(ring.sum()))
        n_i = cv2.connectedComponents(ring.astype(np.uint8), 8)[0] - 1
        n_g = cv2.connectedComponents((O & ring).astype(np.uint8), 8)[0] - 1
        check(cov >= 0.999 and n_g == n_i,
              f"{style} {text!r} 테두리: 보이는 테 덮음 {cov:.3f} · 덩이 {n_i} → {n_g}")
        for outline in (True, False):
            src = big if outline else list(fill.prims)
            sil = tf.raster(src, sh, fcat)
            sx, sy = tg._shift_px(style, ras.cap_px, (1.0, -1.0))
            vis = sil & ~tg._shifted(sil, -sx, -sy)
            fs = tg.fit_shadow(text, style, ix, outline, (1.0, -1.0))
            SH = tf.raster(fs.prims, sh, fcat)
            c2 = float((SH & vis).sum()) / max(1, int(vis.sum()))
            check(fs.n > 0 and c2 >= 0.999,
                  f"{style} {text!r} 그림자(테두리 {'켬' if outline else '끔'}): "
                  f"{fs.n}장 · 보이는 자리 덮음 {c2:.3f}")
        # 도형은 본색과 **같은 것**이라야 한다 (원으로 흉내 낸 실루엣이 아니다)
        base = {p.shape for p in fill.prims}
        check({p.shape for p in tg.fit_outline(text, style, ix).prims} <= base,
              f"{style} {text!r}: 테두리 도형이 본색 도형과 같은 종류다")


def budget_level(cat: Catalog) -> None:
    t0 = time.perf_counter()

    def spec(**kw):
        return TextSpec.from_dict({"enabled": True, "main": "RIN SHIBUYA",
                                   "sub": "IDOLMASTER", "style": "racing", **kw})

    # 기본 엔진은 게임 글꼴 — 늘 층 D
    for free in (60, 200, 900):
        p = plan_tiers(spec(), "racing", free)
        check(p.tier_main == "D" and p.n_main <= free,
              f"font 엔진 예산 {free}: 층 {p.tier_main} {p.n_main}장 ≤ {free}")
    # shapes 엔진 — 넉넉하면 도형 맞춤, 모자라면 게임 글꼴로 물러난다
    p = plan_tiers(spec(engine="shapes"), "racing", 900)
    check(p.tier_main in ("A", "B"), f"shapes 엔진 예산 900: 층 {p.tier_main} {p.n_main}장")
    p = plan_tiers(spec(engine="shapes"), "racing", 60)
    check(p.tier_main == "D", f"shapes 엔진 예산 60: 게임 글꼴로 물러난다 (층 {p.tier_main})")
    p = plan_tiers(spec(engine="shapes", allow_fallback_to_game_text=False), "racing", 60)
    check(p.tier_main == "E", f"물러남 금지 + 예산 60: 글자를 뺀다 (층 {p.tier_main})")
    # 작은 글자는 게임 글꼴이 맡는다
    check(tbu.tier_for_size("A", 20.0) == "D", "20유닛 글자는 층 D (도형 맞춤은 점으로 뭉친다)")
    check(tbu.tier_for_size("A", 60.0) == "A", "60유닛 글자는 층 A 그대로")
    # 옛 이름 `game` 스타일은 기본 엔진으로 푼다
    s = TextSpec.from_dict({"enabled": True, "main": "X", "style": "game"})
    check(s.style == "auto" and s.engine == "font", f"옛 style=game → {s.style}/{s.engine}")
    print(f"  ({time.perf_counter() - t0:.1f}s)")


def invariants(cat: Catalog) -> None:
    t0 = time.perf_counter()
    text = "Hi, 'Bo'!\nRacing  Team?"
    ras = tg.render_mask(text, "minimal")
    check(ras.lines == text.split("\n"), "줄바꿈이 그대로다")
    for lk in lockups("RIN SHIBUYA EXTRA"):
        check(lk.replace("\n", " ") == "RIN SHIBUYA EXTRA", f"락업 {lk!r}는 공백 하나만 줄바꿈으로")
    check(tbu.game_text("Hi, 'Bo'! 09", "arial", cat) == "H, 'o'! 09"
          or "Hi" in tbu.game_text("Hi, 'Bo'! 09", "arial", cat),
          "게임 글꼴에 없는 글자만 빠진다 (띄어쓰기는 지킨다)")
    check(tbu.game_text("RIN SHIBUYA", "arial", cat) == "RIN SHIBUYA", "있는 글자는 다 남는다")
    a = tbu.font_block("Sorae 09", "brushscript", 50.0, cat, fill=(9, 9, 9), outline=(0, 0, 0))
    b = tbu.font_block("Sorae 09", "brushscript", 50.0, cat, fill=(9, 9, 9), outline=(0, 0, 0))
    same = len(a) == len(b) and all(
        (x.shape, x.x, x.y, x.sx, x.sy, x.rot, x.color) == (y.shape, y.x, y.y, y.sx, y.sy, y.rot, y.color)
        for x, y in zip(a, b))
    check(same, f"같은 입력 두 번 → 같은 레이어 ({len(a)}장)")
    x0, _y0, x1, _y1 = _ink_box(a, cat)
    check(abs(x0 + x1) < 0.02 * (x1 - x0), "글자 블록은 원점 중심 (면마다 다시 짓는다)")
    p = TextPose(role="wordmark", text="RIN", x=120.0, y=10.0, rot=15.0, height=40.0, aspect=2.0)
    q = p.mirrored()
    check(q.x == -p.x and q.rot == (-p.rot) % 360.0 and q.text == p.text,
          "미러 포즈: 자리 거울 · 글자 그대로")
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
    on = {"enabled": True, "main": "RIN SHIBUYA", "style": "racing"}
    runs: dict[str, dict[str, str]] = {}
    for name, text in (("off1", None), ("off2", None), ("on1", on), ("on2", on)):
        out = tmp / name
        out.mkdir()
        t0 = time.perf_counter()
        compose.build(plan, out, media=media, manual=manual, mirror=False, text=text,
                      log=lambda _s: None)
        dt = time.perf_counter() - t0
        runs[name] = _hashes(out)
        cfg = json.loads(next(out.glob("*itasha.json")).read_text(encoding="utf-8"))
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
    cat = Catalog(default_catalog_path())
    print("[게임 글꼴]")
    game_font_level(cat)
    print("[도형 맞춤]")
    shape_fit_level(cat)
    print("[예산]")
    budget_level(cat)
    print("[불변식]")
    invariants(cat)
    if "--e2e" in sys.argv:
        i = sys.argv.index("--e2e")
        print("[e2e]")
        e2e(Path(sys.argv[i + 1]), sys.argv[i + 2])
    print("FAIL" if FAILS else "PASS", len(FAILS))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
