"""다른 면의 글자 — 자리를 **못 박았을 때**(rear · hood · roof · window)의 배치.

옆면 글자는 후보 루프 안에서 필드가 자리를 정한다 (`textlayout`). 사람이
자리를 리어·후드·지붕·유리로 못 박으면 그 면에는 인물이 없으므로 (또는
후드 인물뿐) 필드가 없다 — 면 도색 상자에 **가운데 정렬**로 앉히고, 층은 그
면의 남은 장수가 정한다. 글자는 그 면의 그룹 한 장(`text-<면>.json`)이다 —
게임 글꼴 글리프든 도형 맞춤이든 레이어라 같은 그룹에 실린다. 그룹 좌표는 면
유닛 그대로다 (스케일 1/group_unit).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from ...game import surface as gsurf
from ...i18n import msg
from ..catalog import Catalog
from ..model import UNITS_PER_SCALE, LayerPlan, rnd
from . import sponsor
from .boxes import _rel
from .place import ROLE_EXTRA, ROLE_REAR, _refit_canvas, drawable
from .roof import ROOF_DARK, hood_index, top_segments
from .stack import EDGE_SHAPES, _native_half
from .textbudget import plan_tiers
from .textbuild import pose_layers, text_box
from .textlayout import TextPose
from .textstyle import choose_style


# 면 상자에서 글자가 차지할 수 있는 몫 (높이 · 폭)
FACE_H_FRAC = 0.50


FACE_W_FRAC = 0.86


def _target_faces(placement: str) -> list[str]:
    return {"rear": [ROLE_REAR], "hood": [ROLE_EXTRA], "roof": [ROLE_EXTRA],
            "window": ["window_left", "window_right"]}.get(placement, [])


pinned_faces = _target_faces


def _group_used(item: dict, out_dir: Path) -> int:
    """면 항목이 이미 쥔 장수 — 도형 + 그룹 파일들."""
    return (len(item.get("shapes") or []) + len(item.get("post_shapes") or [])
            + sum(len(LayerPlan.load(out_dir / g["plan"]).layers)
                  for g in (item.get("groups") or []) + (item.get("pre_groups") or [])))


def _box_for(placement: str, sm: gsurf.SurfaceMap, hood_u: float | None, aspect: float
             ) -> tuple[tuple[float, float, float, float] | None, float]:
    """(글자 상자, 회전) — 윗면은 후드/지붕 구간, 나머지는 내접 상자."""
    if placement in ("hood", "roof"):
        segs = top_segments(sm)
        if not segs:
            return None, 0.0
        hi = hood_index(segs, hood_u)
        idx = hi if placement == "hood" else min(len(segs) - 1, hi + 1)
        if placement == "roof" and idx == hi:
            return None, 0.0
        box = segs[idx]
        # 글자의 위(캔버스 +y)는 **차 뒤쪽**(윈드실드)이다 — 앞에서 보는 사람이
        # 바로 읽는다 (`build._hood_place`와 같은 규약). 뒤 방향은 야코비안 부호.
        assert sm.warp is not None
        bcx, bcy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        rdir = 1.0 if sm.warp.jac(bcx, bcy)[0, 0] > 0 else -1.0
        return box, (-rdir * 90.0) % 360.0
    box = sm.fit(aspect, coverage=0.99, anchor="center") or sm.paint
    return box, 0.0


def face_text(spec, design, items: list[dict], maps: dict, rigs: dict, cat: Catalog,
              out_dir: Path, plan: LayerPlan, *, group_unit: float, hood_u: float | None,
              notes: list[str], written: list[Path]) -> dict | None:
    """못 박은 면에 글자 그룹을 더한다.

    되돌림: 구성 기록용 요약 (`itasha.json`의 `design.text`) — 없으면 None."""
    style = design.text_style or choose_style(spec.style, design.family, None) \
        if spec.style != "auto" else (design.text_style or "minimal")
    done: list[str] = []
    summary: dict = {}
    for name in _target_faces(spec.placement):
        sm = drawable(name, maps, rigs)
        if sm is None or (sm.uncertain and name != ROLE_EXTRA):
            continue
        item = next((it for it in items if it["surface"] == name), None)
        if item is None:
            item = {"surface": name, "fit": False}
            items.append(item)
        used = (len(item.get("shapes") or []) + len(item.get("post_shapes") or [])
                + sum(int(LayerPlan.load(out_dir / g["plan"]).layers.__len__())
                      for g in (item.get("groups") or []) + (item.get("pre_groups") or [])))
        free = (sm.cap or 1000) - used - 8
        tplan = plan_tiers(spec, style, max(0, free))
        notes += tplan.notes
        if tplan.tier_main == "E":
            continue
        # 상자 비율은 **그 층이 실제로 그릴 글꼴**의 것이다 (게임 글꼴과 OFL 글꼴의
        # 폭이 다르다 — 남의 비율로 재면 글자가 자리보다 넓거나 좁게 선다)
        aspect, hratio = text_box(spec.main, style, tplan, cat)
        box, rot = _box_for(spec.placement, sm, hood_u, aspect)
        if box is None:
            continue
        bw, bh = box[2] - box[0], box[3] - box[1]
        if spec.placement in ("hood", "roof"):
            bw, bh = bh, bw                        # 글자가 v를 따라 달린다
        h = max(4.0, min(FACE_H_FRAC * bh, FACE_W_FRAC * bw / max(0.5, aspect)) / hratio)
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        pose = TextPose(role="face", text=spec.main, x=0.0, y=0.0, rot=0.0, height=h,
                        aspect=aspect, hratio=hratio, on_bed=(spec.placement == "roof"))
        pal = design.pal
        if spec.placement == "roof":
            pal = replace(pal, bed=ROOF_DARK)      # 블랙아웃 위 — 판 위 색 규칙
        layers = pose_layers(pose, pal, cat, style=style, plan=tplan)
        if not layers:
            continue
        tp = LayerPlan(source_image=plan.source_image, image_size=plan.image_size,
                       units_per_px=plan.units_per_px,
                       layers=[replace(l, x=rnd(l.x, 4), y=rnd(l.y, 4),
                                       sx=rnd(l.sx, 4), sy=rnd(l.sy, 4),
                                       rot=rnd(l.rot % 360.0, 4)) for l in layers])
        tp = _refit_canvas(tp, cat)
        tpath = out_dir / f"text-{name}.json"
        tp.save(tpath)
        written.append(tpath)
        item.setdefault("groups", [])
        item["groups"] = list(item["groups"]) + [{
            "plan": _rel(tpath, out_dir), "x": round(cx, 1), "y": round(cy, 1),
            "scale": round(1.0 / max(1e-6, group_unit), 4), "rot": round(rot, 1),
            "mirror": False}]
        done.append(f"{name}={tplan.tier_main}({len(layers)})")
        summary = {"style": style, "tier": tplan.tier_main, "role": "face",
                   "layers": len(layers), "faces": [name]}
    if not done:
        return None
    notes.append(msg("면 글자 ({placement}, {style}): {faces}", placement=spec.placement,
                     style=style, faces=" · ".join(done)))
    summary["faces"] = [d.split("=")[0] for d in done]
    return summary


# ── 면 배정의 글자 (계획 5단계) ─────────────────────────────────────────
#
# 자동 자리(`auto`·`side`)일 때 **다른 면이 받는 글자**다 — 옆면 워드마크는 그대로
# 서고, 사람 판이 같은 이름을 되풀이하는 자리에 다시 앉힌다: 리어 워드마크(로고
# 줄 위) · 윈드실드 글자 띠 · 뒷유리 워드마크(+ 계열의 무늬 도형 날개) · 도어 유리의
# 작은 문구. 사람 판 실측 (`work/lab/humanref/liv`, 글리프 레이어 · 작가 7~13인):
# 이 면들의 글자는 전부 면 좌표에서 **바로 선다**(rot 0 — 윈드실드 132/132 ·
# 유리 138/138), 윈드실드 글자는 지붕 쪽 끝(사다리꼴의 좁은 끝)에 붙고, 뒷유리는
# 위쪽, 도어 유리는 위·가운데(레이어 폭의 0.65)다.


# 면 높이 대비 글자 블록 높이 상한 · 면(또는 내접 상자) 폭 대비 블록 폭 상한 — 역할마다.
ROLE_H = {"wordmark": 0.24, "glass_wordmark": 0.30, "strip": 0.14, "phrase": 0.13}
ROLE_W = {"wordmark": 0.80, "glass_wordmark": 0.84, "strip": 0.72, "phrase": 0.66}


# 자리 (면 높이 몫, 아래에서) — 리어 워드마크는 로고 줄(`sponsor.FACE_ROW_V`) 위,
# 도어 유리 문구는 위·가운데. 뒷유리는 내접 상자 가운데, 윈드실드는 지붕 끝.
REAR_V = 0.62
PHRASE_V = 0.62
STRIP_PAD = 0.07              # 지붕 끝에서의 여백 (면 높이 몫)


# 뒷유리 워드마크 날개 — 계열의 가장자리 무늬 도형(`stack.EDGE_SHAPES`) 둘을 글자
# 양옆에 점대칭으로 (도형 명세에는 뒤집기가 없다 — 180° 회전이 그 짝이다).
WING_K = 1.4                  # 날개 높이 = 글자 블록 높이 × 이것
WING_GAP = 0.35               # 글자와 날개 사이 (글자 블록 높이 몫)
WING_ROOM = 0.62              # 날개가 설 계열에서는 워드마크 폭 상한을 이만큼으로


ROLE_OF_FACE = {ROLE_REAR: "wordmark", "rear_window": "glass_wordmark",
                "windshield": "strip"}


def _roof_end(sm: gsurf.SurfaceMap) -> tuple[float, float]:
    """윈드실드의 **지붕 쪽 끝** v와 그 끝의 칠해진 폭 (면 유닛).

    유리는 지붕 쪽이 좁다 (`game.fold` — 축 표의 z 부호는 유리에서 못 믿고
    사다리꼴이 정한다). 마스크의 위·아래 1/8 띠의 칠해진 폭을 견준다."""
    m = sm.mask
    u0, v0, u1, v1 = sm.paint
    if m.size <= 1 or not m.any():
        return v1, u1 - u0
    mh, mw = m.shape
    band = max(1, mh // 8)
    top = float(m[:band].sum(axis=1).mean())          # 행 0 = v1
    bot = float(m[-band:].sum(axis=1).mean())
    upp = (u1 - u0) / mw
    return (v1, top * upp) if top <= bot else (v0, bot * upp)


def _wings(design, cat: Catalog, sm: gsurf.SurfaceMap, cx: float, cy: float,
           bw: float, bh: float) -> list[dict]:
    """워드마크 양옆의 날개 도형 명세 — 면을 나가거나 마스크 밖이면 빈 목록."""
    names = EDGE_SHAPES.get(design.family.name, ())
    if not names:
        return []
    shape = names[0]
    hx, hy = _native_half(cat, shape)
    u0, v0, u1, v1 = sm.paint
    pad = 0.04 * min(u1 - u0, v1 - v0)
    half_h = WING_K * bh / 2
    # 글자 양옆에 남은 자리에 맞춘다 — 워드마크가 면을 거의 다 쓰면 날개가 작아지고,
    # 절반 아래로 줄어야 들면 안 세운다 (티끌 날개는 없느니만 못하다)
    room = min(cx - u0, u1 - cx) - pad - bw / 2 - WING_GAP * bh
    half_w = half_h * hx / max(1e-6, hy)
    if room < 2 * half_w:
        k = max(0.0, room) / max(1e-6, 2 * half_w)
        if k < 0.5:
            return []
        half_h *= k
        half_w *= k
    s = half_h / (hy * UNITS_PER_SCALE)
    dx = bw / 2 + WING_GAP * bh + half_w
    if cy - half_h < v0 + pad or cy + half_h > v1 - pad:
        return []
    if not (sm.masked_at(cx - dx, cy) and sm.masked_at(cx + dx, cy)):
        return []
    rgb = [int(v) for v in design.pal.primary]
    return [{"shape": shape, "x": round(cx - dx, 1), "y": round(cy, 1),
             "sx": round(s, 4), "sy": round(s, 4), "rot": 0.0, "rgb": rgb},
            {"shape": shape, "x": round(cx + dx, 1), "y": round(cy, 1),
             "sx": round(s, 4), "sy": round(s, 4), "rot": 180.0, "rgb": rgb}]


def assigned_text(spec, design, items: list[dict], maps: dict, rigs: dict, cat: Catalog,
                  out_dir: Path, plan: LayerPlan, *, faces: list[str], group_unit: float,
                  notes: list[str], written: list[Path],
                  reserve: dict[str, float] | None = None) -> dict:
    """면 배정이 준 면들에 글자 그룹을 더한다 (`text-<면>.json`).

    `faces`는 부르는 쪽이 이미 거른 목록이다 (배정이 `sponsor`이고, 글자 자리를 못
    박은 면이 아니고, 사람 덩어리가 없는 면). `maps`는 **온전히 보이는 지도**
    (`place.usable`)다 — 글자 상자는 그 안에 내접한다 (coverage 0.99). `reserve`는
    면 → 내접 상자 아래쪽에서 **로고 줄에 남길 몫**(높이 비) — 리어·프론트의 노출
    띠는 좁아서(실비아 리어 24%) 워드마크가 다 차지하면 로고 줄이 못 선다.
    되돌림: 면 → {role, tier, layers, text}."""
    style = design.text_style or choose_style(spec.style, design.family, None)
    out: dict = {}
    done: list[str] = []
    for name in faces:
        sm = drawable(name, maps, rigs)
        if sm is None or sm.uncertain:
            continue
        item = next((it for it in items if it["surface"] == name), None)
        if item is None:
            item = {"surface": name, "fit": False}
            items.append(item)
        used = _group_used(item, out_dir)
        cap = sm.cap or 1000
        free = cap - used - 8
        role = ROLE_OF_FACE.get(name, "phrase")
        text = (spec.sub or spec.main) if role == "phrase" else spec.main
        if not text or not text.strip():
            continue
        one = replace(spec, main=text, sub=None)
        tplan = plan_tiers(one, style, max(0, free))
        if tplan.tier_main == "E":
            notes += tplan.notes
            continue
        aspect, hratio = text_box(text, style, tplan, cat)
        u0, v0, u1, v1 = sm.paint
        W, H = u1 - u0, v1 - v0
        if role == "strip":
            v_roof, w_roof = _roof_end(sm)
            wmax = ROLE_W[role] * (w_roof if w_roof > 0 else W)
            h = min(ROLE_H[role] * H, wmax / max(0.5, aspect)) / hratio
            bh = h * hratio
            cx = (u0 + u1) / 2
            cy = (v_roof - (STRIP_PAD * H + bh / 2) if v_roof == v1
                  else v_roof + (STRIP_PAD * H + bh / 2))
            box = sm.paint
        else:
            box = sm.fit(aspect, coverage=0.99, anchor="center") or sm.paint
            # 도어 유리처럼 마스크가 덩이 둘이면(B필러) 문구는 **큰 덩이** 안에 —
            # 내접 상자는 필러를 건너 두 창에 걸쳤다 (로고 줄과 같은 규칙, `sponsor._pane`).
            pane = sponsor._pane(sm) if role == "phrase" else None
            if pane is not None:
                box = (pane[0], box[1], pane[1], box[3])
            rv = float((reserve or {}).get(name, 0.0))
            if rv > 0.0:                        # 아래 몫은 로고 줄 (`sponsor.face_row`)
                box = (box[0], box[1] + rv * (box[3] - box[1]), box[2], box[3])
            bw_box = box[2] - box[0]
            wcap = ROLE_W[role]
            if role == "glass_wordmark" and design.family.name in EDGE_SHAPES:
                wcap *= WING_ROOM                 # 날개 둘의 자리를 남긴다
            h = min(ROLE_H[role] * H, wcap * bw_box / max(0.5, aspect)) / hratio
            bh = h * hratio
            cx = (box[0] + box[2]) / 2
            if role == "wordmark":
                cy = v0 + REAR_V * H
            elif role == "phrase":
                cy = v0 + PHRASE_V * H
            else:
                cy = (box[1] + box[3]) / 2
            cy = min(max(cy, box[1] + bh / 2), box[3] - bh / 2)
        if h < 4.0:
            continue
        pose = TextPose(role="face", text=text, x=0.0, y=0.0, rot=0.0, height=h,
                        aspect=aspect, hratio=hratio)
        layers = pose_layers(pose, design.pal, cat, style=style, plan=tplan)
        if not layers:
            continue
        tp = LayerPlan(source_image=plan.source_image, image_size=plan.image_size,
                       units_per_px=plan.units_per_px,
                       layers=[replace(l, x=rnd(l.x, 4), y=rnd(l.y, 4),
                                       sx=rnd(l.sx, 4), sy=rnd(l.sy, 4),
                                       rot=rnd(l.rot % 360.0, 4)) for l in layers])
        tp = _refit_canvas(tp, cat)
        tpath = out_dir / f"text-{name}.json"
        tp.save(tpath)
        written.append(tpath)
        item["groups"] = list(item.get("groups") or []) + [{
            "plan": _rel(tpath, out_dir), "x": round(cx, 1), "y": round(cy, 1),
            "scale": round(1.0 / max(1e-6, group_unit), 4), "rot": 0.0,
            "mirror": False}]
        n_wing = 0
        if role == "glass_wordmark":
            wings = _wings(design, cat, sm, cx, cy, bh * aspect, bh)
            if wings and used + len(layers) + len(wings) <= cap:
                item["post_shapes"] = list(item.get("post_shapes") or []) + wings
                n_wing = len(wings)
        out[name] = {"role": role, "tier": tplan.tier_main, "layers": len(layers),
                     "wings": n_wing, "text": text}
        done.append(f"{name}={role}({len(layers)}" + (f"+{n_wing}" if n_wing else "") + ")")
    if done:
        notes.append(msg("면 배정 글자 ({style}): {faces}", style=style, faces=" · ".join(done)))
    return out
