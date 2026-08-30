"""다른 면의 글자 — 자리를 **못 박았을 때**(rear · hood · roof · window)의 배치.

옆면 글자는 후보 루프 안에서 필드가 자리를 정한다 (`textlayout`). 사람이
자리를 리어·후드·지붕·유리로 못 박으면 그 면에는 인물이 없으므로 (또는
후드 인물뿐) 필드가 없다 — 면 도색 상자에 **가운데 정렬**로 앉히고, 층은 그
면의 남은 장수가 정한다. 그룹 좌표는 면 유닛 그대로다 (스케일 1/group_unit).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from ...game import surface as gsurf
from ...i18n import msg
from ..catalog import Catalog
from ..model import LayerPlan
from .. import textglyph as tg
from .boxes import _rel
from .place import ROLE_EXTRA, ROLE_REAR, _refit_canvas, drawable
from .roof import ROOF_DARK, hood_index, top_segments
from .textbudget import plan_tiers
from .textbuild import pose_layers
from .textlayout import TextPose
from .textstyle import choose_style


# 면 상자에서 글자가 차지할 수 있는 몫 (높이 · 폭)
FACE_H_FRAC = 0.50


FACE_W_FRAC = 0.86


def _target_faces(placement: str) -> list[str]:
    return {"rear": [ROLE_REAR], "hood": [ROLE_EXTRA], "roof": [ROLE_EXTRA],
            "window": ["window_left", "window_right"]}.get(placement, [])


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
    box = sm.fit(aspect, coverage=0.85, anchor="center") or sm.paint
    return box, 0.0


def face_text(spec, design, items: list[dict], maps: dict, rigs: dict, cat: Catalog,
              out_dir: Path, plan: LayerPlan, *, group_unit: float, hood_u: float | None,
              notes: list[str], written: list[Path]) -> dict | None:
    """못 박은 면에 글자 그룹(또는 게임 글자 명세)을 더한다.

    되돌림: 구성 기록용 요약 (`itasha.json`의 `design.text`) — 없으면 None."""
    style = design.text_style or choose_style(spec.style, design.family, None) \
        if spec.style != "auto" else (design.text_style or "minimal")
    if style == "game":
        style = "minimal"
    ras = tg.render_mask(spec.main, style)
    aspect, hratio = ras.aspect, ras.hratio
    done: list[str] = []
    summary: dict = {}
    for name in _target_faces(spec.placement):
        sm = drawable(name, maps, rigs)
        if sm is None or (sm.uncertain and name != ROLE_EXTRA):
            continue
        box, rot = _box_for(spec.placement, sm, hood_u, aspect)
        if box is None:
            continue
        bw, bh = box[2] - box[0], box[3] - box[1]
        if spec.placement in ("hood", "roof"):
            bw, bh = bh, bw                        # 글자가 v를 따라 달린다
        h = max(4.0, min(FACE_H_FRAC * bh, FACE_W_FRAC * bw / max(0.5, aspect)) / hratio)
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
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        pose = TextPose(role="face", text=spec.main, x=0.0, y=0.0, rot=0.0, height=h,
                        aspect=aspect, hratio=hratio, on_bed=(spec.placement == "roof"))
        pal = design.pal
        if spec.placement == "roof":
            pal = replace(pal, bed=ROOF_DARK)      # 블랙아웃 위 — 판 위 색 규칙
        layers, job = pose_layers(pose, pal, cat, style=style, plan=tplan)
        if tplan.tier_main == "D" and job is not None:
            item.setdefault("text", [])
            item["text"] = list(item["text"]) + [{
                "text": job["text"], "font": job["font"],
                "center": [round(cx, 1), round(cy, 1)], "height": round(h, 1),
                "rot": round(rot, 1), "color": job["color"],
                **({"outline": job["outline"]} if job.get("outline") else {})}]
            done.append(f"{name}=D")
            summary = {"style": style, "tier": "D", "role": "face", "layers": 0, "faces": [name]}
            continue
        tp = LayerPlan(source_image=plan.source_image, image_size=plan.image_size,
                       units_per_px=plan.units_per_px,
                       layers=[replace(l, x=round(l.x, 4), y=round(l.y, 4),
                                       sx=round(l.sx, 4), sy=round(l.sy, 4),
                                       rot=round(l.rot % 360.0, 4)) for l in layers])
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
