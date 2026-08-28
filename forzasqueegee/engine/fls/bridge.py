"""우리 산출물 ↔ FLS 사이의 다리.

세 가지를 잇는다:

1. **도안(plan.json) → FLS** — 편집기 프로젝트(`.3so`)와 게임 컨테이너 폴더
   (`LayerGroup_*/C_group`). 프로젝트에는 원화를 안내 레이어로 같이 심는다 —
   사람이 반투명 원화를 깔고 따라 긋는 화면 그대로다.
2. **FLS → 도안** — `.3so`·`C_group`·`C_livery` 어느 것을 줘도 plan.json이
   된다 (리버리는 면별 plan + 구성 파일로 편다).
3. **이타샤 구성(itasha.json) → FLS 리버리** — 면마다의 배치를 절대 좌표
   도형으로 구워 `C_livery` 한 장에 싣는다. **창 조작이 하던 일 전부**가
   여기로 온다: 그룹 준비·불러오기·이동·베이스 도색까지 파일 하나다.

면 유닛 좌표는 `engine.preview.surface_layers`가 낸다 — 미리보기가 그리는
바로 그 목록이라 그림과 파일이 안 갈린다.
"""

from __future__ import annotations

import json
from pathlib import Path

from ...paths import find_run_file, run_file, run_label
from ..model import Layer, LayerPlan
from . import folder, header as hdr, livery, project


# ────────────────────────────── 도안 → FLS ──────────────────────────────


def _guide_for(plan: LayerPlan, plan_path: Path) -> dict | None:
    """원화를 편집기 안내 레이어로 — 캔버스(게임 유닛)에 정합해 심는다.

    `cutout.png`(노선이 실제로 받은 입력)가 있으면 작업 캔버스와 같은 구도라
    축마다 정확히 맞고, 없으면 플랜이 적어 둔 원화를 세로에 맞춘다.
    (`kfpseditor._overlay_state`와 같은 규칙 — 편집기만 다르다.)"""
    import cv2
    import numpy as np

    from ...paths import data_root

    cands: list[tuple[Path, bool]] = [
        (find_run_file(plan_path.parent, "cutout.png"), True)]
    if plan.source_image:
        src = Path(plan.source_image)
        cands.append((src if src.is_absolute() else data_root() / src, False))
    for p, exact in cands:
        if not p.is_file():
            continue
        try:
            img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_UNCHANGED)
        except (OSError, cv2.error):
            continue
        if img is None:
            continue
        ih, iw = img.shape[:2]
        if max(ih, iw) > 2048:              # 프로젝트에 통째로 실린다 — 줄여 담는다
            k = 2048.0 / max(ih, iw)
            img = cv2.resize(img, (max(1, round(iw * k)), max(1, round(ih * k))),
                             interpolation=cv2.INTER_AREA)
            ih, iw = img.shape[:2]
        ok, buf = cv2.imencode(".png", img)
        if not ok:
            continue
        w_u = plan.image_size[0] * plan.units_per_px
        h_u = plan.image_size[1] * plan.units_per_px
        sx, sy = (w_u / iw, h_u / ih) if exact else (h_u / ih, h_u / ih)
        # 안내 레이어의 (0,0)은 그림 중심이다 — 캔버스 중심에 맞춰 놓는다
        return project._guide_node(buf.tobytes(), iw, ih, x=0.0, y=0.0,
                                   sx=sx, sy=sy)
    return None


def plan_project(plan_path: str | Path, out_path: str | Path, *,
                 name: str | None = None, guide: bool = True) -> tuple[Path, dict]:
    """plan.json → `.3so` 편집기 프로젝트. 반환은 (경로, 통계)."""
    plan_path = Path(plan_path)
    plan = LayerPlan.load(plan_path)
    label = name or run_label(plan_path)
    g = _guide_for(plan, plan_path) if guide else None
    doc, st = project.group_project(plan.layers, name=label, guide=g)
    p = project.write(doc, Path(out_path))
    st["path"] = str(p)
    st["guide"] = g is not None
    return p, st


def plan_folder(plan_path: str | Path, out_root: str | Path, *,
                name: str | None = None, creator: str = "") -> tuple[Path, dict]:
    """plan.json → 게임 컨테이너 폴더 (`LayerGroup_<이름>/C_group`+`header`)."""
    plan_path = Path(plan_path)
    plan = LayerPlan.load(plan_path)
    label = name or run_label(plan_path)
    out = folder.export_folder(out_root, label, livery_kind=False)
    st = folder.write_group(out, plan.layers, name=label, creator=creator,
                            thumb=_plan_thumb(plan))
    return out, st


def _plan_thumb(plan: LayerPlan, size: int = 256):
    """도안 렌더 축소본 (RGB) — 컨테이너 미리보기 그림. 실패하면 None."""
    try:
        import cv2

        from ..catalog import Catalog, default_catalog_path
        from ..render import render_plan

        rgb = render_plan(plan, Catalog(default_catalog_path()))
        h, w = rgb.shape[:2]
        k = size / max(h, w, 1)
        if k < 1.0:
            rgb = cv2.resize(rgb, (max(1, round(w * k)), max(1, round(h * k))),
                             interpolation=cv2.INTER_AREA)
        return rgb
    except Exception:                       # noqa: BLE001 — 그림칸일 뿐이다
        return None


# ────────────────────────────── FLS → 도안 ──────────────────────────────


def _write_plan(layers: list[Layer], out_dir: Path, *,
                source_image: str = "") -> Path:
    """레이어 목록 → 도안 (캔버스는 내용에 맞춰 잡는다) + 프리뷰.

    이름은 폴더를 따른다 — `<out_dir>/<폴더이름>.plan.json`."""
    from ..kfpsjson import _fit_canvas, write_preview
    from ..catalog import Catalog, default_catalog_path

    size, upp = _fit_canvas(layers)
    plan = LayerPlan(source_image=source_image, image_size=size,
                     units_per_px=upp, layers=layers)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = run_file(out_dir, "plan.json")
    plan.save(out)
    try:
        write_preview(plan, Catalog(default_catalog_path()),
                      run_file(out_dir, "preview.png"))
    except Exception:                       # noqa: BLE001 — 미리보기는 보조다
        pass
    return out


def import_any(path: str | Path, out_root: str | Path) -> tuple[Path, dict]:
    """FLS/게임 파일 → 도안. 반환은 (대표 경로, 통계).

    - `.3so`(그룹)·`C_group` → `<out_root>/<이름>/<이름>.plan.json`
    - `.3so`(리버리)·`C_livery` → 면마다 도안 + 그것들을 묶는
      `<이름>.itasha.json` (대표 경로가 그 구성 파일이다 — [Itasha] 메뉴가
      그대로 문다)
    """
    p = Path(path)
    kind = folder.sniff(p)
    if kind is None:
        raise ValueError(f"FLS 파일로 안 읽힌다 — {p.name} "
                         f"(.3so · C_group · C_livery)")
    out_root = Path(out_root)
    stem = folder.safe_name(p.stem if p.is_file() else p.name, "fls")
    for pref in (folder.LIVERY_PREFIX, folder.GROUP_PREFIX):
        if stem.lower().startswith(pref.lower()):
            stem = stem[len(pref):] or stem
    if kind == "project":
        doc = project.read(p)
        label = str(doc.get("name") or stem)
        if doc.get("is_livery"):
            sections, st = project.sections_of(doc)
            st["paint_rgb"] = project.paint_of(doc).car_color()
            st["car_id"] = int(doc.get("car_id") or 0)
            return _write_livery_plans(sections, out_root / stem, label, st)
        layers, st = project.layers_of(doc)
        st["name"] = label
        return _write_plan(layers, out_root / stem), st
    if kind == "cgroup":
        layers, st = folder.read_group(p)
        return _write_plan(layers, out_root / stem), st
    sections, st = folder.read_livery(p)    # 통계에 `paint_rgb`·`car_id`가 실린다
    return _write_livery_plans(sections, out_root / stem,
                               str(st.get("name") or stem), st)


def _write_livery_plans(sections: dict[str, list[Layer]], out_dir: Path,
                        label: str, st: dict) -> tuple[Path, dict]:
    """면별 도안 + 이 면들을 그대로 되돌리는 `<이름>.itasha.json`.

    구획 좌표가 이미 **면 유닛 절대값**이라 배치는 자리 (0,0)·크기 1·회전 0이다
    — 다시 내보내면 같은 파일이 나온다."""
    out_dir.mkdir(parents=True, exist_ok=True)
    items: list[dict] = []
    for surface, layers in sections.items():
        if not layers:
            continue
        d = out_dir / surface
        got = _write_plan(layers, d)
        items.append({"plan": f"{surface}/{got.name}", "surface": surface,
                      "x": 0.0, "y": 0.0, "scale": 1.0, "rot": 0.0})
    cfg: dict = {"apply": True}
    if st.get("paint_rgb"):
        cfg["paint"] = {"rgb": list(st["paint_rgb"])}
    if st.get("car_id"):
        cfg["fls_car_id"] = int(st["car_id"])
    # **이미 다 구워진 판이다** — 면마다의 도안이 꾸밈까지 포함한 최종 그림이라,
    # 이걸 다시 열 때 꾸밈을 또 얹으면 두 벌이 된다 (`engine.fls.studio`가
    # 이 표시를 보고 꾸밈 체크를 끈다).
    cfg["fls_baked"] = True
    cfg["placements"] = items
    path = run_file(out_dir, "itasha.json")
    path.write_text(json.dumps(cfg, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    st["config"] = str(path)
    st["surfaces"] = {k: len(v) for k, v in sections.items() if v}
    return path, st


# ────────────────────────────── 이타샤 → FLS 리버리 ──────────────────────────────


def _mirrored(layers: list[Layer]) -> list[Layer]:
    """반대편 면에 붙일 좌우 반전 사본."""
    return [Layer(shape=l.shape, x=-l.x, y=l.y, sx=-l.sx, sy=l.sy,
                  rot=(-l.rot) % 360.0, skew=l.skew, color=l.color,
                  alpha=l.alpha, mask=l.mask) for l in layers]


def itasha_chunks(cfg_path: str | Path,
                  cat=None) -> tuple[dict[str, list[tuple[str, list[Layer]]]], dict]:
    """itasha.json → 면 이름 → **이름 붙은 덩어리** 목록 + 메타.

    `itasha_sections`가 이것을 평평하게 편 것이다. 덩어리 이름이 살아 있어야
    FLS 리버리 프로젝트가 편집기 레이어 나무에 구성기 몫을 갈라 실을 수 있다
    (`project.livery_project` · `engine.fls.studio`)."""
    from ..catalog import Catalog, default_catalog_path
    from ..preview import surface_chunks

    cfg_path = Path(cfg_path)
    raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    cat = cat or Catalog(default_catalog_path())
    out: dict[str, list[tuple[str, list[Layer]]]] = {}
    for item in raw.get("placements") or []:
        name = item.get("surface")
        if not name:
            continue
        if item.get("copy_from"):
            # 반대편 복사 — 그쪽 면의 것을 좌우 반전해 그대로 쓴다
            src = out.get(item["copy_from"])
            if src:
                out[name] = [(f"{n}-mirror", _mirrored(ls)) for n, ls in src]
            continue
        chunks = [(n, ls) for n, ls in surface_chunks(item, cfg_path.parent, cat)
                  if ls]
        if chunks:
            out.setdefault(name, []).extend(chunks)
    meta = {"car": raw.get("car"), "media": raw.get("media"),
            "paint_rgb": (tuple(int(v) for v in raw["paint"]["rgb"])
                          if (raw.get("paint") or {}).get("rgb") else None),
            "car_id": int(raw.get("fls_car_id") or 0) or _car_id(raw)}
    return out, meta


def itasha_sections(cfg_path: str | Path,
                    cat=None) -> tuple[dict[str, list[Layer]], dict]:
    """itasha.json → 면 이름 → **면 유닛 좌표** 레이어 목록 + 메타.

    미리보기가 그리는 목록과 같은 것이다 (`engine.preview.surface_layers`)."""
    chunks, meta = itasha_chunks(cfg_path, cat)
    out: dict[str, list[Layer]] = {}
    for name, cs in chunks.items():
        flat: list[Layer] = []
        for _n, ls in cs:
            flat += ls
        out[name] = flat
    return out, meta


def _car_id(raw: dict) -> int:
    """구성이 지어진 차의 **게임 id** — 리버리가 어느 차에 붙을지를 정한다.

    설치 파일에서 그대로 나온다 (`game.carfiles.car_id`). 구성이 설치 차량을
    적어 뒀으면 그것으로, 아니면 표시 이름을 면 지도와 **같은 문**으로 고른다
    (`compose.carfiles_pick`) — 다른 차의 id를 적으면 그 차에서 안 뜬다.
    설치본을 못 찾으면 0이고, 그때는 부르는 쪽이 그 사실을 말한다."""
    try:
        from ...engine import compose
        from ...game import carfiles

        media = raw.get("media") or compose.carfiles_pick(raw.get("car"))
        return carfiles.car_id(media) if media else 0
    except Exception:                       # noqa: BLE001 — 설치본이 없어도 판은 선다
        return 0


def _livery_paint(meta: dict) -> livery.PaintState | None:
    rgb = meta.get("paint_rgb")
    if not rgb:
        return None
    st = livery.PaintState()
    st.set_car_color(tuple(rgb))
    return st


def itasha_project(cfg_path: str | Path, out_path: str | Path, *,
                   name: str | None = None,
                   extra: dict | None = None) -> tuple[Path, dict]:
    """itasha.json → `.3so` 리버리 프로젝트 (FLS가 인자로 받아 여는 그 파일).

    면마다의 덩어리가 **편집기 그룹으로** 선다 (`FS:decal-0-fit` 따위) — 다시
    열 때 구성기 몫과 사람 몫을 그것으로 가른다."""
    cfg_path = Path(cfg_path)
    sections, meta = itasha_chunks(cfg_path)
    label = name or cfg_path.parent.name
    doc, st = project.livery_project(
        sections, name=label, car_id=meta["car_id"],
        paint=_livery_paint(meta), extra=extra,
        header=hdr.draft(folder.safe_name(label), "", meta["car_id"]))
    p = project.write(doc, Path(out_path))
    st.update({"path": str(p), "car": meta.get("car"),
               "media": meta.get("media")})
    return p, st


def itasha_folder(cfg_path: str | Path, out_root: str | Path, *,
                  name: str | None = None, creator: str = "",
                  thumb=None) -> tuple[Path, dict]:
    """itasha.json → 게임 컨테이너 폴더 (`Livery_<이름>/C_livery`+`header`)."""
    cfg_path = Path(cfg_path)
    sections, meta = itasha_sections(cfg_path)
    label = name or cfg_path.parent.name
    out = folder.export_folder(out_root, label, livery_kind=True)
    st = folder.write_livery(out, sections, name=label, creator=creator,
                             car_id=meta["car_id"], paint=_livery_paint(meta),
                             thumb=thumb)
    st.update({"car": meta.get("car"), "media": meta.get("media"),
               "car_id": meta["car_id"]})
    return out, st
