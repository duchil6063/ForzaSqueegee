"""`.3so` — FLS 편집기 프로젝트 (gzip으로 감싼 JSON).

FLS가 **명령줄 인자로 곧장 여는** 유일한 편집 대상이다 (`main.cpp
openStartupFiles` — `.3so`이거나 `C_livery`). 그래서 "우리 도안을 FLS 편집기에서
연다"는 이 파일을 굽고 FLS를 그 경로로 띄우는 일이 된다.

문서 스키마는 `project_codec.cpp`(`fls_editor_project` 판 3)와 장면 트리
`scene_codec.cpp`다. 여기서 짓는 것은 그 스키마의 **우리가 쓰는 부분**이다:

- 도안(그룹) 프로젝트 — 뿌리에 도형이 줄줄이. 원화를 깔면 안내 레이어(guide)로
  같이 실린다 (사람이 반투명 원화를 깔고 따라 긋는 그 화면 그대로).
- 이타샤(리버리) 프로젝트 — 뿌리에 구획 그룹 11개, 그 아래 도형. 베이스 도색은
  `livery_paint`로 실린다.

색은 `[b, g, r, a]` 순서다 (게임 레코드 그대로 — `scene_codec.cpp`가 그렇게 쓴다).
"""

from __future__ import annotations

import base64
import gzip
import json
from pathlib import Path

from ..model import Layer
from . import ids
from .binfmt import normalize_rotation
from .cgroup import color_bytes
from .header import Header
from .livery import SLOTS, PaintColor, PaintMaterial, PaintState

FORMAT = "fls_editor_project"
VERSION = 3
SUFFIX = ".3so"


def _b64(b: bytes) -> str:
    return base64.b64encode(b or b"").decode("ascii")


def _transform(lay: Layer) -> dict:
    return {"x": lay.x, "y": lay.y, "scale_x": lay.sx, "scale_y": lay.sy,
            "rotation": normalize_rotation(lay.rot), "skew": lay.skew}


def _shape_node(lay: Layer, shape_id: int, index: int, prefix: str = "s") -> dict:
    b, g, r, a = color_bytes(lay)
    return {"kind": "shape", "id": f"{prefix}{index}",
            "name": f"{lay.shape}", "transform": _transform(lay),
            "opacity": 1.0, "visible": True, "locked": False,
            "mask": bool(lay.mask), "color": [b, g, r, a],
            "visual": {"kind": "vector", "shape_id": int(shape_id)},
            "debug": {"source_shape": 0, "abs_offset": 0, "marker": "",
                      "flags": 0, "source_logo_id": 0,
                      "has_source_transform": False}}


def _group_debug() -> dict:
    return {"source_abs_pos": 0, "pending_transform_marker": "",
            "inline_transform_marker": "", "effective_transform_marker": "",
            "header_control_bytes": "", "flags": 0, "source_parent_id": "",
            "source_previous_sibling_id": "", "source_previous_group_depth": 0,
            "source_child_ids": []}


def _guide_node(png: bytes, width: int, height: int, *,
                x: float, y: float, sx: float, sy: float,
                opacity: float = 0.5) -> dict:
    """원화 안내 레이어 — FLS의 `guide` 노드 (픽셀 그림 + 변형)."""
    return {"kind": "guide", "id": "guide0", "name": "원화",
            "transform": {"x": x, "y": y, "scale_x": sx, "scale_y": sy,
                          "rotation": 0.0, "skew": 0.0},
            "opacity": float(opacity), "visible": True, "locked": False,
            "source_path": "",
            "image": {"format": "png", "width": int(width),
                      "height": int(height), "image_bytes": _b64(png),
                      "orientation": "top_down"}}


def _paint_json(paint: PaintState) -> dict:
    out = []
    for h, m in paint.materials.items():
        out.append({"material_hash": f"{h:016x}",
                    "primary": {"enabled": m.primary.enabled,
                                "bgra": list(m.primary.bgra)},
                    "secondary": {"enabled": m.secondary.enabled,
                                  "bgra": list(m.secondary.bgra)},
                    "manufacturer_selector": int(m.selector),
                    "finish": int(m.finish)})
    return {"materials": out}


def _paint_from_json(raw: dict) -> PaintState:
    st = PaintState()
    for e in (raw or {}).get("materials") or []:
        try:
            h = int(str(e.get("material_hash")), 16)
        except (TypeError, ValueError):
            continue
        m = PaintMaterial()
        for key, slot in (("primary", "primary"), ("secondary", "secondary")):
            c = e.get(key) or {}
            bgra = list(c.get("bgra") or (0, 0, 0, 0))[:4]
            bgra += [0] * (4 - len(bgra))
            setattr(m, slot, PaintColor(bool(c.get("enabled")),
                                        tuple(int(v) & 255 for v in bgra)))
        m.selector = int(e.get("manufacturer_selector", 0xFFFFFFFF))
        m.finish = int(e.get("finish", 0))
        st.materials[h] = m
    return st


def _header_json(h: Header) -> dict:
    return {"format_version": h.format_version, "name": h.name,
            "published": h.published, "description": h.description,
            "year": h.year, "month": h.month, "day": h.day,
            "field_block": _b64(h.field_block), "creator_tag": _b64(h.creator_tag),
            "creator_name": h.creator_name,
            "section_prefix": _b64(h.section_prefix),
            "type_value": h.type_value, "car_id": h.car_id,
            "guid": _b64(h.guid), "trailing": _b64(h.trailing),
            "published_tail": ""}


# ────────────────────────────── 짓기 ──────────────────────────────


def _base(name: str, header: Header | None, car_id: int) -> dict:
    doc: dict = {"format": FORMAT, "version": VERSION, "name": name,
                 "source_folder": "", "is_livery": False,
                 "horizontal_guidelines": [], "vertical_guidelines": [],
                 "source_dec_prefix": "", "source_header": "",
                 "color_swatches": []}
    if car_id:
        doc["car_id"] = int(car_id)
    if header is not None:
        doc["header_metadata"] = _header_json(header)
    return doc


def group_project(layers: list[Layer], *, name: str = "Untitled",
                  header: Header | None = None,
                  guide: dict | None = None) -> tuple[dict, dict]:
    """도안(비닐 그룹) 프로젝트 문서 + 통계. `guide`는 `_guide_node` 결과."""
    doc = _base(name, header, 0)
    children: list[dict] = []
    if guide is not None:
        children.append(guide)
    skipped: dict[str, int] = {}
    kept = 0
    for lay in layers:
        sid = ids.id_of(lay.shape)
        if sid is None:
            skipped[lay.shape] = skipped.get(lay.shape, 0) + 1
            continue
        children.append(_shape_node(lay, sid, kept))
        kept += 1
    doc["root"] = {"children": children}
    return doc, {"layers": kept, "skipped": skipped}


# 구성기가 지은 덩어리에 붙는 그룹 이름 머리. **이 머리가 곧 소유권 표시**다 —
# FLS에서 사람이 새로 그린 것은 이 머리가 없고, 이타샤 명령은 다시 지을 때
# 제 머리가 붙은 그룹만 갈아 끼운다 (`engine.fls.studio`).
CHUNK_PREFIX = "FS:"


def _plain_group(gid: str, name: str, kids: list[dict]) -> dict:
    return {"kind": "group", "id": gid, "name": name,
            "transform": {"x": 0.0, "y": 0.0, "scale_x": 1.0, "scale_y": 1.0,
                          "rotation": 0.0, "skew": 0.0},
            "opacity": 1.0, "visible": True, "locked": False,
            "debug": _group_debug(), "children": kids}


def _as_chunks(value) -> list[tuple[str, list[Layer]]]:
    """면 하나의 값 → 이름 붙은 덩어리 목록 (평평한 목록도 받는다)."""
    if value and isinstance(value[0], tuple):
        return [(str(n), list(ls)) for n, ls in value]
    return [("", list(value or []))]


def livery_project(sections: dict, *, name: str = "Untitled",
                   car_id: int = 0, paint: PaintState | None = None,
                   header: Header | None = None,
                   extra: dict | None = None) -> tuple[dict, dict]:
    """이타샤(리버리) 프로젝트 문서 + 통계 — 구획 그룹 11칸 아래에 도형.

    면의 값은 **레이어 목록**이거나 `(이름, 레이어들)` 덩어리 목록이다. 덩어리로
    주면 구획 그룹 아래에 `FS:<이름>` 하위 그룹이 서서 편집기 레이어 나무에
    그대로 보이고, 다시 열 때 구성기 몫과 사람 몫이 갈린다
    (`engine.preview.surface_chunks` · `engine.fls.studio`).

    `extra`는 문서 뿌리에 그대로 실리는 칸이다 — 이타샤 명령이 제 상태를
    프로젝트 안에 두고 다닐 때 쓴다 (FLS는 모르는 칸을 안 건드린다).
    """
    doc = _base(name, header, car_id)
    doc["is_livery"] = True
    if paint is not None and paint.materials:
        doc["livery_paint"] = _paint_json(paint)
    if extra:
        doc.update(extra)
    children: list[dict] = []
    skipped: dict[str, int] = {}
    counts: dict[str, int] = {}
    idx = 0
    for slot, (surface, label) in enumerate(SLOTS):
        chunks = _as_chunks(sections.get(surface))
        kids: list[dict] = []
        kept = 0
        for cname, layers in chunks:
            shaped: list[dict] = []
            for lay in layers:
                sid = ids.id_of(lay.shape)
                if sid is None:
                    skipped[lay.shape] = skipped.get(lay.shape, 0) + 1
                    continue
                shaped.append(_shape_node(lay, sid, idx))
                idx += 1
            if not shaped:
                continue
            kept += len(shaped)
            kids += shaped if not cname else [_plain_group(
                f"sec{slot}-{len(kids)}", CHUNK_PREFIX + cname, shaped)]
        if not kids:
            continue
        counts[surface] = kept
        sec = _plain_group(f"sec{slot}", label, kids)
        sec.update({"is_livery_section": True, "livery_section_slot": slot})
        children.append(sec)
    doc["root"] = {"children": children}
    return doc, {"layers": idx, "sections": counts, "skipped": skipped}


def write(doc: dict, path: str | Path) -> Path:
    """문서를 `.3so`(gzip JSON)로 쓴다. FLS가 이 경로를 인자로 받아 연다."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(doc, ensure_ascii=False, indent=4).encode("utf-8")
    p.write_bytes(gzip.compress(body))
    return p


# ────────────────────────────── 읽기 ──────────────────────────────


def read(path: str | Path) -> dict:
    raw = Path(path).read_bytes()
    if len(raw) < 2 or raw[0] != 0x1F or raw[1] != 0x8B:
        raise ValueError("`.3so` 프로젝트가 아니다 (gzip 껍데기가 없다)")
    doc = json.loads(gzip.decompress(raw).decode("utf-8"))
    if not isinstance(doc, dict):
        raise ValueError("프로젝트 문서가 객체가 아니다")
    fmt = doc.get("format")
    if fmt not in (None, FORMAT, "fh6_editor_project"):
        raise ValueError(f"FLS 편집기 프로젝트가 아니다 (format={fmt!r})")
    if int(doc.get("version", 0)) > VERSION:
        raise ValueError(f"프로젝트 판 {doc.get('version')} — 이 판은 못 읽는다")
    return doc


def _layer_of(node: dict, gm) -> Layer | None:
    from .binfmt import decompose, mat_mul, transform_matrix, translation_of

    vis = node.get("visual") or {}
    if vis.get("kind") == "raster":
        return None
    name = ids.name_of(int(vis.get("shape_id", -1)))
    if name is None:
        return None
    t = node.get("transform") or {}
    x, y = float(t.get("x", 0.0)), float(t.get("y", 0.0))
    sx, sy = float(t.get("scale_x", 1.0)), float(t.get("scale_y", 1.0))
    rot, skew = float(t.get("rotation", 0.0)), float(t.get("skew", 0.0))
    shift = translation_of(gm)              # 이동뿐인 부모는 좌표만 옮긴다
    if shift is not None:
        x, y = x + shift[0], y + shift[1]
    else:
        x, y, sx, sy, rot, skew = decompose(mat_mul(
            gm, transform_matrix(x, y, sx, sy, rot, skew)))
    col = list(node.get("color") or (255, 255, 255, 255))[:4]
    col += [255] * (4 - len(col))
    b, g, r, a = (int(v) & 255 for v in col)
    return Layer(shape=name, x=x, y=y, sx=sx, sy=sy, rot=rot, skew=skew,
                 color=(r, g, b), alpha=round(a / 255.0 * 100.0, 2),
                 mask=bool(node.get("mask")))


def _walk(nodes: list, gm, out: list[Layer], unknown: dict[str, int]) -> None:
    """장면 노드 → 레이어. **도형의 `visible`은 안 본다** (FLS와 같은 규칙).

    FLS 캔버스는 리버리에서 지금 보는 구획만 켜 두고 나머지 구획의 도형을
    `visible: false`로 내려 둔다 — 그건 화면 상태이지 내용이 아니라서, FLS의
    내보내기도 그 값을 무시하고 전부 싣는다 (`collectSectionShapes`). 우리가
    그걸 걸러 내면 FLS에서 저장한 리버리를 되읽을 때 구획 하나만 남는다
    (2026-08-26 실측: 7칸 중 Front만 살아 돌아왔다). 그룹의 `visible`은
    FLS도 보므로 그대로 존중한다."""
    from .binfmt import mat_mul, transform_matrix

    for node in nodes:
        if not isinstance(node, dict):
            continue
        kind = node.get("kind")
        if kind == "guide":
            continue
        if kind == "group" and node.get("visible") is False:
            continue
        if kind == "group":
            t = node.get("transform") or {}
            local = transform_matrix(
                float(t.get("x", 0.0)), float(t.get("y", 0.0)),
                float(t.get("scale_x", 1.0)), float(t.get("scale_y", 1.0)),
                float(t.get("rotation", 0.0)), float(t.get("skew", 0.0)))
            _walk(node.get("children") or [], mat_mul(gm, local), out, unknown)
            continue
        lay = _layer_of(node, gm)
        if lay is None:
            key = str((node.get("visual") or {}).get("shape_id", "?"))
            unknown[key] = unknown.get(key, 0) + 1
            continue
        out.append(lay)


def layers_of(doc: dict) -> tuple[list[Layer], dict]:
    """프로젝트 문서 → 평면 레이어 목록 (그룹 변환 합성) + 통계."""
    from .binfmt import IDENTITY

    out: list[Layer] = []
    unknown: dict[str, int] = {}
    _walk((doc.get("root") or {}).get("children") or [], IDENTITY, out, unknown)
    return out, {"layers": len(out), "unknown": unknown,
                 "is_livery": bool(doc.get("is_livery"))}


def sections_of(doc: dict) -> tuple[dict[str, list[Layer]], dict]:
    """리버리 프로젝트 문서 → 면 이름 → 레이어 목록 + 통계."""
    from .binfmt import IDENTITY, mat_mul, transform_matrix

    out: dict[str, list[Layer]] = {}
    unknown: dict[str, int] = {}
    for node in (doc.get("root") or {}).get("children") or []:
        if not isinstance(node, dict) or node.get("kind") != "group":
            continue
        if not node.get("is_livery_section"):
            continue
        slot = int(node.get("livery_section_slot", -1))
        if not 0 <= slot < len(SLOTS):
            continue
        t = node.get("transform") or {}
        local = transform_matrix(
            float(t.get("x", 0.0)), float(t.get("y", 0.0)),
            float(t.get("scale_x", 1.0)), float(t.get("scale_y", 1.0)),
            float(t.get("rotation", 0.0)), float(t.get("skew", 0.0)))
        layers: list[Layer] = []
        _walk(node.get("children") or [], mat_mul(IDENTITY, local), layers,
              unknown)
        if layers:
            out[SLOTS[slot][0]] = layers
    return out, {"layers": sum(len(v) for v in out.values()),
                 "sections": {k: len(v) for k, v in out.items()},
                 "unknown": unknown}


def _xf_of(node: dict) -> dict:
    t = node.get("transform") or {}
    return {"x": float(t.get("x", 0.0)), "y": float(t.get("y", 0.0)),
            "scale_x": float(t.get("scale_x", 1.0)),
            "scale_y": float(t.get("scale_y", 1.0)),
            "rotation": float(t.get("rotation", 0.0)),
            "skew": float(t.get("skew", 0.0))}


def section_chunks(doc: dict) -> dict[str, dict]:
    """리버리 프로젝트 → 면마다 **구성기 덩어리와 사람 몫**을 갈라 본다.

    면 하나의 값은 `{"chunks": {이름: 변환}, "foreign": [원본 노드…]}`다:

    - `chunks` — `FS:` 머리가 붙은 하위 그룹. 값은 그 그룹의 **현재 변환**이라,
      사람이 편집기에서 그룹째 끌어 옮겼으면 그 몫이 여기 남는다 (FLS는 그룹
      이동을 그룹 자신의 변환에 적는다 — `editor_state.transformEntryFrames`).
    - `foreign` — 그 밖 전부. **사람이 FLS에서 직접 그린 것**이라 다시 지을 때
      건드리지 않고 그대로 옮겨 싣는다.
    """
    out: dict[str, dict] = {}
    for node in (doc.get("root") or {}).get("children") or []:
        if not isinstance(node, dict) or node.get("kind") != "group":
            continue
        if not node.get("is_livery_section"):
            continue
        slot = int(node.get("livery_section_slot", -1))
        if not 0 <= slot < len(SLOTS):
            continue
        chunks: dict[str, dict] = {}
        foreign: list[dict] = []
        for kid in node.get("children") or []:
            name = str(kid.get("name") or "")
            if kid.get("kind") == "group" and name.startswith(CHUNK_PREFIX):
                chunks[name[len(CHUNK_PREFIX):]] = _xf_of(kid)
            else:
                foreign.append(kid)
        out[SLOTS[slot][0]] = {"chunks": chunks, "foreign": foreign}
    return out


def paint_of(doc: dict) -> PaintState:
    return _paint_from_json(doc.get("livery_paint") or {})
