"""내보내기 폴더 — 게임 저장 컨테이너의 꼴 그대로 쓰고 읽는다.

게임이 비닐 그룹·리버리를 저장하는 단위는 **폴더 하나**다:

    LayerGroup_<이름>/  C_group  header  [thumb.webp]
    Livery_<이름>/      C_livery header  [bigThumb.webp]

FLS도 같은 이름 규칙으로 내보낸다 (`projectExportFolder` — 접두사
`LayerGroup_`·`Livery_`). 이 폴더를 저장 컨테이너 뿌리에 그대로 놓으면 게임의
저장 그리드에 뜬다 — **창을 한 번도 안 건드리고** 도안이 게임으로 간다.

미리보기 그림은 webp다 (게임이 그 확장자로 읽는다). Pillow가 있으면 굽고,
없으면 건너뛴다 — 없어도 목록에는 뜬다 (그림칸이 빈다).
"""

from __future__ import annotations

import re
from pathlib import Path

from ..model import Layer
from . import cgroup, header as hdr, livery, project
from .binfmt import unwrap_container, wrap_container

GROUP_PREFIX = "LayerGroup_"
LIVERY_PREFIX = "Livery_"
_BAD = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def safe_name(name: str, fallback: str = "Project") -> str:
    """FLS `safeGroupName`과 같은 규칙 — 파일 이름으로 못 쓸 글자를 `_`로."""
    out = _BAD.sub("_", str(name or "")).rstrip(". ")
    return out or fallback


def export_folder(base: str | Path, name: str, livery_kind: bool) -> Path:
    """`base` 아래에 설 폴더 자리 (있으면 `_2`, `_3`… — FLS와 같은 규칙)."""
    base = Path(base)
    prefix = LIVERY_PREFIX if livery_kind else GROUP_PREFIX
    if base.name.lower().startswith(prefix.lower()) and len(base.name) > len(prefix):
        return base
    stem = safe_name(name)
    if not stem.lower().startswith(prefix.lower()):
        stem = prefix + stem
    cand = base / stem
    i = 2
    while cand.exists():
        cand = base / f"{stem}_{i}"
        i += 1
    return cand


def _write_thumb(path: Path, rgb, size: tuple[int, int] | None = None) -> bool:
    """미리보기 webp. `size`를 주면 그 상자 안에 맞춰 어두운 판 위에 얹는다
    (게임 그리드 칸이 정해진 크기다). Pillow가 없으면 조용히 건너뛴다."""
    if rgb is None:
        return False
    try:
        from PIL import Image

        im = Image.fromarray(rgb).convert("RGB")
        if size is not None:
            im.thumbnail(size, Image.LANCZOS)
            plate = Image.new("RGB", size, (24, 26, 30))
            plate.paste(im, ((size[0] - im.width) // 2,
                             (size[1] - im.height) // 2))
            im = plate
        im.save(str(path), "WEBP", quality=88)
        return True
    except Exception:                       # noqa: BLE001 — 그림칸일 뿐이다
        return False


def write_group(out_dir: str | Path, layers: list[Layer], *, name: str,
                creator: str = "", car_id: int = 0, thumb=None) -> dict:
    """`C_group` + `header` 폴더 하나. 반환은 통계 + 쓴 경로."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload, st = cgroup.encode_group(layers)
    (out / "C_group").write_bytes(wrap_container(payload))
    meta = hdr.draft(safe_name(name), creator, car_id, st["layers"])
    tag = hdr.creator_tag_from_save_path(out)
    if tag:
        meta.creator_tag = tag
    (out / "header").write_bytes(meta.to_bytes())
    st["folder"] = str(out)
    st["thumb"] = _write_thumb(out / "thumb.webp", thumb)
    return st


def write_livery(out_dir: str | Path, sections: dict[str, list[Layer]], *,
                 name: str, creator: str = "", car_id: int = 0,
                 paint: livery.PaintState | None = None, thumb=None) -> dict:
    """`C_livery` + `header` 폴더 하나 — 이타샤 한 벌이 이 폴더다."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = hdr.creator_tag_from_save_path(out)
    payload, st = livery.encode_livery(sections, car_id=car_id, paint=paint,
                                       creator_tag=tag)
    (out / "C_livery").write_bytes(wrap_container(payload))
    meta = hdr.draft(safe_name(name), creator, car_id, st["layers"])
    if tag:
        meta.creator_tag = tag
    (out / "header").write_bytes(meta.to_bytes())
    st["folder"] = str(out)
    # 게임 리버리 그리드 칸 크기 (FLS가 `renderThumbnail`로 쓰는 그 값)
    st["thumb"] = _write_thumb(out / "bigThumb.webp", thumb, (670, 376))
    return st


# ────────────────────────────── 판별·읽기 ──────────────────────────────


def sniff(path: str | Path) -> str | None:
    """이 경로가 무엇인가 — `project` · `cgroup` · `clivery` · None."""
    p = Path(path)
    if p.is_dir():
        if (p / "C_livery").is_file():
            return "clivery"
        if (p / "C_group").is_file():
            return "cgroup"
        return None
    if not p.is_file():
        return None
    if p.suffix.lower() == project.SUFFIX:
        return "project"
    if p.name.lower() == "c_livery" or p.name.lower().endswith(".c_livery"):
        return "clivery"
    if p.name.lower() == "c_group" or p.name.lower().endswith(".c_group"):
        return "cgroup"
    try:
        with p.open("rb") as f:             # 머리 두 바이트면 된다 (큰 plan.json)
            head = f.read(2)
    except OSError:
        return None
    return "project" if head == b"\x1f\x8b" else None


def _payload_of(p: Path, stem: str) -> bytes:
    f = p / stem if p.is_dir() else p
    return unwrap_container(f.read_bytes())


def read_group(path: str | Path) -> tuple[list[Layer], dict]:
    """`C_group` 파일·폴더 → 평면 레이어 목록. header가 있으면 이름도 준다."""
    p = Path(path)
    layers, st = cgroup.decode_group(_payload_of(p, "C_group"))
    st.update(_header_info(p))
    return layers, st


def read_livery(path: str | Path) -> tuple[dict[str, list[Layer]], dict]:
    """`C_livery` 파일·폴더 → 면 이름 → 레이어 목록. 도색은 통계의 `paint_rgb`."""
    p = Path(path)
    sections, st, _paint = livery.decode_livery(_payload_of(p, "C_livery"))
    st.update(_header_info(p))
    return sections, st


def _header_info(p: Path) -> dict:
    f = (p if p.is_dir() else p.parent) / "header"
    if not f.is_file():
        return {}
    try:
        h = hdr.parse(f.read_bytes())
    except (OSError, ValueError):
        return {}
    return {"name": h.name, "creator": h.creator_name,
            "header_car_id": h.car_id, "decals": h.type_value}
