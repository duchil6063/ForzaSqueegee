"""차 면 기하 — **FLS가 실제 게임 메시에서 떠 준 것**을 읽는다.

내장 편집기가 차의 `.carbin` 메시를 그대로 투영해 **면마다 깊이 지도**를 떠
준다 (`--itasha-dump` — 창 없이 3초). 설치 마스크의 실루엣과 어림 껍질
(`game.hull`·`game.seam`)로 세우는 면 지도는 캠백·웨지에서 유리 조각을 겹쳐
놓거나 무릎을 카울 높이로 잡는데, 메시 투영은 그 자리들이 기하로 풀린다.

## 덤프가 담는 것

면(구획) 11칸마다:

- **닫힌 식 조각** — 월드 한 점이 그 면의 구획 좌표 어디에 앉나. 마스크 상자
  (left/right/top/bottom), 원점, 축·부호·배율, 투영 범위, 전치·반전 플래그.
- **깊이 래스터** — 마스크 상자를 유닛 격자로 잘라, 그 칸에서 **면을 마주 보는
  차 표면**의 남은 한 축(깊이) 값. NaN이면 그 칸에는 차가 없다.

앞의 것만으로 월드 → 구획이 풀리고, 뒤의 것이 있어야 구획 → 월드가 풀린다
(면 위의 한 점은 광선일 뿐이고 어디서 멈추는지는 차 표면이 정한다). 둘을
이어 붙이면 **A면의 점 → 월드 → B면의 점**이 된다 — 이것이 면 넘어 좌표를
옮기는 자다 (꾸밈 뿌리를 이웃 면으로 투영하는 데 쓴다).

## 좌표계

월드는 FLS가 쓰는 **미러 차 공간**(x 부호 반전)이다 — 면마다 같은 규약이라
면 사이를 오갈 때 상쇄된다. 구획 좌표는 우리 레이어가 쓰는 그 면 유닛이다
(`game.carfiles`의 에디터 유닛과 같은 공간).
"""

from __future__ import annotations

import json
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MAGIC = b"FSITG1\x00\x00"
FORMAT = "fls_itasha_geometry"
# 덤프의 구획 순서 = `livery.SLOTS` 순서 (Front · Back · Top · Left · Right ·
# Spoiler · FrontWindshield · BackWindshield · TopWindow · LeftWindow · RightWindow)


@dataclass
class SideGeometry:
    """면 하나 — 닫힌 식 조각 + 깊이 래스터."""

    slot: int
    name: str
    valid: bool
    left: float = 0.0
    right: float = 0.0
    top: float = 0.0
    bottom: float = 0.0
    x_origin: float = 0.0
    y_origin: float = 0.0
    x_axis: int = 0
    y_axis: int = 1
    depth_axis: int = 2
    x_axis_scale: float = 1.0
    y_axis_scale: float = 1.0
    rotation_deg: float = 0.0
    transpose: bool = False
    flip_x: bool = False
    flip_y: bool = False
    projection_min: tuple[float, float] = (0.0, 0.0)
    projection_max: tuple[float, float] = (0.0, 0.0)
    depth: np.ndarray | None = field(default=None, repr=False)
    surface: str = ""          # 우리 면 이름 (side_left, top, …)

    # ── 격자 ↔ 아틀라스 ↔ 구획 ──
    @property
    def shape(self) -> tuple[int, int]:
        return (0, 0) if self.depth is None else self.depth.shape

    @property
    def has_surface(self) -> bool:
        return self.depth is not None and bool(np.isfinite(self.depth).any())

    def _uv(self, col: np.ndarray, row: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h, w = self.shape
        return col / max(1, w - 1), row / max(1, h - 1)

    def canvas_of(self, col, row) -> tuple[np.ndarray, np.ndarray]:
        """격자 칸 → 아틀라스 캔버스 좌표."""
        u, v = self._uv(np.asarray(col, float), np.asarray(row, float))
        return (self.left + (self.right - self.left) * u,
                self.top + (self.bottom - self.top) * v)

    def section_of_canvas(self, cx, cy) -> tuple[np.ndarray, np.ndarray]:
        """아틀라스 캔버스 → 구획 좌표 (FLS `liverySectionPoint`)."""
        cx = np.asarray(cx, float)
        cy = np.asarray(cy, float)
        if self.transpose:
            x, y = cy - self.y_origin, cx - self.x_origin
        else:
            x, y = cx - self.x_origin, cy - self.y_origin
        if self.flip_x:
            x = -x
        if self.flip_y:
            y = -y
        return x, y

    def canvas_of_section(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        """구획 좌표 → 아틀라스 캔버스 (위의 역)."""
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        if self.flip_x:
            x = -x
        if self.flip_y:
            y = -y
        if self.transpose:
            return y + self.x_origin, x + self.y_origin
        return x + self.x_origin, y + self.y_origin

    def cell_of_canvas(self, cx, cy) -> tuple[np.ndarray, np.ndarray]:
        h, w = self.shape
        cx = np.asarray(cx, float)
        cy = np.asarray(cy, float)
        du = self.right - self.left
        dv = self.bottom - self.top
        u = (cx - self.left) / (du if du else 1.0)
        v = (cy - self.top) / (dv if dv else 1.0)
        return u * max(1, w - 1), v * max(1, h - 1)

    # ── 월드 ↔ 격자 (FLS 제 공간 — 래스터의 임자다) ──
    def cell_of_world(self, world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """월드(미러 차 공간) → 깊이 래스터의 칸. 래스터를 뜬 그 투영 그대로다."""
        world = np.asarray(world, float)
        px = self.x_axis_scale * world[..., self.x_axis]
        py = self.y_axis_scale * world[..., self.y_axis]
        sx = self.projection_max[0] - self.projection_min[0]
        sy = self.projection_max[1] - self.projection_min[1]
        u = (px - self.projection_min[0]) / (sx if sx else 1.0)
        v = (py - self.projection_min[1]) / (sy if sy else 1.0)
        h, w = self.shape
        return u * max(1, w - 1), v * max(1, h - 1)

    def world_of_cell(self, col, row, depth) -> np.ndarray:
        """칸 + 깊이 → 월드 (위의 역)."""
        u, v = self._uv(np.asarray(col, float), np.asarray(row, float))
        px = self.projection_min[0] + u * (self.projection_max[0]
                                           - self.projection_min[0])
        py = self.projection_min[1] + v * (self.projection_max[1]
                                           - self.projection_min[1])
        out = np.full(np.broadcast(px, py, depth).shape + (3,), np.nan, float)
        out[..., self.x_axis] = px / (self.x_axis_scale or 1.0)
        out[..., self.y_axis] = py / (self.y_axis_scale or 1.0)
        out[..., self.depth_axis] = np.asarray(depth, float)
        return out

    # ── 월드 ↔ 구획 ──
    #
    # **자리도 FLS가 정한다** (사용자 결정 2026-08-26). 우리가 프로젝트에
    # 적는 수치를 3D 차에 그리는 것이 FLS 자신이므로, 그 자가 아닌 좌표를 쓰면
    # 편집기 화면과 우리 계산이 갈린다 — 보정을 걸지 않는다.
    #
    # **FLS 월드는 게임의 거울이다** (x 반대). 게임 자신의 `Locators.xml`이
    # 왼쪽 휠을 x 음수에 두는데 FLS의 왼면 깊이 지도는 x 양수에 있다 — 설치본
    # 13대 전부. 면마다 같은 규약이라 면 사이를 오갈 때는 상쇄된다.
    def section_of_world(self, world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """월드(미러 차 공간, (...,3)) → 구획 좌표."""
        col, row = self.cell_of_world(np.asarray(world, float))
        return self.section_of_canvas(*self.canvas_of(col, row))

    def world_of_section(self, x, y) -> np.ndarray:
        """구획 좌표 → 월드. 차 표면이 없는 자리는 NaN이다 (깊이가 없다)."""
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        cx, cy = self.canvas_of_section(x, y)
        col, row = self.cell_of_canvas(cx, cy)
        return self.world_of_cell(col, row, self.sample_depth(col, row))

    def cell_of_section(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        """구획 좌표 → 깊이 래스터의 칸."""
        return self.cell_of_canvas(*self.canvas_of_section(
            np.asarray(x, float), np.asarray(y, float)))

    def section_of_cell(self, col, row) -> tuple[np.ndarray, np.ndarray]:
        """깊이 래스터의 칸 → 구획 좌표."""
        return self.section_of_canvas(*self.canvas_of(col, row))

    def sample_depth(self, col, row) -> np.ndarray:
        """가장 가까운 칸의 깊이 (격자 밖·표면 없음은 NaN)."""
        col = np.asarray(col, float)
        row = np.asarray(row, float)
        if self.depth is None:
            return np.full(np.broadcast(col, row).shape, np.nan)
        h, w = self.shape
        ci = np.rint(col).astype(int)
        ri = np.rint(row).astype(int)
        inside = (ci >= 0) & (ci < w) & (ri >= 0) & (ri < h)
        out = np.full(ci.shape, np.nan)
        if inside.any():
            out[inside] = self.depth[ri[inside], ci[inside]]
        return out

    # ── 면 지도 ──

    def coverage(self) -> np.ndarray:
        """차 표면이 있는 칸 (격자 그대로, 행 0 = 캔버스 top)."""
        if self.depth is None:
            return np.zeros((0, 0), bool)
        return np.isfinite(self.depth)


@dataclass
class CarGeometry:
    media: str = ""
    source: str = ""
    sides: list[SideGeometry] = field(default_factory=list)

    def side(self, slot: int) -> SideGeometry | None:
        return self.sides[slot] if 0 <= slot < len(self.sides) else None

    def by_name(self, surface: str) -> SideGeometry | None:
        from .livery import SLOT_OF

        slot = SLOT_OF.get(surface)
        return self.side(slot) if slot is not None else None

    def carry(self, src: int, dst: int, x, y) -> tuple[np.ndarray, np.ndarray]:
        """A면의 구획 좌표 → 월드 → B면의 구획 좌표 (**면 넘어 옮기기**).

        차 표면이 없는 자리는 NaN으로 돌아온다 — 잇지 말라는 뜻이다."""
        a, b = self.side(src), self.side(dst)
        if a is None or b is None:
            raise ValueError(f"구획 {src}·{dst} 중 없는 것이 있다")
        world = a.world_of_section(x, y)
        return b.section_of_world(world)


def load(path: str | Path) -> CarGeometry:
    """`.fsgeom` 덤프를 읽는다."""
    raw = Path(path).read_bytes()
    if len(raw) < 12 or raw[:8] != MAGIC:
        raise ValueError("FLS 이타샤 기하 덤프가 아니다")
    n = struct.unpack_from("<I", raw, 8)[0]
    head = json.loads(raw[12 : 12 + n].decode("utf-8"))
    if head.get("format") != FORMAT:
        raise ValueError(f"모르는 덤프 판 — {head.get('format')!r}")
    pos = 12 + n
    car = CarGeometry(media=str(head.get("media", "")),
                      source=str(head.get("source", "")))
    for meta in head.get("sides") or []:
        w, h, packed = struct.unpack_from("<III", raw, pos)
        pos += 12
        depth = None
        if packed:
            # qCompress는 zlib 앞에 원본 길이 4바이트(빅엔디언)를 붙인다
            blob = zlib.decompress(raw[pos + 4 : pos + packed])
            depth = np.frombuffer(blob, "<f4").reshape(h, w)
            pos += packed
        car.sides.append(_side_of(meta, depth))
    return car


def _side_of(meta: dict, depth: np.ndarray | None) -> SideGeometry:
    from .livery import SLOTS

    slot = int(meta.get("slot", -1))
    side = SideGeometry(slot=slot, name=str(meta.get("name", "")),
                        valid=bool(meta.get("valid")),
                        surface=SLOTS[slot][0] if 0 <= slot < len(SLOTS) else "")
    if not side.valid:
        return side
    for key in ("left", "right", "top", "bottom", "x_origin", "y_origin",
                "x_axis_scale", "y_axis_scale", "rotation_deg"):
        setattr(side, key, float(meta.get(key, 0.0)))
    for key in ("x_axis", "y_axis", "depth_axis"):
        setattr(side, key, int(meta.get(key, 0)))
    for key in ("transpose", "flip_x", "flip_y"):
        setattr(side, key, bool(meta.get(key)))
    side.projection_min = tuple(float(v) for v in meta.get("projection_min", (0, 0)))
    side.projection_max = tuple(float(v) for v in meta.get("projection_max", (0, 0)))
    side.depth = depth
    return side


# ────────────────────────────── 덤프 뽑기 ──────────────────────────────


def dumped_cars(out_dir: str | Path | None = None) -> list[str]:
    """덤프가 있는 차 미디어명."""
    return sorted(p.stem for p in _dump_dir(out_dir).glob("*.fsgeom"))


def dump_for_car(media: str, out_dir: str | Path | None = None,
                 *, exe: str | Path | None = None,
                 refresh: bool = False) -> Path:
    """그 차의 기하 덤프를 뽑아 둔다 (있으면 그대로 쓴다).

    내장 편집기를 **창 없이** 부른다 (`--itasha-dump`) — 3초짜리다. 그래서 잰
    적 없는 차도 그 자리에서 선다: 떠 둔 것만 읽던 때는 `work/geom/`에 있는
    차만 이타샤가 됐다."""
    import subprocess

    from ... import flseditor
    from ...game import carfiles

    out_dir = _dump_dir(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{media}.fsgeom"
    root = carfiles.install_dir()
    if root is None:
        raise FileNotFoundError("FH6 설치 폴더를 못 찾았다 — 기하는 차 파일에서 나온다")
    car = root / "media" / "Cars" / f"{media}.zip"
    if not car.is_file():
        raise FileNotFoundError(f"설치 폴더에 그 차가 없다 — {car}")
    if out.is_file() and not refresh and out.stat().st_mtime >= car.stat().st_mtime:
        return out
    binary = Path(exe) if exe else flseditor.find_exe()
    if binary is None:
        raise FileNotFoundError(
            "FLS를 못 찾았다 — `python tools/fls_build.py`로 편집기를 짓거나 "
            "`FS_FLS_EXE`로 자리를 알려 주세요")
    r = subprocess.run([str(binary), "--itasha-dump", str(car), str(out)],
                       capture_output=True, text=True, timeout=600)
    if r.returncode != 0 or not out.is_file():
        raise RuntimeError(f"기하 덤프 실패 — {(r.stderr or r.stdout).strip()}")
    return out


def _dump_dir(out_dir: str | Path | None) -> Path:
    from ...paths import work_root

    return Path(out_dir) if out_dir else work_root() / "geom"
