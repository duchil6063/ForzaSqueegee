r"""FLS 기하 덤프(`.fsgeom`) — 차 **메시**가 면마다 어디에 있나.

내장 편집기 포크가 뜨는 파일이다 (`--itasha-dump`, 패치 0002의
`dumpItashaGeometry`). 지금까지 우리는 이 파일의 **이름만** 썼다
(`fls.studio._learn_car` — 미디어명을 얻는 자리). 안에 든 것은 이것이다:

| 무엇 | 쓸모 |
|---|---|
| 면마다 세계↔구획 투영 상수 | 유닛/m 배율과 원점이 **닫힌 식**으로 나온다 |
| 면마다 **깊이 래스터** | 그 방향에서 보이는 차 표면의 셋째 축 좌표 (없으면 NaN) |

## 왜 이것이 마스크보다 나은가

설치 마스크(`carfiles`)는 "게임이 어디를 칠하나"까지만 말한다. **차가 거기서
어떻게 생겼나**는 지금까지 실루엣 둘로 지은 껍질(`game.hull`)과 마스크 형상
어림(`game.seam`·`game.fold`)이 답했다. 깊이 래스터는 그 자리를 조회로 바꾼다 —
투영을 되짚어 세계 점을 얻고 이웃 면으로 다시 던질 수 있다.

## 격자는 이미 우리 것이다 (2026-08-31 실측)

덤프의 래스터는 구획 마스크 상자 위에 **한 칸 = 캔버스 유닛 하나**로 깔린다.
실비아 `side_left` 893×236이 `carfiles.tab_boxes`의 `[-446.5,-118,446.5,118]`과
그대로 맞았다 (top 916×371 · front 308×105도). 그래서 재표본이 필요 없고,
여기서는 마스크와 **같은 격자 규약**으로 돌려 놓는다 (행 0 = v 최대 = 위,
열 0 = u 최소) — `carfiles.InstallSurface.mask`에 그대로 겹친다.

## 축

`x_axis`·`y_axis`는 모델 벡터의 축 번호(0=x·1=y·2=z)이고 부호는
`x_axis_scale`·`y_axis_scale`이다. 세계 규약은 `game.fold`·`game.locators`와
같다 (**+x = 차의 오른쪽 · +y = 위 · +z = 차 앞**, 단위는 미터). 실측: 실비아
`side_left`가 축 2·부호 −1 = −z로 `fold.AXES`의 표와 글자 그대로 맞는다.
표를 못 박아 두는 대신 **차마다 파일이 말하는 것**을 쓴다는 것이 차이다.

**주의**: 덤프는 세계 점에 `mirroredCarSpace`를 먹인다. 그래서 여기의 미터
좌표가 `Locators.xml`의 날 좌표와 한 축의 부호가 다를 수 있다 — 둘을 대는
자(`tools/geom_check.py`)가 그것을 먼저 판정한다.

## 자기끼리 맞는 것과 게임과 맞는 것은 다르다

면 짝 다섯의 깊이 일관성이 1유닛 아래라는 것(`tools/geom_check.py`)은 **덤프
안에서** 좌표계가 어긋나지 않는다는 뜻이지, 게임이 그 자로 그린다는 뜻이 아니다.
앞·뒤 면은 갈린다: 인게임 프로브와 견준 옛 실측에서 FLS 프레임은 옆면이 잘
맞았고(IoU 0.702) 앞·뒤는 0.397로, 우리 실측 등록(`game.locators`)의 0.601에
5/5로 졌다. 그래도 FLS 좌표를 쓰는 것은 **사용자 결정**이다 (2026-08-26,
"자리까지 FLS로") — 프로젝트에 적는 수치를 3D 차에 그리는 것이 편집기 자신이라
화면과 계산이 갈리지 않는 쪽을 골랐다.

## 못 주는 것

도어 유리·선루프 래스터는 **덤프 18대 전부 빈칸**이다 (`liveryProjectionMeshes`가
그 면을 안 담는다). 유리 이음새는 그대로 마스크 실측이 쥔다 (`game.seam`).
"""

from __future__ import annotations

import json
import struct
import subprocess
import zlib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..i18n import msg

MAGIC = b"FSITG1\0\0"

# 슬롯 번호 → 우리 면 이름. `livery.SLOTS`·`carfiles.XML_TO_TAB`과 같은 순서다
# (덤프는 `LiveryMaskSet.sides`를 슬롯 차례로 적는다).
SLOT_TAB: tuple[str, ...] = (
    "front", "rear", "top", "side_left", "side_right", "spoiler",
    "windshield", "rear_window", "sunroof", "window_left", "window_right")

# 세계 축 번호 → 글자 (`game.fold`의 규약).
AXIS_LETTER = ("x", "y", "z")


@dataclass
class SideGeom:
    """면 하나의 메시 기하 — 좌표는 **면 유닛**(에디터 유닛)이다."""

    tab: str
    slot: int
    box: tuple[float, float, float, float]      # (u0, v0, u1, v1)
    origin: tuple[float, float]                 # 아틀라스 원점
    rotation: float
    x_axis: int
    y_axis: int
    depth_axis: int
    x_scale: float
    y_scale: float
    proj_min: tuple[float, float]               # 투영 상자 (미터)
    proj_max: tuple[float, float]
    # 아틀라스 상자 — 파일이 준 그대로 (`plane`·`to_face`가 되짚는 자).
    atlas: tuple[float, float, float, float]    # (left, right, top, bottom)
    # 면 유닛 격자 위의 깊이 (미터, NaN = 그 자리에서 보이는 표면이 없다).
    # 행 0 = v 최대(위) · 열 0 = u 최소 — `carfiles.InstallSurface.mask`와 같다.
    depth: np.ndarray | None = field(default=None, repr=False)

    # ---------- 자 ----------
    @property
    def units_per_m(self) -> tuple[float, float]:
        """면 유닛 / 미터 — u축·v축 따로. 투영 상자가 **닫힌 식**으로 준다."""
        u0, v0, u1, v1 = self.box
        du = self.proj_max[0] - self.proj_min[0]
        dv = self.proj_max[1] - self.proj_min[1]
        # 회전 90/-90은 저장이 돌아 있어 구획 축과 투영 축이 바뀐다
        if abs(self.rotation) == 90.0:
            du, dv = dv, du
        return (abs(u1 - u0) / max(1e-9, abs(du)),
                abs(v1 - v0) / max(1e-9, abs(dv)))

    @property
    def seen(self) -> float:
        """래스터에서 표면이 보이는 칸의 몫 (0~1). 래스터가 없으면 0."""
        if self.depth is None or self.depth.size == 0:
            return 0.0
        return float(np.isfinite(self.depth).mean())

    # ---------- 조회 ----------
    def at(self, u, v):
        """면 유닛 (u, v)의 깊이 (미터). 밖이거나 표면이 없으면 NaN."""
        if self.depth is None:
            return np.full(np.shape(u), np.nan, float)
        h, w = self.depth.shape
        u0, v0, u1, v1 = self.box
        c = np.round((np.asarray(u, float) - u0) / max(1e-9, u1 - u0) * (w - 1))
        r = np.round((v1 - np.asarray(v, float)) / max(1e-9, v1 - v0) * (h - 1))
        ok = (c >= 0) & (c < w) & (r >= 0) & (r < h)
        ci = np.clip(c, 0, w - 1).astype(int)
        ri = np.clip(r, 0, h - 1).astype(int)
        return np.where(ok, self.depth[ri, ci], np.nan)

    def plane(self, u, v) -> dict[str, np.ndarray]:
        """면 유닛 → 그 면이 쥔 **세계 좌표 두 개** (미터). 깊이는 안 본다."""
        ax, ay = _to_atlas(np.asarray(u, float), np.asarray(v, float),
                           self.origin[0], self.origin[1], self.rotation)
        left, right, top, bottom = self.atlas
        got: dict[str, np.ndarray] = {}
        for val, lo, hi, axis, sgn, i in (
                (ax, left, right, self.x_axis, self.x_scale, 0),
                (ay, top, bottom, self.y_axis, self.y_scale, 1)):
            t = (val - lo) / max(1e-9, hi - lo)
            px = self.proj_min[i] + t * (self.proj_max[i] - self.proj_min[i])
            got[AXIS_LETTER[axis]] = px / (sgn if sgn else 1.0)
        return got

    def world(self, u, v) -> dict[str, np.ndarray]:
        """면 유닛 → **차 표면의 세계 좌표 셋** (깊이가 없으면 그 축이 NaN)."""
        got = self.plane(u, v)
        got[AXIS_LETTER[self.depth_axis]] = self.at(u, v)
        return got

    def to_face(self, w: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """세계 좌표 → 이 면의 유닛 (u, v). `plane`의 역이다."""
        left, right, top, bottom = self.atlas
        vals = []
        for lo, hi, axis, sgn, i in (
                (left, right, self.x_axis, self.x_scale, 0),
                (top, bottom, self.y_axis, self.y_scale, 1)):
            px = np.asarray(w[AXIS_LETTER[axis]], float) * (sgn if sgn else 1.0)
            span = self.proj_max[i] - self.proj_min[i]
            vals.append(lo + (px - self.proj_min[i]) / max(1e-9, span) * (hi - lo))
        return _to_face(vals[0], vals[1], self.origin[0], self.origin[1],
                        self.rotation)


@dataclass
class CarGeom:
    """차 한 대의 기하 덤프."""

    media: str
    source: str
    sides: dict[str, SideGeom]
    path: Path | None = None

    def get(self, tab: str) -> SideGeom | None:
        return self.sides.get(tab)


# ---------- 아틀라스 ↔ 면 유닛 ----------
# `carfiles._editor_to_atlas`·`_atlas_corners_to_editor`와 **같은 식**이다.
# 거기 두고 부르지 않고 여기 다시 쓴 이유는 하나뿐이다: 저쪽은 마스크
# 텍스처를 읽는 자라 이 모듈이 그 파일을 안 열어도 서야 한다.
def _to_atlas(u, v, ox: float, oy: float, rot: float):
    if rot == 180.0:
        return -u + ox, -v + oy
    if rot == 90.0:
        return v + ox, u + oy
    if rot == -90.0:
        return -v + ox, -u + oy
    return u + ox, v + oy


def _to_face(ax, ay, ox: float, oy: float, rot: float):
    if rot == 180.0:
        return -(ax - ox), -(ay - oy)
    if rot == 90.0:
        return (ay - oy), (ax - ox)
    if rot == -90.0:
        return -(ay - oy), -(ax - ox)
    return (ax - ox), (ay - oy)


def _unit_box(left, right, top, bottom, ox, oy, rot):
    """아틀라스 상자 네 귀 → 면 유닛 상자 (u0, v0, u1, v1)."""
    cx = np.array([left, right, left, right], float)
    cy = np.array([top, top, bottom, bottom], float)
    cu, cv = _to_face(cx, cy, ox, oy, rot)
    return (float(cu.min()), float(cv.min()), float(cu.max()), float(cv.max()))


# ---------- 읽기 ----------
def _rasters(blob: bytes, off: int, n: int) -> list[np.ndarray | None]:
    """슬롯 차례의 깊이 래스터들 (덤프 순서 그대로 — 회전 안 먹인 판)."""
    out: list[np.ndarray | None] = []
    for _ in range(n):
        w, h, size = struct.unpack_from("<III", blob, off)
        off += 12
        if size == 0 or w <= 0 or h <= 0:
            out.append(None)
            continue
        # `qCompress`는 zlib 앞에 압축 전 크기 4바이트(빅엔디안)를 단다
        raw = zlib.decompress(blob[off + 4:off + size])
        off += size
        out.append(np.frombuffer(raw, np.float32).reshape(h, w))
    return out


def _to_unit_grid(raster: np.ndarray, side: dict,
                  box: tuple[float, float, float, float]) -> np.ndarray:
    """덤프 래스터 → **면 유닛 격자** (행 0 = 위).

    덤프의 칸은 아틀라스 좌표다: 열 i = `left + i` · 행 j = `top + j`
    (패치의 `canvas = (left + (right-left)*u, top + (bottom-top)*v)`).
    마스크와 같은 격자에 얹으려면 회전 라벨을 먹여 되돌려야 한다 —
    `carfiles._surface_from_xml`이 마스크에 하는 것과 같은 걸음이다.
    """
    h, w = raster.shape
    u0, v0, u1, v1 = box
    gw = max(2, int(round(u1 - u0)))
    gh = max(2, int(round(v1 - v0)))
    us = np.linspace(u0, u1, gw)
    vs = np.linspace(v1, v0, gh)                 # 행 0 = 위 (v1)
    U, V = np.meshgrid(us, vs)
    ax, ay = _to_atlas(U, V, side["x_origin"], side["y_origin"],
                       side["rotation_deg"])
    left, right = side["left"], side["right"]
    top, bottom = side["top"], side["bottom"]
    ci = np.round((ax - left) / max(1e-9, right - left) * (w - 1))
    ri = np.round((ay - top) / max(1e-9, bottom - top) * (h - 1))
    ok = (ci >= 0) & (ci < w) & (ri >= 0) & (ri < h)
    ci = np.clip(ci, 0, w - 1).astype(int)
    ri = np.clip(ri, 0, h - 1).astype(int)
    return np.where(ok, raster[ri, ci], np.nan).astype(np.float32)


def read(path: str | Path) -> CarGeom:
    """`.fsgeom` 한 장 → 차 하나의 기하."""
    p = Path(path)
    blob = p.read_bytes()
    if blob[:8] != MAGIC:
        raise ValueError(msg("{path}는 FLS 기하 덤프가 아니다", path=p))
    n = struct.unpack_from("<I", blob, 8)[0]
    head = json.loads(blob[12:12 + n].decode("utf-8"))
    sides_meta = head.get("sides") or []
    rasters = _rasters(blob, 12 + n, len(sides_meta))
    sides: dict[str, SideGeom] = {}
    for meta, raster in zip(sides_meta, rasters):
        slot = int(meta.get("slot", -1))
        if not meta.get("valid") or not 0 <= slot < len(SLOT_TAB):
            continue
        rot = float(meta.get("rotation_deg", 0.0))
        box = _unit_box(meta["left"], meta["right"], meta["top"], meta["bottom"],
                        meta["x_origin"], meta["y_origin"], rot)
        depth = None
        if raster is not None and np.isfinite(raster).any():
            depth = _to_unit_grid(raster, meta, box)
        tab = SLOT_TAB[slot]
        sides[tab] = SideGeom(
            tab=tab, slot=slot, box=box,
            origin=(float(meta["x_origin"]), float(meta["y_origin"])),
            rotation=rot,
            x_axis=int(meta["x_axis"]), y_axis=int(meta["y_axis"]),
            depth_axis=int(meta["depth_axis"]),
            x_scale=float(meta["x_axis_scale"]), y_scale=float(meta["y_axis_scale"]),
            proj_min=tuple(float(v) for v in meta["projection_min"]),
            proj_max=tuple(float(v) for v in meta["projection_max"]),
            atlas=(float(meta["left"]), float(meta["right"]),
                   float(meta["top"]), float(meta["bottom"])),
            depth=depth)
    return CarGeom(media=str(head.get("media") or p.stem),
                   source=str(head.get("source") or ""), sides=sides, path=p)


# ---------- 덤프 뜨기 ----------
def geom_dir() -> Path:
    """덤프가 사는 자리 (`work/geom`)."""
    from ..paths import work_root

    return work_root() / "geom"


def path_for(media: str) -> Path:
    return geom_dir() / f"{media}.fsgeom"


def editor_exe() -> Path | None:
    """동봉 FLS 편집기 (덤프를 뜨는 자). 없으면 None."""
    p = Path(__file__).resolve().parents[2] / "vendor" / "fls-editor" \
        / "ForzaLiveryStudio.exe"
    return p if p.exists() else None


def dump(media: str, out: str | Path | None = None,
         *, timeout: float = 180.0) -> Path:
    """차 하나의 기하를 **창 없이** 뜬다 (`--itasha-dump`).

    편집기가 `[Itasha]` 메뉴로 부를 때는 제가 덤프를 떠서 `--geometry`로 주므로
    이 길은 CLI 쪽 폴백이다. 2.5 MB쯤 나오고 몇 초 걸린다.
    """
    from . import carfiles

    exe = editor_exe()
    if exe is None:
        raise FileNotFoundError(msg("동봉 FLS 편집기가 없다 — 기하 덤프를 못 뜬다"))
    root = carfiles.install_dir()
    if root is None:
        raise FileNotFoundError(msg("FH6 설치 폴더를 못 찾았다"))
    car = root / "media" / "Cars" / f"{media}.zip"
    if not car.exists():
        raise FileNotFoundError(msg("설치본에 {media}가 없다", media=media))
    dst = Path(out) if out is not None else path_for(media)
    dst.parent.mkdir(parents=True, exist_ok=True)
    # 덤프는 `QApplication`보다 먼저 답하므로 창도 플랫폼 플러그인도 안 탄다
    # (패치 0002) — 환경을 안 건드린다.
    proc = subprocess.run([str(exe), "--itasha-dump", str(car), str(dst)],
                          capture_output=True, timeout=timeout)
    if not dst.exists() or dst.stat().st_size < len(MAGIC) + 12:
        err = (proc.stderr or b"").decode("utf-8", "replace").strip()
        raise RuntimeError(msg("기하 덤프 실패 ({media}): {error}",
                               media=media, error=err or proc.returncode))
    for_car.cache_clear()          # 방금 뜬 것을 캐시된 "없음"이 가리면 안 된다
    return dst


@lru_cache(maxsize=8)
def for_car(media: str | None) -> CarGeom | None:
    """차 하나의 기하 — 떠 둔 덤프가 있으면 그것. 없으면 None (안 뜬다).

    **여기서 덤프를 안 뜨는 이유**: 뜨는 데 몇 초·2.5 MB가 들고, 부르는 쪽은
    대개 "있으면 쓰고 없으면 마스크로 물러난다"이다. 일부러 뜨려면
    `dump`를 직접 부른다.
    """
    if not media:
        return None
    p = path_for(media)
    if not p.exists():
        return None
    try:
        return read(p)
    except (OSError, ValueError, KeyError, struct.error, zlib.error):
        return None
