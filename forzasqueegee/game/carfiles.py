r"""FH6 **설치 파일**에서 차량 리버리 면 지도를 읽는다 — 프로브 없이.

`media/Cars/<차>.zip` 안의 `LiveryMasks/`가 게임이 쓰는 면별 도색 마스크 그
자체다 (2026-08-19 실측 해부):

- `Masks.xml` — 면 11종의 유효 여부·아틀라스 상자·원점·회전. **유효 플래그의
  정식 순서가 곧 인게임 면 탭 구성이다** (스포일러 없는 차는 Wing invalid).
- `<면>.swatchbin` — 1024×1024 BC5 텍스처 (Grub/TXCB 컨테이너, 페이로드 오프셋
  140). 두 채널이 거의 같고 (차이 1%대) ch0을 쓴다.

## 아틀라스 좌표
마스크 텍스처는 모든 면이 한 장을 나눠 쓰는 **아틀라스**다:
x ∈ [-1024, +1024] → 픽셀 0..1024 (2유닛/px), y ∈ [-512, +512] → 픽셀
1024..0 (1유닛/px, 상하 반전). 세 차종·아홉 면의 XML 상자 ↔ 픽셀 상자가
1~2px 안에서 맞았다.

## 에디터 유닛
변형 박스에 치는 수치 = 아틀라스 − (xorigin, yorigin), 회전 라벨만큼 축을
돌린다. 1994 Miata side_left에서 인게임 프로브 마스크의 **99.2%**가 이
변환으로 설치 마스크 안에 들어갔다 (남는 차이는 프로브가 색 차분로 유리·
소프트탑을 못 재는 것 — 설치 마스크가 실측보다 완전하다).

여기서 만드는 `SurfaceMap`은 화면 warp가 없다 (`origin_px`·`px_per_unit`은
마스크 픽셀 공간이다). 배치 계산(`fit`·`masked_at`·`blob_box`)은 유닛 공간이라
그대로 서고, 화면을 쓰는 것(오토핏·프로브)만 실측 지도가 필요하다.
"""

from __future__ import annotations

import json
import os
import re
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..i18n import msg
from .surface import SurfaceMap

# 정식 면 순서 — 인게임 탭 순서와 같다 (catalog/body_tabs.json 실측 순서).
# XML 파일 안의 원소 순서(Front, Back, Top, ...)와는 다르다.
XML_TO_TAB: tuple[tuple[str, str], ...] = (
    ("Front", "front"),
    ("Left", "side_left"),
    ("Top", "top"),
    ("Right", "side_right"),
    ("Back", "rear"),
    ("Wing", "spoiler"),
    ("Glass_Front", "windshield"),
    ("Glass_Back", "rear_window"),
    ("Glass_Top", "sunroof"),
    ("Glass_Left", "window_left"),
    ("Glass_Right", "window_right"),
)
TAB_CAPS = {"side_left": 3000, "top": 3000, "side_right": 3000}   # 나머지 1,000

_PAYLOAD_OFF = 140
_TEX = 1024


# ---------- 설치 폴더 ----------
# 자동 탐색은 **Steam 규약**만 안다 (레지스트리 → 라이브러리 → common/ForzaHorizon6).
# Game Pass·MS Store 설치본이나 옮겨 온 폴더는 그 규약 밖이라 사람이 못 박는다.
ENV_DIR = "FS_FH6_DIR"
_OVERRIDE: Path | None = None          # 이번 실행만 (CLI `--game-dir`)


def settings_file() -> Path:
    """못 박은 게임 자리를 적어 두는 파일 (`work/`는 저장소가 안 따라간다).

    설치 폴더(`install_dir`)와 저장 컨테이너 뿌리(`save_dir` — `savedir`)가
    한 파일에 산다. 그래서 쓰기는 **읽고 고쳐 쓴다** — 한쪽을 못 박는 것이
    다른 쪽을 지우면 안 된다.
    """
    from ..paths import work_root
    return work_root() / "state" / "gamedir.json"


def read_settings() -> dict:
    """저장 파일 그대로 (없거나 깨졌으면 빈 칸)."""
    try:
        raw = json.loads(settings_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return raw if isinstance(raw, dict) else {}


def write_setting(key: str, value: str | None) -> None:
    """열쇠 하나만 고쳐 쓴다. `None`이면 지운다 (남는 값이 없으면 파일도)."""
    raw = read_settings()
    if value is None:
        raw.pop(key, None)
    else:
        raw[key] = value
    f = settings_file()
    if not raw:
        f.unlink(missing_ok=True)
        return
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(raw, ensure_ascii=False, indent=1), encoding="utf-8")


def _root_of(p: str | Path | None) -> Path | None:
    """이 경로가 설치 폴더인가 — 판정자는 `media/Cars`다. 아니면 None.

    폴더 고르기는 `media`나 `media/Cars`까지 들어가서 멈추기 쉬우므로 위로 두
    칸까지 되짚는다 (사람이 게임 폴더를 고른 것은 맞다).
    """
    if not p:
        return None
    q = Path(p).expanduser()
    for c in (q, *list(q.parents)[:2]):
        if (c / "media" / "Cars").is_dir():
            return c
    return None


def _steam_roots() -> list[Path]:
    """Steam이 적어 둔 자리들 — 본체(레지스트리) + `libraryfolders.vdf`의 라이브러리."""
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam") as k:
            steam = Path(winreg.QueryValueEx(k, "SteamPath")[0])
    except OSError:
        return []
    libs = [steam]
    vdf = steam / "steamapps" / "libraryfolders.vdf"
    if vdf.exists():
        try:
            libs += [Path(m.group(1).replace("\\\\", "\\"))
                     for m in re.finditer(r'"path"\s+"([^"]+)"',
                                          vdf.read_text(errors="replace"))]
        except OSError:
            pass
    return [lib / "steamapps" / "common" / "ForzaHorizon6" for lib in libs]


def saved_dir() -> Path | None:
    """저장 파일에 적힌 폴더 (검사 없이 적힌 그대로). 없으면 None."""
    d = read_settings().get("install_dir")
    return Path(d) if d else None


def resolve() -> tuple[Path | None, str]:
    """설치 폴더 + **어디서 나온 자리인가** (사람에게 그대로 보여 줄 한 줄).

    못 박은 것이 이긴다: `--game-dir`(이번 실행) → 환경변수 `FS_FH6_DIR` →
    저장해 둔 폴더 → Steam 자동 탐색. 자동 탐색이 사람이 고른 자리를 덮으면 못
    박은 뜻이 없다.

    **승격은 환경변수를 안 물려받는다** (`elevate.py` — AppInfo 서비스가 띄운다).
    UAC를 타고 다시 뜨는 제품 경로에서 살아남는 것은 **인자와 저장 파일**이다.
    """
    bad = ""
    for src, cand in (("--game-dir", _OVERRIDE),
                      (ENV_DIR, os.environ.get(ENV_DIR)),
                      (msg("저장해 둔 폴더"), saved_dir())):
        if not cand:
            continue
        root = _root_of(cand)
        if root is not None:
            return root, src + bad
        # 못 박았는데 그 자리가 아니면 **말은 해 준다** — 조용히 자동 탐색으로
        # 물러나면 사람은 못 박았다고 믿는 채로 프리셋 배치를 받는다.
        bad = msg(" ({src} `{cand}`에 media/Cars가 없다)", src=src, cand=cand)
    for c in _steam_roots():
        if (c / "media" / "Cars").is_dir():
            return c, msg("Steam 자동 탐색") + bad
    return None, msg("못 찾았다") + bad


def install_dir() -> Path | None:
    """FH6 설치 폴더 (`media/Cars`가 있는 곳). 못 찾으면 None."""
    return resolve()[0]


def _clear_caches() -> None:
    """폴더가 바뀌면 읽어 둔 차를 버린다 — 캐시 키가 `root=None`이라 안 갈린다."""
    read_car.cache_clear()
    from . import locators
    locators.read.cache_clear()
    locators.for_car.cache_clear()


def use_dir(path: str | Path | None) -> Path | None:
    """이번 실행만 이 폴더를 쓴다 (CLI `--game-dir`) — 저장하지 않는다."""
    global _OVERRIDE
    root = None if path is None else _root_of(path)
    if path is not None and root is None:
        raise ValueError(msg("FH6 설치 폴더가 아니다 (media/Cars가 없다) — {path}",
                             path=path))
    _OVERRIDE = root
    _clear_caches()
    return root


def set_install_dir(path: str | Path | None) -> Path | None:
    """설치 폴더를 못 박아 **저장한다**. `None`이면 저장을 지운다 (자동 탐색으로).

    `media/Cars`가 없는 자리는 거절한다 — 받아 두면 다음 실행이 통째로
    프리셋으로 물러나는데 사람은 못 박았다고 믿는다.
    """
    if path is None:
        write_setting("install_dir", None)
        _clear_caches()
        return None
    root = _root_of(path)
    if root is None:
        raise ValueError(msg("FH6 설치 폴더가 아니다 (media/Cars가 없다) — {path}",
                             path=path))
    write_setting("install_dir", str(root))
    _clear_caches()
    return root


def list_cars(root: Path | None = None) -> list[str]:
    """설치된 차량 미디어명 목록 (zip 이름, 확장자 없이)."""
    root = root or install_dir()
    if root is None:
        return []
    return sorted(p.stem for p in (root / "media" / "Cars").glob("*.zip")
                  if "_Traffic" not in p.stem)


# ---------- BC4/BC5 디코드 ----------
def _bc4_channel(blocks: np.ndarray) -> np.ndarray:
    """(N,8) BC4 블록 → (N,16) 텍셀 (4×4 행우선)."""
    n = blocks.shape[0]
    r0 = blocks[:, 0].astype(np.int32)
    r1 = blocks[:, 1].astype(np.int32)
    bits = np.zeros(n, dtype=np.uint64)
    for i in range(6):
        bits |= blocks[:, 2 + i].astype(np.uint64) << np.uint64(8 * i)
    idx = np.empty((n, 16), dtype=np.int32)
    for t in range(16):
        idx[:, t] = ((bits >> np.uint64(3 * t)) & np.uint64(7)).astype(np.int32)
    pal = np.zeros((n, 8), dtype=np.int32)
    pal[:, 0] = r0
    pal[:, 1] = r1
    gt = r0 > r1
    for j in range(2, 8):
        pal[gt, j] = ((8 - j) * r0[gt] + (j - 1) * r1[gt]) // 7
    for j in range(2, 6):
        pal[~gt, j] = ((6 - j) * r0[~gt] + (j - 1) * r1[~gt]) // 5
    pal[~gt, 7] = 255
    return np.take_along_axis(pal, idx, axis=1).astype(np.uint8)


def _decode_bc5_ch0(raw: bytes) -> np.ndarray:
    """swatchbin 페이로드 → (1024,1024) uint8 (첫 채널).

    보통 BC5(블록 16B — 채널 두 벌)인데 **BC4(블록 8B) 차종도 있다**
    (57 벨에어·혼다 액티 등 — 페이로드가 정확히 절반이다). 페이로드 크기로
    가른다: BC4의 첫 채널이 곧 마스크라 디코드는 같은 자다.
    """
    data = np.frombuffer(raw, dtype=np.uint8)[_PAYLOAD_OFF:]
    bw = _TEX // 4
    if len(data) >= bw * bw * 16:                  # BC5
        blocks = data[: bw * bw * 16].reshape(-1, 16)[:, :8]
    else:                                          # BC4
        blocks = data[: bw * bw * 8].reshape(-1, 8)
    tex = _bc4_channel(blocks)
    return tex.reshape(bw, bw, 4, 4).transpose(0, 2, 1, 3).reshape(_TEX, _TEX)


# ---------- 면 지도 ----------
@dataclass
class InstallSurface:
    tab: str                       # 우리 면 이름 (front, side_left, ...)
    xml: str                       # XML 태그 이름
    index: int                     # 인게임 탭 인덱스 (유효 면만 센 것)
    origin: tuple[float, float]    # 아틀라스 원점 (= 에디터 (0,0))
    box: tuple[float, float, float, float]   # 에디터 유닛 (u0, v0, u1, v1)
    rotation: float
    mask: np.ndarray = field(repr=False)     # bool, box 격자 (행 0 = v1 = 위)

    def to_surface_map(self, cap: int | None = None) -> SurfaceMap:
        mh, mw = self.mask.shape
        u0, v0, u1, v1 = self.box
        kx = mw / max(1e-6, u1 - u0)          # 마스크 px / 유닛
        ky = mh / max(1e-6, v1 - v0)
        return SurfaceMap(
            name=self.tab, index=self.index,
            origin_px=(-u0 * kx, v1 * ky),     # 유닛 (0,0)의 마스크 px 자리
            px_per_unit=(kx, ky),
            paint=self.box, fill=round(float(self.mask.mean()), 4),
            mask=self.mask, cap=cap if cap is not None else TAB_CAPS.get(self.tab, 1000),
            uncertain=False, note="install")


def _atlas_to_px(ax: np.ndarray, ay: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (ax + 1024.0) / 2.0, 512.0 - ay


def _editor_to_atlas(u: np.ndarray, v: np.ndarray, ox: float, oy: float,
                     rot: float) -> tuple[np.ndarray, np.ndarray]:
    """에디터 유닛 → 아틀라스. 회전 라벨별 축 배치 (실측 확정: 0·180).

    90/-90(스포일러·유리 세로 저장)은 마스크 저장이 90도 돌아 있다 —
    (u,v)를 그 각으로 돌려 얹는다.
    """
    if rot == 180.0:               # 우측면: 진짜 180° 회전 — u·v 둘 다 뒤집힌다.
        # 인게임 우측면 y+가 화면 아래로 가는 것과 정합한다 (Tab 미러가
        # 상하 뒤집기인 실측 근거와 같은 뿌리). 8대 대조에서 y반전이 6대
        # 압승(실비아 99.8%·두란고 100%), 진 2대는 프로브 병리 기록이 있다.
        return -u + ox, -v + oy
    if rot == 90.0:
        return v + ox, u + oy
    if rot == -90.0:
        return -v + ox, -u + oy
    return u + ox, v + oy


def _surface_from_xml(attrs: dict[str, str], tab: str, xml_name: str, index: int,
                      tex: np.ndarray) -> InstallSurface:
    ox, oy = float(attrs["xorigin"]), float(attrs["yorigin"])
    rot = float(attrs.get("rotation", 0))
    ax0, ax1 = float(attrs["left"]), float(attrs["right"])
    ay0, ay1 = float(attrs["top"]), float(attrs["bottom"])   # 아틀라스 y (top < bottom)
    # 아틀라스 상자 네 귀를 에디터 유닛으로 되돌려 유닛 상자를 얻는다
    cu, cv = _atlas_corners_to_editor(ax0, ax1, ay0, ay1, ox, oy, rot)
    u0, u1 = float(cu.min()), float(cu.max())
    v0, v1 = float(cv.min()), float(cv.max())
    # 유닛 격자 (1유닛 촘촘) 위에서 마스크를 샘플한다 — x는 텍스처가 2유닛/px라
    # 유닛 격자가 이미 과샘플이다
    w = max(2, int(round(u1 - u0)))
    h = max(2, int(round(v1 - v0)))
    us = np.linspace(u0, u1, w)
    vs = np.linspace(v1, v0, h)                 # 행 0 = 위 (v1)
    U, V = np.meshgrid(us, vs)
    ax, ay = _editor_to_atlas(U, V, ox, oy, rot)
    px, py = _atlas_to_px(ax, ay)
    pxi = np.clip(np.round(px).astype(int), 0, _TEX - 1)
    pyi = np.clip(np.round(py).astype(int), 0, _TEX - 1)
    mask = tex[pyi, pxi] > 128
    return InstallSurface(tab=tab, xml=xml_name, index=index, origin=(ox, oy),
                          box=(u0, v0, u1, v1), rotation=rot, mask=mask)


def _atlas_corners_to_editor(ax0, ax1, ay0, ay1, ox, oy, rot):
    """아틀라스 상자 네 귀 → 에디터 유닛 (역변환)."""
    cx = np.array([ax0, ax1, ax0, ax1], float)
    cy = np.array([ay0, ay0, ay1, ay1], float)
    if rot == 180.0:
        return -(cx - ox), -(cy - oy)
    if rot == 90.0:
        return (cy - oy), (cx - ox)
    if rot == -90.0:
        return -(cy - oy), -(cx - ox)
    return (cx - ox), (cy - oy)


@lru_cache(maxsize=8)
def read_car(media: str, root: Path | None = None) -> dict[str, InstallSurface]:
    """차 하나의 유효 면 전부 — {면 이름: InstallSurface}. 탭 인덱스 포함."""
    base = root or install_dir()
    if base is None:
        raise FileNotFoundError(msg("FH6 설치 폴더를 못 찾았다"))
    zp = base / "media" / "Cars" / f"{media}.zip"
    with zipfile.ZipFile(zp) as z:
        xml_root = ET.fromstring(z.read("LiveryMasks/Masks.xml"))
        attrs = {el.tag: el.attrib for el in xml_root}
        out: dict[str, InstallSurface] = {}
        idx = 0
        for xml_name, tab in XML_TO_TAB:
            a = attrs.get(xml_name)
            # 파일 이름은 XML 태그와 대소문자가 다를 수 있다 (Glass_Front ↔
            # glass_Front) — 실제 엔트리에서 찾는다
            if not a or a.get("valid") != "true":
                continue
            entry = _find_entry(z, xml_name)
            if entry is None:
                continue
            tex = _decode_bc5_ch0(z.read(entry))
            out[tab] = _surface_from_xml(a, tab, xml_name, idx, tex)
            idx += 1
        return out


def _find_entry(z: zipfile.ZipFile, xml_name: str) -> str | None:
    want = f"liverymasks/{xml_name.lower()}.swatchbin"
    for n in z.namelist():
        if n.lower() == want:
            return n
    return None


_CLIP_ID = re.compile(r"carclips_(\d+)\.clipd", re.IGNORECASE)


@lru_cache(maxsize=512)
def car_id(media: str, root: Path | None = None) -> int:
    """차의 **게임 id** — 리버리 파일이 "어느 차의 것인가"를 이 수로 적는다.

    설치 파일에서 그대로 나온다: 차 zip 안 애니메이션 클립 이름이
    `Scene/animations/Mojo/clip/carclips_<id>.clipd`이고 그 숫자가 곧 차 id다
    (2026-08-26 전수 실측: 설치본 596대에서 클립이 정확히 하나씩이고, FLS의
    차 id 표와 **596/596 일치**. 그 표에 없는 49대도 여기서는 나온다).
    못 찾으면 0 — 그때 리버리는 어느 차에도 안 붙는다.
    """
    root = root or install_dir()
    if root is None:
        return 0
    z = root / "media" / "Cars" / f"{media}.zip"
    if not z.is_file():
        return 0
    try:
        with zipfile.ZipFile(z) as f:
            for n in f.namelist():
                m = _CLIP_ID.search(n)
                if m:
                    return int(m.group(1))
    except (OSError, zipfile.BadZipFile):
        return 0
    return 0


def surface_maps(media: str, root: Path | None = None) -> dict[str, SurfaceMap]:
    """차 하나의 면 지도 (`SurfaceMap`) — 배치 계산·미리보기용 (화면 warp 없음)."""
    return {tab: s.to_surface_map() for tab, s in read_car(media, root).items()}


def tab_boxes(media: str, root: Path | None = None
              ) -> dict[str, tuple[float, float, float, float]]:
    """이 차의 면별 **에디터 유닛 상자** — 유효 면의 정식 순서 그대로.

    **텍스처를 안 푼다** — 탭 구성도 상자도 `Masks.xml`의 유효 플래그·아틀라스
    상자와 zip 목록이면 서고, BC5 디코드는 그 열 배 값을 한다 (한 대 0.26초 →
    0.004초). 636대를 훑는 자(`game.cars.sync`)가 이 차이로 산다.

    상자는 **마스크에 꼭 맞는다** — 그래서 면 크기를 이 상자로 잰다. 텍스처를
    풀어 대조하면 (30대 560축) 상자와 칠해진 자리의 차이가 중앙 1유닛·90%가
    3유닛 안이다. 크기를 재는 세 축(옆면 가로·세로, 윗면 세로)은 최대 10유닛.
    예외는 **윗면 가로** 하나로 104유닛까지 벌어진 차가 있다 — 길이를 옆면에서
    재는 이유다 (`size_of_boxes`).
    """
    base = root or install_dir()
    if base is None:
        return {}
    zp = base / "media" / "Cars" / f"{media}.zip"
    out: dict[str, tuple[float, float, float, float]] = {}
    try:
        with zipfile.ZipFile(zp) as z:
            attrs = {el.tag: el.attrib
                     for el in ET.fromstring(z.read("LiveryMasks/Masks.xml"))}
            have = {n.lower() for n in z.namelist()}
            for xml_name, tab in XML_TO_TAB:
                a = attrs.get(xml_name)
                if (a or {}).get("valid") != "true" or \
                        f"liverymasks/{xml_name.lower()}.swatchbin" not in have:
                    continue
                cu, cv = _atlas_corners_to_editor(
                    float(a["left"]), float(a["right"]),
                    float(a["top"]), float(a["bottom"]),
                    float(a["xorigin"]), float(a["yorigin"]),
                    float(a.get("rotation", 0)))
                out[tab] = (float(cu.min()), float(cv.min()),
                            float(cu.max()), float(cv.max()))
    except (OSError, KeyError, ValueError, ET.ParseError, zipfile.BadZipFile):
        return {}
    return out


def tab_names(media: str, root: Path | None = None) -> list[str]:
    """이 차의 인게임 면 탭 구성 예측 (유효 면의 정식 순서)."""
    return list(tab_boxes(media, root))


def size_of_boxes(boxes: dict[str, tuple[float, float, float, float]]
                  ) -> tuple[int, int, int] | None:
    """면 상자들 → 차 한 대를 줄인 세 수 **(길이, 폭, 높이) 유닛**. 없으면 None.

    잴 면은 **사람이 도안을 앉히는 면**에서 고른다: 길이·높이는 옆면 상자,
    폭은 윗면 상자. 옆면이 없으면 윗면·앞뒤면으로 물러난다.

    이것은 **면 상자 유닛이지 밀리미터가 아니다.** 면마다 제 텍스처를 꽉 채워
    쓰므로 옆면 가로는 484~1023유닛에 몰려 있고(636대), 차가 길수록 큰 수가
    나오지는 않는다. 윗면은 **편 것**이라 굽은 차일수록 실제 폭보다 넓다
    (카운타크 폭 882유닛 — 같은 차 앞면은 349유닛이다). 그래도 이 수가 곧
    **그 면에서 쓸 수 있는 칸**이라, 도안을 앉히는 쪽에는 이쪽이 맞는 답이다.
    """
    side = boxes.get("side_left") or boxes.get("side_right")
    top = boxes.get("top")
    face = boxes.get("front") or boxes.get("rear")
    lng = side or top
    length = (lng[2] - lng[0]) if lng else None
    width = (top[3] - top[1]) if top else ((face[2] - face[0]) if face else None)
    height = (side[3] - side[1]) if side else ((face[3] - face[1]) if face else None)
    if length is None or width is None or height is None:
        return None
    return round(length), round(width), round(height)


def car_size(media: str, root: Path | None = None) -> tuple[int, int, int] | None:
    """그 차의 (길이, 폭, 높이) 유닛 — zip에서 바로 (색인은 `game.cars.size_of`)."""
    return size_of_boxes(tab_boxes(media, root))


# ---------- 차명 매칭 ----------
_STOP = {"the", "a", "srt", "gt", "rs", "type", "spec", "edition", "coupe"}
# 미디어명 제조사 약칭 → 표시명 토큰 (약칭이 표시명의 접두가 아닌 것만 적는다)
_MAKE = {"che": "chevrolet", "dod": "dodge", "vw": "volkswagen", "mer": "mercedes",
         "lam": "lamborghini", "mit": "mitsubishi", "nis": "nissan", "por": "porsche",
         "sub": "subaru", "toy": "toyota", "maz": "mazda", "hon": "honda",
         "acu": "acura", "aud": "audi", "alf": "alfa", "ast": "aston",
         "fer": "ferrari", "for": "ford", "bmw": "bmw", "lex": "lexus"}


def _tokens(s: str) -> set[str]:
    toks = {t for t in re.split(r"[^a-z0-9]+", s.lower()) if len(t) >= 2 and t not in _STOP}
    # 구분자를 없앤 붙임꼴도 넣는다 — "RX-7" ↔ "RX7", "MX-5" ↔ "MX5"
    parts = [t for t in re.split(r"[^a-z0-9]+", s.lower()) if t]
    for i in range(len(parts) - 1):
        toks.add(parts[i] + parts[i + 1])
    return toks


# 실측 옆면으로 후보를 거를 때의 근거 (프로브 11대 대조): 프로브가 잰 옆면 가로
# ÷ 설치 옆면 상자 가로 = 0.82~0.96, 중앙 0.89. 프로브는 색 차분이라 유리·어두운
# 자리를 못 재서 **늘 작게** 나온다. 되돌린 창을 ±30%로 넉넉히 여는 것은 맞는
# 차를 안 떨어뜨리기 위해서다 (실측 산포가 ±8%니 3.7배 여유 — 좁히면 설치본
# 636대의 옆면 가로가 484~1023에 몰려 있어 진짜 차부터 잘린다). 그래도 격이
# 다른 차는 걸린다: 미아타(실측 옆면 847) 후보에서 경트럭 액티(529)가 빠진다.
PROBE_SIDE_RATIO = 0.89
SIZE_TOL = 0.30


def size_band(side_width: float) -> tuple[float, float]:
    """실측 옆면 가로 → 설치 옆면 상자 가로가 들 만한 창 (`PROBE_SIDE_RATIO`)."""
    mid = side_width / PROBE_SIDE_RATIO
    return mid * (1 - SIZE_TOL), mid * (1 + SIZE_TOL)


def _size_ok(media: str, band: tuple[float, float], root: Path | None) -> bool:
    """이 차의 옆면 상자가 창 안인가. **크기를 모르면 안 거른다** (모르는 것은 죄가 아니다)."""
    from . import cars as gcars

    got = gcars.size_of(media, root)
    return got is None or band[0] <= got[0] <= band[1]


def match_media(car_display: str, root: Path | None = None,
                side_width: float | None = None) -> list[tuple[float, str]]:
    """OCR로 읽은 차 이름 → 미디어명 후보 (점수 내림차순).

    미디어명 규약: `MAKE_Model_YY` — 제조사 약칭·연도 두 자리·모델 토큰
    겹침으로 점수를 매긴다. 확정이 아니라 **후보**다 — 부르는 쪽(GUI)이
    top-1을 보이고 사람이 바꿀 수 있게 한다. 설치본에 없는 차(대여 프로토
    등)는 빈 후보로 나온다.

    `side_width`는 **실측한 옆면 가로**다 (프로브 지도 · 유닛). 주면 그 크기와
    격이 다른 후보를 뺀다 — 이름만으로는 경트럭과 로드스터가 같은 점수를
    받는다. 전부 빠지면 안 거른 목록을 그대로 돌려준다 (거르는 자가 답을
    없애면 안 된다).
    """
    disp = _tokens(car_display)
    year = None
    m = re.search(r"\b(19|20)(\d\d)\b", car_display)
    if m:
        year = m.group(2)
    out = []
    for media in list_cars(root):
        parts = media.split("_")
        yy = parts[-1] if parts[-1].isdigit() else None
        body = parts[:-1] if yy else parts
        toks = _tokens(" ".join(body))
        # 모델 토큰은 붙은 이름도 가른다 (MX5Cup → mx5, cup)
        for p in body[1:]:
            toks |= _tokens(re.sub(r"(?<=[a-z])(?=[A-Z0-9])", " ", p))
        inter = len(disp & toks)
        score = inter * 2.0
        make = body[0].lower()
        make_full = _MAKE.get(make, make)
        if any(t == make_full or t.startswith(make) for t in disp):
            score += 1.5
        if year and yy == year:
            score += 3.0
        elif year and yy and yy != year:
            score -= 1.0
        # 부분 문자열 겹침 (miata ⊂ mx-5 miata 등 토큰이 안 갈리는 경우)
        joined = media.lower()
        score += sum(0.5 for t in disp if len(t) >= 4 and t in joined)
        if score > 0:
            out.append((score, media))
    # 동점이면 짧은 이름(기본 트림) 우선 — Miata_94 > MiataFE_94
    out.sort(key=lambda t: (-t[0], len(t[1]), t[1]))
    if side_width:
        band = size_band(side_width)
        kept = [c for c in out if _size_ok(c[1], band, root)]
        out = kept or out
    return out[:8]


# 이 점수 아래면 **설치 지도를 안 쓴다** — 틀린 차의 면 지도는 없는 것보다 나쁘다
# (배치 수치도 탭 번호도 어긋난다). 실측 점수: 이름이 미디어명 규약과 가까운
# 차만 넘긴다 (Integra_23 6.5 ↔ Type2 2.5 · F-150 2.5 · 911 Carrera RS 3.0).
MEDIA_MIN = 5.0


def pick_media(car_display: str | None, root: Path | None = None,
               side_width: float | None = None
               ) -> tuple[str | None, list[tuple[float, str]]]:
    """표시 이름 → **쓸 미디어명**(문턱을 넘었을 때만) + 후보 목록.

    문턱을 못 넘으면 `(None, 후보들)`이다 — 부르는 쪽이 후보를 사람에게 보이고
    "`--media`로 못 박아라"를 알려 준다. 문턱이 한 곳에만 있어야 CLI·구성 설계·
    탭 해석이 **같은 차**를 고른다 (셋이 갈리면 미리보기와 실행이 딴 면 지도로
    돈다).

    `side_width`(실측 옆면 가로)를 주면 격이 다른 후보를 먼저 뺀다 —
    `match_media` 참고.
    """
    if not car_display:
        return None, []
    cands = match_media(car_display, root, side_width)
    if cands and cands[0][0] >= MEDIA_MIN:
        return cands[0][1], cands
    return None, cands


def resolve_media(name: str, root: Path | None = None) -> str:
    """사람이 준 `--media` 값 → 정확한 미디어명. 못 고르면 ValueError.

    대소문자만 다른 것은 받아 준다 (설치 폴더는 대소문자를 안 가린다). 아예
    다른 글자면 **말없이 매칭으로 물러나지 않는다** — 오타가 조용히 다른 차의
    면 지도를 물어 오는 것이 이 레버를 만든 이유이기 때문이다. 대신 후보를
    들려 준다.
    """
    cars = list_cars(root)
    if name in cars:
        return name
    low = {c.lower(): c for c in cars}
    if name.lower() in low:
        return low[name.lower()]
    cands = [c for _, c in match_media(name, root)][:6] \
        or [c for c in cars if name.lower() in c.lower()][:6]
    hint = msg("  이런 것들이 있다: {cands}", cands=", ".join(cands)) if cands else \
        msg("  설치 폴더에서 media/Cars/*.zip 이름을 볼 것")
    raise ValueError(msg("설치본에 그런 차가 없다 — {name}\n{hint}",
                         name=name, hint=hint))
