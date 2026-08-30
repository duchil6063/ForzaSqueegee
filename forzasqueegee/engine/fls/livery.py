"""`C_livery` — 리버리 한 벌(차 전체) 저장 파일 (읽기·쓰기).

규격은 `docs/CLIVERY.md`(FLS)다. 구획(section) **11칸**이 순서대로 늘어선 판이고
그 순서가 우리 면 이름과 1:1이다:

    Front · Back · Top · Left · Right · Spoiler ·
    FrontWindshield · BackWindshield · TopWindow · LeftWindow · RightWindow

칸마다의 레이어 상한도 우리 표와 같다 — 2·3·4번(윗면·좌·우)이 3,000, 나머지
1,000 (`game.carfiles.TAB_CAPS`와 독립적으로 같은 값이 FLS `liverySectionShapeLimit`에
있다). **이 파일 하나가 이타샤 한 벌이다**: 면마다 도형을 절대 좌표로 싣고
베이스 도색까지 재질 레코드로 실으므로, 창을 조작해 그룹을 불러오고 옮기고
칠하던 일이 파일 쓰기 한 번이 된다.

## 좌표

구획 안 좌표는 **면 유닛**(에디터 변형 상자에 치는 그 수치)이다.
`game.carfiles`가 설치 마스크에서 뽑는 유닛 공간과 같다. 딱 한 칸,
스포일러(5번)만 FLS가 장면↔파일 사이에 180° 축 보정을 둔다
(`liverySectionCanvasTransform`) — 우리도 같은 보정을 걸어 FLS와 바이트가
맞도록 한다.

## 안 채우는 자리

원본 리버리를 물고 고치는 길(FLS의 source-backed 경로)은 안 쓴다 — 우리는 늘
**새로 짓는다**. 그래서 빈 칸은 규격 그대로의 23바이트 뼈대, 찬 칸은 구획
머리 + 레코드 + 18바이트 잔재로 나간다 (FLS의 `sourcePtr == nullptr` 갈래와
같은 바이트).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ...i18n import msg
from ..model import Layer
from . import materials
from .binfmt import (
    IDENTITY,
    Mat,
    decompose,
    f32,
    invert_affine,
    is_identity,
    mat_mul,
    r_u32,
    transform_matrix,
    u16,
    u32,
    u64,
)
from .cgroup import Walker, shape_payload, usable_layers

# 구획 순서 = 파일 순서. 우리 면 이름 ↔ 게임 구획 이름.
SLOTS: tuple[tuple[str, str], ...] = (
    ("front", "Front"),
    ("rear", "Back"),
    ("top", "Top"),
    ("side_left", "Left"),
    ("side_right", "Right"),
    ("spoiler", "Spoiler"),
    ("windshield", "FrontWindshield"),
    ("rear_window", "BackWindshield"),
    ("sunroof", "TopWindow"),
    ("window_left", "LeftWindow"),
    ("window_right", "RightWindow"),
)
SLOT_OF = {name: i for i, (name, _) in enumerate(SLOTS)}
N_SLOTS = len(SLOTS)

# 구획 한 칸의 레이어 상한 (FLS `liverySectionShapeLimit`)
SLOT_CAPS = tuple(3000 if i in (2, 3, 4) else 1000 for i in range(N_SLOTS))

# 압축 푼 스트림 머리 (FLS `kGyvlHeader`) — 판 0, 뿌리 스케일 1.0
GYVL_HEADER = (b"gyvl" + bytes(4) + bytes(4) + b"\x00\x00\x80\x3f" + bytes(4)
               + b"\x00")
# 칸을 닫는 잔재·빈 칸 뼈대에 실리는 회전값 (FLS가 실측 파일에서 뽑은 표)
SLOT_ROTATION = (0.0, 0.0, 0.0, 180.0, 90.0, -90.0, 90.0, 0.0, 0.0, 180.0, 0.0)
EMPTY_SLOT_ROTATION = (0.0, 0.0, 0.0, 0.0, 180.0, 90.0, -90.0, 90.0, 0.0, 0.0,
                       180.0)
REMNANT_SIZE = 18
EMPTY_SLOT_SIZE = 23
BODY_TRUNCATE = 17


def canvas_transform(slot: int) -> Mat:
    """장면 ↔ 파일 축 보정 (FLS `liverySectionCanvasTransform`)."""
    if slot == 5:
        return ((-1.0, 0.0, 0.0), (0.0, -1.0, 0.0))
    return IDENTITY


def _default_remnant(slot: int) -> bytes:
    return bytes(9) + f32(1.0) + f32(SLOT_ROTATION[slot]) + b"\x00"


def _default_empty_slot(slot: int) -> bytes:
    return bytes(8) + f32(1.0) + f32(EMPTY_SLOT_ROTATION[slot]) + bytes(7)


# ────────────────────────────── 도색 ──────────────────────────────


@dataclass
class PaintColor:
    enabled: bool = False
    bgra: tuple[int, int, int, int] = (0, 0, 0, 0)


@dataclass
class PaintMaterial:
    primary: PaintColor = field(default_factory=PaintColor)
    secondary: PaintColor = field(default_factory=PaintColor)
    selector: int = 0xFFFFFFFF
    finish: int = 0


class PaintState:
    """재질 해시 → 도색. `car_color`가 인게임 "차 색"에 해당한다."""

    def __init__(self) -> None:
        self.materials: dict[int, PaintMaterial] = {}

    def set_car_color(self, rgb: tuple[int, int, int]) -> None:
        r, g, b = (int(v) & 255 for v in rgb)
        for h in materials.DEFAULT_CAR_PAINT_GROUPS:
            m = self.materials.setdefault(h, PaintMaterial())
            m.primary = PaintColor(True, (b, g, r, 255))
            m.selector = 0xFFFFFFFF

    def car_color(self) -> tuple[int, int, int] | None:
        for h in (materials.BODY_PAINT, *materials.DEFAULT_CAR_PAINT_GROUPS):
            m = self.materials.get(h)
            if m is not None and m.primary.enabled:
                b, g, r, _ = m.primary.bgra
                return r, g, b
        return None


def _descriptor_table(paint: PaintState) -> bytes:
    """기술자 청크 — 재질 46 + 패널 11 레코드 (FLS `buildLiveryDescriptorTable`)."""
    order: list[int] = list(materials.LIVERY_MATERIALS)
    for h in paint.materials:
        if h not in order:
            order.append(h)
    for h in materials.LIVERY_PANELS:
        if h not in order:
            order.append(h)

    out = bytearray(b"yrvl")
    out += b"\x00\x02" + u16(len(order)) + bytes(6)
    panels = set(materials.LIVERY_PANELS)
    for h in order:
        m = paint.materials.get(h)
        out += u64(h) + b"\x02"
        out += b"\x01" if (m and m.primary.enabled) else b"\x00"
        out += bytes(m.primary.bgra if m else (0, 0, 0, 0))
        out += b"\x01" if (m and m.secondary.enabled) else b"\x00"
        out += bytes(m.secondary.bgra if m else (0, 0, 0, 0))
        out += u32(m.selector if m else (0xFFFFFFFF if h in panels else 0))
        finish = m.finish if m else 0
        if h == materials.WINDOW_GLASS:
            finish = 0
        elif finish == 0 and m and (m.primary.enabled or m.secondary.enabled):
            finish = 1                      # 색을 넣으면 마감이 무광 아닌 1로 선다
        out += u32(finish)
    out += u32(len(materials.LIVERY_PANELS))
    for h in materials.LIVERY_PANELS:
        out += u64(h)
    return bytes(out)


# ────────────────────────────── 쓰기 ──────────────────────────────


def _section_body(layers: list[Layer], slot: int) -> tuple[bytes, int, bool,
                                                           dict[str, int]]:
    """구획 한 칸의 몸통 — 표식 없는 머리 + 레코드들. (바이트, 장수, 끝장 마스크)"""
    usable, skipped = usable_layers(layers)
    if not usable:
        return b"", 0, False, skipped
    adj = invert_affine(canvas_transform(slot))
    plain = is_identity(canvas_transform(slot))
    blocks = (len(usable) + 7) // 8
    out = bytearray(u16(len(usable)) + u16(blocks) + b"\x00\x00" + bytes(blocks))
    prev_mask = False
    for i, (lay, sid) in enumerate(usable):
        if plain:
            packed = lay
        else:
            x, y, sx, sy, rot, skew = decompose(mat_mul(
                adj, transform_matrix(lay.x, lay.y, lay.sx, lay.sy,
                                      lay.rot, lay.skew)))
            packed = Layer(shape=lay.shape, x=x, y=y, sx=sx, sy=sy, rot=rot,
                           skew=skew, color=lay.color, alpha=lay.alpha,
                           mask=lay.mask)
        if i == 0:
            out += b"\x02"                  # 구획의 첫 레코드는 표식이 없다
        else:
            out += (b"\x01" if prev_mask else b"\x00") + b"\x02"
        out += shape_payload(packed, sid)
        prev_mask = bool(lay.mask)
    return bytes(out), len(usable), prev_mask, skipped


def encode_livery(sections: dict[str, list[Layer]], *, car_id: int = 0,
                  paint: PaintState | None = None,
                  creator_tag: bytes = b"") -> tuple[bytes, dict]:
    """면 이름 → 면 유닛 레이어 목록 → `C_livery` 페이로드(압축 전) + 통계."""
    per_slot: list[list[Layer]] = [[] for _ in range(N_SLOTS)]
    unknown_surfaces: list[str] = []
    for name, layers in sections.items():
        slot = SLOT_OF.get(name)
        if slot is None:
            unknown_surfaces.append(name)
            continue
        per_slot[slot] = list(layers)

    bodies: list[tuple[bytes, int, bool]] = []
    skipped: dict[str, int] = {}
    over: list[str] = []
    for slot in range(N_SLOTS):
        body, n, mask, sk = _section_body(per_slot[slot], slot)
        for k, v in sk.items():
            skipped[k] = skipped.get(k, 0) + v
        if n > SLOT_CAPS[slot]:
            over.append(f"{SLOTS[slot][0]} {n:,}/{SLOT_CAPS[slot]:,}")
        bodies.append((body, n, mask))
    if over:
        raise ValueError(msg("면 레이어 상한을 넘는다 — {over}",
                             over=" · ".join(over)))
    if not any(n for _, n, _ in bodies):
        raise ValueError(msg("실을 수 있는 레이어가 하나도 없다"))

    last_pop = max(i for i in range(N_SLOTS) if bodies[i][1] > 0)
    counts = [0] * N_SLOTS
    body = bytearray()
    for slot in range(N_SLOTS):
        chunk, n, terminal_mask = bodies[slot]
        if n == 0:
            empty = _default_empty_slot(slot)
            if slot == last_pop + 1:
                empty = empty[BODY_TRUNCATE:]
            body += empty
            continue
        body += chunk
        if slot + 1 < N_SLOTS:
            remnant = bytearray(_default_remnant(slot))
            remnant[0] = 0x01 if terminal_mask else 0x00
            body += remnant
        else:
            body += b"\x01" if terminal_mask else b"\x00"
        counts[slot] = n

    gyvl = GYVL_HEADER + bytes(body)
    paint = paint or PaintState()
    tag = (creator_tag or b"").ljust(8, b"\x00")[:8]

    out = bytearray(b"vlrc")
    out += u32(2) + u32(0) + u32(0) + u32(car_id) + u32(0) + u16(0)
    out += b"yrvl" + u32(19) + tag + u32(1) + b"\x00" + u32(len(gyvl))
    out += gyvl
    out += b"yrvl"
    for n in counts:
        out += u32(n)
    out += u32(0)
    out += _descriptor_table(paint)
    out += b"yrvl" + u32(0)
    st = {"sections": {SLOTS[i][0]: counts[i] for i in range(N_SLOTS) if counts[i]},
          "layers": sum(counts), "skipped": skipped,
          "unknown_surfaces": unknown_surfaces, "car_id": car_id}
    return bytes(out), st


# ────────────────────────────── 읽기 ──────────────────────────────


def _find(raw: bytes, tag: bytes, start: int = 0) -> int:
    return raw.find(tag, start)


def decode_paint(raw: bytes, start: int, end: int) -> PaintState:
    """기술자 청크의 도색 레코드 → `PaintState` (FLS `readPaintMaterialsAt`).

    표 머리는 `모드(0|1) · 타입 2 · u16 레코드수 · 예약 6`이고 레코드는
    27바이트다. 못 읽으면 빈 상태를 준다 — 도색은 있으면 좋은 것이지 파일이
    서는 조건이 아니다."""
    st = PaintState()
    if start < 0 or start + 10 > len(raw):
        return st
    if raw[start] > 1 or raw[start + 1] != 0x02:
        return st
    count = int.from_bytes(raw[start + 2 : start + 4], "little")
    if count <= 0 or count > 256:
        return st
    stop = start + 10 + count * 27
    if stop > len(raw) or (end >= 0 and stop > end):
        return st
    pos = start + 10
    for _ in range(count):
        h = int.from_bytes(raw[pos : pos + 8], "little")
        m = PaintMaterial()
        m.primary = PaintColor(raw[pos + 9] != 0,
                               tuple(raw[pos + 10 : pos + 14]))
        m.secondary = PaintColor(raw[pos + 14] != 0,
                                 tuple(raw[pos + 15 : pos + 19]))
        m.selector = int.from_bytes(raw[pos + 19 : pos + 23], "little")
        m.finish = int.from_bytes(raw[pos + 23 : pos + 27], "little")
        st.materials[h] = m
        pos += 27
    return st


def decode_livery(raw: bytes) -> tuple[dict[str, list[Layer]], dict, PaintState]:
    """`C_livery` 페이로드 → (면 이름 → 레이어 목록, 통계, 도색)."""
    gy = _find(raw, b"gyvl")
    if gy < 0:
        raise ValueError(msg("C_livery 안에 gyvl 청크가 없다"))
    car_id = 0
    vl = _find(raw, b"vlrc")
    if vl >= 0 and vl + 0x14 <= len(raw):
        car_id = r_u32(raw, vl + 0x10)
    body_start = gy + 0x15
    stats_tag = _find(raw, b"yrvl", gy)
    if stats_tag < 0 or stats_tag < body_start:
        stats_tag = len(raw)
    body = bytearray(raw[body_start:stats_tag])
    counts: list[int] = []
    if stats_tag + 4 <= len(raw) and raw[stats_tag : stats_tag + 4] == b"yrvl":
        for i in range(N_SLOTS):
            off = stats_tag + 4 + i * 4
            counts.append(r_u32(raw, off) if off + 4 <= len(raw) else 0)
    else:
        counts = [0] * N_SLOTS

    populated = [i for i, n in enumerate(counts) if n > 0]
    if populated:
        trailing = N_SLOTS - populated[-1] - 1
        body += bytes(REMNANT_SIZE + trailing * EMPTY_SLOT_SIZE)

    out: dict[str, list[Layer]] = {}
    unknown: dict[int, int] = {}
    rasters = 0
    pos = 0
    for slot in range(N_SLOTS):
        if counts[slot] <= 0:
            pos = min(len(body), pos + EMPTY_SLOT_SIZE)
            continue
        if pos + 6 > len(body):
            raise ValueError(msg("{section}: 구획 머리가 잘렸다",
                                 section=SLOTS[slot][1]))
        n = int.from_bytes(body[pos : pos + 2], "little")
        blocks = int.from_bytes(body[pos + 2 : pos + 4], "little")
        if n <= 0 or blocks != (n + 7) // 8:
            raise ValueError(msg("{section}: 구획 머리가 규격과 다르다 "
                                 "(자식 {n} · 블록 {blocks})",
                                 section=SLOTS[slot][1], n=n, blocks=blocks))
        bitmap = bytes(body[pos + 6 : pos + 6 + blocks])
        w = Walker(bytes(body), pos + 6 + blocks)
        w.read_children(n, bitmap, False, canvas_transform(slot))
        pos = w.p
        if pos < len(body) and body[pos] == 0x01 and w.out:
            w.out[-1].mask = True
        pos = min(len(body), pos + REMNANT_SIZE)
        out[SLOTS[slot][0]] = w.out
        rasters += w.rasters
        for k, v in w.unknown.items():
            unknown[k] = unknown.get(k, 0) + v

    # 도색은 구획 카운터 다음의 `yrvl` 청크에 있다 (그 다음 `yrvl`이 끝)
    paint = PaintState()
    if stats_tag < len(raw):
        ptag = _find(raw, b"yrvl", stats_tag + 4)
        if ptag >= 0:
            pend = _find(raw, b"yrvl", ptag + 4)
            paint = decode_paint(raw, ptag + 4, pend if pend >= 0 else len(raw))
    st = {"car_id": car_id, "counts": {SLOTS[i][0]: counts[i]
                                       for i in range(N_SLOTS) if counts[i]},
          "layers": sum(len(v) for v in out.values()), "rasters": rasters,
          "unknown": {str(k): v for k, v in sorted(unknown.items())},
          "paint_rgb": paint.car_color()}
    return out, st, paint
