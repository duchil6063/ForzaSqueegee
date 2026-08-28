"""`C_group` — 게임의 비닐 그룹 저장 파일 (읽기·쓰기).

규격은 `docs/CGROUP.md`(FLS)다. 우리가 **쓰는** 것은 FLS의 정식 내보내기
(`nested_payload.cpp buildNestedPayload`)와 같은 판이다 — 뿌리 하나에 도형이
줄줄이 달린 평면 구조. 우리 도안(plan.json)이 평면 레이어 목록이라 그 이상의
구조가 없고, 게임은 그룹을 열 때 내용 상자의 중심을 커서에 맞추므로 **좌표는
내용 중심 기준 상대값**으로 나간다 (FLS와 같은 규칙 — 그래야 같은 자리에 선다).

읽기는 그보다 넓다: 뿌리 아래 **중첩 그룹**을 따라가며 그룹 변환을 도형에
합성해 평면 레이어 목록으로 편다 (게임·FLS가 쓴 그룹을 도안으로 들여오는 길).

## 마스크 비트는 **뒤 레코드**가 든다

`01 02`는 제 도형이 아니라 *앞* 도형이 뺄셈 마스크라는 뜻이다 (CGROUP.md
"trailing flag"). 그런데 앞 형제가 **그룹**이면 같은 `01`이 순회 상태(되감기)를
뜻한다 — 그래서 읽는 쪽이 "앞 형제가 도형이었나"를 들고 다닌다. 마지막 장의
마스크는 뿌리를 닫는 두 바이트의 앞자리가 든다.

## 안 하는 것

- **잠긴 그룹**(0x1D == 0x21)은 거절한다 — FLS도 같은 자리에서 막는다.
- 래스터 로고(도형 id 최상위 비트)는 카탈로그에 없는 그림이라 세고 버린다.
- 모토스포츠(FM) 방언은 안 읽는다 — 우리 대상은 FH6다.
"""

from __future__ import annotations

from ..model import Layer
from . import ids
from .binfmt import (
    IDENTITY,
    Mat,
    decompose,
    f32,
    mat_mul,
    normalize_rotation,
    r_f32,
    r_u16,
    r_u32,
    transform_matrix,
    translation_of,
    u16,
)

MAGIC = b"gyvl"
# 압축 푼 페이로드의 머리 0x1d바이트 (FLS `defaultPrefix` 그대로):
# "gyvl" + 판 1 + 예약 + 0x03(뿌리 변환 표식) + 뿌리 변환 (0,0,1,0)
DEFAULT_PREFIX = (MAGIC + b"\x01\x00\x00\x00" + b"\x00\x00\x00\x00" + b"\x03"
                  + b"\x00\x00\x00\x00" + b"\x00\x00\x00\x00"
                  + b"\x00\x00\x80\x3f" + b"\x00\x00\x00\x00")
ROOT_MARKER_OFF = 0x1D
MAX_DIRECT_CHILDREN = 0xFFFF
# 스케일 1.0인 도형의 반폭 — FLS가 `spriteSize` 없이 쓰는 기본값(128×128)과 같다.
# 내용 상자 중심을 재는 데만 쓴다 (레코드 값에는 안 들어간다).
HALF_EXTENT = 64.0


# ────────────────────────────── 레코드 ──────────────────────────────


def color_bytes(lay: Layer) -> bytes:
    """우리 RGB + 알파(0~100) → 레코드의 b, g, r, a (주입이 쓰는 그 바이트)."""
    r, g, b = (int(v) & 255 for v in lay.color)
    a = int(round(max(0.0, min(100.0, float(lay.alpha))) / 100.0 * 255.0))
    return bytes((b, g, r, max(0, min(255, a))))


def shape_payload(lay: Layer, shape_id: int, ox: float = 0.0,
                  oy: float = 0.0) -> bytes:
    """레코드 몸통 30바이트 — 도형id·회전·x·y·sx·sy·기울기·색."""
    return (u16(shape_id) + f32(normalize_rotation(lay.rot))
            + f32(lay.x - ox) + f32(lay.y - oy)
            + f32(lay.sx) + f32(lay.sy) + f32(lay.skew) + color_bytes(lay))


def usable_layers(layers: list[Layer]) -> tuple[list[tuple[Layer, int]],
                                                dict[str, int]]:
    """(레이어, 도형id) 목록 + 카탈로그 id를 모르는 도형의 이름별 장수."""
    out: list[tuple[Layer, int]] = []
    skipped: dict[str, int] = {}
    for lay in layers:
        sid = ids.id_of(lay.shape)
        if sid is None:
            skipped[lay.shape] = skipped.get(lay.shape, 0) + 1
            continue
        out.append((lay, sid))
    return out, skipped


# ────────────────────────────── 쓰기 ──────────────────────────────


def content_center(layers: list[Layer]) -> tuple[float, float]:
    """내용 상자의 중심 (FLS `shapesOrigin` — 반폭 64 고정 기준).

    도형마다 실제 뻗는 반경이 다르지만 FLS의 기본 경로와 **같은 자를 쓴다** —
    같은 도안을 두 길로 내보내도 그룹이 같은 자리에 서야 하기 때문이다."""
    if not layers:
        return 0.0, 0.0
    xs: list[float] = []
    ys: list[float] = []
    for lay in layers:
        (a, b, e), (c, d, f) = transform_matrix(
            lay.x, lay.y, lay.sx, lay.sy, lay.rot, lay.skew)
        for px, py in ((-HALF_EXTENT, -HALF_EXTENT), (HALF_EXTENT, -HALF_EXTENT),
                       (HALF_EXTENT, HALF_EXTENT), (-HALF_EXTENT, HALF_EXTENT)):
            xs.append(a * px + b * py + e)
            ys.append(c * px + d * py + f)
    return (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0


def _flip_mask(marker: bytes, prev_mask: bool) -> bytes:
    """앞 도형이 마스크면 표식의 첫 `00`이 `01`이 된다 (`maybeFlipMaskFlag`)."""
    if prev_mask and marker and marker[0] == 0x00:
        return b"\x01" + marker[1:]
    return marker


def _pack_shapes(chunk: list[tuple[Layer, int]], ox: float, oy: float,
                 prev_mask: bool) -> tuple[bytes, bool]:
    out = bytearray()
    for i, (lay, sid) in enumerate(chunk):
        if i == 0:
            out += b"\x02"                  # 첫 자식은 표식 없는 31바이트다
        else:
            out += (b"\x01" if prev_mask else b"\x00") + b"\x02"
        out += shape_payload(lay, sid, ox, oy)
        prev_mask = bool(lay.mask)
    return bytes(out), prev_mask


def _pack_subgroup(chunk: list[tuple[Layer, int]], parent: tuple[float, float],
                   marker: bytes, prev_mask: bool) -> tuple[bytes, bool]:
    """중첩 그룹 하나 — 이동만 있는 원점 변환 + 머리 + 상대 좌표 레코드들."""
    ox, oy = content_center([l for l, _ in chunk])
    blocks = (len(chunk) + 7) // 8
    out = bytearray(marker)
    out += f32(ox - parent[0]) + f32(oy - parent[1]) + f32(1.0) + f32(0.0)
    out += b"\x20" + u16(len(chunk)) + u16(blocks) + b"\x00\x00" + bytes(blocks)
    body, prev_mask = _pack_shapes(chunk, ox, oy, prev_mask)
    return bytes(out) + body, prev_mask


def encode_group(layers: list[Layer], *, center: bool = True,
                 flat_limit: int = 2040) -> tuple[bytes, dict]:
    """평면 레이어 목록 → `C_group` 페이로드(압축 전) + 통계.

    `center=False`면 좌표를 그대로 쓴다 (대조·시험용 — 그룹 파일은 늘 중심
    기준이다).

    **2,040장을 넘으면 중첩 그룹으로 나눈다.** 뿌리 머리의 비트맵 블록 칸이
    한 바이트로 읽히는 판(`docs/CGROUP.md` 0x20 u8 · FLS `getLayerData`)이
    있어서 블록이 256을 넘으면 (= 자식 2,041장) 다음 바이트로 흘러넘친다.
    중첩 그룹의 같은 칸은 어느 쪽이든 u16이라(FLS `markerlessHeaderFieldsMatch`)
    애매함이 없다 — 그래서 큰 도안은 뿌리에 그룹 몇 개만 두고 장은 그 아래
    둔다. 3,000장짜리도 이 길로 안전하게 선다."""
    usable, skipped = usable_layers(layers)
    if not usable:
        raise ValueError("내보낼 수 있는 레이어가 하나도 없다 "
                         "(카탈로그 도형 id를 아는 장이 없다)")
    if len(usable) > MAX_DIRECT_CHILDREN:
        raise ValueError(f"레이어 {len(usable):,}장 — 자식 비트맵 상한"
                         f"({MAX_DIRECT_CHILDREN:,})을 넘는다")
    ox, oy = content_center([l for l, _ in usable]) if center else (0.0, 0.0)

    n = len(usable)
    groups = 0 if n <= flat_limit else -(-n // flat_limit)
    out = bytearray(DEFAULT_PREFIX)
    if groups:
        # 고르게 나눈다 — 마지막 덩이가 한 장이면 FLS가 그룹으로 안 받는다
        step = -(-n // groups)
        chunks = [usable[i : i + step] for i in range(0, n, step)]
        chunks = [c for c in chunks if c]
        blocks = (len(chunks) + 7) // 8
        bitmap = bytearray(blocks)
        for i in range(len(chunks)):
            bitmap[i // 8] |= 1 << (i % 8)  # 자식은 전부 그룹(1)
        out += b"\x20" + u16(len(chunks)) + u16(blocks) + b"\x00\x00" + bitmap
        prev_mask = False
        for i, chunk in enumerate(chunks):
            # 첫 그룹은 표식 하나, 그 뒤는 앞 형제 그룹의 깊이(1)만큼 되감는다
            marker = b"\x03" if i == 0 else b"\x00\x01\x03"
            # 표식이 앞 도형의 마스크를 이미 들었으므로 여기서 상태를 턴다
            body, prev_mask = _pack_subgroup(
                chunk, (ox, oy), _flip_mask(marker, prev_mask), False)
            out += body
        out += _flip_mask(b"\x00\x01\x01", prev_mask)   # 뿌리 닫기 (깊이 1)
    else:
        blocks = (n + 7) // 8
        out += b"\x20" + u16(n) + u16(blocks) + b"\x00\x00" + bytes(blocks)
        body, prev_mask = _pack_shapes(usable, ox, oy, False)
        out += body
        # 뿌리 닫기 — `00 01`, 마지막 장이 마스크면 앞 바이트가 `01`이 된다
        out += _flip_mask(b"\x00\x01", prev_mask)
    st = {"layers": n, "masks": sum(1 for l, _ in usable if l.mask),
          "center": [ox, oy], "skipped": skipped, "subgroups": groups}
    return bytes(out), st


# ────────────────────────────── 읽기 ──────────────────────────────


class Walker:
    """페이로드 위를 걷는 커서 — 자식 비트맵이 무엇을 읽을지 정한다.

    리버리 구획(`livery.py`)도 이 걸음을 그대로 쓴다 — 구획 안의 문법이
    `C_group`의 그것이기 때문이다."""

    def __init__(self, data: bytes, start: int = 0):
        self.d = data
        self.p = start
        self.out: list[Layer] = []
        self.groups = 0
        self.rasters = 0
        self.unknown: dict[int, int] = {}
        self._last_shape: Layer | None = None   # 앞 형제가 낸 장 (마스크 표시 대상)

    # ── 낱개 ──
    def _skip_unwind(self) -> None:
        """그룹이 닫힌 뒤의 `00`·`01` 되감기 바이트 — 레코드 표식 앞까지 민다."""
        d = self.d
        while self.p + 1 < len(d) and d[self.p] in (0x00, 0x01) \
                and d[self.p + 1] != 0x02:
            self.p += 1

    def _read_shape(self, first: bool, prev_was_shape: bool, parent_mask: bool,
                    gm: Mat) -> None:
        d = self.d
        if not first:
            self._skip_unwind()
        p = self.p
        if p >= len(d):
            raise ValueError("도형 레코드가 없다")
        if first and d[p] == 0x02:
            lead, body = 0x00, p + 1        # 표식 없는 31바이트 (그룹의 첫 자식)
        elif p + 1 < len(d) and d[p] in (0x00, 0x01) and d[p + 1] == 0x02:
            lead, body = d[p], p + 2
        elif d[p] == 0x02:
            lead, body = 0x00, p + 1
        else:
            raise ValueError(f"0x{p:x}: 도형 레코드 표식이 아니다 "
                             f"(0x{d[p]:02x} — 자식 비트맵은 도형이라 한다)")
        if body + 30 > len(d):
            raise ValueError(f"0x{p:x}: 도형 레코드가 잘렸다")
        # `01 02`는 **앞 도형**이 마스크라는 뜻이다. 앞 형제가 그룹이었으면
        # 같은 `01`이 순회 되감기라 마스크가 아니다.
        if lead == 0x01 and prev_was_shape and self._last_shape is not None:
            self._last_shape.mask = True
        self._last_shape = self._add_shape(body, parent_mask, gm)
        self.p = body + 30

    def _add_shape(self, off: int, parent_mask: bool, gm: Mat) -> Layer | None:
        d = self.d
        sid = r_u16(d, off)
        if sid & 0x8000:                    # 래스터 로고 — 카탈로그 밖이다
            self.rasters += 1
            return None
        name = ids.name_of(sid)
        if name is None:
            self.unknown[sid] = self.unknown.get(sid, 0) + 1
            return None
        px, py = r_f32(d, off + 6), r_f32(d, off + 10)
        sxr, syr = r_f32(d, off + 14), r_f32(d, off + 18)
        rotr, skewr = r_f32(d, off + 2), r_f32(d, off + 22)
        shift = translation_of(gm)          # 부모가 이동뿐이면 좌표만 옮긴다
        if shift is not None:
            x, y = px + shift[0], py + shift[1]
            sx, sy, rot, skew = sxr, syr, rotr, skewr
        else:
            x, y, sx, sy, rot, skew = decompose(mat_mul(
                gm, transform_matrix(px, py, sxr, syr, rotr, skewr)))
        b, g, r, a = d[off + 26], d[off + 27], d[off + 28], d[off + 29]
        lay = Layer(shape=name, x=x, y=y, sx=sx, sy=sy, rot=rot, skew=skew,
                    color=(r, g, b), alpha=round(a / 255.0 * 100.0, 2),
                    mask=parent_mask)
        self.out.append(lay)
        return lay

    def _read_transform(self) -> Mat:
        """그룹 앞의 변환 — 표식(00·01·df 반복 + 03[ 03]) + f32 4개[ + 30 f32].

        `03 03`·`df 03 03` 꼴이 있어 표식의 끝이 한 바이트 애매하다. 두 해석을
        다 세워 **뒤에 그룹 머리가 오는 쪽**을 고른다 (CGROUP.md의 판정과 같다)."""
        d = self.d
        p = self.p
        while p < len(d) and d[p] in (0x00, 0x01, 0xDF):
            p += 1
        if p >= len(d) or d[p] != 0x03:
            self.p = p
            return IDENTITY                 # 변환 없이 곧장 그룹이 온다
        starts = [p + 1]
        if p + 1 < len(d) and d[p + 1] == 0x03:
            starts.append(p + 2)
        for i, start in enumerate(starts):
            if start + 16 > len(d):
                continue
            end = start + 16
            sy = None
            if end + 5 <= len(d) and d[end] == 0x30:
                sy = r_f32(d, end + 1)
                end += 5
            last = i == len(starts) - 1
            if not last and (end >= len(d) or d[end] not in (0x20, 0x60)):
                continue                    # 그룹 머리가 안 오면 다른 해석이다
            sx = r_f32(d, start + 8)
            self.p = end
            return transform_matrix(r_f32(d, start), r_f32(d, start + 4),
                                    sx, sx if sy is None else sy,
                                    r_f32(d, start + 12), 0.0)
        raise ValueError(f"0x{self.p:x}: 그룹 변환을 못 읽었다")

    def _read_group(self, parent_mask: bool, gm: Mat) -> None:
        d = self.d
        local = self._read_transform()
        p = self.p
        if p >= len(d):
            raise ValueError("그룹 머리가 없다")
        if d[p] in (0x20, 0x60):
            mask = d[p] == 0x60
            p += 1
        else:
            mask = False                    # 표식 없는 그룹 (markerless)
        if p + 6 > len(d):
            raise ValueError(f"0x{p:x}: 그룹 머리가 잘렸다")
        count = r_u16(d, p)
        blocks = r_u16(d, p + 2)
        if count <= 0 or blocks != (count + 7) // 8:
            raise ValueError(f"0x{p:x}: 자식 수 {count}와 비트맵 블록 {blocks}이 "
                             f"안 맞는다")
        bitmap = d[p + 6 : p + 6 + blocks]
        self.p = p + 6 + blocks
        self.groups += 1
        self.read_children(count, bitmap, parent_mask or mask, mat_mul(gm, local))

    def read_children(self, count: int, bitmap: bytes, parent_mask: bool,
                      gm: Mat) -> None:
        prev_was_shape = False
        for i in range(count):
            if bitmap[i // 8] >> (i % 8) & 1:
                self._read_group(parent_mask, gm)
                prev_was_shape = False
                self._last_shape = None
            else:
                self._read_shape(i == 0, prev_was_shape, parent_mask, gm)
                prev_was_shape = True

    def stats(self) -> dict:
        return {"layers": len(self.out),
                "masks": sum(1 for l in self.out if l.mask),
                "groups": self.groups, "rasters": self.rasters,
                "unknown": {str(k): v for k, v in sorted(self.unknown.items())}}


def decode_group(payload: bytes) -> tuple[list[Layer], dict]:
    """`C_group` 페이로드 → 평면 레이어 목록 + 통계 (그룹 변환은 합성한다)."""
    if len(payload) < 0x25 or payload[:4] != MAGIC:
        raise ValueError("gyvl 페이로드가 아니다")
    marker = payload[ROOT_MARKER_OFF]
    if marker == 0x21:
        raise ValueError("잠긴 그룹이다 (0x1D == 0x21) — 남의 저장본은 안 연다")
    if marker not in (0x20, 0x60, 0x00):
        raise ValueError(f"뿌리 그룹 표식이 아니다 (0x{marker:02x})")
    count = r_u16(payload, 0x1E)
    want = (count + 7) // 8
    # 뿌리 블록 칸은 판마다 u8(0x20 한 바이트)로도 u16으로도 읽힌다 — 자식이
    # 2,040장 이하면 두 해석이 같은 값이라 갈릴 일이 없고, 넘으면 u16 쪽만 선다
    if count <= 0 or want not in (payload[0x20], r_u16(payload, 0x20)):
        raise ValueError(f"뿌리 자식 수 {count}와 블록 "
                         f"{payload[0x20]}/{r_u16(payload, 0x20)}이 안 맞는다")
    blocks = want
    bitmap = payload[0x24 : 0x24 + blocks]
    scale = r_f32(payload, 0x15)
    root = transform_matrix(r_f32(payload, 0x0D), r_f32(payload, 0x11),
                            scale, scale, r_f32(payload, 0x19), 0.0)
    w = Walker(payload, 0x24 + blocks)
    w.read_children(count, bitmap, marker == 0x60, root)
    # 마지막 장의 마스크는 뿌리를 닫는 두 바이트의 앞자리가 든다 (`01 01`)
    if w.p < len(payload) and payload[w.p] == 0x01 and w.out:
        w.out[-1].mask = True
    st = w.stats()
    st["version"] = r_u32(payload, 4)
    return w.out, st
