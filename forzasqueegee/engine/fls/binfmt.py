"""FLS/게임 이진 포맷의 바닥 — 리틀엔디언 · zlib 껍데기 · 아핀 변환.

`C_group`·`C_livery`는 **게임이 제 저장 폴더에 쓰는 파일 그대로**다. 규격은
ForzaLiveryStudio(AGPL-3.0)의 문서와 코덱에서 왔고, 여기 있는 것은 그 규격을
파이썬으로 다시 쓴 것이다 (판정 기준·상수는 `docs/CGROUP.md`·`docs/CLIVERY.md`).
저쪽에서 가져온 것은 **파일이 어떻게 생겼나라는 사실**뿐이고 코드는 한 줄도
옮기지 않았다 — 그래서 이 파일은 AGPL의 파생물이 아니다 (`THIRD_PARTY_NOTICES.md`).

## 좌표·색은 우리 레이어와 1:1이다

레코드 한 장은 `u16 도형id · f32 회전 · f32 x · f32 y · f32 sx · f32 sy ·
f32 기울기 · u8 b,g,r,a`다. 우리 `engine.model.Layer`가 **게임 레코드 그대로**를
들고 있으므로 (메모리 주입이 같은 값을 +0x18/+0x28/+0x50/+0x74에 쓴다) 단위
변환이 없다 — 색 바이트 순서(BGRA ↔ RGB+알파)만 뒤집는다.

## 변환 합성은 FLS와 같은 순서다

`이동 ∘ 회전 ∘ 기울기 ∘ 스케일` (matrix_math.cpp `shapeMatrix`). 분해도 같은
규칙(`decompose`)이라 합성→분해 왕복이 제자리다 — 리버리 구획의 축 보정
(스포일러 180°)이 이 길로 간다.
"""

from __future__ import annotations

import math
import struct
import zlib

from ...i18n import msg

# ── 리틀엔디언 원시값 ────────────────────────────────────────────────

_U16 = struct.Struct("<H")
_U32 = struct.Struct("<I")
_U64 = struct.Struct("<Q")
_F32 = struct.Struct("<f")


def u16(v: int) -> bytes:
    return _U16.pack(int(v) & 0xFFFF)


def u32(v: int) -> bytes:
    return _U32.pack(int(v) & 0xFFFFFFFF)


def u64(v: int) -> bytes:
    return _U64.pack(int(v) & 0xFFFFFFFFFFFFFFFF)


def f32(v: float) -> bytes:
    """f32 한 개. NaN/inf는 0으로 — 게임이 읽는 값에 넣으면 안 된다."""
    v = float(v)
    if not math.isfinite(v):
        v = 0.0
    return _F32.pack(v)


def r_u16(b: bytes, off: int) -> int:
    return _U16.unpack_from(b, off)[0]


def r_u32(b: bytes, off: int) -> int:
    return _U32.unpack_from(b, off)[0]


def r_f32(b: bytes, off: int) -> float:
    return _F32.unpack_from(b, off)[0]


# ── zlib 껍데기 (C_group·C_livery 공통) ──────────────────────────────


def wrap_container(payload: bytes) -> bytes:
    """`u32 압축길이 · u32 원본길이 · zlib` (cgroup_codec.cpp `writeCGroupFile`)."""
    comp = zlib.compress(payload)
    return u32(len(comp)) + u32(len(payload)) + comp


def unwrap_container(blob: bytes) -> bytes:
    """껍데기를 벗긴다. 길이가 안 맞으면 이유를 대고 죽는다."""
    if len(blob) < 8:
        raise ValueError(msg("컨테이너가 껍데기(8바이트)보다 짧다"))
    n_comp, n_raw = r_u32(blob, 0), r_u32(blob, 4)
    if n_comp != len(blob) - 8:
        raise ValueError(msg("압축 길이 머리({header})가 파일 크기({actual})와 다르다",
                             header=n_comp, actual=len(blob) - 8))
    out = zlib.decompress(blob[8 : 8 + n_comp])
    if len(out) != n_raw:
        raise ValueError(msg("원본 길이 머리({header})가 실제({actual})와 다르다",
                             header=n_raw, actual=len(out)))
    return out


# ── 아핀 변환 (matrix_math.cpp 그대로) ───────────────────────────────

# 행렬은 ((a, b, c), (d, e, f)) — 세 번째 행은 늘 (0, 0, 1)이라 안 들고 다닌다.
Mat = tuple[tuple[float, float, float], tuple[float, float, float]]

IDENTITY: Mat = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))


def affine(a: float, b: float, c: float, d: float, e: float, f: float) -> Mat:
    return ((a, b, c), (d, e, f))


def is_identity(m: Mat) -> bool:
    """항등이면 합성·분해를 건너뛴다 — 왕복이 **값 그대로**여야 하는 자리가 있다
    (분해는 같은 변환의 다른 매개화를 낼 수 있다: sy<0이 rot+180·sx<0으로 선다)."""
    return m == IDENTITY


def translation_of(m: Mat) -> tuple[float, float] | None:
    """이동만인 행렬이면 (tx, ty), 아니면 None.

    중첩 그룹의 원점 변환은 늘 이동만이다 (FLS `packTranslationTransform`).
    그때는 분해를 타지 말고 좌표만 옮겨야 **매개화가 안 바뀐다** — sx·sy가 둘
    다 음수인 장이 분해를 지나면 양수 + 회전 180°로 다시 쓰인다 (같은 변환이지만
    값 대조가 거짓 경보를 낸다)."""
    (a, b, e), (c, d, f) = m
    if a == 1.0 and b == 0.0 and c == 0.0 and d == 1.0:
        return e, f
    return None


def mat_mul(l: Mat, r: Mat) -> Mat:
    (a, b, c), (d, e, f) = l
    (g, h, i), (j, k, m) = r
    return ((a * g + b * j, a * h + b * k, a * i + b * m + c),
            (d * g + e * j, d * h + e * k, d * i + e * m + f))


def normalize_rotation(v: float) -> float:
    """0 ≤ 각 < 360 (FLS `normalizeRotation` — 360에 붙으면 0으로 접는다)."""
    if not math.isfinite(v):
        return 0.0
    n = math.fmod(v, 360.0)
    if n < 0.0:
        n += 360.0
    if abs(n - 360.0) < 1e-9:
        return 0.0
    return 0.0 if n == 0.0 else n


def transform_matrix(x: float, y: float, sx: float, sy: float,
                     rot: float, skew: float) -> Mat:
    """이동 ∘ 회전 ∘ 기울기 ∘ 스케일 (`Transform2D::matrix`·`shapeMatrix`)."""
    th = math.radians(rot)
    c, s = math.cos(th), math.sin(th)
    m = affine(1.0, 0.0, x, 0.0, 1.0, y)
    m = mat_mul(m, affine(c, -s, 0.0, s, c, 0.0))
    m = mat_mul(m, affine(1.0, skew, 0.0, 0.0, 1.0, 0.0))
    return mat_mul(m, affine(sx, 0.0, 0.0, 0.0, sy, 0.0))


def decompose(m: Mat) -> tuple[float, float, float, float, float, float]:
    """행렬 → (x, y, sx, sy, rot도, skew). `decomposeTransform2D` 그대로."""
    (a, b, e), (c, d, f) = m
    x, y = e, f
    sx_mag = math.hypot(a, c)
    if sx_mag < 1e-8:
        return x, y, 0.0, math.hypot(b, d), 0.0, 0.0
    if a * d - b * c < 0.0:
        sx = -sx_mag
        rot = math.atan2(-c, -a)
    else:
        sx = sx_mag
        rot = math.atan2(c, a)
    cr, sr = math.cos(rot), math.sin(rot)
    m01 = cr * b + sr * d
    m11 = -sr * b + cr * d
    skew = m01 / m11 if abs(m11) > 1e-8 else 0.0
    return x, y, sx, m11, normalize_rotation(math.degrees(rot)), skew


def invert_affine(m: Mat) -> Mat:
    (a, b, e), (c, d, f) = m
    det = a * d - b * c
    if abs(det) < 1e-12:
        return IDENTITY
    inv = 1.0 / det
    ia, ib, ic, id_ = d * inv, -b * inv, -c * inv, a * inv
    return ((ia, ib, -(ia * e + ib * f)), (ic, id_, -(ic * e + id_ * f)))
