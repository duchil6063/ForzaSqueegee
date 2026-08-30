"""`header` — `C_group`·`C_livery` 옆에 서는 메타데이터 곁파일.

규격은 `docs/HEADER.md`(FLS)다. 게임의 저장 그리드가 읽는 것이 이것이라
(이름·만든이·차 id·데칼 수) 컨테이너만 있고 header가 없으면 목록에 안 뜬다.

**만든이 태그**는 저장 폴더 이름에서 나온다 — WGS 저장 뿌리의 `u_<프로필id>_…`
폴더가 제 프로필 id를 이름에 달고 있어서, 거기에 쓰는 순간 그 프로필의 것이
된다 (FLS `creatorTagFromSavePath`와 같은 규칙). 그 밖의 자리에 쓰면 0이다.
"""

from __future__ import annotations

import datetime as _dt
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

from ...i18n import msg
from .binfmt import r_u16, r_u32, u16, u32

FORMAT_VERSION = 7          # FLS `kCurrentHeaderFormatVersion`
_TAG = 8                    # 만든이 신원 태그
_FIELD_BLOCK = 16
_GUID = 16
_SECTION_PREFIX = 28
_SEC3 = b"\x01\x02"


def _utf16(text: str) -> bytes:
    return str(text).encode("utf-16-le")


def _read_utf16(b: bytes, off: int, chars: int) -> str:
    n = chars * 2
    if off < 0 or off + n > len(b):
        raise ValueError(msg("header: UTF-16 문자열이 파일 끝을 넘는다"))
    return b[off : off + n].decode("utf-16-le")


@dataclass
class Header:
    format_version: int = FORMAT_VERSION
    name: str = ""
    published: bool = False
    description: str = ""
    year: int = 0
    month: int = 0
    day: int = 0
    field_block: bytes = b""
    creator_tag: bytes = b""
    creator_name: str = ""
    section_prefix: bytes = b""
    type_value: int = 0         # 데칼(레이어) 총수 — 게임 그리드가 보여 주는 수
    car_id: int = 0
    guid: bytes = b""
    trailing: bytes = b""

    def to_bytes(self) -> bytes:
        out = bytearray()
        out += u32(self.format_version)
        out += u32(len(self.name))
        out += _utf16(self.name)
        if self.published:
            out += u32(len(self.description))
            out += _utf16(self.description)
        else:
            out += u32(0)
        out += u16(self.year) + bytes((self.month & 255, self.day & 255))
        out += (self.field_block or bytes(_FIELD_BLOCK)).ljust(_FIELD_BLOCK,
                                                               b"\x00")[:_FIELD_BLOCK]
        out += (self.creator_tag or bytes(_TAG)).ljust(_TAG, b"\x00")[:_TAG]
        out += u32(len(self.creator_name))
        out += _utf16(self.creator_name)
        out += self.section_prefix[:_SECTION_PREFIX].ljust(_SECTION_PREFIX, b"\x00")
        out += _SEC3 + bytes(7)
        out += u32(self.type_value) + u32(self.car_id)
        guid = self.guid or uuid.uuid4().bytes
        out += guid.ljust(_GUID, b"\x00")[:_GUID]
        out += self.trailing
        return bytes(out)


def parse(b: bytes) -> Header:
    """`header` 바이트 → `Header` (FLS `parseHeader`와 같은 검사·경계)."""
    if len(b) < 8:
        raise ValueError(msg("header가 너무 짧다"))
    h = Header()
    off = 0
    h.format_version = r_u32(b, off); off += 4
    n = r_u32(b, off); off += 4
    h.name = _read_utf16(b, off, n); off += n * 2
    desc = r_u32(b, off); off += 4
    if desc:
        h.published = True
        h.description = _read_utf16(b, off, desc); off += desc * 2
    h.year = r_u16(b, off)
    h.month, h.day = b[off + 2], b[off + 3]
    off += 4
    h.field_block = b[off : off + _FIELD_BLOCK]
    if len(h.field_block) != _FIELD_BLOCK:
        raise ValueError(msg("header: 메타 블록에서 잘렸다"))
    off += _FIELD_BLOCK
    h.creator_tag = b[off : off + _TAG]
    if len(h.creator_tag) != _TAG:
        raise ValueError(msg("header: 만든이 태그에서 잘렸다"))
    off += _TAG
    n = r_u32(b, off); off += 4
    h.creator_name = _read_utf16(b, off, n); off += n * 2
    h.section_prefix = b[off : off + _SECTION_PREFIX]
    if len(h.section_prefix) != _SECTION_PREFIX:
        raise ValueError(msg("header: 구획 앞자리에서 잘렸다"))
    off += _SECTION_PREFIX
    if b[off : off + 2] != _SEC3:
        raise ValueError(msg("header: 구획 표식(01 02)이 제자리에 없다"))
    off += 9
    h.type_value = r_u32(b, off); off += 4
    h.car_id = r_u32(b, off); off += 4
    h.guid = b[off : off + _GUID]
    if len(h.guid) != _GUID:
        raise ValueError(msg("header: GUID에서 잘렸다"))
    off += _GUID
    h.trailing = b[off:]
    return h


def draft(name: str, creator: str = "", car_id: int = 0,
          layers: int = 0, today: _dt.date | None = None) -> Header:
    """새 초안 header (FLS `defaultDraftHeader` + 데칼 수).

    `day`가 0인 것은 FLS 그대로다 — 게임이 초안에 넣는 값이 그렇다."""
    d = today or _dt.date.today()
    fb = bytearray(_FIELD_BLOCK)
    fb[12] = 2
    return Header(format_version=FORMAT_VERSION, name=name, published=False,
                  year=d.year, month=d.month, day=0,
                  field_block=bytes(fb), creator_tag=bytes(_TAG),
                  creator_name=creator, section_prefix=bytes(_SECTION_PREFIX),
                  type_value=layers, car_id=car_id, guid=uuid.uuid4().bytes)


def creator_tag_from_save_path(path: str | Path) -> bytes:
    """저장 폴더 경로에서 만든이 태그 8바이트 — `u_<프로필id>_…`을 되짚는다."""
    cur = Path(path).resolve()
    for cand in (cur, *cur.parents):
        m = re.match(r"^u_(\d+)", cand.name, flags=re.IGNORECASE)
        if m:
            try:
                pid = int(m.group(1))
            except ValueError:
                continue
            return bytes((pid >> (8 * i)) & 0xFF for i in range(8))
    return b""
