r"""메모리 주입 경로 (58차, 사용자 지시로 허용) — FH6 비닐 그룹 레이어 직접 쓰기.

ForzaPainter가 쓰는 경로다. 창 조작 드라이버(`auto/run_plan.py`)가 장당 5.98초를
쓰는 데 반해, 이쪽은 레이어 표에 값을 바로 적어 3,000장을 몇 초에 끝낸다.
**두 경로 모두 지원한다** — 같은 `plan.json`을 읽고, 어디로 그릴지만 다르다.

    창 조작:   python -m forzasqueegee run  out/내도안/plan.json
    메모리:    python -m forzasqueegee inject out/내도안/plan.json

## 배치는 확정됐다 (61차)

`catalog/fh6_layout.json`에 `validated: true`로 들어 있고, 없으면 쓰기를 막는다.
확정 절차의 요지는 **아는 값 6장으로 되짚기**다:

    ① 값을 아는 6장짜리 플랜을 캔버스에 올린다
    ② (x, y) float 쌍을 찾는다 — 캔버스 유닛 그대로다
    ③ 후보마다 sx·sy·회전까지 **5필드 서명**으로 검증한다 (6/6 통과)
    ④ 레코드 주소를 u64로 되찾으면 **포인터 표**가 나온다
    ⑤ 한 장의 y를 바꿔 화면에서 확인하고 되돌린다

**필드 오프셋은 FH5판과 같았다.** 58차가 막힌 것은 오프셋이 아니라 **표**였다 —
FH6에는 `CLiveryGroup` RTTI가 없어 그룹 객체로 못 가고, 레코드 자체도 힙에
흩어져 있어 "등간격 배열" 가정이 서지 않는다. 이어져 있는 것은 표 쪽이다.

## 3,000장까지 닫혔다 (62차)

    python -m forzasqueegee inject out/내도안/plan.json --template   # 템플릿+주입

`--template`은 **모자란 장수를 창 조작으로 채운 뒤** 주입한다 (`auto/template.py`).
빈 캔버스에서 3,000장이면 22분 + 14초이고, 템플릿이 이미 있으면 게임을 안 건드리고
14초다. 창의 [메모리 주입] 단추도 이 길로 온다.

캔버스 캡처 ↔ 플랜 렌더 **IoU 0.9932 · 평균 색오차 2.54/255**. 창 조작 8.6시간이
23분 + 14초가 된다. 템플릿은 슬롯 `fp61joy3000`에 있으므로 다음부터는 14초다.

- **도형 id 표**는 `catalog/fh6_layout.json`의 `shape_ids`(457종)에 있고 선택
  가능·단색 353종을 전부 덮는다. 규칙은 **id = 100·페이지 + 페이지 안 번호**다
- **알파는 색 4바이트의 넷째다** — a8 128짜리 한 장으로 확인했다
- **마스크·기울기 자리는 아직 모른다** — 레코드에서 못 찾았다. 지금 파이프라인은
  두 노선 다 이 둘을 안 내지만(실측 0장), 들어오면 `apply_plan`이 **막는다** —
  그냥 쓰면 그 축만 빠진 채 그려져 플랜과 인게임이 조용히 갈린다. 배치의
  좌표하강은 기울기 축을 아예 안 민다 (`engine/celfit/scoring._descend`)

## 도형 id가 다시 연 그룹에서 안 먹던 자리 (65차, 닫힘)

레코드는 **0xA8(168)바이트**다. +0x60이 부모
그룹이고 그 그룹의 +0x5A·+0x78이 레이어 수·표라 셋이 서로 검산된다 —
`find_groups_by_record`가 그 길로 **살아 있는** 그룹만 낸다. +0x80은 이 레이어의
모델 경로 문자열(`GAME:\Media\Livery\Vinyls\<이름>.modelbin`)인데 **게임이 다시
안 읽는다** (3,000장을 통째로 바꿔도 화면이 안 변한다) — 읽기 전용 단서다.

막고 있던 것은 오프셋이 아니라 **에셋**이었다: 다시 연 비닐 그룹은 **제 저장본이
참조한 도형만** 그릴 수 있고, 그 밖의 id를 쓰면 조용히 템플릿 도형으로 그려진다.
전부 A_02로 채운 템플릿을 저장했으니 A_02밖에 못 그렸던 것이다. 고치는 자리는
주입이 아니라 **템플릿**이다 (`auto/template.seed`) — 어휘의 도형마다 한 장씩
심어 두면 다시 열어도 전 어휘가 선다.

## 주의

- **관리자 권한은 게임이 승격됐을 때만 필요하다.** 게임이 보통(비승격)으로 돌면
  같은 사용자·같은 무결성 수준이라 `OpenProcess`가 그냥 열린다 — 승격 안 된
  셸에서 3,000장을 넣어 확인했다 (2026-08-13). `--no-admin`으로 UAC를 건너뛴다
- 게임 프로세스 메모리를 고치는 행위다. 치트가 아니라 에디터가 안 주는 임포트
  기능이지만, **약관 위반 판정과 그에 따른 제재는 사용자 책임**이다
- 레이어 **개수는 늘리지 못한다.** FP와 같이 인게임에서 도형을 필요한 장수만큼
  미리 만들어 둔 그룹(템플릿)을 열고, 그 레이어들을 덮어쓰는 방식이다 —
  그 템플릿을 만드는 것이 `--template`이고, 만드는 일 자체는 창 조작이다
"""

from __future__ import annotations

import ctypes
import json
import struct
import sys
from collections.abc import Callable
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path

from ..engine.model import Layer, LayerPlan
from ..i18n import msg

MEM_COMMIT = 0x1000
MEM_PRIVATE = 0x20000
MEM_IMAGE = 0x1000000
PAGE_GUARD = 0x100
PAGE_NOACCESS = 0x01
_RW = (0x04, 0x08, 0x40, 0x80)          # READWRITE·WRITECOPY·EXECUTE_*
PROCESS_ALL = 0x0FFF | 0x100000

# 에디터 스케일 최소 스텝 0.01 — float32에서 0.00999999976이라 여유를 둔다
SCALE_MIN = 0.0099

# 반환 코드 — 0 기록함 · 1 못 찾음/실패 · 2 막음(검증·마스크·도형 id) · 3 사람이 중단
STOPPED = 3

# CLiveryGroup RTTI 이름 — 게임 실행 파일 안에 그대로 있다 (FP exe에서도 확인)
RTTI_NAME = b".?AVCLiveryGroup@@"

_k32 = ctypes.WinDLL("kernel32", use_last_error=True)
_adv = ctypes.WinDLL("advapi32", use_last_error=True)
_psapi = ctypes.WinDLL("psapi", use_last_error=True)


class _MBI(ctypes.Structure):
    _fields_ = [("BaseAddress", ctypes.c_void_p), ("AllocationBase", ctypes.c_void_p),
                ("AllocationProtect", wintypes.DWORD), ("__align", wintypes.DWORD),
                ("RegionSize", ctypes.c_size_t), ("State", wintypes.DWORD),
                ("Protect", wintypes.DWORD), ("Type", wintypes.DWORD),
                ("__align2", wintypes.DWORD)]


@dataclass(frozen=True)
class Layout:
    """레이어 구조체 배치. `catalog/fh6_layout.json`에서 읽는다 (61차 확정).

    **필드 오프셋은 FH5판과 같았다** — 막혀 있던 것은 오프셋이 아니라 표를
    못 찾은 것이었다(FH6에 `CLiveryGroup` RTTI가 없다). 레코드는 **0xA8(168)바이트**
    힙 객체이고 **연속 배열이 아니다**(프로브 6장 중 둘이 5.5MB 떨어져 있었다) —
    이어져 있는 것은 **레코드 포인터 표** 쪽이다. 이웃끼리 붙어 있는 자리도 있어
    거기서는 +0xA8이 다음 레코드의 vtable로 보인다 (65차에 크기를 그렇게 쟀다)."""
    count_offset: int = 0x5A        # 그룹 → 레이어 수 (u16) — 65차 확정
    table_offset: int = 0x78        # 그룹 → 레이어 포인터 배열 — 65차 확정
    position: int = 0x18            # 레이어 → x, y (float 2, 캔버스 유닛)
    scale: int = 0x28               # 레이어 → sx, sy (float 2)
    rotation: int = 0x50            # 레이어 → 회전 (float, 도)
    parent: int = 0x60              # 레이어 → 부모 그룹 객체 (65차)
    color: int = 0x74               # 레이어 → **RGBA** (byte 4)
    shape_id: int = 0x7A            # 레이어 → 도형 id (u16)
    model_path: int = 0x80          # 레이어 → 모델 경로 문자열(0x30) 포인터 (읽기용)
    record_stride: int = 0xA8       # 레코드 한 개 크기 (168) — 65차 실측
    blob_size: int = 0xA8
    table_by_run: bool = False      # 그룹 객체 없이 표를 모양으로 찾는다 (FH6)
    validated: bool = False         # 인게임에서 확인됐나
    shape_ids: dict | None = None   # 카탈로그 이름 → 게임 도형 id

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Layout":
        p = Path(path) if path else (Path(__file__).resolve().parents[2]
                                     / "catalog" / "fh6_layout.json")
        if not p.exists():
            return cls()
        d = json.loads(p.read_text(encoding="utf-8"))
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ------------------------------------------------------------------ 프로세스

def _enable_debug_privilege() -> bool:
    class LUID(ctypes.Structure):
        _fields_ = [("Low", wintypes.DWORD), ("High", ctypes.c_long)]

    class LAA(ctypes.Structure):
        _fields_ = [("Luid", LUID), ("Attributes", wintypes.DWORD)]

    class TP(ctypes.Structure):
        _fields_ = [("Count", wintypes.DWORD), ("Privileges", LAA * 1)]

    tok = wintypes.HANDLE()
    if not _adv.OpenProcessToken(_k32.GetCurrentProcess(), 0x20 | 0x8,
                                 ctypes.byref(tok)):
        return False
    luid = LUID()
    if not _adv.LookupPrivilegeValueW(None, "SeDebugPrivilege", ctypes.byref(luid)):
        return False
    tp = TP(1, (LAA * 1)(LAA(luid, 0x2)))
    return bool(_adv.AdjustTokenPrivileges(tok, False, ctypes.byref(tp), 0, None, None))


def find_pid(title_hint: str | None = None) -> int | None:
    """FH6 창에서 PID를 얻는다 (창 조작 경로와 같은 창을 쓴다)."""
    from . import io as gio

    hwnd = gio.find_hwnd(title_hint) if title_hint else gio.find_hwnd()
    if not hwnd:
        return None
    pid = wintypes.DWORD()
    ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return int(pid.value) or None


class Proc:
    def __init__(self, pid: int):
        _enable_debug_privilege()
        self.pid = pid
        self.h = _k32.OpenProcess(PROCESS_ALL, False, pid)
        if not self.h:
            raise OSError(msg("프로세스 열기 실패 (관리자 권한 필요) pid={pid}", pid=pid))

    def close(self) -> None:
        if self.h:
            _k32.CloseHandle(self.h)
            self.h = None

    def read(self, addr: int, size: int) -> bytes:
        buf = ctypes.create_string_buffer(size)
        got = ctypes.c_size_t()
        if not _k32.ReadProcessMemory(self.h, ctypes.c_void_p(addr), buf, size,
                                      ctypes.byref(got)):
            return b""
        return buf.raw[:got.value]

    def write(self, addr: int, data: bytes) -> bool:
        put = ctypes.c_size_t()
        return bool(_k32.WriteProcessMemory(self.h, ctypes.c_void_p(addr), data,
                                            len(data), ctypes.byref(put)))

    def u16(self, addr: int) -> int | None:
        r = self.read(addr, 2)
        return struct.unpack("<H", r)[0] if len(r) == 2 else None

    def u64(self, addr: int) -> int | None:
        r = self.read(addr, 8)
        return struct.unpack("<Q", r)[0] if len(r) == 8 else None

    def regions(self, type_filter: int, writable: bool):
        addr = 0
        mbi = _MBI()
        while addr < 0x7FFFFFFFFFFF:
            n = _k32.VirtualQueryEx(self.h, ctypes.c_void_p(addr), ctypes.byref(mbi),
                                    ctypes.sizeof(mbi))
            if not n:
                break
            base = mbi.BaseAddress or 0
            size = mbi.RegionSize
            ok = (mbi.State == MEM_COMMIT and mbi.Type == type_filter
                  and not (mbi.Protect & PAGE_GUARD)
                  and mbi.Protect != PAGE_NOACCESS
                  and (not writable or mbi.Protect in _RW))
            if ok:
                yield base, size
            addr = base + size

    def scan(self, pattern: bytes, type_filter: int, writable: bool,
             align: int = 1, limit: int = 0):
        """전 영역에서 패턴 위치를 찾는다 (영역 경계 걸침은 무시)."""
        found = 0
        for base, size in self.regions(type_filter, writable):
            if size > 512 * 1024 * 1024:
                continue
            mem = self.read(base, size)
            if not mem:
                continue
            pos = mem.find(pattern)
            while pos != -1:
                if pos % align == 0:
                    yield base + pos
                    found += 1
                    if limit and found >= limit:
                        return
                pos = mem.find(pattern, pos + 1)


# --------------------------------------------------------------- 그룹 찾기

def find_groups(p: Proc, layout: Layout, expect_count: int | None = None,
                verbose: bool = True) -> list[dict]:
    """RTTI로 CLiveryGroup vtable을 찾고, 힙에서 그 vtable을 쓰는 객체를 모은다.

    FP와 같은 절차다: RTTI 이름 → TypeDescriptor → CompleteObjectLocator →
    vtable → 힙 객체. 레이어 수·표 포인터가 그럴듯한 것만 남긴다."""
    desc = [a - 0x10 for a in p.scan(RTTI_NAME, MEM_IMAGE, False, 1, limit=4)]
    if not desc:
        if verbose:
            print(msg("CLiveryGroup RTTI를 못 찾았다 (게임 빌드가 다르거나 미실행)"))
        return []
    base = _module_base(p)
    cols: list[int] = []
    for d in desc:
        off = d - base
        if not 0 <= off <= 0xFFFFFFFF:
            continue
        for a in p.scan(struct.pack("<I", off), MEM_IMAGE, False, 4):
            col = a - 0xC
            if p.read(col, 1) == b"\x01":     # MSVC x64 COL 서명
                cols.append(col)
    vtables = []
    for col in sorted(set(cols)):
        for a in p.scan(struct.pack("<Q", col), MEM_IMAGE, False, 8):
            vtables.append(a + 8)
    vtables = sorted(set(vtables))
    if verbose:
        print(msg("RTTI {n_rtti}건 · COL {n_col}건 · vtable {n_vt}건",
                  n_rtti=len(desc), n_col=len(set(cols)), n_vt=len(vtables)))
    groups = []
    for vt in vtables:
        pat = struct.pack("<Q", vt)
        for addr in p.scan(pat, MEM_PRIVATE, True, 8):
            cnt = p.u16(addr + layout.count_offset)
            tbl = p.u64(addr + layout.table_offset)
            if not cnt or not tbl or cnt > 4000 or tbl < 0x10000:
                continue
            first = p.u64(tbl)
            if not first or first < 0x10000:
                continue
            if expect_count and cnt != expect_count:
                continue
            groups.append({"group": addr, "vtable": vt, "count": cnt, "table": tbl})
    return groups


def find_tables_by_run(p: Proc, count: int, tol: int = 0,
                       verbose: bool = True) -> list[dict]:
    """**이름 없이** 레이어 표를 찾는다 — 유효 포인터가 `count`개 연속인 자리.

    58차 실측: FH6 실행 파일에는 `CLiveryGroup` RTTI가 **없다**(4,808개 RTTI
    이름 전수 조사 — 나온 것은 전부 Steam 클라이언트 것이다). FP의 FH5 경로가
    그대로 오지 않으므로, 구조체 이름 대신 **모양**으로 찾는다: 레이어 표는
    "힙을 가리키는 8바이트 포인터가 레이어 수만큼 연속된 배열"이다.

    numpy로 영역을 u64 배열로 보고 포인터다운 값의 연속 구간을 잰다."""
    import numpy as np

    hits: list[dict] = []
    for base, size in p.regions(MEM_PRIVATE, True):
        if size < count * 8 or size > 256 * 1024 * 1024:
            continue
        mem = p.read(base, size)
        if len(mem) < count * 8:
            continue
        arr = np.frombuffer(mem[:len(mem) // 8 * 8], np.uint64)
        good = (arr > 0x10000) & (arr < 0x7FFFFFFFFFFF) & (arr % 8 == 0)
        if not good.any():
            continue
        # 연속 True 구간 길이
        idx = np.flatnonzero(np.diff(np.concatenate(([0], good.view(np.int8), [0]))))
        for s, e in zip(idx[::2], idx[1::2]):
            run = int(e - s)
            if run + tol < count:
                continue
            addr = base + int(s) * 8
            first = int(arr[s])
            second = int(arr[s + 1]) if run > 1 else 0
            hits.append({"table": addr, "run": run, "first": first,
                         "stride": second - first if second else 0})
    hits.sort(key=lambda d: abs(d["run"] - count))
    if verbose:
        print(msg("포인터 연속 {count}개 이상인 표 후보 {n}건",
                  count=count, n=len(hits)))
    return hits


MODULE_SPAN = 1 << 28           # 실행 파일 포인터로 볼 범위 (base + 256MB)


def find_groups_by_record(p: Proc, layout: Layout, expect_count: int | None = None,
                          verbose: bool = True) -> list[dict]:
    """레코드의 **모듈 포인터 서명**으로 레이어를 모으고 부모 그룹으로 묶는다.

    `find_tables_by_run`은 "포인터가 N개 연속"이라는 모양만 보므로 두 가지로
    샌다: ① 지운 그룹의 잔재가 후보에 섞이고(64차) ② 진짜 표가 긴 잡동사니
    구간 안에 들어앉으면 16점 선별에서 통째로 떨어진다 (65차 실측: 6장짜리
    그룹에서 후보 13건이 전부 죽은 표였고 살아 있는 표는 아예 안 나왔다).

    여기서는 **레코드 쪽에서 출발한다**. 레이어 레코드는 +0x10·0x20·0x30·0x40이
    전부 같은 실행 파일 포인터(필드 래퍼 vtable)이고 +0x00·0x08도 실행 파일
    포인터다 — 사유 힙에서 이 서명은 사실상 레이어 레코드에만 있다. 모은
    레코드를 **부모(+0x60)** 로 묶으면 그룹이 자기 표를 스스로 증명한다:

        부모(+0x5A) == 레코드 수 · 부모(+0x78) == 표 · 표의 원소 == 그 레코드들

    셋이 서로 맞물리므로 죽은 표는 통과할 수 없다 (그 표를 가리키는 산 그룹이
    없다). 반환은 `{group, table, count}` 목록이고 `expect_count`를 주면 그
    장수인 것만 낸다."""
    import numpy as np

    base = _module_base(p)
    if not base:
        return []
    lo, hi = np.uint64(base), np.uint64(base + MODULE_SPAN)
    recs: dict[int, list[int]] = {}       # 부모 → 레코드 주소
    for addr, size in p.regions(MEM_PRIVATE, True):
        if size < layout.record_stride or size > 512 * 1024 * 1024:
            continue
        mem = p.read(addr, size)
        if len(mem) < layout.record_stride:
            continue
        a = np.frombuffer(mem[:len(mem) // 8 * 8], np.uint64)
        n = len(a) - (layout.parent // 8) - 1
        if n <= 0:
            continue
        v = a[2:n + 2]
        ok = ((v == a[4:n + 4]) & (v == a[6:n + 6]) & (v == a[8:n + 8])
              & (v >= lo) & (v < hi)
              & (a[0:n] >= lo) & (a[0:n] < hi)
              & (a[1:n + 1] >= lo) & (a[1:n + 1] < hi))
        idx = np.flatnonzero(ok)
        if not len(idx):
            continue
        par = a[layout.parent // 8:][idx]
        for j, g in zip(idx.tolist(), par.tolist()):
            if 0x10000 < g < 0x7FFFFFFFFFFF:
                recs.setdefault(int(g), []).append(addr + j * 8)
    out = []
    for g, rs in recs.items():
        cnt = p.u16(g + layout.count_offset)
        tbl = p.u64(g + layout.table_offset)
        if not cnt or not tbl or tbl < 0x10000 or cnt != len(rs):
            continue
        raw = p.read(tbl, cnt * 8)
        if len(raw) != cnt * 8:
            continue
        if set(struct.unpack(f"<{cnt}Q", raw)) != set(rs):
            continue
        if expect_count and cnt != expect_count:
            continue
        out.append({"group": g, "table": tbl, "count": cnt})
    out.sort(key=lambda d: -d["count"])
    if verbose:
        print(msg("레코드 서명으로 찾은 살아 있는 그룹 {n}건", n=len(out))
              + (msg(" (레이어 {count}장)", count=expect_count)
                 if expect_count else "")
              + "".join(msg("\n  그룹 0x{group:x} · 표 0x{table:x} · {count}장",
                            group=d["group"], table=d["table"], count=d["count"])
                        for d in out[:6]))
    return out


def _read_span(layout: Layout) -> int:
    """레코드에서 **실제로 읽는 끝자리** (352바이트 통째가 아니다 — 아래 이유).

    `blob_size`(0x160)만큼 읽으면 **페이지 끝에 걸린 레코드가 통째로 떨어진다**.
    `ReadProcessMemory`는 범위 중 한 바이트라도 못 읽으면 전부 실패인데, 힙 객체가
    페이지 경계를 넘고 다음 페이지가 안 잡혀 있으면 그렇게 된다. 실측: 수천 장을
    주입한 뒤 주소가 페이지 끝에 걸린 레코드 몇 개가 0x160으로는 실패하고 0x7c로는
    값까지 플랜과 정확히 맞았다 — 게임이 지운 것이 아니라 **우리가 너무 많이 읽은
    것**이었다. 필드는 전부 앞쪽 0x7c 안에 있으므로 그만 읽는다."""
    return max(layout.position + 8, layout.scale + 8, layout.rotation + 4,
               layout.color + 4, layout.shape_id + 2)


def _record(p: Proc, addr: int, layout: Layout) -> dict | None:
    span = _read_span(layout)
    raw = p.read(addr, span)
    if len(raw) != span:
        return None
    x, y = struct.unpack_from("<ff", raw, layout.position)
    sx, sy = struct.unpack_from("<ff", raw, layout.scale)
    (rot,) = struct.unpack_from("<f", raw, layout.rotation)
    if any(v != v for v in (x, y, sx, sy, rot)):      # NaN
        return None
    return {"addr": addr, "x": x, "y": y, "sx": sx, "sy": sy, "rot": rot,
            "rgba": tuple(raw[layout.color:layout.color + 4]),
            "shape_id": struct.unpack_from("<H", raw, layout.shape_id)[0]}


def _plausible_record(p: Proc, addr: int, layout: Layout) -> bool:
    """레이어 레코드로 보이나 — **에디터 실제 범위**로 본다.

    범위를 헐겁게 잡으면 안 된다: `0 < sx`처럼 두면 1e-30 같은 쓰레기가 다 통과해
    6장짜리 표를 찾을 때 후보가 7,000건씩 남는다 (실측). 캔버스는 900유닛
    높이이고 에디터 스케일은 0.01~100이다.

    **아래 끝은 0.01이 아니라 `SCALE_MIN`이다.** 스케일 최소 스텝 0.01은 float32로
    0.00999999976이라 `0.01 <= sx`로 자르면 **가장 작은 도형이 통째로 떨어진다** —
    `painter` 3,000장에는 그런 레이어가 늘 있어서, 주입한 그룹을 다시 찾으려 하면
    표를 못 찾는다 (62차 실측).

    **스케일은 크기로 본다 (`abs`).** 음수 스케일은 좌우/상하 반전이고 에디터가
    정상으로 쓰는 값이다 — cel 노선의 곡선 아핀 맞춤이 미러를 자유롭게 고르므로
    우리 플랜의 8~19%가 여기 걸린다(실측). 부호로 자르면 8점 검사가 통과할
    확률이 18~49%로 떨어져
    **이미 주입한 그룹의 표를 다시 못 찾는다**(이어 주입 후보 0건의 원인).
    같은 표본에서 크기가 범위를 벗어나는 레이어는 0장이라, 자르는 것은 부호뿐이다."""
    r = _record(p, addr, layout)
    if r is None:
        return False
    return (abs(r["x"]) <= 5000.0 and abs(r["y"]) <= 5000.0
            and SCALE_MIN <= abs(r["sx"]) <= 100.0
            and SCALE_MIN <= abs(r["sy"]) <= 100.0
            and -0.5 <= r["rot"] <= 360.5 and r["rgba"][3] > 0)


def match_plan(p: Proc, table: int, plan: LayerPlan, layout: Layout,
               tol: float = 2e-3) -> int:
    """표의 레코드가 플랜의 값과 몇 장이나 맞나 (확정용 자).

    아는 도안을 캔버스에 올려 두고 이걸 쓰면 표가 맞는지 의심할 여지가 없다."""
    ok = 0
    for i, lay in enumerate(plan.layers):
        ptr = p.u64(table + i * 8)
        r = _record(p, ptr, layout) if ptr else None
        if r and all(abs(r[k] - getattr(lay, k)) < tol
                     for k in ("x", "y", "sx", "sy", "rot")):
            ok += 1
    return ok


def find_layer_table(p: Proc, count: int, layout: Layout,
                     expect: LayerPlan | None = None,
                     verbose: bool = True) -> list[int]:
    """레이어 포인터 표를 찾는다 (FH6 경로 — 그룹 객체를 안 거친다).

    `find_tables_by_run`이 낸 후보를 **레코드가 레이어처럼 생겼는가**로 거른다.
    포인터가 연속인 자리는 힙에 흔하지만, 그 끝이 전부 에디터 범위 안의 변환을
    담고 있는 자리는 드물다. `expect`를 주면 값까지 대 보므로 확실해진다.

    **표는 연속 구간의 첫 자리가 아닐 수 있다** (62차, 3,000장에서 처음 나왔다).
    표 바로 앞의 포인터가 우연히 "포인터답게" 생기면 구간이 그만큼 앞으로
    늘어난다 — 3,000장 실측에서 구간 3,002 중 **+2번째**가 표였다. 6·46·206장에서
    안 걸린 것은 운이었지 규칙이 아니다. 그래서 구간 안을 **밀어 가며** 본다:
    먼저 구간 전체에서 16개를 뽑아 대충 레이어 표인지 보고(213/214가 여기서
    떨어진다), 살아남은 것만 시작 자리를 하나씩 밀며 8점 검사를 한다.

    **먼저 `find_groups_by_record`를 쓴다** (65차). 그쪽은 레코드 서명 →
    부모 그룹 → 표를 서로 검산하므로 죽은 표가 원리적으로 안 섞이고, 모양
    검색이 통째로 놓치는 자리(잡동사니 구간 안의 표)도 찾는다. 아래 모양
    검색은 서명이 안 걸릴 때를 위한 갈래로 남긴다."""
    live = [g["table"] for g in find_groups_by_record(p, layout, count, verbose)]
    if live:
        if expect is not None:
            ok = [t for t in live if match_plan(p, t, expect, layout) == count]
            if ok:
                return ok
        else:
            return live
    out = []
    for t in find_tables_by_run(p, count, verbose=verbose):
        addr, run = t["table"], t["run"]
        raw = p.read(addr, run * 8)
        if len(raw) < count * 8:
            continue
        ptrs = struct.unpack(f"<{len(raw) // 8}Q", raw[:len(raw) // 8 * 8])
        # ① 싼 선별 — 구간 전체에서 16개를 떠 본다 (대부분 여기서 떨어진다)
        probe = [ptrs[int(i * (len(ptrs) - 1) / 15)] for i in range(16)]
        if sum(_plausible_record(p, q, layout) for q in probe) < 12:
            continue
        # ② 시작 자리를 밀며 8점 검사
        step = max(1, count // 8)
        for off in range(len(ptrs) - count + 1):
            sub = ptrs[off:off + count]
            if len(set(sub)) != count:       # 같은 레코드를 두 번 가리키면 표가 아니다
                continue
            if not all(_plausible_record(p, sub[i], layout)
                       for i in range(0, count, step)):
                continue
            here = addr + off * 8
            if expect is not None and match_plan(p, here, expect, layout) != count:
                continue
            out.append(here)
            break
    if verbose:
        print(msg("레이어 표로 보이는 자리 {n}건", n=len(out))
              + (msg(" (플랜 값까지 일치)") if expect is not None else ""))
    if not out and expect is not None:      # 낡은 포인터가 섞인 배열 — 값으로 되짚는다
        out = find_table_by_anchor(p, expect, layout, verbose)
    return out


def find_table_by_anchor(p: Proc, plan: LayerPlan, layout: Layout,
                         verbose: bool = True) -> list[int]:
    """**첫 레이어의 값으로 되짚어** 표를 찾는다 (모양 검색이 놓칠 때).

    `find_layer_table`의 싼 선별은 구간 전체에서 16개를 떠 본다. 그런데 포인터
    배열은 **더 컸던 그룹의 낡은 포인터**를 뒤에 달고 있을 수 있다 — 3,000장
    그룹을 지우고 526장을 새로 만든 자리에서 실측으로 그랬다. 그러면 표본이
    죽은 레코드에 떨어져 진짜 표가 통째로 떨어진다.

    여기서는 **아는 값**에서 출발하므로 그 문제가 없다 (61차 확정 절차와 같은
    길이다): ① 1번 레이어의 (x, y) float 쌍을 찾고 ② 5필드로 레코드를 확인하고
    ③ 그 주소를 가리키는 u64를 찾아 표 시작으로 삼고 ④ 플랜 전체로 검산한다.
    첫 레이어는 값이 **뚜렷한 앵커**여야 한다 (기본값 0,0,1,1,0이면 후보가 쏟아진다)."""
    n = len(plan.layers)
    a = plan.layers[0]
    pat = struct.pack("<ff", a.x, a.y)
    recs = []
    for hit in p.scan(pat, MEM_PRIVATE, True, 4):
        addr = hit - layout.position
        r = _record(p, addr, layout)
        if r and all(abs(r[k] - getattr(a, k)) < 2e-3
                     for k in ("sx", "sy", "rot")):
            recs.append(addr)
    if verbose:
        print(msg("1번 레이어 레코드 후보 {n}건", n=len(recs)))
    out = []
    for addr in recs:
        for t in p.scan(struct.pack("<Q", addr), MEM_PRIVATE, True, 8):
            if match_plan(p, t, plan, layout) == n:
                out.append(t)
    if verbose:
        print(msg("표 {n_table}건 (플랜 {n}장 전부 일치)", n_table=len(out), n=n))
    return sorted(set(out))


def find_table_by_sentinel(p: Proc, layout: Layout, xy: tuple[float, float],
                           count: int, verbose: bool = True) -> list[int]:
    """**마지막 레이어에 방금 박은 값**으로 표를 못 박는다 (소형 캔버스용).

    장수 검색은 스테일 잔재와 겹친다 (12장 후보 51건·2장 9,344건 — 실측).
    방금 위저드로 박은 (x, y)는 살아 있는 표의 마지막 레코드에만 있으므로:
    ① 그 float 쌍을 찾고 ② 레코드로 그럴듯한지 보고 ③ 그 주소를 가리키는
    포인터에서 `count-1` 슬롯을 되짚어 표 시작을 얻고 ④ 표의 포인터 표본이
    전부 레코드로 그럴듯한지 검산한다.
    """
    pat = struct.pack("<ff", xy[0], xy[1])
    recs = []
    for hit in p.scan(pat, MEM_PRIVATE, True, 4):
        addr = hit - layout.position
        if _plausible_record(p, addr, layout):
            recs.append(addr)
    if verbose:
        print(msg("센티널 레코드 후보 {n}건", n=len(recs)))
    out = []
    for addr in recs:
        for ptr_at in p.scan(struct.pack("<Q", addr), MEM_PRIVATE, True, 8):
            t0 = ptr_at - (count - 1) * 8
            raw = p.read(t0, count * 8)
            if len(raw) != count * 8:
                continue
            ptrs = struct.unpack(f"<{count}Q", raw)
            if not all(ptrs):
                continue
            # 표본 검산 — 앞 4·뒤 4 슬롯이 레코드로 그럴듯한가
            idx = list(range(min(4, count))) + list(range(max(0, count - 4), count))
            if all(_plausible_record(p, ptrs[i], layout) for i in set(idx)):
                out.append(t0)
    out = sorted(set(out))
    if verbose:
        print(msg("센티널 표 {n}건 ({count}장)", n=len(out), count=count))
    return out


def find_folded_table(p: Proc, layout: Layout, xy: tuple[float, float],
                      expect: int, verbose: bool = True) -> tuple[int, int] | None:
    """**다시 연 저장 그룹**의 접힌 그룹 **(내부 표, 장수)**를 찾는다 (2026-08-24).

    '내 비닐 그룹'에서 다시 연 그룹은 편집 캔버스에 **"1-N 접힌 그룹"**으로
    들어온다 — 레이어가 평면 표가 아니라 **그룹 안의 그룹**이라, 평면 표를
    가정하는 `find_layer_table`·`find_table_by_sentinel`이 통째로 실패한다
    (센티널 레코드는 찾아도 표는 0건).

    구조 (센티널 한 장을 위저드로 심어 놓고 되짚는다):

        센티널 레코드 → 부모(+0x60) = **G_top** (편집 캔버스 그룹)
        G_top 표(+0x78)의 원소 중 하나 = **접힌 그룹 객체** (count == expect)
        접힌 그룹의 표(+0x78) = **내부 표** ← 실제 N장이 여기 있다

    `expect`는 접힌 그룹의 원래 장수 힌트다 — 정확 매칭을 우선하되 안 맞으면
    가장 큰 그럴듯한 그룹으로 물러난다 (재저장이 센티널을 함께 저장해
    `canvas_count`가 접힌 count보다 큰 등). **실제 장수를 함께 돌려주므로**
    부르는 쪽은 이 값으로 표를 읽어야 초과분을 안 건드린다. 내부 표에 쓰면
    화면이 바뀐다 (편집 캔버스의 렌더 소스 — 실측 화면차분 8.53).
    """
    pat = struct.pack("<ff", xy[0], xy[1])
    for hit in p.scan(pat, MEM_PRIVATE, True, 4):
        srec = hit - layout.position
        r = _record(p, srec, layout)
        if not (r and abs(r["x"] - xy[0]) < 1e-2 and abs(r["y"] - xy[1]) < 1e-2):
            continue
        gtop = p.u64(srec + layout.parent)
        if not gtop or gtop < 0x10000:
            continue
        gcount = p.u16(gtop + layout.count_offset)
        gtable = p.u64(gtop + layout.table_offset)
        if not gtable or gtable < 0x10000 or not gcount or gcount > 5000:
            continue
        raw = p.read(gtable, gcount * 8)
        if len(raw) != gcount * 8:
            continue
        # G_top 표 원소 중 **내부 표가 그럴듯한 그룹**을 고른다. `expect`(원본
        # 장수)와 정확히 맞는 것을 우선하되, 안 맞으면(재저장으로 접힌 count가
        # 바뀐 그룹 등) **가장 큰 그럴듯한 그룹**으로 물러난다 — G_top 표에는
        # 접힌 그룹 하나와 센티널·잡동사니뿐이라 가장 큰 유효 그룹이 곧 그것이다.
        best: tuple[int, int] | None = None
        for e in struct.unpack(f"<{gcount}Q", raw):
            if not e or e < 0x10000:
                continue
            ecount = p.u16(e + layout.count_offset)
            etable = p.u64(e + layout.table_offset)
            if not ecount or ecount < 2 or ecount > 4000 or not etable \
                    or etable <= 0x10000:
                continue
            iraw = p.read(etable, min(20, ecount) * 8)
            if len(iraw) < min(20, ecount) * 8:
                continue
            iptrs = struct.unpack(f"<{min(20, ecount)}Q", iraw)
            if sum(_plausible_record(p, q, layout) for q in iptrs) < 15:
                continue
            if ecount == expect:
                if verbose:
                    print(msg("접힌 그룹 내부 표 0x{etable:x} ({ecount}장, "
                              "expect 일치) — G_top 0x{gtop:x}",
                              etable=etable, ecount=ecount, gtop=gtop))
                return etable, ecount
            if best is None or ecount > best[1]:
                best = (etable, ecount)
        if best is not None:
            if verbose:
                print(msg("접힌 그룹 내부 표 0x{etable:x} ({ecount}장, expect "
                          "{expect} 불일치 — 최대 그럴듯 그룹) — G_top 0x{gtop:x}",
                          etable=best[0], ecount=best[1], expect=expect, gtop=gtop))
            return best
    if verbose:
        print(msg("접힌 그룹 내부 표를 못 찾았다 (expect {expect}장)", expect=expect))
    return None


def _module_base(p: Proc) -> int:
    arr = (ctypes.c_void_p * 1024)()
    need = wintypes.DWORD()
    if not _psapi.EnumProcessModules(p.h, arr, ctypes.sizeof(arr), ctypes.byref(need)):
        return 0
    return int(arr[0] or 0)


# ------------------------------------------------------------------- probe

def read_layers(p: Proc, table: int, count: int, layout: Layout) -> list[dict]:
    """표에서 레코드를 꺼내 확정된 배치로 해독한다 (읽기 전용)."""
    out = []
    tbl = p.read(table, count * 8)
    ptrs = struct.unpack(f"<{count}Q", tbl) if len(tbl) == count * 8 else ()
    for i in range(count):
        ptr = ptrs[i] if i < len(ptrs) else p.u64(table + i * 8)
        if not ptr:
            continue
        raw = p.read(ptr, _read_span(layout))       # 페이지 끝에 걸린 레코드 대응
        if len(raw) != _read_span(layout):
            continue
        x, y = struct.unpack_from("<ff", raw, layout.position)
        sx, sy = struct.unpack_from("<ff", raw, layout.scale)
        (rot,) = struct.unpack_from("<f", raw, layout.rotation)
        out.append({"i": i, "addr": ptr, "x": x, "y": y, "sx": sx, "sy": sy,
                    "rot": rot,
                    "rgba": tuple(raw[layout.color:layout.color + 4]),
                    "shape_id": struct.unpack_from("<H", raw, layout.shape_id)[0]})
    return out


def probe(expect_count: int | None = None, dump: int = 8,
          expect_plan: str | Path | None = None) -> int:
    """읽기 전용 조사 — 레이어 표를 찾고 확정된 배치로 값을 해독해 보여 준다.

    `--count`로 캔버스의 레이어 수를 준다. **캔버스에 올린 플랜을 `--plan`으로
    같이 주면** 값까지 대 보므로 표가 맞는지 의심할 여지가 없다 (확정 절차)."""
    pid = find_pid()
    if not pid:
        print(msg("FH6 창을 못 찾았다"))
        return 1
    p = Proc(pid)
    layout = Layout.load()
    print(f"pid={pid} base=0x{_module_base(p):x} layout.validated={layout.validated}")
    plan = LayerPlan.load(expect_plan) if expect_plan else None
    if plan is not None and not expect_count:
        expect_count = len(plan.layers)
    if not expect_count:
        print(msg("레이어 수를 줄 것: `inject --probe --count <장수>` "
                  "(또는 `--plan <올려 둔 plan.json>`)"))
        p.close()
        return 2
    tables = (find_layer_table(p, expect_count, layout, expect=plan)
              if layout.table_by_run
              else [g["table"] for g in find_groups(p, layout, expect_count)])
    for t in tables[:4]:
        print(msg("\n표 0x{t:x}", t=t))
        for r in read_layers(p, t, min(dump, expect_count), layout)[:dump]:
            print(f"  [{r['i']:4d}] @0x{r['addr']:x}  "
                  f"pos({r['x']:9.2f},{r['y']:9.2f}) "
                  f"scale({r['sx']:6.3f},{r['sy']:6.3f}) rot {r['rot']:7.2f}  "
                  f"rgba{r['rgba']} shape 0x{r['shape_id']:04x}")
    p.close()
    return 0


# ------------------------------------------------------------------- apply

def apply_plan(plan_path: str | Path, *, force: bool = False,
               layout_path: str | Path | None = None,
               table: int | None = None, template: bool = False,
               canvas: int | None = None, prepare: bool = True,
               stop: Callable[[], bool] | None = None,
               reuse: bool = True, avoid: set[int] | None = None) -> int:
    """`plan.json`의 레이어 값을 열려 있는 비닐 그룹에 직접 쓴다.

    `template=True`면 쓰기 전에 **캔버스 장수를 플랜에 맞춘다** — 모자란 만큼
    창 조작으로 채운다(`auto/template.py`). 빈 캔버스에 그냥 쓸 수는 없다:
    주입은 값을 덮을 뿐 레이어를 못 만든다. 중단은 `stop`이 참을 낼 때이고,
    안 주면 **`run`과 같은 규약** — 플랜 폴더의 `STOP` 파일이다.

    `prepare=True`(기본)면 쓰기 전에 **캔버스를 쓸 수 있는 상태로 만든다** —
    없으면 만들고, 맞으면 그대로 쓰고, 씨앗이 틀렸으면 다시 심는다
    (`auto/template.ensure_ready`). 맞는 템플릿이면 게임을 아예 안 건드린다.
    `prepare=False`는 값만 쓴다 (준비 자신이 주입을 부르므로 그 길이 필요하다).

    `reuse=False`면 준비가 심는 씨앗을 **이 플랜이 쓰는 도형만**으로 좁힌다 —
    만들고 저장하면 끝인 그룹(이타샤)에 쓴다. 다시 열어 다른 플랜을 올릴
    템플릿은 기본값(어휘 전체)이라야 한다.

    `avoid`는 **이미 쓰인 캔버스 장수**다 (`auto.itasha`의 저장 그룹 장수).
    준비가 내놓은 장수가 그 안에 있으면 **주입 전에** 한 장 더 채워 비켜 간다 —
    장수가 곧 그룹의 신원이라 겹치면 불러오기가 남의 그룹을 문다. 주입 **뒤에**
    고치면 안 되는 이유가 둘이다: 준비를 다시 돌게 되는데 그때 `seed_missing`이
    **이미 주입된 캔버스**를 보고 "이 도형을 못 그린다"로 잘못 판정하고,
    두 번째 주입은 센티널을 다시 못 심어 표 후보가 여럿인 채로 첫 번째에 써
    **지운 그룹의 잔재**를 건드린다 (둘 다 2026-08-21 실측).

    **캔버스가 플랜보다 크면 알아서 맞춘다** — 화면의 레이어 카운터를 읽어
    남는 자리를 밀어낸다. `canvas=M`은 그 값을 사람이 못 박는 자리다 (카운터가
    안 보이는 화면에서 쓰거나, 읽은 값을 못 믿을 때).

    그 길은 **플랜보다 큰 그룹에 그냥 쓰는 것**이다. 가격 설계는 장수가
    그림마다 다른데(998~2,033장), 남는 레이어를 **지울 필요가 없다** — 주입이
    알파를 쓰므로(레코드의 색 넷째 바이트) 남는 자리에 **알파 0·최소 크기·
    캔버스 밖** 레코드를 써 넣으면 안 보인다. 그래서 3,000장 템플릿 하나를
    계속 열어 두고 어떤 플랜이든 바로 올릴 수 있다 (그림마다 22분씩 드는
    템플릿 채우기가 통째로 없어진다).

    안 주면 표를 플랜 장수로 찾고 플랜 장수만 쓴다.

    **지운 그룹의 표가 메모리에 남는다.** 3,000장 그룹을 지우고 526장을 새로
    만든 자리에서 후보가 6건 나왔고, 첫 번째는 **지운 그룹의 잔재**였다
    (64차 실측 — 거기에 쓰면 화면이 안 바뀐다). 후보가 여럿이면 `--table`로
    자리를 지정할 것. 자리는 `inject --probe --plan <올려 둔 플랜>`이 낸다."""
    layout = Layout.load(layout_path)
    if not layout.validated and not force:
        print(msg("레이어 배치가 아직 FH6에서 검증되지 않았다.\n"
                  "  1) 인게임에서 아는 그룹을 열고  2) `inject --probe`로 오프셋 확인\n"
                  "  3) catalog/fh6_layout.json에 적고 validated=true\n"
                  "검증 없이 쓰면 그룹이 깨질 수 있다 (--force로 무시)"))
        return 2
    plan = LayerPlan.load(plan_path)
    n_mask = sum(1 for l in plan.layers if l.mask)
    if n_mask and not force:
        print(msg("마스크 레이어가 {n_mask}장 들어 있다 — 레코드의 마스크 자리를 아직\n"
                  "  모른다. 주입하면 일반 레이어로 그려져 플랜과 인게임이 조용히 갈린다.\n"
                  "  창 조작(`run`)은 마스크 위저드를 쓰므로 그쪽으로 올릴 것 (--force로 무시)",
                  n_mask=n_mask))
        return 2
    n_skew = sum(1 for l in plan.layers if abs(l.skew) > 1e-9)
    if n_skew and not force:
        print(msg("기울기가 든 레이어가 {n_skew}/{total}장 있다 — 레코드의 "
                  "기울기 자리를\n  모른다. 마스크와 같은 사정인데 이쪽은 오래 안 막혀"
                  " 있었다: 주입이 기울기만\n  빼고 나머지를 써서 **플랜 렌더와 인게임이"
                  " 조용히 갈렸다** (실측 7장에서 픽셀\n  0.70~1.89% · 셀 일치도 lpips"
                  " +0.0026~+0.0185). 창 조작(`run`)도 그 도구가 없어\n  멈춘다 — 기울기를"
                  " 안 내는 판으로 다시 구울 것 (--force로 무시)",
                  n_skew=n_skew, total=len(plan.layers)))
        return 2
    ids = layout.shape_ids or {}
    unknown = sorted({l.shape for l in plan.layers} - set(ids))
    if unknown and not force:
        print(msg("게임 도형 id를 모르는 도형이 있다: {unknown}\n"
                  "  주입은 도형을 못 바꾸므로 그 레이어는 템플릿의 도형 그대로 그려진다.\n"
                  "  `painter`는 A_02 하나만 쓴다 — 다른 도형을 쓰는 플랜은 id 표가 먼저다"
                  " (--force로 무시)", unknown=unknown))
        return 2
    n = len(plan.layers)
    if template and canvas is not None:
        print(msg("--template과 --canvas는 같이 못 쓴다 — 앞은 캔버스를 플랜 장수에"
                  " 맞추고 뒤는 큰 캔버스를 그대로 쓴다. 하나만 줄 것"))
        return 2
    # 표를 찾는 자가 곧 **캔버스 장수**다 (플랜 장수가 아니다). 남는 자리는
    # 아래에서 안 보이는 레코드로 덮으므로, 캔버스가 커도 그대로 쓴다
    n_slot = n
    if canvas is not None:
        if canvas < n:
            print(msg("캔버스 {canvas}장이 플랜 {n}장보다 적다 — 모자란 만큼은 "
                      "못 올린다 (준비를 켜거나 큰 그룹을 열 것)", canvas=canvas, n=n))
            return 1
        n_slot = canvas
    elif template or prepare:
        # 막는 검사(위)를 다 통과한 뒤에 준비한다 — 22분을 쓰고 나서 거부하면 안 된다
        from ..auto.driver import DriverError
        from ..auto.run_plan import StopRequested
        from ..auto import template as T

        if stop is None:
            stop_file = Path(plan_path).parent / "STOP"
            stop = stop_file.exists
        try:
            got = (T.ensure(n, stop=stop) if template else
                   T.ensure_ready(n, tuple(sorted({l.shape for l in plan.layers})),
                                  stop=stop, reuse=reuse))
        except StopRequested as e:
            print(msg("{err} — 주입은 안 했다", err=e))
            return STOPPED
        except DriverError as e:
            print(msg("템플릿을 못 맞췄다: {err}", err=e))
            return 1
        if got and avoid:
            # **신원 충돌은 주입 전에 비켜 간다** (위 독스트링). 남는 자리는
            # 아래에서 안 보이는 덮개로 덮으므로 그림이 안 바뀐다.
            #
            # **`got`과 `got+1`이 둘 다 비어야 한다** — 아래 센티널 경로가
            # 표를 못 박느라 레이어를 한 장 더 심을 수 있고(그때 신원은 `got+1`이
            # 된다), 그 결정은 여기보다 뒤에서 난다. 두 자리를 다 확보해 두면
            # 센티널이 서든 안 서든 신원이 안 겹친다.
            from ..auto.driver import Driver

            for _ in range(12):
                if got not in avoid and (got + 1) not in avoid:
                    break
                print(msg("캔버스 {got:,}장은 이미 쓰인 그룹 신원이다 (또는 센티널 "
                          "자리 {sentinel:,}장이) — 한 장 더 채워 비켜 간다",
                          got=got, sentinel=got + 1))
                try:
                    got = T.fill(Driver(), got + 1, stop=stop)
                except (DriverError, StopRequested) as e:
                    print(msg("신원 비켜 가기 실패 ({err}) — 그대로 간다", err=e))
                    break
        if got:
            n_slot = max(n, got)
    else:
        # 준비를 껐어도 **캔버스 장수는 읽는다** — 카운터는 창을 안 뺏는다.
        # 잘못 읽어도 안전한 쪽으로만 움직인다: 작게 읽히면 플랜 장수 경로이고,
        # 크게 읽히면 그 장수짜리 표가 없어 "못 찾았다"로 멈춘다
        try:
            from ..auto.template import canvas_count

            m = canvas_count()
        except Exception:
            m = None
        if m and m > n:
            n_slot = m
            print(msg("캔버스가 {m}장이다 (플랜 {n}장) — 남는 {extra}장은 밀어낸다",
                      m=m, n=n, extra=m - n))
    pid = find_pid()
    if not pid:
        print(msg("FH6 창을 못 찾았다"))
        return 1
    p = Proc(pid)
    if table is not None:
        tables = [table]
    elif layout.table_by_run:
        tables = find_layer_table(p, n_slot, layout)
    else:
        tables = [g["table"] for g in find_groups(p, layout, expect_count=n_slot)]
    # **후보가 하나가 아니면 센티널로 못 박는다** — 소형 캔버스(수십 장)는
    # 장수 검색이 스테일 잔재와 수십~수천 건 겹친다 (12장 51건·2장 9,344건 —
    # 실측). 위저드로 레이어 한 장을 더 만들어 아는 값을 박고 그 값으로
    # 되짚으면 살아 있는 표만 남는다. 준비 흐름을 우리가 쥐고 있을 때만
    # (창이 레이어 리스트 상태라는 보장이 그때뿐이다).
    if len(tables) != 1 and (prepare or template) and canvas is None:
        import time as _time

        print(msg("표 후보 {n}건 — 센티널 레이어로 못 박는다", n=len(tables)))
        try:
            from ..auto.template import plant_sentinel

            xy = plant_sentinel()
        except Exception as e:                   # noqa: BLE001 — 폴백이 있다
            print(msg("센티널 경로 실패 ({kind}: {err})",
                      kind=type(e).__name__, err=e))
            xy = None
        if xy is not None:
            n_slot += 1
            got: list[int] = []
            for _ in range(3):                   # 커밋 반영이 한 박자 늦을 수 있다
                got = find_table_by_sentinel(p, layout, xy, n_slot)
                if got:
                    break
                _time.sleep(1.0)
            if got:
                tables = got
            else:
                print(msg("센티널 레코드를 못 찾았다 — 새 장수로 다시 장수 검색"))
                tables = find_layer_table(p, n_slot, layout)
    if not tables:
        print(msg("레이어 {n_slot}장짜리 비닐 그룹을 못 찾았다 — 그 장수만큼 도형이 든 "
                  "템플릿을 열 것 (`--template`을 주면 모자란 만큼 채우고 주입한다)",
                  n_slot=n_slot))
        p.close()
        return 1
    if len(tables) > 1:
        print(msg("경고: 표 후보가 {n}건이다 — 첫 번째에 쓴다"
                  " ({tables}). 지운 그룹의 잔재가"
                  " 섞였을 수 있다 — 화면이 안 바뀌면 --table로 지정할 것",
                  n=len(tables), tables=", ".join(hex(t) for t in tables[:6])))
    table = tables[0]
    wrote = write_plan_to_table(p, table, plan, layout, n_slot)
    p.close()
    pad = n_slot - n
    print(msg("레이어 {wrote}/{n_slot}장 기록 (표 0x{table:x})",
              wrote=wrote, n_slot=n_slot, table=table)
          + (msg(" — 그중 {pad}장은 안 보이는 덮개(알파 0)", pad=pad) if pad else "")
          + msg(" → 게임 화면에서 확인할 것"))
    return 0


# 남는 레이어를 치우는 값 — **이미 인게임에서 도는 수법이다.**
# 인게임 대조가 같은 값으로 3,000장 그룹을 통째로 밀어내 빈 캔버스를 찍고,
# 게이트가 그 빈 캔버스를 기준으로 섰다. 여기서
# 새로 하는 것은 "전부"가 아니라 "남는 자리만" 민다는 것뿐이다.
# 알파 0은 그 위에 얹은 보험이다 (주입이 알파를 쓰는 것은 별도 실측 —
# 인게임에서 알파 128과 255가 색오차로 갈렸다).
_PAD = {"x": 2000.0, "y": 2000.0, "sx": 0.01, "sy": 0.01, "alpha": 0.0}


def _padded(plan: LayerPlan, n_slot: int):
    """플랜 레이어 + 남는 자리를 채울 **안 보이는 레이어** (총 `n_slot`장)."""
    yield from plan.layers
    for _ in range(n_slot - len(plan.layers)):
        yield Layer(shape=plan.layers[0].shape if plan.layers else "A_01",
                    rot=0.0, skew=0.0, color=(0, 0, 0), label="pad", **_PAD)


def write_plan_to_table(p: Proc, table: int, plan: LayerPlan, layout: Layout,
                        count: int) -> int:
    """플랜 레이어 값을 **표의 레코드에 직접 쓴다**. 반환: 쓴 장수.

    `apply_plan`의 쓰기 루프를 뽑은 것 — **저장 그룹 재사용**(다시 연 그룹의
    접힌 그룹 내부 표, `find_folded_table`)도 같은 쓰기를 타게 한다. 플랜이
    `count`보다 짧으면 남는 자리는 안 보이는 덮개(`_padded`)로 채운다."""
    ids = layout.shape_ids or {}
    raw = p.read(table, count * 8)      # 표는 한 번에 읽는다 (3,000장이면 읽기 1회)
    ptrs = struct.unpack(f"<{count}Q", raw) if len(raw) == count * 8 else ()
    wrote = 0
    for i, lay in enumerate(_padded(plan, count)):
        ptr = ptrs[i] if i < len(ptrs) else p.u64(table + i * 8)
        if not ptr:
            continue
        ok = p.write(ptr + layout.position, struct.pack("<ff", lay.x, lay.y))
        ok &= p.write(ptr + layout.scale, struct.pack("<ff", lay.sx, lay.sy))
        ok &= p.write(ptr + layout.rotation, struct.pack("<f", lay.rot % 360.0))
        r, gg, b = lay.rgb()
        av = int(round(max(0.0, min(100.0, lay.alpha)) / 100.0 * 255.0))
        ok &= p.write(ptr + layout.color, bytes((r, gg, b, av)))
        sid = ids.get(lay.shape)
        if sid is not None:
            ok &= p.write(ptr + layout.shape_id, struct.pack("<H", int(sid)))
        wrote += bool(ok)
    return wrote


if __name__ == "__main__":  # 편의용 (정식 진입점은 __main__.py의 inject 명령)
    sys.exit(probe())
