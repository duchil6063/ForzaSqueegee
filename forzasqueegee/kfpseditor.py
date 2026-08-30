# SPDX-License-Identifier: MIT
# 서버 API 표면은 kloudys-forza-painter-suite(MIT)의 start_fabric_editor.py를
# 다시 쓴 것이다. Portions © 2021 AE (forza-painter) · Sam Twidale (geometrize-lib) ·
# Michael Fogleman (Primitive). 전문·유래는 THIRD_PARTY_NOTICES.md.
"""내장 KFPS Fabric 편집기 — 로컬 서버 + 도안 왕복 배선.

`vendor/kfps-editor/`(KFPS 0af4f21f 무수정 사본)를 서빙하는 로컬 HTTP 서버다.
KFPS의 `start_fabric_editor.py`(원본 서버)와 **같은 API 표면**을 우리 저장소
배선으로 다시 쓴 것 — editor.js는 `/tools/fabric-editor/…` 절대 URL과
`/api/fabric-editor/*` 계약을 하드코딩하므로 서버가 그 모양을 그대로 지킨다.

배선이 다른 곳 (KFPS 폴더 규약 → 우리 규약):

- **JSON 브라우저**: KFPS는 `imgs/generated`의 생성물을 내보이지만, 우리는
  `out/**/*.plan.json`(셀·선화·페인터 — 세 노선이 같은 스키마다)을 요청 시점에
  타입코드로 변환해 내보인다. 편집기는 plan.json의 존재를 모른다.
- **Export JSON**: 받은 타입코드를 원본 그대로 `work/editor/exports/`에 남기고
  (이력), 곧장 도안으로 역변환해 `out/kfpsedit/<이름>/`에 놓는다.
  변환 마커(`editor-output-change.json`)로 GUI가 그 자리에서 문다.
- **현재 도안 열기**: 편집기가 URL로 받는 것은 `?project=`뿐이라, 도안을
  편집기 프로젝트(`kloudy_fabric_editor_project_v1` — 타입코드 shapes의
  상위집합)로 구워 프로젝트 폴더에 놓고 그 id로 연다. 이때 원화(cutout.png
  우선)를 **레퍼런스 오버레이로 캔버스에 정합해** 같이 심는다 — 사람이
  원화를 반투명으로 깔고 따라 긋는 그 화면이 바로 나온다.
- **자동 복구본**: 편집기는 조작마다 `runtime/autosave.json`에 지금 상태를
  적어 둔다 — 창이 죽어도 남는다. 그런데 프로젝트를 물려 열면 편집기가 그것을
  **지우므로**(`loadProjectPayload` → `clearAutosave`), 도안을 스테이징해 열기
  전에 `recovery_state()`로 먼저 묻고 `recover_project()`로 굳혀야 한다.
- **표시 언어(한국어/영어)**: 동봉본이 무수정 사본이라 vendor 파일 대신
  index.html **서빙 사본**에 오버레이(`editor_i18n/` — 엔진+한국어 사전)를
  끼운다 (`_inject_overlays`). 편집기 하단 [언어] 셀렉트로 즉시 전환, 선택은
  localStorage(고정 포트 = 고정 origin), 기본은 앱 `--lang`. 커버리지는
  `tools/check_editor_i18n.py`가 vendor 문자열 전수로 대조한다.
- **선으로 가르기**: 같은 길로 기능도 하나 얹는다 (`editor_ext/fs-split.js` —
  editor.js **뒤에** 끼운다). 고른 레이어·그룹을 기준선 하나로 두 묶음으로
  가르고, 선에 걸친 레이어는 사본을 하나 더 만들어 양쪽에 다 넣는다.

편집기 상태(프로젝트·자동 복구·테마·설정)는 `work/editor/`에 산다. 서버
포트는 기본 고정(47615)이다 — 편집기의 즐겨찾기·단축키·리소스 캐시가
localStorage(origin = 호스트:포트)에 붙어 있어서, 포트가 흔들리면 그때마다
백지가 된다. 막혀 있으면 임시 포트로 물러난다.

색·좌표는 왕복 무손실이다 (`engine/kfpsjson.py` — 레이어가 게임 레코드
그대로라 단위 변환이 없고, 색 정본이 RGB 바이트라 몇 번을 오가도 안 변한다).
"""

from __future__ import annotations

import base64
import hmac
import json
import os
import re
import secrets
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

from .i18n import msg
from .paths import data_root, find_run_file, out_root, run_file, work_root

ROOT = data_root()
VENDOR = ROOT / "vendor" / "kfps-editor"
# 도형 메시 2,800파일 — **저장소에 없다.** 게임 에셋에서 나온 것이라 우리가
# 재배포하지 않고 KFPS 고정 커밋에서 받는다 (`tools/get_kfps.py`).
RESOURCES = VENDOR / "Resources"
I18N_DIR = Path(__file__).parent / "editor_i18n"   # 표시 언어 오버레이 (우리 파일)
EXT_DIR = Path(__file__).parent / "editor_ext"     # 기능 오버레이 (우리 파일)
# 편집기 자산 이름 → 그 파일이 사는 우리 폴더. vendor 동봉본이 무수정 사본이라
# 이것들만 저장소에서 나간다.
_OURS = {"fs-i18n.js": I18N_DIR, "fs-i18n-ko.js": I18N_DIR,
         "fs-split.js": EXT_DIR}
OUT = out_root()
STATE = work_root() / "editor"             # 편집기 상태의 뿌리 (도안이 아니다)
RUNTIME = STATE / "runtime"
PROJECTS = STATE / "projects"              # *.fabric-project.json (편집용 저장)
EXPORTS = STATE / "exports"                # Export JSON 원본 이력 (타입코드)
EDIT_OUT = OUT / "kfpsedit"                # 역변환된 도안 (도안 + 프리뷰)
CHANGE_MARKER = RUNTIME / "editor-output-change.json"
SOURCE_MAP = RUNTIME / "source-map.json"   # 프로젝트 이름 → 원 도안 정보
AUTOSAVE = RUNTIME / "autosave.json"    # 편집기 자동 복구본 (슬롯 하나)
SERVER_LOG = RUNTIME / "server.log"

DEFAULT_PORT = 47615        # 고정이 기본 — localStorage(origin)가 포트에 붙는다
EDITOR_PAGE = "/tools/fabric-editor/index.html"
_API = "/api/fabric-editor/"
_MUTATION_HEADER = "X-KFPS-Editor-Session"
_MUTATION_APIS = {"startup-help-confirmed", "preferences", "themes", "autosave",
                  "save-editor-json", "save-project", "open-project-folder"}
_MAX_BODY = 25 * 1024 * 1024               # KFPS와 같은 상한

# 브라우저에 내보이는 도안 파일 이름 → 표시 접미사. 도안 하나뿐인 폴더가
# 대부분이고, 정렬본·프루닝본도 같은 스키마라 그대로 편집이 된다. 실제
# 파일에는 폴더 이름이 앞에 붙는다 (`line-01.plan.json`) — `_out_kind`가 뗀다.
_PLAN_NAMES = {"plan.json": "", "plan_sorted.json": "-sorted",
               "plan_pruned.json": "-pruned"}

_CTYPE = {".html": "text/html; charset=utf-8",
          ".js": "text/javascript; charset=utf-8",
          ".css": "text/css; charset=utf-8",
          ".json": "application/json; charset=utf-8",
          ".md": "text/markdown; charset=utf-8",
          ".png": "image/png", ".ico": "image/x-icon",
          ".svg": "image/svg+xml"}

_count_cache: dict[str, tuple[float, int, int]] = {}    # path → (mtime, size, n)
_preview_cache: dict[str, tuple[float, bytes]] = {}     # path → (mtime, png)


# ────────────────────────────── 공용 ──────────────────────────────


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _clean_name(name: str, fallback: str = "vinyl") -> str:
    """파일 이름으로 안전한 밑동 — KFPS `_clean_filename_base`와 같은 규칙."""
    base = str(name or fallback).replace("\\", "/").split("/")[-1].strip()
    base = re.sub(r"\.json$", "", base, flags=re.IGNORECASE)
    base = re.sub(r"\.(fabric-project|fabric-export|normal-import|fh6-import)$",
                  "", base, flags=re.IGNORECASE)
    base = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", base)
    base = re.sub(r"\s+", " ", base).strip(" .")
    return base or fallback


def _unique_path(folder: Path, base: str, suffix: str = ".json") -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    cand = folder / f"{base}{suffix}"
    if not cand.exists():
        return cand
    for i in range(2, 10000):
        cand = folder / f"{base}-{i}{suffix}"
        if not cand.exists():
            return cand
    return folder / f"{base}-{time.strftime('%Y%m%d-%H%M%S')}{suffix}"


def _count_marker(path: Path, needle: bytes) -> int:
    """큰 JSON의 레이어 수 — 파싱 없이 표식 문자열을 센다 (mtime 캐시).

    plan.json은 레이어마다 `"shape"`가 정확히 한 번, 타입코드 JSON은
    `"type_word"`가 한 번이다 (engine/model·kfpsjson의 저장 꼴)."""
    key = str(path)
    try:
        st = path.stat()
    except OSError:
        return 0
    hit = _count_cache.get(key)
    if hit and hit[0] == st.st_mtime and hit[1] == st.st_size:
        return hit[2]
    try:
        n = path.read_bytes().count(needle)
    except OSError:
        return 0
    _count_cache[key] = (st.st_mtime, st.st_size, n)
    return n


def _rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _load_source_map() -> dict:
    try:
        return json.loads(SOURCE_MAP.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _remember_source(name: str, plan, staged_sha: str | None = None) -> None:
    """프로젝트 이름 → 원 도안 정보. Export 역변환이 원 캔버스·원화 경로를
    되찾는 데 쓴다 (편집기 프로젝트에는 실릴 자리가 없다 — 저장 시 버린다).

    extent(reach 어림)는 **같은 추정자끼리 견주기 위한 기준값**이다 — 추정이
    보수적 과대라 절대값으로 캔버스를 정하면 무편집 왕복도 캔버스가 커진다.
    편집본 extent가 이 기준을 넘은 비율만큼만 원 캔버스를 키운다.

    staged_sha는 우리가 구운 프로젝트 파일의 해시다 — 다음 스테이징이 "그
    파일이 아직 우리 것인가(사람이 Save로 덮지 않았나)"를 이것으로 가른다."""
    from .engine.kfpsjson import content_extent

    ex, ey = content_extent(plan.layers)
    m = _load_source_map()
    entry = dict(m.get(name) or {})    # 병합 — 없는 필드(staged_sha)를 지우면 안 된다
    entry.update({"source_image": str(plan.source_image or ""),
                  "image_size": list(plan.image_size),
                  "units_per_px": float(plan.units_per_px),
                  "extent": [float(ex), float(ey)]})
    if staged_sha:
        entry["staged_sha"] = staged_sha
    m[name] = entry
    _write_json_atomic(SOURCE_MAP, m)


# ────────────────────────────── 도안 → 편집기 프로젝트 ──────────────────────────────


def _overlay_state(plan, plan_path: Path) -> dict | None:
    """원화를 편집기 레퍼런스 오버레이로 — 캔버스(게임 유닛)에 정합해 심는다.

    `cutout.png`(노선이 실제로 받은 입력)가 있으면 그것이 작업 캔버스와 같은
    구도라 축마다 정확히 맞고, 없으면 플랜이 적어 둔 원화를 세로에 맞춰
    깐다 (크롭 전 원본이면 어림 — 옮기는 손잡이가 편집기에 있다)."""
    import cv2
    import numpy as np

    cand: list[tuple[Path, bool]] = [
        (find_run_file(plan_path.parent, "cutout.png"), True)]
    src = Path(plan.source_image) if plan.source_image else None
    if src is not None:
        cand.append((src if src.is_absolute() else ROOT / src, False))
    for p, exact in cand:
        if not p.is_file():
            continue
        try:
            raw = np.fromfile(str(p), np.uint8)          # 유니코드 경로 안전
            img = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
        except (OSError, cv2.error):
            continue
        if img is None:
            continue
        ih, iw = img.shape[:2]
        # 오버레이는 프로젝트/자동 복구 페이로드에 data URL로 실린다 (상한
        # 25MB) — 큰 원본은 2048로 줄여 담는다. 따라 긋기·색 표본에 충분하다.
        if max(ih, iw) > 2048:
            k = 2048.0 / max(ih, iw)
            img = cv2.resize(img, (max(1, round(iw * k)), max(1, round(ih * k))),
                             interpolation=cv2.INTER_AREA)
            ih, iw = img.shape[:2]
        ok, buf = cv2.imencode(".png", img)
        if not ok:
            continue
        if buf.size > 6 * 1024 * 1024:                   # 그래도 크면 JPEG (흰 바탕 합성)
            if img.ndim == 3 and img.shape[2] == 4:
                a = img[:, :, 3:4].astype(np.float32) / 255.0
                img = (img[:, :, :3].astype(np.float32) * a
                       + 255.0 * (1 - a)).astype(np.uint8)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            mime = "image/jpeg"
            if not ok:
                continue
        else:
            mime = "image/png"
        w_units = plan.image_size[0] * plan.units_per_px
        h_units = plan.image_size[1] * plan.units_per_px
        if exact:
            sx, sy = w_units / iw, h_units / ih
        else:
            sx = sy = h_units / ih
        pct = round(sx / (1800.0 / max(iw, ih)) * 100.0, 2)
        url = f"data:{mime};base64," + base64.b64encode(buf.tobytes()).decode()
        return {"version": 1, "kind": "image", "file_name": p.name,
                "mime_type": mime, "data_url": url, "svg_text": None,
                "intrinsic_width": iw, "intrinsic_height": ih,
                "object_width": iw, "object_height": ih,
                "rendered_width": iw * sx, "rendered_height": ih * sy,
                "transform": {"left": 0, "top": 0, "scaleX": sx, "scaleY": sy,
                              "angle": 0, "skewX": 0, "skewY": 0,
                              "flipX": False, "flipY": False,
                              "opacity": 0.5, "visible": True},
                "controls": {"scale_percent": pct, "opacity_percent": 50}}
    return None


def stage_plan_project(plan_path: str | Path, *, name: str | None = None) -> str:
    """도안 → 편집기 프로젝트 파일. 반환은 `?project=`에 넣을 id.

    같은 이름이 **우리가 구운 그대로**면 덮는다 — [편집] 단추를 다시 누르는
    것은 "지금 도안을 다시 열겠다"는 뜻이다. 그런데 사람이 편집기 [Save]로
    그 이름에 프로젝트(안내선·레퍼런스 포함)를 남겼을 수 있다 — 그건 사람의
    작업이라 안 덮고, 뒤에 번호를 붙여 비켜 선다 (지난 스테이징의 해시를
    소스맵에 적어 두고 그것과 다르면 사람 것이다)."""
    import hashlib

    from .engine.catalog import Catalog, default_catalog_path
    from .engine.kfpsjson import MAX_LAYERS, export_typecode
    from .engine.model import LayerPlan

    plan_path = Path(plan_path)
    plan = LayerPlan.load(plan_path)
    data, _st = export_typecode(plan, Catalog(default_catalog_path()))
    if not data["shapes"]:
        raise ValueError(msg("내보낼 수 있는 레이어가 하나도 없다"))
    if len(data["shapes"]) > MAX_LAYERS:
        raise ValueError(msg("레이어 {n}장 — 편집기 상한"
                             "({cap})을 넘는다. pruneplan으로 줄일 것",
                             n=len(data['shapes']), cap=MAX_LAYERS))
    base = _clean_name(name or plan_path.parent.name or plan_path.stem)
    suffix = _PLAN_NAMES.get(plan_path.name)
    if suffix:
        base += suffix
    PROJECTS.mkdir(parents=True, exist_ok=True)
    src_map = _load_source_map()
    cand = base
    for i in range(2, 10000):
        target = PROJECTS / f"{cand}.fabric-project.json"
        if not target.exists():
            break
        cur = hashlib.sha256(target.read_bytes()).hexdigest()
        if cur == (src_map.get(cand) or {}).get("staged_sha"):
            break                         # 우리가 구운 그대로다 — 덮어도 된다
        cand = f"{base}-{i}"
    base = cand
    payload = {"format": "kloudy_fabric_editor_project_v1", "name": base,
               "layer_count": len(data["shapes"]), "shapes": data["shapes"],
               "editor_guides": None, "editor_collapsed_groups": []}
    overlay = _overlay_state(plan, plan_path)
    if overlay is not None:
        payload["editor_source_overlay"] = overlay
    _write_json_atomic(target, payload)
    _remember_source(base, plan,
                     staged_sha=hashlib.sha256(target.read_bytes()).hexdigest())
    return target.name


def recovery_state() -> dict | None:
    """자동 복구본의 얼개 — 없으면 `None`.

    **도안을 스테이징해 열기 전에 반드시 본다.** 편집기는 `?project=`로 뜨면
    복구 제안을 건너뛰고, 프로젝트를 물리는 길에서 이 파일을 지운다 — 물어보지
    않고 열면 창이 죽기 직전까지 하던 편집이 그대로 사라진다."""
    try:
        payload = json.loads(AUTOSAVE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    shapes = payload.get("shapes") if isinstance(payload, dict) else None
    if not isinstance(shapes, list) or not shapes:
        return None
    return {"name": str(payload.get("name") or ""), "layers": len(shapes),
            "saved_at": str(payload.get("saved_at") or ""),
            "path": str(AUTOSAVE)}


def recover_project() -> str:
    """자동 복구본 → 편집기 프로젝트 파일. 반환은 `?project=`에 넣을 id.

    임시 사본을 **프로젝트로 굳혀 놓고** 그것을 연다 — 편집기가 여는 길에 임시
    사본을 지워도 잃는 것이 없다. 이름은 늘 새로 잡는다 (아무것도 안 덮는다).
    복구본에는 레이어·안내선·접힘·레퍼런스 원화가 다 실려 있어서 프로젝트
    스키마와 같고, 머리 두 칸(`format`·`layer_count`)만 갈아 끼우면 된다."""
    state = recovery_state()
    if state is None:
        raise ValueError(msg("자동 복구본이 없다"))
    payload = json.loads(AUTOSAVE.read_text(encoding="utf-8"))
    base = _clean_name(f"{state['name'] or 'autosave'}-recovered", "recovered")
    target = _unique_path(PROJECTS, base, ".fabric-project.json")
    payload["format"] = "kloudy_fabric_editor_project_v1"
    payload["name"] = re.sub(r"\.fabric-project\.json$", "", target.name,
                             flags=re.IGNORECASE)
    payload["layer_count"] = state["layers"]
    _write_json_atomic(target, payload)
    return target.name


# ────────────────────────────── 편집기 Export → 도안 ──────────────────────────────


def _convert_export(clean: str, payload: dict, raw_path: Path) -> tuple[Path, dict]:
    """편집기가 내보낸 타입코드 → `out/kfpsedit/<이름>/<이름>.plan.json`.

    원 도안에서 스테이징한 이름이면 원 캔버스·원화 경로를 되살린다 — 내용이
    원 캔버스를 넘으면 그만큼만 키운다 (렌더가 잘리면 대조가 거짓말이 된다)."""
    from .engine.catalog import Catalog, default_catalog_path
    from .engine.kfpsjson import content_extent, import_kfps, write_preview

    plan, st = import_kfps(payload)
    src = _load_source_map()
    m = src.get(clean) or src.get(re.sub(r"-\d+$", "", clean))
    if m:
        import math

        plan.source_image = m.get("source_image", "")
        upp = float(m.get("units_per_px") or plan.units_per_px)
        ow, oh = (int(v) for v in m.get("image_size", plan.image_size))
        # 원 캔버스가 정답이다 — 편집이 원 내용 범위(같은 reach 추정자)를
        # 넘어서면 그 비율만큼만 키운다 (무편집 왕복은 정확히 원 캔버스)
        ex0, ey0 = (float(v) for v in (m.get("extent") or (0.0, 0.0)))
        ex, ey = content_extent(plan.layers)
        w = ow if ex0 <= 0 or ex <= ex0 * 1.001 else math.ceil(ow * ex / ex0)
        h = oh if ey0 <= 0 or ey <= ey0 * 1.001 else math.ceil(oh * ey / ey0)
        plan.units_per_px = upp
        plan.image_size = (w, h)
    out_dir = EDIT_OUT / clean
    out_dir.mkdir(parents=True, exist_ok=True)
    plan_path = run_file(out_dir, "plan.json")
    plan.save(plan_path)
    run_file(out_dir, "kfps.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    write_preview(plan, Catalog(default_catalog_path()),
                  run_file(out_dir, "preview.png"))
    st["raw"] = _rel(raw_path)
    return plan_path, st


# ────────────────────────────── JSON 브라우저 ──────────────────────────────


def _out_kind(path: Path, names) -> str | None:
    """이 파일이 `names` 중 무엇인가 — 폴더 이름 접두를 떼고 본다.

    `line-01/line-01.plan.json`도 `line-01/plan.json`(예전 이름)도 `plan.json`
    으로 답한다. 아무것도 아니면 None."""
    fn = path.name
    if fn in names:
        return fn
    pre = f"{path.parent.name}."
    if fn.startswith(pre) and fn[len(pre):] in names:
        return fn[len(pre):]
    return None


def _walk_out(names: set[str]) -> list[Path]:
    """out/ 아래에서 이름이 맞는 파일 전부 (폴더 이름 접두 포함).

    `out/`에는 이제 도안만 산다 — 편집기 살림은 `work/`로 나갔다."""
    found: list[Path] = []
    if not OUT.exists():
        return found
    for dirpath, _dirnames, filenames in os.walk(OUT):
        d = Path(dirpath)
        for fn in filenames:
            if _out_kind(d / fn, names):
                found.append(d / fn)
    return found


def _plan_entry(path: Path) -> dict:
    st = path.stat()
    kind = _out_kind(path, _PLAN_NAMES) or path.name
    label = f"{path.parent.name}{_PLAN_NAMES.get(kind, '')}.json"
    return {"id": _rel(path), "name": label, "source": "generated",
            "layers": _count_marker(path, b'"shape"'),
            "mtime": st.st_mtime, "mtime_label": f"{st.st_mtime:.0f}",
            "preview_url": f"{_API}json-preview?id={quote(_rel(path))}"}


def _plan_groups() -> list[dict]:
    groups: dict[str, dict] = {}
    for path in _walk_out(set(_PLAN_NAMES)):
        try:
            entry = _plan_entry(path)
        except OSError:
            continue
        key = path.parent.relative_to(OUT).as_posix()
        g = groups.setdefault(key, {"key": key, "title": key,
                                    "source": "generated", "mtime": 0.0,
                                    "entries": []})
        g["entries"].append(entry)
        g["mtime"] = max(g["mtime"], entry["mtime"])
    for g in groups.values():
        g["entries"].sort(key=lambda e: (e["layers"], e["mtime"], e["name"]),
                          reverse=True)
        g["count"] = len(g["entries"])
        g["max_layers"] = max((e["layers"] for e in g["entries"]), default=0)
    return sorted(groups.values(), key=lambda g: (g["mtime"], g["title"]),
                  reverse=True)


def _single_groups(paths: list[Path], source: str, needle: bytes,
                   root: Path) -> list[dict]:
    groups = []
    for path in paths:
        try:
            st = path.stat()
        except OSError:
            continue
        rel_parent = path.parent.relative_to(root)
        title = path.name if str(rel_parent) == "." \
            else f"{rel_parent.as_posix()}/{path.name}"
        entry = {"id": _rel(path), "name": path.name, "source": source,
                 "layers": _count_marker(path, needle),
                 "mtime": st.st_mtime, "mtime_label": f"{st.st_mtime:.0f}",
                 "preview_url": f"{_API}json-preview?id={quote(_rel(path))}"}
        groups.append({"key": entry["id"], "title": title, "source": source,
                       "mtime": entry["mtime"], "count": 1,
                       "max_layers": entry["layers"], "entries": [entry]})
    return sorted(groups, key=lambda g: (g["mtime"], g["title"]), reverse=True)


def _editor_groups() -> list[dict]:
    paths = sorted(EXPORTS.rglob("*.json")) if EXPORTS.exists() else []
    return _single_groups(paths, "editor", b'"type_word"', EXPORTS)


def _kfps_groups() -> list[dict]:
    return _single_groups(_walk_out({"kfps.json"}), "exported",
                          b'"type_word"', OUT)


def _resolve_out_json(path_id: str) -> Path:
    """브라우저 id → 실제 파일.

    두 뿌리만 연다 — 도안(`out/`)과 편집기가 남긴 내보내기 이력
    (`work/editor/exports/`). 그 밖·JSON 아님·없음은 전부 거절."""
    if not path_id or "\x00" in path_id:
        raise ValueError("missing JSON id")
    cand = (ROOT / path_id).resolve()
    roots = (OUT.resolve(), EXPORTS.resolve())
    if not any(cand.is_relative_to(r) for r in roots):
        raise ValueError("JSON path is outside the editable browser roots")
    if cand.suffix.lower() != ".json" or not cand.is_file():
        raise ValueError("JSON file was not found")
    return cand


def _json_payload(path: Path) -> dict:
    """브라우저에서 고른 JSON → 편집기가 먹는 타입코드 페이로드.

    도안(`layers` 키)은 여기서 변환한다 — 편집기 쪽에는 plan이라는
    개념이 없다. 이미 도형 목록(타입코드·생성기 legacy)이면 그대로 준다
    (legacy는 편집기가 스스로 옮겨 읽는다)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "layers" in data:
        from .engine.catalog import Catalog, default_catalog_path
        from .engine.kfpsjson import export_typecode
        from .engine.model import LayerPlan

        plan = LayerPlan.load(path)
        out, _st = export_typecode(plan, Catalog(default_catalog_path()))
        # 편집기에서 이 이름으로 저장·내보내게 된다 — 원 도안 정보를 걸어 둔다
        kind = _out_kind(path, _PLAN_NAMES) or path.name
        _remember_source(_clean_name(f"{path.parent.name}"
                                     f"{_PLAN_NAMES.get(kind, '')}"), plan)
        return out
    return data


def _preview_bytes(path: Path) -> bytes | None:
    """미리보기 PNG — 도안 폴더의 프리뷰가 있으면 그대로 (0원),
    없으면 들여와서 렌더한다 (420px, mtime 캐시)."""
    sib = find_run_file(path.parent, "preview.png")
    if sib.is_file():
        try:
            return sib.read_bytes()
        except OSError:
            pass
    key = str(path)
    try:
        mt = path.stat().st_mtime
    except OSError:
        return None
    hit = _preview_cache.get(key)
    if hit and hit[0] == mt:
        return hit[1]
    try:
        import cv2

        from .engine.catalog import Catalog, default_catalog_path
        from .engine.kfpsjson import import_kfps
        from .engine.model import LayerPlan
        from .engine.render import render_plan

        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "layers" in data:
            plan = LayerPlan.load(path)
        else:
            plan, _ = import_kfps(data)
        rgb = render_plan(plan, Catalog(default_catalog_path()))
        h, w = rgb.shape[:2]
        if max(h, w) > 420:
            k = 420.0 / max(h, w)
            rgb = cv2.resize(rgb, (max(1, round(w * k)), max(1, round(h * k))),
                             interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if not ok:
            return None
        png = buf.tobytes()
    except Exception:                      # noqa: BLE001 — 미리보기 한 칸일 뿐이다
        return None
    _preview_cache[key] = (mt, png)
    return png


# ────────────────────────────── 프로젝트 ──────────────────────────────


def _resolve_project(path_id: str) -> Path:
    if not path_id or "\x00" in path_id:
        raise ValueError("missing project id")
    cand = (PROJECTS / path_id).resolve()
    if not cand.is_relative_to(PROJECTS.resolve()):
        raise ValueError("project path is outside the internal project folder")
    if cand.suffix.lower() != ".json" or not cand.is_file():
        raise ValueError("project file was not found")
    return cand


def _project_entries() -> list[dict]:
    if not PROJECTS.exists():
        return []
    entries = []
    for path in PROJECTS.rglob("*.fabric-project.json"):
        try:
            st = path.stat()
            title = re.sub(r"\.fabric-project\.json$", "", path.name,
                           flags=re.IGNORECASE)
            head = path.read_text(encoding="utf-8")[: 64 * 1024]
            mc = re.search(r'"layer_count"\s*:\s*(\d+)', head)
            nm = re.search(r'"name"\s*:\s*"([^"]+)"', head)
            entries.append({
                "id": path.relative_to(PROJECTS).as_posix(), "name": path.name,
                "title": nm.group(1) if nm else title,
                "layers": int(mc.group(1)) if mc else
                _count_marker(path, b'"type_word"'),
                "mtime": st.st_mtime})
        except (OSError, ValueError):
            continue
    return sorted(entries, key=lambda e: (e["mtime"], e["title"]), reverse=True)


# ────────────────────────────── 테마·설정 (KFPS 그대로) ──────────────────────────────

THEMES = RUNTIME / "themes"
_BUILTIN_THEMES = ({"id": "pastel", "name": "Signature Pink", "builtin": True,
                    "values": {}},
                   {"id": "dark", "name": "Dark", "builtin": True, "values": {}})


def _theme_entries() -> list[dict]:
    entries = list(_BUILTIN_THEMES)
    if not THEMES.exists():
        return entries
    for path in sorted(THEMES.glob("*.json"),
                       key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            tid = str(payload.get("id") or path.stem)
            values = payload.get("values")
            if not tid or tid in {"pastel", "dark"} or not isinstance(values, dict):
                continue
            entries.append({"id": tid, "name": str(payload.get("name") or tid),
                            "builtin": False,
                            "values": {str(k): str(v) for k, v in values.items()}})
        except Exception:                  # noqa: BLE001 — 깨진 테마는 건너뛴다
            continue
    return entries


def _theme_exists(tid: str) -> bool:
    if tid in {"pastel", "dark"}:
        return True
    if not tid or "\x00" in tid:
        return False
    cand = (THEMES / f"{tid}.json").resolve()
    return cand.is_relative_to(THEMES.resolve()) and cand.is_file()


def _inject_overlays(body: bytes) -> bytes:
    """index.html 서빙 사본에 **우리 오버레이 둘**을 끼운다 (vendor 파일 불변).

    자리가 다르다:

    - 표시 언어(`fs-i18n*.js`)는 **editor-core.js 앞**이다. 정적 DOM이 첫 페인트
      전에 번역되고, editor.js가 이후에 쓰는 텍스트는 오버레이의
      MutationObserver가 받는다. 뜰 때의 언어는 이 프로세스의 언어 설정
      (`work/state/lang.json`·`--lang`)이 **이긴다** — 편집기 안에서 바꾼
      선택은 그 세션 동안만 유지된다.
    - 선으로 가르기(`fs-split.js`)는 **editor.js 뒤**다. 편집기가 세운 캔버스와
      전역 손잡이를 그대로 쓰기 때문이다.

    앵커를 못 찾으면 (vendor 갱신으로 판이 바뀌면) 그 몫만 건너뛴다 — 영어
    원판·가르기 없는 판으로 그대로 동작한다."""
    from .i18n import current_language

    head = b'<script src="editor-core.js'
    if head in body:
        lang = "ko" if current_language() == "ko" else "en"
        inject = (f'<script>window.FS_EDITOR_LANG = "{lang}";</script>\n'
                  f'  <script src="fs-i18n-ko.js?engine=editor-2.1"></script>\n'
                  f'  <script src="fs-i18n.js?engine=editor-2.1"></script>\n'
                  f'  ').encode("utf-8")
        body = body.replace(head, inject + head, 1)
    for tail in (b'<script src="editor.js?engine=editor-2.1"></script>',
                 b'<script src="editor.js"></script>'):
        if tail in body:
            return body.replace(
                tail,
                tail + b'\n  <script src="fs-split.js?engine=editor-2.1"></script>',
                1)
    return body


# ────────────────────────────── HTTP ──────────────────────────────


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    # ── 공용 응답 ──
    def _json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _png(self, body: bytes) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or "0")
        if length <= 0 or length > _MAX_BODY:
            raise ValueError("invalid payload size")
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _authorized(self) -> bool:
        """변이 요청은 편집기 페이지에서만 — KFPS와 같은 3중 검사."""
        expected = str(getattr(self.server, "session_token", ""))
        supplied = str(self.headers.get(_MUTATION_HEADER) or "")
        if not expected or not hmac.compare_digest(supplied, expected):
            return False
        port = int(self.server.server_address[1])
        hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        if str(self.headers.get("Host") or "").casefold() not in hosts:
            return False
        origin = str(self.headers.get("Origin") or "").rstrip("/").casefold()
        if origin and origin not in {f"http://{h}" for h in hosts}:
            return False
        return str(self.headers.get("Sec-Fetch-Site") or "").casefold() \
            not in {"cross-site"}

    # ── GET ──
    def do_GET(self) -> None:  # noqa: N802 — http.server 규약
        parsed = urlparse(self.path)
        route = parsed.path
        try:
            if route.startswith(_API):
                self._api_get(route[len(_API):], parse_qs(parsed.query))
            else:
                self._static(route)
        except BrokenPipeError:
            pass
        except Exception as e:             # noqa: BLE001 — 서버가 죽으면 안 된다
            try:
                self._json({"error": f"{type(e).__name__}: {e}"}, status=500)
            except OSError:
                pass

    def _api_get(self, api: str, query: dict) -> None:
        if api == "health":
            self._json({"ok": True, "service": "kfps-fabric-editor",
                        "host": "forzasqueegee", "pid": os.getpid(),
                        "root": str(ROOT)})
        elif api == "startup-help-confirmed":
            marker = RUNTIME / "startup-help-confirmed.json"
            self._json({"confirmed": marker.exists(), "marker": str(marker)})
        elif api == "preferences":
            marker = RUNTIME / "preferences.json"
            theme = ""
            try:
                theme = str(json.loads(marker.read_text(encoding="utf-8"))
                            .get("theme") or "")
            except (OSError, ValueError):
                pass
            self._json({"theme": theme if _theme_exists(theme) else None,
                        "marker": str(marker)})
        elif api == "themes":
            self._json({"themes": _theme_entries(), "folder": str(THEMES)})
        elif api == "autosave":
            marker = AUTOSAVE
            if not marker.exists():
                self._json({"exists": False, "marker": str(marker)})
                return
            try:
                payload = json.loads(marker.read_text(encoding="utf-8"))
            except (OSError, ValueError) as e:
                self._json({"exists": False, "error": str(e),
                            "marker": str(marker)})
                return
            shapes = payload.get("shapes") if isinstance(payload, dict) else None
            ok = isinstance(shapes, list)
            self._json({"exists": ok, "payload": payload if ok else None,
                        "marker": str(marker)})
        elif api == "json-browser":
            source = (query.get("source") or ["generated"])[0]
            if source == "editor":
                groups = _editor_groups()
            elif source in {"exported", "handmade"}:
                source, groups = "exported", _kfps_groups()
            else:
                source, groups = "generated", _plan_groups()
            self._json({"source": source, "groups": groups,
                        "total_entries": sum(len(g["entries"]) for g in groups)})
        elif api == "json-file":
            try:
                path = _resolve_out_json((query.get("id") or [""])[0])
                payload = _json_payload(path)
            except Exception as e:         # noqa: BLE001 — 이유를 편집기에 준다
                self._json({"error": str(e)}, status=400)
                return
            self._json({"id": _rel(path), "name": path.name, "payload": payload})
        elif api == "json-preview":
            try:
                path = _resolve_out_json((query.get("id") or [""])[0])
                body = _preview_bytes(path)
                if not body:
                    raise ValueError("JSON preview could not be rendered")
            except Exception as e:         # noqa: BLE001
                self._json({"error": str(e)}, status=400)
                return
            self._png(body)
        elif api == "project-browser":
            entries = _project_entries()
            self._json({"entries": entries, "total_entries": len(entries),
                        "folder": str(PROJECTS)})
        elif api == "project-file":
            try:
                path = _resolve_project((query.get("id") or [""])[0])
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception as e:         # noqa: BLE001
                self._json({"error": str(e)}, status=400)
                return
            self._json({"id": path.relative_to(PROJECTS).as_posix(),
                        "name": path.name, "payload": payload})
        else:
            self._json({"error": "not found"}, status=404)

    def _static(self, route: str) -> None:
        """정적 자산 — URL은 KFPS 그대로, 파일은 vendor/kfps-editor/에서."""
        if route in ("", "/"):
            self.send_response(302)
            self.send_header("Location", EDITOR_PAGE)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        prefix = "/tools/fabric-editor/"
        if route.startswith(prefix):
            rel = route[len(prefix):]
        elif route == "/assets/kfps-logo.ico":
            rel = "assets/kfps-logo.ico"
        else:
            self._json({"error": "not found"}, status=404)
            return
        from urllib.parse import unquote
        rel = unquote(rel).replace("\\", "/")
        # 표시 언어 오버레이는 vendor가 아니라 우리 저장소에서 나간다 —
        # 동봉본은 무수정 사본(vendor README의 SHA-256 대조)이어야 하기 때문
        if rel in _OURS:
            cand = _OURS[rel] / rel
            if not cand.is_file():
                self._json({"error": "not found"}, status=404)
                return
        else:
            cand = (VENDOR / rel).resolve()
            if not cand.is_relative_to(VENDOR.resolve()) or not cand.is_file():
                self._json({"error": "not found"}, status=404)
                return
        body = cand.read_bytes()
        if cand.name == "index.html" and cand.parent == VENDOR:
            body = _inject_overlays(body)
        # 도형 메시(무확장 2,800파일)는 판이 커밋에 박혀 있다 — 오래 캐시.
        # 코어(html/js/css)는 no-cache — vendor 갱신이 새로고침으로 바로 선다
        ctype = _CTYPE.get(cand.suffix.lower(),
                           "application/json; charset=utf-8" if not cand.suffix
                           else "application/octet-stream")
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control",
                         "max-age=86400" if "Resources/" in rel else "no-cache")
        self.end_headers()
        self.wfile.write(body)

    # ── POST ──
    def do_POST(self) -> None:  # noqa: N802 — http.server 규약
        parsed = urlparse(self.path)
        route = parsed.path
        if not route.startswith(_API):
            self._json({"error": "not found"}, status=404)
            return
        api = route[len(_API):]
        if api in _MUTATION_APIS and not self._authorized():
            self._json({"error": "editor session authorization failed"},
                       status=403)
            return
        try:
            self._api_post(api)
        except BrokenPipeError:
            pass
        except Exception as e:             # noqa: BLE001 — 서버가 죽으면 안 된다
            try:
                self._json({"error": f"{type(e).__name__}: {e}"}, status=500)
            except OSError:
                pass

    def _api_post(self, api: str) -> None:
        if api == "startup-help-confirmed":
            marker = RUNTIME / "startup-help-confirmed.json"
            _write_json_atomic(marker, {"confirmed": True})
            self._json({"confirmed": True, "marker": str(marker)})
        elif api == "preferences":
            try:
                data = self._body()
                theme = str(data.get("theme") or "")
                if not _theme_exists(theme):
                    raise ValueError("invalid editor theme")
                _write_json_atomic(RUNTIME / "preferences.json", {"theme": theme})
            except (ValueError, OSError) as e:
                self._json({"error": str(e)}, status=400)
                return
            self._json({"ok": True, "theme": theme,
                        "marker": str(RUNTIME / "preferences.json")})
        elif api == "themes":
            try:
                data = self._body()
                name = str(data.get("name") or "Custom Theme").strip() \
                    or "Custom Theme"
                values = data.get("values")
                if not isinstance(values, dict) or not values:
                    raise ValueError("theme values must be a non-empty object")
                tid = re.sub(r"[^a-z0-9._-]+", "-",
                             _clean_name(str(data.get("id") or name),
                                         "custom-theme").lower()).strip(".-_") \
                    or "custom-theme"
                if tid in {"pastel", "dark"}:
                    tid += "-custom"
                THEMES.mkdir(parents=True, exist_ok=True)
                base_tid = tid
                i = 2
                while (THEMES / f"{tid}.json").exists():
                    tid = f"{base_tid}-{i}"
                    i += 1
                payload = {"format": "kfps_fabric_editor_theme_v1", "id": tid,
                           "name": name,
                           "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                        time.gmtime()),
                           "values": {str(k): str(v) for k, v in values.items()}}
                _write_json_atomic(THEMES / f"{tid}.json", payload)
                _write_json_atomic(RUNTIME / "preferences.json", {"theme": tid})
            except (ValueError, OSError) as e:
                self._json({"error": str(e)}, status=400)
                return
            self._json({"ok": True, "theme": payload,
                        "path": str(THEMES / f"{tid}.json")})
        elif api == "autosave":
            marker = AUTOSAVE
            try:
                data = self._body()
                if data.get("action") == "clear":
                    marker.unlink(missing_ok=True)
                    self._json({"ok": True, "cleared": True,
                                "marker": str(marker)})
                    return
                if not isinstance(data.get("shapes"), list):
                    raise ValueError("autosave payload must contain a shapes list")
                _write_json_atomic(marker, data)
            except (ValueError, OSError) as e:
                self._json({"error": str(e)}, status=400)
                return
            self._json({"ok": True, "marker": str(marker)})
        elif api == "save-editor-json":
            self._save_export()
        elif api == "save-project":
            try:
                data = self._body()
                payload = data.get("payload")
                if not isinstance(payload, dict) \
                        or not isinstance(payload.get("shapes"), list):
                    raise ValueError("project payload must contain a shapes list")
                name = _clean_name(str(data.get("name")
                                       or payload.get("name") or "project"),
                                   "project")
                payload["name"] = name
                payload["layer_count"] = len(payload["shapes"])
                target = PROJECTS / f"{name}.fabric-project.json"
                if target.exists() and not bool(data.get("overwrite")):
                    self._json({"error": f'A project named "{name}" already '
                                         f"exists. Choose a different name or "
                                         f"open it before using Save.",
                                "code": "project_exists"}, status=409)
                    return
                _write_json_atomic(target, payload)
            except (ValueError, OSError) as e:
                self._json({"error": str(e)}, status=400)
                return
            self._json({"ok": True,
                        "id": target.relative_to(PROJECTS).as_posix(),
                        "path": str(target), "name": target.name,
                        "title": name})
        elif api == "open-project-folder":
            try:
                PROJECTS.mkdir(parents=True, exist_ok=True)
                os.startfile(str(PROJECTS))  # noqa: S606 — 사용자 요청 폴더 열기
            except OSError as e:
                self._json({"error": str(e)}, status=400)
                return
            self._json({"ok": True, "folder": str(PROJECTS)})
        else:
            self._json({"error": "not found"}, status=404)

    def _save_export(self) -> None:
        """편집기 [Export JSON] — 원본 보존 + 즉시 도안 역변환 + GUI 알림."""
        try:
            data = self._body()
            payload = data.get("payload")
            if not isinstance(payload, dict) \
                    or not isinstance(payload.get("shapes"), list):
                raise ValueError("editor export payload must contain a shapes list")
            clean = _clean_name(str(data.get("name") or "vinyl"))
            raw = _unique_path(EXPORTS, clean)
            raw.write_text(json.dumps(payload, ensure_ascii=False),
                           encoding="utf-8")
        except (ValueError, OSError) as e:
            self._json({"error": str(e)}, status=400)
            return
        # 역변환 실패해도 Export 자체는 성공이다 — 원본이 남았고, 실패 이유는
        # 마커로 GUI에 간다 (편집기 쪽에는 성공으로 답해야 다운로드 폴백이 안 뜬다)
        marker: dict = {"name": clean, "raw": _rel(raw),
                        "changed_at_ns": time.time_ns()}
        try:
            plan_path, st = _convert_export(clean, payload, raw)
            marker.update({"path": str(plan_path), "layers": st["layers"],
                           "masks": st["masks"], "unknown": st["unknown"]})
        except Exception as e:             # noqa: BLE001 — 이유를 남긴다
            marker["error"] = f"{type(e).__name__}: {e}"
        _write_json_atomic(CHANGE_MARKER, marker)
        self._json({"ok": True, "id": _rel(raw), "path": str(raw),
                    "name": raw.name})

    def log_message(self, fmt: str, *args) -> None:
        try:
            RUNTIME.mkdir(parents=True, exist_ok=True)
            with SERVER_LOG.open("a", encoding="utf-8") as f:
                f.write(f"{time.strftime('%H:%M:%S')} {fmt % args}\n")
        except OSError:
            pass


class EditorServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False   # 고정 포트의 임자를 확실히 가른다

    def __init__(self, *args, **kwargs):
        self.session_token = secrets.token_urlsafe(32)
        super().__init__(*args, **kwargs)


_server: EditorServer | None = None
_server_thread: threading.Thread | None = None


def resources_available() -> bool:
    """도형 메시가 자리에 있나 — 없으면 편집기가 빈 도형으로 뜬다.

    저장소에 안 실리는 것이라(게임 자료) 처음 편집기를 열 때 받는다. 부르는
    쪽이 사람에게 물어보고 `tools/get_kfps.py`를 돌린다.
    """
    return RESOURCES.is_dir() and any(RESOURCES.rglob("*.png"))


def ensure_server(port: int | None = None) -> EditorServer:
    """이 프로세스의 편집기 서버 — 이미 돌면 그대로 재사용한다.

    기본은 고정 포트(localStorage가 origin에 붙는다) · 막혀 있으면 임시 포트."""
    global _server, _server_thread
    if _server is not None:
        return _server
    if not (VENDOR / "index.html").is_file():
        raise FileNotFoundError(msg("편집기 동봉본이 없다 — {path}", path=VENDOR))
    if not resources_available():
        raise FileNotFoundError(msg(
            "편집기 도형 리소스가 없다 — {path}\n"
            "  받으려면: python tools/get_kfps.py", path=RESOURCES))
    for p in ([port] if port else [DEFAULT_PORT, 0]):
        try:
            _server = EditorServer(("127.0.0.1", p), _Handler)
            break
        except OSError:
            if p == 0 or port:
                raise
    _server_thread = threading.Thread(target=_server.serve_forever,
                                      name="kfps-editor", daemon=True)
    _server_thread.start()
    return _server


def stop_server() -> None:
    global _server, _server_thread
    if _server is not None:
        _server.shutdown()
        _server.server_close()
    _server = _server_thread = None


def editor_url(server: EditorServer, project: str | None = None) -> str:
    port = server.server_address[1]
    url = f"http://127.0.0.1:{port}{EDITOR_PAGE}"
    if project:
        url += f"?project={quote(project, safe='')}"
    return url + f"#session={quote(server.session_token, safe='')}"


def read_change_marker() -> dict | None:
    try:
        return json.loads(CHANGE_MARKER.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# ────────────────────────────── CLI ──────────────────────────────


def _offer_resources() -> bool:
    """도형 리소스가 없다 — 콘솔에서 받을지 묻고, 받겠다면 그 자리에서 받는다.

    사람이 안 붙어 있으면(파이프·스크립트) 묻지 않고 길만 알려 준다.
    """
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        import get_kfps
    except ImportError as e:
        print(msg("오류: 수신 도구를 못 읽었다 — {err}", err=e))
        return False
    print(msg("편집기 도형 리소스가 없다 (2,800파일 · 30MB).\n"
              "  게임 도형 메시라 저장소에 안 싣고 KFPS 고정 커밋에서 받는다."))
    if sys.stdin and sys.stdin.isatty():
        try:
            yes = input(msg("  지금 받을까요? [Y/n] ")).strip().lower() \
                in ("", "y", "예")
        except (EOFError, KeyboardInterrupt):
            print()
            yes = False
    else:
        yes = False
    if not yes:
        print(msg("  받으려면: python tools/get_kfps.py"))
        return False
    return get_kfps.fetch() == 0


def serve_cli(plan: str | None = None, port: int = 0,
              open_browser: bool = True, recover: bool = False) -> int:
    """`python -m forzasqueegee edit` — 서버를 앞에서 돌린다 (Ctrl+C로 종료).

    plan.json 대신 KFPS JSON(타입코드·생성기 finals)을 줘도 된다 — GUI의
    [도안 불러오기]와 같은 판별·변환(`out/kfpsimport/`)을 거쳐 연다.

    자동 복구본이 있으면 **묻지 않고 프로젝트로 굳힌다** — 여기서는 물을
    사람이 없고, 도안을 물려 열면 편집기가 그것을 지우기 때문이다. `--recover`는
    도안 대신 그 복구본을 연다."""
    if not resources_available() and not _offer_resources():
        return 1
    project = None
    if recover:
        try:
            project = recover_project()
        except (OSError, ValueError) as e:
            print(msg("오류: {err}", err=e))
            return 1
        print(msg("복구본을 프로젝트로 굳혔다 → {path}", path=PROJECTS / project))
        plan = None
    elif plan and recovery_state():
        try:
            print(msg("자동 복구본을 프로젝트로 굳혀 둔다 → {path}",
                      path=PROJECTS / recover_project()))
        except (OSError, ValueError) as e:
            print(msg("경고: 복구본을 못 굳혔다 — {err}", err=e))
    if plan:
        from .engine.kfpsjson import resolve_plan

        plan_path, st = resolve_plan(Path(plan), OUT / "kfpsimport")
        if st is not None:
            print(msg("KFPS JSON을 도안으로 변환했다 → {path}",
                      path=plan_path.parent))
        try:
            project = stage_plan_project(plan_path)
        except (OSError, ValueError) as e:
            print(msg("오류: {err}", err=e))
            return 1
    try:
        server = ensure_server(port or None)
    except OSError as e:
        print(msg("오류: 포트를 못 연다 — {err}", err=e))
        return 1
    url = editor_url(server, project)
    print(msg("KFPS 편집기 (내장)"))
    print(msg("  주소: {url}", url=url))
    print(msg("  Export JSON → work/editor/exports/ 원본 보존 + "
              "out/kfpsedit/<이름>/<이름>.plan.json 자동 변환"))
    if open_browser:
        import webbrowser
        threading.Timer(0.3, lambda: webbrowser.open(url)).start()
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("\n" + msg("종료."))
    finally:
        stop_server()
    return 0
