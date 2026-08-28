# SPDX-License-Identifier: MIT
# 스키마는 kloudys-forza-painter-suite(MIT)의 타입코드 JSON 규약을 따른다.
# Portions © 2021 AE (forza-painter) · Sam Twidale (geometrize-lib) ·
# Michael Fogleman (Primitive). 전문·유래는 THIRD_PARTY_NOTICES.md.
"""KFPS(kloudys-forza-painter-suite) JSON ↔ plan.json.

KFPS의 Fabric 편집기·통합 임포터가 쓰는 **타입코드 JSON**이 정본 스키마다 —
레이어 하나가 게임 메모리 레코드 그대로다 (KFPS
`fh6_export_typecode_json.decode_layer`, KFPS 3.1.40 @ 0af4f21f 대조):

    {"type": 0x100000 + word, "type_word": word,
     "data": [x, y, sx, sy, rot, skew, mask], "color": [r, g, b, a], "mask": bool}

우리 plan.json 레이어와 **필드가 1:1**이라 단위 변환이 없다 — 주입이 레코드
(+0x18 x·y / +0x28 sx·sy / +0x50 rot / +0x74 RGBA / +0x7A word)에 쓰는 값
그대로다. 좌표 y-up·회전 CCW·전단 x+=skew·y 전부 같은 규약임을 편집기 수식
(`fh6MatrixFromData` = T(x,−y)·R(−rot)·SkewX(−skew)·S)과 대조해 확인했다.
word는 `catalog/fh6_layout.json`의 인게임 실측 520종이고 전부 편집기의
도형 라이브러리(Primitives·Gradient·Stripes·…·Community 1~4) 안에 있다.

**어느 카탈로그 도형이든 word로 정확히 오간다** (타원 근사 없음) — 뺄셈
마스크도 mask 플래그로 그대로 살아서, 편집기에서 열고 고쳐 다시 들여와도
렌더가 보존된다. word가 없는 도형(글꼴 글리프 — 두 생성 노선은 안 쓴다)만
등가 타원으로 근사하고 개수를 센다.

**KFPS 생성기 JSON**(finals *.v2.json — type 1·2 사각형 / 8·16 타원)도
들여온다. data는 KFPS 규약인 **중심 기준** [cx, cy, 폭, 높이|rx, ry, rot]
(px, y-down)이고 shapes[0]이 [0, 0, W, H] 배경이면 그것이 캔버스 크기다 —
galatea 이식의 `_shape_to_layer`와 같은 해석이라 painter 노선 plan과 값이
같다. (구 ForzaPainter의 모서리 사각형 스키마는 더 안 받는다 — KFPS 편집기로
열어 타입코드로 내보내면 된다.)

편집기 표시 한계 (데이터는 무변경): KFPS 메시는 bbox 중심 정렬이라 설계
원점이 중심 밖인 소수 도형은 편집기 화면에서만 약간 밀려 보인다. 게임
배치는 우리 카탈로그의 native 보정(인게임 실측)이 맞다.

색은 **바이트 그대로 오간다** — 정본이 RGB 바이트라(모델 문서) 왕복 손실이
0이고, 몇 번을 오가도 불변이다 (HSB는 창 조작이 적용 시점에만 유도).
알파는 왕복에서 1바이트 해상도로 떨어진다 (73.0% → 186/255 → 72.94%) —
손실이 아니라 **게임 레코드가 알파를 1바이트로 저장**하기 때문이다 (주입도
같은 byte를 쓴다). 왕복값이 오히려 인게임 실제값이고, 렌더 차이는 0이다.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from ..paths import run_file
from .catalog import Catalog, default_catalog_path
from .model import UNITS_PER_SCALE, Layer, LayerPlan

TYPE_CODE_BASE = 0x100000    # KFPS: type = 0x100000 + 게임 도형 word(u16)
_APPROX_WORD = 102           # A_02 회전 타원 — word 없는 도형의 근사 몸통
MAX_LAYERS = 3000            # 편집기·게임 공통 상한 (넘으면 편집기가 거부)
_LEGACY_TYPES = frozenset({1, 2, 8, 16})   # KFPS 생성기 사각형·타원


def _word_maps() -> tuple[dict[str, int], dict[int, str]]:
    """카탈로그 이름 ↔ 게임 word (fh6_layout.json 실측 520종)."""
    from ..game.inject import Layout

    ids = Layout.load().shape_ids or {}
    by_name = {n: int(w) for n, w in ids.items()}
    return by_name, {w: n for n, w in by_name.items()}


# ------------------------------------------------------------------ 내보내기

def export_typecode(plan: LayerPlan, catalog: Catalog) -> tuple[dict, dict]:
    """plan.json → KFPS 타입코드 JSON dict. (data, 통계).

    통계: exact(word 그대로)·masks(그중 마스크)·approx(word 없어 등가 타원
    근사)·invisible(알파 0 — 편집기도 버리므로 미리 뺀다).
    """
    by_name, _ = _word_maps()
    shapes: list[dict] = []
    n_exact = n_mask = n_approx = invisible = 0
    for lay in plan.layers:
        a255 = int(round(float(np.clip(lay.alpha, 0, 100)) * 2.55))
        if a255 <= 0 and not lay.mask:
            invisible += 1
            continue
        word = by_name.get(lay.shape)
        if word is None:
            sp = _approx_shape(lay, plan, catalog, a255)
            if sp is None:
                invisible += 1
            else:
                shapes.append(sp)
                n_approx += 1
            continue
        # 마스크는 게임이 색을 안 쓰지만 알파 0이면 편집기가 버리므로 255로
        shapes.append({
            "type": TYPE_CODE_BASE + word, "type_word": word,
            "data": [_r6(lay.x), _r6(lay.y), _r6(lay.sx), _r6(lay.sy),
                     _r6(lay.rot % 360.0), _r6(lay.skew), 1 if lay.mask else 0],
            "color": [*lay.rgb(), 255 if lay.mask else a255],
            "mask": bool(lay.mask), "score": 0,
            "shape_name": lay.shape,
        })
        n_exact += 1
        n_mask += bool(lay.mask)
    return {"shapes": shapes}, {"exact": n_exact, "masks": n_mask,
                                "approx": n_approx, "invisible": invisible}


def _r6(v: float) -> float:
    return round(float(v), 6)


def _polys_px(lay: Layer, plan: LayerPlan, catalog: Catalog) -> list[np.ndarray]:
    """레이어의 변환 폴리곤들 (이미지 px, y-down) — `render._draw_layer`와 같은
    변환 (전단 포함, 글꼴은 잉크 루프만: em 상자 표식은 게임이 안 그린다)."""
    from .textvinyl import is_font

    sh = catalog[lay.shape]
    loops = sh.loops
    if is_font(lay.shape):
        from .textvinyl import ink_loops
        loops = tuple(ink_loops(catalog, lay.shape))
    w, h = plan.image_size
    upp = plan.units_per_px
    rot = np.radians(lay.rot)
    c, s = np.cos(rot), np.sin(rot)
    out = []
    for loop in loops:
        pts = loop * np.array([lay.sx, lay.sy], np.float32) * UNITS_PER_SCALE
        if lay.skew:
            pts = pts + np.stack([pts[:, 1] * lay.skew,
                                  np.zeros(len(pts), np.float32)], axis=1)
        pts = pts @ np.array([[c, s], [-s, c]], np.float32)
        pts += np.array([lay.x, lay.y], np.float32)
        px = pts[:, 0] / upp + w / 2
        py = h / 2 - pts[:, 1] / upp
        out.append(np.stack([px, py], axis=1))
    return out


def _moment_ellipse(polys: list[np.ndarray]) -> list[float] | None:
    """폴리곤(짝홀 채움)의 등가 타원 [cx, cy, rx, ry, 이미지 좌표 각].

    균일 타원의 공분산 고유값은 r²/4라 r = 2√λ가 참 반지름이다 — 타원이면
    정확히 복원되고, 다른 도형은 무게중심·주축이 맞는 근사가 된다. 반지름은
    마스크 면적에 맞춰 다시 재서(πrxry = 면적) 잉크 양을 보존한다."""
    allp = np.concatenate(polys, axis=0)
    x0, y0 = np.floor(allp.min(axis=0))
    x1, y1 = np.ceil(allp.max(axis=0))
    bw, bh = int(x1 - x0) + 1, int(y1 - y0) + 1
    if bw < 1 or bh < 1:
        return None
    # 작은 도형은 슈퍼샘플로 모멘트를 안정시킨다 (긴 변 ≥ 96px 목표)
    sc = int(np.clip(np.ceil(96.0 / max(bw, bh)), 1, 8))
    m = np.zeros((bh * sc, bw * sc), np.uint8)
    off = np.array([x0, y0], np.float64)
    for p in polys:
        mm = np.zeros_like(m)
        cv2.fillPoly(mm, [np.round((p - off) * sc).astype(np.int32)], 1)
        m ^= mm
    mo = cv2.moments(m, binaryImage=True)
    if mo["m00"] < 1:
        return None
    cx, cy = mo["m10"] / mo["m00"], mo["m01"] / mo["m00"]
    mu20, mu02 = mo["mu20"] / mo["m00"], mo["mu02"] / mo["m00"]
    mu11 = mo["mu11"] / mo["m00"]
    half = np.sqrt(((mu20 - mu02) / 2) ** 2 + mu11 ** 2)
    lam1 = max((mu20 + mu02) / 2 + half, 1e-9)
    lam2 = max((mu20 + mu02) / 2 - half, 1e-9)
    ang = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
    rx, ry = 2 * np.sqrt(lam1), 2 * np.sqrt(lam2)
    k = np.sqrt(mo["m00"] / max(np.pi * rx * ry, 1e-9))
    rx, ry = rx * k, ry * k
    return [float(x0 + cx / sc), float(y0 + cy / sc),
            max(float(rx / sc), 0.5), max(float(ry / sc), 0.5),
            float(np.degrees(ang)) % 360.0]


def _approx_shape(lay: Layer, plan: LayerPlan, catalog: Catalog,
                  a255: int) -> dict | None:
    """word 없는 도형(글꼴 등) → 등가 타원 근사 (모멘트법, 잉크 양 보존)."""
    polys = _polys_px(lay, plan, catalog)
    fit = _moment_ellipse(polys) if polys else None
    if fit is None:
        return None
    w, h = plan.image_size
    upp = plan.units_per_px
    cx, cy, rx, ry, ang = fit
    return {
        "type": TYPE_CODE_BASE + _APPROX_WORD, "type_word": _APPROX_WORD,
        "data": [_r6((cx - w / 2) * upp), _r6((h / 2 - cy) * upp),
                 _r6(max(0.01, rx * upp / UNITS_PER_SCALE)),
                 _r6(max(0.01, ry * upp / UNITS_PER_SCALE)),
                 _r6((-ang) % 360.0), 0.0, 1 if lay.mask else 0],
        "color": [*lay.rgb(), 255 if lay.mask else a255],
        "mask": bool(lay.mask), "score": 0,
        "shape_name": f"~{lay.shape}",
    }


# ------------------------------------------------------------------ 들여오기

def sniff_kfps(path: str | Path) -> str | None:
    """이 JSON이 KFPS 도형 목록인가 — "typecode"|"legacy"|None.

    plan.json(`layers` 키)은 None이다. legacy = KFPS 생성기 사각형·타원
    (type 1·2·8·16, **중심 기준** data) — finals *.v2.json이 이 꼴이다.
    """
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    shapes = data.get("shapes") if isinstance(data, dict) else data
    if isinstance(data, dict) and "layers" in data:
        return None
    if not isinstance(shapes, list) or not shapes:
        return None
    kinds = set()
    for sp in shapes[:64]:
        if not isinstance(sp, dict):
            return None
        try:
            t = int(sp.get("type", 0))
        except (TypeError, ValueError):
            return None
        if t > 1000000 or "type_word" in sp:
            kinds.add("typecode")
        elif t in _LEGACY_TYPES and len(sp.get("data") or []) >= 4:
            kinds.add("legacy")
        else:
            return None
    if "typecode" in kinds:
        return "typecode"
    return "legacy" if kinds else None


def import_kfps(src: str | Path | dict, *,
                source_image: str = "") -> tuple[LayerPlan, dict]:
    """KFPS JSON(편집기·게임 내보내기 타입코드, 생성기 legacy) → LayerPlan.

    (plan, 통계) — 통계: layers·masks·invisible(알파 0)·unknown({word: 수} —
    카탈로그에 없는 word, 글꼴 글리프 등)·kind. 레이어 값은 게임 입력 스텝으로
    양자화한다 (플랜 렌더 = 인게임 원칙 — 편집기의 자유 실수를 게임이 실제로
    받는 그리드로 스냅).
    """
    src_path = None if isinstance(src, dict) else Path(src)
    data = src if isinstance(src, dict) else \
        json.loads(src_path.read_text(encoding="utf-8"))
    shapes = data.get("shapes") if isinstance(data, dict) else data
    if not isinstance(shapes, list) or not shapes:
        raise ValueError("KFPS JSON이 아니다 — {'shapes': [...]}여야 한다")
    typed = [sp for sp in shapes if isinstance(sp, dict)]
    if any(int(sp.get("type", 0) or 0) > 1000000 or "type_word" in sp
           for sp in typed):
        return _import_typecode(typed, source_image)
    return _import_legacy(typed, source_image, src_path)


def _import_typecode(shapes: list[dict],
                     source_image: str) -> tuple[LayerPlan, dict]:
    _, by_word = _word_maps()
    layers: list[Layer] = []
    invisible = 0
    unknown: dict[int, int] = {}
    for sp in shapes:
        try:
            t = int(sp.get("type", 0))
            word = int(sp.get("type_word", t & 0xFFFF)) & 0xFFFF
        except (TypeError, ValueError):
            continue
        if t <= 1000000 and "type_word" not in sp:
            continue                       # 타입코드 도형이 아니다
        d = [float(v) for v in (sp.get("data") or [])]
        if len(d) < 4:
            continue
        mask = bool(sp.get("mask") or (len(d) > 6 and d[6]))
        col = list(sp.get("color") or [255, 255, 255, 255])
        while len(col) < 4:
            col.append(255)
        a255 = float(col[3])
        if a255 <= 0 and not mask:
            invisible += 1
            continue
        name = by_word.get(word)
        if name is None:
            unknown[word] = unknown.get(word, 0) + 1
            continue
        rgb = tuple(int(np.clip(round(float(v)), 0, 255)) for v in col[:3])
        layers.append(Layer(
            shape=name, x=d[0], y=d[1],
            sx=d[2] if abs(d[2]) >= 0.01 else 0.01,
            sy=d[3] if abs(d[3]) >= 0.01 else 0.01,
            rot=(d[4] % 360.0) if len(d) > 4 else 0.0,
            skew=d[5] if len(d) > 5 else 0.0,
            color=rgb,
            alpha=100.0 if mask else round(a255 / 255.0 * 100.0, 2),
            label="mask" if mask else "kfps", mask=mask).quantized())
    if not layers:
        raise ValueError("옮길 수 있는 도형이 하나도 없다")
    plan = LayerPlan(source_image=source_image, layers=layers)
    plan.image_size, plan.units_per_px = _fit_canvas(layers)
    return plan, {"kind": "typecode", "layers": len(layers),
                  "masks": sum(1 for l in layers if l.mask),
                  "invisible": invisible, "unknown": unknown}


def _sibling_canvas(src_path: Path | None) -> tuple[int, int] | None:
    """finals/*.v2.json의 원 캔버스 — 형제 체크포인트의 배경 [0,0,W,H]에서.

    KFPS finals에는 배경(캔버스 크기)이 없다 (원본 V2도 drawables만 적는다).
    painter 출력 트리라면 `finals/<이름>.<장수>v2.json` 옆에 같은 장수의
    체크포인트 `checkpoints/<이름>.<장수>.json`이 있고 그 shapes[0]이 캔버스다.
    """
    if src_path is None or not src_path.name.endswith("v2.json"):
        return None
    ck = (src_path.parent.parent / "checkpoints"
          / (src_path.name[:-len("v2.json")] + ".json"))
    try:
        sp0 = json.loads(ck.read_text(encoding="utf-8"))["shapes"][0]
        d0 = [float(v) for v in sp0.get("data", [])]
        if int(sp0.get("type", 0) or 0) == 1 and len(d0) >= 4 \
                and abs(d0[0]) < 0.5 and abs(d0[1]) < 0.5 and d0[2] > 1 and d0[3] > 1:
            return int(round(d0[2])), int(round(d0[3]))
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        pass
    return None


def _import_legacy(shapes: list[dict], source_image: str,
                   src_path: Path | None = None) -> tuple[LayerPlan, dict]:
    """KFPS 생성기 도형 목록 (중심 기준 px, y-down) → LayerPlan.

    shapes[0]이 [0, 0, W, H] 사각형이면 KFPS 배경 규약 — 캔버스 크기를 얻고
    알파가 있으면 전면 배경 레이어로 놓는다. 나머지는 galatea 이식과 같은
    `_shape_to_layer` 해석이라 painter plan과 같은 값이 나온다. finals에는
    배경이 없으므로 형제 체크포인트에서 캔버스를 찾고, 그마저 없으면 도형
    범위로 물러난다 (KFPS 자체 임포터와 같은 어림 — 유닛 배율이 어림만큼
    어긋난다).
    """
    from .render import _BASE_HEIGHT_UNITS
    from .galatea.quantize import _shape_to_layer

    bg = None
    drawables = shapes
    sp0 = shapes[0]
    d0 = [float(v) for v in (sp0.get("data") or [])]
    sib = None
    if int(sp0.get("type", 0) or 0) == 1 and len(d0) >= 4 \
            and abs(d0[0]) < 0.5 and abs(d0[1]) < 0.5 and d0[2] > 1 and d0[3] > 1:
        bg, drawables = sp0, shapes[1:]
        full_w, full_h = int(round(d0[2])), int(round(d0[3]))
        how = "배경 사각형"
    elif (sib := _sibling_canvas(src_path)) is not None:
        full_w, full_h = sib
        how = "형제 체크포인트"
    else:
        mx = my = 1.0
        for sp in shapes:
            d = [float(v) for v in (sp.get("data") or [])]
            if len(d) < 4:
                continue
            # 사각형 data[2·3]은 전체 폭·높이, 타원은 반지름 (KFPS 중심 규약)
            k = 0.5 if int(sp.get("type", 0) or 0) in (1, 2) else 1.0
            mx = max(mx, abs(d[0]) + abs(d[2]) * k)
            my = max(my, abs(d[1]) + abs(d[3]) * k)
        full_w, full_h = int(np.ceil(mx)), int(np.ceil(my))
        how = "도형 범위"
    upp = _BASE_HEIGHT_UNITS / full_h
    plan = LayerPlan(source_image=source_image, image_size=(full_w, full_h),
                     units_per_px=upp)
    invisible = 0
    cat = Catalog(default_catalog_path())
    if bg is not None:
        col = list(bg.get("color") or [0, 0, 0, 0])
        a255 = float(col[3]) if len(col) > 3 else 255.0
        if a255 > 0:
            plan.layers.append(Layer(
                shape=cat.square, x=0.0, y=0.0,
                sx=full_w * upp / (2 * UNITS_PER_SCALE),
                sy=full_h * upp / (2 * UNITS_PER_SCALE),
                color=tuple(int(np.clip(round(float(v)), 0, 255))
                            for v in col[:3]),
                alpha=round(np.clip(a255, 0, 255) / 255.0 * 100.0, 2),
                label="fp_bg").quantized())
    for sp in drawables:
        lay = _shape_to_layer(sp, full_w, full_h, upp)
        if lay is None:
            continue
        if float(lay.alpha) <= 0:
            invisible += 1
            continue
        plan.layers.append(lay)
    if not plan.layers:
        raise ValueError("옮길 수 있는 도형이 하나도 없다")
    return plan, {"kind": "legacy", "layers": len(plan.layers),
                  "masks": 0, "invisible": invisible, "unknown": {},
                  "size_from": how}


def content_extent(layers: list[Layer]) -> tuple[float, float]:
    """레이어 내용의 원점 기준 최대 도달 (x·y, 게임 유닛) — reach 어림."""
    cat = Catalog(default_catalog_path())
    ex = ey = 1.0
    for l in layers:
        r = cat[l.shape].reach if l.shape in cat.shapes else 1.0
        span = r * UNITS_PER_SCALE * max(abs(l.sx), abs(l.sy)) * (1.0 + abs(l.skew))
        ex = max(ex, abs(l.x) + span)
        ey = max(ey, abs(l.y) + span)
    return ex, ey


def _fit_canvas(layers: list[Layer]) -> tuple[tuple[int, int], float]:
    """내용 범위에 맞는 렌더 캔버스 — 1유닛 = 1px (900유닛 캔버스 관행 유지).

    편집기 캔버스는 ±1000유닛이지만 도안은 대개 900유닛 세로 안에 있다.
    내용이 그보다 크면 캔버스를 내용에 맞춰 키운다 (렌더가 잘리면 대조가
    거짓말이 된다). reach 어림이라 여백 8%를 얹는다.
    """
    ex, ey = content_extent(layers)
    w = max(900, int(np.ceil(ex * 1.08)) * 2)
    h = max(900, int(np.ceil(ey * 1.08)) * 2)
    return (w, h), 1.0


def import_kfps_to(src: str | Path, out_dir: str | Path, *,
                   source_image: str = "") -> tuple[LayerPlan, dict, Path]:
    """KFPS JSON → `out_dir`에 도안 + 프리뷰. (plan, 통계, 도안 경로).

    파일 이름은 폴더 이름을 앞에 단다 — `out/내도안/내도안.plan.json`."""
    plan, st = import_kfps(src, source_image=source_image)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plan_path = run_file(out, "plan.json")
    plan.save(plan_path)
    write_preview(plan, Catalog(default_catalog_path()),
                  run_file(out, "preview.png"))
    return plan, st, plan_path


def resolve_plan(path: str | Path, out_root: str | Path) -> tuple[Path, dict | None]:
    """GUI용: 고른 JSON이 KFPS 도형 목록이면 `out_root/<이름>/`에 변환해 두고
    그 plan.json 경로를 준다 — (plan 경로, 변환 통계). plan.json이거나 판별이
    안 되면 그대로 (path, None)이다 (그때는 plan 로더가 제 이유를 적는다)."""
    p = Path(path)
    if sniff_kfps(p) is None:
        return p, None
    _plan, st, plan_path = import_kfps_to(p, Path(out_root) / p.stem)
    return plan_path, st


# ------------------------------------------------------------------ 자가 점검

def write_preview(plan: LayerPlan, catalog: Catalog, path: str | Path) -> None:
    """플랜 완성 예상도 PNG (흰 배경 — 다른 노선의 preview.png와 같은 렌더)."""
    from .render import render_plan

    rgb = render_plan(plan, catalog)
    cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))[1].tofile(str(path))


def roundtrip_diff(plan: LayerPlan, data: dict, catalog: Catalog) -> dict:
    """내보낸 타입코드 도형을 도로 플랜으로 읽어 렌더를 원본과 대조한다.

    word 그대로 오간 도형은 양자화 재적용뿐이라 0에 붙어야 한다 (근사·미지원
    word가 섞이면 그만큼 뜬다).
    """
    from .render import render_plan
    from .sortplan import plan_pad_px

    back, _ = import_kfps(data)
    back.image_size, back.units_per_px = plan.image_size, plan.units_per_px
    pad = plan_pad_px(plan, catalog)
    ra = render_plan(plan, catalog, pad=pad).astype(np.int16)
    rb = render_plan(back, catalog, pad=pad).astype(np.int16)
    d = np.abs(ra - rb).max(axis=2)
    return {"changed_frac": float((d > 8).mean()), "max": int(d.max())}
