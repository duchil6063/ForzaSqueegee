r"""이타샤 구성 **미리보기** — 게임에 올리기 전에 면별 결과를 그림으로 본다.

구성 파일(itasha.json)의 면 배치를 설치 파일 면 지도(`game/carfiles` —
`compose.surfaces_for`) 위에 합성한다. 렌더는 공용 렌더러(`engine.render`)
하나로 한다: 면마다 **면 유닛 캔버스**를 만들고, 그룹·도형을 전부 그 캔버스의
Layer로 변환해 순서대로 그린다 (그려지는 순서 = 게임과 같다:
도형 → 꾸밈 그룹 → 도안 그룹 → 덮개 도형 → 글자).

한계 (미리보기의 정의): 게임의 3D 투영·반사·유리 반투명은 없다 — 여기 보이는
것은 **면 유닛 공간의 평면 합성**이다. 배치 수치 검증 경계와 같은 좌표계라
"어디에 얼마나 크게"는 그대로 맞는다.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

from ..i18n import msg
from ..paths import run_file
from . import textvinyl as tv
from .catalog import Catalog, default_catalog_path
from .model import Layer, LayerPlan
from .render import render_plan
from . import compose

# 면 배열 (시트에 놓는 순서 — 없는 면은 건너뛴다)
SHEET_ORDER = ("side_left", "side_right", "top", "front", "rear",
               "windshield", "rear_window", "window_left", "window_right",
               "spoiler", "sunroof")
MAX_W = 880                    # 면 하나의 미리보기 폭 상한 (px)


def _compose_group(plan: LayerPlan, *, x: float, y: float, scale: float,
                   rot: float, mirror: bool) -> list[Layer]:
    """그룹 플랜의 레이어들을 **면 유닛 좌표**로 변환한 사본.

    표시 변환 = 이동 ∘ R(rot) ∘ (미러면 수평뒤집기) ∘ 균등 스케일.
    미러는 음수 sx로 낸다 (게임의 미러 표현과 같다). 회전 합성은
    미러가 끼면 부호가 뒤집힌다 (M·R(θ) = R(−θ)·M).
    """
    th = math.radians(rot)
    c, s = math.cos(th), math.sin(th)
    out: list[Layer] = []
    for l in plan.layers:
        lx, ly = l.x, l.y
        lsx, lrot = l.sx, l.rot
        if mirror:
            lx = -lx
            lsx = -lsx
            lrot = -lrot
        nx = x + scale * (lx * c - ly * s)
        ny = y + scale * (lx * s + ly * c)
        out.append(Layer(shape=l.shape, x=nx, y=ny,
                         sx=lsx * scale, sy=l.sy * scale,
                         rot=(rot + lrot) % 360.0, skew=l.skew,
                         color=l.color, alpha=l.alpha,
                         label=l.label, mask=l.mask))
    return out


def _text_layers(job: dict, cat: Catalog) -> list[Layer]:
    """글자 명세(`auto.gametext` 규약) → 면 유닛 Layer들.

    이타샤 어휘는 글자를 안 쓰지만 (`engine.compose`) 구성 파일 규약에는 남아
    있으므로 (사람이 손으로 적은 `text`) 미리보기도 그린다. 게임 텍스트 도구의
    세 벌(그림자→테두리→본색)을 재현한다 — 테두리는 **같은 크기 4벌을 대각
    오프셋**으로 깔아 실전 경로와 같은 상수를 쓴다 (`OUTLINE_SHIFT` 0.06 ×
    대문자 높이). 확대 1벌로 흉내 내면 긴 문구에서 글자가 바깥으로 밀려 서로
    겹쳐 **흰 덩어리**로 뭉친다 (게임에는 없는 꼴이다).
    """
    text = job.get("text") or ""
    if not text:
        return []
    font = job.get("font") or tv.DEFAULT_FONT
    h = float(job.get("height") or 20.0)
    cx, cy = job.get("center") or (0.0, 0.0)
    rot = float(job.get("rot") or 0.0)
    passes: list[tuple[tuple[int, int, int], float, tuple[float, float]]] = []
    if job.get("shadow"):
        off = 0.08 * h
        passes.append((tuple(job["shadow"]), 1.0, (off, -off)))
    if job.get("outline"):
        s = tv.OUTLINE_SHIFT * h
        passes += [(tuple(job["outline"]), 1.0, (ox, oy))
                   for ox, oy in ((s, s), (s, -s), (-s, s), (-s, -s))]
    passes.append((tuple(job.get("color") or (255, 255, 255)), 1.0, (0.0, 0.0)))
    out: list[Layer] = []
    th = math.radians(rot)
    c, s = math.cos(th), math.sin(th)
    for color, k, (dx, dy) in passes:
        try:
            tp = tv.text_plan(text, font=font, height=h * k, color=color, cat=cat)
        except Exception:
            continue
        lk = compose.look(tp, cat)
        lcx, lcy = lk.center
        # 잉크 중심이 (cx,cy)+오프셋에 오도록 이동을 역산하고 rot으로 돌린다
        ox = cx + dx * c - dy * s - (lcx * c - lcy * s)
        oy = cy + dx * s + dy * c - (lcx * s + lcy * c)
        out += _compose_group(tp, x=ox, y=oy, scale=1.0, rot=rot, mirror=False)
    return out


def _shape_layer(spec: dict) -> Layer:
    """위저드 도형 명세 → Layer (면 유닛 그대로)."""
    r, g, b = spec.get("rgb") or (255, 255, 255)
    return Layer(shape=spec.get("shape") or "A_01",
                 x=float(spec["x"]), y=float(spec["y"]),
                 sx=float(spec["sx"]), sy=float(spec.get("sy", spec["sx"])),
                 rot=float(spec.get("rot") or 0.0),
                 color=(int(r), int(g), int(b)))


def _chunk_name(plan_ref: str) -> str:
    """구성 파일이 가리키는 플랜 경로 → **그 덩어리의 이름**.

    이름은 파일 이름 그대로다 (`decal-0-fit.json` → `decal-0-fit`) — 구성기가
    쓰는 이름 규약(`compose.groups`·`compose.build`)이 곧 사람이 편집기 레이어
    나무에서 읽을 이름이 된다."""
    return Path(str(plan_ref)).stem


def surface_chunks(item: dict, cfg_dir: Path,
                   cat: Catalog) -> list[tuple[str, list[Layer]]]:
    """면 배치 하나 → **이름 붙은 덩어리** 목록 (그리는 순서대로).

    `surface_layers`가 이것을 평평하게 편 것이다 — 자를 하나만 둔다. 이름이
    따로 필요한 데는 FLS 리버리 프로젝트다: 덩어리마다 편집기 그룹으로 서야
    다시 열었을 때 **무엇이 사람 몫이고 무엇이 구성기 몫인지** 갈린다
    (`engine.fls.studio`).
    """
    out: list[tuple[str, list[Layer]]] = []
    shapes = [_shape_layer(s) for s in item.get("shapes") or []]
    if shapes:
        out.append(("shapes", shapes))
    for g in item.get("pre_groups") or []:
        gp = LayerPlan.load(cfg_dir / g["plan"])
        out.append((_chunk_name(g["plan"]), _compose_group(
            gp, x=float(g.get("x", 0)), y=float(g.get("y", 0)),
            scale=float(g.get("scale", 1)), rot=float(g.get("rot", 0)),
            mirror=bool(g.get("mirror")))))
    if item.get("plan"):
        gp = LayerPlan.load(cfg_dir / item["plan"])
        out.append((_chunk_name(item["plan"]), _compose_group(
            gp, x=float(item.get("x", 0)), y=float(item.get("y", 0)),
            scale=float(item.get("scale", 1)), rot=float(item.get("rot", 0)),
            mirror=bool(item.get("mirror")))))
    # 사람이 앉힌 도안들 — 목록 순서대로 (뒤가 위). `place_one`의 순서와 같다.
    for g in item.get("groups") or []:
        gp = LayerPlan.load(cfg_dir / g["plan"])
        out.append((_chunk_name(g["plan"]), _compose_group(
            gp, x=float(g.get("x", 0)), y=float(g.get("y", 0)),
            scale=float(g.get("scale", 1)), rot=float(g.get("rot", 0)),
            mirror=bool(g.get("mirror")))))
    post = [_shape_layer(s) for s in item.get("post_shapes") or []]
    if post:
        out.append(("shapes-over", post))
    text: list[Layer] = []
    for job in _as_list(item.get("text")):
        text += _text_layers(job, cat)
    if text:
        out.append(("text", text))
    return out


def surface_layers(item: dict, cfg_dir: Path, cat: Catalog) -> list[Layer]:
    """면 배치 하나 → **면 유닛 좌표**의 레이어 목록 (그리는 순서대로).

    게임이 그 면에 올리는 것 전부다: 바닥 도형 → 꾸밈 그룹 → 도안 그룹 →
    사람이 앉힌 그룹들 → 덮개 도형 → 글자. 미리보기(`render_surface`)와
    리버리 파일 내보내기(`engine.fls.bridge`)가 **같은 이 목록**을 받는다 —
    그림과 파일이 갈리면 미리보기가 거짓말이 된다.
    """
    out: list[Layer] = []
    for _name, chunk in surface_chunks(item, cfg_dir, cat):
        out += chunk
    return out


def render_surface(item: dict, smap, cfg_dir: Path, cat: Catalog,
                   base_rgb: tuple[int, int, int] | None,
                   mask_map=None, exposure=None) -> np.ndarray:
    """면 하나의 미리보기 (RGB).

    `mask_map`을 주면 **어둡게 덮는 마스크만** 그쪽 것을 쓴다 — 옆면은 설치
    마스크가 그린하우스까지 흰 판으로 갖고 있지만 게임은 벨트라인 위에 안
    그리므로(`game.seam`), 차체 지도로 덮어야 미리보기가 실물과 같아진다.

    `exposure`(`compose.surface_exposure` — `mask_map` 격자 위 0~1)를 주면 **면이
    차에서 달아나 도안이 문질리는 자리**를 그만큼 더 어둡게 깐다.
    """
    u0, v0, u1, v1 = smap.paint
    w_u, h_u = u1 - u0, v1 - v0
    upp = max(w_u / MAX_W, 0.5)              # 유닛/px
    W = max(64, int(round(w_u / upp)))
    H = max(48, int(round(h_u / upp)))
    ucx, ucy = (u0 + u1) / 2, (v0 + v1) / 2

    layers: list[Layer] = []
    layers.append(Layer(shape="A_01", x=0.0, y=0.0,
                        sx=w_u / 2 / compose.UNITS_PER_SCALE,
                        sy=h_u / 2 / compose.UNITS_PER_SCALE,
                        color=base_rgb or (128, 130, 134)))
    # 미리보기 캔버스는 도색 상자 중심이 원점이다 — 면 유닛에서 그만큼 민다
    for l in surface_layers(item, cfg_dir, cat):
        l.x -= ucx
        l.y -= ucy
        layers.append(l)

    plan = LayerPlan(source_image="preview", image_size=(W, H),
                     units_per_px=upp, layers=layers)
    img = render_plan(plan, cat)

    return shade(img, smap.paint, upp,
                 mask_map if mask_map is not None else smap, exposure)


def shade(img: np.ndarray, box: tuple[float, float, float, float], upp: float,
          mm, exposure=None) -> np.ndarray:
    """도색 마스크 밖은 어둡게, **차에서 안 보이는 자리는 그만큼** 어둡게 (제자리).

    면의 실제 모양이 보이게 하는 자다. `exposure`는 `mm` 마스크 격자 위의 0~1
    노출도이고 (`compose.surface_exposure`), 마스크 밖과 같은 바닥까지 내려간다.
    """
    m = mm.mask
    if m.size <= 1:
        return img
    u0, _v0, _u1, v1 = box
    H, W = img.shape[:2]
    a0, b0, a1, b1 = mm.paint
    mh, mw = m.shape
    us = (np.arange(W) + 0.5) * upp + u0
    vs = v1 - (np.arange(H) + 0.5) * upp
    xs = np.clip((us - a0) / max(1e-6, a1 - a0) * (mw - 1), 0, mw - 1).astype(int)
    ys = np.clip((b1 - vs) / max(1e-6, b1 - b0) * (mh - 1), 0, mh - 1).astype(int)
    sel = m[np.ix_(ys, xs)]
    sel &= (us[None, :] >= a0) & (us[None, :] <= a1)
    sel &= (vs[:, None] >= b0) & (vs[:, None] <= b1)
    k = np.where(sel, 1.0, compose.EXPOSED_FLOOR)
    if exposure is not None:
        e = np.asarray(exposure, np.float32)[np.ix_(ys, xs)]
        k = np.minimum(k, np.clip(
            compose.EXPOSED_FLOOR + (1.0 - compose.EXPOSED_FLOOR)
            * e / max(1e-6, compose.EXPOSED_FULL), compose.EXPOSED_FLOOR, 1.0))
    img[:] = np.clip(img * k[..., None], 0, 255).astype(np.uint8)
    return img


def seam_view(side_img: np.ndarray, side_smap, win_img: np.ndarray, win_smap,
              rig) -> np.ndarray:
    """옆면 + **유리 면을 옆면 좌표로 되돌린 것**을 한 장에 — 이음새 검사판.

    머리가 벨트라인에서 끊기지 않고 유리로 이어지는지는 두 면을 따로 봐서는
    알 수 없다. 유리 면 렌더를 이음새 변환의 역으로 옆면 유닛에 되쏘면 실차에서
    보일 옆모습이 나온다 (게임의 3D 투영·유리 반투명은 여전히 없다).
    """
    out = side_img.copy()
    sm = rig.seam
    if sm is None or win_img is None:
        return out
    H, W = out.shape[:2]
    u0, v0, u1, v1 = side_smap.paint
    us = u0 + (np.arange(W) + 0.5) / W * (u1 - u0)
    vs = v1 - (np.arange(H) + 0.5) / H * (v1 - v0)
    U, V = np.meshgrid(us, vs)
    wu, wv = sm.to_window(U, V)
    a0, b0, a1, b1 = win_smap.paint
    gh, gw = win_img.shape[:2]
    xi = np.round((wu - a0) / max(1e-6, a1 - a0) * (gw - 1)).astype(int)
    yi = np.round((b1 - wv) / max(1e-6, b1 - b0) * (gh - 1)).astype(int)
    ok = (xi >= 0) & (xi < gw) & (yi >= 0) & (yi < gh) & (V >= rig.geom.belt)
    # 유리 도색 마스크 안만 (면 밖은 게임도 안 그린다)
    m = win_smap.mask
    mh, mw = m.shape
    mxi = np.clip((wu - a0) / max(1e-6, a1 - a0) * (mw - 1), 0, mw - 1).astype(int)
    myi = np.clip((b1 - wv) / max(1e-6, b1 - b0) * (mh - 1), 0, mh - 1).astype(int)
    ok &= m[myi, mxi]
    out[ok] = win_img[np.clip(yi, 0, gh - 1), np.clip(xi, 0, gw - 1)][ok]
    # 벨트라인을 가는 선으로 표시 — 이음새가 어디인지 눈에 보이게
    r = int(round((v1 - rig.geom.belt) / max(1e-6, v1 - v0) * (H - 1)))
    if 0 <= r < H:
        out[r, ::6] = (255, 90, 40)
    return out


def _as_list(v) -> list[dict]:
    if not v:
        return []
    return [v] if isinstance(v, dict) else list(v)


def smap_of(maps: dict, name: str):
    return maps[name]


def render_config(cfg_path: Path, out_path: Path | None = None,
                  car: str | None = None, media: str | None = None,
                  cat: Catalog | None = None, log=print,
                  faces_out: "dict[str, np.ndarray] | None" = None
                  ) -> Path | None:
    """구성 파일 → 면별 미리보기 시트 PNG. 지도가 없으면 None.

    `faces_out`을 주면 시트에 붙이기 전의 **면별 그림**(RGB)을 면 이름으로
    담아 준다 — 구성 전체를 한 번에 입혀 볼 때 쓴다.
    """
    cat = cat or Catalog(default_catalog_path())
    raw = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
    car = car or raw.get("car")
    # 구성이 설치 차량을 적어 뒀으면 그걸 쓴다 — 이름 매칭이 딴 차를 물면
    # 미리보기가 있지도 않은 면 위에 그려진다
    maps = compose.surfaces_for(car, media=media or raw.get("media"))
    if not maps:
        log(msg("미리보기 생략: 면 지도가 없다 (실측도 설치 파일도)"))
        return None
    base = None
    if raw.get("paint", {}).get("rgb"):
        base = tuple(int(v) for v in raw["paint"]["rgb"])
    try:
        rigs = compose.side_rigs(maps, media=media or raw.get("media"))
    except Exception:                              # 뼈대가 안 서도 미리보기는 나온다
        rigs = {}
    panels: dict[str, np.ndarray] = {}
    for item in raw.get("placements", []):
        name = item.get("surface")
        smap = maps.get(name)
        if smap is None:
            continue
        mm = compose.drawable(name, maps, rigs) or smap
        try:
            panels[name] = render_surface(
                item, smap, Path(cfg_path).parent, cat, base, mask_map=mm,
                exposure=compose.surface_exposure(
                    name, mm, maps, media or raw.get("media")))
        except Exception as e:
            log(msg("미리보기 {name} 실패: {kind}: {err}",
                    name=name, kind=type(e).__name__, err=e))
    if not panels:
        return None
    if faces_out is not None:
        faces_out.update(panels)       # 이음새 검사판이 붙기 전 — 면 이름 그대로
    # 이음새 검사판 — 옆면 + 유리를 옆면 좌표로 되돌린 합성 (맨 위에 둔다)
    order = list(SHEET_ORDER)
    for sname in ("side_left", "side_right"):
        rig = rigs.get(sname)
        wname = "window_" + sname.split("_")[1]
        if rig is None or rig.seam is None or sname not in panels:
            continue
        wmap = maps.get(wname)
        if wmap is None or wname not in panels:
            continue
        try:
            key = sname + "+glass"     # 라벨은 OpenCV가 그리므로 ASCII만 쓴다
            panels[key] = seam_view(panels[sname], smap_of(maps, sname),
                                    panels[wname], wmap, rig)
            order.insert(0, key)
        except Exception as e:
            log(msg("미리보기 이음새({name}) 실패: {kind}: {err}",
                    name=sname, kind=type(e).__name__, err=e))
    # 세로로 쌓는다 (면 이름 라벨 띠 포함)
    cols: list[np.ndarray] = []
    W = max(p.shape[1] for p in panels.values()) + 8
    for name in order:
        p = panels.get(name)
        if p is None:
            continue
        strip = np.full((22, W, 3), 30, np.uint8)
        cv2.putText(strip, name, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 220, 240), 1, cv2.LINE_AA)
        pad = np.full((p.shape[0], W, 3), 46, np.uint8)
        pad[:, :p.shape[1]] = p
        cols += [strip, pad]
    sheet = np.concatenate(cols, axis=0)
    out_path = out_path or run_file(Path(cfg_path).parent, "preview_itasha.png")
    cv2.imencode(".png", sheet[:, :, ::-1])[1].tofile(str(out_path))
    # `gui.window.shell._log`가 이 문구로 미리보기를 건다 — 같은 msg 템플릿이다
    log(msg("미리보기: {path}", path=out_path))
    return out_path
