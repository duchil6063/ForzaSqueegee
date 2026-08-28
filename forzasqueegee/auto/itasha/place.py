"""차체 배치 — `비닐 & 데칼 적용`에서 면 탭마다 그룹을 불러와 앉힌다.

올린 뒤 화면으로 **발자국을 재서** 목표 상자와 견주고(`measure_placement`),
어긋나면 변형 편집에서 다시 고친다(`autofit`). 사람이 앉힌 자리는 되맞출 목표가
없으므로 구성이 fit을 끈다."""

from __future__ import annotations

import json
import time
from pathlib import Path

from ...game import body
from ..bodyedit import BodyEditor
from ..driver import DriverError
from .config import Config, Placement
from .progress import Clock, save_progress
from .cartabs import check_car_tabs
from .shapes import add_shape_jobs, add_text_job


def place_all(cfg: Config, prog: dict, log=print,
              replace: bool = True, fit: bool = True,
              clock: Clock | None = None) -> BodyEditor:
    from ...engine import compose
    from ...game import surface as gsurf
    from ..fav import FavStack

    clock = clock or Clock()
    b = BodyEditor()
    with clock.stage("place.enter"):
        b.enter_editor()
        check_car_tabs(cfg, b, log=log)  # 다른 차에 올리기 전에 여기서 멈춘다
    shots = cfg.path.parent / "_itasha"
    # 오토핏은 **화면 warp**를 쓴다 — 못 박은 차와 다른 차의 실측 지도로 재면
    # 도안을 엉뚱한 쪽으로 민다 (`compose.probe_ok`).
    smaps = (gsurf.load(cfg.car)
             if cfg.car and compose.probe_ok(cfg.car, cfg.media) else {})
    fav = FavStack()      # 면 도형 색 — 같은 모티프 색이 면마다 되풀이된다
    for p in cfg.placements:
        if p.key in prog["placed"]:
            log(f"{p.surface}: 이미 올렸다 — 건너뛴다")
            continue
        with clock.stage("place", of=p.surface,
                         n=p.group_layers + len(p.shapes) + len(p.post_shapes)):
            place_one(b, p, log=log, shots=shots, replace=replace,
                      smaps=smaps, fit=fit, clock=clock, fav=fav)
        prog["placed"].append(p.key)
        save_progress(cfg, prog)
    return b


def _shot(b: BodyEditor, path: Path) -> None:
    """면 캡처를 남긴다 — 배치 검증의 증거 (`ITASHA_PLAN` §1의 검증 경계)."""
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".png", b.cap()[:, :, ::-1])[1].tofile(str(path))


TINY_SCALE = 0.02          # 발자국을 재려고 잠깐 줄이는 크기


def footprint_of(b: BodyEditor, p: Placement, log=print):
    """올라간 그룹의 **발자국 마스크** — 같은 화면에서 두 번 잡아 차분한다.

    기준 캡처를 **목록 화면**에서 뜨면 안 된다: 변형 편집으로 들어가며 카메라·노출이
    바뀌어 차분이 데칼 밖까지 문다 (2026-08-17 실측: 발자국이 도색 상자보다 넓게
    나와 오토핏이 스케일을 0.045로 깎았다). 대신 **그룹을 잠깐 줄인 화면**을
    기준으로 쓴다 — 화면 상태가 같으니 차분에 데칼만 남는다.
    """
    from ...game import surface as gsurf

    try:
        b.d.set_axis("sx", TINY_SCALE)
    except DriverError:
        b.d.set_axis("sx", TINY_SCALE * 2)     # 레일 근처 플레이크 — 한 발 물러서 재시도
    time.sleep(0.5)
    ref = b.cap()
    time.sleep(0.3)
    b.d.set_axis("sx", p.scale)
    time.sleep(0.5)
    now = b.cap()
    return gsurf.footprint_mask(ref, now), now


def measure_placement(b: BodyEditor, bg, smap, p: Placement,
                      log=print) -> dict | None:
    """화면으로 **올라간 도안의 발자국**을 재서 목표 상자와 견준다.

    반환: {`box`(면 유닛), `target`, `err`(중심 오차·크기비), `scale_fix`} 또는 None.
    비닐 캔버스에서 끝나던 검증이 **면 위 수치까지** 오는 자리다 — 곡면 렌더
    품질은 여전히 안 잰다 (게임의 투영이라 우리가 고칠 축이 없다).
    """
    from ...game import surface as gsurf

    if smap is None:
        return None
    m, _cap = footprint_of(b, p, log=log)
    bb = gsurf.bbox(m)
    if bb is None:
        return None
    u0, v1 = smap.px_to_unit(bb[0], bb[1])
    u1, v0 = smap.px_to_unit(bb[2], bb[3])
    got = (u0, v0, u1, v1)
    out: dict = {"box": [round(v, 1) for v in got],
                 "px": [int(v) for v in bb]}
    # **면 밖으로 나간 발자국은 발자국이 아니다** — 차분이 데칼 아닌 것을 물었다는
    # 뜻이라, 그런 판독으로 보정하면 오히려 망친다.
    pad = 0.15 * max(smap.width, 1.0)
    if (u0 < smap.paint[0] - pad or u1 > smap.paint[2] + pad
            or v0 < smap.paint[1] - pad or v1 > smap.paint[3] + pad):
        out["outside"] = True
        return out
    if p.target is not None:
        t = p.target
        gw, gh = max(1e-6, u1 - u0), max(1e-6, v1 - v0)
        tw, th = t[2] - t[0], t[3] - t[1]
        out["target"] = [round(v, 1) for v in t]
        out["size_ratio"] = [round(gw / max(1e-6, tw), 3), round(gh / max(1e-6, th), 3)]
        out["center_err"] = [round((u0 + u1) / 2 - (t[0] + t[2]) / 2, 1),
                             round((v0 + v1) / 2 - (t[1] + t[3]) / 2, 1)]
        out["scale_fix"] = round(min(tw / gw, th / gh), 3)
    return out


def autofit(b: BodyEditor, bg, smap, p: Placement, got: dict[str, float],
            log=print, rounds: int = 2, tol: float = 0.06) -> dict | None:
    """변형 편집에서 **재고 고친다** — 발자국이 목표 상자에 들어맞을 때까지.

    확정(`commit`) 전에 도는 것이 값이다: 캔버스 유닛 ↔ 면 유닛 환산(`group_unit`)을
    몰라도, 차종·도안이 무엇이어도 화면에서 잰 것으로 수렴한다. 두 바퀴면 충분하다
    (스케일은 곱셈 보정이라 한 바퀴에 거의 맞는다).
    """
    if p.target is None or smap is None:
        return None
    meas = None
    scale0 = p.scale
    for k in range(rounds):
        meas = measure_placement(b, bg, smap, p, log=log)
        if meas is None or "scale_fix" not in meas or meas.get("outside"):
            if meas is not None and meas.get("outside"):
                log("    발자국이 면 밖까지 물렸다 — 판독을 버리고 계산값을 쓴다")
            return meas
        # 보정은 **한 바퀴에 크게 못 움직인다** — 판독이 한 번 새면 그 한 번으로
        # 도안이 사라지는 것을 막고, **일부러 면 밖으로 흘린 요소**(띠·데코) 때문에
        # 발자국이 목표에 구조적으로 못 미치는 상황에서 상한까지 밀어붙이는 것도
        # 막는다 (2026-08-18 실측: ±40%를 주니 인물이 벨트라인을 넘게 커졌다).
        fix = min(1.2, max(0.8, meas["scale_fix"]))
        cx, cy = meas["center_err"]
        tw = max(1e-6, p.target[2] - p.target[0])
        off = max(abs(cx), abs(cy)) / tw
        log(f"    발자국 {meas['box']} · 크기비 {meas['size_ratio']} "
            f"· 중심오차 {meas['center_err']}")
        if abs(fix - 1.0) <= tol and off <= tol:
            return meas
        p.scale = round(min(scale0 * 1.25, max(scale0 * 0.75, p.scale * fix)), 3)
        p.x = round(p.x - cx, 1)
        p.y = round(p.y - cy, 1)
        log(f"    보정 {k + 1}회 → scale {p.scale:g} x {p.x:g} y {p.y:g}")
        # **회전은 지금 읽힌 값을 쓴다.** 미러는 Tab + 180°인데 Tab은 값 칸에 안
        # 뜨므로, 여기서 p.rot(=0)을 다시 넣으면 미러가 반쪽만 남아 인물이 뒤집힌다.
        got.update(b.place(x=p.x, y=p.y, scale=p.scale,
                           rot=got.get("rot", p.rot), mirror=False,
                           soft=True, log=log))
    return measure_placement(b, bg, smap, p, log=log)


def place_one(b: BodyEditor, p: Placement, log=print,
              shots: Path | None = None, replace: bool = True,
              smaps: dict | None = None, fit: bool = True,
              clock: Clock | None = None, fav=None) -> None:
    clock = clock or Clock()
    idx = p.tab if p.tab is not None else body.surface_index(p.surface)
    b.goto_tab(idx)
    n0, cap = b.count_stable(), b.counts()[1]
    if n0 is None or cap is None:
        raise DriverError(f"{p.surface}: 면 카운터를 못 읽었다 — 차체 에디터 화면이 맞나")
    # **면을 비우고 올리는 것이 기본이다.** 구성 파일이 "이 면은 이 도안"이라고
    # 말하고 있고, 마지막에 차 디자인을 통째로 덮으므로 그 면의 옛 레이어만
    # 남겨 봐야 얻는 것이 없다 — 게다가 안 비우면 두 번째 실행이 예산 초과로
    # 통째로 멈춘다. 남기고 싶으면 `--keep-existing`.
    if n0:
        if not replace:
            raise DriverError(
                f"{p.surface}: 이미 {n0:,}장이 있다 (--keep-existing) — "
                f"게임에서 그 면을 비우고 다시 할 것")
        log(f"{p.surface}: 이미 {n0:,}장이 있다 — 비우고 올린다")
        with clock.stage("place.clear", of=p.surface, n=n0):
            b.clear_surface()
        n0 = b.count_stable() or 0
    if p.copy_from is not None:
        log(f"{p.surface}: 반대편({p.copy_from}) 레이어 붙여넣기")
        b.paste_other_side()
        n1 = b.count_stable()
        log(f"  카운터 {n0} → {n1} / {cap}")
        return
    if (p.group_layers == 0 and not p.texts and not p.shapes
            and not p.post_shapes):
        raise DriverError(f"{p.surface}: 빈 플랜 (글자·도형도 없다)")
    want = p.group_layers + p.text_layers + len(p.shapes) + len(p.post_shapes)
    if n0 + want > cap:
        raise DriverError(
            f"{p.surface}: 이미 {n0:,}장이 있어 {want:,}장을 "
            f"더 못 올린다 (상한 {cap:,}) — 그 면을 비우고 다시 할 것")
    # 순서 = 그려지는 순서다 (나중 것이 위): 밑 도형(블랙아웃·관통 띠·모티프) →
    # 꾸밈 그룹 → 그룹(도안·넘어온 조각) → 덮개 도형 → 글자. 도안 뒤에 깔릴
    # 것은 전부 그룹을 불러오기 전에 면에 넣는다.
    n_shapes = add_shape_jobs(b, p.shapes, log=log, clock=clock,
                              of=p.surface, fav=fav)
    for g in p.pre_groups:
        log(f"{p.surface}: 보조 그룹 '{g.group}' ({g.layers:,}장) 불러오는 중")
        with clock.stage("place.group", of=f"{p.surface}/{g.group}", n=g.layers):
            b.open_group_grid()
            b.load_group(g.layers)
            gg = b.place(x=g.x, y=g.y, scale=g.scale, rot=g.rot,
                         mirror=g.mirror, soft=True, log=log)
            b.commit()
        log(f"  배치 {gg}")
    got = {}
    meas = None
    if p.layers:
        log(f"{p.surface}: 그룹 '{p.group}' ({p.layers:,}장) 불러오는 중")
        with clock.stage("place.group", of=f"{p.surface}/{p.group}", n=p.layers):
            bg = b.cap()               # (참고용 캡처 — 발자국은 `footprint_of`가 잰다)
            b.open_group_grid()
            b.load_group(p.layers)
            got = b.place(x=p.x, y=p.y, scale=p.scale, rot=p.rot,
                          mirror=p.mirror, soft=True, log=log)
            smap = (smaps or {}).get(p.surface)
            if fit and p.fit and smap is not None and p.target is not None:
                with clock.stage("place.autofit", of=p.surface):
                    meas = autofit(b, bg, smap, p, got, log=log)
            b.commit()
        log(f"  배치 {got}")
    # 사람이 앉힌 도안들 — 목록 순서대로, 뒤가 위다. 오토핏은 안 돈다: 자리가
    # 사람의 결정이라 화면으로 다시 맞출 목표가 없다 (`engine.compose`가 fit을 끈다).
    for g in p.groups:
        log(f"{p.surface}: 도안 그룹 '{g.group}' ({g.layers:,}장) 불러오는 중")
        with clock.stage("place.group", of=f"{p.surface}/{g.group}", n=g.layers):
            b.open_group_grid()
            b.load_group(g.layers)
            gg = b.place(x=g.x, y=g.y, scale=g.scale, rot=g.rot,
                         mirror=g.mirror, soft=True, log=log)
            b.commit()
        log(f"  배치 {gg}")
    n_post = add_shape_jobs(b, p.post_shapes, log=log, clock=clock,
                            of=p.surface, fav=fav)
    for spec in p.texts:
        with clock.stage("place.text", of=p.surface,
                         n=int(spec.get("layers") or 0)):
            add_text_job(b, spec, log=log)
    want = p.group_layers + p.text_layers + n_shapes + n_post
    n1 = b.count_stable()
    log(f"  카운터 {n0} → {n1} / {cap}")
    if shots is not None:
        _shot(b, shots / f"{p.surface}.png")
        if meas is not None:
            (shots / f"{p.surface}.json").write_text(
                json.dumps({"surface": p.surface, "group": p.group,
                            "layers": p.layers, "place": got,
                            "measured": meas}, ensure_ascii=False, indent=1),
                encoding="utf-8")
    if n1 is None or n1 - n0 != want:
        raise DriverError(
            f"{p.surface}: 카운터가 {n0} → {n1}이다 (기대 {want:,}장 = 그룹 "
            f"{p.group_layers:,} + 글자 {p.text_layers} + 도형 {n_shapes + n_post}) — "
            f"다른 그룹을 불러왔거나 배치가 안 확정됐다")
