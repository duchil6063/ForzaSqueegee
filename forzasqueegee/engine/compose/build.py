"""구성 한 대 짜기 — 이 패키지의 진입점."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path

from ...game import fold as gfold, seam as gseam, surface as gsurf
from ...i18n import msg
from ...paths import run_file
from ..catalog import Catalog, default_catalog_path
from ..model import LayerPlan, rgb_to_hsb
from .boxes import (
    CANVAS_UNITS, _clamp_box, _face_phase, _gap, _group_unit, _overlap, _rel,
    _union)
from .look import Look, look, person_ink, rot_ink_box
from .palette import accent_color, accent_third, accent_tint, base_paint, contrast_ink
from .vocabulary import MOTIF_FAMILIES, MOTIF_SETS, edge_shapes, motif_shapes
from .scatter import DECO_FRONT_N, DECO_FRONT_SIZE
from .bands import ROCKER_BASE_MIN
from .roof import ROOF_DARK, hood_index, roof_blackout, top_segments
from .place import (
    BODY_BIAS, BODY_FILL, ROLE_EXTRA, ROLE_MAIN, ROLE_REAR, ManualPlace, dodge_parts,
    drawable, manual_box, person_pose, person_tilt, place_in_rect, place_xf)
from .folds import _all_folds
from .autoplace import auto_place
from .surfshapes import GLASS, DecoAnchor, deco_anchor, flow_shapes, surface_deco_shapes
from .intent import read_intent
from .design import Design, compose_design
from .families import FAMILIES
from .textspec import TextSpec
from .textbuild import mirrored_set
from .facetext import face_text
from .rigs import (
    _bumper_seed, _hood_seed, _place_for, carfiles_pick, side_rigs, surfaces_for)
from .groups import (
    _hand_group_job, _hand_spread, _unique_group_counts)


@dataclass
class Recipe:
    """구성 결과 — 새로 쓴 도안 파일들과 면 배치."""

    config: dict
    written: list[Path] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def build(main_plan: Path, out_dir: Path, *, car: str | None = None,
          media: str | None = None, extra_plans: list[Path] | None = None,
          mirror: bool = True, apply: bool = True, paint: bool = True,
          base_rgb: tuple[int, int, int] | None = None, flip: bool = False,
          preset: dict[str, dict[str, float]] | None = None,
          cat: Catalog | None = None,
          manual: list[ManualPlace] | None = None, deco: bool = True,
          motif: str | None = None, family: str | None = None,
          text: "TextSpec | dict | None" = None, log=print) -> Recipe:
    """도안 + 실측 → **이타샤 구성 파일**을 쓴다 (게임은 안 건드린다).

    나오는 것은 `auto.itasha`가 그대로 먹는 구성이다. 자동이든 손 배치든 **같은
    길**이다: 자동 경로는 도안을 앉힐 자리(`_place_people` — 눕히기 각·차체 밴드
    예산·후드 재사용)를 계산해 손 배치 목록으로 바꿔 놓고, 그 뒤는 손 배치와 한
    코드가 짓는다 — 면 밖 레이어 빼기(`_hand_spread`), 꾸밈 그룹(베드·띠·산포)·
    관통 띠·지붕 블랙아웃. 그래서 편집기에 도안을 넣고 아무것도 안 만지면
    자동 이타샤가 그대로 나온다.

    `flip`은 **도안 좌우반전**이다 — 같은 말로 "왼쪽·오른쪽 중 어느 면을 원본
    그대로 쓸 것인가"이고, 둘은 같은 레버다 (좌우는 서로의 거울이므로 한쪽을
    뒤집으면 다른 쪽도 뒤집힌다). 기본은 왼쪽이 원본, 오른쪽이 미러다.

    **왜 필요한가**: 인물을 눕히면 그림의 좌우축이 세로로 선다 — 회전 −80°에서
    캔버스 +x가 아래(사이드실)로 간다. 그래서 인물이 어느 쪽 옆구리를 바닥에
    두고 눕는지가 이 한 비트로 갈리고, 그림에 따라 한쪽은 **얼굴이 땅을 보는**
    꼴이 된다. 기하만으로는 어느 쪽이 옳은지 못 정한다 (그림이 어디를 보는지에
    달렸다) — 그래서 자동으로 안 고르고 레버로 낸다.

    `base_rgb`는 **사람이 정한 베이스 도색**이다 — 없으면 주역 도안에서 고른다
    (`base_paint`). 편집기의 [Base Paint]가 이 레버다: 계산된 색으로 시작하고 사람이
    바꾸면 그 색이 온다.

    ## 손 배치 (`manual`)

    `manual`을 주면 **도안 자리는 사람이 정한 것을 그대로 쓴다** — 눕히기·면
    예산·후드 같은 자동 앉히기를 전부 건너뛴다. 나머지는 하나도 안 달라진다:
    베이스 도색·꾸밈 그룹·관통 띠·지붕 블랙아웃이 사람이 앉힌 상자를 기준으로
    그대로 선다. `main_plan`은 손 배치에서도 **주역**이다 — 베이스 도색이 이
    도안의 팔레트에서 나온다 (부르는 쪽이 가장 크게 앉은 도안을 넘긴다).

    **면을 넘긴 몫은 그 자리에서 잘린다** — 이웃 면으로 안 잇는다 (사용자 지시
    2026-08-27). 감아 돌리고 싶으면 편집기에서 도안을 이음선으로 가르고
    (KFPS·FLS의 [선으로 가르기]) 한쪽을 그 면에 따로 올린다.

    ## 꾸밈 (`deco`)

    끄면 **도안만** 올린다 — 꾸밈 그룹(로커·산포)·관통 밴드·
    지붕 블랙아웃·모티프가 전부 빠지고, 그것만 있던 면은 구성에서 사라진다.
    베이스 도색은 별개 레버다 (`paint`): 도안만 올리더라도 차 색은 정해야 한다.

    ## 모티프 계열 (`motif`)

    안 주면 도안의 테마색이 고른다 (`motif_family`). 주면 그 계열로 못 박는다 —
    **계열은 원래 캐릭터 의미에서 오는 것**이라 팔레트로는 거기까지 못 간다
    (수이세이가 별인 것은 이름이 '별마을 혜성'이라서다). 베이스 도색의
    `base_rgb`와 같은 자리의 레버다: 자동으로 정해 주고 사람이 바꾼다.

    ## 구성 계열 (`family`)

    옆면 꾸밈은 **후보를 여럿 지어 점수로 고른다** (`design.compose_design` —
    계열 × 흐름 × 팔레트 변종 × 베드 크기). `family`를 주면 그 계열 안에서만
    고른다 (`families.FAMILIES`). 사람이 앉힌 도안은 어느 후보에서도 안 움직인다.

    ## 글자 (`text`)

    기본은 **안 넣는다** (이타샤 어휘에 글자가 없다는 규칙은 그대로다). 스펙
    (`textspec.TextSpec` 또는 그 dict)을 켜서 주면 캐릭터 이름(+작품명)이 꾸밈의
    한 요소로 후보에 들어간다 — 커스텀 텍스트 도안(동봉 OFL 글꼴, `engine.textglyph`)
    이 기본이고 면 예산이 모자라면 층을 낮추다 게임 글꼴 비닐로 물러나고 그래도
    안 되면 뺀다 (`textbudget`). 옆면 글자는 면마다 제 그룹(`text-<면>.json`)이다
    — 꾸밈 그룹은 좌우를 미러로 나눠 쓰지만 글자는 뒤집히면 안 된다.
    """
    text_spec = (text if isinstance(text, TextSpec) else TextSpec.from_dict(text)) \
        if text is not None else TextSpec()
    mirror_side = "side_left" if flip else "side_right"
    from ...auto.itasha import PRESET             # 순환 참조를 피해 늦게 들여온다

    if motif is not None and motif not in MOTIF_SETS:
        raise ValueError(msg("모르는 모티프 계열: {motif!r} (있는 것: {families})",
                             motif=motif, families=", ".join(MOTIF_FAMILIES)))
    if family is not None and family not in FAMILIES:
        raise ValueError(msg("모르는 구성 계열: {family!r} (있는 것: {families})",
                             family=family, families=", ".join(FAMILIES)))
    cat = cat or Catalog(default_catalog_path())
    preset = preset if preset is not None else PRESET
    extra_plans = list(extra_plans or [])
    out_dir.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []
    maps = surfaces_for(car, media=media, notes=notes)
    if car and not maps:
        notes.append(msg("'{car}' 면 실측이 없다 — 전부 프리셋으로 앉힌다", car=car))
    group_unit = _group_unit(car)

    plan = LayerPlan.load(main_plan)
    lk = look(plan, cat)
    # 베이스는 **자동차 도색**이다 — 비닐이 아니라 도색 메뉴에서 칠한다
    # (레퍼런스 이타샤의 베이스가 전부 그렇다. 장수 0장 · 도료 질감 공짜).
    # 사람이 정한 색이 있으면 그것이 이기고, 없으면 도안에서 고른다.
    if base_rgb is not None:
        base_rgb = tuple(int(v) for v in base_rgb)
        base_hsb = tuple(round(v, 2) for v in rgb_to_hsb(*base_rgb))
        notes.append(msg("베이스 도색은 사람이 정했다 — RGB {rgb}", rgb=base_rgb))
    else:
        base_rgb, base_hsb = base_paint(lk)
    car_rgb = base_rgb if paint else (gsurf.car_color(car) if car else None)
    # 후광·톱니·지붕이 쓸 무채색 — 차 색의 반대다 (`contrast_ink`)
    ocol = contrast_ink(car_rgb)
    log(msg("도안 {name}: {layers:,}장 · 잉크 {w:.0f}×{h:.0f}유닛 "
            "(비 {aspect:.2f}, {kind}) · 팔레트 {colors}색",
            name=main_plan.name, layers=lk.layers, w=lk.w, h=lk.h,
            aspect=lk.aspect, kind=lk.kind, colors=len(lk.palette))
        + (msg(" · 베이스 도색 {rgb} (HSB {hsb})", rgb=base_rgb, hsb=base_hsb)
           if paint else ""))

    # ---- 옆면 뼈대 — 벨트라인·로커·루프라인·유리 이음새 + 부품 자리 ----
    pinned = media or carfiles_pick(car)
    rigs = side_rigs(maps, notes, media=pinned)
    # 윗면에서 **어느 구간이 후드인가**를 고르는 씨앗 (`game.locators.hood_u`).
    # 경계는 여전히 마스크가 낸다 — 로케이터로 경계를 재면 p90이 1 m를 넘는다.
    hood_u = _hood_seed(pinned)

    hand = list(manual or [])
    anchor = "bottom" if lk.kind == "tall" else "center"

    def _place_people(tilt: float) -> None:
        """인물을 **차체 밴드**에 앉힌다 — 폭은 문짝, 아래는 사이드실.

        상자는 `person_budget`이 주는 레퍼런스 실측 예산이고, 벨트라인은 안
        넘는다 — 넘긴 몫은 이웃 면으로 안 가고 그 자리에서 잘린다.
        """
        for r in rigs.values():
            r.tilt = tilt if r.name == "side_left" else -tilt
            r.mirror = bool(mirror and r.name == mirror_side)
            wb, hb = gseam.person_budget(r.body, r.geom)
            iw, ih = person_ink(lk, r.tilt, r.mirror)
            s = min(wb / max(1e-6, iw), hb / max(1e-6, ih))   # 예산에 내접
            box = gseam.person_span(r.body, r.geom, (iw * s, ih * s), r.rear_dir)
            # 도어 노브·주유구가 얼굴에 안 겹치게 민다 (업계 지침) — 부품 자리를
            # 아는 차에서만 (`game.locators`). 크기는 안 건드린다.
            box, dodge = dodge_parts(box, r, lk, r.tilt)
            r.place = place_in_rect(box, r.name, lk, anchor="bottom", fill=1.0,
                                    group_unit=group_unit, tilt=r.tilt,
                                    mirror=r.mirror, paint=r.smap.paint)
            over = max(0.0, box[3] - r.geom.belt) / max(1e-6, box[3] - box[1])
            r.place.why = (
                msg("차체 밴드 {sill:.0f}~{belt:.0f}유닛 · 문짝 폭의 {frac:.2f}"
                    " · 벨트라인 위 {over:.0f}%",
                    sill=r.geom.sill, belt=r.geom.belt,
                    frac=iw * s / max(1e-6, wb / gseam.PERSON_DOOR_FILL),
                    over=over * 100)
                + (msg(" · 눕힘 {tilt:g}°", tilt=r.tilt) if r.tilt else "")
                + (f" · {dodge.split(': ', 1)[-1]}" if dodge else ""))

    # 윗면은 **실제로 그리는 지도**로 짠다 — 유리를 잰 차에서는 앞·뒷유리가 구멍
    # 이라 후드·지붕·데크가 진짜 경계로 갈린다 (`top_segments`). 안 잰 차는
    # 지금까지처럼 실루엣 허리로 가르므로 아무것도 안 바뀐다.
    ts = drawable(ROLE_EXTRA, maps, rigs)
    if not hand:
        # ---- 자동 앉히기 → 손 배치 목록으로 ----
        # 눕히기 — 세로 도안은 차체 밴드를 **가로로** 채운다. 각도는 면 예산이
        # 푼다. 뼈대가 없는 차만 기울임 자로 물러난다.
        tilt, tilt_why = person_pose(lk, rigs)
        if not rigs:
            tilt = person_tilt(lk)
            tilt_why = msg("세로 도안(비 {aspect:.2f}) — {tilt:g}° 기울여 앉힌다 "
                           "(옆면 뼈대 없음 — 폴백)",
                           aspect=lk.aspect, tilt=tilt) if tilt else ""
        if tilt_why:
            notes.append(tilt_why)
        _place_people(tilt)
        for i, name in enumerate(ROLE_MAIN):
            rig = rigs.get(name)
            mirror_this = bool(mirror and name == mirror_side)
            if rig is not None and rig.place is not None:
                hand.append(ManualPlace(plan=Path(main_plan), surface=name,
                                        x=rig.place.x, y=rig.place.y,
                                        scale=rig.place.scale, rot=rig.place.rot,
                                        mirror=rig.mirror))
                notes.append(f"{name}: {rig.place.why}")
            else:
                tsign = tilt if name == "side_left" else -tilt
                pl = _place_for(name, lk, maps, preset, anchor=anchor,
                                bias_x=BODY_BIAS if i == 0 else 1.0 - BODY_BIAS,
                                fill=BODY_FILL, group_unit=group_unit,
                                notes=notes, tilt=tsign, mirror=mirror_this)
                hand.append(ManualPlace(plan=Path(main_plan), surface=name,
                                        x=pl.x, y=pl.y, scale=pl.scale,
                                        rot=pl.rot, mirror=mirror_this))
        # 윗면 — 보조 도안이 있으면 그것, 없으면 **후드 인물** (도안 그룹 재사용:
        # 파일이 같아 준비가 공짜다. HINATA 문법 — 본넷을 기울인 인물이 덮는다).
        if extra_plans:
            ep = Path(extra_plans[0])
            ek = look(LayerPlan.load(ep), cat)
            mp = auto_place(ROLE_EXTRA, ep, ek, maps, rigs, group_unit=group_unit)
            if mp is not None:
                hand.append(mp)
            else:
                notes.append(msg("{surface}: 보조 도안을 못 앉힌다 (면 지도가 없다)",
                                 surface=ROLE_EXTRA))
        elif (ts is not None and not ts.uncertain
                and len(plan.layers) <= (ts.cap or 3000)):
            hood = _hood_place(ts, lk, group_unit, hood_u)
            if hood is not None:
                hx, hy, hs, hrot, hwhy = hood
                hand.append(ManualPlace(plan=Path(main_plan), surface=ROLE_EXTRA,
                                        x=hx, y=hy, scale=hs, rot=hrot))
                notes.append(msg("후드에 인물을 기울여 앉힌다 ({why})", why=hwhy))

    # ---- 손 배치 읽기 — 면별 (자동 경로가 만든 것도 여기서부터는 같다) ----
    hand_ix = {id(mp): i for i, mp in enumerate(hand)}
    by_surface: dict[str, list[ManualPlace]] = {}
    for mp in hand:
        by_surface.setdefault(mp.surface, []).append(mp)
    # 도안마다 한 번만 읽는다 (같은 도안을 여러 면에 올리는 것이 기본이다)
    main_key = str(Path(main_plan).resolve())
    hand_look: dict[str, tuple[LayerPlan, Look]] = {}
    for mp in hand:
        if mp.key() == main_key:
            hand_look.setdefault(main_key, (plan, lk))
        elif mp.key() not in hand_look:
            hp = LayerPlan.load(mp.plan)
            hand_look[mp.key()] = (hp, look(hp, cat))
    # 면마다 사람이 덮은 상자 (합집합) — 꾸밈 그룹의 베드가 이걸 자로 쓴다
    hand_box: dict[str, tuple[float, float, float, float]] = {}
    for name, mps in by_surface.items():
        box = None
        for mp in mps:
            b = manual_box(hand_look[mp.key()][1], mp, group_unit)
            box = b if box is None else _union(box, b)
        hand_box[name] = box

    # ---- 그룹 파일 — 도안마다 하나, 필요하면 면에 맞게 자른 사본 ----
    hand_path: dict[str, Path] = {}
    hand_group: dict[int, tuple[Path, int]] = {}   # 배치 색인 → (쓸 파일, 장수)
    for i, key in enumerate(hand_look, 1):
        p = out_dir / f"decal-{i}.json"
        hand_look[key][0].save(p)
        hand_path[key] = p
    _hand_spread(hand, hand_look, hand_path, hand_group, maps, rigs, cat,
                 out_dir, notes, group_unit=group_unit)
    written = list(hand_path.values()) \
        + sorted({p for p, _n in hand_group.values() if p not in hand_path.values()},
                 key=str)

    # ---- 꾸밈 그룹 — 베드·띠·산포를 **면을 자로** 짠다 ----
    # 프레임은 옆면 차체 밴드를 캔버스(900유닛) 좌표로 옮긴 상자이고, 그 안에서
    # 사람이(또는 자동이) 앉힌 도안 상자를 같이 넘겨 베드·모티프 크기가 도안을
    # 따라가게 한다. 좌우는 미러 대칭이라 **한 벌로 두 면이 다 선다** (반대편은
    # 배치 미러가 x를 뒤집는다 — 그룹은 게임에서 한 번만 만들면 된다).
    deco_src = next((n for n in ROLE_MAIN if hand_box.get(n)), None)
    side0 = maps.get(deco_src) if deco_src else None
    deco_plan = deco_place = deco_front = front_place = None
    design: Design | None = None
    face_summary: dict | None = None
    text_groups: dict[str, dict] = {}         # 면 → 글자 그룹 항목 (커스텀 층)
    text_jobs: dict[str, list[dict]] = {}     # 면 → 게임 글자 명세 (층 D)
    if not deco:
        notes.append(msg("꾸밈을 끈 판이다 — 도안(과 넘친 조각)만 올린다"))
    if deco and side0 is not None and not side0.uncertain:
        r0 = rigs.get(deco_src)
        p0, q0, p1, q1 = side0.paint
        vlo, vhi = (r0.geom.sill, r0.geom.belt) if r0 is not None else (q0, q1)
        ds = (p1 - p0) / CANVAS_UNITS / max(1e-6, group_unit)   # 꾸밈 배치 스케일
        u = ds * group_unit                      # 캔버스 1유닛 = 면 u유닛
        fcu, fcv = (p0 + p1) / 2, (vlo + vhi) / 2
        band = vhi - vlo
        pbox = hand_box[deco_src]
        # **안 그려질 자리를 미리 거른다** — 꾸밈은 캔버스 좌표로 앉으므로 면
        # 도색 마스크를 모른다. 휠아치 구멍·벨트라인 위에 떨어진 모티프는 게임이
        # 통째로 안 그려 장수만 먹는다 (미리보기 실측: 26장 중 여섯).
        dm0 = drawable(deco_src, maps, rigs) or side0

        def _drawable_at(cx: float, cy: float, _m=dm0, _u=u,
                         _c=(fcu, fcv)) -> bool:
            return bool(_m.masked_at(_c[0] + _u * cx, _c[1] + _u * cy))

        frame_box = (-CANVAS_UNITS / 2, -band / u / 2,
                     CANVAS_UNITS / 2, band / u / 2)
        person_box = ((pbox[0] - fcu) / u, (pbox[1] - fcv) / u,
                      (pbox[2] - fcu) / u, (pbox[3] - fcv) / u)
        # **사람 배치를 읽는다** — 이 면에서 가장 크게 앉은 도안이 설계의 뿌리다.
        # 그 배치 변환으로 도안 뜻(실루엣·머리·축)을 프레임 좌표에 얹고, 그
        # 위에서 후보를 지어 점수로 고른다 (`design.compose_design`). 도안은
        # 어느 후보에서도 움직이지 않는다.
        root_mp = max(by_surface[deco_src],
                      key=lambda m: (lambda b: (b[2] - b[0]) * (b[3] - b[1]))(
                          manual_box(hand_look[m.key()][1], m, group_unit)))
        root_plan, root_lk = hand_look[root_mp.key()]
        intent = read_intent(root_plan, root_lk, cat)
        L, t = place_xf(root_mp, group_unit)
        side_cap = min([m.cap or 3000 for n in ROLE_MAIN
                        if (m := maps.get(n)) is not None] or [3000])
        side_person = max([sum(hand_group[hand_ix[id(m)]][1]
                               for m in by_surface.get(n) or [])
                           for n in ROLE_MAIN] or [0])
        # 글자가 옆면에 서는 것은 자리가 옆면(또는 자동)일 때다 — 다른 면을 못
        # 박았으면 옆면 설계는 글자 없이 돌고 그 면이 따로 짓는다 (`face_text`)
        side_text = text_spec if (text_spec.active
                                  and text_spec.placement in ("auto", "side")) else None
        design = compose_design(
            root_plan, root_lk, intent, cat, car_rgb, frame_box=frame_box,
            person_box=person_box, L=L, t=t, frame_center=(fcu, fcv), u=u,
            rear_sign=(r0.rear_dir if r0 is not None else 1.0),
            drawable_at=_drawable_at, motif=motif, halo=ocol, family=family,
            phase=_face_phase(deco_src), text=side_text, cap=side_cap,
            n_person=side_person)
        notes += design.notes
        deco_plan = design.plan(plan, cat)
        # **빈 꾸밈 그룹은 안 만든다.** 이미 어두운 차라 로커가 빠지고
        # (`ROCKER_BASE_MIN`) 인물이 차체 밴드를 거의 다 덮으면 남는 것이 하나도
        # 없을 수 있다 — 빈 그룹은 게임 슬롯만 먹고, 장수 0장은 신원(장수)이
        # 없어 불러올 수도 없다.
        if deco_plan is None:
            notes.append(msg("옆면에 깔 꾸밈이 없다 — 인물이 차체 밴드를 다 쓰고 "
                             "차가 이미 어두워 로커도 안 선다"))
        else:
            deco_path = out_dir / "deco.json"
            deco_plan.save(deco_path)
            written.append(deco_path)
            deco_place = {"plan": _rel(deco_path, out_dir), "x": round(fcu, 1),
                          "y": round(fcv, 1), "scale": round(ds, 3), "rot": 0.0}
            notes.append(msg(
                "꾸밈 그룹 {n:,}장 (로커·베드·산포·에코) — 캔버스 "
                "{canvas:.0f}유닛이 옆면 {span:.0f}유닛에 맞게 스케일 "
                "{scale:.3f}로 앉는다 (도안 스케일에 매이면 캔버스가 면의 1/3밖에 "
                "못 덮는다)",
                n=len(deco_plan.layers), canvas=CANVAS_UNITS, span=p1 - p0,
                scale=ds))
        # 전경 벌 — 도안 **위**에 얹는다 (레퍼런스의 꽃·별은 인물을 덮고 지난다)
        deco_front = design.plan(plan, cat, front=True)
        # 배경 벌이 안 섰으면 전경도 안 쓴다 (`use_deco`가 둘을 같이 켠다) —
        # 쓰지도 않을 그룹 파일을 남기지 않는다
        if deco_front is not None and deco_place is not None:
            fp = out_dir / "deco-front.json"
            deco_front.save(fp)
            written.append(fp)
            front_place = {"plan": _rel(fp, out_dir), "x": round(fcu, 1),
                           "y": round(fcv, 1), "scale": round(ds, 3), "rot": 0.0}
            notes.append(msg("전경 모티프 {n:,}장 — 도안 위에 "
                             "얹는다 (인물이 장면 안에 들어간다)",
                             n=len(deco_front.layers)))
        else:
            deco_front = None
        # ---- 옆면 글자 — 면마다 제 그룹 (미러 금지) 또는 게임 글자 명세 ----
        if design.text is not None and deco_place is not None:
            sets = {deco_src: design.text}
            other = next((n for n in ROLE_MAIN if n != deco_src and by_surface.get(n)), None)
            if other is not None:
                sets[other] = mirrored_set(design.text, design.pal, cat, design.text_plan)
            for sname, tset in sets.items():
                if tset.game_jobs:
                    text_jobs[sname] = [
                        {"text": j["text"], "font": j["font"],
                         "center": [round(fcu + u * j["x"], 1), round(fcv + u * j["y"], 1)],
                         "height": round(u * j["height"], 1), "rot": round(j["rot"], 1),
                         "color": j["color"],
                         **({"outline": j["outline"]} if j.get("outline") else {}),
                         **({"shadow": j["shadow"],
                             "shadow_shift": [round(0.06 * u * j["height"], 1),
                                              round(-0.06 * u * j["height"], 1)]}
                            if j.get("shadow") else {})}
                        for j in tset.game_jobs]
                custom = [l for l in tset.layers if not l.label.startswith("game")]
                if not custom:
                    continue
                tp = design.plan(plan, cat)          # 캔버스 메타는 도안의 것
                tp.layers = [replace(l, x=round(l.x, 4), y=round(l.y, 4), sx=round(l.sx, 4),
                                     sy=round(l.sy, 4), rot=round(l.rot % 360.0, 4))
                             for l in custom]
                tpath = out_dir / f"text-{sname}.json"
                tp.save(tpath)
                written.append(tpath)
                text_groups[sname] = {"plan": _rel(tpath, out_dir), "x": round(fcu, 1),
                                      "y": round(fcv, 1), "scale": round(ds, 3),
                                      "rot": 0.0, "mirror": False}
            notes.append(msg("옆면 글자 그룹 {n}벌 — {what}", n=len(sets),
                             what=(msg("게임 글꼴 비닐 (층 D)") if design.text.tier_main == "D"
                                   else msg("커스텀 도안 {m:,}장", m=design.text.n))))

    # ---- 예산 사다리 — 넘치면 꾸밈부터 버린다 (도안이 주역이다) ----
    # 기준은 **가장 무거운 옆면**이다 (한 면에 여러 장을 올릴 수 있다). 장수는
    # 면에 실제로 올라갈 값이다 — 어느 면에도 안 그려질 레이어는 이미 빠졌다
    # (`_hand_spread`).
    cap = min([m.cap or 3000 for n in ROLE_MAIN
               if (m := maps.get(n)) is not None] or [3000])
    n_person = max([sum(hand_group[hand_ix[id(m)]][1]
                        for m in by_surface.get(n) or [])
                    for n in ROLE_MAIN] or [0])
    use_deco = deco_place is not None
    n_front = len(deco_front.layers) if deco_front is not None else 0
    n_text = max([len(LayerPlan.load(out_dir / g["plan"]).layers)
                  for g in text_groups.values()] or [0])
    if use_deco and n_person + len(deco_plan.layers) + n_front + n_text > cap:
        use_deco = False                         # 도안만 남긴다 (`_check`가 나머지를 잡는다)
        text_groups.clear()
        notes.append(msg("측면이 상한 {cap:,}을 넘는다 — 꾸밈 그룹을 뺀다", cap=cap))

    # 면에 직접 놓는 꾸밈의 색·어휘 — 캔버스 산포와 **같은 세 벌**이다 (액센트 +
    # 밝은 자매 + 색조가 갈린 셋째). 옆면과 다른 팔레트를 쓰면 차를 한 바퀴 돌
    # 때 색이 갈아입혀진다. 근검정/근백은 안 섞는다 (어두운 베이스에서 검은
    # 구멍으로 읽힌다).
    if design is not None:
        motif_c = design.motif_colors
    else:
        main_c = accent_color(lk, car_rgb)
        motif_c = (main_c, accent_tint(main_c, car_rgb), accent_third(main_c, lk, car_rgb))
    motifs_v = motif_shapes(lk, cat, motif)
    # 다른 면의 모티프 수·윗면 스트라이프·로커는 **옆면이 고른 계열**을 따른다 —
    # 면마다 따로 짜면 차를 한 바퀴 돌 때 밀도가 널뛴다.
    fam = design.family if design is not None else None
    dens = fam.other_density if fam is not None else 1.0
    flow_rear = design.flow_rear if design is not None else True

    def _n(k: int) -> int:
        return max(1, int(round(k * dens)))
    # 관통 밴드·톱니는 **옆면 로커와 한 색·한 어휘**다 — 그래야 이어져 보인다.
    # 그래서 **옆면 로커가 안 서면 범퍼 밴드도 안 세운다**: 이을 것이 없는데
    # 범퍼에만 띠가 서면 도색 견본처럼 떠 있는 띠가 된다 (옆면 지도를 못 믿거나
    # 예산이 모자라 꾸밈 그룹을 버렸거나, 차가 이미 어두워 로커를 뺀 판이다).
    flow_v = edge_shapes(lk, cat, motif)
    rocker_on = use_deco and (fam is None or fam.rocker) and (
        car_rgb is None or rgb_to_hsb(*car_rgb)[2] >= ROCKER_BASE_MIN)
    stripe_on = use_deco and (fam is None or fam.top_stripe)

    def _flow(sm, mode: str = "rocker", **kw) -> list[dict]:
        # 윗면 세로 줄은 로커가 아니라 **레이싱 스트라이프**라 무채 대비색이고
        # (Chihaya의 흰·청록 두 줄) 로커 유무와 무관하다.
        if mode == "stripe":
            return flow_shapes(ocol, sm, shapes=flow_v, mode=mode, cat=cat, **kw) \
                if stripe_on else []
        return flow_shapes(ROOF_DARK, sm, shapes=flow_v, mode=mode, cat=cat, **kw) \
            if rocker_on else []

    # 면마다 **나갈 수 있는 이음새**를 미리 푼다. 쓰는 자리는 하나다: 도안 앵커가
    # 이웃 면의 도안 상자를 이 면으로 **투영**한다 (유리 포함 — 도어 유리의
    # 모티프도 문짝의 도안에서 자란다).
    _fold_memo: dict[str, list[gfold.Fold]] = {}

    def _face_folds(name: str) -> list[gfold.Fold]:
        if name not in _fold_memo:
            try:
                got = [f for f in _all_folds(name, maps, rigs, media=pinned)
                       if f.dst in maps]
            except Exception:                    # 지도가 모자란 차 — 투영을 접는다
                got = []
            _fold_memo[name] = got
        return _fold_memo[name]

    # ---- 면마다 **도안 앵커** — 꾸밈이 자랄 뿌리 (`DecoAnchor`) ----
    # 이 면에 도안이 있으면 그 상자가 뿌리다. 없으면 이웃 면의 도안 상자를
    # 이음새 너머로 투영해 쓴다 — 레퍼런스에서 도안 없는 면의 무리는 예외 없이
    # 이웃에서 흘러 들어온다 (Fate R34의 리어 별무리는 리어 쿼터에서 들어와
    # 리어를 채우고, 수이세이의 별은 쿼터와 범퍼에 반씩 걸친다). 둘 다 없으면
    # None이고 산포가 면 상자 한가운데에서 자란다.
    _anchor_memo: dict[str, DecoAnchor | None] = {}

    def _anchor(sm: gsurf.SurfaceMap) -> DecoAnchor | None:
        name = sm.name
        if name in _anchor_memo:
            return _anchor_memo[name]
        got = None
        own = hand_box.get(name)
        if own is not None:
            # 흐름은 옆면 설계가 정한 쪽이다 (`Design.flow_rear` — 빈 자리·얼굴
            # 방향·포즈 축이 고른다). 윗면도 u가 차 뒤다 (`flow_shapes`의 축 규약).
            r = rigs.get(name)
            rd = r.rear_dir if r is not None else 1.0
            got = deco_anchor(own, (rd if flow_rear else -rd, 0.0),
                              why=msg("이 면의 도안"), avoid=own)
        else:
            # **넘쳤나를 묻지 않는다.** 접기 변환은 평면 전체의 아핀이라 도안이
            # 이음선을 안 건드려도 "이 패널에서 보면 도안이 어느 쪽에 있나"를
            # 그대로 준다 — 그것이 여기서 필요한 전부다 (레퍼런스의 리어 별무리는
            # 인물이 문짝에 있어도 리어 쿼터 쪽에서 들어온다). 투영 상자는 대개
            # 이음선 **너머**라 패널 밖이므로 안으로 물려 뿌리로 쓴다.
            best = None
            for src, sbox in hand_box.items():
                for f in _face_folds(src):
                    if f.dst != name:
                        continue
                    pb = f.box(sbox)
                    # 가까운 투영이 임자다 — 겹치면 겹치는 넓이가 큰 쪽
                    key = (_gap(pb, sm.paint), -_overlap(pb, sm.paint))
                    if best is None or key < best[0]:
                        best = (key, pb, src)
            if best is not None:
                _k, pb, src = best
                # 투영은 **어느 쪽에서 흘러 드나**를 말할 뿐이다 — 이 면에
                # 도안이 올라온 것은 아니므로 비울 자리는 없다.
                ink = None
                pb = _clamp_box(pb, sm.paint)
                # 흐름은 **면 안쪽**이다 — 물린 상자는 이음새 가장자리에 붙어
                # 있으니 무리가 패널 안으로 퍼져야 이어져 보인다.
                cu, cv = (pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2
                fu = (sm.paint[0] + sm.paint[2]) / 2
                fv = (sm.paint[1] + sm.paint[3]) / 2
                d = math.hypot(fu - cu, fv - cv)
                flow = ((fu - cu) / d, (fv - cv) / d) if d > 1e-6 else (1.0, 0.0)
                got = deco_anchor(pb, flow, avoid=ink,
                                  why=msg("{src}의 도안을 이 면으로 투영", src=src))
        _anchor_memo[name] = got
        return got

    def _motifs(colors, sm, cat_, over: bool = False, **kw) -> list[dict]:
        """면에 직접 흩는 모티프 — 꾸밈을 끈 판에서는 빈손이다."""
        if not deco:
            return []
        an = _anchor(sm)
        if over:
            if an is None:
                return []                        # 얹을 도안이 없다
            # 전경은 **그림 위**라 배경과 같은 크기면 얼굴을 덮는다
            an = replace(an, ref=an.ref * DECO_FRONT_SIZE)
        return surface_deco_shapes(colors, sm, cat_, anchor=an,
                                   halo=ocol, over=over,
                                   phase=_face_phase(sm.name), **kw)

    items: list[dict] = []
    for name in ROLE_MAIN:
        mps = by_surface.get(name) or []
        if not mps and not use_deco:
            continue                             # 이 면에는 올릴 것이 없다
        item = {"surface": name, "fit": False}
        # 꾸밈 그룹이 **맨 아래**다 — 도안이 그 위에 얹힌다. 미러는 프레임을 뜬
        # 면이 기준이다 (반대편에서 x를 뒤집으면 같은 그림이 선다).
        if use_deco:
            item["pre_groups"] = [dict(deco_place, mirror=name != deco_src)]
            # 글자 그룹은 꾸밈 위·도안 아래, 면마다 제 것 (미러 안 한다)
            if name in text_groups:
                item["pre_groups"].append(text_groups[name])
            if name in text_jobs:
                item["text"] = text_jobs[name]
        if mps:
            item["groups"] = [
                _hand_group_job(m, hand_ix, hand_group, out_dir) for m in mps]
        # 전경 모티프는 **맨 위**다 — 도안 위에 얹혀 인물을 덮고 지난다
        if use_deco and front_place is not None and mps:
            item["groups"] = list(item.get("groups") or []) \
                + [dict(front_place, mirror=name != deco_src)]
        items.append(item)

    # ---- 지붕·데크 블랙아웃 — 윗면의 맨 아래 도형이다 (투톤 문법) ----
    roof_sh = (roof_blackout(ts, shapes=flow_v, hood_u=hood_u, cat=cat)
               if (deco and ts is not None) else [])
    if roof_sh:
        notes.append(msg("지붕·데크 블랙아웃 {n}장 (투톤 문법)", n=len(roof_sh)))

    # ---- 옆면 말고 사람이 배치를 올린 나머지 면 ----
    # 꾸밈은 여기서도 **자동으로** 깔린다 (관통 띠 + 산포 + 지붕 블랙아웃) —
    # 사람이 정한 것은 도안 자리이고, 문법을 짓는 것은 여전히 구성기다.
    for name in list(by_surface):
        mps = by_surface.get(name) or []
        if name in ROLE_MAIN or any(it["surface"] == name for it in items):
            continue
        item = {"surface": name, "fit": False,
                "groups": [_hand_group_job(m, hand_ix, hand_group, out_dir)
                           for m in mps]}
        sm = drawable(name, maps, rigs)
        if sm is not None and name in GLASS and not sm.uncertain:
            # 유리에는 **띠 없이 모티프만** (ARIS 문법 — 레퍼런스의 유리에는
            # 작은 모티프와 낙서뿐이고 차체 띠는 안 올라온다).
            got = _motifs(motif_c, sm, cat, n=_n(3), shapes=motifs_v)
            n_group = sum(hand_group[hand_ix[id(m)]][1] for m in mps)
            if got and n_group + len(got) <= (sm.cap or 1000):
                item["shapes"] = got
                notes.append(msg("{name}: 도안 옆에 모티프 {n}장을 "
                                 "흩는다 (ARIS 문법)", name=name, n=len(got)))
        elif sm is not None and name not in GLASS and (not sm.uncertain
                                                       or _deco_usable(sm)):
            # 윗면 산포는 **후드 구간**에만 흩는다 — 블랙아웃이 덮는 지붕 위에
            # 흩으면 검은 판 위에 모티프가 떠서 투톤이 무너진다 (빈 면 채우기
            # 가지와 같은 규칙이다).
            tsegs = top_segments(sm) if name == ROLE_EXTRA else []
            if tsegs:                          # 후드 구간에만 흩는다 (씨앗은 로케이터)
                tsegs = tsegs[hood_index(tsegs, hood_u):]
            sh = (roof_sh if name == ROLE_EXTRA else []) \
                + _flow(sm, mode="stripe" if name == ROLE_EXTRA else "rocker",
                        **({} if name == ROLE_EXTRA
                           else {"center_v": _bumper_seed(media, name)})) \
                + _motifs(motif_c, sm, cat, n=_n(7), shapes=motifs_v,
                          box=tsegs[0] if tsegs else None)
            # **전경 몫** — 이 면에 도안이 있으면 몇 장은 그 위로 얹는다 (옆면
            # `deco-front`와 같은 문법: 레퍼런스의 꽃·별은 팔·다리를 스치고
            # 지나가고, 전부 뒤에만 깔면 "배경에 스티커를 얹은" 꼴이 된다).
            # `post_shapes`라야 그룹 **위**다 — `shapes`는 맨 아래라 도안이
            # 통째로 덮는다 (제로투 실측: 후드 모티프 일곱이 800장 밑에 깔려
            # 하나도 안 보였다).
            fg = (_motifs(motif_c, sm, cat, n=_n(fam.front_n if fam else DECO_FRONT_N),
                          shapes=motifs_v,
                          over=True, box=tsegs[0] if tsegs else None)
                  if name in hand_box else [])
            n_group = sum(hand_group[hand_ix[id(m)]][1] for m in mps)
            scap = sm.cap or 1000
            if n_group + len(sh) + len(fg) <= scap:
                item["shapes"] = sh
                if fg:
                    item["post_shapes"] = fg
            else:
                notes.append(msg("{name}: 도안 {n:,}장이 상한 {cap:,}에 "
                                 "가깝다 — 꾸밈 도형을 뺀다",
                                 name=name, n=n_group, cap=scap))
        items.append(item)

    # ---- 빈 면 채우기 — 옆면의 띠·산포를 차 전체로 잇는다 ----
    # **꾸밈을 끈 판에서는 채울 것이 없다** — 도안이 안 올라간 면은 구성에서 빠진다.
    used = {it["surface"] for it in items}
    # 윗면 = 지붕 블랙아웃 + 관통 띠 + 후드 모티프 (무사시 후드의 별 문법).
    # 산포 상자는 **후드 구간**이다 — 블랙아웃이 덮는 지붕 위에는 안 흩는다.
    if deco and ROLE_EXTRA not in used and ts is not None \
            and (not ts.uncertain or _deco_usable(ts)):
        segs = top_segments(ts)
        segs = segs[hood_index(segs, hood_u):] if segs else segs
        items.append({"surface": ROLE_EXTRA,
                      "shapes": roof_sh
                      + _flow(ts, mode="stripe")
                      + _motifs(motif_c, ts, cat, n=_n(8),
                                box=segs[0] if segs else None,
                                shapes=motifs_v)})
        used.add(ROLE_EXTRA)
        notes.append(msg("윗면에 관통 띠 + 모티프를 잇는다"))
    # 리어 — 옆면 사이드실 띠가 뒤로 돈다. 띠만 두면 리어가 도색 견본처럼
    # 보인다 (미리보기 판정) — 레퍼런스의 리어도 모티프가 흩어져 있다.
    rs = maps.get(ROLE_REAR)
    if deco and ROLE_REAR not in used and rs is not None \
            and (not rs.uncertain or _deco_usable(rs)):
        items.append({"surface": ROLE_REAR,
                      "shapes": _flow(rs, center_v=_bumper_seed(media, ROLE_REAR))
                      + _motifs(motif_c, rs, cat, n=_n(7), shapes=motifs_v)})
        used.add(ROLE_REAR)
        notes.append(msg("리어에 관통 띠 + 모티프를 잇는다"))
    # 프론트 — **높이**는 내접 상자가 잡는다: 도색 상자 비율로 놓으면 띠가
    # 그릴(비도색)에 떨어져 안 보인다 (2026-08-18 캡처 실측). 길이는 면 폭이고
    # (`flow_shapes`), 상한을 넘는 몫은 나눠 깐다 — front는 스케일이 ±2.3쯤에서
    # 원형으로 감기므로 한 장으로는 면을 못 건넌다.
    fs = maps.get("front")
    if deco and "front" not in used and fs is not None \
            and (not fs.uncertain or _deco_usable(fs)):
        items.append({"surface": "front",
                      "shapes": _flow(fs, box=fs.fit(2.5, coverage=0.85,
                                                     anchor="center"),
                                      max_sx=2.2)
                      + _motifs(motif_c, fs, cat, n=_n(6), shapes=motifs_v)})
        used.add("front")
        notes.append(msg("프론트에 관통 띠 + 모티프를 잇는다"))
    # 도어 유리 = 작은 모티프 (ARIS 문법). 사람이 도안을 올린 면은 이미 `used`다.
    n_motif = 0
    for wname in (("window_left", "window_right") if deco else ()):
        wm = maps.get(wname)
        if wname in used or wm is None or wm.uncertain:
            continue
        motifs = _motifs(motif_c, wm, cat, n=_n(3), shapes=motifs_v)
        if motifs:
            items.append({"surface": wname, "shapes": motifs})
            used.add(wname)
            n_motif += 1
    if n_motif:
        notes.append(msg("도어 유리에 모티프를 흩는다 (ARIS 문법)"))

    # ---- 다른 면의 글자 — 자리를 못 박았을 때 (rear · hood · roof · window) ----
    if (deco and text_spec.active and text_spec.placement not in ("auto", "side")
            and design is not None):
        face_summary = face_text(text_spec, design, items, maps, rigs, cat, out_dir, plan,
                                 group_unit=group_unit, hood_u=hood_u, notes=notes,
                                 written=written)

    # 모티프가 선 면마다 **어느 도안에서 자랐나**를 적는다 — 꾸밈이 엉뚱한 자리에
    # 섰을 때 사람이 먼저 볼 것이 이 뿌리다 (면을 잘못 짚었나, 투영이 딴 면에서
    # 왔나).
    roots = [f"{it['surface']}={an.why}"
             for it in items
             if (it.get("shapes") or it.get("post_shapes"))
             and (an := _anchor_memo.get(it["surface"])) is not None]
    if roots:
        notes.append(msg("꾸밈이 자란 뿌리: {roots}", roots=" · ".join(roots)))

    # ---- 장수 신원 정리 — 그룹마다 장수가 달라야 한다 ----
    # 게임 그리드는 이름을 그림으로만 보여 줘서 **장수로** 그룹을 고른다
    # (`auto.itasha`). 겹치면 구성이 아예 안 서므로(=이 도안은 이타샤가 안 된다)
    # 여기서 투명 패딩으로 비켜 놓는다.
    _unique_group_counts(items, out_dir, notes)

    cfg = {"apply": apply, "car": car, "placements": items}
    if design is not None:
        # 설계 기록 — 어느 계열·팔레트·흐름이 이겼고 점수가 어땠나. 사람이
        # 결과를 보고 "왜 이렇게 짰나"를 되짚는 자리이고, 검증 도구가 읽는다.
        cfg["design"] = {
            "family": design.family.name, "variant": design.pal.variant,
            "flow": "rear" if design.flow_rear else "front",
            "bed_level": round(design.level, 2),
            "score": round(design.score.total, 4),
            "parts": {k: round(v, 3) for k, v in design.score.parts.items()},
            "ranking": design.ranking,
            "palette": {k: list(getattr(design.pal, k)) for k in
                        ("base", "bed", "bed_alt", "primary", "secondary",
                         "shadow", "highlight", "dark")}}
        if text_spec.active:
            tset = design.text
            cfg["design"]["text"] = {
                "enabled": True, "main": text_spec.main, "sub": text_spec.sub,
                "style": (tset.style if tset else design.text_style),
                "tier": (tset.tier_main if tset else "E"),
                "tier_sub": (tset.tier_sub if tset else "E"),
                "role": (tset.poses[0].role if tset else None),
                "layers": (tset.n if tset else 0),
                "placement": text_spec.placement, "priority": text_spec.priority}
            if face_summary:
                cfg["design"]["text"].update(face_summary)
    # 설치 차량을 못 박고 지었으면 **구성이 그걸 기억한다** — 안 적어 두면 다시
    # 돌릴 때 이름 매칭이 다른 차를 물어 미리보기와 검증이 딴 면 지도로 돈다.
    if media:
        cfg["media"] = media
    if paint:
        cfg["paint"] = {"rgb": list(base_rgb), "hsb": list(base_hsb)}
    cfg_path = run_file(out_dir, "itasha.json")
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
    written.append(cfg_path)
    for n in notes:
        log(f"  · {n}")
    return Recipe(config=cfg, written=written, notes=notes)


# 후드 인물의 대각 기울기 (도) — HINATA 레퍼런스의 후드 인물은 차축 대비
# 25~40° 사선이다. 완전 정렬(0°/90°)은 스티커 티가 난다.
HOOD_TILT = 25.0


# 후드 덩어리가 도색 상자의 이 몫보다 작으면 인물을 안 얹는다 (좁은 후드에
# 인물을 구겨 넣으면 유리·펜더로 넘쳐 파편으로 읽힌다) — 기존 보조 아트로.
HOOD_MIN_FRAC = 0.22


# 회전 규약 (2026-08-19 챌린저 캡처로 검증): 게임 회전값 r에서 캔버스 +y(머리)는
# 면 유닛 (-sin r, cos r) 방향이다 — rot 295°가 머리를 +u로 보냈고 캡처가 그대로
# 나왔다. gametext._place_xy의 CCW 오프셋 회전과 같은 규약이다.
HOOD_ROT_SIGN = 1.0


def _hood_place(smap: gsurf.SurfaceMap, lk: Look, group_unit: float,
                hood_u: float | None = None
                ) -> tuple[float, float, float, float, str] | None:
    """윗면 최대 덩어리(후드)에 인물 합성을 **기울여** 앉힌다 (HINATA 문법).

    반환: (x, y, scale, rot, why) 또는 None (후드가 좁다/마스크가 없다).
    머리(캔버스 +y)는 **루프 쪽**을 향한다 — 앞에서 차를 볼 때 인물이 바로
    선다 (레퍼런스 전부 그렇다). 루프 방향은 후드 덩어리 중심에서 전체 마스크
    무게중심 쪽이다 (루프·데크가 그쪽에 있다).

    후드 상자는 **u-프로파일 첫 구간**이 우선이다 — 설치 마스크는 A필러가
    후드와 지붕을 이어 한 덩어리라 blob으로는 윗면 전체를 문다 (인테그라
    실측). 구간이 안 나뉘는 차만 blob으로 물러난다.
    """
    segs = top_segments(smap)
    blob = segs[hood_index(segs, hood_u)] if len(segs) >= 2 else smap.blob_box()
    m = smap.mask
    if blob is None or m.size <= 1:
        return None
    bw, bh = blob[2] - blob[0], blob[3] - blob[1]
    if bw * bh < HOOD_MIN_FRAC * smap.width * smap.height:
        return None
    bcx, bcy = (blob[0] + blob[2]) / 2, (blob[1] + blob[3]) / 2
    # 머리는 **차 뒤쪽** — 앞에 선 사람이 후드/지붕을 볼 때 그림이 바로 서려면
    # 그림의 위(머리)가 관찰자 반대편(뒤)이어야 한다. HINATA 레퍼런스의 후드
    # 인물도 머리가 윈드실드(뒤) 쪽이다. 반대(머리=앞, rot 115°)는 2026-08-19
    # 챌린저 캡처에서 그림이 거꾸로 읽히고 앞유리로 흘러넘쳤다 — rot 295°
    # (머리=뒤)가 지붕에 온전히 앉았다. 뒤 방향은 카메라 규약에서 나온다:
    # top 카메라는 차 앞을 화면 왼쪽에 둔다 (챌린저·MX-5 캡처 일치) → 뒤 = +px.
    # 유닛으로는 야코비안 ∂px/∂u의 부호가 가른다.
    assert smap.warp is not None
    rdir = 1.0 if smap.warp.jac(bcx, bcy)[0, 0] > 0 else -1.0
    rot = HOOD_ROT_SIGN * (-rdir * 90.0 + HOOD_TILT)
    # 회전된 **잉크** 상자 (껍질 실측). 사각형 공식은 65~115°에서 후드 인물을
    # 1.5배 넘게 과대평가해 그만큼 축소시켰다 — 옆면과 같은 결함이다.
    ib = rot_ink_box(lk, rot)
    rw, rh = ib[2] - ib[0], ib[3] - ib[1]
    scale = min(bw / max(1e-6, rw), bh / max(1e-6, rh)) * 0.92 / max(1e-6, group_unit)
    rcx, rcy = (ib[0] + ib[2]) / 2, (ib[1] + ib[3]) / 2
    x = bcx - scale * group_unit * rcx
    y = bcy - scale * group_unit * rcy
    return (round(x, 1), round(y, 1), round(scale, 3), round(rot % 360.0, 1),
            msg("후드 덩어리 {w:.0f}×{h:.0f}유닛 · 기울기 {rot:.0f}°",
                w=bw, h=bh, rot=rot))


def _deco_usable(smap: gsurf.SurfaceMap) -> bool:
    """도형·글자만 올리는 면은 **아핀 어림으로도 충분한가** — `uncertain`이어도
    배율·도색 상자가 성하면 쓴다. 그룹 오토핏은 정밀 매핑이 없으면 위험하지만
    (도안이 면 밖으로 나간다), 모티프·타이포는 몇 유닛 비껴 앉는 것이 최악이고
    면 캡처 검증이 남는다. 격자 표본이 모자라는 윗면(선루프 반사가 프로브를
    문다)이 이 완화의 대상이다.
    """
    kx, ky = smap.px_per_unit
    return (kx > 0.05 and ky > 0.05 and smap.fill > 0.12
            and smap.width > 1.0 and smap.height > 1.0
            and smap.width / smap.height < 15.0)
