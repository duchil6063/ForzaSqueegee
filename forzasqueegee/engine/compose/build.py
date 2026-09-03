"""구성 한 대 짜기 — 이 패키지의 진입점."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from ...game import fold as gfold, seam as gseam, surface as gsurf
from ...i18n import msg
from ...paths import run_file
from ..catalog import Catalog, default_catalog_path
from ..model import UNITS_PER_SCALE, LayerPlan, rgb_to_hsb, rnd
from .boxes import (
    CANVAS_UNITS, _clamp_box, _face_phase, _gap, _group_unit, _overlap, _rel,
    _union)
from .look import Look, layer_points, look, person_ink, rot_ink_box
from .palette import (BASE_BLACK, accent_color, accent_third, accent_tint, base_paint,
                      contrast_ink, material_roles, pastel_base)
from .presets import resolve as resolve_style
from .vocabulary import MOTIF_FAMILIES, MOTIF_SETS, edge_shapes, motif_shapes
from .scatter import DECO_FRONT_N, DECO_FRONT_SIZE
from .bands import ROCKER_BASE_MIN
from .roof import ROOF_DARK, hood_index, roof_blackout, top_segments
from .place import surface_exposure
from .place import (
    BODY_BIAS, BODY_FILL, ROLE_EXTRA, ROLE_MAIN, ROLE_REAR, ManualPlace, dodge_parts,
    drawable, ink_outside, layers_on, manual_box, person_pose, person_scale, person_tilt,
    place_in_rect, place_xf, take_layers, usable)
from .atlas import build_atlas
from . import seams as gseams
from .folds import _all_folds, seam_fold
from .facespec import FACE_OF, FaceSpec
from .autoplace import auto_place
from .surfshapes import GLASS, DecoAnchor, deco_anchor, flow_shapes, surface_deco_shapes
from .intent import read_intent, with_head
from . import whole as wholecar
from .design import Design, _macro_colors, compose_design
from .families import FAMILIES
from .textspec import TextSpec
from .textbuild import mirrored_set
from .facetext import assigned_text, face_text, pinned_faces
from .logokit import LogoItem, LogoSpec, resolve as resolve_logos, watermark_plan
from . import sponsor
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
    # 이긴 옆면 설계 — 계측·디버그 도구가 필드·레이어·점수를 그대로 본다
    # (`work/lab/deco/heat.py`). 구성 파일에는 요약만 실린다.
    design: "Design | None" = None


def build(main_plan: Path, out_dir: Path, *, car: str | None = None,
          media: str | None = None, extra_plans: list[Path] | None = None,
          mirror: bool = True, apply: bool = True, paint: bool = True,
          base_rgb: tuple[int, int, int] | None = None, flip: bool = False,
          preset: dict[str, dict[str, float]] | None = None,
          cat: Catalog | None = None,
          manual: list[ManualPlace] | None = None, deco: bool = True,
          whole: bool | None = None,
          motif: str | None = None, family: str | None = None,
          text: "TextSpec | dict | None" = None,
          logos: "LogoSpec | dict | None" = None,
          faces: "FaceSpec | dict | None" = None,
          style: str | None = None,
          mass_hint: dict | None = None, log=print) -> Recipe:
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
    배치마다 **역할**이 붙어 온다 (`ManualPlace.role` — `compose.cast`): 옆면
    설계의 뿌리는 그림(주역·보조)의 상자이고, 로고·글자·그대로는 두르지 않는다.
    차 전체 구성도 같이 돈다 — 사람이 올린 덩어리가 앉은 면은 변주를 안 받고,
    그 덩어리의 무게는 고정 질량으로 배분에 든다.

    **면을 넘긴 몫은 그 자리에서 잘린다** — 이웃 면으로 안 잇는다 (사용자 지시
    2026-08-27). 감아 돌리고 싶으면 편집기에서 도안을 이음선으로 가르고
    (KFPS·FLS의 [선으로 가르기]) 한쪽을 그 면에 따로 올린다.

    ## 꾸밈 (`deco`)

    끄면 **도안만** 올린다 — 꾸밈 그룹(로커·산포)·관통 밴드·
    지붕 블랙아웃·모티프가 전부 빠지고, 그것만 있던 면은 구성에서 사라진다.
    도안도 **통째로** 올린다 — 면 밖 레이어 빼기(`_hand_spread`)는 켠 판에서만
    한다 (편집기에서 자리를 옮길 도안을 처음 자리로 잘라 두면 구멍이 난다).
    베이스 도색은 별개 레버다 (`paint`): 도안만 올리더라도 차 색은 정해야 한다.

    ## 차 전체 구성 (`whole`)

    끄면 **옛 길**이다 — 옆면·윗면만 도안을 받고 나머지 면은 모티프 몇 장이
    전부다. 켜면 도안 하나에서 결정적으로 변주(얼굴 크롭·상반신·색 줄인 전신·
    2색 엠블럼)를 뽑아 남은 면에 역할을 주고 한계효용으로 장수를 나눈다
    (`compose.whole`). 새 그림을 지어내지 않는다 — 있는 레이어를 다시 자르고
    색을 줄이는 것뿐이다.

    ## 모티프 계열 (`motif`)

    안 주면 도안의 테마색이 고른다 (`motif_family`). 주면 그 계열로 못 박는다 —
    **계열은 원래 캐릭터 의미에서 오는 것**이라 팔레트로는 거기까지 못 간다
    (수이세이가 별인 것은 이름이 '별마을 혜성'이라서다). 베이스 도색의
    `base_rgb`와 같은 자리의 레버다: 자동으로 정해 주고 사람이 바꾼다.

    ## 스타일 프리셋 (`style`) · 구성 계열 (`family`)

    옆면 꾸밈은 **후보를 여럿 지어 점수로 고른다** (`design.compose_design` —
    계열 × 흐름 × 팔레트 변종 × 베드 크기). `style`(`presets.STYLE_PRESETS` —
    레이싱 스폰서 · 무늬·꽃 · 스플래시·찢김 · 미니멀 · 다크 그래피티)을 주면 그
    프리셋의 계열 안에서만 고르고 바탕 도색 기본·글자 스타일과 크기·레이싱 번호·
    로고 줄·리어 배정·예산 사다리가 한 벌로 온다. 사람이 따로 정한 것(바탕 색·
    글자 스타일·면 배정)이 프리셋보다 앞이다. `family`는 엔진 레버 — 계열만 못
    박는다 (`families.FAMILIES`). 사람이 앉힌 도안은 어느 후보에서도 안 움직인다.

    ## 글자 (`text`)

    기본은 **안 넣는다** (이타샤 어휘에 글자가 없다는 규칙은 그대로다). 스펙
    (`textspec.TextSpec` 또는 그 dict)을 켜서 주면 캐릭터 이름(+작품명)이 꾸밈의
    한 요소로 후보에 들어간다 — **게임 글꼴 글리프**(한 글자 한 장, `engine.textvinyl`)
    가 기본 엔진이고, 사람이 `engine: shapes`를 고르면 동봉 OFL 글꼴을 도형으로
    되짓는 커스텀 도안(`engine.textglyph`)을 고운 층이 예산에 들 때만 쓴다
    (`textbudget`). 옆면 글자는 면마다 제 그룹(`text-<면>.json`)이다 — 꾸밈 그룹은
    좌우를 미러로 나눠 쓰지만 글자는 뒤집히면 안 된다.

    ## 로고 (`logos`)

    내장 워터마크(기본 켬)와 사용자 로고 이미지(0~N)를 **스폰서 문법**으로
    앉힌다 (`logokit`·`sponsor` — 사용자 결정 ② 2026-09-02): 옆면 로커 위 한 줄 ·
    리어 범퍼 가운데 + 좌우 · 프론트 범퍼 · 윈드실드 귀퉁이. 워터마크는 그중 한
    자리(`auto`면 리어, 없으면 윈드실드, 그것도 없으면 옆면 줄 끝)다. 로고 그룹은
    면마다 제 것(`logos-<면>.json`)이고 **미러하지 않는다** — 반대편 옆면은
    자리만 거울이다. 꾸밈을 끈 판에는 안 선다 (로고도 꾸밈이다).

    ## 면 배정 (`faces`)

    도어 유리·뒷유리·리어·프론트·윈드실드가 **맡는 일** (`facespec.FaceSpec`,
    계획 5단계). 사람 판은 그 면들에 주역 크롭을 거의 안 돌린다 — 리어는 워드마크
    + 로고 줄 + 색면 이음, 윈드실드는 글자 띠, 도어 유리는 작은 로고 열·문구 또는
    옆면 머리카락을 벨트라인 너머로 이어 그린 것. `auto`는 로고·글자가 있으면
    그것을 앉히고 없으면 크롭으로 물러난다 (리어·뒷유리는 비운다). `continue`
    (이어 그리기)는 옆면 주역의 벨트라인 위 몫을 유리에 사본으로 세운다 — 편집기의
    [선으로 가르기]를 프로그램이 대신 부르는 꼴이라 기본이 아니다.
    """
    if whole is None:                             # 스윕용 스위치 (기본 켬)
        whole = os.environ.get("FS_WHOLE", "1").strip() != "0"
    text_spec = (text if isinstance(text, TextSpec) else TextSpec.from_dict(text)) \
        if text is not None else TextSpec()
    logo_spec = (logos if isinstance(logos, LogoSpec) else LogoSpec.from_dict(logos)) \
        if logos is not None else LogoSpec()
    face_spec = (faces if isinstance(faces, FaceSpec) else FaceSpec.from_dict(faces)) \
        if faces is not None else FaceSpec()
    mirror_side = "side_left" if flip else "side_right"
    from ...auto.itasha import PRESET             # 순환 참조를 피해 늦게 들여온다

    if motif is not None and motif not in MOTIF_SETS:
        raise ValueError(msg("모르는 모티프 계열: {motif!r} (있는 것: {families})",
                             motif=motif, families=", ".join(MOTIF_FAMILIES)))
    if family is not None and family not in FAMILIES:
        raise ValueError(msg("모르는 구성 계열: {family!r} (있는 것: {families})",
                             family=family, families=", ".join(FAMILIES)))
    pre = resolve_style(style)                    # 모르는 이름이면 ValueError
    # 프리셋의 글자 기본 — 사람이 auto로 둔 스타일·보통으로 둔 우선순위만 채운다.
    # 레이싱 번호는 레이싱 프리셋에서만 선다 (다른 프리셋·자동에서는 없는 값이다).
    if pre is not None:
        if text_spec.style == "auto" and pre.text_style != "auto":
            text_spec = replace(text_spec, style=pre.text_style)
        if pre.text_priority and text_spec.priority == "normal":
            text_spec = replace(text_spec, priority=pre.text_priority)
    if text_spec.number and not (pre is not None and pre.number):
        text_spec = replace(text_spec, number=None)
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
    lk = with_head(look(plan, cat), plan, cat)      # 옆면 자리가 얼굴을 벨트 아래에 잡는다
    # 베이스는 **자동차 도색**이다 — 비닐이 아니라 도색 메뉴에서 칠한다
    # (레퍼런스 이타샤의 베이스가 전부 그렇다. 장수 0장 · 도료 질감 공짜).
    # 사람이 정한 색이 있으면 그것이 이기고, 없으면 도안에서 고른다.
    if base_rgb is not None:
        base_rgb = tuple(int(v) for v in base_rgb)
        base_hsb = tuple(round(v, 2) for v in rgb_to_hsb(*base_rgb))
        notes.append(msg("베이스 도색은 사람이 정했다 — RGB {rgb}", rgb=base_rgb))
    elif pre is not None and pre.base == "black":
        base_rgb, base_hsb = BASE_BLACK, tuple(round(v, 2) for v in rgb_to_hsb(*BASE_BLACK))
        notes.append(msg("베이스 도색은 프리셋이 정했다 — 검정 ({name})", name=pre.name))
    elif pre is not None and pre.base == "pastel":
        base_rgb, base_hsb = pastel_base(lk)
        base_hsb = tuple(round(v, 2) for v in base_hsb)
        notes.append(msg("베이스 도색은 프리셋이 정했다 — 파스텔 RGB {rgb} ({name})",
                         rgb=base_rgb, name=pre.name))
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

        상자는 `person_budget`이 주는 사람 판 실측 예산이고(벨트 위로 조금
        나간다), 얼굴은 벨트 아래다 (`person_scale`) — 넘긴 몫은 이웃 면으로
        안 가고 그 자리에서 잘린다.
        """
        for r in rigs.values():
            r.tilt = tilt if r.name == "side_left" else -tilt
            r.mirror = bool(mirror and r.name == mirror_side)
            wb, hb = gseam.person_budget(r.body, r.geom)
            iw, ih = person_ink(lk, r.tilt, r.mirror)
            s, head_why = person_scale(lk, r.tilt, r.mirror, r)   # 예산에 내접, 얼굴은 벨트 아래
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
                + (f" · {head_why}" if head_why else "")
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
            mp = auto_place(ROLE_EXTRA, ep, ek, maps, rigs, group_unit=group_unit,
                            hood=True, media=pinned, notes=notes)
            if mp is not None:
                hand.append(mp)
            else:
                notes.append(msg("{surface}: 보조 도안을 못 앉힌다 (면 지도가 없다)",
                                 surface=ROLE_EXTRA))
        elif (ts is not None and not ts.uncertain
                and len(plan.layers) <= (ts.cap or 3000)):
            hood = _hood_place(ts, lk, group_unit, hood_u,
                               glass=maps[ROLE_EXTRA].drawn is not None)
            if hood is not None:
                hx, hy, hs, hrot, hwhy = hood
                hand.append(ManualPlace(plan=Path(main_plan), surface=ROLE_EXTRA,
                                        x=hx, y=hy, scale=hs, rot=hrot))
                notes.append(msg("후드에 인물을 기울여 앉힌다 ({why})", why=hwhy))

    # ---- 로고 재료 — 사용자 로고는 도안으로 굽고(캐시), 워터마크는 키트에서 ----
    # 면 배정보다 앞이다: 프론트·도어 유리의 `auto`가 "사용자 로고가 있나"로 갈린다.
    logo_cache: dict = {}
    logo_protos: list[sponsor.Proto] = []
    wm_proto: sponsor.Proto | None = None
    if deco and logo_spec.active:
        for it in resolve_logos(logo_spec, out_dir / "logos", cat=cat, log=log, notes=notes):
            logo_protos.append(sponsor.load_proto(it, cat, logo_cache))
        wmp = watermark_plan(False) if logo_spec.watermark else None
        if wmp is not None:
            wm_proto = sponsor.load_proto(
                LogoItem(plan=wmp, kind="watermark", name="ForzaSqueegee"), cat, logo_cache)
        elif logo_spec.watermark:
            notes.append(msg("내장 로고 키트가 없다 — tools/make_kit.py로 굽는다"))

    # ---- 면 배정 — 유리·리어·프론트·윈드실드가 맡는 일 (`compose.facespec`) ----
    # `auto`는 로고·글자가 있으면 그것(로고 줄·워드마크·글자 띠), 없으면 크롭으로
    # 물러난다. 크롭을 안 받는 면은 차 전체 구성(`plan_car`)에 `taken`으로 든다.
    assign = {n: face_spec.resolve(n, logos=bool(logo_protos),
                                   text=bool(deco and text_spec.active),
                                   poster=bool(pre is not None and pre.rear_poster))
              for n in FACE_OF}
    if deco and not face_spec.all_auto:
        notes.append(msg("면 배정: {what}", what=face_spec.describe()))
    # 이어 그리기(`continue`) — 옆면 주역의 벨트라인 위 몫을 도어 유리에 **사본**으로
    # 세운다. 편집기의 [선으로 가르기]를 프로그램이 대신 부르는 꼴이다: 걸친 그림은
    # 양쪽에 서고 면마다 제 마스크가 제 몫만 그린다 (split-at-a-line 규약).
    if deco:
        hand += _continuations(hand, assign, rigs, maps, cat, out_dir, group_unit, notes)

    # ---- 차 전체 구성 — 남은 면에 역할과 예산을 준다 (`compose.whole`) ----
    # 옛 길에서 여기 오는 면은 옆면 둘(+윗면)뿐이었고 나머지는 모티프 몇 장이
    # 전부였다. 사람 판 28벌을 같은 자로 재면 그 자리에 리어 335 · 도어 유리
    # 740 · 뒷유리 589장이 있다 — **맡은 일이 없어서** 비어 있던 것이다.
    # 여기서 도안 하나에서 변주를 뽑아 역할을 주고, 예산은 한계효용이 나눈다.
    main_intent = None
    wcp = None
    # 손 배치(편집기) 판도 같은 길이다 — 사람이 올린 덩어리가 앉은 면은 `taken`
    # 이라 변주를 안 받고(보조 그림·로고를 크롭으로 덮어씌우지 않는다), 그
    # 덩어리들의 무게는 **고정 질량**으로 배분에 든다 (역할표 1단계).
    if whole and deco:
        main_intent = read_intent(plan, lk, cat)
        _tm_key = str(Path(main_plan).resolve())
        _tm_ink: dict[str, float] = {}

        def _taken_mass(mp: ManualPlace) -> float:
            """이미 앉은 배치 하나의 예상 시각 무게 (`whole.ink_weight`)."""
            k = mp.key()
            w = _tm_ink.get(k)
            if w is None:
                hp = plan if k == _tm_key else LayerPlan.load(mp.plan)
                w = float(wholecar.ink_weight(hp.layers, cat).sum())
                _tm_ink[k] = w
            g = float(mp.scale) * group_unit
            return g * g * w

        # 면마다 **합**이다 — 한 면에 덩어리가 여럿이면(주역 + 로고 둘) 무게도
        # 장수도 더해진다. 자동 경로는 면당 하나라 옛 답 그대로다.
        _sure = {mp.surface for mp in hand
                 if maps.get(mp.surface) is not None
                 and not maps[mp.surface].uncertain}
        taken_mass: dict[str, float] = {}
        taken_n: dict[str, int] = {}
        for mp in hand:
            taken_n[mp.surface] = taken_n.get(mp.surface, 0) + len(
                (plan if mp.key() == _tm_key else LayerPlan.load(mp.plan)).layers)
            if mp.surface in _sure:
                taken_mass[mp.surface] = taken_mass.get(mp.surface, 0.0) + _taken_mass(mp)

        wcp = wholecar.plan_car(
            plan, lk, main_intent, cat, maps,
            # 사람(또는 자동)이 덩어리를 앉힌 면 + 면 배정이 크롭을 안 주는 면
            taken={mp.surface for mp in hand} | {n for n, a in assign.items() if not a.crop},
            caps={n: (m.cap or 1000) for n, m in maps.items()},
            # 주역이 이미 앉은 면의 **예상 시각 무게** — 배치가 이미 정해져
            # 있으므로 어림이 아니라 실제 배율로 잰다: (배율×그룹유닛)² ×
            # Σ(칠한 넓이 × 대비). 평가기의 `ruler.visual_weight`와 같은 식이다
            # (`whole.ink_weight`). 옛 길은 여기 **면 넓이**를 넣었는데, 넓지만
            # 그림이 다 안 채우는 윗면이 옆면과 같은 주역으로 읽혔다 (실측
            # 예상 0.368 ↔ 실측 0.231).
            taken_mass=taken_mass,
            # **위계를 아는 배분** (§4) — 앞 판에서 잰 "이 배분이 안 건드리는
            # 것"의 무게에 사람이(또는 자동이) 앉힌 덩어리를 더한 것. 없으면 옛
            # 배분 그대로다 (바이트 동일).
            base_mass=(None if mass_hint is None else
                       {n: float(v) for n, v in mass_hint.items()
                        if maps.get(n) is not None
                        and not maps[n].uncertain}
                       | {s: float(mass_hint.get(s, 0.0)) + m
                          for s, m in taken_mass.items()}),
            base_counts=(None if mass_hint is None else dict(taken_n)))
        art_paths: dict[tuple[str, int, float], Path] = {}
        for name, job in sorted(wcp.jobs.items()):
            var = wcp.variants[job.kind]
            # 옅게 한 변주는 **다른 파일**이다 — 같은 장수라도 색이 다르다
            # (`whole.SurfaceJob.fade`). 키에 안 넣으면 먼저 쓴 면의 색을
            # 뒤 면이 그대로 물려받는다
            key = (job.kind, job.budget, round(job.fade, 3))
            vpath = art_paths.get(key)
            if vpath is None:
                suf = "" if job.fade <= 0.0 else "-f%02d" % round(job.fade * 100)
                vpath = out_dir / f"art-{job.kind}-{job.budget}{suf}.json"
                var.budgeted(job.budget, job.fade).save(vpath)
                art_paths[key] = vpath
            # 배치는 **자른 상자**를 면에 맞춘다 — 상자에 걸친 큰 색면이
            # 캔버스를 원본만큼 넓히므로 잉크 범위로 맞추면 얼굴이 작아진다.
            vlk = replace(look(LayerPlan.load(vpath), cat),
                          box=var.box, hull=None)
            # 좌우 짝은 옆면과 **같은 쪽**을 뒤집는다 — 그래야 그룹 한 벌로 둘이 선다
            mir = bool(mirror and name.endswith("_right")
                       and mirror_side == "side_right")
            mp = auto_place(name, vpath, vlk, maps, rigs, group_unit=group_unit,
                            mirror=mir, fill=job.fill)
            if mp is None:
                notes.append(msg("{surface}: {kind} 변주를 못 앉힌다 (면 지도가 없다)",
                                 surface=name, kind=job.kind))
                continue
            hand.append(replace(mp, role="variant"))
            notes.append(msg("{surface}: {kind} {n:,}장 — {why}",
                             surface=name, kind=job.kind, n=job.budget,
                             why=job.why))
        if wcp.jobs:
            log(msg("차 전체 구성: {items}",
                    items=" · ".join(f"{n}={j.kind}({j.budget:,})"
                                     for n, j in sorted(wcp.jobs.items()))))

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
    # 면 밖 레이어 빼기는 **꾸밈을 켠 판에서만** — 도안만 올리는 판은 통째로
    # (사용자 지시 2026-09-02: 편집기에서 옮길 자리를 미리 자르면 안 된다).
    _hand_spread(hand, hand_look, hand_path, hand_group, maps, rigs, cat,
                 out_dir, notes, group_unit=group_unit, trim=deco)
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

    # ---- 워터마크 자리 — **하나**다: `auto`면 리어 범퍼, 없으면 윈드실드 귀퉁이, 그것도
    # 없으면 옆면 줄 끝. 사람 판의 로고 무리는 24/30벌에 있다 — 없던 자리다. 면 배정이
    # 그 면을 비웠거나 크롭에 줬으면(`assign`) 건너뛴다.
    wm_face: str | None = None
    logo_targets = sponsor.spec_targets(logo_spec)
    logo_groups: dict[str, dict] = {}         # 면 → 로고 그룹 항목
    logo_summary: dict[str, int] = {}
    side_w = (side0.paint[2] - side0.paint[0]) if side0 is not None else None
    if wm_proto is not None:

        def _face_ok(name: str) -> bool:
            sm = maps.get(name)
            return (sm is not None and (not sm.uncertain or _deco_usable(sm))
                    and assign[name].sponsor)

        order = {"auto": ("rear", "windshield", "side"), "rear": ("rear",),
                 "front": ("front",), "windshield": ("windshield",),
                 "rocker": ("side",)}[logo_spec.placement]
        for cand in order:
            if cand == "side":
                if side0 is not None and not side0.uncertain:
                    wm_face = cand
            elif _face_ok(cand):
                wm_face = cand
            if wm_face:
                break
        if wm_face is None:
            notes.append(msg("워터마크를 앉힐 면이 없다 ({place})", place=logo_spec.placement))
    deco_plan = deco_place = deco_front = front_place = None
    side_logos: list = []
    # 옆면이 이음새로 내보내는 두 선 — 큰 색면의 띠와 하부 투톤의 윗선 (면 유닛).
    # 잇는 자는 `compose.seams`이고, 띠는 **어디서 쟀는지**(`Band.at_u`)를 같이 든다.
    side_band: "gseams.Band | None" = None
    # 스택의 관통 띠들 — (띠, 색, 조각 이름). 벨트 블랙아웃이 여기로 나간다.
    side_stack: list[tuple["gseams.Band", tuple[int, int, int], str]] = []
    side_rocker_top: float | None = None
    design: Design | None = None
    face_summary: dict | None = None
    text_groups: dict[str, dict] = {}         # 면 → 글자 그룹 항목
    number_groups: dict[str, dict] = {}       # 면 → 레이싱 번호 그룹 항목 (프리셋)
    number_poses: list = []
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
        # 설계가 두르는 상자는 **그림**(주역·보조)의 것이다 — 로고·글자·그대로는
        # 재료이지 뿌리가 아니라, 그것들까지 두르면 베드가 로고 자리로 늘어난다.
        # 그림이 하나도 없는 면(로고만 올린 면)은 전부를 두른다.
        _anc = [m for m in by_surface[deco_src] if m.anchors] or by_surface[deco_src]
        pbox = None
        for m in _anc:
            b = manual_box(hand_look[m.key()][1], m, group_unit)
            pbox = b if pbox is None else _union(pbox, b)
        # **안 그려질 자리를 미리 거른다** — 꾸밈은 캔버스 좌표로 앉으므로 면
        # 도색 마스크를 모른다. 휠아치 구멍·벨트라인 위에 떨어진 모티프는 게임이
        # 통째로 안 그려 장수만 먹는다 (미리보기 실측: 26장 중 여섯).
        dm0 = drawable(deco_src, maps, rigs) or side0

        def _drawable_at(cx: float, cy: float, _m=dm0, _u=u,
                         _c=(fcu, fcv)) -> bool:
            return bool(_m.masked_at(_c[0] + _u * cx, _c[1] + _u * cy))

        # **글자가 설 수 있는 자리** = 그려지는 자리 ∧ 눈에 보이는 자리. 껍질이 재는
        # 노출(`surface_exposure`)이 낮은 띠 — 사이드실 아랫단·벨트 바로 아래 — 는
        # 마스크 안이라도 게임이 눌러 그려 사람 눈에는 잘린 글자다 (P-A 실측).
        # 산포·색면은 종전대로 마스크만 본다 (노출은 표시용이라는 규약 그대로).
        dm_ex = dm0
        if side0 is not None:
            _ex = surface_exposure(deco_src, side0, maps, media=media)
            if _ex is not None and _ex.shape == dm0.mask.shape:
                dm_ex = replace(dm0, mask=dm0.mask & (_ex >= TEXT_EXPOSURE_MIN))

        def _exposed_at(cx: float, cy: float, _m=dm_ex, _u=u,
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
        root_mp = max(_anc,
                      key=lambda m: (lambda b: (b[2] - b[0]) * (b[3] - b[1]))(
                          manual_box(hand_look[m.key()][1], m, group_unit)))
        root_plan, root_lk = hand_look[root_mp.key()]
        intent = (main_intent if (main_intent is not None
                                  and root_plan is plan)
                  else read_intent(root_plan, root_lk, cat))
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
            drawable_at=_drawable_at, exposed_at=_exposed_at, motif=motif, halo=ocol,
            family=family, phase=_face_phase(deco_src), text=side_text, cap=side_cap,
            n_person=side_person, style=pre)
        notes += design.notes
        # ---- **차 한 대의 지도** — 옆면의 큰 색면이 이음새를 건너간다 ----
        # 프레임 좌표는 면 유닛의 균등 축소라(회전 없음) 각은 그대로고 자리·
        # 두께만 `u`를 곱하면 된다.
        prim = next((l for l in design.back if l.label == "itasha_bed"), None)
        if prim is not None:
            a = prim.rot % 180.0
            # **잰 자리를 같이 들고 간다** (`at_u`) — 기울인 띠의 높이는 어디서
            # 쟀는지를 모르면 이음선 위의 높이로 못 옮긴다 (`compose.seams`).
            side_band = gseams.Band(
                v=fcv + prim.y * u, angle=a - 180.0 if a > 90.0 else a,
                thickness=2 * abs(prim.sy) * UNITS_PER_SCALE * u,
                at_u=fcu + prim.x * u)
        # 스택의 **관통 띠**(벨트 블랙아웃)도 이음새로 내보낸다 — 프레임 좌표는
        # 면 유닛의 균등 축소라 자리·두께에 `u`만 곱한다. 색은 그 조각의 역할색.
        _scol = _macro_colors(design.pal)
        for pc in design.stack:
            if pc.kind == "belt":
                side_stack.append((gseams.Band(
                    v=fcv + pc.at[1] * u, angle=pc.ang, thickness=pc.width * u,
                    at_u=fcu + pc.at[0] * u), _scol.get(pc.role, design.pal.dark), pc.kind))
        # 하부 투톤의 **윗선** — 이것이 이음새를 건너 앞·뒤 범퍼로 이어진다.
        # 눈이 따라가는 것은 밴드의 두께가 아니라 이 선이다.
        rk = next((l for l in design.back if l.label == "itasha_stripe"), None)
        if rk is not None:
            side_rocker_top = fcv + (rk.y + abs(rk.sy) * UNITS_PER_SCALE) * u
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
        # ---- 옆면 글자 — 면마다 제 그룹 (미러 금지) ----
        # 글리프 실물이 차체 밴드 밖으로 나가면 들인다 (필드 격자의 포즈 상자는
        # 글리프 잉크와 어긋난다 — 미아타 사인 글자의 15%가 벨트 위였다)
        if design.text is not None and deco_place is not None:
            design.text = _side_text_guard(design.text, design.fld, cat, dm0, u,
                                           (fcu, fcv), notes)
        if design.text is not None and deco_place is not None and dm_ex is not dm0:
            # 저노출 띠는 **너그럽게** — 잉크의 15%까지는 둔다. 1%로 걸면 설계가 고른
            # 글자(점수 .91)를 굽기가 통째로 빼서 옆면에 이름이 없는 판이 11/60이었다
            # (P-B 실측). 못 들이면 빼지 않고 그대로 둔다 — 없는 이름보다 낫다.
            design.text = _side_text_guard(design.text, design.fld, cat, dm_ex, u,
                                           (fcu, fcv), notes, out_max=TEXT_LOWEXPO_MAX,
                                           drop=False)
        if design.text is not None and deco_place is not None:
            sets = {deco_src: design.text}
            other = next((n for n in ROLE_MAIN if n != deco_src and by_surface.get(n)), None)
            if other is not None:
                sets[other] = mirrored_set(design.text, design.pal, cat, design.text_plan)
            for sname, tset in sets.items():
                if not tset.layers:
                    continue
                tp = design.plan(plan, cat)          # 캔버스 메타는 도안의 것
                tp.layers = [replace(l, x=rnd(l.x, 4), y=rnd(l.y, 4), sx=rnd(l.sx, 4),
                                     sy=rnd(l.sy, 4), rot=rnd(l.rot % 360.0, 4))
                             for l in tset.layers]
                tpath = out_dir / f"text-{sname}.json"
                tp.save(tpath)
                written.append(tpath)
                text_groups[sname] = {"plan": _rel(tpath, out_dir), "x": round(fcu, 1),
                                      "y": round(fcv, 1), "scale": round(ds, 3),
                                      "rot": 0.0, "mirror": False}
            notes.append(msg(
                "옆면 글자 그룹 {n}벌 — {what} {m:,}장", n=len(sets),
                what=(msg("게임 글꼴 {font}", font=design.text_plan.font)
                      if design.text.tier_main == "D"
                      else msg("도형 맞춤 (층 {tier})", tier=design.text.tier_main)),
                m=design.text.n))
        # ---- 레이싱 번호 — 리어 쿼터, 면마다 제 그룹 (프리셋 · 미러 금지) ----
        # 이름 글자 묶음과 겨루지 않는다 — 이긴 설계 뒤에 로고 줄처럼 따로 앉힌다.
        if (pre is not None and pre.number and text_spec.number and deco_place is not None):
            other = next((n for n in ROLE_MAIN if n != deco_src and by_surface.get(n)), None)
            number_groups, number_poses = _side_number(
                design, text_spec, cat, out_dir, plan, [deco_src] + ([other] if other else []),
                (fcu, fcv), ds, notes, written)
        # ---- 옆면 로고 줄 — 로커 위, 면마다 제 그룹 (미러 금지) ----
        row = list(logo_protos) + ([wm_proto] if (wm_proto is not None and wm_face == "side")
                                   else [])
        if row and logo_targets["side"]:
            side_logos = sponsor.side_row(
                row, design.fld,
                (design.text.poses if design.text is not None else []) + number_poses,
                notes, size_k=(pre.logo_row if pre is not None else 1.0))
            side_logos = [
                (sponsor.pick_watermark(
                    pl, sponsor.under_layers(design.back, pl.x, pl.y,
                                             car_rgb or (255, 255, 255)),
                    cat, logo_cache)
                 if pl.proto.item.kind == "watermark" else pl)
                for pl in side_logos]
            if side_logos:
                lsets = {deco_src: side_logos}
                other = next((n for n in ROLE_MAIN if n != deco_src and by_surface.get(n)),
                             None)
                if other is not None:
                    lsets[other] = [pl.mirrored() for pl in side_logos]
                for sname, pls in lsets.items():
                    lpath = out_dir / f"logos-{sname}.json"
                    logo_summary[sname] = sponsor.write_group(pls, lpath, plan, cat)
                    written.append(lpath)
                    logo_groups[sname] = {"plan": _rel(lpath, out_dir), "x": round(fcu, 1),
                                          "y": round(fcv, 1), "scale": round(ds, 3),
                                          "rot": 0.0, "mirror": False}
                notes.append(msg("옆면 로커 줄에 로고 {k}개 — {names} ({n:,}장, 반대편은 "
                                 "자리만 거울)", k=len(side_logos),
                                 names=" · ".join(pl.proto.item.name for pl in side_logos),
                                 n=logo_summary[deco_src]))

    # ---- 예산 사다리 — 넘치면 꾸밈부터 버린다 (도안이 주역이다) ----
    # ---- 옆면 모티프가 글자·로고에 **양보한다** (사용자 지시 2026-09-03) ----
    # 산포·전경 모티프는 글자·로고보다 먼저 흩어진다 — 글자·로고 상자(프레임 좌표)에
    # 닿는 모티프를 두 벌(배경·전경)에서 뺀다. 반대편은 꾸밈도 글자·로고도 자리가
    # 거울이라 같은 자로 맞는다. 큰 색면(띠·블록·베드)은 바탕이라 둔다.
    if design is not None and deco_place is not None:
        marks = [sponsor.pose_box(p)
                 for p in (design.text.poses if design.text is not None else [])
                 + number_poses]
        marks += [pl.box for pl in side_logos]
        cut_all = 0
        for dp, dpath in ((deco_plan, out_dir / "deco.json"),
                          (deco_front, out_dir / "deco-front.json")):
            if dp is None or not marks:
                continue
            kept, cut = sponsor.prune_motif_layers(dp.layers, marks, CANVAS_UNITS)
            if cut:
                dp.layers = kept
                dp.save(dpath)
                cut_all += cut
        if cut_all:
            notes.append(msg("옆면 글자·로고 상자에 닿는 모티프 {n}장을 뺀다", n=cut_all))

    # 기준은 **가장 무거운 옆면**이다 (한 면에 여러 장을 올릴 수 있다). 장수는
    # 면에 실제로 올라갈 값이다 — 꾸밈을 켠 판이면 어느 면에도 안 그려질
    # 레이어는 이미 빠졌다 (`_hand_spread`; 끈 판은 통째라 `_check`가 잡는다).
    cap = min([m.cap or 3000 for n in ROLE_MAIN
               if (m := maps.get(n)) is not None] or [3000])
    n_person = max([sum(hand_group[hand_ix[id(m)]][1]
                        for m in by_surface.get(n) or [])
                    for n in ROLE_MAIN] or [0])
    use_deco = deco_place is not None
    n_front = len(deco_front.layers) if deco_front is not None else 0
    n_text = max([len(LayerPlan.load(out_dir / g["plan"]).layers)
                  for g in text_groups.values()] or [0]) \
        + max([len(LayerPlan.load(out_dir / g["plan"]).layers)
               for g in number_groups.values()] or [0])
    n_logo = max([logo_summary.get(n, 0) for n in ROLE_MAIN] or [0])
    if use_deco and n_logo and \
            n_person + len(deco_plan.layers) + n_front + n_text + n_logo > cap:
        for n in ROLE_MAIN:                      # 로고 줄부터 뺀다 (꾸밈이 먼저다)
            logo_groups.pop(n, None)
            logo_summary.pop(n, None)
        notes.append(msg("측면이 상한 {cap:,}을 넘는다 — 로고 줄을 뺀다", cap=cap))
    if use_deco and n_person + len(deco_plan.layers) + n_front + n_text > cap:
        use_deco = False                         # 도안만 남긴다 (`_check`가 나머지를 잡는다)
        text_groups.clear()
        number_groups.clear()
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
    # 차 지도 — 이음새·차체 선을 한 번만 푼다 (`atlas`). 못 푸는 차는 빈 지도라
    # 이어 붙이기만 조용히 접힌다.
    try:
        atlas = build_atlas(maps, rigs, media=pinned) if deco else None
    except Exception:                              # 지도가 모자란 차 — 이 판은 안 잇는다
        atlas = None

    # §17 **이음새를 건너 이어질 수 있는 짝** — 관계만 세운다 (`whole.seam_links`).
    # 지도가 여기서야 서므로 구성(`plan_car`)보다 뒤다. 그리는 손은 아래
    # `_carry_macro`·`compose.seams.carry`가 이미 쥐고 있다.
    if wcp is not None and atlas is not None:
        wcp.links = wholecar.seam_links(wcp, atlas, {mp.surface for mp in hand})

    carried: dict = {}
    carried_stack: list[dict] = []
    rocker_carry: dict = {}

    def _carry_macro(name: str) -> list[dict]:
        """옆면 큰 색면을 이 면으로 **이어 그린다** (없으면 빈 목록).

        레퍼런스의 큰 색면은 한 면에서 끝나지 않는다 — 옆면의 띠가 범퍼를 돌아
        같은 높이·같은 기울기로 이어지고, 그 전환이 이음새에 숨는다. 지금까지
        이어진 것은 무채 로커 띠뿐이라(`flow_shapes`의 `rocker`) 차를 돌아보면
        옆면에만 구도가 있었다.

        가파른 색면은 안 잇는다: 위아래(벨트라인·로커)로 나가므로 애초에
        앞뒤 이음새에 안 닿는다.
        """
        if atlas is None or not use_deco or design is None:
            return []
        if side_band is None and not side_stack:
            return []
        # **흐름이 가리키는 면에만** 잇는다. 양쪽으로 다 이으면 색면이 차를 한
        # 바퀴 두르는 띠가 되고, 실측으로도 흐름 반대쪽에서는 오히려 어긋났다
        # (바닥 높이 어긋남 dv_front 0.036 → 0.162 — 이미 맞아 있던 것을 흔들었다.
        # 흐름 쪽 리어는 0.225 → 0.136으로 좋아졌다).
        if name != (ROLE_REAR if flow_rear else "front"):
            return []
        sm = maps.get(name)
        seam = atlas.seam_to(deco_src, name)

        def _one(band: "gseams.Band", color, rec: dict) -> list[dict]:
            rec["from"] = deco_src
            if sm is None or sm.uncertain or seam is None:
                rec["why"] = msg("이음새를 못 푼다")
                return []
            if abs(band.angle) > MACRO_CARRY_TILT:
                rec["why"] = msg("색면이 가파르다 ({ang:.0f}°) — 앞뒤 이음새에 안 닿는다",
                                 ang=band.angle)
                return []
            # 건너온 띠의 **두께는 지키고**, 못 지킬 만큼 좁은 면이면 아예 안 잇는다
            # (`seams.WIDTH_MIN`). 반으로 깎인 띠를 이어 붙이면 이음새에서 만나는 두
            # 띠가 서로 다른 띠로 읽힌다 — 이을 바에는 끊는 편이 사람 문법이다.
            ph = sm.paint[3] - sm.paint[1]
            con = gseams.carry(seam.fold, band, "macro", dst_box=sm.paint,
                               tilt_max=MACRO_CARRY_OUT, thick_max=MACRO_CARRY_H * ph)
            rec.update({"to": name, "policy": con.policy, "why": con.why,
                        "v_side": rnd(band.v, 1), "ang_side": rnd(band.angle, 1)})
            rec.update(con.metrics)
            if not con.carried:
                return []
            b2 = con.band
            rec.update({"v": rnd(b2.v, 1), "ang": rnd(b2.angle, 1),
                        "seam_err": {k: rnd(v, 3) for k, v in
                                     gseams.seam_error(seam.fold, band, b2).items()}})
            return flow_shapes(color, sm, mode="macro", center_v=b2.v,
                               anchor_u=b2.at_u, rot=b2.angle,
                               height=b2.thickness, cat=cat)

        out: list[dict] = []
        if side_band is not None:
            out += _one(side_band, design.pal.bed, carried)
        # **스택의 관통 띠**(벨트 블랙아웃)도 같은 자로 건넌다 — 블록 위에 얹힌
        # 층이라 블록 뒤에 그린다 (목록 뒤가 위다). 기록은 띠마다 한 벌.
        for band, color, kind in side_stack:
            rec = {"kind": kind}
            out += _one(band, color, rec)
            carried_stack.append(rec)
        return out

    def _carry_rocker(name: str) -> float | None:
        """옆면 하부 투톤의 **윗선**이 이 면에서 갖는 높이 (없으면 None).

        옛 판은 앞·뒤 범퍼의 밴드를 **그 면의 상자 몫**(또는 범퍼 로케이터)으로
        따로 잡았다. 두 면의 자가 다르므로 이음새에서 선이 어긋나고 두께가
        달라진다 — 차를 돌아보면 하부 투톤이 모서리마다 계단이 진다. 옆면의
        윗선 하나를 건너보내면 그 선이 차를 한 바퀴 돈다.
        """
        if atlas is None or side_rocker_top is None or design is None:
            return None
        sm = maps.get(name)
        seam = atlas.seam_to(deco_src, name)
        if sm is None or sm.uncertain or seam is None:
            return None
        src_map = maps.get(deco_src)
        band = gseams.Band(
            v=side_rocker_top, angle=0.0, thickness=1.0,
            at_u=(0.0 if src_map is None
                  else (src_map.paint[0] + src_map.paint[2]) / 2))
        con = gseams.carry(seam.fold, band, "rocker", dst_box=sm.paint,
                           tilt_max=90.0, thick_max=1.0)
        rocker_carry[name] = {"policy": con.policy, "why": con.why, **con.metrics}
        return con.band.v if con.carried else None

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

    # 이음새에서 넘어온 무리가 패널 안으로 밀려 들어가는 상한 (패널 크기 대비).
    SEAM_PUSH = 0.22


    # 이음새에서 넘어온 무리의 크기 자 (패널 짧은 변 대비) — 이보다 작게는 안 잰다.
    SEAM_REF = 0.50


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
                u0, v0, u1, v1 = sm.paint
                fu, fv = (u0 + u1) / 2, (v0 + v1) / 2
                d = math.hypot(fu - cu, fv - cv)
                flow = ((fu - cu) / d, (fv - cv) / d) if d > 1e-6 else (1.0, 0.0)
                got = deco_anchor(pb, flow, avoid=ink,
                                  why=msg("{src}의 도안을 이 면으로 투영", src=src))
                # **이음새에서 너무 멀리 밀지 않는다.** 물린 상자가 크면
                # `deco_anchor`의 밀어내기(상자 폭의 1.35배)가 무리를 패널
                # 한복판에 세운다 — 리어의 별무리가 범퍼 가운데 떠 있고 이웃
                # 면과 아무 관계가 없어진다 (미리보기 판정). 레퍼런스의 리어
                # 무리는 옆면에서 넘어온 쪽 모서리에 붙는다 (Fate R34의 별).
                # 밀어내기 상한은 **패널 크기**의 몫이다.
                pw, ph = u1 - u0, v1 - v0
                au = min(max(got.at[0], cu - SEAM_PUSH * pw), cu + SEAM_PUSH * pw)
                av = min(max(got.at[1], cv - SEAM_PUSH * ph), cv + SEAM_PUSH * ph)
                # 크기 자도 **패널**이 준다 — 물린 상자는 이음새에서 잘린 조각이라
                # 그 긴 변으로 재면 모티프가 티끌이 된다 (리어 실측: 무리 전체가
                # 패널 폭의 10%였다).
                ref = max(got.ref, SEAM_REF * min(pw, ph))
                got = replace(got, at=(au, av), ref=ref)
        _anchor_memo[name] = got
        return got

    def _glass_anchor(sm: gsurf.SurfaceMap) -> DecoAnchor:
        """도어 유리의 앵커 — **아래 이음새(벨트라인) 쪽, 흐름 끝**에 뭉친다.

        투영 앵커는 인물 상자를 유리 안으로 물려 놓으므로 무리가 유리 **한가운데**
        에 선다 — 큰 별 둘이 유리 복판에 뜬 꼴이고, 차체 그래픽과 아무 관계가
        없다. 레퍼런스의 유리 그래픽은 예외 없이 차체에서 올라온다: ARIS의
        픽셀은 벨트라인 바로 위에 붙고, RIN의 아네모네는 리어 쿼터에서 유리로
        이어진다. 그래서 뭉치는 자리를 **아래 가장자리 · 옆면이 고른 흐름 쪽**에
        둔다 (후보 구름은 여전히 유리 전체다).
        """
        u0, v0, u1, v1 = sm.paint
        r = rigs.get("side_" + sm.name.split("_")[-1])
        rd = r.rear_dir if r is not None else 1.0
        fs = rd if flow_rear else -rd
        w, h = u1 - u0, v1 - v0
        return DecoAnchor(box=sm.paint, center=((u0 + u1) / 2, (v0 + v1) / 2),
                          at=((u0 + u1) / 2 + fs * 0.30 * w, v0 + 0.30 * h),
                          ref=0.62 * h, why=msg("옆면 무리가 유리로 이어진다"))

    def _motifs(colors, sm, cat_, over: bool = False, anchor=None, **kw) -> list[dict]:
        """면에 직접 흩는 모티프 — 꾸밈을 끈 판에서는 빈손이다."""
        if not deco:
            return []
        an = anchor if anchor is not None else _anchor(sm)
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
            if name in number_groups:
                item["pre_groups"].append(number_groups[name])
        if mps:
            item["groups"] = [
                _hand_group_job(m, hand_ix, hand_group, out_dir) for m in mps]
        # 로고 줄은 도안 위 — 사람 판의 스폰서 로고는 z가 맨 위다
        if use_deco and name in logo_groups:
            item["groups"] = list(item.get("groups") or []) + [logo_groups[name]]
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
            # 리어·프론트는 **옆면의 띠를 건너 받는다** — 큰 색면(과 스택의 벨트
            # 띠)은 `_carry_macro`, 하부 투톤의 윗선은 `_carry_rocker`. 도안이 앉은
            # 면이라고 제 몫(범퍼 로케이터)으로 따로 잡으면 이음새에서 계단이
            # 진다. 옛 판은 이 길에서 잇기를 안 불러 33판 전부 `carry`가 비었다.
            rtop = _carry_rocker(name) if name in (ROLE_REAR, "front") else None
            band_kw = ({} if name == ROLE_EXTRA
                       else {"top_v": rtop} if rtop is not None
                       else {"center_v": _bumper_seed(media, name)})
            sh = (roof_sh if name == ROLE_EXTRA else []) \
                + (_carry_macro(name) if name in (ROLE_REAR, "front") else []) \
                + _flow(sm, mode="stripe" if name == ROLE_EXTRA else "rocker", **band_kw) \
                + _motifs(motif_c, sm, cat, n=_n(7), shapes=motifs_v,
                          box=tsegs[0] if tsegs else None)
            # **리어 데크에도 흩는다** — 블랙아웃은 지붕·필러만 덮고 데크는 본색
            # 그대로다 (`roof_blackout`이 넓은 구간을 일부러 건너뛴다). 그런데
            # 산포가 후드 구간에만 갇혀 있어서 데크가 늘 비어 있었다 — 레퍼런스의
            # 데크에는 모티프가 있다 (KOTONE의 자홍 물감 · ARIS의 픽셀).
            deck = _deck_box(tsegs, roof_sh) if name == ROLE_EXTRA else None
            if deck is not None:
                sh += _motifs(motif_c, sm, cat, n=_n(4), shapes=motifs_v, box=deck)
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
        rtop = _carry_rocker(ROLE_REAR)
        band_kw = ({"top_v": rtop} if rtop is not None
                   else {"center_v": _bumper_seed(media, ROLE_REAR)})
        items.append({"surface": ROLE_REAR,
                      "shapes": _carry_macro(ROLE_REAR)
                      + _flow(rs, **band_kw)
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
        ftop = _carry_rocker("front")
        items.append({"surface": "front",
                      "shapes": _carry_macro("front")
                      + _flow(fs, box=fs.fit(2.5, coverage=0.85,
                                             anchor="center"),
                              max_sx=2.2,
                              **({"top_v": ftop} if ftop is not None else {}))
                      + _motifs(motif_c, fs, cat, n=_n(6), shapes=motifs_v)})
        used.add("front")
        notes.append(msg("프론트에 관통 띠 + 모티프를 잇는다"))
    # 도어 유리 = 작은 모티프 (ARIS 문법). 사람이 도안을 올린 면은 이미 `used`다.
    n_motif = 0
    for wname in (("window_left", "window_right") if deco else ()):
        wm = maps.get(wname)
        if wname in used or wm is None or wm.uncertain:
            continue
        motifs = _motifs(motif_c, wm, cat, n=_n(5), shapes=motifs_v,
                         anchor=_glass_anchor(wm))
        if motifs:
            items.append({"surface": wname, "shapes": motifs})
            used.add(wname)
            n_motif += 1
    if n_motif:
        notes.append(msg("도어 유리에 모티프를 흩는다 (ARIS 문법)"))

    # ---- 글자·로고가 앉는 지도 — **온전히 보이는 자리**만 (`place.usable`: 그리는
    # 지도 ∧ 정면도 ∧ 가장자리 여유). 사용자 지시 2026-09-03: 글자·로고는 면을 벗어나면
    # 안 된다. 옆면은 설계 필드가 같은 일을 한다 (`textlayout.DRAWABLE_MIN`).
    umaps = {n: (usable(n, maps, rigs, pinned) or sm) if n not in ROLE_MAIN else sm
             for n, sm in maps.items()}

    # ---- 다른 면의 글자 — 자리를 못 박았을 때 (rear · hood · roof · window) ----
    if (deco and text_spec.active and text_spec.placement not in ("auto", "side")
            and design is not None):
        face_summary = face_text(text_spec, design, items, umaps, rigs, cat, out_dir, plan,
                                 group_unit=group_unit, hood_u=hood_u, notes=notes,
                                 written=written)
    # ---- 면 배정의 글자 — 자동 자리일 때 다른 면이 받는 몫 (사람 판: 리어 워드마크 ·
    # 윈드실드 글자 띠 · 뒷유리 워드마크 · 도어 유리 작은 문구). 옆면 글자는 그대로다.
    # 로고 줄보다 앞이다 — 로고가 글자 상자를 피해 앉는다 (`_face_busy`).
    assigned_summary: dict = {}
    if deco and text_spec.active and design is not None:
        _pinned = set(pinned_faces(text_spec.placement))
        want = [n for n in ("rear", "rear_window", "windshield", "window_left", "window_right")
                if assign[n].sponsor and n not in _pinned and n not in hand_box]
        if want:
            # 리어·프론트에 로고 줄이 설 판이면 워드마크가 내접 상자의 아래 몫을 남긴다
            # — 노출 띠가 좁은 차(실비아 리어 24%)에서 둘이 한 띠를 나눠 쓴다
            _row = {}
            if logo_protos and logo_spec.placement in ("auto", "rear") or wm_face == "rear":
                _row["rear"] = ROW_RESERVE
            if logo_protos and logo_spec.placement in ("auto", "front"):
                _row["front"] = ROW_RESERVE
            assigned_summary = assigned_text(
                text_spec, design, items, umaps, rigs, cat, out_dir, plan, faces=want,
                group_unit=group_unit, notes=notes, written=written, reserve=_row)

    # ---- 다른 면의 로고 — 리어 범퍼(가운데 워터마크 + 좌우) · 프론트 · 윈드실드 ----
    if deco and (logo_protos or wm_proto is not None):
        def _face_busy(item: dict, name: str) -> list:
            boxes = [hand_box[name]] if hand_box.get(name) else []
            for g in (item.get("groups") or []) + (item.get("pre_groups") or []):
                try:
                    bx = look(LayerPlan.load(out_dir / g["plan"]), cat).box
                except Exception:              # noqa: BLE001 — 못 읽는 그룹은 안 피한다
                    continue
                s = float(g.get("scale", 1.0)) * group_unit
                boxes.append((g["x"] + s * bx[0], g["y"] + s * bx[1],
                              g["x"] + s * bx[2], g["y"] + s * bx[3]))
            return boxes

        # 면 배정이 그 면을 비웠거나 크롭에 줬으면 로고 줄도 안 선다 (`assign`). 도어
        # 유리의 작은 로고 열은 `auto`에서만 — 사람 판의 유리 로고는 옆면 무리의
        # 되풀이라, 자리를 못 박은 판(리어·프론트·윈드실드·로커)에는 안 간다.
        _users_on = {"rear": logo_protos if logo_spec.placement in ("auto", "rear") else [],
                     "front": (logo_protos[:3] if logo_spec.placement == "auto"
                               else logo_protos if logo_spec.placement == "front" else []),
                     "windshield": logo_protos if logo_spec.placement == "windshield" else [],
                     "window_left": logo_protos[:3] if logo_spec.placement == "auto" else [],
                     "window_right": logo_protos[:3] if logo_spec.placement == "auto" else []}
        for face in ("rear", "front", "windshield", "window_left", "window_right"):
            if not logo_targets.get(face, True) or not assign[face].sponsor:
                continue
            # 사람이 그림을 올린 면은 그 그림이 그 면의 일이다 — 로고 열도 글자
            # (`assigned_text`)처럼 건너뛴다. 종전에는 도어 유리의 치비 옆에 로고
            # 하나를 끼워 넣었다 (다중 그림 픽스처 W10M reze-win: 700+109장).
            if face in hand_box:
                continue
            users = _users_on[face]
            center = wm_proto if wm_face == face else None
            if not users and center is None:
                continue
            sm = umaps.get(face)
            if sm is None or (sm.uncertain and not _deco_usable(sm)):
                continue
            item = next((it for it in items if it["surface"] == face), None)
            if item is None:
                item = {"surface": face, "fit": False}
                items.append(item)
            busy = _face_busy(item, face)
            used = (len(item.get("shapes") or []) + len(item.get("post_shapes") or [])
                    + sum(len(LayerPlan.load(out_dir / g["plan"]).layers)
                          for g in (item.get("groups") or []) + (item.get("pre_groups") or [])))
            free = (sm.cap or 1000) - used - 8
            floor_v = _carry_rocker(face) if face in (ROLE_REAR, "front") else None
            if face == "windshield" and not users:
                got = sponsor.corner(center, sm, busy, side_w=side_w, notes=notes)
                placed = [got] if got is not None else []
            else:
                placed = sponsor.face_row(users, sm, busy, side_w=side_w, floor_v=floor_v,
                                          center=center, notes=notes)
                # 가운데 줄이 막혀 있으면(변주·글자가 아래까지 내려온 면) 워터마크는
                # 귀퉁이로 물러난다 — 사람 판의 작은 로고도 범퍼 귀퉁이에 선다
                if center is not None and not any(
                        pl.proto.item.kind == "watermark" for pl in placed):
                    got = sponsor.corner(center, sm, busy + [pl.box for pl in placed],
                                         side_w=side_w, notes=notes)
                    if got is not None:
                        placed.append(got)
            if not placed:
                continue
            fixed: list = []
            for pl in placed:
                if pl.proto.item.kind == "watermark":
                    bg = car_rgb or (255, 255, 255)
                    if floor_v is not None and pl.y < floor_v:
                        bg = ROOF_DARK
                    elif (carried.get("to") == face and design is not None
                          and "v" in carried and side_band is not None
                          and abs(pl.y - carried["v"]) < 0.5 * side_band.thickness):
                        bg = design.pal.bed
                    pl = sponsor.pick_watermark(pl, bg, cat, logo_cache)
                fixed.append(pl)
            n_need = sum(len(pl.proto.layers) for pl in fixed)
            if n_need > free:
                notes.append(msg("{surface}: 로고 {n}장이 남은 예산 {free}장을 넘는다 — 뺀다",
                                 surface=face, n=n_need, free=max(0, free)))
                continue
            entry, n = sponsor.face_group(fixed, sm, out_dir / f"logos-{face}.json", plan,
                                          cat, out_dir, group_unit)
            item["groups"] = list(item.get("groups") or []) + [entry]
            written.append(out_dir / f"logos-{face}.json")
            logo_summary[face] = n
            notes.append(msg("{surface}: 로고 {k}개 — {names} ({n:,}장)", surface=face,
                             k=len(fixed), names=" · ".join(pl.proto.item.name for pl in fixed),
                             n=n))

    # 모티프가 선 면마다 **어느 도안에서 자랐나**를 적는다 — 꾸밈이 엉뚱한 자리에
    # 섰을 때 사람이 먼저 볼 것이 이 뿌리다 (면을 잘못 짚었나, 투영이 딴 면에서
    # 왔나).
    roots = [f"{it['surface']}={an.why}"
             for it in items
             if (it.get("shapes") or it.get("post_shapes"))
             and (an := _anchor_memo.get(it["surface"])) is not None]
    if roots:
        notes.append(msg("꾸밈이 자란 뿌리: {roots}", roots=" · ".join(roots)))

    # ---- 모티프가 글자·로고에 **양보한다** (사용자 지시 2026-09-03) ----
    # 면 위저드 도형(산포·전경 모티프)은 글자·로고보다 먼저 흩어져 그 위에 앉는다
    # (W11F: 리어 워드마크 33/33에 별이 겹쳤다). 글자·로고 상자에 닿는 모티프를
    # 뺀다 — 큰 색면(띠·블록)은 바탕이라 둔다.
    for it in items:
        if it["surface"] in ROLE_MAIN or not (it.get("shapes") or it.get("post_shapes")):
            continue
        marks = []
        for g in it.get("groups") or []:
            stem = Path(str(g["plan"])).stem
            if not (stem.startswith("text-") or stem.startswith("logos-")):
                continue
            try:
                bx = look(LayerPlan.load(out_dir / g["plan"]), cat).box
            except Exception:              # noqa: BLE001 — 못 읽는 그룹은 안 피한다
                continue
            s = float(g.get("scale", 1.0)) * group_unit
            marks.append((g["x"] + s * bx[0], g["y"] + s * bx[1],
                          g["x"] + s * bx[2], g["y"] + s * bx[3]))
        if not marks:
            continue
        fw = maps[it["surface"]].width if it["surface"] in maps else 1.0
        cut = 0
        for key in ("shapes", "post_shapes"):
            if it.get(key):
                it[key], k = sponsor.prune_shape_specs(it[key], marks, fw)
                cut += k
        if cut:
            notes.append(msg("{surface}: 글자·로고 상자에 닿는 모티프 {n}장을 뺀다",
                             surface=it["surface"], n=cut))

    # ---- 장수 신원 정리 — 그룹마다 장수가 달라야 한다 ----
    # 게임 그리드는 이름을 그림으로만 보여 줘서 **장수로** 그룹을 고른다
    # (`auto.itasha`). 겹치면 구성이 아예 안 서므로(=이 도안은 이타샤가 안 된다)
    # 여기서 투명 패딩으로 비켜 놓는다.
    _unique_group_counts(items, out_dir, notes)

    cfg = {"apply": apply, "car": car, "placements": items}
    # 역할표 — 어느 덩어리가 무엇으로 읽혔나 (`compose.cast`). 계측 도구가 읽는
    # 자리다; 자동 경로(전부 `hero`)에는 안 적는다 — 옛 구성 파일 그대로다.
    if manual:
        cfg["cast"] = [
            {"surface": mp.surface, "plan": _rel(hand_path[mp.key()], out_dir),
             "role": mp.role, "no_mirror": bool(mp.no_mirror),
             "pinned": bool(mp.pinned),
             "layers": hand_group[hand_ix[id(mp)]][1]}
            for mp in hand if mp.role not in ("variant", "continue")]
    if design is not None:
        # 설계 기록 — 어느 계열·팔레트·흐름이 이겼고 점수가 어땠나. 사람이
        # 결과를 보고 "왜 이렇게 짰나"를 되짚는 자리이고, 검증 도구가 읽는다.
        cfg["design"] = {
            # 프리셋은 골랐을 때만 적는다 — 자동 판의 기록은 종전과 바이트가 같다
            **({"style": pre.name} if pre is not None else {}),
            "family": design.family.name, "variant": design.pal.variant,
            "flow": "rear" if design.flow_rear else "front",
            "bed_level": round(design.level, 2),
            # 이긴 매크로 어휘 짝과 좌표하강 손잡이 — "왜 이 꼴인가"의 첫 줄이다
            "macro": list(design.macro),
            # 블록 위의 색면 스택 — 어느 조각이 섰나 (`compose.stack`)
            "stack": [p.kind for p in design.stack],
            "tweak": {k: round(v, 3) for k, v in vars(design.tweak).items()},
            "score": round(design.score.total, 4),
            "parts": {k: round(v, 3) for k, v in design.score.parts.items()},
            # 항목이 왜 그 값인지 되짚는 원자료 (테두리 명도차·묻은 몫·커버리지…)
            "info": {k: round(v, 4) for k, v in design.score.info.items()},
            "ranking": design.ranking,
            # 이음새를 건너간 큰 색면 — 어느 면으로, 어느 높이·각으로 (`compose.seams`).
            # 비면 안 이었다는 뜻이다 (가파른 색면이거나 이음새를 못 푸는 차).
            "carry": dict(carried),
            # 스택의 관통 띠가 건너간 기록 — 띠마다 한 벌 (벨트 블랙아웃)
            "carry_stack": [dict(r) for r in carried_stack],
            # **이음새 원자료** — 점수에 안 들어간다. 옆면이 이음새로 내보낸 선과
            # 면마다의 결정(이었나·끊었나·왜)이다. 이어 붙인 결과가 실제로 맞았나는
            # 이 값과 면 도형을 대 보면 나온다 (`work/lab/deco/seamcheck.py`).
            "seam": {
                # 값은 `rnd`로 — `round`는 음의 0을 남겨 같은 판이 다른 파일로 보인다
                "side": ({"rocker_top": rnd(side_rocker_top, 2)}
                         if side_rocker_top is not None else {})
                | ({"macro_v": rnd(side_band.v, 2),
                    "macro_ang": rnd(side_band.angle, 2),
                    "macro_h": rnd(side_band.thickness, 2),
                    "macro_at_u": rnd(side_band.at_u, 2)}
                   if side_band is not None else {}),
                "rocker": {k: dict(v) for k, v in rocker_carry.items()},
            },
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
                "placement": text_spec.placement, "priority": text_spec.priority,
                **({"number": text_spec.number,
                    "number_layers": max([len(LayerPlan.load(out_dir / g["plan"]).layers)
                                          for g in number_groups.values()] or [0])}
                   if text_spec.number else {})}
            if face_summary:
                cfg["design"]["text"].update(face_summary)
            if assigned_summary:
                cfg["design"]["text"]["assigned"] = assigned_summary
    # 설치 차량을 못 박고 지었으면 **구성이 그걸 기억한다** — 안 적어 두면 다시
    # 돌릴 때 이름 매칭이 다른 차를 물어 미리보기와 검증이 딴 면 지도로 돈다.
    if media:
        cfg["media"] = media
    # 로고 — 무엇을 어디에 앉혔나 (`sponsor`). 꺼진 판에는 이 칸이 없다.
    if deco and logo_spec.active:
        cfg["logos"] = {"watermark": bool(wm_proto is not None), "watermark_face": wm_face,
                        "images": len(logo_protos), "placement": logo_spec.placement,
                        "faces": dict(sorted(logo_summary.items()))}
    # 면 배정 — 사람이 정한 모드와 `auto`를 푼 결과 (`compose.facespec`). 꺼진 판에는 없다.
    if deco:
        cfg["faces"] = {
            "spec": face_spec.to_dict(),
            "got": {n: {"crop": a.crop, "sponsor": a.sponsor, "continue": a.cont,
                        "why": a.why} for n, a in sorted(assign.items())}}
    # 차 전체 구성이 무엇을 어디에 맡겼나 — 계측 도구가 읽는 자리다
    # (`work/lab/whole/ours.py`). 없는 판에는 이 칸이 아예 없다.
    if wcp is not None and wcp.jobs:
        cfg["whole"] = {n: {"role": j.kind, "layers": j.budget, "why": j.why}
                        | ({} if j.fill >= 1.0 else {"fill": rnd(j.fill, 3)})
                        for n, j in sorted(wcp.jobs.items())}
    if paint:
        cfg["paint"] = {"rgb": list(base_rgb), "hsb": list(base_hsb)}
        # 차 색 27칸 위에 사람이 더 칠하는 둘 — 캘리퍼·유리 틴트
        # (`palette.material_roles`의 실측 표). 파일 노선이 이걸 그대로 쓴다.
        # `FS_MATERIALS=0`이면 안 적는다 (스윕용 — 끄면 옛 구성 파일 그대로다).
        if os.environ.get("FS_MATERIALS", "1").strip() != "0":
            cfg["paint"]["materials"] = {k: list(v) for k, v in
                                         material_roles(lk, base_rgb).items()}
    cfg_path = run_file(out_dir, "itasha.json")
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
    written.append(cfg_path)
    for n in notes:
        log(f"  · {n}")
    return Recipe(config=cfg, written=written, notes=notes, design=design)


# 이어 그리기(면 배정 `continue`) — 벨트라인 위 몫이 이 장수 아래면 안 잇는다:
# 머리끝 몇 장을 유리에 따로 세우면 파편으로 읽힌다.
CONT_MIN_LAYERS = 12
# 벨트라인 위 몫이 유리 높이의 이 몫은 올라와야 잇는다 — 그 아래는 마스크 여백에 걸린
# 머리끝이다 (실측 줄리아 giulia-01: 7유닛/134 = 5%가 49장으로 잡혔다).
CONT_MIN_RISE = 0.12
# 유리 이음새의 배율이 이만큼 넘게 비등방이면 세로 배율이 어긋난다고 적는다 (그룹
# 변환은 균등 배율뿐이라 가로 배율을 쓰고 이음선만 맞춘다).
CONT_ANISO_NOTE = 1.15
# 리어·프론트 워드마크가 내접 상자(온전히 보이는 지도) 아래쪽에 **로고 줄에 남기는
# 몫** (높이 비) — 노출 띠가 좁은 차에서 워드마크와 로고 줄이 한 띠를 나눠 쓴다.
ROW_RESERVE = 0.5
# 옆면 글자 글리프가 차체 밴드 밖으로 나가도 되는 몫 — 사실상 0 (사용자 지시
# 2026-09-03). 1%는 마스크 계단이다.
TEXT_OUT_MAX = 0.01


# 들일 자리가 없을 때 그래도 **두는** 상한 — 이만큼 넘는 글자는 빼지 않는다
TEXT_OUT_KEEP = 0.10


# 글자가 서려면 이만큼은 **눈에 보여야** 한다 (`place.surface_exposure`, 0~1). 옆면
# 실측(줄리아·실비아): 차체 밴드 한가운데 .8~.9, 사이드실 아랫단·벨트 바로 아래 .2~.5.
TEXT_EXPOSURE_MIN = 0.5


# 글리프 잉크가 저노출 띠에 걸려도 되는 몫 — 마스크 밖(`TEXT_OUT_MAX`)과 달리 너그럽다
TEXT_LOWEXPO_MAX = 0.15


def _side_number(design, spec: TextSpec, cat: Catalog, out_dir: Path, plan: LayerPlan,
                 sides: list[str], center: tuple[float, float], ds: float,
                 notes: list[str], written: list[Path]) -> tuple[dict[str, dict], list]:
    """레이싱 번호 그룹 — 면마다 제 것(`number-<면>.json`), 이름 글자를 피해 리어 쿼터에.

    되돌림: (면 → 그룹 항목, 프레임 좌표 포즈 목록 — 로고 줄과 모티프 양보가 쓴다).
    반대편은 자리만 거울이고 숫자는 바로 읽힌다 (로고·글자 미러 금지 규칙)."""
    from .score import raster_layers
    from .textbudget import plan_tiers
    from .textbuild import _on_bed, pose_layers, text_box
    from .textlayout import number_pose, pose_mask
    from .textstyle import choose_style

    fld = design.fld
    style = design.text_style or choose_style(spec.style, design.family, None)
    tplan = design.text_plan or plan_tiers(spec, style, 400)
    aspect, hratio = text_box(spec.number, style, tplan, cat)
    avoid = None
    if design.text is not None and design.text.poses:
        avoid = np.zeros((fld.grid.rows, fld.grid.cols), bool)
        for p in design.text.poses:
            avoid |= pose_mask(fld, p)
    p = number_pose(fld, spec.number, aspect, hratio, avoid=avoid)
    if p is None:
        notes.append(msg("레이싱 번호 '{num}'을(를) 리어 쿼터에 앉힐 자리가 없다 — 뺀다",
                         num=spec.number))
        return {}, []
    _b, bed_a = raster_layers([l for l in design.back if l.label == "itasha_bed"], fld, cat)
    p.on_bed = _on_bed(fld, bed_a, p)
    groups: dict[str, dict] = {}
    poses: list = []
    fcu, fcv = center
    for i, sname in enumerate(sides):
        q = p if i == 0 else p.mirrored()
        layers = pose_layers(q, design.pal, cat, style=style, plan=tplan)
        if not layers:
            continue
        tp = design.plan(plan, cat)
        tp.layers = [replace(l, x=rnd(l.x, 4), y=rnd(l.y, 4), sx=rnd(l.sx, 4),
                             sy=rnd(l.sy, 4), rot=rnd(l.rot % 360.0, 4)) for l in layers]
        path = out_dir / f"number-{sname}.json"
        tp.save(path)
        written.append(path)
        groups[sname] = {"plan": _rel(path, out_dir), "x": round(fcu, 1), "y": round(fcv, 1),
                         "scale": round(ds, 3), "rot": 0.0, "mirror": False}
        poses.append(q)
    if groups:
        notes.append(msg("레이싱 번호 {num} — 리어 쿼터 (높이 {h:.0f}유닛 · {n}장, 반대편은 자리만 거울)",
                         num=spec.number, h=p.height, n=len(layers)))
    return groups, poses


def _side_text_guard(tset, fld, cat: Catalog, body: gsurf.SurfaceMap, u: float,
                     fc: tuple[float, float], notes: list[str],
                     out_max: float = TEXT_OUT_MAX, drop: bool = True):
    """옆면 글자 몫의 **글리프 실물**을 차체 밴드 마스크에 대고 본다 — 나가면 안으로
    밀고, 그래도 안 들면 줄이고, 그래도 안 되면 뺀다.

    설계는 필드 격자(5유닛)의 포즈 상자로 자리를 골랐다 (`textlayout.pose_fit`).
    글리프 잉크는 그 상자와 어긋날 수 있다 (기울인 사인 글자의 끝, 디센더). 미는
    양은 밖 잉크가 넘은 만큼(`place.ink_outside`)이고, 민 자리도 인물을 덮으면 안
    되므로 포즈 자로 다시 본다. 되돌림은 고친 글자 몫이거나 None(뺀다)."""
    from .textbuild import reposed
    from .textlayout import _ok, pose_fit

    def _xf(ts):
        L = np.array([[u, 0.0], [0.0, u]])
        return ink_outside(ts.layers, cat, L, np.array(fc, float), body)

    def _poses_ok(ts) -> bool:
        return all(_ok(pose_fit(fld, p), p.role) for p in ts.poses)

    frac, _push = _xf(tset)
    if frac <= out_max:
        return tset
    for k in (1.0, 0.85, 0.7):
        cand = reposed(tset, k=k) if k != 1.0 else tset
        for _ in range(3):
            frac, (du, dv) = _xf(cand)
            if frac <= out_max and _poses_ok(cand):
                notes.append(msg("옆면 글자가 차체 밴드 밖으로 나가 안으로 들였다 "
                                 "(크기 {k:.2f}배)", k=k))
                return cand
            if frac <= out_max:
                break                             # 안에는 들었는데 인물을 덮는다 — 줄여 본다
            pad = 2.0
            cand = reposed(cand, dx=(du + (pad if du > 0 else -pad if du < 0 else 0.0)) / u,
                           dy=(dv + (pad if dv > 0 else -pad if dv < 0 else 0.0)) / u)
            if not cand.layers:
                break
    if not drop:
        notes.append(msg("옆면 글자의 {pct:.0f}%가 눌린 띠에 걸리는데 들일 자리가 없다 — 그대로 둔다",
                         pct=100 * frac))
        return tset
    if frac <= TEXT_OUT_KEEP:
        # 조금 넘는 것은 둔다 — 글리프 끝이 벨트를 스치는 정도는 잘린 이름이 아니라
        # 이름이다. 빼면 옆면에 이름이 없는 판이 된다 (사용자: 없는 것보다 낫다)
        notes.append(msg("옆면 글자의 {pct:.0f}%가 차체 밴드 밖인데 들일 자리가 없다 — 그대로 둔다",
                         pct=100 * frac))
        return tset
    notes.append(msg("옆면 글자가 차체 밴드 밖으로 나가고 들일 자리가 없다 — 뺀다"))
    return None


def _continuations(hand: list[ManualPlace], assign: dict, rigs: dict, maps: dict,
                   cat: Catalog, out_dir: Path, group_unit: float,
                   notes: list[str]) -> list[ManualPlace]:
    """옆면 주역의 **벨트라인 위 몫**을 도어 유리에 세우는 배치들 (면 배정 `continue`).

    사본은 제 파일(`continue-<유리>.json` — 유리에 닿는 레이어만)이라 옆면 파일은
    종전대로 벨트라인에서 잘린 채고, 유리 쪽은 유리 마스크가 제 몫만 그린다 — 두
    묶음이 이웃한 두 면에 올라가 이음새가 안 벌어지는 [선으로 가르기]의 꼴이다.

    변환은 옆면 배치에 유리 이음새(`folds.seam_fold` — 옆면 유닛 → 유리 유닛의
    아핀 `A·p + b`, `A = diag(su, sv)`)를 곱한 것이다. 그룹 변환은 균등 배율뿐이라
    `A`가 비등방이면 그대로 못 낸다 — 가로 배율(`su`)을 쓰고 **이음선(벨트라인)이
    맞게** 세로 이동을 잡는다: 이음새에서 만나는 것이 이어 그리기의 전부이고, 그
    위의 세로 어긋남은 `sv/su`만큼이다 (실측 도어 유리 세로 늘림 1.0~3.2 중 대개
    1.0~1.6 — `game.fold.GLASS_ANISO`).
    """
    out: list[ManualPlace] = []
    for side in ROLE_MAIN:
        wname = "window_" + side.split("_")[-1]
        a = assign.get(wname)
        if a is None or not a.cont:
            continue
        wm = maps.get(wname)
        if wm is None or wm.uncertain:
            notes.append(msg("{w}: 유리 지도를 못 믿어 이어 그리기를 접는다", w=wname))
            continue
        if any(mp.surface == wname for mp in hand):
            notes.append(msg("{w}: 사람이 올린 덩어리가 있어 이어 그리기를 접는다", w=wname))
            continue
        rig = rigs.get(side)
        f = (seam_fold(side, wname, rig)
             if (rig is not None and rig.seam is not None) else None)
        if f is None:
            notes.append(msg("{w}: 옆면↔유리 이음새를 못 풀어 이어 그리기를 접는다", w=wname))
            continue
        su, sv = float(f.A[0, 0]), float(f.A[1, 1])
        b0, b1 = float(f.b[0]), float(f.b[1])
        dm = wm.drawn or wm
        # 이음선은 **유리가 그리기 시작하는 아랫변**이다 — 그 변에 닿는 옆면 v를
        # 이음새 자체에서 되짚는다 (`rig.geom.belt`는 옆면 자의 벨트라인이라 유리
        # 이음새의 아랫변과 십여 유닛 어긋난다 — 실측 줄리아 27.6 ↔ 14.5).
        pivot = (float(wm.paint[1]) - b1) / max(1e-6, sv)
        gh = float(wm.paint[3] - wm.paint[1])
        srcs = [mp for mp in hand if mp.surface == side and mp.anchors and not mp.pinned]
        for i, mp in enumerate(srcs, 1):
            k = su
            x2 = su * mp.x + b0
            y2 = k * mp.y + b1 + (sv - k) * pivot
            cp = ManualPlace(plan=mp.plan, surface=wname, x=x2, y=y2, scale=mp.scale * k,
                             rot=mp.rot, mirror=mp.mirror, role="continue")
            hp = LayerPlan.load(mp.plan)
            L, t = place_xf(cp, group_unit)
            keep = layers_on(hp, cat, L, t, dm.paint, mask=None if dm.uncertain else dm.mask)
            # 유리 안으로 얼마나 올라오나 — 머리끝 몇 유닛이 마스크 여백에 걸린 것은
            # 잇지 않는다 (사람 판의 이어 그린 머리카락은 유리 높이의 수십 %다)
            rise = 0.0
            for j in keep:
                q = layer_points(hp.layers[j], cat) @ L.T + t
                if len(q):
                    rise = max(rise, float(q[:, 1].max()) - float(wm.paint[1]))
            if len(keep) < CONT_MIN_LAYERS or rise < CONT_MIN_RISE * gh:
                notes.append(msg("{w}: 벨트라인 위 몫이 {n}장 · 유리 높이의 {r:.0%}뿐이라 안 잇는다 — {name}",
                                 w=wname, n=len(keep), r=rise / max(1e-6, gh),
                                 name=Path(mp.plan).name))
                continue
            if len(keep) > (wm.cap or 1000):
                notes.append(msg("{w}: 벨트라인 위 몫 {n:,}장이 유리 상한 {cap:,}을 넘는다 — 안 잇는다",
                                 w=wname, n=len(keep), cap=wm.cap or 1000))
                continue
            path = out_dir / (f"continue-{wname}.json" if i == 1 else f"continue-{wname}-{i}.json")
            take_layers(hp, sorted(keep)).save(path)
            out.append(replace(cp, plan=path, x=round(x2, 1), y=round(y2, 1),
                               scale=round(cp.scale, 4)))
            an = sv / max(1e-6, su)
            notes.append(msg("{w}: {name}의 벨트라인 위 {n:,}장을 이어 그린다 (유리 배율 {k:.2f}{extra})",
                             w=wname, name=Path(mp.plan).name, n=len(keep), k=su,
                             extra=(msg(" · 세로 배율이 {r:.2f}배 달라 이음선만 맞췄다", r=an)
                                    if abs(an - 1.0) > CONT_ANISO_NOTE - 1.0 else "")))
    return out


# 큰 색면을 이음새 너머로 **이어 그리는** 기울기 상한 (도). 이보다 가파른 색면은
# 위아래(벨트라인·로커)로 나가므로 앞뒤 이음새에 애초에 안 닿는다.
MACRO_CARRY_TILT = 32.0


# 이어 온 색면이 **그 면**에서 가질 수 있는 두께 (면 높이 대비)와 기울기 (도).
MACRO_CARRY_H = 0.30
MACRO_CARRY_OUT = 18.0


# 후드 인물의 대각 기울기 (도) — HINATA 레퍼런스의 후드 인물은 차축 대비
# 25~40° 사선이다. 완전 정렬(0°/90°)은 스티커 티가 난다.
HOOD_TILT = 25.0


# 후드 덩어리가 도색 상자의 이 몫보다 작으면 인물을 안 얹는다 (좁은 후드에
# 인물을 구겨 넣으면 유리·펜더로 넘쳐 파편으로 읽힌다) — 기존 보조 아트로.
HOOD_MIN_FRAC = 0.22
# **유리를 아는 차의 자는 따로다.** 위 값은 후드 덩어리가 윈드실드를 삼킨
# 상자에서 잰 것이다 — 윗면 유리를 빼고 나면 같은 차의 몫이 미아타 0.600 →
# 0.302 · 챌린저 0.384 → 0.231로 내려앉는다 (덤프 14대). 상자의 뜻이 바뀌었으니
# 자도 같이 옮긴다: 0.22를 그대로 대면 종전에 통과하던 차 셋(두랑고·시빅
# 타입R·에보 VIII)이 조용히 인물을 잃는다. 0.18은 14대 전부를 종전과 같은
# 판정으로 되돌린다 (그때 최저가 0.231이었다).
#
# 막으려던 실패(유리로 넘침)는 애초에 상자가 유리를 담고 있어서 났다 — 상자가
# 후드 그 자체인 지금은 그 넘침이 구조적으로 안 난다.
HOOD_MIN_FRAC_GLASS = 0.18


# 회전 규약 (2026-08-19 챌린저 캡처로 검증): 게임 회전값 r에서 캔버스 +y(머리)는
# 면 유닛 (-sin r, cos r) 방향이다 — rot 295°가 머리를 +u로 보냈고 캡처가 그대로
# 나왔다. gametext._place_xy의 CCW 오프셋 회전과 같은 규약이다.
HOOD_ROT_SIGN = 1.0


def _deck_box(tsegs: list, roof_sh: list[dict]) -> tuple | None:
    """윗면 구간들에서 **블랙아웃이 안 덮은 뒤 구간**(리어 데크) 상자.

    `tsegs`는 후드 구간부터의 목록이다. 블랙아웃 사각과 u가 절반 넘게 겹치는
    구간은 지붕이므로 뺀다.
    """
    if len(tsegs) < 2:
        return None
    for b in reversed(tsegs[1:]):
        bu0, _bv0, bu1, _bv1 = b
        wid = max(1e-6, bu1 - bu0)
        covered = 0.0
        for r in roof_sh:
            ru0 = r["x"] - r["sx"] * UNITS_PER_SCALE
            ru1 = r["x"] + r["sx"] * UNITS_PER_SCALE
            covered = max(covered, (min(bu1, ru1) - max(bu0, ru0)) / wid)
        if covered < 0.5:
            return b
    return None


def _hood_place(smap: gsurf.SurfaceMap, lk: Look, group_unit: float,
                hood_u: float | None = None, glass: bool = False
                ) -> tuple[float, float, float, float, str] | None:
    """윗면 최대 덩어리(후드)에 인물 합성을 **기울여** 앉힌다 (HINATA 문법).

    반환: (x, y, scale, rot, why) 또는 None (후드가 좁다/마스크가 없다).
    머리(캔버스 +y)는 **루프 쪽**을 향한다 — 앞에서 차를 볼 때 인물이 바로
    선다 (레퍼런스 전부 그렇다). 루프 방향은 후드 덩어리 중심에서 전체 마스크
    무게중심 쪽이다 (루프·데크가 그쪽에 있다).

    후드 상자는 **u-프로파일 첫 구간**이 우선이다 — 설치 마스크는 A필러가
    후드와 지붕을 이어 한 덩어리라 blob으로는 윗면 전체를 문다 (인테그라
    실측). 구간이 안 나뉘는 차만 blob으로 물러난다.

    `glass`는 이 지도가 **유리를 뺀 것인가**이다 (`game.seam.top_body`) — 그때는
    후드 상자가 윈드실드를 안 삼키므로 좁다는 자도 같이 옮긴다
    (`HOOD_MIN_FRAC_GLASS`).
    """
    segs = top_segments(smap)
    blob = segs[hood_index(segs, hood_u)] if len(segs) >= 2 else smap.blob_box()
    m = smap.mask
    if blob is None or m.size <= 1:
        return None
    bw, bh = blob[2] - blob[0], blob[3] - blob[1]
    frac = HOOD_MIN_FRAC_GLASS if glass else HOOD_MIN_FRAC
    if bw * bh < frac * smap.width * smap.height:
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
