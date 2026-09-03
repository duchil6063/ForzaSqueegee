"""자동 자리 — 편집기가 도안을 처음 앉히는 그 자리."""

from __future__ import annotations

from dataclasses import replace

from pathlib import Path

from ...game import seam as gseam, surface as gsurf
from ..catalog import Catalog, default_catalog_path
from ..model import LayerPlan
from .boxes import DEFAULT_GROUP_UNIT
from .look import Look, person_ink, rot_ink_box
from .place import (
    BODY_FILL, ManualPlace, Place, dodge_parts, fit_on, person_pose, person_scale,
    place_in_rect)


def _side_place(rig: "SideRig", lk: Look, group_unit: float, mirror: bool,
                notes: list[str] | None = None) -> tuple[Place, float]:
    """옆면 하나의 인물 자리 — 차체 밴드 예산에 내접(얼굴은 벨트 아래), 발은 로커.
    (배치, 기울기)."""
    t, _ = person_pose(lk, {rig.name: rig})
    t = t if rig.name == "side_left" else -t
    iw, ih = person_ink(lk, t, mirror)
    s, head_why = person_scale(lk, t, mirror, rig)
    box = gseam.person_span(rig.body, rig.geom, (iw * s, ih * s), rig.rear_dir)
    box, why = dodge_parts(box, rig, lk, t)
    if notes is not None:
        notes.extend(w for w in (head_why, why) if w)
    return place_in_rect(box, rig.name, lk, anchor="bottom", fill=1.0,
                         group_unit=group_unit, tilt=t, mirror=mirror,
                         paint=rig.smap.paint), t


def auto_place(name: str, plan_path: Path, lk: Look,
               maps: dict[str, gsurf.SurfaceMap], rigs: dict[str, "SideRig"], *,
               group_unit: float = DEFAULT_GROUP_UNIT, mirror: bool = False,
               fill: float = 1.0, cat: Catalog | None = None,
               notes: list[str] | None = None) -> ManualPlace | None:
    """이 면에 이 도안을 **자동 경로가 앉힐 자리** — 편집기의 첫 자리다.

    옆면은 자동 구성과 **똑같은 수**를 쓴다 (눕히기 각 + 차체 밴드 예산 +
    로커 앵커). 그래서 도안 하나를 넣고 아무것도 안 만지면 지금까지의 이타샤가
    그대로 나오고, 사람은 거기서부터 손댄다. 나머지 면은 도색 마스크 내접이다.
    못 앉히면 None.

    인물은 사람 판처럼 벨트라인 위로 조금 나간다 (`person_budget` — 머리카락·
    어깨·팔) — 넘긴 몫은 그 자리에서 잘리고 **얼굴은 벨트 아래**에 잡는다
    (`person_scale`). 유리까지 쓰려면 편집기에서 도안을 벨트라인으로 가르고
    위쪽 반을 유리 면에 따로 올린다.

    `fill`은 **투영 몫**이다 (1.0 = 종전) — 그림을 면의 그만큼 크기로 줄여
    앉힌다. 장수를 안 깎고 시각 무게만 낮추는 자리라 조연·받침 면이 쓴다
    (`whole.SurfaceJob.fill`). 옆면 뼈대 길에는 안 먹인다 — 그 자리의 배율은
    로커·벨트라인이 정하는 것이라 임의로 줄이면 발이 뜬다.
    """
    rig = rigs.get(name)
    if rig is not None:
        if not lk.head_known and Path(plan_path).is_file():
            # 얼굴을 벨트 아래에 잡는 자 — 편집기 경로는 `look`만 들고 온다
            from .intent import with_head
            lk = with_head(lk, LayerPlan.load(Path(plan_path)),
                           cat or Catalog(default_catalog_path()))
        why: list[str] = []
        pl, t = _side_place(rig, lk, group_unit, mirror, why)
        if notes is not None:
            notes.extend(w for w in why if w)
    else:
        smap = maps.get(name)
        if smap is None:
            return None
        pl = fit_on(smap, lk, anchor="bottom" if lk.kind == "tall" else "center",
                    fill=BODY_FILL * fill, bias_x=0.5, group_unit=group_unit,
                    mirror=mirror)
        if pl is None:                            # 마스크에 이 비율이 안 들어간다
            p0, q0, p1, q1 = smap.paint
            ib = rot_ink_box(lk, 0.0, mirror)
            s = min((p1 - p0) / max(1e-6, ib[2] - ib[0]),
                    (q1 - q0) / max(1e-6, ib[3] - ib[1])) * 0.8 * fill / max(1e-6, group_unit)
            g = s * group_unit
            pl = Place(surface=name, plan=Path(), scale=round(s, 3), rot=0.0,
                       x=round((p0 + p1) / 2 - g * (ib[0] + ib[2]) / 2, 1),
                       y=round((q0 + q1) / 2 - g * (ib[1] + ib[3]) / 2, 1))
    return ManualPlace(plan=Path(plan_path), surface=name, x=pl.x, y=pl.y,
                       scale=pl.scale, rot=pl.rot, mirror=mirror)


def mirror_place(mp: ManualPlace, src: gsurf.SurfaceMap,
                 dst: gsurf.SurfaceMap, surface: str) -> ManualPlace:
    """배치 하나를 **반대편 면의 거울 자리**로 옮긴다 (좌우 대칭).

    좌우 옆면은 서로의 거울이다 — 왼쪽 카메라는 차가 왼쪽을 보므로 +u가 뒤고,
    오른쪽은 그 거울이라 −u가 뒤다 (2026-08-17 캡처 확인). 그래서 "차에서 같은
    자리"는 **각 면의 도색 상자 중심을 축으로 뒤집은 곳**이다.

    표시 변환 = R(rot)·(미러면 수평뒤집기)·스케일이므로, 면 유닛 x를 뒤집는 것은
    `mirror`를 켜고 `rot` 부호를 뒤집는 것과 같다 (M·R(θ) = R(−θ)·M). 그래서
    돌아온 배치는 **거울 대칭이면서 그림 자체는 안 뒤집힌 채로** 읽힌다.
    """
    scx = (src.paint[0] + src.paint[2]) / 2
    scy = (src.paint[1] + src.paint[3]) / 2
    dcx = (dst.paint[0] + dst.paint[2]) / 2
    dcy = (dst.paint[1] + dst.paint[3]) / 2
    return ManualPlace(plan=mp.plan, surface=surface,
                       x=round(dcx + (scx - mp.x), 1),
                       y=round(dcy + (mp.y - scy), 1),
                       scale=mp.scale, rot=round((-mp.rot) % 360.0, 1),
                       mirror=not mp.mirror)


def reseat_place(mp: ManualPlace, src: gsurf.SurfaceMap,
                 dst: gsurf.SurfaceMap, surface: str) -> ManualPlace:
    """배치 하나를 반대편의 **거울 자리에 읽는 방향 그대로** 앉힌다 (로고·글자).

    `mirror_place`와 자리는 같고 뒤집기만 없다 — 거울에 비친 글자는 읽히지 않으니
    (사용자 결정 ③ 2026-09-02) 자리만 옮기고 그림은 그대로 둔다. 각은 거울
    (`-rot`)이라 기울인 글자가 반대편에서도 같은 쪽으로 기운다
    (`textlayout.TextPose.mirrored`와 같은 규약).
    """
    got = mirror_place(mp, src, dst, surface)
    return replace(got, mirror=mp.mirror, role=mp.role, no_mirror=mp.no_mirror,
                   pinned=mp.pinned)
