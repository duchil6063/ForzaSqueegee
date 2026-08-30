"""면 이음새 **접기 그래프** — 한 면의 좌표를 이웃 면 좌표로 옮기는 변환들.

이것으로 그림을 이웃 면에 **올리지는 않는다** — 면을 넘긴 몫은 그 자리에서
잘린다. 쓰임은 하나다: **꾸밈이 자랄 뿌리를 이웃 면으로 투영하는 것**
(`build._anchor`). 레퍼런스에서 도안 없는 면의 모티프 무리는 예외 없이 이웃
면에서 흘러 들어오므로(Fate R34의 리어 별무리는 리어 쿼터에서 들어온다),
"이 패널에서 보면 저 도안이 어느 쪽에 있나"를 답할 자가 필요하다.
"""

from __future__ import annotations

import numpy as np

from ...game import fold as gfold, hull as ghull, surface as gsurf
from ...i18n import msg


def seam_fold(name: str, wname: str, rig: "SideRig") -> gfold.Fold | None:
    """옆면 → 도어 유리 **실측 이음새**를 변환의 꼴로 (`game.seam.Seam`).

    차체 면끼리는 배율이 1이라 등거리지만(`game.fold`) 유리는 제 배율로 저장돼
    있어 비등방이다. 너무 늘어난 이음새(`gfold.GLASS_ANISO`)는 옮겨 봐야 자리가
    안 맞으므로 안 낸다.
    """
    s = rig.seam
    if s is None or s.su <= 1e-6 or s.sv <= 1e-6:
        return None
    if max(s.su, s.sv) / min(s.su, s.sv) > gfold.GLASS_ANISO:
        return None
    return gfold.Fold(
        src=name, dst=wname, axis="v", sign=1.0, edge=rig.geom.belt,
        A=np.diag([s.su, s.sv]),
        b=np.array([s.gu - s.cu * s.su, s.gv0 - s.belt * s.sv]),
        why=msg("유리 이음새 su {su:.2f} · 겹침 {iou:.2f}", su=s.su, iou=s.iou))


def _pillar_hints(rigs: dict[str, "SideRig"]
                  ) -> dict[str, tuple[float, float]]:
    """옆면 필러 프로필에서 읽은 윈드실드·뒷유리의 **윗면 u 띠** (`pillar_bands`).

    윗면 마스크에 유리 구멍이 없는 차(설치본 다수)의 유리 이음새 근거다.
    """
    r = rigs.get("side_left") or rigs.get("side_right")
    if r is None:
        return {}
    try:
        ws, rw = gfold.pillar_bands(r.smap, r.geom.belt, r.geom.roof,
                                    r.geom.cabin, r.rear_dir)
    except Exception:                              # 프로필이 안 서도 판은 선다
        return {}
    out: dict[str, tuple[float, float]] = {}
    if ws is not None:
        out["windshield"] = ws
    if rw is not None:
        out["rear_window"] = rw
    return out


def _all_folds(name: str, maps: dict[str, gsurf.SurfaceMap],
               rigs: dict[str, "SideRig"],
               box: tuple[float, float, float, float] | None = None
               ) -> list[gfold.Fold]:
    """이 면에서 이웃 면으로 나가는 변환 **전부** — 차체 모서리 + 유리 이음새.

    유리도 차체와 같은 그래프의 면이다 (2026-08-21): 옆면 ↔ 도어 유리는
    그린하우스 실측 이음새(`game.seam`), 윗면 ↔ 윈드실드·뒷유리·선루프는 마스크
    유리 구멍 또는 필러 프로필(`gfold.glass_folds` + `_pillar_hints`)이고, 유리
    쪽에서 차체로는 같은 변환의 역이다 (`gfold.invert`).

    차체 모서리의 이음선은 **껍질**이 되짚는다 (`game.hull`) — 앞·뒤 면과 옆·윗면은
    같은 모서리를 서로 다른 깊이에서 쥐어서, 마스크 끝선끼리 붙이면 자리가 수십
    유닛 어긋난다.
    """
    cand = gfold.folds_for(maps, name, box=box, hull=ghull.of(maps))
    if name in ("side_left", "side_right"):
        wname = "window_" + name.split("_")[-1]
        if name in rigs and wname in maps:
            got = seam_fold(name, wname, rigs[name])
            if got is not None:
                cand.append(got)
    elif name in ("window_left", "window_right"):
        sname = "side_" + name.split("_")[-1]
        if sname in rigs and sname in maps:
            got = seam_fold(sname, name, rigs[sname])
            inv = gfold.invert(got) if got is not None else None
            if inv is not None:
                cand.append(inv)
    elif name == "top":
        cand += gfold.glass_folds(maps, name, hints=_pillar_hints(rigs))
    elif name in ("windshield", "rear_window", "sunroof"):
        for f in gfold.glass_folds(maps, "top", dsts=(name,),
                                   hints=_pillar_hints(rigs)):
            inv = gfold.invert(f)
            if inv is not None:
                cand.append(inv)
    return cand
