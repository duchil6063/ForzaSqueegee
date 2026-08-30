"""구성 파일의 그룹 항목 — 플랜 파일을 쓰고 그것을 가리킨다."""

from __future__ import annotations

import json
from pathlib import Path

from ...i18n import msg
from ..catalog import Catalog
from ..model import LayerPlan
from .boxes import _rel
from .place import (
    ManualPlace, drawable, layers_on, place_xf, take_layers)


def _plan_sig(plan: LayerPlan) -> tuple:
    """도안 내용 서명 — 같은 조각을 두 번 쓰지 않게 하는 열쇠다."""
    return tuple((l.shape, round(l.x, 2), round(l.y, 2), round(l.sx, 3),
                  round(l.sy, 3), round(l.rot, 2), l.color,
                  round(l.alpha, 2), l.mask) for l in plan.layers)


def _hand_group_job(mp: ManualPlace, hand_ix: dict, hand_group: dict,
                    out_dir: Path) -> dict:
    """손 배치 하나 → 구성 파일의 그룹 항목 (면에 맞게 자른 도안을 가리킨다)."""
    return {"plan": _rel(hand_group[hand_ix[id(mp)]][0], out_dir),
            "x": round(mp.x, 1), "y": round(mp.y, 1),
            "scale": round(mp.scale, 3), "rot": round(mp.rot % 360.0, 1),
            "mirror": bool(mp.mirror)}


def _hand_spread(hand: list[ManualPlace], hand_look: dict, hand_path: dict,
                 hand_group: dict, maps: dict, rigs: dict, cat: Catalog,
                 out_dir: Path, notes: list[str], *, group_unit: float) -> None:
    """손 배치를 **면에 맞게 나눈다** — 어디서도 안 그려질 레이어를 뺀다.

    둘을 한자리에서 한다:

    1. **어느 배치에서도 안 그려질 레이어를 뺀다** — 게임은 면 도색 상자 밖을 안
       그리므로 그 레이어들은 면 상한만 잡아먹는다. 자르는 단위는 배치가 아니라
       **도안**이다 (모든 배치의 합집합): 한 그룹을 좌우·후드가 나눠 쓰므로
       배치마다 따로 자르면 큰 그룹이 배치 수만큼 늘어 준비 시간(장당 0.44초)이
       그만큼 는다. 합집합이어도 자리마다 제 면 마스크가 알아서 자르므로 그림은
       같고, 파일 장수 = 게임에 올라가는 장수(상한 검사 값)다.
    2. 같은 내용은 **한 벌만** 쓴다 — 좌우 대칭 배치는 잘린 조각도 같은 그림이라
       그룹 하나를 나눠 쓴다 (그룹마다 게임에서 따로 만들어 저장해야 한다).

    **면을 넘긴 몫은 이웃 면으로 안 간다** (사용자 지시 2026-08-27) — 그 자리에서
    잘린 그림이 된다. 이어 붙이고 싶으면 편집기에서 도안을 이음선으로 가르고
    (KFPS·FLS의 [선으로 가르기]) 한쪽을 그 면에 따로 올린다.
    """
    seen: dict[tuple, Path] = {}
    used: set[Path] = set(hand_path.values())

    def _put(base: str, part: LayerPlan) -> tuple[Path, int]:
        sig = _plan_sig(part)
        got = seen.get(sig)
        if got is not None:
            return got, len(part.layers)
        p = out_dir / f"{base}.json"
        k = 2
        while p in used:
            p, k = out_dir / f"{base}-{k}.json", k + 1
        used.add(p)
        part.save(p)
        seen[sig] = p
        return p, len(part.layers)

    # 1) 어느 면에서도 안 그려질 레이어 — 도안마다 배치 합집합으로 잰다
    keep_by_plan: dict[str, set[int]] = {}
    for mp in hand:
        hp, _hlk = hand_look[mp.key()]
        got = keep_by_plan.setdefault(mp.key(), set())
        dm = drawable(mp.surface, maps, rigs)
        if dm is None:
            got.update(range(len(hp.layers)))
            continue
        L, t = place_xf(mp, group_unit)
        keep = layers_on(hp, cat, L, t, dm.paint)
        if not keep:
            notes.append(msg("{surface}: 도안이 통째로 면 밖이다 — 자리를 다시 볼 것",
                             surface=mp.surface))
        got.update(keep)
    files: dict[str, tuple[Path, int]] = {}
    for key, keeps in keep_by_plan.items():
        hp, _hlk = hand_look[key]
        base = hand_path[key]
        if len(keeps) >= len(hp.layers):
            files[key] = (base, len(hp.layers))
            continue
        p, n = _put(f"{base.stem}-fit", take_layers(hp, sorted(keeps)))
        files[key] = (p, n)
        notes.append(msg("{name}: 어느 면에서도 안 그려질 "
                         "{cut:,}장을 빼고 {n:,}장으로 올린다",
                         name=base.name, cut=len(hp.layers) - n, n=n))
    for i, mp in enumerate(hand):
        hand_group[i] = files[mp.key()]




def _unique_group_counts(items: list[dict], out_dir: Path,
                         notes: list[str]) -> None:
    """구성 안의 그룹 플랜들이 **서로 다른 장수**를 갖게 투명 패딩을 덧댄다.

    패딩은 마지막 레이어 자리에 alpha 0·최소 크기로 얹으므로 그림도 상자도 안
    바뀐다 — 바뀌는 것은 장수(신원)뿐이다 (`auto.itasha._dodge_count`와 같은 수).
    우리가 쓴 파일(`out_dir` 안)만 건드린다.
    """
    paths: list[Path] = []
    for it in items:
        for g in (list(it.get("pre_groups") or []) + list(it.get("groups") or [])
                  + ([it] if it.get("plan") else [])):
            p = (out_dir / g["plan"]).resolve()
            if p not in paths:
                paths.append(p)
    taken: dict[int, Path] = {}
    for p in paths:
        try:
            n = len(LayerPlan.load(p).layers)
        except (OSError, ValueError):               # 없는 파일은 `_check`가 잡는다
            continue
        if n not in taken:
            taken[n] = p
            continue
        if not p.is_relative_to(out_dir.resolve()):  # 남의 파일은 못 고친다
            notes.append(msg("{name}: {other}과 장수가 {n:,}장으로 같다 "
                             "— 우리 폴더 밖이라 못 비킨다",
                             name=p.name, other=taken[n].name, n=n))
            continue
        raw = json.loads(p.read_text(encoding="utf-8"))
        pad = dict(raw["layers"][-1])
        pad.update({"alpha": 0.0, "sx": 0.01, "sy": 0.01, "label": "pad",
                    "mask": False, "stroke": -1})
        add = 0
        while n in taken:
            raw["layers"].append(dict(pad))
            n += 1
            add += 1
        p.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
        taken[n] = p
        notes.append(msg("{name}: 다른 그룹과 장수가 겹쳐 투명 패딩 {add}장을 "
                         "덧대 {n:,}장으로 비킨다 (그룹은 장수로 고른다)",
                         name=p.name, add=add, n=n))
