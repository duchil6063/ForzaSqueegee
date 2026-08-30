r"""내장 편집기가 부르는 **이타샤 엔진** — 리버리 프로젝트를 열어 고쳐 다시 쓴다.

편집기(FLS)의 [Itasha] 메뉴가 여기로 온다. 메뉴 항목 하나가 늘 같은 세 걸음이다
(`main_window_itasha.cpp`): 프로젝트를 저장하고 → 이것을 부르고 → 쓴 것을 다시
연다. 그래서 이 모듈이 받는 것도 늘 같다 — `.3so` 하나, 그리고 지금 보고 있는
구획 번호.

## 무엇이 상태인가

`.3so`는 **구워진 그림**이라 그것만으로는 "이 도안을 저 면에 0.25배로 앉혔다"를
되살릴 수 없다. 그 조리법은 프로젝트 옆의 `<이름>.fsitasha.json`에 산다. 명령이
하는 일은 조리법을 고치고 → `engine.compose`로 구성을 짓고 → 그 결과를 `.3so`로
다시 굽는 것이다.

## 사람이 편집기에서 손댄 것은 안 잃는다

두 갈래로 지킨다.

1. **그룹 이동은 조리법으로 되돌아온다.** 구성기가 지은 덩어리는 면 그룹 아래에
   `FS:decal-0-fit` 같은 이름으로 서는데(`project.livery_project`), FLS는 그룹을
   끌면 **그룹 자신의 변환**에 적는다(`editor_state.transformEntryFrames`). 다음
   명령이 그 변환을 배치 수치에 접어 넣으므로, 편집기에서 밀어 놓은 자리가 곧
   조리법의 자리가 된다.
2. **`FS:` 머리가 없는 것은 안 건드린다.** 사람이 편집기에서 직접 그린 도형은
   면 그룹 아래에 그대로 있고, 다시 구울 때 원본 노드째 옮겨 실린다.

## 시키지 않은 것은 안 한다

**꾸밈은 부르기 전에는 안 짠다** (사용자 지시 2026-08-27). 도안을 올리면 올라가는
것은 **그 면의 그 도안뿐**이다 — 띠·산포 모티프·지붕 블랙아웃은 [Grow Decoration]을
누른 뒤에야 선다. 면을 넘긴 몫도 이웃 면에 안 간다: 감아 돌리려면 편집기에서
도안을 이음선으로 **가르고**([Edit → Split Selection at a Line]) 한쪽 그룹을 그
면으로 옮긴다.

## 차가 누구인가

편집기가 `--geometry <미디어명>.fsgeom`을 같이 준다 — 그 파일 이름이 곧 설치
파일의 차다. 그래서 창에서 차를 고를 일이 없다: **지금 편집기에 올라간 차 모델이
그대로 이타샤의 차**다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ...i18n import msg
from ...paths import find_run_file, run_file
from ..catalog import Catalog, default_catalog_path
from ..model import LayerPlan
from . import bridge, folder, header as hdr, project
from .binfmt import decompose, mat_mul, transform_matrix
from .livery import SLOTS

STATE_SUFFIX = ".fsitasha.json"
STATE_VERSION = 1
# 구성기가 쓰는 도안 파일 이름 머리 (`compose.build` — `decal-<i>.json`).
DECAL = "decal-"


# ────────────────────────────── 조리법 ──────────────────────────────


@dataclass
class Studio:
    """리버리 프로젝트 하나 + 그 조리법."""

    path: Path                      # `.3so`
    doc: dict
    state: dict
    notes: list

    @property
    def work(self) -> Path:
        """구성기가 도안·구성 파일을 쓰는 자리 (프로젝트 옆)."""
        return self.path.with_suffix("").with_name(
            self.path.with_suffix("").name + ".fsitasha")

    @property
    def designs(self) -> list:
        return self.state.setdefault("designs", [])


def _blank_state() -> dict:
    return {"version": STATE_VERSION, "car": None, "media": None,
            # **꾸밈은 부르기 전에는 안 짠다** (사용자 지시 2026-08-27
            # "자동 적용 금지"). 도안을 올리면 올라가는 것은 그 도안뿐이고,
            # 띠·모티프·지붕 블랙아웃은 [Grow Decoration]을 누른 뒤에야 선다.
            # 한 번 켜면 그 뒤 굽기에는 계속 실린다 — 되돌리는 것은
            # [Drop Decoration]이다.
            "paint": None, "deco": False, "motif": None, "designs": [],
            # 글자 — 기본 꺼짐. 켜면 캐릭터 이름(+작품명)이 꾸밈의 한 요소로
            # 선다 (`compose.textspec.TextSpec`의 꼴). 꾸밈이 꺼져 있으면 안 선다.
            "text": {"enabled": False, "main": None, "sub": None, "style": "auto",
                     "placement": "auto", "priority": "normal",
                     "allow_fallback_to_game_text": True, "max_layers": None,
                     "outline": "auto", "shadow": "auto"}}


def state_path(project_path: str | Path) -> Path:
    p = Path(project_path)
    return p.with_suffix("").with_name(p.with_suffix("").name + STATE_SUFFIX)


def open_project(project_path: str | Path, *,
                 geometry: str | Path | None = None) -> Studio:
    """`.3so` + 조리법을 연다. 리버리 프로젝트가 아니면 `ValueError`."""
    p = Path(project_path).resolve()
    doc = project.read(p)
    if not doc.get("is_livery"):
        raise ValueError(msg("리버리 프로젝트가 아니다 — 이타샤 명령은 리버리에만 선다"))
    sp = state_path(p)
    state = _blank_state()
    if sp.is_file():
        try:
            state.update(json.loads(sp.read_text(encoding="utf-8")))
        except ValueError:
            pass                    # 깨진 조리법은 빈 것으로 시작한다
    st = Studio(path=p, doc=doc, state=state, notes=[])
    _learn_car(st, geometry)
    _absorb(st)
    return st


def save_state(st: Studio) -> Path:
    sp = state_path(st.path)
    sp.write_text(json.dumps(st.state, ensure_ascii=False, indent=1),
                  encoding="utf-8")
    return sp


def _learn_car(st: Studio, geometry: str | Path | None) -> None:
    """차를 정한다 — **편집기에 올라간 모델이 임자**다.

    `--geometry`의 파일 이름이 설치 미디어명이다. 그것이 없으면 조리법에 적힌
    것, 그것도 없으면 프로젝트의 차 id로 설치본을 뒤져 본다."""
    if geometry:
        media = Path(geometry).stem
        if media and media != st.state.get("media"):
            st.state["media"] = media
            st.state["car"] = None      # 차가 바뀌면 표시 이름도 다시 잡는다
        st.state["geometry"] = str(Path(geometry).resolve())
    if not st.state.get("media") and int(st.doc.get("car_id") or 0):
        st.state["media"] = _media_of_id(int(st.doc["car_id"]))


def _media_of_id(car_id: int) -> str | None:
    """게임 차 id → 설치 미디어명 (`game.carfiles`의 역)."""
    try:
        from ...game import carfiles

        for media in carfiles.list_cars():
            if carfiles.car_id(media) == car_id:
                return media
    except Exception:                   # noqa: BLE001 — 설치본이 없어도 판은 선다
        pass
    return None


# ────────────────────────────── 편집기가 손댄 것 되받기 ──────────────────────────────


def _place_matrix(d: dict):
    """배치 하나의 표시 변환 (`engine.preview._compose_group`과 같은 규약)."""
    s = float(d.get("scale", 0.25))
    return transform_matrix(float(d.get("x", 0.0)), float(d.get("y", 0.0)),
                            -s if d.get("mirror") else s, s,
                            float(d.get("rot", 0.0)), 0.0)


def _group_matrix(xf: dict):
    return transform_matrix(xf["x"], xf["y"], xf["scale_x"], xf["scale_y"],
                            xf["rotation"], xf["skew"])


def decal_numbers(designs: list) -> list[int]:
    """배치마다 구성기가 붙일 도안 파일 번호 (`decal-<n>.json`).

    번호는 **서로 다른 도안 파일마다 하나**이고 1부터다 (`compose.build` —
    같은 도안을 여러 면에 올리는 것이 기본이라 배치 수와 안 같다). 여기서 같은
    셈을 다시 해 두면 편집기 그룹 이름 ↔ 배치가 이어진다."""
    seen: dict[str, int] = {}
    out: list[int] = []
    for d in designs:
        key = str(Path(d["plan"]).resolve())
        out.append(seen.setdefault(key, len(seen) + 1))
    return out


def _absorb(st: Studio) -> None:
    """편집기에서 **그룹째 옮긴 몫**을 배치 수치로 접어 넣는다.

    구성기가 구운 그룹은 변환이 항등으로 서므로, 항등이 아닌 것은 전부 사람이
    민 것이다. 그 변환을 배치 위에 곱하고 다시 분해하면 같은 자리를 배치
    수치로 표현한 것이 나온다 — 그 다음 굽기부터는 조리법이 그 자리를 안다."""
    chunks = project.section_chunks(st.doc)
    st.state["foreign"] = {
        name: entry["foreign"] for name, entry in chunks.items()
        if entry["foreign"]}
    kept: list = []
    for d, n in zip(st.designs, decal_numbers(st.designs)):
        entry = chunks.get(d.get("surface") or "")
        if entry is None:
            kept.append(d)
            continue
        # 편집기에서 그 그룹을 **지웠으면** 조리법에서도 뺀다 — 안 그러면 다음
        # 굽기에 도로 올라온다 (`FS:decal-<n>` · `-fit`·`-top` 따위 전부).
        if not any(k == f"{DECAL}{n}" or k.startswith(f"{DECAL}{n}-")
                   for k in entry["chunks"]):
            st.notes.append(msg("{surface}: 편집기에서 지운 도안을 조리법에서 뺐다 — {name}",
                                surface=d["surface"], name=Path(d["plan"]).name))
            continue
        kept.append(d)
        xf = None
        for key in (f"{DECAL}{n}-fit", f"{DECAL}{n}"):
            if key in entry["chunks"]:
                xf = entry["chunks"][key]
                break
        if xf is None or (xf["x"] == 0.0 and xf["y"] == 0.0
                          and xf["scale_x"] == 1.0 and xf["scale_y"] == 1.0
                          and xf["rotation"] == 0.0 and xf["skew"] == 0.0):
            continue
        x, y, sx, sy, rot, _skew = decompose(
            mat_mul(_group_matrix(xf), _place_matrix(d)))
        d["x"], d["y"] = round(x, 1), round(y, 1)
        d["scale"] = round((abs(sx) + abs(sy)) / 2.0, 4)
        d["rot"] = round(rot, 1)
        d["mirror"] = sx < 0.0
        st.notes.append(msg("{surface}: 편집기에서 옮긴 자리를 받았다",
                            surface=d["surface"]))
    if len(kept) != len(st.designs):
        st.state["designs"] = kept


# ────────────────────────────── 굽기 ──────────────────────────────


def _main_index(st: Studio) -> int:
    """주역 = **면에서 가장 크게 덮는** 도안 (색·모티프·베이스 도색을 이게 정한다)."""
    from .. import compose

    cat = Catalog(default_catalog_path())
    best, area = 0, -1.0
    for i, d in enumerate(st.designs):
        try:
            plan = LayerPlan.load(Path(d["plan"]))
            lk = compose.look(plan, cat)
            b = compose.manual_box(lk, _manual(d), compose.DEFAULT_GROUP_UNIT)
            a = (b[2] - b[0]) * (b[3] - b[1])
        except Exception:               # noqa: BLE001 — 못 재면 후보에서 빠진다
            continue
        if a > area:
            best, area = i, a
    return best


def _manual(d: dict):
    from .. import compose

    return compose.ManualPlace(
        plan=Path(d["plan"]), surface=d["surface"], x=float(d.get("x", 0.0)),
        y=float(d.get("y", 0.0)), scale=float(d.get("scale", 0.25)),
        rot=float(d.get("rot", 0.0)), mirror=bool(d.get("mirror")))


def rebuild(st: Studio, *, log=None) -> dict:
    """조리법 → 구성 → `.3so`. 반환은 통계.

    구성은 자동 경로와 **같은 코드**가 짓는다 (`compose.build(manual=…)`) —
    사람이 정하는 것은 도안의 자리·크기·각도·색이고, 꾸밈 그룹·관통 밴드·지붕
    블랙아웃은 그 위에서 구성기가 짓는다."""
    from ...auto import itasha as auto_itasha

    log = log or (lambda _s: None)
    if not st.designs:
        raise ValueError(msg("올린 도안이 없다 — [Load Design into Section]으로 넣으세요"))
    work = st.work
    work.mkdir(parents=True, exist_ok=True)
    clean_work(st)          # 지난 굽기의 조각이 남으면 이름이 `-2`·`-3`으로 샌다
    main = Path(st.designs[_main_index(st)]["plan"])
    cfg = auto_itasha.compose_config(
        main, work,
        car=st.state.get("car"), media=st.state.get("media"),
        manual=[_manual(d) for d in st.designs],
        paint=True, base_rgb=(tuple(st.state["paint"])
                              if st.state.get("paint") else None),
        deco=bool(st.state.get("deco")), motif=st.state.get("motif"),
        # 구성 계열은 자동(후보 점수)이 기본이다 — 조리법에 `family`가 적혀
        # 있으면 그 계열로 못 박는다 (메뉴는 없다, 사람이 조리법에 적는 레버)
        family=st.state.get("family"),
        text=st.state.get("text") or None,
        mirror=False, preview=False, log=log)
    cfg_path = Path(cfg.path)
    doc, stats = _write_project(st, cfg_path)
    st.doc = doc
    save_state(st)
    return stats


def _write_project(st: Studio, cfg_path: Path) -> tuple[dict, dict]:
    """구성 파일 → 프로젝트 문서를 굽고 사람 몫을 도로 얹어 쓴다."""
    label = st.path.with_suffix("").name
    sections, meta = bridge.itasha_chunks(cfg_path)
    doc, stats = project.livery_project(
        sections, name=label, car_id=meta["car_id"],
        paint=bridge._livery_paint(meta),
        header=hdr.draft(folder.safe_name(label), "", meta["car_id"]))
    _restore_foreign(doc, st.state.get("foreign") or {})
    project.write(doc, st.path)
    stats.update({"car_id": meta["car_id"], "config": str(cfg_path)})
    return doc, stats


def _restore_foreign(doc: dict, foreign: dict) -> None:
    """사람이 편집기에서 직접 그린 노드를 면 그룹에 도로 얹는다 (맨 위)."""
    if not foreign:
        return
    slot_of = {surface: slot for slot, (surface, _l) in enumerate(SLOTS)}
    children = (doc.get("root") or {}).setdefault("children", [])
    by_slot = {int(n.get("livery_section_slot", -1)): n for n in children
               if isinstance(n, dict) and n.get("is_livery_section")}
    for surface, nodes in foreign.items():
        slot = slot_of.get(surface)
        if slot is None or not nodes:
            continue
        sec = by_slot.get(slot)
        if sec is None:
            sec = project._plain_group(f"sec{slot}", SLOTS[slot][1], [])
            sec.update({"is_livery_section": True, "livery_section_slot": slot})
            children.append(sec)
            by_slot[slot] = sec
        sec.setdefault("children", []).extend(nodes)


# ────────────────────────────── 명령 ──────────────────────────────


def surface_of_slot(slot: int) -> str:
    if not 0 <= slot < len(SLOTS):
        raise ValueError(msg("구획 번호가 범위 밖이다 — {slot}", slot=slot))
    return SLOTS[slot][0]


def _maps(st: Studio):
    from .. import compose

    return compose.surfaces_for(st.state.get("car"), media=st.state.get("media"))


def act_load_design(st: Studio, design: str | Path, surface: str) -> str:
    """도안 하나를 이 면에 올린다 — 우리 도안 · KFPS JSON · FLS 파일 아무거나.

    셋을 한 문으로 받는다 (사용자 요청 2026-08-26): 어느 것을 골랐든 우리
    도안(plan.json)으로 바꿔 작업 폴더에 두고 그것을 앉힌다."""
    plan = import_design(design, st.work / "designs")
    d = {"plan": str(plan), "surface": surface, "x": 0.0, "y": 0.0,
         "scale": 0.25, "rot": 0.0, "mirror": False}
    st.designs.append(d)
    _auto_place_one(st, d)
    return f"{Path(design).name} → {surface}"


def import_design(src: str | Path, out_root: str | Path) -> Path:
    """도안 · KFPS 타입코드 JSON · `.3so`/`C_group` → **우리 도안**.

    이미 우리 도안이면 그대로 쓴다 (베끼지 않는다 — 원본이 고쳐지면 그것을
    본다)."""
    from ..kfpsjson import import_kfps_to, sniff_kfps

    p = Path(src).resolve()
    out_root = Path(out_root)
    if p.is_dir():
        cand = find_run_file(p, "plan.json")
        if cand.is_file():
            return cand
    if p.suffix.lower() == ".json":
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise ValueError(msg("JSON을 못 읽는다 — {name}: {error}",
                                 name=p.name, error=e)) from e
        if isinstance(raw, dict) and "layers" in raw:
            return p                    # 우리 도안 그대로
        if sniff_kfps(p):
            out = out_root / folder.safe_name(p.stem, "design")
            return import_kfps_to(p, out)[2]
        raise ValueError(msg("도안으로 못 읽는다 — {name}", name=p.name))
    kind = folder.sniff(p)
    if kind is None:
        raise ValueError(msg("도안으로 못 읽는다 — {name} "
                             "(우리 도안 · KFPS JSON · .3so · C_group)", name=p.name))
    got, _stats = bridge.import_any(p, out_root)
    if not got.name.endswith("plan.json"):
        raise ValueError(msg("{name}은 비닐 그룹이 아니라 리버리다 — "
                             "[Load Design]은 그룹 하나를 받는다", name=p.name))
    return got


def _auto_place_one(st: Studio, d: dict) -> None:
    """도안 하나를 **자동 경로가 앉히는 자리**에 앉힌다."""
    from .. import compose

    maps = _maps(st)
    plan = LayerPlan.load(Path(d["plan"]))
    cat = Catalog(default_catalog_path())
    lk = compose.look(plan, cat)
    rigs = compose.side_rigs(maps, [], media=st.state.get("media"))
    mp = compose.auto_place(d["surface"], Path(d["plan"]), lk, maps, rigs,
                            mirror=bool(d.get("mirror")), notes=st.notes)
    if mp is None:
        st.notes.append(msg("{surface}: 자동 자리를 못 잡았다 — 프리셋으로 둔다",
                            surface=d["surface"]))
        return
    d.update({"x": mp.x, "y": mp.y, "scale": mp.scale, "rot": mp.rot,
              "mirror": mp.mirror})


def designs_of_groups(st: Studio, groups: list) -> list[dict]:
    """편집기에서 **고른 그룹**들이 가리키는 도안.

    편집기가 넘기는 이름은 `FS:` 머리를 뗀 덩어리 이름이다 (`decal-1-fit` ·
    `decal-1-window_left` · `deco` …). 그중 도안 덩어리만 배치로 되짚는다 —
    `decal-<n>`의 `<n>`이 `decal_numbers`가 매긴 번호다. 한 도안을 여러 면에
    올렸으면 그 번호는 여럿을 가리키므로, **어느 면의 것인가**까지 이름에서
    읽어 좁힌다."""
    import re

    want: list[tuple[int, str | None]] = []
    for raw in groups or []:
        name = str(raw)
        if name.startswith(project.CHUNK_PREFIX):
            name = name[len(project.CHUNK_PREFIX):]
        m = re.match(r"^decal-(\d+)(?:-(.+))?$", name)
        if m:
            tail = m.group(2)
            want.append((int(m.group(1)), None if tail == "fit" else tail))
    if not want:
        return []
    numbers = decal_numbers(st.designs)
    out: list[dict] = []
    for d, n in zip(st.designs, numbers):
        for num, surface in want:
            if num == n and (surface is None or surface == d["surface"]):
                if d not in out:
                    out.append(d)
    # 면 이름이 안 맞으면 그 번호의 배치를 전부 잡는다
    if not out:
        for d, n in zip(st.designs, numbers):
            if any(num == n for num, _s in want) and d not in out:
                out.append(d)
    return out


def _targets(st: Studio, surface: str | None, groups: list | None, *,
             what: str, fallback_all: bool) -> list[dict]:
    """이 명령이 손댈 도안 — **고른 그룹 > 보고 있는 면 > (허락하면) 전부**.

    편집기는 늘 지금 열린 구획을 같이 넘기는데 창이 막 떴을 때 그것은 대개
    Front다. 면 하나를 고르는 뜻이 아닌 명령(좌우 대칭 따위)은 그래서 전부로
    물러난다."""
    if not st.designs:
        raise ValueError(msg("올린 도안이 없다 — [Load Design into Section]으로 넣으세요"))
    picked = designs_of_groups(st, groups or [])
    if picked:
        return picked
    hit = [d for d in st.designs if surface and d["surface"] == surface]
    if hit:
        return hit
    if not fallback_all:
        raise ValueError(
            msg("{what}: 어느 도안에 걸지 못 정했다 — 레이어 나무에서 그 도안 "
                "그룹(FS:decal-…)을 고르거나, 도안이 있는 면을 열고 누르세요",
                what=what)
            + (msg(" (지금 면: {surface})", surface=surface) if surface else ""))
    if surface:
        st.notes.append(msg("{surface}에 올린 도안이 없다 — 리버리 전체로 돈다",
                            surface=surface))
    return list(st.designs)


def act_auto_place(st: Studio, surface: str | None,
                   groups: list | None = None) -> str:
    """고른 도안(또는 이 면·전부)을 자동 자리로 되돌린다."""
    hit = _targets(st, surface, groups, what=msg("자동 배치"), fallback_all=True)
    for d in hit:
        _auto_place_one(st, d)
    return msg("도안 {n}장을 자동 자리로", n=len(hit))


def act_decoration(st: Studio, on: bool) -> str:
    """꾸밈(관통 띠·산포 모티프·지붕 블랙아웃)을 짜 넣거나 뺀다.

    **부르기 전에는 안 짠다** (사용자 지시 2026-08-27 "자동 적용 금지") — 도안을
    올리는 것과 꾸밈이 자라는 것은 별개의 일이고, 후자는 사람이 시킬 때만 한다.
    한 번 켠 뒤로는 그 조리법에 남아 다음 굽기에도 실린다."""
    st.state["deco"] = bool(on)
    return (msg("꾸밈을 짠다 (띠·모티프·지붕)") if on
            else msg("꾸밈을 뺀다 — 도안만 올린다"))


def act_text(st: Studio, fields: dict) -> str:
    """캐릭터 이름 글자를 켜거나 고친다 — 문자열은 **그대로** 둔다 (띄어쓰기·대소문자).

    `fields`는 `TextSpec`의 열쇠들(있는 것만 갱신). `main`이 비면 끈다."""
    from ..compose.textspec import TextSpec

    cur = dict(st.state.get("text") or {})
    cur.update({k: v for k, v in fields.items() if v is not None})
    if "main" in fields and fields["main"] is None:    # 명시적으로 지운다 (no-text)
        cur["main"] = None
    cur["enabled"] = bool(cur.get("main"))
    spec = TextSpec.from_dict(cur)           # 값 검사 (모르는 스타일이면 ValueError)
    st.state["text"] = spec.to_dict()
    if not spec.active:
        return msg("글자를 뺀다")
    if not st.state.get("deco"):
        st.notes.append(msg("꾸밈이 꺼져 있어 글자도 지금은 안 선다 — "
                            "[Grow Decoration]을 누르면 같이 선다"))
    return msg("글자 {main!r}{sub} · 스타일 {style} · 자리 {place}",
               main=spec.main, sub=(msg(" + {sub!r}", sub=spec.sub) if spec.sub else ""),
               style=spec.style, place=spec.placement)


def act_decorate(st: Studio, *, composition: str | None = None,
                 motif: str | None = None, paint: str | None = None,
                 auto_paint: bool = False, text: dict | None = None,
                 drop_text: bool = False) -> str:
    """**자동 꾸밈 창** — 구성 계열·모티프 계열·바탕 도색·글자를 한 번에 받아 켠다.

    편집기의 [Auto Decoration...] 대화상자가 부른다 (`flsedit decorate`). 준 것만
    바꾼다 — `composition`/`motif`는 "auto"면 자동(None), `paint`는 #RRGGBB,
    `auto_paint`면 도안에서 고르고, `text`는 스펙 열쇠들(`main`이 비면 끈다),
    `drop_text`면 글자를 뺀다. 마지막에 꾸밈을 켠다 — 이 창의 뜻이 그것이다."""
    from .. import compose

    said: list[str] = []
    if composition is not None:
        said.append(act_family(st, None if composition == "auto" else composition))
    if motif is not None:
        said.append(act_motif(st, None if motif == "auto" else motif))
    if auto_paint:
        said.append(act_base_paint(st, None, True))
    elif paint is not None:
        said.append(act_base_paint(st, paint, False))
    if drop_text:
        said.append(act_text(st, {"main": None}))
    elif text is not None:
        said.append(act_text(st, text))
    st.state["deco"] = True
    st.notes = [n for n in st.notes if "[Grow Decoration]" not in n]   # 이제 켰다
    del compose
    return msg("자동 꾸밈: {what}", what=" · ".join(said) if said else msg("그대로 짠다"))


def act_family(st: Studio, family: str | None) -> str:
    """옆면 꾸밈의 **구성 계열**을 못 박거나(None이면 자동) 푼다.

    기본은 자동 — 후보를 다 지어 점수로 고른다 (`compose.design`). 사람이 계열을
    고르면 그 계열 안에서만 고른다. 꾸밈이 꺼져 있으면 지금은 안 보인다."""
    from .. import compose

    if family is not None and family not in compose.FAMILIES:
        raise ValueError(msg("모르는 구성 계열: {family!r} (있는 것: {families})",
                             family=family, families=", ".join(compose.FAMILIES)))
    st.state["family"] = family
    if not st.state.get("deco"):
        st.notes.append(msg("꾸밈이 꺼져 있어 지금은 안 보인다 — "
                            "[Grow Decoration]을 누르면 이 계열로 짠다"))
    return msg("구성 계열 {family}", family=family or msg("자동"))


def act_motif(st: Studio, family: str | None) -> str:
    from .. import compose

    if family is not None and family not in compose.MOTIF_SETS:
        raise ValueError(msg("모르는 모티프 계열: {family} (있는 것: {families})",
                             family=family,
                             families=", ".join(compose.MOTIF_FAMILIES)))
    st.state["motif"] = family
    if not st.state.get("deco"):
        # 계열만 골라 두는 것은 **꾸밈을 켜는 일이 아니다** (자동 적용 금지) —
        # 그래도 아무 일도 안 일어난 것처럼 보이므로 그 사실을 말한다.
        st.notes.append(msg("꾸밈이 꺼져 있어 지금은 안 보인다 — "
                            "[Grow Decoration]을 누르면 이 계열로 자란다"))
    return msg("모티프 계열 {family}", family=family or msg("자동"))


def act_mirror(st: Studio, surface: str | None,
               groups: list | None = None) -> str:
    """좌우 대칭 — 옆면·도어 유리의 도안을 반대편에 거울로 세운다.

    같은 면에 이미 도안이 있으면 **그것을 갈아 끼운다** (두 벌이 되지 않는다)."""
    from .. import compose

    pairs = {"side_left": "side_right", "side_right": "side_left",
             "window_left": "window_right", "window_right": "window_left"}
    maps = _maps(st)
    # **보고 있는 면이 대칭할 수 있는 면이 아니면 전부 한다.** 편집기는 늘 지금
    # 열린 구획을 같이 넘기는데, 창이 막 떴을 때 그것은 대개 Front다 — 거기서
    # 멈추면 메뉴가 "옆면에서만 된다"는 말만 하고 아무 일도 안 한다.
    if surface is not None and surface not in pairs:
        st.notes.append(msg("{surface}은 대칭할 수 있는 면이 아니다 — 옆면·도어 "
                            "유리를 전부 대칭한다", surface=surface))
        surface = None
    srcs = [d for d in st.designs
            if d["surface"] in pairs
            and (surface is None or d["surface"] == surface)]
    if not srcs:
        raise ValueError(msg("좌우 대칭은 옆면·도어 유리에서만 선다 — "
                             "그 면에 올린 도안이 없다"))
    done: list[str] = []
    for d in list(srcs):
        dst = pairs[d["surface"]]
        sm, dm = maps.get(d["surface"]), maps.get(dst)
        if sm is None or dm is None:
            st.notes.append(msg("{surface}: 면 지도가 없다 — 대칭을 건너뛴다",
                                surface=dst))
            continue
        mp = compose.mirror_place(_manual(d), sm, dm, dst)
        st.state["designs"] = [o for o in st.designs
                               if not (o["surface"] == dst
                                       and o["plan"] == d["plan"])]
        st.designs.append({"plan": d["plan"], "surface": dst, "x": mp.x,
                           "y": mp.y, "scale": mp.scale, "rot": mp.rot,
                           "mirror": mp.mirror})
        done.append(f"{d['surface']} → {dst}")
    if not done:
        raise ValueError(msg("대칭할 면을 못 찾았다"))
    return msg("좌우 대칭: {done}", done=" · ".join(done))


def act_base_paint(st: Studio, color: str | None, auto: bool) -> str:
    """베이스 도색 — 차 전체에 깔리는 단색 (비닐이 아니라 **도색**이다).

    안 주면 도안에서 고른다 (`compose.base_paint` — 레퍼런스 분포 규칙: 단일
    색조가 지배하는 도안만 테마색, 나머지는 도안 명도의 반대인 흰/검)."""
    if auto or color is None:
        st.state["paint"] = None
        rgb = _auto_paint(st)
        return (msg("베이스 도색을 도안에서 고른다 — #{r:02X}{g:02X}{b:02X}",
                    r=rgb[0], g=rgb[1], b=rgb[2])
                if rgb else msg("베이스 도색을 도안에서 고른다"))
    st.state["paint"] = list(_parse_rgb(color))
    r, g, b = st.state["paint"]
    return msg("베이스 도색 #{r:02X}{g:02X}{b:02X}", r=r, g=g, b=b)


def _auto_paint(st: Studio) -> tuple[int, int, int] | None:
    from .. import compose

    if not st.designs:
        return None
    try:
        plan = LayerPlan.load(Path(st.designs[_main_index(st)]["plan"]))
        rgb, _hsb = compose.base_paint(
            compose.look(plan, Catalog(default_catalog_path())))
        return tuple(int(v) for v in rgb)
    except Exception:                   # noqa: BLE001 — 못 고르면 굽기가 고른다
        return None


def _parse_rgb(text: str) -> tuple[int, int, int]:
    s = str(text).strip().lstrip("#")
    if len(s) != 6:
        raise ValueError(msg("색은 #RRGGBB로 준다 — {text!r}", text=text))
    try:
        return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))
    except ValueError as e:
        raise ValueError(msg("색은 #RRGGBB로 준다 — {text!r}", text=text)) from e


def act_export(st: Studio, out_root: str | Path | None) -> str:
    """게임이 읽는 리버리 컨테이너 한 벌 (`Livery_<이름>/C_livery` + `header`).

    저장 컨테이너 뿌리에 그 폴더를 두면 인게임 리버리 목록에 뜬다. 베이스
    도색까지 그 파일에 실리므로 자동차 도색 메뉴를 누를 일이 없다."""
    cfg = find_run_file(st.work, "itasha.json")
    if not cfg.is_file():
        raise ValueError(msg("아직 구운 구성이 없다 — 도안을 올리고 한 번 지으세요"))
    root = Path(out_root) if out_root else st.path.parent
    label = st.path.with_suffix("").name
    out, stats = bridge.itasha_folder(cfg, root, name=label)
    return msg("리버리 컨테이너: {out}  ({layers:,}장 · 차 id {car_id})",
               out=out, layers=stats["layers"], car_id=stats.get("car_id", 0))


# ────────────────────────────── 비닐 그룹 프로젝트 내보내기 ──────────────────────────────


def export_group(project_path: str | Path, fmt: str,
                 out: str | Path | None = None) -> tuple[Path, str]:
    """**비닐 그룹 프로젝트**를 셋 중 하나로 내보낸다 (사용자 요청 2026-08-26).

    | 갈래 | 무엇 | 어디에 쓰나 |
    |---|---|---|
    | `fls` | `LayerGroup_<이름>/C_group` + `header` | 게임 저장 폴더에 두면 인게임 그리드에 뜬다 |
    | `kfps` | KFPS 타입코드 JSON | 내장 KFPS 편집기·KFPS 도구 |
    | `plan` | 우리 도안 `<이름>.plan.json` (+프리뷰) | 이 도구의 모든 명령 |

    `fls`는 편집기 제 [File → Export]와 같은 것을 쓰지만 **자리를 묻지 않는다**
    (프로젝트 옆이 기본) — 셋을 한 메뉴에서 고르게 하려고 여기 같이 둔다.
    """
    p = Path(project_path).resolve()
    doc = project.read(p)
    if doc.get("is_livery"):
        raise ValueError(msg("리버리 프로젝트다 — 이 내보내기는 비닐 그룹의 것이다"))
    label = str(doc.get("name") or p.with_suffix("").name)
    layers, stats = project.layers_of(doc)
    if not layers:
        raise ValueError(msg("프로젝트에 도형이 없다"))
    root = Path(out) if out else p.parent
    if fmt == "fls":
        got = folder.export_folder(root, label, livery_kind=False)
        wst = folder.write_group(got, layers, name=label, creator="")
        return got, msg("비닐 그룹 컨테이너: {out}  ({layers:,}장)",
                        out=got, layers=wst["layers"])
    plan_dir = root if out and Path(root).suffix == "" else root / folder.safe_name(
        label, "group")
    plan_path = bridge._write_plan(layers, plan_dir)
    if fmt == "plan":
        return plan_path, msg("도안: {path}  ({layers:,}장)",
                              path=plan_path, layers=len(layers))
    if fmt != "kfps":
        raise ValueError(msg("모르는 갈래: {fmt} (fls · kfps · plan)", fmt=fmt))
    from ..kfpsjson import export_typecode

    data, est = export_typecode(LayerPlan.load(plan_path),
                                Catalog(default_catalog_path()))
    if not data["shapes"]:
        raise ValueError(msg("KFPS로 내보낼 수 있는 도형이 하나도 없다"))
    kp = run_file(plan_path.parent, "kfps.json")
    kp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return kp, msg("KFPS JSON: {path}  ({shapes:,}장 — "
                   "정확 {exact} · 마스크 {masks} · 근사 {approx})",
                   path=kp, shapes=len(data["shapes"]), exact=est["exact"],
                   masks=est["masks"], approx=est["approx"])


def clean_work(st: Studio) -> None:
    """작업 폴더를 비운다 (구성기가 매번 새로 쓴다 — 옛 조각이 섞이면 안 된다)."""
    for name in ("itasha.json", "preview_itasha.png"):
        (st.work / name).unlink(missing_ok=True)          # 예전 이름
        run_file(st.work, name).unlink(missing_ok=True)
    for p in st.work.glob("decal-*.json"):
        p.unlink(missing_ok=True)
    for p in st.work.glob("deco*.json"):
        p.unlink(missing_ok=True)
    for p in st.work.glob("shapes-*.json"):
        p.unlink(missing_ok=True)
