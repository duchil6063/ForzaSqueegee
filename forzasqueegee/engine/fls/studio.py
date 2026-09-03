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
2. **조리법이 모르는 덩어리는 도안으로 받는다** (`_adopt`). 도안 그룹을 선으로
   가르면 조각은 `FS:` 머리를 잃고, 사람이 직접 그린 도형도 그렇다. 베낀 사본은
   머리를 그대로 두고 이름 뒤에 " (Copy)"가 붙는다(`FS:decal-1 (Copy)`) — 머리가
   아니라 **조리법이 그것을 다시 지을 수 있느냐**(`_claimed`)가 구성기 몫을 가른다 — 그것이 **지금 차에
   실린 그림**이다. 그래서 면 아래 `FS:` 없는 덩어리는
   레이어를 면 유닛 그대로 도안 파일로 떠서 항등 배치로 조리법에 올리고, 꾸밈은
   그 위에 짓는다. 다음 굽기부터는 `FS:decal-<n>` 덩어리라 1번 규칙을 탄다.
   숨긴 그룹·래스터 로고·안내선만 종전처럼 원본 노드째 옮겨 실린다.
   **구성기 덩어리 안을 손댄 것도 같다** — `FS:decal-<n>` 그룹 안에서 레이어를
   지우거나 더하면 장수가 구울 때 적어 둔 값(`state.baked`)과 달라지고, 그 덩어리는
   조리법이 다시 못 짓는 그림이 되어 지금 있는 그대로 도안으로 받는다. 안 그러면
   다음 굽기가 도안 파일에서 지운 레이어를 도로 올린다.

## 실린 것 전부가 재료다 — 역할표

사용자 결정 2026-09-02: 편집기에 실린 덩어리 **전부**가 꾸밈의 재료다. 열 때마다
덩어리마다 역할을 읽어 조리법에 싣는다 (`cast` → `compose.cast`): **주역**(옆면
구성의 앵커 — 가장 크게 덮는 주역이 색·모티프·도색을 정한다) · **보조**(사람이
놓은 면에 그대로 두고 그 면의 변주는 안 짓는다) · **로고** · **글자**(미러하지
않는다) · **그대로**(꾸밈이 건드리지 않는다). 사람이 [Auto Decoration] 창의 실린
그림 표에서 고치면 `role_user`로 못 박혀 다시 열어도 그대로다 (`act_roles`).
차 예산(`compose.whole.allocate_hier`)은 사람이 올린 덩어리를 **고정 질량**으로
받고, 그 덩어리가 앉은 면에는 변주를 안 앉힌다.

## 로고와 좌우 (3단계)

로고는 두 곳에서 온다 (사용자 결정 ②): **내장 워터마크**(기본 켬, `catalog/kit`)와
**사용자 로고 이미지**(대화상자 슬롯 0~N — 열 때 벡터화해 `<작업 폴더>/logos/`에
캐시, `act_logos`). 편집기에 올린 덩어리 중 역할이 로고인 것은 그 자리 그대로다.
앉히는 문법은 `compose.sponsor`다. **로고·글자는 미러하지 않는다** (사용자 결정
③) — 좌우 대칭이 그것들을 만나면 반대편의 거울 자리에 읽는 방향 그대로 다시
앉힌다 (`compose.reseat_place`). "한쪽에만 있으면 반대편에 대칭"(`symmetry`,
기본 켬)은 자동 꾸밈 창이 켜질 때 한 번 돈다 (`_symmetrize`) — 그림은 거울,
로고·글자는 다시 앉히기, 우리가 세운 사본은 `symmetry` 표시를 달아 끄면 걷는다.

## 시키지 않은 것은 안 한다

**꾸밈은 부를 때만 짠다** (사용자 지시 2026-08-27 · 2026-09-01). 도안을 올리면
올라가는 것은 **그 면의 그 도안뿐**이다 — 띠·산포 모티프·지붕 블랙아웃은
[Grow Decoration]·[자동 꾸밈]을 누른 **그 판에만** 선다. 도안을 건드리는 다음
명령(도안 올리기·좌우 대칭·자동 자리)은 꾸밈을 도로 끈다 (`hush_deco`) — 고른
계열·이름 글자는 조리법에 남으니 다시 누르면 그 설정으로 자란다. 면을 넘긴 몫도 이웃 면에 안 간다: 감아 돌리려면 편집기에서
도안을 이음선으로 **가르고**([Edit → Split Selection at a Line]) 한쪽 그룹을 그
면으로 옮긴다.

## 차가 누구인가

편집기가 `--geometry <미디어명>.fsgeom`을 같이 준다 — 그 파일 이름이 곧 설치
파일의 차다. 그래서 창에서 차를 고를 일이 없다: **지금 편집기에 올라간 차 모델이
그대로 이타샤의 차**다.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from ...i18n import msg
from ...paths import find_run_file, run_file
from ..catalog import Catalog, default_catalog_path
from ..model import Layer, LayerPlan
from . import bridge, folder, header as hdr, project
from .binfmt import IDENTITY, decompose, mat_mul, transform_matrix
from .livery import SLOTS

STATE_SUFFIX = ".fsitasha.json"
STATE_VERSION = 1
# 구성기가 쓰는 도안 파일 이름 머리 (`compose.build` — `decal-<i>.json`).
DECAL = "decal-"
# 편집기에서 받은 도안이 사는 작업 폴더 안 자리 (`_adopt`).
ADOPTED_DIR = "adopted"

# 구성기가 짓는 덩어리 이름 (`preview.surface_chunks` · `compose.build`):
# `decal-1-fit` · `deco-front` · `text-side_left` · `shapes-over`(+`-mirror`).
# **이 문법에 안 맞는 `FS:` 그룹은 사람 몫이다** — 편집기의 그룹 복사는 이름을
# 그대로 두고 " (Copy)"를 붙이므로(`FS:decal-1 (Copy)`) 머리만 보면 구성기 몫과
# 안 갈린다. 그 사본은 조리법이 모르는 그림이라 `_adopt`가 도안으로 받는다.
MADE_CHUNK = re.compile(r"^(?:decal-\d+|deco|shapes|text|logos)(?:-[A-Za-z0-9_.]+)*$")
DECAL_CHUNK = re.compile(r"^decal-(\d+)(?:-.*)?$")


def _claimed(chunk: str, live: set[int], deco_on: bool) -> bool:
    """이 덩어리를 **조리법이 다시 지을 수 있나** (`_adopt`가 건너뛰어도 되나).

    이름이 구성기 문법이라는 것만으로는 모자란다 — 조리법이 없거나(다른 이름으로
    저장하면 `.3so`만 따라간다) 그 도안이 조리법에서 빠졌으면 **아무도 그것을 다시
    안 짓는다**. 그때 건너뛰면 차에 실린 그림이 통째로 사라진다.

    - `decal-<n>…` — 그 번호의 도안이 조리법에 살아 있을 때만 (넘친 조각은 이웃
      면에 앉으므로 면은 안 본다).
    - `deco` · `shapes` · `text` — 꾸밈이 켜져 있을 때만 (꺼진 판에서는 안 자란다).
    """
    m = DECAL_CHUNK.match(chunk)
    if m:
        return int(m.group(1)) in live
    return deco_on if MADE_CHUNK.match(chunk) else False


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
            # **꾸밈은 부를 때만 짠다** (사용자 지시 2026-08-27 "자동 적용 금지",
            # 2026-09-01 "부를 때만"). 도안을 올리면 올라가는 것은 그 도안뿐이고,
            # 띠·모티프·지붕 블랙아웃은 [Grow Decoration]·[자동 꾸밈]을 누른
            # 그 판에만 선다 — 도안을 건드리는 다음 명령이 도로 끈다
            # (`hush_deco`). 고른 계열·이름 글자는 남으니 다시 누르면 그 설정으로
            # 자란다.
            "paint": None, "deco": False, "motif": None, "designs": [],
            # 글자 — 기본 꺼짐. 켜면 캐릭터 이름(+작품명)이 꾸밈의 한 요소로
            # 선다 (`compose.textspec.TextSpec`의 꼴). 꾸밈이 꺼져 있으면 안 선다.
            "text": {"enabled": False, "main": None, "sub": None, "style": "auto",
                     "engine": "font",
                     "placement": "auto", "priority": "normal",
                     "allow_fallback_to_game_text": True, "max_layers": None,
                     "outline": "auto", "shadow": "auto"},
            # 로고 — 내장 워터마크(기본 켬) + 사용자 로고 이미지 (`compose.LogoSpec`).
            "logos": {"watermark": True, "images": [], "placement": "auto"},
            # 한쪽 옆면(도어 유리)에만 있으면 반대편에 세운다 — 그림은 거울, 로고·
            # 글자는 읽는 방향 그대로 (`_symmetrize`).
            "symmetry": True,
            # 면 배정 — 유리·리어·프론트·윈드실드가 맡는 일 (`compose.FaceSpec`).
            # `auto`는 로고·글자가 있으면 그것, 없으면 크롭으로 물러난다.
            "faces": {"window": "auto", "rear_window": "auto", "rear": "auto",
                      "front": "auto", "windshield": "auto"}}


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
    live, force = _absorb(st)
    _adopt(st, live, force)
    cast(st)
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
        # 덤프를 **우리 자리로 들인다** — 구성기는 미디어명 하나로 기하를 찾는다
        # (`game.fsgeom.for_car`). 편집기가 제 작업 폴더에 뜬 것을 그대로 두면
        # 꾸밈이 메시를 못 보고 마스크 노선으로 물러난다.
        if st.state.get("media"):
            from ...game import fsgeom

            fsgeom.adopt(st.state["media"], geometry)
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


def _absorb(st: Studio) -> tuple[set[int], set[tuple[str, str]]]:
    """편집기에서 **그룹째 옮긴 몫**을 배치 수치로 접어 넣는다.

    구성기가 구운 그룹은 변환이 항등으로 서므로, 항등이 아닌 것은 전부 사람이
    민 것이다. 그 변환을 배치 위에 곱하고 다시 분해하면 같은 자리를 배치
    수치로 표현한 것이 나온다 — 그 다음 굽기부터는 조리법이 그 자리를 안다.

    **덩어리 안을 손댄 것**은 그 반대로 간다: `FS:decal-<n>` 그룹의 장수가 구울 때
    적어 둔 값(`state.baked`)과 다르면 — 레이어를 지웠거나 더했다 — 그 면의 그
    도안을 조리법에서 빼고 덩어리를 `_adopt`에 넘긴다(둘째 반환값). 조리법이 도안
    파일에서 다시 지으면 지운 레이어가 도로 올라오기 때문이다.

    첫째 반환값은 **아직 조리법이 짓는 덩어리 번호**다 — 문서의 `decal-<n>`은 구울
    때의 번호라, 여기서 뺀 도안 때문에 번호를 다시 매기면 남은 도안의 덩어리가
    남의 번호로 읽혀 두 겹으로 실린다. 꾸밈이 켜진 판에서는 차 전체 구성의 변주
    덩어리(조리법 도안 뒤 번호)도 장수가 그대로면 여기 든다."""
    chunks = project.section_chunks(st.doc)
    counts = _chunk_counts(st.doc)
    baked = st.state.get("baked") or {}
    st.state["foreign"] = {
        name: entry["foreign"] for name, entry in chunks.items()
        if entry["foreign"]}
    kept: list = []
    live: set[int] = set()
    force: set[tuple[str, str]] = set()
    recipe = set(decal_numbers(st.designs))      # 구울 때의 번호 (문서와 같은 셈)
    for d, n in zip(st.designs, decal_numbers(st.designs)):
        entry = chunks.get(d.get("surface") or "")
        if entry is None:
            kept.append(d)
            live.add(n)
            continue
        # 편집기에서 그 그룹을 **지웠으면** 조리법에서도 뺀다 — 안 그러면 다음
        # 굽기에 도로 올라온다 (`FS:decal-<n>` · `-fit`·`-top` 따위 전부).
        mine = [k for k in entry["chunks"]
                if k == f"{DECAL}{n}" or k.startswith(f"{DECAL}{n}-")]
        if not mine:
            st.notes.append(msg("{surface}: 편집기에서 지운 도안을 조리법에서 뺐다 — {name}",
                                surface=d["surface"], name=Path(d["plan"]).name))
            continue
        # 그룹 안에서 지웠거나 더했으면 — 지금 있는 그대로가 도안이다
        was = (baked.get(d["surface"]) or {})
        edited = [k for k in mine
                  if k in was and counts.get(d["surface"], {}).get(k) != was[k]]
        if edited:
            for k in edited:
                force.add((d["surface"], project.CHUNK_PREFIX + k))
            now = sum(counts.get(d["surface"], {}).get(k, 0) for k in edited)
            st.notes.append(msg(
                "{surface}: 편집기에서 손댄 도안을 지금 있는 그대로 받는다 — "
                "{name} ({was:,}장 → {now:,}장)",
                surface=d["surface"], name=Path(d["plan"]).name,
                was=sum(was[k] for k in edited), now=now))
            continue
        kept.append(d)
        live.add(n)
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
    # **차 전체 구성의 변주** (`compose.whole` — 리어·유리의 얼굴 크롭·상반신)도
    # `FS:decal-<n>` 덩어리로 서는데 그 번호는 조리법 도안 뒤다. 꾸밈이 켜진 판에서
    # 장수가 구울 때와 같으면 구성기가 다시 짓는 몫이라 건너뛰고, 안에서 손댔으면
    # 도안처럼 지금 그대로 받는다. 꺼진 판(또는 조리법 없는 판)에서는 아무도 다시
    # 안 지으니 `_adopt`가 받는다.
    if st.state.get("deco"):
        for surface, entry in chunks.items():
            for k in entry["chunks"]:
                m = DECAL_CHUNK.match(k)
                if not m or int(m.group(1)) in recipe:
                    continue
                was = (baked.get(surface) or {}).get(k)
                if was is not None and counts.get(surface, {}).get(k) == was:
                    live.add(int(m.group(1)))
                else:
                    force.add((surface, project.CHUNK_PREFIX + k))
    return live, force


def _chunk_counts(doc: dict) -> dict[str, dict[str, int]]:
    """면마다 `FS:` 덩어리의 **도형 장수** — 구울 때 적어 두고(`state.baked`) 다음에
    열 때 대조하면 그룹 안에서 지우거나 더한 것이 보인다. 숨김은 안 본다 (화면
    상태이지 내용이 아니다)."""
    def _n(node: dict) -> int:
        n = 0
        for kid in node.get("children") or []:
            if not isinstance(kid, dict):
                continue
            if kid.get("kind") == "group":
                n += _n(kid)
            elif kid.get("kind") == "shape":
                n += 1
        return n

    out: dict[str, dict[str, int]] = {}
    for node in (doc.get("root") or {}).get("children") or []:
        if not isinstance(node, dict) or not node.get("is_livery_section"):
            continue
        slot = int(node.get("livery_section_slot", -1))
        if not 0 <= slot < len(SLOTS):
            continue
        got: dict[str, int] = {}
        for kid in node.get("children") or []:
            if not isinstance(kid, dict) or kid.get("kind") != "group":
                continue
            name = str(kid.get("name") or "")
            if name.startswith(project.CHUNK_PREFIX):
                got[name[len(project.CHUNK_PREFIX):]] = _n(kid)
        if got:
            out[SLOTS[slot][0]] = got
    return out


def _adopt(st: Studio, live: set[int] | None = None,
           force: set[tuple[str, str]] | None = None) -> None:
    """조리법이 모르는 덩어리 — 가른 조각·사본·손으로 그린 것 — 를 **도안으로 받는다**.

    편집기에서 도안 그룹을 [선으로 가르기]로 가르면 조각은 `FS:` 머리를 잃는다.
    베낀 사본은 머리를 그대로 두고 이름 뒤에 " (Copy)"가 붙는다 — 그래서 구성기
    몫인지는 머리가 아니라 **조리법이 그것을 다시 지을 수 있느냐**가 가른다
    (`_claimed`) — 조리법 없이 열린 판(`.3so`만 옮겨 온 것)은 그 안의
    `FS:decal-…`이 곧 도안이다. 그 덩어리들이 지금
    차에 실린 그림인데 조리법은 모르니 꾸밈이 "올린 도안이 없다"로 섰다. 그래서
    **지금 프로젝트에 있는 것이 도안**이다: 면 아래 조리법 밖 덩어리마다 레이어를 면 유닛 그대로 도안 파일로 뜨고, 항등
    배치(x·y 0 · 캔버스 1유닛 = 면 1유닛 · 회전 0)로 조리법에 올린다. 배치·색을
    다시 계산하는 것은 없다 — 사람이 둔 자리가 그대로 도안의 자리다. 다음
    굽기부터 그 조각은 `FS:decal-<n>` 덩어리라 옮긴 자리도 되돌아온다 (`_absorb`).

    받는 단위는 면 아래 **최상위 덩어리 하나**다 — 그룹은 그룹째 도안 하나, 그룹
    밖 낱 도형은 면마다 한 도안으로 묶는다. 안 받는 것은 셋이다: 숨긴 그룹(화면
    상태이지 내용이 아니다), 래스터 로고(카탈로그 밖 그림), 안내선. 그것들은
    종전처럼 원본 노드째 남아 다시 구울 때 그대로 실린다 (`_restore_foreign`).

    `live`는 조리법이 아직 짓는 덩어리 번호(`_absorb`가 문서의 번호로 센 것),
    `force`는 이름이 구성기 문법이어도 **받아야 하는** (면, 그룹 이름) — 안에서
    손댄 덩어리다. 그것은 원래 도안(`origin`)을 기억해 좌우 대칭이 반대편의 옛
    사본을 갈아 끼울 수 있게 한다.
    """
    gu = _group_unit_of(st)
    if live is None:
        live = set(decal_numbers(st.designs))
    force = force or set()
    deco_on = bool(st.state.get("deco"))
    leftover: dict[str, list] = {}
    added = 0
    for node in (st.doc.get("root") or {}).get("children") or []:
        if not isinstance(node, dict) or node.get("kind") != "group":
            continue
        if not node.get("is_livery_section"):
            continue
        slot = int(node.get("livery_section_slot", -1))
        if not 0 <= slot < len(SLOTS):
            continue
        surface = SLOTS[slot][0]
        gm = mat_mul(IDENTITY, _node_matrix(node))
        loose: list[Layer] = []
        rest: list[dict] = []
        for kid in node.get("children") or []:
            if not isinstance(kid, dict):
                continue
            name = str(kid.get("name") or "")
            forced = (surface, name) in force
            if (kid.get("kind") == "group" and not forced
                    and name.startswith(project.CHUNK_PREFIX)
                    and _claimed(name[len(project.CHUNK_PREFIX):],
                                 live, deco_on)):
                continue                       # 구성기 몫 — 조리법이 이미 안다
            layers, keep = _peel(kid, gm)
            if keep is not None:
                rest.append(keep)
            if not layers:
                continue
            if kid.get("kind") == "group":
                _adopt_one(st, surface, name or msg("이름 없는 그룹"), layers, gu,
                           origin=_origin_of(st, name) if forced else None)
                added += 1
            else:
                loose += layers
        if loose:
            _adopt_one(st, surface, msg("낱 도형"), loose, gu)
            added += 1
        if rest:
            leftover[surface] = rest
    st.state["foreign"] = leftover
    if added:
        st.notes.append(msg("편집기의 덩어리 {n}개를 도안으로 받았다 — 지금 자리 그대로",
                            n=added))


def _node_matrix(node: dict):
    t = node.get("transform") or {}
    return transform_matrix(
        float(t.get("x", 0.0)), float(t.get("y", 0.0)),
        float(t.get("scale_x", 1.0)), float(t.get("scale_y", 1.0)),
        float(t.get("rotation", 0.0)), float(t.get("skew", 0.0)))


def _peel(node: dict, gm) -> tuple[list[Layer], dict | None]:
    """노드 하나 → (받을 레이어들, 못 받아 그대로 둘 노드).

    그룹은 변환을 접어 내려가며 벡터 도형을 레이어로 뜬다 (`project._layer_of`
    — 되읽기와 같은 식). 못 받는 것(숨긴 그룹·래스터·안내선)만 남긴 사본이
    둘째 값이다 — 받은 도형이 사본에도 남으면 두 겹으로 실린다."""
    kind = node.get("kind")
    if kind == "guide":
        return [], node
    if kind == "group":
        if node.get("visible") is False:
            return [], node
        m = mat_mul(gm, _node_matrix(node))
        layers: list[Layer] = []
        kept: list[dict] = []
        for kid in node.get("children") or []:
            if not isinstance(kid, dict):
                continue
            ls, keep = _peel(kid, m)
            layers += ls
            if keep is not None:
                kept.append(keep)
        if not kept:
            return layers, None
        rest = dict(node)
        rest["children"] = kept
        return layers, rest
    lay = project._layer_of(node, gm)
    if lay is None:
        return [], node
    return [lay], None


def _origin_of(st: Studio, chunk: str) -> str | None:
    """손댄 덩어리 `FS:decal-<n>…`가 **어느 도안에서 왔나** — 구울 때 적어 둔 표
    (`state.baked_plans`: 번호 → 도안 파일)로 되짚는다."""
    m = DECAL_CHUNK.match(chunk[len(project.CHUNK_PREFIX):]
                          if chunk.startswith(project.CHUNK_PREFIX) else chunk)
    if not m:
        return None
    plans = st.state.get("baked_plans") or {}
    return plans.get(m.group(1))


def _adopt_one(st: Studio, surface: str, name: str, layers: list[Layer],
               group_unit: float, *, origin: str | None = None) -> None:
    """레이어 묶음 하나를 도안 파일로 떠서 항등 배치로 조리법에 올린다.

    파일 이름은 **내용 서명**이다 — 같은 덩어리를 다시 열어도 같은 파일이라
    `state`를 몇 번 읽어도 늘지 않고, 구성기가 지우는 이름(`decal-*`)과도 안
    겹친다. 캔버스 1유닛 = 면 1유닛이 되게 스케일은 `1/group_unit`이다
    (`compose.ManualPlace` — 표시 스케일이 `scale × group_unit`)."""
    plan = LayerPlan(source_image=name, image_size=(0, 0), units_per_px=1.0,
                     layers=list(layers))
    sig = hashlib.sha1(json.dumps([asdict(l) for l in layers], sort_keys=True)
                       .encode("utf-8")).hexdigest()[:10]
    p = st.work / ADOPTED_DIR / f"{surface}-{sig}.json"
    if not p.is_file():
        p.parent.mkdir(parents=True, exist_ok=True)
        plan.save(p)
    d = {"plan": str(p), "surface": surface, "x": 0.0, "y": 0.0,
         "scale": round(1.0 / max(1e-9, group_unit), 6), "rot": 0.0,
         "mirror": False, "adopted": name}
    if origin:
        d["origin"] = origin
    st.designs.append(d)
    st.notes.append(msg("{surface}: 편집기의 '{name}' {n:,}장을 도안으로 받았다",
                        surface=surface, name=name, n=len(layers)))


def _group_unit_of(st: Studio) -> float:
    """구성기가 이 판에 쓸 `group_unit` — 차 이름을 **같은 문**으로 정한다
    (`auto.itasha.compose_config`: 조리법의 차, 없으면 면 탭 실측표의 차)."""
    from ...game import body as gbody
    from ..compose.boxes import _group_unit

    car = st.state.get("car") or gbody.tab_table().get("car")
    return _group_unit(car)


# ────────────────────────────── 역할표 ──────────────────────────────


def cast(st: Studio) -> None:
    """실린 덩어리마다 **역할**을 읽어 조리법에 싣는다 (`compose.cast`).

    배치마다 `role`(쓰는 값) · `role_auto`(추정) · `role_why` · `no_mirror` ·
    `pinned` · `layers` · `label`이 선다. 사람이 표에서 고친 것(`role_user`)은
    다시 열어도 그대로고, 나머지는 열 때마다 다시 읽는다 — 편집기에서 레이어를
    지우거나 가르면 장수가 바뀌어 역할도 바뀔 수 있다. 같은 도안 파일은 한 번만
    읽는다 (좌우에 같은 그림 = 같은 역할)."""
    from .. import compose

    memo: dict[str, compose.CastEntry] = {}
    for d in st.designs:
        key = str(Path(d["plan"]).resolve())
        ce = memo.get(key)
        if ce is None:
            try:
                ce = compose.cast_estimate(LayerPlan.load(Path(d["plan"])))
            except Exception as e:      # noqa: BLE001 — 못 읽는 도안은 굽기가 잡는다
                ce = compose.CastEntry("support", msg("못 읽었다: {e}", e=e), 0, 0, 0.0, 0.0)
            memo[key] = ce
        d["role_auto"] = ce.role
        d["role_why"] = ce.why
        d["layers"] = ce.layers
        d["label"] = d.get("adopted") or Path(d["plan"]).name
        if not d.get("role_user") or d.get("role") not in compose.CAST_ROLES:
            d["role"] = ce.role
            d.pop("role_user", None)
        role = d["role"]
        d["no_mirror"] = role in compose.NO_MIRROR_ROLES
        d["pinned"] = role == "pinned"


def act_roles(st: Studio, roles: dict[int, str]) -> str:
    """실린 그림 표에서 사람이 고른 역할 — `번호 → 역할` (번호는 `designs` 차례).

    `auto`(또는 추정과 같은 값)면 사람 손을 뗀다 — 다음에 열 때 다시 읽는다.
    다른 값이면 못 박는다 (`role_user`)."""
    from .. import compose

    changed: list[str] = []
    for i, role in sorted(roles.items()):
        if not 0 <= i < len(st.designs):
            raise ValueError(msg("역할 번호가 범위 밖이다 — {i} (실린 덩어리 {n}개)",
                                 i=i, n=len(st.designs)))
        d = st.designs[i]
        if role == "auto" or role == d.get("role_auto"):
            d.pop("role_user", None)
            d["role"] = d.get("role_auto") or "hero"
        elif role in compose.CAST_ROLES:
            d["role"] = role
            d["role_user"] = True
        else:
            raise ValueError(msg("모르는 역할: {role!r} (있는 것: {roles} · auto)",
                                 role=role, roles=", ".join(compose.CAST_ROLES)))
        d["no_mirror"] = d["role"] in compose.NO_MIRROR_ROLES
        d["pinned"] = d["role"] == "pinned"
        changed.append(f"{d.get('label') or Path(d['plan']).name}={compose.CAST_LABELS[d['role']]}")
    if not any(d.get("role") in ("hero", "support") for d in st.designs):
        st.notes.append(msg("그림(주역·보조)이 하나도 없다 — 로고·글자만으로는 "
                            "구도의 뿌리가 없어 가장 큰 덩어리를 뿌리로 쓴다"))
    return msg("역할 {what}", what=" · ".join(changed) if changed else msg("그대로"))


# ────────────────────────────── 굽기 ──────────────────────────────


def _main_index(st: Studio) -> int:
    """주역 = 역할이 `hero`인 것 중 **면에서 가장 크게 덮는** 도안 (색·모티프·
    베이스 도색을 이게 정한다). 주역이 없으면 그림(보조) 중에서, 그것도 없으면
    전부에서 고른다 — 로고만 올린 판도 서야 한다."""
    from .. import compose

    cat = Catalog(default_catalog_path())
    pool = [i for i, d in enumerate(st.designs) if d.get("role") == "hero"] \
        or [i for i, d in enumerate(st.designs) if d.get("role") == "support"] \
        or list(range(len(st.designs)))
    best, area = pool[0] if pool else 0, -1.0
    for i in pool:
        d = st.designs[i]
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

    role = str(d.get("role") or "hero")
    return compose.ManualPlace(
        plan=Path(d["plan"]), surface=d["surface"], x=float(d.get("x", 0.0)),
        y=float(d.get("y", 0.0)), scale=float(d.get("scale", 0.25)),
        rot=float(d.get("rot", 0.0)), mirror=bool(d.get("mirror")),
        role=role, no_mirror=bool(d.get("no_mirror", role in compose.NO_MIRROR_ROLES)),
        pinned=bool(d.get("pinned", role == "pinned")))


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
        logos=st.state.get("logos") or None,
        faces=st.state.get("faces") or None,
        mirror=False, preview=False, log=log)
    cfg_path = Path(cfg.path)
    doc, stats = _write_project(st, cfg_path)
    st.doc = doc
    # 구운 덩어리의 장수와 번호 → 도안 — 다음에 열 때 그룹 안에서 손댄 것을
    # 알아보는 자다 (`_absorb`).
    st.state["baked"] = _chunk_counts(doc)
    st.state["baked_plans"] = {
        str(n): d["plan"] for d, n in zip(st.designs, decal_numbers(st.designs))}
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
    """도안으로 못 받은 노드(숨긴 그룹·래스터·안내선)를 면 그룹에 도로 얹는다 (맨 위)."""
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
    hush_deco(st)
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


def hush_deco(st: Studio) -> None:
    """**꾸밈은 부를 때만 자란다** (사용자 지시 2026-09-01).

    도안을 건드리는 명령(도안 올리기·좌우 대칭·자동 자리)은 조리법의 `deco`를
    끈다. 안 그러면 [자동 꾸밈]을 한 번 누른 뒤로는 도안 하나 올릴 때마다 **옛
    설정으로 짠 꾸밈**(못 박은 계열·옛 캐릭터 이름 글자)이 새 판에 따라 들어온다
    — 시킨 적 없는 일이다. 고른 계열·이름은 조리법에 그대로 남으니 [자동 꾸밈]을
    다시 누르면 그 설정으로 다시 자란다."""
    if not st.state.get("deco"):
        return
    st.state["deco"] = False
    st.notes.append(msg("꾸밈은 부를 때만 자란다 — 이번 판은 도안만 올린다. "
                        "다시 자라게 하려면 [자동 꾸밈]을 누르세요"))


def act_auto_place(st: Studio, surface: str | None,
                   groups: list | None = None) -> str:
    """고른 도안(또는 이 면·전부)을 자동 자리로 되돌린다."""
    hit = _targets(st, surface, groups, what=msg("자동 배치"), fallback_all=True)
    for d in hit:
        _auto_place_one(st, d)
    hush_deco(st)
    return msg("도안 {n}장을 자동 자리로", n=len(hit))


def act_decoration(st: Studio, on: bool) -> str:
    """꾸밈(관통 띠·산포 모티프·지붕 블랙아웃)을 짜 넣거나 뺀다.

    **부르기 전에는 안 짠다** (사용자 지시 2026-08-27 "자동 적용 금지") — 도안을
    올리는 것과 꾸밈이 자라는 것은 별개의 일이고, 후자는 사람이 시킬 때만 한다.
    켠 것은 이 판에만 실린다 — 도안을 건드리는 다음 명령이 도로 끈다
    (`hush_deco`)."""
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
                 drop_text: bool = False, roles: dict[int, str] | None = None,
                 logos: dict | None = None, symmetry: bool | None = None,
                 faces: list | dict | None = None) -> str:
    """**자동 꾸밈 창** — 구성 계열·모티프 계열·바탕 도색·글자·역할표·로고·좌우·면 배정을 한 번에 받아 켠다.

    편집기의 [Auto Decoration...] 대화상자가 부른다 (`flsedit decorate`). 준 것만
    바꾼다 — `composition`/`motif`는 "auto"면 자동(None), `paint`는 #RRGGBB,
    `auto_paint`면 도안에서 고르고, `text`는 스펙 열쇠들(`main`이 비면 끈다),
    `drop_text`면 글자를 뺀다. `roles`는 실린 그림 표에서 사람이 고른 역할
    (`act_roles`), `logos`는 워터마크·로고 이미지·자리(`act_logos`), `symmetry`는
    "한쪽에만 있으면 반대편에"(`act_symmetry`), `faces`는 면 배정(`act_faces`).
    마지막에 꾸밈을 켠다 — 이 창의 뜻이 그것이다."""
    from .. import compose

    said: list[str] = []
    if roles:
        said.append(act_roles(st, roles))
    if logos is not None:
        said.append(act_logos(st, **logos))
    if faces is not None:
        said.append(act_faces(st, faces))
    if symmetry is not None:
        st.state["symmetry"] = bool(symmetry)
    said.append(act_symmetry(st, bool(st.state.get("symmetry", True))))
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


def act_logos(st: Studio, *, watermark: bool | None = None,
              images: list | None = None, placement: str | None = None) -> str:
    """로고 옵션 — 내장 워터마크 · 사용자 로고 이미지(0~N) · 자리. 준 것만 바꾼다.

    이미지는 **지금** 벡터화한다 (`compose.vectorize_logo` — 셀 노선, 110장 상한,
    내용 서명 캐시 `<작업 폴더>/logos/`) — 굽기 안에서 하면 오류가 상태줄에
    안 뜬다. 못 굽는 이미지는 빼고 말한다."""
    from .. import compose

    cur = dict(st.state.get("logos") or {"watermark": True, "images": [], "placement": "auto"})
    if watermark is not None:
        cur["watermark"] = bool(watermark)
    if placement is not None:
        cur["placement"] = placement
    if images is not None:
        had = {str(it.get("image")): it for it in (cur.get("images") or [])
               if isinstance(it, dict)}
        got: list[dict] = []
        for img in images:
            key = str(Path(str(img)).resolve())
            it = had.get(key) or had.get(str(img)) or {"image": key, "plan": None}
            if not (it.get("plan") and Path(it["plan"]).is_file()):
                try:
                    it["plan"] = str(compose.vectorize_logo(
                        key, st.work / "logos", log=lambda t: st.notes.append(t)))
                except (ValueError, OSError, RuntimeError, SystemExit) as e:
                    st.notes.append(msg("로고를 못 굽는다 — {name}: {e}",
                                        name=Path(key).name, e=e))
                    continue
            got.append({"image": key, "plan": it["plan"]})
        cur["images"] = got
    spec = compose.LogoSpec.from_dict(cur)          # 값 검사 (모르는 자리면 ValueError)
    st.state["logos"] = spec.to_dict()
    return msg("로고: 워터마크 {wm} · 이미지 {n}장 · 자리 {place}",
               wm=msg("켬") if spec.watermark else msg("끔"), n=len(spec.images),
               place=spec.placement)


def act_faces(st: Studio, faces: list | dict) -> str:
    """면 배정 — 도어 유리·뒷유리·리어·프론트·윈드실드가 맡는 일 (`compose.FaceSpec`).

    `faces`는 CLI 꼴(`["window=continue", "rear=crop"]`)이거나 dict다. 준 면만
    바꾸고 나머지는 그대로다. 값 검사는 스펙이 한다 (모르는 모드면 ValueError)."""
    from .. import compose

    cur = dict(st.state.get("faces") or {})
    if isinstance(faces, dict):
        got = {compose.FACE_OF.get(str(k), str(k)): str(v) for k, v in faces.items()}
    else:
        parsed = compose.FaceSpec.from_args(list(faces))
        got = {} if parsed is None else {
            f: getattr(parsed, f) for f in compose.FACE_NAMES
            if any(compose.FACE_OF.get(str(it).partition("=")[0].strip(),
                                       str(it).partition("=")[0].strip()) == f
                   for it in faces)}
    spec = compose.FaceSpec.from_dict(cur | got)
    st.state["faces"] = spec.to_dict()
    return msg("면 배정: {what}", what=spec.describe())


def act_symmetry(st: Studio, on: bool) -> str:
    """**한쪽에만 있으면 반대편에** — 옆면·도어 유리 한 쌍에서 한쪽만 도안이 있으면
    반대편에 세운다 (그림은 거울, 로고·글자는 자리만 거울 — `_mirror_one`).

    우리가 세운 사본은 `symmetry` 표시를 단다. 끄면 그 사본만 걷는다 — 사람이 올린
    것은 안 건드린다. 양쪽에 다 있으면(우리 사본이든 사람 것이든) 아무것도 안 한다
    — 사람이 반대편을 지웠으면 다음 판에 다시 세운다."""
    st.state["symmetry"] = bool(on)
    if not on:
        before = len(st.designs)
        st.state["designs"] = [d for d in st.designs if not d.get("symmetry")]
        gone = before - len(st.designs)
        return msg("좌우: 한쪽만 둔다") + (msg(" (세웠던 사본 {n}개를 걷는다)", n=gone)
                                        if gone else "")
    done = _symmetrize(st)
    return msg("좌우: 한쪽에만 있으면 반대편에") + (
        msg(" — {what}", what=" · ".join(done)) if done else "")


def _symmetrize(st: Studio) -> list[str]:
    maps = _maps(st)
    done: list[str] = []
    for a, b in (("side_left", "side_right"), ("window_left", "window_right")):
        on_a = [d for d in st.designs if d["surface"] == a]
        on_b = [d for d in st.designs if d["surface"] == b]
        if bool(on_a) == bool(on_b):
            continue
        src, dst = (on_a, b) if on_a else (on_b, a)
        for d in list(src):
            new = _mirror_one(st, d, dst, maps)
            if new is None:
                continue
            new["symmetry"] = True
            st.designs.append(new)
            done.append(f"{d['surface']} → {dst}"
                        + (msg(" (자리만)") if d.get("no_mirror") else ""))
    return done


def _mirror_one(st: Studio, d: dict, dst: str, maps: dict) -> dict | None:
    """배치 하나의 반대편 사본 — 그림은 거울, 로고·글자는 **읽는 방향 그대로**
    (`compose.reseat_place`). 면 지도가 없으면 None. 역할표는 그대로 물려받는다."""
    from .. import compose

    sm, dm = maps.get(d["surface"]), maps.get(dst)
    if sm is None or dm is None:
        st.notes.append(msg("{surface}: 면 지도가 없다 — 대칭을 건너뛴다", surface=dst))
        return None
    mp = (compose.reseat_place if d.get("no_mirror") else compose.mirror_place)(
        _manual(d), sm, dm, dst)
    new = {"plan": d["plan"], "surface": dst, "x": mp.x, "y": mp.y,
           "scale": mp.scale, "rot": mp.rot, "mirror": mp.mirror}
    if d.get("origin"):
        new["origin"] = d["origin"]
    for k in ("role", "role_auto", "role_why", "role_user", "label",
              "layers", "no_mirror", "pinned", "adopted"):
        if k in d:
            new[k] = d[k]
    return new


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


MIRROR_PAIRS = {"side_left": "side_right", "side_right": "side_left",
                "window_left": "window_right", "window_right": "window_left"}


def _mirror_sources(st: Studio, surface: str | None, groups: list | None) -> list[dict]:
    """좌우 대칭의 **원본** 도안 — 고른 그룹 > 보고 있는 면 > 도안이 있는 쪽.

    **한 방향만 간다.** 한 쌍의 양쪽이 다 후보에 들면 보고 있는 면이 가르고, 그래도
    못 가르면 멈춘다 — 양방향으로 돌면 방금 고친 쪽이 옛 반대편의 거울로
    되돌아간다. 편집기가 넘기는 그룹 이름(`decal-1`)은 좌우가 같은 이름이라
    그룹만으로는 한쪽을 못 고른다."""
    pairs = MIRROR_PAIRS
    if not st.designs:
        raise ValueError(msg("올린 도안이 없다 — [Load Design into Section]으로 넣으세요"))
    picked = designs_of_groups(st, groups or [])
    cands = [d for d in picked if d["surface"] in pairs]
    if picked and not cands:
        raise ValueError(msg("고른 그룹은 옆면·도어 유리의 도안이 아니다 — "
                             "좌우 대칭은 그 면에서만 선다"))
    if not cands and surface in pairs:
        cands = [d for d in st.designs if d["surface"] == surface]
    if not cands:
        # 보고 있는 면이 원본이 못 된다 (옆면이 아니거나 비어 있다) — 편집기는 늘
        # 지금 열린 구획을 넘기는데 창이 막 떴을 때 그것은 대개 Front다.
        cands = [d for d in st.designs if d["surface"] in pairs]
        if not cands:
            raise ValueError(msg("좌우 대칭은 옆면·도어 유리에서만 선다 — "
                                 "그 면에 올린 도안이 없다"))
    have = {d["surface"] for d in cands}
    both = sorted(s for s in have if pairs[s] in have)
    if both:
        if surface not in both:
            raise ValueError(
                msg("양쪽에 다 도안이 있어 어느 쪽이 원본인지 못 정했다 ({sides}) — "
                    "원본 쪽 구획을 열고 누르거나, 레이어 나무에서 그 도안 "
                    "그룹(FS:decal-…)을 고르세요", sides=" · ".join(both))
                + (msg(" (지금 면: {surface})", surface=surface) if surface else ""))
        cands = [d for d in cands if d["surface"] not in both or d["surface"] == surface]
    srcs_of = sorted({d["surface"] for d in cands})
    if surface not in srcs_of:
        st.notes.append(msg("{surface}에서는 원본을 못 정한다 — 도안이 있는 쪽({sides})을 "
                            "반대편에 세운다", surface=surface or msg("이 면"),
                            sides=" · ".join(srcs_of)))
    return cands


def _same_design(a: dict, b: dict) -> bool:
    """같은 도안인가 — 편집기에서 손대 받은 것은 **원래 도안**(`origin`)으로 센다.
    왼쪽에서 레이어를 지우고 좌우 대칭하면 오른쪽의 옛 사본이 갈려야 한다."""
    return (str(a.get("origin") or a["plan"]) == str(b.get("origin") or b["plan"])
            or str(a["plan"]) == str(b["plan"]))


def act_mirror(st: Studio, surface: str | None,
               groups: list | None = None) -> str:
    """좌우 대칭 — 옆면·도어 유리의 도안을 반대편에 거울로 세운다 (`_mirror_sources`).

    반대편에 같은 도안이 이미 있으면 **그것을 갈아 끼운다** (두 벌이 되지 않는다)."""
    from .. import compose

    srcs = _mirror_sources(st, surface, groups)
    maps = _maps(st)
    done: list[str] = []
    for d in list(srcs):
        dst = MIRROR_PAIRS[d["surface"]]
        # 로고·글자는 **절대 미러하지 않는다** (사용자 결정 2026-09-02) — 거울에
        # 비친 글자는 읽히지 않는다. 반대편의 거울 자리에 읽는 방향 그대로 앉힌다.
        new = _mirror_one(st, d, dst, maps)
        if new is None:
            continue
        if d.get("no_mirror"):
            st.notes.append(msg("{surface}: '{name}'은(는) {role}라 자리만 거울로 앉힌다",
                                surface=dst,
                                name=d.get("label") or Path(d["plan"]).name,
                                role=compose.CAST_LABELS.get(d.get("role"), d.get("role"))))
        st.state["designs"] = [o for o in st.designs
                               if not (o["surface"] == dst
                                       and _same_design(o, d))]
        st.designs.append(new)
        done.append(f"{d['surface']} → {dst}")
    if not done:
        raise ValueError(msg("대칭할 면을 못 찾았다"))
    hush_deco(st)
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
    for p in st.work.glob("logos-*.json"):
        p.unlink(missing_ok=True)
    # 받은 도안 중 조리법이 더는 안 가리키는 것 — 가른 조각을 다시 갈랐거나 지운 것
    keep = {Path(d["plan"]).resolve() for d in st.designs}
    for p in (st.work / ADOPTED_DIR).glob("*.json"):
        if p.resolve() not in keep:
            p.unlink(missing_ok=True)
