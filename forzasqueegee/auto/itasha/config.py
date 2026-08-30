"""구성 파일 — 규약·검증·짓기. 게임을 건드리기 전에 여기서 다 걸러 낸다.

`itasha.json`을 읽어 `Config`로 세우고(`load_config`), 면 예산·중복 장수·이름
규칙을 **게임 밖에서** 다 본다(`_check`). 구성을 새로 짓는 두 길도 여기다 —
프리셋(`make_config`)과 설계(`compose_config` → `engine.compose.build`)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from ...engine.model import LayerPlan
from ...i18n import msg
from ...paths import run_label
from ...game import body


PRESET: dict[str, dict[str, float]] = {
    "side_left":    {"x": 0.0, "y": 0.0, "scale": 0.25, "rot": 0.0},
    "side_right":   {"x": 0.0, "y": 0.0, "scale": 0.25, "rot": 0.0},
    "top":          {"x": 0.0, "y": 0.0, "scale": 0.30, "rot": 0.0},
    "front":        {"x": 0.0, "y": 0.0, "scale": 0.15, "rot": 0.0},
    "rear":         {"x": 0.0, "y": 0.0, "scale": 0.15, "rot": 0.0},
    "spoiler":      {"x": 0.0, "y": 0.0, "scale": 0.10, "rot": 0.0},
    "windshield":   {"x": 0.0, "y": 0.0, "scale": 0.15, "rot": 0.0},
    "rear_window":  {"x": 0.0, "y": 0.0, "scale": 0.15, "rot": 0.0},
    "sunroof":      {"x": 0.0, "y": 0.0, "scale": 0.10, "rot": 0.0},
    "window_left":  {"x": 0.0, "y": 0.0, "scale": 0.12, "rot": 0.0},
    "window_right": {"x": 0.0, "y": 0.0, "scale": 0.12, "rot": 0.0},
}


DEFAULT_PLACE = {"x": 0.0, "y": 0.0, "scale": 0.25, "rot": 0.0}


PRESET_ITASHA = ("side_left", "side_right")


NAME_OK = re.compile(r"^[A-Za-z0-9 _.\-]{1,30}$")


@dataclass
class GroupLoad:
    """면에 **먼저 깔리는 보조 그룹** (꾸밈 그룹 — `engine.compose`의 그룹 분리).

    도안 그룹보다 아래에 깔린다. 준비·불러오기 규약은 주 그룹과 같다 (장수가
    신원이다).
    """

    plan: Path
    group: str
    layers: int
    x: float = 0.0
    y: float = 0.0
    scale: float = 1.0
    rot: float = 0.0
    mirror: bool = False


@dataclass
class Placement:
    plan: Path
    surface: str
    group: str
    layers: int
    x: float = 0.0
    y: float = 0.0
    scale: float = 1.0
    rot: float = 0.0
    mirror: bool = False
    copy_from: str | None = None      # 이 면 대신 반대편 레이어를 붙여넣는다
    # **그 차의** 면 탭 인덱스 (`_resolve_tabs`가 못 박는다). 탭표는 잰 차 한
    # 대의 것이라 유효 면이 다른 차에서 어긋난다.
    tab: int | None = None
    # 구성 설계가 노린 **면 유닛 상자** (u0,v0,u1,v1). 있으면 배치를 화면으로 재서
    # 이 상자에 맞추고(오토핏) 결과를 검증에 적는다 (`engine.compose`가 넣는다).
    target: tuple[float, float, float, float] | None = None
    # 오토핏 허용 여부 — 띠·데코를 일부러 면 밖으로 흘리는 합성은 발자국이 목표와
    # 구조적으로 어긋나 보정이 상한에 걸리므로 구성기가 끈다 (`engine.compose`).
    fit: bool = True
    # 게임 텍스트 도구로 **면에 직접** 넣을 글자 — 그룹 뒤(위)에 넣는다. 자동
    # 구성은 글자를 안 내지만 (`engine.compose` — 어휘에 글자가 없다) 손으로 적은
    # 구성 파일은 받는다. plan 없이 글자만 있으면 글자 전용 배치다.
    texts: list[dict] = field(default_factory=list)
    # 도형 위저드로 면에 직접 놓을 띠·도형 (`engine.compose.flow_shapes`).
    # 소형 그룹 주입은 표 식별이 모호해서 (2026-08-18 실측) 이 길로 간다.
    shapes: list[dict] = field(default_factory=list)
    # 주 그룹보다 **먼저 깔리는** 보조 그룹들 (꾸밈 그룹 — 그룹 분리 문법).
    pre_groups: list[GroupLoad] = field(default_factory=list)
    # **사람이 편집기에서 앉힌 도안들** (`engine.fls.studio`). 백드롭 뒤,
    # 덮개 도형 앞에 목록 순서대로 올라간다 — 뒤가 위다. 한 면에 도안 하나라는
    # 제약은 게임이 아니라 자동 구성의 것이었다: 면은 그룹을 몇 개든 받는다
    # (꾸밈 그룹이 이미 그 증거다). `plan`을 쓰는 자동 경로와 배타적이 아니라
    # 서로 겹칠 수 있다 (유리에 머리 조각 + 사람이 올린 사인).
    groups: list[GroupLoad] = field(default_factory=list)
    # 주 그룹 **위에** 위저드로 놓는 도형 — `shapes`는 맨 아래, 이것은 글자
    # 직전이다.
    post_shapes: list[dict] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.surface}"

    @property
    def text_layers(self) -> int:
        """게임 텍스트 도구가 이 면에 더할 장수."""
        return sum(int(t.get("layers") or 0) for t in self.texts)

    @property
    def group_layers(self) -> int:
        """이 면에 불러올 그룹 장수 합 (보조·손 배치 그룹 포함)."""
        return (self.layers + sum(g.layers for g in self.pre_groups)
                + sum(g.layers for g in self.groups))

    def all_groups(self) -> list[GroupLoad]:
        """이 면이 불러오는 그룹 전부 — **놓는 순서대로**."""
        out = list(self.pre_groups)
        if self.layers:
            out.append(GroupLoad(plan=self.plan, group=self.group,
                                 layers=self.layers, x=self.x, y=self.y,
                                 scale=self.scale, rot=self.rot,
                                 mirror=self.mirror))
        return out + list(self.groups)


@dataclass
class Config:
    path: Path
    placements: list[Placement] = field(default_factory=list)
    apply: bool = True
    car: str | None = None
    # 설치 파일 차량 이름 (`media/Cars/<이것>.zip`). 있으면 면 지도를 **이름
    # 매칭 없이** 그 차로 못 박는다 — 편집기에서 차를 골라 지은 구성이 그렇다.
    media: str | None = None
    # 베이스 도색 HSB (`engine.compose.base_paint`가 고른다). 있으면 배치 전에
    # `auto.paintcar`가 자동차 도색 메뉴에서 차 전체를 이 색으로 칠한다.
    paint: tuple[float, float, float] | None = None
    # **이 차의 면 탭 구성** (설치 파일 · `game.cars.tabs_of`). 면 이름 검증·탭
    # 번호·상한이 전부 이걸 본다 — 실측표 한 장으로는 스포일러·선루프가 있는
    # 차를 못 읽는다. 설치본을 못 찾으면 빈 목록이고 실측표로 물러난다.
    tabs: list[str] = field(default_factory=list)

    @property
    def progress_path(self) -> Path:
        return self.path.with_suffix(".progress.json")

    def groups(self) -> dict[str, "Placement | GroupLoad"]:
        """준비해야 하는 그룹 (**플랜 경로** → 대표). 글자 전용 배치는 그룹이 없다.

        보조 그룹(`pre_groups`)도 준비 대상이다 — 규약(plan·group·layers)이 같다.
        열쇠가 이름이 아닌 이유: 표시 이름은 저장 슬롯 제약(ASCII 30자)으로
        잘려 **서로 다른 그룹이 같은 이름**이 된다 (실측: `…-decal-1-window_left/
        right`가 둘 다 `…-decal-1-window`로 잘려 한쪽만 준비되고 다른 쪽 장수가
        그쪽으로 덮였다). 게임은 어차피 장수로 찾으므로 이름은 표시용이다.
        """
        out: dict[str, Placement | GroupLoad] = {}
        for p in self.placements:
            for g in p.pre_groups:
                out.setdefault(str(g.plan), g)
            if p.copy_from is None and p.layers > 0:
                out.setdefault(str(p.plan), p)
            for g in p.groups:
                out.setdefault(str(g.plan), g)
        return out


def ascii_name(text: str) -> str:
    """플랜 폴더 이름에서 ASCII 저장 슬롯 이름을 만든다 (게임 텍스트 필드 제약)."""
    out = re.sub(r"[^A-Za-z0-9 _.\-]", "", text).strip()
    return (out or "Itasha")[:30]


def load_config(path: Path, preset: bool = True,
                media: str | None = None) -> Config:
    """itasha.json을 읽고 **전부 검증한다** — 게임을 건드리기 전에 다 걸러 낸다.

    `media`를 주면 구성이 적어 둔 차를 덮어 못 박는다 (CLI `--media`) — 탭
    해석(`_resolve_tabs`)이 그 차의 유효 면 순서를 쓰므로 **읽는 즉시** 덮어야
    한다.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    paint = (raw.get("paint") or {}).get("hsb")
    cfg = Config(path=path, apply=bool(raw.get("apply", True)),
                 car=raw.get("car"), media=media or raw.get("media"),
                 paint=tuple(float(v) for v in paint) if paint else None)
    items = raw.get("placements")
    if not items:
        raise ValueError(msg("{name}: placements가 비어 있다", name=path.name))
    # **면 이름 검증은 그 차의 목록으로** 한다 — 실측표는 잰 차 한 대의 것이라
    # 스포일러·선루프가 있는 차의 면을 "모르는 이름"으로 죽인다 (실측: 인테그라
    # 23의 sunroof 배치가 읽는 자리에서 죽었다).
    names = _car_tabs(cfg)
    cfg.tabs = list(names)
    for i, it in enumerate(items):
        cfg.placements.append(_placement(path, i, it, preset, names))
    _resolve_tabs(cfg, names)
    _check(cfg, names)
    return cfg


def _car_tabs(cfg: Config) -> list[str]:
    """구성이 지어진 차의 **면 탭 구성** (설치 파일 예측). 못 고르면 빈 목록.

    떠 둔 차량 색인이 먼저다 (`game.cars` — [차량 정보 동기화]가 뜬다). 색인에
    없는 차면 그 차 zip을 직접 읽는다.

    차를 고르는 것은 **면 지도와 같은 문**이다 (`compose.carfiles_pick`) — 탭
    번호가 딴 차에서 나오면 배치가 통째로 다른 면에 앉는다.
    """
    try:
        from ...engine import compose
        from ...game import cars as gcars

        media = cfg.media or compose.carfiles_pick(cfg.car)
        return gcars.tabs_of(media) if media else []
    except Exception:                              # 설치본이 없어도 구성은 선다
        return []


def _resolve_tabs(cfg: Config, names: list[str] | None = None) -> None:
    """면 이름 → **그 차의** 탭 인덱스를 못 박는다.

    탭표(`catalog/body_tabs.json`)는 잰 차 한 대의 것이라 유효 면이 다른 차에서
    어긋난다 — **스포일러가 있는 차는 `rear` 뒤가 통째로 한 칸 밀린다** (CRX
    뮤겐 실측: 탭표의 window_left 7이 그 차에서는 rear_window다. 그대로 가면
    유리 모티프가 뒷유리에 찍힌다). 설치 파일이 그 차의 유효 면 순서를 그대로
    주므로(`carfiles.tab_names` — 정식 순서 = 인게임 탭 구성) 그것을 먼저 쓰고,
    설치본이 없거나 차를 못 고르면 탭표로 물러난다.
    """
    names = names if names is not None else _car_tabs(cfg)
    for p in cfg.placements:
        try:
            p.tab = body.surface_index(p.surface, names)
        except ValueError as e:
            raise ValueError(msg("{err} (구성을 그 차로 다시 지을 것)",
                                 err=e)) from None


def _placement(cfg_path: Path, i: int, it: dict, preset: bool,
               names: list[str] | None = None) -> Placement:
    where = f"placements[{i}]"
    surface = it.get("surface")
    if not surface:
        raise ValueError(msg("{where}: surface가 없다", where=where))
    body.surface_index(surface, names)          # 이 차에 없는 면이면 ValueError
    copy_from = it.get("copy_from")
    if copy_from:
        body.surface_index(copy_from, names)
        return Placement(plan=Path(), surface=surface, group="", layers=0,
                         copy_from=copy_from)
    if not it.get("plan"):
        # 그룹 없는 배치 — 도형만(리어·프론트의 띠·모티프), 글자만, 또는 **손
        # 배치 그룹만**. `post_shapes`도 받는다: 제 글자 밑에 깔릴 판은 그룹이
        # 없는 면에도 온다 (여기서 흘리면 판만 조용히 사라진다).
        if (it.get("text") or it.get("shapes") or it.get("post_shapes")
                or it.get("groups") or it.get("pre_groups")):
            return Placement(plan=Path(), surface=surface, group="", layers=0,
                             texts=_as_list(it.get("text")),
                             shapes=list(it.get("shapes") or []),
                             pre_groups=_groups(cfg_path, where, it, "pre_groups"),
                             groups=_groups(cfg_path, where, it, "groups"),
                             fit=bool(it.get("fit", True)),
                             post_shapes=list(it.get("post_shapes") or []))
        raise ValueError(msg("{where}: plan이 없다 (copy_from·text·shapes·groups도 없다)",
                             where=where))
    plan_path = (cfg_path.parent / it["plan"]).resolve()
    if not plan_path.exists():
        raise ValueError(msg("{where}: 플랜이 없다 — {path}",
                             where=where, path=plan_path))
    layers = len(LayerPlan.load(plan_path).layers)
    # 슬롯 이름은 **폴더 + 파일 이름**이다. 폴더만 쓰면 한 폴더에 도안이 여러 개인
    # 구성(측면 합성 + 리어 글자)에서 이름이 겹쳐 게임 슬롯이 부딪친다.
    group = it.get("group") or ascii_name(run_label(plan_path))
    if not NAME_OK.match(group):
        raise ValueError(msg("{where}: 저장 슬롯 이름은 ASCII 30자 이내여야 한다 "
                             "(받은 것: {got!r})", where=where, got=group))
    base = dict(DEFAULT_PLACE)
    if preset:
        base.update(PRESET.get(surface, {}))
    tgt = it.get("target")
    return Placement(
        plan=plan_path, surface=surface, group=group, layers=layers,
        x=float(it.get("x", base["x"])), y=float(it.get("y", base["y"])),
        scale=float(it.get("scale", base["scale"])),
        rot=float(it.get("rot", base["rot"])),
        mirror=bool(it.get("mirror", False)),
        fit=bool(it.get("fit", True)),
        target=tuple(float(v) for v in tgt) if tgt else None,
        texts=_as_list(it.get("text")),  # 글자는 그룹이 아니라 **면**에 들어간다
        shapes=list(it.get("shapes") or []),
        pre_groups=_groups(cfg_path, where, it, "pre_groups"),
        groups=_groups(cfg_path, where, it, "groups"),
        post_shapes=list(it.get("post_shapes") or []))


def _groups(cfg_path: Path, where: str, it: dict, key: str) -> list[GroupLoad]:
    """`pre_groups`·`groups` 목록 읽기 — 규약(plan·group·layers·변형)이 같다."""
    out: list[GroupLoad] = []
    for j, g in enumerate(it.get(key) or []):
        gp = (cfg_path.parent / g["plan"]).resolve()
        if not gp.exists():
            raise ValueError(msg("{where}: {key}[{j}] 플랜이 없다 — {path}",
                                 where=where, key=key, j=j, path=gp))
        gl = len(LayerPlan.load(gp).layers)
        name = g.get("group") or ascii_name(run_label(gp))
        if not NAME_OK.match(name):
            raise ValueError(msg("{where}: {key}[{j}] 저장 슬롯 이름은 ASCII 30자 "
                                 "이내여야 한다 (받은 것: {got!r})",
                                 where=where, key=key, j=j, got=name))
        out.append(GroupLoad(
            plan=gp, group=name, layers=gl,
            x=float(g.get("x", 0.0)), y=float(g.get("y", 0.0)),
            scale=float(g.get("scale", 1.0)), rot=float(g.get("rot", 0.0)),
            mirror=bool(g.get("mirror", False))))
    return out


def _as_list(v) -> list[dict]:
    if not v:
        return []
    return [v] if isinstance(v, dict) else list(v)


def _check(cfg: Config, names: list[str] | None = None) -> None:
    """면 예산·중복·장수 충돌을 **게임 밖에서** 다 잡는다."""
    used: dict[str, str] = {}
    by_layers: dict[int, str] = {}
    for p in cfg.placements:
        if p.surface in used:
            raise ValueError(msg("면 {surface}에 배치가 둘이다 "
                                 "(하나만 둔다 — 여러 도안은 도안 쪽에서 합칠 것)",
                                 surface=p.surface))
        used[p.surface] = p.group
        if p.copy_from is not None:
            if p.copy_from not in used:
                raise ValueError(msg("{surface}: copy_from={copy_from}이 "
                                     "아직 배치되지 않았다 (앞에 두어야 한다)",
                                     surface=p.surface, copy_from=p.copy_from))
            continue
        if p.group_layers == 0:      # 글자 전용 배치 — 그룹 장수 검사 무관
            continue
        cap = body.surface_cap(p.surface, names)
        total = p.group_layers + p.text_layers + len(p.shapes) + len(p.post_shapes)
        if cap is not None and total > cap:
            heavy = max(p.all_groups(), key=lambda g: g.layers)
            raise ValueError(msg(
                "{surface}: 이 면에 올릴 것이 {total:,}장인데 상한이 {cap:,}장이다 "
                "(그룹 {group_layers:,} + 글자 {text_layers} + 도형 "
                "{shape_count}).\n"
                "  이 면의 도안을 줄이거나 한 장을 빼고 다시 걸 것 — 가장 무거운 것은 "
                "'{heavy_group}' {heavy_layers:,}장이다\n"
                "  (장수를 줄이려면: python -m forzasqueegee pruneplan "
                "{heavy_plan} --min-vis 1)",
                surface=p.surface, total=total, cap=cap,
                group_layers=p.group_layers, text_layers=p.text_layers,
                shape_count=len(p.shapes) + len(p.post_shapes),
                heavy_group=heavy.group, heavy_layers=heavy.layers,
                heavy_plan=heavy.plan))
        for name_, layers_ in [(g.group, g.layers) for g in p.all_groups()]:
            if layers_ in by_layers and by_layers[layers_] != name_:
                raise ValueError(msg(
                    "그룹 {a}과 {b}의 장수가 {layers:,}장으로 "
                    "같다 — 게임 그리드에서 이름을 못 읽어 **장수로** 고르므로 갈 수 없다.\n"
                    "  한 쪽을 `pruneplan --min-vis 1`로 한 장이라도 줄일 것",
                    a=name_, b=by_layers[layers_], layers=layers_))
            by_layers[layers_] = name_


def make_config(plans: list[Path], out: Path,
                surfaces: tuple[str, ...] = PRESET_ITASHA) -> Config:
    """플랜 목록 → 프리셋 구성 파일. 하나면 좌측면 + 우측면(반대편 복사)이다."""
    items: list[dict] = []
    if len(plans) == 1 and surfaces == PRESET_ITASHA:
        # 좌우에 **같은 그룹을 두 번** 올린다 (오른쪽은 미러). 반대편 복사
        # (`copy_from`)가 더 싸 보이지만 그 메뉴 행은 면마다 있다 없다 하고
        # 실패해도 카운터로만 알 수 있다 — 예산이 면마다 따로라 두 번 올려도
        # 손해가 없으므로 예측 가능한 쪽을 기본으로 둔다.
        rel = _rel(plans[0], out)
        items.append({"plan": rel, "surface": "side_left"})
        items.append({"plan": rel, "surface": "side_right", "mirror": True})
    else:
        for plan, surface in zip(plans, surfaces):
            items.append({"plan": _rel(plan, out), "surface": surface})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"apply": True, "placements": items},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    return load_config(out)


def compose_config(main_plan: Path, out: Path, *,
                   extra_plans: list[Path] | None = None,
                   car: str | None = None, media: str | None = None,
                   mirror: bool = True, paint: bool = True,
                   base_rgb: "tuple[int, int, int] | None" = None,
                   flip: bool = False, manual: "list | None" = None,
                   deco: bool = True,
                   motif: str | None = None, family: str | None = None,
                   preview: bool = True, log=print) -> Config:
    """도안 하나(+보조) → **설계된** 이타샤 구성. `engine.compose`가 짠다.

    `make_config`(프리셋 상수)와 갈리는 자리는 셋이다:

    1. **베이스 도색**을 도안에서 고른다 (`compose.base_paint` — `base_rgb`를
       주면 그 색) — 실행이 자동차 도색 메뉴에서 차 전체를 칠한다
       (`auto.paintcar`).
    2. 스케일·이동을 **면 실측 지도**로 계산한다 (인게임 프로브가
       잰 것). 지도가 없거나 의심스러운 면은 프리셋으로 물러난다.
    3. 꾸밈 그룹(로커 띠·산포)·관통 띠·지붕 블랙아웃이 자동으로 짜인다 —
       `deco=False`면 그 전부를 빼고 **도안만** 올린다. 면을 넘친 조각은
       이웃 면에 안 잇고 그 자리에서 잘린다. 모티프 계열은 도안의 테마색이
       고르고 `motif`로 못 박는다 (`compose.motif_family`). 옆면 꾸밈의 **구성
       계열**은 후보를 지어 점수로 고르고 `family`로 못 박는다
       (`compose.FAMILIES`).

    `manual`(`engine.compose.ManualPlace` 목록)을 주면 **도안 자리만** 사람이
    정한 것으로 바뀌고 나머지는 그대로다 — 내장 편집기(`engine.fls.studio`)가
    이 길로 온다.
    """
    from ...engine import compose
    from ...game import body as gbody

    car = car or gbody.tab_table().get("car")
    # `out`이 폴더인지 파일인지는 **있는 그대로** 본다 — 확장자로 짐작하면
    # 스튜디오 작업 폴더(`<이름>.fsitasha/`)가 파일로 오인돼 구성이 부모
    # 폴더로 새고, 폴더 위에 파일을 쓰려다 'Permission denied'로 죽는다.
    as_dir = out.is_dir() or not out.suffix
    rec = compose.build(main_plan, out if as_dir else out.parent,
                        car=car, media=media,
                        extra_plans=list(extra_plans or []),
                        mirror=mirror, paint=paint, base_rgb=base_rgb,
                        flip=flip, manual=manual,
                        deco=deco, motif=motif, family=family, log=log)
    cfg_path = next(p for p in rec.written if p.name.endswith("itasha.json"))
    # `out`이 **폴더**면 구성 파일은 이미 그 안에 쓰였다 — 폴더 위에 덮어쓰려
    # 하면 안 된다 (`-o out/내차`가 'Permission denied: out/내차'로 죽었다).
    if as_dir:
        out = cfg_path
    elif cfg_path != out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(cfg_path.read_text(encoding="utf-8"), encoding="utf-8")
    # 구성 미리보기 — 게임을 건드리기 전에 면별 결과 꼴을 그림으로 남긴다
    # (설치 파일 면 지도 기반, `engine.preview`). 실패해도 실행은 계속한다.
    if preview:
        try:
            from ...engine import preview as _preview

            _preview.render_config(out, media=media, log=log)
        except Exception as e:                    # noqa: BLE001 — 미리보기는 보조다
            log(msg("미리보기 생성 실패: {kind}: {err}",
                    kind=type(e).__name__, err=e))
    return load_config(out)


def _rel(plan: Path, out: Path) -> str:
    try:
        return plan.resolve().relative_to(out.resolve().parent).as_posix()
    except ValueError:
        return plan.resolve().as_posix()
