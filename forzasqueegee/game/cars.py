r"""설치 폴더의 **차량 목록을 한 벌로 떠 둔다** — `work/state/cars.json`.

**저장소에는 이 색인이 없다.** 각자의 설치본이 정본이라(판·DLC마다 차가
다르다) 처음 필요해지는 순간 그 자리에서 뜬다 (`index` → `sync`). 게임
파일에서 나온 자료를 우리가 배포하지 않는다는 뜻이기도 하다.

## 왜 필요한가

차마다 다른 것이 셋이다: **면 탭 구성**(스포일러·선루프가 있고 없고), 면
**상한**, 면 **크기**. 셋 다 설치 파일이 알고 있는데 (`media/Cars/<차>.zip`의
`LiveryMasks/Masks.xml`), 지금까지 그걸 아는 자리는 **인게임 실측표 한 장**
뿐이었다 (`catalog/body_tabs.json` — 사람이 게임에서 재서 만든 차 한 대의 표).

그래서 표를 잰 차에 없는 면은 이름조차 모른다. 실측 재현: 표는 줄리아 GTAm
(9면)인데 인테그라 23은 스포일러·선루프가 있어 11면이고, 그 차의 구성을
읽으면 **`모르는 차체 면 이름: sunroof`로 죽는다** — 설치 파일에는 그 면이
또렷이 적혀 있는데도.

동기화는 그 간극을 메운다: 설치본 전부의 면 목록과 크기를 한 번에 떠서 파일로
두고, 탭 해석·상한 검사·면 이름 검증이 **그 차의** 목록을 본다. 크기는 사람이
차를 고르는 자리(편집기의 [New Project → Livery])와 `--media` 후보 목록이 본다 — 이름만으로는
경트럭과 로드스터가 같은 점수를 받는데, 옆면 상자는 529유닛과 902유닛이다.

## 무엇을 뜨고 무엇을 안 뜨나

뜨는 것은 `Masks.xml`이 아는 것뿐이다 — 면 목록(=탭 순서)과 면마다의 **유닛
상자**다. 상자에서 차 한 대를 세 수로 줄인 것이 **면 크기**다 (길이·폭·높이 —
`carfiles.size_of_boxes`). 유닛이지 밀리미터가 아니다.

**마스크 텍스처는 안 푼다**: 한 대 0.26초씩 636대면 2분이고, 정작 배치에
필요할 때 한 대만 풀면 되기 때문이다 (`carfiles.surface_maps`). XML만 읽으면
**636대가 0.9초**고, 상자를 같이 떠도 그대로다 — 상자는 이미 읽은 XML에 있다.

**게임을 켜야만 아는 것은 여기 없다.** 도색 마스크 실측·윗면 유리·화면 배율은
프로브의 몫이고 (→ `catalog/surfaces/`), 동기화는 그
쪽을 대신하지 못한다 — 어느 차를 쟀는지 보여 줄 뿐이다.

## 낡으면 — 알아서 다시 뜬다

프로세스마다 한 번, 색인이 지금 설치본과 맞는지 본다 (`stale`): 파일이 없거나,
설치 폴더가 바뀌었거나(`root_id`), 차 수가 다르면 그 자리에서 다시 뜬다. 확인
자체는 폴더 목록 한 번이라 공짜에 가깝다.

차 몇 대만 늘었을 때는 다시 뜨기 전에도 답이 틀리지 않는다 — 색인에 없는 차는
**그 차만 zip에서 읽는다** (`tabs_of`·`size_of`). 목록 자체
(`carfiles.list_cars`)는 늘 폴더를 직접 본다.

## 못 뜨면

설치 폴더를 못 찾거나 색인을 못 쓰면 **여기서 죽지 않는다** — 빈 표로 돌아가고
부르는 쪽이 물러난다 (zip 직접 읽기 · 면 지도 없이 프리셋). 사람에게 폴더를
물어야 하는 자리에서는 `ensure(ask=...)`를 쓴다 — 받은 자리는 못 박아 저장하니
다음 실행부터는 안 묻는다.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import date
from pathlib import Path

from . import carfiles

_CACHE: dict | None = None


def path() -> Path:
    """색인을 두는 자리 — **생성물이라 `work/`에 산다** (저장소에 안 실린다).

    못 박은 설치 폴더(`work/state/gamedir.json`) 바로 옆이다. 지워도 다음
    실행이 설치 폴더에서 다시 뜬다.
    """
    from ..paths import work_root
    return work_root() / "state" / "cars.json"


def _root_id(root: Path | str) -> str:
    """설치 폴더의 **신원**만 남긴다 — 경로 자체는 안 적는다.

    낡았는지 보는 데에는 "같은 폴더인가"만 있으면 되고, 이 파일은 저장소에
    실려 나가므로 뜬 사람의 경로가 남을 이유가 없다. 대소문자·구분자 차이로
    헛경고가 나지 않게 normcase를 먼저 태운다.
    """
    norm = os.path.normcase(os.path.abspath(str(root)))
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:12]


def _read() -> dict:
    """떠 둔 색인 파일 (없거나 깨졌으면 빈 표)."""
    p = path()
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except (OSError, ValueError):
        return {}


def stale(raw: dict) -> bool:
    """이 색인이 **지금 설치본과 안 맞나** — 없거나·다른 폴더거나·차 수가 다르면.

    설치 폴더를 못 찾으면 **낡지 않았다고 본다**: 다시 뜰 수가 없는데 낡았다고
    해 봐야 매번 헛되이 훑을 뿐이고, 있는 색인이라도 쓰는 편이 낫다.
    """
    if not raw.get("cars"):
        return True
    root = carfiles.install_dir()
    if root is None:
        return False
    if raw.get("root_id") and _root_id(root) != raw["root_id"]:
        return True
    live = len(carfiles.list_cars(root))
    return bool(live and live != len(raw["cars"]))


def index() -> dict:
    """색인 — **없거나 낡았으면 설치 폴더에서 그 자리에서 뜬다.**

    저장소에 색인을 실어 두지 않으므로(각자의 설치본이 정본이다) 처음 쓰는
    순간이 뜨는 순간이다. 645대 XML 읽기가 찬 캐시에서 4.5초·더운 캐시에서
    1초 안팎이고, 그 뒤로는 파일이 남아 프로세스마다 확인만 한다. 설치 폴더를 못 찾거나 못 쓰면 **빈 표로
    돌아가고 부르는 쪽이 물러난다** (zip 직접 읽기·프리셋) — 여기서 죽지
    않는다. 사람에게 폴더를 물어야 하는 자리는 `ensure`다.
    """
    global _CACHE
    if _CACHE is None:
        _CACHE = _read()
        if stale(_CACHE):
            try:
                sync()
            except (OSError, ValueError):
                pass
    return _CACHE


def ensure(ask=None, log=None) -> dict:
    """색인을 **보장한다** — 못 뜨면 사람에게 설치 폴더를 묻는다.

    `ask(why)`는 사람에게 폴더를 받아 돌려주는 콜백이다 (CLI는 콘솔 입력,
    창은 폴더 고르기). 빈 값을 돌려주면 그만 묻는다. 받은 자리는 못 박아
    저장하므로(`carfiles.set_install_dir`) 다음 실행부터는 안 묻는다.
    """
    global _CACHE
    raw = index()
    for _ in range(3):
        if raw.get("cars") or ask is None:
            break
        got = ask(carfiles.resolve()[1])
        if not got:
            break
        try:
            carfiles.set_install_dir(got)
        except ValueError as e:
            if log:
                log(str(e))
            continue
        _CACHE = None
        raw = index()
    return raw


def cars() -> dict[str, list[str]]:
    """{미디어명: 면 이름 목록} — 색인에 있는 것만."""
    got = index().get("cars")
    return got if isinstance(got, dict) else {}


def tabs_of(media: str | None, root: Path | None = None) -> list[str]:
    """그 차의 **면 탭 구성**. 색인이 알면 색인, 모르면 zip을 직접 읽는다."""
    if not media:
        return []
    got = cars().get(media)
    if got:
        return list(got)
    return carfiles.tab_names(media, root)


def caps_of(media: str | None, root: Path | None = None) -> dict[str, int]:
    """그 차의 면별 레이어 상한 — 면 이름이 정한다 (`carfiles.TAB_CAPS`)."""
    return {n: carfiles.TAB_CAPS.get(n, 1000) for n in tabs_of(media, root)}


def sizes() -> dict[str, list[int]]:
    """{미디어명: [길이, 폭, 높이]} — 색인에 있는 것만."""
    got = index().get("size")
    return got if isinstance(got, dict) else {}


def size_of(media: str | None, root: Path | None = None) -> tuple[int, int, int] | None:
    """그 차의 **면 크기** (길이, 폭, 높이 유닛). 색인이 알면 색인, 모르면 zip.

    무엇을 어느 면에서 재는지는 `carfiles.size_of_boxes`가 정한다 — 유닛이지
    밀리미터가 아니다.
    """
    if not media:
        return None
    got = sizes().get(media)
    if got and len(got) == 3:
        return int(got[0]), int(got[1]), int(got[2])
    return carfiles.car_size(media, root)


def size_text(media: str | None, root: Path | None = None) -> str:
    """후보 목록·차 고르는 칸에 그대로 붙이는 한 토막 — `934×372×274`. 모르면 빈 글."""
    got = size_of(media, root)
    return "×".join(str(v) for v in got) if got else ""


def sync(root: Path | None = None, log=None) -> dict:
    """설치 폴더를 훑어 색인을 다시 뜨고 저장한다. 되돌리는 것은 요약이다.

    요약: `{"cars": 636, "faces": 5881, "sized": 636, "failed": [...],
    "probed": [...]}` — `probed`는 **인게임 프로브까지 잰 차**다
    (`catalog/surfaces/`). 동기화가 그것까지 해 주지는 못하므로 몇 대가
    재였는지만 알려 준다.

    면 구성과 크기를 **한 번 읽어 둘 다 뜬다** — 상자를 읽으면 이름은 그
    열쇠다 (`carfiles.tab_boxes`). 텍스처를 안 푸는 것은 그대로다.
    """
    global _CACHE
    root = root or carfiles.install_dir()
    if root is None:
        raise FileNotFoundError(
            "FH6 설치 폴더를 못 찾았다 — "
            "`python -m forzasqueegee gamedir <경로>`로 못 박을 것")
    names = carfiles.list_cars(root)
    got: dict[str, list[str]] = {}
    size: dict[str, list[int]] = {}
    failed: list[str] = []
    for i, media in enumerate(names, 1):
        boxes = carfiles.tab_boxes(media, root)
        if boxes:
            got[media] = list(boxes)
            lwh = carfiles.size_of_boxes(boxes)
            if lwh:
                size[media] = list(lwh)
        else:
            failed.append(media)
        if log and (i % 100 == 0 or i == len(names)):
            log(f"  {i}/{len(names)}대")
    out = {"synced": date.today().isoformat(), "root_id": _root_id(root),
           "cars": got, "size": size}
    _CACHE = out
    # **쓰기는 덤이다** — 못 써도(읽기 전용 자리·권한) 이번 실행은 뜬 색인으로
    # 그대로 돈다. 다음 실행이 다시 뜰 뿐이다.
    p = path()
    saved = True
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    except OSError as e:
        saved = False
        if log:
            log(f"  색인을 저장하지 못했다 ({e}) — 이번 실행만 쓴다")
    return {"cars": len(got), "faces": sum(len(v) for v in got.values()),
            "sized": len(size), "failed": failed, "probed": probed(),
            "path": p, "root": root, "saved": saved}


def probed() -> list[str]:
    """**인게임 프로브까지 잰 차** 이름들 (`catalog/surfaces/`). 동기화 밖의 일이다."""
    from . import surface as gsurf

    d = gsurf.map_dir()
    if not d.exists():
        return []
    out = []
    for p in sorted(d.glob("*.json")):
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if raw.get("car"):
            out.append(raw["car"])
    return out


def summary() -> str:
    """지금 상태 한 줄 — 사람에게 그대로 보여 준다.

    **낡았으면 낡았다고 말한다**: 설치 폴더가 바뀌었거나 차 수가 안 맞으면 그
    자리에서 보인다 (색인이 없는 차는 zip을 직접 읽으므로 틀린 답이 나오지는
    않는다 — 다시 뜨면 빨라질 뿐이다).
    """
    raw = index()
    n = len(cars())
    if not n:
        root, why = carfiles.resolve()
        if root is None:
            return (f"차량 정보가 없다 — 설치 폴더를 {why} · "
                    "`python -m forzasqueegee gamedir <경로>`로 지정할 것")
        return "차량 정보가 아직 없다 — `python -m forzasqueegee cars --sync`"
    root = carfiles.install_dir()
    live = len(carfiles.list_cars())
    note = ""
    if (root is not None and raw.get("root_id")
            and _root_id(root) != raw["root_id"]):
        note = " · 설치 폴더가 바뀌었다 (다시 뜰 것)"
    elif live and live != n:
        note = f" · 설치 폴더에는 {live}대 (다시 뜰 것)"
    elif not sizes():
        note = " · 크기가 아직 없다 (다시 뜰 것)"
    # 크기가 636대 중 635대인 것은 흠이 아니다 — 옆면도 앞뒤면도 없는 차가 있다
    # (아리엘 노마드는 윗면·앞유리뿐이라 잴 높이가 없다).
    return (f"차량 {n}대 · 크기 {len(sizes())}대 "
            f"({raw.get('synced', '?')} 동기화){note} · "
            f"인게임 프로브 {len(probed())}대")
