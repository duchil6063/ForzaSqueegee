"""**쓰는 곳**의 뿌리 — 어디에 무엇을 쓰는지는 여기서만 정한다.

두 자리로 가른다:

- `out/` — **도안 산출물만.** 사람이 만들어 쓰는 결과물이다 (`make` 결과·
  리버리·편집기 내보내기·들여오기 변환). 폴더 하나가 도안 하나다.
- `work/` — 도안이 아닌 살림. 편집기 상태·기하 덤프·인게임 캡처·캐시·
  검증 스크래치·계측 산출이 여기 산다. 지워도 도안은 안 없어진다
  (`work/geom`·`work/cars`만은 다시 못 뜨는 실측이라 예외).

도안 폴더 안 파일은 **폴더 이름을 앞에 단다** — `out/line-01/line-01.plan.json`.
이름이 `plan.json` 하나로 고정이면 편집기 탭·게임 임포트 목록·열린 파일
대화상자에서 어느 도안인지 안 갈린다. 쓸 때는 `run_file()`, 읽을 때는
`find_run_file()`을 쓴다 (뒤엣것은 예전 이름도 받는다).

읽기 전용 자산(`catalog/`)은 손댈 게 없다 — 모듈들이
`Path(__file__).resolve().parents[2]`로 저장소 안을 그대로 가리킨다.
"""

from __future__ import annotations

from pathlib import Path


def data_root() -> Path:
    """산출물을 쓰는 뿌리 = 저장소 뿌리."""
    return Path(__file__).resolve().parents[1]


def out_root() -> Path:
    """도안 산출물의 뿌리 (`out/`)."""
    return data_root() / "out"


def work_root() -> Path:
    """도안이 아닌 살림의 뿌리 (`work/`)."""
    return data_root() / "work"


def work_file(*parts: str) -> Path:
    """`work/` 아래 파일 하나 — 담긴 폴더까지 만들어 준다."""
    p = work_root().joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def run_file(run_dir: str | Path, name: str) -> Path:
    """도안 폴더에 **쓸** 자리 — `out/line-01/line-01.plan.json`."""
    d = Path(run_dir)
    return d / f"{d.name}.{name}"


def find_run_file(run_dir: str | Path, name: str) -> Path:
    """도안 폴더에서 **읽을** 자리.

    새 이름(`line-01.plan.json`)이 없으면 예전 이름(`plan.json`)을 준다 —
    이미 구운 도안 폴더가 그대로 돌아간다. 둘 다 없으면 새 이름을 주므로
    "없다"는 메시지에 지금 규칙의 이름이 뜬다.
    """
    p = run_file(run_dir, name)
    if p.exists():
        return p
    old = Path(run_dir) / name
    return old if old.exists() else p


def run_label(path: str | Path) -> str:
    """도안 파일 하나에 붙일 **이름** — 담긴 폴더 이름이 기본이다.

    게임 저장 슬롯·컨테이너 폴더·FLS 프로젝트가 이 이름으로 선다.

        out/내도안/내도안.plan.json        → 내도안
        out/내도안/내도안.plan_sorted.json → 내도안-plan_sorted
        out/내도안/plan.json (예전 이름)   → 내도안
    """
    p = Path(path)
    parent = p.parent.name
    stem = p.stem
    pre = f"{parent}."
    if stem.startswith(pre):
        stem = stem[len(pre):]
    return parent if stem == "plan" else f"{parent}-{stem}"


def glob_run_files(root: str | Path, name: str, *, deep: bool = False):
    """`root` 아래 도안 폴더들에서 `name`을 모은다 (새 이름·예전 이름 둘 다).

    `deep`이면 몇 겹이든 내려간다 (`out/**/`), 아니면 바로 아래 칸만 본다.
    폴더 하나에 둘 다 있으면 새 이름 쪽만 준다.
    """
    root = Path(root)
    if not root.is_dir():
        return []
    pre = "**/" if deep else "*/"
    seen: dict[Path, Path] = {}
    for p in sorted(root.glob(f"{pre}*.{name}")):
        if p.name == f"{p.parent.name}.{name}":
            seen[p.parent] = p
    for p in sorted(root.glob(f"{pre}{name}")):
        seen.setdefault(p.parent, p)
    return [seen[d] for d in sorted(seen)]
