"""실행 환경이 `pyproject.toml`이 못 박은 그대로인가 — `ForzaSqueegee.bat`이 묻는다.

**"불러와지나"로는 모자란다.** 옛 판이 이미 깔려 있으면 임포트는 다 되고,
그러면 설치를 건너뛰어 고정한 판이 한 번도 안 선다 (실측: numpy 1.26에서
같은 그림의 레이어 120장이 전부 달랐다). 그래서 **판을 대조한다**.

    python tools/check_env.py          # 맞으면 0, 아니면 1
    python tools/check_env.py --quiet  # 찍지 않고 코드만

표준 라이브러리만 쓴다 — 꾸러미가 아직 없을 때 도는 자리다. 출력은 콘솔
코드페이지(cp949)에 실리므로 ASCII 문장부호만 쓴다.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _pins() -> tuple[list[tuple[str, str]], tuple[int, int] | None, tuple[int, int] | None]:
    """(꾸러미 == 판) 목록과 파이썬 하한/상한."""
    raw = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    proj = raw["project"]
    pins = []
    for dep in proj.get("dependencies", []):
        name, _, ver = str(dep).partition("==")
        if ver:
            pins.append((name.strip(), ver.strip()))
    lo = hi = None
    for part in str(proj.get("requires-python", "")).split(","):
        part = part.strip()
        if part.startswith(">="):
            lo = tuple(int(x) for x in part[2:].strip().split("."))
        elif part.startswith("<"):
            hi = tuple(int(x) for x in part[1:].strip().split("."))
    return pins, lo, hi


def check(quiet: bool = False) -> int:
    say = (lambda *a: None) if quiet else print
    pins, lo, hi = _pins()
    bad = 0

    cur = sys.version_info[:2]
    if (lo and cur < lo) or (hi and cur >= hi):
        want = ".".join(map(str, lo or ())) + " ~ " + (
            ".".join(map(str, (hi[0], hi[1] - 1))) if hi else "")
        say(f"  파이썬 {'.'.join(map(str, cur))} (필요: {want})")
        bad += 1

    import importlib.metadata as md

    for name, want in pins:
        try:
            got = md.version(name)
        except md.PackageNotFoundError:
            say(f"  {name}: 없음 (필요 {want})")
            bad += 1
            continue
        if got != want:
            say(f"  {name}: {got} (필요 {want})")
            bad += 1
    if bad and not quiet:
        say(f"  맞지 않는 것 {bad}개")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(check("--quiet" in sys.argv))
