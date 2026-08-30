"""실행 환경이 `pyproject.toml`이 못 박은 그대로인가 — `ForzaSqueegee.bat`이 묻는다.

**"불러와지나"로는 모자란다.** 옛 판이 이미 깔려 있으면 임포트는 다 되고,
그러면 설치를 건너뛰어 고정한 판이 한 번도 안 선다 (실측: numpy 1.26에서
같은 그림의 레이어 120장이 전부 달랐다). 그래서 **판을 대조한다**.

    python tools/check_env.py          # 맞으면 0, 아니면 1
    python tools/check_env.py --quiet  # 찍지 않고 코드만
    python tools/check_env.py --deep   # 판 대조에 더해 실제로 임포트까지 해 본다

표준 라이브러리만 쓴다 — 꾸러미가 아직 없을 때 도는 자리다. stdout은 UTF-8
(replace)로 돌려 앉히므로 파이프에 물려도 출력이 안 죽는다.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 파이프·리다이렉트에선 stdout이 콘솔 코드페이지(cp949 등)로 떨어져 한글이나
# `—` 한 글자에 UnicodeEncodeError로 죽는다 — elevate.ensure_std_streams와 같은
# 보호인데, 이 스크립트들은 독립 실행이라 여기 따로 둔다.
for _s in (sys.stdout, sys.stderr):
    if _s is not None and hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:               # noqa: BLE001 — 못 바꿔도 그냥 간다
            pass

# i18n은 stdlib뿐이라 실제로는 늘 서지만, 이 스크립트는 꾸러미가 아직 없는
# 자리라 임포트가 무너져도 원문 그대로 말하고 계속 간다.
try:
    sys.path.insert(0, str(ROOT))
    from forzasqueegee.i18n import msg
except Exception:                       # noqa: BLE001
    msg = lambda s, **kw: s.format(**kw) if kw else s   # noqa: E731


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
        say(msg("  파이썬 {got} (필요: {want})",
                got='.'.join(map(str, cur)), want=want))
        bad += 1

    import importlib.metadata as md

    for name, want in pins:
        try:
            got = md.version(name)
        except md.PackageNotFoundError:
            say(msg("  {name}: 없음 (필요 {want})", name=name, want=want))
            bad += 1
            continue
        if got != want:
            say(msg("  {name}: {got} (필요 {want})",
                    name=name, got=got, want=want))
            bad += 1
    if bad and not quiet:
        say(msg("  맞지 않는 것 {n}개", n=bad))
    return 1 if bad else 0


# 꾸러미 이름 → 실제로 임포트해 볼 모듈. 판은 맞는데 못 뜨는 경우가 있다 —
# 네이티브 확장이 시스템 DLL(msvcp140 등)을 못 찾을 때다. 매 실행마다 재면
# 몇 초씩 늘어나므로 설치 직후(--deep)에만 잰다.
_IMPORTS = (("numpy", "numpy"), ("opencv-python", "cv2"),
            ("Pillow", "PIL.Image"), ("PySide6", "PySide6.QtWidgets"),
            ("onnxruntime", "onnxruntime"))


def deep(quiet: bool = False) -> int:
    """핀 임포트를 실제로 해 본다. 못 뜨는 것이 있으면 1."""
    import importlib

    say = (lambda *a: None) if quiet else print
    bad = 0
    for pkg, mod in _IMPORTS:
        try:
            importlib.import_module(mod)
        except Exception as e:                     # noqa: BLE001 — 원인 불문 보고
            bad += 1
            say(msg("  {pkg}: 임포트 실패 - {err}", pkg=pkg, err=e))
            if "DLL" in str(e):
                say(msg("    Visual C++ 재배포 패키지가 없으면 이렇게 된다. 설치:"))
                say("    https://aka.ms/vs/17/release/vc_redist.x64.exe")
    return 1 if bad else 0


if __name__ == "__main__":
    quiet = "--quiet" in sys.argv
    rc = check(quiet)
    if "--deep" in sys.argv:
        rc = deep(quiet) or rc
    sys.exit(rc)
