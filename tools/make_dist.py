r"""배포 꾸러미를 짓는다 — `dist/ForzaSqueegee-<판>.zip`.

받은 사람이 **풀고 `ForzaSqueegee.bat`을 누르면 되는** 한 벌이다. 들어가는 것은
실행에 필요한 것뿐이고(개발 문서·도구·기록은 안 들어간다), 거기에 **라이선스가
요구하는 것**이 붙는다.

    python tools/make_dist.py            # dist/ForzaSqueegee-1.3.1.zip
    python tools/make_dist.py --check    # 무엇이 들어가는지만 센다

## 무거운 것은 안 싣는다 — 받는다

신경망 모델 넷(332MB)과 FLS 편집기(31MB)는 꾸러미에 안 들어간다. 자리와
SHA-256이 `release.json`에 있고, 받은 사람의 프로그램이 **쓸 때 릴리스에서**
받는다 (`forzasqueegee/modelstore.py`·`tools/get_fls.py`). 꾸러미가 360MB
가벼워지고, 모델이나 편집기를 갈아도 꾸러미를 다시 안 지어도 된다.

KFPS 편집기의 도형 리소스(30MB)도 마찬가지로 빠지는데, 이쪽은 이유가 다르다 —
게임 도형의 메시라 **우리가 재배포하지 않는다**. 편집기를 처음 열 때 KFPS
고정 커밋에서 받는다 (`tools/get_kfps.py`).

편집기가 꾸러미에서 빠지면서 **AGPL 의무도 꾸러미를 안 따라온다** — 그 바이너리는
릴리스가 배포하고, 대응 소스가 같은 릴리스에 함께 올라간다
(`tools/fls_build.py --package`가 둘을 짓는다).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
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

sys.path.insert(0, str(ROOT))

# 실행에 필요한 것 — git이 추적하는 것 중에서 고른다 (기록·개인 파일이 안 샌다).
KEEP_PREFIX = ("forzasqueegee/", "catalog/", "vendor/galatea/",
               "vendor/kfps-editor/", "models/", "docs/", "tools/")
KEEP_FILES = ("ForzaSqueegee.bat", "pyproject.toml", "THIRD_PARTY_NOTICES.md",
              "LICENSE", "README.md", "release.json")


def version() -> str:
    import tomllib

    raw = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return str(raw["project"]["version"])


def tracked() -> list[str]:
    out = subprocess.run(["git", "-C", str(ROOT), "ls-files"],
                         capture_output=True, text=True, check=True).stdout
    return [l for l in out.splitlines() if l]


def picked() -> list[Path]:
    """꾸러미에 들어갈 파일 (저장소 상대 경로)."""
    got = [Path(rel) for rel in tracked()
           if rel.startswith(KEEP_PREFIX) or rel in KEEP_FILES]
    return [p for p in got if "__pycache__" not in p.parts]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="무엇이 들어가는지만 센다")
    a = ap.parse_args()

    files = picked()
    total = sum((ROOT / p).stat().st_size for p in files)
    groups: dict[str, tuple[int, int]] = {}
    for p in files:
        key = p.parts[0] if len(p.parts) > 1 else "(뿌리)"
        n, b = groups.get(key, (0, 0))
        groups[key] = (n + 1, b + (ROOT / p).stat().st_size)
    for key, (n, b) in sorted(groups.items()):
        print(f"  {key:24s} {n:5,}개 {b / 1e6:8.1f}MB")
    print(f"  {'합계':24s} {len(files):5,}개 {total / 1e6:8.1f}MB")
    if a.check:
        return 0

    ver = version()
    out = ROOT / "dist" / f"ForzaSqueegee-{ver}.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    # 최상위를 폴더 하나로 감싼다 — 다운로드 폴더에 그대로 풀어도 파일 350여 개가
    # 흩어지지 않고, 풀린 그 폴더가 곧 프로그램 자리다 (`runtime/`·`out/`·`work/`이
    # 다 그 안에 놓인다).
    top = f"ForzaSqueegee-{ver}"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in files:
            z.write(ROOT / p, f"{top}/{p.as_posix()}")
    print(f"\n{out}  ({out.stat().st_size / 1e6:.1f}MB)")
    print("  모델과 FLS 편집기는 안 실렸다 — 쓸 때 릴리스에서 받는다 (release.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
