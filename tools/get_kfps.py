r"""KFPS 편집기의 **도형 리소스**를 `vendor/kfps-editor/Resources/`에 받아 둔다.

이 2,800파일은 게임 비닐 도형의 메시 데이터(정점·인덱스·알파)와 그 미리보기다.
**게임 에셋에서 나온 것이라 우리가 재배포하지 않는다** — 저장소에 넣지 않고
KFPS 고정 커밋에서 그때그때 받는다 (LICENSE·THIRD_PARTY_NOTICES의 "게임 콘텐츠").
FLS 편집기·신경망 모델과 같은 길이다.

    python tools/get_kfps.py            # 받는다
    python tools/get_kfps.py --check    # 있는지만 본다
    python tools/get_kfps.py --verify   # 받아 둔 것을 집계 SHA-256으로 대조

## 받는 길이 둘이다

- **git** (있으면 이 길) — 부분 체크아웃(`--filter=blob:none` + sparse)이라
  필요한 36MB만 받는다. 실측 4초.
- **zip** (git이 없을 때) — codeload가 커밋 통째로만 주므로 250MB급을 받아
  그중 이 폴더만 푼다. 느리지만 git 없이 된다.

편집기를 안 쓰면 받을 필요가 없다 — 도안 만들기·게임 파일 내보내기는 이것
없이 다 된다.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import tempfile
import urllib.request
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

DEST = ROOT / "vendor" / "kfps-editor" / "Resources"

REPO = "heyitshestia/kloudys-forza-painter-suite"
# vendor/kfps-editor/README.md가 적은 그 커밋이다 (KFPS 3.1.40) — 편집기
# 파일들과 **같은 판**이어야 도형 리소스 포맷이 안 갈린다.
PIN = "0af4f21f984ad42f33dcf570ad36ad8e704092b6"
SUB = "tools/fabric-editor/Resources"
ZIP_URL = f"https://codeload.github.com/{REPO}/zip/{PIN}"

FILES = 2800
# 집계 = 정렬한 `<상대경로> <파일 sha256>` 줄들의 sha256 (`aggregate`).
AGG = "291821b134a74d557fca074db447efbfa06bddab866bfe47878b66759227ea07"


def aggregate(root: Path) -> tuple[int, str]:
    """(파일 수, 집계 SHA-256) — 트리 하나를 한 수로 줄인다.

    파일마다 해시를 적어 두는 대신 이렇게 둔다: 2,800개를 표로 들고 있어 봐야
    사람이 못 읽고, 우리가 묻는 것은 "업스트림 그대로인가" 하나뿐이다.
    """
    h = hashlib.sha256()
    n = 0
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        n += 1
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        h.update(f"{p.relative_to(root).as_posix()} {digest}\n".encode())
    return n, h.hexdigest()


def have() -> bool:
    """받아 둔 것이 쓸 만한가 (폴더가 있고 파일 수가 맞나)."""
    if not DEST.is_dir():
        return False
    return sum(1 for p in DEST.rglob("*") if p.is_file()) >= FILES


def check() -> int:
    if not DEST.is_dir():
        print(f"없다 — {DEST}\n  받으려면: python tools/get_kfps.py")
        return 1
    n = sum(1 for p in DEST.rglob("*") if p.is_file())
    size = sum(p.stat().st_size for p in DEST.rglob("*") if p.is_file())
    print(f"있다 — {DEST} ({n:,}파일 · {size / 1e6:.1f}MB)"
          + ("" if n >= FILES else f"  ※ {FILES:,}파일이어야 한다 — 다시 받을 것"))
    return 0 if n >= FILES else 1


def verify() -> int:
    """받아 둔 것이 **고정 커밋 그대로인가**."""
    if not DEST.is_dir():
        print(f"없다 — {DEST}")
        return 1
    n, got = aggregate(DEST)
    ok = n == FILES and got == AGG
    print(f"{'맞다' if ok else '어긋난다'} — {n:,}파일 · {got[:16]}…")
    if not ok:
        print(f"  기대: {FILES:,}파일 · {AGG[:16]}…\n"
              f"  다시 받으려면: python tools/get_kfps.py")
    return 0 if ok else 1


def _fetch_git(work: Path, log) -> Path | None:
    """git 부분 체크아웃 — 이 폴더의 blob만 받는다 (36MB)."""
    git = shutil.which("git")
    if not git:
        return None
    cmds = [
        [git, "init", "-q", "."],
        [git, "remote", "add", "origin", f"https://github.com/{REPO}"],
        [git, "sparse-checkout", "init", "--no-cone"],
        [git, "sparse-checkout", "set", SUB],
        [git, "fetch", "-q", "--filter=blob:none", "--depth", "1", "origin", PIN],
        [git, "checkout", "-q", "FETCH_HEAD"],
    ]
    for cmd in cmds:
        r = subprocess.run(cmd, cwd=work, capture_output=True,
                           encoding="utf-8", errors="replace")
        if r.returncode != 0:
            log(f"  git 실패 ({' '.join(cmd[1:3])}) — {r.stderr.strip()[:200]}")
            return None
    got = work / SUB
    return got if got.is_dir() else None


def _fetch_zip(work: Path, log) -> Path | None:
    """git이 없을 때 — 커밋 zip을 받아 이 폴더만 푼다.

    codeload는 부분 다운로드를 안 주므로 통째로 받는다 (250MB급). 그래서
    git이 있으면 그쪽이 먼저다.
    """
    blob = work / "kfps.zip"
    log(f"  zip을 받는 중… ({ZIP_URL})")
    req = urllib.request.Request(ZIP_URL, headers={"User-Agent": "ForzaSqueegee"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r, blob.open("wb") as f:  # noqa: S310
            shutil.copyfileobj(r, f, 1 << 20)
    except OSError as e:
        log(f"  못 받았다 — {e}")
        return None
    out = work / "unzipped"
    marker = f"/{SUB}/"
    with zipfile.ZipFile(blob) as z:
        names = [n for n in z.namelist() if marker in n and not n.endswith("/")]
        if not names:
            log(f"  zip 안에 {SUB}가 없다")
            return None
        head = names[0][: names[0].index(marker) + len(marker)]
        for n in names:
            target = out / n[len(head):]
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(n) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    blob.unlink(missing_ok=True)
    return out


def fetch(log=print) -> int:
    """받아서 자리에 놓는다. 되돌리는 것은 종료 코드다 (0이면 됐다)."""
    log(f"KFPS 도형 리소스를 받는다 — {REPO} @ {PIN[:12]}")
    with tempfile.TemporaryDirectory(prefix="fs-kfps-") as tmp:
        work = Path(tmp)
        src = _fetch_git(work, log) or _fetch_zip(work, log)
        if src is None:
            log("받지 못했다 — 네트워크나 git을 확인할 것")
            return 1
        n, got = aggregate(src)
        if n != FILES or got != AGG:
            # **어긋나면 안 놓는다** — 서빙하는 파일이 고정 커밋 그대로여야
            # editor.js의 도형 포맷과 우리 타입코드 대조가 같은 판 위에 선다.
            log(f"받은 것이 고정 커밋과 다르다 ({n:,}파일 · {got[:16]}…) — 버린다")
            return 1
        DEST.parent.mkdir(parents=True, exist_ok=True)
        if DEST.exists():
            shutil.rmtree(DEST, ignore_errors=True)
        shutil.move(str(src), str(DEST))
    size = sum(p.stat().st_size for p in DEST.rglob("*") if p.is_file())
    log(f"받았다 — {DEST} ({FILES:,}파일 · {size / 1e6:.1f}MB)")
    return 0


def ensure(log=print) -> bool:
    """없으면 받는다 (있으면 아무것도 안 한다) — 부르는 쪽이 먼저 물어볼 것."""
    return True if have() else fetch(log=log) == 0


def main() -> int:
    ap = argparse.ArgumentParser(description="KFPS 편집기 도형 리소스를 받는다")
    ap.add_argument("--check", action="store_true", help="있는지만 본다")
    ap.add_argument("--verify", action="store_true", help="집계 SHA-256 대조")
    a = ap.parse_args()
    if a.check:
        return check()
    if a.verify:
        return verify()
    if have():
        print(f"이미 있다 — {DEST}  (다시 받으려면 폴더를 지우고 실행할 것)")
        return 0
    return fetch()


if __name__ == "__main__":
    sys.exit(main())
