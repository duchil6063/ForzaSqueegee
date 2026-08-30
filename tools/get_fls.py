r"""FLS(ForzaLiveryStudio) 편집기를 `vendor/fls-editor/`에 받아 둔다.

FLS는 AGPL-3.0이라 **이 저장소에 바이너리를 넣지 않는다** — 받아서 옆에 두고,
우리 프로그램은 그것을 **바깥 프로그램으로 실행만** 한다 (링크되지 않으므로
파생물이 아니다).

받을 곳이 둘이다:

- **우리 릴리스** (기본) — [Itasha] 메뉴가 든 우리 빌드. `release.json`의
  `fls`가 그 자리와 SHA-256을 쥔다. AGPL이 요구하는 대응 소스가 **같은
  릴리스에** 함께 올라가 있다 (`tools/fls_build.py --package`가 둘을 짓는다).
- **업스트림 공식 릴리스** (`--official`) — 도형 편집·3D 미리보기는 다 되지만
  [Itasha] 메뉴는 없다. 우리 릴리스가 아직 없으면 여기로 물러난다.

    python tools/get_fls.py              # 우리 빌드 ([Itasha] 메뉴 있음)
    python tools/get_fls.py --official   # 업스트림 공식 릴리스
    python tools/get_fls.py --tag 1.2.1  # 업스트림 판을 못 박아
    python tools/get_fls.py --check      # 지금 있는 것만 확인

받는 것이 없어도 **내보내기는 다 된다** — 게임이 읽는 것은 파일이지 편집기가
아니다. FLS는 그 파일을 눈으로 고칠 때 쓴다.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import sys
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

DEST = ROOT / "vendor" / "fls-editor"
REPO = "Arstz/ForzaLiveryStudio"
API = f"https://api.github.com/repos/{REPO}/releases"
EXE = "ForzaLiveryStudio.exe"
RELEASE = ROOT / "release.json"


def _ours() -> tuple[str, dict] | None:
    """우리 릴리스의 (내려받을 주소, 자산 기술). 적혀 있지 않으면 None."""
    try:
        r = json.loads(RELEASE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    fls, repo, tag = r.get("fls"), r.get("repo"), r.get("tag")
    if not fls or not repo or not tag or not fls.get("file"):
        return None
    return f"{repo.rstrip('/')}/releases/download/{tag}/{fls['file']}", fls


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "ForzaSqueegee"})
    with urllib.request.urlopen(req, timeout=120) as r:      # noqa: S310
        return r.read()


def _release(tag: str | None) -> dict:
    url = f"{API}/tags/{tag}" if tag else f"{API}/latest"
    return json.loads(_get(url).decode("utf-8"))


def _asset(rel: dict) -> dict:
    for a in rel.get("assets") or []:
        name = str(a.get("name", "")).lower()
        if name.endswith(".zip") and ("win" in name or "x64" in name):
            return a
    raise RuntimeError("릴리스에 윈도우 zip이 없다")


def _clear() -> None:
    """받기 전에 자리를 비운다 — 남은 파일이 섞이면 **원본 그대로**가 아니다."""
    if not DEST.is_dir():
        return
    for item in DEST.iterdir():
        if item.name == "README.md":
            continue
        shutil.rmtree(item, ignore_errors=True) if item.is_dir() else item.unlink()


def check() -> int:
    exe = DEST / EXE
    if not exe.is_file():
        print(f"없다 — {exe}")
        return 1
    stamp = DEST / "VERSION.json"
    ver = ""
    if stamp.is_file():
        try:
            ver = json.loads(stamp.read_text(encoding="utf-8")).get("tag", "")
        except ValueError:
            pass
    print(f"있다 — {exe} ({exe.stat().st_size:,}바이트"
          + (f" · 판 {ver}" if ver else "") + ")")
    return 0


def _unpack(blob: bytes, stamp: dict) -> int:
    """받은 zip을 `vendor/fls-editor/`에 편다 (`Release/` 한 겹은 벗긴다)."""
    DEST.mkdir(parents=True, exist_ok=True)
    _clear()
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        names = z.namelist()
        root = ""
        for n in names:
            if n.lower().endswith(EXE.lower()):
                root = n[: -len(EXE)]
                break
        for n in names:
            if n.endswith("/") or not n.startswith(root):
                continue
            target = DEST / n[len(root):]
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(n) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    (DEST / "VERSION.json").write_text(
        json.dumps(stamp, ensure_ascii=False, indent=1), encoding="utf-8")
    return check()


def get_ours() -> int:
    """우리 빌드를 받는다 ([Itasha] 메뉴). 자리가 없거나 못 받으면 1."""
    got = _ours()
    if got is None:
        print("  release.json에 우리 빌드가 안 적혀 있다")
        return 1
    url, fls = got
    print(f"  우리 빌드 — {fls['file']} ({int(fls.get('size', 0)):,}바이트)")
    try:
        blob = _get(url)
    except Exception as e:                      # noqa: BLE001 — 없으면 물러난다
        print(f"  못 받았다 — {e}")
        return 1
    sha = hashlib.sha256(blob).hexdigest()
    if fls.get("sha256") and sha != fls["sha256"]:
        print(f"  해시가 다르다 — {sha[:16]}… != {fls['sha256'][:16]}…")
        return 1
    print(f"  SHA-256 {sha}")
    return _unpack(blob, {"source": "fork", "asset": fls["file"], "sha256": sha,
                          "url": url, "pin": fls.get("pin", ""),
                          "corresponding_source": fls.get("source", "")})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--official", action="store_true",
                    help="업스트림 공식 릴리스를 받는다 ([Itasha] 메뉴 없음)")
    ap.add_argument("--tag", default=None,
                    help="업스트림 릴리스 태그 (기본: 최신) — --official 쪽 인자")
    ap.add_argument("--check", action="store_true", help="받지 않고 확인만")
    ap.add_argument("--force", action="store_true", help="이미 있어도 다시 받는다")
    a = ap.parse_args()
    if a.check:
        return check()
    if (DEST / EXE).is_file() and not a.force:
        print("이미 있다 (다시 받으려면 --force)")
        return check()
    if not a.official and not a.tag:
        print("릴리스 확인 — 우리 빌드 ([Itasha] 메뉴)")
        if get_ours() == 0:
            return 0
        print("  → 업스트림 공식 릴리스로 물러난다 ([Itasha] 메뉴는 없다)")
    print(f"릴리스 확인 — {REPO}")
    rel = _release(a.tag)
    asset = _asset(rel)
    tag = str(rel.get("tag_name") or a.tag or "?")
    print(f"  판 {tag} · {asset['name']} ({int(asset['size']):,}바이트) 받는 중…")
    blob = _get(asset["browser_download_url"])
    sha = hashlib.sha256(blob).hexdigest()
    print(f"  SHA-256 {sha}")
    return _unpack(blob, {"source": "official", "repo": REPO, "tag": tag,
                          "asset": asset["name"], "sha256": sha,
                          "published": rel.get("published_at", "")})


if __name__ == "__main__":
    sys.exit(main())
