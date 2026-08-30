"""신경망 모델을 **쓰기 직전에 받아 둔다** — `models/`의 자리와 임자.

모델 넷은 합쳐 330MB가 넘어 저장소에 넣지 않는다. 목록·크기·SHA-256은
저장소 뿌리의 `release.json`에 있고 — 이 프로젝트가 내는 릴리스의 자산 목록이다
(`tools/get_fls.py`도 같은 파일을 읽는다) — 여기서 그것을 읽어 필요한 것만
받는다. 받은 파일은 `models/` 그대로 앉으므로 **두 번째 실행부터는 네트워크를
안 탄다**.

- 받는 자리: `models/<파일>` (`FS_MODEL_DIR`로 옮길 수 있다)
- 받는 곳: `release.json`의 `repo`+`tag` 릴리스 (`FS_MODEL_BASE_URL`이 이긴다 —
  미러를 물릴 자리)
- 받고 나서 SHA-256을 대조한다 — 어긋나면 지우고 실패로 친다 (반쯤 받은
  파일이 남아 onnxruntime이 엉뚱한 데서 죽는 일이 없다)

**실패는 경고만이다** (한 버튼 원칙). 네트워크가 없거나 릴리스가 안 열리면
`ensure()`가 None을 주고, 부르는 쪽은 모델이 없을 때의 길로 간다 — 배경
제거는 건너뛰고, 선화는 고전 방식으로 물러서고, SR은 큐빅으로 간다.
`line` 노선만은 선화 모델이 필수라 거기서 멈춘다.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import urllib.error
import urllib.request
from pathlib import Path

from .i18n import msg

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "release.json"
_TIMEOUT = 60
_CHUNK = 1 << 20
_FAILED: set[str] = set()        # 한 번 실패한 것 — 이번 실행에서 다시 안 조른다


def model_dir() -> Path:
    """모델이 앉는 자리."""
    d = os.environ.get("FS_MODEL_DIR")
    return Path(d) if d else ROOT / "models"


def _manifest() -> dict:
    try:
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"models": {}}


def entries() -> dict[str, dict]:
    """모델 목록 (`release.json`의 `models`)."""
    return _manifest().get("models") or {}


def path(name: str) -> Path:
    """그 모델이 앉을 자리 (있든 없든)."""
    ent = entries().get(name)
    return model_dir() / (ent["file"] if ent else f"{name}.onnx")


def have(name: str) -> bool:
    """이미 받아 뒀나 — 크기까지 맞아야 한다 (반쯤 받은 파일을 거른다)."""
    p = path(name)
    if not p.is_file():
        return False
    want = (entries().get(name) or {}).get("size")
    return want is None or p.stat().st_size == int(want)


def url(name: str) -> str | None:
    """그 모델을 받을 주소. `FS_MODEL_BASE_URL`이 릴리스(`base`+`tag`)를 이긴다."""
    ent = entries().get(name)
    if not ent:
        return None
    base = os.environ.get("FS_MODEL_BASE_URL")
    if base:
        return f"{base.rstrip('/')}/{ent['file']}"
    m = _manifest()
    repo, tag = m.get("repo"), m.get("tag")
    if not repo or not tag:
        return None
    return f"{repo.rstrip('/')}/releases/download/{tag}/{ent['file']}"


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for blk in iter(lambda: f.read(_CHUNK), b""):
            h.update(blk)
    return h.hexdigest()


def _download(src: str, dst: Path, size: int, log) -> None:
    """`.part`로 받아 해시를 맞춰 보고 제자리에 놓는다."""
    tmp = dst.with_suffix(dst.suffix + ".part")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(src, headers={"User-Agent": "ForzaSqueegee"})
    got = 0
    step = max(size // 10, 1) if size else 1 << 24
    mark = step
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:   # noqa: S310
        with tmp.open("wb") as f:
            while True:
                blk = r.read(_CHUNK)
                if not blk:
                    break
                f.write(blk)
                got += len(blk)
                if got >= mark:
                    mark += step
                    log(f"    {got / 1e6:,.0f}MB"
                        + (f" / {size / 1e6:,.0f}MB" if size else ""))
    if size and got != size:
        tmp.unlink(missing_ok=True)
        raise OSError(msg("크기가 다르다 — 받은 {got:,} / 적힌 {want:,}",
                          got=got, want=int(size)))
    shutil.move(str(tmp), str(dst))


def ensure(name: str, log=print) -> Path | None:
    """그 모델을 쓸 수 있게 만든다. 자리를 주거나, 못 하면 None.

    이미 있으면 곧바로 자리를 준다 (해시는 받을 때만 본다 — 매번 320MB를
    다시 읽으면 실행이 느려진다).
    """
    ent = entries().get(name)
    if ent is None:
        return None
    dst = path(name)
    if have(name):
        return dst
    if name in _FAILED:
        return None
    src = url(name)
    if not src:
        log(msg("  경고: {file}을 받을 곳이 없다 (release.json)",
                file=ent['file']))
        _FAILED.add(name)
        return None
    if os.environ.get("FS_NO_MODEL_FETCH"):
        log(msg("  경고: {file}이 없다 (FS_NO_MODEL_FETCH — 받지 않는다)",
                file=ent['file']))
        _FAILED.add(name)
        return None
    size = int(ent.get("size") or 0)
    log(msg("  모델을 받는다 — {file} ({mb:,.0f}MB, {what})",
            file=ent['file'], mb=size / 1e6, what=ent.get('what', '')))
    try:
        _download(src, dst, size, log)
        digest = _sha256(dst)
        if ent.get("sha256") and digest != ent["sha256"]:
            dst.unlink(missing_ok=True)
            raise OSError(msg("해시가 다르다 — {got}… ≠ {want}…",
                              got=digest[:16], want=ent['sha256'][:16]))
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        dst.with_suffix(dst.suffix + ".part").unlink(missing_ok=True)
        log(msg("  경고: {file}을 못 받았다 — {err}", file=ent['file'], err=e))
        log(msg("    직접 받아 {dir}에 두어도 된다: {url}",
                dir=model_dir(), url=src))
        _FAILED.add(name)
        return None
    log(msg("  받았다 — {name}", name=dst.name))
    return dst


def fetch_all(log=print) -> int:
    """목록의 모델을 다 받아 둔다. 못 받은 개수를 준다 (0이면 다 됐다)."""
    bad = 0
    todo = entries()
    log(msg("모델 {n}개를 확인한다 — {dir}", n=len(todo), dir=model_dir()))
    for name, ent in todo.items():
        if have(name):
            log(msg("  있다 — {file}", file=ent['file']))
            continue
        if ensure(name, log=log) is None:
            bad += 1
    return bad


def verify(log=print) -> int:
    """받아 둔 모델의 SHA-256을 전수 대조한다. 어긋난 개수를 준다."""
    bad = 0
    for name, ent in entries().items():
        p = path(name)
        if not p.is_file():
            log(msg("  없다 — {file}", file=ent['file']))
            continue
        digest = _sha256(p)
        ok = digest == ent.get("sha256")
        log(msg("  {status} — {file} {digest}…",
                status=msg("맞다") if ok else msg("다르다"),
                file=ent['file'], digest=digest[:16]))
        bad += 0 if ok else 1
    return bad
