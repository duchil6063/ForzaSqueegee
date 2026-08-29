r"""전용 파이썬(`runtime/`)에 꾸러미를 앉힌다 — `ForzaSqueegee.bat`의 둘째 단계.

배치가 python.org **임베더블 배포판**을 `runtime/`에 펴 놓고 이 스크립트를 그
파이썬으로 부른다. 여기서 하는 일:

1. `python312._pth`를 우리 것으로 바꾼다 — 저장소 뿌리와 `Lib\site-packages`를
   넣고 `import site`를 켠다. `._pth`가 있는 파이썬은 PYTHONPATH·PYTHONHOME·
   레지스트리를 **전부 무시**하므로, 이 PC에 딴 파이썬·딴 판 꾸러미가 뭐가
   깔려 있든 여기엔 안 닿는다 (환경이 달라 안 되는 일을 뿌리에서 끊는 자리).
2. pip이 없으면 못 박은 휠을 받아 SHA-256을 대조하고, 그 휠 안의 pip으로 pip
   자신을 앉힌다 (임베더블에는 ensurepip이 없다).
3. `pyproject.toml`의 `==` 핀 **그대로** 꾸러미를 앉힌다 — 판이 다르면 그 판으로
   바꾼다 (핀의 이유는 pyproject의 주석에 있다).
4. 판 대조 + 실제 임포트(`check_env.py --deep`)로 마친다 — DLL이 없어 못 뜨는
   경우까지 여기서 잡아 말해 준다.

표준 라이브러리만 쓴다 — 꾸러미가 아직 없는 파이썬에서 도는 자리다.
출력은 실제 콘솔이 받으므로 한국어가 안전하다 (파이프로 물리면 replace로 간다).
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# pip 부트스트랩 휠 — 임베더블에는 pip이 없어 이것부터 앉힌다. 휠은 zip이라
# sys.path에 얹으면 그 안의 pip이 그대로 돌고, 그 pip으로 자신을 설치한다.
_PIP_FILE = "pip-26.2.1-py3-none-any.whl"
_PIP_URL = ("https://files.pythonhosted.org/packages/"
            "f3/6e/1736e5b4ae2b778ef2f81c47d797de9f891d4d8acb047a24ca37a60294dd/"
            + _PIP_FILE)
_PIP_SHA = "71138adf1f4ca900cdb7d289c21b7494329f2332b6d85f0e1c42108c0384ed3e"


def _say(*a: object) -> None:
    try:
        print(*a, flush=True)
    except UnicodeEncodeError:      # 파이프가 좁은 코드페이지일 때
        print(str(a).encode("ascii", "replace").decode(), flush=True)


def _runtime_dir() -> Path | None:
    """지금 도는 파이썬이 임베더블 `runtime/`인가 — 아니면 손대지 않는다.

    시스템 파이썬으로 잘못 부르면 **그쪽에** 꾸러미를 깔아 버린다. `._pth`
    파일은 임베더블에만 있으므로 그것으로 가른다.
    """
    d = Path(sys.executable).resolve().parent
    return d if any(d.glob("python3*._pth")) else None


def _fix_pth(rt: Path) -> None:
    """`._pth`를 우리 배치로 — 뿌리(`..`)·site-packages·`import site`."""
    pth = next(rt.glob("python3*._pth"))
    want = "\n".join((pth.name[:-len("._pth")] + ".zip",
                      ".", r"Lib\site-packages", "..", "import site", ""))
    if pth.read_text(encoding="utf-8", errors="replace") != want:
        pth.write_text(want, encoding="ascii")


def _download(url: str, dst: Path, sha: str) -> bool:
    req = urllib.request.Request(url, headers={"User-Agent": "ForzaSqueegee"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:      # noqa: S310
            blob = r.read()
    except OSError as e:
        _say(f"  못 받았다 - {e}")
        return False
    got = hashlib.sha256(blob).hexdigest()
    if got != sha:
        _say(f"  SHA-256이 다르다 - {got[:16]}... != {sha[:16]}...")
        _say("  (프록시가 내용을 바꿔치기하는 회선에서 이렇게 된다)")
        return False
    dst.write_bytes(blob)
    return True


def _ensure_pip(rt: Path) -> bool:
    """pip이 서게 한다 — 대조는 서브프로세스로 (지금 프로세스는 옛 경로다)."""
    py = str(rt / "python.exe")
    if subprocess.run([py, "-c", "import pip"], capture_output=True).returncode == 0:
        return True
    _say(f"  pip을 받는다 - {_PIP_FILE} (약 2MB)")
    whl = rt / _PIP_FILE
    if not _download(_PIP_URL, whl, _PIP_SHA):
        return False
    r = subprocess.run([py, "-c",
                        "import sys, runpy; sys.path.insert(0, sys.argv.pop(1)); "
                        "runpy.run_module('pip', run_name='__main__')",
                        str(whl), "install", "-q", "--no-warn-script-location",
                        str(whl)])
    whl.unlink(missing_ok=True)
    return r.returncode == 0


def main() -> int:
    rt = _runtime_dir()
    if rt is None:
        _say("이 스크립트는 runtime\\python.exe로 부르는 자리다 - ForzaSqueegee.bat이 부른다.")
        return 2
    _fix_pth(rt)

    sys.path.insert(0, str(ROOT / "tools"))
    import check_env

    pins, _, _ = check_env._pins()
    _say("꾸러미를 앉힌다 - " + " ".join(f"{n} {v}" for n, v in pins))
    _say("  내려받기 약 310MB, 앉히면 약 890MB - 전부 이 폴더의 runtime/ 안이다.")

    py = str(rt / "python.exe")
    if not _ensure_pip(rt):
        _say("pip을 못 세웠다 - 인터넷 연결을 확인하고 다시 실행할 것.")
        return 1
    r = subprocess.run([py, "-m", "pip", "install", "--no-warn-script-location",
                        "--disable-pip-version-check",
                        *[f"{n}=={v}" for n, v in pins]])
    if r.returncode != 0:
        _say("꾸러미를 못 앉혔다 - 위 pip 메시지를 볼 것 (회선·디스크·프록시).")
        return 1

    # 판 대조 + 실제 임포트 — 새 ._pth로 뜬 서브프로세스가 잰다.
    r = subprocess.run([py, str(ROOT / "tools" / "check_env.py"), "--deep"])
    if r.returncode != 0:
        _say("앉히긴 했는데 검사에 걸렸다 - 위 메시지를 볼 것.")
        return 1
    _say("다 됐다 - 다음 실행부터는 이 과정을 건너뛴다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
