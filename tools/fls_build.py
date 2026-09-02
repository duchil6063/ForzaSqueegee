r"""FLS 포크를 **소스에서 짓는다** — 툴체인 준비부터 `vendor/fls-editor/` 배포까지.

FLS(AGPL-3.0)에 우리 이타샤 기능을 붙이려면 소스를 고쳐 빌드해야 한다. 고친
내용은 저장소에 **패치로** 산다 (`tools/fls-patch/*.patch`) — 업스트림 통째
사본 대신 고정 커밋 + 패치라, 무엇을 바꿨는지가 그대로 보이고 업스트림 갱신도
쉽다. AGPL이 요구하는 "대응 소스"는 고정 커밋 + 이 패치들이다.

    python tools/fls_build.py --setup     # 툴체인 (Qt·MinGW·zlib) — 한 번만
    python tools/fls_build.py             # 소스 동기화 → 패치 → 빌드 → 배포
    python tools/fls_build.py --check     # 지금 상태만 본다
    python tools/fls_build.py --package   # 릴리스에 올릴 두 벌(바이너리·대응 소스)

**릴리스에 올리는 것은 두 벌이다** (`--package`). AGPL은 고친 바이너리를 주면
대응 소스를 **같이** 주라고 하는데, 남의 저장소가 살아 있기를 기대하는 것으로는
그 의무가 안 끝난다("as long as needed to satisfy these requirements"). 그래서
업스트림 고정 커밋에 우리 패치를 얹은 **완전한 소스 트리를 우리가 같은 릴리스에
올린다** — 바이너리 옆 자리다.

**관리자 권한이 필요 없다.** MSVC 대신 포터블 MinGW-w64를, Qt는 공식 바이너리를
`aqtinstall`로 받아 전부 `FS_DEV_ROOT`(기본 `D:/dev`) 아래 둔다 — 시스템에
아무것도 안 깔리고 그 폴더를 지우면 원상복구다. CUDA는 끈다 (선택 기능이다).

빌드가 끝나면 `vendor/fls-editor/`에 실행 파일과 Qt 런타임이 선다 — 제품이
[FLS 편집기에서 편집]으로 띄우는 바로 그 자리다.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
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

PATCH_DIR = ROOT / "tools" / "fls-patch"
VENDOR = ROOT / "vendor" / "fls-editor"

DEV = Path(os.environ.get("FS_DEV_ROOT", "D:/dev"))
QT_VER = "6.8.2"
QT_ARCH = "win64_mingw"
QT_DIR = DEV / "Qt" / QT_VER / "mingw_64"
MINGW_BIN = DEV / "Qt" / "Tools" / "mingw1310_64" / "bin"
PREFIX = DEV / "prefix"                      # zlib 설치 자리
SRC = DEV / "fls-src"                        # 포크 작업 트리
BUILD = DEV / "build" / "fls"
ZLIB_SRC = DEV / "src" / "zlib-1.3.1"
ZLIB_URL = "https://github.com/madler/zlib/archive/refs/tags/v1.3.1.tar.gz"

UPSTREAM = "https://github.com/Arstz/ForzaLiveryStudio.git"
# 패치가 얹히는 업스트림 고정점 (태그 1.2.1). 올릴 때는 여기와 패치를 같이 옮긴다.
PIN = "5e890e1766eedd884cfa0d1234e135431bb7cdde"

# 우리가 고친 것 — AGPL §5(a)가 요구하는 "무엇을 고쳤나"다. `_notice`가 이대로
# 바이너리 옆 고지에 싣고, 개수가 `tools/fls-patch/*.patch`와 어긋나면 멈춘다.
# 패치를 더하거나 뺄 때 여기도 같이 고친다 (순서는 패치 번호순).
CHANGES = (
    "**build** — Qt 플러그인 자리를 vcpkg 배치 가정 없이 찾도록 고침",
    "**itasha** — [Itasha] 메뉴(리버리 한 벌 짓기) · 창 없는 면 기하 덤프\n"
    "   (`--itasha-dump`) · [Edit → Split Selection at a Line]",
    "**i18n** — 한국어 UI(내장 영→한 대응표)와 언어 설정, 한국어가 기본",
    "**itasha** — [Auto Decoration...] 창 하나로 구성 계열 · 무늬 계열 ·\n"
    "   바탕 도색 · 캐릭터 이름 글자를 짜서 엔진에 한 번에 넘긴다",
    "**itasha** — [Auto Decoration...] 창의 실린 그림 표 — 차에 실린 덩어리마다\n"
    "   역할(주역 · 보조 · 로고 · 글자 · 그대로)",
    "**itasha** — [Auto Decoration...] 창의 로고(내장 워터마크 · 로고 이미지 ·\n"
    "   자리)와 좌우(한쪽에만 있으면 반대편에) 묶음",
)

CMAKE_ARGS = [
    "-DCMAKE_BUILD_TYPE=Release",
    "-DFLS_ENABLE_CUDA=OFF",          # 선택 기능 — 툴킷 없이 짓는다
    "-DFLS_BUILD_TESTS=OFF",
    "-DFLS_BUILD_HELPER_TOOLS=OFF",
    "-DFLS_BUILD_LIVERY_COMPARE=OFF",
    "-DFLS_ENABLE_IMGGEN_MENU=OFF",
    "-DFLS_PRIVACY_POLICY=ON",
    "-DENFORCE_SHAPE_LIMITS=ON",
]


def sh(cmd: list[str], cwd: Path | None = None, env: dict | None = None,
       quiet: bool = False) -> subprocess.CompletedProcess:
    if not quiet:
        print("  $ " + " ".join(str(c) for c in cmd))
    return subprocess.run([str(c) for c in cmd], cwd=str(cwd) if cwd else None,
                          env=env, check=True, text=True,
                          capture_output=quiet)


def build_env() -> dict:
    """MinGW·CMake·Ninja가 앞에 서는 환경 — 시스템 PATH를 안 건드린다."""
    env = dict(os.environ)
    scripts = Path(sys.executable).parent / "Scripts"
    env["PATH"] = os.pathsep.join(
        [str(MINGW_BIN), str(QT_DIR / "bin"), str(scripts), env.get("PATH", "")])
    return env


def _tool(name: str) -> Path | None:
    scripts = Path(sys.executable).parent / "Scripts"
    for cand in (scripts / f"{name}.exe", MINGW_BIN / f"{name}.exe"):
        if cand.is_file():
            return cand
    found = shutil.which(name)
    return Path(found) if found else None


def check() -> int:
    rows = [("MinGW g++", MINGW_BIN / "g++.exe"),
            ("Qt6", QT_DIR / "lib" / "cmake" / "Qt6" / "Qt6Config.cmake"),
            ("Qt WebP 플러그인", QT_DIR / "plugins" / "imageformats" / "qwebp.dll"),
            ("zlib", PREFIX / "lib" / "libzlib.dll.a"),
            ("cmake", _tool("cmake")), ("ninja", _tool("ninja")),
            ("포크 소스", SRC / "CMakeLists.txt"),
            ("빌드본", VENDOR / "ForzaLiveryStudio.exe")]
    bad = 0
    for label, path in rows:
        ok = path is not None and Path(path).exists()
        bad += not ok
        print(f"  {'OK ' if ok else '** '}{label:18s} {path}")
    print(f"  패치 {len(sorted(PATCH_DIR.glob('*.patch')))}개 · 업스트림 고정 {PIN[:12]}")
    return 1 if bad else 0


# ────────────────────────────── 준비 ──────────────────────────────


def setup() -> None:
    """Qt·MinGW·zlib — 없는 것만 받아 짓는다 (여러 번 돌려도 안전하다)."""
    print("[툴체인]")
    if not (MINGW_BIN / "g++.exe").is_file():
        print("  MinGW-w64 13.1 받는 중…")
        sh([sys.executable, "-m", "aqt", "install-tool", "windows", "desktop",
            "tools_mingw1310", "--outputdir", DEV / "Qt"])
    if not (QT_DIR / "lib" / "cmake" / "Qt6" / "Qt6Config.cmake").is_file():
        print(f"  Qt {QT_VER} ({QT_ARCH}) 받는 중… (약 1.2GB)")
        sh([sys.executable, "-m", "aqt", "install-qt", "windows", "desktop",
            QT_VER, QT_ARCH, "-m", "qtimageformats", "--outputdir", DEV / "Qt"])
    for tool in ("cmake", "ninja"):
        if _tool(tool) is None:
            sh([sys.executable, "-m", "pip", "install", "--quiet", tool])
    if not (PREFIX / "lib" / "libzlib.dll.a").is_file():
        print("  zlib 1.3.1 짓는 중…")
        _fetch_zlib()
        env = build_env()
        sh([_tool("cmake"), "-G", "Ninja", "-S", ZLIB_SRC,
            "-B", DEV / "build" / "zlib", "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_INSTALL_PREFIX={PREFIX}", "-DZLIB_BUILD_EXAMPLES=OFF"],
           env=env, quiet=True)
        sh([_tool("cmake"), "--build", DEV / "build" / "zlib",
            "--target", "install"], env=env, quiet=True)
    print("  준비 끝")


def _fetch_zlib() -> None:
    import tarfile
    import urllib.request

    ZLIB_SRC.parent.mkdir(parents=True, exist_ok=True)
    tgz = ZLIB_SRC.parent / "zlib.tar.gz"
    if not tgz.is_file():
        urllib.request.urlretrieve(ZLIB_URL, tgz)      # noqa: S310
    with tarfile.open(tgz) as t:
        t.extractall(ZLIB_SRC.parent, filter="data")


# ────────────────────────────── 소스·패치 ──────────────────────────────


def sync_source(keep_local: bool) -> None:
    """업스트림 고정 커밋을 꺼내고 우리 패치를 얹는다.

    `keep_local`이면 작업 트리를 안 건드린다 — 포크를 손으로 고치는 중일 때
    (그 상태에서 `--export-patches`로 패치를 다시 뽑는다)."""
    print("[소스]")
    if not (SRC / ".git").is_dir():
        print(f"  클론 {UPSTREAM}")
        sh(["git", "clone", "--quiet", UPSTREAM, SRC])
    if keep_local:
        head = subprocess.run(["git", "-C", str(SRC), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
        print(f"  작업 트리 그대로 (HEAD {head})")
        return
    sh(["git", "-C", SRC, "fetch", "--quiet", "origin"])
    sh(["git", "-C", SRC, "checkout", "--quiet", "-B", "itasha", PIN])
    patches = sorted(PATCH_DIR.glob("*.patch"))
    if patches:
        print(f"  패치 {len(patches)}개 얹는 중")
        sh(["git", "-C", SRC, "am", "--quiet", *[str(p) for p in patches]])
    head = subprocess.run(["git", "-C", str(SRC), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    print(f"  HEAD {head} (고정 {PIN[:12]} + 패치 {len(patches)})")


def export_patches() -> None:
    """작업 트리의 커밋들을 다시 패치로 뽑는다 (고정점 이후 전부)."""
    PATCH_DIR.mkdir(parents=True, exist_ok=True)
    for old in PATCH_DIR.glob("*.patch"):
        old.unlink()
    sh(["git", "-C", SRC, "format-patch", "--no-signature", f"{PIN}..HEAD",
        "-o", PATCH_DIR])
    got = sorted(PATCH_DIR.glob("*.patch"))
    print(f"  패치 {len(got)}개 뽑았다 → {PATCH_DIR.relative_to(ROOT)}")


# ────────────────────────────── 빌드·배포 ──────────────────────────────


def build(jobs: int | None) -> None:
    print("[빌드]")
    env = build_env()
    sh([_tool("cmake"), "-G", "Ninja", "-S", SRC, "-B", BUILD,
        f"-DCMAKE_PREFIX_PATH={QT_DIR.as_posix()};{PREFIX.as_posix()}",
        *CMAKE_ARGS], env=env, quiet=True)
    cmd = [_tool("cmake"), "--build", BUILD]
    if jobs:
        cmd += ["-j", str(jobs)]
    sh(cmd, env=env)
    exe = BUILD / "ForzaLiveryStudio.exe"
    if not exe.is_file():
        raise SystemExit("빌드가 실행 파일을 안 냈다")
    print(f"  {exe} ({exe.stat().st_size:,}바이트)")


def deploy() -> None:
    """Qt 런타임을 붙이고 `vendor/fls-editor/`로 옮긴다."""
    print("[배포]")
    env = build_env()
    sh([QT_DIR / "bin" / "windeployqt.exe", "--release", "--no-translations",
        "--no-system-d3d-compiler", "--no-opengl-sw", "--compiler-runtime",
        BUILD / "ForzaLiveryStudio.exe"], env=env, quiet=True)
    # zlib을 공유로 지었으므로 그 DLL도 옆에 서야 한다 (없으면 조용히 안 뜬다)
    for dll in (PREFIX / "bin").glob("*.dll"):
        shutil.copy2(dll, BUILD / dll.name)
    # 편집기 자체 문자열은 내장 표(패치 0003)가 옮기고, Qt 표준 대화상자
    # (확인/취소·색 고르기)는 Qt의 한국어 번역이 옮긴다. windeployqt는
    # --no-translations라(전 언어를 다 실어 온다) 한국어 한 벌만 손으로 싣는다.
    qt_ko = QT_DIR / "translations" / "qtbase_ko.qm"
    if qt_ko.is_file():
        (BUILD / "translations").mkdir(exist_ok=True)
        shutil.copy2(qt_ko, BUILD / "translations" / qt_ko.name)
    VENDOR.mkdir(parents=True, exist_ok=True)
    # **떠 있는 편집기를 밟지 않는다.** 실행 중이면 exe가 잠겨 있어 지우다가
    # 반쯤 비운 자리를 남긴다 — 지우기 전에 알아채고 사람에게 닫으라고 한다.
    exe = VENDOR / "ForzaLiveryStudio.exe"
    if exe.is_file():
        try:
            with exe.open("r+b"):             # 도는 실행 파일은 쓰기로 못 연다
                pass
        except OSError as e:
            raise SystemExit(
                f"편집기가 떠 있어 배포를 못 한다 — {exe}\n"
                f"  창을 닫고 다시 돌리세요 ({e.strerror or e})") from e
    keep = {"README.md"}          # 우리 글 — 빌드가 덮지 않는다
    for item in VENDOR.iterdir():
        if item.name in keep:
            continue
        shutil.rmtree(item, ignore_errors=True) if item.is_dir() else item.unlink()
    for item in BUILD.iterdir():
        # 빌드 부산물은 안 옮긴다 — `*_autogen`은 moc/uic가 남긴 것이라
        # 실행에 필요 없고 꾸러미만 불린다 (실측 60여 파일).
        if item.name in {"CMakeFiles", "CMakeCache.txt", ".ninja_deps",
                         ".ninja_log", "build.ninja", "cmake_install.cmake"} \
                or item.name.endswith("_autogen"):
            continue
        if item.is_dir():
            shutil.copytree(item, VENDOR / item.name, dirs_exist_ok=True)
        elif item.suffix.lower() in {".exe", ".dll"} or item.name == "assets":
            shutil.copy2(item, VENDOR / item.name)
    # **라이선스 전문이 바이너리를 따라다녀야 한다** (AGPL-3.0). 대응 소스는
    # 고정 커밋 + 패치 묶음이고 그 자리는 README가 적는다 — 여기서는 원문을 옮긴다.
    src_license = SRC / "LICENSE"
    if src_license.is_file():
        shutil.copy2(src_license, VENDOR / "LICENSE")
    (VENDOR / "VERSION.json").write_text(
        f'{{\n "source": "fork",\n "upstream": "{UPSTREAM}",\n'
        f' "pin": "{PIN}",\n "patches": {len(sorted(PATCH_DIR.glob("*.patch")))}\n}}\n',
        encoding="utf-8")
    n = sum(1 for _ in VENDOR.rglob("*") if _.is_file())
    print(f"  {VENDOR} — 파일 {n}개")


# ────────────────────────────── 릴리스 꾸러미 ──────────────────────────────

# 바이너리에 딸려 가는 제3자 라이선스 전문 — 자리는 툴체인 안이다.
# (Qt는 aqt 설치본에 전문이 없어 고지로 가리킨다 — 아래 `_notice`)
BUNDLED_LICENSES = [
    ("gcc-COPYING.RUNTIME", MINGW_BIN.parent / "licenses" / "gcc" / "COPYING.RUNTIME"),
    ("gcc-COPYING3", MINGW_BIN.parent / "licenses" / "gcc" / "COPYING3"),
    ("zlib-LICENSE", ZLIB_SRC / "LICENSE"),
]


def _head() -> str:
    return subprocess.run(["git", "-C", str(SRC), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()


def _notice(binary_zip: str, source_zip: str) -> str:
    """바이너리 옆에 서는 고지 — AGPL §5(a)·§6(d)가 요구하는 것을 한 장에."""
    patches = sorted(PATCH_DIR.glob("*.patch"))
    if len(patches) != len(CHANGES):
        raise SystemExit(
            f"고지가 실제 수정과 어긋난다 — 패치 {len(patches)}개인데 CHANGES는 "
            f"{len(CHANGES)}줄이다.\n"
            f"AGPL §5(a) 고지가 무엇을 고쳤는지 다 적어야 한다 — "
            f"`tools/fls_build.py`의 CHANGES를 맞추세요.")
    changed = "\n".join(f"{i}. {c}" for i, c in enumerate(CHANGES, 1))
    return f"""# ForzaLiveryStudio — ForzaSqueegee 수정판

이 프로그램은 [ForzaLiveryStudio](https://github.com/Arstz/ForzaLiveryStudio)
(© Arstz, **AGPL-3.0-or-later**)를 **수정한 판**입니다. 전문은 옆의 `LICENSE`.

## 무엇을 고쳤나 (AGPL-3.0 §5(a))

업스트림 고정 커밋 `{PIN}` (태그 1.2.1) 위에 다음을 얹었습니다:

{changed}

수정자는 ForzaSqueegee contributors이고, 날짜는 함께 실린 대응 소스의 각 커밋에
있습니다.

## 대응 소스 (AGPL-3.0 §6)

**같은 릴리스의 `{source_zip}`이 이 바이너리의 대응 소스 전부입니다** — 업스트림
고정 커밋에 위의 것을 모두 얹은 완전한 트리이고, 그것만으로 이 바이너리를 다시
지을 수 있습니다. 짓는 법은 그 안의 `BUILD.md`에 있습니다.

## 함께 실린 제3자 구성 요소

| | 라이선스 | 비고 |
|---|---|---|
| Qt 6.8.2 (`Qt6*.dll`, `plugins/`, `translations/qtbase_ko.qm`) | LGPL-3.0 — https://doc.qt.io/qt-6/lgpl.html | **동적 링크**입니다. 같은 판의 Qt로 DLL을 바꿔 끼워 다시 링크할 수 있습니다. 한국어 번역 파일(qttranslations 모듈)도 같은 LGPL-3.0 선택지로 실었습니다. 소스: https://download.qt.io/archive/qt/6.8/6.8.2/single/ |
| MinGW-w64 GCC 런타임 (`libgcc_s_seh-1.dll` · `libstdc++-6.dll` · `libwinpthread-1.dll`) | GPL-3.0 + GCC Runtime Library Exception (`licenses/gcc-*`) | 예외 조항이 어떤 프로그램과도 함께 배포하도록 허용합니다 |
| zlib 1.3.1 (`libzlib.dll`) | zlib 라이선스 (`licenses/zlib-LICENSE`) | 소스: https://github.com/madler/zlib/archive/refs/tags/v1.3.1.tar.gz |

## 원래 자리

이 꾸러미는 ForzaSqueegee가 함께 씁니다 (`vendor/fls-editor/`에 풀어 둡니다).
ForzaSqueegee 본체는 MIT이고 이 편집기를 **별개 프로세스로 실행만** 합니다 —
어느 쪽도 상대를 링크하지 않으므로 파생물이 아닙니다.

바이너리: `{binary_zip}`
"""


def _build_md() -> str:
    """대응 소스에 딸려 가는 빌드 설명 — 이것만으로 다시 지을 수 있어야 한다."""
    cont = " " + chr(92)
    args = (NL := chr(10)).join("        " + a + cont for a in CMAKE_ARGS)
    return f"""# 짓는 법

이 트리는 [ForzaLiveryStudio](https://github.com/Arstz/ForzaLiveryStudio)의
고정 커밋 `{PIN}`에 ForzaSqueegee의 패치를 얹은 것입니다 (AGPL-3.0-or-later).
`patches/`에 그 패치가 그대로 있어 무엇을 바꿨는지 볼 수 있습니다.

## 필요한 것

- Qt {QT_VER} ({QT_ARCH}) + qtimageformats
- MinGW-w64 13.1 (또는 호환 GCC)
- zlib 1.3.1 (공유 라이브러리)
- CMake · Ninja

## 명령

    cmake -G Ninja -S . -B build{cont}
        -DCMAKE_PREFIX_PATH="<Qt>;<zlib prefix>"{cont}
{args}
    cmake --build build
    windeployqt --release --no-translations --compiler-runtime build/ForzaLiveryStudio.exe

ForzaSqueegee 저장소가 있으면 이 전부를 한 명령이 합니다:

    python tools/fls_build.py --setup    # 툴체인
    python tools/fls_build.py            # 동기화 → 패치 → 빌드 → 배포
"""


def package() -> int:
    """릴리스에 올릴 두 벌 — 바이너리와 그 대응 소스."""
    import zipfile

    print("[꾸러미]")
    exe = VENDOR / "ForzaLiveryStudio.exe"
    if not exe.is_file():
        print(f"  빌드본이 없다 — {exe}\n  `python tools/fls_build.py`로 먼저 지으세요")
        return 2
    if not (SRC / ".git").is_dir():
        print(f"  포크 소스가 없다 — {SRC}")
        return 2
    dirty = subprocess.run(["git", "-C", str(SRC), "status", "--porcelain"],
                           capture_output=True, text=True).stdout.strip()
    if dirty:
        print("  포크 작업 트리가 깨끗하지 않다 — 대응 소스가 바이너리와 갈린다:")
        print("   " + dirty.replace("\n", "\n   "))
        return 2
    head = _head()
    patches = sorted(PATCH_DIR.glob("*.patch"))
    out = ROOT / "dist"
    out.mkdir(parents=True, exist_ok=True)
    # 패치 수도 이름에 넣는다 — 업스트림 고정점이 같아도 우리 패치가 늘면 **다른
    # 바이너리**다. 이름이 갈려야 옛 판을 적어 둔 `release.json`을 가진 사람의
    # sha256 대조가 안 깨진다 (`tools/get_fls.py`가 받은 뒤 그것을 본다).
    stem = f"ForzaLiveryStudio-itasha-{PIN[:12]}-p{len(patches)}"
    binary_zip, source_zip = f"{stem}-win64.zip", f"{stem}-source.zip"

    # ── 대응 소스: 고정 커밋 + 패치가 얹힌 그 트리 그대로 ──
    src_path = out / source_zip
    raw = subprocess.run(["git", "-C", str(SRC), "archive", "--format=zip", head],
                         capture_output=True, check=True).stdout
    src_path.write_bytes(raw)
    with zipfile.ZipFile(src_path, "a", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.writestr("BUILD.md", _build_md())
        for p in patches:
            z.write(p, f"patches/{p.name}")
        z.writestr("COMMIT", f"upstream {PIN}\nfork     {head}\n")
    print(f"  {src_path.name}  ({src_path.stat().st_size / 1e6:.1f}MB)")

    # ── 바이너리: 빌드본 + 라이선스 전문 + 고지 ──
    bin_path = out / binary_zip
    with zipfile.ZipFile(bin_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(VENDOR.rglob("*")):
            if p.is_file() and p.name != "README.md":     # README는 우리 저장소 글이다
                z.write(p, p.relative_to(VENDOR).as_posix())
        z.writestr("NOTICE.md", _notice(binary_zip, source_zip))
        for name, path in BUNDLED_LICENSES:
            if path.is_file():
                z.write(path, f"licenses/{name}")
            else:
                print(f"  ** 라이선스 전문이 없다 — {path}")
    print(f"  {bin_path.name}  ({bin_path.stat().st_size / 1e6:.1f}MB)")

    import hashlib

    for p in (bin_path, src_path):
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        print(f"  sha256 {h}  {p.name}")
    print("\n  둘을 **같은 릴리스에** 올리세요 — AGPL은 대응 소스가 "
          "바이너리를 따라다니기를 요구합니다.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", action="store_true", help="툴체인만 준비하고 끝")
    ap.add_argument("--check", action="store_true", help="상태만 본다")
    ap.add_argument("--keep-local", action="store_true",
                    help="작업 트리를 안 건드린다 (포크를 손으로 고치는 중)")
    ap.add_argument("--export-patches", action="store_true",
                    help="작업 트리 커밋을 tools/fls-patch/로 다시 뽑는다")
    ap.add_argument("--package", action="store_true",
                    help="릴리스에 올릴 두 벌을 dist/에 짓는다 "
                         "(바이너리 + AGPL 대응 소스) — 짓지는 않는다")
    ap.add_argument("--no-deploy", action="store_true")
    ap.add_argument("-j", "--jobs", type=int, default=None)
    a = ap.parse_args()
    if a.check:
        return check()
    if a.package:
        return package()
    if a.export_patches:
        export_patches()
        return 0
    setup()
    if a.setup:
        return 0
    sync_source(a.keep_local)
    build(a.jobs)
    if not a.no_deploy:
        deploy()
    print("\n끝. 제품 창의 [FLS 편집기에서 편집]이 이 판을 띄운다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
