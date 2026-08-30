# SPDX-License-Identifier: MIT
# kloudys-forza-painter-suite(MIT)의 생성 파이프라인을 이식한 파생물이다.
# Portions © 2021 AE (forza-painter) · Sam Twidale (geometrize-lib) ·
# Michael Fogleman (Primitive). 전문·유래는 THIRD_PARTY_NOTICES.md.
"""동봉 GPU 생성기 실행 — 원본: forza_generator_v2.run_generator.

exe 호출·라이브 프리뷰 스냅샷 승격·중단 파일은 원본 그대로이고, I/O가
우리 것이다: 진행 줄(`[n/stop] ...`)을 콜백으로 돌리고 콜백 예외(창의
취소 = pipeline.Cancelled)를 프로세스 종료 후 그대로 전파한다.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
from pathlib import Path

from ...i18n import msg
from .base import GENERATOR_BIN, VENDOR_DIR

# 원시 생성기 진행 줄 — "[123/2000] Added rotated ellipse ..." (실측)
_PROGRESS_RE = re.compile(r"^\[(\d+)/(\d+)\]")


def run_generator(image: Path, settings_path: Path, checkpoint_dir: Path, preview_dir: Path,
                  out_stem: str, stop_file: Path | None = None, seed: int = 0,
                  log=print, progress=None) -> bool:
    """GPU 원시 생성 실행. `progress(n, stop)`은 생성기 진행 줄에서 나온다.

    콜백이 예외를 올리면(창의 취소) 프로세스를 끊고 그대로 전파한다.
    `stop_file`이 생기면 우아하게 멈추고 True를 돌려준다 (있는 체크포인트로
    마무리한다).
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    out_base = checkpoint_dir / out_stem
    preview_path = preview_dir / f"{out_stem}.raw.preview.png"
    cmd = [
        str(GENERATOR_BIN),
        str(image),
        "-settings",
        str(settings_path),
        "-output",
        str(out_base),
        "-preview",
        str(preview_path),
    ]
    if seed:
        cmd.extend(["-seed", str(seed)])
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW") else 0
    proc = subprocess.Popen(
        cmd,
        cwd=str(VENDOR_DIR),
        creationflags=flags,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    interrupted = False
    callback_error: list[BaseException] = []

    def _cleanup_live_preview_snapshots() -> None:
        # 원시 생성기는 라이브 프리뷰 옆에 번호 스냅샷을 남긴다 — 최신 것을
        # 고정 경로로 승격하고 번호 파일은 지운다 (V2 그대로)
        snapshots = sorted(
            preview_dir.glob(f"{out_stem}.raw.preview.*.png"),
            key=lambda path: path.stat().st_mtime if path.exists() else 0.0,
        )
        if snapshots:
            latest = snapshots[-1]
            temp_preview = preview_path.with_suffix(preview_path.suffix + ".tmp")
            try:
                shutil.copy2(latest, temp_preview)
                os.replace(temp_preview, preview_path)
            except OSError:
                try:
                    if temp_preview.exists():
                        temp_preview.unlink()
                except OSError:
                    pass
        for snapshot in snapshots:
            try:
                snapshot.unlink()
            except OSError:
                pass

    def _forward_output():
        try:
            if proc.stdout is None:
                return
            for raw_line in proc.stdout:
                line = raw_line.rstrip("\r\n")
                if not line:
                    continue
                m = _PROGRESS_RE.match(line)
                if m:
                    n, total = int(m.group(1)), int(m.group(2))
                    if progress is not None:
                        try:
                            progress(n, total)
                        except BaseException as exc:  # 창의 취소(Cancelled)
                            callback_error.append(exc)
                            try:
                                proc.terminate()
                            except Exception:
                                pass
                            return
                    if "Added" in line and n % 250 == 0:
                        log(f"  [GPU {n}/{total}]")
                    continue
                log(line)
                if "Saved preview snapshot" in line:
                    _cleanup_live_preview_snapshots()
        finally:
            if proc.stdout is not None:
                try:
                    proc.stdout.close()
                except Exception:
                    pass

    reader = threading.Thread(target=_forward_output, daemon=True)
    reader.start()
    while proc.poll() is None:
        if callback_error:
            break
        if stop_file is not None and stop_file.exists():
            interrupted = True
            log(msg("중단 요청 — 마지막 저장 체크포인트까지로 내부 생성을 끝낸다…"))
            try:
                proc.terminate()
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            break
        try:
            proc.wait(timeout=0.2)
        except subprocess.TimeoutExpired:
            continue
    proc.wait()
    reader.join(timeout=2)
    _cleanup_live_preview_snapshots()
    if callback_error:
        raise callback_error[0]
    if not interrupted and proc.returncode not in (0, None):
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    return interrupted
