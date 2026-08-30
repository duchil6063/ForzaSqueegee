"""CLI 진입점: python -m forzasqueegee <명령> ... (명령 목록은 --help)

문법과 실행은 `cli` 패키지에 있다 — 여기는 그것을 부르고, **죽음을 보이게 하는**
자리다. 창 모드(`pythonw`)에는 콘솔이 없어 트레이스백이 갈 곳이 없다 — 그대로면
"창이 안 떠요"만 남는다. 그래서 여기서 받아 `work/logs/crash.log`에 적고, 콘솔이
없으면 메시지 상자 하나를 띄운다. 네이티브 크래시(Qt·onnxruntime의 접근 위반)는
파이썬 예외 없이 죽으므로 faulthandler가 `work/logs/native-crash.log`에 남긴다.

`pyproject`의 콘솔 스크립트가 이 이름(`forzasqueegee.__main__:main`)을 가리키므로
`main`은 여기서도 닿아야 한다.
"""

from __future__ import annotations

import sys

from .cli import main as _cli_main

_FAULT_LOG = None      # faulthandler가 쥔 파일 — 프로세스가 사는 동안 잡아 둔다


def _console_present() -> bool:
    try:
        import ctypes

        return bool(ctypes.windll.kernel32.GetConsoleWindow())
    except Exception:            # noqa: BLE001 — 윈도가 아니면 콘솔이 있다고 친다
        return True


def _log_crash(text: str) -> str | None:
    """트레이스백을 `work/logs/crash.log`에 덧붙인다. 자리를 주거나 None."""
    try:
        import datetime

        from .paths import work_file

        p = work_file("logs", "crash.log")
        stamp = datetime.datetime.now().isoformat(timespec="seconds")
        with p.open("a", encoding="utf-8") as f:
            f.write(f"\n===== {stamp} · python {sys.version.split()[0]}"
                    f" · {' '.join(sys.argv[1:])} =====\n{text}")
        return str(p)
    except Exception:            # noqa: BLE001 — 기록 실패가 보고를 막으면 안 된다
        return None


def main(argv: list[str] | None = None) -> int:
    global _FAULT_LOG
    try:
        import faulthandler

        from .paths import work_file

        _FAULT_LOG = work_file("logs", "native-crash.log").open("w", encoding="utf-8")
        faulthandler.enable(_FAULT_LOG)
    except Exception:            # noqa: BLE001 — 감시를 못 세워도 본론은 간다
        pass
    try:
        return _cli_main(argv)
    except SystemExit:
        raise
    except KeyboardInterrupt:
        return 130
    except BaseException:        # noqa: BLE001 — 마지막 문지기다
        import traceback

        text = traceback.format_exc()
        where = _log_crash(text)
        try:
            sys.stderr.write(text)
        except Exception:        # noqa: BLE001
            pass
        if not _console_present():
            try:
                import ctypes

                try:                       # i18n마저 깨졌으면 한국어 원문으로 간다
                    from .i18n import msg as _t
                except Exception:          # noqa: BLE001 — 마지막 문지기다
                    _t = lambda s, **kw: s.format(**kw) if kw else s  # noqa: E731
                text = _t("ForzaSqueegee가 오류로 멈췄습니다.")
                if where:
                    text += _t("\n\n기록: {where}\n이 파일을 개발자에게 보내 주세요.",
                               where=where)
                ctypes.windll.user32.MessageBoxW(None, text, "ForzaSqueegee", 0x10)
            except Exception:    # noqa: BLE001
                pass
        return 1


if __name__ == "__main__":
    sys.exit(main())
