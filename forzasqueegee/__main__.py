"""CLI 진입점: python -m forzasqueegee <명령> ... (명령 목록은 --help)

문법과 실행은 `cli` 패키지에 있다 — 여기는 그것을 부르는 자리다. `pyproject`의
콘솔 스크립트가 이 이름(`forzasqueegee.__main__:main`)을 가리키므로 `main`은
여기서도 닿아야 한다.
"""

from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
