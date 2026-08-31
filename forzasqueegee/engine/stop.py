"""중단 신호 — 창의 [취소]가 **엔진 안까지** 닿는 길.

진행 콜백 하나로는 못 멈춘다. `progress`는 빽빽한 반복문(획 배치·영역
채움)에만 걸려 있고, 그 밖의 긴 단(SR·선화·셀 분해·미세 조정·성장·메움·
수리)은 콜백을 아예 안 부른다 — 실측 한 판(318초)에서 콜백 사이 공백이
최대 86초, 실행 시간의 64%가 20초 넘는 공백 안이었다. 그 구간에서 누른
취소는 다음 콜백이 올 때까지 아무 일도 안 한다.

그래서 신호를 **진행 보고와 분리한다**: `make(should_stop=...)`가 받은
콜러블을 이 모듈이 스레드 지역으로 들고, 엔진 어디서든 `stop_here()`를
불러 묻는다. 반복문 머리마다 한 줄이면 되고 서명은 안 바뀐다 (그 한 줄을
넣자고 함수 마흔 개에 인자를 다는 대신이다). 스레드 지역이라 창이 만들기와
주입을 동시에 돌려도 신호가 안 섞인다.

`should_stop`은 **싸야 한다** — 획 하나·레이어 하나마다 불린다 (창은
플래그 하나를 읽는 람다다).
"""

from __future__ import annotations

import threading
from contextlib import contextmanager


class Cancelled(RuntimeError):
    """사람이 멈췄다 — 엔진은 이 예외를 **안 잡는다**.

    그 단계까지 쓴 파일은 남고 report는 안 생긴다. 부르는 쪽(창·CLI)이
    잡아 "취소했다"로 닫는다.
    """


_LOCAL = threading.local()


@contextmanager
def stopping(should_stop):
    """이 스레드에서 `stop_here()`가 물을 곳을 건다 (`None`이면 무동작)."""
    prev = getattr(_LOCAL, "fn", None)
    _LOCAL.fn = should_stop
    try:
        yield
    finally:
        _LOCAL.fn = prev


def stop_here() -> None:
    """중단 요청이면 여기서 `Cancelled`를 올린다 — 긴 반복문의 머리에 둔다."""
    fn = getattr(_LOCAL, "fn", None)
    if fn is not None and fn():
        raise Cancelled
