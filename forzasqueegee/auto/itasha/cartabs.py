"""지금 타는 차가 구성의 차인가 — 차체 에디터의 **면 탭 수**로 본다."""

from __future__ import annotations

from ..bodyedit import BodyEditor
from ..driver import Driver, DriverError
from .config import Config, _car_tabs


PART_TABS = ("sunroof", "spoiler")


def check_car_tabs(cfg: Config, b: BodyEditor, log=print) -> None:
    """지금 차의 **면 탭 수**가 구성의 차와 맞나 — 걸어서 실측하고, 장착 부품
    차이는 보정한다.

    탭 개수는 곧 유효 면 수라 차마다 다르고, **같은 차라도 장착 부품이 탭을
    바꾼다** (스포일러·선루프). 안 맞는 차에 올리면 배치 수치가 다른 크기의 면
    위에 앉고 뒤쪽 면은 탭 번호까지 밀린다 (실측: 예측에 선루프가 낀 채로
    올리면 window_left가 실차의 window_right에 앉는다).

    세는 것은 **걸음**이다 (`BodyEditor.count_tabs` — 왼끝 클램프에서 오른쪽으로
    밑줄이 멈출 때까지): 흰 셀 세기는 배경 벽·선택 반전·회색 화살표에 물린다.

    실측이 예측보다 **부품 면만큼 적으면** 그 면을 빼고 탭을 다시 못 박는다 —
    구성이 그 면을 쓰고 있으면 뺄 수 없으므로 멈춘다. 실측이 더 많으면 어느
    면이 끼었는지 알 수 없으므로 멈춘다.
    """
    want = _car_tabs(cfg)
    if not want:
        return
    b.tabs_n = len(want)
    got = b.count_tabs()
    if got == len(want):
        log(f"차 확인: 면 탭 {got}개 — 구성의 차와 맞는다")
        return
    if got < len(want):
        # 부품 의존 면을 빼서 맞춰 본다 (선루프 → 스포일러 순)
        names = list(want)
        for drop in PART_TABS:
            if len(names) > got and drop in names:
                if any(p.surface == drop for p in cfg.placements):
                    raise DriverError(
                        f"지금 차에는 {drop} 면이 없는데 구성이 그 면을 쓴다 — "
                        f"지금 타는 차로 구성을 다시 지을 것 (`--car`)")
                names.remove(drop)
        if len(names) == got:
            for p in cfg.placements:
                if p.surface not in names:
                    raise DriverError(
                        f"{p.surface}: 지금 차에 없는 면이다 — 있는 면은 "
                        f"{', '.join(names)}")
                p.tab = names.index(p.surface)
            b.tabs_n = got
            dropped = [n for n in want if n not in names]
            log(f"차 확인: 면 탭 {got}개 — 예측({len(want)})에서 장착 안 된 "
                f"{'·'.join(dropped)} 면을 빼고 탭을 다시 못 박았다")
            return
    raise DriverError(
        f"지금 차의 탭이 {got}개인데 구성은 면 {len(want)}개 차"
        f"({cfg.media or cfg.car})로 지었다 — 배치 수치도 탭 번호도 어긋난다. "
        f"**지금 타는 차**로 구성을 다시 지을 것 (`--car`)")


def verify_car(cfg: Config, log=print) -> None:
    """지금 타는 차가 구성의 차인가 — 차체 에디터를 잠깐 열어 탭을 세고 닫는다."""
    from ...auto import design

    want = _car_tabs(cfg)
    if not want:
        return
    d = Driver()
    b = BodyEditor(d)
    design.back_to_menu(d)
    design.goto_row(d, design.ROW_BODY_VINYL)
    b.d._step("enter", lambda: b.screen() == "list", "차체 에디터 진입",
              tries=3, wait=4.0)
    try:
        check_car_tabs(cfg, b, log=log)
    finally:
        design.back_to_menu(d)
