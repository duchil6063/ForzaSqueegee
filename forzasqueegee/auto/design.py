r"""`디자인 및 도색` 메뉴 (에디터 바깥의 허브) 조작.

행 (2026-08-17 실측, 한국어 UI):

    0 자동차 도색
    1 비닐 & 데칼 적용        ← 차체 에디터 (`auto.bodyedit`)
    2 새로운 디자인 찾기
    3 내 디자인
    4 내 비닐 그룹
    5 비닐 그룹 만들기        ← 빈 비닐 그룹 에디터 (템플릿·주입의 출발점)
    6 팔로우 중인 플레이어
    7 자동차 선택

만들기 메뉴와 달리 **선택 행도 흰 박스**다 (라임 테두리만 다르다) — 그래서
`Driver._menu_bands`의 "선택 행은 흰 밴드에서 빠진다" 규약이 여기서는 안 선다.
자리는 흰 행 밴드를 세고 라임 테두리가 **어느 밴드 안에 있나**로 잡는다.
"""

from __future__ import annotations

import time

import numpy as np

from ..game import io as gio
from .driver import Driver, DriverError

ROW_PAINT = 0
ROW_BODY_VINYL = 1
ROW_MY_GROUPS = 4
ROW_NEW_GROUP = 5

MENU_X_REL = (0.040, 0.225)      # 행 박스 x 구간 (1600×899에서 64..360)
MENU_Y_REL = (0.40, 0.90)        # 행이 놓이는 세로 범위


def _bands(img: np.ndarray) -> list[tuple[int, int]]:
    h, w = img.shape[:2]
    x0, x1 = int(MENU_X_REL[0] * w), int(MENU_X_REL[1] * w)
    y0, y1 = int(MENU_Y_REL[0] * h), int(MENU_Y_REL[1] * h)
    white = (img[:, x0:x1].min(axis=2) > 235).mean(axis=1)
    out: list[tuple[int, int]] = []
    s = None
    for y in range(y0, y1):
        if white[y] > 0.6 and s is None:
            s = y
        elif white[y] <= 0.6 and s is not None:
            if y - s > int(25 / 999 * h):
                out.append((s, y))
            s = None
    if s is not None and y1 - s > int(25 / 999 * h):
        out.append((s, y1))
    return out


def _lime_center(img: np.ndarray) -> float | None:
    h, w = img.shape[:2]
    x0, x1 = int(MENU_X_REL[0] * w), int(MENU_X_REL[1] * w)
    y0, y1 = int(MENU_Y_REL[0] * h), int(MENU_Y_REL[1] * h)
    sub = img[y0:y1, x0:x1]
    r = sub[:, :, 0].astype(np.int16)
    g = sub[:, :, 1].astype(np.int16)
    b = sub[:, :, 2].astype(np.int16)
    lime = ((r > 140) & (r < 235) & (g > 200) & (b < 110)).mean(axis=1)
    ys = np.where(lime > 0.5)[0]        # 테두리 위·아래 선
    if len(ys) == 0:
        return None
    return float(ys.min() + ys.max()) / 2 + y0


def menu_open(img: np.ndarray) -> bool:
    """`디자인 및 도색` 메뉴 화면인가 (왼쪽 흰 행 6개 이상 + 라임 선택 테두리)."""
    return len(_bands(img)) >= 6 and _lime_center(img) is not None


def row_index(img: np.ndarray) -> int | None:
    """선택된 행 인덱스. 미검출 시 None.

    선택 행은 보통 **반전(검정)**이라 흰 밴드에서 빠진다 — 그러면 라임 테두리
    위에 있는 흰 밴드 수가 곧 인덱스다. 전환 애니메이션 중에는 선택 행이 흰
    상태로 잡히기도 해서, 테두리가 어떤 밴드 **안**에 있으면 그 밴드로 본다.
    """
    bands = _bands(img)
    cy = _lime_center(img)
    if not bands or cy is None:
        return None
    for i, (a, b) in enumerate(bands):
        if a - 4 <= cy <= b + 4:
            return i
    return sum(1 for a, _ in bands if a < cy)


def wait_menu(d: Driver, timeout: float = 8.0) -> None:
    """`디자인 및 도색` 메뉴가 뜰 때까지 (전환 애니메이션 대기)."""
    t_end = time.time() + timeout
    while time.time() < t_end:
        if menu_open(d.cap()):
            return
        time.sleep(0.3)
    raise DriverError("`디자인 및 도색` 메뉴 화면이 아니다")


def goto_row(d: Driver, idx: int, max_steps: int = 12) -> None:
    """`디자인 및 도색` 메뉴에서 목표 행으로 (매 스텝 재검출 폐루프)."""
    wait_menu(d)
    for _ in range(max_steps):
        img = d.cap()
        if not menu_open(img):
            raise DriverError("`디자인 및 도색` 메뉴 화면이 아니다")
        cur = row_index(img)
        if cur is None:
            time.sleep(0.25)
            continue
        if cur == idx:
            return
        gio.press("down" if idx > cur else "up")
        time.sleep(0.3)
    raise DriverError(f"`디자인 및 도색` 행 이동 실패: 목표 {idx}")


def back_to_menu(d: Driver, tries: int = 6) -> None:
    """어느 에디터에서든 Esc를 눌러 `디자인 및 도색` 메뉴로 돌아온다.

    나가기 대화상자가 뜨면 **마지막 행(저장하지 않고 나가기)**을 고른다 —
    이 길은 '되돌리기'용이라 저장하지 않는 것이 옳다. 차에 적용하려면
    `bodyedit.exit_editor(apply=True)`를 명시적으로 부를 것.

    대화상자는 두 판이다 (실측): 차체·저장된 그룹은 3행, **저장한 적 없는 새
    그룹은 2행**(`새로운 디자인 저장` — 카탈로그에 저장 / 저장하지 않고 나가기).
    둘 다 마지막 행이 "저장하지 않고 나가기"다.
    """
    for _ in range(tries):
        img = d.cap()
        if menu_open(img):
            return
        rows = d._menu_row_index(img)
        if rows is not None and rows[1] in (2, 3):  # 나가기 대화상자 (2행/3행)
            d.menu_goto_row(rows[1] - 1)
            gio.press("enter")
            time.sleep(2.0)
            continue
        gio.press("esc")
        time.sleep(1.5)
    raise DriverError("`디자인 및 도색` 메뉴 복귀 실패")
