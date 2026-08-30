r"""자동차 도색(전체 도색) — 이타샤의 **베이스 컬러**를 도색으로 깐다.

레퍼런스 이타샤 8장(`references/이타샤`)의 베이스는 전부 **단색 자동차 도색**이다
(흰 4·검 2·캐릭터 테마색 2). 비닐로 차를 덮는 것이 아니라 도색 메뉴에서 칠한다 —
장수를 안 먹고, 메탈릭 반짝임·광택 같은 도료 질감은 비닐로 낼 수 없다.

## 길 (2026-08-18 실측, 한국어 UI 1600×899)

    디자인 및 도색 → 0행 `자동차 도색` → 부품 그리드(3×3, (0,0)=전체 도색)
    → Enter → 색 선택 화면 (탭: 제조사 고유 색상 · 일반 색상 · 메탈릭 … )
    → 일반 색상 탭에서 X `색상 미세 조정` → HSB 값 3칸 (비닐 색 편집기와 같은 자)
    → Enter가 **즉시 커밋**하고 부품 그리드로 돌아온다 (색 화면을 안 거친다)
    → Esc → `새로운 디자인 저장` 대화상자 → `현재 자동차에 적용`

- HSB 화면은 비닐 쪽과 같은 위젯이라 `Driver.set_hsb`(OCR 폐루프)가 그대로 선다.
- 색 목록·그리드에서 움직이면 차가 **라이브 프리뷰**된다 — 훑기만 해도 색이
  바뀌어 보이지만 Esc면 되돌아온다. 확정은 미세 조정의 Enter뿐이다.
- 탭 이동은 PgUp 클램프 후 PgDn 하나다 (`goto_font_tab`과 같은 문법). 일반 색상
  탭인지는 `도색 마감` 행의 흰 Tab 배지로 확인한다 — 메탈릭 탭에서 X를 누르면
  **듀얼 색상** 편집기가 열려 엉뚱한 것을 커밋하므로 확인 없이는 X를 안 누른다.
- 도색 마감(광택/무광 등)은 **건드리지 않는다** — 라벨이 한글 텍스트라 읽을 자가
  없고, 기본 광택이 레퍼런스 이타샤의 다수와 같다.
"""

from __future__ import annotations

import time

import numpy as np

from ..game import io as gio
from ..i18n import msg
from .driver import Driver, DriverError

# 판별식 상자 (상대 좌표, 2026-08-18 실측·문턱은 정찰 캡처 11장으로 캘리브레이션)
_PART_TITLE = (0.050, 0.097, 0.291, 0.130)   # `전체 도색` 라임 제목 밴드
_PART_CELLS = (0.047, 0.145, 0.292, 0.578)   # 3×3 흰 셀 영역
_COLOR_TITLE = (0.038, 0.049, 0.211, 0.088)  # 색 화면 라임 제목 (부품 그리드보다 위)
_FINISH_BADGE = (0.178, 0.280, 0.202, 0.304)  # `도색 마감` 행의 흰 Tab 배지
# 부품 그리드 셀 기하 (상대): 원점 셀 중심 + 피치. 선택 셀은 라임 테두리로 읽는다.
_CELL_ORIGIN = (0.089, 0.219)
_CELL_PITCH = (0.0817, 0.146)


def _frac_lime(img: np.ndarray, box) -> float:
    h, w = img.shape[:2]
    x0, y0, x1, y1 = box
    sub = img[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
    r = sub[:, :, 0].astype(np.int16)
    g = sub[:, :, 1].astype(np.int16)
    b = sub[:, :, 2].astype(np.int16)
    return float(((r > 140) & (r < 235) & (g > 200) & (b < 110)).mean())


def _frac_white(img: np.ndarray, box) -> float:
    h, w = img.shape[:2]
    x0, y0, x1, y1 = box
    sub = img[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
    return float((sub.min(axis=2) > 200).mean())


def in_part_grid(img: np.ndarray) -> bool:
    """도색 부품 그리드인가 (제목 0.95 · 셀 0.81 대 다른 화면 <0.2)."""
    return (_frac_lime(img, _PART_TITLE) > 0.6
            and _frac_white(img, _PART_CELLS) > 0.5)


def _looks_hsb(img: np.ndarray) -> bool:
    """HSB 미세 조정 화면의 지문 — 색상 무지개 스트립 + 밝기 트랙 그라데이션.

    `Driver.in_hsb_edit`과 같은 자다 (팔레트의 고채도 행이 무지개에 오탐되는
    것을 밝기 그라데이션이 막는다 — 2026-08-02 실측)."""
    h, w = img.shape[:2]
    strip = img[int(314 / 999 * h):int(334 / 999 * h),
                int(110 / 1776 * w):int(350 / 1776 * w)].astype(np.int16)
    if ((strip.max(axis=2) - strip.min(axis=2)) > 100).mean() < 0.5:
        return False
    btrack = img[int(486 / 999 * h):int(506 / 999 * h),
                 int(110 / 1776 * w):int(350 / 1776 * w)]
    third = btrack.shape[1] // 3
    return float(btrack[:, -third:].mean() - btrack[:, :third].mean()) >= 40


def in_color_screen(img: np.ndarray) -> bool:
    """색 선택 화면(어느 탭이든)인가.

    HSB 미세 조정과는 **HSB 화면의 지문이 아니라는 것**으로 가른다. `현재:`
    스와치의 흼으로 가르면 차가 흰색일 때 오발하고 (2026-08-18 실측: 흰 시빅),
    색 그리드의 유채 비율로 가르면 제조사 탭의 무채 팔레트에서 오발한다
    (같은 날 실측: 혼다 흰·검 팔레트가 유채 0.2로 나왔다)."""
    return (_frac_lime(img, _COLOR_TITLE) > 0.6
            and _frac_lime(img, _PART_TITLE) < 0.3
            and not _looks_hsb(img))


def in_general_tab(img: np.ndarray) -> bool:
    """`일반 색상` 탭인가 — `도색 마감` 행의 Tab 배지가 이 탭에만 있다.

    배지의 흼만 보면 **제조사 탭의 흰 색상 막대**가 같은 자리를 채워 오탐한다
    (2026-08-18 실측: 혼다 흰 팔레트). 배지 왼쪽의 `도색 마감:` 텍스트 행은
    어두운 바탕이므로 함께 요구한다."""
    if not (in_color_screen(img) and _frac_white(img, _FINISH_BADGE) > 0.2):
        return False
    h, w = img.shape[:2]
    left = img[int(0.280 * h):int(0.304 * h), int(0.050 * w):int(0.170 * w)]
    return float((left.max(axis=2) < 120).mean()) > 0.4


def sel_cell(img: np.ndarray) -> tuple[int, int] | None:
    """부품 그리드의 선택 셀 (행, 열) — 라임 테두리 중심을 셀 기하에 댄다.

    **제목 띠를 먼저 버린다.** 셀 이름을 띄우는 라임 제목 띠가 셀 판 바로 위에
    붙어 있어(실측 1600×899: 띠 y 82~122 · 첫 셀 행 133~) 크롭에 몇 줄만 새도
    폭이 판 전체라 무게중심을 통째로 끌어올린다 — 2행이 1행으로, 1행이 0행으로
    읽혔다. 그래서 **판 전체 폭의 0.7을 넘게 채운 행**은 테두리가 아니라 띠로
    보고 뺀다 (테두리는 셀 하나 폭이라 3분의 1을 못 넘는다).
    """
    h, w = img.shape[:2]
    x0, y0, x1, y1 = _PART_CELLS
    px0, py0 = int(x0 * w), int(y0 * h)
    sub = img[py0:int((y1 + 0.01) * h), px0:int((x1 + 0.01) * w)]
    r = sub[:, :, 0].astype(np.int16)
    g = sub[:, :, 1].astype(np.int16)
    b = sub[:, :, 2].astype(np.int16)
    m = (r > 140) & (r < 235) & (g > 200) & (b < 110)
    m[m.mean(axis=1) > 0.7] = False                 # 제목 띠 행을 뺀다
    ys, xs = np.where(m)
    if len(ys) < 30:
        return None
    cx = (xs.mean() + px0) / w
    cy = (ys.mean() + py0) / h
    col = int(round((cx - _CELL_ORIGIN[0]) / _CELL_PITCH[0]))
    row = int(round((cy - _CELL_ORIGIN[1]) / _CELL_PITCH[1]))
    return max(0, min(2, row)), max(0, min(2, col))


N_COLOR_TABS = 5      # 제조사·일반·메탈릭·최근·즐겨찾기 (양끝 화살표 제외)
WHOLE_CAR = (0, 0)    # 실측으로 아는 유일한 셀 — 전체 도색


def goto_cell(d: Driver, cell: tuple[int, int]) -> None:
    """부품 그리드 선택을 그 셀로 옮긴다 (선택 셀을 읽는 폐루프).

    그리드가 **끝에서 순환**하므로 (실측: (0,0)에서 위 두 번 = (1,0)) 걸음을
    세는 길은 못 쓴다 — 라임 테두리를 읽어 한 칸씩 좁힌다. 빈 셀은 선택이 안
    옮겨 가므로 제자리에서 멈추고, 그때 여기서 죽는다 (그 차에 없는 부품이다).
    """
    row, col = cell
    last, stuck = None, 0
    for _ in range(16):
        sel = sel_cell(d.cap())
        if sel is None:
            raise DriverError(msg("도색 그리드 선택 셀 미검출"))
        if sel == (row, col):
            return
        # **한 번 같다고 멈춘 게 아니다** — 부품을 고르면 카메라가 그 부품으로
        # 줌하느라 화면이 한 박자 늦게 바뀐다 (실측: 휠 셀에서 차가 통째로
        # 다시 잡힌다). 세 번 연속 같아야 "그 방향에 셀이 없다"로 본다.
        stuck = stuck + 1 if sel == last else 0
        if stuck >= 3:
            raise DriverError(msg("도색 부품 셀 {cell}로 못 간다 (선택이 {sel}에서 "
                                  "멈춘다 — 그 차에 없는 부품이다)",
                                  cell=cell, sel=sel))
        last = sel
        if sel[0] != row:
            gio.press("up" if sel[0] > row else "down")
        else:
            gio.press("left" if sel[1] > col else "right")
        time.sleep(0.55)
    raise DriverError(msg("도색 부품 셀 {cell}을 못 잡았다", cell=cell))


def open_general_tab(d: Driver) -> None:
    """색 선택 화면에서 `일반 색상` 탭 + HSB 미세 조정까지 연다.

    탭 이동은 PgUp 클램프(왼쪽 끝 = 제조사) 후 PgDn 하나다. 커서가 `현재:`
    스와치에 있으면 X가 안 열리므로 그리드로 내려 `도색 마감` 배지를 띄운다 —
    그 배지가 "일반 색상 탭 맞다"의 판별식이다 (메탈릭 탭에서 X는 **듀얼 색상**
    편집기를 열어 엉뚱한 것을 커밋한다).
    """
    for _ in range(N_COLOR_TABS + 1):
        gio.press("pgup")
        time.sleep(0.25)
    gio.press("pgdn")
    time.sleep(0.8)
    for _ in range(3):
        if in_general_tab(d.cap()):
            break
        gio.press("down")
        time.sleep(0.5)
    else:
        raise DriverError(msg("`일반 색상` 탭이 아니다 (도색 마감 배지 미검출) — "
                              "여기서 X를 누르면 다른 편집기가 열려 멈춘다"))
    d._step("x", d.in_hsb_edit, msg("도색 HSB 미세 조정 진입"))


def set_paint(d: Driver, hsb: tuple[float, float, float],
              part: tuple[int, int] = WHOLE_CAR, log=print) -> dict[str, float]:
    """차를 HSB로 칠하고 `현재 자동차에 적용`까지 간다.

    시작은 아무 데나 — `디자인 및 도색` 메뉴로 스스로 돌아온다. 반환은
    `set_hsb`의 판독값(OCR 재확인 수치)이라 그 자체가 검증이다.

    `part`는 부품 그리드의 (행, 열)이다. 기본은 **전체 도색**(0,0) — 나머지
    셀의 뜻은 아직 실측 전이라 부르는 쪽이 아는 셀만 준다
    (재서 `catalog/paint_parts.json`에 적어 둔다).
    """
    from . import design
    from .bodyedit import BodyEditor

    design.back_to_menu(d)
    design.goto_row(d, design.ROW_PAINT)
    d._step("enter", lambda: in_part_grid(d.cap()), msg("도색 부품 그리드 진입"),
            tries=3, wait=4.0)
    goto_cell(d, part)
    d._step("enter", lambda: in_color_screen(d.cap()), msg("도색 색 선택 진입"),
            tries=3, wait=4.0)
    open_general_tab(d)
    got = d.set_hsb(*hsb)
    log(msg("  도색 HSB {got} (판독값)", got=got))
    # 미세 조정의 Enter는 즉시 커밋하고 **부품 그리드로** 돌아온다 (실측)
    d._step("enter", lambda: in_part_grid(d.cap()), msg("도색 커밋 → 부품 그리드"),
            tries=3, wait=4.0)
    # 나가기 — 대화상자에서 `현재 자동차에 적용` (0행). 도색을 바꿨으므로 반드시 뜬다.
    b = BodyEditor(d)
    gio.press("esc")
    t_end = time.time() + 8.0
    while time.time() < t_end:
        time.sleep(0.3)
        img = d.cap()
        if b.exit_dialog_open(img):
            d.menu_goto_row(0)
            gio.press("enter")
            design.wait_menu(d, timeout=20.0)
            return got
        if design.menu_open(img):
            return got            # 대화상자 없이 나왔다 — 같은 색이었던 경우
    raise DriverError(msg("도색 나가기 실패 (대화상자·메뉴 둘 다 안 왔다)"))
