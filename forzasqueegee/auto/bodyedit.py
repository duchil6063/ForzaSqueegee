r"""차체 에디터 드라이버 — 면 탭 이동·그룹 불러오기·배치·확정 (폐루프).

`디자인 및 도색 → 비닐 & 데칼 적용` 화면을 몬다. 비닐 그룹 에디터를 모는
`auto.driver.Driver`를 **그대로 안고** 화면 판별과 면 탭만 새로 댄다 — 도형
그리드·값 박스 OCR·변형 수치 입력은 두 에디터에서 같은 위젯이라 실측으로
확인했다 (2026-08-17).

두 에디터가 다른 자리 (실측):
- 레이어 카운터가 한 칸 아래다 (`game.ocr.read_body_count`).
- 맨 위에 면 탭 스트립이 있다 (`game.body`).
- 캔버스가 회색 체커가 아니라 **색 있는 차체**다 — 캔버스 채도로 화면을 가르던
  자가 여기서 무너진다 (`Driver.in_transform_edit` 문서 참조).
- **불러온 그룹의 변형 도구는 셋**이다: 이동(x·y) · 스케일(단일 균등) · 회전.
  비닐 레이어처럼 X/Y 개별 스케일이 없다.

되돌리기는 X(잘라내기)다 — 확정한 그룹은 리스트에서 잘라 클립보드로 보낸다.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from ..game import body
from ..game import io as gio
from ..game import ocr
from ..i18n import msg
from .driver import Driver, DriverError

# 그룹 그리드 (내 비닐 그룹) 기하 — 1600×899 실측 비율
GRID_COLS = 5
GRID_CELL0 = (0.2865, 0.3937)     # (0,0) 셀 중심 (x, y)
GRID_STRIDE = (0.1725, 0.3115)    # 셀 간격 (x, y)
# 좌측 정보 패널: 선택 항목의 레이어 수 ("2,835") 자리
GROUP_LAYERS_REL = (0.082, 0.578, 0.198, 0.617)

# 레이어 만들기 메뉴에서 '저장된 비닐 그룹 불러오기'의 자리. 앞 네 행
# (비닐 도안/로고/그룹/마스크)은 늘 있고, 뒤에 붙여넣기·반대편 붙여넣기가
# 상황에 따라 끼며, 마지막이 '모든 레이어 삭제'다 — 그래서 이 자리는 고정이다.
MENU_LOAD_GROUP = 2

# '저장 항목 없음' 대화상자 — 저장된 비닐 그룹이 **하나도 없으면** 그리드 대신
# 이것이 뜬다 (2026-08-20 실측). 라임 제목 띠와 그 아래 어두운 본문 한 줄이고,
# 확인(Enter)을 누르면 레이어 리스트로 돌아온다. 만들기 메뉴는 같은 자리가 흰
# 행이라(라임 0.07 대 0.91) 둘이 안 섞인다.
NO_SAVE_TITLE = (0.34, 0.44, 0.66, 0.49)
NO_SAVE_BODY = (0.34, 0.51, 0.66, 0.545)


class BodyEditor:
    """차체 에디터 폐루프. `Driver`를 물고 화면만 다르게 읽는다."""

    def __init__(self, drv: Driver | None = None):
        self.d = drv or Driver()
        self.hwnd = self.d.hwnd
        # 이 차의 면 탭 수 — 구성의 설치 파일 예측이 못 박는다 (`n_tabs`)
        self.tabs_n: int | None = None
        # 면 변형 편집에서 Y 스탬프가 서는가 — 같은 (색, 도형) 도형 묶음을
        # 위저드 한 바퀴로 놓는 지름길의 전제다 (`auto.itasha.add_shape_jobs`).
        # 첫 묶음이 커밋 수로 실측해 안 서면 끄고, 이 세션에서는 다시 안 켠다.
        self.stamp_ok = True

    # ---------- 화면 판별 ----------
    def cap(self) -> np.ndarray:
        return self.d.cap()

    def screen(self, img: np.ndarray | None = None) -> str | None:
        """'list' | 'menu' | 'groups' | 'fileopt' | 'transform' | 'shapes' | None."""
        if img is None:
            img = self.cap()
        if self.in_transform(img):
            return "transform"
        if self._fileopt_open(img):
            return "fileopt"
        if self.d._menu_open():
            return "menu"
        if self._groups_open(img):
            return "groups"
        if self.d.find_highlight(img) is not None:
            return "shapes"
        if ocr.read_body_cap(img) is not None and (
                body.selected_center(img) is not None
                # 밑줄이 알림 배너에 덮여도 리스트다 — 안 그러면 선택이 배너 아래
                # 탭에 있을 때 에디터 복귀가 통째로 실패한다 (2026-08-18 실측)
                or body.occluded_span(img) is not None):
            return "list"
        return None

    def in_transform(self, img: np.ndarray | None = None) -> bool:
        """변형 편집 화면인가 — **값 칸 하나만 있어도** 참이다.

        그룹의 스케일·회전은 X칸 하나만 쓴다 (Y칸이 아예 없다). `Driver`의 자는
        두 칸을 다 요구하므로 그 두 도구에서 거짓이 된다 — 여기서는 첫 칸
        아이콘만 본다 (색상·HSB 화면은 그 자리가 회색 패널이라 안 걸린다).
        """
        if img is None:
            img = self.cap()
        h, w = img.shape[:2]
        band = img[int(45 / 999 * h):int(90 / 999 * h),
                   int(68 / 1776 * w):int(418 / 1776 * w)]
        r = band[:, :, 0].astype(np.int16)
        g = band[:, :, 1].astype(np.int16)
        b = band[:, :, 2].astype(np.int16)
        if ((r > 150) & (r < 240) & (g > 200) & (b < 100)).mean() < 0.3:
            return False
        icon = img[int(148 / 999 * h):int(193 / 999 * h),
                   int(68 / 1776 * w):int(112 / 1776 * w)].mean()
        if icon < 235:
            return False
        return ocr.read_value(img, "x") is not None

    def _fileopt_open(self, img: np.ndarray) -> bool:
        """'파일 옵션' 대화상자 — 화면 가운데 위쪽 라임 제목 밴드."""
        h, w = img.shape[:2]
        band = img[int(0.150 * h):int(0.220 * h), int(0.36 * w):int(0.64 * w)]
        r = band[:, :, 0].astype(np.int16)
        g = band[:, :, 1].astype(np.int16)
        b = band[:, :, 2].astype(np.int16)
        return bool(((r > 140) & (r < 235) & (g > 200) & (b < 110)).mean() > 0.5)

    def _groups_open(self, img: np.ndarray) -> bool:
        """'내 비닐 그룹' 그리드 — 좌측 정보 패널의 '인기도' 라임 띠."""
        h, w = img.shape[:2]
        band = img[int(0.715 * h):int(0.775 * h), int(0.055 * w):int(0.19 * w)]
        r = band[:, :, 0].astype(np.int16)
        g = band[:, :, 1].astype(np.int16)
        b = band[:, :, 2].astype(np.int16)
        return bool(((r > 140) & (r < 235) & (g > 200) & (b < 110)).mean() > 0.5)

    def counts(self, img: np.ndarray | None = None) -> tuple[int | None, int | None]:
        """(현재 면의 레이어 수, 그 면의 상한)."""
        if img is None:
            img = self.cap()
        return ocr.read_body_count(img), ocr.read_body_cap(img)

    def count_stable(self, tries: int = 8, delay: float = 0.06) -> int | None:
        prev: int | None = None
        for _ in range(tries):
            v = ocr.read_body_count(self.cap())
            if v is not None and v == prev:
                return v
            prev = v
            time.sleep(delay)
        return None

    # ---------- 면 탭 ----------
    def n_tabs(self) -> int:
        """이 차의 면 탭 수 — 구성이 알려 준 값이 우선이다 (`tabs_n`).

        실측 표의 길이는 **표를 잰 차**의 것이라 유효 면이 다른 차에서 어긋난다
        (실측: 줄리아 표 9탭 ↔ CRX 뮤겐 10탭 — 마지막 탭이 "범위 밖"으로 죽었다).
        `place_all`·`check_car_tabs`가 설치 파일 예측으로 못 박는다.
        """
        if self.tabs_n:
            return self.tabs_n
        tabs = body.tab_table().get("tabs") or []
        return len(tabs) or 11

    def current_tab(self, img: np.ndarray | None = None) -> int | None:
        return body.selected_tab(img if img is not None else self.cap(),
                                 n=self.n_tabs())

    def count_tabs(self) -> int:
        """이 차의 면 탭 수를 **걸어서** 실측한다 — 왼끝에 붙이고 오른쪽으로
        선택 밑줄이 안 움직일 때까지 센다 (클램프 = 끝).

        설치 파일 예측은 장착 부품에 어긋날 수 있다 (스포일러·선루프 유무가 탭을
        더하고 뺀다). 흰 셀 세기는 배경 벽·선택 반전에 물리므로 정확한 수는 이
        길뿐이다. 20초쯤 든다 — 30분짜리 배치를 엉뚱한 면에 거는 것보다 싸다.
        """
        guess = self.n_tabs()
        for _ in range(guess + 3):
            gio.press("pgup")
            time.sleep(0.15)
        time.sleep(0.6)
        prev = body.selected_center(self.cap())
        n = 1
        for _ in range(guess + 4):
            gio.press("pgdn")
            time.sleep(0.4)
            cur = body.selected_center(self.cap())
            if cur is None or (prev is not None and abs(cur - prev) < 0.005):
                break
            prev = cur
            n += 1
        return n

    def goto_tab(self, index: int) -> None:
        """면 탭 index로. **왼쪽 끝에 붙이고 세어서** 간다 (양끝 클램프·순환 없음).

        스트립이 스크롤되지 않으므로 실측표의 중심 좌표로 곧장 확인할 수 있다 —
        그래도 이동은 세어서 한다: 양끝에서 화살표 셀이 회색으로 죽어 흰 셀을
        세는 길은 한 칸씩 밀린다.
        """
        n = self.n_tabs()
        if not 0 <= index < n:
            raise DriverError(msg("면 탭 범위 밖: {index} (0..{max})",
                                  index=index, max=n - 1))
        for _ in range(3):                       # 아는 자리에서면 상대 이동이 싸다
            cur = self.current_tab()
            if cur == index:
                return
            if cur is None:
                break
            for _ in range(abs(index - cur)):
                gio.press("pgdn" if index > cur else "pgup")
                time.sleep(0.4)
            time.sleep(0.3)
        for attempt in range(2):                 # 안 맞으면 왼쪽 끝에 붙여 다시 센다
            if self.current_tab() == index:
                return
            for _ in range(n + 2):
                gio.press("pgup")
                time.sleep(0.2)
            time.sleep(0.6)
            prev = body.selected_center(self.cap())
            for _ in range(index):
                gio.press("pgdn")
                time.sleep(0.45)
                cur = body.selected_center(self.cap())
                if cur is None or (prev is not None and abs(cur - prev) < 0.005):
                    break            # 밀림 — 바깥 루프가 다시 클램프한다
                prev = cur
            time.sleep(0.4)
            if self.current_tab() == index:
                return
            del attempt
        # 알림 배너가 목표 탭을 덮었으면 — 밑줄이 안 보여 위 두 경로가 못 간다
        # (실측: 멀티플레이 서버 알림이 탭 3~7을 덮는다). **보이는 탭에서 검증한
        # 뒤 남은 걸음만 세어** 간다. 덮인 자리는 화면 검증이 원리적으로 불가라
        # 걸음 수를 믿는 수밖에 없다 — 걸음을 최소로 만들고 천천히 민다.
        img = self.cap()
        span = body.occluded_span(img)
        tabs = body.tab_table().get("tabs") or []
        if span is not None and tabs:
            tgt = next((t for t in tabs if int(t["index"]) == index), None)
            covered = tgt is not None and span[0] <= tgt["center"] <= span[1]
            vis = [t for t in tabs
                   if not (span[0] <= t["center"] <= span[1])]
            if covered and vis:
                v = min(vis, key=lambda t: abs(int(t["index"]) - index))
                vi = int(v["index"])
                self.goto_tab(vi)                # 보이는 탭은 정상 경로로 확인된다
                for _ in range(abs(index - vi)):
                    gio.press("pgdn" if index > vi else "pgup")
                    time.sleep(0.7)
                time.sleep(0.4)
                got = self.current_tab()
                if got in (index, None):         # None = 여전히 덮여 있다 (정상)
                    return
                raise DriverError(
                    msg("면 탭 이동 실패: 목표 {index}, 배너 우회 후 {got}",
                        index=index, got=got))
        raise DriverError(msg("면 탭 이동 실패: 목표 {index}", index=index))

    # ---------- 레이어 만들기 ----------
    def goto_plus(self, max_steps: int = 8) -> None:
        """리스트 선택을 '+' 칸으로 (불러온 그룹은 한 줄로 접히므로 위로 몇 칸)."""
        self.d.goto_plus(max_steps=max_steps)

    def open_create_menu(self) -> None:
        self.d._open_create_menu()

    def open_wizard(self) -> None:
        """리스트 → 만들기 메뉴 0행(비닐 도안) → 도형 선택 그리드.

        비닐 그룹 에디터와 같은 흐름이다 — 글꼴 탭도 여기에 있어(실측) 면에
        글자를 바로 넣을 수 있다 (`auto.gametext`).
        """
        self.d.open_wizard()

    def menu_rows(self) -> int:
        idx = self.d._menu_row_index()
        if idx is None:
            raise DriverError(msg("레이어 만들기 메뉴 행 미검출"))
        return idx[1]

    # ---------- 저장된 비닐 그룹 불러오기 ----------
    def no_saves(self, img: np.ndarray | None = None) -> bool:
        """'저장 항목 없음' 대화상자인가 — 저장된 비닐 그룹이 하나도 없을 때다."""
        if img is None:
            img = self.cap()
        h, w = img.shape[:2]

        def crop(rel):
            x0, y0, x1, y1 = rel
            return img[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)].astype(np.int16)

        t = crop(NO_SAVE_TITLE)
        lime = ((t[:, :, 0] > 140) & (t[:, :, 0] < 235)
                & (t[:, :, 1] > 200) & (t[:, :, 2] < 110)).mean()
        dark = (crop(NO_SAVE_BODY).max(axis=2) < 90).mean()
        return bool(lime > 0.60 and dark > 0.70)

    def open_group_grid(self, allow_empty: bool = False) -> bool:
        """'+' → 레이어 만들기 → 저장된 비닐 그룹 불러오기 → 그룹 그리드.

        **저장된 그룹이 하나도 없으면** 그리드가 아니라 '저장 항목 없음'
        대화상자가 뜬다 (실측). 확인을 눌러 레이어 리스트로 돌아온 뒤
        `allow_empty`면 False를 돌려주고, 아니면 그 자리에서 죽는다 — 불러올
        그룹이 있어야 하는 자리에서 빈손으로 계속 가면 엉뚱한 것을 문다.
        """
        self.open_create_menu()
        self.d.menu_goto_row(MENU_LOAD_GROUP)
        seen: dict[str, bool] = {}

        def _opened() -> bool:
            img = self.cap()
            if self._groups_open(img):
                seen["grid"] = True
            elif self.no_saves(img):
                seen["empty"] = True
            return bool(seen)

        self.d._step("enter", _opened, msg("비닐 그룹 그리드"))
        if seen.get("grid"):
            return True
        gio.press("enter")                        # 확인 → 레이어 리스트
        time.sleep(0.8)
        if not allow_empty:
            raise DriverError(msg("저장된 비닐 그룹이 하나도 없다 — 불러올 것이 없다 "
                                  "(그룹 준비가 먼저다)"))
        return False

    def group_layers(self, img: np.ndarray | None = None) -> int | None:
        """그룹 그리드 좌측 정보 패널이 말하는 **선택 항목의 레이어 수**.

        이름은 못 읽는다 (글자 템플릿이 숫자뿐이다) — 대신 이 수가 이름표다.
        플랜 장수를 아니까 찾는 쪽에서 곧장 대조할 수 있다.
        """
        if img is None:
            img = self.cap()
        h, w = img.shape[:2]
        x0, y0, x1, y1 = GROUP_LAYERS_REL
        crop = img[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
        gl = ocr._count_glyphs(crop, crop.min(axis=2) > 200)
        n = ocr._count_digits(gl)
        # 저장 그룹은 최소 2장이다 (게임이 그 미만 저장을 거부한다) — 0은 패널
        # 자리표시자/미정착 판독이라 실패(None)로 본다. 0을 값으로 흘리면
        # walk_groups의 조용함 계산이 오염돼 실제 그룹 셀을 못 가 본다.
        return None if n == 0 else n

    def group_cell(self, img: np.ndarray | None = None) -> tuple[int, int] | None:
        """그리드에서 선택된 셀 (row, col). 미검출 시 None (라임 테두리 기준)."""
        if img is None:
            img = self.cap()
        h, w = img.shape[:2]
        r = img[:, :, 0].astype(np.int16)
        g = img[:, :, 1].astype(np.int16)
        b = img[:, :, 2].astype(np.int16)
        lime = ((r > 140) & (r < 235) & (g > 200) & (b < 110)).astype(np.uint8)
        lime[:, :int(0.20 * w)] = 0        # 좌측 정보 패널의 '인기도' 띠 배제
        n, lab, stats, cent = cv2.connectedComponentsWithStats(lime, 8)
        best = None
        for i in range(1, n):
            bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            if bw > 0.13 * w and bh > 0.20 * h:
                if best is None or stats[i, cv2.CC_STAT_AREA] > stats[best, cv2.CC_STAT_AREA]:
                    best = i
        if best is None:
            return None
        cx, cy = cent[best]
        col = round((cx / w - GRID_CELL0[0]) / GRID_STRIDE[0])
        row = round((cy / h - GRID_CELL0[1]) / GRID_STRIDE[1])
        return (int(row), int(col)) if row >= 0 and 0 <= col < GRID_COLS else None

    def group_layers_stable(self, tries: int = 20, delay: float = 0.08) -> int | None:
        """정보 패널의 레이어 수를 **정착 판독**한다 (연속 2회 동일).

        칸을 옮기면 패널이 한 박자 늦게 바뀐다 — 한 번만 읽으면 **앞 칸의 수**를
        읽어 엉뚱한 그룹을 고른다.
        """
        prev: int | None = None
        for _ in range(tries):
            v = self.group_layers()
            if v is not None and v == prev:
                return v
            prev = v
            time.sleep(delay)
        return prev

    def walk_groups(self, max_cells: int = 40, quiet_steps: int = 4,
                    max_rows: int = 6):
        """그리드의 항목을 (걸음, 장수)로 낸다 — **줄마다 오른쪽으로 밀고, 아래로**.

        자리(row, col)로 훑으면 안 된다: 이 그리드는 **회전목마**라 고른 항목이 늘
        같은 자리에 그려진다 (2026-08-17 실측: 서로 다른 세 그룹이 모두
        `group_cell()`에서 (0,0)으로 읽혔다). 그리고 **오른쪽 순환은 그 줄 안에서만**
        돈다 — 오른쪽만 밀면 둘째 줄 그룹을 영영 못 본다.

        순환의 끝을 세지도 않는다: 정보 패널이 한 박자 늦어 읽는 순서가 밀리기
        때문이다. 대신 **새 장수가 `quiet_steps` 걸음 동안 안 나오면 그 줄은 다
        봤다**고 보고 아래로 내려가며, 한 줄이 통째로 새것을 안 내면 끝낸다.

        같은 장수의 그룹이 둘이면 가를 수 없다 — 그래서 `auto.itasha`가 한 구성
        안의 장수 충돌을 미리 거부한다 (이름을 못 읽으니 장수가 이름표다).
        """
        seen: set[int] = set()
        step = 0
        for row in range(max_rows):
            fresh = 0
            quiet = 0
            for _ in range(max_cells):
                n = self.group_layers_stable()
                if n is not None:
                    if n in seen:
                        quiet += 1
                    else:
                        seen.add(n)
                        fresh += 1
                        quiet = 0
                    yield step, n
                    step += 1
                if quiet >= quiet_steps:
                    break
                gio.press("right")
                time.sleep(0.45)
            if row and not fresh:
                return                      # 이 줄에 새것이 없다 = 다 봤다
            self._grid_down()

    def find_group(self, layers: int, max_cells: int = 40) -> None:
        """그리드에서 **레이어 수가 `layers`인** 항목을 찾아 포커스한 채로 멈춘다.

        같은 장수의 그룹이 여럿이면 먼저 만난 것을 고른다 (이름이 안 읽히니
        그 이상은 못 가른다 — 그래서 `auto.itasha`가 장수 충돌을 미리 거부한다).
        """
        seen = []
        for _step, got in self.walk_groups(max_cells):
            if got == layers:
                return
            seen.append(got)
        # **느린 전수 훑기 폴백** — 다운로드 그룹이 낀 줄은 정보 패널이
        # 네트워크 지연으로 한참 늦어, 정착 판독이 앞 칸 값을 물고 "다 본
        # 값"으로 조용함이 차 조기 종료한다 (2026-08-31 실측: 2행의 deco
        # 그룹을 다섯 번 연속 못 찾았다). 줄마다 내려가며 칸마다 길게
        # 기다려 읽고, 맞는 값은 한 번 더 기다려 확인한다 (패널 지연이
        # 앞 칸 값을 이 칸에 씌우는 것을 막는다).
        for _row in range(3):
            for _col in range(max_cells // 2):
                time.sleep(1.3)
                n = self.group_layers()
                if n is None:
                    time.sleep(0.7)
                    n = self.group_layers()
                if n == layers:
                    time.sleep(1.0)
                    if self.group_layers() == layers:
                        return
                if n is not None:
                    seen.append(n)
                gio.press("right")
                time.sleep(0.45)
            self._grid_down()
        raise DriverError(
            msg("레이어 {layers:,}장짜리 비닐 그룹을 못 찾았다 "
                "(본 것: {seen})", layers=layers, seen=sorted(set(seen))))

    def _grid_down(self) -> None:
        """그리드에서 **줄 내리기를 확인하고** 간다 — 전환 애니메이션이 down을
        먹으면 같은 줄을 다시 걸게 된다. 라임 테두리는 행 구분이 안 될 때가
        있어, **좌측 정보 패널이 바뀌었는가**로 본다 — 선택이 실제로 움직이면
        패널(이름·장수·인기도)이 반드시 다시 그려진다."""
        img0 = self.cap()
        hh, ww = img0.shape[:2]
        box = (int(0.17 * hh), int(0.89 * hh), int(0.04 * ww), int(0.22 * ww))
        ref = img0[box[0]:box[1], box[2]:box[3]].astype(np.int16)
        for _try in range(3):
            gio.press("down")
            time.sleep(0.8)
            now = self.cap()[box[0]:box[1], box[2]:box[3]].astype(np.int16)
            if float(np.abs(now - ref).mean()) > 6.0:
                break

    def scan_groups(self, max_cells: int = 40) -> set[int]:
        """그리드를 훑어 저장된 비닐 그룹들의 **장수**를 모은다.

        이름은 못 읽으므로 장수만 모은다 — 준비 단계가 "이미 있나"를 묻는 데 쓴다.
        """
        return {n for _step, n in self.walk_groups(max_cells) if n}

    def paste_other_side(self, expect: int | None = None) -> None:
        """'반대편의 모든 레이어 붙여넣기' — 좌우 대칭 면을 한 번에 채운다.

        이 행은 **반대편이 있는 면에만** 뜨고 자리는 뒤에서 둘째다 (마지막은
        '모든 레이어 삭제'). 행 수가 모자라면 반대편이 없는 면이라 거부한다.
        `expect`를 주면 카운터가 그만큼 늘었는지까지 본다.
        """
        n0 = self.count_stable() or 0
        self.open_create_menu()
        rows = self.menu_rows()
        if rows < 6:
            gio.press("esc")
            time.sleep(0.5)
            raise DriverError(msg("이 면에는 '반대편 붙여넣기'가 없다 (좌우 짝이 없는 면)"))
        self.d.menu_goto_row(rows - 2)
        gio.press("enter")
        for _ in range(40):
            time.sleep(0.5)
            n1 = ocr.read_body_count(self.cap())
            if n1 is not None and n1 > n0:
                if expect is not None and n1 - n0 != expect:
                    raise DriverError(
                        msg("반대편 붙여넣기: {delta:,}장이 들어왔다 "
                            "(기대 {expect:,}장)", delta=n1 - n0, expect=expect))
                return
        raise DriverError(msg("반대편 붙여넣기 후 카운터가 안 늘었다"))

    def load_group(self, layers: int) -> None:
        """그룹 그리드에서 그 장수의 그룹을 골라 면에 불러온다 → 변형 편집.

        그리드 → Enter → '파일 옵션'(첫 행 = 비닐 그룹 불러오기) → Enter → 로드.
        3,000장 그룹은 로드에 수 초가 걸린다.
        """
        self.find_group(layers)
        self.d._step("enter", lambda: self._fileopt_open(self.cap()),
                     msg("파일 옵션 대화상자"))
        gio.press("enter")
        t_end = time.time() + 60.0
        while time.time() < t_end:
            if self.in_transform():
                return
            time.sleep(0.4)
        raise DriverError(msg("그룹 로드 후 변형 편집 미진입"))

    # ---------- 배치 ----------
    def place(self, x: float = 0.0, y: float = 0.0, scale: float = 1.0,
              rot: float = 0.0, mirror: bool = False,
              soft: bool = False, log=print) -> dict[str, float]:
        """변형 편집에서 그룹을 앉힌다. 반환: 최종 판독값.

        순서는 **회전 → 스케일 → 이동**이다. 회전·스케일이 이동값의 의미를
        바꾸지 않는다는 보장이 없어(원점 기준 변환) 이동을 마지막에 맞춘다.

        `soft`면 **이동 축이 못 닿아도 안 죽는다** (`Driver.set_axis_soft`) —
        면마다 이동 범위가 클램프돼 있고 홀드 폐루프가 어긋나는 면이 있다.
        회전·스케일에는 안 건다: 틀린 크기·각은 그림을 바꾸지만 몇 유닛 이동은
        안 바꾸고, 자리는 면 캡처와 카운터가 여전히 검증한다.

        **미러는 Tab + 180° 회전이다.** Tab(레이어 뒤집기)만 누르면 그룹이
        **위아래로** 뒤집힌다 (2026-08-17 실측: 우측면에 그대로 걸었더니 인물이
        거꾸로 섰다). 좌우 미러는 상하 뒤집기에 180° 회전을 얹은 것과 같으므로
        둘을 묶어 보낸다. Tab 자체는 값 칸에 안 뜨므로 폐루프로 못 재고 —
        회전값은 재진다 — 최종 확인은 캡처가 한다.
        """
        got: dict[str, float] = {}
        if mirror:
            gio.press("tab")
            time.sleep(0.6)
            rot = (rot + 180.0) % 360.0
        got["rot"] = self.d.set_axis("rot", rot)
        got["scale"] = self.d.set_axis("sx", scale)   # 그룹은 단일 균등 스케일 = 도구 2 X칸
        if soft:
            got["x"] = self.d.set_axis_soft("x", x, log=log)
            got["y"] = self.d.set_axis_soft("y", y, press_tool=False, log=log)
        else:
            got["x"] = self.d.set_axis("x", x)
            got["y"] = self.d.set_axis("y", y, press_tool=False)
        return got

    def commit(self) -> None:
        """변형 편집 → Enter → 레이어 리스트 복귀 (양성 확인)."""
        gio.press("1")           # 이동 도구로 되돌려 두 칸 판별을 살린다
        time.sleep(0.3)
        self.d._step("enter", lambda: not self.in_transform(), msg("그룹 배치 확정"))
        for _ in range(20):
            img = self.cap()
            if self.d._edit_menu_open(img):
                gio.press("esc")
                time.sleep(0.4)
                continue
            if ocr.read_body_cap(img) is not None:
                return
            time.sleep(0.2)
        raise DriverError(msg("확정 후 리스트 미도착"))

    def clear_surface(self) -> int:
        """이 면의 레이어를 통째로 지운다 (만들기 메뉴 마지막 행 → 확인). 반환: 지운 장수.

        면이 이미 비어 있으면 아무것도 안 한다. 마지막 행이 무엇인지 세지 않고
        **끝까지 내려** 고른다 — 클립보드·반대편 유무로 행 수가 5~7로 변한다.
        """
        n0 = self.count_stable() or 0
        if n0 == 0:
            return 0
        self.open_create_menu()
        rows = self.menu_rows()
        self.d.menu_goto_row(rows - 1)
        self.d._step("enter", self.d._confirm_band, msg("모든 레이어 삭제 확인창"))
        gio.press("enter")
        for _ in range(40):
            time.sleep(0.4)
            if ocr.read_body_count(self.cap()) == 0:
                return n0
        raise DriverError(msg("면을 비우지 못했다 (카운터가 0이 안 됐다)"))

    # ---------- 들어가기·나가기 ----------
    def exit_dialog_open(self, img: np.ndarray | None = None) -> bool:
        """'새로운 디자인 저장' 대화상자인가 (행 3개 + 면 카운터 부재).

        이 창은 제목 밴드 아래에 **설명 문단**이 끼어 있어 `_menu_open`의 고정
        영역 흰 비율로는 안 잡힌다 — 행 세기로 가른다.
        """
        if img is None:
            img = self.cap()
        if ocr.read_body_cap(img) is not None:
            return False
        idx = self.d._menu_row_index(img)
        return idx is not None and idx[1] == 3

    def exit_editor(self, apply: bool) -> bool:
        """Esc → '새로운 디자인 저장' 대화상자에서 고른다. 반환: 대화상자가 떴나.

        행: 0 현재 자동차에 적용 / 1 디자인 카탈로그에 저장 / 2 저장하지 않고 나가기.
        `apply=False`면 마지막 행(저장 안 함)이다.

        **대화상자는 바꾼 게 있을 때만 뜬다** — 아무것도 안 건드렸으면 Esc 한 번에
        곧장 `디자인 및 도색`으로 나간다. 그걸 모르고 Esc를 되풀이하면 메뉴를
        타고 **월드까지 빠져나간다** (2026-08-17 실측으로 그렇게 됐다). 그래서
        Esc는 **한 번만** 보내고 둘 중 무엇이 왔는지 기다린다.
        """
        from . import design

        gio.press("esc")
        t_end = time.time() + 8.0
        while time.time() < t_end:
            time.sleep(0.3)
            img = self.cap()
            if self.exit_dialog_open(img):
                self.d.menu_goto_row(0 if apply else 2)
                gio.press("enter")
                design.wait_menu(self.d, timeout=20.0)
                return True
            if design.menu_open(img):
                return False          # 바꾼 게 없다 — 저장할 것도 없다
        raise DriverError(msg("에디터 나가기 실패 (대화상자·메뉴 둘 다 안 왔다)"))

    def enter_editor(self) -> None:
        """`디자인 및 도색` 메뉴에서 `비닐 & 데칼 적용`으로 들어간다.

        이미 레이어 리스트면 아무것도 안 한다. 에디터 안 다른 화면(위저드·그리드)
        이면 리스트로 되돌린 뒤 판단한다 — 그래야 "이미 들어와 있다"를 잘못 보고
        엉뚱한 화면에서 배치를 시작하지 않는다.
        """
        from . import design

        for _ in range(5):
            s = self.screen()
            if s == "list":
                return
            if s is None:
                break
            gio.press("esc")
            time.sleep(1.0)
        design.goto_row(self.d, design.ROW_BODY_VINYL)
        self.d._step("enter", lambda: self.screen() == "list", msg("차체 에디터 진입"),
                     tries=3, wait=4.0)

    # ---------- 저장 그룹 재사용 (2026-08-24 실측 확립) ----------
    def slot_layers(self, img: np.ndarray | None = None) -> int | None:
        """저장 슬롯 그리드에서 **선택 셀의 장수**. 셀 우하단 아이콘 옆 숫자 OCR.

        '내 비닐 그룹' 그리드는 좌측 정보 패널에 장수가 있지만(`group_layers`),
        **저장 슬롯 그리드**는 없다 — 각 셀 우하단에 레이어 아이콘 + 장수가
        박혀 있다. 콤마 4자리까지 읽는다 (`ocr.read_slot_count`)."""
        if img is None:
            img = self.cap()
        cell = self.group_cell(img)
        if cell is None:
            return None
        r, c = cell
        h, w = img.shape[:2]
        cx = (GRID_CELL0[0] + c * GRID_STRIDE[0]) * w
        cy = (GRID_CELL0[1] + r * GRID_STRIDE[1]) * h
        crop = img[int(cy + 0.075 * h):int(cy + 0.15 * h),
                   int(cx - 0.01 * w):int(cx + 0.12 * w)]
        return ocr.read_slot_count(crop)

    def slot_layers_stable(self, tries: int = 10, delay: float = 0.08) -> int | None:
        """슬롯 셀 장수 정착 판독 (칸 이동 뒤 패널이 한 박자 늦는다)."""
        prev: int | None = None
        for _ in range(tries):
            v = self.slot_layers()
            if v is not None and v == prev:
                return v
            prev = v
            time.sleep(delay)
        return prev

    def open_saved_group(self, layers: int, wait_s: float = 60.0) -> int | None:
        """'내 비닐 그룹'에서 `layers`장 그룹을 **편집 캔버스로 다시 연다**.

        `디자인 및 도색 → 내 비닐 그룹` 그리드 → 장수로 찾아(`find_group`) →
        "파일 옵션" → **비닐 그룹 불러오기** → 편집 캔버스. 반환: 열린 캔버스
        장수 (`template.canvas_count`). 재사용 경로의 1단계다 (`auto.itasha`).

        **다시 연 그룹은 "1-N 접힌 그룹" 중첩 구조**라, 주입은
        `game.inject.find_folded_table`로 접힌 그룹 내부 표를 찾아야 한다.
        """
        from . import design
        from ..game import io as _gio
        from .template import canvas_count

        design.goto_row(self.d, design.ROW_MY_GROUPS)
        _gio.press("enter")
        for _ in range(20):
            time.sleep(0.5)
            if self._groups_open(self.cap()):
                break
        self.find_group(layers)
        self.d._step("enter", lambda: self._fileopt_open(self.cap()),
                     "파일 옵션 대화상자")
        _gio.press("enter")                       # 첫 행 = 비닐 그룹 불러오기
        t_end = time.time() + wait_s
        while time.time() < t_end:
            cc = canvas_count(self.hwnd)
            if cc is not None:
                return cc
            time.sleep(0.5)
        raise DriverError(msg("저장 그룹({layers:,}장) 불러오기 후 편집 캔버스 미진입",
                              layers=layers))

    def remove_sentinel(self) -> int | None:
        """위저드로 방금 심은 **센티널(최상위 레이어)을 잘라낸다** (재사용 정리).

        표 식별용 센티널을 심으면 캔버스가 N→N+1이 된다. 주입까지 끝낸 뒤 이걸
        불러 N으로 되돌리면 재저장이 원래 장수로 커밋돼 **신원이 안 누적된다**.
        위저드 확정 직후엔 그 레이어가 선택돼 있어 X(잘라내기)면 지워진다.

        **비닐 그룹 에디터라 `canvas_count`(레이어 카운터)로 확인한다** —
        차체 에디터 카운터(`read_body_count`)는 이 화면에 없다 (2026-08-24 실측).
        """
        from .template import canvas_count

        n0 = canvas_count(self.hwnd)
        gio.press("x")
        for _ in range(20):
            time.sleep(0.3)
            n1 = canvas_count(self.hwnd)
            if n1 is not None and n0 is not None and n1 < n0:
                return n1
        raise DriverError(msg("센티널 제거 실패 (canvas 카운터가 안 줄었다)"))

    def resave_overwrite(self, layers: int, max_cells: int = 30) -> None:
        """편집 캔버스의 현재 내용을 **`layers`장 슬롯에 덮어쓴다** (재저장).

        레이어 리스트에서 **Backspace → 저장 슬롯 그리드**. 첫 셀은 새 슬롯이라
        건너뛰고(`slot_layers`로 장수를 읽어), 목표 장수 셀을 찾아 → Enter →
        "파일 옵션"의 첫 행 **"<이름> 파일 덮어쓰기"** → Enter.

        **새 슬롯 저장(`Driver.save_group`)은 다시 연 그룹에서 게임이 커밋 안
        한다** (2026-08-24 확정) — 덮어쓰기만 먹는다. 덮어쓰기 뒤 화면은 레이어
        리스트로 복귀한다.
        """
        from ..game import io as _gio

        if self.d._save_screen() == "list":
            self.d._step("backspace",
                         lambda: self.d._save_screen() == "slots",
                         msg("저장 슬롯 그리드"))
        elif self.d._save_screen() != "slots":
            raise DriverError(msg("재저장 시작: 예상 밖 화면({screen})",
                                  screen=self.d._save_screen()))
        # 슬롯 그리드는 **회전목마**다 (선택 항목이 늘 같은 자리에 그려져
        # `group_cell`이 항상 (0,0)) — 자리로는 못 훑고 **장수로** 훑는다.
        # 함정 둘 (2026-08-24 실측): ① **1행 첫 셀은 새 슬롯**이고 그 장수는
        # 현재 캔버스라 목표와 겹칠 수 있다(재사용은 둘 다 N) — 그걸 고르면
        # 이름 대화상자가 떠 덮어쓰기가 안 되므로 **1행 0열은 건너뛴다**.
        # ② right는 그 줄 안에서만 순환하니 **본 장수가 다시 나오면 한 바퀴**로
        # 보고 아래 줄로 내려간다.
        found = False
        for row in range(6):
            seen: set[int] = set()
            for col in range(12):
                n = self.slot_layers_stable()
                if n is None:
                    _gio.press("right")
                    time.sleep(0.4)
                    continue
                if n in seen:                     # 이 줄 한 바퀴 (순환)
                    break
                seen.add(n)
                if n == layers and not (row == 0 and col == 0):
                    found = True
                    break
                _gio.press("right")
                time.sleep(0.4)
            if found:
                break
            _gio.press("down")
            time.sleep(0.4)
        if not found:
            raise DriverError(msg("저장 슬롯에서 {layers:,}장 그룹을 못 찾았다",
                                  layers=layers))
        self.d._step("enter", lambda: self._fileopt_open(self.cap()),
                     msg("덮어쓰기 파일 옵션"))
        _gio.press("enter")                       # 첫 행 = 덮어쓰기
        for _ in range(15):
            if self.d._save_screen() in ("list", "slots"):
                if self.d._save_screen() == "slots":
                    _gio.press("esc")
                    time.sleep(0.8)
                return
            _gio.press("enter")
            time.sleep(1.0)
        raise DriverError(msg("덮어쓰기 후 리스트 미복귀"))
