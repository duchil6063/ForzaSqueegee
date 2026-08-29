"""FH6 에디터 자동 입력 드라이버 (폐루프).

원칙 (인게임 실측):
- 키 드롭/홀드 끊김 실측 → 모든 조작은 캡처·OCR 검증 후 진행
- 대략 이동은 WASD 홀드(속도 적응 비례 제어), 마무리는 화살표 스텝
- 값 판독은 game.ocr (연속 2회 동일 = 정착)

도형 선택: catalog/cell_map.json (2026-08-02 실측) — 탭 16개 PgUp/PgDn 전환·양끝
클램프, 기본 탭 = A그룹 열우선. 탭 검증은 셀 썸네일 ↔ 카탈로그 래스터 IoU.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..game import io as gio
from ..game import ocr

# 도형 그리드 기하 (1776×999 실측, 클라이언트 크기 비율로 환산)
CELL0_CX, CELL0_CY = 205.5 / 1776, 299.5 / 999  # (0,0) 셀 중심
CELL_STRIDE_X, CELL_STRIDE_Y = 152.8 / 1776, 152.0 / 999
GRID_ROWS, GRID_COLS = 4, 10
CELL_HALF = 66 / 999  # 셀 패치 반폭 (높이 비율)

# 도형 이름 대체 — **비어 있다 (64차)**. 창 조작이 딴 셀을 고르면 주입 경로와
# 갈린다: 주입은 도형 id를 그대로 적으므로 대체를 못 따라간다. B_26은 셀을
# 몰라서 A_02로 보내고 있었는데, id 실측으로 자리가 확정돼(탭 6) 이유가 사라졌다.
SHAPE_ALIASES: dict[str, str] = {}


def _cell_map_path() -> Path:
    return Path(__file__).resolve().parents[2] / "catalog" / "cell_map.json"

# 변형 도구: (도구 키, 값 박스, 스텝, 홀드키 +/-, 화살표 +/-, 홀드속도/s, 홀드 시동량)
#
# 홀드 응답 곡선 (56차, 0.02~0.8s 실측 — 짧은 구간만 재면 시동량을 속도로 오인한다):
#   이동 x  : 0.05s→18 · 0.1→37.5 · 0.2→73.5 · 0.4→145.5 · 0.8→288  → 360/s, a0≈0
#   크기 sx : 0.05→0.26 · 0.1→0.48 · 0.2→0.96 · 0.4→1.90            → 4.75/s, a0≈0.02
#   회전 rot: 0.05→13 · 0.1→24 · 0.2→47 · 0.4→97 · 0.8→191          → 237/s, a0≈1.2
# 어느 축도 가속이 없다 — 선형이다. 0.02s는 이 직선보다 덜 나간다(시동 구간).
TOOL_AXES = {
    "x": ("1", "x", 0.5, ("d", "a"), ("right", "left"), 360.0, 0.0),
    "y": ("1", "y", 0.5, ("w", "s"), ("up", "down"), 360.0, 0.0),
    "sx": ("2", "x", 0.01, ("d", "a"), ("right", "left"), 4.75, 0.02),
    "sy": ("2", "y", 0.01, ("w", "s"), ("up", "down"), 4.75, 0.02),
    # 회전(실측): a=증가/d=감소, w/s 무반응, 우측 화살표=감소. 값은 X박스 단일
    "rot": ("3", "x", 0.1, ("a", "d"), ("left", "right"), 237.0, 1.2),
}

TRACE = bool(os.environ.get("FS_TRACE"))  # 수렴 과정 추적 출력

MIN_HOLD_S = 0.02  # 이보다 짧은 홀드는 시동 구간이라 제어가 안 된다 (56차 실측)
# 홀드는 살짝 모자라게 친다 — 초과하면 방향을 뒤집어 왕복 1회(~0.4s)를 더 쓴다.
# 모자란 쪽은 다음 홀드나 화살표가 이어받는다.
HOLD_UNDERSHOOT = 0.92

# 도구별 값 칸 배치 — 도구 전환 확인용 (None = 그 칸은 비어 있어야 함)
TOOL_BOXES: dict[str, dict[str, str | None]] = {
    "1": {"x": "x", "y": "y"},
    "2": {"x": "sx", "y": "sy"},
    "3": {"x": "rot", "y": None},
    "5": {"x": "alpha", "y": None},
}

# ---- 투명도(도구 5) — 이 축만 스텝 규약이 다르다 (60차 실측) ----
# 표시는 0~100인데 내부가 8비트라 **화살표가 비대칭**이다:
#   왼쪽 = 내림 -3/255 (표시 -1.18) · 오른쪽 = 올림 +2/255 (표시 +0.78)
# 표시값으로 세면 위상 잔차가 남는다 → 8비트로 환산해 내림·올림 횟수를 정수로 푼다.
# gcd(3,2)=1이라 모든 알파에 정확히 닿는다 (255 → 128 = 43내림 + 1올림).
#
# **코스 홀드 단계가 없다.** a·d 홀드는 0.12s까지 이동이 0이고 0.14s부터 갑자기
# 전 구간을 넘긴다(0.30s에 240+ = 클램프). 제어 가능한 구간이 아예 없어서 붙일
# 자리가 없다 — 다른 축과 달리 화살표 폐루프 하나로 간다. 44회 = 판독 포함 ~1.0s.
#
# 양끝 클램프는 100%와 **0.78%(=2/255)**다 — 아래 끝이 0이 아니다(실측).
ALPHA_TOOL, ALPHA_BOX = "5", "x"
ALPHA_DOWN_KEY, ALPHA_UP_KEY = "left", "right"
ALPHA_DOWN8, ALPHA_UP8 = 3, 2
ALPHA_MIN8, ALPHA_MAX8 = 2, 255


def alpha8(pct: float) -> int:
    """표시값(0~100) → 내부 8비트 알파. 플랜 렌더(`engine.render`)와 같은 환산이다."""
    return int(round(min(100.0, max(0.0, pct)) * 255.0 / 100.0))


def _alpha_presses(cur8: int, tgt8: int) -> tuple[int, int]:
    """(내림 횟수, 올림 횟수) — 합이 최소인 해. 2·올림 − 3·내림 = 목표 − 현재.

    총 횟수가 (d + 5·내림)/2라 내림 수에 단조증가한다 → 성립하는 가장 작은
    내림 수가 곧 최소해다. 패리티는 세 걸음 안에 반드시 맞는다.
    """
    d = tgt8 - cur8
    a_min = max(0, -(d // 3))  # ceil(-d/3)
    for a in range(a_min, a_min + 3):
        rem = d + ALPHA_DOWN8 * a
        if rem >= 0 and rem % ALPHA_UP8 == 0:
            return a, rem // ALPHA_UP8
    raise DriverError(f"투명도: 도달 불가 ({cur8} → {tgt8})")


def _alpha_batches(cur8: int, tgt8: int) -> list[tuple[str, int]]:
    """전송 순서 — 양끝 클램프를 안 무는 쪽을 고른다.

    내림을 먼저 보내면 아래 클램프를, 올림을 먼저 보내면 위 클램프를 물 수 있다
    (예: 255 → 254는 올림 먼저 보내면 255에 붙어 253으로 착지). 목표가 끝에서
    몇 스텝 안일 때만 갈리므로 두 순서를 모의해 맞는 쪽을 쓴다.
    """
    down, up = _alpha_presses(cur8, tgt8)
    orders = (((ALPHA_DOWN_KEY, down), (ALPHA_UP_KEY, up)),
              ((ALPHA_UP_KEY, up), (ALPHA_DOWN_KEY, down)))
    for order in orders:
        v = cur8
        for key, n in order:
            step = -ALPHA_DOWN8 if key == ALPHA_DOWN_KEY else ALPHA_UP8
            v = min(ALPHA_MAX8, max(ALPHA_MIN8, v + step * n))
        if v == tgt8:
            return [(k, n) for k, n in order if n]
    # 둘 다 무는 경우 — 양끝을 가로지르는 몇 쌍뿐이다(전수 64,516쌍 중 7).
    # 목표 쪽으로 크게 한 번 밀어 끝에서 떼어 놓고 다음 회차에서 정확히 푼다.
    return [(ALPHA_UP_KEY if tgt8 > cur8 else ALPHA_DOWN_KEY, 90)]

# 값 띠 안에서 **도구마다 다르게 그려지는** 아이콘 열 구간 (2325폭 실측 98~134).
# 값과 무관한 도구 식별자다 — 도구 키가 드롭됐을 때 엉뚱한 축을 미는 사고를 막는다.
TOOL_ICON_REL = (98 / 2325, 135 / 2325)

class DriverError(RuntimeError):
    pass


@dataclass
class TransformTarget:
    x: float = 0.0
    y: float = 0.0
    sx: float = 1.0
    sy: float = 1.0
    rot: float = 0.0
    alpha: float = 100.0  # 표시 0~100 (새 레이어 기본값 = 100)


def _wrap_err(target: float, value: float, is_rot: bool) -> float:
    e = target - value
    if is_rot:
        e = (e + 180.0) % 360.0 - 180.0
    return e


class Driver:
    def __init__(self, hwnd: int | None = None):
        gio.set_dpi_aware()
        self.hwnd = hwnd or gio.find_hwnd()
        if not self.hwnd:
            raise DriverError("FH6 창을 찾을 수 없음")
        if not gio.focus(self.hwnd):
            raise DriverError("FH6 포그라운드 전환 실패")
        self._cell_map: dict | None = None
        self._shape_loc: dict[str, tuple[int, int, int]] | None = None
        self._verify_refs: dict[int, list[tuple[int, int, np.ndarray]]] = {}
        self.current_tab: int | None = None  # 검증된 현재 탭 (도형 선택 화면 기준)
        self.stats = {"settle_hit": 0, "settle_miss": 0, "settle_repress": 0}
        self._tool_icons: dict[str, np.ndarray] = {}  # 도구별 아이콘 참조 (첫 확인 때 학습)

    # ---------- 도형 셀 매핑 (catalog/cell_map.json) ----------
    def _load_cell_map(self) -> None:
        if self._cell_map is not None:
            return
        self._cell_map = json.loads(_cell_map_path().read_text(encoding="utf-8"))
        self._shape_loc = {}
        for tab_s, cells in self._cell_map["cells"].items():
            for rc, name in cells.items():
                r, c = (int(v) for v in rc.split(","))
                # 같은 도형이 여러 탭에 있으면 낮은 탭(기본 우선) 유지
                if name not in self._shape_loc or int(tab_s) < self._shape_loc[name][0]:
                    self._shape_loc[name] = (int(tab_s), r, c)

    def shape_loc(self, shape: str) -> tuple[int, int, int]:
        """카탈로그 도형 이름 → (탭, row, col). 미등록 시 DriverError."""
        self._load_cell_map()
        shape = SHAPE_ALIASES.get(shape, shape)
        loc = self._shape_loc.get(shape)
        if loc is None:
            raise DriverError(f"셀 매핑 없는 도형: {shape}")
        return loc

    def _cell_thumb(self, img, r: int, c: int) -> np.ndarray | None:
        """셀 (r,c)의 썸네일 마스크 (배경 적응 임계값). 빈 셀이면 None.

        문턱은 배경과 최고 밝기의 중간값이다 — 셀 매핑을 뜰 때와
        **같은 자**를 써야 한다. 고정 190은 그라데이션·디더·얇은 썸네일을
        통째로 놓쳐 탭 검증이 실패한다 (54차: 탭 2·5·9·12가 그렇게 떨어졌다).
        선택 셀의 lime 테두리는 채도로 거른다.
        """
        h, w = img.shape[:2]
        cy = int((CELL0_CY + r * CELL_STRIDE_Y) * h)
        cx = int((CELL0_CX + c * CELL_STRIDE_X) * w)
        half = int(CELL_HALF * h)
        patch = img[cy - half:cy + half, cx - half:cx + half].astype(np.int16)
        gray = patch.mean(axis=2)
        dull = (patch.max(axis=2) - patch.min(axis=2)) < 40
        border = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
        bg = float(np.median(border))
        peak = float(np.percentile(gray[dull], 99.5)) if dull.any() else bg
        m = (gray > max(bg + 25, (bg + peak) / 2)) & dull
        return m if m.sum() >= 25 else None

    @staticmethod
    def _norm64(m: np.ndarray) -> np.ndarray | None:
        ys, xs = np.nonzero(m)
        if len(ys) < 10:
            return None
        crop = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.uint8)
        return cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA) > 0.5

    def _tab_refs(self, tab: int) -> list[tuple[int, int, np.ndarray]]:
        """탭 검증용 (row, col, 카탈로그 정규화 마스크) 참조 3개.

        **자리로 고르면 안 된다** (54차). 대각 3셀 고정은 그 자리가 얇거나
        페이드인 탭에서 검증이 통째로 실패하고(구판 실측: 탭 2·5·9·12),
        IoU가 높은 셀로 바꾸면 이번엔 흔한 실루엣이 뽑혀 **다른 탭 화면을
        목표 탭으로 오인**한다(13탭 중 16쌍). 변별력은 전 탭 캡처를 같이 봐야
        재므로 매핑 때 골라 `cell_map["refs"]`에 넣어 둔 것을 쓴다.
        """
        if tab in self._verify_refs:
            return self._verify_refs[tab]
        from ..engine.catalog import Catalog, default_catalog_path
        self._load_cell_map()  # ensure_tab을 안 거친 직접 검증 호출 대비
        cells = self._cell_map["cells"].get(str(tab), {})
        pre = self._cell_map.get("refs", {}).get(str(tab))
        if pre:
            picks = [(f"{r},{c}", cells[f"{r},{c}"]) for r, c in pre if f"{r},{c}" in cells]
        else:  # 구판 cell_map 호환 — 대각 분산 셀
            picks = []
            for want in ((0, 0), (1, 5), (3, 9)):
                best = min(cells, key=lambda rc: abs(int(rc.split(",")[0]) - want[0])
                           + abs(int(rc.split(",")[1]) - want[1]), default=None)
                if best and best not in [p[0] for p in picks]:
                    picks.append((best, cells[best]))
        cat = Catalog(default_catalog_path())
        refs = []
        for rc, name in picks:
            r, c = (int(v) for v in rc.split(","))
            n = self._norm64(cat[name].rasterize(128))
            if n is not None:
                refs.append((r, c, n))
        self._verify_refs[tab] = refs
        return refs

    def _verify_tab(self, img, tab: int) -> bool:
        """현재 도형 선택 화면이 tab인지 셀 썸네일 IoU로 확인 (과반)."""
        refs = self._tab_refs(tab)
        if not refs:
            return False
        ok = 0
        for r, c, ref in refs:
            m = self._cell_thumb(img, r, c)
            n = self._norm64(m) if m is not None else None
            if n is not None:
                iou = np.logical_and(n, ref).sum() / max(1, np.logical_or(n, ref).sum())
                if iou > 0.55:
                    ok += 1
        return ok * 2 > len(refs)

    # 탭 스트립 행 (1307 높이 실측 236~247px) — 흰 탭 버튼들이 한 줄로 놓인다
    TAB_ROW_REL = (236 / 1307, 247 / 1307)

    def _tab_boxes(self, img) -> list[tuple[int, int]]:
        """탭 스트립의 탭 슬롯 x구간 목록 (좌→우, 양끝 스크롤 화살표 제외).

        선택된 탭은 **반전 렌더**라 흰 구간이 아니다 — 흰 박스 사이의 넓은 빈틈이
        곧 선택 탭이다. 스트립이 왼쪽 끝에 있으면 슬롯 i = 탭 i다.
        """
        h, w = img.shape[:2]
        y0, y1 = int(self.TAB_ROW_REL[0] * h), int(self.TAB_ROW_REL[1] * h)
        white = (img[y0:y1].min(axis=2) > 200).mean(axis=0)
        cols = np.where(white > 0.5)[0]
        runs: list[tuple[int, int]] = []
        s = p = None
        for c in cols:
            if s is None or c > p + 3:
                if s is not None:
                    runs.append((s, p))
                s = c
            p = c
        if s is not None:
            runs.append((s, p))
        runs = [r for r in runs if r[1] - r[0] > int(0.008 * w)]
        if len(runs) < 3:
            return []
        gap = 0.04 * w  # 이보다 넓은 빈틈 = 선택된(반전) 탭
        boxes: list[tuple[int, int]] = []
        x = runs[0][1]  # 좌측 스크롤 화살표 오른쪽 끝
        for r in runs[1:-1]:
            if r[0] - x > gap:
                boxes.append((x, r[0]))
            boxes.append(r)
            x = r[1]
        if runs[-1][0] - x > gap:
            boxes.append((x, runs[-1][0]))
        return boxes

    def ensure_tab(self, tab: int) -> None:
        """도형 선택 화면에서 목표 탭으로.

        탭 슬롯을 직접 클릭한다 (56차: 화살표 클램프는 PgUp 17회 + PgDn n회로
        4~6초, 클릭은 0.2초). 실제 쓰이는 탭은 1~5뿐이라 스트립 첫 화면에 늘
        보인다. 클릭이 빗나가면(스트립이 스크롤된 상태) 검증이 잡아내고 옛
        클램프 경로로 떨어진다.
        """
        self._load_cell_map()
        n_tabs = len(self._cell_map["meta"]["tabs"])
        for attempt in range(4):
            img = self.cap()
            if self._verify_tab(img, tab):
                self.current_tab = tab
                return
            boxes = self._tab_boxes(img)
            if attempt < 2 and tab < len(boxes):
                x0, x1 = boxes[tab]
                _, hh = gio.client_size(self.hwnd)
                gio.click(self.hwnd, (x0 + x1) / 2,
                          (self.TAB_ROW_REL[0] + self.TAB_ROW_REL[1]) / 2 * hh)
                time.sleep(0.25)
                continue
            for _ in range(n_tabs + 1):  # PgUp 클램프 → 절대 위치 확보
                gio.press("pgup")
                time.sleep(0.22)
            for _ in range(tab):
                gio.press("pgdn")
                time.sleep(0.55)
            time.sleep(0.5)
            # `추천 그룹` 탭(카탈로그 0)은 **세션에 따라 없다** (2026-08-18
            # 실비아 실측: 전 탭이 한 칸 밀려 도형·글꼴이 통째로 어긋났다).
            # 걸음수 뒤 검증이 어긋나면 한 칸 왼쪽(탭 0 부재)·오른쪽을 본다.
            if self._verify_tab(self.cap(), tab):
                self.current_tab = tab
                return
            for key in ("pgup", "pgdn", "pgdn"):
                gio.press(key)
                time.sleep(0.55)
                if self._verify_tab(self.cap(), tab):
                    self.current_tab = tab
                    return
        raise DriverError(f"탭 이동 실패: 목표 {tab}")

    def select_shape(self, shape: str) -> None:
        """도형 선택 화면에서 카탈로그 도형으로 이동 (탭 검증·전환 포함)."""
        tab, row, col = self.shape_loc(shape)
        if self._verify_tab(self.cap(), tab):  # 매번 내용 검증 (캡처 1회, ~30ms)
            self.current_tab = tab
        else:
            self.ensure_tab(tab)
        self.select_cell(row, col)

    # ---------- 화면 상태 감지 ----------
    def cap(self) -> np.ndarray:
        return gio.capture(self.hwnd)

    def find_highlight(self, img) -> tuple[int, int] | None:
        """도형 그리드의 선택 셀 (row, col). 미검출 시 None."""
        h, w = img.shape[:2]
        r = img[:, :, 0].astype(np.int16)
        g = img[:, :, 1].astype(np.int16)
        b = img[:, :, 2].astype(np.int16)
        m = ((g > 180) & (r > 120) & (r < 235) & (b < 90)).astype(np.uint8)
        m[: int(0.22 * h)] = 0  # 탭 밑줄 제외
        n, lab, stats, cent = cv2.connectedComponentsWithStats(m, 8)
        best = None
        for i in range(1, n):
            bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            if 0.06 * w < bw < 0.12 * w and 0.12 * h < bh < 0.19 * h:
                if best is None or stats[i, cv2.CC_STAT_AREA] > stats[best, cv2.CC_STAT_AREA]:
                    best = i
        if best is None:
            return None
        cx, cy = cent[best]
        col = round((cx / w - CELL0_CX) / CELL_STRIDE_X)
        row = round((cy / h - CELL0_CY) / CELL_STRIDE_Y)
        if 0 <= row < GRID_ROWS and 0 <= col < GRID_COLS:
            return row, col
        return None

    def in_hsb_edit(self) -> bool:
        """색상 미세 조정(HSB) 화면 여부.

        판별: 좌상단 연두 제목 밴드 + 색상 슬라이더 무지개 스트립(고채도 다색)
        + 슬라이더 값 OCR 성공.
        """
        img = self.cap()
        h, w = img.shape[:2]
        band = img[int(45 / 999 * h):int(90 / 999 * h), int(68 / 1776 * w):int(418 / 1776 * w)]
        r = band[:, :, 0].astype(np.int16)
        g = band[:, :, 1].astype(np.int16)
        b = band[:, :, 2].astype(np.int16)
        if ((r > 150) & (r < 240) & (g > 200) & (b < 100)).mean() < 0.3:
            return False
        strip = img[int(314 / 999 * h):int(334 / 999 * h), int(110 / 1776 * w):int(350 / 1776 * w)]
        sat = strip.max(axis=2).astype(np.int16) - strip.min(axis=2).astype(np.int16)
        if (sat > 100).mean() < 0.5:  # 무지개 스트립 없음 = HSB 화면 아님
            return False
        # 밝기(B) 트랙의 검정→밝음 그라데이션 확인 — 색상 화면의 고채도 행(팔레트·
        # 즐겨찾기 리스트)이 무지개 스트립 검사에 오탐되는 것을 차단 (2026-08-02 실측)
        btrack = img[int(486 / 999 * h):int(506 / 999 * h), int(110 / 1776 * w):int(350 / 1776 * w)]
        third = btrack.shape[1] // 3
        if btrack[:, -third:].mean() - btrack[:, :third].mean() < 40:
            return False
        return ocr.read_hsb_value(img, "h") is not None

    def in_transform_edit(self, img=None) -> bool:
        """변형 편집 화면 여부.

        판별: 좌상단 연두 제목 밴드 + **값 박스 아이콘 타일 2개가 모두 밝음** +
        값 OCR 성공. (레이어 리스트에서 값 영역이 썸네일과 겹쳐 OCR 오탐 가능 →
        밴드 확인 필수)

        **캔버스를 안 본다.** 좌측 팔레트 자리의 채도로 색상 선택 화면을
        거르면 안 된다 — 그 자는 캔버스가 회색(비닐 그룹 체커)일 때만 서고,
        차체 에디터에서는 그 자리에 파란 차가 있어 변형 편집을 색상 화면으로
        오판한다 (2026-08-17 실측). 아이콘 타일은 화면 종류만 가리키고 캔버스와
        무관하므로 두 에디터에서 같이 선다 (색상·HSB 화면은 그 자리가 회색 패널).
        """
        if img is None:
            img = self.cap()
        h, w = img.shape[:2]
        band = img[int(45 / 999 * h):int(90 / 999 * h), int(68 / 1776 * w):int(418 / 1776 * w)]
        r = band[:, :, 0].astype(np.int16)
        g = band[:, :, 1].astype(np.int16)
        b = band[:, :, 2].astype(np.int16)
        if ((r > 150) & (r < 240) & (g > 200) & (b < 100)).mean() < 0.3:
            return False
        # 값 박스 아이콘 타일 2개가 모두 밝음 = 변형 편집 (색상·HSB 화면 배제)
        icon1 = img[int(148 / 999 * h):int(193 / 999 * h), int(68 / 1776 * w):int(112 / 1776 * w)].mean()
        icon2 = img[int(148 / 999 * h):int(193 / 999 * h), int(246 / 1776 * w):int(290 / 1776 * w)].mean()
        if icon1 < 235 or icon2 < 235:
            return False
        return ocr.read_value(img, "x") is not None

    # ---------- 단계 전이 ----------
    def _step(self, key: str, check, desc: str, tries: int = 3, wait: float = 1.4,
              poll: float = 0.12):
        """key 전송 후 check를 폴링. 재전송은 타임아웃 후에만.

        Enter류는 멱등이 아님(변형 편집에서 재전송 = 조기 커밋) — 전이 애니메이션
        동안 성급히 재전송하지 않도록 wait 동안 poll 간격으로 확인한다.
        """
        for _ in range(tries):
            gio.unlatch_menu()  # Alt 래치 = 키 전량 무시 (42차 실측). 재전송 전 해제
            gio.press(key)
            t_end = time.time() + wait
            while time.time() < t_end:
                time.sleep(poll)
                if check():
                    return
        raise DriverError(f"단계 전이 실패: {desc}")

    def _menu_open(self) -> bool:
        """레이어 만들기 메뉴(중앙 흰 행 4개) 표시 여부 — P 배경 상태와 무관."""
        img = self.cap()
        h, w = img.shape[:2]
        rows = img[int(435 / 999 * h):int(635 / 999 * h), int(580 / 1776 * w):int(1196 / 1776 * w)]
        return float((rows.min(axis=2) > 235).mean()) > 0.4  # 순백 메뉴 행 (격자 ~205와 구분)

    def _edit_menu_open(self, img=None) -> bool:
        """레이어 편집 메뉴(기존 레이어 Enter, 연두 '편집' 제목 밴드) 표시 여부."""
        if img is None:
            img = self.cap()
        h, w = img.shape[:2]
        band = img[int(98 / 999 * h):int(138 / 999 * h), int(200 / 1776 * w):int(560 / 1776 * w)]
        r = band[:, :, 0].astype(np.int16)
        g = band[:, :, 1].astype(np.int16)
        b = band[:, :, 2].astype(np.int16)
        return bool(((r > 150) & (r < 240) & (g > 200) & (b < 100)).mean() > 0.3)

    def list_selection(self, img) -> tuple[int, int] | None:
        """레이어 리스트 선택 셀 lime 링의 (y_top, y_bot) px. 미검출 시 None."""
        h, w = img.shape[:2]
        r = img[:, :, 0].astype(np.int16)
        g = img[:, :, 1].astype(np.int16)
        b = img[:, :, 2].astype(np.int16)
        lime = (r > 140) & (r < 235) & (g > 200) & (b < 110)
        sub = lime[:, int(60 / 1776 * w):int(175 / 1776 * w)]
        prof = sub.sum(axis=1)
        edges = np.where(prof > 0.5 * sub.shape[1])[0]  # 링 상/하변 (하단 카운트 텍스트 배제)
        if len(edges) < 2:
            return None
        return int(edges.min()), int(edges.max())

    def plus_selected(self, img) -> bool | None:
        """선택 셀이 '+'(새 레이어) 셀인가. 리스트 선택 미검출 시 None.

        '+' 셀 = 순백 배경(체커 없음). 레이어 썸네일 = 체커 배경(회색조 195~238
        비율 0.28+, 흰 도형이어도 모서리에 체커 노출). 실측: '+' 0.07 vs 썸네일 0.28~0.39.
        """
        sel = self.list_selection(img)
        if sel is None:
            return None
        y0, y1 = sel
        h, w = img.shape[:2]
        inner = img[y0 + 8:y1 - 8, int(80 / 1776 * w):int(156 / 1776 * w)]
        gray = inner.mean(axis=2)
        checker = ((gray > 195) & (gray <= 238)).mean()
        return bool(checker < 0.15 and (gray > 238).mean() > 0.6)

    def goto_plus(self, max_steps: int = 64) -> None:
        """리스트 선택을 '+' 셀로.

        커밋 직후에는 새 레이어가 '+' 바로 아래라 한 걸음이지만, **탭을 다시
        연 리스트는 선택이 수십 장 아래**일 수 있다 (2026-08-18 실측: top 43장
        리스트에서 8걸음 상한이 두 번 연속 모자라 실행이 죽었다).
        """
        for _ in range(max_steps):
            img = self.cap()
            if self._edit_menu_open(img):  # 열려 있으면 상하 이동이 메뉴를 조작 — 먼저 닫기
                gio.press("esc")
                time.sleep(0.4)
                continue
            plus = self.plus_selected(img)
            if plus is None:  # 전환 애니메이션 중일 수 있음 — 잠시 후 재검출
                time.sleep(0.25)
                continue
            if plus:
                return
            gio.press("up")
            time.sleep(0.25)
        raise DriverError("'+' 셀 이동 실패 (리스트 선택 미검출 포함)")

    def _open_create_menu(self) -> None:
        """레이어 리스트('+' 셀로 이동) → 레이어 만들기 메뉴.

        '+' 오탐으로 레이어 편집 메뉴가 열렸으면 Esc 후 '+' 재이동 (자기 치유).
        """
        self.goto_plus()
        for _ in range(4):
            gio.press("enter")
            t_end = time.time() + 1.2
            while time.time() < t_end:
                time.sleep(0.12)
                if self._menu_open():
                    return
                if self._edit_menu_open():  # '+' 오탐 → 레이어 편집 메뉴가 열림
                    break
            gio.press("esc")  # 잘못 열린 메뉴 정리
            time.sleep(0.4)
            self.goto_plus()
        raise DriverError("단계 전이 실패: 레이어 만들기 메뉴")

    def open_wizard(self) -> None:
        """레이어 리스트 → 레이어 만들기 메뉴 → 도형 선택."""
        self._open_create_menu()
        self._step("enter", lambda: self.find_highlight(self.cap()) is not None,
                   "도형 선택 화면")

    def _menu_row_index(self, img=None) -> tuple[int, int] | None:
        """만들기 메뉴의 (선택 행 인덱스, 전체 행 수). 검출 실패 시 None.

        선택 행은 흰 밴드에서 빠지고 lime 테두리로 렌더(실측 10차) →
        "선택 밴드 위에 있는 흰 밴드 수 = 선택 행 인덱스". 행 수는 흰 밴드 + 1.
        붙여넣기 행 유무로 4/5행 가변이지만 마스크(2행)는 항상 위쪽이라 무관.
        """
        if img is None:
            img = self.cap()
        bands, sel = self._menu_bands(img)
        if sel is None:
            return None
        idx = sum(1 for b in bands if b[0] < sel[0])
        return idx, len(bands) + 1

    def menu_goto_row(self, idx: int, max_steps: int = 8) -> None:
        """만들기 메뉴에서 목표 행으로 (매 스텝 재검출 폐루프)."""
        for _ in range(max_steps):
            cur = self._menu_row_index()
            if cur is None:
                raise DriverError("만들기 메뉴 선택 행 미검출")
            if cur[0] == idx:
                return
            gio.press("down" if idx > cur[0] else "up")
            time.sleep(0.3)
        raise DriverError(f"만들기 메뉴 행 이동 실패: 목표 {idx}")

    # 마스크 도형 선택 화면: 추천 그룹 탭 없음 → 기본 탭 = 탭 0. 배열은 비닐
    # 기본 탭(cell_map 탭 1)과 동일 (실측 10차). 내용 검증은 같은 참조 재사용.
    MASK_BASIC_TAB = 1  # cell_map 기준 기본 탭 (검증 참조용)

    def ensure_mask_basic_tab(self) -> None:
        """마스크 도형 선택 화면에서 기본 탭으로 (PgUp 클램프 = 왼쪽 끝이 기본)."""
        self._load_cell_map()
        n_tabs = len(self._cell_map["meta"]["tabs"]) - 1  # 추천 그룹 없음
        for _ in range(3):
            if self._verify_tab(self.cap(), self.MASK_BASIC_TAB):
                return
            for _ in range(n_tabs + 1):
                gio.press("pgup")
                time.sleep(0.22)
            time.sleep(0.5)
        raise DriverError("마스크 기본 탭 이동 실패")

    def open_mask_wizard(self, shape: str) -> None:
        """리스트 → 만들기 메뉴 3행(마스크 도안 적용) → 도형 선택 → 변형 편집.

        마스크 위저드는 색상 단계가 없어 도형 확정 Enter가 바로 변형 편집으로
        진입한다 (실측 10차). 도형은 기본 탭(A그룹)만 지원.
        """
        tab, row, col = self.shape_loc(shape)
        if tab != self.MASK_BASIC_TAB:
            raise DriverError(f"마스크 도형은 기본 탭만 지원: {shape} (탭 {tab})")
        self._open_create_menu()
        self.menu_goto_row(2)
        self._step("enter", lambda: self.find_highlight(self.cap()) is not None,
                   "마스크 도형 선택 화면")
        self.current_tab = None  # 비닐 선택 화면과 별개 탭 상태 — 캐시 무효화
        self.ensure_mask_basic_tab()
        self.select_cell(row, col)
        self._step("enter", self.in_transform_edit, "마스크 변형 편집 진입")

    @staticmethod
    def _menu_title_bottom(sub, h: int) -> int:
        """메뉴 제목(두꺼운 lime 띠)의 아래 y. 못 찾으면 옛 고정값.

        제목은 통짜 lime 밴드(높이 ~45/999)이고 선택 행 테두리는 얇은 선(~5)이다.
        위에서부터 훑어 처음 만나는 두꺼운 띠가 제목이다.
        """
        r = sub[:, :, 0].astype(np.int16)
        g = sub[:, :, 1].astype(np.int16)
        b = sub[:, :, 2].astype(np.int16)
        lime = ((r > 140) & (r < 235) & (g > 200) & (b < 110)).mean(axis=1)
        thick = int(20 / 999 * h)
        y_start = None
        for y in range(h):
            if lime[y] > 0.7 and y_start is None:
                y_start = y
            elif lime[y] <= 0.7 and y_start is not None:
                if y - y_start >= thick:
                    return y
                y_start = None
        return int(400 / 999 * h)

    # 만들기 메뉴 기하 (1776×999): 행 영역 x 560~1220
    def _menu_bands(self, img) -> tuple[list[tuple[int, int]], tuple[int, int] | None]:
        """만들기 메뉴의 (흰 행 밴드들, 선택 행 밴드).

        선택 행 = 검은 배경 + lime 테두리. 제목 밴드의 lime 글로우가 아래로
        번질 수 있어 lime 연속 구간(run) 중 마지막 구간을 선택으로 본다.

        행 시작 y는 **제목 밴드를 찾아서** 정한다. 고정값(옛 405/999)은 행 수가
        늘면 무너진다 — 메뉴가 세로 가운데 정렬이라 행이 많을수록 위로 자라고,
        차체 에디터에서 클립보드가 차면 7행이 되어 앞 두 행이 고정선 위로
        올라간다 (2026-08-17 실측: 그 때문에 '그룹 불러오기' 대신 엉뚱한 행을
        골라 레이어를 하나 만들었다). 제목 밴드는 두꺼운 lime 띠라 얇은 선택
        테두리와 두께로 갈린다.
        """
        h, w = img.shape[:2]
        x0, x1 = int(560 / 1776 * w), int(1220 / 1776 * w)
        sub = img[:, x0:x1]
        y_top = self._menu_title_bottom(sub, h)
        white = (sub.min(axis=2) > 235).mean(axis=1)
        r = sub[:, :, 0].astype(np.int16)
        g = sub[:, :, 1].astype(np.int16)
        b = sub[:, :, 2].astype(np.int16)
        lime = ((r > 140) & (r < 235) & (g > 200) & (b < 110)).mean(axis=1)
        bands, in_band, y_start = [], False, 0
        for y in range(y_top, h):
            if white[y] > 0.5 and not in_band:
                in_band, y_start = True, y
            elif white[y] <= 0.5 and in_band:
                in_band = False
                if y - y_start > int(25 / 999 * h):
                    bands.append((y_start, y))
        runs, in_run, r_start = [], False, 0
        for y in range(y_top, h):
            if lime[y] > 0.25 and not in_run:
                in_run, r_start = True, y
            elif lime[y] <= 0.25 and in_run:
                in_run = False
                if y - r_start >= max(2, int(3 / 999 * h)):  # 테두리 선(~5px)도 포함
                    runs.append((r_start, y))
        if in_run:
            runs.append((r_start, h))
        sel = runs[-1] if runs else None
        return bands, sel

    def _confirm_band(self) -> bool:
        """확인창이 떠 있나 — lime 제목 밴드(y 432~500)."""
        img = self.cap()
        h, w = img.shape[:2]
        band = img[int(432 / 999 * h):int(500 / 999 * h),
                   int(560 / 1776 * w):int(1220 / 1776 * w)]
        r = band[:, :, 0].astype(np.int16)
        g = band[:, :, 1].astype(np.int16)
        b = band[:, :, 2].astype(np.int16)
        return bool(((r > 140) & (r < 235) & (g > 200) & (b < 110)).mean() > 0.25)

    def select_cell(self, row: int, col: int, max_steps: int = 30) -> None:
        """현재 탭에서 (row,col) 셀 선택 — 클릭 우선, 실패 시 화살표 이동.

        56차 실측: 클릭은 어느 칸이든 0.21초, 화살표는 1.6~4.8초(칸당 0.25s).
        클릭 결과는 하이라이트 재검출로 확인하므로 빗나가면 화살표가 이어받는다.
        """
        w, h = gio.client_size(self.hwnd)
        for _ in range(2):
            gio.click(self.hwnd, (CELL0_CX + col * CELL_STRIDE_X) * w,
                      (CELL0_CY + row * CELL_STRIDE_Y) * h)
            t_end = time.time() + 0.8
            while time.time() < t_end:
                if self.find_highlight(self.cap()) == (row, col):
                    return
        for _ in range(max_steps):
            cur = self.find_highlight(self.cap())
            if cur is None:
                raise DriverError("선택 하이라이트 미검출")
            if cur == (row, col):
                return
            if cur[0] != row:
                key = "down" if row > cur[0] else "up"
            else:
                key = "right" if col > cur[1] else "left"
            gio.press(key)
            time.sleep(0.25)
        raise DriverError(f"셀 이동 실패: 목표 ({row},{col})")

    def confirm_shape_and_color(self, hsb: tuple[float, float, float] | None = None,
                                fav=None, tol: float = 0.0) -> None:
        """도형 확정 → 색상 확정 → 변형 편집 진입.

        hsb 지정 시: fav 스택(auto.fav.FavStack)에 있으면 즐겨찾기 인덱스 점프로
        선택(~3s), 없으면 X → HSB 미세 조정(~10s) 후 Y 등록 + 스택 갱신.
        fav=None이면 항상 HSB 경로. 미지정 시: 기본 색상(흰색) 그대로 Enter.

        `tol`은 색 축의 **받아들일 오차**다 (`set_hsb_axis`) — 부르는 쪽이
        "이 도형은 한 표시 스텝쯤 어긋나도 좋다"고 말하는 자리다.
        """
        gio.press("enter")  # → 색상 선택
        time.sleep(0.5)
        if hsb is not None:
            used_fav = False
            if fav is not None:
                idx = fav.index(hsb)
                if idx is not None:
                    from .fav import hsb_rgb
                    try:
                        self.fav_pick(idx, expect_rgb=hsb_rgb(hsb))
                        used_fav = True
                    except DriverError as e:
                        fav.forget(hsb)  # 모델 불신 — HSB 폴백 + 재등록
                        print(f"  즐겨찾기 폴백({e})")
            if not used_fav:
                self._step("x", self.in_hsb_edit, "HSB 미세 조정 진입")
                self.set_hsb(*hsb, tol=tol)
                if fav is not None:
                    self.fav_register()
                    fav.register(hsb)
        self._step("enter", self.in_transform_edit, "변형 편집 진입")

    # ---------- HSB 색상 설정 ----------
    HSB_KEYS = ("h", "s", "b")
    # 선택(연두 테두리) 감지용 행 세로 범위 (좌측 밴드 x 68..82, 1776×999 실측)
    HSB_ROW_Y = {"h": (255, 352), "s": (341, 438), "b": (427, 528)}

    def _hsb_selected(self, img) -> str | None:
        """현재 선택된 슬라이더 키. 미검출 시 None (선택 행 lime 비율 ~0.23, 비선택 <0.04)."""
        h, w = img.shape[:2]
        r = img[:, :, 0].astype(np.int16)
        g = img[:, :, 1].astype(np.int16)
        b = img[:, :, 2].astype(np.int16)
        lime = (r > 140) & (r < 235) & (g > 200) & (b < 110)
        best, best_ratio = None, 0.1
        for key, (y0, y1) in self.HSB_ROW_Y.items():
            ratio = lime[int(y0 / 999 * h):int(y1 / 999 * h),
                         int(68 / 1776 * w):int(82 / 1776 * w)].mean()
            if ratio > best_ratio:
                best, best_ratio = key, ratio
        return best

    def select_hsb_row(self, key: str, max_steps: int = 8) -> None:
        """상하 화살표로 목표 슬라이더 행 선택 (매 스텝 재검출)."""
        order = self.HSB_KEYS
        for _ in range(max_steps):
            cur = None
            # 전이 직후 한두 프레임은 선택 테두리가 아직 안 그려져 있고, 커서가
            # 세 슬라이더 **밖**(위쪽 현재/새 색 스와치)에 열리는 일도 있다.
            # 더 고약한 것은 **패널이 열렸다 곧장 닫히는** 경합이다 — X 입력이
            # 겹치면 토글돼 색 픽커로 돌아가고, 그때는 몇 프레임을 기다려도
            # 테두리가 없다 (2026-08-18 실측: top 면 글자 색에서 세 번 연속,
            # 디버그 캡처가 픽커 화면이었다). 닫혔으면 X로 다시 열고, 열려
            # 있는데 행이 안 잡히면 down으로 커서를 슬라이더로 밀어 넣는다.
            for i in range(8):
                cur = self._hsb_selected(self.cap())
                if cur is not None:
                    break
                if i >= 1:
                    if self.in_hsb_edit():
                        gio.press("down")
                    else:
                        gio.press("x")
                        time.sleep(0.8)
                time.sleep(0.4)
            if cur is None:
                img = self.cap()
                try:
                    import cv2
                    from ..paths import work_file

                    cv2.imencode(".png", img[:, :, ::-1])[1].tofile(
                        str(work_file("verify", "hsbfail.png")))
                except Exception:      # noqa: BLE001 — 디버그 캡처는 최선노력
                    pass
                raise DriverError("HSB 선택 행 미검출")
            if cur == key:
                return
            gio.press("down" if order.index(key) > order.index(cur) else "up")
            time.sleep(0.2)
        raise DriverError(f"HSB 행 선택 실패: 목표 {key}")

    # 슬라이더 트랙 기하 (1776×999): 클릭 x=100 → 0.00, x=356 → 1.00 (선형 256px),
    # 트랙 클릭 = 비례 값 점프 + 해당 행 선택 동시
    HSB_TRACK_Y = {"h": 324, "s": 410, "b": 496}
    HSB_TRACK_X0, HSB_TRACK_X1 = 100, 356

    HSB_TOL_AT = 4           # 몇 바퀴째부터 `tol`을 받아들이나

    def set_hsb_axis(self, key: str, target: float, tol: float = 0.0) -> float:
        """슬라이더 하나를 목표값으로 — 끝점 클릭 앵커 + 반 스텝 카운트 배치.

        실측 메커니즘: 화살표 1회 = 내부값 0.005(드롭 없음), 표시는 소수 2자리
        절사(표시 스텝 0.01 = press 2회). 트랙 끝 클릭은 내부값을 정확히
        0.0/1.0으로 클램프 + 행 선택 → 가까운 끝에서 목표까지 press 수를
        세어 보내면 내부값이 위상 잔차 없이 정확히 목표에 도달.

        **판독은 내부값을 반 스텝까지밖에 못 좁힌다** — 표시가 절사라 `0.07`은
        내부값 [0.070, 0.080)을 통째로 뜻한다. 그래서 잔차 보정은 늘 온전한
        표시 스텝(짝수 press)으로 나오고, 그러면 내부값이 제 표시 칸 안 어디에
        있는지(위상)가 안 바뀐다. 한 번 어긋난 위상은 그대로 왕복한다 — 실측한
        증상이 그것이다 (지붕 블랙아웃 b: 목표 0.08 ↔ 판독 0.07 왕복, 3대 중
        1대꼴). **같은 판독이 다시 오면 반 스텝을 덜 보내** 위상을 깨고 칸
        가운데에 앉힌다.

        `tol`은 그래도 못 맞췄을 때 **받아들일 차이**다. 색 한 스텝(b에서 2.5/255)
        때문에 도형을 통째로 버리는 것이 더 나쁜 자리가 있다 (`auto.itasha.
        _add_shape_job` — 실패하면 그 도형을 폐기한다). 0이면 못 맞출 때 죽는다.
        """
        target = min(1.0, max(0.0, round(target, 2)))
        h, w = self.cap().shape[:2]
        ty = self.HSB_TRACK_Y[key] / 999 * h
        # 끝점 앵커 클릭 = 행 선택 + 코스 점프. 단 노브가 끝 근처에 있으면 클릭이
        # 그랩이 되어 값이 안 변할 수 있음 — 클램프를 전제하지 않고 잔차 루프로 수렴
        anchor_right = target > 0.5
        ax = self.HSB_TRACK_X1 if anchor_right else self.HSB_TRACK_X0
        gio.click(self.hwnd, ax / 1776 * w, ty)
        time.sleep(0.25)
        if self._hsb_selected(self.cap()) != key:
            self.select_hsb_row(key)
        v = None
        seen: list[float] = []
        for k in range(8):  # 잔차 보정 폐루프 — 배치 대량 키 드롭도 잔차만 재전송
            v = ocr.read_hsb_stable(self.hwnd, key, tries=15)
            if v is None:
                raise DriverError(f"HSB {key}: 값 판독 실패")
            if abs(v - target) < 0.0051:
                return v
            if k >= self.HSB_TOL_AT and abs(v - target) <= tol + 1e-9:
                print(f"  HSB {key}: 목표 {target} ↔ 판독 {v} — 허용차 {tol} "
                      f"안이라 받는다 (표시 절사 한 스텝)")
                return v
            resid = round((target - v) * 200)
            if v in seen and abs(resid) > 1:     # 제자리 — 위상을 깬다 (반 스텝)
                resid -= 1 if resid > 0 else -1
            elif resid == 0:                     # 표시 절사 경계 — 반 스텝만 보정
                resid = 1 if target > v else -1
            seen.append(v)
            gio.press_batch("right" if resid > 0 else "left", abs(resid))
            time.sleep(0.15 + 0.002 * abs(resid))
        raise DriverError(f"HSB {key}: 수렴 실패 (목표 {target}, 현재 {v})")

    def set_hsb(self, h: float, s: float, b: float,
                tol: float = 0.0) -> dict[str, float]:
        """HSB 미세 조정 화면에서 3슬라이더를 목표값으로."""
        got = {}
        for key, target in (("h", h), ("s", s), ("b", b)):
            got[key] = self.set_hsb_axis(key, target, tol=tol)
        return got

    # ---------- 즐겨찾기 색상 팔레트 (색상 선택 화면, 2247×1264 실측·비율 환산) ----------
    # 세로 1열 리스트, 10행 표시. 행 = 색 스와치 그 자체(밴드 RGB 직독).
    # 스크롤바: 트랙 클릭 무반응, 썸 드래그만 유효(~0.7s, 목표 y 정확 착지).
    # 드래그 후 포커스 = 이전 포커스 절대 인덱스를 표시 범위로 클램프(실측):
    # 포커스 없음 → down 1회 = 최상단 표시 행 진입.
    FAV_TRACK_X = 76 / 2247            # 스크롤바 트랙 중심 x
    FAV_TRACK_Y0, FAV_TRACK_Y1 = 337 / 1264, 1038 / 1264
    FAV_ROW0_CY = 370 / 1264           # 표시 행 0 중심 y
    FAV_ROW_PITCH = 70.3 / 1264
    FAV_VISIBLE = 10
    # 서브탭 스트립(팔레트/이전 색상/즐겨찾기♥): 선택 탭 셀 = 반전(흰 비율<0.4),
    # 비선택 = 흰 박스(>0.5). PgDn은 우측 끝(즐겨찾기)에서 클램프.
    FAV_TAB_Y = (135 / 1264, 170 / 1264)
    FAV_TAB_X = {"pal": (196, 243), "prev": (252, 299), "fav": (308, 355)}  # /2247

    def color_subtab(self, img=None) -> str | None:
        """색상 선택 화면의 선택 서브탭 ('pal'|'prev'|'fav'). 판별 불가 시 None."""
        if img is None:
            img = self.cap()
        h, w = img.shape[:2]
        y0, y1 = int(self.FAV_TAB_Y[0] * h), int(self.FAV_TAB_Y[1] * h)
        ratios = {}
        for k, (x0, x1) in self.FAV_TAB_X.items():
            cell = img[y0:y1, int(x0 / 2247 * w):int(x1 / 2247 * w)]
            ratios[k] = float((cell.min(axis=2) > 180).mean())
        sel = [k for k, v in ratios.items() if v < 0.4]
        if len(sel) == 1 and all(v > 0.5 for k, v in ratios.items() if k != sel[0]):
            return sel[0]
        return None

    def ensure_fav_tab(self) -> None:
        """색상 선택 화면에서 즐겨찾기 서브탭으로 (PgDn 클램프로 절대 도달)."""
        for _ in range(3):
            if self.color_subtab() == "fav":
                return
            for _ in range(3):
                gio.press("pgdn")
                time.sleep(0.45)
        raise DriverError("즐겨찾기 서브탭 이동 실패")

    def _fav_thumb(self, img) -> tuple[int, int] | None:
        """스크롤바 썸 (y_top, y_bot) px. 스크롤바 없음(항목 ≤10) 시 None."""
        h, w = img.shape[:2]
        x = int(self.FAV_TRACK_X * w)
        y0, y1 = int(self.FAV_TRACK_Y0 * h), int(self.FAV_TRACK_Y1 * h)
        c = img[y0:y1 + 1, x - 3:x + 3].mean(axis=(1, 2))
        ys = np.where(c > 120)[0]
        if len(ys) < int(20 / 1264 * h):
            return None
        return int(ys.min()) + y0, int(ys.max()) + y0

    def _fav_focus(self, img) -> int | None:
        """포커스 행(lime 링)의 표시 행 인덱스 0~9. 미검출 시 None.

        검출은 리스트 y범위로 제한(제목 밴드·'현재' 스와치 테두리 lime 오탐 방지).
        한계: 링과 같은 lime 계열의 스와치 행이 비포커스로 존재하면 오검출 가능
        — 호출측 색 대조 검증이 최종 방어선.
        """
        h, w = img.shape[:2]
        ya = int(self.FAV_TRACK_Y0 * h) - int(12 / 1264 * h)
        yb = int(self.FAV_TRACK_Y1 * h) + int(12 / 1264 * h)
        sub = img[ya:yb, int(84 / 2247 * w):int(478 / 2247 * w)]
        r = sub[:, :, 0].astype(np.int16)
        g = sub[:, :, 1].astype(np.int16)
        b = sub[:, :, 2].astype(np.int16)
        lime = (r > 140) & (r < 235) & (g > 200) & (b < 110)
        ys = np.where(lime.mean(axis=1) > 0.5)[0]
        if len(ys) == 0:
            return None
        cy = (ys.min() + ys.max()) / 2 + ya
        k = round((cy / h - self.FAV_ROW0_CY) / self.FAV_ROW_PITCH)
        return int(k) if 0 <= k < self.FAV_VISIBLE else None

    def _fav_row_rgb(self, img, k: int) -> tuple[float, float, float]:
        """표시 행 k의 스와치 RGB (행 안쪽 밴드 평균)."""
        h, w = img.shape[:2]
        cy = (self.FAV_ROW0_CY + k * self.FAV_ROW_PITCH) * h
        half = int(18 / 1264 * h)
        inner = img[int(cy) - half:int(cy) + half,
                    int(150 / 2247 * w):int(400 / 2247 * w)]
        m = inner.reshape(-1, 3).mean(axis=0)
        return (float(m[0]), float(m[1]), float(m[2]))

    def fav_pick(self, idx: int, expect_rgb: tuple[float, float, float] | None = None,
                 tol: float = 28.0) -> None:
        """즐겨찾기 리스트에서 idx(0=맨 위) 행을 포커스 + 색 검증 (Enter는 호출측).

        썸 드래그 점프로 idx를 표시 중앙 부근에 → down/up 잔여 이동(≤9회).
        expect_rgb 지정 시 행 스와치 RGB 대조(불일치 = DriverError → HSB 폴백용).
        """
        self.ensure_fav_tab()
        img = self.cap()
        h, w = img.shape[:2]
        y0, y1 = int(self.FAV_TRACK_Y0 * h), int(self.FAV_TRACK_Y1 * h)
        track_h = y1 - y0
        top = 0
        tb = self._fav_thumb(img)
        if tb is not None:
            th = tb[1] - tb[0] + 1
            n = max(self.FAV_VISIBLE + 1,
                    round(self.FAV_VISIBLE * track_h / th))
            t_max = n - self.FAV_VISIBLE
            top = round((tb[0] - y0) / (track_h - th) * t_max)
            want = min(max(idx - 4, 0), t_max)  # 목표를 표시 중앙 부근에
            if want != top:
                ty = y0 + want / t_max * (track_h - th) + th / 2
                tx = self.FAV_TRACK_X * w
                gio.drag(self.hwnd, tx, (tb[0] + tb[1]) / 2, tx, ty)
                time.sleep(0.4)
                img = self.cap()
                tb2 = self._fav_thumb(img)
                if tb2 is None or abs(tb2[0] + th / 2 - ty) > 6:
                    raise DriverError(f"즐겨찾기 스크롤 점프 실패 (목표 top {want})")
                top = want
        k = self._fav_focus(img)
        if k is None:  # 포커스 미진입 — down 1회 = 최상단 표시 행
            gio.press("down")
            time.sleep(0.35)
            img = self.cap()
            k = self._fav_focus(img)
            if k is None:
                raise DriverError("즐겨찾기 포커스 진입 실패")
        kt = idx - top
        if not 0 <= kt < self.FAV_VISIBLE:
            raise DriverError(f"즐겨찾기 목표가 표시 범위 밖 (idx {idx}, top {top})")
        for _ in range(4):  # 잔여 이동 폐루프 (키 드롭 대비)
            if k == kt:
                break
            delta = kt - k
            for _ in range(abs(delta)):
                gio.press("down" if delta > 0 else "up")
                time.sleep(0.16)
            time.sleep(0.25)
            img = self.cap()
            k2 = self._fav_focus(img)
            if k2 is None:  # 리스트 이탈(상단 초과 등) — 재진입
                gio.press("down")
                time.sleep(0.35)
                img = self.cap()
                k2 = self._fav_focus(img)
                if k2 is None:
                    raise DriverError("즐겨찾기 포커스 소실")
            k = k2
        else:
            raise DriverError(f"즐겨찾기 행 이동 수렴 실패 (목표 {kt}, 현재 {k})")
        if expect_rgb is not None:
            got = self._fav_row_rgb(img, k)
            if max(abs(a - b) for a, b in zip(got, expect_rgb)) > tol:
                raise DriverError(
                    f"즐겨찾기 색 불일치 idx {idx}: 기대 "
                    f"{tuple(round(v) for v in expect_rgb)}, "
                    f"실측 {tuple(round(v) for v in got)}")

    def fav_register(self) -> None:
        """HSB 미세 조정 화면에서 현재 색을 즐겨찾기에 등록.

        S/B행 선택 상태에서만 Y가 등록됨(H행이면 무시, 8차 실측). 화면 내
        피드백 없음 — 검증은 이후 fav_pick의 색 대조가 담당.
        """
        if self._hsb_selected(self.cap()) == "h":
            self.select_hsb_row("b")
        gio.press("y", hold_s=0.09)
        time.sleep(0.35)

    # ---------- 변형값 설정 ----------
    # 드래그 실측(2026-08-02): 보낸 px 중 축별 데드존(x≈10px, y≈22px)이 소실되고
    # 나머지는 등방 ppu(1유닛=렌더1px) 그대로 반영. 그랩 지점 의존은 미미.
    # 상수 비율 모델은 장거리에서 과이동 → px = |유닛|×ppu + 데드존 (부호별 가산).
    # 잔차 ±3% 이하는 화살표 폐루프가 흡수 → 코스 전용.
    DRAG_DEAD_PX = {"x": 10.0, "y": 22.0}

    @staticmethod
    def _tool_icon(band: np.ndarray) -> np.ndarray:
        w = band.shape[1]
        x0, x1 = int(TOOL_ICON_REL[0] * w), int(TOOL_ICON_REL[1] * w)
        return band[:, x0:x1].astype(np.int16)

    def _icon_says(self, band: np.ndarray, tool: str) -> bool | None:
        """아이콘으로 본 현재 도구가 tool인가. 참조 미학습 시 None."""
        ref = self._tool_icons.get(tool)
        if ref is None:
            return None
        cur = self._tool_icon(band)
        if cur.shape != ref.shape:
            return None
        near = float(np.abs(cur - ref).mean())
        others = [float(np.abs(cur - r).mean())
                  for t, r in self._tool_icons.items() if t != tool and r.shape == cur.shape]
        return near < 6.0 and (not others or near < min(others))

    def _settle_tool(self, tool: str, prev: dict[str, float] | None,
                     min_wait: float = 0.06, timeout: float = 0.6,
                     retries: int = 2) -> bool:
        """도구 키 전송 직후, 그 도구가 떠 있고 값이 아는 값과 맞을 때까지 대기.

        고정 대기(0.25s) + 안정 판독(~0.15s)을 한 번의 부분 캡처 폴링으로 합친다.
        확인은 두 겹이다 — **값 두 칸이 모두 맞을 것**(한 칸만 보면 도구 키가
        드롭됐을 때 직전 도구의 값이 우연히 일치해 엉뚱한 축을 민다. 회전은 X칸
        단일 표시라 Y칸이 비어 있는 것 자체가 판별자다) + **도구 아이콘 일치**
        (값과 무관한 식별자, 첫 성공 때 학습한다).
        시간 안에 못 맞추면 도구 키를 다시 눌러 본다 — 드롭이면 이걸로 살아나고,
        끝내 실패해야 False(호출측이 정규 안정 판독으로 폴백)다.
        """
        want: dict[str, float | None] = {}
        if prev is not None:
            for box, axis in TOOL_BOXES[tool].items():
                if axis is None:
                    want[box] = None
                elif axis in prev:
                    want[box] = prev[axis]
                else:
                    want = {}
                    break
        _, h = gio.client_size(self.hwnd)
        rows = ocr.val_rows(h)
        for attempt in range(retries + 1):
            time.sleep(min_wait)
            t_end = time.time() + timeout
            while time.time() < t_end:
                band = gio.capture(self.hwnd, rows=rows)
                icon = self._icon_says(band, tool)
                if icon is False:  # 다른 도구가 떠 있다 — 값 일치는 우연일 수 있다
                    break
                if want and all(ocr.read_value_band(band, b) == v for b, v in want.items()):
                    self._tool_icons.setdefault(tool, self._tool_icon(band))
                    self.stats["settle_hit"] += 1
                    return True
            if attempt < retries:
                self.stats["settle_repress"] += 1
                gio.unlatch_menu()
                gio.press(tool)
        self.stats["settle_miss"] += 1
        return False

    def _assert_tool(self, tool: str) -> None:
        """폴백 경로 안전핀 — 아이콘으로 도구를 확인, 아니면 재전송 후 재확인."""
        _, h = gio.client_size(self.hwnd)
        rows = ocr.val_rows(h)
        for _ in range(3):
            said = self._icon_says(gio.capture(self.hwnd, rows=rows), tool)
            if said is None or said:  # 미학습(None)이면 판단을 보류한다
                return
            gio.unlatch_menu()
            gio.press(tool)
            time.sleep(0.25)
        raise DriverError(f"도구 {tool} 전환 확인 실패")

    def set_move_xy(self, tx: float, ty: float,
                    prev: dict[str, float] | None = None) -> dict[str, float]:
        """이동 x·y 동시 설정 — 드래그 1회 코스 배치 후 축별 화살표 보정.

        드래그 끝점은 캔버스 안쪽으로 클램프(레이어 리스트·클립보드·핫키 바 회피,
        부분 드래그 잔차는 set_axis의 홀드+화살표가 처리). 도형 중심을 그랩.
        prev = 소프트웨어가 아는 현재 변형 상태 (맞으면 판독 생략, 아니면 자동 폴백).
        """
        gio.press("1")
        if self._settle_tool("1", prev):
            x0, y0 = prev["x"], prev["y"]
        else:
            x0 = ocr.read_stable(self.hwnd, "x")
            y0 = ocr.read_stable(self.hwnd, "y")
        w, h = gio.client_size(self.hwnd)
        if x0 is None or y0 is None:
            raise DriverError("이동: 값 판독 실패")
        # 1유닛 = 렌더 1px, 클라 스케일 = 클라높이/렌더세로 (창 크기 무관 실측).
        # 1440 = 게임 그래픽 설정의 렌더 해상도 — 다른 설정에선 코스가 어긋나지만
        # 드래그는 코스 전용이라 화살표 폐루프가 흡수한다.
        ppu = h / 1440.0
        dragged = False
        if max(abs(tx - x0), abs(ty - y0)) > 60.0:  # 드래그가 홀드보다 이득인 거리
            sx = w / 2 + x0 * ppu
            sy = h / 2 - y0 * ppu
            if 0 < sx < w and 0 < sy < h:  # 그랩 지점이 화면 안일 때만
                dux, duy = tx - x0, ty - y0
                dxp = abs(dux) * ppu + self.DRAG_DEAD_PX["x"] if abs(dux) > 2 else 0.0
                dyp = abs(duy) * ppu + self.DRAG_DEAD_PX["y"] if abs(duy) > 2 else 0.0
                ex = sx + (dxp if dux > 0 else -dxp)
                ey = sy - (dyp if duy > 0 else -dyp)  # 화면 y+아래 = 유닛 y-
                # UI 회피 가드 (1776×999 실측 240/90/40px의 비율 환산 — 절대 px면
                # 큰 창에서 레이어 리스트·핫키 바가 비율만큼 커져 가드를 뚫는다)
                ex = min(w * (1 - 240 / 1776), max(w * 240 / 1776, ex))
                ey = min(h * (1 - 90 / 999), max(h * 40 / 999, ey))
                if abs(ex - sx) > 8 or abs(ey - sy) > 8:
                    gio.drag(self.hwnd, sx, sy, ex, ey)
                    time.sleep(0.15)
                    dragged = True
        # 보정: 드래그 착지 오차는 평균 20/11유닛(56차 실측)이라 홀드 한 번을 더
        # 얹고, 마무리는 두 축을 묶어 판독을 한 번만 문다.
        targets = {"x": tx, "y": ty}
        cur = {"x": x0, "y": y0}
        if dragged:
            r = ocr.read_stable_multi(self.hwnd, ("x", "y"), tries=40)
            if r is None:
                raise DriverError("이동: 드래그 후 판독 실패")
            cur = {"x": r["x"], "y": r["y"]}
        if any(abs(targets[a] - cur[a]) > TOOL_AXES[a][5] * MIN_HOLD_S
               for a in ("x", "y")):
            cur = self._coarse_pair(("x", "y"), targets, cur)
        return self._fine_pair(("x", "y"), targets, v0=cur)

    def set_scale(self, tsx: float, tsy: float,
                  prev: dict[str, float] | None = None) -> dict[str, float]:
        """크기 sx·sy 동시 설정 — 도구 키 1회, 홀드 코스, 쌍 마무리."""
        gio.press("2")
        ok = self._settle_tool("2", prev)
        cur = {"sx": prev["sx"], "sy": prev["sy"]} if (ok and prev) else None
        if cur is None:
            self._assert_tool("2")
            r = ocr.read_stable_multi(self.hwnd, ("x", "y"), tries=40)
            if r is None:
                raise DriverError("크기: 값 판독 실패")
            cur = {"sx": r["x"], "sy": r["y"]}
        targets = {"sx": tsx, "sy": tsy}
        if any(abs(targets[a] - cur[a]) > TOOL_AXES[a][5] * MIN_HOLD_S
               for a in ("sx", "sy")):
            cur = self._coarse_pair(("sx", "sy"), targets, cur)
        return self._fine_pair(("sx", "sy"), targets, v0=cur)

    def _coarse_pair(self, names: tuple[str, str], targets: dict[str, float],
                     cur: dict[str, float]) -> dict[str, float] | None:
        """같은 도구의 두 축을 홀드로 정지 밴드까지 몰아넣는다 (판독은 라운드당 1회).

        **반복해야 한다** — 한 번만 치면 이동량이 홀드 상한(1회 최대 ~2s)을 넘는
        경우(예: 크기 1.00→14.48) 나머지가 통째로 화살표로 넘어가 배치 상한
        200스텝씩 기어간다 (56차 추적에서 장당 17~21초). 속도는 축별로 적응한다.
        """
        boxes = tuple(TOOL_AXES[n][1] for n in names)
        speed = {n: TOOL_AXES[n][5] for n in names}
        keys = {n: TOOL_AXES[n][3] for n in names}
        for _ in range(8):
            plan: dict[str, float] = {}
            for name in names:
                _, _, _, _, _, _, a0 = TOOL_AXES[name]
                err = targets[name] - cur[name]
                if abs(err) <= a0 + speed[name] * MIN_HOLD_S:
                    continue
                dur = min(2.0, max(MIN_HOLD_S,
                                   HOLD_UNDERSHOOT * (abs(err) - a0) / speed[name]))
                kp, kn = keys[name]
                gio.hold(kp if err > 0 else kn, dur)
                plan[name] = dur
            if not plan:
                return cur
            time.sleep(0.06)
            r = ocr.read_stable_multi(self.hwnd, boxes, tries=40)
            if r is None:
                return None
            new = {n: r[TOOL_AXES[n][1]] for n in names}
            for name, dur in plan.items():
                moved = abs(new[name] - cur[name])
                a0 = TOOL_AXES[name][6]
                if moved > 0 and dur > MIN_HOLD_S * 1.5:
                    speed[name] = 0.5 * speed[name] + 0.5 * max(1e-6, moved - a0) / dur
            cur = new
        return cur

    def _fine_pair(self, names: tuple[str, str], targets: dict[str, float],
                   v0: dict[str, float] | None = None) -> dict[str, float]:
        """같은 도구의 두 축을 함께 화살표로 마무리 — 판독을 한 번으로 합친다.

        축마다 따로 돌리면 롤 정착 대기(전체 시간의 큰 몫)를 두 번 문다. 두 축의
        값 칸은 한 띠에 같이 들어오므로, 두 축분 화살표를 연달아 보내고 **한 번**
        안정 판독한다. 게인·방향 뒤집힘 적응은 축별로 따로 둔다.
        """
        boxes = tuple(TOOL_AXES[n][1] for n in names)
        steps = {n: TOOL_AXES[n][2] for n in names}
        arrows = {n: TOOL_AXES[n][4] for n in names}
        eff = {n: TOOL_AXES[n][2] for n in names}  # 프레스당 실측 이동량 학습
        cur: dict[str, float] | None = dict(v0) if v0 else None
        for _ in range(10):
            if cur is None:
                r = ocr.read_stable_multi(self.hwnd, boxes, tries=40)
                if r is None:
                    continue
                cur = {n: r[TOOL_AXES[n][1]] for n in names}
            sent: dict[str, tuple[int, float]] = {}
            for n in names:
                err = targets[n] - cur[n]
                if abs(err) < steps[n] * 0.51:
                    continue
                cnt = min(200, round(abs(err) / eff[n]))
                if cnt == 0:      # 한 프레스가 오차보다 크다 — 이 축은 여기가 최선
                    continue
                ap, an = arrows[n]
                gio.press_batch(ap if err > 0 else an, cnt)
                sent[n] = (cnt, err)
            if not sent:
                return cur
            if TRACE:
                print(f"    fine{names} 목표{targets} 현재{cur} 전송{sent}")
            time.sleep(0.05)
            r = ocr.read_stable_multi(self.hwnd, boxes, tries=40)
            if r is None:
                cur = None
                continue
            new = {n: r[TOOL_AXES[n][1]] for n in names}
            for n, (cnt, err) in sent.items():
                moved = new[n] - cur[n]
                if moved == 0:
                    continue
                # 근거 있는 역방향은 몇 번이든 뒤집는다 (_arrow_fine과 같은 근거)
                if (moved > 0) != (err > 0) and abs(moved) > 2 * eff[n]:
                    arrows[n] = (arrows[n][1], arrows[n][0])
                eff[n] = max(steps[n] * 0.25,
                             0.6 * eff[n] + 0.4 * abs(moved) / cnt)
            cur = new
        raise DriverError(f"{names}: 미세 보정 수렴 실패 (목표 {targets}, 현재 {cur})")

    def _arrow_fine(self, axis: str, box: str, step: float, ap: str, an: str,
                    target: float, is_rot: bool = False,
                    v0: float | None = None, slow: bool = False) -> float:
        """화살표 배치 전송 → 일괄 판독 검증 반복 (반영률 적응 게인).

        v0 = 소프트웨어가 아는 현재값. 주면 첫 판독을 생략하고 바로 보낸다
        (틀려도 전송 뒤 판독이 잡아 다음 회차에서 수렴).
        slow = 프레스를 30ms/20ms로 보낸다 — 병리 축 복구용 (front 실측:
        4ms 배치는 10~30% 유실, 느린 전송은 100% 반영).
        """
        kw = {"hold_s": 0.03, "gap_s": 0.02} if slow else {}
        eff = step        # 프레스당 실측 이동량 — 반영률을 온라인으로 배운다
        pinned = 0        # 전송에도 값이 안 움직인 연속 횟수 — 클램프 조기 감지
        v = v0
        for _ in range(12 if slow else 10):
            if v is None:
                v = ocr.read_stable(self.hwnd, box, tries=20)
                if v is None:
                    continue
            err = _wrap_err(target, v, is_rot)
            if abs(err) < step * 0.51:
                return v
            n = min(200, round(abs(err) / eff))
            if n == 0:
                # 한 프레스가 오차보다 크게 움직이는 축(front류 증폭 축) —
                # 지금 값이 도달 가능한 최선이다
                return v
            gio.press_batch(ap if err > 0 else an, n, **kw)
            # 롤 애니메이션이 전송보다 느리다 — 고정 대기 대신 폴링이 기다린다
            time.sleep(0.05)
            v3 = ocr.read_stable(self.hwnd, box, tries=40)
            if TRACE:
                print(f"    fine[{axis}] 목표{target:g} 현재{v:g} "
                      f"전송{'+' if err > 0 else '-'}{n} → {v3}", flush=True)
            if v3 is None:
                v = None
                continue
            moved = _wrap_err(v3, v, is_rot)
            if moved != 0:
                pinned = 0
                # 근거 있는 역방향은 몇 번이든 뒤집는다 — 한 번 잠그면 노이즈
                # 반전 뒤 영영 역주행한다 (2026-08-19 front 실측: x가 클램프까지)
                if (moved > 0) != (err > 0) and abs(moved) > 2 * eff:
                    ap, an = an, ap
                eff = max(step * 0.25, 0.6 * eff + 0.4 * abs(moved) / n)
            else:
                pinned += 1
                if pinned >= 3:
                    raise DriverError(
                        f"{axis}: 값이 {v}에 꽂혀 있다 (목표 {target} — 클램프)")
            v = v3
        raise DriverError(f"{axis}: 미세 보정 수렴 실패 (목표 {target}, 현재 {v})")

    def set_axis(self, axis: str, target: float,
                 prev: dict[str, float] | None = None,
                 press_tool: bool = True, gentle: bool = False) -> float:
        """축 하나를 목표값으로 (도구 전환 포함). 반환: 최종 판독값.

        prev = 소프트웨어가 아는 현재 변형 상태 (도구 전환 확인과 동시에 판독 생략).
        press_tool=False = 같은 도구를 쓰는 축을 방금 처리해 키를 다시 안 눌러도 됨.
        gentle=True = 홀드 없이 **느린 화살표만** — front류 병리 축(랩·증폭·유실)
        복구용. 홀드는 그 축에서 상태를 망가뜨리고 4ms 배치는 10~30% 유실되지만,
        느린 화살표(30ms)는 100% 반영이 실측돼 있다 (2026-08-19 챌린저 front).
        """
        tool, box, step, (kp, kn), (ap, an), speed, a0 = TOOL_AXES[axis]
        is_rot = axis == "rot"
        if press_tool:
            gio.press(tool)
            ok = self._settle_tool(tool, prev)
        else:  # 같은 도구를 쓰는 앞 축이 이미 눌러 확인했다
            ok = prev is not None
        v = prev[axis] if (ok and prev is not None) else None
        if v is None:
            self._assert_tool(tool)
            v = ocr.read_stable(self.hwnd, box)
        if v is None:
            raise DriverError(f"{axis}: 값 판독 실패(도구 {tool})")
        if gentle:
            return self._arrow_fine(axis, box, step, ap, an, target, is_rot,
                                    v0=v, slow=True)
        fresh = True  # v가 최신 판독인가 (홀드 후 판독 실패 시 False → 화살표 단계에서 재판독)
        pinned = 0   # 홀드에도 값이 안 움직인 연속 횟수 — 클램프 조기 감지
        # 대략: 홀드 비례 제어 (속도·시동량 적응, 방향 자동 감지).
        # 정지 밴드 = 최소 홀드가 내는 이동량 — 이보다 가까우면 홀드로는 못 줄인다
        for _ in range(20):
            stop_band = a0 + speed * MIN_HOLD_S
            err = _wrap_err(target, v, is_rot)
            if abs(err) <= stop_band:
                break
            dur = min(2.0, max(MIN_HOLD_S,
                               HOLD_UNDERSHOOT * (abs(err) - a0) / speed))
            if is_rot:
                # 스윙을 60° 이하로 제한 — mod 360 랩핑에 의한 방향 오판 방지
                dur = min(dur, 60.0 / speed)
            gio.hold(kp if err > 0 else kn, dur)
            time.sleep(0.08)  # 나머지 정착 대기는 read_stable 폴링이 적응적으로 맡는다
            v2 = ocr.read_stable(self.hwnd, box, tries=30)
            if TRACE:
                print(f"    hold[{axis}] 목표{target:g} 현재{v:g} "
                      f"{kp if err > 0 else kn}×{dur:.2f}s → {v2} (속도추정 {speed:.1f})",
                      flush=True)
            if v2 is None:
                fresh = False
                continue
            moved = _wrap_err(v2, v, is_rot)
            if moved != 0:
                pinned = 0
                # 근거 있는 역방향은 **몇 번이든** 뒤집는다 — 한 번 잠그면 노이즈
                # 반전 한 번에 영영 역주행해 클램프까지 달아난다 (2026-08-19
                # 챌린저 front 실측: x 목표 125.2에 -343). 문턱은 정지 밴드 —
                # 이보다 작은 이동은 시동량·롤 노이즈일 수 있어 방향 근거가 못 된다.
                if (moved > 0) != (err > 0) and abs(moved) > stop_band:
                    kp, kn = kn, kp  # 홀드키 방향이 반대인 축(예: 회전, front 미러)
                # 반전 홀드도 속도 근거다 — front 스케일처럼 방향이 반대이고
                # 몇십 배 빠른 축은 여기서 속도를 안 배우면 다음 홀드가 또
                # 클램프까지 날아가 진동한다 (실측: sx 1.0→-60, ~230/s vs 4.75/s)
                if dur > MIN_HOLD_S * 1.5:  # 시동량이 지배하지 않는 홀드만 속도 근거
                    speed = 0.6 * speed + 0.4 * max(1e-6, abs(moved) - a0) / dur
            else:
                pinned += 1
                if pinned >= 3 and dur >= 0.1:
                    break  # 실한 홀드에도 값이 꽂혀 있다 = 이동 클램프 — 홀드로는 끝
            v = v2
            fresh = True
        # 마무리: 오차만큼 화살표 배치 전송 → 일괄 판독 검증 반복
        # (변형 축 반영률 100% 실측 → 게인 1.0 시작, 반영률 변동 대비 적응)
        return self._arrow_fine(axis, box, step, ap, an, target, is_rot,
                                v0=v if fresh else None)

    def set_axis_soft(self, axis: str, target: float,
                      press_tool: bool = True, log=print) -> float | None:
        """이동 축을 목표로 두되 **못 닿아도 안 죽는다**. 반환: 최종 판독값.

        면마다 이동 범위가 클램프돼 있고(실측: `top`의 x는 ±554에서 멈춘다) 홀드
        폐루프가 어긋나는 면도 있다 (2026-08-19 챌린저 front x가 목표 125.2에
        -184.5/-343 · 2026-08-21 인테그라 front x가 목표 9.4에 8.0). 못 닿으면
        정착을 기다렸다가 **느린 화살표 전용**으로 한 번 더 가고(느린 전송은
        100% 반영이 실측돼 있어 단조 접근이 보장된다), 그래도 안 되면 멈춘
        자리에 두고 간다.

        **몇 유닛 비껴 앉는 것이 40분짜리 실행이 죽는 것보다 낫다** — 자리는
        면 캡처와 카운터가 여전히 검증한다. 크기(스케일)는 이 완충을 안 쓴다:
        틀린 크기는 그림을 바꾸지만 몇 유닛 이동은 안 바꾼다.
        """
        try:
            return self.set_axis(axis, target, press_tool=press_tool)
        except DriverError:
            pass
        time.sleep(2.0)
        try:
            got = self.set_axis(axis, target, press_tool=True, gentle=True)
            log(f"    {axis} 느린 화살표 재시도로 앉혔다 (목표 {target:g})")
            return got
        except DriverError as e:
            v = ocr.read_stable(self.hwnd, "y" if axis == "y" else "x", tries=20)
            log(f"    {axis}가 {v}에서 멈춘다 (목표 {target:g}) — 그대로 둔다 ({e})")
            return v

    def set_alpha(self, target: float, prev: dict[str, float] | None = None,
                  press_tool: bool = True) -> float:
        """투명도(도구 5)를 목표 표시값으로. 반환: 최종 판독값.

        폐루프 규약은 다른 축과 같다 — 도구 전환 확인(값 두 칸 + 아이콘 두 겹) →
        배치 전송 → 안정 판독 → 잔차 재계산. 다른 것은 스텝뿐이라 표시값이 아니라
        **8비트로 환산해** 내림·올림 횟수를 푼다 (위 ALPHA_* 주석).
        코스 홀드 단계는 이 축에 없다 — 실측으로 제어 가능한 홀드 구간이 없다.

        prev = 소프트웨어가 아는 현재 변형 상태. 맞으면 첫 판독을 생략한다
        (새 레이어는 항상 100에서 시작하므로 painter 경로는 늘 이 길로 간다).
        """
        tgt8 = min(ALPHA_MAX8, max(ALPHA_MIN8, alpha8(target)))
        if press_tool:
            gio.press(ALPHA_TOOL)
            ok = self._settle_tool(ALPHA_TOOL, prev)
        else:  # 같은 도구를 쓰는 앞 축이 이미 눌러 확인했다 (지금은 투명도 단독)
            ok = prev is not None
        v = prev.get("alpha") if (ok and prev is not None) else None
        if v is None:
            self._assert_tool(ALPHA_TOOL)
            v = ocr.read_stable(self.hwnd, ALPHA_BOX)
        if v is None:
            raise DriverError("투명도: 값 판독 실패")
        for _ in range(8):
            cur8 = alpha8(v)
            if cur8 == tgt8:
                return v
            for key, n in _alpha_batches(cur8, tgt8):
                gio.press_batch(key, n)
            time.sleep(0.05)
            v2 = ocr.read_stable(self.hwnd, ALPHA_BOX, tries=40)
            if v2 is None:  # 롤이 길어졌을 뿐일 수 있다 — 한 번 더 기다려 본다
                v2 = ocr.read_stable(self.hwnd, ALPHA_BOX, tries=40)
                if v2 is None:
                    raise DriverError("투명도: 전송 후 판독 실패")
            if TRACE:
                print(f"    alpha 목표{tgt8} 현재{cur8} → {alpha8(v2)}")
            v = v2
        raise DriverError(f"투명도: 수렴 실패 (목표 {target}, 현재 {v})")

    # ---------- 마무리 ----------
    def commit(self) -> None:
        """변형 편집에서 Enter → 레이어 확정, 리스트 복귀(양성 확인)."""
        self._step("enter", lambda: not self.in_transform_edit(), "레이어 커밋")
        for _ in range(15):  # 리스트 도착 대기 (전환 애니메이션)
            img = self.cap()
            if self._edit_menu_open(img):  # 잉여 Enter로 열린 편집 메뉴 정리
                gio.press("esc")
                time.sleep(0.4)
                continue
            if self.list_selection(img) is not None:
                return
            time.sleep(0.15)
        raise DriverError("커밋 후 리스트 미도착")

    def discard(self) -> None:
        """변형 편집에서 Esc×3 → 미확정 폐기, 리스트 복귀 (Esc 과다 금지)."""
        for _ in range(3):
            gio.press("esc")
            time.sleep(0.6)
        if self.in_transform_edit():
            raise DriverError("폐기 실패: 여전히 변형 편집")

    # ---------- 비닐 그룹 저장 (실측 2026-08-02) ----------
    # 플로우: 리스트 Backspace → 슬롯 그리드(기본 선택=새 저장 슬롯, lime 링)
    # → Enter → "저장 파일 이름 지정" 텍스트 대화상자(기본 텍스트 "Forza" 잔존, 64자)
    # → 클리어+이름 입력 → Enter → 저장 처리 → "공유" 대화상자 → Esc(공유 안 함) → 리스트

    def _save_screen(self) -> str | None:
        """저장 플로우 화면 판별: 'slots' | 'name' | 'share' | 'list' | None.

        판별 순서 중요: 슬롯 화면의 좌측 "인기도" lime 밴드(y 727~765)가
        list_selection의 선택 링 검출에 걸리므로 슬롯/대화상자 확인이 먼저.
        중앙 lime 제목 밴드 세로 위치로 이름(y 402~462)/공유(y 363~428) 구분.
        """
        img = self.cap()
        h, w = img.shape[:2]

        def lime_rows(x0u: int, x1u: int) -> np.ndarray:
            x0, x1 = int(x0u / 1776 * w), int(x1u / 1776 * w)
            r = img[:, x0:x1, 0].astype(np.int16)
            g = img[:, x0:x1, 1].astype(np.int16)
            b = img[:, x0:x1, 2].astype(np.int16)
            return ((r > 140) & (r < 235) & (g > 200) & (b < 110)).mean(axis=1)

        center = lime_rows(600, 1180)
        ys = np.where(center > 0.5)[0]
        if len(ys) > int(20 / 999 * h):
            mid = ys.mean() / h * 999
            if 350 <= mid <= 430:
                return "share"
            if 430 < mid <= 480:
                return "name"
        # 슬롯 화면: 좌측 정보 패널의 "인기도" solid lime 밴드 (x 95~345, y 715~775)
        left = lime_rows(95, 345)
        y0, y1 = int(715 / 999 * h), int(775 / 999 * h)
        if (left[y0:y1] > 0.5).sum() > int(25 / 999 * h):
            return "slots"
        if self.list_selection(img) is not None:
            return "list"
        return None

    def save_group(self, name: str, wait_s: float = 30.0) -> None:
        """레이어 리스트에서 현재 레이어 전체를 새 비닐 그룹 슬롯에 저장.

        신규 슬롯 전용 (슬롯 화면의 기본 선택 = 새 저장 슬롯). 공유는 건너뜀.
        """
        s = self._save_screen()
        if s == "list":
            self._step("backspace", lambda: self._save_screen() == "slots", "저장 슬롯 화면")
        elif s != "slots":  # 이미 슬롯 화면이면 바로 진행
            raise DriverError(f"저장 시작: 예상 밖 화면({s})")
        self._step("enter", lambda: self._save_screen() == "name", "이름 입력 대화상자")
        gio.press_batch("backspace", 70, hold_s=0.02, gap_s=0.02)  # 기본 텍스트 클리어(64자 상한)
        gio.type_text(name)
        time.sleep(0.4)
        gio.press("enter")  # 저장 처리(가변 시간) → 공유 대화상자
        t_end = time.time() + wait_s
        name_seen: float | None = None
        while True:
            if time.time() >= t_end:
                raise DriverError("저장 후 공유 대화상자/리스트 미도착")
            s = self._save_screen()
            if s == "share":
                break
            if s == "list":  # 공유 없이 바로 복귀하는 경로 대비
                return
            if s == "name":  # Enter 드롭 — 이름 대화상자가 2.5초+ 지속될 때만 재전송
                now = time.time()  # (공유 화면 재전송 = 공유 실행이므로 금지)
                if name_seen is None:
                    name_seen = now
                elif now - name_seen > 2.5:
                    gio.press("enter")
                    name_seen = None
            else:
                name_seen = None
            time.sleep(0.5)
        # 공유 안 함 — 대화상자 표시 직후 수 초간 입력 무시(저장 처리 중) 실측(9차),
        # Esc 드롭 대비 재시도. 복귀 지점(리스트/슬롯) 모두 수용
        time.sleep(1.5)
        self._step("esc", lambda: self._save_screen() in ("list", "slots"), "공유 취소",
                   tries=5, wait=2.5)
        if self._save_screen() == "slots":
            self._step("esc", lambda: self._save_screen() == "list", "슬롯 화면 닫기")
