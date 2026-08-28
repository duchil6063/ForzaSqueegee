r"""차체 에디터(비닐 & 데칼 적용) 화면을 **화면으로** 읽는 자들.

비닐 그룹 에디터와 같은 계열의 UI지만 좌표가 다르다 — 면 탭 스트립이 맨 위에
있고, 레이어 카운터가 비닐 쪽보다 한 칸 아래에 있다 (`game.ocr.read_body_count`).

탭 스트립 (2026-08-17 실측, 1600×899 클라, 셀 y 41..78 · 밑줄 y 76..78):
- 셀 13개 = 좌 스크롤 화살표 + **면 탭 11개** + 우 스크롤 화살표.
- 비선택 셀은 흰 박스, **선택 셀은 반전(검정) + 아래 라임 밑줄**.
- PgUp/PgDn으로 이동하고 양끝에서 클램프한다 (순환 없음).
- **양끝에서는 그쪽 화살표 셀이 회색으로 죽는다** — 흰 박스 세기로 자리를 세면
  그 순간 한 칸씩 밀린다. 그래서 자리는 **라임 밑줄의 x중심** 하나로 읽고,
  중심↔인덱스 대조표는 실측 표(`catalog/body_tabs.json`)가 쥔다.

스트립은 스크롤되지 않는다(탭 11개가 늘 한 화면) — 그래서 중심 좌표가 인덱스의
안정된 이름표가 된다. 흰 구간은 셀 띠 **전체**의 흰 행 비율로 잡는다: 아이콘이
셀 가운데를 가로질러 가운데 몇 행만 보면 한 셀이 둘로 갈라진다.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

import numpy as np

# 탭 셀 띠의 세로 구간 (클라 높이 비율) — 셀 위/아래 테두리 안쪽
TAB_ROW_REL = (0.046, 0.086)
# 선택 밑줄 띠 (셀 아래 라임 3px)
UNDERLINE_REL = (0.075, 0.095)
WHITE_FRAC = 0.35        # 이 비율 이상 흰 행이면 셀 안쪽 열
MIN_RUN_REL = 0.02       # 이보다 좁은 흰 구간은 잡티
LINK_GAP_REL = 0.09      # 이보다 먼 흰 구간은 스트립 바깥 (배경 밝은 벽 등)
UNDERLINE_FRAC = 0.08    # 밑줄 띠에서 라임 행 비율 문턱


def _tab_table_path() -> Path:
    return Path(__file__).resolve().parents[2] / "catalog" / "body_tabs.json"


@cache
def tab_table() -> dict:
    """면 탭 실측 표 (`catalog/body_tabs.json`). 없으면 빈 표."""
    p = _tab_table_path()
    if not p.exists():
        return {"tabs": []}
    return json.loads(p.read_text(encoding="utf-8"))


def surfaces() -> list[dict]:
    """실측표의 면 목록 (index·center·cap·name·ko)."""
    return list(tab_table().get("tabs") or [])


def surface_names(names: list[str] | None = None) -> list[str]:
    """이 차가 가진 면 이름들. `names`(설치 파일이 준 그 차의 목록)가 이긴다."""
    return list(names) if names else [t["name"] for t in surfaces() if t.get("name")]


def surface_index(surface: str | int, names: list[str] | None = None) -> int:
    """면 이름(또는 인덱스 그대로) → 탭 인덱스. 모르는 이름이면 ValueError.

    `names`는 **그 차의 면 목록**이다 (`game.cars.tabs_of` — 설치 파일이 준
    정식 순서). 주면 그것이 이긴다: 실측표는 잰 차 한 대의 것이라 스포일러
    (설치본 636대 중 362대)나 선루프(60대)가 있는 차에서 이름조차 모른다.
    """
    if isinstance(surface, int) or str(surface).isdigit():
        return int(surface)
    if names:
        if surface in names:
            return names.index(surface)
        raise ValueError(f"이 차에 없는 면이다: {surface} "
                         f"(있는 면: {', '.join(names)})")
    for t in surfaces():
        if t.get("name") == surface:
            return int(t["index"])
    raise ValueError(f"모르는 차체 면 이름: {surface} "
                     f"(아는 것: {', '.join(surface_names()) or '없음'})")


def surface_cap(surface: str | int, names: list[str] | None = None) -> int | None:
    """그 면의 레이어 상한. 모르면 None.

    상한은 **면 이름이 정한다** (옆면·윗면 3,000 · 나머지 1,000 — 설치 파일의
    `carfiles.TAB_CAPS`). 실측표는 그 값을 인덱스에 적어 둔 사본일 뿐이라,
    표에 없는 면은 이름으로 곧장 답한다.
    """
    if not isinstance(surface, int) and not str(surface).isdigit():
        from . import carfiles
        if names or surface not in surface_names():
            surface_index(surface, names)         # 이 차에 없는 면이면 여기서 죽는다
            return carfiles.TAB_CAPS.get(str(surface), 1000)
    idx = surface_index(surface, names)
    for t in surfaces():
        if int(t["index"]) == idx:
            return t.get("cap")
    return None


def _lime(a: np.ndarray) -> np.ndarray:
    r = a[:, :, 0].astype(np.int16)
    g = a[:, :, 1].astype(np.int16)
    b = a[:, :, 2].astype(np.int16)
    return (r > 140) & (r < 235) & (g > 200) & (b < 110)


def _white_runs(img: np.ndarray) -> list[tuple[int, int]]:
    h, w = img.shape[:2]
    y0, y1 = int(TAB_ROW_REL[0] * h), int(TAB_ROW_REL[1] * h)
    white = (img[y0:y1].min(axis=2) > 200).mean(axis=0) > WHITE_FRAC
    runs: list[tuple[int, int]] = []
    s = None
    for x in range(w):
        if white[x] and s is None:
            s = x
        elif not white[x] and s is not None:
            runs.append((s, x - 1))
            s = None
    if s is not None:
        runs.append((s, w - 1))
    return [r for r in runs if r[1] - r[0] > MIN_RUN_REL * w]


def strip_runs(img: np.ndarray) -> list[tuple[int, int]]:
    """스트립에 속한 흰 셀 구간만 (가장 긴 사슬) — 배경의 밝은 벽을 떼어 낸다."""
    runs = _white_runs(img)
    if not runs:
        return []
    w = img.shape[1]
    chains: list[list[tuple[int, int]]] = [[runs[0]]]
    for prev, cur in zip(runs, runs[1:]):
        if cur[0] - prev[1] <= LINK_GAP_REL * w:
            chains[-1].append(cur)
        else:
            chains.append([cur])
    return max(chains, key=len)


def selected_center(img: np.ndarray) -> float | None:
    """선택 탭 라임 밑줄의 x중심 (클라 폭 비율 0~1). 미검출 시 None."""
    h, w = img.shape[:2]
    band = img[int(UNDERLINE_REL[0] * h):int(UNDERLINE_REL[1] * h)]
    cols = np.where(_lime(band).mean(axis=0) > UNDERLINE_FRAC)[0]
    if len(cols) == 0:
        return None
    return float(cols.mean()) / w


def tab_centers(n: int | None = None,
                table: dict | None = None) -> list[tuple[int, float]]:
    """탭 인덱스 → 셀 중심 (클라 폭 비율) — 실측 표 + **등간격 외삽**.

    스트립은 왼쪽 붙박이 등간격이라 (실측 1600×899: 첫 칸 0.055 · 간격 0.05)
    표를 잰 차보다 면이 많은 차의 뒤쪽 탭은 표의 간격으로 외삽한다 — 실측:
    표는 9탭 차(줄리아)인데 CRX 뮤겐은 스포일러가 끼어 10탭이라, 표 길이를
    그대로 쓰면 마지막 탭(window_right)이 "범위 밖"으로 죽는다.
    """
    tabs = sorted((table or tab_table()).get("tabs") or [],
                  key=lambda t: int(t["index"]))
    if not tabs:
        return []
    got = [(int(t["index"]), float(t["center"])) for t in tabs]
    if n is None or n <= got[-1][0] + 1:
        return got
    if len(got) >= 2:
        diffs = sorted(b - a for (_i, a), (_j, b) in zip(got, got[1:]))
        pitch = diffs[len(diffs) // 2]
    else:
        pitch = 0.05
    last_i, last_c = got[-1]
    for i in range(last_i + 1, n):
        last_c += pitch
        got.append((i, last_c))
    return got


def selected_tab(img: np.ndarray, table: dict | None = None,
                 n: int | None = None) -> int | None:
    """선택된 면 탭 인덱스. 실측 표가 없거나 중심을 못 읽으면 None.

    중심에서 가장 가까운 탭을 고르되 **반 칸보다 멀면 버린다** — 스트립이 아닌
    화면에서 우연히 잡힌 라임을 탭으로 오인하지 않기 위한 문턱이다. `n`은 이
    차의 면 수다 (표를 잰 차와 다르면 뒤쪽 탭 중심을 외삽한다).
    """
    cx = selected_center(img)
    if cx is None:
        return None
    tabs = tab_centers(n, table)
    if not tabs:
        return None
    best = min(tabs, key=lambda t: abs(t[1] - cx))
    pitch = (max(c for _i, c in tabs) - min(c for _i, c in tabs)) \
        / max(1, len(tabs) - 1)
    return int(best[0]) if abs(best[1] - cx) < 0.5 * pitch else None


def occluded_span(img: np.ndarray) -> tuple[float, float] | None:
    """알림 배너가 탭 스트립을 덮는 x구간 (클라 폭 비율). 없으면 None.

    멀티플레이 서버 알림 배너(노란 제목 띠 + 흰 본문 상자)가 화면 상단 가운데를
    덮으면 그 아래 탭 셀·선택 밑줄이 통째로 안 보인다 (실측: 탭 3~7 검출 사망).
    배너 열의 지문은 **제목 띠의 채도 있는 노랑**이다 — 흰 벽 배경은 노랑을 못
    낸다. 본문 상자가 밑줄 띠까지 희게 덮는 것도 함께 본다.
    """
    h, w = img.shape[:2]
    title = img[int(0.030 * h):int(0.070 * h)]
    r = title[:, :, 0].astype(np.int16)
    g = title[:, :, 1].astype(np.int16)
    b = title[:, :, 2].astype(np.int16)
    yellow = ((r > 200) & (g > 170) & (b < 120)).mean(axis=0) > 0.3
    body_band = img[int(UNDERLINE_REL[0] * h):int(UNDERLINE_REL[1] * h)]
    white = (body_band.min(axis=2) > 200).mean(axis=0) > 0.8
    cols = np.where(yellow & white)[0]
    if len(cols) < 0.08 * w:
        return None
    return float(cols.min()) / w, float(cols.max()) / w
