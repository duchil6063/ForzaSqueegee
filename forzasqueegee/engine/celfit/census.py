"""단위 인구조사 — bar/mop **단위가 왜 이렇게 많이 생기나** (디버그 전용).

`FS_UNIT_CENSUS=1`일 때만 산다. 끄면 이 모듈의 어느 손도 안 불리고 산출물은
바이트 동일이다 (호출부가 전부 `if census.ON:` 뒤에 있고, 계보를 받는 인자는
`rec=None` 기본값이다).

**무엇을 재나.** 최종 레이어 라벨만 세면 "막대 1,394 · 마무리 1,027"에서
거꾸로 추측하는 수밖에 없다. 여기서는 그 앞의 계보를 그대로 남긴다:

    영역 → 잔여 마스크 → 뼈대/성분 → **단위** → 피터 → 레이어

단위와 레이어를 섞지 않는 것이 요점이다. 막대는 경로 하나(단위)가 도형
여럿(레이어)을 부를 수 있고, 마무리는 성분 하나가 언제나 한 장이다 — 그래서
"장수가 많다"의 원인이 개수인지 장당 비용인지가 이 자에서만 갈린다.

새 문턱을 세우지 않는다. 여기 적히는 사유 코드는 전부 **지금 코드가 이미
내린 판정**을 이름 붙여 내보내는 것이다.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

ON = os.environ.get("FS_UNIT_CENSUS", "").strip() not in ("", "0", "false")

_PLATE: dict | None = None
_REG: dict | None = None


# ── 판·영역 ──────────────────────────────────────────────────────────
def begin_plate(name: str, size) -> None:
    global _PLATE, _REG
    _PLATE = {"plate": name, "size": [int(size[0]), int(size[1])],
              "regions": []}
    _REG = None


def plate() -> dict | None:
    return _PLATE


def begin_region(**kv) -> dict:
    """영역 하나의 기록을 연다 — 반환값이 그 영역의 dict다."""
    global _REG
    _REG = dict(kv)
    _REG.setdefault("bar", None)
    _REG.setdefault("mop", None)
    _REG["res"] = {}                       # 단계별 남은 잔여 px (§14)
    if _PLATE is not None:
        _PLATE["regions"].append(_REG)
    return _REG


def region() -> dict | None:
    return _REG


def end_region(**kv) -> None:
    global _REG
    if _REG is not None:
        _REG.update(kv)
    _REG = None


def res_mark(tag: str, n: int) -> None:
    """단계 하나가 끝난 자리의 잔여 px — 막대↔마무리 인수인계 원장(§14)."""
    if _REG is not None:
        _REG["res"][tag] = int(n)


# ── 기하 자 (계측 전용 — 판정에 안 쓴다) ─────────────────────────────
def shape_stats(m: np.ndarray) -> dict:
    """성분 하나의 모양 자 — 면적·둘레·조밀도·연장·bbox."""
    u = m.astype(np.uint8)
    area = int(u.sum())
    if not area:
        return {"area": 0, "peri": 0, "compact": 0.0, "elong": 0.0,
                "bbox": [0, 0, 0, 0]}
    peri = int((u & ~cv2.erode(u, np.ones((3, 3), np.uint8))).sum())
    ys, xs = np.nonzero(u)
    dt = cv2.distanceTransform(np.pad(u, 1), cv2.DIST_L2, 3)[1:-1, 1:-1]
    wmed = 2.0 * float(np.median(dt[m]))
    return {"area": area, "peri": peri,
            # 등주 조밀도 — 원판이면 1, 너덜하면 그 아래
            "compact": round(4.0 * np.pi * area / max(peri * peri, 1.0), 4),
            # 가로세로비 = 면적/폭² (`fill.region_shape`와 같은 자)
            "elong": round(area / max(wmed * wmed, 1.0), 3),
            "wmed": round(wmed, 3),
            "bbox": [int(xs.min()), int(ys.min()),
                     int(xs.max()) + 1, int(ys.max()) + 1]}


def where(sel: np.ndarray, dedge: np.ndarray | None,
          ink: np.ndarray | None) -> dict:
    """이 단위가 **어디서** 생겼나 (§22) — 기존 거리 지도만 쓴다.

    `dedge`는 영역 마스크의 거리변환(경계까지 px), `ink`는 놓인 획의 커버
    지도다. 새 의미 검출기를 만들지 않는다.
    """
    out: dict = {}
    if dedge is not None and sel.any():
        d = dedge[sel]
        out["dedge_med"] = round(float(np.median(d)), 2)
        # "경계 인접" = 경계에서 2px 안 — 격자 양자화가 남기는 껍질의 두께다
        out["edge_frac"] = round(float((d <= 2.0).mean()), 4)
    if ink is not None and sel.any():
        out["ink_frac"] = round(float(ink[sel].mean()), 4)
    return out


def _maps() -> dict:
    """영역이 들고 있는 거리 지도들 — 단위마다 "어디서 생겼나"를 붙일 재료."""
    if _REG is None:
        return {}
    return {k: _REG[k] for k in ("_dedge", "_ink") if _REG.get(k) is not None}


# ── 막대 (§7·§8·§9·§12) ──────────────────────────────────────────────
def bar_open(**kv) -> dict:
    """`_fit_bars` 진입 — 깔때기 칸을 연다."""
    rec = {"joins": [], "units": [], "drop": {}, **kv, **_maps()}
    if _REG is not None:
        _REG["bar"] = rec
    return rec


def bar_unit(bar: dict, **kv) -> dict:
    """단위 하나 — 여기 실린 카운터를 `stroke._fit_path`가 채운다."""
    u = {"curve": 0, "recurse": 0, "seg_nodes": 0, "seg_bars": 0,
         "seg_curves": 0, "eps_grow": 0, "equal_split": 0, "layers": 0, **kv}
    bar["units"].append(u)
    return u


def join_rec(bar: dict):
    """`skeleton._join_paths`에 넘길 접합 기록기 (없으면 None)."""
    return bar["joins"] if bar is not None else None


# ── 마무리 (§15·§16·§17) ─────────────────────────────────────────────
def mop_open(**kv) -> dict:
    rec = {"comps": [], **kv, **_maps()}
    if _REG is not None:
        _REG["mop"] = rec
    return rec


def mop_comp(mop: dict, **kv) -> dict:
    c = dict(kv)
    mop["comps"].append(c)
    return c
