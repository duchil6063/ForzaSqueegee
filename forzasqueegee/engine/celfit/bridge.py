"""끊긴 획 잇기 — 신경망 선화의 점선을 사람처럼 한 획으로 잇는다.

배치 **전에** 이어야 뼈대 경로가 틈을 관통하는 한 경로가 되고, 곡선 한 장
맞춤·파편 필터가 긴 경로 기준으로 돈다. 판정과 상수 근거는 `bridge_line_gaps`.
"""

from __future__ import annotations

import cv2
import numpy as np

from .skeleton import (_dt_along, _end_dir, _join_paths, _paths, _prune_spurs,
                       _thin)


# 끊긴 획 잇기 상수 (line 노선 전용) — 판정은 `bridge_line_gaps` 문서.
# 틈은 **뼈대 끝점 사이 거리**로 재는데 끝점은 마스크 끝보다 획 반폭만큼
# 안쪽이다 (합성 실측: 굵기 2px 막대의 눈 8px 틈이 끝점으로는 11px). 그래서
# 상한은 "가시 틈 상한 + 두 끝의 반폭"이다 — 반폭 보정이 없으면 굵기 배수를
# 아무리 올려도 그만큼 안쪽에서 깎인다.
# 굵기 10배·천장 20px (사용자 확정 2026-08-25): 선화 밝기 증거가 없는 틈도
# 기하가 맞으면 잇는다 — "오버레이를 벗어나더라도 선끼리 자연스럽게 잇는"
# 레퍼런스의 문법이 우선이다 (옅은 헤일로의 대시 틈이 10~20px). 대신 긴
# 다리(12px 초과)는 마주봄 각을 45°→30°로 조인다 — 멀수록 외삽이라 확신을
# 더 요구한다. 나란한 가닥 끝은 접선이 평행이라 어느 각에서도 안 붙는다.
_GAP_ANG = 45.0        # 마주봄 판정 각 (도) — 끝 접선이 상대를 향하는 허용 폭
_GAP_ANG_FAR = 30.0    # 긴 다리(12px 초과)의 마주봄 각
_GAP_NEAR = 12.0       # 이 거리까지는 기본 각 (px)
_GAP_MUL = 10.0        # 가시 틈 상한 = 획 굵기 × 이 배수
_GAP_MIN, _GAP_MAX = 6.0, 20.0   # 가시 틈 상한의 바닥·천장 (px, 작업 해상도 1200)


def bridge_line_gaps(lm: np.ndarray, sel: np.ndarray, log=print
                     ) -> tuple[np.ndarray, np.ndarray, int]:
    """선 지도의 **끊긴 획을 잇는다** (line 노선 전용). 반환 (이은 지도, 다리 마스크, 쌍 수).

    신경망 선화는 옅은 구간(히스테리시스 문턱 언저리)에서 획이 점선으로
    끊긴다 — 실측(01): 마주보는 자유 끝이 ≤6px에 57쌍, ≤10px에 78쌍. 사람은
    오버레이의 선을 픽셀대로 따르지 않고 그 자리를 **한 획으로 이어** 긋는다
    (레퍼런스 사진의 선따기 문법). cel 노선은 끊긴 자리가 밑의 면 색에 가려져
    문제가 안 되지만, 바탕이 비는 line 노선에서는 점선이 그대로 보인다.

    잇기 전에 이어야 배치도 그 덕을 본다: 다리가 성분을 합치면 뼈대 경로가
    틈을 관통하는 **한 경로**가 되고, 곡선 한 장 맞춤·파편 필터가 긴 경로
    기준으로 돌아간다 (다리를 나중에 채우면 짧은 조각들이 그대로 파편이다).

    "끊긴 획" 판정 — 자유 끝 둘이
    ① 서로를 마주보고 (끝의 바깥 접선이 상대 끝을 겨눈다 — ±45°, 12px를
       넘는 긴 다리는 ±30°: 나란한 머리칼 가닥의 이웃한 끝은 접선이
       평행이라 안 붙는다),
    ② 접선끼리도 반대 방향이며 (한 획의 연장선상),
    ③ 가시 틈이 획 굵기의 10배 안이고 (바닥 6px·천장 20px + 뼈대 끝점의
       반폭 보정 — 상수 근거는 `_GAP_MUL` 문서),
    ④ 다리 전체가 실루엣(±2px) 안에 놓일 때만 잇는다.
    가까운 쌍부터 탐욕 1:1 — 결정적이다. 다리 굵기는 두 끝 중 가는 쪽이다.
    """
    h, w = lm.shape
    ncc, cc, cstats, _ = cv2.connectedComponentsWithStats(
        lm.astype(np.uint8), connectivity=8)
    ends = []                              # (y, x, dy, dx, pid, wmed)
    pid = 0
    for ci in range(1, ncc):
        # 3px부터 — 배치는 6px 미만 성분을 안 그리지만, 잇기는 그 작은
        # 조각이 관건이다: 옅은 선(헤일로)의 대시 조각이 바로 이 크기라,
        # 빼면 다리가 못 서고 점선이 그대로 남는다. 접선이 서야 하므로
        # 경로 3표본 미만인 조각은 아래에서 걸린다
        if cstats[ci, cv2.CC_STAT_AREA] < 3:
            continue
        x0 = max(0, int(cstats[ci, cv2.CC_STAT_LEFT]) - 4)
        y0 = max(0, int(cstats[ci, cv2.CC_STAT_TOP]) - 4)
        x1 = min(w, x0 + int(cstats[ci, cv2.CC_STAT_WIDTH]) + 8)
        y1 = min(h, y0 + int(cstats[ci, cv2.CC_STAT_HEIGHT]) + 8)
        m = cc[y0:y1, x0:x1] == ci
        dt = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 3)
        on = m & (dt > 0)
        res_w = 2.0 * float(np.median(dt[on])) if on.any() else 2.0
        skel = _prune_spurs(_thin(m), max(3.0, 1.2 * res_w))
        for path, hj, tj in _join_paths(_paths(skel)):
            if len(path) < 3:              # 접선을 잴 표본이 안 된다
                continue
            wmed = 2.0 * float(np.median(_dt_along(dt, path)))
            pid += 1
            for j, head in ((hj, True), (tj, False)):
                if j >= 0:
                    continue               # 접합점 끝은 끊긴 게 아니라 교차다
                d = _end_dir(path, head)
                p = path[0] if head else path[-1]
                ends.append((float(p[0] + y0), float(p[1] + x0),
                             float(d[0]), float(d[1]), pid, wmed))
    if not ends:
        return lm, np.zeros_like(lm), 0
    arr = np.array([[e[0], e[1]] for e in ends])
    dirs = np.array([[e[2], e[3]] for e in ends])
    pids = np.array([e[4] for e in ends])
    wms = np.array([e[5] for e in ends])
    cos_near = float(np.cos(np.radians(_GAP_ANG)))
    cos_far = float(np.cos(np.radians(_GAP_ANG_FAR)))
    ok_zone = cv2.dilate(sel.astype(np.uint8),
                         np.ones((5, 5), np.uint8)).astype(bool)
    cell = _GAP_MAX
    grid: dict = {}
    for i, (y, x) in enumerate(arr):
        grid.setdefault((int(y // cell), int(x // cell)), []).append(i)
    cand = []
    for i, (y, x) in enumerate(arr):
        gy, gx = int(y // cell), int(x // cell)
        for ay in (gy - 1, gy, gy + 1):
            for ax in (gx - 1, gx, gx + 1):
                for j in grid.get((ay, ax), ()):
                    if j <= i or pids[j] == pids[i]:
                        continue
                    v = arr[j] - arr[i]
                    d = float(np.hypot(*v))
                    lim = (min(_GAP_MAX, max(_GAP_MIN,
                                             _GAP_MUL * max(wms[i], wms[j])))
                           + 0.5 * (wms[i] + wms[j]))   # 뼈대 끝점 반폭 보정
                    if d > lim or d < 1e-9:
                        continue
                    u = v / d
                    cl = cos_near if d <= _GAP_NEAR else cos_far
                    if (float(np.dot(dirs[i], u)) >= cl
                            and float(np.dot(dirs[j], -u)) >= cl
                            and float(np.dot(dirs[i], dirs[j])) <= -cl):
                        cand.append((d, i, j))
    cand.sort()
    used: set = set()
    bridge = np.zeros((h, w), np.uint8)
    n = 0
    for d, i, j in cand:
        if i in used or j in used:
            continue
        # ④ 다리 전체가 실루엣(±2px) 안 — 밖으로 나가는 다리는 남남이다
        ts = np.linspace(0.0, 1.0, max(4, int(d) + 2))[:, None]
        if d > 6.0:
            # 곡선 다리 (에르미트) — 끝 접선을 따라 잇는다. 직선 다리는 굽은
            # 획(헤일로 고리·곡선 가닥)에 꺾임 마디를 남긴다 — 사람은 획의
            # 흐름 그대로 곡선으로 잇는다. 짧은 틈은 직선과 차이가 없다
            h00 = 2 * ts ** 3 - 3 * ts ** 2 + 1
            h10 = ts ** 3 - 2 * ts ** 2 + ts
            h01 = -2 * ts ** 3 + 3 * ts ** 2
            h11 = ts ** 3 - ts ** 2
            pts = (h00 * arr[i][None] + h10 * (dirs[i] * d)[None]
                   + h01 * arr[j][None] + h11 * (-dirs[j] * d)[None])
        else:
            pts = arr[i][None] * (1 - ts) + arr[j][None] * ts
        yy = np.clip(np.round(pts[:, 0]), 0, h - 1).astype(int)
        xx = np.clip(np.round(pts[:, 1]), 0, w - 1).astype(int)
        if not bool(ok_zone[yy, xx].all()):
            continue
        used.update((i, j))
        cv2.polylines(bridge, [np.stack([xx, yy], axis=1).astype(np.int32)],
                      False, 1, max(1, int(round(min(wms[i], wms[j])))))
        n += 1
    if not n:
        return lm, np.zeros_like(lm), 0
    bridge = bridge.astype(bool) & ok_zone
    return lm | bridge, bridge & ~lm, n
