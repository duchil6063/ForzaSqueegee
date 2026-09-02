"""세선화·경로 — 마스크 하나를 **획의 중심선**으로 바꾼다.

Zhang-Suen 세선화 + 계단 잉여 제거로 진짜 단위폭 뼈대를 얻고, 분기점 뭉치를
노드로 접어 가닥 사슬만 경로로 삼는다. 사람이 교차를 무시하고 한 획으로 긋는
문법은 `_join_paths`가 재현한다. 선 도안·영역 껍질·획 어휘 계측이 공유한다.

경로 평활(`smooth_path`)도 여기 있다 — 계단은 펴되 **의도된 각은 지킨다**.
"""

from __future__ import annotations

import os

import cv2
import numpy as np


# 8-이웃 배치(256가지) → (이웃의 8-연결 성분 수, 이웃 수). 성분 수가 곧
# "그 픽셀에서 뻗는 가닥 수"다 — 1이면 이웃끼리 이미 붙어 있어 그 픽셀은
# 없어도 선이 안 끊긴다(계단 잉여), 2면 통과점, 3 이상이 진짜 분기점.
_NB_OFF = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))


def _nb_lut() -> tuple[np.ndarray, np.ndarray]:
    adj = [[j for j in range(8)
            if j != i and abs(_NB_OFF[i][0] - _NB_OFF[j][0]) <= 1
            and abs(_NB_OFF[i][1] - _NB_OFF[j][1]) <= 1] for i in range(8)]
    ncomp = np.zeros(256, np.int8)
    nnb = np.zeros(256, np.int8)
    for code in range(256):
        on = {i for i in range(8) if code >> i & 1}
        nnb[code] = len(on)
        seen, c = set(), 0
        for s in on:
            if s in seen:
                continue
            c += 1
            stack = [s]
            seen.add(s)
            while stack:
                for v in adj[stack.pop()]:
                    if v in on and v not in seen:
                        seen.add(v)
                        stack.append(v)
        ncomp[code] = c
    return ncomp, nnb


_NCOMP, _NNB = _nb_lut()


def _nb_codes(img: np.ndarray) -> np.ndarray:
    c = np.zeros(img.shape, np.uint8)
    for i, (dy, dx) in enumerate(_NB_OFF):
        c |= (np.roll(np.roll(img, -dy, 0), -dx, 1) << i).astype(np.uint8)
    return c


def _unit_width(skel: np.ndarray) -> np.ndarray:
    """Zhang-Suen이 남긴 **계단 잉여 픽셀**을 걷어 진짜 단위폭으로 만든다.

    대각선 획에서 병렬 세선화는 "└" 모양 계단을 남긴다. 그 모서리 픽셀은
    8-이웃이 3개라 `_paths`가 분기점으로 오인하고, 획이 3~7px 조각으로
    부서져 파편 필터에 전멸한다 (실측: 백 px짜리 뼈대 하나가 경로 열몇 개로
    쪼개져 굵은 절반이 통째로 누락되고, 그 자리를 옅은 영역 타원이 대신 채워
    "바랜 선"이 됐다. 제거 후에는 경로 하나로 온전히 남는다).

    삭제 기준은 "이웃이 한 덩어리"뿐이라 위상이 안 변한다 — 이웃끼리 이미
    서로 붙어 있으니 그 픽셀을 빼도 이어진 것은 이어진 채다. 끝점(이웃 1개)은
    획이 짧아지므로 남긴다. 래스터 순서 순차 삭제 = 결정적.
    """
    img = np.pad(skel, 1).astype(np.uint8)
    while True:
        code = _nb_codes(img)
        cand = np.argwhere((img == 1) & (_NCOMP[code] == 1) & (_NNB[code] >= 2))
        removed = False
        for y, x in cand:
            if not img[y, x]:
                continue
            c = 0
            for i, (dy, dx) in enumerate(_NB_OFF):   # 삭제 반영 재확인
                c |= int(img[y + dy, x + dx]) << i
            if _NCOMP[c] == 1 and _NNB[c] >= 2:
                img[y, x] = 0
                removed = True
        if not removed:
            break
    return img[1:-1, 1:-1].astype(bool)


def _thin(mask: np.ndarray) -> np.ndarray:
    """Zhang-Suen 세선화 + 계단 잉여 제거 (bool, 소영역 ROI 전용)."""
    img = np.pad(mask, 1).astype(np.uint8)
    while True:
        changed = False
        for step in (0, 1):
            p = img
            p2 = np.roll(p, -1, 0); p3 = np.roll(np.roll(p, -1, 0), 1, 1)
            p4 = np.roll(p, 1, 1);  p5 = np.roll(np.roll(p, 1, 0), 1, 1)
            p6 = np.roll(p, 1, 0);  p7 = np.roll(np.roll(p, 1, 0), -1, 1)
            p8 = np.roll(p, -1, 1); p9 = np.roll(np.roll(p, -1, 0), -1, 1)
            nb = [p2, p3, p4, p5, p6, p7, p8, p9]
            B = sum(x.astype(np.int8) for x in nb)
            seq = nb + [p2]
            A = sum(((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.int8)
                    for i in range(8))
            if step == 0:
                c1 = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                c1 = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            rm = (p == 1) & (B >= 2) & (B <= 6) & (A == 1) & c1
            if rm.any():
                img[rm] = 0
                changed = True
        if not changed:
            break
    return _unit_width(img[1:-1, 1:-1].astype(bool))


def _paths(skel: np.ndarray) -> list[tuple[np.ndarray, int, int]]:
    """뼈대 → (경로, 머리 분기점 id, 꼬리 분기점 id) 목록. id −1 = 자유 끝.

    분기점(차수 ≥3)은 8-연결에서 2~3px **뭉치**로 나타난다 — 픽셀 단위로 걷던
    옛 구현은 뭉치 내부 간선이 길이 2 경로를 양산했다 (실측: 뼈대 160px에
    경로 118개). 뭉치를 통째로 노드로 접어 가닥 사슬만 경로로 삼는다.
    """
    ys, xs = np.nonzero(skel)
    pts = set(zip(ys.tolist(), xs.tolist()))
    if not pts:
        return []
    nbrs = {p: [q for q in ((p[0] + dy, p[1] + dx)
                            for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                            if dy or dx) if q in pts] for p in pts}
    deg = {p: len(n) for p, n in nbrs.items()}

    # 분기점 뭉치 → 노드 id (연결 성분)
    jmask = np.zeros_like(skel, np.uint8)
    for p, d in deg.items():
        if d >= 3:
            jmask[p] = 1
    ncc, cc = cv2.connectedComponents(jmask, connectivity=8)
    jid = {p: int(cc[p]) for p in pts if jmask[p]}          # 1..ncc-1
    jcent = {}
    for p, j in jid.items():
        jcent.setdefault(j, []).append(p)
    jcent = {j: np.mean(np.array(v, np.float64), axis=0) for j, v in jcent.items()}

    # 가닥 = 분기점 뭉치를 뺀 나머지 (차수 ≤2 사슬·고리)
    chain = pts - set(jid)
    cn = {p: [q for q in nbrs[p] if q in chain] for p in chain}
    out = []
    seen: set = set()

    def walk(start):
        path = [start]
        seen.add(start)
        cur, prev = start, None
        while True:
            nxt = [q for q in cn[cur] if q != prev and q not in seen]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            seen.add(cur)
            path.append(cur)
        return path

    def jends(path):
        """경로 양끝에 붙은 분기점 뭉치 — 있으면 중심점을 잇고 id를 단다."""
        hj = tj = -1
        head_j = [jid[q] for q in nbrs[path[0]] if q in jid]
        tail_j = [jid[q] for q in nbrs[path[-1]] if q in jid]
        arr = [np.array(p, np.float64) for p in path]
        if head_j:
            hj = head_j[0]
            arr.insert(0, jcent[hj])
        if tail_j:
            tj = tail_j[0]
            arr.append(jcent[tj])
        return np.array(arr), hj, tj

    for p in chain:                       # 사슬 끝(차수 ≤1)에서 출발
        if p in seen or len(cn[p]) >= 2:
            continue
        out.append(jends(walk(p)))
    for p in chain:                       # 남은 순수 고리
        if p not in seen:
            out.append(jends(walk(p)))
    return [(a, hj, tj) for a, hj, tj in out if len(a) >= 2]


def _prune_spurs(skel: np.ndarray, min_len: float) -> np.ndarray:
    """끝점에서 min_len 안에 분기점이 나오는 곁가지를 지운다 (2회 반복).

    선 폭 남짓의 스퍼가 분기점을 양산해 경로를 파편내는 것을 막는다
    (실측: 경로의 90%가 12px 미만 파편이었다).
    """
    sk = skel.copy()
    for _ in range(2):
        ys, xs = np.nonzero(sk)
        pts = set(zip(ys.tolist(), xs.tolist()))
        nbrs = {p: [q for q in ((p[0] + dy, p[1] + dx)
                                for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                                if dy or dx) if q in pts] for p in pts}
        deg = {p: len(n) for p, n in nbrs.items()}
        removed = False
        for p in list(pts):
            if deg.get(p, 0) != 1:
                continue
            walk, cur, prev = [p], p, None
            while len(walk) <= min_len:
                nxt = [q for q in nbrs[cur] if q != prev]
                if not nxt or deg.get(cur, 0) >= 3:
                    break
                prev, cur = cur, nxt[0]
                walk.append(cur)
            if deg.get(cur, 0) >= 3 and len(walk) - 1 <= min_len:
                for q in walk[:-1]:            # 분기점 자신은 남긴다
                    if q in pts:
                        pts.discard(q)
                        sk[q] = False
                        removed = True
        if not removed:
            break
    return sk


def _end_dir(path: np.ndarray, head: bool) -> np.ndarray:
    """경로 끝의 접선 (밖을 향한 단위 벡터, (y,x))."""
    k = min(5, len(path) - 1)
    v = (path[0] - path[k]) if head else (path[-1] - path[-1 - k])
    n = np.hypot(*v)
    return v / n if n > 1e-9 else np.array([0.0, 0.0])


def _join_paths(paths: list[tuple[np.ndarray, int, int]],
                max_angle_deg: float = 35.0, rec: list | None = None
                ) -> list[tuple[np.ndarray, int, int]]:
    """분기점 노드에서 접선이 이어지는 경로 쌍을 하나로 합친다 (수렴까지 반복).

    합치지 않으면 획이 분기점마다 끊긴다. 사람은 교차를 무시하고 한 획으로
    긋는다 — 그 문법의 재현이다. 반환은 (경로, 머리 접합점, 꼬리 접합점) —
    접합점 id ≥ 0이면 그 끝이 다른 획과 만나는 교차점이다 (다리 조각 판정용).

    `rec`(계측 전용, `celfit.census`): 접합 노드마다 후보 쌍의 코사인과 판정을
    append한다. **판정은 안 바꾼다** — 지금 코드가 이미 내린 결정에 이름만
    붙인다 (`angle` = 각 문턱 미달, `paired` = 이 회차에 이미 합쳐진 짝,
    `singleton` = 그 노드에 끝이 하나뿐, `ok` = 합쳐진 쌍).
    """
    cos_lim = float(np.cos(np.radians(max_angle_deg)))
    items = [(a, hj, tj) for a, hj, tj in paths if len(a) >= 2]
    changed = True
    npass = 0
    while changed:
        changed = False
        npass += 1
        ends: dict[int, list[tuple[int, bool]]] = {}
        for i, (p, hj, tj) in enumerate(items):
            if hj >= 0:
                ends.setdefault(hj, []).append((i, True))
            if tj >= 0:
                ends.setdefault(tj, []).append((i, False))
        merged: set[int] = set()
        new_items = []
        for key, lst in ends.items():
            best = None
            cand_rec: list = []
            for a in range(len(lst)):
                for b in range(a + 1, len(lst)):
                    (i, hi), (j, hjd) = lst[a], lst[b]
                    if i == j or i in merged or j in merged:
                        if rec is not None:
                            cand_rec.append((None, "paired"))
                        continue
                    c = -float(np.dot(_end_dir(items[i][0], hi),
                                      _end_dir(items[j][0], hjd)))
                    if rec is not None:
                        cand_rec.append((c, None))
                    if c >= cos_lim and (best is None or c > best[0]):
                        best = (c, i, hi, j, hjd)
            if rec is not None:
                rec.append({"pass": npass,
                            "ends": len(lst), "cos_lim": round(cos_lim, 6),
                            "cos": [None if c is None else round(c, 4)
                                    for c, _ in cand_rec],
                            "paired": sum(1 for _, w in cand_rec
                                          if w == "paired"),
                            "ok": best is not None,
                            "best": None if best is None else round(best[0], 4)})
            if best is None:
                continue
            _, i, hi, j, hjd = best
            (pi, ihj, itj), (pj, jhj, jtj) = items[i], items[j]
            if hi:                                   # i의 접점 끝이 꼬리가 되게
                pi, ihj, itj = pi[::-1], itj, ihj
            if not hjd:                              # j의 접점 끝이 머리가 되게
                pj, jhj, jtj = pj[::-1], jtj, jhj
            new_items.append((np.concatenate([pi, pj[1:]], axis=0), ihj, jtj))
            merged.update((i, j))
            changed = True
        if changed:
            items = new_items + [it for k, it in enumerate(items) if k not in merged]
    return items


# ── 각을 지키는 평활 ──────────────────────────────────────────────────
# 각 판정의 자 (도) — **기선 L만큼 떨어진 두 현이 이루는 각**이다. 점 세 개의
# 국소 각을 쓰면 안 된다: 선 지도의 계단(±1px 지그재그)이 국소로는 45°까지
# 나오는데 기선 3px에서는 상쇄돼 30° 아래로 내려간다. 그래서 아래 두 문턱은
# 계단(≤37°)과 사람이 의도한 꺾임(눈꼬리·턱선·옷 주름 ≥60°) 사이에 선다.
_CORNER_LO = float(os.environ.get("FS_CORNER_LO", 50.0))
_CORNER_HI = float(os.environ.get("FS_CORNER_HI", 85.0))


def corner_strength(path: np.ndarray, base: int = 3) -> np.ndarray:
    """경로 표본마다 **의도된 각일 확률** 0~1 (기선 `base`px의 현 각).

    극대만 남긴다 (기선 안 비최대 억제) — 안 그러면 반경이 작은 매끈한 굽음이
    통째로 각으로 잡혀 평활이 아예 안 걸린다. 각은 한 점이고 굽음은 구간이라
    극대 여부가 둘을 가른다.
    """
    n = len(path)
    if n < 2 * base + 1:
        return np.zeros(n, np.float64)
    a = path[base:-base] - path[:-2 * base]
    b = path[2 * base:] - path[base:-base]
    na = np.hypot(a[:, 0], a[:, 1])
    nb = np.hypot(b[:, 0], b[:, 1])
    cos = np.clip((a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1])
                  / np.maximum(na * nb, 1e-9), -1.0, 1.0)
    ang = np.degrees(np.arccos(cos))
    out = np.zeros(n, np.float64)
    out[base:-base] = ang
    # 비최대 억제 — 기선 창 안에서 가장 크게 꺾인 표본만 각이다
    keep = np.zeros(n, bool)
    for i in range(base, n - base):
        lo, hi = max(base, i - base), min(n - base, i + base + 1)
        if out[i] >= out[lo:hi].max() - 1e-9 and out[i] > 0:
            keep[i] = True
    c = np.where(keep, (out - _CORNER_LO) / max(_CORNER_HI - _CORNER_LO, 1e-6),
                 0.0)
    return np.clip(c, 0.0, 1.0)


def smooth_path(path: np.ndarray, ker: np.ndarray) -> np.ndarray:
    """경로 평활 — **완만한 굽음은 적극적으로, 의도된 각은 그대로**.

    선 지도의 계단(±1px 고주파)은 지도 쪽에서 못 없애므로(`lineart.hysteresis`
    문서) 여기서 편다. 그런데 창을 통째로 걸면 눈꼬리·턱선·옷 주름의 각까지
    같은 만큼 깎인다 — 5탭이면 각 하나가 5px에 걸쳐 둥글어지고, 그 자리가
    "자동 벡터화 티"의 한 축이다.

    그래서 평활을 **끄는 게 아니라 각에서만 물린다**: 각 세기
    (`corner_strength`)만큼 원래 표본으로 되돌린다. 세기는 창 반폭에 걸쳐
    삼각으로 퍼뜨려 각의 양옆 다리도 각을 향해 곧게 남는다 — apex만 남기면
    다리가 안쪽으로 빨려 들어가 각이 되레 뭉툭해진다.

    끝은 **복제해서 채운다**. 창 몫만큼 표본을 잃으면(`valid` + 양끝 원본
    한 점) 획 끝 두 표본이 통째로 없어지고, 그 자리가 곧 획이 기계적으로
    잘려 보이는 끝이다.
    """
    n, k = len(path), len(ker)
    if n < k + 2:
        return path
    half = (k - 1) // 2
    pad = np.concatenate([np.repeat(path[:1], half, axis=0), path,
                          np.repeat(path[-1:], half, axis=0)], axis=0)
    sm = np.stack([np.convolve(pad[:, 0], ker, "valid"),
                   np.convolve(pad[:, 1], ker, "valid")], axis=1)[:n]
    c = corner_strength(path, base=max(3, half))
    if not c.any():
        return sm
    # 삼각 퍼짐 — 각에서 1, 창 반폭 밖에서 0
    tri = 1.0 - np.abs(np.arange(-half, half + 1)) / (half + 1.0)
    w = np.clip(np.convolve(c, tri, "same"), 0.0, 1.0)[:, None]
    return sm * (1.0 - w) + path * w


def _cross2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """2차원 외적 (스칼라) — `np.cross`를 **쓰면 안 된다**.

    NumPy 2.0에서 2차원 입력이 폐기되고 2.x에서 **제거**됐다(ValueError). 옛
    numpy가 깔린 개발 환경에서는 안 보이다가, 새로 설치한 사용자에게서 도안
    생성이 통째로 죽는다 (실측: 깨끗한 venv에 numpy 2.5.2가 깔려 `_try_curve`
    에서 예외). 정의가 한 줄이라 의존할 이유도 없다.
    """
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _rdp(path: np.ndarray, eps: float = 0.8) -> np.ndarray:
    """더글러스-포이커 단순화."""
    if len(path) < 3:
        return path
    a, b = path[0], path[-1]
    ab = b - a
    L = np.hypot(*ab)
    if L < 1e-9:
        d = np.hypot(*(path - a).T)
    else:
        d = np.abs(_cross2(ab, path - a)) / L
    i = int(np.argmax(d))
    if d[i] <= eps:
        return np.array([a, b])
    left = _rdp(path[:i + 1], eps)
    return np.concatenate([left[:-1], _rdp(path[i:], eps)])


def _rdp_idx(path: np.ndarray, eps: float = 0.8) -> list[int]:
    """더글러스-포이커 단순화의 **마디 인덱스** (`_rdp`와 같은 자·같은 마디).

    점만 받으면 마디 사이의 **원래 호**를 되찾을 수 없다 — 그 호가 굽었는지가
    곧 "여기 막대를 써도 되는가"라서 (`stroke._fit_segments`) 인덱스가 필요하다.
    """
    if len(path) < 3:
        return list(range(len(path)))
    keep = [0, len(path) - 1]
    stack = [(0, len(path) - 1)]
    while stack:
        s, e = stack.pop()
        if e - s < 2:
            continue
        a, ab = path[s], path[e] - path[s]
        L = np.hypot(*ab)
        seg = path[s:e + 1]
        d = (np.hypot(*(seg - a).T) if L < 1e-9
             else np.abs(_cross2(ab, seg - a)) / L)
        i = int(np.argmax(d))
        if d[i] <= eps or i == 0 or i == e - s:
            continue
        keep.append(s + i)
        stack += [(s, s + i), (s + i, e)]
    return sorted(keep)


def _dt_along(dt: np.ndarray, path: np.ndarray) -> np.ndarray:
    p = path.astype(int)
    return dt[p[:, 0].clip(0, dt.shape[0] - 1), p[:, 1].clip(0, dt.shape[1] - 1)]


def _resample(pts: np.ndarray, n: int) -> np.ndarray:
    """폴리라인을 호길이 등간격 n점으로 다시 뽑는다 — 아핀 맞춤의 대응점."""
    d = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])
    if d[-1] < 1e-9:
        return np.repeat(pts[:1].astype(np.float64), n, axis=0)
    t = np.linspace(0.0, float(d[-1]), n)
    return np.stack([np.interp(t, d, pts[:, 0]),
                     np.interp(t, d, pts[:, 1])], axis=1)
