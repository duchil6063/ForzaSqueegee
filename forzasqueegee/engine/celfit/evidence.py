"""선 증거 — 신경망 선화를 **이진 마스크가 아니라 증거원**으로 다룬다.

AniLines의 출력을 그대로 "그릴 선"으로 삼으면 그 모델의 판단이 곧 획의 중요도가
된다. 여기서는 선화를 여러 증거 중 **하나**로 내리고, 획마다 원화·기하·위상에서
읽은 값을 함께 실어 뒤 단계(역할 분류·후보 경쟁·정책 선택)가 쓰게 한다.

두 층이다:

- `EvidenceMaps` — 그림 한 장의 증거 지도 묶음 (신뢰도·어두움·색 경계·실루엣·값).
  이진화 **전** 신뢰도를 그대로 들고 있는 것이 요점이다.
- `StrokeEvidence` — 경로 하나가 그 지도들에서 읽은 값 (`sample`).

Detail 모델은 있으면 쓰고 없으면 Basic만으로 돈다 (`lineart.extract` 문서) —
`detail_only`(Detail에만 있는 선)는 우선순위가 낮은 후보로 따로 센다.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


# 양옆 표본의 법선 여유 (px) — `select._side_pts`와 같은 자.
_SIDE_PAD = 2.5
# 나란한 이웃을 세는 반경 = 폭 × 이 배수 (하한 3px). `select._THIN_R`와 같은 자다.
_PAR_R = 5.0
# 반복성 판정의 접선 각차 상한 (배각 내적) — `select._THIN_COS`와 같은 자.
_PAR_COS = 0.6428


@dataclass(frozen=True)
class EvidenceMaps:
    """그림 한 장의 선 증거 지도 묶음 (전부 작업 해상도, 전장 좌표)."""

    basic: np.ndarray            # float32 [0,1] — AniLines basic 신뢰도
    detail: np.ndarray | None    # float32 [0,1] — detail 모델 (없으면 None)
    mask: np.ndarray             # bool — 뼈대를 뽑는 선 지도 (다리 포함)
    basic_mask: np.ndarray       # bool — 이진화한 basic
    detail_only: np.ndarray | None   # bool — detail에만 있는 선
    dark: np.ndarray             # float32 [0,1] — 원화 어두움
    edge: np.ndarray             # float32 [0,1] — 원화 색 경계 강도
    sil: np.ndarray              # bool — 실루엣 테 근방
    value: np.ndarray            # float32 — 지각 값 (importance.place_weight)
    bridge: np.ndarray | None    # bool — 끊긴 획을 이어 붙인 다리 px

    @property
    def has_detail(self) -> bool:
        return self.detail is not None


@dataclass
class StrokeEvidence:
    """경로 하나가 증거 지도에서 읽은 값 — 역할 분류·정책이 이것만 본다."""

    basic: float = 0.0          # 선화 basic 신뢰도 (경로 위 중앙값)
    detail: float = 0.0         # detail 신뢰도 (없으면 basic과 같다)
    detail_only: float = 0.0    # detail에만 있는 px 비율
    dark: float = 0.0           # 원화 어두움
    side_de: float = 0.0        # 양옆 색차 (Lab 노름)
    sil: float = 0.0            # 실루엣 인접 표본 비율
    edge_agree: float = 0.0     # 주변 색 경계와의 일치도
    bnd: float = 0.0            # 양옆 셀 영역 라벨이 다른 비율
    length: float = 0.0         # 경로 길이 px
    continuity: float = 0.0     # 선 지도 위에 실제로 얹힌 표본 비율
    bridged: float = 0.0        # 다리(이은 구간)로 채운 표본 비율
    curvature: float = 0.0      # 평균 |곡률| (1/px)
    width: float = 0.0          # 폭 중앙값 px
    width_cv: float = 0.0       # 폭 변동계수
    # ── 접합점 구조 ──────────────────────────────────────────────
    # 개수만으로는 "교차에 걸려 있다"까지밖에 못 말한다. **차수**(그 접합점에서
    # 몇 갈래가 뻗는가)가 있어야 3갈래 교차와 선망 뭉치가 갈리고, 자유 끝 수가
    # 있어야 "완전히 고립된 획"이 기하가 아니라 **위상**으로 선다.
    junctions: int = 0          # 양끝 접합점 수 (0~2)
    free_ends: int = 2          # 자유 끝 수 (2 = 양끝이 다 열려 있다)
    j_deg_max: int = 0          # 양끝 접합점 차수의 큰 쪽 (0 = 접합점 없음)
    j_deg_sum: int = 0          # 두 끝 차수의 합 — 선망 안쪽일수록 크다
    repeat: float = 0.0         # 반복성 — 나란한 이웃에 덮인 표본 비율
    parallel: float = 0.0       # 주변 평행 경로 밀도 (표본당 이웃 수)
    enclosure: float = 0.0      # 폐쇄/갇힘 — 경로가 잉크에 둘러싸인 정도
    importance: float = 0.0     # 지각 값 (경로 위 평균)
    # **주변 획 대비** 지각 값. 절대값은 획 판정에 못 쓴다 — 값 지도는 그림
    # 전체의 중앙을 1로 잡는데 획은 정의상 값이 가장 높은 자리라, 획만 모아
    # 보면 전부 상한에 붙는다 (실측 10장 중앙 9.9·상한 16). "주변보다 눈에
    # 띄는가"는 이웃 획들과 견주어야 답이 나온다
    imp_rel: float = 1.0

    def as_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in self.__dict__.items()}


def build_maps(basic_gray: np.ndarray | None, detail_gray: np.ndarray | None,
               mask: np.ndarray, basic_mask: np.ndarray,
               src_rgb: np.ndarray, sel: np.ndarray, value: np.ndarray,
               bridge: np.ndarray | None = None) -> EvidenceMaps:
    """증거 지도 묶음을 짓는다 (전부 작업 해상도).

    `basic_gray`·`detail_gray`는 **이진화 전** 밝기 지도(255=배경·0=선)다 —
    255를 빼 [0,1] 신뢰도로 둔다. 없으면(고전 폴백) 마스크 자체가 신뢰도다.
    """
    h, w = mask.shape
    if basic_gray is not None:
        basic = np.clip((255.0 - basic_gray.astype(np.float32)) / 255.0, 0.0, 1.0)
    else:
        basic = basic_mask.astype(np.float32)
    detail = det_only = None
    if detail_gray is not None:
        detail = np.clip((255.0 - detail_gray.astype(np.float32)) / 255.0, 0.0, 1.0)
        det_only = (detail > 0.35) & ~basic_mask & sel
    lab = cv2.cvtColor(src_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    # 어두움 — L*을 뒤집는다. 선은 어두운 쪽이라는 원화 쪽 증거다
    dark = np.clip(1.0 - lab[..., 0] / 255.0, 0.0, 1.0).astype(np.float32)
    # 색 경계 — Lab 세 채널의 기울기 크기. 선화가 없는 경계(색만 갈리는 자리)와
    # 선화만 있는 자리(그림자 해칭)를 가르는 축이다
    gx = cv2.Sobel(lab, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(lab, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt((gx * gx + gy * gy).sum(axis=2))
    hi = float(np.percentile(mag[sel], 97)) if sel.any() else 1.0
    edge = np.clip(mag / max(hi, 1e-6), 0.0, 1.0).astype(np.float32)
    # 실루엣 테 근방 — 알파 경계에서 몇 px 안. 배경이 없는 입력은 전부 False
    if bool((~sel).any()):
        rim = sel & ~cv2.erode(sel.astype(np.uint8),
                               np.ones((3, 3), np.uint8)).astype(bool)
        r = max(2, int(round(0.004 * min(w, h))))
        sil = cv2.dilate(rim.astype(np.uint8), cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))).astype(bool)
    else:
        sil = np.zeros((h, w), bool)
    return EvidenceMaps(basic=basic, detail=detail, mask=mask,
                        basic_mask=basic_mask, detail_only=det_only,
                        dark=dark, edge=edge, sil=sil,
                        value=value.astype(np.float32), bridge=bridge)


def _side_pts(path: np.ndarray, wmed: float, shape: tuple[int, int],
              rx0: int, ry0: int):
    """경로 표본마다 양옆(폭/2+여유 법선) 전장 좌표 — `select._side_pts`와 같은 식."""
    h, w = shape
    idx = np.arange(0, len(path), 3)
    if not len(idx):
        return idx, idx, idx, idx, idx
    j0 = np.maximum(idx - 2, 0)
    j1 = np.minimum(idx + 2, len(path) - 1)
    tan = path[j1] - path[j0]
    norm = np.hypot(tan[:, 0], tan[:, 1])
    norm[norm < 1e-9] = 1.0
    off = wmed / 2.0 + _SIDE_PAD
    oy = -tan[:, 1] / norm * off
    ox = tan[:, 0] / norm * off
    ys = path[idx, 0] + ry0
    xs = path[idx, 1] + rx0
    return (idx,
            np.clip(np.round(ys + oy), 0, h - 1).astype(np.int64),
            np.clip(np.round(xs + ox), 0, w - 1).astype(np.int64),
            np.clip(np.round(ys - oy), 0, h - 1).astype(np.int64),
            np.clip(np.round(xs - ox), 0, w - 1).astype(np.int64))


def _curvature(path: np.ndarray) -> float:
    """평균 |곡률| (1/px) — 세 점 원 근사의 중앙값."""
    if len(path) < 5:
        return 0.0
    step = max(2, len(path) // 24)
    a, b, c = path[:-2 * step:step], path[step:-step:step], path[2 * step::step]
    if not len(a):
        return 0.0
    v1, v2 = b - a, c - b
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    l1 = np.hypot(v1[:, 0], v1[:, 1])
    l2 = np.hypot(v2[:, 0], v2[:, 1])
    l3 = np.hypot(c[:, 0] - a[:, 0], c[:, 1] - a[:, 1])
    den = l1 * l2 * l3
    k = np.where(den > 1e-9, 2.0 * np.abs(cross) / np.maximum(den, 1e-9), 0.0)
    return float(np.median(k))


def junction_degrees(paths) -> dict[int, int]:
    """접합점 id → **차수** (그 뭉치에 붙은 가닥 끝의 수).

    `skeleton._paths`가 준 (경로, 머리 접합점, 꼬리 접합점) 목록만으로 나온다 —
    같은 id를 끝으로 삼는 가닥을 세면 그것이 그 교차점의 갈래 수다. 3은 T자
    교차, 4 이상은 선망 뭉치다.
    """
    deg: dict[int, int] = {}
    for _, hj, tj in paths:
        for j in (hj, tj):
            if j >= 0:
                deg[j] = deg.get(j, 0) + 1
    return deg


def sample(path: np.ndarray, wmed: float, widths: np.ndarray,
           maps: EvidenceMaps, labels: np.ndarray, rx0: int, ry0: int,
           j_head: int, j_tail: int, lab_img: np.ndarray,
           j_deg: dict | None = None) -> StrokeEvidence:
    """경로 하나의 증거 벡터 — 지도들에서 읽는다 (반복성·평행 밀도는 뒤에서 채운다).

    `widths`는 경로 표본마다의 폭(px)이고 `lab_img`는 원화 Lab (전장)이다.
    `j_deg`는 성분의 접합점 차수 표다 (`junction_degrees`).
    """
    h, w = maps.mask.shape
    p = path.round().astype(int)
    gy = np.clip(p[:, 0] + ry0, 0, h - 1)
    gx = np.clip(p[:, 1] + rx0, 0, w - 1)
    ev = StrokeEvidence()
    ev.length = float(np.hypot(*np.diff(path, axis=0).T).sum()) if len(path) > 1 else 0.0
    ev.basic = float(np.median(maps.basic[gy, gx]))
    ev.detail = (float(np.median(maps.detail[gy, gx])) if maps.detail is not None
                 else ev.basic)
    ev.detail_only = (float(np.mean(maps.detail_only[gy, gx]))
                      if maps.detail_only is not None else 0.0)
    ev.dark = float(np.median(maps.dark[gy, gx]))
    ev.edge_agree = float(np.mean(maps.edge[gy, gx]))
    ev.importance = float(np.mean(maps.value[gy, gx]))
    ev.continuity = float(np.mean(maps.mask[gy, gx]))
    ev.bridged = (float(np.mean(maps.bridge[gy, gx]))
                  if maps.bridge is not None else 0.0)
    ev.curvature = _curvature(path)
    if len(widths):
        med = float(np.median(widths))
        ev.width = med
        ev.width_cv = float(np.std(widths) / med) if med > 1e-6 else 0.0
    else:
        ev.width = wmed
    ev.junctions = int(j_head >= 0) + int(j_tail >= 0)
    ev.free_ends = 2 - ev.junctions
    if j_deg:
        dh, dt = j_deg.get(j_head, 0), j_deg.get(j_tail, 0)
        ev.j_deg_max = int(max(dh, dt))
        ev.j_deg_sum = int(dh + dt)
    idx, ay, ax, by, bx = _side_pts(path, wmed, (h, w), rx0, ry0)
    if len(idx):
        la, lb = labels[ay, ax], labels[by, bx]
        ev.bnd = float(np.mean(la != lb))
        ev.sil = float(np.mean((la < 0) | (lb < 0) | maps.sil[gy[idx], gx[idx]]))
        d = lab_img[ay, ax] - lab_img[by, bx]
        ev.side_de = float(np.median(np.sqrt((d * d).sum(axis=1))))
        # 폐쇄/갇힘 — 양옆이 둘 다 잉크면 이 획은 선망 **안**에 갇혀 있다.
        # 무늬(그물·레이스)의 안쪽 가닥이 여기서 높게 나오고, 실루엣·고립
        # 특징은 한쪽이 반드시 빈 자리라 낮다
        ink = maps.mask
        ev.enclosure = float(np.mean(ink[ay, ax] & ink[by, bx]))
    return ev


def fill_neighborhood(evs: list[StrokeEvidence], paths: list[np.ndarray],
                      widths: list[float], offs: list[tuple[int, int]]) -> None:
    """반복성·평행 밀도·**주변 대비 중요도**를 채운다 — 전체를 모은 뒤라야 잰다.

    표본마다 반경(폭 비례) 안에서 **접선이 나란한 다른 경로의 표본 수**를 센다.
    배각(2θ) 비교라 진행 방향 부호를 무시한다. 같은 반경 안 이웃 획들의 지각
    값 중앙과 견줘 `imp_rel`도 낸다 — "주변보다 눈에 띄는가"의 자다. 결정적
    이다: 격자 순회 순서가 입력 순서에만 의존한다.
    """
    cell = 8
    grid: dict[tuple[int, int], list] = {}
    samples = []
    for k, (path, off) in enumerate(zip(paths, offs)):
        idx = np.arange(0, len(path), 3)
        if not len(idx):
            samples.append((idx, idx, idx, idx))
            continue
        j0 = np.maximum(idx - 2, 0)
        j1 = np.minimum(idx + 2, len(path) - 1)
        tan = path[j1] - path[j0]
        ang2 = 2.0 * np.arctan2(tan[:, 0], tan[:, 1])
        ys = path[idx, 0] + off[1]
        xs = path[idx, 1] + off[0]
        samples.append((ys, xs, np.cos(ang2), np.sin(ang2)))
        for i in range(len(idx)):
            grid.setdefault((int(ys[i] // cell), int(xs[i] // cell)), []).append(
                (float(ys[i]), float(xs[i]), float(np.cos(ang2[i])),
                 float(np.sin(ang2[i])), k))
    for k, (ys, xs, c2, s2) in enumerate(samples):
        if not len(ys):
            continue
        r = max(3.0, _PAR_R * max(widths[k], 1.0))
        rr = int(np.ceil(r / cell))
        hits = 0
        near = 0
        seen: set[int] = set()               # 반경 안에 있는 다른 획 (중요도 비교용)
        for i in range(len(ys)):
            cy, cx = int(ys[i] // cell), int(xs[i] // cell)
            cnt = 0
            for gy in range(cy - rr, cy + rr + 1):
                for gx in range(cx - rr, cx + rr + 1):
                    for (py, px, pc2, ps2, pk) in grid.get((gy, gx), ()):
                        if pk == k:
                            continue
                        if (ys[i] - py) ** 2 + (xs[i] - px) ** 2 > r * r:
                            continue
                        seen.add(pk)
                        if pc2 * c2[i] + ps2 * s2[i] >= _PAR_COS:
                            cnt += 1
            hits += cnt > 0
            near += cnt
        evs[k].repeat = hits / len(ys)
        evs[k].parallel = near / len(ys)
        if seen:
            med = float(np.median([evs[j].importance for j in seen]))
            evs[k].imp_rel = (evs[k].importance / med if med > 1e-9 else 1.0)
