"""구성 필드 — 사람이 앉힌 도안 둘레의 **어디를 채우고 어디를 비우나**.

사람이 만든 이타샤의 핵심은 채움/비움의 배분이다. 옆면 차체 밴드를 꾸밈
캔버스 좌표(`design`의 `frame_box` — 밴드 가운데가 원점, 폭 900)의
격자로 놓고, 배치 변환을 거꾸로 먹여 도안의 실루엣·머리·디테일을 그 격자에
얹는다. 그 위에서 다섯 구역을 낸다:

- **protected** — 얼굴·핵심 실루엣. 전경 장식이 침범하면 안 된다.
- **support**   — 인물을 받치는 색면(베드)이 앉기 좋은 구역 (실루엣 후광 + 포즈 축 슬래브).
- **decoration** — 띠·모티프가 서도 되는 구역 (흐름 쪽, 실루엣에서 떨어진 도색면).
- **negative**  — 일부러 비우는 구역 (흐름 반대쪽 먼 자리).
- **flow**      — 그래픽이 흐르는 방향.

전부 같은 격자(`FieldGrid`) 위의 0~1 래스터라 점수기가 그대로 겹쳐 잰다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from .intent import DesignIntent


# 격자 칸 (캔버스 유닛). 900유닛 프레임이 180칸 — 베드·모티프 판정에 충분하다.
CELL = 5.0


# 실루엣 둘레의 **여유 띠** (인물 높이 대비) — 배경 모티프는 이 안에 안 선다.
GAP_FRAC = 0.06


# 베드 후광 — 실루엣에서 이만큼 나간 자리까지가 베드의 본 자리 (인물 높이 대비).
HALO_FRAC = 0.14


# 머리 보호 띠 (머리 상자 크기 대비).
HEAD_PAD = 0.18


@dataclass
class FieldGrid:
    """꾸밈 캔버스 좌표의 격자 — 원점은 프레임 왼쪽 위 칸의 모서리."""

    x0: float
    y_top: float
    cols: int
    rows: int
    cell: float = CELL

    def centers(self) -> tuple[np.ndarray, np.ndarray]:
        xs = self.x0 + (np.arange(self.cols) + 0.5) * self.cell
        ys = self.y_top - (np.arange(self.rows) + 0.5) * self.cell
        return np.meshgrid(xs, ys)

    def to_cell(self, x: float, y: float) -> tuple[int, int]:
        return int((x - self.x0) / self.cell), int((self.y_top - y) / self.cell)

    def at(self, ras: np.ndarray, x: float, y: float) -> float:
        c, r = self.to_cell(x, y)
        if 0 <= c < self.cols and 0 <= r < self.rows:
            return float(ras[r, c])
        return 0.0

    def box_of(self, ras: np.ndarray, thr: float = 0.5
               ) -> tuple[float, float, float, float] | None:
        ys, xs = np.where(ras >= thr)
        if len(xs) == 0:
            return None
        return (self.x0 + xs.min() * self.cell, self.y_top - (ys.max() + 1) * self.cell,
                self.x0 + (xs.max() + 1) * self.cell, self.y_top - ys.min() * self.cell)

    def px(self, units: float) -> int:
        return max(1, int(round(units / self.cell)))


@dataclass
class CompositionField:
    grid: FieldGrid
    frame_box: tuple[float, float, float, float]
    person_box: tuple[float, float, float, float]
    char: np.ndarray = field(repr=False)          # 실루엣 알파
    char_rgb: np.ndarray = field(repr=False)      # 실루엣 색 (H,W,3)
    detail: np.ndarray = field(repr=False)
    drawable: np.ndarray = field(repr=False)      # 이 자리가 실제로 그려지나
    head: np.ndarray = field(repr=False)
    protected: np.ndarray = field(repr=False)
    support: np.ndarray = field(repr=False)
    decoration: np.ndarray = field(repr=False)
    negative: np.ndarray = field(repr=False)
    flow: tuple[float, float] = (1.0, 0.0)
    axis: tuple[float, float] = (0.0, 1.0)        # 포즈 장축 (프레임 좌표, 머리 쪽 +)
    head_center: tuple[float, float] | None = None
    face_dir: float = 0.0                         # 얼굴이 향하는 프레임 x 방향
    visual_center: tuple[float, float] = (0.0, 0.0)
    texture: tuple[float, float] = (1.0, 0.0)     # 결(머리카락·주름)의 방향 (프레임 좌표)
    texture_coherence: float = 0.0
    rear_sign: float = 1.0                        # +x가 차 뒤면 +1
    free: dict[str, float] = field(default_factory=dict)   # 방향별 빈 도색면 몫

    @property
    def char_h(self) -> float:
        return self.person_box[3] - self.person_box[1]

    @property
    def char_w(self) -> float:
        return self.person_box[2] - self.person_box[0]

    def flow_angle(self) -> float:
        """흐름 방향의 각 (도, 프레임 x축 기준 반시계)."""
        return math.degrees(math.atan2(self.flow[1], self.flow[0]))


def _sample(ras: np.ndarray, cols: np.ndarray, rows: np.ndarray) -> np.ndarray:
    h, w = ras.shape[:2]
    ci = np.floor(cols).astype(int)
    ri = np.floor(rows).astype(int)
    ok = (ci >= 0) & (ci < w) & (ri >= 0) & (ri < h)
    out = np.zeros(cols.shape, np.float32)
    out[ok] = ras[ri[ok], ci[ok]]
    return out


def build_field(it: DesignIntent, L: np.ndarray, t: np.ndarray,
                frame_center: tuple[float, float], u: float,
                frame_box: tuple[float, float, float, float],
                person_box: tuple[float, float, float, float],
                rear_sign: float, drawable_at=None,
                flow: tuple[float, float] | None = None) -> CompositionField:
    """도안 뜻 + 배치 변환 → 구성 필드.

    `L`·`t`는 `place.place_xf`(캔버스 점 p → 면 유닛 `L p + t`), 프레임 좌표는
    `(면 유닛 − frame_center) / u`다 (`build`의 꾸밈 프레임과 같은 식).
    `flow`를 주면 그 방향을 못 박고, 안 주면 여기서 고른다 (`choose_flow`).
    """
    fx0, fy0, fx1, fy1 = frame_box
    # 프레임을 위아래로 조금 넉넉히 — 인물이 밴드를 넘길 수 있다 (사이드실 아래 발)
    y_lo = min(fy0, person_box[1]) - 2 * CELL
    y_hi = max(fy1, person_box[3]) + 2 * CELL
    cols = int(math.ceil((fx1 - fx0) / CELL))
    rows = int(math.ceil((y_hi - y_lo) / CELL))
    g = FieldGrid(x0=fx0, y_top=y_hi, cols=cols, rows=rows)
    X, Y = g.centers()
    # 프레임 → 면 유닛 → 캔버스(도안) 좌표.
    #
    # **행렬 곱도 역행렬도 안 쓴다.** 둘 다 BLAS·LAPACK을 거치는데 거기서 나오는
    # 마지막 비트가 스레드 수에 따라 흔들리고, 그 흔들림이 필드 래스터 한 칸을
    # 뒤집어 점수 순위를 바꾼다 — 같은 입력이 프로세스마다 다른 `deco.json`을
    # 냈다 (2026-09-01 실측 · `boxes.major_axis`에 같은 사정을 적었다).
    # 2×2는 닫힌 식이 정확하고 더 싸다.
    det = float(L[0, 0]) * float(L[1, 1]) - float(L[0, 1]) * float(L[1, 0])
    det = det if abs(det) > 1e-12 else math.copysign(1e-12, det or 1.0)
    i00, i01 = float(L[1, 1]) / det, -float(L[0, 1]) / det
    i10, i11 = -float(L[1, 0]) / det, float(L[0, 0]) / det
    qx = X * u + (frame_center[0] - float(t[0]))
    qy = Y * u + (frame_center[1] - float(t[1]))
    px_ = qx * i00 + qy * i01
    py_ = qx * i10 + qy * i11
    pc = (px_ - it.origin[0]) / it.upp
    pr = (it.origin[1] - py_) / it.upp
    char = _sample(it.alpha, pc, pr)
    detail = _sample(it.detail, pc, pr)
    char_rgb = np.stack([_sample(it.rgb[..., k].astype(np.float32), pc, pr)
                         for k in range(3)], -1).astype(np.uint8)
    if drawable_at is not None:
        draw = np.array([[1.0 if drawable_at(float(X[r, c]), float(Y[r, c])) else 0.0
                          for c in range(cols)] for r in range(rows)], np.float32)
    else:
        draw = np.ones((rows, cols), np.float32)
        draw[(Y < fy0) | (Y > fy1)] = 0.0
    # 머리 — 도안 상자를 변환해 폴리곤으로
    head = np.zeros((rows, cols), np.float32)
    head_c = None
    if it.head is not None:
        hx0, hy0, hx1, hy1 = it.head
        pts = np.array([[hx0, hy0], [hx1, hy0], [hx1, hy1], [hx0, hy1]], float)
        q = (pts @ L.T + t - np.array(frame_center)) / u
        poly = np.stack([(q[:, 0] - g.x0) / CELL, (g.y_top - q[:, 1]) / CELL], 1)
        cv2.fillPoly(head, [np.round(poly).astype(np.int32)], 1.0)
        head_c = (float(q[:, 0].mean()), float(q[:, 1].mean()))
    ch = person_box[3] - person_box[1]
    k_gap = g.px(GAP_FRAC * ch)
    k_halo = g.px(HALO_FRAC * ch)
    sil = (char > 0.5).astype(np.uint8)
    sil_gap = cv2.dilate(sil, np.ones((2 * k_gap + 1, 2 * k_gap + 1), np.uint8))
    sil_halo = cv2.dilate(sil, np.ones((2 * k_halo + 1, 2 * k_halo + 1), np.uint8))
    # 보호 — 머리(+여유) ∪ 디테일 높은 실루엣 핵
    kh = g.px(HEAD_PAD * ch) if head_c is not None else 1
    prot = cv2.dilate(head.astype(np.uint8), np.ones((2 * kh + 1, 2 * kh + 1), np.uint8)).astype(np.float32)
    prot = np.maximum(prot, ((char > 0.5) & (detail > 0.45)).astype(np.float32))
    # 포즈 축·시각 중심 (프레임 좌표)
    a = np.array(it.axis, float) @ L.T
    n = np.linalg.norm(a)
    axis = (float(a[0] / n), float(a[1] / n)) if n > 1e-6 else (0.0, 1.0)
    tx = np.array(it.flow, float) @ L.T
    tn = np.linalg.norm(tx)
    texture = (float(tx[0] / tn), float(tx[1] / tn)) if tn > 1e-6 else (1.0, 0.0)
    vc = (np.array(it.visual_center, float) @ L.T + t - np.array(frame_center)) / u
    vcx, vcy = float(vc[0]), float(vc[1])
    face_dir = float(np.sign(L[0, 0]) * it.face_dir) if it.head_confident else 0.0
    # 방향별 **빈 도색면** — 인물 상자 밖에서 그려지는 칸의 몫
    outside = (draw > 0.5) & (sil_gap == 0)
    free = {"pos": float(outside[:, X[0] > person_box[2]].mean()) if (X[0] > person_box[2]).any() else 0.0,
            "neg": float(outside[:, X[0] < person_box[0]].mean()) if (X[0] < person_box[0]).any() else 0.0}
    fl = flow if flow is not None else choose_flow(axis, face_dir, free, rear_sign)
    fnorm = math.hypot(*fl) or 1.0
    fl = (fl[0] / fnorm, fl[1] / fnorm)
    # 지지 — 실루엣 후광 ∪ 포즈 축 슬래브 (흐름 쪽으로 더 뻗는다)
    supp = sil_halo.astype(np.float32)
    slab = np.zeros((rows, cols), np.float32)
    ax, ay = axis
    # 슬래브 축은 포즈 축과 흐름의 섞임 — 세운 인물은 사선, 누운 인물은 흐름 그대로
    sx, sy = 0.55 * ax + 0.45 * fl[0], 0.55 * ay + 0.45 * fl[1]
    sn = math.hypot(sx, sy) or 1.0
    sx, sy = sx / sn, sy / sn
    half_len = 0.80 * max(ch, person_box[2] - person_box[0])
    half_wid = 0.28 * ch
    c1 = np.array([vcx, vcy]) + np.array([fl[0], fl[1]]) * 0.15 * ch
    dirs = np.array([[sx, sy], [-sy, sx]])
    corners = [c1 + dirs[0] * half_len + dirs[1] * half_wid,
               c1 + dirs[0] * half_len - dirs[1] * half_wid,
               c1 - dirs[0] * half_len - dirs[1] * half_wid,
               c1 - dirs[0] * half_len + dirs[1] * half_wid]
    poly = np.array([[(p[0] - g.x0) / CELL, (g.y_top - p[1]) / CELL] for p in corners])
    cv2.fillPoly(slab, [np.round(poly).astype(np.int32)], 1.0)
    supp = np.maximum(supp, slab) * (draw > 0.5)
    # 장식 — 그려지는 자리 ∧ 실루엣 여유 밖 ∧ 흐름 쪽 (거리로 감쇠)
    dx, dy = X - vcx, Y - vcy
    along = (dx * fl[0] + dy * fl[1]) / max(1.0, 0.5 * (fx1 - fx0))
    deco = np.clip(0.35 + along * 1.2, 0.0, 1.0) * (draw > 0.5) * (sil_gap == 0)
    # 여백 — 흐름 반대쪽, 인물에서 인물 폭 이상 떨어진 자리
    back = -(dx * fl[0] + dy * fl[1])
    neg = ((back > 0.55 * (person_box[2] - person_box[0]) + 0.5 * (person_box[3] - person_box[1]))
           & (draw > 0.5) & (sil_halo == 0)).astype(np.float32)
    return CompositionField(
        grid=g, frame_box=frame_box, person_box=person_box, char=char,
        char_rgb=char_rgb, detail=detail, drawable=draw, head=head, protected=np.clip(prot, 0, 1),
        support=supp.astype(np.float32), decoration=deco.astype(np.float32),
        negative=neg, flow=fl, axis=axis, head_center=head_c, face_dir=face_dir,
        visual_center=(vcx, vcy), texture=texture,
        texture_coherence=it.flow_coherence, rear_sign=rear_sign, free=free)


def choose_flow(axis: tuple[float, float], face_dir: float, free: dict[str, float],
                rear_sign: float) -> tuple[float, float]:
    """흐름 방향 — 뒤 고정이 아니라 **빈 자리 · 얼굴 방향 · 포즈 축**이 정한다.

    점수: 빈 도색면이 넓은 쪽 +, 얼굴이 향하는 쪽 + (시선 앞을 비우지 않고
    채우는 것이 레퍼런스의 다수 — RIN의 글자는 얼굴 앞에 있다), 차 뒤쪽 +
    (동률이면 레퍼런스의 다수인 리어 쿼터). 세로 성분은 포즈 축의 기울기를
    조금 따른다 — 대각 포즈면 대각선 흐름.
    """
    pos = 0.55 * free.get("pos", 0.0) + 0.25 * max(0.0, face_dir) + 0.20 * (rear_sign > 0)
    neg = 0.55 * free.get("neg", 0.0) + 0.25 * max(0.0, -face_dir) + 0.20 * (rear_sign < 0)
    sx = 1.0 if pos >= neg else -1.0
    # 포즈 축이 기울었으면 그 기울기를 흐름의 세로 성분으로 (상한 ±0.45)
    ax, ay = axis
    tilt = 0.0
    if abs(ax) > 0.25:                       # 축이 가로에 가깝다 (눕힌 인물·대각)
        tilt = max(-0.45, min(0.45, (ay / max(1e-6, abs(ax))) * math.copysign(1.0, ax) * sx * 0.6))
    return (sx, tilt)
