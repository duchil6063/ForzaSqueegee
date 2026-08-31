"""도안 **읽기 2단계** — `Look`(상자·팔레트)이 못 보는 것을 래스터로 잰다.

`Look`은 잉크 상자·볼록 껍질·팔레트만 안다. 사람이 배치한 도안에 **맞는**
꾸밈을 짜려면 그림 안이 필요하다 — 얼굴이 어디고, 포즈 축이 어느 쪽으로
기울었고, 어디가 빽빽하고 어디가 비었는지. 전부 도안 자체의 래스터에서
잰다 (모델 없음): 두 바탕색으로 렌더한 차이가 실루엣이고, 살색 덩어리가
머리이며, 소벨 크기가 디테일이고, 구조 텐서가 결의 방향이다.

좌표는 전부 **캔버스 유닛**이다 (도안 좌표) — 면에 앉힌 뒤의 자리는
`field`가 배치 변환을 먹여 푼다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import cv2
import numpy as np

from ..catalog import Catalog
from ..model import LayerPlan, rgb_to_hsb
from ..render import render_plan
from .boxes import major_axis
from .look import Look
from .palette import is_skin


# 래스터 긴 변 (px). 얼굴·축·디테일을 재는 데는 이만하면 넉넉하고 (2,000장
# 도안도 두 번 렌더에 0.3초), 더 키우면 후보 루프가 느려진다.
INTENT_PX = 320


# 머리를 찾는 창 — 잉크 상자 **위쪽 몫**에서 살색 덩어리를 찾는다. 전신
# 애니 그림의 머리는 키의 1/5, 버스트는 절반이다 (`place.FACE_FRAC_*`).
HEAD_SEARCH = 0.45


# 살색 덩어리 → 머리 상자로 넓히는 몫 (머리카락까지). 가로는 얼굴 폭의
# 이만큼 양쪽으로, 위로는 얼굴 높이만큼 (정수리·머리카락).
HEAD_GROW_W = 0.55


HEAD_GROW_UP = 1.05


# 디테일 지도의 흐림 반경 (래스터 긴 변의 몫).
DETAIL_BLUR = 0.04


@dataclass
class DesignIntent:
    """도안 하나의 **읽은 뜻** — 꾸밈이 무엇을 살리고 무엇을 피해야 하나."""

    lk: Look
    # 래스터 — 캔버스 유닛 ↔ 픽셀. 원점은 잉크 상자 왼쪽 위 (행 0 = 상자 위끝).
    upp: float                                 # 유닛/px
    origin: tuple[float, float]                # 픽셀 (0,0) 모서리의 캔버스 (x, y_top)
    alpha: np.ndarray = field(repr=False)      # (H,W) 0~1 실루엣
    rgb: np.ndarray = field(repr=False)        # (H,W,3) uint8
    detail: np.ndarray = field(repr=False)     # (H,W) 0~1 디테일 밀도
    # 읽은 것들 (캔버스 유닛)
    visual_center: tuple[float, float] = (0.0, 0.0)
    axis: tuple[float, float] = (0.0, 1.0)     # 포즈 장축 (머리 쪽이 +)
    elongation: float = 1.0                    # 장축/단축
    head: tuple[float, float, float, float] | None = None
    head_confident: bool = False
    face_dir: float = 0.0                      # 얼굴이 향하는 x 방향 (-1~1, 0=모름)
    flow: tuple[float, float] = (1.0, 0.0)     # 결(머리카락·옷 주름)의 방향
    flow_coherence: float = 0.0                # 결이 한 방향인가 (0~1)
    angularity: float = 0.0                    # 실루엣이 뾰족한가 (0~1)
    density: float = 0.0                       # 잉크 상자 안 실루엣 몫
    detail_mean: float = 0.0                   # 실루엣 안 평균 디테일
    occupancy: np.ndarray = field(repr=False, default_factory=lambda: np.zeros((1, 1)))
    # 색 역할의 씨앗 — 실루엣 픽셀에서 잰 것 (rgb). None이면 없다.
    shadow_rgb: tuple[int, int, int] | None = None
    highlight_rgb: tuple[int, int, int] | None = None
    dark_neutral_rgb: tuple[int, int, int] | None = None
    light_neutral_rgb: tuple[int, int, int] | None = None
    edge_rgb: tuple[int, int, int] = (128, 128, 128)   # 실루엣 테두리 평균색
    edge_lum: float = 0.5
    # 실루엣 **속**의 평균 명도 — 테두리가 아니라 덩어리의 밝기다.
    #
    # 둘은 자주 반대다: 파스텔 인물은 속이 밝은데 윤곽선이 검어서 `edge_lum`이
    # 낮다. 그것만 보고 판 색을 고르면 밝은 인물 뒤에 **밝은 판**이 깔리고,
    # 가까이서는 윤곽선 덕에 읽히지만 멀리서는 인물이 판에 녹는다 (실측
    # silvia-01: 테두리 명도차 0.46인데 far 배율 끌림이 0.029 — 꾸밈 덩어리의
    # 0.199에 밀려 주역이 뒤집혔다). 멀리서 읽히는 것은 덩어리의 밝기다.
    body_lum: float = 0.5

    # ---- 좌표 ----
    def to_px(self, x: float, y: float) -> tuple[float, float]:
        return (x - self.origin[0]) / self.upp, (self.origin[1] - y) / self.upp

    def to_xy(self, col: float, row: float) -> tuple[float, float]:
        return self.origin[0] + col * self.upp, self.origin[1] - row * self.upp

    def alpha_at(self, x: float, y: float) -> float:
        c, r = self.to_px(x, y)
        h, w = self.alpha.shape
        ci, ri = int(c), int(r)
        if 0 <= ci < w and 0 <= ri < h:
            return float(self.alpha[ri, ci])
        return 0.0

    @property
    def impression(self) -> str:
        """`sharp`(뾰족·기계적) · `soft`(둥글·부드러움) · `mixed`."""
        if self.angularity >= 0.55:
            return "sharp"
        if self.angularity <= 0.30:
            return "soft"
        return "mixed"

    @property
    def airy(self) -> bool:
        """성긴 그림인가 — 실루엣이 상자를 덜 채우고 디테일이 낮다."""
        return self.density < 0.42 or self.detail_mean < 0.16


def _raster(plan: LayerPlan, lk: Look, cat: Catalog
            ) -> tuple[np.ndarray, np.ndarray, float, tuple[float, float]]:
    """도안을 잉크 상자 크기로 두 번 렌더해 (rgb, alpha, upp, origin)을 낸다.

    바탕 흰/검 두 벌의 차이가 알파다 — 뺄셈 마스크·반투명 그라데이션까지
    렌더러가 처리한 결과에서 나오므로 폴리곤을 직접 채우는 것보다 정확하다.
    """
    x0, y0, x1, y1 = lk.box
    w, h = max(1.0, x1 - x0), max(1.0, y1 - y0)
    upp = max(w, h) / INTENT_PX
    W, H = max(8, int(math.ceil(w / upp)) + 2), max(8, int(math.ceil(h / upp)) + 2)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    shifted = LayerPlan(source_image=plan.source_image, image_size=(W, H),
                        units_per_px=upp,
                        layers=[replace(l, x=l.x - cx, y=l.y - cy)
                                for l in plan.layers])
    a = render_plan(shifted, cat, bg=0).astype(np.int16)
    b = render_plan(shifted, cat, bg=255).astype(np.int16)
    diff = np.abs(b - a).mean(axis=2) / 255.0
    alpha = np.clip(1.0 - diff, 0.0, 1.0).astype(np.float32)
    # 색은 검은 바탕 렌더를 알파로 되돌린 것 — 반투명은 어두워지지만 역할 판정에는 충분하다
    rgb = np.clip(b, 0, 255).astype(np.uint8)
    rgb[alpha < 0.5] = 255
    origin = (cx - W / 2 * upp, cy + H / 2 * upp)
    return rgb, alpha, upp, origin


def _pca_axis(mask: np.ndarray) -> tuple[tuple[float, float], float]:
    """실루엣의 장축 (px, y-down) 과 장/단축 비. 픽셀이 모자라면 세로축."""
    ys, xs = np.where(mask)
    if len(xs) < 16:
        return (0.0, -1.0), 1.0
    # 2차 모멘트는 **명시적 합 + 닫힌 식**이다 (`boxes.major_axis`) — BLAS·LAPACK을
    # 거치면 스레드 수에 따라 마지막 비트가 흔들려 결정성이 깨진다
    return major_axis(xs, ys)


def _head_box(rgb: np.ndarray, alpha: np.ndarray, lk: Look
              ) -> tuple[tuple[int, int, int, int] | None, bool, float]:
    """살색으로 머리 상자(px)를 찾는다. 되돌림: (상자, 확신, 얼굴 x방향).

    얼굴은 **가장 큰 살색 원판**이다 — 팔·목은 가늘고 얼굴은 둥글다. 살색
    덩어리 하나를 통째로 쓰면 올린 팔까지 이어져 상자가 그림 절반이 된다
    (실측: B1-03). 거리 변환의 극대 중 반지름이 큰 것들을 놓고, 그중 **위쪽**
    (머리는 위에 있다)을 고른다. 누운 그림은 위가 아니라 장축 끝이지만 그때도
    얼굴 원판이 가슴·허벅지 원판보다 작지 않아 반지름이 가른다.
    """
    H, W = alpha.shape
    sil = alpha > 0.5
    n_sil = int(sil.sum())
    if n_sil < 32:
        return None, False, 0.0
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hh = hsv[..., 0].astype(np.float32) / 180.0
    ss = hsv[..., 1].astype(np.float32) / 255.0
    vv = hsv[..., 2].astype(np.float32) / 255.0
    skin = sil & (hh <= 0.11) & (ss < 0.55) & (ss > 0.03) & (vv > 0.55)
    skin = cv2.morphologyEx(skin.astype(np.uint8), cv2.MORPH_OPEN,
                            np.ones((3, 3), np.uint8)).astype(bool)
    if skin.sum() < 0.003 * n_sil:
        return None, False, 0.0
    dist = cv2.distanceTransform(skin.astype(np.uint8), cv2.DIST_L2, 5)
    rmax = float(dist.max())
    if rmax < 3.0:
        return None, False, 0.0
    # 극대 후보 — 반지름이 최대의 3할 이상인 자리들 (얼굴은 허벅지·가슴
    # 원판보다 작을 수 있다 — 수영복 그림)
    k = max(3, int(rmax) | 1)
    peak = (dist >= cv2.dilate(dist, np.ones((k, k), np.uint8)) - 1e-3) & (dist >= 0.3 * rmax)
    ys, xs = np.where(peak)
    rows = np.where(sil.any(axis=1))[0]
    top, bot = int(rows.min()), int(rows.max())
    span = max(1, bot - top)
    hair = sil & ~skin
    best, bs = None, -1e9
    for y, x in zip(ys, xs):
        r = float(dist[y, x])
        topness = 1.0 - (y - top) / span
        # **얼굴 위에는 머리카락이 있다** — 원판 위 상자에 살색 아닌 잉크가
        # 얼마나 있나. 올린 팔(위가 빈 곳)과 가슴(위가 얼굴 살)을 이것이 가른다.
        y0, y1 = max(0, int(y - 2.6 * r)), max(0, int(y - 1.2 * r))
        x0, x1 = max(0, int(x - r)), min(W, int(x + r) + 1)
        above = hair[y0:y1, x0:x1]
        hair_above = float(above.mean()) if above.size else 0.0
        sc = 0.45 * topness + 0.25 * (r / rmax) + 0.30 * min(1.0, hair_above * 1.5)
        if sc > bs:
            best, bs = (int(x), int(y), r), sc
    if best is None:
        return None, False, 0.0
    cx, cy, r = best
    # 얼굴 원판 → 얼굴 상자 (원판은 볼 안쪽이라 1.35배가 얼굴 윤곽). 살색이
    # 목·가슴까지 한 덩이인 그림은 원판이 얼굴보다 커지므로 잉크 폭으로 막는다.
    cols = np.where(sil.any(axis=0))[0]
    fr = min(1.35 * r, 0.22 * (cols.max() - cols.min() + 1), 0.16 * span)
    x, y = int(cx - fr), int(cy - fr)
    w = h = int(2 * fr)
    # 얼굴 안에서 살색 무게중심이 원판 중심에서 벗어난 쪽이 **얼굴이 향하는**
    # 쪽 (반측면은 얼굴 살이 코 쪽으로 몰린다). 약한 근거라 크기를 죽인다.
    sub = skin[max(0, y):y + h, max(0, x):x + w]
    if sub.any():
        cols = np.arange(sub.shape[1]) + max(0, x)
        ccx = float((sub.sum(axis=0) * cols).sum() / sub.sum())
        face_dir = float(max(-1.0, min(1.0, (ccx - cx) / max(1.0, fr) * 3.0)))
    else:
        face_dir = 0.0
    gx = int(HEAD_GROW_W * w)
    hx0, hx1 = max(0, x - gx), min(W - 1, x + w + gx)
    hy0, hy1 = max(0, y - int(HEAD_GROW_UP * h)), min(H - 1, y + h + int(0.15 * h))
    # 위쪽 6할 밖의 얼굴은 못 믿는다 (누운 그림이 아니면 머리는 위에 있다)
    confident = (math.pi * r * r) >= 0.010 * n_sil and (cy - top) / span < 0.6
    return (hx0, hy0, hx1, hy1), confident, face_dir


def _angularity(alpha: np.ndarray) -> float:
    sil = (alpha > 0.5).astype(np.uint8)
    cnts, _ = cv2.findContours(sil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return 0.0
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 16:
        return 0.0
    hull = cv2.convexHull(c)
    solidity = area / max(1.0, cv2.contourArea(hull))
    per = cv2.arcLength(c, True)
    poly = cv2.approxPolyDP(c, 0.012 * per, True).reshape(-1, 2).astype(np.float64)
    sharp = 0
    n = len(poly)
    for i in range(n):
        p0, p1, p2 = poly[i - 1], poly[i], poly[(i + 1) % n]
        a, b = p0 - p1, p2 - p1
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-6 or nb < 1e-6:
            continue
        ang = math.degrees(math.acos(max(-1.0, min(1.0, float(a @ b) / (na * nb)))))
        if ang < 75.0:
            sharp += 1
    frac = sharp / max(1, n)
    # 세 자를 섞는다 (테스트 11장 실측으로 눈금을 맞췄다): 뾰족한 꼭짓점 몫
    # (B1-07 0.00 ↔ B1-09 0.50) · 껍질 대비 빈 몫 (0.09 ↔ 0.44) · 둘레 대비
    # 껍질 둘레 (1.16 ↔ 2.09). 0.10(미카·둥글다) ~ 0.77(히나타·다리와 의자살).
    rough = cv2.arcLength(c, True) / max(1e-6, cv2.arcLength(hull, True)) - 1.0
    return float(max(0.0, min(1.0, 0.45 * frac + 0.35 * (1.0 - solidity) / 0.45
                                  + 0.20 * rough / 1.1)))


def _flow(rgb: np.ndarray, alpha: np.ndarray) -> tuple[tuple[float, float], float]:
    """구조 텐서로 **결의 방향**(px, y-down 단위벡터)과 일관성을 잰다."""
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    m = alpha > 0.5
    if m.sum() < 16:
        return (1.0, 0.0), 0.0
    jxx, jyy, jxy = float((gx * gx)[m].sum()), float((gy * gy)[m].sum()), float((gx * gy)[m].sum())
    # 결의 방향은 기울기와 **직교**한다
    theta = 0.5 * math.atan2(2 * jxy, jxx - jyy)
    coh = math.hypot(jxx - jyy, 2 * jxy) / max(1e-6, jxx + jyy)
    dx, dy = -math.sin(theta), math.cos(theta)
    return (dx, dy), float(max(0.0, min(1.0, coh)))


def _role_seeds(rgb: np.ndarray, alpha: np.ndarray, lk: Look) -> dict:
    """실루엣 픽셀을 팔레트 색으로 양자화해 그림자·하이라이트·무채 씨앗을 뽑는다."""
    m = alpha > 0.5
    out: dict = {"shadow": None, "highlight": None, "dark": None, "light": None}
    if not m.any() or not lk.palette:
        return out
    pal = np.array(lk.palette[:12], np.float32)
    px = rgb[m].astype(np.float32)
    if len(px) > 20000:
        px = px[np.linspace(0, len(px) - 1, 20000).astype(int)]
    d = ((px[:, None, :] - pal[None, :, :]) ** 2).sum(axis=2)
    idx = np.argmin(d, axis=1)
    share = np.bincount(idx, minlength=len(pal)) / max(1, len(px))
    rows = []
    for c, w in zip(lk.palette[:12], share):
        h, s, b = rgb_to_hsb(*c)
        rows.append((c, float(w), h, s, b))
    chroma = [r for r in rows if r[3] >= 0.25 and not is_skin(r[2], r[3], r[4]) and r[1] >= 0.01]
    if chroma:
        sh = min(chroma, key=lambda r: r[4])
        if sh[4] < 0.50:
            out["shadow"] = sh[0]
        hi = max(chroma, key=lambda r: r[4] * (0.5 + r[3]))
        if hi[4] > 0.70:
            out["highlight"] = hi[0]
    grey = [r for r in rows if r[3] < 0.22 and r[1] >= 0.01]
    if grey:
        dk = min(grey, key=lambda r: r[4])
        if dk[4] < 0.35:
            out["dark"] = dk[0]
        lt = max(grey, key=lambda r: r[4])
        if lt[4] > 0.85:
            out["light"] = lt[0]
    return out


def read_intent(plan: LayerPlan, lk: Look, cat: Catalog) -> DesignIntent:
    """도안 한 장 → `DesignIntent`. 결정적이다 (같은 도안 → 같은 값)."""
    rgb, alpha, upp, origin = _raster(plan, lk, cat)
    H, W = alpha.shape
    sil = alpha > 0.5
    n_sil = int(sil.sum())
    x0, y0, x1, y1 = lk.box

    # 디테일 — 소벨 크기를 흐린 것 (실루엣 안만)
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    mag = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3),
                        cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
    k = max(3, int(DETAIL_BLUR * max(H, W)) | 1)
    det = cv2.blur(mag * sil.astype(np.float32), (k, k))
    if det.max() > 1e-6:
        det = det / float(np.percentile(det[sil], 97) if n_sil else det.max())
    detail = np.clip(det, 0.0, 1.0).astype(np.float32)

    head_px, confident, face_dir = _head_box(rgb, alpha, lk)
    head = None
    if head_px is not None:
        hx0, hy0, hx1, hy1 = head_px
        head = (origin[0] + hx0 * upp, origin[1] - hy1 * upp,
                origin[0] + hx1 * upp, origin[1] - hy0 * upp)
    elif n_sil:
        rows = np.where(sil.any(axis=1))[0]
        frac = 0.22 if lk.kind == "tall" else 0.42
        top = int(rows.min())
        hy1 = top + int(frac * (rows.max() - top))
        cols = np.where(sil[top:hy1 + 1].any(axis=0))[0]
        if len(cols):
            head = (origin[0] + cols.min() * upp, origin[1] - hy1 * upp,
                    origin[0] + cols.max() * upp, origin[1] - top * upp)

    # 시각 중심 — 알파 × (0.4 + 디테일) 가중, 머리 상자는 두 배
    wgt = alpha * (0.4 + detail)
    if head_px is not None:
        hx0, hy0, hx1, hy1 = head_px
        wgt[hy0:hy1 + 1, hx0:hx1 + 1] *= 2.0
    tot = float(wgt.sum())
    if tot > 1e-6:
        ys, xs = np.mgrid[0:H, 0:W]
        vc_px = (float((wgt * xs).sum() / tot), float((wgt * ys).sum() / tot))
    else:
        vc_px = (W / 2, H / 2)
    vc = (origin[0] + vc_px[0] * upp, origin[1] - vc_px[1] * upp)

    (ax, ay), elong = _pca_axis(sil)
    # 머리 쪽이 +가 되게 부호를 맞춘다 (y-down px → 캔버스 y-up)
    axis = (ax, -ay)
    if head is not None:
        hc = ((head[0] + head[2]) / 2 - vc[0], (head[1] + head[3]) / 2 - vc[1])
        if axis[0] * hc[0] + axis[1] * hc[1] < 0:
            axis = (-axis[0], -axis[1])
    elif axis[1] < 0:
        axis = (-axis[0], -axis[1])

    (fx, fy), coh = _flow(rgb, alpha)
    seeds = _role_seeds(rgb, alpha, lk)

    # 실루엣 테두리 평균색 — 베드·베이스가 이 색과 갈려야 실루엣이 산다
    er = cv2.erode(sil.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    ring = sil & ~er
    if ring.any():
        e = rgb[ring].astype(np.float32).mean(axis=0)
    elif n_sil:
        e = rgb[sil].astype(np.float32).mean(axis=0)
    else:
        e = np.array([128.0, 128.0, 128.0])
    edge_rgb = tuple(int(v) for v in e)
    edge_lum = float((0.299 * e[0] + 0.587 * e[1] + 0.114 * e[2]) / 255.0)
    if n_sil:
        b = rgb[sil].astype(np.float32).mean(axis=0)
        body_lum = float((0.299 * b[0] + 0.587 * b[1] + 0.114 * b[2]) / 255.0)
    else:
        body_lum = 0.5

    # 점유 격자 (12×12) — 빈 자리 판정용
    occ = cv2.resize(alpha, (12, 12), interpolation=cv2.INTER_AREA)

    return DesignIntent(
        lk=lk, upp=upp, origin=origin, alpha=alpha, rgb=rgb, detail=detail,
        visual_center=vc, axis=axis, elongation=elong, head=head,
        head_confident=confident, face_dir=face_dir, flow=(fx, -fy),
        flow_coherence=coh, angularity=_angularity(alpha),
        density=n_sil / max(1, H * W),
        detail_mean=float(detail[sil].mean()) if n_sil else 0.0,
        occupancy=occ,
        shadow_rgb=seeds["shadow"], highlight_rgb=seeds["highlight"],
        dark_neutral_rgb=seeds["dark"], light_neutral_rgb=seeds["light"],
        edge_rgb=edge_rgb, edge_lum=edge_lum, body_lum=body_lum)


def empty_regions(it: DesignIntent, thr: float = 0.12
                  ) -> list[tuple[float, float, float, float]]:
    """잉크 상자 안에서 **비어 있는** 격자 칸들 (캔버스 유닛 상자)."""
    occ = it.occupancy
    H, W = occ.shape
    x0, y0, x1, y1 = it.lk.box
    cw, ch = (x1 - x0) / W, (y1 - y0) / H
    out = []
    for r in range(H):
        for c in range(W):
            if occ[r, c] < thr:
                out.append((x0 + c * cw, y1 - (r + 1) * ch, x0 + (c + 1) * cw, y1 - r * ch))
    return out
