"""스폰서 문법 — 로고 무리를 차에 앉힌다 (내장 워터마크 + 사용자 로고).

사람 판(30벌)에서 읽은 자리다: **로커 위 한 줄**(글자 높이 4~7유닛), 리어 범퍼
가운데 워드마크 + 좌우 로고, 프론트 범퍼 작은 로고 2~3, 윈드실드 띠. 로고 폭은
옆면 폭의 9~10%(중앙)이고 워터마크는 그 절반이다 (`work/lab/humanref`).

- **옆면 줄**은 옆면 설계의 필드(`CompositionField`)에서 앉힌다 — 로커 글자
  (`textlayout.rocker_pose`)와 같은 높이, 인물의 흐름 쪽에서 시작해 흐름 방향으로
  늘어선다. 자리 규칙(그려지는 몫·인물 가림·보호구역·이음새)은 글자와 같은 자
  (`textlayout._settle`)로 잰다.
- **다른 면**(리어·프론트·윈드실드)은 면 도색 상자에서 앉힌다 — 아래 띠 위의
  줄이고, 그 면에 이미 있는 덩어리(변주·글자)는 피한다.
- **로고는 미러하지 않는다** (사용자 결정 ③). 반대편 옆면은 자리만 거울이고
  로고는 읽는 방향 그대로다 — 글자의 `mirrored_set`과 같은 방식.
- 워터마크는 **바탕 밝기**로 두 벌 중 하나를 고른다 (밝은 바탕 → 검정 잉크).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from ...game import surface as gsurf
from ...i18n import msg
from ..catalog import Catalog
from ..model import Layer, LayerPlan, UNITS_PER_SCALE, rnd
from .boxes import _overlap, _rel
from .bands import ROCKER_FRAC
from .field import CompositionField
from .logokit import LogoItem, LogoSpec, watermark_plan
from .look import look
from .place import _refit_canvas
from .textlayout import ROCKER_TEXT_H, TextPose, _settle, pose_mask


# 옆면 로고 폭 (옆면 폭 대비) — 사람 판 중앙 9~10%.
SIDE_LOGO_W = 0.095


# 워터마크는 사람 로고 폭의 절반.
WATERMARK_K = 0.5


# 옆면 줄의 높이 상한 (차체 밴드 대비) — 로커 글자와 같다.
ROW_H = ROCKER_TEXT_H


# 로고 사이 틈 (줄 높이 대비).
GAP = 0.35


# 다른 면의 사용자 로고 = 옆면 로고 폭 × 이것 (범퍼의 로고는 옆면 것보다 작다 —
# 사람 판의 프론트 로고는 65~448장짜리 작은 것 2~3개다).
FACE_LOGO_K = 0.6


# 한 면에서 로고 하나가 차지할 수 있는 폭 상한 · 줄 전체 상한 (면 폭 대비).
FACE_W_MAX = 0.24
FACE_ROW_W = 0.90


# 다른 면 줄의 중심 높이 (면 아래에서, 높이 몫) — 범퍼 띠 바로 위.
FACE_ROW_V = 0.20


# 윈드실드 귀퉁이 워터마크 — 가장자리에서의 여백 (면 크기 몫).
CORNER_PAD = 0.08


# 다른 면 줄의 로고가 **읽히는 크기의 하한** — 기준 폭(옆면 로고 폭 × `FACE_LOGO_K`)
# 대비. 노출 띠가 좁아 이 아래로 줄어야 들어가면 뺀다. W13 실측: 리어 줄은 0.37~0.45
# 배에서 읽히고(실비아·미아타), 프론트 0.21배는 점이었다.
FACE_LOGO_MIN = 0.33


# 면 위의 다른 덩어리와 이만큼 넘게 겹치면 그 자리는 안 쓴다 (로고 상자 몫).
# 로고는 도안·글자에 가려지면 안 된다 (사용자 지시 2026-09-03) — 상자끼리 스치는
# 몫만 남긴다.
BUSY_MAX = 0.02


@dataclass
class Proto:
    """로고 원형 — 원점 중심 레이어와 잉크 크기 (캔버스 유닛)."""

    item: LogoItem
    layers: list[Layer]
    w: float
    h: float

    @property
    def aspect(self) -> float:
        return self.w / max(1e-6, self.h)


@dataclass
class Placed:
    """앉힌 로고 하나 — 중심·폭·각 (좌표계는 부르는 쪽이 안다)."""

    proto: Proto
    x: float
    y: float
    w: float
    rot: float = 0.0

    @property
    def h(self) -> float:
        return self.w / self.proto.aspect

    @property
    def box(self) -> tuple[float, float, float, float]:
        return (self.x - self.w / 2, self.y - self.h / 2,
                self.x + self.w / 2, self.y + self.h / 2)

    def layers(self) -> list[Layer]:
        return posed(self.proto, self.x, self.y, self.w, self.rot)

    def mirrored(self) -> "Placed":
        """반대편 옆면 — 자리는 거울, 로고는 그대로 (읽는 방향)."""
        return Placed(proto=self.proto, x=-self.x, y=self.y, w=self.w,
                      rot=(-self.rot) % 360.0)


def load_proto(item: LogoItem, cat: Catalog, cache: dict | None = None) -> Proto:
    key = str(item.plan.resolve())
    if cache is not None and key in cache:
        return replace(cache[key], item=item)
    plan = LayerPlan.load(item.plan)
    lk = look(plan, cat)
    cx, cy = (lk.box[0] + lk.box[2]) / 2, (lk.box[1] + lk.box[3]) / 2
    layers = [replace(l, x=l.x - cx, y=l.y - cy, label="logo") for l in plan.layers]
    pr = Proto(item=item, layers=layers, w=max(1e-6, lk.w), h=max(1e-6, lk.h))
    if cache is not None:
        cache[key] = pr
    return pr


def posed(pr: Proto, x: float, y: float, w: float, rot: float = 0.0) -> list[Layer]:
    """원형을 폭 `w`로 (x, y)에 각 `rot`으로 앉힌 레이어."""
    k = w / pr.w
    r = math.radians(rot)
    c, s = math.cos(r), math.sin(r)
    out: list[Layer] = []
    for l in pr.layers:
        lx, ly = l.x * k, l.y * k
        out.append(Layer(shape=l.shape, x=x + lx * c - ly * s, y=y + lx * s + ly * c,
                         sx=l.sx * k, sy=l.sy * k, rot=(l.rot + rot) % 360.0,
                         skew=l.skew, color=l.color, alpha=l.alpha, label=l.label,
                         mask=l.mask))
    return out


# ────────────────────────────── 옆면 줄 ──────────────────────────────


def _row_size(pr: Proto, frame_w: float, band: float) -> tuple[float, float]:
    w = SIDE_LOGO_W * frame_w * (WATERMARK_K if pr.item.kind == "watermark" else 1.0)
    h = w / pr.aspect
    if h > ROW_H * band:
        h = ROW_H * band
        w = h * pr.aspect
    return w, h


def side_row(protos: list[Proto], fld: CompositionField,
             text_poses: list[TextPose] | None,
             notes: list[str], size_k: float = 1.0) -> list[Placed]:
    """로커 위 한 줄 — 인물의 흐름 쪽에서 흐름 방향으로 (프레임 좌표).

    로커 글자가 이미 그 줄에 있으면 그 뒤에서 시작한다. 흐름 쪽에 못 앉는 로고는
    반대쪽 끝에서 안쪽으로 세워 본다. 그래도 안 되면 뺀다. `size_k`는 프리셋의
    로고 줄 배율 (`presets.StylePreset.logo_row` — 레이싱은 크게, 미니멀은 0 = 줄 없음)."""
    if size_k <= 0:
        return []
    fx0, fy0, fx1, fy1 = fld.frame_box
    band = fy1 - fy0
    frame_w = fx1 - fx0
    flow = 1.0 if fld.flow[0] >= 0 else -1.0
    sizes = [_row_size(p, frame_w * size_k, band * size_k) for p in protos]
    hrow = max([h for _w, h in sizes] or [0.0])
    if hrow <= 0:
        return []
    y = fy0 + ROCKER_FRAC * band + 0.06 * band + 0.5 * hrow
    gap = GAP * hrow
    acc = np.zeros((fld.grid.rows, fld.grid.cols), bool)
    # 글자가 이미 쓴 자리 — 로커 글자·사인은 같은 줄이거나 그 끝이다
    starts = {+1.0: fld.person_box[2] + 0.08 * fld.char_w,
              -1.0: fld.person_box[0] - 0.08 * fld.char_w}
    for p in (text_poses or []):
        acc |= pose_mask(fld, p)
        if abs(p.y - y) < 0.5 * (p.h + hrow):
            for sgn in (+1.0, -1.0):
                edge = p.x + sgn * p.w / 2
                if sgn * (edge - starts[sgn]) > 0:
                    starts[sgn] = edge + sgn * gap
    cursor = {+1.0: starts[+1.0], -1.0: starts[-1.0]}
    ends = {+1.0: fx1 - 0.03 * frame_w, -1.0: fx0 + 0.03 * frame_w}
    placed: list[Placed] = []
    for pr, (w, h) in zip(protos, sizes):
        got = None
        for sgn in (flow, -flow):
            if sgn == flow:
                x = cursor[sgn] + sgn * w / 2
            else:
                x = ends[sgn] - sgn * w / 2           # 반대쪽 끝에서 안쪽으로
            pose = TextPose(role="rocker", text="", x=x, y=y, rot=0.0, height=h,
                            aspect=w / max(1e-6, h), hratio=1.0)
            q = _settle(fld, pose, (1.0, 0.0), sgn, 0.45 * h, avoid=acc)
            if q is None:
                continue
            got = Placed(proto=pr, x=q.x, y=q.y, w=q.w)
            acc |= pose_mask(fld, q)
            if sgn == flow:
                cursor[sgn] = q.x + sgn * (q.w / 2 + gap)
            else:
                ends[sgn] = q.x - sgn * (q.w / 2 + gap)
            break
        if got is None:
            notes.append(msg("옆면 로커 줄에 '{name}'을(를) 앉힐 자리가 없다 — 뺀다",
                             name=pr.item.name))
            continue
        placed.append(got)
    return placed


# ────────────────────────────── 다른 면 ──────────────────────────────


def _drawn(sm: gsurf.SurfaceMap, box: tuple[float, float, float, float]) -> float:
    """상자 안 **5×5 격자점** 중 마스크 안인 몫 — 귀퉁이·변 한가운데까지 다 든다.

    종전 다섯 점(중심·네 귀)은 변 한가운데가 마스크 계단에 걸린 상자를 통과시켰다
    (W11F 프론트: 엠블럼 윗변이 후드 코에 걸쳤다)."""
    u0, v0, u1, v1 = box
    pts = [(u0 + (u1 - u0) * i / 4.0, v0 + (v1 - v0) * j / 4.0)
           for i in range(5) for j in range(5)]
    return sum(1 for u, v in pts if sm.masked_at(u, v)) / 25.0


# ────────────────────────────── 모티프 양보 ──────────────────────────────
# 글자·로고 상자 둘레로 모티프가 비워 줘야 하는 여유 (상자 짧은 변의 몫).
MARK_PAD = 0.15


def rect_box(x: float, y: float, hw: float, hh: float, rot: float
             ) -> tuple[float, float, float, float]:
    """중심·반폭·반높이·각 → 축 정렬 상자."""
    r = math.radians(rot)
    c, s = abs(math.cos(r)), abs(math.sin(r))
    ex, ey = hw * c + hh * s, hw * s + hh * c
    return (x - ex, y - ey, x + ex, y + ey)


def _pad(box, k: float = MARK_PAD):
    p = k * min(box[2] - box[0], box[3] - box[1])
    return (box[0] - p, box[1] - p, box[2] + p, box[3] + p)


def hits_marks(box, marks: list) -> bool:
    return any(_overlap(box, _pad(m)) > 0.0 for m in marks)


def prune_shape_specs(specs: list[dict], marks: list, face_w: float) -> tuple[list[dict], int]:
    """면 위저드 도형 명세에서 글자·로고 상자에 닿는 **모티프**를 뺀다.

    큰 색면(면 폭 25%↑ — 띠·블록)은 바탕이라 둔다: 글자가 띠 위에 서는 것은 사람
    판의 문법이다. 되돌림: (남은 명세, 뺀 수). 사용자 지시 2026-09-03 — 글자·로고는
    도안·꾸밈에 가려지면 안 된다. 꾸밈이 양보한다.
    """
    if not marks:
        return specs, 0
    keep, cut = [], 0
    for s in specs:
        hw, hh = UNITS_PER_SCALE * abs(float(s["sx"])), UNITS_PER_SCALE * abs(float(s.get("sy", s["sx"])))
        if 2 * max(hw, hh) >= 0.25 * face_w:
            keep.append(s)
            continue
        if hits_marks(rect_box(float(s["x"]), float(s["y"]), hw, hh, float(s.get("rot") or 0.0)),
                      marks):
            cut += 1
            continue
        keep.append(s)
    return keep, cut


# 옆면 꾸밈에서 **바탕**인 라벨 — 로커 띠·베드·색면 스택. 글자·로고가 그 위에 서는
# 것은 사람 판의 문법이라 양보하지 않는다. 나머지(산포·에코·전경 모티프)는 양보한다.
BACKGROUND_LABELS = frozenset({"itasha_bed", "itasha_stripe", "itasha_stack"})


def prune_motif_layers(layers: list[Layer], marks: list, frame_w: float
                       ) -> tuple[list[Layer], int]:
    """레이어 목록(프레임 좌표)에서 글자·로고 상자에 닿는 모티프를 뺀다 — 옆면 꾸밈
    그룹용. 바탕 라벨(`BACKGROUND_LABELS`)과 큰 색면은 `prune_shape_specs`와 같은
    자로 둔다."""
    if not marks:
        return layers, 0
    keep, cut = [], 0
    for l in layers:
        hw, hh = UNITS_PER_SCALE * abs(float(l.sx)), UNITS_PER_SCALE * abs(float(l.sy))
        if l.label in BACKGROUND_LABELS or 2 * max(hw, hh) >= 0.25 * frame_w:
            keep.append(l)
            continue
        if hits_marks(rect_box(float(l.x), float(l.y), hw, hh, float(l.rot)), marks):
            cut += 1
            continue
        keep.append(l)
    return keep, cut


def pose_box(p: TextPose) -> tuple[float, float, float, float]:
    """글자 포즈의 축 정렬 상자 (프레임 좌표)."""
    return rect_box(p.x, p.y, p.w / 2, p.h / 2, p.rot)


def _busy(box, busy: list[tuple[float, float, float, float]]) -> float:
    a = max(1e-6, (box[2] - box[0]) * (box[3] - box[1]))
    return max([_overlap(box, b) / a for b in busy] or [0.0])


def _fits(sm: gsurf.SurfaceMap, box, busy) -> bool:
    return _drawn(sm, box) >= 0.999 and _busy(box, busy) <= BUSY_MAX


def _mask_without(sm: gsurf.SurfaceMap, busy: list) -> gsurf.SurfaceMap:
    """다른 덩어리 상자(`busy`)를 판 마스크 사본 — 내접 상자를 찾을 때 글자·도안이
    차지한 자리를 빈 자리로 세지 않게 (리어: 워드마크 위·아래에서 줄 자리를 찾는다)."""
    if not busy or sm.mask.size <= 1:
        return sm
    m = sm.mask.copy()
    mh, mw = m.shape
    u0, v0, u1, v1 = sm.paint
    kx = (mw - 1) / max(1e-6, u1 - u0)
    ky = (mh - 1) / max(1e-6, v1 - v0)
    for b in busy:
        x0 = max(0, int((b[0] - u0) * kx)); x1 = min(mw, int((b[2] - u0) * kx) + 1)
        y0 = max(0, int((v1 - b[3]) * ky)); y1 = min(mh, int((v1 - b[1]) * ky) + 1)
        if x1 > x0 and y1 > y0:
            m[y0:y1, x0:x1] = False
    return replace(sm, mask=m, fill=round(float(m.mean()), 4), drawn=None)


def _nudge(sm: gsurf.SurfaceMap, pl: Placed, busy, steps: tuple[float, ...]) -> Placed | None:
    """자리를 위로 조금씩 밀어 보고, 안 들면 줄여 본다 (0.55배까지 — 좁은 노출 띠)."""
    _u0, v0, _u1, v1 = sm.paint
    H = v1 - v0
    for k in (1.0, 0.85, 0.7, 0.55):
        for dv in steps:
            q = Placed(proto=pl.proto, x=pl.x, y=pl.y + dv * H, w=pl.w * k, rot=pl.rot)
            if _fits(sm, q.box, busy):
                return q
    return None


NUDGES = (0.0, 0.04, 0.08, 0.12, -0.04, 0.16, 0.20)


def face_row(protos: list[Proto], sm: gsurf.SurfaceMap, busy: list, *,
             side_w: float | None, floor_v: float | None,
             center: Proto | None, notes: list[str]) -> list[Placed]:
    """아래 띠 위의 로고 줄 (면 유닛). `center`는 가운데 자리(워터마크)다.

    사용자 로고는 가운데 좌우로 번갈아 선다. 폭은 옆면 로고 폭에 매인다
    (`FACE_LOGO_K`) — 면마다 따로 재면 차를 돌 때 로고 크기가 널뛴다."""
    u0, v0, u1, v1 = sm.paint
    W, H = u1 - u0, v1 - v0
    ref = side_w if side_w else W / 0.4 * 1.0
    # 도어 유리처럼 마스크가 **덩이 둘**(B필러가 가른 앞·뒤 창)이면 줄은 큰 덩이의
    # 폭에 앉는다 — 면 상자 가운데는 필러라 줄이 반씩 잘렸다 (W9T 줄리아).
    pane = _pane(sm)
    if pane is not None:
        u0, u1 = pane
        W = u1 - u0
    widths: dict[int, float] = {}
    for pr in ([center] if center else []) + protos:
        w = SIDE_LOGO_W * ref * (WATERMARK_K if pr.item.kind == "watermark" else FACE_LOGO_K)
        widths[id(pr)] = min(w, FACE_W_MAX * W)
    order: list[Proto] = []
    left: list[Proto] = []
    right: list[Proto] = []
    for i, pr in enumerate(protos):
        (right if i % 2 == 0 else left).append(pr)
    order = list(reversed(left)) + ([center] if center else []) + right
    if not order:
        return []
    hrow = max(widths[id(pr)] / pr.aspect for pr in order)
    gap = GAP * hrow
    total = sum(widths[id(pr)] for pr in order) + gap * (len(order) - 1)
    if total > FACE_ROW_W * W:
        k = FACE_ROW_W * W / total
        for key in widths:
            widths[key] *= k
        hrow *= k
        gap *= k
        total = FACE_ROW_W * W
    nominal = dict(widths)                        # 읽히는 크기의 기준 — 이 아래로 줄면 뺀다
    # 줄이 **온전히 드는 상자** — 마스크(그리는 지도 ∧ 정면도 ∧ 여유, `place.usable`)에
    # 내접하는 줄 비율의 상자. 범퍼 아랫단은 마스크에 있어도 정면도가 낮아 안 보이므로
    # 상수 높이(`FACE_ROW_V`)로 못 박으면 줄이 통째로 빠진다 (W12: 프론트 33/33).
    # 상자가 줄보다 좁으면 줄을 그만큼 줄인다.
    fitbox = _mask_without(sm, busy).fit(max(0.5, total / max(1e-6, hrow)), coverage=0.995,
                                         anchor="bottom", margin=0.0)
    if fitbox is not None and pane is not None:
        fitbox = (max(fitbox[0], u0), fitbox[1], min(fitbox[2], u1), fitbox[3])
    if fitbox is not None and fitbox[2] - fitbox[0] < total:
        k = max(0.3, (fitbox[2] - fitbox[0]) / max(1e-6, total))
        for key in widths:
            widths[key] *= k
        hrow *= k
        gap *= k
        total *= k
    v = v0 + FACE_ROW_V * H
    if floor_v is not None:
        v = max(v, floor_v + 0.5 * hrow + 0.03 * H)
    ucen = (u0 + u1) / 2
    if fitbox is not None:
        v = min(max(v, fitbox[1] + hrow / 2), fitbox[3] - hrow / 2)
        ucen = (fitbox[0] + fitbox[2]) / 2
    out: list[Placed] = []
    if fitbox is None:
        steps = NUDGES
    else:
        # 상자 높이 전체를 훑는다 — 아래에서 위로, 그다음 아래로. 리어의 노출 띠는
        # 워드마크가 아래를 차지하므로 줄이 그 위로 올라가야 한다 (W12S2 리어 0/4).
        lo, hi = fitbox[1] + hrow / 2, fitbox[3] - hrow / 2
        up = [dv for dv in np.arange(0.0, (hi - v) / H + 1e-9, 0.03)]
        down = [-dv for dv in np.arange(0.03, (v - lo) / H + 1e-9, 0.03)]
        steps = tuple(float(d) for d in up + down) or (0.0,)
    u = ucen - total / 2
    for pr in order:
        w = widths[id(pr)]
        pl = Placed(proto=pr, x=u + w / 2, y=v, w=w)
        u += w + gap
        q = _nudge(sm, pl, busy, steps or (0.0,))
        if q is None:
            notes.append(msg("{surface}: '{name}'을(를) 앉힐 자리가 없다 — 뺀다",
                             surface=sm.name, name=pr.item.name))
            continue
        if q.w < FACE_LOGO_MIN * nominal[id(pr)]:
            # 들어가긴 하는데 점이 된다 (실비아 프론트: 노출 띠가 좁아 0.16배) —
            # 안 읽히는 로고는 없는 것보다 나쁘다
            notes.append(msg("{surface}: '{name}'은(는) 자리가 좁아 {k:.2f}배로 줄어야 "
                             "든다 — 뺀다", surface=sm.name, name=pr.item.name,
                             k=q.w / max(1e-6, nominal[id(pr)])))
            continue
        busy = list(busy) + [q.box]
        out.append(q)
    return out


def _pane(sm: gsurf.SurfaceMap) -> tuple[float, float] | None:
    """마스크가 덩이 둘 이상이면 **가장 큰 덩이의 u 범위** (면 유닛), 아니면 None.

    작은 부스러기(마스크 넓이의 5% 미만)는 덩이로 안 친다 — 한 덩이 면(리어·
    프론트·윈드실드)은 그대로라 답이 안 바뀐다."""
    m = sm.mask
    if m is None or m.size <= 1:
        return None
    import cv2
    n, lbl, stats, _c = cv2.connectedComponentsWithStats(
        m.astype("uint8"), connectivity=8)
    tot = max(1, int(m.sum()))
    blobs = [(int(stats[i, cv2.CC_STAT_AREA]), int(stats[i, cv2.CC_STAT_LEFT]),
              int(stats[i, cv2.CC_STAT_WIDTH])) for i in range(1, n)
             if stats[i, cv2.CC_STAT_AREA] >= 0.05 * tot]
    if len(blobs) < 2:
        return None
    area, left, width = max(blobs)
    kx = sm.px_per_unit[0]
    ox = sm.origin_px[0]
    return ((left - ox) / kx, (left + width - ox) / kx)


def corner(pr: Proto, sm: gsurf.SurfaceMap, busy: list, *, side_w: float | None,
           notes: list[str]) -> Placed | None:
    """면 아래 귀퉁이의 워터마크 — 오른쪽 아래, 안 되면 왼쪽 아래."""
    u0, v0, u1, v1 = sm.paint
    W, H = u1 - u0, v1 - v0
    ref = side_w if side_w else W / 0.4
    w = min(SIDE_LOGO_W * ref * WATERMARK_K, FACE_W_MAX * W)
    h = w / pr.aspect
    for sgn in (+1.0, -1.0):
        # 귀퉁이도 마스크(∧ 정면도)에 **드는 상자** 안이다 — 그 쪽 아래 귀퉁이에
        # 내접하는 상자를 찾아 그 구석에 앉힌다 (`place.usable` — 범퍼 아랫단은
        # 안 보인다). 상자가 없으면 종전 자리(면 상자 귀퉁이)에서 밀어 본다.
        box = _mask_without(sm, busy).fit(pr.aspect, coverage=0.995, anchor="bottom",
                                          bias_x=1.0 if sgn > 0 else 0.0, margin=0.0)
        if box is not None and (box[2] - box[0]) >= w and (box[3] - box[1]) >= h:
            x = (box[2] if sgn > 0 else box[0]) - sgn * (CORNER_PAD * W + w / 2)
            x = min(max(x, box[0] + w / 2), box[2] - w / 2)
            y = min(box[1] + CORNER_PAD * H + h / 2, box[3] - h / 2)
            pl = Placed(proto=pr, x=x, y=y, w=w)
            q = _nudge(sm, pl, busy, tuple(
                dv for dv in NUDGES if box[1] + h / 2 <= y + dv * H <= box[3] - h / 2) or (0.0,))
            if q is not None:
                return q
        x = (u1 if sgn > 0 else u0) - sgn * (CORNER_PAD * W + w / 2)
        pl = Placed(proto=pr, x=x, y=v0 + CORNER_PAD * H + h / 2, w=w)
        q = _nudge(sm, pl, busy, NUDGES)
        if q is not None:
            return q
    notes.append(msg("{surface}: 워터마크를 앉힐 귀퉁이가 없다", surface=sm.name))
    return None


# ────────────────────────────── 바탕 밝기 ──────────────────────────────


def _lum(c) -> float:
    return (0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]) / 255.0


def _inside(l: Layer, x: float, y: float) -> bool:
    """점이 레이어의 (회전한) 상자 안인가 — 반폭은 `|sx|×64`."""
    r = math.radians(-l.rot)
    c, s = math.cos(r), math.sin(r)
    dx, dy = x - l.x, y - l.y
    px, py = dx * c - dy * s, dx * s + dy * c
    return abs(px) <= abs(l.sx) * UNITS_PER_SCALE and abs(py) <= abs(l.sy) * UNITS_PER_SCALE


def under_layers(layers: list[Layer], x: float, y: float, fallback) -> tuple[int, int, int]:
    """(x, y) 밑의 색 — 위에서부터 첫 불투명 레이어, 없으면 `fallback`."""
    for l in reversed(layers):
        if l.mask or l.alpha < 50.0:
            continue
        if _inside(l, x, y):
            return tuple(int(v) for v in l.color)
    return tuple(int(v) for v in fallback)


def pick_watermark(pl: Placed, bg: tuple[int, int, int], cat: Catalog,
                   cache: dict) -> Placed:
    """바탕이 어두우면 흰 잉크 워터마크로 바꿔 앉힌다 (자리는 그대로)."""
    dark = _lum(bg) < 0.5
    want = watermark_plan(dark)
    if want is None or want.resolve() == pl.proto.item.plan.resolve():
        return pl
    pr = load_proto(replace(pl.proto.item, plan=want), cat, cache)
    return replace(pl, proto=pr)


# ────────────────────────────── 그룹 파일 ──────────────────────────────


def write_group(placed: list[Placed], path: Path, meta: LayerPlan, cat: Catalog) -> int:
    """앉힌 로고들 → 그룹 도안 한 장. 되돌림은 장수."""
    layers: list[Layer] = []
    for pl in placed:
        layers += pl.layers()
    tp = LayerPlan(source_image=meta.source_image, image_size=meta.image_size,
                   units_per_px=meta.units_per_px,
                   layers=[replace(l, x=rnd(l.x, 4), y=rnd(l.y, 4), sx=rnd(l.sx, 4),
                                   sy=rnd(l.sy, 4), rot=rnd(l.rot % 360.0, 4))
                           for l in layers])
    tp = _refit_canvas(tp, cat)
    tp.save(path)
    return len(tp.layers)


def face_group(placed: list[Placed], sm: gsurf.SurfaceMap, path: Path, meta: LayerPlan,
               cat: Catalog, out_dir: Path, group_unit: float) -> tuple[dict, int]:
    """면 유닛 자리 → 면 상자 가운데에 앉는 그룹 항목 (스케일 1/group_unit)."""
    cx, cy = (sm.paint[0] + sm.paint[2]) / 2, (sm.paint[1] + sm.paint[3]) / 2
    local = [replace(pl, x=pl.x - cx, y=pl.y - cy) for pl in placed]
    n = write_group(local, path, meta, cat)
    return ({"plan": _rel(path, out_dir), "x": round(cx, 1), "y": round(cy, 1),
             "scale": round(1.0 / max(1e-6, group_unit), 4), "rot": 0.0,
             "mirror": False}, n)


def spec_targets(spec: LogoSpec) -> dict[str, bool]:
    """자리 옵션 → 어느 줄을 쓰나."""
    p = spec.placement
    return {"side": p in ("auto", "rocker"), "rear": p in ("auto", "rear"),
            "front": p in ("auto", "front"), "windshield": p in ("auto", "windshield")}
