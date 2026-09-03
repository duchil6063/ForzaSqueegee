"""색면 **스택** — 사람 옆면의 바닥 문법 (띠 2~4 + 블록 + 무늬 가장자리 + 뺄셈 마스크).

## 왜

사람 판 30벌의 옆면 바닥은 큰 색면 **13장 · 6색**이고(작가 단위 p10~p90 = 6~22장),
그 정체는 차체 선을 따르는 띠들이다 — 벨트라인 아래의 어두운 띠, 뒷휠 아치에서
솟는 날, 블록을 따라 달리는 가는 핀, 그리고 블록의 가장자리를 찢거나 튀기는
무늬 도형 한두 장. 우리 판(W7L)의 큰 색면은 12장인데 판 1~2장 + 키라인 원판 +
로커 톱니라 **수는 같고 정체가 다르다** (`work/lab/humanref/structure2.py`).

`macro`의 어휘 아홉은 판 **하나**의 모양이다 (`bed_level`·좌표하강이 그 판을
다듬는다). 여기는 그 판 위에 **스택**을 한 단위로 얹는다 — 판의 모양을 더
늘리는 것이 아니라 재료를 늘리는 것이다 (계획서 §2 · 2026-09-02).

## 조각 다섯 (`PIECES`)

    belt   벨트라인 바로 아래의 얇은 관통 띠 (블랙아웃 — 사람 판의 무채 띠)
    arch   흐름 쪽 휠아치에서 벨트라인으로 솟는 날 (blade — 위아래로 나간다)
    pin    블록과 나란한 가는 선 두 줄 (레이싱 핀스트라이프)
    edge   블록 가장자리에 겹친 무늬 도형 (D 찢김 · G 튐 · V 스월) — 블록 색
    gap    블록 속을 달리는 얇은 **뺄셈 마스크** — 차 색이 드러나는 홈

어느 조각을 쓰는지는 **계열이 정한다** (`families.Family.stack`) — 후보 축이
아니다. 축으로 두면 후보가 배로 늘고, 사람 판의 스택은 계열(레이싱·그래픽·
스플래시)의 문법이지 도안마다 고르는 것이 아니다. 매개변수는 전부 인물·블록·
차체 선에서 나온다 (`plan`).

## 사람 실측 (2026-09-03, 옆면 50장)

- 큰 색면의 각: 절반이 10° 아래(벨트·로커를 따르는 띠), 4분의 1이 45° 위(아치
  에서 솟는 날). 두께는 상자 높이의 0.18~0.68 (p25~p75).
- 전단은 큰 색면 26% · 작은 것 27%로 **같다** — 전단 비는 도안(획)의 몫이라 이
  스택으로는 못 닫는다.
- 큰 색면의 알파는 96%가 100 — 반투명 판은 안 쓴다 (알파는 글로우·그림자 몫).
- 마스크는 옆면당 중앙 39장인데 크기 중앙이 면 폭의 1%라 **도안 안의 지우개**다.
  큰 색면을 도려내는 마스크는 드물다 — 여기서는 홈 한 줄만 판다.
- 무늬 페이지의 큰 색면은 D_03(찢긴 띠)이 압도적, 작은 것은 V_19·V_69(거친
  결)·D_03·V_70·C_16이다.

좌표는 프레임 좌표(꾸밈 캔버스, y-up). 라벨은 `itasha_stack` — 예산 사다리
(`design.TRIM_ORDER`)·바닥 요소 판정(`score.GROUND`)이 그 이름을 읽는다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..catalog import Catalog
from ..model import UNITS_PER_SCALE, Layer
from .macro import STEEP_KINDS, MacroSpec, _band, _cap_width, _dir, _rect, _run


LABEL = "itasha_stack"


PIECES = ("belt", "arch", "pin", "edge", "gap", "streak")


# 스트릭 — 인물 뒤에서 흐름 쪽으로 **길게 나가는 가는 선** (개수 · 두께 · 간격 · 각 상한 ·
# 인물 시각 중심에서의 세로 치우침, 밴드·인물 높이 몫). 실루엣 키라인(울퉁불퉁한
# 흰 테)을 대신한다 — 사람 판의 색면은 인물 뒤로 한 방향으로 뻗는다 (사용자 지시
# 2026-09-03: "차라리 없는 게 낫다, 라인 같은 걸로 뒤에 길게").
STREAK_N = 2
STREAK_T = 0.03
STREAK_GAP = 0.10
STREAK_ANG = 12.0
STREAK_DY = -0.05


# 벨트 띠의 두께 (밴드 높이 몫, level이 그 사이를 걷는다). 사람 큰 색면 두께의
# 아래 사분위(상자 높이의 0.12~0.18)에 맞춘다 — 벨트 띠는 스택의 가장 얇은 판이다.
BELT_T = (0.08, 0.12)


# 벨트라인에서 띠가 내려온 몫 (밴드 높이 대비) — 라인에 딱 붙이면 유리 몰딩과 겹친다.
BELT_INSET = 0.02


# 아치 날의 기울기 (도, 수평 기준) — 사람 판의 가파른 색면은 45° 위에 몰린다.
ARCH_ANG = 58.0


# 아치 날의 폭 (인물 폭 몫). 위아래로 나가는 날이라 넓어도 패널을 안 덮는다
# (`macro` 문서의 가파른 색면 셈). 넓이 상한은 `_cap_width`가 따로 건다.
ARCH_W = (0.32, 0.52)
ARCH_SHARE = 0.14


# 인물 머리가 아치 위에 있으면 날을 세우지 않는다 (인물 폭 몫) — 머리 뒤의 큰
# 사선은 멀리서 읽히는 실루엣을 깨뜨린다.
ARCH_HEAD_CLEAR = 0.30


# 핀스트라이프 두께 (밴드 높이 몫)와 하한(유닛), 블록 가장자리에서 떨어진 거리
# (인물 높이 몫).
PIN_T = 0.035
PIN_MIN = 6.0
PIN_OFF = 0.09


# 무늬 가장자리 — 길이(인물 폭 몫)·높이(블록 폭 몫)·자리(블록 축을 따라 인물
# 폭의 몫만큼 흐름 쪽으로).
EDGE_LEN = 0.90
EDGE_LEN_MAX = 0.40           # 프레임 폭 몫 상한
EDGE_H = 0.55
EDGE_ALONG = 0.55


# 한쪽으로만 나가는 조각(벨트·홈)이 인물 뒤에서 시작하는 자리 — 시각 중심에서
# 흐름 반대쪽으로 인물 폭의 이 몫. 인물이 그 시작을 가린다.
HALF_BACK = 0.22


# 홈(뺄셈 마스크) — 두께(밴드 높이 몫)·하한·블록 중심선에서의 자리(블록 폭 몫).
GAP_T = 0.045
GAP_MIN = 5.0
GAP_OFF = 0.30


# 계열 → 가장자리 무늬 도형 (게임 도형 id 표에 전부 있다 — `catalog/fls_shape_ids.json`).
# D_03·D_02는 찢긴 띠, G_12·G_13은 옆으로 튄 물감, V_01·V_11은 스월이다.
EDGE_SHAPES: dict[str, tuple[str, ...]] = {
    "graphic_bed": ("D_03", "D_02"),
    "diagonal_flow": ("V_01", "V_11"),
    "dark": ("V_01", "V_11"),
    "splash": ("G_12", "G_13"),
}


# 벨트 띠가 무채(`dark`)로 서려면 블록과 이만큼은 갈려야 한다 (Lab ΔE). 아니면
# 주 액센트로 간다 — 검은 판 위 검은 띠는 없는 것과 같다.
BELT_DE_MIN = 30.0


# 휠아치 검출 — 밴드 아래쪽 이 몫의 칸이 이만큼 비도색이면 아치 열이다.
ARCH_ROWS = 0.30
ARCH_HOLE = 0.60
ARCH_MIN_W = 0.06             # 프레임 폭 몫


@dataclass(frozen=True)
class StackPiece:
    """스택 조각 하나의 매개변수 (프레임 좌표)."""

    kind: str
    at: tuple[float, float] = (0.0, 0.0)
    ang: float = 0.0
    width: float = 10.0        # 축 법선 방향 두께
    cut: float = 0.0           # 끝 전단 (`macro.MacroSpec.cut`)
    taper: float = 0.0
    role: str = "dark"
    shape: str = ""            # edge — 무늬 도형 이름
    length: float = 0.0        # edge — 축 방향 길이 (관통 조각은 0)
    side: float = 0.0          # 0 = 양쪽으로 관통 · ±1 = 그쪽으로만 나간다 (시작은 인물 뒤)
    z: float = 2.0


def _de(a, b) -> float:
    import cv2
    la = cv2.cvtColor(np.array([[list(a)]], np.uint8), cv2.COLOR_RGB2LAB)[0, 0].astype(float)
    lb = cv2.cvtColor(np.array([[list(b)]], np.uint8), cv2.COLOR_RGB2LAB)[0, 0].astype(float)
    return float(np.linalg.norm(la - lb))


def arches(fld) -> list[float]:
    """휠아치 구멍의 **x 중심들** (프레임 좌표) — 밴드 아래쪽의 비도색 열에서 읽는다.

    `fld.drawable`은 면 도색 마스크를 프레임 격자에 얹은 것이라 (`build`의
    `_drawable_at`) 아치 구멍이 그대로 비도색 칸으로 남아 있다. 지도가 없는
    판(드로어블이 전부 1)이면 빈 목록 — 그때 날은 흐름 쪽 프레임 4분의 1 자리에 선다.
    """
    g = fld.grid
    fx0, fy0, fx1, fy1 = fld.frame_box
    band = fy1 - fy0
    r_top = max(0, min(g.rows - 1, int((g.y_top - (fy0 + ARCH_ROWS * band)) / g.cell)))
    r_bot = max(r_top + 1, min(g.rows, int(math.ceil((g.y_top - fy0) / g.cell))))
    sub = fld.drawable[r_top:r_bot, :]
    if sub.size == 0:
        return []
    hole = (sub < 0.5).mean(axis=0) >= ARCH_HOLE
    out: list[float] = []
    span = fx1 - fx0
    start = None
    for c in range(g.cols + 1):
        on = c < g.cols and bool(hole[c])
        if on and start is None:
            start = c
        elif not on and start is not None:
            if (c - start) * g.cell >= ARCH_MIN_W * span:
                out.append(g.x0 + (start + c) / 2 * g.cell)
            start = None
    return out


def _flat(sp: MacroSpec) -> bool:
    return sp.kind not in STEEP_KINDS and sp.kind != "none"


def plan(fld, family: str, pieces: tuple[str, ...], specs: tuple[MacroSpec, ...],
         level: float, *, colors: dict[str, tuple[int, int, int]],
         rocker: bool = False) -> tuple[StackPiece, ...]:
    """구도 필드 + 계열의 조각 목록 + 블록 명세 → 스택 조각들.

    `specs[0]`(z가 가장 낮은 것)이 블록이다. 블록이 얕은 띠(ribbon·blade·stack)면
    핀·가장자리·홈이 그 축을 따르고, `split`이면 가른 선을 따른다. 그 밖의
    가파른 어휘(corner·burst·chevron·sweep·bracket)에는 블록 종속 조각을 안
    붙인다 — 벨트·아치만 선다.
    """
    if not pieces or not specs:
        return ()
    fx0, fy0, fx1, fy1 = fld.frame_box
    band = fy1 - fy0
    span = fx1 - fx0
    ch, cw = fld.char_h, fld.char_w
    vcx, vcy = fld.visual_center
    fsign = 1.0 if fld.flow[0] >= 0 else -1.0
    block = min(specs, key=lambda s: s.z)
    flat = _flat(block)
    # 핀·가장자리의 **숙주**는 얕은 띠다 — 블록이 가파르면(burst·sweep·corner)
    # 짝(ribbon·blade)이 얕을 때 그쪽을 따른다. 홈은 블록에만 판다.
    host = block if flat else next((s for s in specs if _flat(s)), None)
    out: list[StackPiece] = []
    bed_rgb = colors.get(block.role, colors["bed"])
    for kind in pieces:
        if kind == "belt":
            t = (BELT_T[0] + (BELT_T[1] - BELT_T[0]) * max(0.0, min(1.0, level))) * band
            # 잉크는 로커와 **같은 무채**다 — 벨트와 로커가 다른 검정이면 큰 색면의
            # 색 수만 는다 (사람 옆면 6색: 색을 늘리지 않고 판을 늘린다). 로커가
            # 안 서는 어두운 차에서는 무채 잉크가 밝은 쪽이라 `dark`가 맞다.
            ink = "rocker" if (rocker and "rocker" in colors
                               and _de(colors["dark"], colors["rocker"]) < 20.0) else "dark"
            role = ink if _de(colors[ink], bed_rgb) >= BELT_DE_MIN else "primary"
            ang = max(-6.0, min(6.0, 0.4 * block.ang)) if flat else 0.0
            # 인물 뒤에서 시작해 **흐름 쪽으로만** 나간다 — 시작은 인물이 가리고
            # 끝은 차가 자른다. 관통시키면 흐름 반대쪽의 여백 덩이를 가르고
            # 커버리지가 넘쳐 벨트를 쓰는 계열이 한 판도 못 이겼다 (W8S1 실측:
            # graphic_bed 7 → 0 · motorsport 15 → 1, presence 0.73 → 0.00).
            out.append(StackPiece(kind="belt", at=(vcx - fsign * HALF_BACK * cw,
                                                   fy1 - BELT_INSET * band - t / 2),
                                  ang=ang, width=t, cut=0.30, role=role, z=5.0,
                                  side=fsign))
        elif kind == "arch":
            holes = arches(fld)
            end = fx1 if fsign > 0 else fx0
            ax = (min(holes, key=lambda x: abs(x - end)) if holes
                  else end - fsign * 0.22 * span)
            hx = fld.head_center[0] if fld.head_center is not None else vcx
            if abs(ax - hx) < ARCH_HEAD_CLEAR * cw:
                continue                          # 머리 뒤에 사선을 세우지 않는다
            ang = 90.0 - fsign * (90.0 - ARCH_ANG)
            w = _cap_width(fld.frame_box, ang,
                           (ARCH_W[0] + (ARCH_W[1] - ARCH_W[0]) * level) * cw,
                           share=ARCH_SHARE)
            # 색은 **블록과 같다** — 새 색을 들이지 않는다 (큰 색면 색 수는 이미
            # 사람 p90 언저리다: W8S1에서 bed_alt·secondary를 들이니 7 → 9색으로
            # 범위 밖). 같은 색의 두 번째 방향이 곧 사람 판의 "판 + 날"이다.
            out.append(StackPiece(kind="arch", at=(ax, fy0), ang=ang, width=w,
                                  taper=0.40, role=block.role, z=3.0))
        elif kind == "pin" and host is not None:
            d = _dir(host.ang)
            nx, ny = -d[1], d[0]
            t = max(PIN_MIN, PIN_T * band)
            off = host.width / 2 + PIN_OFF * ch + t / 2
            for s in (1.0, -1.0):
                out.append(StackPiece(kind="pin",
                                      at=(host.at[0] + nx * off * s, host.at[1] + ny * off * s),
                                      ang=host.ang, width=t, cut=host.cut,
                                      role="primary", z=4.0))
        elif kind == "streak":
            ang = host.ang if host is not None else max(-STREAK_ANG, min(STREAK_ANG, 0.4 * block.ang))
            t = max(PIN_MIN, STREAK_T * band)
            gap = STREAK_GAP * band
            d = _dir(ang)
            nx, ny = -d[1], d[0]
            # 색은 주 액센트 — 판과 안 갈리면 무채 잉크
            role = "primary" if _de(colors.get("primary", bed_rgb), bed_rgb) >= BELT_DE_MIN else "dark"
            x0, y0 = vcx - fsign * HALF_BACK * cw, vcy + STREAK_DY * ch
            for k in range(STREAK_N):
                off = (k - (STREAK_N - 1) / 2) * gap
                out.append(StackPiece(kind="streak", at=(x0 + nx * off, y0 + ny * off),
                                      ang=ang, width=t, cut=0.30, role=role, z=4.5,
                                      side=fsign))
        elif kind == "edge":
            shapes = EDGE_SHAPES.get(family, ())
            if not shapes:
                continue
            if host is not None:
                d = _dir(host.ang)
                nx, ny = -d[1], d[0]
                L = min(EDGE_LEN * cw, EDGE_LEN_MAX * span)
                h = EDGE_H * host.width
                along = EDGE_ALONG * cw
                for k, s in enumerate((1.0, -1.0)):
                    px = host.at[0] + d[0] * fsign * s * along + nx * s * host.width / 2
                    py = host.at[1] + d[1] * fsign * s * along + ny * s * host.width / 2
                    out.append(StackPiece(kind="edge", at=(px, py),
                                          ang=host.ang + (0.0 if s > 0 else 180.0),
                                          width=h, length=L, role=host.role,
                                          shape=shapes[k % len(shapes)], z=host.z + 0.5))
            elif block.kind == "split":
                L = min(1.1 * band, EDGE_LEN_MAX * span)
                out.append(StackPiece(kind="edge", at=block.at, ang=block.ang,
                                      width=0.35 * band, length=L, role=block.role,
                                      shape=shapes[0], z=block.z + 0.5))
        elif kind == "gap":
            t = max(GAP_MIN, GAP_T * band)
            if flat:
                d = _dir(block.ang)
                nx, ny = -d[1], d[0]
                off = GAP_OFF * block.width
                # 홈도 인물 뒤에서 흐름 쪽으로만 — 관통시키면 블록이 두 덩어리로
                # 갈려 위계 자(`critic.macro`)가 둘째 덩어리로 읽는다
                sx0 = vcx - fsign * HALF_BACK * cw
                out.append(StackPiece(kind="gap",
                                      at=(sx0 + nx * off, block.at[1] + ny * off
                                          + d[1] / max(1e-6, d[0]) * (sx0 - block.at[0])),
                                      ang=block.ang, width=t, cut=block.cut,
                                      z=block.z + 0.7, side=fsign))
            elif block.kind == "split":
                d = _dir(block.ang)
                nx, ny = -d[1] * block.side, d[0] * block.side
                off = 0.10 * span
                out.append(StackPiece(kind="gap",
                                      at=(block.at[0] + nx * off, block.at[1] + ny * off),
                                      ang=block.ang, width=t, z=block.z + 0.7))
    return tuple(out)


def _native_half(cat: Catalog, name: str) -> tuple[float, float]:
    """도형의 로컬 반폭·반높이 — 무늬 도형은 ±1이 아니다 (`CatShape.reach`)."""
    sh = cat.shapes.get(name)
    if sh is None or not sh.loops:
        return 1.0, 1.0
    pts = np.concatenate([np.asarray(l) for l in sh.loops if len(l)])
    hx = float(max(abs(pts[:, 0].min()), abs(pts[:, 0].max()))) or 1.0
    hy = float(max(abs(pts[:, 1].min()), abs(pts[:, 1].max()))) or 1.0
    return hx, hy


def _half_band(frame, p: StackPiece, color, cat: Catalog) -> list[Layer]:
    """`p.at`에서 `p.side` 쪽으로만 프레임을 나가는 띠 — 시작 쪽은 인물 뒤에 숨는다.

    시작 끝은 프레임 안에 있으므로 전단(`cut`)을 그 끝에도 먹인다 — 곧게 자른
    끝이 인물 뒤에서 삐져나와도 눕힌 단면이라 띠의 꼴로 읽힌다.
    """
    d = _dir(p.ang)
    half = p.width / 2
    fwd = _run(frame, p.at[0], p.at[1], d, half, p.side) + abs(p.cut) * half
    if fwd <= 1e-6:
        return []
    cx = p.at[0] + d[0] * p.side * fwd / 2
    cy = p.at[1] + d[1] * p.side * fwd / 2
    return [_rect(cx, cy, fwd, p.width, p.ang, color, cat, 100.0, skew=p.cut)]


def _relabel(layers: list[Layer]) -> list[Layer]:
    for l in layers:
        l.label = LABEL
    return layers


def build(pieces: tuple[StackPiece, ...], frame: tuple[float, float, float, float],
          colors: dict[str, tuple[int, int, int]], cat: Catalog
          ) -> list[tuple[float, list[Layer]]]:
    """조각들 → (z, 레이어들) 묶음. 부르는 쪽이 매크로의 z와 섞어 정렬한다."""
    out: list[tuple[float, list[Layer]]] = []
    for p in pieces:
        color = colors.get(p.role, colors["bed"])
        if p.kind in ("belt", "pin", "arch", "streak"):
            ls = (_half_band(frame, p, color, cat) if p.side
                  else _band(frame, p.at, p.ang, p.width, p.cut, p.taper, color, cat, 100.0))
            out.append((p.z, _relabel(ls)))
        elif p.kind == "gap":
            ls = (_half_band(frame, p, (0, 0, 0), cat) if p.side
                  else _band(frame, p.at, p.ang, p.width, p.cut, 0.0, (0, 0, 0), cat, 100.0))
            for l in ls:
                l.mask = True
            out.append((p.z, _relabel(ls)))
        elif p.kind == "edge" and p.shape in cat.shapes:
            hx, hy = _native_half(cat, p.shape)
            out.append((p.z, [Layer(shape=p.shape, x=p.at[0], y=p.at[1],
                                    sx=p.length / 2 / UNITS_PER_SCALE / hx,
                                    sy=p.width / 2 / UNITS_PER_SCALE / hy,
                                    rot=p.ang % 360.0, color=color, label=LABEL)]))
    return out


def n_layers(pieces: tuple[StackPiece, ...]) -> int:
    n = 0
    for p in pieces:
        n += 2 if (p.kind == "arch" and p.taper >= 0.12) else 1
    return n
