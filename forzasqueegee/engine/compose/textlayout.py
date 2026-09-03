"""텍스트 배치 — 워드마크가 **캐릭터를 받치는 자리**를 필드에서 찾는다.

글자는 산포 모티프처럼 흩지 않는다. 사람이 만든 이타샤의 이름은 늘 정해진
자리에 선다 (레퍼런스 실측): 인물 뒤 사선 판을 따라가는 대형 워드마크(RIN
SHIBUYA), 로커 위 얇은 레이싱 글자(ARIS의 스폰서 행), 리어 쿼터의 작은 사인
(EVELYNE). 이 모듈은 그 세 문법을 필드 좌표(꾸밈 프레임)의 **포즈**로 낸다 —
어느 포즈가 이기는지는 점수가 정한다 (`textscore`).

자리는 넷이다: 인물 곁 사선 워드마크(`wordmark_poses`), 인물 **뒤**를 가로지르는
큰 워드마크(`behind_pose` — RIN SHIBUYA 문법, 인물이 글자를 일부 덮는다), 로커
위 글자, 리어 쿼터 사인. 이름이 한 줄로 길면(공백이 있으면) **두 줄 락업**도
후보다 (`lockups`) — 옆면 밴드는 높이가 150유닛 남짓이라 한 줄 이름은 자리
길이가 높이를 정하는데, 두 줄이면 같은 자리에서 글자가 두 배 선다.

규칙:
- 보호 구역(얼굴)은 **밑에 깔려도** 안 건드린다 — 글자가 얼굴 뒤에서 비치면 얼굴이 어수선해진다.
- 인물이 글자를 덮는 몫은 45% 아래, 인물 뒤 워드마크는 50% (뒤에 깔린 워드마크가 인물에 반쯤 가려지는 것은 문법이지만 그 이상이면 안 읽힌다).
- 그려지는 자리(도색 마스크) 90% 이상 — 휠아치·벨트라인에 잘린 글자는 안 읽힌다.
  자리 길이도 휠아치에서 끊는다 (구멍을 건너뛰면 글자가 구멍 위를 지난다).
- 메인과 서브의 위계: 서브는 메인 대문자 높이의 0.42배, 메인 밑(축의 수직 방향)에 붙는다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .bands import ROCKER_FRAC
from .bed import slab_axis
from .field import CompositionField


# 워드마크 대문자 높이 상한 (인물 높이 대비) — 이보다 크면 글자가 주인공이 된다.
WORDMARK_H_MAX = 0.40


# 로커 위 글자 높이 (차체 밴드 대비) · 사인 높이
ROCKER_TEXT_H = 0.15


SIGNATURE_H = 0.24


# 서브 텍스트 높이 (메인 대비)
SUB_RATIO = 0.42


# 인물이 글자를 덮어도 되는 몫 — **사실상 0**이다 (사용자 지시 2026-09-03: 글자·로고는
# 도안에 가려지면 안 된다). 종전 0.45(인물 뒤 워드마크 0.5 — RIN SHIBUYA의 양끝만
# 읽히는 문법)는 W11F 33판에서 옆면 글자 10/26이 인물 밑에 들어가게 했다. 2%는
# 상자 귀퉁이가 인물 껍질 격자에 스치는 몫이다.
OCCLUDE_MAX = 0.02


# 인물 뒤 워드마크도 같다 — 자리는 남겨 두되(글자가 인물보다 커서 양끝이 읽히는
# 도안이 있다) 덮이는 몫은 같은 자로 잰다.
OCCLUDE_MAX_BEHIND = 0.02


def occlude_max(role: str) -> float:
    return OCCLUDE_MAX_BEHIND if role == "behind" else OCCLUDE_MAX


# 그려지는 자리의 하한 — 글자·로고는 면을 벗어나면 안 된다 (사용자 지시 2026-09-03).
# 1.0이 아닌 것은 격자 계단 때문이다 (프레임 격자 셀이 마스크 경계에 걸친다).
DRAWABLE_MIN = 0.985


# 자리 길이를 끊는 구멍의 최소 길이 (프레임 유닛) — 이보다 짧은 틈(마스크 계단)은 건넌다
HOLE_MIN = 14.0


@dataclass
class TextPose:
    role: str                  # wordmark · rocker · signature · sub
    text: str
    x: float
    y: float
    rot: float                 # 도
    height: float              # 대문자 높이 (프레임 유닛)
    aspect: float              # 잉크 상자 w/h (글자 블록)
    hratio: float = 1.0        # 잉크 상자 높이 / 대문자 높이 (두 줄이면 ~2.3)
    on_bed: bool = False

    @property
    def w(self) -> float:
        return self.h * self.aspect

    @property
    def h(self) -> float:
        """잉크 상자 높이 — 배치·잘림·덮임은 전부 이 상자로 잰다."""
        return self.height * self.hratio

    def mirrored(self) -> "TextPose":
        """반대편 옆면의 포즈 — 자리는 거울, 글자는 그대로 (읽히는 방향)."""
        return TextPose(role=self.role, text=self.text, x=-self.x, y=self.y,
                        rot=(-self.rot) % 360.0, height=self.height,
                        aspect=self.aspect, hratio=self.hratio, on_bed=self.on_bed)


def pose_mask(fld: CompositionField, p: TextPose) -> np.ndarray:
    """포즈의 회전 상자를 필드 격자에 채운 마스크."""
    g = fld.grid
    m = np.zeros((g.rows, g.cols), np.uint8)
    r = math.radians(p.rot)
    c, s = math.cos(r), math.sin(r)
    hw, hh = p.w / 2, p.h / 2
    pts = []
    for dx, dy in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)):
        x = p.x + dx * c - dy * s
        y = p.y + dx * s + dy * c
        pts.append([(x - g.x0) / g.cell, (g.y_top - y) / g.cell])
    cv2.fillPoly(m, [np.round(np.array(pts)).astype(np.int32)], 1)
    return m.astype(bool)


def pose_fit(fld: CompositionField, p: TextPose, avoid: np.ndarray | None = None
             ) -> tuple[float, float, float]:
    """(그려지는 몫, 인물이 덮는 몫, 보호구역 몫) — 포즈 상자 안에서.

    **격자 밖은 안 그려지는 자리로 센다.** 상자가 프레임을 넘으면 넘은 몫은
    마스크에 아예 안 담기므로, 담긴 칸만 보면 "면 밖으로 반쯤 나간 글자"가
    만점을 받는다 (실측: 미아타 옆면의 인물 뒤 워드마크가 리어 범퍼 너머로
    나가 'RIN SHIBUY'로 잘렸는데 그려지는 몫이 1.00이었다). 상자가 덮어야 할
    칸 수를 넓이로 따로 세어 분모에 둔다.

    `avoid`(다른 글자 상자 마스크)를 주면 보호구역처럼 센다 — 서브가 메인 위에
    올라앉는 것을 막는다 (실측: 휠아치를 피하려다 메인 안으로 들어갔다)."""
    m = pose_mask(fld, p)
    n = int(m.sum())
    if n == 0:
        return 0.0, 1.0, 1.0
    want = max(float(n), p.w * p.h / (fld.grid.cell ** 2))
    prot = fld.protected > 0.5
    if avoid is not None:
        prot = prot | avoid
    return (float((fld.drawable[m] > 0.5).sum()) / want,
            float((fld.char[m] > 0.5).mean()),
            float(prot[m].mean()))


def _ok(fit: tuple[float, float, float], role: str = "wordmark") -> bool:
    draw, occ, prot = fit
    return draw >= DRAWABLE_MIN and occ <= occlude_max(role) and prot <= 0.02


def seam_margin(fld: CompositionField, p: TextPose) -> float:
    """포즈 상자에서 **프레임 양 끝(= 앞·뒤 이음새)**까지의 여유 (프레임 폭의 몫).

    옆면 프레임은 차체 밴드 통째라 그 u 양 끝이 곧 앞·뒤 패널 경계다. 음수면
    이미 넘어간 것이다.
    """
    r = math.radians(p.rot)
    c, s = math.cos(r), math.sin(r)
    hw, hh = p.w / 2, p.h / 2
    xs = [p.x + dx * c - dy * s
          for dx, dy in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh))]
    fx0, fx1 = fld.frame_box[0], fld.frame_box[2]
    return min(min(xs) - fx0, fx1 - max(xs)) / max(1.0, fx1 - fx0)


# 글자가 이음새에서 지켜야 하는 여유 (프레임 폭의 몫).
#
# 글자의 이음새 정책은 **피하기**다 (`compose.seams.ROLE_POLICY`): 이름이
# 패널 경계를 넘으면 그 자리에서 잘리고, 안 잘려도 모서리의 곡률에서 글자가
# 휘어 안 읽힌다. 사람이 만든 이타샤의 워드마크는 예외 없이 한 패널 안에 선다.
#
# 실측 T0(33판 · 글자 그룹 58벌): 여유의 중앙값은 프레임 폭의 0.259지만 **둘이
# 0**이었다 — 그 둘이 이음새에 그대로 걸려 있었다. 0.02면 그 둘만 걸린다.
SEAM_PAD = 0.02


def _seam_ok(fld: CompositionField, p: TextPose) -> bool:
    return seam_margin(fld, p) >= SEAM_PAD


def _settle(fld: CompositionField, p: TextPose, axis: tuple[float, float],
            away: float, min_h: float, avoid: np.ndarray | None = None) -> TextPose | None:
    """포즈를 규칙에 맞게 **밀고, 그래도 안 되면 줄인다**.

    순서가 중요하다: 크기를 먼저 줄이면 휠아치 하나 때문에 워드마크가 로커 글자
    크기로 오그라든다 (실측: 65 → 24유닛). 먼저 축을 따라 인물 반대쪽으로 밀고,
    다음 위아래로 비켜 보고, 그래도 안 들면 그때 줄인다.
    """
    fy0, fy1 = fld.frame_box[1], fld.frame_box[3]
    for _ in range(8):
        if _ok(pose_fit(fld, p, avoid), p.role) and _seam_ok(fld, p):
            return p
        best, bs = None, -1.0
        # 밀어 보기 — 축 방향(인물 반대쪽) 몇 단, 수직 방향 몇 단
        for k in (1, 2, 3, 4):
            for dx, dy in ((away * axis[0] * 0.12 * p.w * k, away * axis[1] * 0.12 * p.w * k),
                           (0.0, 0.10 * (fy1 - fy0) * k), (0.0, -0.10 * (fy1 - fy0) * k)):
                q = TextPose(**{**p.__dict__, "x": p.x + dx, "y": p.y + dy})
                if not (fy0 + 0.4 * q.h <= q.y <= fy1 - 0.4 * q.h):
                    continue
                draw, occ, prot = pose_fit(fld, q, avoid)
                sm = seam_margin(fld, q)
                if _ok((draw, occ, prot), p.role) and sm >= SEAM_PAD:
                    return q
                # 이음새를 넘은 만큼은 **못 그리는 자리와 같은 무게**로 벌한다 —
                # 넘긴 몫은 패널 경계에서 잘린다
                sc = draw - 0.5 * occ - prot + min(0.0, sm - SEAM_PAD)
                if sc > bs:
                    best, bs = q, sc
        if p.height * 0.85 < min_h:
            return None
        base = best if best is not None else p
        p = TextPose(**{**base.__dict__, "height": base.height * 0.85})
    return p if (_ok(pose_fit(fld, p, avoid), p.role) and _seam_ok(fld, p)) else None


def _run_length(fld: CompositionField, x: float, y: float, axis: tuple[float, float],
                sign: float) -> float:
    """(x, y)에서 축 방향으로 그려지는 자리가 이어지는 길이 (프레임 유닛).

    휠아치 같은 구멍(`HOLE_MIN` 이상)이 나오면 **그 앞에서 끝난다** — 구멍을
    건너뛰면 워드마크가 아치 위를 지나 잘린다 (실측: 26유닛 글자가 뒷아치에 걸렸다)."""
    g = fld.grid
    d = 0.0
    step = g.cell
    hole = 0.0
    while d < 1200.0:
        px, py = x + sign * axis[0] * d, y + sign * axis[1] * d
        if not (fld.frame_box[0] <= px <= fld.frame_box[2]):
            break
        if g.at(fld.drawable, px, py) < 0.5:
            hole += step
            if hole >= HOLE_MIN:
                return max(0.0, d - hole)
        else:
            hole = 0.0
        d += step
    return d


def wordmark_poses(fld: CompositionField, text: str, aspect: float, hratio: float,
                   rocker: bool) -> list[TextPose]:
    """인물 곁 **사선 워드마크** 후보 — 흐름 쪽 우선, 안 되면 반대쪽(여백 활용)."""
    ax = slab_axis(fld)
    ang = math.degrees(math.atan2(ax[1], ax[0]))
    ch, cw = fld.char_h, fld.char_w
    vcx, vcy = fld.visual_center
    band = fld.frame_box[3] - fld.frame_box[1]
    out: list[TextPose] = []
    flow_sign = 1.0 if fld.flow[0] >= 0 else -1.0
    for sign in (flow_sign, -flow_sign):
        # 인물 상자 안쪽 1/3에서 휠아치까지가 글자 줄의 자리 — 인물 상자 끝에서
        # 시작하면 문짝 하나가 자리라 이름이 20유닛으로 오그라든다 (실측: 실비아
        # 옆면, 상자 끝 → 아치 110유닛). 인물 실루엣이 덮는 몫은 `_settle`이 잰다
        # (상자는 실루엣보다 넓어 끝 1/3은 대개 빈 도색이다).
        edge = fld.person_box[2] if sign > 0 else fld.person_box[0]
        start = edge - sign * 0.33 * cw
        run = _run_length(fld, start, vcy, ax, sign) - 0.04 * (fld.frame_box[2] - fld.frame_box[0])
        if run < 0.5 * cw:
            continue
        # 상자 높이의 상한 → 대문자 높이
        bh = min(WORDMARK_H_MAX * ch * hratio, 0.6 * band, run / max(0.5, aspect))
        h = max(bh / hratio, 0.12 * band)
        w = h * hratio * aspect
        cx = start + sign * (0.5 * w + 0.02 * run)
        cy = vcy + ax[1] * (cx - vcx) / max(1e-6, ax[0]) * 0.6 - 0.05 * ch
        cy = max(fld.frame_box[1] + 0.5 * bh, min(fld.frame_box[3] - 0.5 * bh, cy))
        p = TextPose(role="wordmark", text=text, x=cx, y=cy, rot=ang, height=h, aspect=aspect,
                     hratio=hratio)
        p = _settle(fld, p, ax, sign, 0.12 * band)
        if p is not None:
            out.append(p)
    return out


def behind_pose(fld: CompositionField, text: str, aspect: float, hratio: float
                ) -> TextPose | None:
    """인물 **뒤**를 가로지르는 큰 워드마크 — 휠아치 사이 도색면을 다 쓴다.

    한 줄 이름의 높이는 자리 길이가 정하므로 인물 곁 자리(`wordmark_poses`)로는
    30유닛을 못 넘는다. 사람이 만든 이타샤의 이름은 인물 뒤에 크게 깔린다
    (RIN SHIBUYA) — 인물이 35%까지 덮어도 읽힌다. 얼굴 보호 구역과 덮임 몫은
    `_settle`이 지킨다 (안 되면 밀고, 그래도 안 되면 줄인다)."""
    ax = slab_axis(fld)
    ang = math.degrees(math.atan2(ax[1], ax[0]))
    ch = fld.char_h
    vcx, vcy = fld.visual_center
    band = fld.frame_box[3] - fld.frame_box[1]
    px, py = (fld.person_box[0] + fld.person_box[2]) / 2, vcy
    fwd = _run_length(fld, px, py, ax, 1.0)
    back = _run_length(fld, px, py, ax, -1.0)
    run = 0.92 * (fwd + back)
    if run < 0.8 * fld.char_w:
        return None
    bh = min(WORDMARK_H_MAX * ch * hratio, 0.6 * band, run / max(0.5, aspect))
    h = bh / hratio
    if h < 0.14 * band:
        return None
    # 자리 가운데에 두되 흐름 쪽으로 조금 — 인물이 덮는 몫이 한쪽으로 몰리지 않게
    cx = px + ax[0] * (fwd - back) / 2
    cy = py + ax[1] * (fwd - back) / 2 - 0.04 * ch
    cy = max(fld.frame_box[1] + 0.5 * bh, min(fld.frame_box[3] - 0.5 * bh, cy))
    flow_sign = 1.0 if fld.flow[0] >= 0 else -1.0
    p = TextPose(role="behind", text=text, x=cx, y=cy, rot=ang, height=h, aspect=aspect,
                 hratio=hratio)
    return _settle(fld, p, ax, flow_sign, 0.14 * band)


def rocker_pose(fld: CompositionField, text: str, aspect: float, hratio: float
                ) -> TextPose | None:
    """로커 띠 바로 위의 얇은 글자 — 흐름 쪽에 붙는다 (스폰서 행 문법)."""
    fx0, fy0, fx1, fy1 = fld.frame_box
    band = fy1 - fy0
    h = ROCKER_TEXT_H * band
    y = fy0 + ROCKER_FRAC * band + 0.5 * h + 0.06 * band
    flow_sign = 1.0 if fld.flow[0] >= 0 else -1.0
    edge = fld.person_box[2] if flow_sign > 0 else fld.person_box[0]
    room = (fx1 - edge) if flow_sign > 0 else (edge - fx0)
    w = h * hratio * aspect
    if w > 0.9 * room:
        h = max(0.10 * band, 0.9 * room / max(0.5, aspect) / hratio)
        w = h * hratio * aspect
    x = edge + flow_sign * (0.08 * fld.char_w + 0.5 * w)
    p = TextPose(role="rocker", text=text, x=x, y=y, rot=0.0, height=h, aspect=aspect,
                 hratio=hratio)
    return _settle(fld, p, (1.0, 0.0), flow_sign, 0.09 * band)


def signature_pose(fld: CompositionField, text: str, aspect: float, hratio: float
                   ) -> TextPose | None:
    """리어 쿼터(흐름 끝)의 작은 사인."""
    fx0, fy0, fx1, fy1 = fld.frame_box
    band = fy1 - fy0
    ax = slab_axis(fld)
    ang = math.degrees(math.atan2(ax[1], ax[0]))
    flow_sign = 1.0 if fld.flow[0] >= 0 else -1.0
    h = SIGNATURE_H * band / hratio
    w = h * hratio * aspect
    x = (fx1 if flow_sign > 0 else fx0) - flow_sign * (0.5 * w + 0.05 * (fx1 - fx0))
    y = fld.visual_center[1] + 0.08 * fld.char_h
    p = TextPose(role="signature", text=text, x=x, y=y, rot=ang, height=h, aspect=aspect,
                 hratio=hratio)
    return _settle(fld, p, ax, -flow_sign, 0.10 * band)


def sub_pose(fld: CompositionField, main: TextPose, text: str, aspect: float, hratio: float
             ) -> TextPose | None:
    """메인 밑에 붙는 서브 — 같은 각, 높이는 메인의 `SUB_RATIO`."""
    h = SUB_RATIO * main.height
    r = math.radians(main.rot)
    # 축의 수직(아래쪽)으로 메인 반높이 + 서브 반높이 + 틈
    off = 0.5 * main.h + 0.5 * h * hratio + 0.18 * main.height
    nx, ny = math.sin(r), -math.cos(r)
    p = TextPose(role="sub", text=text, x=main.x + nx * off, y=main.y + ny * off,
                 rot=main.rot, height=h, aspect=aspect, hratio=hratio)
    ax = (math.cos(r), math.sin(r))
    sign = 1.0 if (main.x - fld.visual_center[0]) * ax[0] >= 0 else -1.0
    # 메인 상자(+여유)는 서브가 못 들어가는 자리다
    pad = TextPose(**{**main.__dict__, "height": main.height * 1.15,
                      "aspect": main.aspect * 1.08})
    return _settle(fld, p, ax, sign, 0.4 * h, avoid=pose_mask(fld, pad))


def lockups(main: str) -> list[str]:
    """이름의 줄 나눔 후보 — 원문 그대로, 그리고 (한 줄에 공백이 있으면) 가장 고르게
    갈리는 공백에서 두 줄로. 글자는 안 바꾼다 (공백 하나가 줄바꿈이 될 뿐)."""
    out = [main]
    if "\n" in main:
        return out
    words = main.split(" ")
    if len(words) < 2:
        return out
    best, bd = None, None
    for k in range(1, len(words)):
        a, b = " ".join(words[:k]), " ".join(words[k:])
        d = abs(len(a) - len(b))
        if bd is None or d < bd:
            best, bd = a + "\n" + b, d
    if best is not None:
        out.append(best)
    return out


ROLES = ("wordmark", "behind", "rocker", "signature")


def layout_sets(fld: CompositionField, main: str, sub: str | None,
                box_main: tuple[float, float], box_sub: tuple[float, float], rocker: bool,
                roles: tuple[str, ...] = ROLES) -> list[list[TextPose]]:
    """포즈 묶음 후보들 — 각 묶음은 메인 하나(+서브 하나). `box_*`는 (잉크 상자 w/h,
    상자 높이/대문자 높이). `main`은 이미 줄이 갈린 문자열일 수 있다 (`lockups`) —
    로커 글자는 한 줄일 때만."""
    sets: list[list[TextPose]] = []
    mains: list[TextPose] = []
    am, hm = box_main
    if "wordmark" in roles:
        mains += wordmark_poses(fld, main, am, hm, rocker)
    if "behind" in roles:
        p = behind_pose(fld, main, am, hm)
        if p is not None:
            mains.append(p)
    if "rocker" in roles and "\n" not in main:
        p = rocker_pose(fld, main, am, hm)
        if p is not None:
            mains.append(p)
    if "signature" in roles:
        p = signature_pose(fld, main, am, hm)
        if p is not None:
            mains.append(p)
    for m in mains:
        group = [m]
        if sub:
            s = sub_pose(fld, m, sub, box_sub[0], box_sub[1])
            if s is not None:
                group.append(s)
        sets.append(group)
    return sets
