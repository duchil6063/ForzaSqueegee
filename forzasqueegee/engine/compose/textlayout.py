"""텍스트 배치 — 워드마크가 **캐릭터를 받치는 자리**를 필드에서 찾는다.

글자는 산포 모티프처럼 흩지 않는다. 사람이 만든 이타샤의 이름은 늘 정해진
자리에 선다 (레퍼런스 실측): 인물 뒤 사선 판을 따라가는 대형 워드마크(RIN
SHIBUYA), 로커 위 얇은 레이싱 글자(ARIS의 스폰서 행), 리어 쿼터의 작은 사인
(EVELYNE). 이 모듈은 그 세 문법을 필드 좌표(꾸밈 프레임)의 **포즈**로 낸다 —
어느 포즈가 이기는지는 점수가 정한다 (`textscore`).

규칙:
- 보호 구역(얼굴)은 **밑에 깔려도** 안 건드린다 — 글자가 얼굴 뒤에서 비치면 얼굴이 어수선해진다.
- 인물이 글자를 덮는 몫은 35% 아래 (뒤에 깔린 워드마크가 인물에 반쯤 가려지는 것은 문법이지만 그 이상이면 안 읽힌다).
- 그려지는 자리(도색 마스크) 85% 이상.
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


# 인물이 글자를 덮어도 되는 몫 · 그려지는 자리의 하한
OCCLUDE_MAX = 0.35


DRAWABLE_MIN = 0.82


@dataclass
class TextPose:
    role: str                  # wordmark · rocker · signature · sub
    text: str
    x: float
    y: float
    rot: float                 # 도
    height: float              # 대문자 높이 (프레임 유닛)
    aspect: float              # 잉크 상자 w/h (글자 블록)
    on_bed: bool = False

    @property
    def w(self) -> float:
        return self.height * self.aspect

    @property
    def h(self) -> float:
        return self.height

    def mirrored(self) -> "TextPose":
        """반대편 옆면의 포즈 — 자리는 거울, 글자는 그대로 (읽히는 방향)."""
        return TextPose(role=self.role, text=self.text, x=-self.x, y=self.y,
                        rot=(-self.rot) % 360.0, height=self.height,
                        aspect=self.aspect, on_bed=self.on_bed)


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

    `avoid`(다른 글자 상자 마스크)를 주면 보호구역처럼 센다 — 서브가 메인 위에
    올라앉는 것을 막는다 (실측: 휠아치를 피하려다 메인 안으로 들어갔다)."""
    m = pose_mask(fld, p)
    n = int(m.sum())
    if n == 0:
        return 0.0, 1.0, 1.0
    prot = fld.protected > 0.5
    if avoid is not None:
        prot = prot | avoid
    return (float((fld.drawable[m] > 0.5).mean()),
            float((fld.char[m] > 0.5).mean()),
            float(prot[m].mean()))


def _ok(fit: tuple[float, float, float]) -> bool:
    draw, occ, prot = fit
    return draw >= DRAWABLE_MIN and occ <= OCCLUDE_MAX and prot <= 0.02


def _settle(fld: CompositionField, p: TextPose, axis: tuple[float, float],
            away: float, min_h: float, avoid: np.ndarray | None = None) -> TextPose | None:
    """포즈를 규칙에 맞게 **밀고, 그래도 안 되면 줄인다**.

    순서가 중요하다: 크기를 먼저 줄이면 휠아치 하나 때문에 워드마크가 로커 글자
    크기로 오그라든다 (실측: 65 → 24유닛). 먼저 축을 따라 인물 반대쪽으로 밀고,
    다음 위아래로 비켜 보고, 그래도 안 들면 그때 줄인다.
    """
    fy0, fy1 = fld.frame_box[1], fld.frame_box[3]
    for _ in range(8):
        if _ok(pose_fit(fld, p, avoid)):
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
                if _ok((draw, occ, prot)):
                    return q
                sc = draw - 0.5 * occ - prot
                if sc > bs:
                    best, bs = q, sc
        if p.height * 0.85 < min_h:
            return None
        base = best if best is not None else p
        p = TextPose(**{**base.__dict__, "height": base.height * 0.85})
    return p if _ok(pose_fit(fld, p, avoid)) else None


def _run_length(fld: CompositionField, x: float, y: float, axis: tuple[float, float],
                sign: float) -> float:
    """(x, y)에서 축 방향으로 그려지는 자리가 이어지는 길이 (프레임 유닛)."""
    g = fld.grid
    d = 0.0
    step = g.cell
    while d < 1200.0:
        px, py = x + sign * axis[0] * d, y + sign * axis[1] * d
        if g.at(fld.drawable, px, py) < 0.5:
            # 휠아치 같은 구멍은 건너뛰되, 프레임 밖이면 끝이다
            if not (fld.frame_box[0] <= px <= fld.frame_box[2]):
                break
        d += step
    return d


def wordmark_poses(fld: CompositionField, text: str, aspect: float,
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
        # 인물 상자 끝 + 여유에서 프레임 끝까지가 글자 줄의 자리
        edge = fld.person_box[2] if sign > 0 else fld.person_box[0]
        start = edge + sign * 0.10 * cw
        run = _run_length(fld, start, vcy, ax, sign) - 0.04 * (fld.frame_box[2] - fld.frame_box[0])
        if run < 0.35 * cw:
            continue
        h = min(WORDMARK_H_MAX * ch, 0.42 * band, run / max(0.5, aspect))
        h = max(h, 0.12 * band)
        w = h * aspect
        cx = start + sign * (0.5 * w + 0.02 * run)
        cy = vcy + ax[1] * (cx - vcx) / max(1e-6, ax[0]) * 0.6 - 0.05 * ch
        cy = max(fld.frame_box[1] + 0.5 * h, min(fld.frame_box[3] - 0.5 * h, cy))
        p = TextPose(role="wordmark", text=text, x=cx, y=cy, rot=ang, height=h, aspect=aspect)
        p = _settle(fld, p, ax, sign, 0.12 * band)
        if p is not None:
            out.append(p)
    return out


def rocker_pose(fld: CompositionField, text: str, aspect: float) -> TextPose | None:
    """로커 띠 바로 위의 얇은 글자 — 흐름 쪽에 붙는다 (스폰서 행 문법)."""
    fx0, fy0, fx1, fy1 = fld.frame_box
    band = fy1 - fy0
    h = ROCKER_TEXT_H * band
    y = fy0 + ROCKER_FRAC * band + 0.5 * h + 0.06 * band
    flow_sign = 1.0 if fld.flow[0] >= 0 else -1.0
    edge = fld.person_box[2] if flow_sign > 0 else fld.person_box[0]
    room = (fx1 - edge) if flow_sign > 0 else (edge - fx0)
    w = h * aspect
    if w > 0.9 * room:
        h = max(0.10 * band, 0.9 * room / max(0.5, aspect))
        w = h * aspect
    x = edge + flow_sign * (0.08 * fld.char_w + 0.5 * w)
    p = TextPose(role="rocker", text=text, x=x, y=y, rot=0.0, height=h, aspect=aspect)
    return _settle(fld, p, (1.0, 0.0), flow_sign, 0.09 * band)


def signature_pose(fld: CompositionField, text: str, aspect: float) -> TextPose | None:
    """리어 쿼터(흐름 끝)의 작은 사인."""
    fx0, fy0, fx1, fy1 = fld.frame_box
    band = fy1 - fy0
    ax = slab_axis(fld)
    ang = math.degrees(math.atan2(ax[1], ax[0]))
    flow_sign = 1.0 if fld.flow[0] >= 0 else -1.0
    h = SIGNATURE_H * band
    w = h * aspect
    x = (fx1 if flow_sign > 0 else fx0) - flow_sign * (0.5 * w + 0.05 * (fx1 - fx0))
    y = fld.visual_center[1] + 0.08 * fld.char_h
    p = TextPose(role="signature", text=text, x=x, y=y, rot=ang, height=h, aspect=aspect)
    return _settle(fld, p, ax, -flow_sign, 0.10 * band)


def sub_pose(fld: CompositionField, main: TextPose, text: str, aspect: float
             ) -> TextPose | None:
    """메인 밑에 붙는 서브 — 같은 각, 높이는 메인의 `SUB_RATIO`."""
    h = SUB_RATIO * main.height
    r = math.radians(main.rot)
    # 축의 수직(아래쪽)으로 메인 반높이 + 서브 반높이 + 틈
    off = 0.5 * main.h + 0.5 * h + 0.18 * main.height
    nx, ny = math.sin(r), -math.cos(r)
    p = TextPose(role="sub", text=text, x=main.x + nx * off, y=main.y + ny * off,
                 rot=main.rot, height=h, aspect=aspect)
    ax = (math.cos(r), math.sin(r))
    sign = 1.0 if (main.x - fld.visual_center[0]) * ax[0] >= 0 else -1.0
    # 메인 상자(+여유)는 서브가 못 들어가는 자리다
    pad = TextPose(**{**main.__dict__, "height": main.height * 1.15,
                      "aspect": main.aspect * 1.08})
    return _settle(fld, p, ax, sign, 0.4 * h, avoid=pose_mask(fld, pad))


def layout_sets(fld: CompositionField, main: str, sub: str | None, aspect_main: float,
                aspect_sub: float, rocker: bool, roles: tuple[str, ...] = ("wordmark", "rocker", "signature")
                ) -> list[list[TextPose]]:
    """포즈 묶음 후보들 — 각 묶음은 메인 하나(+서브 하나)."""
    sets: list[list[TextPose]] = []
    mains: list[TextPose] = []
    if "wordmark" in roles:
        mains += wordmark_poses(fld, main, aspect_main, rocker)
    if "rocker" in roles:
        p = rocker_pose(fld, main, aspect_main)
        if p is not None:
            mains.append(p)
    if "signature" in roles:
        p = signature_pose(fld, main, aspect_main)
        if p is not None:
            mains.append(p)
    for m in mains:
        group = [m]
        if sub:
            s = sub_pose(fld, m, sub, aspect_sub)
            if s is not None:
                group.append(s)
        sets.append(group)
    return sets
