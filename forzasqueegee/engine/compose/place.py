"""배치 — 도안을 면 어디에 얼마로 앉히나 (자리 수학과 그 밑감)."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from ...game import hull as ghull, seam as gseam, surface as gsurf
from ...i18n import msg
from ..catalog import Catalog
from ..model import LayerPlan
from .boxes import DEFAULT_GROUP_UNIT
from .look import Look, layer_points, person_ink, rot_ink, rot_ink_box


# 인물이 면 높이의 몇 몫까지 차지하나. 1.0을 쓰면 머리가 유리·루프로 올라가고
# 발이 사이드실 밑으로 내려간다 — 실차 랩과 이타샤 모두 위아래에 여백을 둔다.
BODY_FILL = 0.94


# ---- 인물은 **차체 밴드를 가로로 채운다** (2026-08-20 레퍼런스 픽셀 실측) ----
# 상자는 기하로 곧장 잡는다 (커버리지 내접 상자는 휠아치를 피하려 위로 올라가
# 인물이 뜨고 작아진다 — 실차 캡처). 자는 `game.seam.person_budget`이다: 폭 =
# 문짝(휠아치 사이)의 0.75, 높이 = 차체 밴드의 1.10 (레퍼런스 실측 중앙값).
# 그 예산에서 인물이 가장 커지는 각도를 `person_pose`가 푼다. 높이만 예산으로
# 잡으면 폭이 종횡비에 딸려 와 세로 도안이 가는 세로 띠로 서고 인물의 상당
# 부분이 유리로 올라간다 (레퍼런스가 벨트라인을 넘기는 것은 0~29%, 중앙 9%이고
# 그것도 머리카락 끝·팔·다리다). 발은 사이드실이고 다리가 휠아치를 가로지르며
# 잘리는 것은 허용이다.
#
# **벨트라인 위는 옆면이 아니라 도어 유리 면이 그린다** — 설치 마스크는 옆면에
# 그린하우스를 흰 판으로 갖고 있지만 게임은 거기에 안 그린다 (프로브 15면 대조).
# 그래서 인물은 벨트라인을 안 넘게 앉힌다 (`game.seam.PERSON_BAND_FILL`) —
# 넘긴 몫은 아무 데도 안 가고 그 자리에서 잘린다. 유리까지 쓰려면 편집기에서
# 도안을 벨트라인으로 **가르고** 위쪽 반을 유리 면에 따로 올린다.
# ---- 세로 도안 눕히기 (2026-08-20 레퍼런스 픽셀 실측) ----
# 세로로 긴 전신 도안은 세우지 말고 **눕힌다**. 레퍼런스 옆면 여섯 장에서 인물
# 장축이 수직에서 벗어난 각을 격자 오버레이로 쟀다: EVELYNE 43° · KOTONE 60° ·
# ARIS 65° · 마린 ~70° · RIN SHIBUYA 79° · Evo IX ~80° (중앙값 67°). 세운 셋은
# 전부 **버스트 크롭**이고 10~15°다 — 작은 전신 인물이 서 있는 장면은 열 장
# 어디에도 없다. 각도는 상수로 안 준다: `person_pose`가 면 예산 안에서 인물이
# 가장 커지는 각을 푼다. 여기 있는 것은 그 탐색의 **울타리**다.
LIE_MAX = 80.0             # 실측한 누운 각도의 최대 (RIN 79° · Evo IX ~80°)


LIE_GAIN_MIN = 1.12        # 이만큼도 안 커지면 안 눕힌다 (돌린 값을 못 한다)


LIE_TIE = 0.01             # 배율 동률 판정 — 같으면 덜 돌린 각을 남긴다


# **머리는 차 뒤쪽**으로 눕힌다.
#
# 레퍼런스는 이 축으로 안 갈린다 — 앞 4(RIN SHIBUYA·ARIS·KOTONE·Evo IX) 대 뒤
# 3(마린·EVELYNE·Fate)이라 실측이 방향을 안 정해 준다. 그래서 이 값은 **레버**다:
# 윗면 후드 인물도 머리가 윈드실드 쪽(차 뒤)이라 (`_hood_place`) 같은 방향으로
# 두면 옆·윗면이 한 방향으로 읽힌다. 어느 옆구리를 바닥에 두나는 별개 레버다
# (`--flip`).
#
# 부호는 `side_left` 기준이다 (오른쪽은 미러라 반대). 게임 회전 r에서 캔버스
# +y(머리)는 면 유닛 (−sin r, cos r)이므로, side_left(+u가 뒤)에서 머리를 +u로
# 보내려면 r이 음수여야 한다.
LIE_HEAD_REAR = True


# 뼈대를 못 세운 면(폴백)과 후드가 쓰는 기울임 자. 옆면은 위 `person_pose`다.
TILT_ASPECT = 0.55         # 이 종횡비(w/h)보다 세로면 기울이기 시작


TILT_FULL = 0.30           # 이 종횡비에서 최대 기울기에 도달


TILT_MAX = 24.0            # 최대 기울기 (도) — 업계 관찰 범위 5~20°대 + 여유


# 인물 자리의 좌우 비율. 문·리어펜더에 오게 **뒤로** 살짝 민다.
# 어느 쪽이 뒤인가는 면마다 다르다 (2026-08-17 캡처 확인): `side_left` 카메라는
# 차가 왼쪽을 보므로 화면 오른쪽(+u)이 뒤고, `side_right`는 그 거울이라 -u가 뒤다.
# 그래서 오른쪽 면은 `1 - BODY_BIAS`를 쓴다.
BODY_BIAS = 0.56


def person_tilt(lk: Look) -> float:
    """세로 도안의 측면 기울기(도, 크기만). 0이면 세운다.

    종횡비가 TILT_ASPECT보다 세로면 선형으로 키워 TILT_FULL에서 TILT_MAX에
    닿는다. 정방·가로 도안은 기울이지 않는다 (영상 실측: 버스트는 수직이다).

    옆면 뼈대가 서는 면은 이 자를 안 쓴다 — 거기서는 `person_pose`가 면 예산
    에서 각도를 **푼다**. 이 자는 뼈대를 못 세운 면(폴백)과 후드가 쓴다.
    """
    if lk.aspect >= TILT_ASPECT:
        return 0.0
    t = (TILT_ASPECT - lk.aspect) / max(1e-6, TILT_ASPECT - TILT_FULL)
    return round(min(1.0, t) * TILT_MAX, 1)


def person_pose(lk: Look, rigs: "dict[str, SideRig]") -> tuple[float, str]:
    """옆면 인물의 **눕히는 각도**를 면 예산에서 푼다. 되돌림: (각도, 설명).

    레퍼런스 실측이 시키는 것은 하나다 — **차체 밴드를 가로로 채워라.** 옆면은
    가로로 긴 판(줄리아 실측: 문짝 455유닛 × 차체 밴드 156유닛 = 2.9:1)인데
    전신 도안은 세로로 길다. 세우면 예산 중 **높이만** 다 쓰고 폭은 남는다:
    줄리아 × 미쿠(비 0.61)에서 인물 폭이 문짝의 0.35뿐이었고(세로일수록 더 작아
    celtest-01은 0.20) — 레퍼런스는 0.62~0.84다 — 대신 인물의 **40%**가
    벨트라인 위로 올라가 잘렸다.

    그래서 각도를 상수로 안 주고 **예산 안에서 가장 커지는 각**을 찾는다:

        s(θ) = min(폭예산 / 잉크폭(θ), 높이예산 / 잉크높이(θ))

    잉크 크기는 껍질 실측이다 (`rot_ink`) — 사각형 공식으로 재면 회전한 그림을
    과대평가해 눕힐수록 손해가 나서 최적이 0°로 붙는다.

    두 가지를 안 한다:

    - **버스트는 안 눕힌다** (`kind != "tall"`). 레퍼런스의 세운 인물 셋은 전부
      버스트 크롭이고(Cygames 86 ~10° · 수이세이 ~15° · 치하야 수직), 정방에
      가까운 그림은 돌려도 기울어진 사진으로 읽힌다.
    - **90°까지는 안 간다** (`LIE_MAX`). 실측한 누운 각도의 최대가 80°다
      (RIN SHIBUYA 79° · Evo IX ~80°) — 정확히 직각이면 '누운 인물'이 아니라
      '돌린 그림'이 된다.

    좌우는 **한 각도를 나눠 쓴다** (부호만 반대) — 같은 도안 파일 하나가 두 면에
    서야 준비가 공짜이기 때문이다. 그래서 예산은 두 면 중 **빡빡한 쪽**이다.
    """
    if lk.kind != "tall" or not rigs:
        return 0.0, ""
    wb = hb = None
    for r in rigs.values():
        w, h = gseam.person_budget(r.body, r.geom)
        wb = w if wb is None else min(wb, w)
        hb = h if hb is None else min(hb, h)
    if not wb or not hb:
        return 0.0, ""

    # 탐색은 **부호를 넣어** 돈다 — 잉크 껍질은 좌우 대칭이 아니라 +θ와 −θ의
    # 회전 상자가 다르다. 부호를 나중에 뒤집으면 그때 잰 크기가 안 맞는다.
    sgn = -1.0 if LIE_HEAD_REAR else 1.0

    def _s(deg: float) -> float:
        iw, ih = person_ink(lk, deg)
        return min(wb / max(1e-6, iw), hb / max(1e-6, ih))

    up = _s(0.0)
    best, bs = 0.0, up
    for d in range(1, int(LIE_MAX) + 1):
        s = _s(sgn * d)
        if s > bs * (1.0 + LIE_TIE):           # 동률이면 **덜 돌린 쪽**을 남긴다
            best, bs = sgn * d, s
    if not best or bs < up * LIE_GAIN_MIN:
        return 0.0, ""
    iw, ih = person_ink(lk, best)
    return best, msg("세로 도안(비 {aspect:.2f})을 **{deg:g}° 눕혀** 차체 "
                     "밴드를 가로로 채운다 (머리 = 차 {head}) — 인물이 "
                     "{gain:.2f}배 커진다 (면에서 {w:.0f}×{h:.0f}유닛 "
                     "· 예산 {wb:.0f}×{hb:.0f})",
                     aspect=lk.aspect, deg=abs(best),
                     head=msg("뒤") if LIE_HEAD_REAR else msg("앞"),
                     gain=bs / up, w=iw * bs, h=ih * bs, wb=wb, hb=hb)


def _refit_canvas(plan: LayerPlan, cat: Catalog) -> LayerPlan:
    """캔버스(image_size)를 레이어 실제 범위로 넓힌다 — 렌더·검증이 잘리지 않게."""
    lo = np.array([1e9, 1e9], np.float32)
    hi = np.array([-1e9, -1e9], np.float32)
    for l in plan.layers:
        pts = layer_points(l, cat)
        if len(pts):
            lo = np.minimum(lo, pts.min(axis=0))
            hi = np.maximum(hi, pts.max(axis=0))
    if hi[0] < lo[0]:
        return plan
    upp = plan.units_per_px
    half = np.maximum(np.abs(lo), np.abs(hi)) / upp * 1.02
    w = int(max(plan.image_size[0], math.ceil(2 * half[0])))
    h = int(max(plan.image_size[1], math.ceil(2 * half[1])))
    plan.image_size = (w, h)
    return plan


# ---------- 면 배치 계산 ----------
@dataclass
class Place:
    surface: str
    plan: Path
    x: float
    y: float
    scale: float
    rot: float = 0.0
    mirror: bool = False
    why: str = ""
    # 노린 면 유닛 상자 — 배치가 화면으로 스스로 맞추는 목표다 (`auto.itasha.autofit`)
    target: tuple[float, float, float, float] | None = None


@dataclass
class ManualPlace:
    """**사람이 편집기에서 직접 앉힌** 도안 한 장 (`engine.fls.studio`).

    수치는 자동 배치가 내는 것과 **같은 좌표계**다 — 면 유닛 이동 `x·y`, 그룹
    균등 스케일 `scale`(캔버스 1유닛 = `scale × group_unit` 면 유닛), 표시 회전
    `rot`, 좌우반전 `mirror`. 그래서 사람이 앉힌 자리도 자동 배치와 똑같이
    꾸밈·글자의 기준이 된다 (`build(manual=...)`).

    **면을 넘긴 몫은 그 자리에서 잘린다** — 게임이 면 도색 상자 밖을 안 그린다.
    이웃 면으로 이어 올리고 싶으면 편집기에서 도안을 이음선으로 **가르고**
    (KFPS·FLS의 [선으로 가르기]) 한쪽을 그 면에 올린다.
    """

    plan: Path
    surface: str
    x: float = 0.0
    y: float = 0.0
    scale: float = 0.25
    rot: float = 0.0
    mirror: bool = False
    # **역할표** (`compose.cast`) — 이 덩어리가 차에서 무엇인가. `hero`가 옆면
    # 구성의 앵커이고, `support`는 사람이 놓은 면에 그대로 두며(그 면의 변주는
    # 안 짓는다), `logo`·`text`는 미러하지 않고, `pinned`는 꾸밈이 안 건드린다
    # (면 밖 자르기도 안 한다). 자동 경로의 배치는 전부 `hero`다.
    role: str = "hero"
    no_mirror: bool = False
    pinned: bool = False

    def key(self) -> str:
        return str(Path(self.plan).resolve())

    @property
    def anchors(self) -> bool:
        """옆면 설계의 뿌리가 될 수 있나 — 그림(주역·보조)만. 로고·글자·그대로는
        구도의 재료이지 뿌리가 아니다."""
        return self.role in ("hero", "support")


def manual_box(lk: Look, mp: ManualPlace,
               group_unit: float) -> tuple[float, float, float, float]:
    """사람이 앉힌 도안이 면에서 **실제로 덮는 상자** (면 유닛).

    자동 배치의 인물 상자와 같은 자다 — 회전 잉크 껍질(`rot_ink_box`)로 잰다.
    사각형 공식으로 재면 눕힌 도안의 상자가 실제보다 커져(80°에서 1.5배) 베드가
    있지도 않은 그림 둘레까지 부풀고 모티프도 그만큼 커진다.
    """
    s = mp.scale * group_unit
    ib = rot_ink_box(lk, mp.rot, mp.mirror)
    return (mp.x + s * ib[0], mp.y + s * ib[1],
            mp.x + s * ib[2], mp.y + s * ib[3])


def place_xf(mp: ManualPlace, group_unit: float) -> tuple[np.ndarray, np.ndarray]:
    """배치 하나의 **표시 변환** — 캔버스 점 p가 앉는 면 유닛은 `L @ p + t`다.

    규약은 `engine.preview._compose_group`·`manual_box`와 같다:
    이동 ∘ R(rot) ∘ (미러면 수평뒤집기) ∘ 균등 스케일.
    """
    s = mp.scale * group_unit
    th = math.radians(mp.rot)
    c, sn = math.cos(th), math.sin(th)
    M = np.diag([-1.0 if mp.mirror else 1.0, 1.0])
    return s * (np.array([[c, -sn], [sn, c]]) @ M), np.array([mp.x, mp.y], float)


def drawable(name: str, maps: dict[str, gsurf.SurfaceMap],
             rigs: dict[str, "SideRig"]) -> gsurf.SurfaceMap | None:
    """그 면이 **실제로 그리는** 지도.

    둘이 겹친다: 옆면은 벨트라인 아래(차체)까지이고(`SideRig.body`), 그 밖의 면은
    지도가 제 것을 들고 있다 (`SurfaceMap.drawn` — 윗면 유리를 잰 차가 그렇다,
    `surfaces_for`). 아무것도 없으면 도색 마스크 그대로다.
    """
    rig = rigs.get(name)
    if rig is not None:
        return rig.body
    sm = maps.get(name)
    return (sm.drawn or sm) if sm is not None else None


# 표시 밝기 — 도색 마스크 밖과 **안 보이는 자리**를 이만큼 어둡게 깐다.
EXPOSED_FLOOR = 0.22          # 아주 안 보이는 자리 (마스크 밖과 같은 값)


EXPOSED_FULL = ghull.HEAD_ON_MIN   # 이 정면도부터는 온전히 밝다


def surface_exposure(name: str, smap: gsurf.SurfaceMap,
                     maps: dict[str, gsurf.SurfaceMap],
                     media: str | None = None) -> np.ndarray | None:
    """이 면이 도안을 **실제로 내보이는 정도** (0~1) — `smap` 마스크 격자 위.

    도색 마스크는 "게임이 여기에 칠한다"까지만 말한다. 그 위에 한 겹이 더
    걸린다: **면이 차에서 달아나는 자리**다. 후드의 코끝, 데크의 꽁무니, 범퍼
    아랫단은 칠해지기는 하지만 도안이 몇 배로 눌려 사람 눈에는 안 그려진 것과
    같다. 껍질이 그것을 잰다 (`game.hull.head_on` — 깊이 지도의 기울기).

    되돌리는 것은 **표시용**이다 — 배치·예산은 지금까지와 같은 마스크로 돈다.
    껍질을 못 지으면 None이고 부르는 쪽은 마스크만 그린다.

    **게임이 아예 안 그리는 자리는 여기 없다** — 그건 그리는 지도가 이미 쥔다
    (`drawable` — 옆면 그린하우스는 벨트라인 아래로, 윗면 앞·뒷유리는 프로브가
    잰 띠로 잘려 있다, `game.seam`). 이 자는 "그려지기는 하는데 눌려서 안
    보이는" 자리만 잰다.

    **메시 껍질이면 한 가지가 더 0이 된다** (`media`를 주고 그 차의 기하 덤프가
    떠 있을 때 — `hull.MeshHull`): 마스크는 칠한다는데 **그 방향에서 보이는
    표면이 아예 없는** 칸이다 (옆면 그린하우스·윗면 유리 — 차체 메시에 없다).
    표시용이라는 뜻은 같다 — 둘 다 "도안이 여기서는 안 보인다"이다.
    """
    h = ghull.of(maps, media)
    return h.head_on(name, smap) if h is not None else None


# 면 밖 판정의 **여유** — 도색 마스크를 면 긴 변의 이 몫만큼 부풀린 뒤 잰다.
# 마스크는 실측이라 가장자리가 한두 유닛 어긋날 수 있고, 걸친 레이어를 버리면
# 이음새에 빈 띠가 생긴다 — 그래서 조금 넉넉히 두고 그 밖만 뺀다.
OFF_CAR_MARGIN = 0.02


def layers_on(plan: LayerPlan, cat: Catalog, L: np.ndarray, t: np.ndarray,
              box: tuple[float, float, float, float], *,
              mask: np.ndarray | None = None,
              margin: float = OFF_CAR_MARGIN) -> list[int]:
    """면에 **닿는** 레이어 색인 — 그 면에서 그려지는 것들이다.

    `mask`(면 도색 마스크, `box` 상자 위의 격자)를 주면 판정은 **차 위에 있나**다:
    레이어 껍질의 볼록 껍데기를 격자에 찍어 `margin`만큼 부풀린 마스크와 겹치는지
    본다. 상자 안이지만 차 밖인 자리(휠아치·그린하우스·실루엣 밖)의 레이어는
    게임이 안 그리면서 면 상한만 잡아먹으므로 여기서 빠진다. 볼록 껍데기라 상자를
    가로지르는 큰 도형(베드·띠)도 놓치지 않는다 — "점 하나가 안에 드나"로 재면
    껍질 점이 전부 밖인 그것을 놓친다.

    `mask`가 없으면 레이어 잉크 상자와 면 상자의 겹침이다. 어느 쪽이든 걸친
    레이어는 남긴다: 면이 알아서 자르므로 남기면 이음새가 이어지고 버리면 빈
    띠가 생긴다.
    """
    keep: list[int] = []
    grid = None
    if mask is not None and mask.size > 1:
        import cv2

        mh, mw = mask.shape
        r = int(round(max(0.0, margin) * max(mw, mh)))
        grid = mask.astype(np.uint8)
        if r > 0:
            grid = cv2.dilate(grid, np.ones((2 * r + 1, 2 * r + 1), np.uint8))
        kx = (mw - 1) / max(1e-6, box[2] - box[0])
        ky = (mh - 1) / max(1e-6, box[3] - box[1])
    for i, l in enumerate(plan.layers):
        pts = layer_points(l, cat)
        if len(pts) == 0:
            continue
        q = pts @ L.T + t
        if not (float(q[:, 0].max()) >= box[0] and float(q[:, 0].min()) <= box[2]
                and float(q[:, 1].max()) >= box[1] and float(q[:, 1].min()) <= box[3]):
            continue
        if grid is None:
            keep.append(i)
            continue
        px = np.stack([(q[:, 0] - box[0]) * kx, (box[3] - q[:, 1]) * ky], 1)
        hull = cv2.convexHull(px.astype(np.float32)).reshape(-1, 2)
        x0, y0 = np.floor(hull.min(0)).astype(int)
        x1, y1 = np.ceil(hull.max(0)).astype(int)
        x0, y0 = max(int(x0), 0), max(int(y0), 0)
        x1, y1 = min(int(x1), mw - 1), min(int(y1), mh - 1)
        if x1 < x0 or y1 < y0:
            continue
        win = grid[y0:y1 + 1, x0:x1 + 1]
        if not win.any():
            continue
        stamp = np.zeros(win.shape, np.uint8)
        cv2.fillConvexPoly(stamp, np.round(hull - [x0, y0]).astype(np.int32), 1)
        if (stamp & win).any():
            keep.append(i)
    return keep


def take_layers(plan: LayerPlan, keep: list[int]) -> LayerPlan:
    """색인만 남긴 **사본** — 순서를 지킨다 (뒤가 위라 순서가 곧 그림이다).

    레이어까지 복사한다: 조각은 원점을 옮겨 앉히므로 원본 레이어를 그대로 물면
    그 도안을 쓰는 다른 면까지 같이 밀린다.
    """
    return LayerPlan(source_image=plan.source_image, image_size=plan.image_size,
                     units_per_px=plan.units_per_px,
                     layers=[replace(plan.layers[i]) for i in keep])


# 얼굴권 — 인물 상자에서 **머리 쪽 몫**이다. 전신 애니 그림의 머리는 키의 1/5쯤
# 이고(7등신 실사보다 크다) 버스트 크롭은 절반이 얼굴이다. 이 몫 안에 부품이
# 들어오면 민다.
FACE_FRAC_TALL = 0.22


FACE_FRAC_BUST = 0.50


# 부품 둘레로 더 비우는 여유 (인물 상자 폭의 몫) — 노브·주유구가 얼굴 **가장자리**에
# 걸쳐도 지침 위반이라 조금 넉넉히 잡는다.
PART_PAD = 0.04


def face_zone(box: tuple[float, float, float, float], lk: Look, tilt: float,
              rear_dir: float) -> tuple[float, float, float, float]:
    """인물 상자 안에서 **얼굴이 있는 자리** (면 유닛).

    머리 방향은 문법이 정한다: 눕힌 인물은 머리가 **차 뒤**(`LIE_HEAD_REAR`)라
    상자의 `rear_dir` 쪽 끝이고, 세운 인물(버스트)은 위쪽이다. 그림 안에서
    얼굴이 정확히 어디인지는 안 재고 **머리 쪽 몫**으로 어림한다 — 지침이 묻는
    것은 "노브가 얼굴에 걸치나"이고 몇 유닛의 정밀도가 필요한 판단이 아니다.
    """
    u0, v0, u1, v1 = box
    frac = FACE_FRAC_BUST if lk.kind != "tall" else FACE_FRAC_TALL
    if abs(tilt) >= 45.0:                           # 누웠다 — 머리가 차 뒤 끝
        w = (u1 - u0) * frac
        return (u1 - w, v0, u1, v1) if rear_dir > 0 else (u0, v0, u0 + w, v1)
    h = (v1 - v0) * frac
    return (u0, v1 - h, u1, v1)


def dodge_parts(box: tuple[float, float, float, float], rig: "SideRig",
                lk: Look, tilt: float) -> tuple[tuple[float, float, float, float], str]:
    """도어 노브·주유구가 **얼굴에 안 겹치게** 인물 상자를 u로 민다.

    업계 지침("도어 노브·주유구가 얼굴에 안 겹치게")이 이 자를
    기다리고 있었다: 도색 마스크에는 그 구멍이 없어서 지금까지 "안 잰다"였고,
    설치 파일 로케이터가 그 자리를 준다 (`game.locators`).

    **크기는 안 건드린다** — 작게 만들면 지침 하나 지키려고 레퍼런스 문법(문짝
    채움 0.75)을 깨는 셈이다. 미는 것도 **문짝 구간 안**이고, 그 안에서 못
    비키면 그냥 둔다 (완벽히 지킬 수 없는 차가 있는 것이 정상이다).
    """
    if not rig.parts:
        return box, ""
    u0, v0, u1, v1 = box
    fz = face_zone(box, lk, tilt, rig.rear_dir)
    pad = PART_PAD * (u1 - u0)
    hits = [(u, lab) for u, v, lab in rig.parts
            if fz[0] - pad <= u <= fz[2] + pad and fz[1] <= v <= fz[3]]
    if not hits:
        return box, ""
    # 밀 수 있는 범위 = 문짝 구간 (없으면 도색 상자)
    door = door_span(rig)
    lo, hi = door if door is not None else (rig.body.paint[0], rig.body.paint[2])
    lo, hi = lo, hi - (u1 - u0)
    if lo > hi:                                    # 인물이 문짝보다 넓다
        return box, ""
    # 후보: 걸린 부품 전부를 얼굴권 **밖**으로 내보내는 최소 이동
    best = None
    # 후보 이동량은 **딱 맞게** 재면 안 된다 — 부품 u가 0.1유닛으로 반올림돼
    # 있어 경계에서 "여전히 걸린다"로 돌아온다 (줄리아 실측: 정확히 194.0에서
    # 되튕겼다). 반 유닛을 더 민다.
    eps = 0.5
    for d in sorted({s for u, _ in hits
                     for s in (u + pad - fz[0] + eps, u - pad - fz[2] - eps)}
                    | {0.0}, key=abs):
        nx = min(max(u0 + d, lo), hi)
        d = nx - u0
        nfz = (fz[0] + d, fz[1], fz[2] + d, fz[3])
        if any(nfz[0] - pad <= u <= nfz[2] + pad and nfz[1] <= v <= nfz[3]
               for u, v, _ in rig.parts):
            continue
        best = d
        break
    if best is None:
        return box, msg("{name}: {part}가 얼굴권에 걸리는데 문짝 안에서 "
                        "못 비킨다 — 그대로 둔다",
                        name=rig.name, part=hits[0][1])
    if abs(best) < 0.5:                            # 이미 비켜 있다 (반올림 몫)
        return box, ""
    return ((u0 + best, v0, u1 + best, v1),
            msg("{name}: {parts}를 피해 인물을 {shift:+.0f}유닛 민다 (업계 지침)",
                name=rig.name, parts=" · ".join(sorted({h[1] for h in hits})),
                shift=best))


def fit_on(smap: gsurf.SurfaceMap, lk: Look, *, anchor: str = "bottom",
           fill: float = BODY_FILL, bias_x: float = 0.5,
           group_unit: float = DEFAULT_GROUP_UNIT,
           coverage: float = 0.88, overshoot: float = 1.0,
           full_box: tuple[float, float, float, float] | None = None,
           tilt: float = 0.0, mirror: bool = False) -> Place | None:
    """면 지도 + 도안 생김새 → 배치 수치. 지도가 못 앉히면 None.

    `fit`이 준 상자(면 유닛)에 도안 잉크 상자를 **비율 그대로** 넣는다:

        scale = min(상자폭/잉크폭, 상자높이/잉크높이) × fill ÷ group_unit
        이동  = 상자 중심 − scale × (표시 변환된 잉크 중심)

    이동이 잉크 상자 중심을 쓰는 것이 중요하다 — 캔버스 원점이 그림 가운데가
    아니라(투명 여백·소품으로 치우친다) 원점을 맞추면 인물이 면 밖으로 나간다.

    `overshoot`는 fit 뒤에 곱하는 확대다 (발 앵커는 유지된다). 측면 인물은 이
    길을 안 쓴다 — 벨트라인 위는 옆면이 안 그리므로 확대는 잘리는 몫만 키운다.
    뼈대를 못 세운 면의 폴백에만 남는다.

    `tilt`는 **표시 회전값**(도, 부호 포함 — 그대로 배치 rot이 된다)이다. 회전된
    잉크 상자의 종횡비로 fit을 돌리고, 중심 오프셋도 같은 각으로 돌려 뺀다
    (`_hood_place`와 같은 규약 — 게임 회전 r에서 캔버스 +y는 (-sin r, cos r)).
    `mirror`는 우측면의 Tab+180 미러다: 표시 변환 = R(rot)·수평뒤집기이므로
    잉크 중심의 x부호를 뒤집고 나서 돌린다.
    """
    rw_ink, rh_ink = rot_ink(lk, tilt)           # 회전된 잉크 크기 (껍질 실측)
    rect = smap.fit(rw_ink / max(1e-6, rh_ink), coverage=coverage,
                    anchor=anchor, bias_x=bias_x)
    if rect is None:
        return None
    return place_in_rect(rect, smap.name, lk, anchor=anchor, fill=fill,
                         group_unit=group_unit, overshoot=overshoot,
                         full_box=full_box, tilt=tilt, mirror=mirror,
                         paint=smap.paint)


def place_in_rect(rect: tuple[float, float, float, float], name: str, lk: Look, *,
                  anchor: str = "bottom", fill: float = BODY_FILL,
                  group_unit: float = DEFAULT_GROUP_UNIT, overshoot: float = 1.0,
                  full_box: tuple[float, float, float, float] | None = None,
                  tilt: float = 0.0, mirror: bool = False,
                  paint: tuple[float, float, float, float] | None = None) -> Place:
    """**주어진 면 유닛 상자**에 도안을 앉힌다 — 상자를 누가 골랐든 수학은 같다.

    `fit_on`은 상자를 마스크 내접 탐색으로 고르고, 인물 배치는 `game.seam.
    person_span`이 기하로 고른다 (로커~루프라인). 그 뒤는 여기 한 자리다.
    """
    th = math.radians(tilt)
    c, s_ = math.cos(th), math.sin(th)
    ca, sa = abs(c), abs(s_)
    # 표시 변환을 먹인 잉크 상자 — 크기와 **중심**이 둘 다 여기서 나온다.
    ib = rot_ink_box(lk, tilt, mirror)
    rw_ink, rh_ink = ib[2] - ib[0], ib[3] - ib[1]
    rw, rh = rect[2] - rect[0], rect[3] - rect[1]
    s = min(rw / max(1e-6, rw_ink), rh / max(1e-6, rh_ink)) * fill / max(1e-6, group_unit)
    s *= overshoot
    cx, cy = (rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2
    if anchor == "bottom":                      # 상자 아래에 발을 붙인다
        cy = rect[1] + s * group_unit * rh_ink / 2
    ox_ = (ib[0] + ib[2]) / 2
    oy_ = (ib[1] + ib[3]) / 2
    x = cx - s * group_unit * ox_
    y = cy - s * group_unit * oy_
    # 노린 상자 = **화면에서 실제로 보일 상자** — 캔버스 전체(띠까지)를 앉힌 자리를
    # 면의 도색 상자로 자른다. 오토핏이 발자국과 견주는 상대라 "보일 것"이어야 한다.
    fb = full_box if full_box is not None else lk.box
    fcx, fcy = (fb[0] + fb[2]) / 2, (fb[1] + fb[3]) / 2
    if mirror:
        fcx = -fcx
    fw = (fb[2] - fb[0]) * ca + (fb[3] - fb[1]) * sa
    fh = (fb[2] - fb[0]) * sa + (fb[3] - fb[1]) * ca
    tcx = x + s * group_unit * (fcx * c - fcy * s_)
    tcy = y + s * group_unit * (fcx * s_ + fcy * c)
    tgt = (tcx - s * group_unit * fw / 2, tcy - s * group_unit * fh / 2,
           tcx + s * group_unit * fw / 2, tcy + s * group_unit * fh / 2)
    if paint is not None:
        p0, q0, p1, q1 = paint
        tgt = (max(tgt[0], p0), max(tgt[1], q0), min(tgt[2], p1), min(tgt[3], q1))
    return Place(surface=name, plan=Path(), x=round(x, 1), y=round(y, 1),
                 scale=round(s, 3), rot=round(tilt % 360.0, 1),
                 target=tuple(round(v, 1) for v in tgt),
                 why=msg("도색상자 {rw:.0f}×{rh:.0f}유닛 · 잉크 {w:.0f}×{h:.0f}유닛",
                         rw=rw, rh=rh, w=lk.w, h=lk.h)
                     + (msg(" · 기울기 {tilt:g}°", tilt=tilt) if tilt else ""))


def door_span(rig: "SideRig") -> tuple[float, float] | None:
    """**휠아치 사이** 유닛 구간 (문짝) — 글자가 통째로 그려지는 자리다.

    아치는 도색 마스크의 구멍이라 그 위의 획은 사라진다. 아치를 못 찾으면 None
    (부르는 쪽이 도색 상자 전체로 물러난다). 인물의 좌우 예산도 같은 구간이다
    (`game.seam.person_budget`) — 자가 하나여야 글자와 인물이 같은 판에 선다.
    """
    return gseam.door_span(rig.geom)


@dataclass
class SideRig:
    """옆면 한 짝의 뼈대 묶음 — 배치 계산이 이걸 들고 다닌다."""

    name: str
    smap: gsurf.SurfaceMap          # 원본 (설치) 지도
    body: gsurf.SurfaceMap          # 벨트라인 아래만 (배치·마스크 판정은 이걸 쓴다)
    geom: gseam.SideGeom
    seam: gseam.Seam | None
    rear_dir: float                 # +1이면 +u가 차 뒤
    place: Place | None = None
    tilt: float = 0.0
    mirror: bool = False
    # 얼굴이 피해야 할 부품 자리 (u, v, 이름) — `game.locators`가 설치 파일에서
    # 잰다. 등록이 없거나 미심쩍은 차는 빈 목록이라 지금까지와 똑같이 앉는다.
    parts: tuple[tuple[float, float, str], ...] = ()


# 면 역할 (레퍼런스 8장 조사): 주역은 측면, 보조 아트는 윗면, 뒤는 띠·모티프.
ROLE_MAIN = ("side_left", "side_right")


ROLE_EXTRA = "top"


ROLE_REAR = "rear"
