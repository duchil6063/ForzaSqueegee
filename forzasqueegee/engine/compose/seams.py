"""이음새를 건너는 **정책과 기하** — 이을까, 끊을까, 피할까.

## 왜 정책이 필요한가

사람이 만든 리버리는 모든 요소를 이음새 너머로 이어 붙이지 않는다. 큰 색면과
하부 투톤은 차를 돌아 이어지고(그 전환이 이음새에 숨는다), 얼굴·이름 글자는
애초에 이음새를 피해 앉으며, 잔 모티프는 이음새 앞에서 그냥 끝난다. 억지로
다 이으면 자동 생성 티가 나고, 하나도 안 이으면 옆면에만 구도가 있는 차가
된다. 그래서 **역할마다 정책이 다르다** (`ROLE_POLICY`).

## 왜 자리를 다시 재는가

띠 하나를 이웃 면으로 옮길 때 물어야 하는 것은 "**이음선 위에서** 이 띠가
어느 높이·어느 각·어느 두께인가"다. 두 자리에서 그걸 틀리고 있었다:

- **재는 자리** — 옆면 띠의 높이는 면 **한가운데**에서 잰 값이었다. 띠가 기울어
  있으면 이음선 위의 높이는 그와 `tan(각) × (이음선 − 가운데)`만큼 다르다
  (실비아 옆면: 이음선이 가운데에서 412유닛, 10° 기울기면 **73유닛**).
- **놓는 자리** — 건너간 높이를 목적 면의 **한가운데**에 놓고 거기서 기울였다.
  같은 크기의 어긋남이 반대편에서 한 번 더 난다.

두 어긋남이 겹치면 띠가 이음새에서 백 유닛 넘게 어긋난 채 만난다 — 그것이
"패널 seam에서 갑자기 끊기거나 방향이 달라 보인다"의 기하다. 이 모듈은 띠를
**이음선 위의 점과 각**으로 말하고, 놓을 때도 그 점을 지나게 한다.

## 못 이으면 끊는다

이음새는 늘 믿을 수 있는 자가 아니다 (`game.fold.SeamSegment.confidence` —
두 마스크가 그 자리에서 서로 다른 것을 쥐면 신뢰가 0으로 간다). 신뢰가 낮거나,
건너간 높이가 면 밖이거나, 두께를 반 넘게 깎아야 들어가면 **안 잇는다** —
그 면은 제 문법으로 끝낸다. 어정쩡하게 이어 붙인 띠보다 낫다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ...game import fold as gfold
from ..model import rnd


# 역할 → 이음새 정책.
#
#   avoid      이음새에 닿지 않게 앉힌다 (얼굴·글자 — 곡면에서 읽히지 않는다)
#   continue   이웃 면으로 이어 그린다 (큰 색면·하부 투톤·레이싱 스트라이프)
#   terminate  이음새 앞에서 끝낸다 (잔 모티프·에코 — 반쪽으로 잘리면 파편이다)
AVOID = "avoid"
CONTINUE = "continue"
TERMINATE = "terminate"

ROLE_POLICY: dict[str, str] = {
    # 이어 간다 — 차를 한 바퀴 도는 큰 흐름
    "itasha_bed": CONTINUE,
    "itasha_stripe": CONTINUE,
    "itasha_stack": CONTINUE,
    "macro": CONTINUE,
    "rocker": CONTINUE,
    "stripe": CONTINUE,
    # 피한다 — 이음새 위에서 읽히지 않는 것
    "text": AVOID,
    "text_sub": AVOID,
    "face": AVOID,
    "itasha_keyline": AVOID,
    # 끊는다 — 이어 봐야 파편이 되는 잔 것
    "itasha_deco": TERMINATE,
    "itasha_echo": TERMINATE,
    "motif": TERMINATE,
    "echo": TERMINATE,
}


def policy_for(role: str) -> str:
    """모르는 역할은 **끊는다** — 이어 붙이는 것이 기본값이면 안 된다."""
    return ROLE_POLICY.get(role, TERMINATE)


# 이음새를 이어도 되는 최소 신뢰 (`SeamSegment.confidence`). 실측 세 대의
# 차체 이음새 분포: 옆↔앞·옆↔뒤는 0.24~1.00, 윗↔옆은 0.00~0.97이고 0에 가까운
# 토막은 두 마스크가 그 자리에서 서로 다른 것을 쥔 자리다 (미러·스포일러·
# 데크리드로 감아 도는 뒤 면).
CONF_MIN = 0.45


# 건너간 띠의 두께가 이만큼 밑으로 깎여야 면에 들어가면 **안 잇는다**. 두께가
# 반으로 줄면 이음새에서 만나는 두 띠가 다른 띠로 보인다 — 이을 바에는 끊는다.
WIDTH_MIN = 0.55


# 건너간 각을 이만큼 넘게 꺾어야 하면 안 잇는다 (도). 방향이 다른 띠를 억지로
# 눕히면 이음새에서 꺾인 자국이 남는다.
TILT_CLAMP_MAX = 12.0


@dataclass(frozen=True)
class Band:
    """이음새를 건너려는 **띠 하나** — 전부 그 면의 유닛이다.

    `v`는 `at_u`에서 잰 중심선 높이다. 이음새 이야기를 할 때 `at_u`는 늘
    **이음선 위**라야 한다 — 면 한가운데에서 잰 높이를 이음새의 높이로 쓰는 것이
    이 모듈이 고치는 잘못이다.
    """

    v: float
    angle: float          # 도 (면 u축 기준)
    thickness: float
    at_u: float

    def v_at(self, u: float) -> float:
        """중심선이 `u`에서 갖는 높이."""
        return self.v + math.tan(math.radians(self.angle)) * (u - self.at_u)


@dataclass(frozen=True)
class Continuation:
    """이음새 하나를 건너기로 한(또는 안 하기로 한) 결정과 그 근거."""

    policy: str
    band: Band | None = None
    confidence: float = 0.0
    why: str = ""
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def carried(self) -> bool:
        return self.policy == CONTINUE and self.band is not None


def seam_cross(f: gfold.Fold, band: Band) -> tuple[float, float]:
    """띠가 **이음선과 만나는 점** (src 유닛).

    이음선은 넘치는 축의 상수선이므로 만나는 자리는 그 축의 값이 이음선일 때다.
    이음선 자체가 공유 축을 따라 기울어 있으므로(`Fold.segments`) 한 번 되짚는다:
    띠의 높이로 이음선을 고르고, 그 이음선 위에서 띠의 높이를 다시 잰다.
    """
    if f.axis == "v":
        # 이음선이 v 상수선 — 띠의 높이가 곧 이음선이라 만나는 u가 하나가 아니다.
        # 그 면에서는 띠가 세로로 서므로 자리는 이음선 위 아무 u여도 같다.
        return band.at_u, f.edge
    sg = f.segment_at(band.v)
    u_seam = f.edge if sg is None else sg.src_edge
    v_seam = band.v_at(u_seam)
    sg2 = f.segment_at(v_seam)
    if sg2 is not None and sg2 is not sg:
        u_seam = sg2.src_edge
        v_seam = band.v_at(u_seam)
    return u_seam, v_seam


def carry(f: gfold.Fold, band: Band, role: str, *,
          dst_box: tuple[float, float, float, float],
          tilt_max: float, thick_max: float) -> Continuation:
    """띠 하나를 이웃 면으로 — 이을지 끊을지와 그 기하.

    `dst_box`는 목적 면의 도색 상자, `tilt_max`는 그 면에서 허용하는 기울기(도),
    `thick_max`는 두께 상한(면 유닛)이다. 되돌림의 `metrics`는 원자료다 —
    가중합에 안 넣고 기록만 한다 (`work/lab/deco/seamcheck.py`가 읽는다).
    """
    pol = policy_for(role)
    if pol != CONTINUE:
        return Continuation(policy=pol, why=f"역할 {role}")
    u_seam, v_seam = seam_cross(f, band)
    conf = f.confidence_at(v_seam if f.axis == "u" else u_seam)
    u2, v2 = f.to_local(u_seam, v_seam)
    d = _direction(f, band.angle)
    ang2 = math.degrees(math.atan2(d[1], d[0]))
    ang2 = ang2 - 180.0 if ang2 > 90.0 else (ang2 + 180.0 if ang2 < -90.0 else ang2)
    s_slope, d_slope = f.seam_tangent(v_seam if f.axis == "u" else u_seam)
    metrics = {
        "conf": rnd(conf, 4),
        "u_seam": rnd(float(u_seam), 2),
        "v_seam": rnd(float(v_seam), 2),
        "v_dst": rnd(float(v2), 2),
        "ang_src": rnd(band.angle, 2),
        "ang_dst": rnd(ang2, 2),
        # 이음선 자체가 양쪽에서 같은 각으로 놓였나 (0이면 같다)
        "tangent_err": rnd(abs(abs(s_slope) - abs(d_slope)), 4),
    }
    q0, q1 = dst_box[1], dst_box[3]
    pad = 0.10 * (q1 - q0)
    if conf < CONF_MIN:
        return Continuation(policy=TERMINATE, confidence=conf, metrics=metrics,
                            why=f"이음새 신뢰 {conf:.2f} < {CONF_MIN}")
    if not (q0 - pad <= v2 <= q1 + pad):
        return Continuation(policy=TERMINATE, confidence=conf, metrics=metrics,
                            why="건너간 높이가 면 밖")
    ang_c = max(-tilt_max, min(tilt_max, ang2))
    if abs(ang_c - ang2) > TILT_CLAMP_MAX:
        metrics["tilt_clamp"] = rnd(abs(ang_c - ang2), 2)
        return Continuation(policy=TERMINATE, confidence=conf, metrics=metrics,
                            why=f"각을 {abs(ang_c - ang2):.0f}° 꺾어야 든다")
    th = min(band.thickness, thick_max)
    ratio = th / max(1e-6, band.thickness)
    metrics["width_ratio"] = rnd(ratio, 4)
    metrics["tilt_clamp"] = rnd(abs(ang_c - ang2), 2)
    if ratio < WIDTH_MIN:
        return Continuation(policy=TERMINATE, confidence=conf, metrics=metrics,
                            why=f"두께가 {ratio:.2f}배로 깎인다")
    # 실제로 놓일 띠 — 이음선 위의 점 (u2, v2)를 지나고 각은 클램프한 값
    return Continuation(policy=CONTINUE, confidence=conf, metrics=metrics,
                        band=Band(v=float(v2), angle=float(ang_c), thickness=float(th),
                                  at_u=float(u2)),
                        why="이음선 위에서 만난다")


def _direction(f: gfold.Fold, angle: float) -> tuple[float, float]:
    """방향 하나를 dst 유닛으로 — 아핀의 선형부만 (평행이동 없음)."""
    r = math.radians(angle)
    dx, dy = math.cos(r), math.sin(r)
    x = float(f.A[0, 0]) * dx + float(f.A[0, 1]) * dy
    y = float(f.A[1, 0]) * dx + float(f.A[1, 1]) * dy
    n = math.hypot(x, y)
    return (x / n, y / n) if n > 1e-9 else (1.0, 0.0)


def seam_error(f: gfold.Fold, src: Band, dst: Band) -> dict[str, float]:
    """이음선 위에서 두 띠가 얼마나 어긋나 만나나 — **결과를 재는 자**.

    `src`는 이 면의 띠, `dst`는 이웃 면에 실제로 놓인 띠다 (클램프까지 먹은 것).
    되돌림:

        pos     이음선 위 중심선 높이 차 (면 유닛)
        tilt    각 차 (도)
        width   두께 비 (dst / src, 1이면 같다)
    """
    u_seam, v_seam = seam_cross(f, src)
    u2, v2 = f.to_local(u_seam, v_seam)
    d = _direction(f, src.angle)
    want = math.degrees(math.atan2(d[1], d[0]))
    want = want - 180.0 if want > 90.0 else (want + 180.0 if want < -90.0 else want)
    return {
        "pos": abs(dst.v_at(u2) - v2),
        "tilt": abs(dst.angle - want),
        "width": dst.thickness / max(1e-6, src.thickness),
    }
