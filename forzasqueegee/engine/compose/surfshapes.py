"""면에 직접 놓는 꾸밈 — 관통 띠 · 산포 모티프의 도형 명세."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ...game import surface as gsurf
from ..catalog import Catalog
from ..model import UNITS_PER_SCALE, rnd
from .vocabulary import _RING8
from .scatter import (
    DECO_ANCHOR_GAP, DECO_GAP_MAX, DECO_HERO_CAP, DECO_TIER_SIZE, HALO_GROW,
    scatter_motifs)
from .bands import ROCKER_FRAC, TEETH_AMP, TEETH_OVERLAP


# 유리 면 — 관통 띠를 안 올린다 (레퍼런스의 유리는 작은 모티프뿐이다)
GLASS = ("windshield", "rear_window", "sunroof", "window_left", "window_right")


# 면 산포가 도색 상자에서 뻗는 반경 (상자 크기의 몫).
DECO_REACH = 0.44


FLOW_TEETH = 6                 # 관통 밴드 윗선을 뜯는 조각 수 (면당)


# **씨앗(`center_v`)이 없을 때의** 하부 밴드 윗선 (도색 상자의 몫). 옆면
# 로커(`ROCKER_FRAC` 0.26)보다 높다 — 앞·뒤 마스크는 아래쪽 몫이 범퍼 **밑면**
# 이라 그만큼을 얹어야 보이는 자리에 걸친다.
#
# 2026-08-21 머스탱 다크호스 실측: front는 0.26으로도
# 면 위 89%·폭 88%였고 0.42에서 면 위 81%·폭 88%로 스플리터를 제대로 덮는다.
# rear는 **어느 몫으로도 안 된다** — v를 훑으니 상자 8%에서 잉크 0px, 18%에서
# 179px, 28%에서 3.7천이고, 0.42로 올려도 범퍼 아래 모서리에만 걸렸다. 리어
# 구역이 범퍼 밑면과 데크리드를 같이 담아 상자 몫이 자가 못 되기 때문이라,
# 리어는 범퍼 로케이터를 씨앗으로 쓴다 (`game.locators.bumper_v`).
FACE_ROCKER_FRAC = 0.42


# ---- 띠는 면 끝을 **넘겨서** 끝난다 ----
# 사각이 면 안에서 끝나면 패널 한가운데에 곧은 단면이 남는다 — 사람이 그은
# 띠는 언제나 차 밖으로 달아나고, 끝을 자르는 것은 차의 실루엣이다. 넘긴 몫은
# 면 마스크가 자르므로 공짜다 (그려지지도 않는다).
BAND_OVERSHOOT = 1.06


# 나눠 깐 조각끼리 겹치는 몫 — 안 겹치면 반올림 한 칸에 흰 실선이 난다.
# 같은 색·같은 도형이라 겹쳐도 이음매가 안 보인다 (`_shape_batches`가 이 묶음을
# 위저드 한 바퀴로 놓는다).
TILE_OVERLAP = 1.08


def surface_sx_cap(smap: gsurf.SurfaceMap) -> float:
    """면 도형 **한 장**이 가질 수 있는 스케일 상한 — 면 제 크기다.

    면 스케일은 원형(랩) 축이라 상한 밖 목표는 수렴이 원리적으로 불가하고,
    폐루프는 5%씩 물러나다 도형을 통째로 잃는다 (`auto.itasha.shapes`). 하드
    랩을 실측한 것은 front의 ±2.3 하나뿐이라(2026-08-19) 그 면은 부르는 쪽이
    `max_sx`로 따로 못 박고, 여기서는 **폭 ÷ 128**로 어림한다 — 도형이 제가
    앉은 면보다 클 수 없다는 자다.

    **이 자는 일부러 짜다.** 2026-08-31 911 터보 인게임 실측에서 top은 8.296
    (이 자로는 7.977) · rear는 2.519(2.422)가 그대로 통과했다 — 실제 상한은
    적어도 4% 더 후하다. 그래도 안 늘리는 이유는 값이 싸기 때문이다: 넘게
    잡으면 띠 하나를 통째로 잃거나 15% 짧아지는데, 짜게 잡아 갈린 조각은
    묶음 스탬프로 나가 **장당 오히려 빠르다** (홑장 17.7초 vs 묶음 12.8초).
    그림도 같다 (같은 색 사각이 겹칠 뿐 — 실측 픽셀 차 0.1%).
    """
    return max(0.5, (smap.paint[2] - smap.paint[0]) / 2 / UNITS_PER_SCALE)


def band_tiles(u0: float, u1: float, cap: float) -> list[tuple[float, float]]:
    """`[u0,u1]`을 덮는 사각 조각들 `(중심 u, sx)` — 상한을 넘으면 나눈다.

    한 장으로 안 되는 길이를 **짧게 자르는 대신** 여러 장으로 잇는다. 옛
    길(`sx = min(sx, max_sx)`)은 띠를 면 절반에서 사각으로 끊었다 (실측:
    RX-7 프론트 면 294유닛에 띠 42유닛).
    """
    span = max(1e-6, u1 - u0)
    lim = max(1e-6, cap * UNITS_PER_SCALE)
    n = max(1, math.ceil(TILE_OVERLAP * span / 2 / lim))
    half = span / 2 / n * (TILE_OVERLAP if n > 1 else 1.0)
    return [(u0 + span * (i + 0.5) / n, half / UNITS_PER_SCALE) for i in range(n)]


def flow_shapes(color: tuple[int, int, int], smap: gsurf.SurfaceMap,
                box: tuple[float, float, float, float] | None = None,
                max_sx: float | None = None,
                shapes: tuple[str, ...] | None = None,
                mode: str = "rocker", center_v: float | None = None,
                cat: Catalog | None = None, rot: float = 0.0,
                height: float | None = None) -> list[dict]:
    """**관통 요소** — 옆면의 문법을 이웃 면으로 이어 가는 도형 명세.

    레퍼런스의 배경 요소는 한 면에서 끝나지 않고 여러 면을 관통해 전체를
    접착한다 — 제작자들은 인물을 이음새에 정밀하게 잇는 게 아니라 **면 요소가
    비슷한 높이로 이어지게** 하고 전환을 이음새에 숨긴다.

    이어 가는 것은 옆면에서 실제로 서는 것이라야 한다:

    - `rocker` (앞·뒤 범퍼) — **하부 투톤 밴드 + 찢긴 윗선**. 옆면의
      `stripe_layers`가 만드는 그 밴드가 범퍼를 돌아 이어진다 (Evo IX의 검정
      하부가 앞뒤로 이어지고, ARIS의 흰 하부도 그렇다).
    - `stripe` (윗면) — **차 길이로 달리는 세로 줄 두 벌**. 지붕에는 '하부'가
      없다. 레퍼런스의 윗면 관통 요소는 후드·지붕을 타고 넘는 레이싱 스트라이프
      두 줄이다 (Chihaya의 청록·흰 두 줄). 윗면 축이 u = 차 뒤 · v = 차 오른쪽
      이므로 v를 중심선 양옆에 두고 u로 길게 뻗은 사각이 곧 그 줄이다.

    그룹 주입이 아니라 **도형 위저드로 면에 직접** 놓는 명세다 (`auto.itasha.
    add_shape_job`) — 소형 그룹 주입은 레이어 표 식별이 모호해서 못 쓴다
    (2026-08-18 실측: 12장 그룹의 표 후보가 51건, 2장은 9,344건 — 엉뚱한 표에
    쓰여 씨앗 사각이 그대로 찍혔다). 값은 게임 변형 칸에 그대로 들어간다
    (x·y = 면 유닛, sx·sy = 스케일 값 = 반폭/128).

    `box`를 주면 밴드의 **높이·자리(v)**를 그 상자로 잡는다 — 프론트처럼 도색
    상자 가운데가 비도색(그릴)인 면은 내접 상자를 줘야 밴드가 보이는 자리에
    앉는다. **길이(u)는 언제나 면 도색 폭**이고 거기서 더 넘긴다: 내접 상자로
    길이까지 재던 옛 길은 띠를 면 한가운데에서 사각으로 끊었다 (실측: RX-7
    프론트 면 294유닛에 띠 42유닛 · CRX 306에 79 · 챌린저 502에 282).

    `max_sx`는 이 면의 스케일 축 상한이다 (`surface_sx_cap`과 함께 걸린다).
    상한을 넘는 길이는 **짧게 자르지 않고 나눠 깐다** (`band_tiles`).
    """
    q0, q1 = (box[1], box[3]) if box is not None else (smap.paint[1], smap.paint[3])
    hh = q1 - q0
    # 길이는 면 도색 폭 — 여기서 `BAND_OVERSHOOT`만큼 더 넘겨 끝을 면 밖에 둔다
    fu0, fu1 = smap.paint[0], smap.paint[2]
    fcx, fw = (fu0 + fu1) / 2, (fu1 - fu0) * BAND_OVERSHOOT
    u0, u1 = fcx - fw / 2, fcx + fw / 2
    cap = surface_sx_cap(smap)
    if max_sx is not None:
        cap = min(cap, max_sx)
    tiles = band_tiles(u0, u1, cap)
    if mode == "macro":
        # **옆면의 큰 색면이 이음새를 건너온 것** (`atlas.carry_band`).
        #
        # 로커 띠와 갈리는 자리 셋: 색이 판 색이고(무채 잉크가 아니다), 높이와
        # 각을 옆면에서 받아 오며(제 면의 몫으로 잡지 않는다), 찢긴 윗선이
        # 없다 — 이어지는 것이지 이 면에서 시작하는 것이 아니다.
        #
        # 조각을 나눌 때 각이 있으면 조각마다 중심 v가 달라야 한다 — 한 줄로
        # 놓고 전부 같은 각으로 돌리면 계단이 진다.
        hb = height if height is not None else ROCKER_FRAC * hh
        cv = center_v if center_v is not None else (q0 + q1) / 2
        tr = math.radians(rot)
        out2 = []
        for tx, tsx in tiles:
            out2.append({"shape": "A_01", "x": rnd(tx, 1),
                         "y": rnd(cv + (tx - fcx) * math.tan(tr), 1),
                         "sx": rnd(tsx / max(0.2, math.cos(tr)), 3),
                         "sy": rnd(hb / 2 / UNITS_PER_SCALE, 4),
                         "rot": rnd(rot % 360.0, 1), "rgb": list(color)})
        return out2
    if mode == "stripe":
        return [{"shape": "A_01", "x": rnd(tx, 1),
                 "y": rnd(q0 + f * hh, 1), "sx": rnd(tsx, 3),
                 "sy": rnd(0.055 * hh / 2 / UNITS_PER_SCALE, 4),
                 "rot": 0.0, "rgb": list(color)}
                for f in (0.42, 0.58) for tx, tsx in tiles]
    # 하부 밴드 — **도색 상자 안에서** 잡는다. 상자 아래로 빼면 도형 중심이
    # 상자 밖으로 나가고, 면의 이동 축은 상자보다 좁게 클램프되므로 실행 중에
    # 엉뚱한 자리에 꽂힌다 (`add_shape_job`의 폐루프가 못 닿는다).
    # 높이는 `FACE_ROCKER_FRAC`다 — 옆면 로커보다 높다 (그 상수 설명: 상자
    # 아래쪽 몫이 범퍼 밑면이라 화면에 안 나온다). `center_v`(범퍼 로케이터)가
    # 있으면 **그 높이에 앉힌다** — 리어처럼 상자가 보이는 면보다 넓은 면은
    # 상자 몫으로는 못 앉는다 (`game.locators.bumper_v`).
    if center_v is not None:
        half = ROCKER_FRAC * hh / 2
        lo, top = center_v - half, center_v + half
    else:
        top = q0 + FACE_ROCKER_FRAC * hh
        lo = q0 - 0.10 * hh      # 상자 아래를 조금 물어 범퍼 밑선까지 덮는다
    out = [{"shape": "A_01", "x": rnd(tx, 1), "y": rnd((lo + top) / 2, 1),
            "sx": rnd(tsx, 3),
            "sy": rnd((top - lo) / 2 / UNITS_PER_SCALE, 4),
            "rot": 0.0, "rgb": list(color)} for tx, tsx in tiles]
    # 찢긴 윗선 — 옆면 로커와 **같은 자**다 (`_teeth`): 가로는 이웃과 겹칠 만큼,
    # 세로는 밴드의 몇 할만. 등방으로 재던 옛 자는 면 높이의 0.39배짜리 조각을
    # 내서 범퍼 위로 검은 가시가 솟았다 (frag0-03 미리보기).
    vocab = shapes or ("A_21",)
    band = top - lo
    for i in range(FLOW_TEETH):
        k = 0.55 + 0.45 * (i % 3) / 2.0
        name = vocab[i % len(vocab)]
        reach = (cat.shapes[name].reach
                 if cat is not None and name in cat.shapes else 1.0)
        out.append({"shape": name,
                    "x": rnd(u0 + (u1 - u0) * (i + 0.5) / FLOW_TEETH, 1),
                    "y": rnd(top - 0.38 * band * ((i * 3 % 5) / 4.0), 1),
                    "sx": rnd(TEETH_OVERLAP * ((u1 - u0) / FLOW_TEETH) / 2 * k
                                / UNITS_PER_SCALE / reach, 3),
                    "sy": rnd(TEETH_AMP * band / 2 * k
                                / UNITS_PER_SCALE / reach, 4),
                    "rot": rnd((17.0 * i) % 24.0 - 12.0, 1),
                    "rgb": list(color)})
    return out


@dataclass
class DecoAnchor:
    """면 하나의 **도안 기준점** — 그 면의 산포가 이걸 자로 쓴다 (면 유닛).

    레퍼런스 8장에서 꾸밈은 예외 없이 도안에서 자란다: 무리의 핵이 도안 뒤쪽에
    있고(EVELYNE 백합 · 수이세이 별무리 · RIN 아네모네), 크기가 도안 크기의
    몫이며(최대형이 인물 높이의 0.4~0.7), 도안이 없는 면의 무리는 이웃 면에서
    이음새를 넘어 흘러 들어온다(Fate R34의 리어 별은 리어 쿼터에서 온다).
    """

    box: tuple[float, float, float, float]   # 뿌리 — 구름 중심·크기 자를 내는 상자
    center: tuple[float, float]              # 후보 구름의 중심 — 뿌리 상자 중심
    at: tuple[float, float]                  # 무리가 뭉치는 자리 (상자 모서리 너머)
    ref: float                               # 크기 자 — 뿌리 상자의 긴 변
    # 배경 모티프가 **피할 자리** — 도안 잉크가 이 면에 실제로 있을 때만이다.
    # 뿌리와 갈리는 이유: 이웃 면에서 투영한 뿌리는 "이 패널에서 보면 도안이
    # 어느 쪽이냐"일 뿐 여기 그려지는 것이 아니라, 그것으로 비우면 멀쩡한 패널이
    # 통째로 금지 구역이 된다 (실측: 리어 130×170이 비어 모티프가 다 버려졌다).
    avoid: tuple[float, float, float, float] | None = None
    why: str = ""


def deco_anchor(box: tuple[float, float, float, float],
                flow: tuple[float, float] = (1.0, 0.0),
                why: str = "",
                avoid: tuple[float, float, float, float] | None = None
                ) -> DecoAnchor:
    """도안 상자 하나 → 그 면의 산포 앵커.

    뭉치는 자리는 상자 **바깥 모서리 너머**다 (`DECO_ANCHOR_GAP`) — 상자 안에
    두면 무리가 도안을 피해 **양쪽으로 갈라진다** (2026-08-22 판정: 꽃 무리가
    인물 앞에, 큰 꽃 한 송이가 인물 뒤에 따로 섰다). 레퍼런스의 무리는 갈라지지
    않고 도안 뒤쪽에 한 덩이로 붙는다.

    면 안으로 물리는 것은 **산포가** 한다 (`surface_deco_shapes`) — 흩어도 되는
    범위는 면이 아니라 그때 준 상자(후드 덩어리 같은)라 여기서는 알 수 없다.
    """
    u0, v0, u1, v1 = box
    w, h = u1 - u0, v1 - v0
    cu, cv = (u0 + u1) / 2, (v0 + v1) / 2
    dx, dy = flow
    return DecoAnchor(
        box=box, center=(cu, cv), ref=max(w, h), why=why, avoid=avoid,
        at=(cu + dx * (w / 2 + DECO_ANCHOR_GAP * w),
            cv + dy * (h / 2 + DECO_ANCHOR_GAP * h)))


def surface_deco_shapes(colors: tuple[tuple[int, int, int], ...],
                        smap: gsurf.SurfaceMap, cat: Catalog,
                        n: int = 8,
                        box: tuple[float, float, float, float] | None = None,
                        shapes: tuple[str, ...] | None = None,
                        anchor: "DecoAnchor | None" = None,
                        halo: tuple[int, int, int] | None = None,
                        over: bool = False,
                        phase: float = 0.0) -> list[dict]:
    """면 산포 모티프 — 도형 위저드 명세 (기성 모티프, 도색 마스크 안만).

    옆면 말고는 인물 그룹을 또 준비하는 것이 비싸다 (장당 0.44초 + 그룹 슬롯
    하나). 십수 장 규모의 모티프는 위저드가 몇 분에 놓고, 그 산포 자체가
    레퍼런스의 윗면·리어 문법이다 (무사시 후드의 별 · FDgUB4의 스플래터).

    **자리를 정하는 것은 `anchor`(도안)다.** 레퍼런스에서 꾸밈은 예외 없이
    도안에서 자란다 — 무리의 핵이 도안 뒤쪽에 있고, 크기가 도안 크기의 몫이며,
    도안이 없는 면의 무리는 이웃 면에서 이음새를 넘어 흘러 들어온다. 앵커가
    없으면(둘 다 못 찾은 면) 면 상자 중심으로 물러난다.

    `box`를 주면 산포를 그 상자(예: 후드 덩어리)로 좁힌다 — 윗면은 이동 축이
    클램프되는 면이라 (실측: top x ±554) 판 끝까지 흩으면 못 가는 자리가 생긴다.

    **마스크 안에 온전히 드는 것만 남는다** — 면을 걸터앉은 모티프는 그 자리에서
    잘려 조각으로 남는다 (이웃 면에 이어 붙이지 않는다).

    `over=True`면 도안 **위에** 얹는 전경 몫이다 (옆면 `deco-front`와 같은 문법) —
    부르는 쪽이 `post_shapes`에 넣어 그룹 위로 올린다.
    """
    p0, q0, p1, q1 = box if box is not None else smap.paint
    cx, cy = (p0 + p1) / 2, (q0 + q1) / 2
    rx, ry = DECO_REACH * (p1 - p0), (DECO_REACH - 0.04) * (q1 - q0)
    if anchor is not None:
        avoid = anchor.avoid
        # 크기 자는 도안이되 **패널이 상한을 쥔다** (`DECO_HERO_CAP`) — 층 비율은
        # 옆면에서 잰 것이라 좁은 패널에 그대로 쓰면 최대형이 판을 덮는다.
        ref = min(anchor.ref,
                  DECO_HERO_CAP * min(p1 - p0, q1 - q0) / DECO_TIER_SIZE[0])
        # 후보 구름은 **도안을 중심으로** 뜬다 — 면 한가운데에서 뜨면 도안이
        # 면 끝에 앉은 판(후드에 눕힌 인물·리어 쿼터에서 넘어온 앵커)에서
        # 무리가 도안과 반대쪽에 선다. 반경은 여전히 면의 것이다: 얼마나 멀리
        # 흩어도 되는지는 패널이 정하고, 어디서 자라는지는 도안이 정한다.
        cx, cy = anchor.center
        # 뭉치는 자리를 **흩어도 되는 상자 안**으로 물린다. 밖에 두면 무리가
        # 통째로 그 밖에 서서 자리 검사에 전멸한다 (제로투 실측: 후드 인물의
        # 앵커가 지붕 위에 앉아 후드 모티프 일곱이 다 버려졌다). 최대형
        # 반지름만큼은 안쪽이라야 큰 것이 모서리에서 반쪽으로 잘리지 않는다.
        hero = DECO_TIER_SIZE[0] * ref / 2.0

        def _in(v: float, lo: float, hi: float) -> float:
            return min(max(v, lo + hero), max(lo + hero, hi - hero))

        at = (_in(anchor.at[0], p0, p1), _in(anchor.at[1], q0, q1))
        # 구름 중심도 **상자 안**이라야 한다 — 뭉치는 자리만 물리고 중심을 밖에
        # 두면 후보가 통째로 상자 밖에 떠서 `_ok`에 전멸한다 (윗면 데크: 앵커가
        # 후드 도안이라 후보 넷이 하나도 안 남았다).
        if box is not None:
            cx, cy = _in(cx, p0, p1), _in(cy, q0, q1)
    else:
        # 뿌리를 못 찾은 면 — 면 상자 한가운데에서 자란다. 도안이 어느 면에도
        # 안 앉았거나(있을 수 없다) 이 면으로 오는 접기 변환이 하나도 안 풀리는
        # 차다. 크기 자는 면 상자로 물러난다: 옛 최대형이 면 높이의 0.20이었고
        # `DECO_TIER_SIZE[0]`이 0.55이므로 그 값이 되게 나눈다.
        at, ref, avoid = (cx, cy), 0.20 / DECO_TIER_SIZE[0] * (q1 - q0), None

    def _ok(u: float, v: float, rq: float) -> bool:
        # 자리 검사는 **모티프가 실제로 덮는 넓이**로 한다 — 중심 한 점만 보면
        # 큰 모티프가 마스크 가장자리에 걸터앉아 반쪽으로 찍힌다 (13호차 리어
        # 실차 캡처: 별들이 범퍼 아랫선에 잘려 조각으로 남았다). 반지름은
        # 내접 몫(MOTIF_INSCRIBE)이다 — 별·꽃은 외접 사각의 모서리를 안 채우므로
        # 사각 모서리로 재면 들어갈 자리도 다 물리친다.
        if not smap.masked_at(u, v):
            return False                   # 중심은 언제나 제 면 안이라야 한다
        # `box`(후드 덩어리)는 **경계**다 — 구름 중심이 도안으로 옮겨 가면서
        # 후보가 상자 밖으로 새면 지붕 블랙아웃 위에 모티프가 떠서 투톤이
        # 무너진다. 상자를 준 뜻이 그 자리에만 흩으라는 것이므로 여기서 막는다.
        if box is not None and not (p0 <= u <= p1 and q0 <= v <= q1):
            return False
        # 이동 축은 **도색 상자 밖에서 클램프에 꽂힌다** (`add_shape_job`) —
        # 상자 밖 중심은 실행 중에 엉뚱한 자리에 앉으므로 아예 안 낸다.
        pp0, qq0, pp1, qq1 = smap.paint
        if not (pp0 <= u <= pp1 and qq0 <= v <= qq1):
            return False
        return all(smap.masked_at(u + math.cos(t) * rq, v + math.sin(t) * rq)
                   for t in _RING8)

    out: list[dict] = []
    for mo in scatter_motifs(
            center=(cx, cy), radii=(rx, ry), ref=ref, n=n,
            vocab=shapes or (cat.circle,), cat=cat, colors=colors,
            anchor_at=at, avoid=avoid, over=over, place_ok=_ok, phase=phase,
            gap=None if over else DECO_GAP_MAX):
        spec = {"shape": mo.shape, "x": rnd(mo.x, 1), "y": rnd(mo.y, 1),
                "sx": rnd(mo.half, 3), "sy": rnd(mo.half, 3),
                "rot": rnd(mo.rot, 1), "rgb": list(mo.color)}
        # 후광은 캔버스 산포와 같은 자다 (`deco_layers`) — 다만 **최대형 하나**
        # 뒤에만 깐다. 면 도형은 한 장이 폐루프 일곱이라 캔버스처럼 두 층에 다
        # 두르면 면마다 넷씩, 차 한 대에 스무 장이 붙는다. 무리를 차체에서
        # 떼어 놓는 일은 구도를 쥔 큰 것 하나가 거의 다 한다.
        if halo is not None and mo.tier == 0:
            out.append(dict(spec, sx=rnd(mo.half * HALO_GROW, 3),
                            sy=rnd(mo.half * HALO_GROW, 3), rgb=list(halo)))
        out.append(spec)
    return out
