"""구성 계열 — 꾸밈 문법 **한 벌이 아니라 여러 벌**이다.

같은 도안이라도 사람마다 다른 리버리를 만든다. 계열은 그 갈래의 뼈대다 —
배경 색면의 구조, 흐름, 밀도, 여백, 모티프 크기 분포, 다른 면으로 이어지는
법. 어느 계열이 이 도안에 맞는지는 **후보를 다 지어 점수로** 고른다
(`design.compose_design`) — 여기서는 도안 뜻(`DesignIntent`)으로 후보 순서만
정한다.
"""

from __future__ import annotations

from dataclasses import dataclass

from .intent import DesignIntent
from .look import Look


@dataclass(frozen=True)
class Family:
    name: str
    bed: str                    # none · plate · wedge · blob · slab
    bed_level: float            # 베드 크기 (0~1)
    motif_n: int                # 옆면 배경 모티프 수
    tier_scale: float           # 모티프 층 크기 배율
    rocker: bool                # 하부 투톤 로커
    top_stripe: bool            # 윗면 레이싱 스트라이프
    front_n: int                # 인물 위에 얹는 전경 모티프 수
    empty_target: float         # 여백 구역이 비어야 하는 몫 (점수 목표)
    clutter: tuple[float, float]   # 장식 커버리지 목표 구간 (밴드 빈 면적 대비)
    echo: bool                  # 그래픽 에코 (결 조각·샤드·블록)
    flows: tuple[str, ...]      # 흐름 후보 — auto · rear · front
    other_density: float        # 다른 면 모티프 수 배율
    torn: bool = False          # 베드 흐름 끝을 뜯는다 (스플래시)
    text_budget: float = 1.0    # 글자 장수 예산 배율 (여백 계열은 글자도 절제한다)


FAMILIES: dict[str, Family] = {
    # 비움 — 큰 베드 없이 얇은 슬래브 하나와 잔 모티프 몇, 여백이 주역
    "minimal": Family("minimal", bed="slab", bed_level=0.35, motif_n=7,
                      tier_scale=0.75, rocker=False, top_stripe=False, front_n=1,
                      empty_target=0.92, clutter=(0.04, 0.16), echo=False,
                      flows=("auto",), other_density=0.45, text_budget=0.5),
    # 그래픽 베드 — 인물 뒤 큰 색면이 구도를 잡고 모티프는 그 가장자리에
    "graphic_bed": Family("graphic_bed", bed="plate", bed_level=0.75, motif_n=14,
                          tier_scale=0.95, rocker=True, top_stripe=False, front_n=2,
                          empty_target=0.75, clutter=(0.18, 0.42), echo=True,
                          flows=("auto", "rear", "front"), other_density=0.8),
    # 사선 흐름 — 대각 판 둘이 인물을 지나 흐르고 모티프가 그 결을 따른다
    "diagonal_flow": Family("diagonal_flow", bed="wedge", bed_level=0.65, motif_n=16,
                            tier_scale=0.9, rocker=True, top_stripe=True, front_n=3,
                            empty_target=0.65, clutter=(0.22, 0.48), echo=True,
                            flows=("auto", "rear"), other_density=1.0),
    # 모터스포츠 — 로커·스트라이프 등 직선 요소, 베드는 낮은 슬래브, 모티프는 적게
    "motorsport": Family("motorsport", bed="slab", bed_level=0.55, motif_n=9,
                         tier_scale=0.8, rocker=True, top_stripe=True, front_n=1,
                         empty_target=0.80, clutter=(0.12, 0.30), echo=True,
                         flows=("rear", "front"), other_density=0.6, text_budget=0.8),
    # 스플래시 — 덩어리 베드에 뜯긴 가장자리, 모티프가 많고 인물 위로도 얹힌다
    "splash": Family("splash", bed="blob", bed_level=0.8, motif_n=22,
                     tier_scale=1.05, rocker=True, top_stripe=False, front_n=4,
                     empty_target=0.5, clutter=(0.30, 0.60), echo=True,
                     flows=("auto",), other_density=1.25, torn=True),
}


FAMILY_NAMES = tuple(FAMILIES)


def rank_families(it: DesignIntent, lk: Look, person_frac: float) -> list[str]:
    """이 도안에 맞는 계열 **후보 순서** (전부 후보다 — 점수가 최종을 정한다).

    `person_frac`은 인물이 차체 밴드 폭을 얼마나 덮나 (0~1).
    """
    sc: dict[str, float] = {n: 0.0 for n in FAMILIES}
    # 인물이 밴드를 크게 덮으면 베드가 설 자리가 없다 → minimal·motorsport
    if person_frac > 0.72:
        sc["minimal"] += 1.0
        sc["motorsport"] += 0.6
        sc["splash"] -= 0.6
        sc["graphic_bed"] -= 0.3
    else:
        sc["graphic_bed"] += 0.6
        sc["diagonal_flow"] += 0.4
    # 성긴·파스텔 그림은 받쳐야 읽힌다 → 베드 계열
    if it.airy or lk.pale > 0.4:
        sc["graphic_bed"] += 0.8
        sc["splash"] += 0.3
        sc["minimal"] -= 0.5
    # 대각 포즈·누운 인물 → 사선 흐름
    if abs(it.axis[0]) > 0.3 or it.elongation > 2.2:
        sc["diagonal_flow"] += 0.7
    # 뾰족·기계적 인상 → 모터스포츠·사선, 둥글면 스플래시·베드
    if it.impression == "sharp":
        sc["motorsport"] += 0.6
        sc["diagonal_flow"] += 0.3
    elif it.impression == "soft":
        sc["splash"] += 0.5
        sc["graphic_bed"] += 0.2
    # 디테일이 빽빽하면 장식을 줄인다
    if it.detail_mean > 0.45:
        sc["minimal"] += 0.4
        sc["splash"] -= 0.4
    return [n for n, _ in sorted(sc.items(), key=lambda kv: (-kv[1], kv[0]))]
