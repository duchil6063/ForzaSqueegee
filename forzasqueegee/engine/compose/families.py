"""구성 계열 — 꾸밈 문법 **한 벌이 아니라 여러 벌**이다.

같은 도안이라도 사람마다 다른 리버리를 만든다. 계열은 그 갈래의 뼈대다 —
배경 색면의 구조, 흐름, 밀도, 여백, 모티프 크기 분포, 다른 면으로 이어지는
법. 어느 계열이 이 도안에 맞는지는 **후보를 다 지어 점수로** 고른다
(`design.compose_design`) — 여기서는 도안 뜻(`DesignIntent`)으로 후보 순서만
정한다.
"""

from __future__ import annotations

from dataclasses import dataclass

from .graph import DEFAULT_GRAMMAR
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
    clutter: tuple[float, float]   # 모티프 커버리지 목표 구간 (밴드 빈 면적 대비 — 판·로커는 안 센다)
    echo: bool                  # 그래픽 에코 (결 조각·샤드·블록)
    flows: tuple[str, ...]      # 흐름 후보 — auto · rear · front
    other_density: float        # 다른 면 모티프 수 배율
    torn: bool = False          # 베드 흐름 끝을 뜯는다 (스플래시)
    text_budget: float = 1.0    # 글자 장수 예산 배율 (여백 계열은 글자도 절제한다)
    # **매크로 어휘 짝** — (주 색면, 가로지르는 짝). 계열마다 둘 안팎을 후보로
    # 돌린다 (`macro.KINDS`). 이것이 계열을 "완성된 도형 묶음"에서 **문법 프리셋**
    # 으로 낮추는 자리다: 계열은 어느 어휘로 시작할지만 말하고, 실제 매개변수는
    # 인물이 정하고(`macro.plan`) 어느 짝이 이기는지는 점수가 정한다.
    macro: tuple[tuple[str, str], ...] = (("ribbon", "ribbon"),)
    # 계열이 **덧붙이는** 관계 문법 (`graph`) — 공통 문법 위에 더한다.
    # (관계, 노드 a, 노드 b, 가중치). 없는 노드를 가리키면 안 센다.
    grammar: tuple[tuple[str, str, str, float], ...] = ()
    # **색면 스택** — 블록 위에 얹는 조각 (`stack.PIECES`: belt · arch · pin ·
    # edge · gap). 후보 축이 아니라 계열의 문법이다 — 사람 판의 스택은
    # 레이싱·그래픽·스플래시라는 계열이 정하지 도안마다 고르는 것이 아니다.
    stack: tuple[str, ...] = ()
    # 이 계열이 도는 **팔레트 변종** (`roles.ROLE_VARIANTS`). 비면 공통 셋
    # (`design.VARIANTS_TRIED`)이다 — 검정 바탕의 형광 액센트처럼 계열이 곧
    # 팔레트인 자리만 제 것을 든다.
    variants: tuple[str, ...] = ()

    def rels(self) -> tuple[tuple[str, str, str, float], ...]:
        """이 계열이 지키려는 관계 전부 — 공통 문법 + 제 몫."""
        return DEFAULT_GRAMMAR + self.grammar


FAMILIES: dict[str, Family] = {
    # 비움 — 큰 베드 없이 얇은 슬래브 하나와 잔 모티프 몇, 여백이 주역
    "minimal": Family("minimal", bed="slab", bed_level=0.35, motif_n=7,
                      tier_scale=0.75, rocker=False, top_stripe=False, front_n=1,
                      empty_target=0.92, clutter=(0.03, 0.12), echo=False,
                      flows=("auto",), other_density=0.45, text_budget=0.5,
                      macro=(("ribbon", "none"), ("ribbon", "ribbon")),
                      # 덧붙일 것이 없다 — 공통 문법(품되 삼키지 않는 판 · 무리와
                      # 여백의 맞섬)이 그대로 이 계열의 이치다
                      ),
    # 그래픽 베드 — 인물 뒤 큰 색면이 구도를 잡고 모티프는 그 가장자리에
    "graphic_bed": Family("graphic_bed", bed="plate", bed_level=0.75, motif_n=14,
                          tier_scale=0.95, rocker=True, top_stripe=False, front_n=2,
                          empty_target=0.75, clutter=(0.10, 0.28), echo=True,
                          flows=("auto", "rear", "front"), other_density=0.8,
                          macro=(("split", "ribbon"), ("split", "none"), ("ribbon", "blade")),
                          # 벨트 블랙아웃 + 아치 날 + 찢긴 가장자리 + 홈 — 사람
                          # 그래픽 판의 바닥 (structure2 실측: 띠 13장·6색)
                          stack=("belt", "arch", "edge", "gap", "streak"),
                          # 무리는 판 **가장자리**에 선다 — 판 위에 얹으면 판이
                          # 얼룩이 되고 무리도 안 읽힌다
                          grammar=(("avoids", "motif", "macro0", 0.8),)),
    # 사선 흐름 — 대각 판 둘이 인물을 지나 흐르고 모티프가 그 결을 따른다.
    # 프리셋 짝은 없다 (자동과 CLI 레버 `family`로만 선다) — 밝은 차의 자동 후보
    # 다섯이 종전 그대로여야 자동 판이 기준판과 같다 (W14D→W15A 실측: 이것을 빼니
    # 33판 중 17판이 갈리고 minimal이 9→12, 차 H 중앙 .744→.586).
    "diagonal_flow": Family("diagonal_flow", bed="wedge", bed_level=0.65, motif_n=16,
                            tier_scale=0.9, rocker=True, top_stripe=True, front_n=3,
                            empty_target=0.65, clutter=(0.12, 0.32), echo=True,
                            flows=("auto", "rear"), other_density=1.0,
                            macro=(("blade", "chevron"), ("split", "stack"), ("blade", "none")),
                            # 아치에서 솟는 날 + 사선을 따르는 핀 + 스월 가장자리
                            stack=("arch", "pin", "edge", "streak"),
                            # 사선 둘이 서로를 **가로질러야** 흐름이 난다
                            grammar=(("counter_to", "macro1", "macro0", 1.0),)),
    # 다크 그래피티 — 검정 바탕 위에 형광 사선 날 둘이 인물을 지나고, 큰
    # 워드마크가 한 요소다 (사람 판의 "검정 + 형광 액센트 + 큰 이름"). 사선
    # 흐름 계열의 뼈대(쐐기 판·날·핀·스월 가장자리)를 검정 바탕용으로 조인 것 —
    # 로커는 없다 (검은 차에 검은 로커는 없는 것과 같다), 모티프는 적고 크다.
    # 자동에서는 **검은 바탕에서만** 후보다 (`design.DARK_BASE_LUM`).
    "dark": Family("dark", bed="wedge", bed_level=0.55, motif_n=10,
                   tier_scale=0.95, rocker=False, top_stripe=True, front_n=2,
                   empty_target=0.72, clutter=(0.08, 0.24), echo=True,
                   flows=("auto", "rear"), other_density=0.8, text_budget=1.4,
                   macro=(("blade", "chevron"), ("blade", "none"), ("split", "stack")),
                   # 사선을 따르는 핀 + 스월 가장자리 + 인물 뒤 스트릭
                   stack=("pin", "edge", "streak"),
                   # 사선 둘이 서로를 **가로질러야** 흐름이 난다
                   grammar=(("counter_to", "macro1", "macro0", 1.0),),
                   # 형광 액센트가 이 계열의 팔레트다 (`roles` neon)
                   variants=("neon", "primary")),
    # 모터스포츠 — 로커·스트라이프 등 직선 요소, 베드는 낮은 슬래브, 모티프는 적게
    "motorsport": Family("motorsport", bed="slab", bed_level=0.55, motif_n=9,
                         tier_scale=0.8, rocker=True, top_stripe=True, front_n=1,
                         empty_target=0.80, clutter=(0.08, 0.24), echo=True,
                         flows=("rear", "front"), other_density=0.6, text_budget=0.8,
                         macro=(("stack", "corner"), ("stack", "none"), ("ribbon", "stack")),
                         # 벨트 띠 + 핀스트라이프 + 띠 속의 홈 (레이싱 그래픽)
                         stack=("belt", "pin", "gap"),
                         # 로커가 띠를 이어 받는다 (레이싱 그래픽의 직선 계열)
                         grammar=(("continues", "rocker", "macro0", 0.6),)),
    # 스플래시 — 덩어리 베드에 뜯긴 가장자리, 모티프가 많고 인물 위로도 얹힌다
    "splash": Family("splash", bed="blob", bed_level=0.8, motif_n=22,
                     tier_scale=1.05, rocker=True, top_stripe=False, front_n=4,
                     empty_target=0.5, clutter=(0.18, 0.45), echo=True,
                     flows=("auto",), other_density=1.25, torn=True,
                     macro=(("burst", "ribbon"), ("sweep", "blade"), ("burst", "none")),
                     # 아치 날 + 튄 물감 가장자리 (찢김·스플래시는 무늬 도형이 낸다)
                     stack=("arch", "edge", "streak"),
                     # 전경 조각이 인물을 스치고 지난다 (장면 안의 인물)
                     grammar=(("overlaps", "front", "hero", 0.6),)),
}


FAMILY_NAMES = tuple(FAMILIES)


def rank_families(it: DesignIntent, lk: Look, person_frac: float,
                  base_lum: float | None = None) -> list[str]:
    """이 도안에 맞는 계열 **후보 순서** (전부 후보다 — 점수가 최종을 정한다).

    `person_frac`은 인물이 차체 밴드 폭을 얼마나 덮나 (0~1). `base_lum`은 베이스
    도색의 명도 (0~1) — 다크 계열은 검은 바탕에서만 앞에 선다.
    """
    sc: dict[str, float] = {n: 0.0 for n in FAMILIES}
    # 다크 그래피티는 **바탕이 검을 때**의 문법이다 — 흰 차 위 형광 사선은
    # 그래피티가 아니라 얼룩이다 (자동에서는 `design`이 밝은 바탕에서 아예 뺀다).
    # 바탕을 모르면(옛 호출) 중립이다.
    if base_lum is not None:
        sc["dark"] += 1.0 if base_lum < 0.25 else -0.8
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
    # 대각 포즈·누운 인물 → 사선 흐름 (다크도 같은 뼈대다)
    if abs(it.axis[0]) > 0.3 or it.elongation > 2.2:
        sc["diagonal_flow"] += 0.7
        sc["dark"] += 0.7
    # 뾰족·기계적 인상 → 모터스포츠·사선, 둥글면 스플래시·베드
    if it.impression == "sharp":
        sc["motorsport"] += 0.6
        sc["diagonal_flow"] += 0.3
        sc["dark"] += 0.3
    elif it.impression == "soft":
        sc["splash"] += 0.5
        sc["graphic_bed"] += 0.2
    # 디테일이 빽빽하면 장식을 줄인다
    if it.detail_mean > 0.45:
        sc["minimal"] += 0.4
        sc["splash"] -= 0.4
    return [n for n, _ in sorted(sc.items(), key=lambda kv: (-kv[1], kv[0]))]
