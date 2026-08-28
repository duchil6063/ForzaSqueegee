"""꾸밈 그룹 — 베드 · 띠 · 산포를 **캔버스 한 장**으로 조립한다."""

from __future__ import annotations

from dataclasses import replace

from ..catalog import Catalog
from ..model import Layer, LayerPlan
from .look import Look
from .palette import accent_color, accent_third, accent_tint
from .vocabulary import edge_shapes, motif_shapes
from .scatter import (
    DECO_ANCHOR_GAP, DECO_FRONT_N, DECO_FRONT_SIZE, DECO_TIER_SIZE, deco_layers)
from .bands import stripe_layers
from .roof import ROOF_DARK
from .place import _refit_canvas


# 띠가 캔버스 끝에 딱 붙으면 기울일 때 모서리가 밖으로 나간다 — 조금 줄인다
DECO_FRAME_FILL = 0.98


def compose_deco(plan: LayerPlan, lk: Look, cat: Catalog,
                 car_rgb: tuple[int, int, int] | None,
                 frame_box: tuple[float, float, float, float],
                 person_box: tuple[float, float, float, float],
                 stripes: bool = True,
                 deco: bool = True, front: bool = False,
                 anchor: float = 0.0,
                 halo: tuple[int, int, int] | None = None,
                 drawable_at=None, family: str | None = None) -> LayerPlan:
    """**꾸밈 그룹** — 베드 + 띠 + 산포. 자는 **인물이 아니라 면**이다.

    그룹 분리: 도안은 도안 그대로 제 그룹으로 남고, 꾸밈은 이 그룹이 쥔다. 사람
    제작자도 요소별 그룹으로 관리한다 (ARIS 레이어 리스트 — 로고·모티프가
    각각 별개 그룹; EVELYNE의 백합 = 30장짜리 미니 그룹).

    **왜 인물 좌표를 안 쓰나** (실차 캡처): 꾸밈이 인물과 변환 하나를 나눠 쓰면
    띠가 면을 덮으려고 캔버스를 3.4배 넘어간다 (줄리아 실측: x −1,732~1,311).
    캔버스는 **900유닛 고정**이라 그 밖은 게임이 안 그린다 — 옆면에 띠도 산포도
    없이 베드만 남는다 (베드는 ±433이라 유일하게 캔버스 안이다). 인물
    스케일(0.31)에 매인 한 꾸밈이 덮을 수 있는 최대는 900×0.31 = **283유닛**,
    옆면의 3분의 1이라 원리적으로 못 덮는다.

    그래서 꾸밈은 **제 스케일**로 앉는다. `frame_box`는 옆면 차체 밴드를 이
    캔버스 좌표로 옮긴 상자(가운데가 원점, 폭 900)이고 `person_box`는 그 안에서
    인물이 덮는 자리다. 띠·산포는 프레임을 자로 쓰고(면을 덮는다) 베드와 모티프
    크기는 인물을 자로 쓴다(인물을 띄운다).
    """
    frame = replace(lk, box=frame_box, hull=None)
    person = replace(lk, box=person_box, hull=None)
    # 무리는 인물 **옆**에 선다 — 뭉치는 자리를 차체 밴드 한가운데에 두면 그
    # 자리가 인물 상자 안이라 무리가 인물을 피해 **양쪽으로 갈라진다**
    # (2026-08-22 판정: 꽃 무리가 인물 앞에, 큰 꽃 한 송이가 인물 뒤에 따로
    # 섰다). 레퍼런스의 무리는 갈라지지 않고 인물 뒤쪽 리어 쿼터에 **한 덩이로**
    # 붙는다 (EVELYNE의 백합 · RIN의 꽃 · 수이세이의 별). 그래서 뭉치는 자리를
    # 인물 상자 **바깥 모서리**로 옮긴다.
    rear = 1.0 if anchor >= 0 else -1.0
    spread = 0.50 * DECO_FRAME_FILL * frame.w
    edge = person_box[2] if rear > 0 else person_box[0]
    ax = edge + rear * DECO_ANCHOR_GAP * person.w
    # 무리 한가운데가 면 끝에 너무 붙으면 **최대형이 범퍼 밖으로 반쯤 걸린다**
    # (2026-08-22 판정: 도안 넷 다 큰 별·꽃 하나가 뒤끝에서 잘렸다). 최대형
    # 반지름만큼은 안쪽에 둔다.
    hero = DECO_TIER_SIZE[0] * person.h / 2.0
    lim = max(0.0, DECO_FRAME_FILL * frame.w / 2.0 - hero)
    anchor = max(-lim, min(lim, ax)) / max(1e-6, spread)
    main = accent_color(lk, car_rgb)
    # 산포는 **유채색 세 벌**이다 (주색 + 밝은 자매 + 색조가 갈린 셋째) —
    # 레퍼런스의 배경은 예외 없이 세 색 이상이고, 두 벌은 한 색조의 농담이라
    # 배경이 단벌로 읽힌다. 근검정/근백은 안 섞는다 (어두운 베이스에서 검은
    # 구멍으로 읽힌다 — 미리보기 실측).
    trio = (main, accent_tint(main, car_rgb), accent_third(main, lk, car_rgb))
    layers: list[Layer] = []
    if front:
        # 전경 벌 — 도안 **위**에 얹는 몇 장뿐이다. 배경 벌과 정확히 반대 조건이다
        # (`over=True`): 인물에 **걸치는 것만** 남기고, 크기는 잔것으로 줄인다.
        # 레퍼런스의 전경은 팔·다리를 스치고 지나가는 꽃 몇 송이지 얼굴을 덮는
        # 큰 판이 아니다.
        return _refit_canvas(LayerPlan(
            source_image=plan.source_image, image_size=plan.image_size,
            units_per_px=plan.units_per_px,
            layers=deco_layers(frame, trio, cat, n=DECO_FRONT_N,
                               spread=0.44 * DECO_FRAME_FILL * frame.w,
                               size_ref=person.h * DECO_FRONT_SIZE,
                               anchor=anchor, halo=halo,
                               shapes=motif_shapes(lk, cat, family),
                               drawable_at=drawable_at,
                               avoid=person_box, over=True)), cat)
    if stripes:
        # 로커 밴드는 **맨 아래**다 — 판·산포가 그 위에 얹힌다. 색은 어두운
        # 쪽이다 (레퍼런스의 하부 투톤은 수이세이·ARIS·Evo IX·KOTONE·EVELYNE이
        # 전부 검정 계열이다) — 이미 어두운 차에서는 아예 안 깐다.
        layers = stripe_layers(frame, ROOF_DARK, cat,
                               shapes=edge_shapes(lk, cat, family), car=car_rgb,
                               length=DECO_FRAME_FILL * frame.w) + layers
    if deco:
        # 산포는 **프레임 절반**을 반경으로 — 띠가 덮는 범위를 그대로 덮어야
        # 차 전체가 접착된다. 레퍼런스의 별·꽃은 범퍼 모서리까지 간다.
        # 인물 상자는 비운다 (`avoid`) — 배경 모티프는 인물 **곁**에 선다.
        layers += deco_layers(frame, trio, cat,
                              spread=0.50 * DECO_FRAME_FILL * frame.w,
                              size_ref=person.h, anchor=anchor, halo=halo,
                              shapes=motif_shapes(lk, cat, family),
                              drawable_at=drawable_at, avoid=person_box)
    out = LayerPlan(source_image=plan.source_image, image_size=plan.image_size,
                    units_per_px=plan.units_per_px, layers=layers)
    return _refit_canvas(out, cat)
