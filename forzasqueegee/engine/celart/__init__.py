"""cel 노선 1단 — 원화를 셀 애니 중간 표현(평면 색 영역 목록)으로 재해석.

여기서 정하는 것은 **무엇을 그릴 것인가**뿐이다 (몇 장으로 그릴지는 `celfit`,
값이 되는지는 `price`). 사람이 셀 일러스트를 다시 그릴 때처럼, 의미 있는 큰
색 덩어리와 시각적으로 중요한 작은 색면을 계층적으로 갈라 놓는다.

## 모듈 구성 — 아래로 갈수록 위를 쓴다

    model       자료형 `Region`·`CelArt` (영역 지도 + 그리기 순서 + 선화)
    geodesic    측지 전파 — 장벽을 안 넘는 최근접 (선 귀속·스냅이 함께 쓴다)
    prep        배경 채움 · 평활 · 라벨 다수결
    inkfill     §1 선 제거 = 면 귀속 복원 — 선을 장벽으로, 양쪽 면에 나눠 준다
    palette     §5 팔레트 — 색 표현만. 꼬리(작은 고색차)를 끝까지 지킨다
    atoms       §2 원자 — watershed + 그리드 재분할 (**최종 영역이 아니다**)
    dense       §4 밀집 시각 특징 (선택) — 없으면 없는 대로 돈다
    marks       무늬 보호 조각 — 작지만 없어지면 특징이 사라지는 영역
    rag         §3 영역 인접 그래프 + MDL 병합 — 무엇이 한 덩어리인가
    legacy      RAG 이전의 문턱 사다리 병합 (§14 a0 대조군)
    snap        §10 획 스냅(측지) · 영역 표 · 경계 펴기
    decompose   진입점 `decompose`

밖에서 쓰는 것은 이 파일이 다시 내보내는 이름뿐이다.
"""

from __future__ import annotations

from .decompose import _MAX_REGIONS, decompose
from .inkfill import faces_of
from .marks import mark_mask
from .model import _ALPHA_OPAQUE, CelArt, Region
from .prep import _fill_bg_nearest
from .snap import (rebuild_regions, region_table, regularize,
                   snap_labels_to_ink, with_regions)

__all__ = [
    "CelArt", "Region", "decompose", "_MAX_REGIONS", "_ALPHA_OPAQUE",
    "_fill_bg_nearest", "mark_mask", "faces_of",
    "snap_labels_to_ink", "rebuild_regions", "regularize", "region_table",
    "with_regions",
]
