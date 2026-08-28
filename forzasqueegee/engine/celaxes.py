"""cel 노선의 **단계 스위치** — 축을 하나씩 끄고 켜서 무엇이 무엇을 했는지 가른다.

전부 기본 켬이고, 끄면 그 축이 들어오기 전의 동작으로 돌아간다. 회귀 계측이
이 스위치로 판을 여러 벌 구워 나란히 세운다. 어느 축이 어떤 오차를 줄였는지는 그 표가 답한다.

    FS_CEL_INKFILL   §1  선 제거를 면 귀속 복원으로 (끄면 유클리드 최근접)
    FS_CEL_OVERSEG   §2  원자 그리드 재분할 (끄면 watershed 원자 그대로)
    FS_CEL_RAG       §3  MDL 그래프 병합 (끄면 문턱 사다리 — `celart.legacy`)
    FS_CEL_DENSE     §4  밀집 시각 특징을 병합 보조 증거로 (모델 있을 때만)
    FS_CEL_TAIL      §5  팔레트 꼬리 센터 보강
    FS_CEL_DESCFIT   §6  서술자 후보 순위 — **기본 꺼짐** (아래 근거)
    FS_CEL_LAYERED   §7  영역 전체를 덮는 바탕 도형 먼저 (끄면 봉우리 탐욕)
    FS_CEL_SETCOVER  §8  후보 집합 덮개 · 작은 빔 탐색
    FS_CEL_PAIR      §9  이웃 영역 공유 경계 (이음 당김)
    FS_CEL_SNAP_GEO  §10 획 스냅을 측지로 (끄면 유클리드 최근접)
    FS_CEL_RESIDUAL  §12 잔차를 **기존 도형을 움직여** 먼저 고친다
    FS_CEL_GROWFIRST §14 잔여를 **사기 전에** 기존 도형을 늘려 먹는다

`FS_CEL_NEW=0` 한 방이면 전부 끈 옛 동작이다 (a0 대조판).

**§6만 기본이 꺼져 있다.** 서술자 순위 자체는 제 일을 한다 — 잔여를 닮은
도형이 앞에 서고, 굽은 껍질에 초승달이 한 장으로 선다. 다만 그 순위가 여는
넓은 어휘(불투명·뚱뚱한 76종)의 값이 회귀 열 장에서 안 나왔다: 최종 설정에서
이 축만 빼 보면(대조판 b1-nodesc) 총 도형이 1,984 → 1,997로
**늘어나는 대신** 선/색면 틈이 580 → 659px, 중요도 가중 오차가 15.09 → 15.28로
나빠진다. 어휘가 넓어지면 가장자리가 너덜한 도형이 잔여를 잘 주워 점수로
이기지만 육안이 나빠진다는 종전 관찰(`celfit.vocabulary` 문서)이 순위를 얹어도
남는다는 뜻이다. 기계는 그대로 두고 스위치만 꺼 둔다 — 어휘가 바뀌거나
카탈로그가 늘면 다시 재 볼 자리다 (`FS_CEL_DESCFIT=1`).
"""

from __future__ import annotations

import os

_AXES = ("INKFILL", "OVERSEG", "RAG", "DENSE", "TAIL", "DESCFIT", "LAYERED",
         "SETCOVER", "PAIR", "SNAP_GEO", "RESIDUAL", "GROWFIRST")


# 기본이 꺼진 축 — 근거는 위 문서 (회귀 열 장의 축 빼 보기 실측)
_OFF = ("DESCFIT",)


def on(axis: str) -> bool:
    """이 축이 켜져 있나 — `FS_CEL_NEW=0`이면 개별 스위치와 무관하게 전부 꺼짐."""
    if os.environ.get("FS_CEL_NEW", "1") == "0":
        return False
    return os.environ.get("FS_CEL_" + axis,
                          "0" if axis in _OFF else "1") != "0"


def names() -> dict:
    """지금 켜진 축 — report에 실어 판을 되짚는다."""
    return {a.lower(): on(a) for a in _AXES}
