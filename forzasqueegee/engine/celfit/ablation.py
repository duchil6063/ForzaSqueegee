"""단계 스위치 — 엔진의 각 축을 **끄고 켜서** 무엇이 무엇을 했는지 가른다.

전부 기본 켬이고, 끄면 그 축이 들어오기 전의 동작으로 돌아간다. 회귀 계측이
이 스위치로 판을 여러 벌 구워 나란히 세운다. 어느 축이 어떤 오차를
줄였는지는 그 표가 답한다.

    FS_LRE_EVIDENCE  선 증거 (basic 신뢰도 · detail 판)
    FS_LRE_GRAPH     획 그래프 (비용 기반 이어긋기 · 역할 판정)
    FS_DESC_VOCAB    도형 서술자 어휘 (가는 막대 포함)
    FS_LRE_CAND      후보 경쟁 (끄면 곡선 게이트 그대로의 탐욕 배치)
    FS_LINE_WIDTH    획 폭 충실도 (놓인 폭을 원화 띠에 맞춘다)
    FS_LINE_CORE_GAIN 이상 띠 밖 선 픽셀의 이득 배율 (1 = 옛 동작)
    FS_CORE_PROFILE  이상 띠를 폭 프로파일로 (끄면 폭 중앙값 한 수)
    FS_LINE_WFLOOR   폭 바닥을 어휘 실측으로 (끄면 둥근사각 폭이 바닥)
    FS_JOIN_MAIN     접합점에서 주요 윤곽 우선 (끄면 가장 싼 짝 하나)
    FS_CORNER_AWARE  각 보존 평활 (끄면 창을 통째로 거는 균일 평활)
    FS_STROKE_INTENT 각에서 끊기 — 분절이 이탈 최대점 대신 의도된 각을 쓴다
    FS_STROKE_SPAN   한 획의 도형 상한을 길이에 비례시킨다 (0 = 상수 상한)
    FS_WIDTH_PROFILE 폭 프로파일 (0이면 폭을 중앙값 한 수로만 본다)
    FS_STROKE_END    끝 뭉툭함 (0이면 물방울·쐐기도 획 도형으로 쓴다)
    FS_STROKE_BULGE  몸통 배부름 (0이면 비등방이 만든 쐐기도 획으로 쓴다)
    FS_CHAIN_JOINT   사슬 이음 정리 (끄면 마디가 각자 제 채점만 보고 선다)
    FS_JOINT_DESCEND 이음을 **놓는 동안** 맞춘다 (끄면 다 놓은 뒤 정리만)
    FS_STROKE_GRAMMAR 획 도형 문법을 두 노선 공통으로 (끄면 line 노선만)
    FS_TEX_SIMPLIFY  무늬 단순화 (기본 **꺼짐** — 근거는 `policy` 문서)

**여기는 선 재구성 엔진의 축만 있다.** cel 노선의 재해석·채움 축(§1~§12)은
`engine.celaxes`에 따로 있다 — 두
엔진이 서로를 안 임포트해야 하므로(celfit → celart 한 방향) 스위치도 갈라
둔다.
"""

from __future__ import annotations

import os


def _on(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default) != "0"


def evidence() -> bool:
    """이진화 전 신뢰도·detail 판을 증거로 쓸까 (끄면 마스크만)."""
    return _on("FS_LRE_EVIDENCE")


def graph() -> bool:
    """비용 기반 이어긋기·역할 판정을 쓸까 (끄면 접선 각도 이음 + 3분류)."""
    return _on("FS_LRE_GRAPH")


def candidates() -> bool:
    """후보 경쟁을 쓸까 (끄면 곡선 자격 게이트를 그대로 둔 탐욕 배치)."""
    return _on("FS_LRE_CAND")


def width() -> bool:
    """놓인 획 폭을 원화 띠에 맞출까 (끄면 밴드 안에서 공짜로 굵어진다).

    끄면 종전 동작이다: 1px 물림 면제와 이어긋기 값이 **성분 전체**의 선
    픽셀에 걸려, 조밀한 선망에서 도형이 이웃 가닥까지 먹는 폭으로 부푼다
    (`scoring._Scorer.set_band`·`stroke._STROKE_WMAX` 문서).
    """
    return _on("FS_LINE_WIDTH")


def core_profile() -> bool:
    """이상 띠를 **폭 프로파일**로 지을까 (끄면 폭 중앙값 한 수 — `engine._core_band`).

    끄면 종전 동작이다: 획 하나의 띠 두께가 한 수라, 가는 쪽에서는 부푸는
    것이 공짜고 굵은 쪽에서는 제 몸통이 띠 밖으로 나간다.
    """
    return _on("FS_CORE_PROFILE")


def core_gain() -> float:
    """이상 띠 **밖** 선 픽셀의 이득 배율 (`scoring._CORE_GAIN`)."""
    from .scoring import _CORE_GAIN

    return _CORE_GAIN


def profile() -> bool:
    """폭을 **프로파일**로 볼까 (끄면 중앙값 한 수 — `stroke._W_PROF`).

    끄면 종전 동작이다: `placed_profile`이 가운데 80%만 재므로 끝이 뾰족한
    물방울이 테이퍼 게이트를 그대로 통과한다 (실측 획 도형의 35%).
    """
    return (float(os.environ.get("FS_WIDTH_PROFILE", "1")) > 0.0
            and float(os.environ.get("FS_STROKE_END", "0.55")) > 0.0)


def grammar() -> bool:
    """획 도형 문법(테이퍼·가늘기·폭·끝 뭉툭함)을 **cel 노선에도** 걸까.

    끄면 종전 동작이다: line 노선만 걸리고 cel의 선 도안은 잎사귀·쐐기를
    그대로 쓴다. 선은 모든 면 위에 마지막으로 얹히므로 그 티가 셀에서도
    그대로 보인다 — 실측(01)에서 앞머리 선이 검은 쐐기 덩어리로 나왔다.
    면 영역의 가는 잔여 경로(`fill._fit_bars`)는 이 문법 밖이다.
    """
    return _on("FS_STROKE_GRAMMAR")


def joint() -> bool:
    """사슬 이음을 맞출까 (끄면 마디가 각자 제 채점만 본다 — `chain` 문서)."""
    return _on("FS_CHAIN_JOINT")


def corner() -> bool:
    """의도된 각에서 평활을 물릴까 (끄면 모든 경로를 같은 창으로 민다).

    끄면 종전 동작이다: 5탭이 눈꼬리·턱선·옷 주름의 꺾임을 5px에 걸쳐 둥글게
    깎고, 획 양끝 표본도 창 몫만큼 잃는다 (`skeleton.smooth_path` 문서).
    """
    return _on("FS_CORNER_AWARE")


def intent() -> bool:
    """분절을 **의도된 각**에서 할까 (`FS_STROKE_INTENT=0`이면 이탈 최대점).

    끄면 종전 동작이다: 쪼갤 자리·RDP 마디·DP 마디 후보가 전부 현에서의 최대
    이탈만 본다 — 매끈한 호의 한가운데가 마디가 되어 그 자리에 각이 서고,
    다리가 짧은 진짜 꺾임은 마디가 안 되어 도형 하나에 뭉개진다
    (`intent` 문서).
    """
    from . import intent as _I

    return _I.on()


def names() -> dict:
    """지금 켜진 축 — report에 실어 판을 되짚는다."""
    return {"evidence": evidence(), "graph": graph(), "candidates": candidates(),
            "width": width(), "corner": corner(), "profile": profile(),
            "intent": intent(),
            "span": float(os.environ.get("FS_STROKE_SPAN", "0.0225")),
            "core_profile": core_profile(), "core_gain": core_gain(),
            "wfloor": _on("FS_LINE_WFLOOR"), "join_main": _on("FS_JOIN_MAIN"),
            "bulge": float(os.environ.get("FS_STROKE_BULGE", "1.8")),
            "joint": joint(),
            "joint_descend": os.environ.get("FS_JOINT_DESCEND", "1") != "0",
            "grammar": grammar(),
            "desc_vocab": _on("FS_DESC_VOCAB"),
            "texture_simplify": _on("FS_TEX_SIMPLIFY", "0")}
