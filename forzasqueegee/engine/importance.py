"""중요도 가중 — "어디가 눈에 띄는가"를 픽셀 맵으로 (모델 없음, 대상 불문).

`prune_impact`는 시각 영향을 "소유 px × 드러나는 색과의 ΔE"로 잰다. 이 잣대는
같은 색차를 어디서든 같게 본다 — 그래서 **평평한 뺨 위의 눈동자 하이라이트
30px와 머리칼 속 가닥 30px가 동급**이다. 12차 판정에서 확정된 사실이 이것과
어긋난다: 눈에 띄는 정도는 색차 단독이 아니라 **주변이 평평한가**에 달렸다
(코 그림자는 색차가 한 자리인데도 평평한 볼 위라 사라지면 바로 보인다).

여기서는 그 관찰을 시각 마스킹의 교과서 형태로 옮긴다 —

    W = (작은 창의 색 편차) / (상수 + 큰 창의 색 편차)

작은 창은 그 자리의 대비, 큰 창은 주변의 소란(마스킹)이다. 평평한 면 위의
고대비 특징은 크게, 이미 시끄러운 곳(머리칼 속)의 같은 대비는 작게 나온다.
얼굴을 찾지 않으므로 초상·전신·메카·소품 어디서나 같은 규칙으로 돈다
(전신 구도에서도 눈이 이미지 전체 최고값으로 나온다 — 얼굴 검출 없이).

쓰는 쪽은 이 맵을 **곱 배수로만** 쓴다. 상하한을 두고 √로 압축하는 것이
핵심 — 원 비율은 꼬리가 중앙값의 7배까지 가서, 그대로 곱하면 가중이 아니라
컷 순서를 지배해 버린다.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

# 창 크기는 작업 해상도 비례 — 1200에서 7px·41px. 스윕(창 7/41·11/61·15/91 ×
# 배수 0.5~2.0·0.7~1.5·0.85~1.25)에서 두 타깃 모두 개선인 설정은 7/41과 11/61
# 둘뿐이었고, 배수를 키우거나(0.5~2.0) 줄이면(0.85~1.25) joy가 악화로 뒤집힌다.
# 설정 사이 차이는 노이즈 수준이라 소수점으로 고를 자리가 아니다 (육안 판정 채택)
_SMALL_REL, _BIG_REL = 0.006, 0.034
_FLOOR = 4.0          # 큰 창 편차의 바닥 — 완전 평면(편차 0)에서 발산 방지
_LO, _HI = 0.7, 1.5   # 곱 배수 상하한


def _odd(v: int) -> int:
    return max(3, int(v) | 1)


def _activity(lab: np.ndarray, k: int) -> np.ndarray:
    """창 k 안의 Lab 표준편차 합 = 그 자리 색이 얼마나 요동치는가."""
    m = cv2.boxFilter(lab, -1, (k, k))
    m2 = cv2.boxFilter(lab * lab, -1, (k, k))
    return np.sqrt(np.maximum(m2 - m * m, 0)).sum(2)


def _raw(rgb: np.ndarray, sel: np.ndarray) -> tuple[np.ndarray, float]:
    """정규화 전 원 비율과 캔버스 안 중앙값."""
    s = max(sel.shape)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    raw = _activity(lab, _odd(round(_SMALL_REL * s))) / (
        _FLOOR + _activity(lab, _odd(round(_BIG_REL * s))))
    return raw, (float(np.median(raw[sel])) if sel.any() else 0.0)


def masking_weight(rgb: np.ndarray, sel: np.ndarray) -> np.ndarray:
    """RGB(작업 해상도) → 곱 배수 맵 (float32, 캔버스 밖은 1.0).

    이미지마다 중앙값으로 정규화한다 — 절대 편차 수치는 그림체마다 다르고,
    타깃별 상수를 두지 않기 위한 것이다 (한 버튼 원칙).
    """
    raw, med = _raw(rgb, sel)
    if med <= 1e-6:
        return np.ones(sel.shape, np.float32)
    out = np.clip(np.sqrt(raw / med), _LO, _HI).astype(np.float32)
    out[~sel] = 1.0
    return out


# 값 맵(가격 설계) — 컷 **순서**가 아니라 "살까 말까"를 가르므로 압축하면
# 안 된다. 순서만 정하는 `masking_weight`는 꼬리가 순서를 지배하는 것을
# 막으려고 √+0.7~1.5로 눌러 두었는데, 값으로 쓰면 그 압축이 곧 "눈이든
# 머리칼이든 값이 거의 같다"가 되어 가격이 균일 조악화로 떨어진다.
#
# 압축을 푸는 데 그치지 않고 **볼록하게(γ>1)** 쓴다. 레퍼런스의 배분이
# 그것을 요구한다: 사람은 몸통 블록인 전체에 40장을 쓰고 눈 하나에 50장을
# 쓴다 (프리렌 영상 22s ↔ 44s). 대비 비율에 선형이면 눈처럼 **작지만 결정적인**
# 구조가 총량에서 구조적으로 진다 — 전신 구도에서 실제로 그랬다(06 라이덴:
# 눈 상자의 값 비율이 중앙의 1.4~2.8배인데 넓이가 작아 앞머리 가닥 획이
# 전부 가격에 밀렸고, 눈이 통째로 뭉갠 덩어리가 됐다). γ=2면 그 2.8배가
# 7.8배가 되어 값을 한다. 중앙값은 γ와 무관하게 1이라 λ의 뜻은 그대로다.
_V_GAMMA = float(os.environ.get("FS_V_GAMMA", 2.0))
_V_LO = float(os.environ.get("FS_V_LO", 0.0625))    # 1/16
_V_HI = float(os.environ.get("FS_V_HI", 16.0))


def place_weight(rgb: np.ndarray, sel: np.ndarray) -> np.ndarray:
    """RGB → **값 맵** (float32, 캔버스 밖 0) — 이 픽셀을 맞히는 것의 값어치.

    중앙값이 1이 되게 정규화하므로 가격 λ의 단위는 "중앙 픽셀 몇 개어치"다.
    캔버스 밖은 0 — 배경을 맞히는 데는 값이 없다 (침범 벌점은 따로 문다).
    """
    raw, med = _raw(rgb, sel)
    if med <= 1e-6:
        return sel.astype(np.float32)
    out = np.clip((raw / med) ** _V_GAMMA, _V_LO, _V_HI).astype(np.float32)
    out[~sel] = 0.0
    return out
