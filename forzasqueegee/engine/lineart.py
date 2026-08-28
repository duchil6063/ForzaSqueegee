"""신경망 선화 추출 — AniLines basic (ONNX, MIT) 래퍼.

cel 노선의 획 품질은 선화의 품질이 상한이다. 영역 뼈대에서 뽑던 고전 방식은
색 양자화 요철이 그대로 획에 실렸는데, 신경망 선화는 원화에서 사람이 그린
듯한 **이어진 매끈한 선**을 준다 (세 모델 비교 실측: MangaLine은 가닥 누락,
AniLines detail은 해칭 노이즈, basic이 두 시험 이미지 모두 최선).

배포 요건(torch 금지)에 맞춰 onnxruntime CPU로 돈다 — `models/anilines_basic.onnx`
(69MB). 저장소에 없고 **쓰기 직전에 받는다** (`modelstore`). 모델을 못 받거나
onnxruntime이 없으면 None을 돌려주고,
celart가 고전 방식으로 대체한다 (한 버튼 원칙 — 실패는 경고만).
"""

from __future__ import annotations

import os

import cv2
import numpy as np

from .. import modelstore

# detail 판은 Basic이 놓친 세부선·추가 경계를 **낮은 우선순위 증거**로 얹는
# 자리다 (`celfit.evidence`). 못 받으면 Basic만으로 돈다 (한 버튼 원칙 —
# 실패는 알림만)
_NAMES = {"basic": "anilines_basic", "detail": "anilines_detail"}
_PAD = 8          # UNet 다운샘플 배수
# 추출 입력 상한 — UNet에 타일 처리가 없어 메모리가 입력 px에 그대로 붙는다.
# 실측(이 개발 환경): 6.2Mpx 통과 · 8.8Mpx는 33GB 버퍼를 요구하며 OOM.
# 상한을 넘으면 줄여서 넣는다 (실패는 경고만 — 한 버튼 원칙)
_MAX_PX = 5.0e6
_T = 160          # 선 판정 문턱 (255=배경·0=선) — 히스테리시스의 강한 쪽
# 히스테리시스 약한 문턱 — 신경망 출력은 옅은 획(저대비 원화의 연분홍 선·
# 머리칼 끝)에서 _T를 넘나들어, 문턱 하나로 자르면 넘는 픽셀만 빠져 점선이
# 된다. 강한 픽셀과 **이어진** 약한 픽셀만 선으로 받으면 옅은 구간이 획에
# 붙어 돌아오고, 획과 무관한 옅은 얼룩은 이어짐이 없어 그대로 걸러진다
_T_WEAK = int(os.environ.get("FS_LINE_T_WEAK", 200))
# 축소 시 문턱 완화 — AREA 평균은 가는 선을 배경과 섞어 씻어낸다. 굵기를
# 안 늘리는 대신 문턱을 열어 이어짐을 되찾는 쪽 (min-pool은 반대 선택)
_T_AREA = int(os.environ.get("FS_LINE_T_AREA", 200))


def hysteresis(gray: np.ndarray) -> np.ndarray:
    """선 밝기 지도 → 선 마스크 (이중 문턱) — `_T_WEAK` 문서 참조.

    작업 해상도에서 선은 폭 2px 이진이라 대각선마다 1px 계단이 남는다. 그
    계단은 **여기서 못 없앤다** — 밝기 지도를 흐린 뒤 자르면 잉크가 번져 선
    px가 +48%가 되고(시그마 0.8 실측), 부호 거리장을 흐린 뒤 0에서 잘라도
    뼈대 마디 밀도는 3.74 → 3.70/100px으로 그대로면서 폭 변동계수만 나빠진다
    (마스크를 직접 흐리는 쪽은 1px 선을 지운다). 계단이 실제로 걷히는 자리는
    경로 평활뿐이다 (`celfit.lines._SMOOTH`).
    """
    strong = gray < _T
    if not strong.any():
        return strong
    weak = (gray < _T_WEAK).astype(np.uint8)
    _, cc = cv2.connectedComponents(weak, connectivity=8)
    keep = np.zeros(int(cc.max()) + 1, bool)
    keep[cc[strong]] = True
    keep[0] = False
    return keep[cc]


def down_mode() -> str:
    """선 지도 축소 방식 — `FS_LINE_DOWN=area|minpool`."""
    import os
    m = os.environ.get("FS_LINE_DOWN", "minpool")
    return m if m in ("area", "minpool") else "minpool"


def fit_gray(gray: np.ndarray, w: int, h: int) -> tuple[np.ndarray, bool]:
    """선 밝기 지도를 작업 해상도로 — (지도, 히스테리시스를 쓸까).

    `to_mask`가 이진화 **직전**에 쓰는 지도가 바로 이것이다. 신뢰도를 남기는
    쪽(`to_conf`)도 같은 지도를 써야 마스크와 신뢰도가 같은 자리를 가리킨다.
    """
    gh, gw = gray.shape[:2]
    if (gh, gw) == (h, w):
        return gray, True
    if gh * gw < 1.2 * h * w:
        # 사실상 같은 크기 — 줄이는 게 아니니 붓 선택이 무의미하다. 여기서
        # 방식을 태우면 침식만 남아 선이 굵어진다 (수요 적응 재생성의 1px
        # 어긋남에서 실제로 났다: SR ×4 = 2016인데 작업 해상도는 2017)
        return cv2.resize(gray, (w, h), interpolation=(
            cv2.INTER_AREA if gh * gw > h * w else cv2.INTER_LINEAR)), True
    if down_mode() == "minpool":
        # 최소 필터(선 = 어두움)로 가는 선을 살린 뒤 평균 축소. 이어짐이 가장
        # 좋지만 선 폭이 늘어난다
        g = cv2.erode(gray, np.ones((3, 3), np.uint8))
        return cv2.resize(g, (w, h), interpolation=cv2.INTER_AREA), True
    return cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA), False


def to_mask(gray: np.ndarray, w: int, h: int) -> np.ndarray:
    """선 밝기 지도 → (h,w) 선 마스크."""
    g, hyst = fit_gray(gray, w, h)
    return hysteresis(g) if hyst else (g < _T_AREA)


def to_conf(gray: np.ndarray, w: int, h: int) -> np.ndarray:
    """선 밝기 지도 → (h,w) **이진화 전 지도** (uint8, 255=배경·0=선).

    마스크가 버리는 정보다: 옅은 획과 진한 획, 문턱을 살짝 못 넘은 자리가
    전부 0/1로 눌린다. 획 평가(`evidence`)는 그 세기를 그대로 쓴다.
    """
    return fit_gray(gray, w, h)[0]


def available(variant: str = "basic") -> bool:
    """쓸 수 있는 자리인가 — 모델 파일은 `extract()`가 받으므로 여기서 안 본다."""
    import os
    if os.environ.get("FS_CEL_NOLINES"):   # 비교 실험용 — 고전(선화 없음) 강제
        return False
    if variant == "detail" and os.environ.get("FS_NO_DETAIL"):
        return False
    try:
        import onnxruntime  # noqa: F401
        return True
    except ImportError:
        return False


def _session(model, variant: str = "basic"):
    """추론 세션 하나 — **CPU 아레나를 끄고** 만든다.

    onnxruntime의 기본 CPU 아레나는 중간 텐서를 캐시해 두고 프로세스가 끝날
    때까지 안 돌려준다. 이 UNet은 타일 처리가 없어 중간 텐서가 입력 px에 그대로
    비례하므로 아레나가 그 덩치를 통째로 문다. 끄면 텐서 수명이 끝나는 대로
    반납된다 — 실측(3.3Mpx 입력, 독립 프로세스 커널 피크):

        아레나 켬  피크 WS 7.23GB · 커밋 9.22GB · 5.1s
        아레나 끔  피크 WS 3.42GB · 커밋 4.12GB · 6.2s   ← 출력 바이트 동일

    잔류도 같이 없어진다(추출 직후 7GB → 0.1GB). 비용은 추출 1회당 +1.1초로
    전체 런의 1% 수준이고, 얻는 것은 **작은 기기에서 물러서지 않는 것**이다 —
    물러서면 같은 그림이 다른 선 지도를 내 도안이 갈린다(한 버튼 원칙).
    `enable_mem_pattern`은 이 모델에서 피크·시간 어느 쪽도 안 바꾼다(실측).

    세션을 모듈에 캐시하지 않는다. 한 런에 한두 번 부르는 자리라 재사용 이득이
    모델 보유 73MB보다 작고, 호출이 끝나면 반드시 놓는 쪽이 뒤 단계(셀 재해석·
    배치)가 낮은 바닥에서 돌게 한다.
    """
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.enable_cpu_mem_arena = False
    return ort.InferenceSession(str(model), so,
                                providers=["CPUExecutionProvider"])


def _preprocess(rgb: np.ndarray, variant: str) -> np.ndarray:
    """RGB(uint8) → 모델 입력 텐서 (1,C,H,W) float32. 업스트림 `infer.py`와 같다.

    두 판이 갈리는 자리다. basic은 **선명화 ×6 한 RGB 3채널**이고, detail은
    **회색조와 뒤집은 Sobel 크기 2채널**이다 (선명화 없음) — 채널 수부터
    다르므로 같은 입력을 먹이면 모델이 아예 안 돈다. 패딩은 UNet 4단
    다운샘플에 맞춘 8의 배수 반사 패딩이다.
    """
    if variant == "detail":
        g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        sx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
        sob = 255 - cv2.normalize(cv2.magnitude(sx, sy), None, 0, 255,
                                  cv2.NORM_MINMAX, cv2.CV_8UC1)
        img = np.stack([g, sob], axis=0).astype(np.float32) / 255.0
    else:
        # PIL ImageEnhance.Sharpness(6.0)과 동일한 선명화 (SMOOTH 커널 역보간)
        from PIL import Image, ImageEnhance

        sharp = np.array(ImageEnhance.Sharpness(Image.fromarray(rgb)).enhance(6.0))
        img = sharp.transpose(2, 0, 1).astype(np.float32) / 255.0
    h, w = img.shape[1:]
    ph, pw = (-h) % _PAD, (-w) % _PAD
    return np.pad(img, ((0, 0), (0, ph), (0, pw)), mode="reflect")[None]


def extract(rgb: np.ndarray, log=print, cap: bool = False,
            variant: str = "basic") -> np.ndarray | None:
    """RGB(uint8) → 선화 밝기 지도 (uint8, 255=배경·0=선). 불가하면 None.

    전처리는 AniLines 원본 `infer.py`와 동일하고 **판마다 다르다**:

        basic   3채널 — 선명화 ×6 한 RGB / 255
        detail  2채널 — [회색조, 255−정규화 Sobel 크기] / 255 (선명화 없음)

    두 판은 채널 수부터 다르므로 같은 입력을 먹이면 안 된다 (`_preprocess`).
    `cap=True`면 `_MAX_PX`를 넘는 입력을 줄여서 넣는다 (OOM 방지).
    `variant="detail"`은 세부선 판이다 — 없으면 조용히 None (증거가 하나 줄 뿐).
    """
    if not available(variant):
        if variant != "basic":
            return None
        log("  경고: 선화 추출을 못 쓴다(onnxruntime 없음) — 고전 방식으로 진행")
        return None
    model = modelstore.ensure(_NAMES[variant], log=log)
    if model is None:
        if variant != "basic":
            return None
        log("  경고: 선화 모델을 못 받았다 — 고전 방식으로 진행")
        return None
    if cap and rgb.shape[0] * rgb.shape[1] > _MAX_PX:
        s = (_MAX_PX / (rgb.shape[0] * rgb.shape[1])) ** 0.5
        nw, nh = max(1, round(rgb.shape[1] * s)), max(1, round(rgb.shape[0] * s))
        log(f"  선화 입력 상한 — {rgb.shape[1]}×{rgb.shape[0]} → {nw}×{nh}")
        rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    sess = _session(model, variant)
    try:
        # 메모리 상한은 기기마다 다르다 — 실패하면 절반 px로 물러서며 두 번 더
        # 시도하고, 끝내 안 되면 경고만 내고 고전 방식에 넘긴다 (한 버튼 원칙)
        for attempt in range(3):
            h, w = rgb.shape[:2]
            x = _preprocess(rgb, variant)
            try:
                y = sess.run(None, {"image": x})[0]
            except Exception as e:                    # onnxruntime OOM 등
                if attempt == 2:
                    log(f"  경고: 선화 추출 실패 — 고전 방식으로 진행 ({type(e).__name__})")
                    return None
                nw, nh = max(1, round(w / 1.42)), max(1, round(h / 1.42))
                log(f"  선화 추출 실패({type(e).__name__}) — {nw}×{nh}로 재시도")
                rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
                continue
            return np.clip(y[0, 0, :h, :w] * 255.0 + 0.5, 0, 255).astype(np.uint8)
        return None
    finally:
        del sess       # 세션을 놓아야 뒤 단계가 낮은 바닥에서 돈다
