"""신경망 배경 제거 — isnet-anime (ONNX, Apache-2.0) 래퍼.

요구사항의 전처리(배경 제거) 담당이다. 알파 없는 입력(사진·JPG·불투명 PNG)은
이미지 전체가 캔버스로 잡혀 배경까지 도형으로 그려진다 — 인물만 딸 알파를
신경망으로 만든다. 모델은 rembg의 isnet-anime(SkyTNT anime-segmentation
계열, IS-Net 구조)로, 애니 인물화 특화라 이 프로젝트의 입력과 맞는다.

배포 요건(torch 금지)에 맞춰 onnxruntime CPU로 돈다 — `models/isnet_anime.onnx`
(169MB). 저장소에 없고 **쓰기 직전에 받는다** (`modelstore`). 모델을 못 받거나
onnxruntime이 없으면 None을 돌려주고,
파이프라인은 기존 경고("알파가 없다")로 진행한다 (한 버튼 원칙 — 실패는
경고만). 전·후처리는 rembg의 DisSession과 동일하다 (1024² LANCZOS, /max
정규화, mean (0.485, 0.456, 0.406), 출력 min-max 정규화 후 LANCZOS 복원).
"""

from __future__ import annotations

import numpy as np

from .. import modelstore
from ..i18n import msg

_NAME = "isnet_anime"
_SIZE = 1024
_MEAN = (0.485, 0.456, 0.406)

# 인물 본체의 1/N보다 작은 고립 알파 덩어리는 소품으로 본다 (아래 `_keep_subject`)
_ISO_KEEP = 8
# 그중 **본체의 볼록 껍질 안에 앉은 눈에 띄는 덩어리**는 가림에 잘린 인물의
# 일부로 보고 되살린다 — 껍질 안 비율과 본체 대비 크기 (아래 `_keep_subject`)
_PART_HULL, _PART_MIN = 0.5, 0.01


def available() -> bool:
    """쓸 수 있는 자리인가 — 모델 파일은 `matte()`가 받으므로 여기서 안 본다."""
    try:
        import onnxruntime  # noqa: F401
        return True
    except ImportError:
        return False


def _keep_subject(alpha: np.ndarray, log=print) -> np.ndarray:
    """인물과 **떨어져 있는 작은 덩어리**를 알파에서 지운다 (소품·잔재).

    isnet은 인물만 따라고 시켜도 인물 가까이 있는 반투명 소품(유리잔·피처 등)을
    부분적으로 남기고, 그 자리에 인물과 동떨어진 얼룩이 그려진다. 자가 점검
    (커버리지·구멍)은 그것도 "덮어야 할 픽셀"로 보므로 **수치로는 안 잡힌다**.
    알파 단계에서 거르는 것이 유일한 자리다.

    **잣대는 거리가 아니라 크기다.** 소품은 손 옆에 붙어 있어 본체에서 몇십 px
    떨어졌을 뿐이고, 정당한 테두리 부스러기도 그 정도 거리라 거리로는 둘이 안
    갈린다. 반면 크기는 확실히 갈린다 — 남는 소품은 본체의 몇 % 이하이고 본체는
    90%대다. **그 사이가 통째로 비어 있다.**

    크기 잣대는 상수 하나(`_ISO_KEEP`)다: **본체의 1/8 미만인 성분은 버린다.**
    표본 7장에서 이 상수는 **1 < N < 27.1 구간 전체가 산출물 바이트 동일**이다
    (버릴 것 중 제일 큰 aru 유리잔이 본체의 1/27.1이고, 살릴 것 중 제일 작은
    것이 본체 자신이다). 즉 표본은 이 값을 못 고른다 — 절벽이 아니라 **띠**이고
    8은 그 안에서 양쪽 여유가 다 크다.

    **크기만으로는 가림에 잘린 인물을 못 지킨다.** 다리에 가려 몸에서 끊긴
    머리 뭉치는 본체의 1/23.7이라 이 잣대에 걸려 통째로 사라진다 (실측 09:
    9,141px · 본체의 4.21%). 그렇다고 잣대를 그 위로 올리면 소품이 다시
    살아난다 — 살릴 것(1/23.7)과 버릴 것(1/27.1)이 붙어 있어 크기로는 못
    가른다. 낮은 문턱 연결성·본체까지 거리·알파 확신·색도 다 뒤집혀 있다
    (실측: 09 머리 뭉치가 배경 글자보다 더 멀고·더 옅다).

    가르는 것은 **본체가 그 자리를 감싸고 있는가**다: 가림에 잘린 부분은 인물
    자신의 윤곽 안에 들어앉고, 옆에 놓인 남은 물건은 밖에 선다. 그래서 크기로
    버릴 성분 중 **본체 볼록 껍질 안 비율이 `_PART_HULL` 이상이고 본체의
    `_PART_MIN` 이상**인 것만 되살린다. 표본 11장에서 두 잣대 다 양쪽 여유가
    크다 — 껍질 안 비율은 살릴 것 98.5% 대 버릴 것 0%, 크기는 4.2% 대 0.03%
    (껍질 안이지만 눈에 안 띄는 부스러기가 그 0.03%다).

    **부드러운 테두리는 상수 없이 가른다** — 옅은 알파(<128) 픽셀은 가까운 쪽
    코어에 딸려 간다. 옅은 마스크로 성분을 나누는 길은 못 쓴다: isnet의 옅은
    안개가 소품과 인물을 이어 붙여 7장 모두 성분이 하나가 된다(실측).
    """
    import cv2

    hi = (alpha >= 128).astype(np.uint8)       # 문턱 128 = 크롭·셀 캔버스 판정과 같다
    n, lab, stats, _ = cv2.connectedComponentsWithStats(hi, 8)
    if n <= 2:                                 # 성분이 하나면 거를 것이 없다
        return alpha
    areas = stats[1:, cv2.CC_STAT_AREA]
    drop_ids = 1 + np.flatnonzero(areas * _ISO_KEEP < areas.max())
    if drop_ids.size == 0:
        return alpha
    main = 1 + int(np.argmax(areas))           # 본체 = 제일 넓은 성분
    hull = np.zeros_like(hi)
    cv2.fillConvexPoly(
        hull, cv2.convexHull(cv2.findNonZero((lab == main).astype(np.uint8))), 1)
    part = [i for i in drop_ids.tolist()
            if areas[i - 1] >= _PART_MIN * areas.max()
            and hull[lab == i].mean() >= _PART_HULL]
    if part:
        drop_ids = np.array([i for i in drop_ids.tolist() if i not in part])
        log(msg("  가림에 잘린 인물 {n}개 되살림 "
                "({px:,}px · 본체의 {pct:.2f}%)",
                n=len(part), px=int(areas[np.array(part) - 1].sum()),
                pct=areas[np.array(part) - 1].sum() / areas.max() * 100))
        if drop_ids.size == 0:
            return alpha
    drop = np.isin(lab, drop_ids)
    keep = hi.astype(bool) & ~drop
    # 코어 둘 중 가까운 쪽에 딸려 보낸다 (버린 코어 위는 거리 0이라 그대로 지워진다)
    d_keep = cv2.distanceTransform((~keep).astype(np.uint8), cv2.DIST_L2, 3)
    d_drop = cv2.distanceTransform((~drop).astype(np.uint8), cv2.DIST_L2, 3)
    out = alpha.copy()
    out[d_drop < d_keep] = 0
    log(msg("  고립 알파 {n}개 제거 ({px:,}px · 본체의 {pct:.2f}%)",
            n=drop_ids.size, px=int(areas[drop_ids - 1].sum()),
            pct=areas[drop_ids - 1].sum() / areas.max() * 100))
    return out


def matte(rgb: np.ndarray, log=print) -> np.ndarray | None:
    """RGB(uint8) → 인물 알파 (uint8, 255=인물·0=배경). 불가하면 None."""
    if not available():
        log(msg("  경고: 배경 제거를 못 쓴다(onnxruntime 없음) — 알파 없이 진행"))
        return None
    model = modelstore.ensure(_NAME, log=log)
    if model is None:
        log(msg("  경고: 배경 제거 모델을 못 받았다 — 알파 없이 진행"))
        return None
    import onnxruntime as ort
    from PIL import Image

    # CPU 아레나를 끄고, 쓰고 나면 바로 놓는다 — 이 모델은 런 맨 앞에서 한 번만
    # 돌고 그 뒤 100초 넘게 노는데, 기본 설정이면 가중치 176MB에 중간 텐서까지
    # 물고 끝까지 남는다. 실측(1024² 입력): 피크 WS 0.92 → 0.80GB · 커밋
    # 2.01 → 1.50GB, 출력 바이트 동일, 비용 +0.1초. 근거는 lineart._session()
    so = ort.SessionOptions()
    so.enable_cpu_mem_arena = False
    sess = ort.InferenceSession(str(model), so, providers=["CPUExecutionProvider"])

    im = Image.fromarray(rgb).resize((_SIZE, _SIZE), Image.Resampling.LANCZOS)
    a = np.asarray(im).astype(np.float64)
    a = a / max(float(a.max()), 1e-6)
    x = (a - np.array(_MEAN)).transpose(2, 0, 1)[None].astype(np.float32)
    pred = sess.run(None, {sess.get_inputs()[0].name: x})[0][:, 0, :, :]
    del sess
    pred = (pred - pred.min()) / max(float(pred.max() - pred.min()), 1e-12)
    mask = Image.fromarray((np.squeeze(pred) * 255).astype(np.uint8), mode="L")
    mask = mask.resize((rgb.shape[1], rgb.shape[0]), Image.Resampling.LANCZOS)
    return _keep_subject(np.asarray(mask), log=log)
