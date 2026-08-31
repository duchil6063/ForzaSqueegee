"""입력 확대 — 큐빅 대신 애니 특화 SR (Real-ESRGAN anime6B).

모델이 없으면 `None`을 돌려주고 호출부가 큐빅으로 간다 (한 버튼 원칙).
`FS_NO_SR=1`로 끈다.

타일 처리: ×4라 큰 입력은 메모리를 크게 문다. 타일 240px + 겹침 16px로 나눠
돌리고 겹침은 잘라 붙인다 (SR은 국소 연산이라 이음매가 안 보인다).

`prepare()`가 노선이 실제로 받을 **중간본**을 만든다. 모드 셋 — `FS_SR_MODE`:

- `fit` (**기본**, 2026-08-13 육안 채택) — **무조건 SR, 그리고 짧은 변을 작업
  해상도로.** 확대 배수 게이트도 "원본이 이미 크다"도 안 본다. 기본
  `FS_SR_IN=full`이라 원본을 통째로 SR에 넣고, ×4 결과를 짧은 변 기준으로
  줄인다. 중간본이 작업 해상도와 같으므로 선 지도를 줄일 일이 없다
  (`FS_LINE_DOWN`이 무력해진다).
  - 원본 짧은 변이 300 미만이면 ×4로도 작업 해상도에 못 미쳐 마지막 단계가
    축소가 아니라 큐빅 확대가 된다.
- `off` — 원본 < 작업 해상도일 때만 SR. 원본이 크면 SR을 안 대고 호출부가
  INTER_AREA로 줄인다 (변인 분리용).
- `cap` — 해상도 무관 SR이되 버리는 양을 최소로. SR 입력을 `_SR_IN_MAX`까지만
  줄이고 ×4 한 중간본을 그대로 넘긴다 (선화를 작업 해상도보다 큰 데서 뽑기
  위한 모드). SR 비용은 입력 px에 비례한다 — 실측 21~24 s/입력Mpx.
  - **저해상 입력에는 안 건다** — 원본이 작업 해상도에 못 미쳐 SR이 확대를
    맡는 그림(`_SR_MIN_GAIN` 게이트가 여는 자리)에서는 중간본을 작업 해상도
    위로 더 키워 봐야 **SR이 지어낸 것을 더 뽑을 뿐이다**. 그런 입력은 off와
    같은 길로 간다. 새 상수가 아니라 이미 있는 SR 게이트를 재사용한다.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

from .. import modelstore
from ..i18n import msg
from .stop import stop_here

_TILE, _OVER = 240, 16
_SCALE = 4
# SR 입력 상한 — `FS_SR_IN`으로 고른다. 기본은 `full`(상한 없음)이다:
# - `full`(기본) — 원본을 통째로 SR에 넣는다. 도안↔원 지각차가 전역·얼굴
#   양쪽에서 가장 좋고, 비용·메모리는 견딘다(피크 WS 3GB 안쪽).
# - `short` — 짧은 변을 `_SR_IN_MAX`로. 작업 해상도 규칙과 같은 축이라 인물의
#   폭은 안 줄지만 짧은 변이 큰 원본에서 실제 픽셀을 버린다.
# - `long` — 긴 변을 `_SR_IN_MAX`로. 비용은 제일 잘 잡히는데 세로로 긴 구도에서
#   **짧은 변이 같이 무너진다** — 실제 픽셀을 버린 뒤 ×4로 되살리는 꼴이다.
_SR_IN_MAX = 1200
# ×4 출력의 픽셀 상한 — 이보다 크면 SR을 접고 원본으로 간다 (fit 모드).
# 96Mpx ≈ RGBA 384MB. 실측 사고는 383Mpx(1.4GiB)였다 — 여유를 4배 둔다.
_SR_OUT_MAX_PX = 96_000_000


# SR을 걸 최소 확대 배수 — 작업 해상도가 짧은 변 기준이 되면서 "짧은 변만 살짝
# 모자란" 큰 원본이 저해상으로 판정된다. 그런 입력을 ×4 SR(수십 Mpx)에 태우는
# 것은 순수 낭비다 — 1.25배 미만은 큐빅으로 간다
_SR_MIN_GAIN = 1.25
_MODES = ("off", "cap", "fit")
_NAME = "realesrgan_anime6b"
_SESS = None
_WARNED = False


def available() -> bool:
    """쓸 수 있나 — 받아 둔 모델이 있어야 한다 (`prepare()`가 미리 받는다)."""
    return modelstore.have(_NAME) and not os.environ.get("FS_NO_SR")


def _sess():
    global _SESS
    if _SESS is None:
        import onnxruntime as ort

        _SESS = ort.InferenceSession(str(modelstore.path(_NAME)),
                                     providers=["CPUExecutionProvider"])
    return _SESS


def release() -> None:
    """세션을 놓는다 — `prepare`가 끝나면 부른다.

    여기는 아레나를 **켜 둔다**: 240px 타일을 수백 번 돌리는 자리라 텐서 크기가
    일정하고 작아서, 아레나 재사용이 정확히 이 형태를 위한 것이다 (선화 쪽과
    반대다 — 그쪽은 입력 px에 비례하는 덩어리를 한 번 잡는다). 대신 SR이
    끝나면 남길 이유가 없으므로 세션째 놓는다.
    """
    global _SESS
    _SESS = None


def _run(rgb: np.ndarray) -> np.ndarray:
    """RGB uint8 → ×4 RGB uint8 (타일 처리)."""
    s = _sess()
    h, w = rgb.shape[:2]
    out = np.zeros((h * _SCALE, w * _SCALE, 3), np.uint8)
    for y0 in range(0, h, _TILE):
        for x0 in range(0, w, _TILE):
            stop_here()          # SR이 앞단 시간의 태반이다 — 타일마다 묻는다
            y1, x1 = min(y0 + _TILE, h), min(x0 + _TILE, w)
            # 겹침을 물려 자른 뒤 결과에서 겹침만큼 다시 떼어낸다
            py0, px0 = max(0, y0 - _OVER), max(0, x0 - _OVER)
            py1, px1 = min(h, y1 + _OVER), min(w, x1 + _OVER)
            tile = rgb[py0:py1, px0:px1].astype(np.float32) / 255.0
            y = s.run(None, {"img": tile.transpose(2, 0, 1)[None]})[0][0]
            y = np.clip(y.transpose(1, 2, 0), 0, 1)
            cy0, cx0 = (y0 - py0) * _SCALE, (x0 - px0) * _SCALE
            out[y0 * _SCALE:y1 * _SCALE, x0 * _SCALE:x1 * _SCALE] = np.round(
                y[cy0:cy0 + (y1 - y0) * _SCALE,
                  cx0:cx0 + (x1 - x0) * _SCALE] * 255).astype(np.uint8)
    return out


def fit(rgba: np.ndarray, size: int) -> np.ndarray:
    """**짧은 변**을 `size`로 맞춘다 (축소 AREA·확대 큐빅) — 노선 공통 규칙.

    작업 해상도는 짧은 변 기준이다. 긴 변 기준이면 세로로 긴 구도에서 인물의
    폭이 몇백 px로 줄어 얼굴이 무너진다. 짧은 변을 고정하면 어떤 구도든
    인물의 굵기가 같은 해상도로 들어온다.
    """
    h, w = rgba.shape[:2]
    s = size / min(h, w)
    if abs(s - 1.0) < 1e-6:
        return rgba
    return cv2.resize(rgba, (max(1, round(w * s)), max(1, round(h * s))),
                      interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)


def fit_long(rgba: np.ndarray, size: int) -> np.ndarray:
    """**긴 변**을 `size`로 맞춘다 — 비용·메모리 상한용 (작업 해상도 규칙 아님)."""
    h, w = rgba.shape[:2]
    s = size / max(h, w)
    if abs(s - 1.0) < 1e-6:
        return rgba
    return cv2.resize(rgba, (max(1, round(w * s)), max(1, round(h * s))),
                      interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)


def fit_min(rgba: np.ndarray, size: int) -> np.ndarray:
    """**짧은 변**을 `size`로 맞춘다 (확대는 안 한다).

    SR 입력의 축소 기준 — 긴 변으로 맞추면 세로로 긴 구도가 폭 몇십 px로
    뭉개져, 얼굴이 백 px 남짓 안에 들어가고 SR이 복원이 아니라 창작을 한다.
    짧은 변 기준이면 폭이 유지된다.
    """
    h, w = rgba.shape[:2]
    s = min(1.0, size / min(h, w))
    if abs(s - 1.0) < 1e-6:
        return rgba
    return cv2.resize(rgba, (max(1, round(w * s)), max(1, round(h * s))),
                      interpolation=cv2.INTER_AREA)


def _sr_input(rgba: np.ndarray) -> np.ndarray:
    """SR에 넣을 입력으로 줄인다 — `FS_SR_IN`(full|short|long, `_SR_IN_MAX` 참조)."""
    h, w = rgba.shape[:2]
    m = os.environ.get("FS_SR_IN", "full")
    if m == "full":
        return rgba                        # 안 줄인다 — 원본을 통째로 SR에
    if m == "long":
        return fit_long(rgba, min(max(h, w), _SR_IN_MAX))
    return fit_min(rgba, _SR_IN_MAX)


def _sr_rgba(rgba: np.ndarray) -> np.ndarray:
    """RGBA → ×4 RGBA.

    RGB만 SR을 태우고 알파는 큐빅 — 알파는 경계 한 겹이라 SR이 줄 게 없고,
    RGB는 투명부를 최근접 불투명 색으로 메우고 넣는다 (투명 픽셀의 쓰레기
    색이 SR을 타고 실루엣 안으로 번지는 것을 막는다).
    """
    from .celart import _fill_bg_nearest

    h, w = rgba.shape[:2]
    sel = rgba[..., 3] >= 128
    sr = _run(_fill_bg_nearest(rgba[..., :3], sel) if not sel.all() else rgba[..., :3])
    a = cv2.resize(rgba[..., 3], (w * _SCALE, h * _SCALE), interpolation=cv2.INTER_CUBIC)
    return np.dstack([sr, a])


def upscale_rgba(rgba: np.ndarray, size: int, log=print,
                 force: bool = False) -> np.ndarray | None:
    """저해상 RGBA를 **짧은 변** `size`로 키운다. SR을 못 쓰면 None (호출부가 큐빅).

    `force=True`면 확대 배수 게이트를 건너뛴다 — `fit` 모드가 쓴다.
    """
    global _WARNED
    if not available():
        if not _WARNED and not os.environ.get("FS_NO_SR"):
            log(msg("  경고: SR 모델이 없다 — 큐빅 확대로 진행 "
                    "(models/realesrgan_anime6b.onnx)"))
            _WARNED = True
        return None
    h, w = rgba.shape[:2]
    if not force and size < _SR_MIN_GAIN * min(h, w):
        return None                       # 확대가 미미하다 — 호출부가 큐빅
    # SR 입력 상한 — 비용이 입력 px에 비례한다(실측 21~24 s/Mpx). 어차피 ×4라
    # 상한까지 줄여 넣어도 작업 해상도를 넘는다
    pre = _sr_input(rgba)
    log(msg("  저해상 입력 {w}×{h} — 애니 SR로 확대", w=w, h=h)
        + (msg(" ({w}×{h} 경유)", w=pre.shape[1], h=pre.shape[0])
           if pre.shape != rgba.shape else ""))
    return fit(_sr_rgba(pre), size)


def mode() -> str:
    m = os.environ.get("FS_SR_MODE", "fit")
    return m if m in _MODES else "fit"      # 모르는 값이면 기본값으로 (off 아님)


def prepare(rgba: np.ndarray, size: int, log=print) -> np.ndarray:
    """노선이 받을 **중간본**을 만든다 — 호출부가 `fit(…, size)`로 줄여 쓴다.

    반환값의 최대변은 작업 해상도 이상이거나(SR 경로) 원본 그대로다(SR을 못
    쓰거나 off 모드에서 원본이 이미 큰 경우 — 호출부의 AREA 축소가 맡는다).
    선화는 이 중간본에서 뽑는다 (작업 해상도로 줄인 뒤 뽑으면 가는 선이
    씻겨 점선이 된다).

    끝나면 세션을 놓는다 — 바로 다음이 선화 추출이라 두 세션이 겹칠 이유가 없다.
    """
    try:
        return _prepare(rgba, size, log)
    finally:
        release()


def _prepare(rgba: np.ndarray, size: int, log) -> np.ndarray:
    h, w = rgba.shape[:2]
    m = mode()
    # SR 모델은 저장소에 없다 — 쓸 자리에서 받아 둔다 (18MB). 못 받으면
    # 아래 `available()`이 False가 되어 큐빅으로 간다 (한 버튼 원칙)
    if m != "off" and not os.environ.get("FS_NO_SR"):
        modelstore.ensure(_NAME, log=log)
    if not available():
        if m != "off":
            log(msg("  경고: SR 모델이 없다 — 원본 해상도로 진행"))
        if min(h, w) >= size:
            return rgba
        sr = upscale_rgba(rgba, size, log=log)
        return rgba if sr is None else sr
    if m == "off":
        if min(h, w) >= size:
            return rgba
        sr = upscale_rgba(rgba, size, log=log)
        return rgba if sr is None else sr
    if m == "fit":
        # 무조건 SR — 게이트는 안 본다. 단 ×4 출력이 메모리를 뚫을 크기면
        # (실측: 6080×4444 원본 → 22796×16812, 1.4GiB 할당 실패로 죽었다)
        # 원본을 그대로 중간본으로 쓴다. 대개 그런 원본은 이미 작업 해상도를
        # 넘어 SR이 줄 것이 없고, 축소는 호출부의 AREA가 맡는다. 짧은 변이
        # 작업 해상도에 못 미치는 극단적으로 긴 구도(예: 900×20000)도 ×4에서
        # 상한을 넘으므로 여기로 온다 — 그때는 호출부가 큐빅으로 키운다.
        # **판단은 ×4 출력 크기 하나로 한다**: 짧은 변을 같이 보면 그런 구도가
        # 가드를 비켜가 상한을 만든 그 할당 실패로 되돌아간다.
        if h * w * _SCALE * _SCALE > _SR_OUT_MAX_PX:
            log(msg("  원본 {w}×{h} — ×{scale} SR 출력이 메모리 상한을 넘어 "
                    "원본 해상도로 진행", w=w, h=h, scale=_SCALE))
            return rgba
        sr = upscale_rgba(rgba, size, log=log, force=True)
        return rgba if sr is None else sr
    if m == "cap" and size >= _SR_MIN_GAIN * min(h, w):
        # 저해상 입력 — SR이 확대를 맡는 자리다. 중간본을 더 키우면 SR이 지어낸
        # 것을 선화가 더 뽑는다 (위 docstring). off와 같은 길로 보낸다
        sr = upscale_rgba(rgba, size, log=log)
        return rgba if sr is None else sr
    pre = _sr_input(rgba)
    log(f"  SR(cap): {w}×{h} → {pre.shape[1]}×{pre.shape[0]} → ×4 "
        f"({pre.shape[1] * _SCALE}×{pre.shape[0] * _SCALE})")
    return _sr_rgba(pre)
