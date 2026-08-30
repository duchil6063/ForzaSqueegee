"""§4 밀집 시각 특징 — **있으면 쓰고 없으면 없는 대로 돈다.**

DINO 계열의 범용 밀집 특징은 "이 두 자리가 같은 구조인가"를 색과 독립으로
안다 — 머리칼의 밝은 톤과 어두운 톤은 색이 달라도 같은 구조이고, 살구색
벽과 뺨은 색이 같아도 다른 구조다. 그 증거를 그래프 병합(`rag.cost`)의
**보조 항**으로만 쓴다.

지키는 선이 둘이다.

- **최종 라벨을 직접 만들지 않는다.** 특징은 간선 비용의 한 항이고, 색·위상
  파이프라인이 여전히 결정을 내린다. 모델이 무엇을 보든 선 밑에서 경계가
  서는 규칙·무늬 보호·상한은 그대로다.
- **필수 의존이 아니다.** 모델 파일이 없으면 `None`을 돌려주고 그걸로 끝이다
  (한 버튼 원칙 — 실패는 경고도 아니고 한 줄 기록이다).

모델은 `models/dense_feat.onnx`(또는 `FS_DENSE_MODEL`)에서 찾는다. 입력은
NCHW float, 출력은 `(1,C,h,w)` 격자이거나 `(1,N,C)` 토큰이면 된다 — 토큰
쪽은 정사각 격자로 되접고 CLS 한 장은 버린다.

영역 특징은 **격자 특징을 PCA로 줄여 올린 뒤** 영역 평균이다. 원 차원(384~768)
을 화면 해상도로 올리면 수 GB라 못 든다 — 줄이는 축은 이 이미지 안에서
결정적으로(공분산 고유벡터, 부호 고정) 잡는다.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from ...i18n import msg

_MODEL = os.environ.get("FS_DENSE_MODEL", "")
_DEFAULT = Path(__file__).resolve().parents[3] / "models" / "dense_feat.onnx"
_SIDE = int(os.environ.get("FS_DENSE_SIDE", 518))     # 모델 입력 한 변
_DIMS = int(os.environ.get("FS_DENSE_DIMS", 16))      # PCA 뒤 차원
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)
_SESS: dict = {}


def model_path() -> Path | None:
    """쓸 모델 파일 (없으면 None)."""
    p = Path(_MODEL) if _MODEL else _DEFAULT
    return p if p.is_file() else None


def _session():
    # 이미 지어 둔 세션이 있으면 그것을 쓴다 — 모델 파일 없이도 이 경로를
    # 태워 볼 수 있어야 한다 (세션을 끼워 넣는 것이 유일한 방법이다).
    got = _SESS.get("s", False)
    if got is not False:
        return got
    p = model_path()
    if p is None:
        return None
    if "s" not in _SESS:
        try:
            import onnxruntime as ort

            so = ort.SessionOptions()
            so.log_severity_level = 3
            _SESS["s"] = ort.InferenceSession(str(p), so,
                                              providers=["CPUExecutionProvider"])
        except Exception:
            _SESS["s"] = None
    return _SESS["s"]


def _grid(rgb: np.ndarray) -> np.ndarray | None:
    """RGB → 격자 특징 (gh, gw, C). 모델이 없거나 못 읽으면 None."""
    sess = _session()
    if sess is None:
        return None
    try:
        inp = sess.get_inputs()[0]
        side = _SIDE
        shp = list(inp.shape)
        if len(shp) == 4 and isinstance(shp[2], int) and shp[2] > 0:
            side = int(shp[2])
        img = cv2.resize(rgb, (side, side), interpolation=cv2.INTER_AREA)
        x = ((img.astype(np.float32) / 255.0 - _MEAN) / _STD)
        x = np.ascontiguousarray(x.transpose(2, 0, 1)[None])
        out = sess.run(None, {inp.name: x})[0]
    except Exception:
        return None
    a = np.asarray(out)
    if a.ndim == 4:                        # (1,C,h,w)
        return np.ascontiguousarray(a[0].transpose(1, 2, 0).astype(np.float32))
    if a.ndim == 3:                        # (1,N,C) 토큰
        n = a.shape[1]
        g = int(round(np.sqrt(n)))
        if g * g != n:                     # CLS·레지스터 토큰이 앞에 붙은 판
            g = int(np.sqrt(n - 1))
            if g * g != n - 1:
                return None
            a = a[:, n - g * g:]
        return np.ascontiguousarray(a[0].reshape(g, g, -1).astype(np.float32))
    return None


def region_features(rgb: np.ndarray, labels: np.ndarray, sel: np.ndarray,
                    n: int, log=print) -> np.ndarray | None:
    """영역별 L2 정규화 특징 (n, D) — 모델이 없으면 None."""
    g = _grid(rgb)
    if g is None:
        return None
    gh, gw, c = g.shape
    flat = g.reshape(-1, c)
    flat = flat - flat.mean(0, keepdims=True)
    # 결정적 PCA — 공분산 고유벡터, 첫 성분의 최대 절댓값 성분을 양으로 고정
    cov = flat.T @ flat / max(len(flat) - 1, 1)
    ev, evec = np.linalg.eigh(cov.astype(np.float64))
    idx = np.argsort(-ev)[:min(_DIMS, c)]
    basis = evec[:, idx]
    sign = np.sign(basis[np.argmax(np.abs(basis), axis=0), np.arange(basis.shape[1])])
    sign[sign == 0] = 1.0
    basis = basis * sign
    red = (flat @ basis).astype(np.float32).reshape(gh, gw, -1)
    h, w = labels.shape
    up = cv2.resize(red, (w, h), interpolation=cv2.INTER_LINEAR)
    d = up.shape[2]
    acc = np.zeros((max(n, 1), d), np.float64)
    idx_flat = labels[sel].ravel()
    np.add.at(acc, idx_flat, up[sel].astype(np.float64))
    cnt = np.bincount(idx_flat, minlength=max(n, 1)).astype(np.float64)
    acc /= np.maximum(cnt, 1)[:, None]
    nrm = np.linalg.norm(acc, axis=1, keepdims=True)
    log(msg("  밀집 특징 {c}차 → {d}차 (격자 {gh}×{gw}) — 병합 보조 증거",
            c=c, d=d, gh=gh, gw=gw))
    return (acc / np.maximum(nrm, 1e-9)).astype(np.float32)
