"""painter 노선 — KFPS(kloudys-forza-painter-suite) 동일 로직 (galatea).

해상도는 프리셋이 정하므로 작업 해상도 인자를 안 쓴다. 공용 전처리·io는
`pipeline`에 있다.
"""

from __future__ import annotations

from pathlib import Path

from .pipeline import read_rgba


# painter 노선의 프리셋 — KFPS(kloudys-forza-painter-suite) 원본 3종 중
# 기본은 shaded (KFPS UI 기본값, 애니/디지털 아트용). GUI는 이 기본만 쓰고
# CLI(`painter` 명령)에서 프리셋을 고를 수 있다.
PAINTER_PRESET = "shaded"
PAINTER_NOTE = ("KFPS 동일 로직 — GPU 원시 생성(OpenCL/Vulkan) + 체크포인트 마무리. "
                "채점 마스크는 A_02 48각형, 값은 게임 입력 스텝 그리드다")


def _make_painter(image: Path, out: Path, shapes: int, size: int,
                  log, progress=None) -> dict:
    """painter 노선 — KFPS 동일 로직 (galatea.generate). 해상도는 프리셋이
    정하므로(`maxResolution`) `size`는 여기서 안 쓴다.

    셀 노선의 자가 점검 지표는 셀화·배치 구조를 전제하므로 여기서는 **입력
    적합성만** 잰다.
    """
    from .galatea import generate

    rgba = read_rgba(image)
    opaque = bool(rgba[..., 3].min() >= 250)
    if opaque:
        log("  경고: 알파가 없다 — 캔버스 경계 제약과 투명 침범 벌점이 놀게 된다")
    rep = generate(image, out, shapes=shapes, preset=PAINTER_PRESET,
                   log=log, progress=progress)
    checks = [{"id": "alpha", "ok": not opaque,
               "text": "투명 배경 있음" if not opaque
                       else "알파 없음 — 침범 벌점이 안 걸린다"}]
    bad = [c for c in checks if not c["ok"]]
    sel = rep.get("selected", {})
    return {"input": {"size": [int(rgba.shape[1]), int(rgba.shape[0])],
                      "alpha": not opaque},
            "plan": {"layers": sel.get("final_drawables", 0),
                     "error": sel.get("error"),
                     "candidate": sel.get("candidate"),
                     "shape_types": sel.get("shape_types"),
                     "preset": rep.get("preset"),
                     "checkpoints": len(rep.get("candidates", []))},
            "notes": [PAINTER_NOTE], "checks": checks,
            "verdict": ("판정: 걸린 것 없음" if not bad else
                        "판정: " + " · ".join(c["text"] for c in bad))}
