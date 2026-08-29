"""line 노선 — 원화의 **선만** 획으로 딴다 (면 채움 없음).

`_line_design`은 그 획 배치 절반이라 **cel 노선도 이것을 쓴다** (구조 변경
2026-08-26 — 선 도안 먼저). 공용 전처리·판정·io는 `pipeline`에 있다.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from ..paths import run_file
from .pipeline import (_one_region, _reach_check, _source_bundle, read_rgba,
                       write_png)


def _line_design(rgba: np.ndarray, sel: np.ndarray, lm0: np.ndarray,
                 shapes: int, cat, source_image: str, log, progress=None,
                 value: np.ndarray | None = None, price: float = 0.0,
                 route: str = "line", basic_gray=None, detail_gray=None):
    """**선 도안** — 공통 선 재구성 엔진을 부르는 자리.

    lm0(작업 해상도 선 지도)를 이어(`bridge_line_gaps`) 증거 지도를 짓고
    (`celfit.evidence.build_maps`) 엔진에 태운다. line 노선과 cel 노선이
    **같은 함수·같은 엔진**을 쓰고, 갈리는 것은 넘기는 `route`(정책 이름)
    하나다. 반환 선 지도는 다리를 이은 "긋기로 한 선"이고, src_rgb는 다리
    px에 제 선 색을 입힌 색 표본이다.

    `basic_gray`·`detail_gray`는 **이진화 전** 선화 지도다 (중간본 해상도) —
    있으면 신뢰도 증거로 실린다. `value`·`price`는 cel 노선의 잉크 가격이다.
    """
    from .celart import CelArt, _fill_bg_nearest
    from .celfit import bridge_line_gaps, build_maps, fit_line_plan
    from .celfit import policy as P

    from . import lineart

    h, w = lm0.shape
    src_rgb = _fill_bg_nearest(rgba[..., :3], sel) if not sel.all() \
        else np.ascontiguousarray(rgba[..., :3])
    # detail 전용 선을 **경로 원천에 합류**시킨다.
    # 증거로만 두면 detail이 처음 찾아 준 선은 값만 재고 못 긋는다 — 뼈대를
    # 뽑는 원천이 이 마스크 하나이기 때문이다. 합류시키면 그 자리에도 경로가
    # 서고, 역할 등급이 한 단 낮아 예산 컷·가격이 먼저 문다
    # (`graph.LogicalStroke.rank`).
    #
    # **근거 (표준 10장 실측)**: 배치의 천장인 `outline_src`(실루엣 테를
    # 선 지도가 덮는 몫)가 0.834 → 0.938로 오르고, 실제 실루엣 윤곽이
    # 0.799 → 0.919(+12.0%p)가 된다. 즉 "원화가 실루엣에 선을 안 그렸다"고
    # 보고해 오던 자리의 태반은 **원화가 아니라 basic 판의 사각지대**였다.
    # 밴드 밖 스필도 .0028 → .0017로 준다. 값은 도형 +40%(910 → 1,274장/장)와
    # 시간 +35%다. detail을 **증거로만** 쓰는 것은 거의 무효였다 (+34장·
    # 커버리지 +0.04%p) — 신뢰도는 역할 판정의 입력일 뿐 경로를 안 만든다.
    lm_paths = lm0
    if detail_gray is not None:
        extra = lineart.to_mask(detail_gray, w, h) & sel & ~lm0
        n_ex = int(extra.sum())
        if n_ex:
            lm_paths = lm0 | extra
            log(f"  detail 전용 선 {n_ex:,}px 합류 (basic 대비 "
                f"+{100.0 * n_ex / max(1, int(lm0.sum())):.0f}%) — 낮은 우선순위")
    # 끊긴 획 잇기 — 신경망 선화의 점선(옅은 구간)을 사람처럼 한 획으로 잇는다.
    # 배치 전에 이어야 곡선 맞춤·파편 필터가 이어진 경로 기준으로 돈다
    line_mask, bridge, n_bridge = bridge_line_gaps(lm_paths, sel, log)
    if n_bridge:
        log(f"  끊긴 획 {n_bridge}쌍 이음 (마주보는 자유 끝, 굵기 비례 틈)")
        # 다리 px의 색은 **가장 가까운 원래 선 px의 색** — 색 표본과 선화
        # 목표가 다리를 바탕색이 아니라 제 선 색으로 보게 한다
        src_rgb = np.where(bridge[..., None], _fill_bg_nearest(src_rgb, lm_paths),
                           src_rgb)
    # 배치용 셀 — 실루엣 한 영역. 획 채점판의 의미(배경 침범 벌점·우선순위)가
    # cel 노선의 선화 단계와 같아지는 최소 구성이다
    cel = CelArt(size=(w, h), labels=np.where(sel, 0, -1).astype(np.int32),
                 regions=_one_region(sel, src_rgb[sel].mean(axis=0)),
                 line_mask=line_mask, src_rgb=src_rgb)
    # 값 지도는 **가격과 별개로** 늘 짓는다 — 획 평가가 "주변보다 눈에 띄는가"를
    # 여기서 읽기 때문이다 (`evidence.StrokeEvidence.importance`). 가격을 안
    # 무는 line 노선에서 이 지도를 생략하면 중요도가 상수 1.0으로 죽어, 무늬
    # 판정과 부스러기 보호가 통째로 성립하지 않는다
    from .importance import place_weight

    val = value if value is not None else place_weight(
        np.ascontiguousarray(src_rgb), sel)
    maps = build_maps(
        lineart.to_conf(basic_gray, w, h) if basic_gray is not None else None,
        lineart.to_conf(detail_gray, w, h) if detail_gray is not None else None,
        line_mask, lm0, src_rgb, sel, val, bridge)
    log("획 배치 중…")
    plan, stats = fit_line_plan(cel, cat, budget=shapes,
                                source_image=source_image, log=log,
                                progress=progress, value=value, price=price,
                                maps=maps, pol=P.get(route))
    stats["bridged_gaps"] = n_bridge
    return plan, stats, line_mask, src_rgb


def _make_line(image: Path, out: Path, shapes: int, size: int,
               log, progress=None) -> dict:
    """line 노선 — 원화의 선만 획 레이어로 딴다 (면 채움 없음). 레이어 수는 자동.

    레퍼런스(`references/사람작업/오버레이-선*.png`)의
    사람 방식 재현이다: 원화를 반투명으로 깔고 그 위에 선만 따라 긋는다 —
    여기서는 신경망 선화(AniLines)가 그 "따라 그을 선"이고, 획 배치는 cel
    노선의 선화 단계와 같은 기계다 (`celfit.fit_line_plan` — 무엇이 왜
    다른지는 그 문서). 선화가 곧 목표라서 **모델이 없으면 이 노선은 못
    돈다** — cel처럼 고전 방식 대체가 없다 (실패는 명확히 알린다).

    자가 점검: 선 지도 대비 잉크 커버리지(배치 품질)와 실루엣 테의 선 유무
    (원화가 실루엣에 선을 안 그린 그림은 여기서 드러난다 — 도안 결함이
    아니라 입력의 성질이므로 문구가 원인을 말한다).
    """
    from .catalog import Catalog, default_catalog_path
    from .celart import CelArt, _ALPHA_OPAQUE
    from .render import render_plan

    from . import lineart, upscale

    big, line_gray, detail_gray = _source_bundle(read_rgba(image), size, log)
    if line_gray is None:
        raise SystemExit(
            "line 노선은 선화 모델이 필수다 — models/anilines_basic.onnx와 "
            "onnxruntime이 있어야 한다 "
            "(`python -m forzasqueegee models`로 받는다)")
    rgba = upscale.fit(big, size)
    h, w = rgba.shape[:2]
    opaque = bool(rgba[..., 3].min() >= 250)
    if opaque:
        log("  경고: 알파가 없다 — 이미지 전체에서 선을 딴다 (배경 제거 권장)")
    sel = rgba[..., 3] >= _ALPHA_OPAQUE
    lm0 = lineart.to_mask(line_gray, w, h) & sel
    if not lm0.any():
        raise SystemExit("선을 하나도 못 찾았다 — 선화가 없는 그림이다 "
                         "(cel 노선을 쓸 것)")
    log(f"  선화: 선 픽셀 {int(lm0.sum()):,}개")
    cat = Catalog(default_catalog_path())
    if progress:
        progress(0.02, "획 배치")
    plan, stats, line_mask, src_rgb = _line_design(
        rgba, sel, lm0, shapes, cat, str(image), log,
        progress=(lambda f, t: progress(0.02 + f * 0.88, t)) if progress else None,
        route="line", basic_gray=line_gray, detail_gray=detail_gray)
    rec = stats.pop("_rec", None)

    # 선화 목표 — cel.png의 자리다. labels가 선 마스크인 CelArt의 flat_render가
    # 정확히 "흰 바탕 + 원화 색 선"이고, 미세 조정의 목표로도 그대로 쓴다
    # (실루엣 = 선 px — 덮인 선을 새로 노출하는 이동은 기각, 선 밖 스필을
    # 줄이거나 안 덮인 선을 덮는 이동만 남는다)
    tgt = CelArt(size=(w, h), labels=np.where(line_mask, 0, -1).astype(np.int32),
                 regions=_one_region(line_mask,
                                     np.median(src_rgb[line_mask], axis=0)),
                 line_mask=line_mask, src_rgb=src_rgb)
    write_png(run_file(out, "line.png"),
              cv2.cvtColor(tgt.flat_render(), cv2.COLOR_RGB2BGR))
    # 선 재구성 자취 — 회귀 계측·육안 대조가 같은 근거를 본다
    from .celfit import policy as _P
    from .celfit import stroke_metrics

    from . import linedebug

    struct = stroke_metrics(plan, rec, cat, 900.0 / h)
    stats.update(linedebug.save(out, rec, _P.LINE, (w, h), struct=struct))
    if os.environ.get("FS_LINE_DEBUG", "1") != "0":
        linedebug.overlay(out, rec, (w, h), line_mask)

    from .finetune import refine_plan

    log("전역 미세 조정 중…")
    stats["finetune"] = refine_plan(
        plan, tgt, cat, log=log,
        progress=(lambda f, t: progress(0.92 + f * 0.07, t))
        if progress else None)
    plan.save(run_file(out, "plan.json"))

    # 2× 렌더 후 축소 — cel 노선과 같은 이유 (인게임은 벡터라 경계가 매끈하다)
    render = cv2.resize(render_plan(plan, cat, scale=2), (w, h),
                        interpolation=cv2.INTER_AREA)
    write_png(run_file(out, "preview.png"),
              cv2.cvtColor(render, cv2.COLOR_RGB2BGR))
    flat = tgt.flat_render().astype(np.float32)
    rmse_line = float(np.sqrt(((render.astype(np.float32) - flat) ** 2).mean()))

    outline = stats.get("outline_cover")
    checks = [
        {"id": "alpha", "ok": not opaque,
         "text": "투명 배경 있음" if not opaque else "알파 없음 — 전체에서 선을 딴다"},
        {"id": "budget", "ok": stats.get("skipped_strokes", 0) == 0,
         "text": "예산 안에 전 획 배치" if stats.get("skipped_strokes", 0) == 0
                 else f"예산 소진 — 획 {stats['skipped_strokes']}개 못 그림"},
        # 문턱 0.88은 병리 감지용이다 — 건강한 결과의 실측 분포(표준 10장)가
        # ±1px 90~96%이고 그 꼬리는 최소 도형보다 작아 못 그리는 반점·파편이라,
        # 그 위에 문턱을 세우면 매번 운다. 예산 컷·획 대량 실패만 잡는다.
        # 옅은 선이 덜 선 것은 아래 실루엣 검사가 따로 말한다
        {"id": "ink", "ok": stats["ink_near"] >= 0.88,
         "text": f"선 커버리지 {stats['ink_near']:.1%} (±1px · 정밀 "
                 f"{stats['ink_cover']:.1%})"},
        _reach_check(plan, cat),
    ]
    if outline is not None:
        # 원화 탓과 배치 탓을 가른다 — outline_src(테에 선 지도가 있는 몫)가
        # 상한이다. 상한과의 차의 몸통(표준 10장 실측 6~11%p)은 최소 도형보다
        # 작아 못 그리는 반점·대시의 손실이라 노선의 성질이고, 그보다 큰
        # 꼬리(14%p+)만 알린다 — 문턱 12%p는 그 분포의 무릎이다
        src_lim = stats.get("outline_src") or 0.0
        followed = outline >= src_lim - 0.12
        checks.insert(3, {
            "id": "outline", "ok": outline >= 0.90 or followed,
            "text": (f"실루엣 윤곽선 {outline:.1%}" if outline >= 0.90
                     else (f"실루엣의 {1 - src_lim:.0%}에 원화 선이 없다 — "
                           f"도안은 원화를 따른다 (윤곽선 {outline:.1%})")
                     if followed
                     else (f"실루엣 윤곽선 {outline:.1%} (원화에는 "
                           f"{src_lim:.0%} 있음) — 옅은 윤곽 획이 덜 섰다"))})
    bad = [c for c in checks if not c["ok"]]
    return {"input": {"size": [w, h], "alpha": not opaque},
            "plan": {"layers": len(plan.layers), **stats},
            "structure": struct,
            "metrics": {"rmse_line": round(rmse_line, 1),
                        "ink_cover": stats["ink_cover"],
                        "ink_near": stats["ink_near"],
                        "ink_stray": stats["ink_stray"],
                        "outline_cover": outline,
                        "outline_src": stats.get("outline_src")},
            "checks": checks,
            "verdict": ("판정: 걸린 것 없음" if not bad else
                        "판정: " + " · ".join(c["text"] for c in bad))}
