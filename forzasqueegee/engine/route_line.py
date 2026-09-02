"""line 노선 — 원화의 **선만** 획으로 딴다 (면 채움 없음).

`_line_design`은 그 획 배치 절반이라 **cel 노선도 이것을 쓴다** (구조 변경
2026-08-26 — 선 도안 먼저). 공용 전처리·판정·io는 `pipeline`에 있다.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from ..i18n import msg
from ..paths import run_file
from .celfit.affine import report as skew_report
from .progress import LINE as _LINE_STAGES
from .progress import Clock as _Clock
from .pipeline import (_one_region, _reach_check, _source_bundle, read_rgba,
                       write_png)


def _line_design(rgba: np.ndarray, sel: np.ndarray, lm0: np.ndarray,
                 shapes: int, cat, source_image: str, log, progress=None,
                 value: np.ndarray | None = None, price: float = 0.0,
                 route: str = "line", basic_gray=None, detail_gray=None,
                 native_gray=None, labels=None, regions=None):
    """**선 도안** — 공통 선 재구성 엔진을 부르는 자리.

    lm0(작업 해상도 선 지도)를 이어(`bridge_line_gaps`) 증거 지도를 짓고
    (`celfit.evidence.build_maps`) 엔진에 태운다. line 노선과 cel 노선이
    **같은 함수·같은 엔진**을 쓰고, 갈리는 것은 넘기는 `route`(정책 이름)
    하나다. 반환 선 지도는 다리를 이은 "긋기로 한 선"이고, src_rgb는 다리
    px에 제 선 색을 입힌 색 표본이다.

    `basic_gray`·`detail_gray`는 **이진화 전** 선화 지도다 (중간본 해상도) —
    있으면 신뢰도 증거로 실린다. `native_gray`는 **SR을 안 태운 원화 해상도**
    basic 판이다 (§25) — "SR 판만 본 선인가"를 가르는 자로만 쓰고 경로 원천
    으로는 안 쓴다. `value`·`price`는 cel 노선의 잉크 가격이다.

    `labels`·`regions`는 **잠정 색 영역**이다 (§26, cel 노선만). 없으면 배치용
    셀은 실루엣 한 장이고 획의 양옆 판정은 색차(`side_de`)만 본다. 주면 획이
    **무엇을 가르고 있는지**를 실제 면 지도에서 읽어(`evidence.sample`의 `bnd`)
    역할 판정과 잉크 가격이 그것을 쓴다 — 색이 거의 같은 두 면을 가르는 획은
    색면이 절대 못 그리는 자리라 값을 깎아 준다 (`celfit.engine._ink_mul` ⑤).
    영역은 이 단이 끝난 뒤 **놓인 획에 다시 스냅**되므로(`celart.snap`),
    선과 면이 서로를 한 번씩 보정하는 두 방향이 된다.
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
            log(msg("  detail 전용 선 {n:,}px 합류 (basic 대비 "
                    "+{pct:.0f}%) — 낮은 우선순위",
                    n=n_ex, pct=100.0 * n_ex / max(1, int(lm0.sum()))))
    # 끊긴 획 잇기 — 신경망 선화의 점선(옅은 구간)을 사람처럼 한 획으로 잇는다.
    # 배치 전에 이어야 곡선 맞춤·파편 필터가 이어진 경로 기준으로 돈다
    line_mask, bridge, n_bridge = bridge_line_gaps(lm_paths, sel, log)
    if n_bridge:
        log(msg("  끊긴 획 {n}쌍 이음 (마주보는 자유 끝, 굵기 비례 틈)",
                n=n_bridge))
        # 다리 px의 색은 **가장 가까운 원래 선 px의 색** — 색 표본과 선화
        # 목표가 다리를 바탕색이 아니라 제 선 색으로 보게 한다
        src_rgb = np.where(bridge[..., None], _fill_bg_nearest(src_rgb, lm_paths),
                           src_rgb)
    # 배치용 셀 — 잠정 색 영역이 오면 그것을, 아니면 실루엣 한 영역. 획
    # 채점판이 보는 것은 어느 쪽이든 `labels < 0`(배경)뿐이고, 갈리는 것은
    # 획의 양옆 판정이다 (`evidence.sample`의 `bnd` — 이 획이 무엇을 가르나)
    cel = CelArt(size=(w, h),
                 labels=(labels if labels is not None
                         else np.where(sel, 0, -1).astype(np.int32)),
                 regions=(regions if regions is not None
                          else _one_region(sel, src_rgb[sel].mean(axis=0))),
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
        line_mask, lm0, src_rgb, sel, val, bridge,
        native_gray=lineart.to_conf(native_gray, w, h)
        if native_gray is not None else None)
    log(msg("획 배치 중…"))
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
    from .render import render_plan, render_plan_rgba

    from . import lineart, upscale

    # 앞단(SR + 선화 3판)도 진행을 알린다 — cel 노선과 같은 이유로 여기가
    # 판 시간의 첫 뭉텅이인데 오래도록 막대가 0에 붙어 있었다
    clk = _Clock("line", _LINE_STAGES, progress)
    big, line_gray, detail_gray, native_gray = _source_bundle(
        read_rgba(image), size, log, progress=clk.sub("prep", msg("앞단")))
    if line_gray is None:
        raise SystemExit(msg(
            "line 노선은 선화 모델이 필수다 — models/anilines_basic.onnx와 "
            "onnxruntime이 있어야 한다 "
            "(`python -m forzasqueegee models`로 받는다)"))
    rgba = upscale.fit(big, size)
    h, w = rgba.shape[:2]
    opaque = bool(rgba[..., 3].min() >= 250)
    if opaque:
        log(msg("  경고: 알파가 없다 — 이미지 전체에서 선을 딴다 (배경 제거 권장)"))
    sel = rgba[..., 3] >= _ALPHA_OPAQUE
    lm0 = lineart.to_mask(line_gray, w, h) & sel
    if not lm0.any():
        raise SystemExit(msg("선을 하나도 못 찾았다 — 선화가 없는 그림이다 "
                             "(cel 노선을 쓸 것)"))
    log(msg("  선화: 선 픽셀 {n:,}개", n=int(lm0.sum())))
    cat = Catalog(default_catalog_path())
    plan, stats, line_mask, src_rgb = _line_design(
        rgba, sel, lm0, shapes, cat, str(image), log,
        progress=clk.sub("draw", msg("획 배치")),
        route="line", basic_gray=line_gray, detail_gray=detail_gray,
        native_gray=native_gray)
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
              cv2.cvtColor(tgt.flat_render_rgba(), cv2.COLOR_RGBA2BGRA))
    # 선 재구성 자취 — 회귀 계측·육안 대조가 같은 근거를 본다
    from .celfit import policy as _P
    from .celfit import stroke_metrics

    from . import linedebug

    struct = stroke_metrics(plan, rec, cat, 900.0 / h)
    stats.update(linedebug.save(out, rec, _P.LINE, (w, h), struct=struct))
    if os.environ.get("FS_LINE_DEBUG", "1") != "0":
        linedebug.overlay(out, rec, (w, h), line_mask)

    from .finetune import refine_plan

    log(msg("전역 미세 조정 중…"))
    stats["finetune"] = refine_plan(
        plan, tgt, cat, log=log, progress=clk.sub("ft", msg("전역 미세 조정")))
    clk.enter("write", msg("파일 쓰기"))
    plan.save(run_file(out, "plan.json"))

    # 2× 렌더 후 축소 — cel 노선과 같은 이유 (인게임은 벡터라 경계가 매끈하다)
    render2 = render_plan(plan, cat, scale=2)
    render = cv2.resize(render2, (w, h), interpolation=cv2.INTER_AREA)
    write_png(run_file(out, "preview.png"),
              cv2.cvtColor(render_plan_rgba(plan, cat, scale=2, out_size=(w, h),
                                            white=render2),
                           cv2.COLOR_RGBA2BGRA))
    flat = tgt.flat_render().astype(np.float32)
    rmse_line = float(np.sqrt(((render.astype(np.float32) - flat) ** 2).mean()))

    outline = stats.get("outline_cover")
    checks = [
        {"id": "alpha", "ok": not opaque,
         "text": msg("투명 배경 있음") if not opaque
                 else msg("알파 없음 — 전체에서 선을 딴다")},
        {"id": "budget", "ok": stats.get("skipped_strokes", 0) == 0,
         "text": msg("예산 안에 전 획 배치")
                 if stats.get("skipped_strokes", 0) == 0
                 else msg("예산 소진 — 획 {n}개 못 그림",
                          n=stats["skipped_strokes"])},
        # 문턱 0.88은 병리 감지용이다 — 건강한 결과의 실측 분포(표준 10장)가
        # ±1px 90~96%이고 그 꼬리는 최소 도형보다 작아 못 그리는 반점·파편이라,
        # 그 위에 문턱을 세우면 매번 운다. 예산 컷·획 대량 실패만 잡는다.
        # 옅은 선이 덜 선 것은 아래 실루엣 검사가 따로 말한다
        {"id": "ink", "ok": stats["ink_near"] >= 0.88,
         "text": msg("선 커버리지 {near:.1%} (±1px · 정밀 {cover:.1%})",
                     near=stats["ink_near"], cover=stats["ink_cover"])},
        _reach_check(plan, cat),
    ]
    stats.update(skew_report(plan.layers))          # §14 기울기 계측
    if outline is not None:
        # 원화 탓과 배치 탓을 가른다 — outline_src(테에 선 지도가 있는 몫)가
        # 상한이다. 상한과의 차의 몸통(표준 10장 실측 6~11%p)은 최소 도형보다
        # 작아 못 그리는 반점·대시의 손실이라 노선의 성질이고, 그보다 큰
        # 꼬리(14%p+)만 알린다 — 문턱 12%p는 그 분포의 무릎이다
        src_lim = stats.get("outline_src") or 0.0
        followed = outline >= src_lim - 0.12
        checks.insert(3, {
            "id": "outline", "ok": outline >= 0.90 or followed,
            "text": (msg("실루엣 윤곽선 {outline:.1%}", outline=outline)
                     if outline >= 0.90
                     else msg("실루엣의 {miss:.0%}에 원화 선이 없다 — "
                              "도안은 원화를 따른다 (윤곽선 {outline:.1%})",
                              miss=1 - src_lim, outline=outline)
                     if followed
                     else msg("실루엣 윤곽선 {outline:.1%} (원화에는 "
                              "{src:.0%} 있음) — 옅은 윤곽 획이 덜 섰다",
                              outline=outline, src=src_lim))})
    bad = [c for c in checks if not c["ok"]]
    clk.close()                            # 끝까지 온 판 — 실측을 눈금에 배운다
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
            "verdict": (msg("판정: 걸린 것 없음") if not bad else
                        msg("판정: {items}",
                            items=" · ".join(c["text"] for c in bad)))}
