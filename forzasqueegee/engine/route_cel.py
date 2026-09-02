"""cel 노선 — **선 도안 먼저, 색은 그 아래.** 레이어 수는 자동.

노선 본체 하나다 — 공용 전처리·판정·io는 `pipeline`에, 획 배치 절반은
`route_line._line_design`에, 배치·마무리 기계는 `celfit`에 있다.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from .celfit.affine import report as skew_report
from .progress import CEL as _CEL_STAGES
from .progress import Clock as _Clock
from .pipeline import _reach_check, _source_bundle, read_rgba, write_png
from ..i18n import msg
from ..paths import run_file
from .route_line import _line_design


# **"보이는 구멍"의 크기.** 이보다 작은 잔여 군집은 메우지도, 자가 점검에서
# 세지도 않는다 — 정의와 게이트가 같은 값을 써야 "메우고 나면 통과"가 성립한다.
# 작업 해상도 1,200에서 4px 군집은 화면 폭의 0.3%이고 차에 올리면 더 줄어든다
# (가이드: "어차피 마지막에 차에 붙이면 별로 티도 안 남").
HOLE_MIN_PX = 4


# 진행 막대의 눈금은 **시간**이다 — 단계 수가 아니라 각 단이 실제로 무는
# 초로 잰다. 기본값·학습·환산은 전부 `engine.progress`에 있다.


def _make_cel(image: Path, out: Path, shapes: int, size: int,
              log, progress=None) -> dict:
    """cel 노선 — **선 도안 먼저, 색은 그 아래.** 레이어 수는 자동.

    사람 순서 그대로 세 단이다:

    ① **선 도안** — line 노선과 같은 기계(`_line_design`)로 SR 중간본의
       선화를 사람 문법으로 긋는다 (이어긋기·이음 보수·뚱뚱 덩어리 채움).
    ② **셀 재해석** — 같은 중간본을 celart.decompose로 평면 색 영역으로
       누른다 (원본 선 지도가 watershed 안내).
    ③ **색 채움** — 셀 영역 라벨을 **배치된 획 라스터에 스냅**해
       (`celart.snap_labels_to_ink`) 그 영역들을 fit_plan으로 채운다.

    스냅이 사용자 제약을 기하로 만든다: 다른 색 영역의 경계가 획 아래(밴드
    중앙선)에 서므로 **색이 선을 못 넘고**, 채움 목표가 획 밑 제 몫까지
    포함하므로 색면과 획 사이에 **빈 틈이 없으며** (구멍·커버리지 게이트가
    그것을 지킨다), 같은 영역(= 같은 색)만 획 아래를 지나간다. 획 위
    스필은 공짜(`_Scorer`의 ink 지도)·나중 면 위 스필은 나중 면이 덮는다.

    선화 모델이 없거나 선이 안 나오면 **선·면 동시 배치로 폴백**한다 (한 버튼
    원칙 — cel 노선은 모델 없이도 끝까지 돈다).

    자가 점검: 원본 대비 RMSE(참고)와 **배치 목표(cel.png) 대비** 커버리지,
    선 커버리지(line 노선과 같은 자 — 병리 감지용). 장수는 **가격 설계**가
    정한다: 한 장이 지각 가중 px로 λ(`price._PRICE_REL`)만큼 못 벌면 안
    산다 — 선 도안도 같은 λ의 잉크 몫(×`_PRICE_INK`)을 문다. 예산(상한 3,000)은
    목표가 아니라 인게임 한계일 뿐이고, 선+채움이 넘친 몫은 재컷이 시각 영향
    하위부터 걷는다 (획·메움은 보호한다).
    """
    from dataclasses import replace

    from .catalog import Catalog, default_catalog_path
    from .celart import (_ALPHA_OPAQUE, _MAX_REGIONS, decompose,
                         rebuild_regions, snap_labels_to_ink)
    from .celfit import (_ink_cover, align_to_regions, count_hole_clusters,
                         coverage, fill_holes, fit_plan, fix_min_gain,
                         grow_covers, price_of, repair_min_gain,
                         repair_mismatch, silhouette_cover)
    from .model import UNITS_PER_SCALE
    from .render import render_plan, render_plan_rgba

    from . import lineart, upscale

    # 중간본(SR 결과 또는 원본) → 작업 해상도. 캔버스 유닛은 해상도 독립이라
    # 인게임 결과는 같고, 셀 분해(평활·팔레트·뼈대)의 정밀도만 올라간다.
    # 선화는 줄이기 전 중간본에서 뽑아 여기서 작업 해상도로 맞춘다
    clk = _Clock("cel", _CEL_STAGES, progress)
    big, line_gray, detail_gray, native_gray = _source_bundle(
        read_rgba(image), size, log, progress=clk.sub("prep", msg("앞단")))
    rgba = upscale.fit(big, size)
    h, w = rgba.shape[:2]
    opaque = bool(rgba[..., 3].min() >= 250)
    if opaque:
        log(msg("  경고: 알파가 없다 — 이미지 전체를 캔버스로 본다 (배경 제거 권장)"))
    sel = rgba[..., 3] >= _ALPHA_OPAQUE
    lm0 = (lineart.to_mask(line_gray, w, h) & sel) if line_gray is not None \
        else None
    cat = Catalog(default_catalog_path())

    # ① 선 도안 — line 노선과 같은 기계. 선화가 없으면 동시 배치로 폴백
    line_plan = line_rec = None
    line_stats: dict = {}
    line_mask = src_line = ink = None
    classic = lm0 is None or not lm0.any()
    if classic:
        log(msg("  경고: 선화가 없다 — 선·면 동시 배치로 폴백"))
    # 값 맵과 가격 — 한 장이 무엇을 벌어야 살 값인가 (`price._PRICE_REL`).
    # 값은 원화의 국소 대비/주변 소란으로 잰다 (얼굴 검출 없음, 대상 불문).
    # **선 도안보다 먼저 짓는다** — 선 도안(잉크 몫 λ×0.25)과 채움이 같은 λ
    # 자를 나눠 쓴다. 무엇을 놓을지는 가격이 정하고 상한 배분은 재컷(시각
    # 영향)이 한다 — 예산을 미리 선·채움으로 가르지 않는다
    from .celart import _fill_bg_nearest
    from .importance import place_weight

    src0 = _fill_bg_nearest(rgba[..., :3], sel) if not sel.all() \
        else np.ascontiguousarray(rgba[..., :3])
    val = place_weight(np.ascontiguousarray(src0), sel)
    lam = price_of(val)
    log(msg("  가격 λ = {lam:.0f} (값 픽셀) — 한 장이 이만큼 못 벌면 안 산다",
            lam=lam))
    log(msg("셀 재해석 중…"))
    clk.enter("celart", msg("셀 재해석"))
    # 영역 상한은 예산에 안 묶는다 — 무엇을 그릴지(분해)와 몇 장을
    # 쓸지(가격)는 다른 물음이다. 묶어 두면 예산을 내릴 때 눈·코·입이
    # **분해 단계에서** 병합돼 사라진다 (실측 700장: 영역 120개, 입이 통째로
    # 없어졌다). 분해는 늘 상한(`_MAX_REGIONS`)까지 내고, 그중 값이 안 되는
    # 영역이 도형을 못 받을 뿐이라 특징이 통째로 사라지지 않는다
    #
    # §26 **분해가 선 도안보다 먼저다.** 분해는 놓인 획이 아니라 원본 선
    # 지도(`lm0`)를 안내로 쓰므로 순서를 앞당겨도 입력이 같고, 그렇게 얻은
    # **잠정 영역**을 선 도안이 증거로 쓴다 — 획마다 "무엇을 가르고 있나"를
    # 실제 면 지도에서 읽는다 (`evidence.sample`의 `bnd`). 그러고 나서 영역
    # 라벨을 **놓인 획에 다시 스냅**하므로(아래), 선과 면이 서로를 한 번씩
    # 보정하는 두 방향이 된다: 면 지도가 선을 고르고, 선이 면 경계를 앉힌다.
    cel = decompose(rgba, max_regions=_MAX_REGIONS, line_mask=lm0, log=log,
                    value=val, price=lam, debug=bool(os.environ.get("FS_CEL_DEBUG")))
    # 감사 — 스냅 **전**의 영역 넓이 (스냅이 무엇을 줄이고 지웠나, §33).
    # 계보 스위치가 꺼져 있으면 빈 dict다
    _pre_snap_area = ({str(r.rid): int(r.area) for r in cel.regions}
                      if (cel.trace or {}).get("lineage") else {})
    if not classic:
        log(msg("  선화: 선 픽셀 {n:,}개", n=int(lm0.sum())))
        line_plan, line_stats, line_mask, src_line = _line_design(
            rgba, sel, lm0, shapes, cat, str(image), log,
            progress=clk.sub("line", msg("선 도안")),
            value=val, price=lam, route="cel",
            basic_gray=line_gray, detail_gray=detail_gray,
            native_gray=native_gray,
            labels=cel.labels, regions=cel.regions)
        line_rec = line_stats.pop("_rec", None)
        # 삼킨 선 되칠 (celart.decompose 문서) — 귀속·평활이 밝게 덧칠한
        # 어두운 잔선 px를 선 얹기에 합친다. src_line은 원화색이라 그 자리가
        # 도로 어두워지고, 목표(cel.png)의 밝은 반점이 사라진다
        _sw = (cel.trace or {}).get("line_swallowed")
        if _sw is not None and _sw.any():
            line_mask = line_mask | _sw
        # ③의 채비 — 채움 목표 = **선 도안에 스냅한 셀** (`snap_labels_to_ink`
        # 문서). 획 라스터·캔버스 배율은 배치와 같은 식이고(`fit_plan` upp),
        # 스냅 반경은 게임 격자(최소 도형 반폭)다. line_mask(이은 선 지도)와
        # src_line(다리 색 반영)을 실은 cel이 이후 전 단계의 목표다 —
        # flat_render = 채움 색 + 선 도안 색이 미세 조정·수리·구멍 메움의
        # 한 기준이 된다
        upp = 900.0 / h
        # §26-b **스냅 전에 획을 색 경계로 한 칸 민다** — 여기가 선과 면이
        # 서로를 보는 두 번째 방향이다 (첫째는 위 §26: 면 지도가 어느 선을
        # 그릴지 골랐다). 제 선 지도 위 잉크를 잃지 않는 수만 받으므로 선
        # 충실도가 자격이고, 그러고 나서 아래 스냅이 색 경계를 그 획 밑에
        # 앉힌다 — 순서가 이렇게 서야 밀림이 굳지 않는다
        stats_align = align_to_regions(line_plan.layers, cat, upp, w, h,
                                       line_mask, cel.labels, log)
        ink = _ink_cover(line_plan.layers, cat, upp, w, h)
        r = max(1, int(round(0.01 * UNITS_PER_SCALE / upp)))
        # 색 가드 (snap 문서) — 선 지도가 안 덮는 픽셀은 원화색이 라벨 교체를
        # 심사한다. 어두운 선 주제가 밝은 이웃에게 넘어가 흰 점선·경계 침범이
        # 되는 것을 막는다 (X7 검수 실측)
        lut = np.zeros((int(cel.labels.max()) + 1, 3), np.uint8)
        for _r in cel.regions:
            lut[_r.rid] = _r.color
        labels = snap_labels_to_ink(cel.labels, sel, ink, r,
                                    src=np.ascontiguousarray(rgba[..., :3]),
                                    colors=lut, visible=~line_mask)
        # `replace`로 갈아 끼운다 — 면 지도(faces)·분해 자취(trace)가 그대로
        # 따라와야 계측·디버그 겹판이 같은 판을 본다
        cel = replace(cel, labels=labels,
                      regions=rebuild_regions(labels, cel.regions),
                      line_mask=line_mask, src_rgb=src_line)
        log(msg("  선 도안에 스냅: 획 {ink:,}px · 영역 {n}개 (반경 {r}px)",
                ink=int(ink.sum()), n=len(cel.regions), r=r))
    write_png(run_file(out, "cel.png"),
              cv2.cvtColor(cel.flat_render_rgba(), cv2.COLOR_RGBA2BGRA))

    stats_trace = {k: v for k, v in (cel.trace or {}).items()
                   if not isinstance(v, (np.ndarray, set))}

    log(msg("도형 배치 중…"))
    # 단위 인구조사 (`FS_UNIT_CENSUS=1`) — bar/mop **단위**의 계보를 남긴다.
    # 끄면 아무 일도 안 하고 산출물도 그대로다 (`celfit.census`)
    from .celfit import census as _census
    if _census.ON:
        _census.begin_plate(out.name, cel.size)
        # 분해 쪽 계보를 같은 판에 싣는다 (`celart.rag.lineage`) — 여기서만
        # 최종 영역 id로 "원자 → 병합 → 못 산 최선 이웃"이 이어진다.
        # 스냅은 라벨 id를 안 바꾸므로(`snap.rebuild_regions`) 같은 키다
        _census.plate_meta(
            lineage=(cel.trace or {}).pop("lineage", None),
            lam=(cel.trace or {}).get("lam"),
            atoms_ws=(cel.trace or {}).get("atoms_ws"),
            snap_area={str(r.rid): int(r.area) for r in cel.regions},
            pre_snap_area=_pre_snap_area)
        # 영역 마스크의 정본 — 겹판이 이 지도로 영역 하나를 되짚는다
        # (0 = 배경, rid+1). 계보 스위치를 끄면 안 나온다
        write_png(run_file(out, "cel_labels.png"),
                  (cel.labels.astype(np.int32) + 1).astype(np.uint16))
    # 여유 배치는 안 한다 — 잘라 맞출 것이 없으므로 상한 그대로다.
    if classic:
        # 폴백 — 선·면을 fit_plan이 함께 배치한다.
        # 선 예산도 상한이 아니라 가격이 정한다 — 선을 미리 떼어 두는 것은
        # "예산을 나눠 쓴다"는 전제이고, 가격에는 나눌 예산이 없다
        plan, stats = fit_plan(
            cel, cat, budget=shapes, line_budget=shapes,
            source_image=str(image), log=log, value=val, price=lam,
            progress=clk.sub("line", msg("도형 배치"), thru="fill"))
    else:
        # ③ 색 채움 — 선 도안은 이미 섰으니 **면만** 채운다. cel의
        # line_mask는 목표(cel.png·미세 조정·수리·메움)용이라 배치에는 떼고
        # 준다 (붙이면 획을 또 놓는다). 획이 덮는 자리는 ink_free로 공짜
        # (`_Scorer` ink — 같은 색만 선 아래로 지나가는 자유의 채점 쪽 절반).
        # 예산은 상한 그대로 — 무엇을 놓을지는 가격(λ)이 정하고, 선+채움이
        # 상한을 넘친 몫은 아래 재컷이 시각 영향 하위부터 걷는다
        # `sid_start` — 획 그룹 번호를 선 도안 다음부터 잇는다. 두 판이 각자
        # 0부터 매기면 채움의 잔여 막대 사슬과 획이 같은 그룹이 되어 프루닝
        # 원자성과 구조 지표가 함께 깨진다 (`fit_plan` 문서)
        plan, stats = fit_plan(
            replace(cel, line_mask=None), cat, budget=shapes,
            source_image=str(image), log=log, value=val, price=lam,
            ink_free=ink,
            sid_start=max((l.stroke for l in line_plan.layers), default=-1) + 1,
            progress=clk.sub("fill", msg("도형 배치")))
        plan.layers.extend(line_plan.layers)  # 선 도안은 모든 면 위 (마지막 선따기)
        stats.update({k: v for k, v in line_stats.items() if k not in stats})
        stats["line_layers"] = len(line_plan.layers)
        stats["align_moves"] = stats_align
    # 단계 스냅숏 계측 (`FS_CEL_STAGES=1`) — 결함 px가 **어느 단계에서 처음
    # 생기는지** 귀속하는 자리다 (goal 세션의 검증된 디버깅 방법). 단계마다
    # plan만 저장하고 아무것도 안 바꾼다 — 끄면 산출물 바이트 동일
    _stg = os.environ.get("FS_CEL_STAGES")

    def _snap(name):
        if _stg:
            plan.save(run_file(out, f"plan_{name}.json"))
            # 단계별 선 구조 지표 — 어느 단이 이음을 되흔드는지 귀속용
            if line_rec is not None:
                import json as _json
                from .celfit import stroke_metrics as _sm
                run_file(out, f"struct_{name}.json").write_text(
                    _json.dumps(_sm(plan, line_rec, cat, 900.0 / h)),
                    encoding="utf-8")
    if _census.ON:
        import json as _json
        run_file(out, "unit_census.json").write_text(
            _json.dumps(_census.plate()), encoding="utf-8")
    _grown_idx = stats.pop("_grown_idx", [])
    if _stg:
        import json as _json
        run_file(out, "plan_grown.json").write_text(
            _json.dumps(_grown_idx), encoding="utf-8")
    _snap("s1_fit")
    # 유예 덮개 (celfit이 미룬 λ×12~25 구간) — report에 못 실리는 레이어
    # 참조라 여기서 빼 두고, 예산이 확정된 뒤(아래) 잔여만큼 산다
    carve_defer = stats.pop("_carve_defer", [])
    stats.update({k: v for k, v in stats_trace.items() if k not in stats})

    # 예산 배분: 상한까지 컷 → 구멍 메움을 상한 **초과로** 얹음 → 재컷을
    # 수렴까지 반복 (메움 0장 + 상한 이내 = 종료). 재컷은 시각 영향 하위부터
    # 걷되 배경 노출 px에 상수 벌점(_BG_PEN)을 물어 새 핀홀을 안 만든다.
    # 마지막 라운드는 컷만 — 인게임 상한 보장.
    #
    # **보호는 메움뿐이다.** 메움은 벌점만으로는 재컷이 작은 것부터 도로 걷어
    # 잔여 ~800군집 평형에 갇히므로 라벨로 지켜야 한다. 획은 아니다 — 획까지
    # 절대 보호하면 포화 장에서 **채움이 통째로 죽는다**: 실측 04(3,000장
    # 포화)에서 선이 2,524장을 차지하자 채움이 4장까지 깎여 그림에서 색이
    # 사라졌다 (평균 ΔE 17 → 48). 재컷의 영향 순서는 이미 획 그룹 단위로
    # 원자적이라(`pruneplan.prune_impact`) 획이 중간에서 끊기지 않고 통째로
    # 빠지고, 지각 가중 ΔE로 재므로 "덜 보이는 획"이 "면 색"보다 먼저 잘린다 —
    # 그것이 라벨 보호가 하려던 일의 옳은 형태다.
    # 재컷 자체는 포화 장에서만 돈다 (표준 10장 중 04 하나)
    from .importance import masking_weight
    from .pruneplan import prune_impact

    # 중요도 가중 — 같은 색차라도 평평한 면 위(눈·입·손가락·소품)가 더 눈에
    # 띈다. 모델 없이 원화의 중심-주변 대비로 잰다 (importance.py)
    imp = None
    if cel.src_rgb is not None:
        imp = masking_weight(cel.src_rgb, cel.labels >= 0)
    stats["hole_layers"] = 0
    stats["fix_layers"] = 0
    # 구멍 게이트 귀속용 계측 (`FS_DBG_HOLES=1`) — 게이트가 걸렸을 때 **어느
    # 단계가** 구멍을 열고 닫는지는 최종 수치로는 안 보인다. 단계마다 군집 수를
    # 찍고 실루엣(`sil.png`)·미세 조정 전 플랜을 남기면, 이후는 그 둘만으로
    # 오프라인에서 메움·성장을 재현할 수 있다 (셀 재해석을 다시 안 돌린다).
    # 끄면 아무 일도 안 하고 산출물도 그대로다
    _dbg = os.environ.get("FS_DBG_HOLES")
    if _dbg:
        write_png(run_file(out, "sil.png"), (cel.labels >= 0).astype(np.uint8) * 255)

        def _hlog(tag):
            log(msg("  [dbg] {tag}: 4px+ {n4} / 1px+ {n1} / 레이어 {n}",
                    tag=tag,
                    n4=count_hole_clusters(plan, cel, cat, min_px=HOLE_MIN_PX),
                    n1=count_hole_clusters(plan, cel, cat),
                    n=len(plan.layers)))
    else:
        def _hlog(tag):
            pass
    _hlog(msg("배치 직후"))
    # §12 **잔차 초점 패스** — 보정 도형을 사기 전에 기존 도형을 먼저 움직인다.
    # 미세 조정이 맨 마지막이면 구멍 메움·잔차 수리가 이미 도형을 사 버린
    # 뒤다. 같은 기계를 잔차 자리에 초점을 맞춰 **먼저** 돌리면 그 자리 중
    # 상당수가 이동·스케일·회전만으로 닫히고, 남은 것만 산다
    from .celfit import residual as _resid
    from .finetune import refine_plan as _refine

    act_before = None                      # 겹판용 — 배치 직후의 "고칠 자리"
    res0 = _resid.analyze(plan, cel, cat, value=val,
                          price=repair_min_gain(lam), min_px=HOLE_MIN_PX)
    stats.update({k: v for k, v in res0.items() if k.startswith("res_")})
    if os.environ.get("FS_CEL_DEBUG"):
        act_before = res0["actionable"].copy()
    only = _resid.focus_layers(plan, cat, cel, res0["actionable"],
                               res0["owner"])
    if only:
        log(msg("잔차 초점 조정 중…"))
        st = _refine(plan, cel, cat, log=log, max_passes=2, only=only,
                     tag=msg("잔차 초점"),
                     progress=clk.sub("focus", msg("잔차 초점 조정")))
        stats["res_focus_layers"] = len(only)
        stats["res_focus_moves"] = st["accepts"]
    _snap("s2_focus")
    # 수렴하면 조기 종료라 상한은 여유 있게 — 3,000은 3라운드에 끝나고(불변),
    # 낮은 상한은 컷이 깊어 수렴이 느리다. 12는 포화 장 실측이다 — 선+채움
    # 3,355장의 컷↔메움 초과가 라운드마다 십수 장씩 잦아들며 단조 수렴하므로
    # 몇 라운드 더가 값싸다
    rounds = 12
    # 루프 안의 눈금 — 라운드 수는 미리 모른다 (대개 한두 바퀴에 수렴하고
    # 포화 판만 12까지 간다). 그래서 라운드를 세지 않고 **점점 느려지는**
    # 눈금을 준다: 몇 바퀴를 돌든 이 단의 몫 안에서 단조로 찬다
    _rc = (lambda it: 1.0 - 0.75 ** (it + 1))
    _sub_rc = clk.sub("recut", msg("구멍 메움·수리"))
    for it in range(rounds):
        if len(plan.layers) > shapes:
            if progress:
                _sub_rc(_rc(it), msg("시각 영향 정리"))
            before = len(plan.layers)
            keep_layers = list(plan.layers)
            # §18 자격 보호 — 혼자 덮고 있는 장은 값 불문 못 자른다. 라벨
            # 보호(양보 있음)와 급이 다르다 (`celfit.coverage` 문서)
            uniq = coverage.unique_layers(plan, cel, cat)
            # 보호 합이 예산을 넘으면 prune_impact가 보호도 영향 하위부터
            # 양보시킨다 — 항상 정확히 예산으로 내려온다
            plan, _ = prune_impact(plan, cat, budget=shapes,
                                   protect_labels=("hole",), weight=imp,
                                   protect_idx=uniq)
            # 한 장씩은 안전한 두 장이 같은 표본을 나눠 덮고 있을 수 있다 —
            # 동시 제거의 연쇄만 여기서 되짚는다 (되살리면 예산을 몇 장
            # 넘길 수 있고, 다음 라운드의 컷이 그것을 도로 맞춘다)
            fixed, back = coverage.repair_cut(
                keep_layers, plan.layers, cel, cat, plan.units_per_px)
            if back:
                plan.layers[:] = fixed
                log(msg("  컷 되짚기: {n}장 복구 (동시 제거가 연 구멍)", n=back))
            log(msg("  정리{round}: {before} → {after}장 (시각 영향 하위 컷)",
                    round=it + 1, before=before, after=len(plan.layers)))
            stats["pruned_to"] = len(plan.layers)
            _hlog(msg("정리{round} 후", round=it + 1))
        if it == rounds - 1:
            break
        if progress:
            _sub_rc(_rc(it) + 0.02, msg("구멍 메움"))
        # 성장 먼저 — 경계 부스러기(잔여의 74%)를 기존 레이어 확장으로 공짜
        # 흡수하고, 남은 것(오목 포켓 등)만 메움 타원(레이어 소모)이 맡는다
        grow_covers(plan, cel, cat, log=log)
        _hlog(msg("성장{round} 후", round=it + 1))
        # 라운드 상한은 수요 전체를 한 번에 소화할 만큼 — 500이던 시절 수요의
        # 절반만 메우고 라운드를 소진했다. 초과분은 재컷이 배경 노출 벌점
        # (_BG_PEN) 순서로 저대비 겹침부터 걷어 흡수한다. 메움 대상은
        # `HOLE_MIN_PX` 이상 군집 — 그보다 작은 반점은 화면에서 안 보이는
        # 크기라(mop min_blob과 같은 논리) 레이어를 안 쓴다
        # 라운드 헤드룸은 예산 비례 (3,000이면 4,500 = 기존 +1500과 동일) —
        # 고정 +1500은 낮은 상한에서 재컷이 감당 못 할 보호 레이어를 허용한다
        headroom = shapes * 3 // 2
        # 스택 바닥에 끼운다 — 구멍 px는 아무도 안 덮는 자리라 z 불문 보이고,
        # 타원의 스필은 이웃 면·선화가 위에서 덮는다 (`fill_holes`의 `at` 문서)
        n_hole = fill_holes(plan, cel, cat, log=log, min_px=HOLE_MIN_PX,
                            max_layers=max(0, headroom - len(plan.layers)),
                            value=val, price=lam, at=0)
        stats["hole_layers"] += n_hole
        _hlog(msg("메움{round} 후", round=it + 1))
        # 잔차 수리 — "덮였지만 색이 틀린" 응집 자국(얼룩 음영·빗나간 획·끊긴
        # 선)을 보정 도형으로 고친다 (9차 판정 "완성되지 못한 부분" 대응).
        # 첫 라운드 한 번만: 채택 문턱(min_gain)이 컷 바닥보다 높아 수리는
        # 재컷을 살아남고, 매 라운드 돌리면 컷↔수리 진동만 생긴다.
        #
        # **2단이다.** 문턱 λ×25를 넘는 수리는 상한 초과를 허용한다 — 재컷이
        # 그만큼 채움을 걷어도 남는 값이다. 그 아래 λ×12까지의 수리는 **예산
        # 잔여 안에서만** 산다 — 이 단은 재컷 밀어내기를 감수할 값이 아니다.
        # 여유 장에서는 이 단이 경계 초승달·얼룩 잔차를 줍는다
        n_fix = 0
        if it == 0:
            n_fix = repair_mismatch(
                plan, cel, cat, log=log,
                max_layers=max(0, headroom - len(plan.layers)),
                protect_lines=True, min_gain=fix_min_gain(lam))
            if len(plan.layers) < shapes:      # 잔여 0이면 이 단은 소거다
                n_fix += repair_mismatch(
                    plan, cel, cat, log=log,
                    max_layers=shapes - len(plan.layers),
                    protect_lines=True, min_gain=repair_min_gain(lam))
            stats["fix_layers"] = n_fix
        if n_hole == 0 and n_fix == 0 and len(plan.layers) <= shapes:
            break
    # 마지막 라운드는 컷 전용이라 그 컷이 연 구멍은 손대지 못했다 — 성장은
    # 레이어 0장이라 상한을 깨지 않으므로 여기서 한 번 더 흡수한다. 낙오
    # 구멍은 여러 스텝 거리에 있을 수 있어 반복을 넉넉히 준다 (스텝당 지름
    # ~1.7px, 해악 게이트는 스텝마다 그대로). 구멍이 이미 0이면(3,000 기준
    # 전 케이스) 아무것도 안 해 출력 불변
    if count_hole_clusters(plan, cel, cat, min_px=HOLE_MIN_PX,
                           value=val, price=lam):
        grow_covers(plan, cel, cat, log=log, passes=6)
    _hlog(msg("꼬리 성장 후"))
    # 유예 덮개 구매 — 배치 때 λ×25(획 가격 기준)를 못 넘겨 미뤄 둔 획
    # 덮개를, 상한이 확정된 지금 **예산 잔여만큼** 순이득 순으로 산다 (2단
    # 수리와 같은 무늬 — 여기서는 재컷이 없으므로 채움을 밀어낼 길이 없다).
    # 앵커(제 획 또는 첫째 덮개)가 재컷에 잘렸으면 덮개도 버린다 — 덮개만
    # 남으면 선이 아니라 면 색 얼룩이다. 삽입은 항상 앵커 **바로 뒤**다
    if carve_defer and len(plan.layers) < shapes:
        room = shapes - len(plan.layers)
        alive = {id(l): i for i, l in enumerate(plan.layers)}
        buys = []
        for net, anchor, cov in sorted(carve_defer, key=lambda e: -e[0]):
            if len(buys) >= room:
                break
            i = alive.get(id(anchor))
            if i is not None:
                buys.append((i, cov))
        for i, cov in sorted(buys, key=lambda e: -e[0]):
            plan.layers.insert(i + 1, cov)
        stats["carve_late"] = len(buys)
        if buys:
            log(msg("  유예 덮개 {n}장 구매 (예산 잔여 {room}장 중)",
                    n=len(buys), room=room))
    _hlog(msg("유예 덮개 후"))
    # §16 **사후 가격** — 다 그린 판에서 값을 다시 묻는다 (`prune_price` 문서).
    # 살 때의 값은 그 시점 잔여 기준이라, 뒤에 그린 면이 덮은 자리는 값이
    # 사라진다. 문턱은 수리와 같은 λ 환산(`repair_min_gain`)이고, 라벨로 지키는
    # 것은 메움뿐이다 — **획도 같은 저울에 선다** (§22, `prune_price` 문서).
    # 배경이 드러나는 장은 영향의 바닥 벌점이 지킨다
    from .pruneplan import prune_price

    _snap("s3a_preprice")                  # 되판 장을 귀속할 자리 (계측 전용)
    sold_before = list(plan.layers)
    plan, pp = prune_price(
        plan, cat, repair_min_gain(lam), weight=val,
        sil=cel.labels >= 0,
        protect_idx=coverage.unique_layers(plan, cel, cat))
    fixed, back = coverage.repair_cut(sold_before, plan.layers, cel,
                                      cat, plan.units_per_px)
    if back:
        plan.layers[:] = fixed
        pp["removed"] -= back
        pp["after"] = len(plan.layers)
    stats["postprice"] = pp["removed"]
    if pp["removed"]:
        log(msg("  사후 가격: {before} → {after}장 "
                "(기여가 수리 문턱({thr:,.0f})에 못 미치는 "
                "{removed}장 되팜, {rounds}바퀴)",
                before=pp["before"], after=pp["after"],
                thr=repair_min_gain(lam), removed=pp["removed"],
                rounds=pp["rounds"]))
        # 되팔면 그 밑이 드러난다. 영향의 바닥 벌점(`_BG_PEN`)이 배경을
        # 여는 장을 지키지만 그것은 **한 장씩 뺐을 때**의 셈이라, 위아래
        # 두 장이 같은 바퀴에 팔리면 연쇄로 배경이 열릴 수 있다 (실측
        # C20-09: 구멍 게이트 탈락). 그래서 여기서 반드시 되메운다 —
        # 먼저 이웃을 늘려 공짜로 흡수하고(레이어 0장), 그래도 남으면
        # 되판 만큼의 잔여 안에서 메움을 산다. 게이트가 지키는 자리다
        if count_hole_clusters(plan, cel, cat, min_px=HOLE_MIN_PX,
                               value=val, price=lam):
            grow_covers(plan, cel, cat, log=log, passes=6)
            n_h = fill_holes(plan, cel, cat, log=log, min_px=HOLE_MIN_PX,
                             max_layers=max(0, shapes - len(plan.layers)),
                             value=val, price=lam, at=0)
            stats["hole_layers"] += n_h
            if n_h:
                grow_covers(plan, cel, cat, log=log, passes=6)
    _hlog(msg("사후 가격 후"))
    _snap("s3_price")
    if _dbg:
        plan.save(run_file(out, "plan_preft.json"))
    # 전역 미세 조정 (DiffCompositing 이산 이식) — 완성된 스택 전체를 놓고
    # 레이어 기하를 양자화 스텝 이웃으로 좌표하강. 레이어 수·순서·색 불변,
    # 실루엣 신규 노출 기각이라 위 게이트들(상한·구멍)이 그대로 성립한다
    log(msg("전역 미세 조정 중…"))
    stats["finetune"] = _refine(
        plan, cel, cat, log=log,
        progress=clk.sub("ft", msg("전역 미세 조정")))
    _snap("s4_ft")
    # 그리기 순서 미세 조정 — 옳은 색 조각이 이미 덮고 있는 자리를 이웃 영역
    # 조각이 위에서 가리는 px를 순서만 바꿔 돌려준다 (도형 0장, 커버리지 불변.
    # `finetune.reorder_fills` 문서)
    from .finetune import reorder_fills as _reorder

    stats.update(_reorder(plan, cel, cat, log=log))
    _snap("s5_reorder")
    # §29 **교차점** — 획끼리 만나는 자리를 라스터에서 맞춘다 (레이어 0장).
    # 전역 미세 조정 **다음**이라야 그 손이 되돌리지 않고, 봉인 **앞**이라야
    # 옮긴 잉크가 드러낸 표본을 봉인이 막는다 (`celfit.junction` 문서)
    _jrec = stats.get("_rec") or line_rec
    if _jrec is not None and line_mask is not None:
        from .celfit import junction as _junc

        n_j = _junc.close_gaps(plan, cat, 900.0 / h, (w, h), _jrec.strokes,
                               line_mask, cel.labels < 0, stats)
        if n_j:
            log(msg("  교차점 정리: 끝 마디 {n}회 이동 (도형 0장)", n=n_j))
    _snap("s6_junc")                       # 봉인 **직전** — 빚의 원장이 끊기는 자리
    # §18 **봉인** — 여기가 기하를 만지는 마지막 손이다. 위의 모든 단은 값을
    # 물어 λ와 거래하지만 이 단은 안 한다: 실루엣 안에 안 칠한 표본이 하나라도
    # 남으면 그 자리는 인게임에서 차 도색이 비친다 (`celfit.coverage` 문서).
    # 성장(레이어 0장)으로 먼저 먹고, 남은 것만 최소 도형이 맡는다
    log(msg("커버리지 봉인 중…"))
    stats.update(coverage.seal_coverage(
        plan, cel, cat, log=log, budget=shapes, weight=imp,
        progress=clk.sub("seal", msg("커버리지 봉인"))))
    stats.update(coverage.measure(plan, cel, cat))
    stats.update(skew_report(plan.layers))          # §14 기울기 계측
    stats["hole_left"] = count_hole_clusters(plan, cel, cat, min_px=HOLE_MIN_PX,
                                             value=val, price=lam)
    stats["hole_specks"] = count_hole_clusters(plan, cel, cat)   # 1px+ 참고치
    stats["price"] = round(lam, 1)
    _hlog(msg("미세 조정 후"))
    # §19~§22 **팔레트 접기** — 색만 묶는다 (장수·기하는 위에서 다 끝났다).
    # 획 색이 제 발자국 평균이고 면 색이 소유 px 평균이라, 다 그린 판에는
    # 분해 팔레트(32색)의 스무 배가 넘는 색이 있다 (실측 중앙 791색). 사람
    # 판은 옆면 하나가 96색이고 상위 8색이 69%를 덮는다 — 그 차이는 색을
    # 아껴 쓰는 것이 아니라 **같은 역할에 같은 색을 다시 쓰는** 것이다.
    #
    # 기본은 **획 색만** 접는다 — 획 색은 원화에서 잰 값이 아니라 제 발자국
    # 아래 픽셀의 평균이라 지켜야 할 정본이 없고, 면 색은 그 영역의 측정값이라
    # 옮기면 그대로 재현 오차다 (`ramps` 머리말의 표: 획만 접으면 854→517색에
    # `imp_error_seen` 0.104 → 0.106 · 틀린 픽셀 913 → 913으로 무동작이고,
    # 면까지 접으면 280색에 그 자가 4.5배가 된다).
    if os.environ.get("FS_RAMP", "1").strip() != "0":
        from .celart.ramps import fold_colors

        plan, ramp_st = fold_colors(plan, cat, log=log)
        stats["ramp"] = ramp_st
    clk.enter("write", msg("파일 쓰기"))
    plan.save(run_file(out, "plan.json"))
    # 선 재구성 자취 — line 노선과 **같은 파일 형식**이라 두 노선을 나란히
    # 대 볼 수 있다 (같은 논리 획 그래프인지, 갈렸다면 어느 정책 칸인지)
    from .celfit import policy as _P
    from .celfit import stroke_metrics

    from . import linedebug

    _rec = stats.pop("_rec", None) or line_rec
    _pol = _P.CEL_FALLBACK if classic else _P.CEL
    # 선 도안의 구조 지표 — 획 레이어는 `plan`의 뒤쪽에 그대로 있으므로 cel도
    # line과 **같은 자**로 잰다 (두 노선을 나란히 대 보는 자리)
    _joints = [] if _stg else None
    line_struct = stroke_metrics(plan, _rec, cat, 900.0 / h,
                                 joints_out=_joints)
    if _stg and _joints is not None:
        import json as _json
        run_file(out, "plan_joints.json").write_text(
            _json.dumps(_joints, ensure_ascii=False), encoding="utf-8")
    stats.update({k: v for k, v in
                  linedebug.save(out, _rec, _pol, (w, h),
                                 struct=line_struct).items()
                  if k not in stats})
    if os.environ.get("FS_LINE_DEBUG", "1") != "0":
        linedebug.overlay(out, _rec, (w, h), line_mask if not classic else lm0)

    # 2× 렌더 후 축소 — 인게임은 벡터라 경계가 매끈하다. 하드 래스터 프리뷰는
    # 실물보다 거칠어 보이고 RMSE도 억울하게 나온다
    render2 = render_plan(plan, cat, scale=2)
    render = cv2.resize(render2, (w, h), interpolation=cv2.INTER_AREA)
    # 파일은 투명 배경 — 자가 점검(RMSE)은 아래 흰 바탕 `render`로 잰다
    write_png(run_file(out, "preview.png"),
              cv2.cvtColor(render_plan_rgba(plan, cat, scale=2, out_size=(w, h),
                                            white=render2),
                           cv2.COLOR_RGBA2BGRA))

    sel = rgba[..., 3] >= 128
    tgt = np.where(sel[..., None], rgba[..., :3], 255).astype(np.float32)
    rmse_src = float(np.sqrt(((render.astype(np.float32) - tgt) ** 2).mean()))
    flat = np.where(sel[..., None], cel.flat_render(), 255).astype(np.float32)
    rmse_cel = float(np.sqrt(((render.astype(np.float32) - flat) ** 2).mean()))
    # 커버리지 게이트는 **실제로 칠한 실루엣**으로 본다 — `silhouette_cover`
    # 문서. 자가 커버리지는 참고치로 남긴다
    stats["self_cover"] = round(
        1.0 - stats["uncovered_px"] / max(1, int(sel.sum())), 4)
    cover = silhouette_cover(plan, cel, cat)

    checks = [
        {"id": "alpha", "ok": not opaque,
         "text": msg("투명 배경 있음") if not opaque
                 else msg("알파 없음 — 전체가 캔버스로 잡힌다")},
        {"id": "budget", "ok": stats["skipped"] == 0,
         "text": msg("예산 안에 전 영역 배치") if stats["skipped"] == 0
                 else msg("예산 소진 — 영역 {n}개 못 그림", n=stats["skipped"])},
        {"id": "coverage", "ok": cover >= 0.97,
         "text": msg("커버리지 {cover:.1%}", cover=cover)},
        # §18 경성 게이트 — 크기·값과 무관한 자격 판정. 위 `holes`(가격의 자)
        # 보다 엄하고, 통과하면 그쪽은 자동으로 통과한다
        {"id": "seal", "ok": stats["hard_hole_samples"] == 0,
         "text": (msg("실루엣 안 미커버 없음 (2배 표본)")
                  if stats["hard_hole_samples"] == 0
                  else msg("미커버 표본 {n:,}개 ({px:.1f}px · 군집 {c}개)",
                           n=stats["hard_hole_samples"],
                           px=stats["hard_hole_px"],
                           c=stats["hard_hole_clusters"]))},
        {"id": "holes", "ok": stats["hole_left"] == 0,
         "text": (msg("보이는 구멍 없음 ({min_px}px 미만 반점 {n}개)",
                      min_px=HOLE_MIN_PX, n=stats["hole_specks"])
                  if stats["hole_left"] == 0
                  else msg("구멍 잔여 — {min_px}px+ 군집 {n}개",
                           min_px=HOLE_MIN_PX, n=stats["hole_left"]))},
        _reach_check(plan, cat),
    ]
    if not classic:
        # 선 도안 커버리지 — line 노선과 같은 자·같은 문턱 (병리 감지용.
        # 획 밑을 면이 받치므로 문턱은 그대로 0.88이면 충분하다)
        #
        # **분모는 "긋기로 한 선"이다** (`ink_near_drawn`). `ink_near`는 선 지도
        # 전체를 분모로 써서 "선화 모델을 얼마나 따랐나"를 재는데, 표현 결정
        # (`celfit.engine._fill_owns`)이 **일부러** 뺀 경계가 그대로 감점이 되어
        # 정상 판마다 이 검사가 뜬다 (실측 11번 판 79.2%). 병리 검사는 "그리기로
        # 한 것을 못 그렸나"를 물어야 하므로 억제한 경계는 분모에서 뺀다.
        # 억제가 없으면 두 값은 같다.
        checks.insert(3, {
            "id": "ink", "ok": stats["ink_near_drawn"] >= 0.88,
            "text": msg("선 커버리지 {near:.1%} (±1px · 정밀 {cover:.1%})",
                        near=stats["ink_near_drawn"],
                        cover=stats["ink_cover"])})
        if stats.get("skipped_strokes"):
            checks[1] = {"id": "budget", "ok": False,
                         "text": msg("예산 소진 — 획 {n}개 못 그림 (영역 {m}개)",
                                     n=stats["skipped_strokes"],
                                     m=stats["skipped"])}
    # §13 구조 지표 — "같은 품질이면 더 적은 도형"을 재는 자리. 평균 오차만으로는
    # 도형을 더 쓸수록 좋아 보이므로, 영역당 장수·맞물림·보존을 함께 낸다
    from .celfit import metrics as _cm

    # 선 도안 구조 + 색면 구조 — 한 자리에서 두 자를 나란히 낸다
    struct = dict(line_struct)
    struct.update(_cm.plan_metrics(plan, cel, cat, value=val,
                                   price=repair_min_gain(lam),
                                   min_px=HOLE_MIN_PX,
                                   extra={k: stats[k] for k in
                                          ("atoms", "region_merges",
                                           "merge_gain", "merge_forced",
                                           "palette_k", "palette_extra",
                                           "hole_layers", "fix_layers",
                                           "res_focus_moves")
                                          if k in stats}))
    if os.environ.get("FS_CEL_DEBUG"):
        from . import celdebug
        from .celart import mark_mask

        owner, reg_of = _resid.owner_map(plan, cel, cat, with_regions=True)
        res_dbg = _resid.analyze(plan, cel, cat, value=val,
                                 price=repair_min_gain(lam),
                                 min_px=HOLE_MIN_PX, owner=owner)
        celdebug.save(out, cel, plan, cat, res=res_dbg, marks=mark_mask(cel),
                      reg_of=reg_of, act_before=act_before, write=write_png)

    bad = [c for c in checks if not c["ok"]]
    clk.close()                            # 끝까지 온 판 — 실측을 눈금에 배운다
    return {"input": {"size": [w, h], "alpha": not opaque},
            "plan": {"layers": len(plan.layers), **stats},
            "metrics": {"rmse_src": round(rmse_src, 1),
                        "rmse_cel": round(rmse_cel, 1),
                        "coverage": round(cover, 4),
                        "ink_near": stats.get("ink_near"),
                        "ink_stray": stats.get("ink_stray")},
            "structure": struct,
            "checks": checks,
            "verdict": (msg("판정: 걸린 것 없음") if not bad else
                        msg("판정: {items}",
                            items=" · ".join(c["text"] for c in bad)))}
