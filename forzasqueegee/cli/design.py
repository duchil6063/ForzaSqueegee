"""도안을 짓고 고치는 명령 — 게임을 안 건드린다."""

from __future__ import annotations

from ..i18n import msg
from ..paths import run_file


def cmd_make(args) -> int:
    from ..engine.pipeline import make

    make(args.image, args.out, route=args.route, shapes=args.shapes,
         size=args.size, cut_bg=args.cut_bg, no_crop=args.no_crop)
    return 0


def cmd_painter(args) -> int:
    from ..engine.galatea import generate

    generate(args.image, args.out, shapes=args.shapes, preset=args.preset,
             seed=args.seed, repair=(True if args.repair else None),
             luma=(None if args.luma is None else args.luma == "on"),
             heatmap=args.heatmap, boost=args.boost,
             finalize_only=args.finalize_only)
    return 0


def cmd_sortplan(args) -> int:
    from pathlib import Path

    from ..engine.catalog import Catalog, default_catalog_path
    from ..engine.model import LayerPlan
    from ..engine.sortplan import render_equal, sort_plan

    plan_path = Path(args.plan)
    plan = LayerPlan.load(plan_path)
    cat = Catalog(default_catalog_path())
    sorted_plan, stats = sort_plan(plan, cat)
    if not render_equal(plan, sorted_plan, cat):
        print(msg("오류: 정렬 전후 렌더 불일치 — 저장하지 않음"))
        return 1
    out = (Path(args.out) if args.out
           else run_file(plan_path.parent, "plan_sorted.json"))
    sorted_plan.save(out)
    print(msg("정렬 완료 → {out}", out=out))
    print(msg("  레이어 {layers}, 그룹 {groups_before}→{groups_after}, "
              "HSB 세션 {hsb_before}→{hsb_after} (렌더 동일성 검증 통과)",
              layers=stats["layers"],
              groups_before=stats["groups_before"],
              groups_after=stats["groups_after"],
              hsb_before=stats["hsb_before"], hsb_after=stats["hsb_after"]))
    return 0


def cmd_pruneplan(args) -> int:
    from pathlib import Path

    from ..engine.catalog import Catalog, default_catalog_path
    from ..engine.model import LayerPlan
    from ..engine.pruneplan import prune_plan
    from ..engine.sortplan import render_equal

    plan_path = Path(args.plan)
    plan = LayerPlan.load(plan_path)
    cat = Catalog(default_catalog_path())
    pruned, stats = prune_plan(plan, cat, min_vis=args.min_vis)
    out = (Path(args.out) if args.out
           else run_file(plan_path.parent, "plan_pruned.json"))
    if args.min_vis <= 0:
        if not render_equal(plan, pruned, cat):
            print(msg("오류: 프루닝 전후 렌더 불일치 — 저장하지 않음"))
            return 1
        note = msg("렌더 동일성 검증 통과")
    else:
        import numpy as np

        from ..engine.render import render_plan
        from ..engine.sortplan import plan_pad_px

        pad = plan_pad_px(plan, cat)
        ra = render_plan(plan, cat, pad=pad).astype(np.int16)
        rb = render_plan(pruned, cat, pad=pad).astype(np.int16)
        d = np.abs(ra - rb).max(axis=2)
        note = msg("렌더 diff: 변경 {changed:.3%} px, 최대 {max}",
                   changed=float((d > 8).mean()), max=int(d.max()))
    pruned.save(out)
    print(msg("프루닝 완료 → {out}", out=out))
    print(msg("  레이어 {before} → {after} (-{removed}, {note})",
              before=stats["before"], after=stats["after"],
              removed=stats["removed"], note=note))
    return 0


def cmd_kfpsimport(args) -> int:
    import json
    from pathlib import Path

    from ..engine.kfpsjson import import_kfps_to

    try:
        plan, st, plan_path = import_kfps_to(
            args.kfps, args.out, source_image=args.image or "")
    except (OSError, ValueError, json.JSONDecodeError) as e:
        print(msg("오류: {e}", e=e))
        return 1
    w, h = plan.image_size
    kind = msg("타입코드") if st["kind"] == "typecode" else msg("생성기(legacy)")
    print(msg("들여오기 완료 ({kind}) → {plan_path}", kind=kind,
              plan_path=plan_path))
    print(msg("  레이어 {layers}장 (마스크 {masks}) · "
              "캔버스 {w}x{h}px",
              layers=st["layers"], masks=st["masks"], w=w, h=h))
    for word, n in sorted(st["unknown"].items()):
        print(msg("  경고: 카탈로그에 없는 도형 word {word} {n}장을 뺐다 "
                  "(글꼴 글리프 등)", word=word, n=n))
    if st["invisible"]:
        print(msg("  알파 0 도형 {n}장은 뺐다 (안 보이는 도형)",
                  n=st["invisible"]))
    if len(plan.layers) > 3000:
        print(msg("  경고: FH6 비닐 그룹 상한(3,000장)을 넘는다 — "
                  "pruneplan으로 줄일 것"))
    return 0


def cmd_kfpsexport(args) -> int:
    import json
    from pathlib import Path

    from ..engine.catalog import Catalog, default_catalog_path
    from ..engine.kfpsjson import export_typecode, roundtrip_diff
    from ..engine.model import LayerPlan

    plan_path = Path(args.plan)
    plan = LayerPlan.load(plan_path)
    cat = Catalog(default_catalog_path())
    data, st = export_typecode(plan, cat)
    if not data["shapes"]:
        print(msg("오류: 내보낼 수 있는 레이어가 하나도 없다"))
        return 1
    out = (Path(args.out) if args.out
           else run_file(plan_path.parent, "kfps.json"))
    out.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print(msg("내보내기 완료 → {out}  (도형 {n}장: "
              "정확 {exact} · 마스크 {masks} · "
              "근사 {approx})",
              out=out, n=len(data["shapes"]), exact=st["exact"],
              masks=st["masks"], approx=st["approx"]))
    if st["approx"]:
        print(msg("  참고: word 없는 도형 {n}장은 등가 타원으로 "
                  "나갔다 (글꼴 글리프 등)", n=st["approx"]))
    d = roundtrip_diff(plan, data, cat)
    print(msg("  자가 점검 (재들여온 렌더 대조): 변경 {changed_frac:.3%} px, "
              "최대 {max}", changed_frac=d["changed_frac"], max=d["max"]))
    return 0


def cmd_flsedit(args) -> int:
    r"""내장 FLS 편집기의 [Itasha] 메뉴가 부르는 엔진.

    편집기가 늘 같은 세 걸음으로 온다: 프로젝트를 저장 → 이 명령 → 쓴 것을
    다시 열기 (`main_window_itasha.cpp`). 그래서 여기서 하는 일도 하나다 —
    조리법을 고치고 `.3so`를 다시 굽는다 (`engine.fls.studio`).

    **마지막 한 줄이 편집기 상태줄에 뜬다** — 사람이 읽을 문장을 거기에 둔다.
    """
    from pathlib import Path

    from ..engine.fls import studio

    lines: list[str] = []
    log = lines.append
    st = None
    try:
        if args.action == "export-group":
            got, said = studio.export_group(args.project, args.format, args.out)
            print(said)
            return 0
        st = studio.open_project(args.project, geometry=args.geometry)
        surface = (studio.surface_of_slot(args.slot)
                   if args.slot is not None and args.slot >= 0 else None)
        said = ""
        if args.action == "load-design":
            if not args.design:
                print(msg("오류: --design이 필요하다"))
                return 2
            if surface is None:
                print(msg("오류: 도안을 올릴 구획을 못 정했다 (--slot)"))
                return 2
            said = studio.act_load_design(st, args.design, surface)
        elif args.action == "auto-place":
            said = studio.act_auto_place(st, surface, args.group)
        elif args.action in ("decoration", "no-decoration"):
            said = studio.act_decoration(st, args.action == "decoration")
        elif args.action == "motif":
            said = studio.act_motif(st, args.family)
        elif args.action == "family":
            said = studio.act_family(st, args.composition)
        elif args.action == "style":
            said = studio.act_style(st, None if args.style in (None, "auto") else args.style)
        elif args.action == "decorate":
            text_fields = None
            if args.text:
                text_fields = {
                    "main": args.text, "sub": args.subtext, "number": args.text_number,
                    "style": args.text_style,
                    "engine": args.text_engine,
                    "placement": args.text_placement, "priority": args.text_priority,
                    "allow_fallback_to_game_text": (
                        None if args.game_text_fallback is None
                        else args.game_text_fallback != "off"),
                    "max_layers": args.text_max_layers, "outline": args.text_outline,
                    "shadow": args.text_shadow}
            roles = None
            if args.role is not None:
                roles = {}
                for item in args.role:
                    k, sep, v = str(item).partition("=")
                    if not sep or not k.strip().isdigit():
                        print(msg("오류: --role은 `<번호>=<역할>` 꼴이다 — {item!r}",
                                  item=item))
                        return 2
                    roles[int(k)] = v.strip()
            logos = None
            if (args.logo is not None or args.no_logos or args.watermark is not None
                    or args.logo_placement is not None):
                logos = {"images": ([] if args.no_logos else args.logo),
                         "watermark": (None if args.watermark is None
                                       else args.watermark == "on"),
                         "placement": args.logo_placement}
            said = studio.act_decorate(
                st, composition=args.composition, motif=args.family, style=args.style,
                paint=args.color, auto_paint=bool(args.auto_paint),
                text=text_fields, drop_text=bool(args.no_text), roles=roles,
                logos=logos,
                symmetry=(None if args.symmetry is None else args.symmetry == "on"),
                faces=getattr(args, "face", None))
        elif args.action == "text":
            if not args.text:
                print(msg("오류: --text가 필요하다"))
                return 2
            said = studio.act_text(st, {
                "main": args.text, "sub": args.subtext, "number": args.text_number,
                "style": args.text_style,
                "engine": args.text_engine,
                "placement": args.text_placement, "priority": args.text_priority,
                "allow_fallback_to_game_text": (
                    None if args.game_text_fallback is None
                    else args.game_text_fallback != "off"),
                "max_layers": args.text_max_layers, "outline": args.text_outline,
                "shadow": args.text_shadow})
        elif args.action == "no-text":
            said = studio.act_text(st, {"main": None})
        elif args.action == "mirror":
            said = studio.act_mirror(st, surface, args.group)
        elif args.action == "base-paint":
            said = studio.act_base_paint(st, args.color, args.color is None)
        elif args.action == "state":
            import json as _json
            from ..engine import compose as _compose

            # 편집기 창이 읽는 것 — 조리법 그대로에 **드롭다운 목록**을 더한다
            # (프리셋 이름·설명은 엔진의 것이고 언어도 엔진이 안다)
            out = dict(st.state)
            out["style_presets"] = _compose.style_listing()
            print(_json.dumps(out, ensure_ascii=False, indent=1))
            return 0
        stats = studio.rebuild(st, log=log)
    except (ValueError, OSError, FileNotFoundError, RuntimeError) as e:
        print("\n".join(lines))
        # 오류 앞의 알림도 같이 낸다 — 조리법이 무엇을 뺐는지가 대개 오류의
        # **까닭**이다 (편집기에서 지운 도안 따위).
        for n in (st.notes if st is not None else []):
            print(f"  · {n}")
        print(msg("오류: {e}", e=e))
        return 1
    if lines:
        print("\n".join(lines))
    for n in st.notes:
        print(f"  · {n}")
    faces = " · ".join(f"{k} {v:,}" for k, v in
                       sorted((stats.get("sections") or {}).items()))
    print(msg("{said} — {layers:,}장", said=said, layers=stats["layers"])
          + (f" ({faces})" if faces else ""))
    return 0


def cmd_flsexport(args) -> int:
    """도안·이타샤 구성 → 게임이 읽는 컨테이너 폴더 + FLS 프로젝트(`.3so`).

    입력을 보고 갈래를 정한다 — 도안(`*.plan.json`)이면 비닐 그룹
    (`LayerGroup_*/C_group`), `*.itasha.json`이면 리버리 한 벌
    (`Livery_*/C_livery`)이다."""
    import json
    from pathlib import Path

    from ..engine.fls import bridge

    src = Path(args.plan)
    if not src.is_file():
        print(msg("오류: 파일이 없다 — {src}", src=src))
        return 1
    out_root = Path(args.out) if args.out else src.parent
    try:
        raw = json.loads(src.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        print(msg("오류: {e}", e=e))
        return 1
    livery = isinstance(raw, dict) and "placements" in raw
    try:
        if livery:
            folder, st = bridge.itasha_folder(src, out_root, name=args.name)
            proj, _ = bridge.itasha_project(
                src, out_root / f"{folder.name}.3so", name=args.name)
            print(msg("리버리로 내보냈다 → {folder}", folder=folder))
            print(msg("  레이어 {layers:,}장 · 면 {faces}",
                      layers=st["layers"],
                      faces=" · ".join(f"{k} {v:,}"
                                       for k, v in st["sections"].items())))
            if st.get("car_id"):
                print(msg("  차 id {car_id}", car_id=st["car_id"])
                      + (f" ({st['media']})" if st.get("media") else ""))
            else:
                print(msg("  경고: 차 id를 못 정했다 (설치 폴더를 못 찾았거나 차를 "
                          "못 골랐다) — 리버리가 인게임 목록에 안 뜰 수 있다. "
                          "구성 파일에 \"media\": \"<MAKE_Model_YY>\"를 적을 것"))
        else:
            folder, st = bridge.plan_folder(src, out_root, name=args.name)
            proj, _ = bridge.plan_project(
                src, out_root / f"{folder.name}.3so", name=args.name)
            print(msg("비닐 그룹으로 내보냈다 → {folder}", folder=folder))
            print(msg("  레이어 {layers:,}장 (마스크 {masks:,}",
                      layers=st["layers"], masks=st["masks"])
                  + (msg(" · 중첩 그룹 {subgroups}", subgroups=st["subgroups"])
                     if st.get("subgroups") else "") + ")")
    except (OSError, ValueError) as e:
        print(msg("오류: {e}", e=e))
        return 1
    print(msg("  FLS 프로젝트 → {proj}", proj=proj))
    if st.get("skipped"):
        n = sum(st["skipped"].values())
        print(msg("  경고: 카탈로그 도형 id를 모르는 {n}장을 뺐다 "
                  "({skipped})", n=n, skipped=", ".join(sorted(st["skipped"]))))
    print(msg("  게임 저장 컨테이너 뿌리에 폴더를 그대로 두면 저장 그리드에 뜬다"))
    if args.open:
        from .. import flseditor

        try:
            flseditor.open_file(proj)
            print(msg("  FLS 편집기를 열었다 — {exe}", exe=flseditor.find_exe()))
        except FileNotFoundError as e:
            print(msg("  경고: {e}", e=e))
    return 0


def cmd_flsimport(args) -> int:
    """FLS·게임 파일 → 도안. 리버리는 면마다 도안 + `*.itasha.json`으로 편다."""
    from pathlib import Path

    from ..engine.fls import bridge

    try:
        out, st = bridge.import_any(args.path, args.out)
    except (OSError, ValueError) as e:
        print(msg("오류: {e}", e=e))
        return 1
    if out.name.endswith("itasha.json"):
        print(msg("리버리를 폈다 → {out}", out=out))
        print(msg("  면 {faces}",
                  faces=" · ".join(f"{k} {v:,}"
                                   for k, v in (st.get("surfaces") or {}).items())))
    else:
        print(msg("도안으로 변환했다 → {out}", out=out))
        print(msg("  레이어 {layers:,}장 "
                  "(마스크 {masks:,})",
                  layers=st.get("layers", 0), masks=st.get("masks", 0)))
    for sid, n in sorted((st.get("unknown") or {}).items()):
        print(msg("  경고: 카탈로그에 없는 도형 id {sid} {n}장을 뺐다",
                  sid=sid, n=n))
    if st.get("rasters"):
        print(msg("  래스터 로고 {n}장은 뺐다 (카탈로그 밖의 그림이다)",
                  n=st["rasters"]))
    return 0


def cmd_bordermask(args) -> int:
    from pathlib import Path

    from ..engine.bordermask import border_mask_layers
    from ..engine.catalog import Catalog, default_catalog_path
    from ..engine.model import LayerPlan

    plan_path = Path(args.plan)
    plan = LayerPlan.load(plan_path)
    cat = Catalog(default_catalog_path())
    bands = border_mask_layers(plan, cat)
    if not bands:
        print(msg("돌출 없음 — 밴드 마스크 불필요"))
        return 0
    # 밴드는 제 폴더에 둔다 (auto_progress 충돌 방지) — 이름은 도안 폴더를
    # 따라간다 (`out/내도안/border/내도안.plan_border.json`)
    out = (Path(args.out) if args.out else plan_path.parent / "border"
           / f"{plan_path.parent.name}.plan_border.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    mini = LayerPlan(source_image=plan.source_image, image_size=plan.image_size,
                     units_per_px=plan.units_per_px, layers=bands)
    mini.save(out)
    print(msg("밴드 마스크 {n}장 → {out}", n=len(bands), out=out))
    for b in bands:
        print(f"  {b.shape} x={b.x} y={b.y} sx={b.sx} sy={b.sy}")
    return 0


def cmd_edit(args) -> int:
    from ..kfpseditor import serve_cli

    return serve_cli(args.plan, port=args.port,
                     open_browser=not args.no_browser, recover=args.recover)
