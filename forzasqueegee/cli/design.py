"""도안을 짓고 고치는 명령 — 게임을 안 건드린다."""

from __future__ import annotations

from ..paths import run_file


def cmd_make(args) -> int:
    from ..engine.pipeline import make

    make(args.image, args.out, route=args.route, shapes=args.shapes,
         size=args.size, keep_bg=args.keep_bg, no_crop=args.no_crop)
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
        print("오류: 정렬 전후 렌더 불일치 — 저장하지 않음")
        return 1
    out = (Path(args.out) if args.out
           else run_file(plan_path.parent, "plan_sorted.json"))
    sorted_plan.save(out)
    print(f"정렬 완료 → {out}")
    print(f"  레이어 {stats['layers']}, 그룹 {stats['groups_before']}→{stats['groups_after']}, "
          f"HSB 세션 {stats['hsb_before']}→{stats['hsb_after']} (렌더 동일성 검증 통과)")
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
            print("오류: 프루닝 전후 렌더 불일치 — 저장하지 않음")
            return 1
        note = "렌더 동일성 검증 통과"
    else:
        import numpy as np

        from ..engine.render import render_plan
        from ..engine.sortplan import plan_pad_px

        pad = plan_pad_px(plan, cat)
        ra = render_plan(plan, cat, pad=pad).astype(np.int16)
        rb = render_plan(pruned, cat, pad=pad).astype(np.int16)
        d = np.abs(ra - rb).max(axis=2)
        note = (f"렌더 diff: 변경 {float((d > 8).mean()):.3%} px, "
                f"최대 {int(d.max())}")
    pruned.save(out)
    print(f"프루닝 완료 → {out}")
    print(f"  레이어 {stats['before']} → {stats['after']} "
          f"(-{stats['removed']}, {note})")
    return 0


def cmd_kfpsimport(args) -> int:
    import json
    from pathlib import Path

    from ..engine.kfpsjson import import_kfps_to

    try:
        plan, st, plan_path = import_kfps_to(
            args.kfps, args.out, source_image=args.image or "")
    except (OSError, ValueError, json.JSONDecodeError) as e:
        print(f"오류: {e}")
        return 1
    w, h = plan.image_size
    kind = "타입코드" if st["kind"] == "typecode" else "생성기(legacy)"
    print(f"들여오기 완료 ({kind}) → {plan_path}")
    print(f"  레이어 {st['layers']}장 (마스크 {st['masks']}) · "
          f"캔버스 {w}x{h}px")
    for word, n in sorted(st["unknown"].items()):
        print(f"  경고: 카탈로그에 없는 도형 word {word} {n}장을 뺐다 "
              f"(글꼴 글리프 등)")
    if st["invisible"]:
        print(f"  알파 0 도형 {st['invisible']}장은 뺐다 (안 보이는 도형)")
    if len(plan.layers) > 3000:
        print(f"  경고: FH6 비닐 그룹 상한(3,000장)을 넘는다 — "
              f"pruneplan으로 줄일 것")
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
        print("오류: 내보낼 수 있는 레이어가 하나도 없다")
        return 1
    out = (Path(args.out) if args.out
           else run_file(plan_path.parent, "kfps.json"))
    out.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print(f"내보내기 완료 → {out}  (도형 {len(data['shapes'])}장: "
          f"정확 {st['exact']} · 마스크 {st['masks']} · "
          f"근사 {st['approx']})")
    if st["approx"]:
        print(f"  참고: word 없는 도형 {st['approx']}장은 등가 타원으로 "
              f"나갔다 (글꼴 글리프 등)")
    d = roundtrip_diff(plan, data, cat)
    print(f"  자가 점검 (재들여온 렌더 대조): 변경 {d['changed_frac']:.3%} px, "
          f"최대 {d['max']}")
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
                print("오류: --design이 필요하다")
                return 2
            if surface is None:
                print("오류: 도안을 올릴 구획을 못 정했다 (--slot)")
                return 2
            said = studio.act_load_design(st, args.design, surface)
        elif args.action == "auto-place":
            said = studio.act_auto_place(st, surface, args.group)
        elif args.action in ("decoration", "no-decoration"):
            said = studio.act_decoration(st, args.action == "decoration")
        elif args.action == "motif":
            said = studio.act_motif(st, args.family)
        elif args.action == "mirror":
            said = studio.act_mirror(st, surface, args.group)
        elif args.action == "base-paint":
            said = studio.act_base_paint(st, args.color, args.color is None)
        elif args.action == "export":
            # 내보내기는 **지금 프로젝트 그대로**를 쓴다 — 다시 굽지 않는다
            # (편집기에서 손댄 것이 그 자리에서 컨테이너로 간다).
            studio.rebuild(st, log=log)
            print("\n".join(lines))
            print(studio.act_export(st, args.out))
            return 0
        elif args.action == "state":
            import json as _json

            print(_json.dumps(st.state, ensure_ascii=False, indent=1))
            return 0
        stats = studio.rebuild(st, log=log)
    except (ValueError, OSError, FileNotFoundError, RuntimeError) as e:
        print("\n".join(lines))
        print(f"오류: {e}")
        return 1
    if lines:
        print("\n".join(lines))
    for n in st.notes:
        print(f"  · {n}")
    faces = " · ".join(f"{k} {v:,}" for k, v in
                       sorted((stats.get("sections") or {}).items()))
    print(f"{said} — {stats['layers']:,}장" + (f" ({faces})" if faces else ""))
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
        print(f"오류: 파일이 없다 — {src}")
        return 1
    out_root = Path(args.out) if args.out else src.parent
    try:
        raw = json.loads(src.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        print(f"오류: {e}")
        return 1
    livery = isinstance(raw, dict) and "placements" in raw
    try:
        if livery:
            folder, st = bridge.itasha_folder(src, out_root, name=args.name)
            proj, _ = bridge.itasha_project(
                src, out_root / f"{folder.name}.3so", name=args.name)
            print(f"리버리로 내보냈다 → {folder}")
            print(f"  레이어 {st['layers']:,}장 · 면 " +
                  " · ".join(f"{k} {v:,}" for k, v in st["sections"].items()))
            if st.get("car_id"):
                print(f"  차 id {st['car_id']}"
                      + (f" ({st['media']})" if st.get("media") else ""))
            else:
                print("  경고: 차 id를 못 정했다 (설치 폴더를 못 찾았거나 차를 "
                      "못 골랐다) — 리버리가 인게임 목록에 안 뜰 수 있다. "
                      "구성 파일에 \"media\": \"<MAKE_Model_YY>\"를 적을 것")
        else:
            folder, st = bridge.plan_folder(src, out_root, name=args.name)
            proj, _ = bridge.plan_project(
                src, out_root / f"{folder.name}.3so", name=args.name)
            print(f"비닐 그룹으로 내보냈다 → {folder}")
            print(f"  레이어 {st['layers']:,}장 (마스크 {st['masks']:,}"
                  + (f" · 중첩 그룹 {st['subgroups']}" if st.get("subgroups")
                     else "") + ")")
    except (OSError, ValueError) as e:
        print(f"오류: {e}")
        return 1
    print(f"  FLS 프로젝트 → {proj}")
    if st.get("skipped"):
        n = sum(st["skipped"].values())
        print(f"  경고: 카탈로그 도형 id를 모르는 {n}장을 뺐다 "
              f"({', '.join(sorted(st['skipped']))})")
    print("  게임 저장 컨테이너 뿌리에 폴더를 그대로 두면 저장 그리드에 뜬다")
    if args.open:
        from .. import flseditor

        try:
            flseditor.open_file(proj)
            print(f"  FLS 편집기를 열었다 — {flseditor.find_exe()}")
        except FileNotFoundError as e:
            print(f"  경고: {e}")
    return 0


def cmd_flsimport(args) -> int:
    """FLS·게임 파일 → 도안. 리버리는 면마다 도안 + `*.itasha.json`으로 편다."""
    from pathlib import Path

    from ..engine.fls import bridge

    try:
        out, st = bridge.import_any(args.path, args.out)
    except (OSError, ValueError) as e:
        print(f"오류: {e}")
        return 1
    if out.name.endswith("itasha.json"):
        print(f"리버리를 폈다 → {out}")
        print("  면 " + " · ".join(f"{k} {v:,}"
                                   for k, v in (st.get("surfaces") or {}).items()))
    else:
        print(f"도안으로 변환했다 → {out}")
        print(f"  레이어 {st.get('layers', 0):,}장 "
              f"(마스크 {st.get('masks', 0):,})")
    for sid, n in sorted((st.get("unknown") or {}).items()):
        print(f"  경고: 카탈로그에 없는 도형 id {sid} {n}장을 뺐다")
    if st.get("rasters"):
        print(f"  래스터 로고 {st['rasters']}장은 뺐다 (카탈로그 밖의 그림이다)")
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
        print("돌출 없음 — 밴드 마스크 불필요")
        return 0
    # 밴드는 제 폴더에 둔다 (auto_progress 충돌 방지) — 이름은 도안 폴더를
    # 따라간다 (`out/내도안/border/내도안.plan_border.json`)
    out = (Path(args.out) if args.out else plan_path.parent / "border"
           / f"{plan_path.parent.name}.plan_border.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    mini = LayerPlan(source_image=plan.source_image, image_size=plan.image_size,
                     units_per_px=plan.units_per_px, layers=bands)
    mini.save(out)
    print(f"밴드 마스크 {len(bands)}장 → {out}")
    for b in bands:
        print(f"  {b.shape} x={b.x} y={b.y} sx={b.sx} sy={b.sy}")
    return 0


def cmd_edit(args) -> int:
    from ..kfpseditor import serve_cli

    return serve_cli(args.plan, port=args.port,
                     open_browser=not args.no_browser, recover=args.recover)
