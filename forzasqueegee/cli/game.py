"""게임에 적용하는 명령 — 주입 · 창 조작 · 창 열기."""

from __future__ import annotations

from ..i18n import msg


def cmd_inject(args) -> int:
    from ..game.inject import apply_plan, probe

    if args.probe or not args.plan:
        # probe에 plan을 주면 "캔버스에 올려 둔 도안"으로 읽어 표를 대조한다
        return probe(expect_count=args.count, expect_plan=args.plan)
    return apply_plan(args.plan, force=args.force, template=args.template,
                      canvas=args.canvas, prepare=not args.no_prepare,
                      table=int(args.table, 0) if args.table else None)


def cmd_run(args) -> int:
    from ..auto.run_plan import run

    run(args.plan, start=args.start, limit=args.limit)
    return 0


def cmd_overlay(args) -> int:
    from ..overlay.guide import run

    return run(args.plan)


def cmd_itasha(args) -> int:
    from pathlib import Path

    from ..auto import itasha
    from ..game import carfiles

    if args.list_cars:
        return _list_cars(carfiles, args.media or args.car)
    media = args.media
    if media:
        # 오타를 **여기서** 잡는다 — 없는 이름은 조용히 매칭으로 물러나
        # 엉뚱한 차의 면 지도로 40분을 쓰게 한다
        try:
            media = carfiles.resolve_media(media)
        except (ValueError, OSError) as e:
            print(msg("오류: {e}", e=e))
            return 1
        if media != args.media:
            print(msg("설치 차량: {media}", media=media))
    if args.plan:
        plans = [Path(p) for p in args.plan]
        out = Path(args.out or "itasha.json")
        if args.preset_only:
            cfg = itasha.make_config(plans, out)
        else:
            # 기본 경로: **구성 설계**가 도안 생김새와 면 실측으로 짠다
            # (실측이 없는 면은 프리셋으로 물러난다 — engine/compose.py)
            base = None
            if args.base:
                s = args.base.lstrip("#")
                if len(s) != 6:
                    print(msg("오류: --base는 #RRGGBB 꼴이어야 한다 ({base})",
                              base=args.base))
                    return 1
                base = tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))
            from ..engine.compose.textspec import text_from_args

            tspec = text_from_args(args)          # `--text`가 없으면 None (글자 없음)
            try:
                cfg = itasha.compose_config(
                    plans[0], out, extra_plans=plans[1:], car=args.car,
                    media=media,
                    mirror=not args.no_mirror, paint=not args.no_paint,
                    base_rgb=base, flip=args.flip, deco=not args.no_deco,
                    motif=args.motif, family=args.family,
                    text=(tspec.to_dict() if tspec is not None else None))
            except (ValueError, OSError) as e:
                print(msg("오류: {e}", e=e))
                return 1
        # `-o`에 폴더를 주면 구성 파일은 그 안에 선다 — 구성이 아는 경로를 쓴다
        out = cfg.path
        print(msg("구성 파일 → {out}", out=out))
        if args.make_only:
            print(itasha.describe(cfg))
            return 0
        config = out
    elif args.config:
        config = Path(args.config)
    else:
        print(msg("itasha.json 경로나 --plan 중 하나는 있어야 한다"))
        return 2
    try:
        return itasha.run(config, restart=args.restart,
                          prepare=not args.no_prepare, yes=args.yes,
                          dry_run=args.dry_run,
                          replace=not args.keep_existing,
                          fit=not args.no_autofit, media=media)
    except (ValueError, OSError) as e:
        print(msg("오류: {e}", e=e))
        return 1
    return 0


def cmd_gui(args) -> int:
    from ..gui import run as run_gui

    return run_gui(args.image)
