"""설치본을 들여다보는 명령 — 설치 폴더와 차량 목록."""

from __future__ import annotations

from ..i18n import msg


def _list_cars(carfiles, name: str | None) -> int:
    """설치된 차량 미디어명 — 이름을 주면 그 이름의 **후보와 점수**만 찍는다.

    이 목록이 `--media`의 유일한 발견 경로다 (게임 화면의 표시 이름과 파일
    이름은 규약이 다르다). 점수가 갈리지 않을 때 사람이 붙잡을 것은 **크기**라
    후보마다 (길이×폭×높이) 유닛을 같이 찍는다 (`game.cars.size_text`).
    """
    from ..game import cars as gcars

    if name:
        media, cands = carfiles.pick_media(name)
        if not cands:
            print(msg("'{name}' 후보 없음 — 이름 일부로 다시 찾아볼 것",
                      name=name))
            return 1
        print(msg("'{name}' 후보 (문턱 {min:g}점 · 크기는 길이×폭×높이 유닛):",
                  name=name, min=carfiles.MEDIA_MIN))
        for s, m in cands:
            mark = msg("←  자동으로 고른다") if m == media else \
                (msg("   문턱 미달") if s < carfiles.MEDIA_MIN else "")
            print(f"  {s:5.1f}  {m:<30}{gcars.size_text(m):>16}  {mark}".rstrip())
        if media is None:
            print(msg("\n전부 문턱 미달이다 — 설치 면 지도를 통째로 버리고 "
                      "프리셋으로 물러난다.\n  못 박으려면: "
                      "--media {media}", media=cands[0][1]))
        return 0
    cars = carfiles.list_cars()
    if not cars:
        print(msg("설치 폴더를 못 찾았다 (media/Cars)"))
        return 1
    print(msg("설치된 차량 {n}대:", n=len(cars)))
    for c in cars:
        print(f"  {c:<30}{gcars.size_text(c):>16}".rstrip())
    return 0


def _gamedir(carfiles, path: str | None, clear: bool) -> int:
    """설치 폴더를 보여 주거나 못 박는다 — 자동 탐색(Steam 규약)이 실패할 때의 길."""
    if clear:
        carfiles.set_install_dir(None)
        print(msg("저장해 둔 설치 폴더를 지웠다 — 자동 탐색으로 돌아간다"))
    elif path:
        try:
            root = carfiles.set_install_dir(path)
        except (OSError, ValueError) as e:
            print(msg("오류: {e}", e=e))
            return 1
        print(msg("설치 폴더를 못 박았다 → {root}", root=root))
    root, src = carfiles.resolve()
    if root is None:
        print(msg("설치 폴더를 못 찾았다 — {src}\n"
                  "  못 박으려면: python -m forzasqueegee gamedir "
                  '"D:\\...\\ForzaHorizon6"  (media 폴더가 있는 곳)', src=src))
        return 1
    print(msg("설치 폴더: {root}  ({src})", root=root, src=src))
    print(msg("차량 {n}대", n=len(carfiles.list_cars())))
    return 0


def ask_dir(why: str) -> str | None:
    """설치 폴더를 **사람에게 묻는다** — 콘솔에 사람이 붙어 있을 때만.

    자동 탐색이 실패하는 자리는 정해져 있다 (Steam 밖 설치·다른 드라이브).
    거기서 조용히 프리셋으로 물러나면 사람은 왜 결과가 다른지 모른다. 그래서
    묻고, 받은 자리는 못 박아 저장한다 (`gamedir`과 같은 자리).
    """
    import sys

    print(msg("FH6 설치 폴더를 {why}.", why=why))
    if not sys.stdin or not sys.stdin.isatty():
        print(msg(r'  지정하려면: python -m forzasqueegee gamedir "D:\\...\\ForzaHorizon6"'))
        return None
    try:
        got = input(msg("  media 폴더가 있는 자리 (엔터면 건너뜀): ")).strip().strip('"')
    except (EOFError, KeyboardInterrupt):
        print()
        return None
    return got or None


def _cars(sync: bool, car: str | None) -> int:
    """차량 정보 — 떠 둔 색인을 보여 주거나 설치 폴더에서 다시 뜬다.

    게임은 안 건드린다 (`LiveryMasks/Masks.xml`만 읽는다). 인게임 프로브가
    재는 것(도색 마스크·화면 배율·윗면 유리)은 이 길이 못 한다 —
    그건 인게임 프로브의 몫이고, 여기서는 몇 대를 쟀는지만 알려 준다.
    """
    from ..game import cars as gcars

    if sync:
        from ..game import carfiles

        got = None
        for _ in range(3):
            try:
                got = gcars.sync(log=print)
                break
            except FileNotFoundError:
                pick = ask_dir(carfiles.resolve()[1])
                if not pick:
                    return 1
                try:
                    carfiles.set_install_dir(pick)
                except ValueError as e:
                    print(f"  {e}")
            except (OSError, ValueError) as e:
                print(msg("오류: {e}", e=e))
                return 1
        if got is None:
            return 1
        print(msg("차량 {cars}대 · 면 {faces}개 · "
                  "크기 {sized}대 → {path}",
                  cars=got["cars"], faces=got["faces"], sized=got["sized"],
                  path=got["path"])
              + ("" if got["saved"] else msg("  (저장 실패 — 이번 실행만 쓴다)")))
        if got["failed"]:
            print(msg("  못 읽은 차 {n}대: {names}", n=len(got["failed"]),
                      names=", ".join(got["failed"][:8])))
        print(msg("  인게임 프로브까지 잰 차 {n}대", n=len(got["probed"]))
              + (": " + ", ".join(got["probed"][:6]) if got["probed"] else ""))
        return 0
    if car:
        tabs = gcars.tabs_of(car)
        if not tabs:
            print(msg("모르는 차다 — {car}  (`--sync`로 다시 뜨거나 이름을 확인할 것)",
                      car=car))
            return 1
        caps = gcars.caps_of(car)
        size = gcars.size_of(car)
        print(msg("{car} — 면 {n}개", car=car, n=len(tabs))
              + (msg(" · 길이 {l} · 폭 {w} · 높이 {h} 유닛",
                     l=size[0], w=size[1], h=size[2])
                 if size else msg(" · 크기를 못 잰다 (옆면도 앞뒤면도 없다)")))
        for i, n in enumerate(tabs):
            print(msg("  탭 {i:2d}  {name:<14} 상한 {cap:,}",
                      i=i, name=n, cap=caps[n]))
        return 0
    gcars.ensure(ask=ask_dir, log=print)
    print(gcars.summary())
    return 0 if gcars.cars() else 1


def cmd_gamedir(args) -> int:
    from ..game import carfiles

    return _gamedir(carfiles, args.path, args.clear)


def cmd_lang(args) -> int:
    """UI 언어를 보여 주거나 못 박아 저장한다 — GUI·CLI·편집기가 함께 쓴다."""
    from ..i18n import current_language, msg, save_language, saved_language

    if args.value:
        save_language(args.value)
        print(msg("언어를 못 박았다 → {lang}  (GUI·CLI·KFPS·FLS 편집기 공통)",
                  lang=args.value))
        return 0
    saved = saved_language()
    print(msg("언어: {lang}  ({how})", lang=current_language(),
              how=msg("저장값") if saved else msg("기본값 — 저장된 것이 없다")))
    return 0


def cmd_cars(args) -> int:
    return _cars(args.sync, args.car)


def cmd_models(args) -> int:
    """모델을 미리 받아 두거나 확인한다 — `release.json`이 목록이다."""
    from .. import modelstore

    if args.verify:
        return 1 if modelstore.verify() else 0
    if args.check:
        miss = 0
        for name, ent in modelstore.entries().items():
            got = modelstore.have(name)
            miss += 0 if got else 1
            have = msg("있다") if got else msg("없다")
            print(f"  {have} — {ent['file']} "
                  f"({int(ent['size']) / 1e6:,.0f}MB · {ent.get('what', '')})")
        print(msg("  {dir} — 없는 것 {miss}개", dir=modelstore.model_dir(),
                  miss=miss)
              + (msg(" (쓸 때 저절로 받는다)") if miss else ""))
        return 0
    return 1 if modelstore.fetch_all() else 0
