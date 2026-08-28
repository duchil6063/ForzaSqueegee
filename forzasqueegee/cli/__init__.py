"""CLI — `python -m forzasqueegee <명령>`의 문법과 실행.

명령 하나가 함수 하나다 (`cmd_<이름>(args) -> int`). 문법은 `parser`가 한 벌로
쥐고, 실행은 갈래별 모듈이 나눠 갖는다:

    parser  명령줄 문법 — 하위 명령과 그 인자
    design  도안을 짓고 고치는 명령 — 게임을 안 건드린다.
    game    게임에 적용하는 명령 — 주입 · 창 조작 · 창 열기.
    info    설치본을 들여다보는 명령 — 설치 폴더와 차량 목록.

`main`은 셋을 잇는 것뿐이다 — 파싱하고, 실행 전에 서야 할 것(스트림·언어·설치
폴더·권한)을 세우고, 표에서 골라 부른다.
"""

from __future__ import annotations

from . import design, game, info
from .parser import build_parser

# 명령 이름 → 그것을 하는 함수. `parser`의 하위 명령과 정확히 같은 집합이다.
_COMMANDS = {
    "make": design.cmd_make,
    "painter": design.cmd_painter,
    "sortplan": design.cmd_sortplan,
    "pruneplan": design.cmd_pruneplan,
    "kfpsimport": design.cmd_kfpsimport,
    "kfpsexport": design.cmd_kfpsexport,
    "flsexport": design.cmd_flsexport,
    "flsedit": design.cmd_flsedit,
    "flsimport": design.cmd_flsimport,
    "bordermask": design.cmd_bordermask,
    "edit": design.cmd_edit,
    "inject": game.cmd_inject,
    "run": game.cmd_run,
    "overlay": game.cmd_overlay,
    "itasha": game.cmd_itasha,
    "gui": game.cmd_gui,
    "gamedir": info.cmd_gamedir,
    "cars": info.cmd_cars,
    "models": info.cmd_models,
}


def main(argv: list[str] | None = None) -> int:
    from ..elevate import ensure_std_streams

    ensure_std_streams()     # argparse의 오류 출력보다 먼저다 (창 모드엔 stderr이 없다)
    args = build_parser().parse_args(argv)

    from ..elevate import ensure_admin, need_admin
    from ..i18n import set_language

    set_language(args.lang)

    # **인자가 먼저 선다** — 승격은 환경변수를 안 물려받으므로(`elevate.py`) UAC를
    # 타고 다시 뜬 프로세스에서도 살아 있는 것은 이 인자와 저장 파일뿐이다.
    if args.game_dir:
        from ..game import carfiles

        try:
            carfiles.use_dir(args.game_dir)
        except (OSError, ValueError) as e:
            print(f"오류: {e}")
            return 2

    # 주입이 걸린 명령은 **띄우자마자** 권한을 묻는다 — 다만 **정말 필요할 때만**
    # 묻는다. 필요한 것은 관리자 권한 자체가 아니라 게임 프로세스를 여는 권한이고,
    # 게임이 승격 안 된 채로 돌면 같은 무결성 수준이라 그냥 열린다 (실측).
    # 그래서 열어 보고 안 열릴 때만 UAC를 띄운다 — 안 그러면 필요도 없는 승격
    # 요청이 매번 뜬다 (사용자 지적). 거절해도 계속 간다.
    if args.command in ("gui", "inject", "itasha") and not args.no_admin:
        if need_admin():
            ensure_admin()

    return _COMMANDS[args.command](args)
