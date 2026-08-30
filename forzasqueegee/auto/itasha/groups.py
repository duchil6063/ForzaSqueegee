"""비닐 그룹 준비 — 빈 그룹을 만들어 플랜을 주입하고 슬롯에 저장한다.

저장 그룹은 **장수로** 찾는다 (게임이 이름을 그림으로만 보여 준다). 그래서 한
설정 안에서 장수가 겹치면 안 되고, 겹치면 투명 패딩으로 비켜 선다
(`_dodge_count`). 이미 같은 장수의 저장 그룹이 있으면 다시 열어 덮어쓴다
(`prepare_group_reuse`) — 2,000장 재스탬프를 피하는 길이다."""

from __future__ import annotations

import json
import time
from pathlib import Path

from ...engine.model import LayerPlan
from ...game import io as gio
from ...i18n import msg
from ..bodyedit import BodyEditor
from ..driver import Driver, DriverError
from .config import Config, Placement
from .progress import Clock, save_progress


def prepare_group(d: Driver, p: Placement, log=print,
                  clock: Clock | None = None,
                  avoid: set[int] | None = None) -> None:
    """빈 비닐 그룹을 만들어 플랜을 주입하고 슬롯에 저장한다.

    준비(`ensure_ready`)가 씨앗을 심고 장수를 채운다 — 채우기는 장당 0.44초라
    2,835장이면 20분쯤이고, 씨앗은 도형 한 종에 위저드 한 바퀴(3~14초)다.

    **씨앗은 이 플랜이 쓰는 도형까지다** (`reuse=False`). 여기서 만드는 그룹은
    주입하고 저장하면 끝이라 **다시 열 일이 없으므로**, 어휘 전체를 심는 규약
    (다음 플랜을 위한 것이다)이 그대로 낭비가 된다 — 소형 그룹은 준비 시간의
    95%가 씨앗이었다 (실측: 15장짜리 `shapes-top` 155초 중 채우기가 7초).

    `avoid`는 **이미 쓰인 장수**다 (`prepare_groups`의 `have`) — 주입 직전에
    그것과 겹치면 한 장 더 채워 비켜 간다 (`game.inject.apply_plan`의 `avoid`).
    """
    from ...auto import design
    from ...auto import template
    from ...game.inject import apply_plan

    clock = clock or Clock()
    with clock.stage("prepare.open", of=p.group):
        design.back_to_menu(d)
        design.goto_row(d, design.ROW_NEW_GROUP)
        gio.press("enter")
        for _ in range(30):
            time.sleep(0.5)
            if template.canvas_count() == 0:
                break
        else:
            raise DriverError(msg("새 비닐 그룹 에디터가 안 열렸다 (빈 캔버스 미확인)"))
    log(msg("  새 비닐 그룹 · 도안 {layers:,}장 주입 준비", layers=p.layers))
    with clock.stage("prepare.inject", of=p.group, n=p.layers):
        # 준비(씨앗·채우기) + **신원 비켜 가기** + 주입. 셋이 한 호출인 이유는
        # 가운데 것이 앞뒤 사이에서만 안전하기 때문이다 (`inject.apply_plan`의
        # `avoid` 설명 — 주입 뒤에 고치면 준비 재검사와 표 식별이 둘 다 샌다).
        apply_plan(p.plan, reuse=False, avoid=avoid)
    # **실제 캔버스 장수를 읽는다** — 장수가 그룹의 유일한 신원인데, 씨앗이나
    # 센티널 레이어, 흘린 스탬프가 캔버스를 플랜보다 키울 수 있다. 저장 전에
    # 읽어 두면 부르는 쪽이 신원을 실제값으로 갱신한다.
    n_now = template.canvas_count()
    if n_now and n_now != p.layers:
        log(msg("  캔버스 실제 {now:,}장 (플랜 {layers:,} + 준비 잔여) — "
                "그룹 신원을 {now:,}장으로 갱신한다", now=n_now, layers=p.layers))
        p.layers = n_now
    with clock.stage("prepare.save", of=p.group, n=p.layers):
        Driver(d.hwnd).save_group(p.group)
    log(msg("  저장 슬롯 '{group}' ({layers:,}장)",
            group=p.group, layers=p.layers))


def prepare_group_reuse(d: Driver, p: "Placement | GroupLoad",
                        saved_layers: int, log=print,
                        clock: Clock | None = None) -> None:
    """**저장된 그룹을 다시 열어 플랜을 주입하고 덮어쓴다** — 재스탬프 회피.

    같은 도안·같은 장수인데 **배치·색만 바뀐 재적용**에서, 2000장을 다시
    스탬프(판당 ~19분)하는 대신 이미 만들어 둔 그룹을 열어 값만 새로 쓴다
    (열기+주입+재저장 수십 초). 조각은 전부 실측 확립돼 있다 (2026-08-24):

    1. `bodyedit.open_saved_group` — '내 비닐 그룹' 그리드에서 장수로 찾아
       편집 캔버스로 연다 (다시 연 그룹은 **"1-N 접힌 그룹"** 중첩 구조다).
    2. `template.plant_sentinel` + `inject.find_folded_table` — 센티널로
       접힌 그룹 **내부 표**를 찾는다 (평면 표 검색은 이 구조에서 실패한다).
    3. `inject.write_plan_to_table` — 그 표에 플랜 값을 쓴다 (편집 캔버스의
       렌더 소스라 화면이 바뀐다).
    4. `bodyedit.remove_sentinel` — 센티널을 잘라 원래 장수로 되돌린다
       (신원 누적 없음).
    5. `bodyedit.resave_overwrite` — 슬롯에 덮어쓴다 (새 슬롯 저장은 다시 연
       그룹에서 게임이 커밋 안 한다).

    **실패하면 예외를 올린다** — 부르는 쪽(`prepare_groups`)이 새로 만들기로
    폴백한다. 장수가 바뀌는 경우(차분 스탬프 필요)는 아직 이 길을 안 탄다.
    """
    from ...auto import design
    from ...engine.model import LayerPlan
    from ...game.inject import (Layout, Proc, find_folded_table, find_pid,
                               write_plan_to_table)
    from .. import template

    clock = clock or Clock()
    b = BodyEditor(d)
    with clock.stage("reuse.open", of=p.group, n=saved_layers):
        cc = b.open_saved_group(saved_layers)
    with clock.stage("reuse.inject", of=p.group, n=cc):
        xy = template.plant_sentinel()
        pid = find_pid()
        if not pid:
            raise DriverError(msg("FH6 프로세스를 못 찾았다 (재사용 주입)"))
        proc = Proc(pid)
        layout = Layout.load()
        try:
            got = find_folded_table(proc, layout, xy, cc)
            if got is None:
                raise DriverError(msg("{group}: 접힌 그룹 내부 표를 못 찾았다",
                                      group=p.group))
            tbl, tcount = got
            plan = LayerPlan.load(p.plan)
            wrote = write_plan_to_table(proc, tbl, plan, layout, tcount)
            log(msg("  재사용 주입 {wrote}/{total:,}장 (표 0x{table:x})",
                    wrote=wrote, total=tcount, table=tbl))
        finally:
            proc.close()
        b.remove_sentinel()
    with clock.stage("reuse.save", of=p.group):
        b.resave_overwrite(saved_layers)
    design.back_to_menu(d)
    log(msg("  저장 그룹 재사용 완료 — '{group}' ({layers:,}장)",
            group=p.group, layers=saved_layers))


def prepare_groups(cfg: Config, prog: dict, log=print,
                   clock: Clock | None = None) -> None:
    from ...auto import design

    clock = clock or Clock()

    # 진행 파일의 장수는 **준비가 끝난 뒤의 실제 장수**다 — 장수 충돌을 피해
    # 패딩을 덧대거나(`_dodge_count`) 센티널이 한 장 더 붙으면 구성 파일의
    # 선언값보다 크다. 그래서 "같으냐"로 비교하면 **끝난 그룹을 다시 만든다**
    # (실측: 재개가 1,619장 그룹을 두고 1,620장으로 다시 12분을 썼다).
    # 크거나 같으면 준비된 것으로 보고, 그때 늘어난 장수를 배치에 전파한다 —
    # 불러오기도 배치도 전부 **장수로** 그룹을 찾기 때문이다.
    # 재사용 판정 기준 시각 — 진행 파일보다 **새 플랜**은 값이 바뀐 것이다.
    # 판정 전에 캡처한다 (`save_progress`가 뒤에서 이 파일을 갱신하므로).
    prog_mtime = (cfg.progress_path.stat().st_mtime
                  if cfg.progress_path.exists() else 0.0)
    todo = {}
    reuse = {}
    for key, p in cfg.groups().items():
        done = prog["groups"].get(key)
        if done is not None and done >= p.layers:
            # **플랜이 진행 파일보다 새롭고 장수가 같으면** 값만 바뀐 재적용이다
            # — 저장 그룹을 열어 새 값을 주입한다 (2000장 재스탬프 회피). 장수가
            # 바뀌었으면(차분 스탬프 필요) 아직 새로 만든다. 재사용이 실패하면
            # 아래에서 새로 만들기로 폴백하므로 그림은 늘 맞는다.
            plan_new = False
            try:
                plan_new = (Path(key).exists()
                            and Path(key).stat().st_mtime > prog_mtime)
            except OSError:
                plan_new = False
            if plan_new and done == p.layers:
                reuse[key] = p
                continue
            for q in cfg.placements:
                if q.copy_from is None and str(q.plan) == key:
                    q.layers = done
                for g in q.pre_groups + q.groups:
                    if str(g.plan) == key:
                        g.layers = done
            continue
        todo[key] = p
    if not todo and not reuse:
        log(msg("그룹 준비: 할 것 없다 (진행 파일 기준)"))
        return
    d = Driver()
    # ---- 재사용 먼저 (저장 그룹 열어 값만 주입) ----
    for key, p in list(reuse.items()):
        done = prog["groups"][key]
        log(msg("그룹 '{group}' 재사용 ({layers:,}장) — 저장 그룹을 열어 값만 다시 쓴다",
                group=p.group, layers=done))
        try:
            prepare_group_reuse(d, p, done, log=log, clock=clock)
            for q in cfg.placements:
                if q.copy_from is None and str(q.plan) == key:
                    q.layers = done
                for g in q.pre_groups + q.groups:
                    if str(g.plan) == key:
                        g.layers = done
            save_progress(cfg, prog)          # 진행 파일 mtime 갱신 (다음엔 재사용 안 함)
        except DriverError as e:
            log(msg("그룹 '{group}' 재사용 실패 ({err}) — 새로 만들기로 폴백한다",
                    group=p.group, err=e))
            todo[key] = p
    if not todo:
        design.back_to_menu(d)
        return
    with clock.stage("prepare.scan"):
        have = scan_saved_groups(d, log=log)
    for key, p in todo.items():
        if p.layers in have:
            # **장수는 그룹의 유일한 신원이다** — 겹치면 남의 그룹과 못 가른다
            # (2026-08-19 실측: celtest-04와 -08 합성이 둘 다 1,688장 —
            # "이미 있다"로 건너뛰고 제로투 그룹을 듀랑고에 불러왔다). 투명
            # 패딩 레이어로 장수를 비켜 가고, 같은 그룹을 쓰는 모든 배치의
            # 장수를 갱신한다 (불러오기도 장수로 찾으므로).
            p = _dodge_count(p, have, log=log)
            for q in cfg.placements:
                if str(q.plan) == key:
                    q.layers = p.layers
                    q.plan = p.plan
                for g in q.pre_groups + q.groups:
                    if str(g.plan) == key:
                        g.layers = p.layers
                        g.plan = p.plan
        log(msg("그룹 '{group}' 준비 ({layers:,}장) — {minutes}분쯤 걸린다",
                group=p.group, layers=p.layers,
                minutes=max(1, int(p.layers * 0.44 / 60))))
        with clock.stage("prepare", of=p.group, n=p.layers):
            prepare_group(d, p, log=log, clock=clock, avoid=have)
        # 준비가 장수를 키웠을 수 있다 (센티널 레이어) — 신원을 실제값으로
        # 전파한다. 배치·불러오기가 전부 장수로 찾으므로 여기서 안 맞추면
        # 엉뚱한 그룹을 문다. 짝은 **플랜 경로**로 맞춘다 (표시 이름은 30자로
        # 잘려 겹칠 수 있다).
        for q in cfg.placements:
            if q.copy_from is None and str(q.plan) in (key, str(p.plan)):
                q.layers = p.layers
            for g in q.pre_groups + q.groups:
                if str(g.plan) in (key, str(p.plan)):
                    g.layers = p.layers
        have.add(p.layers)
        prog["groups"][key] = p.layers
        save_progress(cfg, prog)
    design.back_to_menu(d)


def _dodge_count(p: Placement, have: set[int], log=print) -> Placement:
    """합성 플랜에 **투명 패딩 레이어**를 덧대 장수를 빈 자리로 옮긴다.

    패딩은 마지막 레이어 자리에 alpha 0·최소 크기로 얹으므로 그룹의 bbox도
    그림도 안 바뀐다 — 바뀌는 것은 장수(신원)뿐이다.
    """
    import dataclasses

    with open(p.plan, encoding="utf-8") as f:
        d = json.load(f)
    pad = dict(d["layers"][-1])
    pad.update({"alpha": 0.0, "sx": 0.01, "sy": 0.01, "label": "pad",
                "mask": False, "stroke": -1})
    n = p.layers
    while n in have:
        d["layers"].append(dict(pad))
        n += 1
    out = Path(str(p.plan) + f".dodge{n}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False)
    log(msg("그룹 '{group}': {layers:,}장이 기존 그룹과 겹친다 — "
            "투명 패딩 {pad}장을 덧대 {n:,}장으로 비켜 간다",
            group=p.group, layers=p.layers, pad=n - p.layers, n=n))
    return dataclasses.replace(p, plan=out, layers=n)


def scan_saved_groups(d: Driver, log=print) -> set[int]:
    """지금 게임에 저장돼 있는 비닐 그룹들의 **장수** 집합.

    차체 에디터의 그룹 그리드를 훑는다 — 이름은 못 읽으므로 장수만 모은다.
    """
    from ...auto import design

    b = BodyEditor(d)
    design.back_to_menu(d)
    design.goto_row(d, design.ROW_BODY_VINYL)
    b.d._step("enter", lambda: b.screen() == "list", msg("차체 에디터 진입"),
              tries=3, wait=4.0)
    try:
        # 저장된 그룹이 하나도 없는 판(새 프로필·그룹을 다 지운 뒤)에서는 그리드가
        # 아예 안 열린다 — 빈 집합이 옳은 답이다 (준비가 처음부터 다 만든다).
        got = b.scan_groups() if b.open_group_grid(allow_empty=True) else set()
    finally:
        for _ in range(4):
            if design.menu_open(d.cap()):
                break
            gio.press("esc")
            time.sleep(1.2)
        design.back_to_menu(d)
    log(msg("게임에 저장된 비닐 그룹 {n}개 (장수 {counts})",
            n=len(got), counts=sorted(got)))
    return got
