"""이타샤 전체 실행 — 준비 → 배치 → 커밋."""

from __future__ import annotations

import time
from pathlib import Path

from ...game import body
from ..bodyedit import BodyEditor
from ..driver import Driver
from .config import Config, load_config
from .progress import Clock, load_progress, save_progress, timing_summary
from .cartabs import verify_car
from .groups import prepare_groups
from .place import place_all


def finish(b: BodyEditor, apply: bool, log=print) -> None:
    if not apply:
        log("적용하지 않고 나간다 (설정 apply=false)")
        b.exit_editor(apply=False)
        return
    log("에디터를 나가 '현재 자동차에 적용'을 고른다")
    if not b.exit_editor(apply=True):
        log("바뀐 게 없어 저장 대화상자가 안 떴다 — 적용할 것이 없다")
        return
    log("적용 완료")


def describe(cfg: Config, finish: bool = True) -> str:
    """구성을 사람이 읽는 표로. `finish=False`면 마무리 줄을 뺀다 —
    파일로 내보내는 길(`engine.fls.studio`)은 게임을 안 건드리므로 '현재
    자동차에 적용'이 거짓말이 된다."""
    lines = [f"이타샤 구성: {cfg.path}"]
    if cfg.car:
        lines.append(f"  기준 차량 메모: {cfg.car}"
                     + (f"  (면 지도는 설치 파일 {cfg.media})" if cfg.media else ""))
    if cfg.paint is not None:
        lines.append(f"  베이스 도색: HSB {cfg.paint} (자동차 도색 메뉴)")
    for p in cfg.placements:
        cap = body.surface_cap(p.surface, cfg.tabs)
        if p.copy_from is not None:
            lines.append(f"  {p.surface:<13} ← 반대편({p.copy_from}) 복사")
            continue
        deco = "".join(
            f"  {tag} '{txt}'" for tag, txt in
            (("글자", " / ".join(t["text"] for t in p.texts)),) if txt)
        if p.shapes:
            deco += f"  면 도형 {len(p.shapes)}"
        if p.post_shapes:
            deco += f"  덮개 도형 {len(p.post_shapes)}"
        if p.pre_groups:
            deco += "  보조그룹 " + " ".join(
                f"{g.group}({g.layers:,})" for g in p.pre_groups)
        if p.layers == 0 and p.groups:
            tot = sum(g.layers for g in p.groups)
            lines.append(f"  {p.surface:<13} 손 배치 도안 {len(p.groups)}개 "
                         f"{tot:,}장/{f'{cap:,}' if cap else '?'}{deco}")
            for g in p.groups:
                lines.append(f"      {g.group:<14} {g.layers:>6,}장 x={g.x:g} "
                             f"y={g.y:g} scale={g.scale:g} rot={g.rot:g}"
                             + ("  좌우반전" if g.mirror else ""))
            continue
        if p.layers == 0:
            lines.append(f"  {p.surface:<13} (그룹 없음){deco}")
            continue
        lines.append(
            f"  {p.surface:<13} {p.group:<14} {p.layers:>6,}장/{cap or '?':>5} "
            f"x={p.x:g} y={p.y:g} scale={p.scale:g} rot={p.rot:g}"
            + ("  미러" if p.mirror else "") + deco)
    if finish:
        lines.append(f"  마무리: {'현재 자동차에 적용' if cfg.apply else '적용 안 함'}")
    return "\n".join(lines)


def run(config: Path, restart: bool = False, prepare: bool = True,
        yes: bool = False, dry_run: bool = False, replace: bool = True,
        fit: bool = True, media: str | None = None, log=print) -> int:
    """이타샤 전체 실행. 반환: 종료 코드."""
    cfg = load_config(config, media=media)
    log(describe(cfg))
    if dry_run:
        # 지난 실행의 시간 기록이 있으면 같이 보인다 — 무엇을 고칠지 정하는
        # 자리가 여기다 (실행을 다시 돌리지 않고 읽는 유일한 길이다)
        past = timing_summary(load_progress(cfg))
        if past:
            log("\n" + past)
        log("\n--dry-run — 게임을 건드리지 않았다")
        return 0
    # 동의는 **아무것도 하기 전에** 받는다 — 그룹 준비가 20분짜리라 그걸 다
    # 돌려 놓고 "동의가 없다"로 멈추면 사람 시간을 버린다
    if cfg.apply and not yes:
        log("\n마지막에 **현재 자동차의 디자인을 덮는다**. 계속하려면 --yes를 주거나\n"
            "  구성 파일에 \"apply\": false 를 두고 배치만 확인할 것.\n"
            "  (되돌리기: 게임 [디자인 및 도색]에서 X = 도색/비닐 기본으로 되돌리기)")
        return 2
    prog = {"groups": {}, "placed": []} if restart else load_progress(cfg)
    if restart and cfg.progress_path.exists():
        cfg.progress_path.unlink()
    # 구성이 진행 파일보다 새로우면 배치 기록은 낡은 것이다 — 배치 키가 면
    # 이름뿐이라 그대로 두면 새 구성을 통째로 건너뛰고 "적용할 것이 없다"로
    # 끝난다 (2026-08-24 실측). `groups`는 남긴다: 비닐 그룹은 게임에 실재하고
    # 준비 단계가 장수로 다시 맞춰 본다.
    elif prog["placed"] and cfg.progress_path.exists() \
            and cfg.path.stat().st_mtime > cfg.progress_path.stat().st_mtime:
        log("구성이 진행 파일보다 새롭다 — 배치 기록을 버리고 전 면을 다시 올린다")
        prog["placed"] = []
        prog.pop("done", None)
    clock = Clock(cfg, prog)
    t_run = time.time()
    # 차가 맞는지 **아무것도 건드리기 전에** 본다 — 도색부터가 즉시 커밋이고
    # 그룹 준비는 20분짜리다. 여는 데 십 초면 된다.
    with clock.stage("verify_car"):
        verify_car(cfg, log=log)
    # 베이스 도색 — **차체 배치보다 먼저** (사람 순서와 같다: 도색 위에 비닐).
    # 도색은 `현재 자동차에 적용`으로 즉시 커밋되므로 apply=false면 건드리지 않는다.
    if cfg.paint is not None and cfg.apply:
        if prog.get("painted") == list(cfg.paint):
            log("베이스 도색: 이미 칠했다 (진행 파일 기준)")
        else:
            from .. import paintcar

            log(f"베이스 도색: HSB {cfg.paint} (자동차 도색 → 전체 도색)")
            with clock.stage("paint"):
                paintcar.set_paint(Driver(), cfg.paint, log=log)
            prog["painted"] = list(cfg.paint)
            save_progress(cfg, prog)
    elif cfg.paint is not None:
        log("베이스 도색: apply=false라 건너뛴다 (도색은 즉시 커밋이라)")
    if prepare:
        prepare_groups(cfg, prog, log=log, clock=clock)
    else:
        log("그룹 준비 건너뜀 (--no-prepare)")
    b = place_all(cfg, prog, log=log, replace=replace, fit=fit, clock=clock)
    with clock.stage("finish"):
        finish(b, cfg.apply, log=log)
    prog["done"] = True
    clock.add("run", time.time() - t_run)
    log(f"\n끝. 진행 파일 {cfg.progress_path.name}")
    log(timing_summary(prog))
    return 0
