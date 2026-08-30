"""면에 **직접** 놓는 것 — 도형 위저드와 게임 텍스트 도구.

그룹을 불러 앉히는 것이 아니라 면 위에 한 장씩 세운다. 같은 (색, 도형) 묶음은
위저드 한 바퀴 + Y 스탬프로 몰아 놓는다(`_add_shape_run`) — 장당 위저드를 다시
여는 것보다 훨씬 싸다."""

from __future__ import annotations

import time

from ...engine.model import game_hsb
from ...game import io as gio
from ...i18n import msg
from ..bodyedit import BodyEditor
from ..driver import Driver, DriverError
from .progress import Clock


def _soft_xy(d: Driver, axis: str, target: float,
             press_tool: bool = True, log=print) -> None:
    """이동 축을 목표로 두되 **클램프에 걸려도 죽지 않는다** (`Driver.set_axis_soft`).

    위저드 도형과 그룹 불러오기가 **같은 자를 나눠 쓴다** — 면 도형을 그룹
    주입으로 옮기면서 그 자가 한쪽에만 있는 것이 드러났다 (인테그라 front의
    그룹 x가 목표 9.4에 8.0에서 멈춰 실행이 죽었다).
    """
    d.set_axis_soft(axis, target, press_tool=press_tool, log=log)


def add_shape_job(b: BodyEditor, spec: dict, log=print,
                  clock: Clock | None = None, of: str = "",
                  fav=None) -> bool:
    """`_add_shape_job` + **시간 기록**.

    이 한 장이 얼마나 걸리는지가 "면 도형을 그룹 주입으로 바꿀 값이 있나"의
    유일한 판정 근거다 (도형마다 [도형·HSB 3축·rot·sx·sy·x·y] 폐루프 일곱).
    버린 도형도 시간을 쓰므로 성패와 무관하게 적는다.
    """
    clock = clock or Clock()
    t0 = time.time()
    try:
        return _add_shape_job(b, spec, log=log, fav=fav)
    finally:
        clock.add("place.shape", time.time() - t0,
                  of=f"{of}/{spec.get('shape')}" if of else str(spec.get("shape")),
                  n=1)


def _rgb_key(spec: dict) -> tuple[int, int, int]:
    return tuple(int(v) for v in (spec.get("rgb") or (255, 255, 255)))


def _shape_batches(specs: list[dict]) -> list[list[dict]]:
    """면 도형 명세를 **위저드 한 바퀴로 놓을 묶음**으로 자른다.

    묶음 = 같은 (색, 도형)이고, **같은 색 연속 구간 안에서만** 모은다. 그 구간
    안에서는 순서를 바꿔도 그림이 같다 — 같은 불투명 색은 겹쳐도 z순서가 안
    보인다 (관통 띠의 이빨·지붕 블랙아웃이 다 이 꼴이다: 한 색에 도형만
    번갈아 나온다). 색이 다른 명세와의 앞뒤는 그림이므로 구간을 안 넘는다.
    """
    out: list[list[dict]] = []
    i = 0
    while i < len(specs):
        j = i
        while j < len(specs) and _rgb_key(specs[j]) == _rgb_key(specs[i]):
            j += 1
        by: dict[str, list[dict]] = {}
        for s in specs[i:j]:
            by.setdefault(str(s.get("shape")), []).append(s)
        out.extend(by.values())
        i = j
    return out


def add_shape_jobs(b: BodyEditor, specs: list[dict], log=print,
                   clock: Clock | None = None, of: str = "",
                   fav=None) -> int:
    """면 도형 명세 목록을 **묶음으로** 놓는다. 반환: 실제로 선 장수.

    같은 (색, 도형) 묶음은 위저드 한 바퀴 + Y 스탬프로 간다 (`_add_shape_run`) —
    면 도형 한 장이 폐루프 일곱(실측 22~45초)인데 그중 위저드·색 지정이
    절반이라, 이빨 띠·산포처럼 같은 색 연속이 많은 구성에서 그 몫이 통째로
    빠진다. 스탬프가 이 화면에서 안 서는 것으로 판정되면(묶음에서 커밋 한
    장만 남으면) 장별 경로로 물러나고 이 에디터 세션에서는 다시 안 시도한다.
    """
    clock = clock or Clock()
    total = 0
    for batch in _shape_batches(list(specs)):
        if len(batch) == 1 or not getattr(b, "stamp_ok", True):
            for spec in batch:
                total += bool(add_shape_job(b, spec, log=log, clock=clock,
                                            of=of, fav=fav))
            continue
        t0 = time.time()
        landed = _add_shape_run(b, batch, log=log, fav=fav)
        clock.add("place.shape", time.time() - t0,
                  of=(f"{of}/{batch[0].get('shape')}x{len(batch)}" if of
                      else f"{batch[0].get('shape')}x{len(batch)}"),
                  n=max(1, landed))
        total += landed
        if landed >= len(batch):
            continue
        if landed == 1:
            # Y 스탬프가 이 화면에서 안 선다는 서명이다 — 스탬프가 다 무시되면
            # 마지막 명세(Enter 커밋) **한 장만** 남는다. 나머지를 장별로 다시
            # 놓고 스탬프는 접는다 (같은 에디터 세션에서 거동이 안 바뀐다).
            log(msg("  스탬프가 안 선다 ({landed}/{total}) — 장별 경로로 물러난다",
                    landed=landed, total=len(batch)))
            b.stamp_ok = False
            retry = batch[:-1]
        elif landed == 0:
            # 한 장도 안 섰다 — 스탬프 문제가 아니라 첫 장의 축·위저드 실패다
            # (스탬프였다면 커밋 한 장은 남는다). 묶음 전체를 장별로 다시 놓는다.
            retry = batch
        else:
            # 드문 키 드롭 — 모자란 만큼 꼬리를 장별로 놓는다. 같은 색·같은
            # 도형이라 어느 장이 떨어졌든 겹쳐 놓여도 그림이 안 바뀐다.
            log(msg("  스탬프 {landed}/{total}장 — 남은 {left}장을 장별로 놓는다",
                    landed=landed, total=len(batch), left=len(batch) - landed))
            retry = batch[landed:]
        for spec in retry:
            total += bool(add_shape_job(b, spec, log=log, clock=clock,
                                        of=of, fav=fav))
    return total


HSB_TOL = 0.01


def _shape_axes(b: BodyEditor, spec: dict, log=print,
                prev_rot: float | None = None) -> float:
    """도형 한 장의 변형(회전 → 스케일 → 이동)을 앉힌다. 반환: 앉힌 회전값.

    `_add_shape_job`의 축 규약을 그대로 떼어 낸 것이다 — 스탬프 런
    (`_add_shape_run`)이 장마다 같은 규약을 타야 해서 함수로 세웠다.
    `prev_rot`가 목표와 같으면 회전을 건너뛴다 — 스탬프는 변형 상태를 이으므로
    (`run_plan`과 같은 실측 규약) 이빨 띠처럼 회전이 다 0인 묶음에서 폐루프
    하나가 통째로 빠진다.

    스케일 부호는 지킨다 — 면에 따라 새 레이어가 음수 스케일(미러)로 시작하고,
    0을 건너는 폐루프는 발산한다. 사각·원은 미러해도 같은 모양이라 부호만
    맞추면 된다. 부호는 장마다 **현재값에서 다시 읽는다** — 스탬프 뒤에도 직전
    장의 부호가 그대로 남아 있어 읽은 값이 곧 이번 장의 출발 부호다.
    """
    from ...game import ocr

    d = b.d
    rot = float(spec.get("rot") or 0.0) % 360.0
    if prev_rot is None or abs(rot - prev_rot) > 1e-9:
        d.set_axis("rot", rot)
    gio.press("2")                   # 스케일 도구 — 부호를 읽는다
    time.sleep(0.3)
    cur = ocr.read_stable(b.hwnd, "x", tries=20)
    sign = -1.0 if (cur is not None and cur < 0) else 1.0
    # 면 스케일은 **원형(랩) 축**이다 (2026-08-19 front 실측: sx가 ±2.3쯤에서
    # 반대 부호로 감긴다 — 1.96 →d→ -2.2, -2.14 →a→ +1.6). 상한 밖 목표는
    # 수렴이 원리적으로 불가라, 실패하면 5%씩 줄여 상한 안쪽에 앉힌다 —
    # 띠가 몇 % 짧아지는 것은 그림에 무해하고 도형을 잃는 것보다 낫다.
    # 랩으로 부호가 넘어가 있으면 그쪽 부호로 따라간다 (도형은 대칭이라 무해).
    fail = None
    for factor in (1.0, 0.95, 0.9, 0.85):
        try:
            for axis, val in (("sx", float(spec["sx"])), ("sy", float(spec["sy"]))):
                try:
                    d.set_axis(axis, sign * val * factor, press_tool=False)
                except DriverError:
                    # 홀드가 어긋난 것일 수 있다 — 느린 화살표 전용으로 재시도
                    time.sleep(1.0)
                    d.set_axis(axis, sign * val * factor,
                               press_tool=False, gentle=True)
                    log(msg("    {axis} 느린 화살표로 앉혔다", axis=axis))
            fail = None
            break
        except DriverError as e:
            fail = e
            now = ocr.read_stable(b.hwnd, "x", tries=20)
            if now is not None and now * sign < 0 and abs(now) > 0.5:
                sign = -sign
                log(msg("    스케일이 랩 너머에 있다 (현재 {now:g}) — 부호를 따라간다",
                        now=now))
            log(msg("    스케일 ×{factor:g} 실패 — 백오프 ({err})",
                    factor=factor, err=e))
    if fail is not None:
        raise fail
    _soft_xy(d, "x", float(spec["x"]), log=log)
    _soft_xy(d, "y", float(spec["y"]), press_tool=False, log=log)
    return rot


def _add_shape_job(b: BodyEditor, spec: dict, log=print, fav=None) -> bool:
    """도형 하나를 **면에 직접** 놓는다 (도형 위저드 — `surface_probe`와 같은 길).
    반환: 확정했나 — 실패하면 미확정 폐기(Esc)하고 False다.

    관통 띠·산포 모티프가 쓴다. 소형 그룹 주입은 레이어 표 식별이 모호해서
    (2장짜리 표 후보 9,344건 — 엉뚱한 표에 쓰인다) 위저드로 간다. 색은
    즐겨찾기 스택(`auto.fav.FavStack`)이 있으면 인덱스 점프(~3초)이고 없으면
    HSB 3축 폐루프(~10초)다 — `run_plan`과 같은 규약이고, 스택이 실제와
    어긋나면 행 색 대조가 잡아 HSB로 물러나 자기 치유한다.
    """
    d = b.d
    log(msg("  면 도형 ({shape} · sx {sx:g} · sy {sy:g})",
            shape=spec['shape'], sx=spec['sx'], sy=spec['sy']))
    n_before = b.count_stable()
    try:
        b.open_wizard()
        d.select_shape(spec["shape"])
        d.confirm_shape_and_color(hsb=game_hsb(*spec["rgb"]), fav=fav,
                                  tol=HSB_TOL)
        _shape_axes(b, spec, log=log)
        b.commit()
        return True
    except DriverError as e:
        log(msg("    도형을 버린다 — {err}", err=e))
        # 커밋이 **늦게 먹었을 수 있다** (2026-08-19 실측: 확정 후 리스트 전환이
        # 4초를 넘겨 폐기 루프가 헛돌았다 — 마지막 raise 시점엔 이미 list였다).
        # 먼저 Esc 없이 기다리고, 리스트가 오면 카운터로 커밋 여부를 가른다.
        for i in range(8):
            if b.screen() == "list":
                n_after = b.count_stable()
                if (n_before is not None and n_after is not None
                        and n_after > n_before):
                    log(msg("    …커밋이 늦게 도착했다 — 놓은 것으로 센다"))
                    return True
                return False
            if i >= 3:                   # 리스트가 안 오면 그때부터 미확정 폐기
                gio.press("esc")
            time.sleep(1.0)
        raise DriverError(msg("도형 폐기 후 리스트 복귀 실패 (화면 {screen})",
                              screen=b.screen()))


def _add_shape_run(b: BodyEditor, specs: list[dict], log=print, fav=None) -> int:
    """같은 (색, 도형) 묶음을 **위저드 1회 + Y 스탬프**로 놓는다. 반환: 선 장수.

    `run_plan.draw_group`과 같은 실측 규약이다 — Y = 사본 즉시 커밋·편집 유지.
    변형은 장마다 절대값으로 앉히므로 스탬프 하나가 떨어져도 다음 장은 안
    어긋난다. 중간에 죽어도 그때까지 스탬프한 장은 이미 커밋돼 있다 — 면
    카운터로 세어 돌려주고, 모자란 만큼은 부르는 쪽(`add_shape_jobs`)이 장별로
    다시 놓는다. 묶음이 같은 색·같은 도형이라 어느 장이 떨어졌는지 몰라도
    꼬리를 다시 놓으면 된다 (겹쳐 놓여도 같은 불투명 색이라 그림이 안 바뀐다).
    """
    d = b.d
    spec0 = specs[0]
    log(msg("  면 도형 묶음 ({shape} × {n} · 같은 색 — 스탬프)",
            shape=spec0['shape'], n=len(specs)))
    n0 = b.count_stable()
    prev_rot: float | None = None
    try:
        b.open_wizard()
        d.select_shape(spec0["shape"])
        d.confirm_shape_and_color(hsb=game_hsb(*spec0["rgb"]), fav=fav,
                                  tol=HSB_TOL)
        for k, spec in enumerate(specs):
            prev_rot = _shape_axes(b, spec, log=log, prev_rot=prev_rot)
            if k < len(specs) - 1:
                gio.press("y", hold_s=0.09)  # 스탬프 = 사본 커밋, 편집 유지
                time.sleep(0.5)              # 차체 장면은 그룹 캔버스보다 무겁다
            else:
                b.commit()
    except DriverError as e:
        log(msg("    묶음이 끊겼다 ({err}) — 스탬프된 장까지만 남긴다", err=e))
        # 커밋이 늦게 먹었을 수 있는 것까지 카운터가 가른다 (`_add_shape_job`의
        # 늦은 커밋과 같은 사정) — 리스트로 돌아와 세면 그게 답이다.
        for i in range(8):
            if b.screen() == "list":
                break
            if i >= 3:                   # 리스트가 안 오면 그때부터 미확정 폐기
                gio.press("esc")
            time.sleep(1.0)
        else:
            raise DriverError(msg("묶음 폐기 후 리스트 복귀 실패 (화면 {screen})",
                                  screen=b.screen()))
    n1 = b.count_stable()
    if n0 is None or n1 is None:
        # 카운터를 못 읽으면 몇 장 섰는지 알 길이 없다 — 다 선 것으로 치고
        # 최종 검증(`place_one`의 n1-n0 대조)에 맡긴다.
        return len(specs)
    return max(0, n1 - n0)


def add_text_job(b: BodyEditor, spec: dict, log=print) -> None:
    """구성이 낸 글자 명세를 **차체 면에 직접** 넣는다 (게임 텍스트 도구).

    그룹에 넣지 않는 이유는 미러다 — 그룹을 뒤집으면 글자까지 뒤집혀 우측면이
    거울 글씨가 된다. 면에 따로 넣으면 좌우 다 바로 읽히고, 글자만 올리는 면
    (리어 로고)도 같은 길로 열린다.

    자리는 **면 유닛의 중심 기준**이다 — 게임은 첫 글리프의 설계 원점을 붙잡으므로
    오프셋을 `engine.textvinyl.text_metrics`가 계산해 준다.
    """
    from ...engine import textvinyl as tv
    from .. import gametext as gt

    cx, cy = spec["center"]
    h = float(spec["height"])
    font = spec.get("font") or tv.DEFAULT_FONT
    col = tuple(spec.get("color") or (255, 255, 255))
    edge = spec.get("outline")
    m = tv.text_metrics(spec["text"], font=font, height=h)
    shadow = spec.get("shadow")
    job = gt.TextJob(
        text=spec["text"], font=font, scale=round(m["scale"], 3),
        # 게임 값 칸은 0~360으로 표시한다 — 음수 각을 그대로 넣으면 판독 폐루프가
        # 352와 -8을 서로 다른 값으로 보고 싸운다
        rot=float(spec.get("rot") or 0.0) % 360.0,
        hsb=game_hsb(*col), outline=game_hsb(*edge) if edge else None,
        shadow=game_hsb(*shadow) if shadow else None,
        shadow_shift=tuple(spec.get("shadow_shift") or (0.0, 0.0)),
        center=(cx, cy), height=h)
    gt.add_text(b.d, job, log=log, host=b, count=b.count_stable)
