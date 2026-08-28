r"""주입용 빈 템플릿 그룹 — 캔버스 레이어 수를 목표까지 채운다 (창 조작).

**주입은 레이어 개수를 못 늘린다** — 값을 덮어쓸 뿐이라 N장짜리 플랜을 넣으려면
캔버스에 N장이 미리 있어야 한다 (`game/inject.py`). 그 N장은 **거의** 무엇이든
상관없으므로 같은 도형·같은 색으로 Y 스탬프만 되풀이해 만든다. 변형을 하나도
안 건드리므로 장당 비용이 스탬프 하나(0.44초)다 — 3,000장 22분이고, 창 조작으로
도안을 그리는 8.6시간과는 다른 값이다.

**"거의"가 붙는 자리가 씨앗이다** (`seed`, 65차). 저장한 그룹을 다시 열면 그
그룹은 **제 저장본이 참조한 도형 에셋만** 그릴 수 있다 — 주입한 도형 id가 그
밖이면 조용히 템플릿 도형(타원)으로 그려진다. 그래서 전부 A_02로 채운 템플릿은
**한 번 저장하고 나면 못 쓴다**. 어휘의 도형마다 한 장씩 심어 두면 그 템플릿은
다시 열어도 전 어휘가 서고, 매 세션 23분을 다시 안 쓴다.

**어휘 전체를 심는 것은 다시 쓰기 위한 값이다** — 다시 열 일이 없는 그룹에는
그 값이 없다 (`ensure_ready(reuse=False)`). 이타샤는 그룹마다 새 캔버스를 만들어
주입하고 저장하면 끝이라, 안 쓰는 도형까지 심으면 위저드 한 바퀴씩이 그대로
버려진다.

부르는 곳은 `game.inject.apply_plan(template=True)`다 — 창의 [메모리 주입]과
`inject --template`이 이 길로 온다.

레이어 수는 좌하단 **"N / 3000" 카운터**로 읽으므로 **레이어 리스트 화면**이어야
한다 (`ocr.read_layer_count`). 스탬프가 몇 장 흘렸어도 모자란 만큼 다시 채우므로
결과는 정확히 `target`이다. 시작·종료 상태는 둘 다 레이어 리스트다.

**목표보다 많으면 손대지 않는다.** 표는 앞에서부터 잡히므로 주입은 앞 `target`장만
덮고 남는 레이어는 템플릿 도형 그대로 캔버스에 남는다.

그 자리에서 **이쪽으로 오지 말고 `inject --canvas M`을 쓰는 편이 낫다** — 남는
레이어를 주입이 캔버스 밖으로 밀어(알파 0·최소 크기) 안 보이게 한다. 3,000장
템플릿 하나를 계속 열어 두고 어떤 플랜이든 바로 올릴 수 있으므로, 그림마다
장수를 맞추느라 드는 22분이 통째로 없어진다 (`game/inject.apply_plan`).
"""

from __future__ import annotations

import time
from collections.abc import Callable

from ..game import io as gio, ocr
from .driver import Driver, DriverError
from .run_plan import StopRequested

STAMP_WAIT = 0.35        # run_plan과 같은 값 (Y 스탬프 후 안정 대기) — 시작값
# 스탬프 대기는 **적응한다** — 채우기는 청크마다 카운터로 수확을 재고 모자란
# 만큼 다시 채우므로, 대기를 줄이다 키가 떨어져도 시간만 조금 잃지 그림이
# 안 상한다. 수확이 온전하면 줄이고 떨어지면 되돌린다 (`fill`). 0.44s/장의
# 대부분이 이 고정 대기였다 — 2,200장 그룹 하나에 몇 분이 걸린 자리다.
STAMP_WAIT_MIN = 0.16    # 홀드 0.09 + 이 대기 = 스탬프 하나 0.25s 바닥
SEC_PER_LAYER = 0.44     # 실측 (3,000장 22분) — 남은 시간 어림에 쓴다 (보수적)
SHAPE = "A_02"           # 템플릿 도형 (주입이 도형 id도 덮으므로 무엇이든 된다)
CHUNK = 250              # 위저드 한 번에 몇 장 (중간 확인 간격)


def canvas_count(hwnd: int | None = None) -> int | None:
    """지금 캔버스의 레이어 수. 못 읽으면 None (레이어 리스트 화면이 아닌 경우).

    **창을 안 뺏는다** — PrintWindow 캡처라 게임이 가려져 있어도 읽는다. 그래서
    게임을 건드리기 전에 물어볼 수 있다 (`Driver`는 만드는 순간 포그라운드를
    가져간다)."""
    hwnd = hwnd or gio.find_hwnd()
    return ocr.read_layer_count_stable(hwnd) if hwnd else None


def count(d: Driver) -> int:
    n = ocr.read_layer_count_stable(d.hwnd)
    if n is None:
        raise DriverError("레이어 수를 못 읽었다 — 레이어 리스트 화면이 맞나")
    return n


def vocabulary() -> tuple[str, ...]:
    """씨앗으로 심을 도형 목록 = cel 노선이 낼 수 있는 도형 전부.

    **어휘가 바뀌면 템플릿도 다시 만들어야 한다** — 저장본이 안 쥔 도형은 조용히
    **다른 도형으로** 그려진다(타원 고정이 아니다 — 실측상 배치마다 다르다).
    어휘가 줄기만 했으면 옛 템플릿이 상위집합이라 그대로 쓴다. 지금 그룹이
    무엇을 못 그리는지는 인게임 캔버스를 캡처해 대조하면 나온다.
    """
    from ..engine.catalog import Catalog, default_catalog_path
    from ..engine.celfit import shape_vocabulary

    # 카탈로그를 준다 — 획 어휘는 서술자가 그때그때 고르므로
    # (`celfit.vocabulary.stroke_vocab`) 목록이 카탈로그에 따라 달라진다
    return shape_vocabulary(Catalog(default_catalog_path()))


def seed(d: Driver, shapes: tuple[str, ...],
         stop: Callable[[], bool] | None = None) -> int:
    """도형마다 한 장씩 찍는다 — **다시 연 그룹에서 그 도형이 서게 하는 유일한 길**.

    65차 실측으로 확정된 규칙이다. 비닐 그룹을 저장했다 다시 열면 그 그룹은
    **저장본이 참조한 도형 에셋만** 쥐고 있고, 레코드에 그 밖의 도형 id를 써도
    화면은 안 바뀐다(템플릿 도형 그대로 그려진다). 인게임 `모양 변경`은 새
    에셋을 들여오지만 **레이어 만들기 위저드는 다시 연 그룹에서 못 들여온다** —
    그래서 씨앗은 **템플릿을 처음 만들 때** 심어야 한다.

    확인: 여섯 도형(A_02·A_09·U_04·U_19·U_23·U_35)을 심어 저장하고 다시 연
    그룹에 **순서를 뒤집은** 도형 id를 주입하니 여섯이 그대로 뒤집혀 그려졌다.
    씨앗 없이 같은 것을 하면 여섯 다 타원이 된다.

    씨앗 레이어는 앞자리를 차지하고 주입이 그 위를 덮으므로 흔적이 안 남는다
    (플랜이 씨앗 수보다 짧을 때만 남는다 — 그건 `ensure`의 '많으면 손대지
    않는다'와 같은 사정이다). 장당 3~14초로 위저드 한 바퀴가 통째로 든다.

    한 종이 실패해도 **멈추지 않는다** — 그 도형만 못 서고 나머지는 그대로다.
    실패한 이름을 로그에 적으므로 필요하면 그것만 인게임에서 더 심으면 된다."""
    from .run_plan import _recover_to_list

    n0 = count(d)
    print(f"씨앗 {len(shapes)}종을 심는다 (다시 연 템플릿에서 이 도형들이 선다)",
          flush=True)
    bad = []
    for k, s in enumerate(shapes):
        if stop is not None and stop():
            raise StopRequested(f"STOP 감지 — 씨앗 {k}/{len(shapes)}종에서 중단")
        try:
            d.open_wizard()
            d.select_shape(s)
            d.confirm_shape_and_color()
            d.commit()
        except DriverError as e:
            bad.append(s)
            print(f"  씨앗 {s} 실패({e}) — 건너뛴다", flush=True)
            _recover_to_list(d)
            time.sleep(1.0)
    n = count(d)
    print(f"  씨앗 {n - n0}장 (요청 {len(shapes)})"
          + (f" · 못 심은 것 {bad}" if bad else "") + f" → {n}장", flush=True)
    return n


def fill(d: Driver, target: int, shape: str = SHAPE, chunk: int = CHUNK,
         stop: Callable[[], bool] | None = None) -> int:
    """카운터가 `target`이 될 때까지 스탬프 묶음을 되풀이한다. 반환: 최종 장수.

    `stop`은 **스탬프마다** 불린다. 참이면 지금까지 찍은 것을 커밋하고(위저드를
    연 채로 두면 다음 실행이 리스트에서 시작 못 한다) `StopRequested`를 올린다."""
    n = count(d)
    print(f"템플릿: 현재 {n}장 → 목표 {target}장 "
          f"(예상 {(target - n) * SEC_PER_LAYER / 60:.0f}분)", flush=True)
    wait = STAMP_WAIT
    while n < target:
        if stop is not None and stop():
            raise StopRequested(f"STOP 감지 — 템플릿 {n}/{target}장에서 중단")
        want = min(chunk, target - n)
        t0 = time.time()
        d.open_wizard()
        d.select_shape(shape)
        d.confirm_shape_and_color()          # 기본 흰색 — 색 단계가 없다
        stopped = False
        for _ in range(want - 1):
            if stop is not None and stop():
                stopped = True
                break
            gio.press("y", hold_s=0.09)      # 스탬프 = 사본 커밋, 편집 유지
            time.sleep(wait)
        d.commit()
        time.sleep(0.6)
        got = count(d)
        dt = time.time() - t0
        print(f"  +{got - n:4d}장 (요청 {want}) → {got}/{target}  "
              f"{dt:.0f}초 · 장당 {dt / max(1, got - n):.2f}초"
              + (f" · 대기 {wait:.2f}s" if wait != STAMP_WAIT else ""), flush=True)
        if got <= n:
            if wait < STAMP_WAIT:            # 너무 줄인 대기가 원인일 수 있다
                print(f"  스탬프가 안 늘었다 — 대기를 {STAMP_WAIT}s로 되돌려 재시도",
                      flush=True)
                wait = STAMP_WAIT
                continue
            raise DriverError(f"스탬프가 한 장도 안 늘었다 ({got}장) — 중단")
        # 수확 기반 적응 — 온전히 들어오면 줄이고, 떨어지기 시작하면 물러선다.
        # 다시 채우는 값이 있으므로 공격적으로 줄여도 손해가 작다. 중단으로
        # 일찍 끊긴 청크는 수확 근거가 아니다.
        if not stopped:
            if got - n >= want:
                wait = max(STAMP_WAIT_MIN, round(wait * 0.85, 3))
            elif (got - n) / max(1, want) < 0.9:
                wait = min(STAMP_WAIT, round(wait * 1.3, 3))
        n = got
        if stopped:
            raise StopRequested(f"STOP 감지 — 템플릿 {n}/{target}장에서 중단")
    return n


# 센티널 자리 — 캔버스 밖(±1000 안 어디든 되지만 그림과 안 겹치게 구석)이고
# 게임 입력 스텝(0.5) 위라 판독이 그대로 돌아온다. 주입이 이 슬롯을 안 보이는
# 덮개로 덮으므로 흔적이 안 남는다.
SENT_X, SENT_Y = 741.5, -333.5


def plant_sentinel(log=print) -> tuple[float, float] | None:
    """위저드로 레이어 한 장을 만들어 **아는 값**을 박는다 — 표 식별의 닻.

    소형 캔버스(수십 장)는 장수로 표를 찾으면 스테일 잔재가 수십~수천 건
    겹친다 (2026-08-18 실측: 12장 후보 51건·2장 9,344건). 방금 박은 값은
    살아 있는 표에만 있으므로 (x, y) 역추적이 표를 유일하게 못 박는다
    (`game.inject.find_table_by_sentinel`). 반환: 판독된 (x, y) — 실패면 None.
    """
    try:
        d = Driver()
        d.open_wizard()
        d.select_shape(SHAPE)
        d.confirm_shape_and_color()
        x = d.set_axis("x", SENT_X)
        y = d.set_axis("y", SENT_Y, press_tool=False)
        d.commit()
        time.sleep(0.5)
        log(f"센티널 레이어를 심었다 (x {x} · y {y}) — 표 식별용")
        return (float(x), float(y))
    except DriverError as e:
        log(f"센티널 실패 ({e}) — 장수 검색으로 물러난다")
        return None


def ensure_ready(need: int, shapes: tuple[str, ...] | None = None,
                 stop: Callable[[], bool] | None = None, log=print,
                 reuse: bool = True) -> int | None:
    """주입 직전 **캔버스를 쓸 수 있는 상태로** 만든다. 반환: 최종 장수 (모르면 None).

    규칙 셋이고, 셋 다 사람이 손대지 않아도 된다:

    1. **씨앗 템플릿이 없으면** (빈 캔버스) 만들고 주입한다 — 어휘를 심고
       모자란 만큼 채운다.
    2. **맞는 씨앗 템플릿이면** 그대로 주입한다. 장수가 많아도 좋다 — 남는
       레이어는 주입이 캔버스 밖으로 민다 (`game/inject`). **이 길은 게임을
       아예 안 건드린다** (읽기만 한다).
    3. **틀린 씨앗 템플릿이면** (플랜이 쓰는 도형을 그룹이 못 그리면) **멈추고
       사람에게 알린다.** 그 자리에서 고칠 수가 없다 — 그룹을 풀고(편집 메뉴
       마지막 행) 통째로 지우고 어휘를 다시 심어도 **에셋이 안 들어온다**
       (실측: 그러고 나서 위저드로 놓은 U_23이 그대로 타원으로 그려졌다).
       65차의 "다시 연 그룹엔 위저드가 못 들여온다"가 그대로 맞고, 그룹 해제는
       리스트를 편집 가능하게 만들 뿐 에셋을 되살리지 못한다. 길은 **새 비닐
       그룹을 만드는 것**뿐인데 그건 에디터 밖 메뉴다.

    판정은 화면으로 한다 (`game/canvas.seed_missing`) — 메모리에는 옳은 도형
    id가 들어가 있고 **그려지는 것만** 다르기 때문이다. 장수가 모자라기만 한
    경우(1의 변형)는 씨앗을 안 건드리고 채우기만 한다.

    **묻는 것과 심는 것이 다르다.** 묻는 것은 `shapes`(이 플랜이 쓰는 도형)뿐이다 —
    안 쓰는 도형 때문에 22분짜리 재생성을 돌릴 이유가 없다. 심을 때는 기본이
    **어휘 전체**다: 이 플랜만 보고 심으면 다음 그림이 다른 도형을 쓰는 순간
    3번 길이 또 돌아 22분을 다시 낸다. 한 번 심어 두고 계속 쓰는 것이 그
    합집합의 값이다.

    **`reuse=False`면 이 플랜이 쓰는 도형만 심는다.** 합집합은 **템플릿을 다시
    쓰기 위한** 규약인데, 재사용이 없는 그룹이 있다: 이타샤는 그룹마다 새 비닐
    그룹을 만들어 주입하고 저장하면 끝이라 그 캔버스를 다시 여는 일이 없다
    (`auto.itasha.prepare_group`). 그런 자리에서 합집합은 **쓰지도 않을 도형에
    위저드 한 바퀴씩**을 쓴다 — 씨앗은 장당 3~14초라 소형 그룹에서는 준비
    시간의 95%가 씨앗이다 (실측: 15장짜리 `shapes-top`이 155초인데 채우기는
    7초). 좁혀도 **그리는 것은 그대로다**: 플랜이 참조하는 도형은 전부
    심으므로 그림에 필요한 에셋은 빠짐없이 그룹에 들어간다.
    """
    from ..game.canvas import seed_missing

    want = tuple(w for w in (vocabulary() if shapes is None else shapes) if w)
    # 심는 것은 **이 플랜의 도형 + 어휘 전체**다 (`reuse`면). 어휘만 심으면 어휘
    # 밖 도형을 쓰는 플랜(텍스트 비닐의 글꼴 글리프)이 다른 도형으로 그려진다.
    # 플랜 도형을 앞에 두는 이유는 `plant[:need]`로 잘릴 때 **그 플랜이 쓰는
    # 것부터** 살아야 하기 때문이다 (글자 30장짜리 플랜에 어휘 49종을 먼저
    # 심으면 글리프가 다 잘린다).
    plant = (tuple(dict.fromkeys(want + tuple(w for w in vocabulary() if w)))
             if reuse else want)
    have = canvas_count()
    if have is None:
        log("레이어 수를 못 읽었다 (레이어 리스트 화면이 아닌 듯) — 준비를 건너뛴다")
        return None
    if have > 0:
        miss = seed_missing(list(want), have, log=log)
        if not miss:
            if have >= need:                       # 2) 그대로 간다
                log(f"템플릿 {have}장 · 도형 {len(want)}종 전부 그린다 — 바로 주입한다"
                    + (f" (남는 {have - need}장은 밀어낸다)" if have > need else ""))
                return have
            d = Driver()                           # 1') 모자란 만큼만 채운다
            return fill(d, need, stop=stop)
        raise DriverError(
            f"이 비닐 그룹은 도형 {len(miss)}종을 못 그린다 ({' '.join(miss[:8])}"
            + (" …" if len(miss) > 8 else "")
            + ") — 주입해도 다른 도형으로 그려진다.\n"
            "  이 그룹으로는 못 고친다: 그룹을 풀고 통째로 지우고 다시 심어도\n"
            "  **에셋이 안 들어온다** (실측: 그러고 나서 위저드로 놓은 U_23이\n"
            "  그대로 타원으로 그려졌다).\n"
            "  **새 비닐 그룹을 만들고**(디자인 및 도색 → 비닐 그룹 만들기) 다시\n"
            "  실행할 것 — 빈 캔버스면 씨앗을 심어 템플릿을 지어 준다.")
    else:
        d = Driver()
    if not reuse:
        log(f"씨앗을 이 플랜의 도형 {len(want)}종으로 좁힌다 "
            f"(어휘 전체는 {len(vocabulary())}종 — 다시 안 여는 그룹이라 "
            f"합집합이 필요 없다)")
    seed(d, plant[:need], stop)                    # 1)·3) 새로 짓는다
    return fill(d, need, stop=stop)


def ensure(target: int, shape: str = SHAPE, chunk: int = CHUNK,
           stop: Callable[[], bool] | None = None,
           seeds: tuple[str, ...] | None = None) -> int:
    """캔버스를 정확히 `target`장으로 맞춘다. 반환: 최종 장수.

    이미 맞으면 게임을 안 건드린다. 많으면 `DriverError`다 (위 독스트링).

    **씨앗은 빈 캔버스에서 시작할 때만 심는다.** 이미 레이어가 있으면 그 그룹이
    어떤 에셋을 쥐고 있는지 화면으로는 못 읽으므로, 있는 것을 믿고 채우기만
    한다 (`seed` 문서). `seeds=()`로 끄고, 안 주면 어휘 전체다."""
    d = Driver()
    n = count(d)
    if n == target:
        print(f"템플릿 {n}장 — 이미 맞다")
        return n
    if n > target:
        # 남는 장은 **주입이 밀어낸다** — 여기서 할 일이 없다 (`game/inject`)
        print(f"템플릿 {n}장 — 플랜 {target}장보다 많다. 남는 {n - target}장은 "
              "주입이 캔버스 밖으로 민다 (창 조작 없음)")
        return n
    if n == 0:
        want = vocabulary() if seeds is None else seeds
        want = tuple(w for w in want if w)[:target]
        if want:
            n = seed(d, want, stop)
    return fill(d, target, shape, chunk, stop)
