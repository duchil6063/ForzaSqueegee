r"""텍스트 비닐을 **게임의 텍스트 도구로** 만든다 (비닐 그룹 에디터 안).

FH6에는 글자를 넣는 자기 도구가 있다 (2026-08-17 실측): 도형 선택 화면의 **글꼴
탭**은 글리프 격자가 아니라 **글꼴 11개 목록**이고, 칸을 고르면 `텍스트 입력`
대화상자가 뜬다 ("비닐을 제작하려면 문구를 입력하세요. 라틴 알파벳만 허용됩니다").
문구를 넣으면 게임이 **글자당 레이어 한 장**을 만들어 변형 편집으로 넘긴다
("MIKU" → 4장).

그래서 글자는 **주입하지 않고 이 길로** 만든다:

- 주입은 도형을 못 바꾼다 — 글리프 도형 id 표가 있어야 하고, 다시 연 그룹은
  저장본이 참조한 도형만 그린다(씨앗). 글꼴 960종을 그렇게 심는 것은 비싸다.
- 이 길은 글자 한 줄이 키 몇 번이다. 자리·크기·색은 우리 폐루프가 그대로 잡고
  (`Driver.set_axis`·`set_hsb`), 커닝은 게임이 제 글꼴 규칙으로 맞춘다.

조판 예측(글자 상자가 얼마나 될까)은 여전히 `engine/textvinyl`이 한다 — 같은
글리프를 같은 순서로 쓰므로 구성 설계가 그 값으로 자리를 정할 수 있다.

## 테두리·그림자는 같은 문구를 여러 벌 겹친 것이다

같은 문구를 **원 위 여덟 자리에** 깔면 테두리가 되고(DC 가이드의 눈동자 기법),
같은 크기로 한 번 비껴 깔면 오프셋 그림자가 된다. 세 벌(그림자 → 테두리 → 본색)이면
레퍼런스 이타샤의 그래피티/스티커 타이포 처리다 (RIN SHIBUYA·EVELYNE 실측).
글자 수 × 벌 수 장이 들지만 면 예산에는 티가 안 난다.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from ..engine import textvinyl as tv
from ..game import io as gio
from ..game import ocr
from ..i18n import msg
from .driver import Driver, DriverError

FONT_TAB = 14                  # 도형 선택 화면의 '글꼴' 탭 (cell_map meta.tabs 인덱스)
DIALOG_WAIT = 8.0              # '텍스트 입력' 대화상자를 기다리는 시간
# 테두리 오프셋은 조판 쪽(`engine.textvinyl`)이 쥔다 — 미리보기도 같은 값으로
# 그려야 테두리를 넣을지 뺄지를 미리보기로 판단할 수 있다.
OUTLINE_SHIFT = tv.OUTLINE_SHIFT


def _table_path() -> Path:
    return Path(__file__).resolve().parents[2] / "catalog" / "font_cells.json"


@cache
def font_cells() -> dict[str, tuple[int, int]]:
    """글꼴 이름 → 그리드 칸 (row, col). `catalog/font_cells.json` 실측표."""
    p = _table_path()
    if not p.exists():
        return {}
    raw = json.loads(p.read_text(encoding="utf-8"))
    out: dict[str, tuple[int, int]] = {}
    for rc, name in (raw.get("cells") or {}).items():
        r, c = (int(v) for v in rc.split(","))
        out[name] = (r, c)
    return out


def font_cell(font: str) -> tuple[int, int]:
    cells = font_cells()
    if font in cells:
        return cells[font]
    raise DriverError(
        msg("게임 글꼴 목록에 '{font}'이 없다 (아는 것: {fonts}).\n"
            "  표는 catalog/font_cells.json이고 근거는 그 안의 note다",
            font=font, fonts=", ".join(sorted(cells))))


@dataclass
class TextJob:
    """게임에 넣을 글자 한 줄. 좌표·크기는 **캔버스 유닛**이다."""

    text: str
    font: str = "arial"
    x: float = 0.0
    y: float = 0.0
    scale: float = 1.0                       # 게임 스케일 값 (글자 상자 배율)
    rot: float = 0.0
    hsb: tuple[float, float, float] | None = None
    outline: tuple[float, float, float] | None = None
    # 오프셋 그림자 — 본색과 같은 크기의 사본을 `shadow_shift`(유닛)만큼 비껴
    # **맨 아래에** 깐다. 테두리와 겹치면 그래피티/스티커 타이포가 된다.
    shadow: tuple[float, float, float] | None = None
    shadow_shift: tuple[float, float] = (0.0, 0.0)
    # 중심 기준 배치 (있으면 x·y를 여기서 계산한다). 게임은 **첫 글리프의 설계
    # 원점**을 붙잡으므로 중심에 맞추려면 오프셋을 빼야 하고, 그 오프셋은 크기에
    # 비례해 바뀐다 — 테두리 사본이 더 크므로 사본마다 다시 계산해야 가운데가 맞는다.
    center: tuple[float, float] | None = None
    height: float | None = None               # 대문자 높이 (캔버스 유닛)

    @property
    def n_layers(self) -> int:
        """게임이 만들 레이어 수 — **공백 아닌 글자 수** × 벌 수."""
        n = sum(1 for c in self.text if not c.isspace())
        passes = (1 + (tv.OUTLINE_PASSES if self.outline is not None else 0)
                  + (self.shadow is not None))
        return n * passes


def goto_font_tab(d: Driver, tab: int = FONT_TAB, n_tabs: int = 16,
                  back: int = 1) -> None:
    """도형 선택 화면에서 글꼴 탭으로 — **오른끝 클램프에서 `back`걸음** 되돌아온다.

    `Driver.ensure_tab`은 셀 썸네일을 `cell_map`과 대 보는데 글꼴 탭에는 그 표가
    없다 (글리프가 아니라 글꼴 목록이라 만들 수도 없다). 대신 도착 확인을 **다음
    단계로** 미룬다: 칸을 고르면 `텍스트 입력` 대화상자가 떠야 하고, 안 뜨면
    호출자가 `back`을 바꿔 다시 온다.
    """
    # 왼끝에서 세어 가면 **차종마다 탭 수가 달라** 어긋난다 (2026-08-18 실측:
    # 실비아 에디터는 인테그라보다 탭이 하나 적어 14걸음이 글꼴을 지나 '내 FH5
    # 비닐' 빈 그리드에 내렸다). 오른끝 기준도 고정은 못 쓴다 — **오른끝 탭
    # 구성은 세션에 따라 변한다** (2026-08-19 실측: 게임 재시작 후 '오른끝−1'이
    # 색상/도형 탭에 내려 색상 패널이 열렸다 — 서버 연결 여부가 탭을 넣고 뺀다).
    for _ in range(n_tabs + 2):
        gio.press("pgdn")
        time.sleep(0.16)
    time.sleep(0.4)
    for _ in range(back):
        gio.press("pgup")
        time.sleep(0.7)


def _discard_to_list(host, times: int = 3, d: Driver | None = None) -> None:
    """잘못 연 것(색상 패널·도형 그리드)을 Esc로 버리고 레이어 리스트로 돌아온다.

    Esc를 세지 않고 던지면 리스트를 지나 **나가기 대화상자**까지 흐른다 — host
    (차체 에디터)가 있으면 매 걸음 화면을 확인하고, 리스트면 멈춘다.

    비닐 그룹 캔버스에는 host가 없다. 거기서 세지 않고 던졌더니 그룹을 통째로
    빠져나가 '저장하시겠습니까' 대화상자에 섰고, 다음 걸음이 '+' 셀을 못 찾아
    실행이 죽었다 (2026-08-20 실측 — 글꼴 탭 탐색이 back을 세 번 바꾸는 동안
    Esc가 여섯 번 나갔다). host가 없으면 `d`로 **레이어 리스트인지**를 본다:
    리스트에는 선택 링이 있고 대화상자에는 없다.
    """
    for _ in range(times + 2):
        if host is not None and host.screen() == "list":
            return
        if host is None and d is not None and d.list_selection(d.cap()) is not None:
            return
        gio.press("esc")
        time.sleep(0.9)


def _dialog_open(d: Driver) -> bool:
    """'텍스트 입력' 대화상자인가 — 가운데 위쪽 라임 제목 밴드 + 흰 입력칸."""
    img = d.cap()
    h, w = img.shape[:2]
    band = img[int(0.38 * h):int(0.47 * h), int(0.33 * w):int(0.67 * w)]
    r = band[:, :, 0].astype(int)
    g = band[:, :, 1].astype(int)
    b = band[:, :, 2].astype(int)
    lime = ((r > 140) & (r < 235) & (g > 200) & (b < 110)).mean()
    box = img[int(0.52 * h):int(0.58 * h), int(0.35 * w):int(0.65 * w)]
    white = (box.min(axis=2) > 200).mean()
    return bool(lime > 0.5 and white > 0.5)


def _typed_ink(d: Driver) -> float:
    """입력칸(흰 박스)에 든 글자의 몫 — 문구가 들어갔나 보는 자."""
    img = d.cap()
    h, w = img.shape[:2]
    box = img[int(0.525 * h):int(0.575 * h), int(0.35 * w):int(0.65 * w)]
    return float((box.max(axis=2) < 120).mean())


def _type_and_confirm(d: Driver, text: str) -> None:
    """문구를 넣고 **들어간 것을 확인한 뒤** 확정한다.

    대화상자가 뜬 직후에는 입력칸이 아직 키를 안 받는다 — 그대로 치면 통째로
    흘린다 (2026-08-17 실측: 빈 칸으로 Enter가 가서 변형 편집이 안 열렸다).
    그래서 잠깐 기다리고, 칸에 글자가 보이는지 보고, 안 보이면 다시 친다.
    """
    for attempt in range(3):
        time.sleep(0.45)
        gio.type_text(text)
        time.sleep(0.4)
        if _typed_ink(d) > 0.01:
            break
        gio.press_batch("backspace", len(text) + 4)
        time.sleep(0.2)
    else:
        raise DriverError(msg("텍스트 입력칸에 '{text}'가 안 들어갔다", text=text))
    d._step("enter", d.in_transform_edit, msg("텍스트 → 변형 편집"),
            tries=3, wait=3.0)


def _place_xy(t: TextJob, scale: float) -> tuple[float, float]:
    """이 벌(스케일 `scale`)을 앉힐 게임 좌표. 중심 기준이면 오프셋을 뺀다.

    회전이 있으면 오프셋도 같이 돈다 — 게임이 붙잡는 기준점(첫 글리프 원점)을
    축으로 글자 블록이 돌므로, 잉크 중심으로 가는 벡터를 같은 각으로 돌려 뺀다.
    """
    if t.center is None or t.height is None:
        return t.x, t.y
    import math

    from ..engine import textvinyl as tv

    h = t.height * (scale / max(1e-6, t.scale))
    m = tv.text_metrics(t.text, font=t.font, height=h)
    cx, cy = m["cx"], m["cy"]
    if t.rot:
        r = math.radians(t.rot)
        c, s = math.cos(r), math.sin(r)
        cx, cy = cx * c - cy * s, cx * s + cy * c
    return round(t.center[0] - cx, 1), round(t.center[1] - cy, 1)


def _one_pass(d: Driver, t: TextJob, scale: float,
              hsb: tuple[float, float, float] | None,
              host=None, shift: tuple[float, float] = (0.0, 0.0)
              ) -> dict[str, float]:
    """글자 한 벌을 만들어 앉히고 확정한다. 반환: 최종 판독값.

    `host`를 주면 **차체 에디터**에서 돈다 (`auto.bodyedit.BodyEditor`) — 글꼴 탭은
    두 에디터에 다 있고(실측), 다른 것은 위저드를 열고 확정하는 손뿐이다.
    `shift`는 이 벌만 비껴 앉히는 오프셋(유닛) — 그림자 벌이 쓴다.
    """
    row, col = font_cell(t.font)
    # 글꼴 탭의 자리는 세션마다 다르다 (오른끝 탭 구성이 서버 상태로 변한다 —
    # 2026-08-19 실측). '텍스트 입력' 대화상자 도착을 판정자로 쓰고, 안 뜨면
    # 열린 것(색상 패널 등)을 버리고 오른끝에서 되돌아오는 걸음 수를 바꿔 본다.
    found = False
    # 오른끝 탭 구성이 세션마다 변해 `back`을 훑는다. **0을 먼저** 본다 —
    # 2026-08-20 세션의 글꼴 탭은 오른끝 그 자체였다 ('자연' 다음 '글꼴').
    for back in (0, 1, 2, 3):
        (host or d).open_wizard()
        goto_font_tab(d, back=back)
        try:
            d.select_cell(row, col)
        except DriverError:
            _discard_to_list(host, times=2, d=d)
            continue
        gio.press("enter")
        t_end = time.time() + DIALOG_WAIT
        while time.time() <= t_end:
            if _dialog_open(d):
                found = True
                break
            time.sleep(0.3)
        if found:
            if back != 1:
                print(msg("    글꼴 탭 = 오른끝−{back} (이 세션)", back=back))
            break
        _discard_to_list(host, times=3, d=d)
    if not found:
        raise DriverError(msg("'텍스트 입력' 대화상자가 안 떴다 — 글꼴 탭 탐색 실패 (오른끝−1..3·0)"))
    _type_and_confirm(d, t.text)
    got: dict[str, float] = {}
    if hsb is not None:
        d._step("x", d.in_hsb_edit, msg("글자 색 HSB 진입"))
        d.set_hsb(*hsb)
        d._step("enter", d.in_transform_edit, msg("색 → 변형 편집 복귀"))
    got["rot"] = d.set_axis("rot", t.rot)
    # 글자 덩어리의 스케일은 **값 칸이 하나**다 (균등) — 불러온 비닐 그룹과 같다.
    # 두 칸을 요구하면 두 번째에서 판독 실패로 죽는다 (2026-08-17 실측).
    got["scale"] = d.set_axis("sx", scale)
    px, py = _place_xy(t, scale)
    # 이동 축은 면마다 클램프가 있다 (실측: top x ±554) — 못 닿는 목표는 멈춘
    # 자리에 두고 간다. 글자가 몇 유닛 비껴 앉는 것이 실행이 죽는 것보다 낫다.
    got["x"] = _soft_axis(d, "x", round(px + shift[0], 1))
    got["y"] = _soft_axis(d, "y", round(py + shift[1], 1), press_tool=False)
    (host or d).commit()
    return got


def _soft_axis(d: Driver, axis: str, target: float,
               press_tool: bool = True) -> float:
    try:
        return d.set_axis(axis, target, press_tool=press_tool)
    except DriverError:
        pass
    # 홀드 폐루프가 어긋난 것일 수 있다 — 정착 후 느린 화살표 전용으로 한 번 더
    # (itasha._soft_xy와 같은 근거, 2026-08-19 챌린저 front 실측)
    time.sleep(2.0)
    try:
        return d.set_axis(axis, target, press_tool=True, gentle=True)
    except DriverError:
        v = ocr.read_stable(d.hwnd, "y" if axis == "y" else "x", tries=20)
        return float(v) if v is not None else target


def add_text(d: Driver, t: TextJob, log=print, host=None,
             count=None) -> dict[str, float]:
    """글자를 캔버스에 넣는다 (테두리가 있으면 뒤에 한 벌 더). 반환: 본색 판독값.

    **뒤부터 넣는다** — 나중에 넣은 것이 위에 그려지므로 테두리를 먼저 깔아야 한다.
    """
    # **레이어 카운터가 보이는 화면**에서만 시작한다. `list_selection`은 변형 편집
    # 화면에서도 값을 내는 일이 있어(2026-08-17) 그것만 믿으면 위저드를 엉뚱한
    # 화면에서 열려다 죽는다. 카운터는 리스트에만 있다.
    read_n = count or (lambda: ocr.read_layer_count_stable(d.hwnd))
    n0 = read_n()
    for _ in range(5):
        if n0 is not None:
            break
        gio.press("esc")
        time.sleep(1.0)
        n0 = read_n()
    if n0 is None:
        raise DriverError(msg("레이어 리스트 화면이 아니다 (레이어 수를 못 읽었다)"))
    if t.shadow is not None:
        log(msg("  글자 '{text}' 그림자 ({font})", text=t.text, font=t.font))
        _one_pass(d, t, t.scale, t.shadow, host=host, shift=t.shadow_shift)
    if t.outline is not None:
        # 140.8 = 스케일 1.0의 대문자 높이(유닛) — `sans` 'H' 실측 2026-08-17,
        # 게임 텍스트 스케일 1.0 = 비닐 레이어 스케일 1.0 (카탈로그 예측 139.5)
        h = t.height if t.height is not None else t.scale * 140.8
        offs = tv.outline_offsets(OUTLINE_SHIFT * h)
        for i, (ox, oy) in enumerate(offs):
            log(msg("  글자 '{text}' 테두리 {i}/{n} ({font})",
                    text=t.text, i=i + 1, n=len(offs), font=t.font))
            _one_pass(d, t, t.scale, t.outline, host=host, shift=(ox, oy))
    log(msg("  글자 '{text}' 본색 ({font})", text=t.text, font=t.font))
    got = _one_pass(d, t, t.scale, t.hsb, host=host)
    n1 = read_n()
    if n0 is not None and n1 is not None:
        log(msg("    레이어 {n0} → {n1} (기대 +{expect})",
                n0=n0, n1=n1, expect=t.n_layers))
        if n1 - n0 != t.n_layers:
            raise DriverError(
                msg("글자 '{text}': 레이어가 {delta}장 늘었다 (기대 {expect}장) — "
                    "게임이 글자를 다르게 나눴다",
                    text=t.text, delta=n1 - n0, expect=t.n_layers))
    return got
