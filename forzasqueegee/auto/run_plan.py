"""plan.json 순차 자동 그리기 루프 — 같은 도형·색 연속 구간은 Y 스탬프로 고속 배치.

그룹 실행 (2026-08-02 실측 검증):
- 연속 레이어가 (도형, HSB) 동일하면 위저드·색 지정을 1회만 하고, 레이어마다
  변형 설정 → Y(스탬프 = 사본 즉시 커밋, 편집 유지) → 마지막 레이어만 Enter 커밋.
- 같은 색 그룹 내부의 z순서는 시각적으로 무의미(동일 색)라 스탬프 순서 제약 없음.
- 변형은 직전 레이어 값에서 이어짐 → 목표가 같은 축은 스킵.

진행 저장: <plan 폴더>/auto_progress.json — 스탬프/커밋마다 갱신, 재실행 시 이어서.
plan 파일명이 진행 파일과 다르면 무시하고 0부터 (plan.json ↔ plan_sorted.json 혼동 방지).
중단: Ctrl+C 또는 <plan 폴더>/STOP 파일. 실패 복구: 리스트 복귀 후 남은 구간 재시도.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from ..engine.model import LayerPlan
from ..game import io as gio
from .driver import Driver, DriverError, TransformTarget, alpha8
from .fav import FavStack

# 새 레이어의 변형 기본값 — 목표가 같으면 축 스킵 (위저드 새 도형은 항상 이 값으로 생성)
# 투명도 100도 실측 확인: Y 스탬프 뒤에는 유지되지만 **새 위저드 레이어는 100으로 돌아온다**
AXIS_DEFAULTS = {"x": 0.0, "y": 0.0, "sx": 1.0, "sy": 1.0, "rot": 0.0, "alpha": 100.0}
# 색상 기본값(흰색): 목표가 같으면 HSB 미세 조정 스킵
DEFAULT_HSB = (0.0, 0.0, 1.0)


class StopRequested(RuntimeError):
    pass


def _progress_path(plan_path: Path) -> Path:
    return plan_path.parent / "auto_progress.json"


def _load_done(plan_path: Path) -> int:
    p = _progress_path(plan_path)
    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
        if data.get("plan") == plan_path.name:  # 다른 플랜의 진행이면 무시
            return int(data.get("done", 0))
    return 0


def _save_done(plan_path: Path, done: int, total: int) -> None:
    _progress_path(plan_path).write_text(
        json.dumps({"plan": plan_path.name, "done": done, "total": total}),
        encoding="utf-8")


def _check_stop(plan_path: Path) -> None:
    stop = plan_path.parent / "STOP"
    if stop.exists():
        stop.unlink()  # 다음 실행을 위해 제거
        raise StopRequested("STOP 파일 감지 — 레이어 경계에서 중단")


def _hsb_key(layer) -> tuple[float, float, float]:
    # 게임 색 입력 UI(HSB 0.01)에 넣을 값 — 인게임 적용 시점에만 변환한다
    return layer.hsb()


def _hsb_or_none(layer) -> tuple[float, float, float] | None:
    hsb = _hsb_key(layer)
    return None if hsb == DEFAULT_HSB else hsb


def _recover_to_list(d: Driver) -> None:
    """오류 후 화면을 레이어 리스트로 복귀 (위저드 어느 단계에 있든)."""
    if d.in_transform_edit():
        d.discard()
        return
    if d.in_hsb_edit():
        for _ in range(3):  # HSB → 색상 선택 → 도형 선택 → 리스트
            gio.press("esc")
            time.sleep(0.6)
        return
    for _ in range(4):
        img = d.cap()
        if d._edit_menu_open(img) or d._menu_open() or d.find_highlight(img) is not None:
            gio.press("esc")
            time.sleep(0.6)
            continue
        if d.list_selection(img) is not None:  # 리스트 복귀 확인
            return
        gio.press("esc")  # 미지 화면(색상 선택 등) — 한 단계 뒤로
        time.sleep(0.6)


def _set_transforms(d: Driver, layer, prev: dict[str, float]) -> dict[str, float]:
    """직전 상태 prev에서 layer 목표로 변형 설정. 반환: 새 상태.

    **기울기(skew) 축은 없다** — 쓰지 않기로 한 축이라 플랜도 내지 않는다.
    그래도 값이 들어오면 조용히 무시하는 대신 멈춘다: 무시하면 플랜 렌더와
    인게임 결과가 소리 없이 갈린다.
    """
    if abs(getattr(layer, "skew", 0.0)) > 1e-9:
        raise DriverError(
            f"기울기가 든 레이어는 못 그린다 (skew={layer.skew}) — driver에 축이 없다")
    t = TransformTarget(x=layer.x, y=layer.y, sx=layer.sx, sy=layer.sy,
                        rot=layer.rot % 360.0, alpha=layer.alpha)
    cur = dict(prev)
    if abs(t.x - prev["x"]) > 1e-9 or abs(t.y - prev["y"]) > 1e-9:
        got = d.set_move_xy(t.x, t.y, prev=cur)
        cur.update(got)
    d_sx = abs(t.sx - prev["sx"]) > 1e-9
    d_sy = abs(t.sy - prev["sy"]) > 1e-9
    if d_sx and d_sy:  # 같은 도구 — 키 1회, 판독 1회로 두 축을 함께
        cur.update(d.set_scale(t.sx, t.sy, prev=cur))
    elif d_sx or d_sy:
        axis = "sx" if d_sx else "sy"
        cur[axis] = d.set_axis(axis, getattr(t, axis), prev=cur)
    if abs(t.rot - prev["rot"]) > 1e-9:
        cur["rot"] = d.set_axis("rot", t.rot, prev=cur)
    # 투명도는 표시값이 아니라 8비트로 견준다 — 표시 0.01 격자에 없는 값도 있어서
    # (알파 128 = 50.196…) 표시 비교로는 이미 맞은 축을 다시 밀 수 있다
    if alpha8(t.alpha) != alpha8(prev["alpha"]):
        cur["alpha"] = d.set_alpha(t.alpha, prev=cur)
    return cur


def draw_group(d: Driver, plan_path: Path, layers: list, start: int, total: int,
               fav: FavStack | None = None) -> int:
    """같은 (도형, HSB) 연속 그룹 실행. 반환: 커밋한 레이어 수.

    layers[start:]에서 그룹을 잘라 위저드 1회 + 레이어별 변형→Y 스탬프,
    마지막 레이어는 Enter 커밋. 스탬프마다 진행 저장.
    색상은 즐겨찾기 스택 우선(인덱스 점프), 첫 사용 색은 HSB 설정 + Y 등록.
    """
    first = layers[start]
    end = start + 1
    while end < len(layers) and layers[end].shape == first.shape \
            and layers[end].mask == first.mask \
            and _hsb_key(layers[end]) == _hsb_key(first):
        end += 1
    group = layers[start:end]

    d.shape_loc(first.shape)  # 매핑 없는 도형은 위저드 진입 전에 실패
    if first.mask:
        # 마스크 위저드: 색상 단계 없음. Y 스탬프는 마스크 변형에서도 동작 (11차 실측)
        d.open_mask_wizard(first.shape)
    else:
        d.open_wizard()
        d.select_shape(first.shape)
        d.confirm_shape_and_color(hsb=_hsb_or_none(first), fav=fav)
    state = dict(AXIS_DEFAULTS)
    committed = 0
    for k, layer in enumerate(group):
        t0 = time.time()
        state = _set_transforms(d, layer, state)
        if k < len(group) - 1:
            gio.press("y", hold_s=0.09)  # 스탬프 = 사본 커밋, 편집 유지
            time.sleep(0.35)
        else:
            d.commit()
        committed += 1
        _save_done(plan_path, start + committed, total)
        tag = "스탬프" if k < len(group) - 1 else "커밋"
        print(f"  [{start + committed}/{total}] {layer.shape} ({layer.label}) "
              f"{tag} {time.time() - t0:.1f}s")
    return committed


def run(plan_path: str | Path, start: int | None = None,
        limit: int | None = None) -> int:
    """plan.json 실행. 반환: 이번 실행에서 커밋한 레이어 수.

    start 미지정 시 auto_progress.json에서 이어서. limit = 이번 실행 최대 레이어 수.
    """
    plan_path = Path(plan_path)
    plan = LayerPlan.load(plan_path)
    layers = [l.quantized() for l in plan.layers]
    total = len(layers)
    done = _load_done(plan_path) if start is None else start
    if done >= total:
        print(f"이미 완료됨 ({done}/{total})")
        return 0

    d = Driver()
    fav = FavStack()
    end = total if limit is None else min(total, done + limit)
    committed = 0
    print(f"run_plan: {plan_path} — {done}/{total}부터 {end}까지 (그룹 스탬프 모드)")
    while done < end:
        _check_stop(plan_path)
        bounded = layers[:end]  # limit는 그룹 경계 기준으로 잘라 초과 방지
        try:
            n = draw_group(d, plan_path, bounded, done, total, fav)
        except DriverError as e:
            print(f"  [{done}] 실패({e}) — 복구 후 재시도")
            _recover_to_list(d)
            time.sleep(1.0)
            done = _load_done(plan_path)  # 그룹 중간 스탬프까지는 커밋됨
            n = draw_group(d, plan_path, bounded, done, total, fav)  # 재시도 실패는 전파
        done += n
        committed += n
    print(f"완료: 이번 실행 {committed}개 커밋, 진행 {done}/{total}")
    return committed
