"""§18 커버리지 불변 — **실루엣 안에 안 칠한 표본은 없다.**

구멍 자(`holes.count_hole_clusters`)는 **가격의 자**다: 4px보다 작은 군집도,
λ에 못 미치는 군집도 안 센다. "메울 값이 있나"를 묻는 자리에서는 그것이
맞다 — 메움 한 장도 λ를 물기 때문이다. 그러나 사용자가 인게임에서
보는 것은 값이 아니라 **차 도색이 비치는 자국**이고, 그 자국에는 크기 하한이
없다. 그래서 자를 둘로 가른다:

- **가격의 자** (`holes`) — 무엇을 살 값이 있나. 배치 도중에 쓴다.
- **불변의 자** (여기) — 안 칠한 표본이 하나라도 있나. 끝에서 쓴다.

불변의 자는 **λ와 거래하지 않는다.** 크기·중요도·값과 무관하게 미커버는
불가(不可)다 — 벌점이 아니라 자격이다.

## 표본은 왜 픽셀보다 잘게 세나

게임은 벡터를 그린다. 도안의 두 도형이 0.4px 어긋나 있으면 1px 격자 래스터는
"둘 다 그 픽셀을 덮었다"고 말하지만 인게임에서는 그 사이로 도색이 비친다.
그래서 이 모듈은 **2배 슈퍼샘플**(`_SS`)에서 센다 — 프루닝의 소유자
래스터(`pruneplan._SS`)·미리보기 렌더가 이미 쓰는 그 배율이고, 같은 배율에서
보이지 않는 틈은 게임 화면에서도 도형 경계의 안티에일리어싱 아래다.

## 최외곽 테는 뺀다

실루엣의 **가장 바깥 한 겹**은 세지 않는다 (`holes._sil_rim`과 같은 자).
도형 가장자리는 게임 이동 스텝 격자에만 설 수 있어 그 테 안쪽으로는 어떤
도형도 경계에 딱 맞출 수 없고, 물러선 자리를 구멍으로 세면 그것을 메우는
도형이 다시 실루엣 밖으로 넘쳐 순환이 된다. 뺀 자리는 "실루엣 **내부**"의
정의 그 자체다 — 도안의 결함이 아니라 격자의 폭이다.

## 파괴적 연산의 자격 판정

컷·되팔기·병합은 커버리지를 **줄일 수 있는** 유일한 손이다. 그 손에는
`unique_layers`(혼자 덮고 있는 레이어)를 보호로 주고, 실행 뒤에는
`repair_cut`이 **동시 제거의 연쇄**를 되짚어 필요한 만큼 되살린다 (한 장씩
보면 안전한 두 장이 같은 표본을 나눠 덮고 있을 수 있다).
"""

from __future__ import annotations

import os

import cv2
import numpy as np

from ...i18n import msg
from ..catalog import Catalog
from ..celart import CelArt
from ..model import LayerPlan
from .geometry import _poly_px, poly_bbox, poly_mask
from .holes import _sil_rim
from .vocabulary import _FILL_SHAPE

# 슈퍼샘플 배율 — 프루닝 소유자 래스터(`pruneplan._SS`)·미리보기와 같은 자
_SS = 2


def sil_core(cel: CelArt, upp: float) -> np.ndarray:
    """**실루엣 내부** (최외곽 테 제외, 1x bool) — 불변이 걸리는 자리."""
    return (cel.labels >= 0) & ~_sil_rim(cel.labels, upp)


def _upsample(m: np.ndarray, ss: int) -> np.ndarray:
    """1x bool → ss 격자 bool (정확한 반복 — 보간 없음)."""
    return np.repeat(np.repeat(m, ss, axis=0), ss, axis=1)


def _layer_mask(cat: Catalog, lay, upp: float, w: int, h: int, ss: int):
    """레이어 → (ss 격자 마스크, x0, y0) 또는 None. 짝홀 규칙은 렌더와 같다.

    꼭짓점을 격자에 반올림하지 **않는다** (`geometry.poly_mask` 문서) — 그
    반올림이 도안에 없는 틈을 만들고, 이 모듈은 하필 그 틈을 세는 자리다.
    """
    polys = _poly_px(cat, lay, upp, w, h)
    if not polys:
        return None
    box = poly_bbox(polys, w, h, ss, pad=1)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    m = poly_mask(polys, (y1 - y0, x1 - x0), x0 / ss, y0 / ss, ss)
    return m, x0, y0


def counts(layers: list, cat: Catalog, upp: float, w: int, h: int,
           ss: int = _SS, dtype=np.uint8) -> np.ndarray:
    """표본마다 **몇 장이 덮고 있나** (기본 uint8, 255에서 포화).

    뺄셈 마스크는 그 자리의 셈을 0으로 되돌린다 (프루닝 소유자 모델과 같은 뜻).

    `dtype`은 **셈을 되돌릴 때**만 넓힌다 (`HardCoverage`) — 포화한 셈은 빼면
    틀리기 때문이다. 읽기만 하는 자리는 0·1·2+만 가르면 되므로 uint8 그대로다.
    """
    cnt = np.zeros((h * ss, w * ss), dtype)
    cap = int(np.iinfo(dtype).max)
    for lay in layers:
        got = _layer_mask(cat, lay, upp, w, h, ss)
        if got is None:
            continue
        m, x0, y0 = got
        box = cnt[y0:y0 + m.shape[0], x0:x0 + m.shape[1]]
        mb = m
        if lay.mask:
            box[mb] = 0
        else:
            box[mb & (box < cap)] += 1     # 상한에서 포화 (0·1·2+만 가르면 된다)
    return cnt


def hard_holes(plan: LayerPlan, cel: CelArt, cat: Catalog, ss: int = _SS,
               cnt: np.ndarray | None = None) -> np.ndarray:
    """실루엣 내부에서 **아무도 안 덮은 표본** (ss 격자 bool)."""
    w, h = cel.size
    if cnt is None:
        cnt = counts(plan.layers, cat, plan.units_per_px, w, h, ss)
    return _upsample(sil_core(cel, plan.units_per_px), ss) & (cnt == 0)


def measure(plan: LayerPlan, cel: CelArt, cat: Catalog, ss: int = _SS) -> dict:
    """report용 경성 지표 — 표본 수·픽셀 환산·군집 수. 게이트가 이것을 본다."""
    w, h = cel.size
    hard = hard_holes(plan, cel, cat, ss)
    n = int(hard.sum())
    if not n:
        return {"hard_hole_samples": 0, "hard_hole_px": 0.0,
                "hard_hole_clusters": 0}
    ncc, _, _, _ = cv2.connectedComponentsWithStats(hard.astype(np.uint8),
                                                    connectivity=8)
    return {"hard_hole_samples": n,
            "hard_hole_px": round(n / (ss * ss), 2),
            "hard_hole_clusters": int(ncc - 1)}


def need_px(hard: np.ndarray, ss: int = _SS) -> np.ndarray:
    """ss 격자 구멍 → **그 표본을 품은 1x 픽셀** (메움·성장이 겨눌 자리)."""
    h2, w2 = hard.shape
    return hard.reshape(h2 // ss, ss, w2 // ss, ss).any(axis=(1, 3))


class HardCoverage:
    """§18 진실을 **도중에도** 묻는 증분 자 — 이 교체가 새 미커버를 여나.

    봉인은 파이프라인 맨 끝에서 이 모듈의 자로 판정한다: 2배 표본 격자에서,
    꼭짓점을 **반올림하지 않고**, 실루엣 내부(테 제외)에 아무도 안 덮은 표본이
    있나. 그런데 그 앞 단들은 각자의 래스터로 "구멍 안 열었다"를 묻는다 —
    전역 미세 조정(`finetune._win_mask`)은 꼭짓점을 1x 격자로 반올림한 소유자
    래스터를 본다. 그 반올림이 도안에 없는 틈을 만들고 있는 틈을 지우므로
    (`geometry.poly_mask` 문서), 두 자는 **같은 이동에 다른 답**을 낸다:
    미세 조정이 통과시킨 이동이 봉인의 자에서는 새 표본을 여는 일이 실제로
    있다 (실측: 표준 판당 s4_ft가 130~260 표본을 새로 연다).

    그래서 커버리지 자격을 묻는 자리는 전부 **여기 하나**를 쓴다. 이 클래스는
    새 2배 마스크 구현이 아니다 — `counts`·`_layer_mask`·`sil_core` 그대로이고,
    더한 것은 "스택 전체를 매번 다시 세지 않는다"는 증분뿐이다.

    판정은 **스택 수준**이다. 비켜난 표본이라도 다른 레이어가 이미 덮고 있으면
    (`cnt > 1`) 퇴행이 아니다 — 자격은 "아무도 안 덮나"이지 "이 장이 덮나"가
    아니다.
    """

    def __init__(self, plan: LayerPlan, cel: CelArt, cat: Catalog,
                 ss: int = _SS) -> None:
        self.cat = cat
        self.ss = ss
        self.upp = plan.units_per_px
        self.w, self.h = cel.size
        self.core = _upsample(sil_core(cel, self.upp), ss)
        # 셈을 **되돌려야** 하므로 포화하지 않는 폭으로 센다
        self.cnt = counts(plan.layers, cat, self.upp, self.w, self.h, ss,
                          dtype=np.uint16)
        self._memo: dict = {}

    def _mask(self, lay):
        """레이어 → (마스크, x0, y0). 받은 이동은 `opens`·`swap`이 잇달아 같은
        두 자세를 묻는다 — 그 한 왕복만 기억한다 (기하가 키라 안전하다)."""
        k = (lay.shape, lay.x, lay.y, lay.sx, lay.sy, lay.rot, lay.skew)
        if k not in self._memo:
            if len(self._memo) > 64:
                self._memo.clear()
            self._memo[k] = _layer_mask(self.cat, lay, self.upp,
                                        self.w, self.h, self.ss)
        return self._memo[k]

    def opens(self, old, new) -> bool:
        """`old`를 `new`로 갈면 **아무도 안 덮는 표본**이 새로 생기나."""
        mo = self._mask(old)
        mn = self._mask(new)
        if mo is None:
            return False                   # 덮던 것이 없으면 열 것도 없다
        m_o, ox, oy = mo
        # 비켜나는 표본 = 옛 마스크 ∖ 새 마스크. 새 마스크를 옛 창에 겹친다
        vac = m_o.copy()
        if mn is not None:
            m_n, nx, ny = mn
            x0 = max(ox, nx); y0 = max(oy, ny)
            x1 = min(ox + m_o.shape[1], nx + m_n.shape[1])
            y1 = min(oy + m_o.shape[0], ny + m_n.shape[0])
            if x0 < x1 and y0 < y1:
                vac[y0 - oy:y1 - oy, x0 - ox:x1 - ox] &= ~m_n[
                    y0 - ny:y1 - ny, x0 - nx:x1 - nx]
        if not vac.any():
            return False
        sub = (slice(oy, oy + m_o.shape[0]), slice(ox, ox + m_o.shape[1]))
        return bool((vac & self.core[sub] & (self.cnt[sub] == 1)).any())

    def swap(self, old, new) -> None:
        """받아들인 교체를 셈에 반영한다 (증분)."""
        got = self._mask(old)
        if got is not None:
            m, x0, y0 = got
            self.cnt[y0:y0 + m.shape[0], x0:x0 + m.shape[1]][m] -= 1
        got = self._mask(new)
        if got is not None:
            m, x0, y0 = got
            self.cnt[y0:y0 + m.shape[0], x0:x0 + m.shape[1]][m] += 1


def unique_layers(plan: LayerPlan, cel: CelArt, cat: Catalog, ss: int = _SS,
                  cnt: np.ndarray | None = None) -> set[int]:
    """**혼자 덮고 있는** 레이어 index 집합 — 파괴적 연산의 제거 금지 목록.

    표본의 커버 수가 1이면 그 표본을 덮는 레이어는 하나뿐이고, 그 장을 빼면
    그 자리는 그대로 구멍이 된다. 값·크기와 무관한 자격 판정이다.
    """
    w, h = cel.size
    if cnt is None:
        cnt = counts(plan.layers, cat, plan.units_per_px, w, h, ss)
    uniq = _upsample(sil_core(cel, plan.units_per_px), ss) & (cnt == 1)
    if not uniq.any():
        return set()
    out: set[int] = set()
    for i, lay in enumerate(plan.layers):
        if lay.mask:
            continue
        got = _layer_mask(cat, lay, plan.units_per_px, w, h, ss)
        if got is None:
            continue
        m, x0, y0 = got
        if (uniq[y0:y0 + m.shape[0], x0:x0 + m.shape[1]] & (m > 0)).any():
            out.add(i)
    return out


def repair_cut(before: list, kept: list, cel: CelArt, cat: Catalog,
               upp: float, ss: int = _SS) -> tuple[list, int]:
    """컷 결과가 연 구멍을 **되살려 닫는다** — (되살린 목록, 되살린 장수).

    `unique_layers` 보호는 **한 장씩 뺐을 때**의 셈이다. 같은 표본을 둘이
    나눠 덮고 있으면 둘 다 보호를 안 받고, 한 바퀴에 같이 잘리면 그 자리가
    열린다 (실측 C20-09에서 사후 가격이 그렇게 배경을 열었다).

    되살리는 순서는 **가장 많은 구멍을 닫는 장부터**다 — 결정적이고, 되살리는
    장수가 최소에 가깝다. 남은 구멍이 없어질 때까지 돈다.
    """
    w, h = cel.size
    kept_id = {id(l) for l in kept}
    cut = [l for l in before if id(l) not in kept_id]
    if not cut:
        return kept, 0
    core = _upsample(sil_core(cel, upp), ss)
    open_ = core & (counts(kept, cat, upp, w, h, ss) == 0)
    # 컷 전에도 열려 있던 자리는 이 컷의 책임이 아니다 (§18 seal이 닫는다)
    open_ &= ~(core & (counts(before, cat, upp, w, h, ss) == 0))
    back: list = []
    taken: set[int] = set()
    while open_.any():
        best = None
        for lay in cut:
            if id(lay) in taken:
                continue
            got = _layer_mask(cat, lay, upp, w, h, ss)
            if got is None:
                continue
            m, x0, y0 = got
            hit = int((open_[y0:y0 + m.shape[0], x0:x0 + m.shape[1]]
                       & (m > 0)).sum())
            if hit and (best is None or hit > best[0]):
                best = (hit, lay, m, x0, y0)
        if best is None:
            break                          # 되살릴 수 있는 장이 없다 — seal이 맡는다
        _hit, lay, m, x0, y0 = best
        taken.add(id(lay))
        open_[y0:y0 + m.shape[0], x0:x0 + m.shape[1]] &= ~(m > 0)
        back.append(lay)
    if not back:
        return kept, 0
    # **원래 자리로** 되돌린다 — 그리기 순서가 곧 색이다
    order = {id(l): i for i, l in enumerate(before)}
    out = sorted(kept + back, key=lambda l: order[id(l)])
    return out, len(back)


# 봉인의 성장이 감수하는 해악 — **얻는 구멍 px당** 목표 대비 오차 증가
# (ΔE×px, `holes.grow_covers` 문서의 새 자). 값의 자가 아니다: 안 칠한 자리는
# 차 도색이 그대로 비치므로 어떤 색 오차보다 나쁘다. 그래도 무한이 아닌 것은
# 성장이 **다른 자리**를 함께 건드리기 때문이다 — 한 픽셀을 얻자고 스무
# 픽셀을 나쁘게 만들면 그 자리가 새 얼룩이다.
#
# **실측 스윕(01·09·11)의 무릎이 200이다.** 60 → 200 → 600에서 성장이
# 317/352/269 → 479/439/343 → 589/479/416회로 늘고 봉인 도형이
# 254/132/185 → 215/112/148 → 174/102/132장으로 준다. 총 도형은 각각
# −34/−24/−35 · −83/−36/−50이고, 평균 ΔE는 200에서 +0.03/+0.04/−0.01
# (노이즈)인데 600에서 +0.09/+0.08/+0.03으로 한 방향으로 밀린다. 반대색
# 슬리버는 두 단 모두 준다 (성장이 곧 슬리버를 덮는 손이라서다).
_SEAL_HARM = float(os.environ.get("FS_SEAL_HARM", 200.0))
# 봉인 메움이 조각을 묶는 반경 (px) — 값은 군집 **수**가 정한다 (`seal_coverage`)
_SEAL_GROUP_R = int(os.environ.get("FS_SEAL_GROUP", 4))
# 봉인 점의 스케일 사다리 (게임 입력 스텝의 배수) — 작은 것부터 훑는다
_DOT_SCALES = (0.01, 0.02, 0.03, 0.05, 0.08)


def _seal_dots(plan: LayerPlan, cel: CelArt, cat: Catalog, hard: np.ndarray,
               ss: int, log=print, at: int | None = None) -> int:
    """남은 표본을 **반드시** 덮는 최소 도형 — 봉인의 마지막 손.

    성장·메움은 모양과 값을 본다. 여기는 안 본다: 남은 자리는 열 몇 표본짜리
    부스러기이고 그 크기에서 남는 물음은 "덮이나" 하나뿐이다. 메움
    (`holes.fill_holes`)이 여기까지 못 오는 것은 그쪽이 **1x 격자**에서
    덮였는지를 묻기 때문이다 — 픽셀 하나를 반만 덮은 도형도 그쪽에서는
    성공이라 같은 자리가 라운드마다 되돌아온다 (실측 S0-09: 4라운드 뒤 10표본).

    후보 자리는 유한하다. 게임 입력이 위치를 0.5유닛·스케일을 0.01 격자에
    세우므로, **작은 스케일부터** 그 격자를 훑어 목표 표본을 다 덮으면서
    실루엣을 안 넘는 첫 수를 고른다 (결정적). 실루엣을 안 넘는 수가 없으면
    넘김이 가장 작은 수를 쓴다 — 미커버는 자격이고 스필은 값이라, 이 한
    자리에서는 자격이 이긴다.
    """
    from ..model import Layer, _q

    w, h = cel.size
    upp = plan.units_per_px
    flat = cel.flat_render()
    sil = _upsample(cel.labels >= 0, ss)
    ncc, cc, cstats, _ = cv2.connectedComponentsWithStats(
        hard.astype(np.uint8), connectivity=8)
    n = 0
    for ci in range(1, ncc):
        cx0 = cstats[ci, cv2.CC_STAT_LEFT]; cy0 = cstats[ci, cv2.CC_STAT_TOP]
        cx1 = cx0 + cstats[ci, cv2.CC_STAT_WIDTH]
        cy1 = cy0 + cstats[ci, cv2.CC_STAT_HEIGHT]
        tgt = cc[cy0:cy1, cx0:cx1] == ci
        ys, xs = np.nonzero(tgt)
        # 표본 격자 중심 → px (표본 하나가 1/ss px이므로 +0.5 표본이 그 중심)
        cx = (cx0 + xs.mean() + 0.5) / ss
        cy = (cy0 + ys.mean() + 0.5) / ss
        color = tuple(int(v) for v in flat[min(h - 1, int(cy)),
                                           min(w - 1, int(cx))])
        bx = _q((cx - w / 2) * upp, 0.5)
        by = _q((h / 2 - cy) * upp, 0.5)
        best = None
        for sc in _DOT_SCALES:
            for dx, dy in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1),
                           (-1, -1), (1, -1), (-1, 1), (1, 1)):
                lay = Layer(shape=_FILL_SHAPE, x=bx + dx * 0.5, y=by + dy * 0.5,
                            sx=sc, sy=sc, rot=0.0, skew=0.0, color=color,
                            alpha=100.0, label="seal")
                polys = _poly_px(cat, lay, upp, w, h)
                box = poly_bbox(polys, w, h, ss, pad=1)
                if box is None:
                    continue
                bx0, by0, bx1, by1 = box
                m = poly_mask(polys, (by1 - by0, bx1 - bx0), bx0 / ss, by0 / ss, ss)
                # 목표를 다 덮나 (군집 bbox가 도형 bbox 안에 들어와야 한다)
                if not (bx0 <= cx0 and by0 <= cy0 and bx1 >= cx1 and by1 >= cy1):
                    continue
                sub = m[cy0 - by0:cy1 - by0, cx0 - bx0:cx1 - bx0]
                if not (sub | ~tgt).all():
                    continue
                spill = int(np.count_nonzero(m & ~sil[by0:by1, bx0:bx1]))
                if best is None or spill < best[0]:
                    best = (spill, lay)
                if spill == 0:
                    break
            if best is not None and best[0] == 0:
                break
        if best is None:                   # 사다리 끝까지 못 덮었다 — 가장 큰 점
            lay = Layer(shape=_FILL_SHAPE, x=bx, y=by, sx=_DOT_SCALES[-1],
                        sy=_DOT_SCALES[-1], rot=0.0, skew=0.0, color=color,
                        alpha=100.0, label="seal")
            best = (0, lay)
        if at is None:
            plan.layers.append(best[1])
        else:
            plan.layers.insert(at, best[1])
            at += 1
        n += 1
    return n


def _make_room(plan: LayerPlan, cel: CelArt, cat: Catalog,
               budget: int | None, ss: int, st: dict, want: int = 0,
               weight: np.ndarray | None = None) -> None:
    """예산 자리를 만든다 — **혼자 안 덮는 장부터** 판다 (자격 보호 + 되짚기).

    **획을 절대 보호하지 않는다.** 라벨 보호는 메움·봉인뿐이다 — 파이프라인의
    재컷이 이미 같은 자리에서 같은 답을 냈다 (`route_cel`: "획까지 절대 보호하면
    포화 장에서 채움이 통째로 죽는다"). 실측 W4-04(예산 포화, 획 2,659장)에서
    획을 보호한 채 287장을 팔았더니 그 287장이 전부 채움이라 — 채움 708장의
    40%다 — 그린 영역이 139개 줄고 작은 중요 영역 보존이 0.94 → 0.46으로
    무너졌다. 획과 채움을 같은 저울(지각 가중 영향)에 올려야 "덜 보이는 획"이
    "면 색"보다 먼저 팔린다.

    `weight`(중요도 가중)도 반드시 넘긴다 — 안 넘기면 그 저울이 면적×색차뿐이라
    작고 또렷한 것이 먼저 팔린다.
    """
    from ..pruneplan import prune_impact

    if budget is None or len(plan.layers) + want <= budget:
        return
    keep_i = unique_layers(plan, cel, cat, ss)
    before = list(plan.layers)
    cut, _ps = prune_impact(plan, cat, budget=max(1, budget - want),
                            protect_labels=("hole", "seal"),
                            protect_idx=keep_i, weight=weight)
    fixed, _back = repair_cut(before, cut.layers, cel, cat,
                              plan.units_per_px, ss)
    st["seal_sold"] += len(before) - len(fixed)
    plan.layers[:] = fixed


def seal_coverage(plan: LayerPlan, cel: CelArt, cat: Catalog, *,
                  log=print, budget: int | None = None, ss: int = _SS,
                  rounds: int = 4, weight: np.ndarray | None = None,
                  progress=None) -> dict:
    """**마지막 봉인** — 실루엣 내부의 안 칠한 표본을 0으로 만든다.

    파이프라인의 맨 끝(전역 미세 조정 **뒤**)에 선다. 그 앞의 모든 단은 값을
    묻는 단이라 λ와 거래하지만, 이 단은 안 한다 — 여기서 남은 자국은 크기가
    얼마든 인게임에서 차 도색이 비치는 자리다.

    손은 세 벌이고 **싼 것부터** 쓴다:

    ① **성장** — 이미 놓인 도형을 양자화 한 스텝 키운다 (`holes.grow_covers`).
       레이어 0장이라 예산도 구조도 안 건드린다. **슈퍼샘플 격자에서** 판정을
       돌리는 것이 요점이다 (`grow_covers`의 `ss`) — 1x에서는 "이미 덮은
       픽셀" 안의 틈을 겨눌 길이 없어 이 손이 거의 안 움직인다.
    ② **메움 도형** — 그러고도 남은 자리에 `holes.fill_holes`를 값 게이트
       없이(min_px 1 · λ 0) 건다. 예산이 찼으면 **혼자 안 덮는 장**부터 팔아
       자리를 만든다 (`unique_layers` 보호 + `repair_cut` 되짚기라 그 컷이 새
       구멍을 못 연다).
    ③ **봉인 점** — 그래도 남은 표본을 `_seal_dots`가 반드시 덮는다. ②가
       1x 격자를 보는 한 반 픽셀짜리 잔여는 영영 안 닫히기 때문이다.

    봉인 뒤로는 어떤 단도 기하를 안 건드린다 — 그것이 이 단이 맨 끝인 이유다.
    """
    from .holes import fill_holes, grow_covers

    st = {"seal_grow": 0, "seal_layers": 0, "seal_sold": 0, "seal_dots": 0}
    hard = hard_holes(plan, cel, cat, ss)
    st["seal_before"] = int(hard.sum())
    if not hard.any():
        st["seal_after"] = 0
        return st
    quiet = lambda *_a, **_k: None         # noqa: E731 — 라운드 로그는 아래 한 줄로
    nr = max(1, rounds)
    for _r in range(nr):
        if progress:                       # 라운드가 눈금이다 (대개 한두 바퀴)
            progress(_r / nr)
        # 획도 **긴 축으로만** 키운다 — 두 획이 만나는 모서리의 쐐기가 잔여
        # 미커버의 태반이고(실측 W4-01), 긴 축 성장은 선을 안 굵힌다
        st["seal_grow"] += grow_covers(plan, cel, cat, log=quiet, passes=4,
                                       need=hard, ss=ss,
                                       harm_per_px=_SEAL_HARM, ink_long=True)
        hard = hard_holes(plan, cel, cat, ss)
        if not hard.any():
            break
        need = need_px(hard, ss)
        # 자리 만들기 — 군집 수만큼은 있어야 한 바퀴가 헛돌지 않는다
        ncc, _, _, _ = cv2.connectedComponentsWithStats(
            need.astype(np.uint8), connectivity=8)
        want = int(ncc - 1)
        _make_room(plan, cel, cat, budget, ss, st, want, weight)
        room = 10 ** 9 if budget is None else max(0, budget - len(plan.layers))
        if room <= 0:
            log(msg("  경고: 봉인할 자리가 없다 — 예산이 꽉 찼다"))
            break
        # 조각을 **멀리까지 묶는다**. 봉인 잔여는 1,500px 남짓이 150군집
        # 안팎으로 흩어져 있어 값은 군집 수가 정한다 (실측 W3-11: 자리
        # 1,502px · 도형 438장). 묶는 반경을 2 → 4px로 넓히면 군집이 그만큼
        # 줄고, 지나치게 뻗은 그룹은 실루엣 초과 판정이 반씩 가른다
        # 스택 **바닥**에 끼운다 — 봉인이 맡는 자리는 아무도 안 덮는 표본이라
        # z가 바닥이어도 그 표본에서는 보이고, 타원의 스필은 이웃 면·선화가
        # 위에서 덮는다. 맨 뒤에 얹으면 그 스필이 선을 끊고 경계 반대편에 색
        # 얼룩을 흩뿌린다 (실측 W7-01: 선·면 6,340px이 봉인 스필에 덮여 있었다)
        n = fill_holes(plan, cel, cat, log=quiet, min_px=1, max_layers=room,
                       holes=need, label="seal", group_r=_SEAL_GROUP_R, at=0)
        st["seal_layers"] += n
        hard = hard_holes(plan, cel, cat, ss)
        if not hard.any() or n == 0:
            break
    # 점은 군집 하나에 한 장이라, 한 장으로 못 덮는 군집은 갈라져 남는다 —
    # 남는 것이 없어질 때까지 돈다 (군집이 매 바퀴 작아지므로 몇 바퀴면 끝난다)
    for _ in range(4):
        if not hard.any():
            break
        _make_room(plan, cel, cat, budget, ss, st, weight=weight)
        st["seal_dots"] += _seal_dots(plan, cel, cat, hard, ss, log, at=0)
        hard = hard_holes(plan, cel, cat, ss)
    # 인게임 상한은 게이트가 아니라 **기계의 한계**다 — 넘긴 채로 내보낼 수
    # 없다. 넘겼으면 자리를 만들고, 그 컷이 연 자리는 점이 도로 닫는다
    for _ in range(3):
        if budget is None or len(plan.layers) <= budget:
            break
        _make_room(plan, cel, cat, budget, ss, st, weight=weight)
        hard = hard_holes(plan, cel, cat, ss)
        if hard.any():
            st["seal_dots"] += _seal_dots(plan, cel, cat, hard, ss, log, at=0)
    hard = hard_holes(plan, cel, cat, ss)
    st["seal_after"] = int(hard.sum())
    log(msg("  봉인: 표본 {before:,} → {after:,} (성장 {grow}회 · 도형 {layers}장{dots}{sold})",
            before=st["seal_before"], after=st["seal_after"],
            grow=st["seal_grow"], layers=st["seal_layers"],
            dots=(msg(" · 점 {n}장", n=st["seal_dots"])
                  if st["seal_dots"] else ""),
            sold=(msg(" · 되팜 {n}장", n=st["seal_sold"])
                  if st["seal_sold"] else "")))
    return st
