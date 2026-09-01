"""cel 노선 마무리 — 전역 미세 조정 (DiffCompositing 개념의 이산 이식).

celfit 배치는 영역별 그리디다: 채점판이 자기 영역 ROI만 보고, 나중 면이
실제로 어디를 덮는지는 벌점 근사(_PEN_WASTE)로만 안다. 이 패스는 **완성된
스택 전체**를 놓고 레이어 하나씩 게임 양자화 스텝(이동 0.5·스케일 0.01·회전
0.1°) 이웃으로 좌표하강시킨다. 전 레이어 불투명이라 픽셀 색 = 그 픽셀을
덮는 최상위 레이어 색 — "레이어 i를 움직이면 정확히 어떤 px가 어떤 색으로
바뀌는지"를 증분으로 계산할 수 있고, 목표 대비 제곱 오차가 줄어드는 이동만
받는다. DiffCompositing(SIGGRAPH Asia 2020)의 "합성 스택 전체를 미분해
요소를 미세 조정"을, 고정 도형·양자화 파라미터 어휘에 맞게 기울기 대신
양자화 이웃 탐색으로 옮긴 것이다.

목표 이미지 = 셀 평면 렌더 (선화 px는 원화 색 — flat_render가 이미 그렇다).
배치(celfit)·선화(_fit_lines)가 각자 겨눴던 목표의 합성이라, 채움 경계와
선화 획이 같은 잣대 위에서 함께 미세 조정된다.

불변 보장:
- 레이어 수·순서·색·도형 불변 — 기하(x·y·sx·sy·rot·skew)만 움직인다.
  기울기 축은 2026-09-01에 열렸다 (레코드 +0x70을 찾아 주입·저장 왕복이
  실측으로 닫혔다) — 그전까지는 주입이 그 축을 조용히 빼고 써서 여기서 밀면
  도안과 인게임이 갈렸다. `FS_SKEW=0`이면 예전처럼 안 민다.
- 실루엣 px를 새로 노출하는 이동은 **무조건 기각** — holes 게이트(4px+
  군집 0)가 패스 후에도 성립한다.
- 캔버스 rect 밖 돌출을 늘리는 이동도 기각 — 밖은 채점이 못 보는 무벌점
  구간이라 열어 두면 경계 레이어가 그리로 샌다 (celfit 가드와 같은 논리).
- 결정적 (난수 없음, 순서 고정).
"""

from __future__ import annotations

import os

import cv2
import numpy as np

from ..i18n import msg
from .catalog import Catalog
from .celart import CelArt
from .celfit import _poly_px
from .celfit.affine import step_visible
from .celfit.chain import _GAP_TOL, placed_line
from .model import Layer, LayerPlan
from .stop import stop_here

# 비용 상수 (제곱 RGB 오차 단위, 채널당 최대 255² × 3 = 195,075)
# 실루엣 밖(캔버스) 침범 px당 상수 — 인게임 배경은 흰 프리뷰가 아니라 차체
# 도색이라, 밝은 색 스필이 "흰 목표와 가깝다"고 싸지면 안 된다 (pruneplan
# _BG_PEN과 같은 논리의 제곱 오차판)
_P_BG = 4000.0
# 실루엣 미커버(구멍) px당 상수 바닥 — 기존 1~3px 반점을 덮는 성장에 상을
# 주고, 밝은 셀 색 위 핀홀도 색 불문 비용이 되게 한다. 신규 노출은 어차피
# 기각이라 이 값은 "기존 구멍을 메우는 이득"으로만 작동한다
_P_HOLE = 8000.0
_EPS = 1.0            # 이보다 못 버는 이동은 무시 (부동소수 채터 방지)
_MAX_WALK = 8         # 같은 방향 연속 스텝 상한
# **JND 대역 벌점** (기본 켜짐 · `FS_FT_JND=0`으로 끈다) — 덮은 px가 목표와
# Lab ΔE 4~12로 틀리면
# px당 이 상수를 더한다. 제곱 오차만 보면 "큰 오차를 고치며 은근한 틀림을
# 흩뿌리는" 이동이 늘 이긴다 — 잔차 초점 패스가 유령 경계(목표에 없는 저대비
# 경계)를 배로 늘리는 원인이 이것이다 (단계 귀속 실측 S1-01: ghost 1,530
# → 3,611px). 4~12 대역은 수리 문턱(12) 아래·JND(4) 위라 어느 손도 다시
# 안 보는 자리이므로, 만들 때 막는 것이 유일한 손이다. 값 1,500은 RGB 노름
# ~39(Lab ΔE 십수)의 제곱 오차에 해당 — 대역 px 하나가 "잘 보이는 오차"
# 하나 값이 되게 한다. 목표 자체 경계 2px 안(경계 양자화 자리)과 획(ink)
# 소유 px는 제외 — 획이 흐린 선 위를 긋는 것은 이 벌점의 대상이 아니다
_P_JND = 1500.0
# **이음 게이트** (기본 켜짐 · `FS_FT_JGATE=0`으로 끈다) — 획 레이어의 이동이
# 같은 획의 이음을
# 벌리면 기각한다. 미세 조정은 채움 목표의 제곱 오차만 보므로 획을 제
# 이웃 마디에서 떼어 놓고도 이득이면 받는다 — 단계 귀속 실측(JS0/JS1,
# 01·03·07): 잔차 초점 한 단이 joint_gap을 3.07→4.26px(+39%)로 벌리고,
# line 노선 최종(3.05)이 cel 배치 직후(3.07)와 같다 — cel의 이음 틈 초과분은
# 전부 이 손이 만든다. 가격이 아니라 자격이다(실루엣 신규 노출 기각과 같은
# 급): 이음 상대 끝까지의 거리가 max(현재, _GAP_TOL)을 넘는 이동은 못 간다.
# 상대는 같은 획 안 다른 레이어의 가장 가까운 끝 — 새 상수 없음.

# 수 하나 = (속성, 게임 양자화 스텝)의 묶음 — 짧은 스텝 먼저 (대부분의 개선은
# 반 스텝이다). 홑 축 여덟 뒤에 **한쪽 변 수** 넷이 선다: 이동과 스케일을 한
# 스텝으로 묶으면 반대쪽 변을 (거의) 붙든 채 한 변만 물러서거나 나아간다.
# 축을 하나씩 밟는 하강에는 이 수가 없다 — 이동만 하면 양 변이 같이 밀리고
# 스케일만 줄이면 양 변이 같이 물러서서, 낮은 대비 경계를 한 변만 넘어간
# 도형이 국소 최적에 갇힌다 (실측 X3-01 허벅지: 목표 celΔ0 자리의 색조 계단.
# 그 계단은 ΔE 4~12라 수리 문턱 아래고, 이 패스만이 만질 수 있는 자리다).
# 바깥 sign 루프가 두 성분의 부호를 함께 뒤집으므로 궤도 대표 넷이면 여덟 수다.
_AXES = ((("x", 0.5),), (("y", 0.5),), (("x", 1.0),), (("y", 1.0),),
         (("sx", 0.01),), (("sy", 0.01),), (("rot", 0.1),), (("rot", 0.4),),
         (("x", 0.5), ("sx", -0.01)), (("x", 0.5), ("sx", 0.01)),
         (("y", 0.5), ("sy", -0.01)), (("y", 0.5), ("sy", 0.01)))
# **기울기 축** (§11) — 배치에만 있고 여기에 없으면 기능이 절반만 구현된 것이다.
# 스텝은 게임 입력 격자 그대로(0.01)이고 바깥 sign 루프가 양쪽을 본다.
#
# 축을 더하는 것만으로는 "모든 레이어에 전단을 낸다"가 안 된다 — `try_move`가
# **엄격한 개선**(`< -_EPS`)일 때만 받으므로, 지금 `skew=0`인 레이어는 실제로
# 목적함수가 좋아질 때만 들어온다. 반대 방향(0으로 돌아가기)도 같은 축이
# 맡는다. 동률에서 0을 고르는 것만 축으로 안 되므로, 레이어마다 축을 밟기
# **전에** 전단 0을 한 번 따로 묻는다 (§11의 "완전 동률이면 skew=0").
_SKEW_AXIS = (("skew", 0.01),)
_SKEW_ON = os.environ.get("FS_SKEW", "0") != "0" and \
    os.environ.get("FS_FT_SKEW", "1") != "0"


def _win_mask(cat: Catalog, lay: Layer, upp: float, w: int, h: int
              ) -> tuple[np.ndarray, np.ndarray, float]:
    """레이어 → (bbox [x0,y0,x1,y1], 창 마스크(bool), 캔버스 밖 돌출 px).

    래스터는 count_hole_clusters·celfit과 같은 식(round → fillPoly, 짝홀 XOR)
    — 게이트가 보는 커버리지와 이 패스의 소유자 모델이 같은 픽셀을 본다.
    """
    polys = _poly_px(cat, lay, upp, w, h)
    xs = np.concatenate([p[:, 0] for p in polys])
    ys = np.concatenate([p[:, 1] for p in polys])
    ext = float(max(0.0, -xs.min(), xs.max() - w, -ys.min(), ys.max() - h))
    x0 = max(0, int(np.floor(xs.min())) - 1)
    y0 = max(0, int(np.floor(ys.min())) - 1)
    x1 = min(w, int(np.ceil(xs.max())) + 2)
    y1 = min(h, int(np.ceil(ys.max())) + 2)
    if x0 >= x1 or y0 >= y1:
        return np.array([0, 0, 0, 0], np.int32), np.zeros((0, 0), bool), ext
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    off = np.array([x0, y0], np.int32)
    rp = [np.round(p).astype(np.int32) - off for p in polys]
    if len(rp) == 1:
        cv2.fillPoly(m, rp, 1)
    else:
        for p in rp:
            mm = np.zeros_like(m)
            cv2.fillPoly(mm, [p], 1)
            m ^= mm
    return np.array([x0, y0, x1, y1], np.int32), m.astype(bool), ext


def refine_plan(plan: LayerPlan, cel: CelArt, cat: Catalog, *,
                log=print, progress=None, max_passes: int = 3,
                only: list[int] | None = None, tag: str = "전역") -> dict:
    """플랜을 제자리에서 미세 조정. 반환: 통계 딕셔너리 (report용).

    `only`를 주면 **그 레이어들만** 민다 (§12 잔차 초점 패스). 소유자 모델은
    여전히 스택 전체를 보므로 판정은 같고, 훑는 대상만 좁아진다 — 잔차가
    남은 자리를 **보정 도형을 사기 전에** 기존 도형의 이동·스케일·회전으로
    먼저 고치는 자리다.
    """
    layers = plan.layers
    n = len(layers)
    w, h = cel.size
    upp = plan.units_per_px
    for l in layers:
        if l.mask or l.alpha < 99.5 or cat[l.shape].gradient is not None:
            log(msg("  경고: 전역 미세 조정 생략 — 불투명 소유자 모델 밖 레이어 포함"))
            return {"moved_layers": 0, "accepts": 0, "skipped": True}

    tgt = cel.flat_render().astype(np.int32)          # 목표 (선화 px = 원화 색)
    sil = cel.labels >= 0
    lut = np.full((n + 1, 3), 255, np.int32)          # [0] = 배경 흰색
    for i, l in enumerate(layers):
        lut[i + 1] = l.rgb()
    # 미커버 px 비용: 흰 노출 오차 + 실루엣 안이면 색 불문 바닥
    ucost = ((255 - tgt) ** 2).sum(2).astype(np.float64)
    ucost[sil] += _P_HOLE
    # JND 대역 벌점 채비 (_P_JND 문서) — 스위치 꺼짐이면 계산도 안 한다
    _jnd = os.environ.get("FS_FT_JND", "1") != "0"
    if _jnd:
        tgt_lab = cv2.cvtColor(tgt.astype(np.uint8),
                               cv2.COLOR_RGB2LAB).astype(np.float32)
        lut_lab = cv2.cvtColor(np.clip(lut, 0, 255).astype(np.uint8)
                               .reshape(-1, 1, 3),
                               cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
        _e = np.zeros((h, w), bool)
        for _dx, _dy in ((1, 0), (0, 1)):
            _d = np.linalg.norm(tgt_lab[_dy:, _dx:]
                                - tgt_lab[:h - _dy, :w - _dx], axis=-1) > 4.0
            _e[_dy:, _dx:] |= _d
            _e[:h - _dy, :w - _dx] |= _d
        band_ok = sil & ~cv2.dilate(
            _e.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))).astype(bool)
        not_ink = np.array([True] + [l.label != "ink" for l in layers])
        # 획 면제의 사각지대 (기본 켜짐 · `FS_FT_INKBAND=0`으로 끈다) — 획이
        # **선 지도 밖**으로
        # 벗어난 px는 면제하지 않는다. 흐린 선 위를 긋는 것은 획의 일이지만,
        # 선 지도가 없는 면 위의 대역색 발자국은 채움의 월경과 같은 유령이다
        # (단계 귀속 실측: 벌점 채택 후에도 잔차 초점이 06에서 ghost +1,543 —
        # 태반이 획 이동). 1px 팽창은 래스터 반 픽셀 몫이다
        _lz = getattr(cel, "line_mask", None)
        _inkband = (os.environ.get("FS_FT_INKBAND", "1") != "0"
                    and _lz is not None)
        if _inkband:
            line_zone = cv2.dilate(_lz.astype(np.uint8),
                                   np.ones((3, 3), np.uint8)).astype(bool)

    boxes = np.zeros((n, 4), np.int32)
    masks: list[np.ndarray] = [None] * n              # type: ignore[list-item]
    exts = np.zeros(n, np.float64)
    owner = np.full((h, w), -1, np.int32)             # 최상위 레이어 (-1 = 배경)
    for i, l in enumerate(layers):
        boxes[i], masks[i], exts[i] = _win_mask(cat, l, upp, w, h)
        x0, y0, x1, y1 = boxes[i]
        owner[y0:y1, x0:x1][masks[i]] = i

    # 이음 게이트 채비 (_P_JND 아래 문서) — 같은 획의 끝끼리 짝을 맺는다
    _jgate = os.environ.get("FS_FT_JGATE", "1") != "0"
    partners: dict[int, list] = {}
    if _jgate:
        ends: dict[int, tuple] = {}
        by_sid: dict[int, list[int]] = {}
        for i, l in enumerate(layers):
            if l.label == "ink" and l.stroke >= 0:
                got = placed_line(cat, l, upp, w, h)
                if got is not None:
                    ends[i] = (got[0], got[1])
                    by_sid.setdefault(l.stroke, []).append(i)
        for sid, ids in by_sid.items():
            if len(ids) < 2:
                continue
            for i in ids:
                pl = []
                for k in (0, 1):
                    best = None
                    for j in ids:
                        if j == i:
                            continue
                        for ej in (0, 1):
                            d = float(np.hypot(*(ends[i][k] - ends[j][ej])))
                            if best is None or d < best[0]:
                                best = (d, j, ej)
                    pl.append(None if best is None else (best[1], best[2]))
                partners[i] = pl

    def widens_joint(i: int, cand: Layer) -> bool:
        """cand로 옮기면 같은 획의 이음이 벌어지나 (게이트)."""
        pl = partners.get(i)
        if not pl:
            return False
        cur = placed_line(cat, layers[i], upp, w, h)
        new = placed_line(cat, cand, upp, w, h)
        if cur is None or new is None:
            return False
        for k, pr in enumerate(pl):
            if pr is None:
                continue
            j, ej = pr
            pe = placed_line(cat, layers[j], upp, w, h)
            if pe is None:
                continue
            d_old = float(np.hypot(*(cur[k] - pe[ej])))
            d_new = float(np.hypot(*(new[k] - pe[ej])))
            if d_new > max(d_old, _GAP_TOL) + 1e-6:
                return True
        return False

    def cost_at(ys: np.ndarray, xs: np.ndarray, o: np.ndarray) -> np.ndarray:
        """픽셀들의 비용 — 소유자 o(-1 = 미커버) 기준."""
        d = ((tgt[ys, xs] - lut[o + 1]) ** 2).sum(1).astype(np.float64)
        cov = o >= 0
        d[cov & ~sil[ys, xs]] += _P_BG
        if _jnd:
            sub = not_ink[o + 1]
            if _inkband:
                sub = sub | ~line_zone[ys, xs]
            bsel = cov & band_ok[ys, xs] & sub
            if bsel.any():
                de = np.linalg.norm(tgt_lab[ys[bsel], xs[bsel]]
                                    - lut_lab[o[bsel] + 1], axis=1)
                d[bsel] += _P_JND * ((de > 4.0) & (de <= 12.0))
        unc = ~cov
        if unc.any():
            d[unc] = ucost[ys[unc], xs[unc]]
        return d

    def unders(i: int, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
        """i가 비켜난 픽셀들이 드러내는 소유자 (i 아래 최상위, 없으면 -1)."""
        u = np.full(len(ys), -1, np.int32)
        dy0, dy1 = int(ys.min()), int(ys.max())
        dx0, dx1 = int(xs.min()), int(xs.max())
        js = np.flatnonzero((boxes[:i, 0] <= dx1) & (boxes[:i, 2] > dx0)
                            & (boxes[:i, 1] <= dy1) & (boxes[:i, 3] > dy0))
        unres = np.ones(len(ys), bool)
        for j in js[::-1]:
            if not unres.any():
                break
            bx0, by0, bx1, by1 = boxes[j]
            sel = (unres & (ys >= by0) & (ys < by1)
                   & (xs >= bx0) & (xs < bx1))
            if not sel.any():
                continue
            idx = np.flatnonzero(sel)
            hit = masks[j][ys[idx] - by0, xs[idx] - bx0]
            idx = idx[hit]
            u[idx] = j
            unres[idx] = False
        return u

    def try_move(i: int, cand: Layer):
        """이동 평가 — (Δ비용, 커밋 데이터) 또는 None(기각)."""
        box_n, m_n, ext_n = _win_mask(cat, cand, upp, w, h)
        if ext_n > exts[i] + 0.5:          # 캔버스 밖 돌출 증가 — 무벌점 구간
            return None
        box_o, m_o = boxes[i], masks[i]
        ux0 = int(min(box_o[0], box_n[0])); uy0 = int(min(box_o[1], box_n[1]))
        ux1 = int(max(box_o[2], box_n[2])); uy1 = int(max(box_o[3], box_n[3]))
        if ux0 >= ux1 or uy0 >= uy1:
            return None
        mo = np.zeros((uy1 - uy0, ux1 - ux0), bool)
        if m_o.size:
            mo[box_o[1] - uy0:box_o[3] - uy0, box_o[0] - ux0:box_o[2] - ux0] = m_o
        mn = np.zeros_like(mo)
        if m_n.size:
            mn[box_n[1] - uy0:box_n[3] - uy0, box_n[0] - ux0:box_n[2] - ux0] = m_n
        ow = owner[uy0:uy1, ux0:ux1]
        add = mn & ~mo & (ow < i)          # 새로 덮고, 위에 아무도 없는 px
        rem = mo & ~mn & (ow == i)         # 비켜나 드러나는 px
        delta = 0.0
        ysA, xsA = np.nonzero(add)
        ysD, xsD = np.nonzero(rem)
        if not len(ysA) and not len(ysD):
            return None
        u = np.zeros(0, np.int32)
        if len(ysD):
            ysD = ysD + uy0; xsD = xsD + ux0
            u = unders(i, ysD, xsD)
            if bool(((u < 0) & sil[ysD, xsD]).any()):
                return None                # 실루엣 신규 노출 = 새 구멍 — 기각
            cur = cost_at(ysD, xsD, np.full(len(ysD), i, np.int32))
            delta += float(cost_at(ysD, xsD, u).sum() - cur.sum())
        if len(ysA):
            ysA = ysA + uy0; xsA = xsA + ux0
            cur = cost_at(ysA, xsA, ow[add])
            delta += float(cost_at(ysA, xsA,
                                   np.full(len(ysA), i, np.int32)).sum()
                           - cur.sum())
        return delta, (ysA, xsA, ysD, xsD, u, box_n, m_n, ext_n)

    def commit(i: int, cand: Layer, data) -> None:
        ysA, xsA, ysD, xsD, u, box_n, m_n, ext_n = data
        if len(ysD):
            owner[ysD, xsD] = u
        if len(ysA):
            owner[ysA, xsA] = i
        boxes[i], masks[i], exts[i] = box_n, m_n, ext_n
        layers[i] = cand

    accepts = 0
    moved = np.zeros(n, bool)
    gain = 0.0
    _axes = _AXES + (_SKEW_AXIS,)
    todo = list(range(n)) if only is None else [i for i in only if 0 <= i < n]
    if not todo:
        return {"moved_layers": 0, "accepts": 0, "cost_gain": 0.0}
    for p in range(max_passes):
        pass_accepts = 0
        for si, i in enumerate(todo):
            stop_here()
            if progress and si % 250 == 0:
                progress((p + si / len(todo)) / max_passes,
                         msg("미세 조정 {p}/{total}", p=p + 1, total=max_passes))
            # **0으로 돌아갈 길을 먼저 연다** (§11) — 전단이 값을 잃은 자세로
            # 흘러간 레이어는 접어야 한다. 비용이 **안 나빠지면** 받는다
            # (엄격한 개선을 요구하면 완전 동률에서 전단이 그대로 남는다)
            _skew_i = _SKEW_ON and step_visible(cat, layers[i])
            if _skew_i and layers[i].skew:
                z = Layer(**{**layers[i].__dict__})
                z.skew = 0.0
                z = z.quantized()
                if not (_jgate and widens_joint(i, z)):
                    r = try_move(i, z)
                    if r is not None and r[0] <= _EPS:
                        commit(i, z, r[1])
                        gain -= min(r[0], 0.0)
                        accepts += 1
                        pass_accepts += 1
                        moved[i] = True
            for combo in (_axes if _skew_i else _AXES):
                for sign in (1.0, -1.0):
                    for _ in range(_MAX_WALK):
                        lay = layers[i]
                        cand = Layer(**{**lay.__dict__})
                        dead = False
                        for name, step in combo:
                            v = getattr(cand, name) + sign * step * (
                                1.0 if getattr(cand, name) >= 0
                                or name not in ("sx", "sy") else -1.0)
                            if name in ("sx", "sy") and abs(v) < 0.01:
                                dead = True
                                break
                            setattr(cand, name, v)
                        if dead:
                            break
                        cand = cand.quantized()
                        if _jgate and widens_joint(i, cand):
                            break
                        res = try_move(i, cand)
                        if res is None or res[0] > -_EPS:
                            break
                        commit(i, cand, res[1])
                        gain -= res[0]
                        accepts += 1
                        pass_accepts += 1
                        moved[i] = True
        log(msg("  {tag} 미세 조정 패스 {p}: 이동 {n}회 수락",
                tag=msg(tag), p=p + 1, n=pass_accepts))
        if pass_accepts < max(8, len(todo) // 100):
            break
    if progress:
        progress(1.0, msg("미세 조정 완료"))
    stats = {"moved_layers": int(moved.sum()), "accepts": accepts,
             "cost_gain": round(gain, 0),
             # §14 — 이 패스가 끝난 뒤 전단을 쓰는 레이어 수
             "skew_layers": sum(1 for l in layers if l.skew)}
    log(msg("  {tag} 미세 조정: 레이어 {moved}/{total}개 이동 (수락 {n}회)",
            tag=msg(tag), moved=stats["moved_layers"], total=len(todo),
            n=accepts))
    return stats


def reorder_fills(plan: LayerPlan, cel: CelArt, cat: Catalog, *,
                  log=print, max_passes: int = 2) -> dict:
    """채움의 **그리기 순서**를 목표 기준으로 미세 조정한다 — 도형 0장.

    배치의 그리기 순서는 넓이 내림차순 한 벌뿐이고, 그 뒤 어느 단도 순서를
    다시 묻지 않는다. 그래서 "옳은 색 조각이 이미 그 자리를 덮고 있는데
    이웃 영역 조각이 **위에서** 가리는" px가 남는다 — 목표에 없는 저대비
    경계(유령 계단)의 몸통이 이것이다 (실측 X4-01: 아래 깔린 채움이 지금
    보이는 색보다 ΔE 4+ 가까운 px 18k, 12+ 가까운 px 6.7k).

    수는 하나다: 채움 a를 **위 조각 b 바로 뒤로** 올린다. 그때 바뀌는 화면은
    정확히 F = mask(a) ∧ {지금 소유자의 순위가 (a, b] 안}이고, 커버리지
    집합은 안 변하므로 실루엣 노출이 원리적으로 없다. F의 목표 대비 제곱
    오차가 줄어드는 이동만 받는다 (미세 조정과 같은 자·같은 게이트).

    대상은 선화 블록 **아래**의 불투명 채움뿐이다 — 선화·덧칠·마스크·
    그라데이션의 순서는 그대로다 (선은 모든 면 위라는 문법이 순서 그 자체다).
    결정적이다: 아래 순위부터 훑고, 후보 b는 가림 px 수 내림차순이다.
    """
    layers = plan.layers
    n = len(layers)
    w, h = cel.size
    upp = plan.units_per_px
    ink0 = next((i for i, l in enumerate(layers) if l.label == "ink"), n)
    # 움직이는 것은 배치가 놓은 채움("cel")뿐 — 메움·봉인은 바닥이 제자리고
    # (스필을 이웃이 덮는 설계), 수리·선화·마스크의 순서도 각자의 문법이다
    movable = [i for i, l in enumerate(layers)
               if i < ink0 and not l.mask and l.alpha >= 99.5
               and (l.label or "cel") == "cel"
               and cat[l.shape].gradient is None]
    if not movable:
        return {"reorder_moves": 0}

    tgt = cel.flat_render().astype(np.int32)
    lut = np.full((n + 1, 3), 255, np.int32)
    for i, l in enumerate(layers):
        lut[i + 1] = l.rgb()
    # 순서 수에도 같은 대역 자 (기본 켜짐 · `FS_RO_JND=0`으로 끈다, _P_JND
    # 문서) — 제곱 오차만
    # 보면 큰 오차 자리를 돌려받는 승급이 그 발자국의 다른 자리에 ΔE 4~12
    # 유령을 새로 깔아도 이긴다 (단계 귀속 실측 S2-05: reorder가 ghost +350)
    _jnd = os.environ.get("FS_RO_JND", "1") != "0"
    if _jnd:
        tgt_lab = cv2.cvtColor(tgt.astype(np.uint8),
                               cv2.COLOR_RGB2LAB).astype(np.float32)
        lut_lab = cv2.cvtColor(np.clip(lut, 0, 255).astype(np.uint8)
                               .reshape(-1, 1, 3),
                               cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
        _e = np.zeros((h, w), bool)
        for _dx, _dy in ((1, 0), (0, 1)):
            _d = np.linalg.norm(tgt_lab[_dy:, _dx:]
                                - tgt_lab[:h - _dy, :w - _dx], axis=-1) > 4.0
            _e[_dy:, _dx:] |= _d
            _e[:h - _dy, :w - _dx] |= _d
        sil_ = cel.labels >= 0
        band_ok = sil_ & ~cv2.dilate(
            _e.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))).astype(bool)

        def _priced(ys, xs, ids):
            d = ((tgt[ys, xs] - lut[ids + 1]) ** 2).sum(1).astype(np.float64)
            de = np.linalg.norm(tgt_lab[ys, xs] - lut_lab[ids + 1], axis=1)
            d += _P_JND * (band_ok[ys, xs] & (de > 4.0) & (de <= 12.0))
            return d
    boxes = np.zeros((n, 4), np.int32)
    masks: list[np.ndarray] = [None] * n              # type: ignore[list-item]
    owner = np.full((h, w), -1, np.int32)             # 레이어 **id** (index 불변)
    for i, l in enumerate(layers):
        boxes[i], masks[i], _ = _win_mask(cat, l, upp, w, h)
        x0, y0, x1, y1 = boxes[i]
        owner[y0:y1, x0:x1][masks[i]] = i
    order = list(range(n))                 # 순위 → id
    rank = np.arange(n, dtype=np.int64)    # id → 순위
    ink_rank = int(min((rank[i] for i in range(n) if layers[i].label == "ink"),
                       default=n))

    moves = 0
    gain = 0.0
    for _ in range(max_passes):
        pass_moves = 0
        for a in sorted(movable, key=lambda i: rank[i]):
            x0, y0, x1, y1 = boxes[a]
            if x0 >= x1 or y0 >= y1:
                continue
            ow = owner[y0:y1, x0:x1][masks[a]]
            ra = int(rank[a])
            covered = ow[(rank[ow] > ra) & (rank[ow] < ink_rank)]
            if not covered.size:
                continue
            ids, cnt = np.unique(covered, return_counts=True)
            # 후보 b — 가림 px 내림차순 상위 넷 (동률은 id 오름차순: 결정적)
            top = ids[np.lexsort((ids, -cnt))][:4]
            best = None
            ys0, xs0 = np.nonzero(masks[a])
            ys0 = ys0 + y0
            xs0 = xs0 + x0
            ow_flat = owner[ys0, xs0]
            d_a = (_priced(ys0, xs0, np.full(len(ys0), a, np.int32)) if _jnd
                   else ((tgt[ys0, xs0] - lut[a + 1]) ** 2).sum(1))
            for b in top:
                rb = int(rank[int(b)])
                sel = (rank[ow_flat] > ra) & (rank[ow_flat] <= rb)
                if not sel.any():
                    continue
                d_o = (_priced(ys0[sel], xs0[sel], ow_flat[sel]) if _jnd
                       else ((tgt[ys0[sel], xs0[sel]]
                              - lut[ow_flat[sel] + 1]) ** 2).sum(1))
                delta = float(d_a[sel].sum() - d_o.sum())
                if delta < -_EPS and (best is None or delta < best[0]):
                    best = (delta, rb, sel)
            if best is None:
                continue
            delta, rb, sel = best
            owner[ys0[sel], xs0[sel]] = a
            order.pop(ra)
            order.insert(rb, a)            # ra 제거로 rb 자리가 곧 "b 바로 뒤"
            for r_, id_ in enumerate(order):
                rank[id_] = r_
            ink_rank = int(min((rank[i] for i in range(n)
                                if layers[i].label == "ink"), default=n))
            gain -= delta
            moves += 1
            pass_moves += 1
        if not pass_moves:
            break
    if moves:
        plan.layers[:] = [layers[i] for i in order]
        log(msg("  그리기 순서 조정 {n}회 (도형 0장)", n=moves))
    return {"reorder_moves": moves, "reorder_gain": round(gain, 0)}
