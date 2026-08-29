"""선 재구성 엔진 — **두 노선이 함께 쓰는 하나**.

`line`은 이 엔진의 결과를 그대로 도안으로 내고, `cel`은 같은 결과를 색면
아래에 깐다. 그래서 여기에는 노선 분기가 없다: 증거 → 그래프 → 이어긋기 →
역할 → 후보 생성까지가 공통이고, **무엇을 그릴지와 어느 후보를 쓸지만**
정책(`policy`)이 정한다. 같은 입력이면 두 노선이 같은 논리 획 그래프를 얻고,
결과가 갈리면 그 이유가 정책 한 칸으로 적힌다 (`Reconstruction.report`).

    build_strokes   성분 → 뼈대 → 그래프 → 이어긋기 → 증거 → 역할 → 덩어리 채움
    place_strokes   후보 경쟁 → 정책 선택 → 확정 → 확장·이음 보수 → 병합

`line` 노선은 이 엔진의 품질을 혼자 재는 자리이기도 하다 — 면이 안 가려 주므로
결함이 그대로 보인다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..catalog import Catalog
from ..celart import CelArt
from ..model import LayerPlan
from . import candidates as C
from . import chain
from . import intent as I
from .evidence import (EvidenceMaps, fill_neighborhood, junction_degrees,
                       sample)
from .geometry import _min_span
from .graph import (_CONF, _DE, _DETAIL_W, _ISO_LEN_REL, _LINE_FRAG_REL,
                    _SIL, _SUP_OK, FEATURE, NOISE, TEXTURE, LogicalStroke,
                    classify, continue_strokes, texture_representatives)
from .grammar import (_FAT_MAX_MUL, _FAT_MIN_AREA, _LINE_W_REL, _WCAP_CV,
                      _fill_fat, _patch_seams)
from .merge import merge_costrokes
from .policy import _SPAN_REL
from .scoring import _PEN_FAR, _PEN_LINE, _RETRACE, _Scorer
from .skeleton import (_dt_along, _paths, _prune_spurs, _thin, smooth_path)
from .stroke import _STROKE_SLIM, _path_worth
from .vocabulary import min_stroke_width_px

# 경로 평활 창 (홀수 탭) — 선 지도의 계단 현상은 지도 쪽에서 못 없앤다
# (`lineart.hysteresis` 문서). 실제로 듣는 자리는 여기뿐이고 5탭이 무릎이다 —
# 9·13탭은 진짜 굽음까지 깎아 끊김·튀어나옴이 도로 는다.
_SMOOTH = int(os.environ.get("FS_LINE_SMOOTH", 5))
# §21 지속성 · §22 경계 설명력이 잉크 가격에 실리는 무게. 1.0이면 "그 조건이
# 완전히 성립하는 획은 값을 **두 배** 해야 산다"는 뜻이다 (`_ink_mul`).
_PERSIST_W = float(os.environ.get("FS_LINE_PERSIST", 1.0))
_EXPL_W = float(os.environ.get("FS_LINE_EXPL", 1.0))
# §25 다중 원천 합치가 잉크 가격에 실리는 무게 (`_ink_mul` ④).
_SUPPORT_W = float(os.environ.get("FS_LINE_SUPPORT", 1.0))
# §26 분리 이득이 잉크 가격을 깎는 몫 (`_ink_mul` ⑤). 0.5면 "색이 못 그리는
# 경계를 온전히 맡는 획은 반값"이다 — 면제가 아니라 할인이다.
_SEP_W = float(os.environ.get("FS_LINE_SEP", 0.5))
# §23 — 획의 값을 **그 획이 부를 도형 수**로 나눈다 (0이면 한 획 = λ 하나).
_SHAPE_PRICE = float(os.environ.get("FS_LINE_SHAPE_PRICE", 1.0))
# 이상 띠의 여유 px (`_Scorer.set_band`의 core) — 원화 띠 폭에 더하는 몫.
# 0이면 띠는 정확히 `max(획 폭, 최소 도형 폭)`이고, 그보다 굵어지는 몫은
# 전부 낭비로 문다. 여유를 주면(1·2px) 굵기 초과가 그만큼 도로 공짜가 된다 —
# 실측(01·07) 여유 2px에서 폭 초과 획이 7.0% → 5.7%로 겨우 줄고 rmse가
# 22.4 → 21.9인데, 0에서는 2.4% · 20.7이다. 최소 도형 폭이 바닥이라 "게임이
# 못 내는 가늘기"는 여기서 이미 면제돼 있다
_CORE_SLACK = float(os.environ.get("FS_LINE_CORE", 0.0))


def _core_band(shape: tuple[int, int], pp: np.ndarray,
               wprof: np.ndarray | None, wmed: float,
               min_w: float) -> np.ndarray:
    """획의 **이상 띠** — "이 자리에서 이 획이 얼마나 굵은가".

    사람 획은 한 획 안에서도 굵기가 변하고(테이퍼), 원화 띠도 다른 선과 만나는
    자리에서 굵어진다. 두께를 한 수(폭 중앙값)로 누르면 가는 쪽에서는 띠가
    남아돌고(부푸는 것이 공짜) 굵은 쪽에서는 제 몸통이 띠 밖으로 나간다
    (제대로 그은 것이 벌점).

    그래서 띠를 **폭 프로파일로** 짓는다: 마디마다 그 자리 원화 띠 폭으로
    두께를 준다. 자와 클립은 후보 순위의 목표 프로파일(`stroke._prof_pen`의
    `wt`)과 같다 — 배치가 맞추려는 폭과 채점이 재는 폭이 같은 자여야 한다.
    """
    from .stroke import _PROF_CAP

    cm = np.zeros(shape, np.uint8)
    poly = np.stack([pp[:, 1], pp[:, 0]], axis=1)
    if wprof is None or len(wprof) != len(pp) or len(pp) < 3:
        cv2.polylines(cm, [poly], False, 1,
                      max(1, int(round(wmed + _CORE_SLACK))))
        return cm.astype(bool)
    w = np.clip(np.asarray(wprof, np.float64), min_w, _PROF_CAP * wmed)
    if len(w) >= 5:                        # dt 표본의 ±1px 톱니를 편다
        k = np.ones(5) / 5.0
        w = np.convolve(np.pad(w, 2, mode="edge"), k, "valid")
    t = np.maximum(1, np.rint(w + _CORE_SLACK).astype(int))
    segs: dict[int, list] = {}
    for i in range(len(poly) - 1):
        segs.setdefault(int(max(t[i], t[i + 1])), []).append(poly[i:i + 2])
    for tt, lst in segs.items():
        cv2.polylines(cm, lst, False, 1, tt)
    return cm.astype(bool)


@dataclass
class Reconstruction:
    """엔진 한 번의 결과 — 획 목록과 그 판단의 자취."""

    strokes: list[LogicalStroke] = field(default_factory=list)
    dropped: list[LogicalStroke] = field(default_factory=list)
    fat_fills: int = 0
    stats: dict = field(default_factory=dict)

    def report(self, pol) -> dict:
        """report·debug가 읽는 요약 — 역할별 수·후보 종류·정책 자취."""
        def tally(items, key):
            out: dict = {}
            for it in items:
                v = key(it)
                if v:
                    out[v] = out.get(v, 0) + 1
            return dict(sorted(out.items()))

        n = [s.shapes for s in self.strokes if s.shapes]
        drawn = len(n)
        return {
            "policy": pol.name,
            "roles": tally(self.strokes, lambda s: s.role),
            "roles_dropped": tally(self.dropped, lambda s: s.role),
            "dropped": tally(self.dropped + self.strokes, lambda s: s.dropped),
            "candidate_kinds": tally(self.strokes, lambda s: s.kind),
            "strokes_drawn": drawn,
            "stroke_shapes": int(sum(n)),
            "shapes_per_stroke": round(float(np.mean(n)), 3) if n else 0.0,
            "one_shape_ratio": round(float(np.mean([v == 1 for v in n])), 4) if n else 0.0,
            "two_or_less_ratio": round(float(np.mean([v <= 2 for v in n])), 4) if n else 0.0,
            "seam_shapes": int(sum(s.seams for s in self.strokes)),
            "extended": int(sum(s.grown for s in self.strokes)),
            "fragmented": int(sum(1 for s in self.strokes
                                  if s.cand.get("breaks", 0))),
            "fat_fills": self.fat_fills,
            **{k: v for k, v in self.stats.items() if not k.startswith("_")},
        }

    def per_stroke(self) -> list[dict]:
        """획마다 한 줄 — 디버그 겹그림·회귀 계측이 읽는다."""
        out = []
        for s in self.strokes + self.dropped:
            out.append({
                "sid": s.sid, "role": s.role, "drawn": bool(s.shapes),
                "dropped": s.dropped, "kind": s.kind, "shapes": s.shapes,
                "seams": s.seams, "grown": s.grown, "members": s.members,
                "len": round(float(s.ev.length), 1),
                "width": round(float(s.width), 2),
                "ev": s.ev.as_dict(), "cand": s.cand,
                "path": [[int(round(p[0] + s.roi[1])), int(round(p[1] + s.roi[0]))]
                         for p in s.path[::max(1, len(s.path) // 32)]],
            })
        return out


def _ink_map(layers, cat: Catalog, upp: float, w: int, h: int) -> np.ndarray:
    from .geometry import _ink_cover
    return (_ink_cover(layers, cat, upp, w, h) if layers
            else np.zeros((h, w), bool))


def _add_ink(ink: np.ndarray, layers, cat: Catalog, upp: float,
             w: int, h: int) -> None:
    """놓은 도형을 잉크 지도에 더한다 — 제 bbox 창 안에서만 (전장 재래스터 금지)."""
    from .geometry import _mask_px, _poly_px

    for lay in layers:
        polys = _poly_px(cat, lay, upp, w, h)
        x0 = max(0, min(int(np.floor(p[:, 0].min())) for p in polys) - 1)
        y0 = max(0, min(int(np.floor(p[:, 1].min())) for p in polys) - 1)
        x1 = min(w, max(int(np.ceil(p[:, 0].max())) for p in polys) + 2)
        y1 = min(h, max(int(np.ceil(p[:, 1].max())) for p in polys) + 2)
        if x0 >= x1 or y0 >= y1:
            continue
        ink[y0:y1, x0:x1] |= _mask_px(cat, lay, upp, w, h, (x0, y0, x1, y1))


def _smooth_kernel() -> np.ndarray:
    n = max(3, _SMOOTH | 1)
    k = np.ones(1, np.float64)
    for _ in range(n - 1):
        k = np.convolve(k, np.ones(2, np.float64))
    return k / k.sum()


def build_strokes(plan: LayerPlan, cel: CelArt, maps: EvidenceMaps,
                  cat: Catalog, upp: float, sids, log, pol) -> Reconstruction:
    """성분마다 뼈대를 그래프로 올려 **논리 획**을 짓는다.

    노선 무관이다 — 정책은 마지막의 "무엇을 그릴까"에서만 본다. 뚱뚱 덩어리
    채움(눈동자·장식)은 여기서 `plan`에 바로 놓는다: 그 자리를 지나는 뼈대
    조각은 획이 아니라 채움이 대신 그린 것이라, 획 분류보다 먼저 서야 한다.
    """
    w, h = cel.size
    lm = cel.line_mask
    min_w = 2.0 * _min_span(upp)
    base = float(min(w, h))
    lab_img = cv2.cvtColor(cel.src_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    bg_all = cel.labels < 0
    rr = max(1, int(round(2.0 * _min_span(upp))))
    near_all = cv2.dilate(lm.astype(np.uint8), cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * rr + 1, 2 * rr + 1))).astype(bool)

    rec = Reconstruction()
    fat_cands: list = []
    raw: list[LogicalStroke] = []
    ncc, cc, cstats, _ = cv2.connectedComponentsWithStats(
        lm.astype(np.uint8), connectivity=8)
    order = np.argsort(-cstats[1:, cv2.CC_STAT_AREA]) + 1
    ker = _smooth_kernel()
    for ci in order:
        if cstats[ci, cv2.CC_STAT_AREA] < 6:
            break
        x0 = max(0, int(cstats[ci, cv2.CC_STAT_LEFT]) - 4)
        y0 = max(0, int(cstats[ci, cv2.CC_STAT_TOP]) - 4)
        x1 = min(w, x0 + int(cstats[ci, cv2.CC_STAT_WIDTH]) + 8)
        y1 = min(h, y0 + int(cstats[ci, cv2.CC_STAT_HEIGHT]) + 8)
        m = cc[y0:y1, x0:x1] == ci
        dt = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 3)
        res_w = 2.0 * float(np.median(dt[m & (dt > 0)])) if (m & (dt > 0)).any() else 2.0
        # 선은 타이트해야 한다 — 조밀한 선망 위에서 뚱뚱한 호가 다른 선 픽셀을
        # 쓸어 담아 이득을 내는 것을 낭비 벌점이 막는다. 최소 도형이 띠보다
        # 굵으면 넘치는 것이 불가피하므로 그 몫만큼 벌점을 깎는다
        pen_w = _PEN_LINE * min(1.0, res_w / max(min_w, 1e-6))
        sc = _Scorer(cat, upp, w, h, (x0, y0, x1, y1), m, np.zeros_like(m),
                     bg_all[y0:y1, x0:x1], pen_waste=pen_w,
                     val=maps.value[y0:y1, x0:x1],
                     soft=near_all[y0:y1, x0:x1],
                     retrace=_RETRACE, pen_far=_PEN_FAR)
        # 뚱뚱 덩어리 후보 — 반폭이 최소 도형 폭을 넘는 **컴팩트한** 덩어리는
        # 획이 아니라 채운 도형이다. 굵어도 길쭉하면 병합 띠·교차 뭉치라
        # 획이 맡을 자리다
        fat = m & (dt > min_w)
        if fat.any():
            nfc, fcc = cv2.connectedComponents(fat.astype(np.uint8), 8)
            for fi in range(1, nfc):
                blob = fcc == fi
                ys, xs = np.nonzero(blob)
                if len(ys) < _FAT_MIN_AREA:
                    continue
                ln = float(_thin(blob).sum())
                if ln > 1 and (len(ys) / ln) / ln <= _STROKE_SLIM:
                    continue
                fat_cands.append((sc, ys, xs))
        skel = _prune_spurs(_thin(m), max(3.0, 1.2 * res_w))
        src = cel.src_rgb[y0:y1, x0:x1]
        comp: list[LogicalStroke] = []
        raw_paths = _paths(skel)
        # 접합점 차수 — **이어긋기 전** 원 가닥으로 센다. 이은 뒤에 세면 우리가
        # 합친 만큼 차수가 줄어 "원래 몇 갈래였나"를 못 읽는다
        jdeg = junction_degrees(raw_paths)
        for path, hj, tj in raw_paths:
            widths = 2.0 * _dt_along(dt, path)
            p = path.round().astype(int)
            cols = src[p[:, 0].clip(0, src.shape[0] - 1),
                       p[:, 1].clip(0, src.shape[1] - 1)]
            comp.append(LogicalStroke(
                sid=-1, path=path, n_raw=len(path),
                width=float(np.median(widths)) if len(widths) else 2.0,
                widths=widths,
                color=tuple(int(v) for v in np.median(cols, axis=0)),
                comp=int(ci), roi=(x0, y0, x1, y1), head_j=hj, tail_j=tj,
                sc=sc, dt=dt, jdeg=jdeg))
        # 이어긋기는 **성분 안에서** — 접합점 id가 성분 지역 번호다
        raw.extend(continue_strokes(comp))

    # 평활 — 계단(±1px 고주파)을 편다. 길이는 **평활 전** 표본 수로 재 둔다.
    # 의도된 꺾임에서는 평활을 물린다 (`skeleton.smooth_path`)
    for s in raw:
        if len(s.path) < _SMOOTH + 2:
            continue
        s.path = smooth_path(s.path, ker)
    # 의도 — **평활 뒤의 경로**에서 각을 읽는다 (`intent` 문서). 평활이
    # 지킨 각이 그대로 남아 있고, 배치가 맞추는 것도 이 경로라 "지킨 각"과
    # "끊는 각"이 같은 각이 된다. 표본마다 한 수라 쪼갤 때 함께 잘린다
    for s in raw:
        s.intent = I.build(s.path)
    # 증거 — 지도에서 읽고, 이웃 통계(반복성·평행 밀도)는 전체를 모은 뒤에
    for s in raw:
        s.ev = sample(s.path, s.width, s.widths, maps, cel.labels,
                      s.roi[0], s.roi[1], s.head_j, s.tail_j, lab_img,
                      j_deg=s.jdeg)
    fill_neighborhood([s.ev for s in raw], [s.path for s in raw],
                      [s.width for s in raw],
                      [(s.roi[0], s.roi[1]) for s in raw])
    classify(raw, _LINE_FRAG_REL * base, max(4.0, _ISO_LEN_REL * base), min_w)

    # 뚱뚱 덩어리 채움 — 획 분류보다 먼저 놓는다
    filled = np.zeros((h, w), bool)
    cap_area = (_FAT_MAX_MUL * min_w) ** 2
    for sc, ys, xs in fat_cands:
        if len(ys) > cap_area:
            continue
        got = _fill_fat(plan, sc, cel, upp, ys, xs,
                        next(sids) if sids else -1, 2)
        if got:
            filled[np.clip(ys + sc.roi[1], 0, h - 1),
                   np.clip(xs + sc.roi[0], 0, w - 1)] = True
            rec.fat_fills += got

    # 정책이 무엇을 그릴지 고른다 — 노선이 갈리는 첫 자리다
    tex_keep = texture_representatives(raw) if pol.texture_simplify else None
    wcap = max(_LINE_W_REL * base, min_w)
    for s in raw:
        p = s.path.round().astype(int)
        gy = np.clip(p[:, 0] + s.roi[1], 0, h - 1)
        gx = np.clip(p[:, 1] + s.roi[0], 0, w - 1)
        if rec.fat_fills and float(filled[gy, gx].mean()) > 0.6:
            s.dropped = "fat_fill"          # 채움이 대신 그렸다
        elif s.role == NOISE:
            s.dropped = "fragment"
        elif tex_keep is not None and s.role == TEXTURE and id(s) not in tex_keep:
            s.dropped = "texture"
        elif not pol.draws(s.role):
            s.dropped = "role:" + s.role
        if s.dropped:
            rec.dropped.append(s)
            continue
        # 폭 정책 — 병합 띠·교차 뭉치의 굵기를 그대로 안 긋는다. 폭이 경로
        # 내내 일관(변동계수 < _WCAP_CV)하고 충분히 길면 의도된 굵은 선
        # (굵은 테두리 그림체·머리핀)으로 보고 그대로 둔다
        if s.width > wcap:
            med = float(np.median(s.widths)) if len(s.widths) else 0.0
            consistent = (s.n_raw >= 4.0 * s.width and med > 1e-6
                          and float(np.std(s.widths)) / med < _WCAP_CV)
            if not consistent:
                s.width = wcap
        s.sid = next(sids) if sids else -1
        rec.strokes.append(s)
    # 컷 순서 — 역할 등급이 먼저고 그 안에서 긴 것이 먼저다. 실루엣 윤곽과
    # 고립 특징이 앞이라 예산 컷이 그 둘을 안 문다
    rec.strokes.sort(key=lambda s: (s.rank, -s.ev.length))
    n_frag = sum(1 for s in rec.dropped if s.dropped == "fragment")
    n_tex = sum(1 for s in rec.dropped if s.dropped == "texture")
    rec.stats.update({
        "strokes_found": len(rec.strokes),
        "fragments": n_frag,
        "texture_dropped": n_tex,
        "iso_kept": sum(1 for s in rec.strokes if s.role == FEATURE),
        "joined_edges": int(sum(s.members - 1 for s in raw)),
        "detail_evidence": bool(maps.has_detail),
        "detail_only_strokes": sum(1 for s in rec.strokes
                                   if s.ev.detail_only >= 0.5),
        # §25 — 원화 해상도 판이 확인해 준 획의 몫 (원천이 하나면 0에 붙는다)
        "native_evidence": maps.native is not None,
        "sr_only_strokes": sum(1 for s in rec.strokes
                               if s.ev.support < _SUP_OK),
        "support_med": (round(float(np.median([s.ev.support
                                               for s in rec.strokes])), 3)
                        if rec.strokes else 1.0),
    })
    if rec.fat_fills:
        log(f"  덩어리 채움 {rec.fat_fills}장 (선으로 못 긋는 폭 — 모멘트 타원)")
    if n_frag or n_tex:
        log(f"  선 파편 {n_frag}개 생략"
            + (f" · 무늬 {n_tex}개 단순화" if n_tex else "")
            + f" (고립 특징 {rec.stats['iso_kept']}개 보호) — "
              f"유의미 획 {len(rec.strokes)}개")
    return rec


def _ink_mul(s, pol, base: float) -> float:
    """이 획의 잉크 가격 **배수** — 사람이 생략할 선일수록 비싸다.

    지우는 것이 아니라 비싸게 만드는 것이 요점이다. 무엇을 그릴지는 이미 λ가
    답하고 있고(`price`), 여기서 하는 일은 그 저울에 **사람이 실제로 쓰는
    이유들**을 얹는 것뿐이다. 값이 크면 이 조건들이 다 성립해도 그린다.
    항은 다섯이고 넷은 값을 올리며 마지막 하나만 깎는다 (§21~§26).

    ① **지속성** (`ev.persist`, §21) — 눈을 가늘게 뜨면 사라지는 선인가.
       머리칼 열 가닥·옷 주름 반복처럼 모아 보면 한 덩어리로 읽히는 선을
       사람은 대표 몇 가닥으로 줄인다. 사라지는 정도만큼 값을 더 받는다 —
       그 안에서 살아남는 가닥이 곧 "대표"다 (몇 개를 남길지 세지 않아도
       된다: 값이 센 가닥부터 남는다).

    ② **경계 설명력** (§22, cel 노선만) — 이 선을 안 그어도 **색이 그 자리를
       설명하나**. 양옆 색차가 역할 판정의 색 문턱(`_DE`)을 넘고 선화 신뢰도가
       그 문턱(`_CONF`)에 못 미치면, 그 자리의 경계는 색면이 이미 그린다.
       사람도 색이 또렷이 갈리는 자리에는 선을 겹쳐 긋지 않는다. 반대로 양옆
       색이 같은데 선 증거가 세면 이 배수가 1이라 그대로 지켜진다 — 그 선은
       색으로는 절대 안 생기는 구조다. line 노선에는 안 건다 (`fill_below`):
       거기는 받쳐 줄 색면이 없어 선이 빠지면 그 자리가 통째로 빈다.

    ④ **원천이 하나인가** (§25) — 아래 코드 주석. ⑤ **분리 이득** (§26) —
       유일하게 값을 **깎는** 항이다.

    문턱을 새로 만들지 않았다 — `_DE`·`_CONF`는 역할 판정이 "색으로 설명되는
    경계"와 "선화가 확실히 본 선"을 가르는 데 이미 쓰는 그 자다.

    **효과는 작다** (표준 11장: 논리 획 531 → 512 · 선 도형 1,185 → 1,168).
    잉크 λ가 거의 안 물기 때문이다: 값 지도는 그림 전체의 중앙을 1로 잡는데
    획은 정의상 값이 가장 높은 자리라(`evidence.imp_rel` 문서), 획의 값은
    대개 λ의 여러 배다 — 배수를 둘로 키워도 문턱을 넘는 획이 판당 열 몇 개
    는다. 이 세트에서 선 도형 수를 정하는 것은 가격이 아니라 **논리 획의
    수**이고, 그것은 파편 판정(`graph.classify`)이 정한다.
    """
    m = 1.0
    if _SHAPE_PRICE and _SPAN_REL > 0.0:
        # ③ **장수 자체가 비용이다** (§11의 선 판). 한 획이 실제로 무는 것은
        # 도형 1.7장이고(실측 중앙), 긴 획은 열 장도 넘는다 — 획 하나당 λ
        # 하나로 물면 그만큼이 공짜다. 길이에 비례해 값을 받으면 "이 획이
        # 부를 도형 수만큼 벌어야 산다"가 되어, 채움 쪽이 이미 쓰는 저울과
        # 같은 저울이 된다. 자는 정책이 도형 상한을 정할 때 쓰는 그 자다
        # (`policy._SPAN_REL` — 가장 곱게 그은 획의 도형당 길이).
        m *= max(1.0, float(s.ev.length) / max(_SPAN_REL * base, 1.0)) ** _SHAPE_PRICE
    if _PERSIST_W:
        m *= 1.0 + _PERSIST_W * (1.0 - min(1.0, max(0.0, s.ev.persist)))
    if _EXPL_W and pol.fill_below:
        conf = max(s.ev.basic, _DETAIL_W * s.ev.detail)
        expl = (min(1.0, max(0.0, s.ev.side_de / _DE - 1.0))
                * min(1.0, max(0.0, 1.0 - conf / _CONF)))
        m *= 1.0 + _EXPL_W * expl
    if _SUPPORT_W:
        # ④ **원천이 하나인가** (§25) — SR 중간본의 선화 판 하나만 이 선을
        # 봤다면 그것은 모델이 지어낸 것일 수 있다. 지우지 않고 **비싸게**
        # 만든다: 원화 해상도 판이 확인해 주거나(`support`), 양옆 색이 실제로
        # 갈리거나(`side_de`), 실루엣을 타고 있으면(`sil`) 지지가 선다 —
        # 셋 중 가장 센 것을 쓴다. 지지가 하나도 없으면 값을 두 배로 받는다.
        #
        # 원화 판이 없으면 `support`가 1이라 이 항은 **무동작**이다
        # (SR을 안 태웠거나 모델이 없는 경우 — 폴백 불변).
        corrob = max(s.ev.support,
                     min(1.0, s.ev.side_de / _DE),
                     min(1.0, s.ev.sil / max(_SIL, 1e-6)))
        m *= 1.0 + _SUPPORT_W * min(1.0, max(0.0, 1.0 - corrob))
    if _SEP_W and pol.fill_below:
        # ⑤ **이 획이 아니면 그 경계가 없다** (§26, cel 노선만). 잠정 색 영역
        # 위에서 읽은 양옆 라벨이 갈리는데(`bnd`) 색은 거의 같으면(`side_de`가
        # 역할 문턱 아래), 그 자리는 색면이 원리적으로 못 그린다 — 채움을 아무리
        # 잘해도 두 면이 같은 색이라 경계가 안 보인다. 그런 획은 값을 깎아 준다.
        #
        # §22(경계 설명력)의 짝이다: 그쪽은 "색이 이미 그렸으니 비싸다"이고
        # 이쪽은 "색이 못 그리니 싸다"라 두 항이 겹치지 않는다 (`side_de`가
        # 높으면 이 항이 0, 낮으면 §22가 0).
        #
        # 잠정 영역을 안 주면(line 노선) `bnd`가 0이라 **무동작**이다.
        need = s.ev.bnd * min(1.0, max(0.0, 1.0 - s.ev.side_de / _DE))
        m *= 1.0 - _SEP_W * need
    return m


def place_strokes(plan: LayerPlan, rec: Reconstruction, cel: CelArt,
                  cat: Catalog, upp: float, budget: int, forms: tuple,
                  pol, log, price: float = 0.0, progress=None) -> int:
    """후보를 지어 정책이 고르게 한다 — 확정 장수를 돌려준다.

    확장·이음 보수는 배치가 다 끝난 뒤다 (§ seam 생성 전에 전체 최적화):
    잔틈은 새 도형이 아니라 **있는 도형을 한 칸 늘려** 메우는 것이 먼저고,
    그래도 남는 틈만 도형을 쓴다.
    """
    w, h = cel.size
    base = float(min(w, h))
    min_w = 2.0 * _min_span(upp)
    rr = max(1, int(round(2.0 * _min_span(upp))))
    near_all = cv2.dilate(cel.line_mask.astype(np.uint8),
                          cv2.getStructuringElement(
                              cv2.MORPH_ELLIPSE, (2 * rr + 1, 2 * rr + 1))
                          ).astype(bool)
    bg_all = cel.labels < 0
    # 잉크가 나가도 되는 자리 — 선 밴드 안. line 노선은 그 위에 배경까지 뺀다
    # (덮어 줄 면이 없어 흰 바탕에 자국이 그대로 남는다)
    allow = near_all if pol.fill_below else (near_all & ~bg_all)
    n = n_cheap = n_joint = 0
    placed: list = []
    owners: list = []
    total = max(1, len(rec.strokes))
    # 지금까지 그은 잉크 — **덮임 판정의 기준**이다 (`candidates.evaluate`).
    # 이미 놓인 덩어리 채움도 여기 실린다. 갱신은 놓은 도형의 bbox 안에서만
    ink_so_far = _ink_map(plan.layers, cat, upp, w, h)
    for k, s in enumerate(rec.strokes):
        if progress and (k & 15) == 0:
            progress(k / total, f"획 {k + 1}/{total}")
        if n >= budget:
            for rest in rec.strokes[k:]:
                rest.dropped = "budget"
            rec.stats["skipped_strokes"] = len(rec.strokes) - k
            log(f"  경고: 선 예산 소진 — 남은 획 {len(rec.strokes) - k}개 생략 "
                f"(덜 보이는 순)")
            break
        sc = s.sc
        # 가격 — 획 하나가 단위다. 사람은 획을 한 번에 긋거나 아예 안 긋는다.
        # 정책이 가격을 무는 역할만 문다 (실루엣·고립 특징은 면제 — 길이로
        # 값을 매기면 구조적으로 지는데, 빠지면 그 자리 경계가 통째로 없어진다)
        if price and pol.prices(s.role) \
                and _path_worth(sc, s.path, s.width) < price * _ink_mul(s, pol, base):
            s.dropped = "price"
            n_cheap += 1
            continue
        # 제 경로 밴드 — 이득·retrace가 이 띠 안에서만 선다. 한정이 없으면
        # 하강이 교차하는 **다른 선** 위로 마디를 늘려도 이득을 얻는다.
        # 여유는 양자화 몫(정책 배수 × 최소 도형 폭)
        pp = s.path.round().astype(np.int32)
        poly = [np.stack([pp[:, 1], pp[:, 0]], axis=1)]
        bm = np.zeros(sc.residual.shape, np.uint8)
        cv2.polylines(bm, poly, False, 1,
                      max(1, int(round(max(s.width, min_w)
                                       + pol.band_slack * min_w))))
        band = bm.astype(bool)
        # 제 획의 **이상 띠** — 원화가 그은 폭(+ 양자화 여유 한 겹). 밴드가
        # "어디까지 나가도 되나"라면 이쪽은 "얼마나 굵은가"다. 이 띠 밖의
        # 물림·재추적은 공짜가 아니다 (`scoring._Scorer.set_band`)
        # 이상 띠의 바닥은 **어휘가 낼 수 있는 가장 가는 폭**이다 — 막대 폭을
        # 바닥으로 쓰면 원화보다 굵은 띠가 공짜가 된다
        # (`vocabulary.min_stroke_width_px`)
        wfloor = min_stroke_width_px(cat, upp)
        core = _core_band(sc.residual.shape, pp,
                          s.widths if len(s.widths) == len(s.path) else None,
                          max(s.width, wfloor), wfloor)
        # 한 획의 도형 상한은 **길이가 정한다** (`policy.shapes_for`) — 상수
        # 상한이 긴 획을 깨던 자리다 (끊김이 길이 400px 위에서 73%)
        cands = C.build(sc, s.dt, s.path, s.width, s.color, s.sid, forms, cat,
                        upp, w, h, allow, pol, band,
                        max(1, min(budget - n,
                                   pol.shapes_for(s.ev.length, base))),
                        ink_so_far,
                        core=core,
                        wprof=s.widths if len(s.widths) == len(s.path) else None,
                        it=s.intent if (s.intent is not None
                                        and len(s.intent.corner) == len(s.path))
                        else None)
        best = C.pick(cands, pol)
        if best is None:
            s.dropped = "nofit"
            continue
        s.kind, s.shapes, s.cand = best.kind, best.n, best.summary()
        if not best.layers:
            # 이미 다른 획이 그 자리를 다 그었다 — 한 장도 안 쓴다
            s.dropped = "covered"
            continue
        lo = len(plan.layers)
        sc.set_band(band, core)
        # 사슬 이음 정리 — 굳히기 **전에** 마디끼리 맞춘다 (`chain` 문서).
        # 굳힌 뒤에 밀면 잔여 회계가 이미 그 자리를 지운 뒤라 점수가 거짓말을
        # 한다. 여기서는 획의 도형이 하나도 안 굳어 있어 마디마다 같은 판을 본다
        n_joint += chain.polish(best.layers, sc, cat, upp,
                                np.stack([pp[:, 0] + s.roi[1],
                                          pp[:, 1] + s.roi[0]], axis=1),
                                max(s.width, min_w), w, h)
        n += C.commit(plan, sc, best)
        sc.set_band(None)
        _add_ink(ink_so_far, best.layers, cat, upp, w, h)
        placed.append((s.sid, s.path, s.width, s.color, sc, lo,
                       len(plan.layers)))
        owners.append(s)
    rec.stats["cheap_strokes"] = n_cheap
    rec.stats["joint_moves"] = n_joint
    if n_cheap:
        log(f"  가격 미달 획 {n_cheap}개 생략 (값 < λ) — "
            f"그은 획 {len(placed)}개")
    if pol.seam_repair and placed and n < budget:
        n += _patch_seams(plan, cat, upp, (w, h), placed, budget - n, log,
                          rec.stats, near_all & ~bg_all, forms=forms,
                          owners=owners)
    # 겹침 병합 — 같은 방향으로 겹친 짧은 막대를 하나로 (배치 뒤)
    n -= merge_costrokes(plan, cat, upp, (w, h), near_all & ~bg_all,
                         cel.src_rgb, lo=0, log=log, st=rec.stats)
    return n
