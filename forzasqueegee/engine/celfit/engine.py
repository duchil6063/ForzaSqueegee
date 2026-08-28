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
from . import ablation
from . import candidates as C
from . import chain
from . import intent as I
from .evidence import (EvidenceMaps, fill_neighborhood, junction_degrees,
                       sample)
from .geometry import _min_span
from .graph import (_ISO_LEN_REL, _LINE_FRAG_REL, FEATURE, NOISE, TEXTURE,
                    LogicalStroke, classify, continue_strokes,
                    texture_representatives)
from .grammar import (_FAT_MAX_MUL, _FAT_MIN_AREA, _LINE_W_REL, _WCAP_CV,
                      _fill_fat, _patch_seams)
from .merge import merge_costrokes
from .scoring import _PEN_FAR, _PEN_LINE, _PEN_LINE_ADAPT, _RETRACE, _Scorer
from .skeleton import (_dt_along, _join_paths, _paths, _prune_spurs,
                       _thin, smooth_path)
from .stroke import _STROKE_SLIM, _path_worth
from .vocabulary import min_stroke_width_px

# 경로 평활 창 (홀수 탭) — 선 지도의 계단 현상은 지도 쪽에서 못 없앤다
# (`lineart.hysteresis` 문서). 실제로 듣는 자리는 여기뿐이고 5탭이 무릎이다 —
# 9·13탭은 진짜 굽음까지 깎아 끊김·튀어나옴이 도로 는다.
_SMOOTH = int(os.environ.get("FS_LINE_SMOOTH", 5))
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

    종전에는 획 하나에 두께가 한 수(폭 중앙값)뿐이었다. 그런데 사람 획은
    한 획 안에서도 굵기가 변하고(테이퍼), 원화 띠도 다른 선과 만나는 자리에서
    굵어진다 — 한 수로 누르면 가는 쪽에서는 띠가 남아돌고(부푸는 것이 공짜)
    굵은 쪽에서는 제 몸통이 띠 밖으로 나간다(제대로 그은 것이 벌점).

    그래서 띠를 **폭 프로파일로** 짓는다: 마디마다 그 자리 원화 띠 폭으로
    두께를 준다. 자와 클립은 후보 순위의 목표 프로파일(`stroke._prof_pen`의
    `wt`)과 같다 — 배치가 맞추려는 폭과 채점이 재는 폭이 같은 자여야 한다.
    """
    from . import ablation
    from .stroke import _PROF_CAP

    cm = np.zeros(shape, np.uint8)
    poly = np.stack([pp[:, 1], pp[:, 0]], axis=1)
    if (wprof is None or len(wprof) != len(pp) or len(pp) < 3
            or not ablation.core_profile()):
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
        pen_w = _PEN_LINE
        if _PEN_LINE_ADAPT:
            pen_w *= min(1.0, res_w / max(min_w, 1e-6))
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
        # 이어긋기 — 그래프 축을 끄면 접선 각도만 보던 옛 이음으로 돌아간다
        raw_paths = (_paths(skel) if ablation.graph()
                     else _join_paths(_paths(skel)))
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
        raw.extend(continue_strokes(comp) if ablation.graph() else comp)

    # 평활 — 계단(±1px 고주파)을 편다. 길이는 **평활 전** 표본 수로 재 둔다.
    # 각 보존 축(`ablation.corner`)이 켜져 있으면 의도된 꺾임에서만 평활을
    # 물린다 (`skeleton.smooth_path`); 꺼지면 창을 통째로 거는 옛 동작이다
    for s in raw:
        if len(s.path) < _SMOOTH + 2:
            continue
        if ablation.corner():
            s.path = smooth_path(s.path, ker)
        else:
            mid = np.stack([np.convolve(s.path[:, 0], ker, "valid"),
                            np.convolve(s.path[:, 1], ker, "valid")], axis=1)
            s.path = np.concatenate([s.path[:1], mid, s.path[-1:]], axis=0)
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
    classify(raw, _LINE_FRAG_REL * base, max(4.0, _ISO_LEN_REL * base), min_w,
             simple=not ablation.graph())

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
        "ablation": ablation.names(),
    })
    if rec.fat_fills:
        log(f"  덩어리 채움 {rec.fat_fills}장 (선으로 못 긋는 폭 — 모멘트 타원)")
    if n_frag or n_tex:
        log(f"  선 파편 {n_frag}개 생략"
            + (f" · 무늬 {n_tex}개 단순화" if n_tex else "")
            + f" (고립 특징 {rec.stats['iso_kept']}개 보호) — "
              f"유의미 획 {len(rec.strokes)}개")
    return rec


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
                and _path_worth(sc, s.path, s.width) < price:
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
        core = None
        if ablation.width():
            # 이상 띠의 바닥은 **어휘가 낼 수 있는 가장 가는 폭**이다 —
            # 막대 폭을 바닥으로 쓰면 원화보다 굵은 띠가 공짜가 된다
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
