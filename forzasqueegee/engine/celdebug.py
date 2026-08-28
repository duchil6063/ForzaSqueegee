"""cel 분해·배치의 **겹판** — 수치가 못 보여 주는 것을 눈으로 본다 (§14).

`FS_CEL_DEBUG=1`이면 도안 폴더에 겹판이 함께 나온다. 끄면 아무 일도 안 하고
산출물도 그대로다 (기본 꺼짐 — 판마다 몇 MB다).

    cel_atoms.png     초기 원자 (병합 전)
    cel_regions.png   병합 후 의미 영역 · 무늬 보호 조각은 흰 테
    cel_bounds.png    선이 받치는 경계(초록) · 병합이 지운 약한 경계(빨강)
    cel_shapes.png    바탕 도형(파랑) · 보정 도형(주황) · 의도한 스필(노랑)
    cel_residual.png  잔차 분류 (구멍·틈·경계·얼룩·새어나감) · **고쳐진 자리**(초록)
                      · 값이 안 되어 안 고친 잔차(어둡게)

색은 라벨에서 결정적으로 낸다 (난수 없음) — 같은 판을 다시 구우면 같은 그림이다.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..paths import run_file
from .catalog import Catalog
from .celart import CelArt
from .model import LayerPlan

# 잔차 분류 색 (BGR) — `celfit.residual`의 클래스 번호 순서
_RES_COLORS = {
    1: (255, 255, 255),    # hole    — 흰 구멍
    2: (0, 255, 255),      # gap     — 선과 색면 사이 틈 (노랑)
    3: (0, 160, 255),      # boundary— 경계 어긋남 (주황)
    4: (0, 0, 255),        # wrong   — 얼룩 (빨강)
    5: (255, 0, 255),      # leak    — 실루엣 밖 (자홍)
}


def _tint(labels: np.ndarray) -> np.ndarray:
    """라벨 지도 → 색판 (BGR). 색은 id의 해시라 결정적이고 이웃끼리 갈린다."""
    n = int(labels.max()) + 1 if labels.max() >= 0 else 0
    lut = np.zeros((max(n, 1), 3), np.uint8)
    if n:
        idx = np.arange(n, dtype=np.int64)
        lut[:, 0] = (idx * 61 + 40) % 216 + 20
        lut[:, 1] = (idx * 149 + 90) % 216 + 20
        lut[:, 2] = (idx * 233 + 150) % 216 + 20
    out = np.zeros(labels.shape + (3,), np.uint8)
    pos = labels >= 0
    out[pos] = lut[labels[pos]]
    return out


def _outline(mask: np.ndarray) -> np.ndarray:
    u = mask.astype(np.uint8)
    return (u & ~cv2.erode(u, np.ones((3, 3), np.uint8))).astype(bool)


def save(out_dir, cel: CelArt, plan: LayerPlan, cat: Catalog, *,
         res: dict | None = None, marks: np.ndarray | None = None,
         reg_of: list[int] | None = None, act_before: np.ndarray | None = None,
         write=None) -> None:
    """겹판 한 벌 — 없는 재료는 조용히 건너뛴다."""
    from .celfit import residual as _res

    if write is None:
        def write(path, img):
            cv2.imwrite(str(path), img)

    trace = cel.trace or {}
    atoms = trace.get("atom_labels")
    if atoms is not None:
        write(run_file(out_dir, "cel_atoms.png"), _tint(atoms))

    reg = _tint(cel.labels)
    if marks is not None and marks.any():
        reg[_outline(marks)] = (255, 255, 255)
    write(run_file(out_dir, "cel_regions.png"), reg)

    # 경계 — 선이 받치는 것과 병합이 지운 것
    lb = cel.labels
    bnd = np.zeros(lb.shape, bool)
    bnd[:, :-1] |= lb[:, :-1] != lb[:, 1:]
    bnd[:-1] |= lb[:-1] != lb[1:]
    bnd &= lb >= 0
    img = np.zeros(lb.shape + (3,), np.uint8)
    ink = cel.line_mask
    if ink is not None:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        near = cv2.dilate(ink.astype(np.uint8), k).astype(bool)
        img[bnd & near] = (0, 220, 0)          # 선이 받치는 경계
        img[bnd & ~near] = (0, 200, 220)       # 선 없는 색 경계
    else:
        img[bnd] = (0, 200, 220)
    if atoms is not None:
        ab = np.zeros(lb.shape, bool)
        ab[:, :-1] |= atoms[:, :-1] != atoms[:, 1:]
        ab[:-1] |= atoms[:-1] != atoms[1:]
        ab &= atoms >= 0
        img[ab & ~bnd] = (0, 0, 200)           # 병합이 지운 약한 경계
    write(run_file(out_dir, "cel_bounds.png"), img)

    # 도형 — 바탕 · 보정 · 의도한 스필
    if reg_of is not None:
        owner = res["owner"] if res and "owner" in res else \
            _res.owner_map(plan, cel, cat)
        img = np.zeros(lb.shape + (3,), np.uint8)
        seen: set[int] = set()
        kind = np.zeros(len(plan.layers), np.uint8)   # 0 = 그 밖, 1 = 바탕, 2 = 보정
        for i, l in enumerate(plan.layers):
            if l.label in ("hole", "fix"):
                kind[i] = 2
            elif l.label != "ink" and reg_of[i] >= 0 and reg_of[i] not in seen:
                kind[i] = 1
                seen.add(reg_of[i])
        pos = owner >= 0
        k_img = np.zeros(lb.shape, np.uint8)
        k_img[pos] = kind[owner[pos]]
        img[k_img == 1] = (220, 120, 0)        # 바탕 (파랑)
        img[k_img == 2] = (0, 140, 255)        # 보정 (주황)
        # 의도한 스필 — 제 영역 밖을 덮었는데 위(나중 면·획)가 가려 주는 자리
        own_reg = np.full(lb.shape, -1, np.int32)
        ro = np.asarray(reg_of + [-1], np.int32)
        own_reg[pos] = ro[owner[pos]]
        spill = pos & (lb >= 0) & (own_reg >= 0) & (own_reg != lb)
        img[spill] = (0, 255, 255)             # 스필 (노랑)
        write(run_file(out_dir, "cel_shapes.png"), img)

    if res is not None and "classes" in res:
        cls = res["classes"]
        img = np.zeros(cls.shape + (3,), np.uint8)
        for v, c in _RES_COLORS.items():
            img[cls == v] = c
        act = res.get("actionable")
        if act is not None:                    # 안 고치기로 한 잔차는 어둡게
            img[(cls > 0) & ~act] //= 3
        if act_before is not None:
            # **고쳐진 잔차** — 배치 직후 "고칠 값이 있다"고 꼽혔던 자리 중
            # 지금은 깨끗한 곳. §12의 사다리(기존 도형 이동·스케일·회전 →
            # 이웃 조정 → 메움·수리)가 통틀어 닫은 몫이다
            img[act_before & (cls == 0)] = (0, 255, 0)
        write(run_file(out_dir, "cel_residual.png"), img)
