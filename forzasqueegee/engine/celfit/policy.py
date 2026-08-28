"""노선 정책 — 두 노선이 **어디서 갈리는가**를 한 자리에 모은다.

선 재구성 자체(증거·그래프·역할·후보 생성)는 노선을 안 본다. 같은 입력이면
line과 cel이 **같은 논리 획 그래프**를 낸다. 갈리는 것은 마지막 두 가지뿐이다:

- **무엇을 그릴까** — 정책이 역할·가격으로 고른다.
- **어느 후보를 쓸까** — 정책이 허용 오차를 준다.

그래서 두 노선의 결과 차이는 반드시 정책 한 칸으로 설명된다. 획마다 어느
칸이 갈랐는지를 `dropped`·`kind`에 적어 두므로 report에서 되짚을 수 있다.

- `line` — 도안이 선 하나뿐이다. 배경 스필을 세게 막고, 덮어 그리기(carve)를
  안 쓰며(덮을 색이 없다), 획 연속성과 그 자체로의 완성도를 앞세운다. 나중에
  면이 가려 줄 것을 전제한 오차를 허용하지 않는다.
- `cel` — 획 아래를 색면이 받친다. **면이 받치는 것은 스필이지 틈이 아니다** —
  선은 모든 면 위에 마지막으로 얹히므로 덮임·끊김·획 도형 문법은 line 노선과
  같은 자를 쓰고, 갈리는 것은 스필 허용치·밴드 여유·덮어 그리기다. 잉크
  가격(λ)으로 획과 채움이 같은 예산을 겨룬다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from .graph import (COLOR_BOUNDARY, FEATURE, INTERNAL_CONTOUR, NOISE,
                    SILHOUETTE, STRUCTURE, TEXTURE)


@dataclass(frozen=True)
class RoutePolicy:
    """한 노선이 선 재구성 결과를 어떻게 쓸지."""

    name: str
    # ── 무엇을 그릴까 ────────────────────────────────────────────────
    draw_roles: tuple[str, ...]      # 그리는 역할
    texture_simplify: bool           # 무늬를 대표 외곽선만 남길까
    price_roles: tuple[str, ...]     # 잉크 가격을 무는 역할 (나머지는 면제)
    # ── 어느 후보를 쓸까 ─────────────────────────────────────────────
    cover_min: float                 # 획 경로 중 덮여야 하는 최소 비율
    stray_max: float                 # 획 잉크 중 허용 밴드 밖 몫의 상한
    breaks_max: int                  # 허용하는 끊김 수
    band_slack: float                # 밴드 여유 배수 (최소 도형 폭의 배)
    carve: bool                      # 덮어서 그리기 허용
    fill_below: bool                 # 획 아래를 면이 받치는가
    seam_repair: bool                # 이음 보수를 돌리나
    max_shapes: int = 8              # 한 획에 쓸 도형 수 상한
    # 후보 비교에서 도형 수보다 기하 오차를 앞세울 여지 (0 = 도형 수 우선)
    err_weight: float = 0.0
    notes: str = ""
    extra: dict = field(default_factory=dict)

    def draws(self, role: str) -> bool:
        return role in self.draw_roles

    def shapes_for(self, length_px: float, base: float) -> int:
        """이 획에 허용할 도형 수 — **길이가 정한다** (`max_shapes`는 바닥).

        상한이 상수(8)이던 동안 그 자리가 긴 획을 깨는 유일한 원인이었다.
        실측(표준 01, 그은 획 545개): 끊김이 남는 획의 비율이 길이 100px
        아래 1%, 100~200px 8%, 200~400px 29%, **400px 위 73%**로 단조로 뛴다.
        길수록 어려워서가 아니다 — `_fit_segments`가 마디 수를 상한에 맞추려고
        허용오차를 키우기 때문이다(`stroke._fit_segments`): 900px 경로가 여덟
        토막이 되면 각 토막이 제 호를 못 따라가고 그 사이가 벌어진다. 그래서
        긴 획일수록 **도형은 적게 쓰고 이음 보수는 많이 무는** 결과가 됐다 —
        1,039px 실루엣 하나가 배치 8장에 이음 보수 25장(덮임 0.41 · 끊김 3)이다.
        길이당 도형은 그 반대여야 한다: 깨끗하게 그은 획(끊김 0)의 도형당
        길이가 100px 아래에서 27px인데 400px 위에서는 135px이었다.

        상한을 걷는 것이 아니라 **길이에 비례시킨다**. 자는 **가장 곱게 그은
        획의 도형당 길이**다: 끊김 0인 획의 도형당 길이가 100px 아래 구간에서
        27px(@ 짧은 변 1200 = 0.0225)이고, 그 구간이 맞춤이 가장 정확한
        자리다. 천장은 목표가 아니라 "우리가 가장 곱게 그을 때만큼은 열어
        둔다"는 뜻이라 가장 고운 쪽으로 잡는 것이 맞다 — 몇 장을 실제로 쓸지는
        후보 경쟁이 정한다 (`candidates.pick`: 같은 단이면 도형 수가 적은 쪽이
        이기고, 그 도형 수에는 끊김이 부를 이음 보수까지 들어 있다).

        하드 상한 `_SHAPE_HARD`는 그래도 한 획이 도안을 먹지 못하게 하는
        뚜껑이다. 실측(01) 1,039px 실루엣이 21장을 썼으니 24는 그 위다.

        **실측(표준 10장, line 노선)**: 상한이 상수 8일 때 → 길이 비례일 때
        (끊김 값을 길이로 재는 짝과 함께 — `candidates._SEAM_PER_BREAK`).
        이음 보수 1,062 → **748**(−30%) · 끊긴 획 230 → **157**(−32%) ·
        한 획 안 꺾임 29.8° → 28.5° · 이음각 25.1° → 23.6° · 계열 섞임
        .323 → .298 · 선 커버리지 .9694 → .9710 · 밴드 밖 스필 .0010 → .0006 ·
        실루엣 윤곽 .9116 → .9155 · RMSE 25.7 → 25.5. 값은 총 도형 +0.6%다.
        길이 400px 위 획만 보면 배치 543 → 838장인데 이음 보수가
        307 → 72장이라 합은 850 → 910장이고, 끊김이 남는 몫이 .66 → **.24**다.
        위 1,039px 실루엣은 8장 + 이음 25장(덮임 .41) → 21장 + 이음 6장
        (덮임 .86)이 됐다.
        """
        if _SPAN_REL <= 0.0 or base <= 0.0:
            return self.max_shapes
        span = max(_SPAN_REL * base, 1.0)
        want = int(-(-float(length_px) // span))       # ceil
        return int(min(_SHAPE_HARD, max(self.max_shapes, want)))

    def prices(self, role: str) -> bool:
        return role in self.price_roles


# 그리는 역할 — 무늬·부스러기 말고는 전부 그린다. 무늬는 노선이 아니라
# `texture_simplify` 스위치가 정한다 (기본 꺼짐 — 아래 근거)
_DRAW = (SILHOUETTE, STRUCTURE, INTERNAL_CONTOUR, COLOR_BOUNDARY, FEATURE,
         TEXTURE)
# 가격을 무는 역할 — 실루엣 윤곽과 고립 특징은 길이로 값을 매기면 구조적으로
# 진다 (짧고 가늘기 때문이다). 그런데 그 둘이 빠지면 그 자리의 경계가 통째로
# 없어진다. 사람이 아끼는 것은 다발 가닥·내부 톤 경계 쪽이라 가격은 거기서 받는다
_PRICED = (STRUCTURE, INTERNAL_CONTOUR, COLOR_BOUNDARY, TEXTURE, NOISE)

# **한 획에 쓸 도형 수의 길이 비례 몫** — 짧은 변 대비 "도형 하나가 맡는
# 경로 길이". 0이면 상한이 상수 `max_shapes` 그대로다 (종전 동작).
# 근거는 `RoutePolicy.shapes_for` 문서.
_SPAN_REL = float(os.environ.get("FS_STROKE_SPAN", 0.0225))
# 그래도 한 획이 도안을 먹지 못하게 하는 뚜껑 — 이보다 많이 드는 것은 획이
# 아니라 면이다 (그 자리는 덩어리 채움·면 채움이 맡는다).
_SHAPE_HARD = int(os.environ.get("FS_STROKE_MAX_HARD", 24))

# 무늬 단순화 — **기본 꺼짐**이다. 무늬/구조의 구분은 의미론이고, 창 통계로는
# 복제할 수 없다는 것이 사람 라벨 10장 대조로 확인됐다 (지목된 레이스·오비
# 문양과 남겨야 할 글자·머리칼 디테일이 같은 분포였다). 여기 역할 판정은
# 갇힘·평행 밀도까지 보므로 그때보다 자가 낫지만, 켤지는 회귀 계측이 정한다.
_TEXTURE_SIMPLIFY = os.environ.get("FS_TEX_SIMPLIFY", "0") != "0"


LINE = RoutePolicy(
    name="line",
    draw_roles=_DRAW,
    texture_simplify=_TEXTURE_SIMPLIFY,
    price_roles=_PRICED,
    # 선만 있는 도안이라 오차가 그대로 보인다 — 덮임은 넉넉히, 스필은 빡빡하게
    cover_min=float(os.environ.get("FS_LINE_COVER", 0.90)),
    stray_max=float(os.environ.get("FS_LINE_STRAY", 0.10)),
    breaks_max=0,
    band_slack=1.0,
    carve=False,
    fill_below=False,
    seam_repair=True,
    max_shapes=int(os.environ.get("FS_STROKE_MAX_LINE", 8)),
    notes="독립 표시 — 배경 스필 제한·carve 금지·획 연속성 우선",
)

CEL = RoutePolicy(
    name="cel",
    draw_roles=_DRAW,
    texture_simplify=_TEXTURE_SIMPLIFY,
    price_roles=_PRICED,
    # 획 아래를 면이 받치므로 밴드 밖 스필도 면 위라면 안 보인다.
    # **덮임과 끊김은 그 자유에 안 든다** — 면이 받치는 것은 스필이지 틈이
    # 아니고, 선은 모든 면 위에 마지막으로 얹히므로 획의 틈은 line 노선과
    # 똑같이 보인다. 게다가 그 틈은 공짜가 아니다: 이음 보수가 끊김 하나당
    # 도형 1.5장을 문다 (`candidates._SEAM_PER_BREAK`). 느슨하게 두었더니
    # 셀 노선의 획 도형 중 **이음 보수가 20~39%**였다 (line 노선은 13%) —
    # 조여 보니 셀 세 장에서 이음 보수 몫 .354→.295 · .202→.144 · .293,
    # 경계 넘김 −.005~−.009, 중요도 가중 오차 −0.01~−0.46, 도형은 ±10장이다.
    cover_min=float(os.environ.get("FS_CEL_COVER", 0.90)),
    stray_max=float(os.environ.get("FS_CEL_STRAY", 0.18)),
    breaks_max=int(os.environ.get("FS_CEL_BREAKS", 0)),
    band_slack=1.4,
    carve=False,          # 선 도안 단계는 면이 아직 없다 — 덮개는 폴백에서만
    fill_below=True,
    seam_repair=True,
    max_shapes=int(os.environ.get("FS_STROKE_MAX_LINE", 8)),
    notes="면이 받친다 — overlap 허용·ink-free 고려·shared boundary 반영",
)

# 선화 모델이 없을 때의 cel 폴백 — 선·면을 함께 놓는다. 덮개(carve)가 여기서만
# 산다: 그 자리에는 면 색이 있어 굵게 긋고 도로 덮을 수 있다. 예산이 선·면
# 양쪽에 걸리므로 한 획의 도형 상한도 낮다
CEL_FALLBACK = RoutePolicy(
    **{**CEL.__dict__, "name": "cel-fallback", "carve": True, "max_shapes": 5,
       "notes": "선화 모델 없음 — 선·면 동시 배치, 덮개 허용"})

POLICIES = {"line": LINE, "cel": CEL, "cel-fallback": CEL_FALLBACK}


def get(name: str) -> RoutePolicy:
    return POLICIES.get(name, LINE)


def diff(a: RoutePolicy, b: RoutePolicy) -> dict:
    """두 정책이 갈리는 칸만 — 노선별 결과 차이를 되짚는 자리."""
    out = {}
    for k in a.__dataclass_fields__:
        if k in ("name", "notes", "extra"):
            continue
        va, vb = getattr(a, k), getattr(b, k)
        if va != vb:
            out[k] = [va, vb]
    return out
