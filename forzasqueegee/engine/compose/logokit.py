"""로고 키트 — 내장 워터마크와 사용자 로고 이미지를 **도안으로** 갖춘다.

사람이 만든 리버리 30벌 중 24벌에 스폰서 로고·워드마크 무리가 있다
(`work/lab/humanref`, 2026-09-02). 우리 판에는 그 자리가 비어 있었고 산포
모티프가 억지로 채우고 있었다 — 그것이 "자동 생성 티"의 실제 출처다. 실제
브랜드는 안 쓴다. 로고는 두 곳에서 온다 (사용자 결정 ② 2026-09-02):

- **내장 워터마크** — `docs/logo.png`·`logo-dark.png`를 한 번 벡터화해 둔
  `catalog/kit/logo.plan.json`·`logo-dark.plan.json` (`tools/make_kit.py`).
  밝은 바탕엔 `logo`(검정 잉크), 어두운 바탕엔 `logo-dark`(흰 잉크).
- **사용자 로고 이미지** — 대화상자 슬롯(0~N장). 같은 노선으로 벡터화한다.
  안 넣으면 사용자 몫만 빠진다.

## 장수 상한 — 로고는 **오브젝트**다

사람 판의 로고는 중앙 41장·2색이고, 평가기(`work/lab/whole/ruler`)는 옆면의
작은 덩어리를 **120장 미만**이면 오브젝트(로고·엠블럼), 그 위면 그림으로
읽는다. 그래서 로고 도안은 `LOGO_LAYERS`(110)장 아래로 깎는다 — 셀 노선은
장수를 그림이 정하므로(봉인 점이 수백 장 붙는다) 시각 기여가 작은 레이어부터
버린다 (`pruneplan.prune_plan`, 이분 탐색). 로고 안의 작은 글자는 이 장수에서
못 읽히지만, 워터마크 크기(옆면 폭의 5%)에서는 어차피 마크만 읽힌다.

벡터화는 **내용 서명으로 캐시**한다 — 같은 이미지는 한 번만 굽고, 같은 이미지
두 판은 같은 도안이다 (결정성 자는 파일 해시).
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ...i18n import msg
from ...paths import find_run_file
from ..catalog import Catalog, default_catalog_path
from ..model import LayerPlan


# 로고 도안 장수 상한 — 평가기의 그림 문턱(120장) 아래.
LOGO_LAYERS = 110


# 벡터화 노선 — 셀 노선(면 채움 + 선). 로고는 평면색이라 선보다 면이 맞다.
ROUTE = "cel"


# 워터마크·로고 무리를 어디에 두나. `auto`는 워터마크를 리어 범퍼 가운데(없으면
# 윈드실드 귀퉁이, 그것도 없으면 옆면 로커 줄 끝)에, 사용자 로고를 옆면 로커 줄 +
# 리어 좌우 + 프론트 범퍼에 앉힌다. 나머지는 그 면 하나에만 (`compose.sponsor`).
PLACEMENTS = ("auto", "rear", "front", "windshield", "rocker")


KIT_DIR = default_catalog_path().parent / "kit"


# 워터마크 파일 — 바탕 밝기로 고른다 (`compose.sponsor`).
WATERMARK = {"light": KIT_DIR / "logo.plan.json",       # 밝은 바탕 → 검정 잉크
             "dark": KIT_DIR / "logo-dark.plan.json"}   # 어두운 바탕 → 흰 잉크


@dataclass
class LogoSpec:
    """로고 옵션 — 조리법(`<이름>.fsitasha.json`)의 `logos` 키와 CLI 인자가 같은 꼴."""

    watermark: bool = True
    # 사용자 로고 — `{"image": <이미지 또는 도안 경로>, "plan": <벡터화한 도안>}`.
    # `plan`이 비면 `resolve`가 채운다.
    images: list[dict] = field(default_factory=list)
    placement: str = "auto"

    @property
    def active(self) -> bool:
        return bool(self.watermark or self.images)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "LogoSpec":
        d = dict(d or {})
        images: list[dict] = []
        for it in d.get("images") or []:
            if isinstance(it, dict):
                if it.get("image") or it.get("plan"):
                    images.append({"image": str(it.get("image") or it.get("plan")),
                                   "plan": (str(it["plan"]) if it.get("plan") else None)})
            elif it:
                images.append({"image": str(it), "plan": None})
        spec = cls(watermark=bool(d.get("watermark", True)), images=images,
                   placement=str(d.get("placement") or "auto"))
        spec.validate()
        return spec

    def validate(self) -> None:
        if self.placement not in PLACEMENTS:
            raise ValueError(msg("모르는 로고 자리: {placement} (있는 것: {places})",
                                 placement=self.placement, places=", ".join(PLACEMENTS)))


@dataclass(frozen=True)
class LogoItem:
    """앉힐 로고 하나 — 도안 파일과 종류."""

    plan: Path
    kind: str              # "watermark" · "user"
    name: str


def _sig(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()[:10]


def is_plan(path: Path) -> bool:
    """이미 우리 도안인가 (`*.json`에 `layers`가 있다)."""
    if path.suffix.lower() != ".json":
        return False
    try:
        return '"layers"' in path.read_text(encoding="utf-8")
    except OSError:
        return False


def cap_layers(plan: LayerPlan, cat: Catalog, cap: int = LOGO_LAYERS) -> LayerPlan:
    """도안을 `cap`장 아래로 — 시각 기여가 작은 레이어부터 (이분 탐색).

    `prune_plan`은 반투명 레이어를 못 받는다 — 그런 도안은 넓이 작은 것부터
    뺀다 (로고는 평면색이라 거의 안 온다)."""
    if len(plan.layers) <= cap:
        return plan
    from ..pruneplan import prune_plan

    try:
        p, st = prune_plan(plan, cat, min_vis=0.0, strict_labels=())
        if st["after"] <= cap:
            return p
        lo, hi = 0.0, 64.0
        while True:                                # 상한을 먼저 찾는다
            p, st = prune_plan(plan, cat, min_vis=hi, strict_labels=())
            if st["after"] <= cap or hi > 1e6:
                break
            lo, hi = hi, hi * 4
        best = p
        while hi - lo > max(1.0, 0.01 * hi):
            mid = (lo + hi) / 2
            p, st = prune_plan(plan, cat, min_vis=mid, strict_labels=())
            if st["after"] <= cap:
                hi, best = mid, p
            else:
                lo = mid
        return best
    except ValueError:
        keep = sorted(range(len(plan.layers)),
                      key=lambda i: -abs(plan.layers[i].sx * plan.layers[i].sy))[:cap]
        return LayerPlan(source_image=plan.source_image, image_size=plan.image_size,
                         units_per_px=plan.units_per_px,
                         layers=[plan.layers[i] for i in sorted(keep)])


def vectorize(image: str | Path, cache_dir: Path, *, cat: Catalog | None = None,
              log=None) -> Path:
    """이미지 → 로고 도안 (`cache_dir/logo-<서명>/…plan.json`). 도안이면 그대로.

    셀 노선으로 굽고 `LOGO_LAYERS`장으로 깎는다. 같은 이미지는 한 번만 굽는다."""
    src = Path(image).resolve()
    if not src.is_file():
        raise ValueError(msg("로고 파일이 없다 — {path}", path=src))
    if is_plan(src):
        return src
    cache_dir = Path(cache_dir)
    out = cache_dir / f"logo-{_sig(src)}"
    plan_path = find_run_file(out, "plan.json")
    if plan_path.is_file():
        return plan_path
    from ..pipeline import make

    quiet = log or (lambda *_a, **_k: None)
    quiet(msg("로고 벡터화 — {name} (셀 노선, {cap}장 상한)", name=src.name, cap=LOGO_LAYERS))
    make(src, out, route=ROUTE, log=lambda *_a, **_k: None)
    plan_path = find_run_file(out, "plan.json")
    plan = LayerPlan.load(plan_path)
    capped = cap_layers(plan, cat or Catalog(default_catalog_path()))
    if len(capped.layers) != len(plan.layers):
        capped.save(plan_path)
        quiet(msg("  {name}: {n:,}장 → {m}장 (시각 기여 하위 컷)",
                  name=src.name, n=len(plan.layers), m=len(capped.layers)))
    return plan_path


def resolve(spec: LogoSpec, cache_dir: Path, *, cat: Catalog | None = None,
            log=None, notes: list[str] | None = None) -> list[LogoItem]:
    """스펙의 사용자 로고를 도안으로 갖춘다 (`spec.images[i]["plan"]`을 채운다).

    못 굽는 이미지는 빼고 말한다 — 한 장 때문에 판이 안 서면 안 된다."""
    items: list[LogoItem] = []
    for it in spec.images:
        try:
            p = Path(it["plan"]) if it.get("plan") and Path(it["plan"]).is_file() \
                else vectorize(it["image"], cache_dir, cat=cat, log=log)
        except (ValueError, OSError, RuntimeError, SystemExit) as e:
            if notes is not None:
                notes.append(msg("로고를 못 굽는다 — {name}: {e}",
                                 name=Path(str(it.get("image"))).name, e=e))
            continue
        it["plan"] = str(p)
        items.append(LogoItem(plan=p, kind="user", name=Path(str(it["image"])).name))
    return items


def watermark_plan(dark_background: bool) -> Path | None:
    """바탕 밝기에 맞는 워터마크 도안 (키트가 없으면 None)."""
    p = WATERMARK["dark" if dark_background else "light"]
    return p if p.is_file() else None
