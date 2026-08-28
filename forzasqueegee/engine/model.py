"""레이어 계획 데이터 모델.

좌표계·수치 체계는 인게임 실측 기준:
- 캔버스 중앙 원점, X+ 오른쪽, Y+ 위쪽
- 도형은 ±1 정규화 좌표(카탈로그)이며 스케일 1.0일 때 반폭 64 캔버스 유닛
- 변형 스텝: 이동 0.5(화살표), 스케일 0.01, 회전 0.1°, 기울기 0.01, 투명도 0~100

**색의 정본은 RGB 바이트다** (게임 레코드 +0x74가 RGBA 바이트 — 주입·KFPS
왕복이 이 값 그대로 오간다). HSB는 저장하지 않는다 — 게임의 **색 입력
UI**(HSB 슬라이더 0.01 스텝)를 실제로 쓰는 곳(창 조작·오버레이 안내)만
`hsb()`로 그 자리에서 유도한다. RGB→HSB 재유도를 저장 경로에 두면 왕복마다
어두운 색의 채도가 한 칸씩 걷는 래칫이 생긴다 (2026-08-25 전수 실측:
그리드 코너 51회 왕복에 Δ49/255) — 정본을 바이트로 두면 원천 소멸한다.
"""

from __future__ import annotations

import colorsys
import json
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

# 스케일 1.0 도형의 정규화 1.0 길이가 차지하는 캔버스 유닛 수
# (2026-08-02 정밀 실측: 스케일 도구 OCR + 차분 에지 → 폭 128유닛/±1, 격자 3칸)
UNITS_PER_SCALE = 64.0


def _q(v: float, step: float) -> float:
    return round(round(v / step) * step, 4)


@dataclass
class Layer:
    """게임 레이어 1개 = 카탈로그 도형 + 변형 + 색."""

    shape: str  # 카탈로그 도형 이름 (예: "A_01")
    x: float = 0.0  # 캔버스 유닛
    y: float = 0.0
    sx: float = 1.0  # 게임 스케일 값 (음수 = 미러)
    sy: float = 1.0
    rot: float = 0.0  # 도(deg), 0~360
    skew: float = 0.0
    color: tuple[int, int, int] = (255, 255, 255)  # sRGB 바이트 (레코드 그대로)
    alpha: float = 100.0  # 0~100
    label: str = ""  # 레이어 분류 (fp/fp_bg/mask — i18n region.* 키)
    # 뺄셈 마스크 도안: 색 없음, 이 레이어보다 먼저 그려진 레이어 전부를
    # 도형 영역만큼 잘라 배경 노출 (이후 레이어는 무영향, 비파괴 — 실측 10차)
    mask: bool = False
    # 획 그룹 id (cel 노선) — 같은 값이면 **한 획에서 나온 마디**다. 사람은 획을
    # 한 번에 긋지 중간에서 끊지 않으므로, 프루닝은 이 단위로 전부 살리거나
    # 전부 버린다 (중간 절단 = 획이 점선이 된다). -1 = 획 아님(면·메움·수리)
    stroke: int = -1

    def __post_init__(self) -> None:
        self.color = tuple(int(v) for v in self.color)  # json 리스트 → 튜플

    def quantized(self) -> "Layer":
        """게임 입력 스텝으로 양자화한 사본 (색은 바이트 정본이라 그대로)."""
        return Layer(
            shape=self.shape,
            x=_q(self.x, 0.5),
            y=_q(self.y, 0.5),
            sx=_q(self.sx, 0.01),
            sy=_q(self.sy, 0.01),
            rot=_q(self.rot % 360.0, 0.1),
            skew=_q(self.skew, 0.01),
            color=self.color,
            alpha=_q(self.alpha, 0.01),
            label=self.label,
            mask=self.mask,
            stroke=self.stroke,
        )

    def rgb(self) -> tuple[int, int, int]:
        return self.color

    def hsb(self) -> tuple[float, float, float]:
        """게임 색 입력 UI(HSB 0.01 스텝)에 넣을 값 — **적용 시점에만** 부른다.

        UI 그리드가 낼 수 있는 색 중 이 레이어의 RGB에 가장 가까운 것을
        고른다 (`game_hsb`). 창 조작·수동 안내 전용이다 — 주입은 RGB 바이트를
        그대로 쓰므로 이 변환이 없다.
        """
        return game_hsb(*self.color)


def rgb_to_hsb(r: int, g: int, b: int) -> tuple[float, float, float]:
    """RGB → HSB 0.01 그리드 반올림 — **분석·휴리스틱용** (색조·밝기 판단 등).

    게임 색 입력값이 필요하면 `game_hsb`를 쓸 것 — 반올림 그리드는 렌더
    최근접이 아니다 (최악 9/255 어긋난다, 2026-08-25 전수 실측).
    """
    h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
    return round(h, 2), round(s, 2), round(v, 2)


def hsb_to_rgb(h: float, s: float, b: float) -> tuple[int, int, int]:
    """게임 색 입력 HSB가 실제로 내는 RGB 바이트 (내림 절단 — 인게임 실측)."""
    r, g, bb = colorsys.hsv_to_rgb(h, s, b)
    return int(r * 255), int(g * 255), int(bb * 255)


@lru_cache(maxsize=65536)
def game_hsb(r: int, g: int, b: int) -> tuple[float, float, float]:
    """이 RGB에 **가장 가까운 색을 내는** 게임 HSB 그리드점 (0.01 스텝).

    반올림 유도는 절단·격자 상호작용 때문에 최근접이 아니다 — 후보를 해석적
    구간(각 채널을 재현하는 v·s·h 그리드 이웃)에서 모아 렌더 거리 최소를
    고른다. 우리 그리드 색(`hsb_to_rgb` 출력)은 정확히 재현되므로 그 색은
    왕복 불변이고, 임의 색(편집기에서 새로 고른 것)은 1회 스냅 후 불변이다.
    """
    r, g, b = int(r) & 255, int(g) & 255, int(b) & 255
    h0, s0, v0 = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
    mx = max(r, g, b)

    def grid_near(v: float, span: int) -> list[float]:
        c = round(v * 100)
        return [k / 100 for k in range(max(0, c - span), min(100, c + span) + 1)]

    # v: max 채널 재현은 반올림 ±1칸이지만 s와의 교환 최적이 ±2에 선다 (전수 대조)
    v_cands = grid_near(v0, 2)
    # s: min = int(255·v·(1−s)) 절단이 1/mx 만큼 미니 — 어두울수록 넓게
    s_span = 2 + (0 if mx >= 200 else min(12, int(255 / max(mx, 8) + 1)))
    s_cands = grid_near(s0, s_span)
    # h: mid 채널 절단 1이 색상 1/(6·채도폭) 만큼 미니 — 저채도일수록 넓게
    chroma = mx - min(r, g, b)
    h_span = 2 + (0 if chroma >= 32 else min(12, int(255 / max(6 * chroma, 16) + 2)))
    hc = round(h0 * 100)
    h_cands = [((hc + k) % 100) / 100 for k in range(-h_span, h_span + 1)]

    best = None
    for v in v_cands:
        for s in s_cands:
            for h in (h_cands if s > 0 else h_cands[:1]):
                rr, gg, bb = hsb_to_rgb(h, s, v)
                d = (rr - r) ** 2 + (gg - g) ** 2 + (bb - b) ** 2
                if best is None or d < best[0]:
                    best = (d, h, s, v)
                    if d == 0:
                        return h, s, v
    return best[1], best[2], best[3]


@dataclass
class LayerPlan:
    """변환 결과: 그리는 순서대로의 레이어 목록 + 소스 메타."""

    source_image: str = ""
    image_size: tuple[int, int] = (0, 0)  # (w, h) px
    units_per_px: float = 1.0
    layers: list[Layer] = field(default_factory=list)

    def save(self, path: str | Path) -> None:
        data = {
            "source_image": self.source_image,
            "image_size": list(self.image_size),
            "units_per_px": self.units_per_px,
            "layers": [asdict(l) for l in self.layers],
        }
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "LayerPlan":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            source_image=data["source_image"],
            image_size=tuple(data["image_size"]),
            units_per_px=data["units_per_px"],
            layers=[Layer(**_layer_dict(d)) for d in data["layers"]],
        )


def _layer_dict(d: dict) -> dict:
    """plan.json 레이어 dict → Layer 인자. 구판(h/s/b 저장)은 한 번 변환한다.

    구판의 색은 그 HSB가 실제로 내던 RGB(`hsb_to_rgb` 절단)로 옮긴다 —
    렌더·주입 바이트가 그대로다. 저장은 항상 RGB다.
    """
    if "h" in d and "color" not in d:
        d = dict(d)
        d["color"] = hsb_to_rgb(d.pop("h", 0.0), d.pop("s", 0.0), d.pop("b", 0.0))
    return d
