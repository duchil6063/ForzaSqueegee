"""산포 문법 — 조각을 어디에 몇 개 흩나 (캔버스·면이 나눠 쓴다)."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from ..catalog import Catalog
from ..model import Layer
from .look import Look
from .vocabulary import _RING8, MOTIF_INSCRIBE, motif_shapes, shape_half


# 산포 데코 개수. 레이어 예산에는 티도 안 나고(면 상한 3,000), 더 흩으면
# 스티커 티가 나기 시작한다 — 레퍼런스의 모티프도 한 면에 십수 개 규모다.
DECO_N = 26


# 산포가 **뭉치는 자리** — 프레임 반경의 몫으로 차 뒤쪽이다. 0이면 균등 산포
# (색종이), 1.0이면 리어 범퍼에 다 몰린다. 레퍼런스의 무리 중심은 차 길이의
# 0.68~0.80 언저리다 (수이세이의 별무리·RIN SHIBUYA의 꽃무리·EVELYNE의 백합).
DECO_ANCHOR = 0.40


# 뭉치는 자리를 인물 상자 모서리에서 얼마나 더 밀어내나 (인물 폭의 몫).
# 0이면 모서리에 딱 붙어 무리 절반이 다시 인물에 걸린다.
DECO_ANCHOR_GAP = 0.35


# 무리의 **기울기** — 1이면 균등, 클수록 중심에 몰린다. 2.2면 절반이 무리 반경
# 안쪽 3할에 들고 나머지가 바깥으로 성기게 빠진다 (레퍼런스 무리의 꼴).
DECO_FALLOFF = 2.2


# 모티프 **층의 경계** — 뭉치는 자리에서의 정규화 거리(`d`, 산포 반경 대비)다.
# 이 거리 안쪽이면 그 층. 마지막 층은 나머지 전부다.
DECO_TIER = (0.30, 0.52, 0.78)


# 층별 크기 (인물 높이 대비) — 무리 가운데가 크고 바깥으로 잔다.
DECO_TIER_SIZE = (0.55, 0.34, 0.20, 0.11)


# 층별 **개수 상한**. 레퍼런스의 큰 모티프는 한둘이고 나머지가 잔것이다
# (EVELYNE의 백합 한 송이가 크고 열 송이가 작다 · 수이세이의 큰 별 하나 ·
# Fate의 큰 별 둘). 거리로만 층을 가르면 뭉치는 자리 둘레에 큰 것이 여럿
# 서서 면이 덩어리로 막힌다 — 상한이 "몇 개가 구도를 잡나"를 못 박는다.
DECO_TIER_MAX = (1, 3, 6, 99)


# 모티프 둘의 최소 간격 (두 크기의 평균 대비). 1.0이면 외접 사각이 안 닿고,
# 0.62면 모서리만 문다 — 레퍼런스의 무리도 그 정도로 겹친다.
DECO_SEP = 0.62


# 최대형 모티프가 **패널 짧은 변**에서 차지할 수 있는 몫의 상한. 크기 자는
# 도안이지만(`DECO_TIER_SIZE`는 인물 높이 대비다) 그 자는 **옆면**에서 잰 것이라
# — 옆면은 인물보다 네 배 넓다 — 좁은 패널에 그대로 쓰면 최대형 하나가 판을
# 덮는다 (제로투 실측: 리어 182유닛에 94유닛짜리 꽃이 앉아 패널이 꽃 세 송이로
# 찼다). 레퍼런스의 좁은 패널 실측: Fate R34 리어의 큰 별 0.28 · 수이세이 리어
# 0.31 · Cygames 86 리어 범퍼 0.22.
DECO_HERO_CAP = 0.30


# 이미 놓인 것에서 **가장 가까운 것까지의 상한** — 자는 `DECO_SEP`과 같다
# (두 크기의 평균). 무리는 뭉치되 이어져야 한다: 레퍼런스 8장 어디에도 사이가
# 텅 빈 채 홀로 뜬 조각이 없다. 기울기(`DECO_FALLOFF`)만으로는 못 막는다 —
# 자리·간격 검사가 무리 안쪽 후보를 대량으로 버리고 나면 목록 끝의 먼 후보가
# 그대로 통과한다 (제로투 실측: 앞펜더 끝과 리어 끝에 꽃 한 송이씩이 무리와
# 끊긴 채 섰다).
#
# 3.0은 실측 분포의 **빈 띠**다 (도안 3장 × 차 3대를 실제 파이프라인으로 짓고
# 놓인 모티프 19개의 최근접 거리를 잰 것): 0.53 0.53 0.54 0.54 0.65 0.65 0.72
# 0.72 1.06 1.06 1.35 1.35 1.59 1.84 1.93 1.93 2.35 **3.65 8.97**. 2.35에서
# 3.65로 뛰는 자리가 무리와 고아의 경계이고, 그 위 둘은 정확히 우리가 잡으려던
# 것이다 — 미쿠 리어 쿼터의 대형 별(지름 187)과 호시노 옆면의 대형 별(125),
# 둘 다 이웃 없이 홀로 서 있었다. 사이가 비어 있어 2.4~3.6 어디에 놓아도 같다.
DECO_GAP_MAX = 3.0


# 도안 **위**에 얹는 전경 모티프 개수. 레퍼런스의 모티프는 인물 뒤에만 있지
# 않다 — EVELYNE의 백합과 RIN SHIBUYA의 꽃은 인물 팔·다리를 덮고 지나간다.
# 전부 뒤에 깔면 "배경에 스티커를 얹은" 꼴로 읽히고, 앞뒤로 걸쳐야 인물이
# 장면 **안에** 있는 것으로 읽힌다. 몇 장이면 되므로 그룹 하나를 더 쓴다.
DECO_FRONT_N = 3


# 전경 모티프의 크기 자 (배경 대비). 배경은 맨 도색 위라 인물 높이의 절반까지
# 커도 되지만, 전경은 **그림 위**라 같은 크기면 얼굴을 덮는다.
DECO_FRONT_SIZE = 0.42


# 후광 사본이 얼마나 큰가 (모티프 대비). 레퍼런스의 테두리는 모티프 폭의
# 5~8% 규모다 — 더 키우면 테두리가 아니라 두 번째 모티프로 읽힌다.
HALO_GROW = 1.13


@dataclass
class Motif:
    """산포 한 조각 — 자리·크기·층. **좌표계를 모른다**.

    캔버스 꾸밈 그룹(`deco_layers`)과 면 도형(`surface_deco_shapes`)이 이걸 같이
    받아 제 꼴로 뱉는다. 문법이 한 벌이어야 차를 한 바퀴 돌 때 같은 무리로
    읽힌다 — 갈라 두었더니 면 도형 쪽에만 도안 기준이 없었다.
    """

    x: float
    y: float
    size: float                     # 지름 (그 좌표계의 유닛)
    half: float                     # 게임 스케일 값 (`shape_half`)
    shape: str
    rot: float
    tier: int
    color: tuple[int, int, int]
    alpha: float


def scatter_motifs(*, center: tuple[float, float], radii: tuple[float, float],
                   ref: float, n: int, vocab: tuple[str, ...], cat: Catalog,
                   colors: tuple[tuple[int, int, int], ...],
                   anchor_at: tuple[float, float] | None = None,
                   avoid: tuple[float, float, float, float] | None = None,
                   over: bool = False, place_ok=None,
                   phase: float = 0.0,
                   gap: float | None = None) -> list[Motif]:
    """**산포 문법 한 벌** — 레퍼런스 8장에서 뽑은 무리의 꼴.

    자리·크기·층을 정하는 자가 전부 여기 있다. 부르는 쪽은 좌표계와 뱉는 꼴만
    다르다.

    - `center`·`radii` — 후보 구름의 중심·반경 (이 면/판에서 흩어도 되는 범위)
    - `anchor_at` — 무리가 **뭉치는 자리**. 안 주면 `center`다. 레퍼런스의 무리는
      균등하지 않고 한쪽(대개 도안 뒤쪽 리어 쿼터)에 몰려 반대쪽으로 성기게
      빠진다 — 뭉침이 곧 구도다.
    - `ref` — 크기 자. **도안 크기**를 준다 (최대형이 그 0.55배 — 레퍼런스 실측
      0.4~0.7). 면 크기를 주면 도안과 무관한 스티커가 된다.
    - `avoid`/`over` — 도안 상자. 배경 벌은 피하고(`over=False`), 전경 벌은
      실루엣 **가장자리에 걸치는 것만** 남긴다.
    - `place_ok(x, y, rq)` — 그 자리에 온전히 놓을 수 있나 (마스크·이음새 판정).
    - `phase` — 나선의 위상. 면마다 흔들어야 같은 배열이 되풀이되지 않는다.
    - `gap` — 이웃이 **두 크기 평균의 이 배수**보다 멀면 그 조각을 잔것으로
      내린다 (고아 금지 — `DECO_SEP`과 같은 자의 반대쪽 끝이다).
    """
    cx, cy = center
    rx, ry = radii
    ax, ay = anchor_at if anchor_at is not None else (cx, cy)

    # 1) 후보를 넉넉히 뜬 뒤 **뭉치는 자리에 가까운 순으로** 고른다. 균등 산포는
    #    색종이로 읽힌다 (수이세이의 별무리 · RIN SHIBUYA의 꽃무리 · EVELYNE의
    #    백합은 예외 없이 한쪽 끝에 몰린다).
    cand: list[tuple[float, float, float, float]] = []   # (점수, x, y, 각)
    for i in range(n * 6):
        a = phase + i * 2.399963                 # 황금각 — 뭉치지 않는 결정적 산포
        rr = 0.22 + 0.78 * ((i * 7 % (n * 3)) / max(1, n * 3 - 1))
        x = cx + math.cos(a) * rx * rr
        y = cy + math.sin(a) * ry * rr
        # 거리는 **등방**이라야 한다. x를 rx로, y를 ry로 나누면 짧은 축(차체
        # 밴드는 면 폭의 6분의 1이다)이 분모를 지배해서 "뭉치는 자리에 가깝다"가
        # 사실상 "가운데 높이다"가 되고, 무리가 자리와 무관하게 가로로 퍼진다
        # (2026-08-21 미리보기 판정). 둘 다 rx로 나눈다 — 밴드는 어차피 좁아
        # 세로 편차가 자연히 작다.
        d = math.hypot(x - ax, y - ay) / max(1e-6, rx)
        cand.append((d, x, y, a))
    cand.sort(key=lambda t: t[0])
    # 가까운 순으로 **앞에서부터 n개**를 그냥 집으면 한 점에 뭉쳐 덩어리가 된다
    # (2026-08-21 미리보기 판정: 리어 휠아치 위 한 자리에 스물여섯이 겹쳤다).
    # 레퍼런스의 무리는 뭉치되 **기울기**가 있다 — 중심이 빽빽하고 바깥으로
    # 성기게 빠진다. 정렬 목록을 거듭제곱으로 훑어 그 기울기를 낸다.
    # 훑는 매개변수는 **0~1**이라야 한다 — `k/n`을 n의 두 배까지 돌리면 지수 항이
    # 1을 넘어 뒤쪽 절반이 통째로 목록 끝값에 붙고, 무리가 뭉치는 자리가 아니라
    # **반대쪽**에 선다 (2026-08-21 미리보기 판정: 리어에 몰려야 할 별무리가
    # 앞펜더에 섰다). 간격 검사가 후보를 버리므로 n의 세 배를 뽑아 훑는다.
    m = max(1, n * 3)
    last = len(cand) - 1
    order = [cand[min(last, int(last * (k / m) ** DECO_FALLOFF))] for k in range(m)]

    out: list[Motif] = []
    j = 0
    put: list[tuple[float, float, float]] = []   # (x, y, 크기) — 간격 검사용
    n_tier = [0] * (len(DECO_TIER) + 1)          # 층별로 몇 장 섰나
    for d, x, y, a in order:
        if j >= n:
            break
        if any(abs(x - px) < 1e-6 and abs(y - py) < 1e-6 for px, py, _s in put):
            continue                             # 같은 후보가 두 번 뽑혔다
        # 크기는 **한 벌이 아니라 층**이다 — 큰 모티프 몇 개가 구도를 잡고
        # 작은 것이 사이를 메운다. 같은 크기로 흩으면 개수만 는 색종이가 된다
        # (12호차 캡처 판정). 층의 실측 범위 (레퍼런스 8장): 큰 모티프는 인물
        # 높이의 **4~7할**이다 — 수이세이의 흰 별 0.7 · Fate 별 0.6 · EVELYNE
        # 백합 0.48 · RIN SHIBUYA 꽃 0.39. 잔것은 1할 아래로도 간다 (Cygames
        # 86의 비말).
        #
        # 층을 가르는 자는 **뭉치는 자리에서의 거리**다 (`DECO_TIER`). 받아들인
        # 순서(j)로 가르던 옛 자는 순서가 거리와 안 맞아서 — 자리 검사와 간격
        # 검사가 후보의 대다수를 버리므로 — 최대형 모티프가 무리에서 떨어진
        # 면 끝에 홀로 섰다 (2026-08-22 미리보기 판정: 도안 셋 다 옆면 앞뒤 끝에
        # 큰 별 하나씩이 떠 있었다. frag0-03의 산포 x 범위 ±352에서 최대형 둘이
        # +352와 +151, 잔것 무리는 −170~−211로 반대쪽이었다).
        # 거리로 가르면 무리 가운데가 크고 바깥이 잔것인 레퍼런스의 기울기가
        # 그대로 나오고, 큰 모티프가 홀로 서는 일이 **원리적으로** 없어진다.
        tier = next((i for i, lim in enumerate(DECO_TIER) if d < lim),
                    len(DECO_TIER))
        # 층이 찼으면 **한 단계 잔것으로** 내린다 (버리지 않는다 — 자리는 이미
        # 무리 안이라 쓸 만하고, 버리면 무리가 성겨진다)
        while tier < len(DECO_TIER) and n_tier[tier] >= DECO_TIER_MAX[tier]:
            tier += 1
        size = DECO_TIER_SIZE[tier] * ref
        # 도형을 **먼저** 고르고 그 뻗음으로 스케일을 낸다 — 계열마다 뻗음이
        # 달라(꽃 1.67 · 별 1.00) 도형을 모르고는 크기를 못 정한다
        sh = vocab[(j * 3 + tier) % len(vocab)]
        half = shape_half(cat, sh, size)
        # **인물을 덮지 않는다** (배경 벌) — 큰 모티프가 인물 높이의 절반까지
        # 커도 되는 것은 레퍼런스에서 그것이 **맨 도색 위**에 있기 때문이다
        # (EVELYNE의 백합·RIN의 꽃·수이세이의 큰 별은 전부 인물 옆 리어 쿼터에
        # 있다). 인물 위에 얹으면 같은 크기가 그림을 지운다 — 2026-08-22 판정:
        # 제로투 얼굴과 몸통을 청록 꽃 한 송이가 통째로 덮었다.
        r = MOTIF_INSCRIBE * size / 2
        if avoid is not None:
            hits = (x + r > avoid[0] and x - r < avoid[2]
                    and y + r > avoid[1] and y - r < avoid[3])
            if not over and hits:
                continue                         # 배경 — 인물을 피한다
            if over:
                # 전경 — 인물 **실루엣 가장자리에 걸친다**. 상자에 닿기만 하면
                # 되게 두면 한가운데(얼굴·몸통)에 앉는다 (2026-08-22 판정:
                # 제로투 얼굴 위에 꽃 한 송이가 그대로 얹혔다). 레퍼런스의
                # 전경은 팔·다리·머리끝을 **스치고 지나간다** — 상자 안에
                # 통째로 들어간 것은 그 문법이 아니다.
                inside = (x - r >= avoid[0] and x + r <= avoid[2]
                          and y - r >= avoid[1] and y + r <= avoid[3])
                if not hits or inside:
                    continue
        # **겹쳐 쌓지 않는다** — 뭉치는 자리에 큰 모티프가 몰리면 서로를 덮어
        # 한 덩어리 색면이 된다 (2026-08-21 미리보기 판정: 리어 쿼터가 통째로
        # 빨간 판이 됐다). 무리는 뭉치되 조각은 각각 읽혀야 한다 — 수이세이의
        # 별무리도 별끼리는 모서리만 문다. 간격은 두 모티프 크기의 평균에
        # 비례한다 (큰 것끼리는 멀리, 잔것은 붙어도 된다).
        if any(math.hypot(x - px, y - py) < DECO_SEP * (size + ps) / 2
               for px, py, ps in put):
            continue
        # **안 그려질 자리는 안 쓴다** — 산포는 마스크를 안 보고 앉으므로
        # 휠아치 구멍·벨트라인 위에 떨어진 모티프가 통째로 사라진다 (미리보기
        # 실측: 26장 중 여섯이 아치에 떨어졌다).
        if place_ok is not None and not place_ok(x, y, r):
            continue
        c = colors[j % len(colors)]
        out.append(Motif(
            x=x, y=y, size=size, half=half, shape=sh,
            rot=(a * 57.29578 + 37.0 * j) % 360.0, tier=tier, color=c,
            # 큰 것은 살짝 비쳐 배경으로 물러나고 잔것은 또렷하다 (레퍼런스의
            # 대형 모티프는 거의 다 반투명 층이다)
            alpha=74.0 if tier == 0 else (86.0 if tier == 1 else 100.0)))
        put.append((x, y, size))
        n_tier[tier] += 1
        j += 1

    # **무리에서 떨어진 것은 잔것이라야 한다.** 레퍼런스에 고아가 없다는 말은
    # "멀리 있는 조각이 없다"가 아니라 — Cygames 86의 비말도 EVELYNE의 백합도
    # 멀리까지 흩어진다 — **멀리 있는 것이 크지 않다**는 뜻이다. 중형·대형
    # 하나가 무리와 끊긴 채 홀로 서면 스티커로 읽힌다 (제로투 실측: 앞펜더 끝과
    # 리어 끝에 중형 꽃 한 송이씩. 미쿠 실측: 리어 쿼터 끝의 대형 별 하나).
    #
    # **놓고 나서** 재는 이유가 둘이다. 하나, 버리면 그 자리가 다음 무리로 가는
    # 다리인 판에서 무리 하나가 통째로 죽는다 (실측: 일곱 중 다섯을 잃었다).
    # 둘, 무리의 핵(최대형)은 **맨 먼저** 놓이므로 놓는 중에 재면 잴 이웃이
    # 아직 없어 영영 안 걸린다 — 정작 홀로 설 위험이 가장 큰 것이 그것이다.
    #
    # 크기만 내리고 자리는 그대로다. 줄어들기만 하므로 간격·회피·마스크 검사가
    # 다시 틀어질 일이 없다. 판정은 **원래 크기 한 벌로** 한 번만 한다 (내린
    # 것으로 다시 재면 문턱이 같이 줄어 연쇄로 다 잔것이 된다).
    if gap is not None and len(out) > 1:
        small = DECO_TIER_SIZE[-1] * ref
        for i, a in enumerate(out):
            if a.tier >= len(DECO_TIER):
                continue
            if all(math.hypot(a.x - b.x, a.y - b.y) > gap * (a.size + b.size) / 2
                   for k, b in enumerate(out) if k != i):
                out[i] = replace(a, size=small, tier=len(DECO_TIER),
                                 half=shape_half(cat, a.shape, small),
                                 alpha=100.0)
    return out


def deco_layers(lk: Look, colors: tuple[tuple[int, int, int], ...],
                cat: Catalog, n: int = DECO_N,
                spread: float | None = None,
                shapes: tuple[str, ...] | None = None,
                size_ref: float | None = None,
                anchor: float = 0.0,
                halo: tuple[int, int, int] | None = None,
                drawable_at=None,
                avoid: tuple[float, float, float, float] | None = None,
                over: bool = False) -> list[Layer]:
    """인물 **곁에** 흩는 액센트 조각 — 레퍼런스의 별·꽃·스플래터의 우리 어휘판.

    레퍼런스 8장 전부에서 배경 모티프는 인물 **밖**에 있다 — EVELYNE의 백합은
    리어 쿼터와 로커에 있고 인물은 문짝에 있다, RIN SHIBUYA의 꽃도 리어 쿼터다,
    수이세이의 큰 별도 리어 쿼터다. 큰 모티프가 인물 높이의 4~7할까지 커도
    괜찮은 이유가 이것이다: **맨 도색 위에** 있기 때문이다. 인물 위에 얹으면
    같은 크기가 그림을 덮어 버린다.

    `avoid`(인물이 덮는 상자, 이 좌표계)를 주면 그 상자에 닿는 자리를 버린다.
    안 주면 온 판에 흩는다.

    `over=True`면 반대로 **인물에 걸치는 것만** 남긴다 — 도안 위에 얹는 전경 벌이
    그 길이다 (레퍼런스의 꽃·백합 몇 송이는 팔·다리를 덮고 지나간다).

    어휘는 **게임 기성 도형**이고 한 계열 두 종뿐이다 (`motif_shapes`) — 종을
    늘리면 무리가 아니라 클립아트 모음이 된다. 자리는 황금각 나선이라 결정적이다
    (같은 도안 → 같은 데코).

    `spread`는 산포 최대 반경(캔버스 유닛, x축) — 띠 길이의 절반쯤을 주면 산포가
    띠와 같은 범위를 덮어 차 전체가 접착된다.

    자리를 정하는 문법은 `scatter_motifs`가 쥔다 — 이 함수는 그것을 캔버스
    레이어(+후광)로 뱉는 껍질이다.
    """
    cx, cy = lk.center
    rx = spread if spread is not None else lk.w * 1.5
    ref = size_ref or lk.h

    def _ok(x: float, y: float, rq: float, _f=drawable_at) -> bool:
        if _f is None:
            return True
        return all(_f(x + math.cos(th) * rq, y + math.sin(th) * rq)
                   for th in _RING8)

    out: list[Layer] = []
    for mo in scatter_motifs(
            center=(cx, cy), radii=(rx, lk.h * 0.55), ref=ref, n=n,
            vocab=shapes if shapes is not None else motif_shapes(lk, cat),
            cat=cat, colors=colors,
            anchor_at=(cx + anchor * rx, cy),    # 뭉치는 자리 (차 뒤쪽 끝)
            avoid=avoid, over=over, place_ok=_ok,
            # 전경 벌은 몇 장이 인물을 스치는 것이 전부라 이어질 무리가 아니다
            gap=None if over else DECO_GAP_MAX):
        # **후광** — 큰 모티프 뒤에 대비색 사본을 조금 키워 깐다. 레퍼런스의
        # 모티프는 예외 없이 테두리를 두른다 (수이세이의 흰 별 뒤 남색 별 ·
        # Fate의 색 별 둘레 흰 테). 없으면 베이스와 비슷한 색조의 모티프가
        # 차체에 묻혀 "얼룩"으로 읽힌다.
        if halo is not None and mo.tier <= 1:
            out.append(Layer(shape=mo.shape, x=mo.x, y=mo.y,
                             sx=mo.half * HALO_GROW, sy=mo.half * HALO_GROW,
                             rot=mo.rot, color=halo, alpha=mo.alpha,
                             label="itasha_deco"))
        out.append(Layer(shape=mo.shape, x=mo.x, y=mo.y, sx=mo.half,
                         sy=mo.half, rot=mo.rot, color=mo.color,
                         alpha=mo.alpha, label="itasha_deco"))
    return out
