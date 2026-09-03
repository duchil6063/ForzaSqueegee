"""색 — 베이스 도색 · 액센트 · 테마색."""

from __future__ import annotations

import colorsys
import math

from ..model import hsb_to_rgb, rgb_to_hsb
from .boxes import _gap
from .look import Look


# ---- 베이스 도색 규칙 (2026-08-21 — 레퍼런스·랩 업계 재판독) ----
# `references/이타샤` 8장의 베이스는 **흰 4 · 검 2 · 캐릭터 테마색 2**다 — 무채가
# 다수(6/8)다. 실차 랩 업계도 같다: 이타샤 랩의 기본 바탕은 흰(인쇄 바탕과
# 캐릭터 발색) 아니면 검이고, 테마색 전면 도장은 **캐릭터의 상징색이 확실할
# 때**(미쿠 청록·마린 빨강)만 쓴다 (10kwraps·itasha-guild·痛車アソビ 사례 조사).
#
# 그래서 테마색은 **단일 색조가 도안을 지배할 때만** 쓴다: 유채 팔레트의
# 면적에서 지배 색조(±0.08) 몫이 BASE_THEME_SHARE 이상. 그 밖은 흰/검이고,
# 가르는 자는 **흰 바탕에서 읽히나** 하나다 (`BASE_PALE_SHARE`).
BASE_THEME_SHARE = 0.58    # 지배 색조 몫이 이만큼이면 "상징색"으로 본다


BASE_THEME_SAT = 0.45      # 지배색 자체가 이만큼 진해야 상징색이다 — 살색·


                           # 갈색 머리처럼 흐린 지배색은 상징색이 아니라 그냥
                           # 제일 넓은 색이다 (14도안 실측: 게이트 없이는 테마가
                           # 8/14로 과반 — 레퍼런스 분포 2/8과 어긋난다)
BASE_HUE_NEAR = 0.08       # 같은 색조로 묶는 색조 거리


BASE_SAT_MIN = 0.22        # 이보다 흐린 색은 유채 후보에서 뺀다


BASE_WHITE = (245, 245, 245)


BASE_BLACK = (18, 18, 20)


# 도안의 **근백 잉크 몫**이 이 위면 검 차, 아래면 흰 차 (2026-08-22 렌더 실측 —
# 도안 12종 × 흰/검 두 바탕 대조). 가르는 것은 "이 그림이 흰 바탕에서 읽히나"
# 하나뿐이고, 흰 바탕이 삼키는 것은 **밝고 흐린** 잉크다 (`Look.pale`).
#
# 옛 자는 `person_value`(상위 6색 평균 명도)였는데 그것이 재는 것은 살색·하이라이트
# 면적이라 도안의 대비 능력과 상관이 없었다: 같은 그림의 변형 둘이 평균 명도
# 0.70/0.82로 갈려 흰 차와 검 차를 따로 받았고(근백 몫은 둘 다 0.12로 같다),
# 팔레트 1위가 **순검정**인 도안(ref01)이 검은 차를 받아 인물이 차체에 잠겼다.
# 실측 표본 12종에서 검이 44% — 레퍼런스 분포(검 2/10)의 두 배였다.
#
# 0.56은 실측 분포의 **틈**이다 (0.586 ↔ 0.539) — 같은 그림의 변형들이 한쪽으로
# 몰려 서고, 결과가 흰 8 · 검 4로 레퍼런스 비(4:2)와 맞는다.
BASE_PALE_SHARE = 0.56


def base_paint(lk: Look) -> tuple[tuple[int, int, int], tuple[float, float, float]]:
    """차 전체에 깔 **베이스 도색** (rgb, hsb) — 레퍼런스·랩 업계의 분포를 따른다.

    상징색이 확실한 도안(단일 색조 지배)만 눌러 앉힌 테마색이고, 나머지는 도안이
    흰 바탕에서 읽히느냐로 갈리는 흰/검이다. 이 값은 **초기값**이다 — 편집기의
    색 칸이 이 색으로 시작하고 사람이 그대로 바꿀 수 있다 (`build(base_rgb=…)`).
    """
    # 유채 후보의 **면적** 무게 (`Look.weights`). 순위 가중(1/(1+0.35·i))으로
    # 어림하던 자리다 — 면적이 곧 "지배 색조 몫"의 정의라 어림할 이유가 없다.
    cand: list[tuple[tuple[float, float, float], float]] = []
    # **상위 12색만** 본다 — 베이스는 "지배 색조가 있나"를 넓은 면적의 비로
    # 재는 자리라 꼬리 색을 넣으면 분모가 흔들린다 (액센트 쪽과 다른 자다).
    for c, w in zip(lk.palette[:12], (lk.weights or [1.0] * len(lk.palette))[:12]):
        h, s, b = rgb_to_hsb(*c)
        if s >= BASE_SAT_MIN and b >= 0.15:
            cand.append(((h, s, b), w))
    if cand:
        total = sum(w for _c, w in cand)
        (dh, ds, db), _w0 = cand[0]
        near = sum(w for (h, _s, _b), w in cand
                   if min(abs(h - dh), 1.0 - abs(h - dh)) <= BASE_HUE_NEAR)
        if (total > 1e-6 and ds >= BASE_THEME_SAT
                and near / total >= BASE_THEME_SHARE):
            return _separate_from_person((dh, ds, db), lk)
    rgb = BASE_BLACK if lk.pale > BASE_PALE_SHARE else BASE_WHITE
    return rgb, rgb_to_hsb(*rgb)


# 파스텔 베이스 — 채도 상한 · 명도 하한, 지배 색조가 없을 때의 채도.
PASTEL_BASE_SAT = 0.22
PASTEL_BASE_VAL = 0.93
PASTEL_BASE_SAT_FLAT = 0.14


def pastel_base(lk: Look) -> tuple[tuple[int, int, int], tuple[float, float, float]]:
    """무늬·꽃 프리셋의 **파스텔 바탕** — `base_paint`의 색조를 옅고 밝게 눌러 낸다.

    테마색이 선 도안은 그 색조의 파스텔, 흰/검으로 물러난 도안은 인물의 지배
    색조(없으면 주 액센트)를 아주 옅게 깐다 — 어느 쪽이든 인물보다 연하다.
    """
    rgb, (h, s, b) = base_paint(lk)
    if s >= BASE_SAT_MIN:
        hsb = (h, min(s, PASTEL_BASE_SAT), max(b, PASTEL_BASE_VAL))
    else:
        dom = dominant(lk)
        if dom is None:
            ah, _s, _b = rgb_to_hsb(*accent_color(lk, rgb))
        else:
            ah = dom[0]
        hsb = (ah, PASTEL_BASE_SAT_FLAT, PASTEL_BASE_VAL)
    out = tuple(int(round(v * 255)) for v in colorsys.hsv_to_rgb(*hsb))
    return out, rgb_to_hsb(*out)


# 인물 **지배색**과 베이스가 이만큼 안쪽이면 같은 색조로 본다
PERSON_HUE_NEAR = 0.10


# 같은 색조일 때 벌려야 할 최소 명도차 — 이만큼 벌어지면 실루엣이 산다
PERSON_VAL_GAP = 0.38


# 같은 색조로 물러날 때의 채도 상한 (물러남 = 채도를 죽여 인물 뒤로 빠진다)
RETREAT_SAT = 0.42


def dominant(lk: Look) -> tuple[float, float, float] | None:
    """도안의 **지배색** (h, s, b) — 면적 1위 색. 팔레트가 비면 None.

    베이스·베드·모티프가 전부 이 색을 피하거나 벌려야 한다. 인물의 지배색은
    보통 머리색이고(미쿠 하늘색·마린 빨강), 차 색이 그것과 같으면 실루엣이
    통째로 동화된다 — 12호차 줄리아 캡처의 증상이 정확히 그것이었다.
    """
    return rgb_to_hsb(*lk.palette[0]) if lk.palette else None


def _separate_from_person(theme: tuple[float, float, float], lk: Look
                          ) -> tuple[tuple[int, int, int], tuple[float, float, float]]:
    """테마색 베이스를 **인물 지배색에서 떼어 놓는다** (목표 4).

    레퍼런스의 테마색 베이스는 인물의 머리색과 같은 색조여도 명도가 확실히
    갈린다 (밝은 인물 위 짙은 테마색, 어두운 인물 위 연한 테마색). 우리는 그
    규칙 없이 팔레트 1위를 그대로 칠해서 미쿠(하늘색 머리)를 하늘색 차에
    올렸고, 인물이 차체에 녹았다 (12호차 실측).

    같은 색조면 둘 중 하나로 푼다: **명도를 벌리거나**(테마 정체성은 남는다),
    벌릴 자리가 없으면 **물러난다**(채도를 죽인다).
    """
    h, s, b = theme
    dom = dominant(lk)
    s = min(s, 0.70)
    b = min(max(b, 0.55), 0.92)
    if dom is not None and dom[1] > 0.18:
        dh = abs(h - dom[0])
        dh = min(dh, 1.0 - dh)
        if dh < PERSON_HUE_NEAR:
            # 인물이 밝으면 베이스를 내리고, 어두우면 올린다
            if dom[2] >= 0.55:
                b = max(0.16, min(b, dom[2] - PERSON_VAL_GAP))
            else:
                b = min(0.94, max(b, dom[2] + PERSON_VAL_GAP))
            if abs(b - dom[2]) < PERSON_VAL_GAP * 0.8:
                s = min(s, RETREAT_SAT)          # 못 벌리면 물러난다
    rgb = tuple(int(round(v * 255)) for v in colorsys.hsv_to_rgb(h, s, b))
    return rgb, (round(h, 2), round(s, 2), round(b, 2))


# ---------- 꾸밈의 색 ----------
def accent_color(lk: Look,
                 car: tuple[int, int, int] | None = None) -> tuple[int, int, int]:
    """도안에서 **띠·모티프에 쓸 색** — 진하고 밝은 것.

    팔레트를 그대로 쓰면 인물 살색·머리색이 나와 배경 띠가 인물을 먹는다. 채도가
    높고 어둡지 않은 색을 고른다.

    `car`(베이스 도색 색)를 주면 **차와 같은 색조를 버린다** — 같은 색조의 띠는
    차체에 묻혀 "붙인 것"으로 안 보인다 (렌더 실측: 청록 베이스 위 청록 띠는
    아예 안 보였다).

    팔레트가 통째로 베이스와 한 색조면(단색 테마 도안) **베이스의 보색**으로 간다.
    무채색(흰/근검정)으로 물러나면 미쿠처럼 팔레트가 하늘색 일색인 도안에서 차
    전체가 청록+흰 단벌이 된다 — 레퍼런스의 "테마색 + 대비색" 이중주가 안 난다
    (실차 캡처 판정). 보색은 인물 색조와 최대로 갈리므로 실루엣도 같이 산다.

    **살색은 액센트가 아니다.** 후보는 이미 `ACCENT_SRC_MIN`만큼 진한 색이라야
    한다 — 그만큼 안 진하면 이 도안에는 테마색이 없는 것이고, 흐린 색을 끌어와
    채도를 올리면(`readable_on`의 하한) 그 색은 예외 없이 **인물의 살색**으로
    떨어진다 (2026-08-22 실측: ref01은 팔레트 최대 채도가 0.18인데 액센트가
    #D17F79 연어색으로 나왔고, frag0-01은 살색 #D6A8A2가 그대로 #D6877C가 됐다.
    두 도안 다 회색 머리·검은 옷이라 테마색이랄 것이 없다). 레퍼런스의 무채
    베이스 두 대가 정확히 그 자리에서 **무채 액센트**를 쓴다 (Cygames 86의 흰
    스플래터 · EVELYNE의 흰 백합).
    """
    ch = rgb_to_hsb(*car) if car is not None else None
    best, score = None, -1e9
    ws = lk.weights or [1.0] * len(lk.palette)
    for c, w in zip(lk.palette, ws):
        h, s, b = rgb_to_hsb(*c)
        if s < ACCENT_SRC_MIN:                      # 흐린 색 — 테마색이 아니다
            continue
        if w < ACCENT_AREA_MIN:                     # 티끌 색 — 상징색이 아니다
            continue
        if is_skin(h, s, b):                        # 살색은 액센트가 아니다
            continue
        v = s * min(1.0, b * 1.4)
        if ch is not None and ch[1] > 0.2:          # 차가 유채색일 때만 색조를 견준다
            dh = abs(h - ch[0])
            dh = min(dh, 1.0 - dh)
            if dh < 0.10:
                v -= 2.0                            # 차와 같은 색조 — 후보에서 사실상 뺀다
        if v > score:
            best, score = c, v
    if best is None or score < 0.15:
        if ch is not None and ch[1] > 0.20:         # 유채색 베이스 → 보색 액센트
            hb = (ch[0] + 0.5) % 1.0
            bb = 0.97 if ch[2] < 0.60 else 0.62
            return tuple(int(round(v * 255))
                         for v in colorsys.hsv_to_rgb(hb, 0.80, bb))
        # 흰/검 베이스(또는 차를 모를 때) → 무채 대비. 폴백은 이미 차를 보고 정했다
        return (22, 26, 38) if (ch is None or ch[2] > 0.55) else (245, 245, 245)
    return readable_on(best, car)


# 액센트가 차와 벌어져야 하는 명도 차 하한. 이보다 붙으면 모티프가 차체에 묻혀
# 얼룩으로 읽힌다 — 어두운 차 위 어두운 모티프는 그냥 구멍이다.
ACCENT_B_GAP = 0.28


# 액센트 채도 하한 — 도안에서 온 유채 액센트는 이만큼은 진해야 차체에서 색으로
# 읽힌다. **무채로 정한 액센트에는 안 먹인다** (`readable_on`).
ACCENT_S_MIN = 0.42


# 팔레트 색이 **액센트 후보가 되는** 채도 하한. 이 아래는 테마색이 아니라 살색·
# 머리색·하이라이트라 끌어오면 안 된다 (`accent_color`).
ACCENT_SRC_MIN = 0.35


# 액센트 후보의 **면적 하한** (도안 잉크 대비). 상징색은 면적이 작다 — 시로코의
# 청록 후광 1.6% · 아리스의 하늘색 1.2% — 그래서 팔레트를 32색까지 보지만,
# 그 아래로 내려가면 안티앨리어싱 부스러기와 그림자 색이 상징색 행세를 한다
# (11번 도안: 0.09%짜리 하늘색 한 점이 팔레트 135위에 있다).
ACCENT_AREA_MIN = 0.008


# 이보다 흐리면 **일부러 고른 무채**로 본다 — 채도 하한을 안 먹인다.
ACCENT_GREY_MAX = 0.18


# 채도가 남아 있어도 이만큼 어둡거나 밝으면 사람 눈에는 검정·흰색이다.
# 무채 폴백 상수(`INK_DARK` = 22,26,38)는 푸른 기가 있어 채도가 0.42로 읽히는데,
# 그 색조를 "이 액센트의 색조"로 쓰면 셋째 액센트가 근검정의 파란 기를 따라
# 보라로 떨어진다 (2026-08-22 미리보기 판정 — 흰 차에 근검정 + 보라가 섰다).
ACCENT_DARK_MAX = 0.22


ACCENT_LIGHT_MIN = 0.92


def achromatic_accent(c: tuple[int, int, int]) -> bool:
    """이 액센트가 **무채로 읽히나** — 흐리거나, 아주 어둡거나, 아주 밝다."""
    _h, s, b = rgb_to_hsb(*c)
    return s <= ACCENT_GREY_MAX or b <= ACCENT_DARK_MAX or b >= ACCENT_LIGHT_MIN


def readable_on(c: tuple[int, int, int],
                car: tuple[int, int, int] | None) -> tuple[int, int, int]:
    """액센트를 **차 위에서 읽히게** 민다 — 명도 하한 + 채도 하한.

    색조 회피만으로는 안 된다: 색조가 갈려도 명도가 붙으면 모티프가 차체에 묻힌다
    (실측 — 짙은 빨강 베이스 b 0.35에 근검정 청록 b 0.21이 뽑혀 별무리가 검은
    구멍이 됐다). 도안 팔레트가 어두운 도안(짙은 옷·밤 장면)에서 늘 걸린다.

    미는 방향은 차의 반대다 — 어두운 차면 올리고 밝은 차면 내린다. 색조는 그대로
    두므로 도안에서 온 색이라는 것은 안 변한다.

    **무채는 무채로 둔다** — 채도 하한은 "도안에서 온 유채 액센트가 차체에서 색을
    잃지 않게" 하는 자인데, `accent_color`가 테마색이 없다고 판정해 흰/근검정을
    고른 자리에까지 먹이면 그 판정을 되돌려 버린다 (흰 액센트가 하늘색이 된다).
    """
    if car is None:
        return c
    h, s, b = rgb_to_hsb(*c)
    cb = rgb_to_hsb(*car)[2]
    if not achromatic_accent(c):
        s = max(s, ACCENT_S_MIN)
    if abs(b - cb) < ACCENT_B_GAP:
        b = min(1.0, cb + ACCENT_B_GAP) if cb < 0.55 \
            else max(0.10, cb - ACCENT_B_GAP)
    return tuple(int(round(v * 255)) for v in colorsys.hsv_to_rgb(h, s, b))


def accent_tint(main: tuple[int, int, int],
                car: tuple[int, int, int] | None = None
                ) -> tuple[int, int, int]:
    """액센트색의 **밝은 자매** — 산포 모티프의 둘째 색.

    채도를 반으로 죽이면 흰 얼룩이 되어 모티프가 색을 잃는다 (2026-08-20
    미리보기 판정). 색조는 그대로 두고 채도를 조금만 낮추고 명도를 올린다.
    밝은 차에서는 올리는 쪽이 차와 붙으므로 `car`를 주면 되민다.

    주색이 **무채면 자매도 무채다** — 흰 액센트의 밝은 자매는 회색이지 하늘색이
    아니다 (`accent_color`가 테마색 없음으로 판정한 도안에서 온다).
    """
    h, s, b = rgb_to_hsb(*main)
    # 무채 주색의 자매는 **명도만** 갈린다 (흰 액센트의 자매는 회색이지 하늘색이
    # 아니다) — 채도 하한은 `readable_on`이 같은 판정으로 건너뛴다
    s = s * 0.78 if achromatic_accent(main) else max(0.35, s * 0.78)
    out = tuple(int(round(v * 255)) for v in colorsys.hsv_to_rgb(
        h, s, min(1.0, b * 1.12 + 0.06)))
    return readable_on(out, car)


# 셋째 액센트가 주색·차와 벌어져야 하는 최소 색조 거리 (0~0.5).
ACCENT_HUE_GAP = 0.11


# 팔레트에 짝이 없을 때 주색에서 돌릴 색조 (이웃 색조 — 조화가 깨지지 않는다).
ACCENT_HUE_STEP = 0.09


# 주색이 무채일 때 셋째 액센트의 색조 — 금·주황. 무채 액센트를 쓰는 레퍼런스의
# 셋째가 둘 다 이쪽이다 (Cygames 86의 빨강·금 · EVELYNE의 주황 그래피티).
ACCENT_WARM = 0.09


def accent_third(main: tuple[int, int, int], lk: Look,
                 car: tuple[int, int, int] | None = None
                 ) -> tuple[int, int, int]:
    """**셋째 액센트** — 주색과 색조가 갈리는 짝.

    레퍼런스의 배경은 예외 없이 **세 색 이상**이다 (KOTONE 자홍+노랑+시안 ·
    수이세이 흰+남+보라 · ARIS 검+시안+형광연두). 주색과 그 밝은 자매만 쓰면
    한 색조의 농담이라 배경이 단벌로 읽힌다 — 산포가 "같은 도장 두 번"이 된다.

    도안 팔레트에서 **주색·차와 둘 다 색조가 벌어진** 유채색을 고르고, 없으면
    주색의 **이웃 색조**로 간다 (주색이 무채면 따뜻한 쪽 — 아래 설명).
    """
    hm = rgb_to_hsb(*main)[0]
    hc = rgb_to_hsb(*car)[0] if car is not None else None
    sc = rgb_to_hsb(*car)[1] if car is not None else 0.0

    def _gap(a: float, b: float) -> float:
        d = abs(a - b)
        return min(d, 1.0 - d)

    best, score = None, 0.0
    # **넓은 색만** 본다 (팔레트 상위 여섯) — 꼬리의 잔색은 도안에 한두 획뿐이라
    # 그것으로 배경을 칠하면 도안과 무관한 색이 면을 덮는다 (미쿠 도안에서
    # 형광 초록이 뽑혀 빨강 액센트와 부딪혔다 — 2026-08-21 미리보기 판정).
    for c in lk.palette[:6]:
        h, s, b = rgb_to_hsb(*c)
        if s < 0.30 or b < 0.28:                  # 무채·근검정은 셋째 색이 아니다
            continue
        if _gap(h, hm) < ACCENT_HUE_GAP:
            continue
        if hc is not None and sc > 0.20 and _gap(h, hc) < ACCENT_HUE_GAP:
            continue
        v = s * min(1.0, b * 1.4)
        if v > score:
            best, score = c, v
    if best is None:
        # 팔레트에 짝이 없으면 (단색 테마 도안 — 미쿠는 상위 여덟 색이 전부
        # 하늘색이다) 주색의 **이웃 색조**로 간다. 3분원(120°)을 돌리면 조화가
        # 아니라 충돌이 난다: 미쿠는 주색이 이미 베이스의 보색(빨강)이라 3분원이
        # 형광 초록으로 떨어져 빨강·초록·청록 세 색이 서로 싸웠다 (2026-08-21
        # 미리보기 판정). 이웃 색조는 주색과 한 무리로 읽히면서 농담을 준다.
        #
        # 주색이 **무채면 그 색조는 뜻이 없다** — 근검정 액센트의 h는 그 상수에
        # 든 푸른 기 때문에 0.62로 읽혀 셋째가 보라로 떨어진다. 무채 액센트를
        # 쓰는 레퍼런스 두 대의 셋째는 둘 다 따뜻한 색이다 (Cygames 86의 빨강·
        # 금, EVELYNE의 주황) — 그쪽을 씨앗으로 쓴다.
        seed = ACCENT_WARM if achromatic_accent(main) \
            else (hm + ACCENT_HUE_STEP) % 1.0
        best = tuple(int(round(v * 255))
                     for v in colorsys.hsv_to_rgb(seed, 0.78, 0.90))
    # **눌러 앉히고 차 위에서 읽히게 민다** — 배경색이 주역보다 세면 인물이 뒤로
    # 밀리고(테마 베이스와 같은 규칙), 차와 명도가 붙으면 묻힌다.
    h, s, b = rgb_to_hsb(*best)
    return readable_on(tuple(int(round(v * 255)) for v in colorsys.hsv_to_rgb(
        h, min(s, 0.80), min(0.94, max(0.55, b)))), car)


INK_LIGHT = (245, 246, 250)


INK_DARK = (22, 27, 46)


# ---------- 프레임 색 — 차 색의 반대 무채색 ----------
# 후광·톱니·지붕처럼 **차체에서 떼어 놓는 일만 하는** 잉크가 쓰는 색이다.
# 레퍼런스의 그 자리는 예외 없이 무채다 — 색을 내는 것은 산포 모티프다.
def contrast_ink(car_rgb: tuple[int, int, int] | None) -> tuple[int, int, int]:
    """차 색의 반대 무채색 (흰 아니면 근검정)."""
    b = rgb_to_hsb(*car_rgb)[2] if car_rgb else 0.35
    return INK_DARK if b > 0.62 else INK_LIGHT


# 지배 색조 → 어휘 한 벌. 캐릭터를 손으로 안 짚어도 팔레트가 결정한다.
MOTIF_THEME: tuple[tuple[float, float, str], ...] = (
    (0.92, 1.00, "flower"),        # 분홍·자홍
    (0.00, 0.06, "flower"),        # 빨강
    (0.06, 0.19, "splat"),         # 주황·노랑
    (0.19, 0.46, "flower"),        # 초록
    (0.46, 0.74, "star"),          # 청록·파랑 (결정질 — 별을 두 벌)
    (0.74, 0.92, "swirl"),         # 보라
)


# ---- 테마색 (2026-08-22 — 계열 판정의 살색 누수 수정) ----
# **살색 구간** — 색조가 주황·빨강인데 흐리고 밝다. 진짜 주황 테마색(EVELYNE의
# 주황 그래피티)은 채도가 0.7을 넘으므로 여기 안 걸린다.
SKIN_HUE = (0.02, 0.11)


SKIN_SAT_MAX = 0.55


SKIN_VAL_MIN = 0.55


# 테마색으로 인정할 **색조 무리의 면적 하한**. 이 아래는 눈 색·장신구 같은
# 디테일이라 차 한 대의 문양을 정할 근거가 못 된다.
#
# 2.5%는 실측 분포의 **틈**이다 (테스트 11장, 색조 무리 면적): 미쿠 31.6 ·
# 제로투 21.6 · 라이덴 17.1 · 히나 5.5 · 키타산 4.2 · 호시노 3.7 · 프리렌 3.5
# ↔ 사오리 1.5. 위쪽 일곱은 의상·머리색이고 사오리의 1.5%는 몇 획짜리라,
# 그 사이 어디에 놓아도 같은 답이 나온다.
THEME_AREA_MIN = 0.025


def is_skin(h: float, s: float, b: float) -> bool:
    """이 색이 **살색**인가 — 계열·액센트 후보에서 빼는 자."""
    return (SKIN_HUE[0] <= h <= SKIN_HUE[1]
            and s < SKIN_SAT_MAX and b > SKIN_VAL_MIN)


def theme_color(lk: Look) -> tuple[int, int, int] | None:
    """도안의 **테마색** — 색조 무리로 묶어 면적이 가장 넓은 유채색. 없으면 None.

    `dominant`(팔레트 1위)와 다르다. 인물 그림의 면적 1위는 거의 항상 **살색
    아니면 흰색**이라 그것으로 문양을 정하면 캐릭터와 무관해진다 (2026-08-22
    실측, 테스트 11장: 여섯은 무채라 폴백으로 떨어지고 셋은 살색 색조가
    계열을 정했다 — 실제 의상·머리색이 정한 것은 둘뿐이었다).

    **색조 무리로 묶는** 이유는 팔레트가 잘게 쪼개져 있기 때문이다 — 미쿠의
    청록은 `#09D5E2` 12.8% · `#08D3E0` 7.1% · `#30BABA` 4.6%… 로 나뉘어 있어
    한 칸씩 보면 어느 것도 넓어 보이지 않는다. 묶으면 27%다.
    """
    cand = []
    # **상위 12색만** 본다 — 계열은 넓은 색이 정한다 (액센트와 다른 자다: 상징색은
    # 작아도 되지만 문양 계열까지 꼬리 색이 정하면 도안과 무관해진다).
    for c, w in zip(lk.palette[:12], (lk.weights or [1.0] * len(lk.palette))[:12]):
        h, s, b = rgb_to_hsb(*c)
        if s < ACCENT_SRC_MIN or b < 0.20 or is_skin(h, s, b):
            continue
        cand.append((c, w, h, s, b))
    if not cand:
        return None
    best, area = None, 0.0
    for c, _w, h, s, b in cand:
        got = sum(w2 for _c2, w2, h2, _s2, _b2 in cand
                  if min(abs(h2 - h), 1.0 - abs(h2 - h)) <= BASE_HUE_NEAR)
        if got > area:
            best, area = c, got
    return best if area >= THEME_AREA_MIN else None


# ── 재질 도색 역할 (goal §26·§27) ───────────────────────────────────
#
# 사람 판 28벌의 `C_livery` 기술자 표를 그대로 읽어 보면 (`work/lab/whole`),
# **차 색 27칸 위에 사람이 더 칠하는 것은 딱 둘**이다:
#
#   body(27칸 한 벌)  28/28   ← `PaintState.set_car_color`가 이미 하는 일
#   캘리퍼            19/28   그중 11이 차체와 ΔE>25 (대비색)
#   유리 틴트         14/28
#   휠(림)             0/28   ← 아무도 안 칠한다
#   트림 따로          0/28
#
# 그래서 여기서 하는 일도 둘뿐이다. 재질 그래프를 통째로 새로 짜지 않는다 —
# 근거가 있는 자리만 칠한다.
MATERIAL_CALIPER_DE = 25.0        # 차체와 이만큼 벌어져야 "대비 캘리퍼"다
# 유리 틴트의 명도 — 위에 올린 그림과 광도차를 두려면 어두워야 한다.
MATERIAL_TINT_B = 0.14


def _lab3(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    r, g, b = (v / 255.0 for v in rgb)
    def _f(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = _f(r), _f(g), _f(b)
    x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883
    def _t(v):
        return v ** (1 / 3) if v > 216 / 24389 else (24389 / 27 * v + 16) / 116
    fx, fy, fz = _t(x), _t(y), _t(z)
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)


def material_roles(lk: Look, base_rgb: tuple[int, int, int]
                   ) -> dict[str, tuple[int, int, int]]:
    """차 색 위에 더 칠할 **재질 역할** — {역할: RGB}. 근거는 위 표다.

    캘리퍼는 도안의 액센트색이고, 차체와 안 벌어지면 무채 대비색으로 간다
    (사람 판의 대비 캘리퍼가 하는 일과 같다). 유리 틴트는 액센트의 어두운
    자매다 — 무채로 두면 그냥 짙은 유리라 구성에 안 낀다.
    """
    acc = accent_color(lk, base_rgb)
    lb = _lab3(base_rgb)
    la = _lab3(acc)
    de = math.dist(lb, la)
    caliper = acc if de >= MATERIAL_CALIPER_DE else contrast_ink(base_rgb)
    h, s, _b = rgb_to_hsb(*acc)
    tint = hsb_to_rgb(h, min(1.0, s * 0.8), MATERIAL_TINT_B)
    return {"caliper": tuple(int(v) for v in caliper),
            "glass": tuple(int(v) for v in tint)}
