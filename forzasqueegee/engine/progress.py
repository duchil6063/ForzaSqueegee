"""진행 막대의 눈금 — **시간이 눈금이다.**

막대가 고르게 차야 사람이 "얼마나 남았나"를 읽는다. 그러려면 단계를 **세면
안 되고**, 단계가 실제로 무는 **시간의 몫**을 세야 한다. 종전 눈금은 단계를
셌고, 그래서 판 시간의 41%가 지나서야 막대가 0.10에 닿고 66%에서 0.80에
닿았다 (실측 11번 판). 사람이 볼 때는 오래 멈춰 있다가 갑자기 뛰는 막대다.

몫은 **판마다 다르다.** 앞단(SR + 선화)은 판이 커져도 거의 안 늘어 40초 언저리
로 고정인데 나머지는 내용에 비례한다 — 그래서 같은 앞단이 150초 판에서는
25%, 312초 판에서는 13%다. 어느 한 수를 박아 두면 둘 중 하나는 반드시 틀린다.

그래서 **초로 예측하고 그때그때 몫으로 환산한다.** 기본값은 두 판 실측의
평균이고, 판을 하나 구울 때마다 실제로 문 시간을 배워 둔다
(`work/state/stagetime.json`, EMA). PC가 느리거나 사람이 늘 비슷한 그림을
넣으면 몇 판 만에 그 사람의 수로 수렴한다.

**남은 시간은 창이 따로 셈하지 않는다.** 막대가 예측 시간 몫이면
`경과 × (1 − 몫) / 몫`이 곧 "지금까지의 속도로 남은 예측 시간"이다 — 예측이
어긋나도 경과가 그만큼 커져 저절로 보정된다. 그래서 창은 그 식 하나만 쓴다
(`gui/window/make._tick_clock`). 눈금이 단계 수였을 때 그 식이 거짓말이 된
까닭도 같다.
"""

from __future__ import annotations

import json
import time

from ..paths import work_file

# 단계와 **기본 예측 초** — 두 판 실측의 평균이다 (11번 150초 · 01번 312초,
# 단독 실행). 첫 판이 이 수로 서고 그 뒤로는 배운 수가 이긴다.
#
#              11번    01번   → 기본
#   prep       38.2    39.2     39   앞단 (SR + 선화 3판)
#   celart      7.3    14.4     11   셀 재해석
#   line       21.4    71.4     46   선 도안
#   fill       21.5    32.8     27   도형 배치
#   focus      18.2    28.2     23   잔차 초점
#   recut       9.3    73.9     42   재컷·메움·수리·사후 가격
#   ft         25.9    38.7     32   전역 미세 조정
#   seal        7.9    12.6     10   봉인
#   write       0.7     0.9      4   미리보기·파일 쓰기
CEL = (("prep", 39.0), ("celart", 11.0), ("line", 46.0), ("fill", 27.0),
       ("focus", 23.0), ("recut", 42.0), ("ft", 32.0), ("seal", 10.0),
       ("write", 4.0))

# line 노선 — 앞단은 두 노선이 같은 기계라 같은 수다. 획 배치는 cel의 `line`
# 단과 같은 기계라 그 수를 물려받고(46), 미세 조정은 스택이 획뿐이라 cel의
# 절반쯤으로 본다. **첫 판용 어림이다** — 실측은 11번 판 하나뿐이고(앞단 36 ·
# 획 배치 23 · 미세 조정 11 · 쓰기 2.6, 총 61초) 그 판이 테스트 세트에서 선이
# 가장 성기다. 한 판만 구우면 이 수는 그 사람의 수로 갈린다
LINE = (("prep", 38.0), ("draw", 46.0), ("ft", 16.0), ("write", 3.0))

# **판 크기에 안 따라가는 단.** 앞단(SR + 선화)은 판이 커져도 거의 같은 초를
# 물고, 파일 쓰기도 그렇다. 나머지는 내용(획·영역·도형 수)에 비례한다 —
# 그래서 "이 판이 예측보다 큰가"를 **비례하는 단으로만** 재고, 그 배수를
# 남은 비례 단에만 건다. 앞단까지 같이 늘리면 짧은 판에서 막대가 앞단 내내
# 뒤처진다 (실측 131초 판: 막대 0.20이 시간 0.35에 섰다).
_FIXED = ("prep", "write")

_STORE = ("state", "stagetime.json")
_ALPHA = 0.35            # 새 실측이 예측을 끄는 무게 (EMA)
_MIN, _MAX = 0.2, 3600.0  # 배운 수의 상하한 — 0이나 폭주는 눈금을 망친다


def _load(key: str, spec: tuple) -> dict:
    """배운 초 (없으면 기본값). 읽기 실패는 무해하다 — 기본값으로 선다."""
    pred = {n: s for n, s in spec}
    try:
        got = json.loads(work_file(*_STORE).read_text(encoding="utf-8"))
        for n, v in (got.get(key) or {}).items():
            if n in pred and isinstance(v, (int, float)):
                pred[n] = min(_MAX, max(_MIN, float(v)))
    except Exception:                      # noqa: BLE001 — 눈금은 없어도 돈다
        pass
    return pred


def _save(key: str, pred: dict, marks: dict) -> None:
    """실측을 예측에 섞어 적는다 (EMA). **판 하나가 눈금을 뒤엎지 않는다.**"""
    if not marks:
        return
    try:
        path = work_file(*_STORE)
        try:
            got = json.loads(path.read_text(encoding="utf-8"))
        except Exception:                  # noqa: BLE001
            got = {}
        if not isinstance(got, dict):
            got = {}
        cur = dict(got.get(key) or {})
        for n, sec in marks.items():
            base = float(cur.get(n, pred.get(n, 1.0)))
            cur[n] = round(min(_MAX, max(_MIN,
                                         (1 - _ALPHA) * base + _ALPHA * sec)), 2)
        got[key] = cur
        path.write_text(json.dumps(got, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    except Exception:                      # noqa: BLE001 — 못 적어도 도안은 났다
        pass


class Clock:
    """단계 → 막대 몫. 노선이 단을 열고(`enter`) 엔진에는 `sub`를 넘긴다.

    `sink`가 None이면(명령줄) 아무것도 안 알리되 **시간은 그대로 배운다** —
    다음 판의 눈금이 그만큼 좋아진다.
    """

    def __init__(self, key: str, spec: tuple, sink=None) -> None:
        self.key, self.sink = key, sink
        self.pred = _load(key, spec)
        self.order = [n for n, _ in spec]
        acc = 0.0
        self.before: dict[str, float] = {}
        for n in self.order:
            self.before[n] = acc
            acc += self.pred[n]
        self.total = acc or 1.0
        self.t0 = time.time()
        self.marks: dict[str, float] = {}
        self.scale = 1.0                   # 이 판이 예측보다 몇 배인가
        self._cur: str | None = None
        self._cur_t0 = self.t0
        self._last = 0.0                   # 막대는 **뒤로 안 간다**

    def _sec(self, name: str) -> float:
        """그 단의 남은 예측 초 — 비례 단에는 이 판에서 잰 배수를 건다."""
        return self.pred[name] * (1.0 if name in _FIXED else self.scale)

    def _pos(self, name: str, u: float, thru: str = "") -> float:
        """막대 몫 = (지나온 **실제** 초 + 지금 구간의 예측 몫) / 예측 총 초.

        지나온 쪽은 예측이 아니라 **실측**이라, 예측이 틀렸어도 그만큼 총합에
        반영된다 — 그래서 판이 클수록 막대가 저절로 느려지고 작을수록 빨라진다.

        `thru`는 한 콜백이 여러 단을 덮는 자리다 (`sub` 문서) — 그 구간 전체가
        `u`의 몫이 된다.
        """
        done = self._cur_t0 - self.t0
        i = self.order.index(name)
        j = self.order.index(thru) if thru in self.before else i
        span = sum(self._sec(n) for n in self.order[i:max(i, j) + 1])
        # **예측을 넘긴 단은 넘긴 만큼이 제 길이다.** 안 그러면 오래 끄는 단이
        # 다음 단의 문턱에 먼저 도착해 거기서 멈춘 채로 남는다 — 사람이 보기에
        # 딱 옛 막대의 증상이다. 남은 시간 어림도 이 쪽이 정직하다
        span = max(span, time.time() - self._cur_t0)
        rest = sum(self._sec(n) for n in self.order[max(i, j) + 1:])
        tot = done + span + rest
        return (done + span * max(0.0, min(1.0, u))) / tot if tot > 0 else 0.0

    def _emit(self, f: float, label: str) -> None:
        f = max(self._last, min(0.999, f))
        self._last = f
        if self.sink:
            self.sink(f, label)

    def _close_cur(self) -> None:
        if self._cur is None:
            return
        self.marks[self._cur] = time.time() - self._cur_t0
        self._cur = None
        # 비례 단만으로 이 판의 배수를 다시 잰다 (`_FIXED` 문서)
        got = [n for n in self.marks if n not in _FIXED]
        pred = sum(self.pred[n] for n in got)
        if pred > 0:
            self.scale = min(8.0, max(0.15, sum(self.marks[n] for n in got) / pred))

    def enter(self, name: str, label: str = "") -> None:
        """단 하나를 연다 — 앞 단은 여기서 닫히고 실측이 적힌다."""
        self._close_cur()
        self._cur, self._cur_t0 = name, time.time()
        self._emit(self._pos(name, 0.0), label)

    def sub(self, name: str, label: str = "", thru: str = ""):
        """그 단 **안의** 0~1을 받는 콜백 (엔진 함수에 그대로 넘긴다).

        엔진 쪽 서명이 둘이다 — `(f, 단계이름)`과 `(f)`. 둘 다 받는다.

        `thru`를 주면 그 단**까지**를 한 콜백이 덮는다 — 폴백 경로에서 두 단이
        한 함수로 합쳐지는 자리가 있다 (선화가 없으면 선·면을 `fit_plan`이
        함께 배치한다). 안 주면 자기 단만 덮는다.
        """
        self.enter(name, label)
        end = thru if thru in self.before else name

        def cb(f, t: str = "") -> None:
            self._emit(self._pos(name, float(f), end), t or label)
        return cb

    def close(self) -> None:
        """마지막 단을 닫고 배운 것을 적는다 — **끝까지 간 판에서만** 부른다."""
        self._close_cur()
        _save(self.key, self.pred, self.marks)
