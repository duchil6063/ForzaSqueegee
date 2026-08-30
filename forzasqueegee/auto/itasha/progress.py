"""진행과 시간 — 어디까지 했나, 무엇이 얼마나 걸렸나.

진행은 설정 파일 옆 `<이름>.progress.json`에 그때그때 적어 두고 이어서 한다.
`Clock`이 단계마다 초를 같은 파일에 적어 `timing_summary`가 단가를 낸다."""

from __future__ import annotations

import json
import time
from contextlib import contextmanager

from ...i18n import msg
from .config import Config


def load_progress(cfg: Config) -> dict:
    p = cfg.progress_path
    if not p.exists():
        return {"groups": {}, "placed": []}
    got = json.loads(p.read_text(encoding="utf-8"))
    got.setdefault("groups", {})
    got.setdefault("placed", [])
    return got


def save_progress(cfg: Config, prog: dict) -> None:
    cfg.progress_path.write_text(json.dumps(prog, ensure_ascii=False, indent=1),
                                 encoding="utf-8")


class Clock:
    """실행의 **단계별 시간**을 진행 파일에 그때그때 적는다.

    왜 그때그때인가: 이 기능의 한 판이 40분짜리라 중간에 죽는 일이 흔한데,
    죽어도 **어디에 시간을 썼는지는 남아야** 무엇을 고칠지 정할 수 있다.
    지금까지 "면 도형이 실행 시간의 몫을 차지한다"가 추정이었던 이유가 기록이
    없어서였다 (그룹 준비만 장당 0.44초로 실측돼 있었다).

    이름에 점이 있으면 **하위 단계**다 (`place` ⊃ `place.shape`) — 합이 겹치므로
    요약이 따로 낸다. `prog`이 없으면 아무것도 안 한다 (검사·GUI 경로).
    """

    def __init__(self, cfg: Config | None = None, prog: dict | None = None):
        self.cfg = cfg
        self.prog = prog
        if prog is not None:
            prog.setdefault("timing", [])

    def add(self, name: str, sec: float, of: str = "", n: int = 0) -> None:
        if self.prog is None:
            return
        rec: dict = {"t": name, "s": round(sec, 2)}
        if of:
            rec["of"] = of
        if n:
            rec["n"] = int(n)
        self.prog["timing"].append(rec)
        if self.cfg is not None:
            save_progress(self.cfg, self.prog)

    @contextmanager
    def stage(self, name: str, of: str = "", n: int = 0):
        """단계 하나를 잰다 — **죽어도 적는다** (죽은 자리가 곧 느린 자리다)."""
        t0 = time.time()
        try:
            yield
        finally:
            self.add(name, time.time() - t0, of=of, n=n)


def timing_summary(prog: dict) -> str:
    """진행 파일의 타이밍 → 사람이 읽는 요약 (단계마다 횟수·합·단가)."""
    recs = prog.get("timing") or []
    if not recs:
        return ""
    order: list[str] = []
    agg: dict[str, list[float]] = {}
    for r in recs:
        k = str(r.get("t"))
        if k not in agg:
            agg[k] = [0.0, 0.0, 0.0]
            order.append(k)
        a = agg[k]
        a[0] += 1
        a[1] += float(r.get("s") or 0.0)
        a[2] += float(r.get("n") or 0.0)
    lines = [msg("단계별 시간 (하위 단계는 상위에 포함된다):")]
    for k in order:
        c, s, n = agg[k]
        if n:
            per = msg("  장당 {sec:.2f}초 ({n:,.0f}장)", sec=s / n, n=n)
        elif c > 1:
            per = msg("  평균 {sec:.1f}초", sec=s / c)
        else:
            per = ""
        lines.append(msg("  {stage:<16} {count:>4.0f}회 {minutes:7.1f}분{per}",
                         stage=k, count=c, minutes=s / 60, per=per))
    top = sum(v[1] for k, v in agg.items() if "." not in k)
    lines.append(msg("  {label:<16} {pad:>4} {minutes:7.1f}분",
                     label=msg("합(상위만)"), pad="", minutes=top / 60))
    return "\n".join(lines)
