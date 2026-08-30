"""JSON 리소스 기반 최소 i18n — 두 겹이다.

- `tr(key)` — GUI가 쓰는 열쇠 기반 사전 (`ko.json`·`en.json`). 위젯 문구처럼
  처음부터 두 벌로 짓는 것들.
- `msg(원문, **값)` — 실행 메시지용. **한국어 원문이 곧 열쇠**다: ko면 원문
  그대로라 기존 출력과 바이트 단위로 같고, 다른 언어면 `messages.<lang>.json`
  (원문 템플릿 → 번역 템플릿)에서 바꿔 낸다. 사전에 없으면 원문이 나간다 —
  빠뜨린 번역이 프로그램을 못 세운다.

언어는 `work/state/lang.json`에 못 박아 저장한다 (`save_language`) — 기본은
ko이고, 프로세스가 뜰 때 그 저장값을 읽으므로 하위 프로세스(도구 스크립트·
UAC 승격 재실행)도 같은 언어로 말한다. `--lang`은 이번 실행만 덮는다.

사용:
    from forzasqueegee.i18n import tr, msg, set_language
    set_language("ko")  # 또는 "en"
    tr("app.title")
    msg("읽기 실패: {path}", path=p)
"""

from __future__ import annotations

import json
from pathlib import Path

_DIR = Path(__file__).parent
_FALLBACK = "en"

LANGUAGES = ("ko", "en")


def _settings_file() -> Path:
    # paths.work_root()와 같은 자리 — 순환을 피해 직접 짚는다 (paths는 i18n을 모른다)
    return _DIR.parents[1] / "work" / "state" / "lang.json"


def saved_language() -> str | None:
    """못 박아 둔 언어 — 없거나 모르는 값이면 None."""
    try:
        raw = json.loads(_settings_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    lang = raw.get("lang")
    return lang if lang in LANGUAGES else None


def save_language(lang: str) -> None:
    """언어를 못 박아 저장하고 이 프로세스에도 바로 적용한다."""
    if lang not in LANGUAGES:
        raise ValueError(f"unknown language: {lang}")
    p = _settings_file()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"lang": lang}, ensure_ascii=False, indent=1),
                 encoding="utf-8")
    set_language(lang)


_language = saved_language() or "ko"
_catalogs: dict[str, dict[str, str]] = {}
_messages: dict[str, dict[str, str]] = {}


def _load(lang: str) -> dict[str, str]:
    if lang not in _catalogs:
        path = _DIR / f"{lang}.json"
        _catalogs[lang] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return _catalogs[lang]


def _load_messages(lang: str) -> dict[str, str]:
    if lang not in _messages:
        path = _DIR / f"messages.{lang}.json"
        _messages[lang] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return _messages[lang]


def set_language(lang: str) -> None:
    global _language
    _language = lang


def current_language() -> str:
    return _language


def tr(key: str, **kwargs: object) -> str:
    text = _load(_language).get(key) or _load(_FALLBACK).get(key) or key
    return text.format(**kwargs) if kwargs else text


def msg(text: str, **kwargs: object) -> str:
    """실행 메시지 — 한국어 원문이 열쇠다. 값이 있으면 템플릿(`{name}`)이다."""
    if _language != "ko":
        text = _load_messages(_language).get(text, text)
    return text.format(**kwargs) if kwargs else text
