"""JSON 리소스 기반 최소 i18n.

사용:
    from forzasqueegee.i18n import tr, set_language
    set_language("ko")  # 또는 "en"
    tr("app.title")
"""

from __future__ import annotations

import json
from pathlib import Path

_DIR = Path(__file__).parent
_FALLBACK = "en"

_language = "ko"
_catalogs: dict[str, dict[str, str]] = {}


def _load(lang: str) -> dict[str, str]:
    if lang not in _catalogs:
        path = _DIR / f"{lang}.json"
        _catalogs[lang] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return _catalogs[lang]


def set_language(lang: str) -> None:
    global _language
    _language = lang


def current_language() -> str:
    return _language


def tr(key: str, **kwargs: object) -> str:
    text = _load(_language).get(key) or _load(_FALLBACK).get(key) or key
    return text.format(**kwargs) if kwargs else text
