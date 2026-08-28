"""내장 KFPS 편집기 창 — 로컬 편집기 서버를 QWebEngineView로 문 창.

편집기는 브라우저용으로 지어져 있다 (localStorage에 즐겨찾기·단축키,
`beforeunload`로 저장 안 한 작업 경고). 그 규약이 그대로 서도록:

- **프로필을 디스크에 박는다** (`work/editor/webprofile`) — 기본(off-the-record)
  프로필이면 localStorage가 메모리뿐이라 창을 닫을 때마다 즐겨찾기가 백지가 된다.
- **닫기는 `RequestClose`로 묻는다** — 편집기의 beforeunload가 살아서, 저장
  안 한 편집이 있으면 크로뮴의 나가기 확인이 뜬다. 허락하면 그때 닫는다.
- **새 창 요청(도움말 등)은 바깥 브라우저로** 보낸다 — 이 창은 편집기 하나다.

QtWebEngine을 못 들여오는 환경(꾸러미 훼손 등)은 부르는 쪽이 ImportError를
받아 기본 브라우저로 물러난다 — 편집 자체는 어느 쪽에서든 같은 서버다.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QUrl, Slot
from PySide6.QtGui import QDesktopServices
from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineProfile
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget

from ..i18n import tr

_profile: QWebEngineProfile | None = None


def _shared_profile(profile_dir: Path) -> QWebEngineProfile:
    """앱 수명 프로필 싱글턴 — 창보다 오래 산다.

    프로필이 페이지보다 먼저 죽으면 크로뮴이 넘어진다. 창을 몇 번을 여닫아도
    페이지만 창과 함께 죽고 프로필은 QApplication에 붙어 남는다."""
    global _profile
    if _profile is None:
        profile_dir.mkdir(parents=True, exist_ok=True)
        _profile = QWebEngineProfile("kfps-editor", QApplication.instance())
        _profile.setPersistentStoragePath(str(profile_dir / "storage"))
        _profile.setCachePath(str(profile_dir / "cache"))
    return _profile


class EditorWindow(QWidget):
    def __init__(self, url: str, profile_dir: Path):
        super().__init__()
        self.setWindowTitle(tr("gui.editor.window_title"))
        self.resize(1440, 900)
        self._close_ok = False

        self.page = QWebEnginePage(_shared_profile(profile_dir), self)
        self.page.newWindowRequested.connect(self._new_window)
        self.page.windowCloseRequested.connect(self._close_granted)
        self.view = QWebEngineView(self)
        self.view.setPage(self.page)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.view)
        self.view.setUrl(QUrl(url))

    def load_url(self, url: str) -> None:
        """다른 도안을 물릴 때 — 페이지를 갈아 끼운다 (편집기의 beforeunload가
        저장 안 한 작업을 지키는 것은 브라우저 탭과 같다)."""
        self.view.setUrl(QUrl(url))

    @Slot()
    def _new_window(self, request) -> None:
        QDesktopServices.openUrl(request.requestedUrl())

    @Slot()
    def _close_granted(self) -> None:
        self._close_ok = True
        self.close()

    def closeEvent(self, event) -> None:
        """편집기의 나가기 확인을 존중한다 — 먼저 페이지에 닫기를 청하고,
        페이지가 허락(`windowCloseRequested`)하면 그때 실제로 닫는다."""
        if self._close_ok:
            super().closeEvent(event)
            return
        event.ignore()
        self.page.triggerAction(QWebEnginePage.WebAction.RequestClose)
