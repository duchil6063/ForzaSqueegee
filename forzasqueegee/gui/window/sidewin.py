"""형제 창 — 오버레이 · 내장 KFPS 편집기.

**이타샤는 여기 없다** — 내장 FLS 편집기의 [Itasha] 메뉴가 짓는다
(`engine.fls.studio`). 이 창에서 그리로 가는 길은 [FLS 편집기 열기]다."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt

from ...i18n import tr
from .parts import ROOT


class _SideWindows:
    """제 창으로 뜨는 것들을 여닫는 갈래.

    Qt 창은 주 스레드에서만 살기 때문에 이것들은 **스레드가 아니라 이 프로세스의
    형제 창**이다. 편집기는 [Export JSON] 마커를 1초마다 봐서 편집본을 문다.

    창이 쥐고 있어야 하는 것: `plan`·`job_kind`·`editor_watch`, 그리고 `_msg`
    (shell)."""

    def _open_overlay(self) -> None:
        """게임 창 위에 오버레이를 띄운다 — 이 프로세스의 창이다 (스레드 아님)."""
        from ...overlay.guide import open_guide
        from ...overlay.win import find_window_client_rect

        if self.plan is None:
            return
        if find_window_client_rect() is None:
            self._msg(tr("overlay.game_not_found"), bad=True)
            return
        if self.guide is not None:
            if self.guide_plan == self.plan:  # 같은 도안이면 앞으로만 올린다
                self.guide[1].raise_()
                self.guide[1].activateWindow()
                return
            # 다른 도안을 물었다 — 옛 오버레이를 닫고 새로 띄운다. 옛 패널의
            # `destroyed`(deleteLater라 늦게 온다)를 먼저 끊는다 — 안 끊으면
            # 그것이 나중에 도착해 **새 오버레이의 참조를** 놓아 버린다
            old = self.guide[1]
            old.destroyed.disconnect(self._guide_closed)
            old.close()
            self.guide = self.guide_plan = None
        try:
            guide = open_guide(self.plan)
        except BaseException as e:            # noqa: BLE001
            self._msg(f"{type(e).__name__}: {e}", bad=True)
            return
        self.guide, self.guide_plan = guide, self.plan
        # 패널을 닫으면 지워지게 해 둔다 — 그래야 다시 눌렀을 때 새로 뜬다
        guide[1].setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        guide[1].destroyed.connect(self._guide_closed)
        self._msg(tr("gui.apply.overlay.opened"))

    def _guide_closed(self) -> None:
        self.guide = self.guide_plan = None

    def _offer_kfps(self) -> bool:
        """편집기 도형 리소스가 없다 — 받을지 묻고, 받겠다면 그 자리에서 받는다.

        게임 도형 메시라 저장소에 안 싣고 KFPS 고정 커밋에서 받는다
        (`tools/get_kfps.py`). FLS 편집기를 받는 길과 같은 모양이다.
        """
        import subprocess
        import sys

        from PySide6.QtWidgets import QMessageBox

        btn = QMessageBox.question(
            self, tr("gui.editor.name"), tr("gui.kfps.missing"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel)
        if btn != QMessageBox.StandardButton.Yes:
            return False
        self._msg(tr("gui.kfps.downloading"))
        self.log.clear()
        self.show_log.setChecked(True)
        try:
            r = subprocess.run([sys.executable, str(ROOT / "tools" / "get_kfps.py")],
                               capture_output=True, text=True, timeout=1800)
            self._log((r.stdout or "") + (r.stderr or ""))
        except BaseException as e:            # noqa: BLE001 — 창이 죽으면 안 된다
            self._msg(f"{type(e).__name__}: {e}", bad=True)
            return False
        from ... import kfpseditor

        if not kfpseditor.resources_available():
            self._msg(tr("gui.kfps.download_failed"), bad=True)
            return False
        self._msg(tr("gui.kfps.downloaded"))
        return True

    def _open_editor(self) -> None:
        """내장 KFPS 편집기 — 물고 있는 도안을 프로젝트로 구워 연다.

        도안이 없으면 빈 캔버스다 (편집기 안 [Import JSON]에 out/의 도안이
        전부 나온다 — 셀·선화·페인터가 같은 스키마라 어느 것이든 열린다).
        편집기의 [Export JSON]은 서버가 곧장 도안으로 되바꾸고, 이 창이
        마커를 보고 그 자리에서 문다 (`_poll_editor_export`).

        **자동 복구본이 있으면 먼저 묻는다** — 도안을 물려 열면 편집기가 그것을
        지운다. 지난번에 창이 죽었다면 그 편집이 거기 있다."""
        from ... import kfpseditor

        if not kfpseditor.resources_available() and not self._offer_kfps():
            return
        try:
            srv = kfpseditor.ensure_server()
        except (OSError, FileNotFoundError) as e:
            self._msg(f"{type(e).__name__}: {e}", bad=True)
            return
        if not self.editor_watch.isActive():
            m = kfpseditor.read_change_marker()   # 지난 세션의 Export는 안 문다
            self._editor_seen_ns = int(m.get("changed_at_ns", 0)) if m else 0
            self.editor_watch.start()
        project = None
        note = None                           # 복구본을 열었으면 그 말이 이긴다
        if self.plan is not None:
            rec = kfpseditor.recovery_state()
            keep = self._ask_recovery(rec) if rec else False
            if keep is None:                  # 취소 — 복구본을 안 건드린다
                return
            try:
                project = (kfpseditor.recover_project() if keep
                           else kfpseditor.stage_plan_project(self.plan))
            except BaseException as e:        # noqa: BLE001 — 창이 죽으면 안 된다
                self._msg(f"{type(e).__name__}: {e}", bad=True)
                return
            if keep:
                note = tr("gui.editor.recover.opened", name=project)
        url = kfpseditor.editor_url(srv, project)
        try:
            from ..kfps_editor_window import EditorWindow
        except ImportError:                   # QtWebEngine이 없다 — 브라우저로
            import webbrowser

            webbrowser.open(url)
            self._msg(note or tr("gui.editor.opened_browser"))
            return
        if self.editor is not None:
            self.editor.raise_()
            self.editor.activateWindow()
            if project:                       # 다른 도안을 물렸다 — 갈아 끼운다
                self.editor.load_url(url)
            if note:
                self._msg(note)
            return
        self.editor = EditorWindow(url, kfpseditor.STATE / "webprofile")
        self.editor.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.editor.destroyed.connect(self._editor_closed)
        self.editor.show()
        self._msg(note or tr("gui.editor.opened"))

    def _ask_recovery(self, rec: dict) -> bool | None:
        """복구본을 열까, 도안을 열까 — `True`/`False`, 취소면 `None`.

        도안을 고르면 편집기가 여는 길에 복구본을 지운다 (되돌릴 수 없다).
        그래서 기본 단추는 복구본 쪽이다."""
        from PySide6.QtWidgets import QMessageBox

        box = QMessageBox(self)
        box.setWindowTitle(tr("gui.editor.recover.name"))
        box.setText(tr("gui.editor.recover.ask", name=rec["name"] or "?",
                       n=rec["layers"],
                       when=rec["saved_at"].replace("T", " ")[:19]))
        take = box.addButton(tr("gui.editor.recover.open"),
                             QMessageBox.ButtonRole.AcceptRole)
        drop = box.addButton(tr("gui.editor.recover.discard"),
                             QMessageBox.ButtonRole.DestructiveRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(take)
        box.exec()
        clicked = box.clickedButton()
        return True if clicked is take else (False if clicked is drop else None)

    def _editor_closed(self) -> None:
        self.editor = None

    def _poll_editor_export(self) -> None:
        """편집기 [Export JSON] 마커 — 새 변환이 서면 그 도안을 문다."""
        from ... import kfpseditor

        m = kfpseditor.read_change_marker()
        if not m:
            return
        ns = int(m.get("changed_at_ns", 0))
        if ns <= self._editor_seen_ns:
            return
        if self.job_kind is not None:         # 도는 중 — 끝나면 다음 틱에 문다
            return
        self._editor_seen_ns = ns
        name = str(m.get("name", ""))
        if m.get("error"):
            self._msg(tr("gui.editor.import_failed", name=name,
                         msg=m["error"]), bad=True)
            return
        p = Path(str(m.get("path", "")))
        if not p.is_file():
            return
        self._set_plan(p)
        if self.plan is not None:
            self._show_plan_assets(p)
            self._msg(tr("gui.editor.imported", name=name,
                         n=int(m.get("layers", 0)),
                         out=str(p.parent.relative_to(ROOT))))
