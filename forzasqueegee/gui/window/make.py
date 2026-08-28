"""도안 만들기 — 값을 세우고 `engine.pipeline.make()`를 스레드에 건다."""

from __future__ import annotations

import time
from pathlib import Path

from PySide6.QtCore import QThread, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QFileDialog

from ...engine.pipeline import MAX_SHAPES, make
from ...i18n import tr
from .parts import ROOT, _Job, _root_out


class _MakeOps:
    """도안 만들기 갈래 — 생성 조건·시작·진행·완료.

    창이 쥐고 있어야 하는 것: 입력 위젯(`drop`·`checks`·`route`·`out_edit`·
    `bgcut`), 진행 표시(`bar`·`log`·`hint`·`clock`), 단추(`go`·`open`), 그리고
    `_busy`·`_log`·`_msg` (shell)."""

    def set_image(self, path: str) -> None:
        self.image = Path(path)
        self.drop.show_image(self.image)
        self.src_pane.load(self.image)
        self.out_pane.clear_image(tr("gui.preview"))
        self.out = _root_out(self.image)
        self.out_edit.setText(str(self.out))
        self.verdict.setText("")
        self.checks.setText("")
        self.open.setVisible(False)
        self.bar.setValue(0)
        self.stage.setText("")
        self._sync_go()

    def _ready(self) -> bool:
        """생성이 열리는 조건 — 셀은 이미지만, 페인터는 레이어 수까지."""
        if not self.image:
            return False
        return not self.r_painter.isChecked() or self.shapes.value() >= 1

    def _route(self) -> str:
        if self.r_cel.isChecked():
            return "cel"
        return "line" if self.r_line.isChecked() else "painter"

    def _sync_go(self, *_a) -> None:
        busy = self.job_kind is not None      # 스레드를 만들기 전부터 잠근다
        self.go.setEnabled(not busy and self._ready())
        route = self._route()
        # 페인터만 장수를 사람이 넣는다. 셀은 장수 칸이 아예 없다 —
        # 가격이 정하므로 넣을 수가 없다
        self.shapes.setVisible(route == "painter")
        self.hint.setText(tr(f"gui.route.{route}.hint") if route != "painter"
                          else tr("gui.route.painter.need_shapes")
                          if self.shapes.value() < 1
                          else tr("gui.route.painter.hint"))
        for b in (self.b_export, self.b_overlay, self.b_inject):
            b.setEnabled(not busy and self.plan is not None)
        # 편집기 둘은 **도안 없이도 연다** — 창 안에서 고르면 된다
        self.b_kfps.setEnabled(not busy)
        self.b_fls.setEnabled(not busy)
        self.b_load.setEnabled(not busy)

    def _pick_out(self) -> None:
        d = QFileDialog.getExistingDirectory(self, tr("gui.pick_out"),
                                             str(self.out or ROOT / "out"))
        if d:
            self.out = Path(d)
            self.out_edit.setText(d)

    def _start(self) -> None:
        if not self._ready() or self.thread:
            return
        self.job_kind = "make"
        self._busy(True)
        self.log.clear()
        self.bar.setRange(0, 1000)
        self.bar.setValue(0)
        self._stage_text = ""
        self.verdict.setText("")
        self.checks.setText("")
        self.open.setVisible(False)
        self._set_plan(None)
        self.t0 = time.time()
        self.clock.start()

        route = self._route()
        shapes = self.shapes.value() if route == "painter" else MAX_SHAPES
        self.job = _Job(self.image, self.out, route, shapes,
                        keep_bg=not self.bgcut.isChecked())
        self.thread = QThread(self)
        self.job.moveToThread(self.thread)
        self.thread.started.connect(self.job.go)
        self.job.line.connect(self._log)
        self.job.step.connect(self._step)
        self.job.done.connect(self._done)
        self.job.failed.connect(self._failed)
        self.thread.start()

    def _cancel(self) -> None:
        if self.job:
            self.job.stop = True
            self.stop.setEnabled(False)
            self.stage.setText(tr("gui.cancelling"))

    def _finish(self) -> None:
        self.clock.stop()
        self.bar.setRange(0, 1000)       # 적용이 미정 막대로 돌려놨을 수 있다
        if self.thread:
            self.thread.quit()
            self.thread.wait()
            self.thread.deleteLater()
        self.thread = self.job = self.job_kind = None
        self._busy(False)

    def _tick_clock(self) -> None:
        """경과 시간. **시계가 도는 동안만** — 끝난 뒤의 완료 문구를 덮으면 안 된다."""
        if not self.clock.isActive():
            return
        el = time.time() - self.t0
        self.stage.setText(tr("gui.stage", stage=self._stage_text,
                              min=int(el // 60), sec=int(el % 60)))

    def _open_folder(self) -> None:
        if self.out and self.out.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.out)))
