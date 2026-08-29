"""도안으로 할 수 있는 일 — 단추 여섯과 그 갈래들.

    불러오기 · 내보내기 · 오버레이 · 인게임 적용 · KFPS 편집기 · FLS 편집기

**갈래가 여럿인 일은 단추 하나로 두고 창에서 고른다** (사용자 지시 2026-08-26,
`choose`): 내보내기는 판 셋(FLS·KFPS·FS)을 한 번에 저장 위치까지 골라서,
인게임 적용은 메모리 주입이냐 파일 저장이냐. 둘 다 취소가 있다.

**편집기는 안 묻는다** (사용자 지시 2026-08-26) — 어느 편집기를 쓸지는 사람이
누르기 전에 이미 정한 것이라 단추를 둘로 두고 누르는 즉시 연다.

**인게임의 기본은 파일이다.** 창 조작은 게임 창에 키를 넣어 장당 6초씩 그리지만
(3,000장이면 5시간), 같은 값을 게임이 읽는 컨테이너 파일로 적으면 몇 초에 끝나고
게임 상태를 아예 안 건드린다. 그래서 이 창에는 창 조작 단추가 없다 — 그 경로는
CLI에만 있다 (`python -m forzasqueegee run`).
"""

from __future__ import annotations

import time

from PySide6.QtCore import QThread
from PySide6.QtWidgets import QGroupBox, QLabel, QMessageBox, QPushButton, QVBoxLayout

from ...i18n import tr
from .parts import _ApplyJob, _plan_layers


class _ApplyOps:
    """도안을 게임에 올리는 갈래.

    창 조작·주입은 길어서 작업 스레드로 가고(`parts._ApplyJob`), 중단은 플랜
    폴더의 `STOP` 파일이다 — 레이어 경계라 즉시는 아니다.

    창이 쥐고 있어야 하는 것: `plan`·`bar`·`log`·`clock`, 그리고 `_busy`·
    `_log`·`_msg` (shell)."""

    def _apply_box(self) -> QGroupBox:
        """만든(또는 불러온) 도안으로 할 수 있는 일 — 단추 여섯.

        **갈래가 여럿인 일은 단추를 하나로 두고 창에서 고른다** (사용자 지시
        2026-08-26): 내보내기는 판 셋(FLS·KFPS·FS)을 한 번에, 인게임 적용은
        메모리 주입이냐 파일 저장이냐. 둘 다 취소가 있다. 편집기 둘은 **안
        묻고 바로 연다**. 전부 같은 `plan.json`을 읽는다."""
        gb = QGroupBox(tr("gui.apply"))
        self.plan_lbl = QLabel("")
        self.plan_lbl.setWordWrap(True)

        self.b_load = QPushButton(tr("gui.load_plan"))
        self.b_load.setToolTip(tr("gui.load_plan.tip"))
        self.b_load.clicked.connect(self._pick_plan)
        self.b_export = QPushButton(tr("gui.export"))
        self.b_export.setToolTip(tr("gui.export.tip"))
        self.b_export.clicked.connect(self._export_any)
        self.b_overlay = QPushButton(tr("gui.apply.overlay"))
        self.b_overlay.clicked.connect(self._open_overlay)
        self.b_inject = QPushButton(tr("gui.apply.ingame"))
        self.b_inject.setToolTip(tr("gui.apply.ingame.tip"))
        self.b_inject.clicked.connect(self._apply_ingame)
        # 편집기는 둘이다 — KFPS(브라우저·Fabric)와 FLS(네이티브·3D 미리보기).
        # 같은 도안을 어느 쪽으로든 연다. **단추마다 제 편집기를 곧장 연다** —
        # 어느 쪽을 쓸지는 누르기 전에 정해져 있어 물어볼 것이 없다.
        self.b_kfps = QPushButton(tr("gui.editor.open.kfps"))
        self.b_kfps.setToolTip(tr("gui.editor.tip"))
        self.b_kfps.clicked.connect(self._open_editor)
        self.b_fls = QPushButton(tr("gui.editor.open.fls"))
        self.b_fls.setToolTip(tr("gui.fls.editor.tip"))
        self.b_fls.clicked.connect(self._open_fls)
        self.b_astop = QPushButton(tr("gui.apply.stop"))
        self.b_astop.setVisible(False)
        self.b_astop.clicked.connect(self._stop_apply)
        self.apply_msg = QLabel("")
        self.apply_msg.setWordWrap(True)

        lay = QVBoxLayout(gb)
        lay.addWidget(self.plan_lbl)
        lay.addWidget(self.b_load)
        lay.addWidget(self.b_export)
        lay.addWidget(self.b_overlay)
        lay.addWidget(self.b_inject)
        lay.addWidget(self.b_kfps)
        lay.addWidget(self.b_fls)
        lay.addWidget(self.b_astop)
        lay.addWidget(self.apply_msg)
        return gb

    # ---------- 갈래를 묻는 단추들 ----------
    def _apply_ingame(self) -> None:
        """인게임 적용 — 메모리 주입이냐 게임이 읽는 파일로 저장이냐.

        둘은 값이 같고 **가는 길만 다르다**: 주입은 도는 게임의 레이어 표에 바로
        쓰고(빠르지만 관리자 권한이 필요하고 게임이 떠 있어야 한다), 파일 저장은
        컨테이너를 적어 저장 폴더에 놓는다(게임을 아예 안 건드린다)."""
        from .choose import ask_one

        if self.plan is None:
            self._msg(tr("gui.apply.no_plan"), bad=True)
            return
        got = ask_one(self, "gui.apply.ingame.title", "gui.apply.ingame.head",
                      (("inject", "gui.apply.ingame.inject",
                        "gui.apply.ingame.inject.tip"),
                       ("file", "gui.apply.ingame.file",
                        "gui.apply.ingame.file.tip")))
        if got == "inject":
            self._apply("inject")
        elif got == "file":
            self._export_fls(ask_where=True)

    def _export_any(self) -> None:
        """내보내기 — 판 셋(FLS·KFPS·FS)을 **고른 자리에 한 번에** 적는다."""
        from PySide6.QtWidgets import QDialog

        from .choose import ExportDialog

        if self.plan is None:
            self._msg(tr("gui.apply.no_plan"), bad=True)
            return
        dlg = ExportDialog(self, self.plan.parent)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        kinds, where = dlg.picked()
        if not kinds:
            self._msg(tr("gui.export.none"), bad=True)
            return
        try:
            where.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            self._msg(f"{type(e).__name__}: {e}", bad=True)
            return
        done: list[str] = []
        for kind in kinds:
            got = ({"fls": self._export_fls_proj, "kfps": self._export_kfps,
                    "fs": self._export_fs}[kind])(where)
            if got:
                done.append(got)
        if done:
            self._msg(tr("gui.export.done", n=len(done),
                         names=" · ".join(done), where=self._rel(where)))

    def _export_fs(self, where) -> str | None:
        """FS 판 — 우리 `plan.json` 그대로 (다시 물리면 이 창이 곧장 읽는다)."""
        import shutil

        if self.plan is None:
            return None
        out = where / f"{self.plan.parent.name}-plan.json"
        try:
            if out.resolve() != self.plan.resolve():
                shutil.copyfile(self.plan, out)
        except BaseException as e:            # noqa: BLE001 — 창이 죽으면 안 된다
            self._msg(f"{type(e).__name__}: {e}", bad=True)
            return None
        return out.name

    def _need_template(self) -> bool | None:
        """주입 전 캔버스 장수 확인. 반환: 채울까 · None이면 **시작하지 말 것**.

        판독은 게임을 **안 건드린다** (PrintWindow 캡처라 포그라운드가 아니어도
        읽힌다) — 22분짜리를 물어보기 전에 끝난다. 못 읽으면 막지 않고 그대로
        간다 (레이어 리스트 화면이 아닐 뿐 그룹은 열려 있을 수 있다)."""
        from ...auto.template import SEC_PER_LAYER, canvas_count

        try:
            need = len(_plan_layers(self.plan))
            have = canvas_count()
        except BaseException as e:            # noqa: BLE001 — 창이 죽으면 안 된다
            self._msg(f"{type(e).__name__}: {e}", bad=True)
            return None
        if have is None:
            self._log(tr("gui.apply.inject.no_count"))
            return False
        if have > need:      # 남는 장은 주입이 캔버스 밖으로 민다 — 막지 않는다
            self._log(tr("gui.apply.inject.park", have=have, need=need,
                         extra=have - need))
            return False
        if have == need:
            return False
        ask = tr("gui.apply.inject.fill_ask", have=have, need=need,
                 miss=need - have,
                 min=max(1, round((need - have) * SEC_PER_LAYER / 60)))
        yes = QMessageBox.StandardButton.Yes
        if QMessageBox.question(self, tr("gui.apply.inject.name"), ask,
                                yes | QMessageBox.StandardButton.Cancel) != yes:
            return None
        return True

    def _apply(self, kind: str) -> None:
        from ...elevate import is_admin, need_admin
        from ...overlay.win import find_window_client_rect

        if self.plan is None or self.thread:
            return
        # 권한 먼저다 — 이건 게임이 아니라 **이 프로세스를 어떻게 켰나**의 문제라
        # 게임을 띄우고 다시 눌러 봐야 같은 데서 막힌다. 여기서 승격하면 창을
        # 새로 띄워야 해 지금까지 한 일이 날아가므로 이유만 적고 둔다.
        # 권한은 **게임을 못 열 때만** 문제다 — 게임이 승격 안 된 채로 돌면 같은
        # 무결성 수준이라 그냥 열린다 (실측). `is_admin()`만 보면 필요도 없는데 막힌다.
        if kind == "inject" and not is_admin() and need_admin():
            self._msg(tr("gui.apply.need_admin"), bad=True)
            return
        # 둘 다 게임 창이 있어야 한다 — 없으면 창 조작은 한참 뒤에 DriverError로,
        # 주입은 판독 실패로 넘어진다. 시작하기 전에 같은 문구로 막는다
        if find_window_client_rect() is None:
            self._msg(tr("overlay.game_not_found"), bad=True)
            return
        if kind == "inject" and self._need_template() is None:  # 물러섰다
            return
        (self.plan.parent / "STOP").unlink(missing_ok=True)   # 지난 중단 흔적 지우기
        self.log.clear()
        self.show_log.setChecked(True)
        what = tr(f"gui.apply.{kind}.name")
        self._msg(tr("gui.apply.running", what=what))
        # 진행률을 아는 길이 없다 (엔진처럼 콜백이 오지 않는다) — 막대를 **미정**으로
        # 돌린다. 안 그러면 방금 끝난 생성의 100%가 그대로 남아 다 된 것처럼 보인다
        self.bar.setRange(0, 0)
        self._stage_text = what
        self.t0 = time.time()
        self.clock.start()
        self._tick_clock()
        self.job_kind = kind
        self.job = _ApplyJob(kind, self.plan)
        self.thread = QThread(self)
        self.job.moveToThread(self.thread)
        self.thread.started.connect(self.job.go)
        self.job.line.connect(self._log)
        self.job.done.connect(self._apply_done)
        self.job.failed.connect(self._apply_failed)
        self._busy(True)
        self.thread.start()

    def _stop_apply(self) -> None:
        """창 조작·템플릿 채우기는 `STOP` 파일로 멈춘다 — 레이어 경계라 즉시는 아니다."""
        if self.plan is None:
            return
        (self.plan.parent / "STOP").write_text("stop", encoding="utf-8")
        self.b_astop.setEnabled(False)
        self._msg(tr("gui.apply.stopping"))
