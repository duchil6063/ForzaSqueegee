"""제품 창 본체 — 화면을 짓고, 갈래들이 함께 쓰는 것을 쥔다.

`__init__`이 위젯을 전부 만들고 신호를 잇는다. 갈래 믹스인(`make`·`plan`·
`apply`·`sidewin`)이 그 위젯을 이름으로 쓰므로, **여기서 만든 이름이 곧 그들과의
계약**이다 (각 믹스인 문서열에 무엇을 쓰는지 적어 두었다). 공용은 셋뿐이다 —
`_busy`(무엇이 돌든 입력을 잠근다)·`_log`·`_msg`.

## `@Slot`은 믹스인에 두면 안 된다 (2026-08-26 실측)

`_log`·`_step`·`_done`·`_failed`·`_apply_done`·`_apply_failed`는 갈래로 보면
생성(`make`)과 적용(`apply`)의 것이지만 **여기** 있다. 작업 스레드가 보내는
신호를 이 창이 받으려면 PySide6가 받는 쪽 QObject를 알아야 큐로 넘기는데, 슬롯이
**QObject가 아닌 평범한 믹스인**에 있으면 그 추론이 안 돼 연결이 직결(direct)로
서고 핸들러가 **보내는 쪽 스레드에서** 돈다 — 거기서 위젯을 만지면 프로세스가
통째로 죽는다. 메타오브젝트에 슬롯으로 **등록은 되므로** 등록 여부로는 못 가른다
(이 규칙을 어겼을 때 실제로 세그폴트를 냈다 — 아래에 그
실측을 적어 두었다)."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QThread, QTimer, Slot
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ...engine.pipeline import MAX_SHAPES, make
from ...i18n import current_language, msg, save_language, tr
from ...overlay.win import set_dpi_aware
from ...paths import find_run_file
from .parts import _ApplyJob, _Drop, _Job, _Pane
from .make import _MakeOps
from .plan import _PlanOps
from .apply import _ApplyOps
from .flsops import _FlsOps
from .sidewin import _SideWindows


class MakeWindow(_MakeOps, _PlanOps, _ApplyOps, _FlsOps, _SideWindows, QWidget):
    def __init__(self, image: str | Path | None = None):
        super().__init__()
        self.setWindowTitle(tr("gui.title"))
        self.resize(1000, 720)
        self.image: Path | None = None
        self.out: Path | None = None
        self.plan: Path | None = None
        self.thread: QThread | None = None
        self.job: _Job | _ApplyJob | None = None
        self.job_kind: str | None = None
        self.guide: tuple | None = None    # 오버레이·패널 참조 (놓으면 닫힌다)
        self.guide_plan: Path | None = None   # 그 오버레이가 문 도안
        self.editor = None                 # 내장 KFPS 편집기 창 (놓으면 닫힌다)
        self._editor_seen_ns = 0           # 이미 문 편집기 Export 마커 시각
        self.t0 = 0.0
        self._stage_text = ""
        self._restart = False              # 언어 변경 — `run`의 재시작 루프가 본다

        self.drop = _Drop()
        self.drop.picked.connect(self.set_image)

        # 배경 자동 제거 — 알파 없는 입력에만 발동하는 전처리라 평소엔 무해하지만,
        # 배경까지 그리려는 사용자는 여기서만 끌 수 있다 (CLI --keep-bg와 같다)
        self.bgcut = QCheckBox(tr("gui.bgcut"))
        self.bgcut.setChecked(True)
        self.bgcut.setToolTip(tr("gui.bgcut.tip"))

        self.r_painter = QRadioButton(tr("gui.route.painter"))
        self.r_cel = QRadioButton(tr("gui.route.cel"))
        self.r_line = QRadioButton(tr("gui.route.line"))
        self.r_cel.setChecked(True)        # 셀이 기본 — 사람 방식 재현 (가격 설계)
        self.r_cel.setToolTip(tr("gui.route.cel.tip"))
        self.r_line.setToolTip(tr("gui.route.line.tip"))
        self.r_painter.toggled.connect(self._sync_go)
        self.r_cel.toggled.connect(self._sync_go)
        self.r_line.toggled.connect(self._sync_go)
        # **페인터는 레이어 수를 넣어야 생성이 열린다.** 0 = 아직 안 넣은 상태이고
        # 그때는 특수 문구가 보인다 — 기본값을 3,000으로 두면 36~57분짜리가
        # 사람이 고른 적 없는 수로 돌아 버린다
        self.shapes = QSpinBox()
        self.shapes.setRange(0, MAX_SHAPES)
        self.shapes.setValue(0)
        self.shapes.setSpecialValueText(tr("gui.shapes.empty"))
        self.shapes.setSingleStep(100)
        self.shapes.setSuffix(tr("gui.shapes.suffix"))
        self.shapes.valueChanged.connect(self._sync_go)
        self.hint = QLabel("")
        self.hint.setWordWrap(True)

        box = QGroupBox(tr("gui.route"))
        row = QHBoxLayout()
        row.addWidget(self.r_cel)
        row.addWidget(self.r_line)
        row.addWidget(self.r_painter)
        row.addWidget(self.shapes)
        row.addStretch(1)
        blay = QVBoxLayout(box)
        blay.addLayout(row)
        blay.addWidget(self.hint)

        self.out_edit = QLineEdit(readOnly=True)
        b_out = QPushButton("…")
        b_out.setFixedWidth(32)
        b_out.clicked.connect(self._pick_out)
        orow = QHBoxLayout()
        orow.addWidget(QLabel(tr("gui.out")))
        orow.addWidget(self.out_edit, 1)
        orow.addWidget(b_out)

        self.go = QPushButton(tr("gui.make"))
        self.go.setMinimumHeight(40)
        self.go.setEnabled(False)
        self.go.clicked.connect(self._start)
        self.stop = QPushButton(tr("gui.cancel"))
        self.stop.setMinimumHeight(40)
        self.stop.setVisible(False)
        self.stop.clicked.connect(self._cancel)
        grow = QHBoxLayout()
        grow.addWidget(self.go, 1)
        grow.addWidget(self.stop)

        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.stage = QLabel("")
        self.verdict = QLabel("")
        self.verdict.setWordWrap(True)
        self.checks = QLabel("")
        self.checks.setWordWrap(True)
        self.checks.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        self.open = QPushButton(tr("gui.open_folder"))
        self.open.setVisible(False)
        self.open.clicked.connect(self._open_folder)

        left = QVBoxLayout()
        left.addWidget(self.drop)
        left.addWidget(self.bgcut)
        left.addWidget(box)
        left.addLayout(orow)
        left.addLayout(grow)
        left.addWidget(self.bar)
        left.addWidget(self.stage)
        left.addWidget(self.verdict)
        left.addWidget(self.checks)
        left.addWidget(self.open)
        left.addWidget(self._apply_box())
        left.addStretch(1)

        # 언어 — 바꾸면 이 창을 다시 세워 적용한다 (모든 문구가 생성 때 박힌다).
        # 저장값이라 KFPS·FLS 편집기·CLI·하위 도구도 다음부터 이 언어로 말한다.
        self.lang_combo = QComboBox()
        self.lang_combo.addItem("한국어", "ko")      # 항목은 제 언어로 적는다
        self.lang_combo.addItem("English", "en")
        self.lang_combo.setCurrentIndex(1 if current_language() == "en" else 0)
        self.lang_combo.setToolTip(tr("gui.lang.tip"))
        self.lang_combo.currentIndexChanged.connect(self._on_lang)
        lrow = QHBoxLayout()
        lrow.addWidget(QLabel(tr("gui.lang")))
        lrow.addWidget(self.lang_combo)
        lrow.addStretch(1)
        left.addLayout(lrow)
        lw = QWidget()
        lw.setLayout(left)
        lw.setFixedWidth(380)

        self.src_pane = _Pane(tr("gui.src"))
        self.out_pane = _Pane(tr("gui.preview"))
        panes = QHBoxLayout()
        panes.addWidget(self.src_pane, 1)
        panes.addWidget(self.out_pane, 1)

        self.log = QPlainTextEdit(readOnly=True)
        self.log.setMaximumBlockCount(4000)
        self.log.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.log.setMinimumHeight(160)
        self.show_log = QCheckBox(tr("gui.show_log"))
        self.show_log.toggled.connect(self.log.setVisible)
        self.log.setVisible(False)

        right = QVBoxLayout()
        right.addLayout(panes, 1)
        right.addWidget(self.show_log)
        right.addWidget(self.log)

        top = QHBoxLayout(self)
        top.addWidget(lw)
        top.addLayout(right, 1)

        self.clock = QTimer(self)
        self.clock.setInterval(500)
        self.clock.timeout.connect(self._tick_clock)

        # 편집기 Export 감시 — 서버가 마커를 갱신하면 그 도안을 문다.
        # 편집기를 처음 연 뒤에만 돈다 (1초 폴링 — 마커 stat 한 번이라 공짜다)
        self.editor_watch = QTimer(self)
        self.editor_watch.setInterval(1000)
        self.editor_watch.timeout.connect(self._poll_editor_export)

        self._sync_go()
        self._set_plan(None)
        if image:
            self.set_image(str(image))

    @Slot(int, float)
    def _apply_done(self, code: int, sec: float) -> None:
        from ...game.inject import STOPPED

        what = tr(f"gui.apply.{self.job_kind}.name")
        self._finish()
        if code == 0:
            self.bar.setValue(self.bar.maximum())   # 미정 막대를 다 찬 것으로 닫는다
            self._msg(tr("gui.apply.done", what=what, sec=int(sec)))
        elif code == STOPPED:                 # 사람이 멈춘 것 — 실패가 아니다
            self._msg(tr("gui.apply.stopped", what=what))
        else:
            self._msg(tr("gui.apply.code", what=what, code=code), bad=True)

    @Slot(str)
    def _apply_failed(self, msg: str) -> None:
        what = tr(f"gui.apply.{self.job_kind}.name")
        self._finish()
        self._log(msg)
        self._msg(tr("gui.apply.failed", what=what, msg=msg), bad=True)

    def _msg(self, text: str, bad: bool = False) -> None:
        self.apply_msg.setText(text)
        self.apply_msg.setStyleSheet("color:#b3261e" if bad else "color:#1a7f37")

    def _busy(self, on: bool) -> None:
        """무엇이 돌든 입력을 잠근다 — 중단 단추만 경로마다 다르다."""
        making = on and self.job_kind == "make"
        self.go.setVisible(not making)
        self.stop.setVisible(making)
        self.stop.setEnabled(making)
        # 주입도 템플릿을 채우는 동안은 길다 — 같은 STOP 파일로 멈춘다
        stoppable = on and self.job_kind in ("run", "inject")
        self.b_astop.setVisible(stoppable)
        self.b_astop.setEnabled(stoppable)
        self.drop.setEnabled(not on)
        self.bgcut.setEnabled(not on)
        self.shapes.setEnabled(not on)
        self.lang_combo.setEnabled(not on)   # 재시작 적용이라 도는 중엔 못 바꾼다
        self._sync_go()

    @Slot(str)
    def _log(self, s: str) -> None:
        self.log.appendPlainText(s.rstrip("\n"))
        # 이타샤 구성 미리보기가 나오면 그 자리에서 건다 — 배치 40분을 기다리기
        # 전에 사람이 결과 꼴을 본다 (`engine.preview`가 이 문구로 알린다)
        mark = msg("미리보기: {path}", path="")
        if mark in s:
            p = Path(s.split(mark, 1)[1].strip())
            if p.exists() and p.suffix.lower() == ".png":
                self.out_pane.load(p)

    @Slot(float, str)
    def _step(self, frac: float, stage: str) -> None:
        self.bar.setValue(int(max(0.0, min(1.0, frac)) * 1000))
        self._stage_text = stage
        self._tick_clock()

    @Slot(dict, str)
    def _done(self, rep: dict, out: str) -> None:
        self._finish()
        self.bar.setValue(1000)
        self.stage.setText(tr("gui.done", sec=rep.get("sec", 0)))
        self.out = Path(out)
        pv = find_run_file(self.out, "preview.png")
        if pv.exists():
            self.out_pane.load(pv)
        ok = all(c["ok"] for c in rep.get("checks", []))
        self.verdict.setText(rep.get("verdict", ""))
        self.verdict.setStyleSheet(
            "font-weight:bold; color:%s" % ("#1a7f37" if ok else "#b26a00"))
        ok_tag, warn_tag = tr("gui.check.ok"), tr("gui.check.warn")
        rows = [(ok_tag if c["ok"] else warn_tag) + " " + c["text"]
                for c in rep.get("checks", [])]
        rows += [warn_tag + " " + n for n in rep.get("notes", [])]
        self.checks.setText("\n".join(rows))
        self.open.setVisible(True)
        # 만들자마자 적용 칸이 이 도안을 문다
        plan = find_run_file(self.out, "plan.json")
        if plan.exists():
            self._set_plan(plan)

    @Slot(str)
    def _failed(self, msg: str) -> None:
        self._finish()
        self.stage.setText("")
        self.verdict.setStyleSheet("font-weight:bold; color:#b3261e")
        self.verdict.setText(msg)
        self._log(msg)
        self.show_log.setChecked(True)

    def _on_lang(self) -> None:
        """언어 콤보 — 저장하고 창을 다시 세운다 (`run`의 재시작 루프).

        문구가 전부 생성 때 박히므로 산 채로는 못 바꾼다. 콤보는 일이 도는
        동안 잠기니(`_busy`) 여기 올 때는 한가한 창이고, `close()`가 편집기·
        오버레이 같은 딸린 창들을 정리한 뒤 재시작 코드로 나간다."""
        lang = self.lang_combo.currentData()
        if lang == current_language():
            return
        save_language(lang)
        self._restart = True
        self.close()
        if self.isVisible():             # closeEvent가 막았다 — 다음 시작에 적용된다
            self._restart = False
        else:
            # 마지막 창 닫힘의 quit(0)이 끼어들 수 있어 코드가 아니라 플래그로
            # 판정한다 — exit()는 루프를 깨우는 용도다
            QApplication.exit(RESTART)

    def closeEvent(self, event) -> None:
        """도는 중에 닫으면 **끝나기를 기다린다** — 스레드를 끊으면 산출물이 깨진다.
        창 조작은 게임에 키를 넣고 있으므로 STOP을 놓고 레이어 경계를 기다린다.

        30초 안에 안 서면 **닫지 않는다.** 스레드가 도는 채로 창을 부수면 그
        스레드의 부모가 사라져 프로세스가 통째로 죽는다 — 게임에 키를 넣던
        중이면 캔버스가 어중간한 상태로 남는다."""
        if self.thread:
            if self.job_kind == "make":
                self._cancel()
            elif self.job_kind in ("run", "inject"):
                self._stop_apply()
            self.thread.quit()
            if not self.thread.wait(30000):
                self._msg(tr("gui.close_busy"), bad=True)
                event.ignore()
                return
        if self.guide:
            self.guide[1].close()
        # 편집기는 스스로 판단한다 — 저장 안 한 편집이 있으면 나가기 확인이 뜬다
        if self.editor is not None:
            self.editor.close()
        super().closeEvent(event)


# 언어를 바꾸면 창을 부수고 다시 세운다 — `app.exec()`이 이 코드로 나온다
RESTART = 7301


def run(image: str | Path | None = None) -> int:
    set_dpi_aware()          # QApplication보다 먼저 — 오버레이가 게임 rect와 맞아야 한다
    # 내장 편집기(QtWebEngine)가 나중에 뜰 수 있다 — 이 속성은 QApplication을
    # 만들기 **전**에만 켤 수 있고, 안 켜면 편집기 창에서 GPU 컨텍스트가 갈린다
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
    app = QApplication.instance() or QApplication([])
    plan: Path | None = None
    while True:
        win = MakeWindow(image)
        if plan is not None and plan.exists():
            win._set_plan(plan)
        win.show()
        code = app.exec()
        if not win._restart:
            return code
        # 물고 있던 것은 새 창이 그대로 문다 — 언어만 바뀐다
        image, plan = win.image, win.plan
