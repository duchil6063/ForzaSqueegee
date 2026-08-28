"""갈래를 고르는 작은 창들 — 내보내기 · 인게임 적용 · 편집기.

단추 하나가 여러 갈래를 쥐면 목록이 길어져 무엇이 무엇인지 안 보인다. 그래서
**단추는 하나로 두고 고르는 것은 창에서** 한다 (사용자 지시 2026-08-26).
셋 다 같은 꼴이다: 갈래를 고르고, 취소가 있고, 고른 것만 돌아온다.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
)

from ...i18n import tr

# 내보내기 갈래 — (열쇠, 이름 키, 설명 키)
EXPORT_KINDS = (("fls", "gui.export.fls", "gui.export.fls.tip"),
                ("kfps", "gui.export.kfps", "gui.export.kfps.tip"),
                ("fs", "gui.export.fs", "gui.export.fs.tip"))


class ExportDialog(QDialog):
    """무엇을 어디에 내보낼까 — 갈래 여럿을 **한 번에** 고른다."""

    def __init__(self, parent, where: Path):
        super().__init__(parent)
        self.setWindowTitle(tr("gui.export.title"))
        lay = QVBoxLayout(self)
        head = QLabel(tr("gui.export.head"))
        head.setWordWrap(True)
        lay.addWidget(head)
        self.boxes: dict[str, QCheckBox] = {}
        for key, name, tip in EXPORT_KINDS:
            cb = QCheckBox(tr(name))
            cb.setToolTip(tr(tip))
            cb.setChecked(True)
            lay.addWidget(cb)
            self.boxes[key] = cb
        row = QHBoxLayout()
        self.where = QLineEdit(str(where))
        pick = QPushButton("…")
        pick.setFixedWidth(32)
        pick.clicked.connect(self._pick)
        row.addWidget(QLabel(tr("gui.export.where")))
        row.addWidget(self.where, 1)
        row.addWidget(pick)
        lay.addLayout(row)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                              | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _pick(self) -> None:
        got = QFileDialog.getExistingDirectory(self, tr("gui.export.where"),
                                               self.where.text())
        if got:
            self.where.setText(got)

    def picked(self) -> tuple[list[str], Path]:
        return ([k for k, cb in self.boxes.items() if cb.isChecked()],
                Path(self.where.text()))


class PickDialog(QDialog):
    """갈래 하나를 고른다 (라디오 + 취소). 반환은 `chosen()`의 열쇠."""

    def __init__(self, parent, title: str, head: str,
                 kinds: tuple[tuple[str, str, str], ...]):
        super().__init__(parent)
        self.setWindowTitle(title)
        lay = QVBoxLayout(self)
        lbl = QLabel(head)
        lbl.setWordWrap(True)
        lay.addWidget(lbl)
        self.buttons: dict[str, QRadioButton] = {}
        for i, (key, name, tip) in enumerate(kinds):
            rb = QRadioButton(name)
            rb.setToolTip(tip)
            rb.setChecked(i == 0)
            lay.addWidget(rb)
            self.buttons[key] = rb
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                              | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def chosen(self) -> str | None:
        for key, rb in self.buttons.items():
            if rb.isChecked():
                return key
        return None


def ask_one(parent, title_key: str, head_key: str,
            kinds: tuple[tuple[str, str, str], ...]) -> str | None:
    """갈래 하나를 묻는다 — 취소하면 None."""
    dlg = PickDialog(parent, tr(title_key), tr(head_key),
                     tuple((k, tr(n), tr(t)) for k, n, t in kinds))
    if dlg.exec() != QDialog.DialogCode.Accepted:
        return None
    return dlg.chosen()
