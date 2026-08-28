"""창이 쓰는 부품 — 작업 스레드 한 벌과 작은 위젯들.

긴 작업은 전부 여기 `_Job`·`_ApplyJob`이 **작업 스레드에서** 돌리고 신호로만
창과 이야기한다 (`print()`는 `_Stream`이 줄 단위 신호로 흘린다). `_Drop`은
이미지 끌어다 놓기, `_Pane`은 창 크기에 맞춰 다시 그리는 그림 칸이다."""

from __future__ import annotations

import time
from contextlib import redirect_stdout
from pathlib import Path

from PySide6.QtCore import QObject, Qt, Signal, Slot
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QFileDialog, QFrame, QLabel, QVBoxLayout

from ...engine.pipeline import Cancelled, make
from ...i18n import tr
from ...paths import data_root, out_root


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


ROOT = data_root()      # 산출물을 쓰는 뿌리 = 저장소 뿌리


def _root_out(image: Path) -> Path:
    """기본 출력 폴더 = `out/make/<이미지 이름>`."""
    return out_root() / "make" / image.stem


def _plan_layers(path: Path) -> list:
    """도안의 레이어 목록. 못 읽으면 예외를 그대로 올린다 (창이 이유를 적는다)."""
    from ...engine.model import LayerPlan

    return LayerPlan.load(path).layers


def _plan_source(path: Path) -> Path | None:
    """플랜이 적어 둔 원화 경로. 상대 경로는 저장소 뿌리 기준. 못 읽으면 None."""
    from ...engine.model import LayerPlan

    try:
        p = Path(LayerPlan.load(path).source_image)
    except Exception:                     # noqa: BLE001 — 그림 한 칸일 뿐이다
        return None
    return p if p.is_absolute() else ROOT / p


class _Job(QObject):
    """`make()` 한 번. 신호로만 창과 이야기한다."""

    line = Signal(str)
    step = Signal(float, str)
    done = Signal(dict, str)
    failed = Signal(str)

    def __init__(self, image: Path, out: Path, route: str, shapes: int,
                 keep_bg: bool):
        super().__init__()
        self.image, self.out, self.route, self.shapes = image, out, route, shapes
        self.keep_bg = keep_bg
        self.stop = False
        self._last = -1.0

    @Slot()
    def go(self) -> None:
        try:
            rep = make(self.image, self.out, route=self.route,
                       shapes=self.shapes, log=self.line.emit,
                       progress=self._progress, keep_bg=self.keep_bg)
        except Cancelled:
            self.failed.emit(tr("gui.cancelled"))
        except SystemExit as e:              # 엔진이 막은 것 (모델 없음 등)
            self.failed.emit(str(e))
        except BaseException as e:           # noqa: BLE001 — 창이 죽으면 안 된다
            self.failed.emit(f"{type(e).__name__}: {e}")
        else:
            self.done.emit(rep, str(self.out))

    def _progress(self, frac: float, stage: str) -> None:
        """엔진의 긴 반복문마다 불린다 — 중단은 여기서 예외로 알린다."""
        if self.stop:
            raise Cancelled
        f = float(frac)
        if f - self._last >= 0.002 or f >= 1.0:   # 장마다 오므로 솎는다
            self._last = f
            self.step.emit(f, str(stage))


class _Stream:
    """`print()`를 줄 단위 신호로 흘려보내는 최소 파일 객체."""

    def __init__(self, emit):
        self._emit, self._buf = emit, ""

    def write(self, s: str) -> int:
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._emit(line)
        return len(s)

    def flush(self) -> None:
        if self._buf:
            self._emit(self._buf)
            self._buf = ""


class _ApplyJob(QObject):
    """도안 하나를 게임에 **메모리 주입**으로 올린다.

    CLI와 **같은 함수**를 부른다 (`game.inject.apply_plan`). 진행은 그 함수가
    `print`로 내므로 표준 출력을 가로채 기록 칸으로 보낸다 (이 스레드에서만
    도는 동안이다). 창 조작(`run`)은 창에 단추가 없다 — 파일 저장이 같은 일을
    몇 초에 하므로 명령줄에만 남는다."""

    line = Signal(str)
    done = Signal(int, float)
    failed = Signal(str)

    def __init__(self, kind: str, plan: Path):
        super().__init__()
        self.kind, self.plan = kind, plan

    @Slot()
    def go(self) -> None:
        t0 = time.time()
        out = _Stream(self.line.emit)
        try:
            with redirect_stdout(out):
                code = self._call()
            out.flush()
        except BaseException as e:            # noqa: BLE001 — 창이 죽으면 안 된다
            out.flush()
            self.failed.emit(f"{type(e).__name__}: {e}")
        else:
            self.done.emit(int(code), time.time() - t0)

    def _call(self) -> int:
        from ...game.inject import apply_plan

        # 중단은 `apply_plan`이 STOP 파일로 본다 (창 조작과 같은 규약).
        # 준비(템플릿 없으면 만들기 · 씨앗 틀리면 다시 심기 · 남는 장 밀어내기)는
        # 주입기가 한다 — 창의 물음은 **긴 작업을 미리 알리는 자리**다
        return apply_plan(self.plan, prepare=True)


class _Drop(QFrame):
    """이미지 끌어다 놓기 · 눌러서 고르기."""

    picked = Signal(str)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setMinimumHeight(150)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.thumb = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.text = QLabel(tr("gui.drop"), alignment=Qt.AlignmentFlag.AlignCenter)
        self.text.setWordWrap(True)
        lay = QVBoxLayout(self)
        lay.addWidget(self.thumb, 1)
        lay.addWidget(self.text)

    def show_image(self, path: Path) -> None:
        pm = QPixmap(str(path))
        if not pm.isNull():
            self.thumb.setPixmap(pm.scaled(220, 220,
                                           Qt.AspectRatioMode.KeepAspectRatio,
                                           Qt.TransformationMode.SmoothTransformation))
        self.text.setText(f"{path.name}  ({pm.width()}×{pm.height()})")

    # ── 끌어다 놓기 ──
    def _url(self, event) -> Path | None:
        md = event.mimeData()
        if not md.hasUrls():
            return None
        p = Path(md.urls()[0].toLocalFile())
        return p if p.suffix.lower() in IMAGE_SUFFIXES and p.is_file() else None

    def dragEnterEvent(self, event) -> None:
        if self._url(event):
            event.acceptProposedAction()

    dragMoveEvent = dragEnterEvent

    def dropEvent(self, event) -> None:
        p = self._url(event)
        if p:
            event.acceptProposedAction()
            self.picked.emit(str(p))

    def mousePressEvent(self, _event) -> None:
        filt = ";;".join((tr("gui.image_filter") + " (*.png *.jpg *.jpeg *.webp *.bmp)",
                          "All files (*)"))
        path, _ = QFileDialog.getOpenFileName(self, tr("gui.pick_image"),
                                              str(ROOT), filt)
        if path:
            self.picked.emit(path)


class _Pane(QLabel):
    """창 크기에 맞춰 다시 그리는 그림 칸."""

    def __init__(self, caption: str):
        super().__init__(caption, alignment=Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(200, 200)
        self._pm: QPixmap | None = None
        self._caption = caption

    def load(self, path: Path) -> None:
        """그림을 건다. **못 읽으면 걸려 있던 것을 지운다** — 안 그러면 앞
        도안의 미리보기가 그대로 남아 이번 것처럼 보인다."""
        pm = QPixmap(str(path))
        if pm.isNull():
            self.clear_image(self._caption)
            return
        self._pm = pm
        self._draw()

    def clear_image(self, caption: str) -> None:
        self._pm = None
        self.setText(caption)

    def _draw(self) -> None:
        if self._pm is None:
            return
        self.setPixmap(self._pm.scaled(self.size(),
                                       Qt.AspectRatioMode.KeepAspectRatio,
                                       Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._draw()
