"""수동 따라 그리기 가이드: plan.json(또는 이미지)을 소비하는 오버레이 + 조작 패널.

- 오버레이 창: FH6 클라이언트 영역을 덮는 클릭 통과 창.
  · 도안 모드: 전체 계획 렌더를 반투명으로 표시
  · 단계 모드: 완료 레이어는 흐리게, 현재 레이어는 강조 외곽선 + 십자선
  · 원본 모드: 플랜의 소스 이미지(배경 투명 처리)를 반투명 표시 — 수동 제작 참조용
- `overlay <이미지.png>`처럼 이미지를 직접 주면 플랜 없이 원본 모드만 실행된다.
- 조작 패널: 현재 레이어의 도형/수치(에디터에 입력할 값) 표시, 이전/다음, 진행 저장.
- 좌표 매핑: 캔버스 중앙 원점 → 게임 클라이언트 중앙. px/유닛은 세로 가시 유닛 수
  (기본 1200, 1600×999 실측 1유닛≈0.83px 근거)로 환산하며 패널에서 보정 가능.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PySide6.QtCore import QPointF, Qt, QTimer
from PySide6.QtGui import QColor, QImage, QKeySequence, QPainter, QPen, QPolygonF, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ..engine.catalog import Catalog, default_catalog_path
from ..engine.model import UNITS_PER_SCALE, Layer, LayerPlan
from ..game import calibrate as gcal
from ..game import io as gio
from ..i18n import tr
from .win import find_window_client_rect, make_click_through, set_dpi_aware

# 게임 캔버스 뷰의 세로 가시 유닛 수. 실측: 1 유닛 = 렌더 해상도 1px (2560×1440 → 1440).
# 시작 시/버튼으로 P 격자 캡처 자동 보정이 이 값을 대체한다.
VIEW_UNITS_V = 1440.0


_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


class GuideState:
    """진행 상태: plan + 현재 레이어 인덱스, progress.json 저장/복원.

    이미지 파일을 직접 받으면 plan 없이 원본 모드 전용으로 동작한다
    (image_path만 설정, 진행 저장 없음)."""

    def __init__(self, plan_path: str | Path):
        self.plan_path = Path(plan_path)
        self.plan: LayerPlan | None = None
        self.image_path: Path | None = None
        self.image_height_units = 900.0  # convert 기본 canvas_height_units와 동일
        self.index = 0
        if self.plan_path.suffix.lower() in _IMAGE_SUFFIXES:
            self.image_path = self.plan_path
            self.progress_path = None
            return
        self.plan = LayerPlan.load(self.plan_path)
        self.progress_path = self.plan_path.with_name("progress.json")
        if self.progress_path.exists():
            data = json.loads(self.progress_path.read_text(encoding="utf-8"))
            self.index = min(int(data.get("index", 0)), len(self.plan.layers))
        src = Path(self.plan.source_image)
        if not src.is_absolute():
            root = Path(__file__).resolve().parents[2]
            src = src if src.exists() else root / src
        if src.exists():
            self.image_path = src
            self.image_height_units = self.plan.image_size[1] * self.plan.units_per_px

    def save(self) -> None:
        if self.progress_path is None:
            return
        self.progress_path.write_text(
            json.dumps({"plan": self.plan_path.name, "index": self.index}),
            encoding="utf-8")

    @property
    def current(self) -> Layer | None:
        if self.plan and 0 <= self.index < len(self.plan.layers):
            return self.plan.layers[self.index]
        return None

    def step(self, delta: int) -> None:
        if self.plan is None:
            return
        self.index = int(np.clip(self.index + delta, 0, len(self.plan.layers)))
        self.save()


class OverlayWindow(QWidget):
    """FH6 위에 얹는 클릭 통과 가이드 창."""

    def __init__(self, state: GuideState, catalog: Catalog):
        super().__init__(None, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.state = state
        self.catalog = catalog
        self.mode = "step" if state.plan else "image"  # "step" | "sketch" | "image"
        self.sketch_opacity = 0.45
        self.view_units_v = VIEW_UNITS_V
        # 원본 모드 배치: 이미지 세로 크기(유닛)·중심 오프셋(유닛) — 패널에서 보정
        self.image_height_units = state.image_height_units
        self.image_dx = 0.0
        self.image_dy = 0.0
        self._sketch_cache: QImage | None = None
        self._image_rgba: np.ndarray | None = None  # 원본 RGBA (배경 투명 처리)
        self._image_cache: QImage | None = None
        self._game_rect: tuple[int, int, int, int] | None = None

        # 게임 창 추적 (위치/크기 가변 → 1초마다 재조회)
        self._track = QTimer(self)
        self._track.timeout.connect(self.reposition)
        self._track.start(1000)
        self.reposition()

    def showEvent(self, ev):
        super().showEvent(ev)
        make_click_through(int(self.winId()))

    def reposition(self) -> None:
        rect = find_window_client_rect()
        if rect is None:
            return
        if rect != self._game_rect:
            self._game_rect = rect
            x, y, w, h = rect
            self.setGeometry(x, y, w, h)
            self._sketch_cache = None
            self._image_cache = None
            self.update()

    # --- 좌표 변환: 캔버스 유닛 → 오버레이 px ---
    def _ppu(self) -> float:
        return self.height() / self.view_units_v if self.view_units_v else 1.0

    def canvas_to_px(self, ux: float, uy: float) -> tuple[float, float]:
        s = self._ppu()
        return self.width() / 2 + ux * s, self.height() / 2 - uy * s

    def _layer_polys(self, layer: Layer) -> list[QPolygonF]:
        sh = self.catalog[layer.shape]
        s = self._ppu()
        rot = np.radians(layer.rot)
        c, sn = np.cos(rot), np.sin(rot)
        polys = []
        for loop in sh.loops:
            pts = loop * np.array([layer.sx, layer.sy], np.float32) * UNITS_PER_SCALE
            pts = pts @ np.array([[c, sn], [-sn, c]], np.float32)
            pts += np.array([layer.x, layer.y], np.float32)
            px = self.width() / 2 + pts[:, 0] * s
            py = self.height() / 2 - pts[:, 1] * s
            polys.append(QPolygonF([QPointF(float(a), float(b)) for a, b in zip(px, py)]))
        return polys

    def _render_sketch(self) -> QImage:
        if self._sketch_cache is None or self._sketch_cache.size() != self.size():
            from ..engine.render import render_plan

            # plan을 오버레이 해상도에 맞게 렌더 후 유닛 스케일로 배치
            img = render_plan(self.state.plan, self.catalog)
            h, w = img.shape[:2]
            qimg = QImage(img.data, w, h, w * 3, QImage.Format_RGB888).copy()
            # 캔버스 유닛 크기 → px
            upp = self.state.plan.units_per_px  # 유닛/이미지px
            target_w = int(w * upp * self._ppu())
            target_h = int(h * upp * self._ppu())
            self._sketch_cache = qimg.scaled(target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        return self._sketch_cache

    def _load_image_rgba(self) -> np.ndarray | None:
        """원본 이미지 → RGBA (배경 투명). 알파 있으면 그대로, 없으면 가장자리
        연결 순백을 배경으로 판정(_foreground_mask) — 게임 화면에서 캐릭터만
        비치게 한다. 한글 경로 대응: np.fromfile + imdecode (imread는 조용히 실패)."""
        if self._image_rgba is not None:
            return self._image_rgba
        if self.state.image_path is None or not Path(self.state.image_path).exists():
            return None
        import cv2

        def _foreground_mask(rgba_img: np.ndarray) -> np.ndarray:
            alpha = rgba_img[..., 3]
            if alpha.min() < 250:  # 알파 채널이 실제로 쓰임
                return alpha > 127
            # 흰 배경 제거: 가장자리에 연결된 순백 영역을 배경으로
            rgb = rgba_img[..., :3]
            ff = (rgb.astype(np.int16).min(axis=2) > 240).astype(np.uint8)
            mask = np.zeros((rgb.shape[0] + 2, rgb.shape[1] + 2), np.uint8)
            for seed in [(0, 0), (rgb.shape[1] - 1, 0), (0, rgb.shape[0] - 1),
                         (rgb.shape[1] - 1, rgb.shape[0] - 1)]:
                if ff[seed[1], seed[0]]:
                    cv2.floodFill(ff, mask, seed, 2)
            return ff != 2

        buf = np.fromfile(str(self.state.image_path), np.uint8)
        bgra = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        if bgra is None:
            return None
        if bgra.ndim == 2:
            bgra = cv2.cvtColor(bgra, cv2.COLOR_GRAY2BGRA)
        if bgra.shape[2] == 3:
            bgra = cv2.cvtColor(bgra, cv2.COLOR_BGR2BGRA)
        rgba = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGBA)
        fg = _foreground_mask(rgba)
        rgba[..., 3] = np.where(fg, rgba[..., 3], 0)
        self._image_rgba = np.ascontiguousarray(rgba)
        return self._image_rgba

    def _render_image(self) -> QImage | None:
        if self._image_cache is not None:
            return self._image_cache
        rgba = self._load_image_rgba()
        if rgba is None:
            return None
        h, w = rgba.shape[:2]
        qimg = QImage(rgba.data, w, h, w * 4, QImage.Format_RGBA8888).copy()
        target_h = max(8, int(self.image_height_units * self._ppu()))
        target_w = max(8, int(target_h * w / h))
        self._image_cache = qimg.scaled(target_w, target_h, Qt.KeepAspectRatio,
                                        Qt.SmoothTransformation)
        return self._image_cache

    def invalidate_image(self) -> None:
        self._image_cache = None
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        if self.mode == "sketch":
            img = self._render_sketch()
            p.setOpacity(self.sketch_opacity)
            p.drawImage((self.width() - img.width()) // 2, (self.height() - img.height()) // 2, img)
            p.setOpacity(1.0)
        elif self.mode == "image":
            img = self._render_image()
            if img is not None:
                s = self._ppu()
                p.setOpacity(self.sketch_opacity)
                p.drawImage(int((self.width() - img.width()) / 2 + self.image_dx * s),
                            int((self.height() - img.height()) / 2 - self.image_dy * s),
                            img)
                p.setOpacity(1.0)
        else:
            self._paint_step(p)
        p.end()

    def _paint_step(self, p: QPainter) -> None:
        cur = self.state.current
        # 완료 레이어: 흐린 채움 (도안이 쌓여가는 모습)
        p.setOpacity(0.25)
        p.setPen(Qt.NoPen)
        for layer in self.state.plan.layers[: self.state.index]:
            r, g, b = layer.rgb()
            p.setBrush(QColor(r, g, b))
            for poly in self._layer_polys(layer):
                p.drawPolygon(poly)
        p.setOpacity(1.0)
        if cur is None:
            return
        # 현재 레이어: 채움 미리보기 + 강조 외곽선 + 중심 십자선
        r, g, b = cur.rgb()
        p.setBrush(QColor(r, g, b, 140))
        p.setPen(QPen(QColor(0, 255, 136), 3))
        for poly in self._layer_polys(cur):
            p.drawPolygon(poly)
        cx, cy = self.canvas_to_px(cur.x, cur.y)
        p.setPen(QPen(QColor(0, 255, 136), 1))
        p.drawLine(int(cx) - 14, int(cy), int(cx) + 14, int(cy))
        p.drawLine(int(cx), int(cy) - 14, int(cx), int(cy) + 14)


class ControlPanel(QWidget):
    """오버레이 옆에 띄우는 일반 창: 현재 레이어 수치 + 진행 조작."""

    def __init__(self, state: GuideState, overlay: OverlayWindow):
        super().__init__(None, Qt.WindowStaysOnTopHint)
        self.state = state
        self.overlay = overlay
        self.setWindowTitle(tr("overlay.panel.title"))

        root = QVBoxLayout(self)
        self.progress_lbl = QLabel()
        root.addWidget(self.progress_lbl)

        self.info = QLabel()
        self.info.setTextFormat(Qt.RichText)
        self.info.setStyleSheet("font-family: Consolas, monospace; font-size: 13px;")
        root.addWidget(self.info)

        nav = QHBoxLayout()
        self.prev_btn = QPushButton(tr("overlay.prev"))
        self.next_btn = QPushButton(tr("overlay.next"))
        self.prev_btn.clicked.connect(lambda: self.step(-1))
        self.next_btn.clicked.connect(lambda: self.step(+1))
        nav.addWidget(self.prev_btn)
        nav.addWidget(self.next_btn)
        root.addLayout(nav)
        if state.plan is None:  # 이미지 전용 실행: 단계 UI 숨김
            self.progress_lbl.hide()
            self.info.hide()
            self.prev_btn.hide()
            self.next_btn.hide()

        opts = QGridLayout()
        self.mode_btn = QPushButton()
        self.mode_btn.clicked.connect(self.toggle_mode)
        if state.plan is None or state.image_path is None:
            self.mode_btn.setEnabled(state.plan is not None)
        opts.addWidget(self.mode_btn, 0, 0)

        self.opacity = QSlider(Qt.Horizontal)
        self.opacity.setRange(10, 90)
        self.opacity.setValue(int(self.overlay.sketch_opacity * 100))
        self.opacity.valueChanged.connect(self.set_opacity)
        opts.addWidget(QLabel(tr("overlay.opacity")), 1, 0)
        opts.addWidget(self.opacity, 1, 1)

        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setRange(400.0, 4000.0)
        self.scale_spin.setSingleStep(25.0)
        self.scale_spin.setValue(self.overlay.view_units_v)
        self.scale_spin.valueChanged.connect(self.set_view_units)
        opts.addWidget(QLabel(tr("overlay.view_units")), 2, 0)
        opts.addWidget(self.scale_spin, 2, 1)

        # 원본 모드 배치 보정: 세로 크기(유닛) + 중심 오프셋 X/Y(유닛)
        self.img_h_spin = QDoubleSpinBox()
        self.img_h_spin.setRange(100.0, 4000.0)
        self.img_h_spin.setSingleStep(25.0)
        self.img_h_spin.setValue(self.overlay.image_height_units)
        self.img_h_spin.valueChanged.connect(self.set_image_height)
        opts.addWidget(QLabel(tr("overlay.image_height")), 3, 0)
        opts.addWidget(self.img_h_spin, 3, 1)

        off = QHBoxLayout()
        self.img_dx_spin = QDoubleSpinBox()
        self.img_dy_spin = QDoubleSpinBox()
        for sp in (self.img_dx_spin, self.img_dy_spin):
            sp.setRange(-2000.0, 2000.0)
            sp.setSingleStep(5.0)
            sp.setValue(0.0)
            sp.valueChanged.connect(self.set_image_offset)
            off.addWidget(sp)
        opts.addWidget(QLabel(tr("overlay.image_offset")), 4, 0)
        opts.addLayout(off, 4, 1)
        if state.image_path is None:
            self.img_h_spin.setEnabled(False)
            self.img_dx_spin.setEnabled(False)
            self.img_dy_spin.setEnabled(False)

        self.calib_btn = QPushButton(tr("overlay.calibrate"))
        self.calib_btn.clicked.connect(self.auto_calibrate)
        opts.addWidget(self.calib_btn, 5, 0)
        self.calib_lbl = QLabel("")
        self.calib_lbl.setStyleSheet("color:#888; font-size:11px;")
        opts.addWidget(self.calib_lbl, 5, 1)
        root.addLayout(opts)

        QShortcut(QKeySequence(Qt.Key_Right), self, activated=lambda: self.step(+1))
        QShortcut(QKeySequence(Qt.Key_Space), self, activated=lambda: self.step(+1))
        QShortcut(QKeySequence(Qt.Key_Left), self, activated=lambda: self.step(-1))

        self.refresh()

    def step(self, delta: int) -> None:
        self.state.step(delta)
        self.overlay.update()
        self.refresh()

    def toggle_mode(self) -> None:
        # 단계 → 도안 → 원본(소스 이미지 있을 때) 순환
        cycle = ["step", "sketch"] + (["image"] if self.state.image_path else [])
        cur = cycle.index(self.overlay.mode) if self.overlay.mode in cycle else 0
        self.overlay.mode = cycle[(cur + 1) % len(cycle)]
        self.overlay.update()
        self.refresh()

    def set_opacity(self, v: int) -> None:
        self.overlay.sketch_opacity = v / 100.0
        self.overlay.update()

    def set_view_units(self, v: float) -> None:
        self.overlay.view_units_v = v
        self.overlay._sketch_cache = None
        self.overlay._image_cache = None
        self.overlay.update()

    def set_image_height(self, v: float) -> None:
        self.overlay.image_height_units = v
        self.overlay.invalidate_image()

    def set_image_offset(self) -> None:
        self.overlay.image_dx = self.img_dx_spin.value()
        self.overlay.image_dy = self.img_dy_spin.value()
        self.overlay.update()

    def auto_calibrate(self, silent: bool = False) -> bool:
        """FH6 캡처에서 P 격자 주기로 px/유닛 산출 → 세로 유닛 갱신."""
        hwnd = gio.find_hwnd()
        if hwnd is None:
            if not silent:
                self.calib_lbl.setText(tr("overlay.game_not_found"))
            return False
        img = gio.capture(hwnd)
        ppu = gcal.px_per_unit(img)
        if ppu is None:
            if not silent:
                self.calib_lbl.setText(tr("overlay.calibrate.fail"))
            return False
        self.scale_spin.setValue(img.shape[0] / ppu)  # valueChanged → set_view_units
        self.calib_lbl.setText(tr("overlay.calibrate.ok", ppu=f"{ppu:.4f}"))
        return True

    def closeEvent(self, ev):
        """패널을 닫으면 오버레이도 닫는다 — 오버레이는 클릭 통과라 혼자 남으면
        사용자가 못 닫는다 (제품 창에서 띄웠을 때 특히)."""
        self.overlay.close()
        super().closeEvent(ev)

    def refresh(self) -> None:
        self.mode_btn.setText(tr(f"overlay.mode.{self.overlay.mode}"))
        if self.state.plan is None:
            return
        n = len(self.state.plan.layers)
        i = self.state.index
        self.progress_lbl.setText(tr("overlay.progress", current=min(i + 1, n), total=n))
        cur = self.state.current
        if cur is None:
            self.info.setText(f"<b>{tr('overlay.finished')}</b>")
            return
        rows = [
            (tr("overlay.field.shape"), cur.shape),
            (tr("overlay.field.label"), tr(f"region.{cur.label}") if cur.label else "-"),
            ("X / Y", f"{cur.x:+.1f} / {cur.y:+.1f}"),
            (tr("overlay.field.scale"), f"{cur.sx:.2f} × {cur.sy:.2f}"),
            (tr("overlay.field.rot"), f"{cur.rot:.1f}°"),
            ("HSB", "{:.2f} / {:.2f} / {:.2f}".format(*cur.hsb())),
        ]
        html = "<table>" + "".join(
            f"<tr><td style='color:#888;padding-right:8px'>{k}</td><td><b>{v}</b></td></tr>"
            for k, v in rows) + "</table>"
        self.info.setText(html)


def open_guide(plan_path: str | Path) -> tuple[OverlayWindow, ControlPanel]:
    """오버레이 + 패널을 띄우고 돌려준다. **이벤트 루프는 안 돈다.**

    이미 창이 떠 있는 프로세스(제품 창)에서 부르는 쪽이다 — 거기서 `run()`을
    부르면 이벤트 루프가 겹친다. 돌려받은 두 창은 **참조를 잡고 있어야 한다**
    (놓으면 파이썬이 거둬 가면서 닫힌다)."""
    state = GuideState(plan_path)
    catalog = Catalog(default_catalog_path())
    overlay = OverlayWindow(state, catalog)
    panel = ControlPanel(state, overlay)
    overlay.show()
    panel.show()
    # 패널을 게임 창 왼쪽 옆(공간 없으면 오른쪽)에 배치
    rect = find_window_client_rect()
    if rect:
        x, y, w, _ = rect
        pw = panel.sizeHint().width()
        panel.move(x - pw - 12 if x - pw - 12 >= 0 else x + w + 12, y)
    panel.auto_calibrate(silent=True)  # 격자 배경이 켜져 있으면 시작부터 정밀 보정
    return overlay, panel


def run(plan_path: str | Path) -> int:
    set_dpi_aware()          # QApplication보다 먼저다 — Qt가 좌표계를 정하기 전에
    app = QApplication.instance() or QApplication([])
    overlay, panel = open_guide(plan_path)   # noqa: F841 — 참조를 잡아 둔다
    return app.exec()
