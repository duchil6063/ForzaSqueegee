"""FLS 갈래 — 도안을 **게임이 읽는 파일로 쓰고**, 편집기로 연다.

창 조작이 하던 일을 이것이 대신한다. 창 조작은 도안을 게임 창에 대고 장당
6초씩 그리지만(3,000장이면 5시간), 여기서는 같은 값을 컨테이너 폴더로 적어
저장 폴더에 놓는다 — 게임의 저장 그리드에 그대로 뜬다.

한 번 누르면 둘이 나간다:

    <도안 폴더>/LayerGroup_<이름>/  C_group · header · thumb.webp   ← 게임이 읽는다
    <도안 폴더>/<이름>.3so                                          ← FLS가 여는 것

`.3so`가 따로 있는 이유: FLS는 명령줄 인자로 `.3so`와 `C_livery`만 받는다
(`C_group`은 안 받는다) — 편집기로 열려면 프로젝트 판이 필요하다.

    창이 쥐고 있어야 하는 것: `plan`·`log`·`show_log`, 그리고 `_msg` (shell)."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QFileDialog, QMessageBox

from ...i18n import tr
from .parts import ROOT


class _FlsOps:
    def _fls_out_dir(self) -> Path | None:
        return self.plan.parent if self.plan is not None else None

    def _export_fls(self, ask_where: bool = False) -> None:
        """물고 있는 도안 → **게임이 읽는 `C_group` 컨테이너 폴더**.

        `ask_where`면 어디에 놓을지 묻는다 — 게임 저장 폴더에 바로 놓으면 저장
        그리드에 그대로 뜬다 (인게임 적용의 '파일 저장' 갈래). 안 물으면 도안
        폴더 옆에 적는다."""
        from ...engine.fls import bridge

        if self.plan is None:
            self._msg(tr("gui.fls.no_plan"), bad=True)
            return
        out = self._fls_out_dir()
        if ask_where:
            got = QFileDialog.getExistingDirectory(
                self, tr("gui.apply.ingame.file.where"), str(out))
            if not got:
                return
            out = Path(got)
        try:
            folder, st = bridge.plan_folder(self.plan, out)
        except BaseException as e:            # noqa: BLE001 — 창이 죽으면 안 된다
            self._msg(f"{type(e).__name__}: {e}", bad=True)
            return
        if st.get("skipped"):
            n = sum(st["skipped"].values())
            self._log(tr("gui.fls.skipped", n=n,
                         names=", ".join(sorted(st["skipped"]))))
            self.show_log.setChecked(True)
        self._msg(tr("gui.fls.exported", n=st["layers"],
                     folder=self._rel(folder)))

    def _export_fls_proj(self, where: Path) -> str | None:
        """FLS 판 — `.3so` 프로젝트 하나. 반환은 파일 이름 (실패하면 None).

        FLS는 명령줄로 `.3so`와 `C_livery`만 받는다 (`C_group`은 안 받는다) —
        편집기로 열려면 이 판이 필요하다."""
        from ...engine.fls import bridge

        if self.plan is None:
            return None
        try:
            proj, _st = bridge.plan_project(
                self.plan, Path(where) / f"{self.plan.parent.name}.3so")
        except BaseException as e:            # noqa: BLE001
            self._msg(f"{type(e).__name__}: {e}", bad=True)
            return None
        return proj.name

    def _rel(self, p: Path) -> str:
        try:
            return str(p.relative_to(ROOT))
        except ValueError:
            return str(p)

    def _open_fls(self) -> None:
        """도안을 `.3so`로 구워 FLS 편집기에 띄운다 (없으면 받겠냐고 묻는다).

        **도안이 없어도 연다** — KFPS 편집기가 빈 캔버스로 열리듯 FLS도 빈
        프로젝트로 뜬다 (편집기 안에서 열거나 새로 그리면 된다)."""
        from ... import flseditor
        from ...engine.fls import bridge

        if not flseditor.available() and not self._offer_fls_download():
            return
        if self.plan is None:
            try:
                flseditor.open_file(None)
            except BaseException as e:        # noqa: BLE001
                self._msg(f"{type(e).__name__}: {e}", bad=True)
                return
            self._msg(tr("gui.fls.opened_empty"))
            return
        out = self._fls_out_dir()
        try:
            proj, st = bridge.plan_project(self.plan, out / f"{self.plan.parent.name}.3so")
            flseditor.open_file(proj)
        except BaseException as e:            # noqa: BLE001
            self._msg(f"{type(e).__name__}: {e}", bad=True)
            return
        self._msg(tr("gui.fls.opened", n=st["layers"], name=proj.name))

    def _offer_fls_download(self) -> bool:
        """FLS가 없다 — 받을지 묻고, 받겠다면 그 자리에서 받는다."""
        from ... import flseditor

        yes = QMessageBox.StandardButton.Yes
        btn = QMessageBox.question(
            self, tr("gui.fls.name"), tr("gui.fls.missing"),
            yes | QMessageBox.StandardButton.Open
            | QMessageBox.StandardButton.Cancel)
        if btn == QMessageBox.StandardButton.Open:      # 이미 받아 둔 것을 고른다
            path, _ = QFileDialog.getOpenFileName(
                self, tr("gui.fls.pick"), str(Path.home()),
                f"{flseditor.EXE_NAME} ({flseditor.EXE_NAME})")
            if not path:
                return False
            try:
                flseditor.set_path(path)
            except ValueError as e:
                self._msg(str(e), bad=True)
                return False
            return True
        if btn != yes:
            return False
        self._msg(tr("gui.fls.downloading"))
        self.log.clear()
        self.show_log.setChecked(True)
        try:
            import subprocess
            import sys

            r = subprocess.run([sys.executable, str(ROOT / "tools" / "get_fls.py")],
                               capture_output=True, timeout=900,
                               encoding="utf-8", errors="replace")
            self._log((r.stdout or "") + (r.stderr or ""))
        except BaseException as e:            # noqa: BLE001
            self._msg(f"{type(e).__name__}: {e}", bad=True)
            return False
        if not flseditor.available():
            self._msg(tr("gui.fls.download_failed"), bad=True)
            return False
        self._msg(tr("gui.fls.downloaded"))
        return True
