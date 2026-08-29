"""물고 있는 도안 — 고르기 · 바꿔 물기 · KFPS로 내보내기."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QFileDialog

from ...i18n import tr
from ...paths import find_run_file, glob_run_files
from .parts import ROOT, _plan_layers, _plan_source


class _PlanOps:
    """도안을 무는 갈래 — 어느 plan.json을 들고 있나.

    창이 쥐고 있어야 하는 것: `plan`(현재 경로)·`plan_lbl`·`out`·`out_pane`·
    `src_pane`·`apply_msg`, 그리고 `_log`·`_msg` (shell)."""

    def _set_plan(self, path: Path | None) -> None:
        self.plan = path
        if path is None:
            self.plan_lbl.setText(tr("gui.apply.no_plan"))
        else:
            try:
                n = len(_plan_layers(path))
            except Exception as e:            # noqa: BLE001 — 못 읽으면 이유를 적는다
                self._set_plan(None)
                self._msg(f"{path.name}: {type(e).__name__}: {e}", bad=True)
                return
            self.plan_lbl.setText(tr("gui.apply.plan", name=path.name, n=n))
        self.apply_msg.setText("")
        self._sync_go()

    def _pick_plan(self) -> None:
        start = str(self.out or ROOT / "out")
        path, _ = QFileDialog.getOpenFileName(
            self, tr("gui.pick_plan"), start,
            "도안 / KFPS JSON / FLS (*.json *.3so C_group C_livery);;"
            "All files (*)")
        if not path:
            return
        p = self._resolve_plan(Path(path))
        if p is None:
            return
        self._set_plan(p)
        if self.plan is not None:
            self._show_plan_assets(self.plan)

    def _resolve_plan(self, path: Path) -> Path | None:
        """고른 파일을 도안으로 만든다 — 셋 중 무엇이든 받는다.

        - **plan**: 그대로 (판별이 안 되는 파일도 그대로 — plan 로더가 제 이유를 적는다)
        - **KFPS**: 도형 목록이면 `out/kfpsimport/<이름>/`에 변환
        - **FLS**: `.3so`·`C_group`·`C_livery`면 `out/flsimport/<이름>/`에 변환.
          리버리는 면마다 plan을 내고 그 면들을 묶는 `*.itasha.json`을 낸다
          (그 구성 파일을 통째로 무는 것은 편집기의 [Itasha] 메뉴다 —
          여기서는 첫 면을 문다)

        변환은 1~2초짜리라 스레드가 필요 없다."""
        from ...engine.fls import bridge, folder as flsfolder
        from ...engine.kfpsjson import resolve_plan

        if flsfolder.sniff(path) is not None:
            try:
                out, st = bridge.import_any(path, ROOT / "out" / "flsimport")
            except BaseException as e:        # noqa: BLE001 — 창이 죽으면 안 된다
                self._msg(f"{path.name}: {type(e).__name__}: {e}", bad=True)
                return None
            if st.get("unknown"):
                for sid, n in sorted(st["unknown"].items()):
                    self._log(tr("gui.fls.unknown", id=sid, n=n))
                self.show_log.setChecked(True)
            if out.name.endswith("itasha.json"):  # 리버리 — 면마다 도안
                self._msg(tr("gui.fls.imported_livery",
                             n=st.get("layers", 0),
                             faces=", ".join(sorted(st.get("surfaces") or {})),
                             out=str(out.parent.relative_to(ROOT))))
                return next(iter(glob_run_files(out.parent, "plan.json")),
                            None)
            self._msg(tr("gui.fls.imported", n=st.get("layers", 0),
                         out=str(out.parent.relative_to(ROOT))))
            return out
        try:
            plan_path, st = resolve_plan(path, ROOT / "out" / "kfpsimport")
        except BaseException as e:            # noqa: BLE001 — 창이 죽으면 안 된다
            self._msg(f"{path.name}: {type(e).__name__}: {e}", bad=True)
            return None
        if st is not None:
            for word, n in sorted(st["unknown"].items()):
                self._log(tr("gui.kfpsimport.unknown", word=word, n=n))
            if st["unknown"]:
                self.show_log.setChecked(True)
            self._msg(tr("gui.kfpsimport.done", name=path.name,
                         out=str(plan_path.parent.relative_to(ROOT))))
        return plan_path

    def _export_kfps(self, where=None) -> str | None:
        """KFPS 판 — 편집기·임포터가 읽는 JSON. 반환은 파일 이름 (실패하면 None).

        `where`가 없으면 도안 폴더 옆이다. 내보내기 + 재들여온 렌더 대조 ≈ 1초라
        스레드가 필요 없다 — 그 대조 결과(`diff`)를 기록에 남긴다."""
        import json as _json

        from ...engine.catalog import Catalog, default_catalog_path
        from ...engine.kfpsjson import export_typecode, roundtrip_diff
        from ...engine.model import LayerPlan

        if self.plan is None:
            return None
        folder = Path(where) if where else self.plan.parent
        try:
            plan = LayerPlan.load(self.plan)
            cat = Catalog(default_catalog_path())
            data, st = export_typecode(plan, cat)
            if not data["shapes"]:
                raise ValueError(tr("gui.kfpsexport.empty"))
            # 이름은 늘 도안 폴더를 따른다 (`where`가 남의 폴더여도)
            out = folder / f"{self.plan.parent.name}.kfps.json"
            out.write_text(_json.dumps(data, ensure_ascii=False),
                           encoding="utf-8")
            diff = roundtrip_diff(plan, data, cat)
        except BaseException as e:            # noqa: BLE001 — 창이 죽으면 안 된다
            self._msg(f"{type(e).__name__}: {e}", bad=True)
            return None
        notes = [tr("gui.kfpsexport.done", name=out.name, n=len(data["shapes"]),
                    diff=f"{diff['changed_frac']:.1%}")]
        if st["approx"]:
            notes.append(tr("gui.kfpsexport.approx", n=st["approx"]))
            self.show_log.setChecked(True)
        self._log(" · ".join(notes))
        return out.name

    def _show_plan_assets(self, plan: Path) -> None:
        """불러온 도안의 그림을 칸에 건다 — `make`가 낸 것과 같은 이름들이다.

        **출력 폴더(`self.out`)는 안 건드린다** — 불러오기는 올리기용이고, 다음
        [도안 생성]이 남의 폴더에 쓰면 안 된다."""
        folder = plan.parent
        pv = find_run_file(folder, "preview.png")
        if pv.exists():
            self.out_pane.load(pv)
        else:
            self.out_pane.clear_image(tr("gui.preview.none"))
        # 원본 칸: 전처리가 돌았으면 `cutout.png`(노선이 실제로 받은 입력)를,
        # 아니면 플랜이 적어 둔 원화를 건다. 원화는 만든 기계의 경로라 여기
        # 없을 수 있고, 그러면 그냥 둔다 (못 걸어도 올리기는 된다)
        for src in (find_run_file(folder, "cutout.png"), _plan_source(plan)):
            if src is not None and src.exists():
                self.src_pane.load(src)
                return
