"""`make` — 이미지 하나 → 도안 하나. 한 명령으로 끝난다.

    python -m forzasqueegee make <이미지> -o <폴더> [--route cel|painter|line] [--shapes N]

노선은 사람이 고른다 (자동 판별은 만들지 않는다). 본체는 노선마다 제 모듈에
있고, 이 파일은 **모든 노선이 함께 쓰는 것**만 갖는다 — 전처리(배경 제거·
크롭)·한글 경로 io·인게임 도달 검사·최소 CelArt 구성·KFPS JSON 내보내기:

- `cel` — **기본** (`route_cel`). 사람 순서 그대로 세 단이다: ① **선 도안** —
  line 노선과 같은 기계로 SR 중간본의 선화를 사람 문법으로 긋고 ② **셀
  재해석** — 같은 중간본을 평면 색 영역으로 누른 뒤(celart) ③ **색 채움** —
  셀 영역을 배치된 획 라스터에 스냅해 그 아래를 채운다 (celfit — 색이 선을
  못 넘고, 빈 틈이 없고, 같은 색만 선 아래를 지난다. `_make_cel` 문서).
  레이어 수는 **가격이 정한다** — 예산을 채우지 않고 값이 되는 레이어만
  산다 (`engine.price`).
- `painter` — KFPS(kloudys-forza-painter-suite) 동일 로직 (`route_painter`
  → galatea) — GPU 원시 생성 + 체크포인트 마무리, 회전 타원·사각형 혼합.
- `line` — 원화의 **선만** 획으로 딴다 (면 채움 없음, `route_line`). 사람이
  원화를 반투명 오버레이로 깔고 선만 따라 긋는 방식의 자동화
  (`celfit.fit_line_plan`). 바탕이 비므로 차 도색 위에 선화만 얹는 도안이
  된다. 그 획 배치 절반(`route_line._line_design`)은 cel 노선도 쓴다.

GUI는 이 함수를 부르는 창일 뿐이다 — 로직을 창에 두지 않는다.

산출물 (`-o` 폴더). **파일마다 폴더 이름이 앞에 붙는다** — `out/내도안/`이면
`내도안.plan.json`이다 (`paths.run_file`). 이름이 겹치지 않아 편집기 탭·게임
임포트 목록에서 어느 도안인지 갈린다:

    <이름>.plan.json     창 조작·주입에 쓰는 레이어 계획
    <이름>.kfps.json     KFPS 편집기·임포터 타입코드 JSON
    <이름>.3so           FLS 편집기가 여는 프로젝트
    LayerGroup_<이름>/   게임이 읽는 비닐 그룹 컨테이너 (C_group + header)
    <이름>.preview.png   플랜을 렌더한 그림
    <이름>.cel.png       (cel) 배치 목표 — 획 라스터에 스냅한 셀 재해석 + 선 도안
    <이름>.line.png      (line) 선화 목표 — 흰 바탕 + 원화 색 선
    <이름>.cutout.png    (전처리 발동 시) 배경 제거·크롭 결과 — 노선이 받은 입력
    <이름>.report.json   자가 점검 — `verdict`에 한 줄 판정
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

from ..i18n import msg
from ..paths import find_run_file, run_file
from .celfit.affine import representable
from .stop import Cancelled, stop_here, stopping

# 작업 해상도 — **짧은 변** 기준. 긴 변 기준이면 세로로 긴 구도에서 인물의 폭이
# 몇백 px로 줄어 얼굴이 무너진다
WORK_SIZE = 1200
MAX_SHAPES = 3000     # 인게임 레이어 상한 (남은 용량이 모자라면 통째로 거부된다)
ROUTES = ("painter", "cel", "line")


def _say(s: str = "") -> None:
    """진행이 몇 분짜리라 **버퍼링하면 안 된다** — 파이프로 받으면 통째로 늦는다."""
    print(s, flush=True)


def read_rgba(path: str | Path) -> np.ndarray:
    """RGBA로 읽는다 (알파 없으면 255로 채운다). 한글 경로 대응."""
    buf = np.fromfile(str(path), np.uint8)
    im = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise SystemExit(msg("읽기 실패: {path}", path=path))
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGRA)
    elif im.shape[2] == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2BGRA)
    return np.dstack([cv2.cvtColor(im[..., :3], cv2.COLOR_BGR2RGB), im[..., 3]])


def write_png(path: Path, img: np.ndarray) -> None:
    """PNG로 쓴다 (BGR 또는 BGRA 또는 회색). **한글 경로 대응.**

    `cv2.imwrite`를 쓰면 안 된다 — 경로에 비ASCII 글자가 하나라도 있으면 예외
    없이 `False`만 돌려주고 **파일을 안 만든다**(읽기의 `imread`와 같다). GUI의
    기본 출력 폴더가 `out/make/<이미지 이름>`이라 한글 파일명이 곧 한글 폴더고,
    그래서 도안은 나오는데 그림만 통째로 빠졌다. 인코딩은 OpenCV에 맡기고
    파일은 파이썬이 쓴다."""
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise SystemExit(msg("인코딩 실패: {path}", path=path))
    path.write_bytes(buf.tobytes())


def _crop_to_subject(rgba: np.ndarray) -> tuple[np.ndarray, list[int] | None]:
    """인물 알파 bbox + 여백으로 자른다 — 작업 해상도를 인물에 몰아준다.

    문턱 128은 셀 노선의 캔버스 판정(`sel`)과 같고, 부드러운 알파 치마는
    여백(최대변의 2%)이 덮는다. 알파가 아예 없거나(전부 불투명 = bbox가 전체)
    절감이 10% 미만이면 안 자른다 — 재인코딩만 생기고 이득이 없다.
    반환: (잘린 RGBA, [x, y, w, h]) — 안 잘랐으면 (원본, None).
    """
    a = rgba[..., 3]
    ys, xs = np.where(a >= 128)
    if ys.size == 0:
        return rgba, None
    h, w = a.shape[:2]
    m = max(8, round(0.02 * max(h, w)))
    y0, y1 = max(int(ys.min()) - m, 0), min(int(ys.max()) + 1 + m, h)
    x0, x1 = max(int(xs.min()) - m, 0), min(int(xs.max()) + 1 + m, w)
    if (y1 - y0) * (x1 - x0) >= 0.90 * h * w:
        return rgba, None
    return rgba[y0:y1, x0:x1], [x0, y0, x1 - x0, y1 - y0]


def make(image: str | Path, out_dir: str | Path, *, route: str = "cel",
         shapes: int = MAX_SHAPES, size: int = WORK_SIZE, log=_say,
         progress=None, cut_bg: bool = False,
         no_crop: bool = False, should_stop=None) -> dict:
    """도안 하나를 끝까지 만든다. 반환값은 `report.json`과 같은 딕셔너리.

    `progress(0~1, 단계 이름)`은 창의 진행 막대용이다.

    중단은 **`should_stop`이 맡는다** — 참을 돌려주면 다음 검사점에서
    `Cancelled`가 올라 그대로 밖으로 나간다 (`engine.stop` 문서). 진행 콜백이
    예외를 올려도 같은 결과지만, 그쪽은 빽빽한 반복문에만 걸려 있어 혼자서는
    분 단위로 늦는다.

    **다 만든 판은 안 버린다** — 노선이 끝난 뒤(KFPS·FLS·report 쓰기)로는
    검사점이 없고, 맨 끝 진행 콜백이 올리는 취소도 여기서 삼킨다. 그 자리에서
    멈춰 봐야 몇 초 아끼자고 완성된 도안을 실패로 닫는 셈이다.
    """
    if route not in ROUTES:
        raise SystemExit(msg("모르는 노선: {route} ({routes})",
                             route=route, routes="|".join(ROUTES)))
    # 하드캡은 clamp가 아니라 **오류**다 — 조용히 3,000으로 줄이면 사용자는
    # 요청한 상한이 지켜졌다고 믿는다 (게임은 3,000 초과 그룹을 통째로 거부한다)
    shapes = int(shapes)
    if not 1 <= shapes <= MAX_SHAPES:
        raise SystemExit(msg("레이어 상한이 범위 밖: {shapes} (1~{cap})",
                             shapes=shapes, cap=MAX_SHAPES))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with stopping(should_stop):
        # 전처리 (요구사항 §4: 배경 제거·크롭) — 실패·미발동은 경고만 (한 버튼 원칙)
        # ① 배경 제거: 알파 없는 입력(사진·JPG)은 배경까지 도형으로 그려지므로
        #    신경망 알파(isnet-anime)로 인물만 딴다. **부를 때만 돈다** —
        #    cut_bg가 참일 때뿐이고, 기본은 프레임을 그대로 도안에 담는다
        src = Path(image)
        bgcut = False
        rgba = read_rgba(src)
        if cut_bg and bool(rgba[..., 3].min() >= 250):
            from .bgremove import matte

            log(msg("배경 제거 중… (신경망 알파)"))
            if progress:
                progress(0.0, msg("배경 제거"))
            a = matte(rgba[..., :3], log=log)
            if a is not None:
                rgba[..., 3] = a
                bgcut = True
                log(msg("  인물 알파 생성"))
        # ② 크롭: 인물 bbox + 여백으로 잘라 작업 해상도를 인물에 몰아준다 —
        #    프레임 구석의 인물이 작게 뭉개지는 것을 막는다. 알파가 프레임을 다
        #    채우면(표준 검증 이미지 포함) 미발동이라 기존 도안과 같다
        crop = None
        if not no_crop and not os.environ.get("FS_NO_CROP"):
            oh, ow = rgba.shape[:2]
            rgba, crop = _crop_to_subject(rgba)
            if crop is not None:
                log(msg("  크롭: {ow}×{oh} → {cw}×{ch} (인물 bbox + 여백 2%)",
                        ow=ow, oh=oh, cw=crop[2], ch=crop[3]))
        if bgcut or crop is not None:
            src = run_file(out, "cutout.png")
            write_png(src, np.dstack(
                [cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2BGR), rgba[..., 3]]))
            log(msg("  전처리 결과 → {name}", name=src.name))
        # 노선 본체는 여기서 부른다 — 노선 모듈이 이 파일의 공용 헬퍼를 쓰므로
        # 임포트를 함수 안에 둔다 (모듈 수준이면 서로 물린다)
        from .route_cel import _make_cel
        from .route_line import _make_line
        from .route_painter import _make_painter

        rep = (_make_painter(src, out, shapes, size, log, progress) if route == "painter"
               else _make_line(src, out, shapes, size, log, progress) if route == "line"
               else _make_cel(src, out, shapes, size, log, progress))
        rep["input"]["bgcut"] = bgcut
        rep["input"]["crop"] = crop
    # 여기부터는 취소 밖이다 — 노선이 끝났으니 판은 완성이고, 게임이 읽는
    # 파일을 안 굽고 닫으면 그 판은 쓸모가 없다 (몇 초짜리다)
    rep["kfps"] = _write_kfps_json(out, log)
    rep["fls"] = _write_fls(out, log)
    rep["route"] = route
    rep["shapes"] = shapes
    rep["source"] = str(image)
    rep["sec"] = round(time.time() - t0, 1)
    run_file(out, "report.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")

    log(msg("\n총 {sec:.0f}s → {out}", sec=rep["sec"], out=out))
    log(rep["verdict"])
    if progress:
        try:
            progress(1.0, msg("완료"))
        except Cancelled:                  # 다 만들었다 — 취소로 닫지 않는다
            pass
    return rep


def _write_kfps_json(out: Path, log) -> dict:
    """도안 → KFPS 편집기·임포터 타입코드 JSON — 두 노선 공통.

    실패해도 도안은 유효하므로 경고만 한다 (한 버튼 원칙). 통계는 report에
    실린다 — approx > 0이면 word 없는 도형이 근사로 나갔다는 뜻이다.
    """
    from .catalog import Catalog, default_catalog_path
    from .kfpsjson import export_typecode
    from .model import LayerPlan

    try:
        plan = LayerPlan.load(find_run_file(out, "plan.json"))
        data, st = export_typecode(plan, Catalog(default_catalog_path()))
        kfps = run_file(out, "kfps.json")
        kfps.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        log(msg("KFPS 편집기 JSON → {name} (도형 {n}장",
                name=kfps.name, n=len(data["shapes"]))
            + (msg(" · 근사 {n}", n=st["approx"]) if st["approx"] else "") + ")")
        return st
    except Exception as e:                       # noqa: BLE001
        log(msg("경고: KFPS JSON 생성 실패 — {err}", err=e))
        return {"error": str(e)}


def _write_fls(out: Path, log) -> dict:
    """도안 → 게임 컨테이너 폴더 + FLS 프로젝트(`.3so`) — 세 노선 공통.

    굽자마자 **게임이 읽는 파일**이 도안 옆에 선다 (`flsexport`를 따로 칠
    일이 없다). 컨테이너 폴더를 게임 저장 컨테이너 뿌리에 두면 저장 그리드에
    뜨고, `.3so`는 FLS 편집기가 그대로 연다.

    실패해도 도안은 유효하므로 경고만 한다 (한 버튼 원칙).
    """
    from ..paths import run_label
    from .fls import bridge
    from .fls.folder import GROUP_PREFIX, safe_name

    try:
        plan_path = find_run_file(out, "plan.json")
        label = run_label(plan_path)
        # 설 자리를 못 박아 준다 — `export_folder`는 이미 있으면 `_2`를 붙이는데,
        # 같은 폴더에 다시 구우면 컨테이너가 쌓인다. 판마다 폴더는 하나다.
        dest = out / (GROUP_PREFIX + safe_name(label))
        folder, st = bridge.plan_folder(plan_path, dest)
        proj, _ = bridge.plan_project(plan_path, run_file(out, "3so"))
        st["project"] = str(proj)
        log(msg("FLS·게임 파일 → {folder}/ · {proj} "
                "(레이어 {layers:,}장, 마스크 {masks:,})",
                folder=folder.name, proj=proj.name,
                layers=st["layers"], masks=st["masks"]))
        if st.get("skipped"):
            n = sum(st["skipped"].values())
            log(msg("  경고: 카탈로그 도형 id를 모르는 {n}장을 뺐다 ({kinds})",
                    n=n, kinds=", ".join(sorted(st["skipped"]))))
        return st
    except Exception as e:                       # noqa: BLE001
        log(msg("경고: FLS·게임 파일 생성 실패 — {err}", err=e))
        return {"error": str(e)}


def _bundle_cache_path(rgba: np.ndarray, size: int):
    """`FS_BUNDLE_CACHE=1`일 때 이 입력의 앞단 캐시 파일 — 아니면 None.

    **계측 전용이고 기본은 꺼져 있다.** 앞단(SR + 선화 두 판)이 장당 ~27초라
    회귀 스윕에서 판을 여러 벌 구울 때 그 몫이 통째로 반복된다 — 배치 축을
    재는데 앞단을 다시 도는 것은 값이 없다. 키는 입력 배열 자체의 해시라
    전처리(배경 제거·크롭)까지 반영된다.

    `work/cache/`는 서술자 캐시가 이미 사는 자리다 (`paths.work_root`).
    """
    if os.environ.get("FS_BUNDLE_CACHE", "0") == "0":
        return None
    import hashlib

    from ..paths import work_root

    h = hashlib.blake2b(np.ascontiguousarray(rgba), digest_size=16).hexdigest()
    d = work_root() / "cache" / "bundle"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{h}-{size}.npz"


def _source_bundle(rgba: np.ndarray, size: int, log):
    """노선이 받을 중간본 + 그 해상도의 선화 지도를 만든다.

    반환 (중간본, basic 지도, detail 지도, **원화 판** 지도). 선화를 **작업
    해상도로 줄이기 전에** 뽑는 것이 요점이다 — 축소가 가는 선을 씻어 점선으로
    만든다 (점선도가 몇 배로 뛴다).

    detail 판은 **있으면 쓴다** — Basic이 놓친 세부선을 낮은 우선순위 증거로
    얹는 자리다 (`celfit.evidence`). 모델이 없으면 None이고 Basic만으로 돈다.

    ## §25 원천이 하나면 그 모델의 판단이 곧 도안이다

    선화 판 하나(SR 중간본의 basic)를 진실로 삼으면 **SR이 지어낸 선**과
    **원화에 실제로 있는 선**이 같은 무게를 받는다. SR은 4배로 늘리며 없던
    윤곽을 매끈하게 만들어 내고, detail 판은 basic의 사각지대를 메우는 대신
    해칭·잎사귀 노이즈를 함께 얹는다 (실측 09: 선 px +80%, 그중 48%가 detail
    전용).

    그래서 **원화 해상도에서 한 판 더 뽑는다** — SR을 안 태운 입력에 같은
    basic 모델을 걸고, 그 지도를 중간본 해상도로 늘려 나란히 둔다. 두 판이
    함께 본 선은 강한 증거이고, SR 판에만 있는 선은 값을 더 받아야 산다
    (`celfit.engine._ink_mul`). 지우지 않고 **비싸게** 만드는 것이 요점이다 —
    사라질 선도 제 값을 하면 그어야 하고, 무엇이 값인지는 이미 λ가 답한다.

    SR을 안 태웠으면(원본이 이미 크거나 모델이 없으면) basic이 곧 원화 판이라
    이 지도는 None이고, 아래 단은 지지를 1로 본다 — **모델 부재 폴백은 그대로**다.

    `FS_BUNDLE_CACHE=1`이면 결과를 `work/cache/bundle`에 재사용한다
    (계측 전용 — `_bundle_cache_path` 문서).
    """
    import cv2

    from . import lineart, upscale
    from .celart import _fill_bg_nearest

    cache = _bundle_cache_path(rgba, size)
    if cache is not None and cache.is_file():
        z = np.load(cache)
        log(msg("  앞단 캐시 재사용 (FS_BUNDLE_CACHE)"))
        return (z["big"], z["line"] if "line" in z else None,
                z["detail"] if "detail" in z else None,
                z["native"] if "native" in z else None)
    stop_here()
    big = upscale.prepare(rgba, size, log=log)
    stop_here()
    detail = native = None
    sel = big[..., 3] >= 128
    rgb = _fill_bg_nearest(big[..., :3], sel) if not sel.all() else big[..., :3]
    line = lineart.extract(rgb, log=log, cap=True)
    stop_here()
    if line is not None and lineart.available("detail"):
        detail = lineart.extract(rgb, log=log, cap=True, variant="detail")
        stop_here()
        if detail is not None:
            log(msg("  선화 detail 판도 증거로 얹는다"))
    if line is not None and big.shape[:2] != rgba.shape[:2]:
        # §25 — SR을 태웠으면 **원화 해상도에서도** 같은 basic 모델을 건다.
        # 지도를 중간본 크기로 늘려 나란히 둘 뿐, 경로 원천으로는 안 쓴다
        # (원천은 여전히 중간본 판이다 — 여기서는 "SR만 본 선인가"만 묻는다)
        sel0 = rgba[..., 3] >= 128
        rgb0 = (_fill_bg_nearest(rgba[..., :3], sel0) if not sel0.all()
                else rgba[..., :3])
        nat = lineart.extract(rgb0, log=log, cap=True)
        stop_here()
        if nat is not None:
            native = cv2.resize(nat, (big.shape[1], big.shape[0]),
                                interpolation=cv2.INTER_LINEAR)
            log(msg("  원화 해상도 선화도 증거로 얹는다 (SR 전용 선을 가른다)"))
    if cache is not None:
        got = {"big": big}
        if line is not None:
            got["line"] = line
        if detail is not None:
            got["detail"] = detail
        if native is not None:
            got["native"] = native
        np.savez_compressed(cache, **got)
    return big, line, detail, native


def _reach_check(plan, cat) -> dict:
    """**도안 = 인게임인가** — 플랜이 게임에 그대로 갈 수 있는 것만 쓰나.

    도안 쪽 지표(RMSE·lpips)는 전부 **플랜 렌더**를 보므로, 렌더가 게임과 다르게
    그리는 축이 섞이면 그 결함을 통째로 못 본다. 실제로 두 번 그랬다 — 반투명
    도형(렌더는 불투명으로 그렸다)과 기울기(주입이 조용히 뺐다). 둘 다 도안
    수치는 멀쩡하고 인게임만 갈렸다. 그래서 **자를 여기에 하나 세운다**: 값이
    싸고(플랜만 읽는다) 게임이 없어도 돌고, `make`의 판정 줄에 바로 뜬다.

    기울기 쪽은 2026-09-01에 자리를 찾아(레코드 +0x70) 주입·저장 왕복이
    실측으로 닫혔다. 그래서 이 자가 묻는 것이 "그 축을 쓰나"에서 **"쓴 값이
    게임 입력 격자 위인가"**로 바뀌었다 — 자를 지운 것이 아니라 옮긴 것이다.

    인게임 대조는 이 자가 못 보는 것까지
    보지만 게임이 켜져 있어야 한다 — 둘은 겹치는 자가 아니라 층이 다르다.
    """
    trans = sorted({l.shape for l in plan.layers
                    if l.shape in cat.shapes and not cat[l.shape].opaque})
    # **기울기는 이제 도달한다** — 주입이 레코드 +0x70에 쓰고 저장 왕복도
    # 정확하다 (2026-09-01 인게임 실측). 그래서 자가 "기울기가 있나"에서
    # **"게임이 그대로 낼 수 있는 값인가"**로 바뀌었다: 유한하고 · 확인된 범위
    # 안이고 · 입력 격자(0.01) 위인가. 검사를 지우는 것이 아니라 옮긴 것이다 —
    # 이 자를 세운 까닭(렌더가 게임과 다르게 그리는 축을 잡는다)은 그대로다.
    n_skew = sum(1 for l in plan.layers if not representable(l.skew))
    n_mask = sum(1 for l in plan.layers if l.mask)
    bad = []
    if trans:
        bad.append(msg("반투명 도형 {kinds}", kinds="·".join(trans)))
    if n_skew:
        bad.append(msg("게임이 못 내는 기울기 {n}장", n=n_skew))
    if n_mask:
        bad.append(msg("마스크 {n}장", n=n_mask))
    return {"id": "reach", "ok": not bad,
            "text": msg("게임에 그대로 감") if not bad
                    else msg("게임에 그대로 못 감 — {reasons}",
                             reasons=" · ".join(bad))}


def _one_region(mask: np.ndarray, color) -> list:
    """마스크 하나 = 영역 하나 (line 노선·선 도안의 최소 CelArt 구성용)."""
    from .celart import Region

    ys, xs = np.nonzero(mask)
    return [Region(rid=0, color=tuple(int(v) for v in color),
                   area=int(mask.sum()),
                   bbox=(int(xs.min()), int(ys.min()),
                         int(xs.max()) + 1, int(ys.max()) + 1))]
