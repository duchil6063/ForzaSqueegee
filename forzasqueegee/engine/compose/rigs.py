"""차 한 대의 면 지도와 옆면 뼈대 — 실측이 프리셋보다 우선한다."""

from __future__ import annotations

from pathlib import Path

from ...game import seam as gseam, surface as gsurf
from ...i18n import msg
from .look import Look
from .place import ROLE_MAIN, Place, SideRig, fit_on


def _hood_seed(media: str | None) -> float | None:
    """윗면 후드 구간을 고를 씨앗 u — 설치본을 못 박았을 때만 (없으면 None)."""
    if not media:
        return None
    try:
        from ...game import locators as glocs

        reg, items = glocs.for_car(media)
        return glocs.hood_u(reg, dict(items))
    except Exception:                              # noqa: BLE001 — 보조 근거다
        return None


def _bumper_seed(media: str | None, surface: str) -> float | None:
    """앞·뒤 하부 밴드를 앉힐 씨앗 v (범퍼 로케이터) — 없으면 None.

    `rear`에만 쓴다 (`game.locators.bumper_v` 설명 — 프론트는 상자 아래끝이
    곧 보이는 아래끝이라 씨앗이 필요 없고, 앞범퍼 로케이터는 범퍼 윗부분이다).
    """
    if not media:
        return None
    try:
        from ...game import locators as glocs

        reg, items = glocs.for_car(media)
        return glocs.bumper_v(reg, dict(items), surface)
    except Exception:                              # noqa: BLE001 — 보조 근거다
        return None


def _avoid_on(media: str | None, maps: dict[str, gsurf.SurfaceMap],
              name: str) -> list[tuple[float, float, str]]:
    """그 면에서 얼굴이 피할 부품 자리 — 설치본을 못 박았을 때만 (없으면 빈 목록)."""
    if not media:
        return []
    try:
        from ...game import locators as glocs

        reg, items = glocs.for_car(media)
        return glocs.avoid_points(reg, dict(items), name)
    except Exception:                              # noqa: BLE001 — 보조 근거다
        return []


def _arch_fallback(media: str | None, maps: dict[str, gsurf.SurfaceMap]
                   ) -> dict[str, dict]:
    """마스크가 아치를 모르는 차의 휠 자리 어림 (`game.locators.arch_fallback`)."""
    if not media:
        return {}
    try:
        from ...game import locators as glocs

        return glocs.arch_fallback(media, maps)
    except Exception:                              # noqa: BLE001 — 보조 근거다
        return {}


def _place_for(name: str, lk: Look, maps: dict[str, gsurf.SurfaceMap],
               preset: dict[str, dict[str, float]], *, anchor: str,
               bias_x: float, fill: float, group_unit: float,
               notes: list[str], overshoot: float = 1.0,
               full_box: tuple[float, float, float, float] | None = None,
               tilt: float = 0.0, mirror: bool = False) -> Place:
    """실측 지도가 있으면 계산하고, 없거나 의심스러우면 프리셋으로 물러난다."""
    smap = maps.get(name)
    if smap is not None and not smap.uncertain:
        got = fit_on(smap, lk, anchor=anchor, bias_x=bias_x, fill=fill,
                     group_unit=group_unit, full_box=full_box,
                     overshoot=overshoot, tilt=tilt, mirror=mirror)
        if got is not None:
            got.surface = name
            return got
        notes.append(msg("{name}: 도색 마스크에 도안 비율({aspect:.2f})이 안 들어갔다 "
                         "— 프리셋으로 앉힌다", name=name, aspect=lk.aspect))
    elif smap is None:
        notes.append(msg("{name}: 실측 지도가 없다 — 프리셋", name=name))
    else:
        notes.append(msg("{name}: 실측이 의심스럽다({note}) — 프리셋",
                         name=name, note=smap.note))
    p = dict(preset.get(name, {"x": 0.0, "y": 0.0, "scale": 0.25, "rot": 0.0}))
    return Place(surface=name, plan=Path(), x=p["x"], y=p["y"],
                 scale=p["scale"], rot=p.get("rot", 0.0), why=msg("프리셋"))


def carfiles_pick(car: str | None) -> str | None:
    """표시 이름 → 설치 미디어명 (문턱을 넘었을 때만). 설치본이 없으면 None.

    **잰 차라면 크기도 본다**: 프로브가 잰 옆면 가로로 격이 다른 후보를 먼저
    뺀다 (`gsurf.probe_side_width`). 미디어명을 고르는 문이 하나여야 면 지도·
    탭 구성·미리보기가 **같은 차**를 본다 — 갈리면 배치 수치와 탭 번호가
    딴 차의 것으로 돈다.
    """
    if not car:
        return None
    try:
        from ...game import carfiles
        return carfiles.pick_media(car, side_width=gsurf.probe_side_width(car))[0]
    except Exception:                              # noqa: BLE001
        return None


def probe_ok(car: str | None, media: str | None) -> bool:
    """실측 지도(`car`)가 **못 박은 설치 차량과 같은 차**인가.

    `--media`로 다른 차를 못 박았는데 실측 지도를 그대로 깔면, 그 차에 없는
    면이 실측 쪽에서 새어 들어오고 오토핏이 **남의 차 화면 warp**로 도안을
    민다. 못 박은 차가 없으면(자동 매칭) 실측은 늘 쓴다 — 그때는 둘이 같은
    차에서 왔다.
    """
    if not media or not car:
        return True
    return carfiles_pick(car) == media


def surfaces_for(car: str | None, media: str | None = None,
                 notes: list[str] | None = None) -> dict[str, gsurf.SurfaceMap]:
    """차 하나의 면 지도 — **설치 파일 마스크가 실측보다 우선한다**.

    설치 마스크는 게임이 쓰는 원본이라 유리·소프트탑까지 완전하고, 프로브
    병리(front 접힘·퇴화 상자)가 없다 (`game/carfiles.py`, 실측 12대 대조
    90~100%). 실측 지도는 화면 warp가 필요한 자(오토핏·프로브)만 쓴다 —
    그쪽은 제 파일을 따로 읽는다. `media`를 주면 매칭을 건너뛰고 그 차를 쓴다.

    **실측이 하나 더 있다**: 설치 마스크가 "칠할 수 있다"고 말하는 자리 중
    윗면 앞·뒷유리는 게임이 안 그린다. 그 자리는 설치 파일에 없고 프로브만
    안다 (`game.seam.top_glass`) — 잰 차면 `top` 지도에 `drawn`으로 달아 둔다.
    """
    notes = notes if notes is not None else []
    probe = gsurf.load(car) if car and probe_ok(car, media) else {}
    maps = dict(probe)
    if car and media and not maps and not probe_ok(car, media):
        notes.append(msg("실측 지도({car})는 못 박은 차({media})의 것이 아니다 — "
                         "안 쓴다 (설치 지도만)", car=car, media=media))
    try:
        from ...game import carfiles, cars as gcars
        if media is None and car:
            side_w = gsurf.probe_side_width(car)
            media, cands = carfiles.pick_media(car, side_width=side_w)
            if media and side_w:
                # 크기가 이름을 이겼으면 **말은 하고 간다** — 고른 차가 조용히
                # 바뀌면 배치 수치가 왜 달라졌는지 아무 데도 안 남는다
                plain = carfiles.pick_media(car)[0]
                if plain and plain != media:
                    notes.append(msg(
                        "이름은 {plain}을(를) 골랐지만 실측 옆면 {width:.0f}유닛과 "
                        "격이 다르다 ({size}) — {media}로 간다",
                        plain=plain, width=side_w,
                        size=gcars.size_text(plain) or msg("크기 모름"),
                        media=media))
            if media is None and cands:
                # **후보를 들려 준다** — 문턱을 못 넘으면 설치 지도를 통째로 버리고
                # 프리셋으로 물러나는데, 사람이 못 박을 이름을 모르면 고칠 수가
                # 없다 (실측: Type2 2.5 · F-150 2.5 · 911 3.0 · CRX Mugen 3.0이
                # 전부 미달이었다). 점수가 다 3점대로 갈리지 않으므로 **크기를
                # 같이 들려 준다** — 이름은 못 갈라도 경트럭 529와 로드스터
                # 902는 사람이 한눈에 가른다.
                top = ", ".join(
                    f"{m}({s:.1f}, {gcars.size_text(m) or msg('크기 모름')})"
                    for s, m in cands[:3])
                notes.append(msg("설치 차량 매칭이 애매하다 (후보 {top}) — 실측 "
                                 "지도만 쓴다. 못 박으려면 `--media <이름>`",
                                 top=top))
        if media:
            imaps = carfiles.surface_maps(media)
            maps.update(imaps)
            notes.append(msg("설치 파일 면 지도를 쓴다 ({media} — 면 {n}개)",
                             media=media, n=len(imaps)))
            # 실루엣이 아예 없는 백지 마스크(상자 전부 흰 판)는 아무것도 말해
            # 주지 않는다 — 실제 보이는 영역은 훨씬 좁을 수 있다 (CRX Mugen의
            # front가 그렇다: 마스크는 353×175 전부인데 실차는 범퍼 띠뿐이다).
            blank = [n for n in ("front", "rear")
                     if n in imaps and imaps[n].fill >= 0.985]
            if blank:
                notes.append(msg("{faces} 마스크가 백지다 (실루엣 없음) — "
                                 "이 면의 미리보기는 실제 보이는 영역보다 넓게 나온다",
                                 faces="·".join(blank)))
        elif carfiles.install_dir() is None:
            # 설치 폴더가 없으면 `list_cars`가 빈손이라 후보도 예외도 안 난다 —
            # 여기서 말 안 하면 프리셋으로 물러난 이유가 아무 데도 안 남는다
            notes.append(msg("게임 설치 폴더를 못 찾았다 — 면 지도 없이 프리셋으로 "
                             "앉힌다. `python -m forzasqueegee gamedir <경로>`로 "
                             "못 박을 것"))
    except Exception as e:                         # 설치본이 없어도 죽지 않는다
        notes.append(msg("설치 파일 면 지도 불가 ({error}) — 실측 지도만 쓴다",
                         error=e))
    top = maps.get("top")
    if top is not None:
        bands = gseam.top_glass(top, probe.get("top"))
        if bands:
            top.drawn = gseam.top_body(top, bands)
            notes.append(msg("윗면 유리를 실측으로 뺀다 (프로브): {bands}",
                             bands=" · ".join(f"u {a:.0f}~{b:.0f}"
                                              for a, b in bands)))
        elif probe.get("top") is not None:
            notes.append(msg("윗면 프로브가 있지만 유리를 못 갈랐다 — 윗면을 통째로 쓴다"))
    return maps


def side_rigs(maps: dict[str, gsurf.SurfaceMap],
              notes: list[str] | None = None,
              media: str | None = None) -> dict[str, SideRig]:
    """좌우 옆면의 뼈대 — 벨트라인으로 자른 차체 지도와 유리 이음새.

    **여기가 '유리에 안 그려진다'를 처음으로 아는 자리다.** 설치 마스크의 옆면은
    그린하우스를 흰 판으로 갖고 있지만 게임은 벨트라인 위에 안 그린다 (프로브
    15면 대조). 그래서 배치·마스크 판정에 쓰는 지도를 `body`로 바꿔 두면 인물도
    꾸밈 그룹도 유리를 안 넘본다.
    """
    notes = notes if notes is not None else []
    est_arch: dict[str, dict] | None = None        # 필요할 때 한 번만 계산
    est_used = False
    out: dict[str, SideRig] = {}
    for name in ROLE_MAIN:
        smap = maps.get(name)
        if smap is None or smap.uncertain or smap.mask.size <= 1:
            continue
        try:
            geom = gseam.side_geom(smap)
        except Exception as e:                     # 기하가 이상해도 죽지 않는다
            notes.append(msg("{name}: 옆면 뼈대를 못 읽었다 ({error}) — 설치 마스크 그대로",
                             name=name, error=e))
            continue
        if geom.body_height <= 8.0:
            notes.append(msg("{name}: 벨트라인이 이상하다 (차체 높이 "
                             "{height:.0f}유닛) — 설치 마스크 그대로",
                             name=name, height=geom.body_height))
            continue
        win = maps.get("window_" + name.split("_")[1])
        sm = None
        if win is not None and not win.uncertain and win.mask.size > 1:
            try:
                sm = gseam.seam(smap, win, geom)
            except Exception as e:
                notes.append(msg("{name}: 유리 이음새 계산 실패 ({error})",
                                 name=name, error=e))
        body = gseam.body_map(smap, geom)
        # 마스크에 아치 구멍이 없는 차(설치본 99대) — 휠 자리를 로케이터 어림으로
        # 채운다: 인물 자리 계산(door_span·person_span)이 뒷휠을 피하고, 배치판·
        # 미리보기가 그 자리를 구멍으로 보인다 (CRX 실차: 우측면 인물이 휠에
        # 삼켜졌는데 배치판은 아무것도 안 보였다).
        est = None
        if len(geom.wheels) < 2:
            if est_arch is None:
                est_arch = _arch_fallback(media, maps)
            est = est_arch.get(name)
        if est is not None:
            geom.wheels = est["wheels"]
            body = gseam.punch_arches(body, est["wheels"], est["vc"])
            est_used = True
        out[name] = SideRig(name=name, smap=smap, body=body,
                            geom=geom, seam=sm,
                            rear_dir=1.0 if name == "side_left" else -1.0,
                            parts=tuple(_avoid_on(media, maps, name)))
    if est_used and est_arch:
        k = next(iter(est_arch.values()))["k"]
        notes.append(msg("휠아치가 마스크에 없다 — 로케이터 어림으로 자리를 뚫고 피한다 "
                         "(범퍼 기반 배율 {k:g}유닛/m · 자리 오차 중앙 0.11 m)", k=k))
    if out:
        first = next((r for r in out.values() if r.parts), None)
        if first is not None:
            what = " · ".join(f"{lab}(u={u:.0f})" for u, _v, lab in first.parts)
            notes.append(msg("부품 자리를 잰다 (설치 파일 로케이터): {what} — "
                             "얼굴이 그 위에 안 앉게 민다", what=what))
        g = next(iter(out.values())).geom
        hv = g.roof - g.sill
        notes.append(
            msg("옆면 뼈대: 로커 {sill:.0f} · 벨트라인 {belt:.0f} · 루프 "
                "{roof:.0f} (차체는 벨트라인 아래 {pct:.0f}%",
                sill=g.sill, belt=g.belt, roof=g.roof,
                pct=g.body_height / max(1e-6, hv) * 100)
            + (msg(", 유리 이음새 겹침 {iou:.2f}",
                   iou=next(iter(out.values())).seam.iou)
               if next(iter(out.values())).seam else msg(", 유리 면 없음")) + ")"
            + (f" [{g.note}]" if g.note else ""))
    return out
