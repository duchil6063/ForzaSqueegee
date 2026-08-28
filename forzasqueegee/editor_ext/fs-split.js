/* ForzaSqueegee — 내장 KFPS 편집기 **선으로 가르기**.
 *
 * vendor/kfps-editor는 무수정 사본이라(README의 SHA-256 대조) 편집기 파일을
 * 고치지 않는다. 대신 서버(kfpseditor.py)가 index.html을 내줄 때 이 파일을
 * editor.js **뒤에** 끼워 넣고, 여기서 기능 하나를 얹는다.
 *
 * ## 무엇을 하나
 *
 * 고른 레이어들(또는 고른 그룹의 레이어들, 통틀어 2장 이상)을 **기준선 하나로
 * 두 묶음으로 가른다.** 게임의 면 이음새를 손으로 타는 자다 — 도안이 옆면을
 * 넘겼으면 벨트라인에 선을 놓고 갈라, 위쪽 묶음을 유리 면에 따로 올린다.
 *
 * ## 걸친 레이어는 **양쪽에 다 들어간다**
 *
 * 게임 도형은 반으로 못 자른다. 그래서 기준선에 걸친 레이어는 사본을 하나 더
 * 만들어 두 묶음에 다 넣는다 — 두 묶음을 각각 이웃한 두 면에 올리면 면이 알아서
 * 제 몫만 그리므로 이음새가 안 벌어진다. **여유 공간**은 그 겹침을 기준선
 * 안쪽으로 넓히는 폭이다 (면 유닛): 진짜 이음선이 우리가 놓은 선보다 안쪽이어도
 * 빈 띠가 안 생긴다. 사본이 느는 만큼 장수도 늘므로 상한(3,000)을 먼저 본다.
 *
 * ## 편집기를 안 고치고 얹는 법
 *
 * editor.js는 IIFE가 아니라 최상위 스크립트라, 그 `function` 선언은 window에
 * 붙고 최상위 `let`/`const`(`canvas`)는 전역 렉시컬 환경에 산다 — 다른 고전
 * 스크립트인 이 파일에서 이름 그대로 닿는다. 쓰는 것은 편집기가 제 단추에
 * 쓰는 것과 같은 손잡이뿐이다 (`queueEditorMutation` · `makeFabricObject` ·
 * `pushHistory` …), 그래서 실행 취소·자동 복구·레이어 목록이 그대로 따라온다.
 */
(function () {
  "use strict";

  const NEED = ["queueEditorMutation", "makeFabricObject", "objectToShape",
                "pushHistory", "refreshLayers", "setStatus", "selectObjects",
                "objectCornerCoords", "membersForGroupIds", "vinylObjects",
                "selectedVinylObjects", "nextLayerGroupName",
                "syncCanvasObjectCoords", "applyMaskVisual",
                "styleObjectTransformControls", "updateSelectionPanel"];

  const T = {
    ko: {
      button: "선으로 가르기",
      buttonTip: "고른 레이어들(또는 고른 그룹, 통틀어 2장 이상)을 기준선 하나로 두 묶음으로 가른다. 선에 걸친 레이어는 양쪽에 다 들어간다.",
      title: "선으로 가르기",
      lead: "기준선을 끌어 옮기고, 끝의 손잡이를 끌어 돌린다.",
      angle: "각도",
      offset: "기준선 위치",
      margin: "여유 공간",
      marginTip: "기준선 양쪽으로 허용하는 겹침 폭 (면 유닛). 이 띠에 닿는 레이어는 두 묶음에 다 들어간다.",
      targets: "고른 레이어",
      sideA: "A쪽",
      sideB: "B쪽",
      both: "양쪽 (사본이 는다)",
      total: "가른 뒤 총 장수",
      apply: "가르기",
      cancel: "취소",
      overCap: (n, cap) => `총 ${n.toLocaleString()}장이 편집기 상한 ${cap.toLocaleString()}장을 넘는다 — 여유 공간을 줄이거나 선을 옮길 것.`,
      needTwo: "레이어를 2장 이상 고르고 누르세요 (그룹을 고르면 그 그룹의 레이어 전부).",
      empty: "한쪽이 비었다 — 기준선을 도안 안으로 옮길 것.",
      done: (a, b, dup) => `가르기: A ${a.toLocaleString()}장 · B ${b.toLocaleString()}장`
        + (dup ? ` (선에 걸친 ${dup.toLocaleString()}장은 양쪽에 다 넣었다)` : ""),
      unit: "유닛",
    },
    en: {
      button: "Split at Line",
      buttonTip: "Split the selected layers (or the selected group, two or more in total) into two groups at one line. Layers the line crosses go into both.",
      title: "Split at a Line",
      lead: "Drag the line to move it; drag an end handle to turn it.",
      angle: "Angle",
      offset: "Line position",
      margin: "Clearance",
      marginTip: "How far past the line each side may reach (surface units). A layer touching this band goes into both groups.",
      targets: "Selected layers",
      sideA: "Side A",
      sideB: "Side B",
      both: "Both (adds copies)",
      total: "Layers after the split",
      apply: "Split",
      cancel: "Cancel",
      overCap: (n, cap) => `${n.toLocaleString()} layers would exceed the editor maximum of ${cap.toLocaleString()}. Reduce the clearance or move the line.`,
      needTwo: "Select two or more layers first (selecting a group takes all of its layers).",
      empty: "One side is empty - move the line into the artwork.",
      done: (a, b, dup) => `Split: side A ${a.toLocaleString()}, side B ${b.toLocaleString()}`
        + (dup ? ` (${dup.toLocaleString()} layer(s) on the line went into both)` : ""),
      unit: "units",
    },
  };

  function lang() {
    const got = window.__fsI18n?.lang;
    return got === "ko" ? "ko" : "en";
  }

  function t() { return T[lang()]; }

  // ── 기하 ──
  //
  // 편집기 장면 좌표는 게임 유닛에 y만 뒤집은 것이다 (`fabricPropsFromFh6Data`
  // — left = x, top = −y). 사람에게 보이는 각도는 **게임 규약**(반시계, y 위)
  // 이므로, 게임 각 θ의 법선 (−sinθ, cosθ)를 장면 좌표로 옮기면
  // (−sinθ, −cosθ)다. 기준선은 `n·p = D` 한 줄이고 부호가 곧 A/B다.
  const state = {
    open: false,
    deg: 0,             // 기준선 각도 (게임 규약, 도)
    offset: 0,          // 원점에서 법선 방향으로 잰 기준선 자리 (유닛)
    margin: 0,          // 여유 공간 (유닛)
    targets: [],        // [{obj, dmin, dmax}] — 코너는 열 때 한 번만 잰다
    corners: [],        // 대상마다 코너 4점 (장면 좌표)
    objects: [],        // 화면에 그린 오버레이 fabric 객체
    counts: { a: 0, b: 0, both: 0 },
    drag: null,
    prevSelection: null,
    prevSkipTargetFind: false,
  };

  function normal() {
    const th = state.deg * Math.PI / 180;
    return { x: -Math.sin(th), y: -Math.cos(th) };
  }

  function direction() {
    const n = normal();
    return { x: -n.y, y: n.x };
  }

  function signed(pt) {
    const n = normal();
    return n.x * pt.x + n.y * pt.y - state.offset;
  }

  /** 기준선 위의 기준점 — 대상 상자 중심을 선에 내린 발. */
  function anchor() {
    const n = normal();
    const c = state.center || { x: 0, y: 0 };
    const d = signed(c);
    return { x: c.x - n.x * d, y: c.y - n.y * d };
  }

  // ── 대상 고르기 ──
  //
  // 고른 것이 레이어면 그 레이어들이고, 고른 것이 그룹에 든 레이어면 **그 그룹
  // 전부**다 (그룹 하나를 고르는 것이 곧 레이어 하나를 고르는 것이라, 편집기의
  // [Ungroup]과 같은 규약으로 넓힌다). 여럿이면 합집합이다.
  function pickTargets() {
    const selected = selectedVinylObjects();
    if (!selected.length) return [];
    const ids = [...new Set(selected.map((o) => o.kloudy?.group_id).filter(Boolean))];
    const members = ids.length ? membersForGroupIds(ids) : [];
    const set = new Set([...selected, ...members]);
    const order = vinylObjects();
    return order.filter((o) => set.has(o));
  }

  function measure(objects) {
    return objects.map((obj) => {
      const c = objectCornerCoords(obj);
      return [c.tl, c.tr, c.br, c.bl];
    });
  }

  function recount() {
    const m = Math.max(0, state.margin);
    let a = 0, b = 0, both = 0;
    for (const corners of state.corners) {
      let lo = Infinity, hi = -Infinity;
      for (const p of corners) {
        const d = signed(p);
        if (d < lo) lo = d;
        if (d > hi) hi = d;
      }
      const inA = lo <= m;
      const inB = hi >= -m;
      if (inA && inB) both += 1;
      else if (inA) a += 1;
      else b += 1;
    }
    state.counts = { a: a + both, b: b + both, both };
    return state.counts;
  }

  /** 이 레이어가 어느 쪽인가 — [A쪽인가, B쪽인가]. 둘 다 참일 수 있다. */
  function sidesOf(corners) {
    const m = Math.max(0, state.margin);
    let lo = Infinity, hi = -Infinity;
    for (const p of corners) {
      const d = signed(p);
      if (d < lo) lo = d;
      if (d > hi) hi = d;
    }
    return [lo <= m, hi >= -m];
  }

  // ── 화면의 기준선 ──
  const LINE_LEN = 40000;
  const HANDLE_GAP = 190;    // 회전 손잡이가 기준점에서 떨어진 거리 (유닛)
  const LABEL_GAP = 90;      // A·B 이름표가 선에서 떨어진 거리 (유닛)

  function clearOverlay() {
    for (const o of state.objects) canvas.remove(o);
    state.objects = [];
  }

  function drawOverlay() {
    clearOverlay();
    if (!state.open) {
      canvas.requestRenderAll();
      return;
    }
    const n = normal();
    const d = direction();
    const a = anchor();
    const half = LINE_LEN / 2;
    const base = {
      selectable: false, evented: false, excludeFromExport: true,
      hasControls: false, hasBorders: false, objectCaching: false,
      strokeUniform: true, fsSplitOverlay: true,
    };
    const seg = (off, style) => new fabric.Line(
      [a.x + n.x * off - d.x * half, a.y + n.y * off - d.y * half,
       a.x + n.x * off + d.x * half, a.y + n.y * off + d.y * half],
      Object.assign({}, base, style));
    const made = [];
    if (state.margin > 0) {
      made.push(seg(state.margin, { stroke: "#ff8ac6", strokeWidth: 1,
                                    strokeDashArray: [6, 6], opacity: 0.9 }));
      made.push(seg(-state.margin, { stroke: "#ff8ac6", strokeWidth: 1,
                                     strokeDashArray: [6, 6], opacity: 0.9 }));
    }
    made.push(seg(0, { stroke: "#ff2f92", strokeWidth: 2 }));
    for (const s of [-1, 1]) {
      made.push(new fabric.Circle(Object.assign({}, base, {
        left: a.x + d.x * s * HANDLE_GAP, top: a.y + d.y * s * HANDLE_GAP,
        originX: "center", originY: "center", radius: 7,
        fill: "#ffffff", stroke: "#ff2f92", strokeWidth: 2,
      })));
    }
    // A·B 이름표 — 어느 쪽이 어느 묶음인지는 이 두 글자가 정한다 (각도에 따라
    // "위/아래"·"좌/우"가 뒤집히므로 방향 이름은 오히려 헷갈린다).
    for (const [side, label] of [[-1, "A"], [1, "B"]]) {
      made.push(new fabric.Text(label, Object.assign({}, base, {
        left: a.x + n.x * side * LABEL_GAP, top: a.y + n.y * side * LABEL_GAP,
        originX: "center", originY: "center", fontSize: 26,
        fontFamily: "system-ui, sans-serif", fill: "#ff2f92",
        opacity: 0.85, fontWeight: "700",
      })));
    }
    for (const o of made) {
      canvas.add(o);
      KfpsFabricAdapter.bringObjectToFront(canvas, o);
    }
    state.objects = made;
    canvas.requestRenderAll();
  }

  // ── 끌기 ──
  function scenePoint(opt) {
    return KfpsFabricAdapter.scenePoint(canvas, opt.e);
  }

  function grabRadius() {
    return 14 / Math.max(0.01, canvas.getZoom());
  }

  function onDown(opt) {
    if (!state.open || (opt.e && opt.e.button !== 0)) return;
    const p = scenePoint(opt);
    const a = anchor();
    const d = direction();
    const r = grabRadius();
    for (const s of [-1, 1]) {
      const hx = a.x + d.x * s * HANDLE_GAP;
      const hy = a.y + d.y * s * HANDLE_GAP;
      if (Math.hypot(p.x - hx, p.y - hy) <= r + 7) {
        state.drag = { mode: "turn", sign: s };
        opt.e.preventDefault?.();
        return;
      }
    }
    if (Math.abs(signed(p)) <= r + 3) {
      state.drag = { mode: "move", grab: signed(p) };
      opt.e.preventDefault?.();
    }
  }

  function onMove(opt) {
    if (!state.open || !state.drag) return;
    const p = scenePoint(opt);
    if (state.drag.mode === "move") {
      const n = normal();
      state.offset = n.x * p.x + n.y * p.y - state.drag.grab;
    } else {
      // 기준점을 축으로 돈다 — 각도만 바뀌고 선이 지나는 점은 그대로다.
      const a = anchor();
      const vx = (p.x - a.x) * state.drag.sign;
      const vy = (p.y - a.y) * state.drag.sign;
      if (Math.hypot(vx, vy) < 1e-6) return;
      // 장면 방향 (vx, vy) → 게임 각도 (y 뒤집기)
      let deg = Math.atan2(-vy, vx) * 180 / Math.PI;
      deg = ((deg % 180) + 180) % 180;
      state.deg = deg;
      const n = normal();
      state.offset = n.x * a.x + n.y * a.y;
    }
    syncFields();
    drawOverlay();
    opt.e?.preventDefault?.();
  }

  function onUp() {
    state.drag = null;
  }

  // ── 패널 ──
  let panel = null;
  const fields = {};

  function css() {
    if (document.getElementById("fsSplitStyle")) return;
    const style = document.createElement("style");
    style.id = "fsSplitStyle";
    style.textContent = `
#fsSplitPanel{position:fixed;right:18px;bottom:64px;z-index:60;width:300px;
 padding:14px 16px;border-radius:14px;font-size:12px;line-height:1.5;
 background:var(--panel,#1d1a22);color:var(--text,#f4eef7);
 border:1px solid var(--border,#3a3340);box-shadow:0 12px 34px rgba(0,0,0,.42)}
#fsSplitPanel h3{margin:0 0 2px;font-size:14px}
#fsSplitPanel p{margin:0 0 10px;opacity:.72}
#fsSplitPanel .fsRow{display:flex;align-items:center;gap:8px;margin:6px 0}
#fsSplitPanel .fsRow label{flex:1 1 auto;white-space:nowrap}
#fsSplitPanel .fsRow input{width:86px;flex:0 0 auto;padding:4px 6px;
 border-radius:7px;border:1px solid var(--border,#3a3340);
 background:var(--input,#151218);color:inherit;font:inherit;text-align:right}
#fsSplitPanel .fsUnit{flex:0 0 auto;opacity:.6;width:30px}
#fsSplitPanel .fsTally{margin:10px 0 4px;padding:8px 10px;border-radius:9px;
 background:rgba(255,255,255,.05)}
#fsSplitPanel .fsTally div{display:flex;justify-content:space-between;gap:10px}
#fsSplitPanel .fsWarn{color:#ff8f8f;margin-top:6px}
#fsSplitPanel .fsBtns{display:flex;gap:8px;margin-top:12px}
#fsSplitPanel .fsBtns button{flex:1 1 0;padding:7px 0;border-radius:9px;
 border:1px solid var(--border,#3a3340);background:var(--input,#151218);
 color:inherit;font:inherit;cursor:pointer}
#fsSplitPanel .fsBtns button.fsGo{background:#ff2f92;border-color:#ff2f92;
 color:#fff;font-weight:600}
#fsSplitPanel .fsBtns button:disabled{opacity:.45;cursor:default}`;
    document.head.appendChild(style);
  }

  function num(id, value, step) {
    const input = document.createElement("input");
    input.type = "number";
    input.id = id;
    input.step = String(step);
    input.value = String(value);
    return input;
  }

  function buildPanel() {
    css();
    const box = document.createElement("div");
    box.id = "fsSplitPanel";
    box.setAttribute("data-fs-i18n-skip", "");
    const s = t();
    const h = document.createElement("h3");
    const lead = document.createElement("p");
    box.append(h, lead);
    const rows = [["angle", "deg", 1], ["offset", "offset", 1],
                  ["margin", "margin", 1]];
    for (const [key, id, step] of rows) {
      const row = document.createElement("div");
      row.className = "fsRow";
      const label = document.createElement("label");
      label.htmlFor = `fsSplit_${id}`;
      const input = num(`fsSplit_${id}`, 0, step);
      const unit = document.createElement("span");
      unit.className = "fsUnit";
      unit.textContent = key === "angle" ? "°" : "";
      row.append(label, input, unit);
      box.append(row);
      fields[key] = { label, input };
      input.addEventListener("input", () => {
        const v = Number(input.value);
        if (!Number.isFinite(v)) return;
        if (key === "angle") state.deg = ((v % 180) + 180) % 180;
        else if (key === "offset") state.offset = v;
        else state.margin = Math.max(0, v);
        drawOverlay();
        syncTally();
      });
    }
    const tally = document.createElement("div");
    tally.className = "fsTally";
    for (const key of ["targets", "sideA", "sideB", "both", "total"]) {
      const row = document.createElement("div");
      const name = document.createElement("span");
      const value = document.createElement("b");
      row.append(name, value);
      tally.append(row);
      fields[key] = { label: name, input: value };
    }
    const warn = document.createElement("div");
    warn.className = "fsWarn";
    warn.hidden = true;
    const btns = document.createElement("div");
    btns.className = "fsBtns";
    const go = document.createElement("button");
    go.type = "button";
    go.className = "fsGo";
    const cancel = document.createElement("button");
    cancel.type = "button";
    btns.append(go, cancel);
    box.append(tally, warn, btns);
    document.body.appendChild(box);
    go.addEventListener("click", () => { apply(); });
    cancel.addEventListener("click", () => close());
    fields.warn = warn;
    fields.go = go;
    fields.cancel = cancel;
    fields.head = { h, lead, s };
    return box;
  }

  function syncFields() {
    if (!panel) return;
    const active = document.activeElement;
    const set = (key, value) => {
      const el = fields[key].input;
      if (el !== active) el.value = String(Math.round(value * 10) / 10);
    };
    set("angle", state.deg);
    set("offset", state.offset);
    set("margin", state.margin);
    syncTally();
  }

  function syncTally() {
    if (!panel) return;
    const s = t();
    const c = recount();
    const total = state.corners.length + c.both;
    const cap = typeof MAX_VINYL_LAYERS === "number" ? MAX_VINYL_LAYERS : 3000;
    const room = cap - (vinylObjects().length + c.both);
    const put = (key, name, value) => {
      fields[key].label.textContent = name;
      fields[key].input.textContent = value.toLocaleString();
    };
    put("targets", s.targets, state.corners.length);
    put("sideA", s.sideA, c.a);
    put("sideB", s.sideB, c.b);
    put("both", s.both, c.both);
    put("total", s.total, total);
    const empty = c.a === 0 || c.b === 0;
    fields.warn.hidden = !(room < 0 || empty);
    if (room < 0) fields.warn.textContent = s.overCap(vinylObjects().length + c.both, cap);
    else if (empty) fields.warn.textContent = s.empty;
    fields.go.disabled = room < 0 || empty;
  }

  function retitle() {
    if (!panel) return;
    const s = t();
    fields.head.h.textContent = s.title;
    fields.head.lead.textContent = s.lead;
    fields.angle.label.textContent = s.angle;
    fields.offset.label.textContent = `${s.offset} (${s.unit})`;
    fields.margin.label.textContent = `${s.margin} (${s.unit})`;
    fields.margin.input.title = s.marginTip;
    fields.go.textContent = s.apply;
    fields.cancel.textContent = s.cancel;
    const button = document.getElementById("fsSplitButton");
    if (button) {
      button.textContent = s.button;
      button.title = s.buttonTip;
    }
    syncFields();
  }

  // ── 열고 닫기 ──
  function open() {
    const targets = pickTargets();
    if (targets.length < 2) {
      setStatus(t().needTwo);
      return;
    }
    state.targets = targets;
    state.corners = measure(targets);
    const pts = state.corners.flat();
    const cx = (Math.min(...pts.map((p) => p.x)) + Math.max(...pts.map((p) => p.x))) / 2;
    const cy = (Math.min(...pts.map((p) => p.y)) + Math.max(...pts.map((p) => p.y))) / 2;
    state.center = { x: cx, y: cy };
    state.deg = 0;
    state.margin = 0;
    state.open = true;
    const n = normal();
    state.offset = n.x * cx + n.y * cy;
    state.prevSelection = canvas.selection;
    state.prevSkipTargetFind = canvas.skipTargetFind;
    canvas.selection = false;
    canvas.skipTargetFind = true;      // 선을 끄는 동안 도형이 안 잡힌다
    panel = panel || buildPanel();
    panel.hidden = false;
    retitle();
    drawOverlay();
  }

  function close() {
    state.open = false;
    state.drag = null;
    clearOverlay();
    canvas.selection = state.prevSelection;
    canvas.skipTargetFind = state.prevSkipTargetFind;
    if (panel) panel.hidden = true;
    canvas.requestRenderAll();
  }

  // ── 가르기 ──
  function uniqueGroupName(base, taken) {
    let name = base;
    for (let i = 2; taken.has(name); i++) name = `${base} ${i}`;
    taken.add(name);
    return name;
  }

  function apply() {
    const s = t();
    const plan = [];
    for (let i = 0; i < state.targets.length; i++) {
      const [inA, inB] = sidesOf(state.corners[i]);
      plan.push({ obj: state.targets[i], a: inA, b: inB || !inA });
    }
    const dup = plan.filter((p) => p.a && p.b).length;
    if (!plan.some((p) => p.a) || !plan.some((p) => p.b)) {
      setStatus(s.empty);
      return;
    }
    if (dup && typeof requireLayerCapacity === "function"
        && !requireLayerCapacity(dup, "split this selection")) {
      return;
    }
    // 이름 — 고른 것이 그룹 하나였으면 그 이름을 물려받는다.
    const ids = [...new Set(state.targets.map((o) => o.kloudy?.group_id).filter(Boolean))];
    const base = (ids.length === 1 && state.targets.every((o) => o.kloudy?.group_id === ids[0]))
      ? (state.targets[0].kloudy?.group_name || nextLayerGroupName())
      : nextLayerGroupName();
    const taken = new Set(vinylObjects().map((o) => o.kloudy?.group_name).filter(Boolean));
    const nameA = uniqueGroupName(`${base} A`, taken);
    const nameB = uniqueGroupName(`${base} B`, taken);
    const stamp = `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
    const idA = `group-a-${stamp}`;
    const idB = `group-b-${stamp}`;
    close();
    queueEditorMutation(async () => {
      const clones = [];
      for (const item of plan) {
        const both = item.a && item.b;
        item.obj.kloudy.group_id = item.a ? idA : idB;
        item.obj.kloudy.group_name = item.a ? nameA : nameB;
        if (!both) continue;
        // 사본은 **원본 바로 위**에 꽂는다 — 같은 그림이 이웃해 있으므로
        // 화면은 하나도 안 바뀌고, 묶음만 둘이 된다.
        const shape = objectToShape(item.obj, { includeEditorMeta: true });
        delete shape.editor_id;
        shape.data = Array.isArray(shape.data) ? shape.data.slice() : [];
        shape.editor_group_id = idB;
        shape.editor_group_name = nameB;
        shape.editor_locked = false;
        const clone = await makeFabricObject(shape);
        delete clone.__kloudySelectionOutline;
        applyMaskVisual(clone);
        if (typeof applyObjectHitTestMode === "function") applyObjectHitTestMode(clone);
        clone.hoverCursor = "pointer";
        clone.moveCursor = "move";
        styleObjectTransformControls(clone);
        clones.push({ clone, after: item.obj });
      }
      for (const { clone, after } of clones) {
        canvas.add(clone);
        const at = canvas.getObjects().indexOf(after);
        if (at >= 0) KfpsFabricAdapter.moveObjectTo(canvas, clone, at + 1);
      }
      if (typeof bringGuidesToBack === "function") bringGuidesToBack();
      syncCanvasObjectCoords();
      refreshLayers();
      updateSelectionPanel();
      canvas.requestRenderAll();
      pushHistory("split at line");
      const a = plan.filter((p) => p.a).length;
      const b = plan.filter((p) => p.b).length;
      selectObjects(state.targets.concat(clones.map((c) => c.clone)), "split");
      setStatus(s.done(a, b, dup));
    }).catch((err) => {
      if (typeof showError === "function") showError("Split failed", err);
      setStatus(`Split failed: ${err?.message || err}`);
    });
  }

  // ── 단추 끼우기 ──
  function injectButton() {
    const host = document.querySelector("#ungroupSelected")?.parentElement
      || document.querySelector(".smallTools");
    if (!host || document.getElementById("fsSplitButton")) return;
    const button = document.createElement("button");
    button.id = "fsSplitButton";
    button.type = "button";
    button.setAttribute("data-fs-i18n-skip", "");
    const after = document.getElementById("ungroupSelected");
    if (after && after.nextSibling) host.insertBefore(button, after.nextSibling);
    else host.appendChild(button);
    button.addEventListener("click", () => (state.open ? close() : open()));
  }

  function boot() {
    for (const name of NEED) {
      if (typeof window[name] !== "function") {
        console.warn(`[fs-split] ${name}가 없다 — 가르기를 안 얹는다`);
        return;
      }
    }
    if (typeof canvas === "undefined" || !canvas) return;
    injectButton();
    const s = t();
    const button = document.getElementById("fsSplitButton");
    if (button) {
      button.textContent = s.button;
      button.title = s.buttonTip;
    }
    canvas.on("mouse:down", onDown);
    canvas.on("mouse:move", onMove);
    canvas.on("mouse:up", onUp);
    document.addEventListener("keydown", (e) => {
      if (state.open && e.key === "Escape") {
        e.stopPropagation();
        e.preventDefault();
        close();
      }
    }, true);
    // 언어를 바꾸면 우리 글도 따라 바뀐다 (i18n 오버레이는 우리 UI를 안 만진다)
    document.getElementById("fsLangSelect")?.addEventListener("change", () => {
      const button2 = document.getElementById("fsSplitButton");
      const s2 = t();
      if (button2) {
        button2.textContent = s2.button;
        button2.title = s2.buttonTip;
      }
      retitle();
    });
  }

  // editor.js가 DOMContentLoaded에서 캔버스를 세운다 — 이 파일은 그 뒤에
  // 끼워져 있으므로 같은 이벤트의 **뒤 차례**로 붙으면 캔버스가 이미 서 있다.
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => setTimeout(boot, 0));
  } else {
    setTimeout(boot, 0);
  }
})();
