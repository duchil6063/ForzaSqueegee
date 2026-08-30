/* ForzaSqueegee — 내장 KFPS 편집기 표시 언어 오버레이.
 *
 * vendor/kfps-editor는 무수정 사본이라(README의 SHA-256 대조) 편집기 파일을
 * 고치지 않는다. 대신 서버(kfpseditor.py)가 index.html을 내줄 때 이 파일과
 * 사전(fs-i18n-ko.js)을 끼워 넣고, 여기서 **표시 계층만** 바꾼다:
 *
 * - 정적 DOM은 로드 즉시(이 스크립트가 editor.js보다 먼저 돈다) 번역하고,
 *   editor.js가 그 뒤에 쓰는 텍스트는 MutationObserver로 그때그때 번역한다.
 * - 원문 영어는 노드에 그대로 얹어 둔다(__fsEn/__fsHtmlEn) — 언어를 바꾸면
 *   새로고침 없이 되돌린다 (편집기 beforeunload가 살아 있어 reload는 못 쓴다).
 * - 로직이 읽는 값(data-tool, option value, 가족·도형 이름, 단축키 조합)은
 *   건드리지 않는다. 번역은 렌더된 텍스트 노드·title/placeholder류 속성과
 *   도움말 산문 블록(innerHTML 통치환 — 어순 때문)뿐이다.
 *
 * 언어 선택은 하단 상태줄 오른쪽에 끼운 셀렉트로, 선택은 localStorage에 남는다
 * (서버가 포트를 고정해 origin이 안 흔들린다). 열릴 때의 언어는 서버가 주입한
 * window.FS_EDITOR_LANG(앱의 언어 설정)이 **이긴다** — 앱에서 고른 언어로
 * 편집기가 뜨는 것이 계약이다. 주입이 없을 때만 localStorage → 브라우저 언어
 * → en 순서로 물러난다; 편집기 안에서 바꾼 선택은 그 세션 동안 유지된다.
 */
(function () {
  "use strict";

  const STORE_KEY = "fsEditorLang";
  const DICTS = { ko: window.FS_I18N_KO || null };
  const ATTRS = ["title", "placeholder", "aria-label", "alt"];
  const SKIP_TAGS = new Set(["SCRIPT", "STYLE", "NOSCRIPT", "TITLE"]);
  // 게임 리소스 식별자·사용자 데이터 구역 — 번역 시도는 하되(사전에 없으니
  // 그대로 남는다) 미번역 집계에서는 뺀다. 도형·계열 이름은 검색이 영문
  // 데이터를 상대하므로 일부러 안 옮기는 정책이다.
  const MISS_EXEMPT = "#shapeFamily, #shapeGrid, #overlaySvgLayerSelect, .layerMain, "
    + ".layerGroupTitle, .layerGroupBadge";
  // 정책상 안 옮기는 문장꼴 — 브랜드와 키 조합 나열("Delete / Backspace")
  const MISS_IGNORE = [
    /^KFPS Vinyl Editor/,
    /^[A-Za-z\[\]\d]+(\+[A-Za-z\[\]\d]+)*( \/ [A-Za-z\[\]\d]+(\+[A-Za-z\[\]\d]+)*)+$/,
  ];

  function storedLang() {
    try { return localStorage.getItem(STORE_KEY) || ""; } catch (_e) { return ""; }
  }

  function defaultLang() {
    const injected = String(window.FS_EDITOR_LANG || "");
    if (DICTS[injected] || injected === "en") return injected;
    const nav = String(navigator.language || "");
    if (nav.toLowerCase().startsWith("ko") && DICTS.ko) return "ko";
    return "en";
  }

  let lang = (function () {
    // 서버 주입(앱의 언어 설정)이 이긴다 — 앱에서 고른 언어로 뜨는 계약.
    const injected = String(window.FS_EDITOR_LANG || "");
    if (DICTS[injected] || injected === "en") return injected;
    const s = storedLang();
    return (DICTS[s] || s === "en") ? s : defaultLang();
  })();

  // ── 사전 컴파일 ──
  const compiled = {};       // lang → {exact, html, rules:[{re,tpl,subs}], phrases}
  function dict(l) {
    if (!DICTS[l]) return null;
    if (!compiled[l]) {
      const d = DICTS[l];
      compiled[l] = {
        exact: d.exact || {},
        html: d.html || {},
        phrases: d.phrases || {},
        scoped: d.scoped || {},        // 셀렉터 → {영문: 번역} — 문맥 동음이의 해소
        rules: (d.rules || []).map((r) => {
          try {
            return { re: new RegExp(r[0]), tpl: r[1], subs: r[2] || [] };
          } catch (_e) { return null; }
        }).filter(Boolean),
      };
    }
    return compiled[l];
  }

  // ── 번역기 ──
  const cache = new Map();   // 원문 → 번역 | null (언어 전환 시 비운다)
  const misses = new Set();  // 화면에 나갔는데 못 번역한 문자열 — 검증 도구용

  function phrase(d, text) {
    if (Object.prototype.hasOwnProperty.call(d.phrases, text)) return d.phrases[text];
    if (Object.prototype.hasOwnProperty.call(d.exact, text)) return d.exact[text];
    // humanizeHistoryReason(첫 글자 대문자·[-_]→공백)를 원형으로 되돌려 본다
    const lower = text.charAt(0).toLowerCase() + text.slice(1);
    if (Object.prototype.hasOwnProperty.call(d.phrases, lower)) return d.phrases[lower];
    return null;
  }

  function translateCore(d, text) {
    if (Object.prototype.hasOwnProperty.call(d.exact, text)) return d.exact[text];
    for (const rule of d.rules) {
      const m = rule.re.exec(text);
      if (!m) continue;
      return rule.tpl.replace(/\$(\d)/g, (_all, g) => {
        const v = m[Number(g)] ?? "";
        return rule.subs.includes(Number(g)) ? (phrase(d, v) ?? v) : v;
      });
    }
    return phrase(d, text);
  }

  function translateText(raw, el = null) {
    if (lang === "en" || !raw) return null;
    const d = dict(lang);
    if (!d) return null;
    const text = raw.trim();
    if (!text || !/[A-Za-z]{2}/.test(text)) return null;   // 좌표·수치·기호는 그대로
    if (/^#[0-9A-Fa-f]{3,8}\b/.test(text)) return null;    // 색 헥스 라벨 (#FF00AA / A 255)
    if (el) {
      // 같은 영문이 화면마다 딴 뜻인 곳 — 문맥(조상 셀렉터) 한정 사전이 먼저
      for (const sel in d.scoped) {
        if (Object.prototype.hasOwnProperty.call(d.scoped[sel], text) && el.closest(sel)) {
          return raw.replace(text, d.scoped[sel][text]);
        }
      }
    }
    if (cache.has(text)) {
      const hit = cache.get(text);
      return hit === null ? null : raw.replace(text, hit);
    }
    let out = translateCore(d, text);
    if (out === null && /[.!?] /.test(text)) {
      // 편집기 상태 문구는 독립된 문장을 이어 붙인 꼴이 많다 — 문장 단위로 재시도
      const parts = text.split(/(?<=[.!?])\s+/);
      if (parts.length > 1) {
        const done = parts.map((p) => translateCore(d, p) ?? p);
        if (done.some((p, i) => p !== parts[i])) out = done.join(" ");
      }
    }
    if (cache.size > 4000) cache.clear();
    cache.set(text, out);
    if (out === null) {
      // 집계 조건: 문장꼴(공백 있음) + 한국어 아님 + 식별자/데이터 구역 밖.
      // 낱말 하나(파일·도형·계열 이름)는 데이터라 안 센다 — vendor 문자열
      // 전수는 tools/check_editor_i18n.py가 정적으로 잡는다.
      if (misses.size < 800 && text.includes(" ") && !/[가-힣]/.test(text)
          && !(el && el.closest(MISS_EXEMPT))
          && !MISS_IGNORE.some((rx) => rx.test(text))) {
        misses.add(text);
      }
      return null;
    }
    return raw.replace(text, out);
  }

  // ── DOM 적용 ──
  function excluded(el) {
    if (!el) return false;
    if (SKIP_TAGS.has(el.tagName)) return true;
    return Boolean(el.closest("[data-shortcut-label], [data-fs-i18n-skip]"));
  }

  function applyTextNode(node) {
    const data = node.data;
    if (lang === "en") {                   // en 모드의 적용 = 우리 번역 되돌리기
      if (node.__fsEn !== undefined && data === node.__fsKo) node.data = node.__fsEn;
      return;
    }
    if (node.__fsKo !== undefined && data === node.__fsKo) return;  // 우리가 쓴 값
    const ko = translateText(data, node.parentElement);
    if (ko !== null && ko !== data) {
      node.__fsEn = data;
      node.__fsKo = ko;
      node.data = ko;
    }
  }

  function applyAttrs(el) {
    for (const name of ATTRS) {
      if (!el.hasAttribute(name)) continue;
      const value = el.getAttribute(name);
      const marks = el.__fsAttr || (el.__fsAttr = {});
      if (marks[name] && value === marks[name].ko) continue;
      const ko = translateText(value, el);
      if (ko !== null && ko !== value) {
        marks[name] = { en: value, ko };
        el.setAttribute(name, ko);
      }
    }
  }

  function restoreAttrs(el) {
    const marks = el.__fsAttr;
    if (!marks) return;
    for (const name of ATTRS) {
      if (marks[name] && el.getAttribute(name) === marks[name].ko) {
        el.setAttribute(name, marks[name].en);
      }
    }
  }

  const NORM_RE = /\s+/g;
  function normHtml(s) { return s.replace(NORM_RE, " ").trim(); }

  function applyHtmlBlocks(root) {
    // 산문 블록(인라인 <b> 사이 어순)은 innerHTML 통치환 — 정적 도움말에만
    // 존재하고 폼 컨트롤·리스너가 없는 요소만 사전에 올라 있다
    const d = dict(lang);
    if (!d) return;
    const scope = root.querySelectorAll ? root : document;
    const targets = Array.from(scope.querySelectorAll("p, li, b, h2, h3, h4, span, small"));
    if (scope.matches && scope.matches("p, li, b, h2, h3, h4, span, small")) targets.push(scope);
    for (const el of targets) {
      if (el.childElementCount === 0 || excluded(el)) continue;
      if (el.__fsHtmlKo !== undefined && el.innerHTML === el.__fsHtmlKo) continue;
      const key = normHtml(el.innerHTML);
      const ko = Object.prototype.hasOwnProperty.call(d.html, key) ? d.html[key] : null;
      if (ko !== null) {
        el.__fsHtmlEn = el.innerHTML;
        el.innerHTML = ko;
        el.__fsHtmlKo = el.innerHTML;      // 브라우저 재직렬화 형태로 기억
      }
    }
  }

  function restoreHtmlBlocks() {
    for (const el of document.querySelectorAll("p, li, b, h2, h3, h4, span, small")) {
      if (el.__fsHtmlEn !== undefined && el.innerHTML === el.__fsHtmlKo) {
        el.innerHTML = el.__fsHtmlEn;
      }
    }
  }

  function walk(root) {
    if (root.nodeType === Node.TEXT_NODE) {
      if (!excluded(root.parentElement)) applyTextNode(root);
      return;
    }
    if (root.nodeType !== Node.ELEMENT_NODE || SKIP_TAGS.has(root.tagName)) return;
    if (root.closest("[data-fs-i18n-skip]")) return;
    applyHtmlBlocks(root);
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    let node;
    while ((node = walker.nextNode())) {
      if (!excluded(node.parentElement)) applyTextNode(node);
    }
    if (!excluded(root)) applyAttrs(root);
    for (const el of root.querySelectorAll("[title], [placeholder], [aria-label], [alt]")) {
      if (!excluded(el)) applyAttrs(el);
    }
  }

  function restoreAll() {
    restoreHtmlBlocks();
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    let node;
    while ((node = walker.nextNode())) applyTextNode(node);   // en 모드 = 복원
    for (const el of document.querySelectorAll("*")) restoreAttrs(el);
  }

  // ── 감시 ──
  let observer = null;
  function observe() {
    if (observer) return;
    observer = new MutationObserver((mutations) => {
      if (lang === "en") return;
      for (const m of mutations) {
        if (m.type === "characterData") {
          if (!excluded(m.target.parentElement)) applyTextNode(m.target);
        } else if (m.type === "attributes") {
          if (m.target.nodeType === Node.ELEMENT_NODE && !excluded(m.target)) {
            applyAttrs(m.target);
          }
        } else {
          for (const node of m.addedNodes) walk(node);
        }
      }
    });
    observer.observe(document.documentElement, {
      subtree: true, childList: true, characterData: true,
      attributes: true, attributeFilter: ATTRS,
    });
  }

  // ── 언어 전환 ──
  function setLang(next) {
    if (next === lang) return;
    lang = next;
    try { localStorage.setItem(STORE_KEY, next); } catch (_e) { /* 무시 */ }
    cache.clear();
    if (lang === "en") {
      restoreAll();
    } else {
      applyHtmlBlocks(document);
      walk(document.body);
    }
    // 산문 블록 재구성으로 단축키 라벨 span이 초기 텍스트로 돌아간다 —
    // editor.js의 전역 함수로 현재 바인딩을 다시 입힌다 (로드 전이면 생략)
    try { window.updateShortcutLabels?.(); } catch (_e) { /* 무시 */ }
    updatePickerLabel();
  }

  // ── 언어 픽커 (하단 상태줄, 테마 픽커와 같은 스타일) ──
  let pickerLabelText = null;
  function updatePickerLabel() {
    if (pickerLabelText) pickerLabelText.data = lang === "ko" ? "언어" : "Language";
  }

  function injectPicker() {
    // 하단 상태줄 오른쪽 끝 — 상단 머리는 1440~1600px에서 이미 비좁아
    // (메뉴 nav가 가로 스크롤) 픽커가 스크롤바와 겹친다. 상태줄은 항상
    // 보이고 자리가 넉넉하다. 스타일은 테마 픽커 것을 그대로 입는다.
    const host = document.querySelector(".bottomStatus")
      || document.querySelector(".brandMeta");
    if (!host || document.getElementById("fsLangSelect")) return;
    const label = document.createElement("label");
    label.className = "themePicker";
    label.setAttribute("data-fs-i18n-skip", "");
    pickerLabelText = document.createTextNode("");
    const select = document.createElement("select");
    select.id = "fsLangSelect";
    select.title = "Display language / 표시 언어";
    for (const [value, name] of [["ko", "한국어"], ["en", "English"]]) {
      if (value !== "en" && !DICTS[value]) continue;
      const option = document.createElement("option");
      option.value = value;
      option.textContent = name;
      select.appendChild(option);
    }
    select.value = lang;
    select.addEventListener("change", () => setLang(select.value));
    label.appendChild(pickerLabelText);
    label.appendChild(select);
    host.appendChild(label);
    updatePickerLabel();
  }

  // ── 부트 ──
  // 이 스크립트는 editor-core.js보다 앞에 끼워져 있어 위쪽 DOM은 전부
  // 파싱돼 있고 editor.js는 아직 안 돌았다 — 정적 화면을 먼저 번역해 두면
  // 첫 페인트부터 한국어다.
  injectPicker();
  if (lang !== "en") {
    applyHtmlBlocks(document);
    walk(document.body);
  }
  observe();

  // 검증 도구·콘솔용 손잡이
  window.__fsI18n = {
    get lang() { return lang; },
    set: setLang,
    misses: () => Array.from(misses),
    translate: (s) => translateText(s),
  };
})();
