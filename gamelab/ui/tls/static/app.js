"use strict";
// TLS viewer front end (served by .tools/tls_web.py).  Captured data is only ever put into the page as text nodes.

const $ = (s, r = document) => r.querySelector(s);
function el(tag, attrs, ...kids) {
  const e = document.createElement(tag);
  if (attrs) {
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null || v === false) continue;
      if (k === "class") e.className = v;
      else if (k === "text") e.textContent = v;
      else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
      else e.setAttribute(k, v === true ? "" : v);
    }
  }
  for (const k of kids.flat(Infinity)) {
    if (k == null || k === false) continue;
    e.append(k instanceof Node ? k : document.createTextNode(String(k)));
  }
  return e;
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const pad = (n, w = 2) => String(n).padStart(w, "0");
function fmtTime(ts) {
  if (!ts) return "";
  const d = new Date(ts * 1000);
  return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}.${pad(d.getMilliseconds(), 3)}`;
}
function fmtDate(ts) {
  if (!ts) return "";
  const d = new Date(ts * 1000);
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${fmtTime(ts)}`;
}
function fmtSize(n) {
  if (n == null) return "";
  if (n < 1024) return n + " B";
  if (n < 1048576) return (n / 1024).toFixed(1) + " KB";
  return (n / 1048576).toFixed(2) + " MB";
}
function fmtDur(s) {
  if (s == null || !isFinite(s) || s < 0) return "";
  if (s < 1) return Math.round(s * 1000) + " ms";
  if (s < 60) return s.toFixed(2) + " s";
  if (s < 3600) return Math.floor(s / 60) + "m " + Math.round(s % 60) + "s";
  return Math.floor(s / 3600) + "h " + Math.floor((s % 3600) / 60) + "m";
}
const num = (n) => (n == null ? "" : Number(n).toLocaleString("en-US"));
function shortType(ct) {
  if (!ct) return "";
  let t = ct.split(";")[0].trim().toLowerCase();
  if (t.includes("json")) return "json";
  if (t.startsWith("text/")) return t.slice(5);
  if (t.includes("x-www-form-urlencoded")) return "form";
  if (t.includes("octet-stream")) return "binary";
  if (t.includes("javascript")) return "js";
  return t.split("/").pop();
}
function statusClass(s) { return s ? "st s" + String(s)[0] : "st s0"; }
function lsGet(k) { try { return localStorage.getItem(k); } catch { return null; } }
function lsSet(k, v) { try { localStorage.setItem(k, v); } catch { /* storage unavailable */ } }

// ------------------------------------------------------------------------------------------ state + api

const S = {
  caps: [], cap: "", local: "", data: null, items: [], conns: [], itemById: new Map(), connByN: new Map(),
  view: "items", sel: null, selConn: null, deepIds: null, dtab: "resp", ctab: "overview", visible: [],
  bodyCache: new Map(), connCache: new Map(), loadToken: 0,
  csauth: null, csauthLoading: false, evById: new Map(), selEv: null,
};

async function api(path, params, as = "json") {
  const u = new URL(path, location.href);
  for (const [k, v] of Object.entries(params || {})) if (v !== undefined && v !== null && v !== "") u.searchParams.set(k, v);
  const r = await fetch(u);
  if (!r.ok) {
    let m = "HTTP " + r.status;
    try { m = (await r.json()).error || m; } catch { /* not json */ }
    throw new Error(m);
  }
  if (as === "json") return r.json();
  if (as === "buf") return r.arrayBuffer();
  return r;
}
const capParams = (extra) => Object.assign({ cap: S.cap, local: S.local }, extra || {});
function apiUrl(path, params) {
  const u = new URL(path, location.href);
  for (const [k, v] of Object.entries(params)) if (v !== "" && v != null) u.searchParams.set(k, v);
  return u.pathname + u.search;
}

// ------------------------------------------------------------------------------------------ hash state

function readHash() {
  const h = new URLSearchParams(location.hash.slice(1));
  return { cap: h.get("cap") || "", local: h.get("local") || "", view: h.get("view") || "", sel: h.get("sel") || "",
           conn: h.get("conn") || "", ev: h.get("ev") || "" };
}
function writeHash() {
  const h = new URLSearchParams();
  if (S.cap) h.set("cap", S.cap);
  if (S.local) h.set("local", S.local);
  h.set("view", S.view);
  if (S.view === "items" && S.sel) h.set("sel", S.sel);
  if (S.view === "conns" && S.selConn != null) h.set("conn", S.selConn);
  if (S.view === "csauth" && S.selEv) h.set("ev", S.selEv);
  history.replaceState(null, "", "#" + h.toString());
}

// ------------------------------------------------------------------------------------------ loading

function overlay(msg, err) {
  const o = $("#overlay");
  if (msg == null) { o.classList.add("hidden"); return; }
  o.classList.remove("hidden");
  $(".box", o).classList.toggle("err", !!err);
  $("#ovmsg").textContent = msg + (err ? "  (click to close)" : "");
  o.onclick = err ? () => overlay(null) : null;
}

// A capture still being recorded (or a key log copied in at the end) makes the loaded data stale: offer a reload.
async function checkStale() {
  if (!S.data || document.hidden) return;
  try {
    const st = await api("api/open", capParams());
    const b = $("#reload");
    b.classList.toggle("on", !!st.stale);
    b.title = st.stale ? "The capture files changed since they were decrypted: click to decrypt again" : "Decrypt the capture again";
  } catch { /* server gone */ }
}

async function init() {
  const h = readHash();
  try {
    S.caps = await api("api/captures");
  } catch (e) { overlay("Could not list captures: " + e.message, true); return; }
  const sel = $("#cap");
  sel.replaceChildren();
  for (const c of S.caps) {
    const kl = c.keylogs.length ? c.keylogs.join(" + ") : "no key log";
    sel.append(el("option", { value: c.id }, `${c.name}  (${fmtSize(c.size)}, ${kl})`));
  }
  if (!S.caps.length) { overlay("No *.pcap found under the folders given to tls_web.py.", true); return; }
  const pick = S.caps.find((c) => c.id === h.cap) || S.caps.find((c) => c.keylogs.length) || S.caps[0];
  S.cap = pick.id;
  S.local = h.local;
  S.view = ["conns", "csauth"].includes(h.view) ? h.view : "items";
  sel.value = S.cap;
  $("#frames").value = S.local;
  await openCap(h);
}

async function openCap(h, reload) {
  const token = ++S.loadToken;
  const info = S.caps.find((c) => c.id === S.cap);
  $("#frames").options[0].text = "Frames: auto (" + (info && info.local_default ? "local" : "router rules") + ")";
  overlay("Decrypting " + (info ? info.name : S.cap));
  S.bodyCache.clear(); S.connCache.clear();
  S.csauth = null; S.evById = new Map();
  $("#reload").classList.remove("on");
  try {
    for (let first = true; ; first = false) {
      const st = await api("api/open", capParams({ reload: first && reload ? 1 : "" }));
      if (token !== S.loadToken) return;
      if (st.state === "ready") break;
      if (st.state === "error") { overlay("Decryption failed: " + st.error, true); return; }
      overlay("Decrypting " + (info ? info.name : "") + ": " + st.progress);
      await sleep(250);
    }
    const d = await api("api/summary", capParams());
    if (token !== S.loadToken) return;
    S.data = d;
    S.items = d.items;
    S.conns = d.conns;
    S.itemById = new Map(S.items.map((it) => [it.id, it]));
    S.connByN = new Map(S.conns.map((c) => [c.n, c]));
  } catch (e) { overlay("Error: " + e.message, true); return; }
  overlay(null);
  renderStats();
  const hs = $("#host"), keep = hs.value;
  hs.replaceChildren(el("option", { value: "" }, "All hosts"));
  for (const [host, n] of S.data.summary.hosts) hs.append(el("option", { value: host }, `${host}  (${n})`));
  hs.value = [...hs.options].some((o) => o.value === keep) ? keep : "";
  S.sel = h && h.sel && S.itemById.has(h.sel) ? h.sel : null;
  S.selConn = h && h.conn && S.connByN.has(+h.conn) ? +h.conn : null;
  S.selEv = h && h.ev ? h.ev : null;
  S.deepIds = null;
  if ($("#deep").checked && $("#q").value.trim()) runDeep();
  setView(S.view, true);
}

function renderStats() {
  const s = S.data.summary, box = $("#stats");
  const it = (label, v, bad, title) => el("span", { class: bad ? "bad" : "", title }, el("b", null, v), " " + label);
  const keys = s.keylogs.map((k) => k.split(/[\\/]/).pop()).join(" + ") || "no key log";
  box.replaceChildren(
    it("TLS decrypted", `${s.decrypted}/${s.tls_conns}`, false, `${s.tcp_conns} TCP connections, ${s.non_tls} not TLS`),
    it("HTTP", num(s.http)), it("WS msgs", num(s.ws)),
    it("records", num(s.records), false, `${num(s.app_bytes)} plaintext bytes`),
    it("failed", num(s.failed), s.failed > 0, "records that did not decrypt"),
    it("lost", fmtSize(s.lost), s.lost > 0, "plaintext bytes lost in the capture (router zero-filled frames)"),
    el("span", { title: fmtDate(s.first) + " .. " + fmtDate(s.last) }, fmtTime(s.first).slice(0, 8) + " \u2013 " + fmtTime(s.last).slice(0, 8)),
    el("span", { title: s.sources + "\n" + s.keylogs.join("\n") + `\n${s.key_used} of ${s.key_randoms} key-log client randoms used` }, keys),
  );
}

// ------------------------------------------------------------------------------------------ list views

function setView(v, keepScroll) {
  S.view = v;
  for (const b of document.querySelectorAll(".tabs button")) b.classList.toggle("on", b.dataset.view === v);
  for (const t of document.querySelectorAll(".toggles")) t.classList.toggle("hidden", t.dataset.for !== v);
  $("#host").style.display = $("#deep").closest("label").style.display = v === "csauth" ? "none" : "";
  renderList();
  if (v === "items" && S.sel) showItem(S.sel);
  else if (v === "conns" && S.selConn != null) showConn(S.selConn);
  else if (v === "csauth" && S.selEv && S.evById.has(S.selEv)) showEvent(S.selEv);
  else $("#detail").replaceChildren(el("div", { class: "placeholder" }, { items: "Select a request.", conns: "Select a connection.",
    csauth: "Select a call: each one shows only what changed since the computer's call before it." }[v]));
  scrollSelIntoView();
  writeHash();
}

function terms() {
  return $("#q").value.toLowerCase().split(/\s+/).filter(Boolean);
}
function matchTerms(hay, ts) {
  for (const t of ts) {
    if (t.startsWith("-") && t.length > 1) { if (hay.includes(t.slice(1))) return false; }
    else if (!hay.includes(t)) return false;
  }
  return true;
}

function itemVisible(it, ts, host) {
  if (it.type === "http" && !$("#t-http").checked) return false;
  if (it.type === "ws") {
    if (!$("#t-ws").checked) return false;
    if (["ping", "pong", "close"].includes(it.op) && !$("#t-ctl").checked) return false;
  }
  if (host && it.host !== host) return false;
  if (S.deepIds) return S.deepIds.has(it.id);
  if (!ts.length) return true;
  const hay = [it.host, it.method, it.path, it.status, it.reason, it.preview, "#" + it.conn, it.proto, it.op,
               it.req && it.req.how, it.resp && it.resp.how, it.resp && it.resp.ctype].join(" ").toLowerCase();
  return matchTerms(hay, ts);
}

function connVisible(c, ts, host) {
  if (!c.decrypted && !$("#t-nokey").checked) return false;
  if (host && c.sni !== host) return false;
  if (!ts.length) return true;
  const hay = ["#" + c.n, c.sni, c.server, c.client, c.status, c.proto, c.tls, c.suite, c.alpn, c.key, c.notes.join(" ")]
    .join(" ").toLowerCase();
  return matchTerms(hay, ts);
}

function itemBadges(it) {
  const b = [];
  const add = (t, cls, title) => b.push(el("span", { class: "chip " + (cls || ""), title }, t));
  if (it.type === "ws") {
    const m = it.msg;
    if (m.how) add("deflate", "", m.how);
    if (m.lost) add("lost " + fmtSize(m.lost), "err", "bytes lost in the capture");
    if (!m.complete) add("partial", "warn");
    return b;
  }
  if (it.proto === "h2") add("h2", "info");
  const rq = it.req, rs = it.resp;
  if (rq.how) add(rq.how.startsWith("NGS AE:G") ? "AE:G" : rq.how.split(/[ (,]/)[0], "purple", "request body: " + rq.how);
  if (rs && rs.how) add(rs.how.startsWith("G:") ? "CE:G" : rs.how.split(/[ (,]/)[0], rs.how.includes("fail") || rs.how.includes("not decoded") ? "warn" : "purple", "response body: " + rs.how);
  if (rq.lost || (rs && rs.lost)) add("lost " + fmtSize((rq.lost || 0) + ((rs && rs.lost) || 0)), "err", "bytes lost in the capture");
  if (!rq.complete || (rs && !rs.complete)) add("partial", "warn", "message incomplete in the capture");
  if (!rs) add("no response", "warn");
  if (it.reason && it.reason.startsWith("RST_STREAM")) add("RST", "warn", it.reason);
  return b;
}

function renderList() {
  const box = $("#list");
  const ts = terms(), host = $("#host").value;
  if (!S.data) { box.replaceChildren(); return; }
  if (S.view === "csauth") { renderCsauth(box, ts); return; }
  if (S.view === "items") {
    const vis = S.items.filter((it) => itemVisible(it, ts, host));
    S.visible = vis.map((it) => it.id);
    $("#count").textContent = `${num(vis.length)} of ${num(S.items.length)}`;
    const t0 = S.data.summary.first;
    const tb = el("tbody");
    const frag = document.createDocumentFragment();
    for (const it of vis) {
      const http = it.type === "http";
      const dur = http && it.resp ? it.te - it.ts : null;
      const pathText = http ? it.path : (it.preview != null ? it.preview : `(${it.op}, ${fmtSize(it.msg.size)})`);
      const tr = el("tr", { "data-id": it.id, class: it.id === S.sel ? "sel" : "" },
        el("td", { class: "t", title: fmtDate(it.ts) + "  (+" + fmtDur(it.ts - t0) + ")" }, fmtTime(it.ts)),
        el("td", { class: "num muted" }, "#" + it.conn),
        el("td", { title: it.host }, it.host),
        el("td", null, el("span", { class: "m" + (http ? "" : " ws") }, it.method)),
        el("td", { class: "path", title: http ? it.path : pathText }, pathText),
        el("td", null, http ? el("span", { class: statusClass(it.status), title: it.reason || "" }, it.status ?? "\u2014") : el("span", { class: "muted" }, it.op)),
        el("td", { class: "muted", title: http && it.resp ? it.resp.ctype : "" }, http ? (it.resp && it.resp.display === "json" ? "json" : shortType(it.resp && it.resp.ctype)) : it.msg.display),
        el("td", { class: "num" }, http ? (it.req.size ? fmtSize(it.req.size) : "") : (it.dir === "c2s" ? fmtSize(it.msg.size) : "")),
        el("td", { class: "num" }, http ? (it.resp ? fmtSize(it.resp.size) : "") : (it.dir === "s2c" ? fmtSize(it.msg.size) : "")),
        el("td", { class: "num muted" }, dur != null ? fmtDur(dur) : ""),
        el("td", null, itemBadges(it)));
      frag.append(tr);
    }
    tb.append(frag);
    const table = el("table", { class: "grid wide" },
      el("colgroup", null, [104, 52, 190, 64, null, 54, 64, 72, 72, 64, 150].map((w) => el("col", w ? { style: `width:${w}px` } : null))),
      el("thead", null, el("tr", null, ["Time", "Conn", "Host", "Method", "Path / payload", "Status", "Type", "Sent", "Received", "Time", "Notes"]
        .map((h, i) => el("th", { class: [1, 7, 8, 9].includes(i) ? "num" : "" }, h)))),
      tb);
    box.replaceChildren(vis.length ? table : el("div", { class: "empty" }, S.items.length ? "Nothing matches the filter." : "No decrypted HTTP traffic in this capture."));
  } else {
    const vis = S.conns.filter((c) => connVisible(c, ts, host));
    S.visible = vis.map((c) => c.n);
    $("#count").textContent = `${num(vis.length)} of ${num(S.conns.length)} TLS connections`;
    const frag = document.createDocumentFragment();
    for (const c of vis) {
      const cls = c.status === "decrypted" ? "ok" : c.status === "damaged" ? "err" : "";
      frag.append(el("tr", { "data-n": c.n, class: (c.n === S.selConn ? "sel " : "") + (c.decrypted ? "" : "dim") },
        el("td", { class: "num muted" }, "#" + c.n),
        el("td", { class: "t", title: fmtDate(c.start) }, fmtTime(c.start)),
        el("td", { class: "num muted" }, fmtDur(c.end - c.start)),
        el("td", { title: c.sni }, c.sni || el("span", { class: "muted" }, "(no SNI)")),
        el("td", { class: "path", title: c.client + " \u2192 " + c.server }, c.server),
        el("td", null, c.tls ? "TLS " + c.tls : ""),
        el("td", { title: c.alpn_offered.join(", ") }, c.proto || c.alpn),
        el("td", null, el("span", { class: "chip " + cls, title: c.notes.join("\n") }, c.status)),
        el("td", { class: "num" }, c.decrypted ? `${c.records[0]}/${c.records[1]}` : ""),
        el("td", { class: "num" }, c.decrypted ? `${fmtSize(c.app[0])} / ${fmtSize(c.app[1])}` : ""),
        el("td", { class: "num" }, c.items || ""),
        el("td", { class: "muted", title: c.notes.join("\n") }, c.notes.join("; "))));
    }
    const table = el("table", { class: "grid wide" },
      el("colgroup", null, [52, 104, 64, 210, 160, 64, 140, 92, 70, 140, 52, null].map((w) => el("col", w ? { style: `width:${w}px` } : null))),
      el("thead", null, el("tr", null, ["#", "Start", "Length", "SNI", "Server", "TLS", "Protocol", "Status", "Records \u2191/\u2193", "Bytes \u2191/\u2193", "Items", "Notes"]
        .map((h, i) => el("th", { class: [0, 2, 8, 9, 10].includes(i) ? "num" : "" }, h)))),
      el("tbody", null, frag));
    box.replaceChildren(vis.length ? table : el("div", { class: "empty" }, "Nothing matches the filter."));
  }
}

function markSel() {
  for (const tr of $("#list").querySelectorAll("tr.sel")) tr.classList.remove("sel");
  const sel = S.view === "items" ? (S.sel && $(`#list tr[data-id="${CSS.escape(S.sel)}"]`))
    : S.view === "csauth" ? (S.selEv && $(`#list tr[data-ev="${CSS.escape(S.selEv)}"]`))
    : (S.selConn != null && $(`#list tr[data-n="${S.selConn}"]`));
  if (sel) sel.classList.add("sel");
  return sel;
}
function scrollSelIntoView() {
  const tr = markSel();
  if (tr) tr.scrollIntoView({ block: "nearest" });
}

function selectItem(id) {
  S.sel = id;
  markSel();
  writeHash();
  showItem(id);
}
function selectEv(id) {
  S.selEv = id;
  markSel();
  writeHash();
  showEvent(id);
}
function selectConn(n) {
  S.selConn = n;
  markSel();
  writeHash();
  showConn(n);
}

// ------------------------------------------------------------------------------------------ item detail

async function getBody(id, part, form) {
  const key = `${S.cap}|${S.local}|${id}|${part}|${form}`;
  if (S.bodyCache.has(key)) return S.bodyCache.get(key);
  const buf = await api("api/body", capParams({ id, part, form }), "buf");
  S.bodyCache.set(key, buf);
  if (S.bodyCache.size > 60) S.bodyCache.delete(S.bodyCache.keys().next().value);
  return buf;
}
async function getConn(n) {
  if (S.connCache.has(n)) return S.connCache.get(n);
  const d = await api("api/conn", capParams({ n }));
  S.connCache.set(n, d);
  return d;
}

let detailToken = 0;
async function showItem(id) {
  const token = ++detailToken;
  const box = $("#detail");
  let d;
  try { d = await api("api/item", capParams({ id })); } catch (e) { box.replaceChildren(el("div", { class: "placeholder" }, e.message)); return; }
  if (token !== detailToken) return;
  const conn = S.connByN.get(d.conn) || {};
  const http = d.type === "http";
  const scheme = http ? "https://" : "wss://";
  const head = el("div", { class: "dhead" },
    el("div", { class: "dtitle" },
      el("span", { class: "m" + (http ? "" : " ws") }, d.method),
      el("span", { class: "url" }, scheme + d.host + (d.path || "")),
      http ? el("span", { class: statusClass(d.status) }, d.status != null ? `${d.status} ${d.reason || ""}` : "no response")
           : el("span", { class: "chip purple" }, d.op)),
    el("div", { class: "dmeta" },
      el("span", { title: fmtDate(d.ts) }, fmtDate(d.ts)),
      http && d.resp ? el("span", { title: "first request byte to last response byte" }, fmtDur(d.te - d.ts)) : null,
      el("span", null, d.proto + (d.sid != null ? " stream " + d.sid : "")),
      el("a", { onclick: () => { S.selConn = d.conn; setView("conns"); } , title: "Show the connection" }, `connection #${d.conn}`),
      el("span", null, `TLS ${conn.tls || "?"} ${conn.suite || ""}`),
      el("span", { class: "mono" }, conn.client + " \u2192 " + conn.server)));
  const tabs = http ? [["req", "Request"], ["resp", "Response"], ["conn", "Connection"]] : [["msg", "Message"], ["conn", "Connection"]];
  const cur = tabs.some(([k]) => k === S.dtab) ? S.dtab : (http ? "resp" : "msg");
  const tabBar = el("div", { class: "dtabs" });
  const body = el("div");
  const draw = async (k, user) => {
    if (user) S.dtab = k === "msg" ? "resp" : k;            // remember the user's choice for the next item
    for (const b of tabBar.children) b.classList.toggle("on", b.dataset.k === k);
    if (k === "conn") body.replaceChildren(await connPane(d.conn, token, true));
    else if (k === "msg") body.replaceChildren(partPane(d, "msg", d.msg));
    else body.replaceChildren(partPane(d, k, d[k]));
  };
  for (const [k, label] of tabs) {
    let extra = "";
    if (k === "req" && d.req && d.req.size) extra = " \u00b7 " + fmtSize(d.req.size);
    if (k === "resp" && d.resp && d.resp.size) extra = " \u00b7 " + fmtSize(d.resp.size);
    tabBar.append(el("button", { type: "button", "data-k": k, onclick: () => draw(k, true) }, label + extra));
  }
  box.replaceChildren(head, tabBar, body);
  box.scrollTop = 0;
  await draw(cur);
}

function partPane(item, part, p) {
  const pane = el("div", { class: "pane" });
  if (!p) {
    pane.append(el("div", { class: "banner warn" }, part === "resp" ? "No response in the capture." : "No request in the capture."));
    return pane;
  }
  pane.append(el("div", { class: "startline", title: fmtDate(p.ts) }, p.start));
  if (p.lost) pane.append(el("div", { class: "banner err" }, `${num(p.lost)} bytes of this message were lost in the capture (router zero-filled frames); they show as NUL / "[N bytes lost]".`));
  if (!p.complete) pane.append(el("div", { class: "banner warn" }, "This message is incomplete in the capture."));
  if (p.headers.length) {
    const t = el("table", { class: "kv" });
    for (const [n, v] of p.headers) t.append(el("tr", null, el("td", { class: "k" }, n), el("td", { class: "v" }, v)));
    pane.append(el("details", { class: "sect", open: true }, el("summary", null, `Headers (${p.headers.length})`), t));
  }
  pane.append(bodySection(item, part, p));
  return pane;
}

function bodyModes(p) {
  switch (p.display) {
    case "json": return ["tree", "pretty", "text", "hex"];
    case "text": return ["text", "hex"];
    case "image": return ["image", "hex"];
    case "empty": return [];
    default: return ["hex", "text"];
  }
}

function bodySection(item, part, p) {
  const sect = el("div", { class: "sect" });
  const modes = bodyModes(p);
  const wireDiffers = p.how && p.wire;
  if (!modes.length && !p.wire) {
    sect.append(el("div", { class: "sect-h" }, "Body"), el("div", { class: "muted" }, "(empty)"));
    return sect;
  }
  const key = "bodymode:" + p.display;
  let mode = lsGet(key);
  if (!modes.includes(mode)) mode = modes[0] || "wire";
  const btns = el("div", { class: "btns" });
  const out = el("div", { class: "bodybox" });
  const all = wireDiffers ? [...modes, "wire"] : modes;
  const labels = { tree: "Tree", pretty: "Pretty", text: "Text", hex: "Hex", image: "Image", wire: "Wire" };
  const ext = p.display === "json" ? "json" : p.display === "text" ? "txt" : p.display === "image" ? (shortType(p.ctype) || "img") : "bin";
  const dl = el("a", { class: "tool", href: apiUrl("api/body", capParams({ id: item.id, part, form: "decoded", dl: 1, ext })), title: "Save the (decoded) body" }, "save");
  const copy = el("a", { class: "tool", title: "Copy the body as text" }, "copy");
  const render = async (m) => {
    mode = m;
    if (m !== "wire") lsSet(key, m);
    for (const b of btns.children) b.classList.toggle("on", b.dataset.m === m);
    out.replaceChildren(el("span", { class: "muted" }, "loading\u2026"));
    try {
      const buf = await getBody(item.id, part, m === "wire" ? "wire" : "decoded");
      out.replaceChildren(renderBytes(new Uint8Array(buf), m === "wire" ? (isTextual(new Uint8Array(buf)) ? "text" : "hex") : m, p.ctype));
    } catch (e) { out.replaceChildren(el("span", { class: "lost" }, e.message)); }
  };
  copy.addEventListener("click", async () => {
    try {
      const buf = await getBody(item.id, part, mode === "wire" ? "wire" : "decoded");
      await navigator.clipboard.writeText(new TextDecoder().decode(buf));
      copy.textContent = "copied";
      setTimeout(() => (copy.textContent = "copy"), 1200);
    } catch (e) { copy.textContent = "copy failed"; }
  });
  for (const m of all) btns.append(el("button", { type: "button", "data-m": m, onclick: () => render(m), title: m === "wire" ? "as sent on the wire (before the decoding below)" : null }, labels[m]));
  const info = [fmtSize(p.size)];
  if (p.how) info.push(`wire ${fmtSize(p.wire)}, ${p.how}`);
  if (p.ctype) info.push(p.ctype);
  sect.append(el("div", { class: "sect-h" }, "Body", btns, el("span", { class: "tool" }, info.join(" \u00b7 ")), dl, copy), out);
  render(mode);
  return sect;
}

function isTextual(bytes) {
  const n = Math.min(bytes.length, 4096);
  let bad = 0;
  for (let i = 0; i < n; i++) {
    const b = bytes[i];
    if (b < 9 || (b > 13 && b < 32 && b !== 27 && b !== 30)) bad++;
  }
  if (bad > n * 0.03) return false;
  try { new TextDecoder("utf-8", { fatal: true }).decode(bytes.subarray(0, n > 4 ? n - 4 : n)); return true; } catch { return false; }
}

function renderBytes(bytes, mode, ctype) {
  if (mode === "hex") return hexView(bytes);
  if (mode === "image") {
    const url = URL.createObjectURL(new Blob([bytes], { type: (ctype || "image/png").split(";")[0] }));
    return el("img", { class: "preview", src: url, alt: "image body" });
  }
  const text = new TextDecoder("utf-8").decode(bytes);
  if (mode === "tree" || mode === "pretty") {
    let v;
    try { v = JSON.parse(text.trim().replace(/\x1e+$/, "")); } catch (e) { return textView(text); }
    if (mode === "pretty") return el("pre", { class: "code" }, highlightJson(JSON.stringify(v, null, 2)));
    return jsonTree(v);
  }
  return textView(text);
}

function textView(text) {
  const pre = el("pre", { class: "code" });
  const re = /\u0000{8,}/g;
  let last = 0, m;
  const LIMIT = 2_000_000;
  const t = text.length > LIMIT ? text.slice(0, LIMIT) : text;
  while ((m = re.exec(t))) {
    pre.append(t.slice(last, m.index), el("span", { class: "lost" }, `[${m[0].length} bytes lost in the capture]`));
    last = m.index + m[0].length;
  }
  pre.append(t.slice(last));
  if (t !== text) pre.append(el("span", { class: "muted" }, `\n\u2026 (${num(text.length)} characters, first ${num(LIMIT)} shown; save the body for all)`));
  return pre;
}

function hexView(bytes) {
  const wrap = el("div");
  const pre = el("pre", { class: "hex" });
  const CH = 64 * 1024;
  let shown = 0;
  const more = el("button", { type: "button", class: "more" });
  const step = () => {
    const end = Math.min(bytes.length, shown + CH);
    const lines = [];
    for (let o = shown; o < end; o += 16) {
      let hx = "", as = "";
      for (let i = 0; i < 16; i++) {
        if (o + i < bytes.length) {
          const b = bytes[o + i];
          hx += (b < 16 ? "0" : "") + b.toString(16) + " ";
          as += b >= 32 && b < 127 ? String.fromCharCode(b) : ".";
        } else hx += "   ";
        if (i === 7) hx += " ";
      }
      lines.push(o.toString(16).padStart(8, "0") + "  " + hx + " " + as);
    }
    pre.append(lines.join("\n") + (end < bytes.length ? "\n" : ""));
    shown = end;
    more.hidden = shown >= bytes.length;
    more.textContent = `Show more (${fmtSize(shown)} of ${fmtSize(bytes.length)})`;
  };
  more.addEventListener("click", step);
  if (!bytes.length) pre.append("(empty)");
  else step();
  wrap.append(pre, more);
  return wrap;
}

function highlightJson(s) {
  const frag = document.createDocumentFragment();
  const re = /("(?:\\.|[^"\\])*")(\s*:)?|\b(true|false)\b|\bnull\b|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?/g;
  let last = 0, m;
  while ((m = re.exec(s))) {
    if (m.index > last) frag.append(s.slice(last, m.index));
    let cls = "j-n";
    if (m[1]) cls = m[2] ? "j-k" : "j-s";
    else if (m[3]) cls = "j-b";
    else if (m[0] === "null") cls = "j-null";
    if (m[1] && m[2]) { frag.append(el("span", { class: cls }, m[1]), m[2]); }
    else frag.append(el("span", { class: cls }, m[0]));
    last = m.index + m[0].length;
  }
  frag.append(s.slice(last));
  return frag;
}

// JSON tree: <details> per object / array, children built when first opened.
let treeBudget = 0;
function jsonTree(v) {
  const root = el("div", { class: "jt" });
  const tools = el("div", { class: "tool", style: "margin-bottom:4px" },
    el("a", { class: "tool", onclick: () => { treeBudget = 30000; for (const d of root.querySelectorAll(":scope > details")) expandAll(d); } }, "expand all"), "  ",
    el("a", { class: "tool", onclick: () => { for (const d of root.querySelectorAll("details")) d.open = false; } }, "collapse all"));
  root.append(tools, jnode(null, v, 0));
  return root;
}
function jval(v) {
  if (v === null) return el("span", { class: "j-null" }, "null");
  if (typeof v === "string") return el("span", { class: "j-s" }, JSON.stringify(v));
  if (typeof v === "number") return el("span", { class: "j-n" }, String(v));
  if (typeof v === "boolean") return el("span", { class: "j-b" }, String(v));
  return el("span", null, String(v));
}
function jkey(key) {
  if (key === null) return null;
  return [el("span", { class: "j-k" }, typeof key === "number" ? String(key) : JSON.stringify(key)), ": "];
}
function jpreview(v) {
  const isArr = Array.isArray(v);
  const parts = [];
  let len = 0;
  for (const [k, x] of Object.entries(v)) {
    const s = (isArr ? "" : k + ": ") + (x && typeof x === "object" ? (Array.isArray(x) ? "[\u2026]" : "{\u2026}") : JSON.stringify(x));
    parts.push(s);
    len += s.length;
    if (len > 80) { parts.push("\u2026"); break; }
  }
  return isArr ? "[" + parts.join(", ") + "]" : "{" + parts.join(", ") + "}";
}
function jnode(key, v, depth) {
  if (!v || typeof v !== "object") return el("div", { class: "leaf" }, jkey(key), jval(v));
  const isArr = Array.isArray(v);
  const n = isArr ? v.length : Object.keys(v).length;
  if (!n) return el("div", { class: "leaf" }, jkey(key), el("span", { class: "muted" }, isArr ? "[]" : "{}"));
  const det = el("details", null, el("summary", null, jkey(key), el("span", { class: "muted" }, isArr ? `Array(${n})` : `{${n}}`), " ", el("span", { class: "pv" }, jpreview(v))));
  det._v = v; det._depth = depth;
  det.addEventListener("toggle", () => { if (det.open) fillNode(det); });
  if (depth < 2 && n <= 400) { det.open = true; fillNode(det); }
  return det;
}
function fillNode(det) {
  if (det._filled) return;
  det._filled = true;
  const v = det._v, kids = el("div", { class: "kids" });
  const entries = Array.isArray(v) ? v.map((x, i) => [i, x]) : Object.entries(v);
  const PAGE = 300;
  let shown = 0;
  const more = el("a", { class: "tool" });
  const step = () => {
    const end = Math.min(entries.length, shown + PAGE);
    for (let i = shown; i < end; i++) kids.insertBefore(jnode(entries[i][0], entries[i][1], det._depth + 1), more);
    shown = end;
    if (shown >= entries.length) more.remove(); else more.textContent = `show ${Math.min(PAGE, entries.length - shown)} more of ${entries.length - shown}`;
  };
  kids.append(more);
  more.addEventListener("click", step);
  step();
  det.append(kids);
}
function expandAll(det) {
  if (treeBudget-- <= 0) return;
  det.open = true;
  fillNode(det);
  for (const c of det.querySelectorAll(":scope > .kids > details")) expandAll(c);
}

// ------------------------------------------------------------------------------------------ connection detail

async function showConn(n) {
  const token = ++detailToken;
  const box = $("#detail");
  const c = S.connByN.get(n);
  if (!c) { box.replaceChildren(el("div", { class: "placeholder" }, "No such connection.")); return; }
  const head = el("div", { class: "dhead" },
    el("div", { class: "dtitle" }, el("b", null, `#${c.n}`), el("span", { class: "url" }, c.sni || c.server),
      el("span", { class: "chip " + (c.status === "decrypted" ? "ok" : c.status === "damaged" ? "err" : "") }, c.status)),
    el("div", { class: "dmeta" }, el("span", null, fmtDate(c.start)), el("span", null, fmtDur(c.end - c.start)),
      el("span", { class: "mono" }, c.client + " \u2192 " + c.server), c.tls ? el("span", null, `TLS ${c.tls} ${c.suite}`) : null));
  box.replaceChildren(head, el("div", { class: "pane muted" }, "loading\u2026"));
  const pane = await connPane(n, token, false);
  if (token !== detailToken) return;
  box.replaceChildren(head, pane);
}

async function connPane(n, token, inItem) {
  let d;
  try { d = await getConn(n); } catch (e) { return el("div", { class: "pane" }, el("div", { class: "banner err" }, e.message)); }
  const c = d.conn;
  const wrap = el("div");
  const tabs = [["overview", "Overview"]];
  if (c.items) tabs.push(["items", `Requests (${c.items})`]);
  if (d.records.length) tabs.push(["records", `TLS records (${d.records.length})`]);
  if (d.frames.length) tabs.push(["frames", `HTTP/2 frames (${d.frames.length})`]);
  if (c.decrypted) tabs.push(["streams", "Streams"]);
  let cur = tabs.some(([k]) => k === S.ctab) ? S.ctab : "overview";
  const bar = el("div", { class: "dtabs", style: inItem ? "top:37px" : null });
  const body = el("div", { class: "pane" });
  const draw = (k) => {
    cur = k; S.ctab = k;
    for (const b of bar.children) b.classList.toggle("on", b.dataset.k === k);
    body.replaceChildren(connSection(d, k));
  };
  for (const [k, label] of tabs) bar.append(el("button", { type: "button", "data-k": k, onclick: () => draw(k) }, label));
  wrap.append(bar, body);
  draw(cur);
  return wrap;
}

function kvTable(rows, plain) {
  const t = el("table", { class: "kv" + (plain ? " plain" : "") });
  for (const [k, v] of rows) if (v != null && v !== "" && !(Array.isArray(v) && !v.length)) t.append(el("tr", null, el("td", { class: "k" }, k), el("td", { class: "v" }, v)));
  return t;
}

function connSection(d, k) {
  const c = d.conn;
  if (k === "overview") {
    const frag = el("div");
    if (c.notes.length) frag.append(el("div", { class: "banner " + (c.decrypted ? "warn" : "info") }, c.notes.join("; ")));
    frag.append(el("div", { class: "sect" }, kvTable([
      ["connection", `#${c.n}`], ["time", `${fmtDate(c.start)} \u2013 ${fmtTime(c.end)}  (${fmtDur(c.end - c.start)})`],
      ["client", c.client], ["server", c.server], ["SNI", c.sni], ["TLS", c.tls && `${c.tls}  ${c.suite}`],
      ["ALPN offered", c.alpn_offered.join(", ")], ["ALPN chosen", c.alpn], ["key", c.key], ["status", c.status],
      ["protocol", c.proto], ["records c2s / s2c", c.decrypted ? `${c.records[0]} / ${c.records[1]}` : ""],
      ["failed", c.failed || ""], ["unauthenticated", c.unauth ? `${c.unauth} (records with lost bytes, decrypted without the tag check)` : ""],
      ["plaintext c2s / s2c", c.decrypted ? `${num(c.app[0])} / ${num(c.app[1])} bytes` : ""],
      ["lost c2s / s2c", c.lost[0] || c.lost[1] ? `${num(c.lost[0])} / ${num(c.lost[1])} bytes` : ""],
      ["alerts", c.alerts.join(", ")], ["client random", d.ch.random],
    ])));
    if (c.certs.length) {
      const cs = el("div", { class: "sect" }, el("div", { class: "sect-h" }, `Server certificates (${c.certs.length})`));
      for (const x of c.certs) {
        cs.append(el("div", { class: "cert" },
          el("div", { class: "subj" }, x.subject),
          el("div", { class: "muted" }, "issuer ", x.issuer),
          x.not_before ? el("div", { class: "muted" }, `valid ${x.not_before} \u2013 ${x.not_after}  \u00b7  serial ${x.serial}`) : null,
          x.san && x.san.length ? el("div", { class: "muted" }, "SAN " + x.san.join(", ")) : null,
          x.sha256 ? el("div", { class: "muted mono", style: "font-size:11px" }, "sha256 " + x.sha256) : null));
      }
      frag.append(cs);
    }
    return frag;
  }
  if (k === "items") {
    const t0 = c.start;
    const tb = el("tbody");
    for (const it of d.items) {
      const http = it.type === "http";
      tb.append(el("tr", { onclick: () => gotoItem(it.id) },
        el("td", { class: "t", title: fmtDate(it.ts) }, "+" + fmtDur(it.ts - t0)),
        el("td", null, el("span", { class: "m" + (http ? "" : " ws") }, it.method)),
        el("td", { class: "path", title: it.path }, http ? it.path : (it.preview != null ? it.preview : it.op)),
        el("td", null, http ? el("span", { class: statusClass(it.status) }, it.status ?? "\u2014") : el("span", { class: "muted" }, it.op)),
        el("td", { class: "num" }, http ? (it.resp ? fmtSize(it.resp.size) : "") : fmtSize(it.msg.size))));
    }
    return el("div", { class: "subgrid" }, el("table", { class: "grid" },
      el("colgroup", null, [90, 64, null, 64, 80].map((w) => el("col", w ? { style: `width:${w}px` } : null))),
      el("thead", null, el("tr", null, ["Time", "Method", "Path / payload", "Status", "Size"].map((h, i) => el("th", { class: i === 4 ? "num" : "" }, h)))), tb));
  }
  if (k === "records") {
    const tb = el("tbody");
    for (const r of d.records) {
      const st = r.status === "ok" ? "ok" : r.status === "plain" ? "" : r.status === "unauth" ? "warn" : "err";
      const tr = el("tr", { title: r.has_pt ? "Click for the plaintext" : "" },
        el("td", { class: "t", title: fmtDate(r.ts) }, fmtTime(r.ts)),
        el("td", null, r.dir === "c2s" ? "\u2191 c2s" : "\u2193 s2c"),
        el("td", null, r.ct),
        el("td", { class: "num" }, num(r.len)),
        el("td", null, r.inner || ""),
        el("td", { class: "num" }, r.ptlen != null ? num(r.ptlen) : ""),
        el("td", null, el("span", { class: "chip " + st, title: r.lost ? `${r.lost} plaintext bytes lost` : "" }, r.status + (r.lost ? ` (${r.lost} lost)` : ""))),
        el("td", { class: "muted", title: r.info }, r.info + (r.app_off != null ? `app data @${num(r.app_off)}` : "")));
      if (r.has_pt) {
        tr.addEventListener("click", async () => {
          const nx = tr.nextElementSibling;
          if (nx && nx.classList.contains("exp")) { nx.remove(); return; }
          const cell = el("td", { colspan: 8 }, el("span", { class: "muted" }, "loading\u2026"));
          tr.after(el("tr", { class: "exp" }, cell));
          try {
            const buf = new Uint8Array(await api("api/record", capParams({ n: c.n, dir: r.dir, i: r.i }), "buf"));
            cell.replaceChildren(el("div", { class: "bodybox" }, isTextual(buf) && r.inner === "ApplicationData" ? textView(new TextDecoder().decode(buf)) : hexView(buf)));
          } catch (e) { cell.replaceChildren(el("span", { class: "lost" }, e.message)); }
        });
      }
      tb.append(tr);
    }
    return el("div", { class: "subgrid" }, el("table", { class: "grid" },
      el("colgroup", null, [104, 64, 128, 64, 128, 64, 92, null].map((w) => el("col", w ? { style: `width:${w}px` } : null))),
      el("thead", null, el("tr", null, ["Time", "Dir", "Record", "Length", "Inner type", "Plain", "Status", "Info"].map((h, i) => el("th", { class: [3, 5].includes(i) ? "num" : "" }, h)))), tb));
  }
  if (k === "frames") {
    const tb = el("tbody");
    for (const f of d.frames) {
      tb.append(el("tr", null,
        el("td", { class: "t", title: fmtDate(f.ts) }, fmtTime(f.ts)),
        el("td", null, f.dir === "c2s" ? "\u2191 c2s" : "\u2193 s2c"),
        el("td", null, f.type),
        el("td", { class: "num" }, f.sid),
        el("td", { class: "mono" }, "0x" + f.flags.toString(16).padStart(2, "0")),
        el("td", { class: "num" }, num(f.len)),
        el("td", { title: f.info }, f.info, f.lost ? el("span", { class: "lost" }, ` ${f.lost} lost`) : null)));
    }
    return el("div", { class: "subgrid" }, el("table", { class: "grid" },
      el("colgroup", null, [104, 64, 120, 60, 56, 64, null].map((w) => el("col", w ? { style: `width:${w}px` } : null))),
      el("thead", null, el("tr", null, ["Time", "Dir", "Type", "Stream", "Flags", "Length", "Info"].map((h, i) => el("th", { class: [3, 5].includes(i) ? "num" : "" }, h)))), tb));
  }
  if (k === "streams") {
    const frag = el("div");
    for (const [dir, label, size] of [["c2s", "client \u2192 server", c.app[0]], ["s2c", "server \u2192 client", c.app[1]]]) {
      const out = el("div", { class: "bodybox" }, el("span", { class: "muted" }, "\u2026"));
      const btns = el("div", { class: "btns" });
      let mode = lsGet("streammode") || "text";
      const LIM = 1 << 20;
      const render = async (m) => {
        mode = m; lsSet("streammode", m);
        for (const b of btns.children) b.classList.toggle("on", b.dataset.m === m);
        try {
          const buf = new Uint8Array(await api("api/stream", capParams({ n: c.n, dir, off: 0, len: LIM }), "buf"));
          const view = m === "hex" ? hexView(buf) : textView(new TextDecoder("utf-8").decode(buf));
          out.replaceChildren(view, size > LIM ? el("div", { class: "muted" }, `first ${fmtSize(LIM)} of ${fmtSize(size)} shown`) : null);
        } catch (e) { out.replaceChildren(el("span", { class: "lost" }, e.message)); }
      };
      for (const m of ["text", "hex"]) btns.append(el("button", { type: "button", "data-m": m, onclick: () => render(m) }, m === "text" ? "Text" : "Hex"));
      frag.append(el("div", { class: "sect" }, el("div", { class: "sect-h" }, label, btns, el("span", { class: "tool" }, `${num(size)} bytes decrypted application data`)), out));
      if (size) render(mode); else out.replaceChildren(el("span", { class: "muted" }, "(empty)"));
    }
    return frag;
  }
  return el("div");
}

function gotoItem(id) {
  const it = S.itemById.get(id);
  if (!it) return;
  S.sel = id;
  S.deepIds = null;
  if (!itemVisible(it, terms(), $("#host").value)) {
    $("#q").value = ""; $("#host").value = "";
    if (it.type === "ws") { $("#t-ws").checked = true; if (["ping", "pong", "close"].includes(it.op)) $("#t-ctl").checked = true; }
    else $("#t-http").checked = true;
  }
  setView("items");
}

// ------------------------------------------------------------------------------------------ x-csauth timeline
// One section per computer (the NGS hwid), its calls in time order, each showing only what changed (ui/tls/csauth.py
// says what counts): runs of heartbeats with nothing new folded into one row.

async function loadCsauth() {
  if (S.csauthLoading) return;
  S.csauthLoading = true;
  const token = S.loadToken;
  try {
    const d = await api("api/csauth", capParams());
    if (token !== S.loadToken) return;
    S.csauth = d;
    S.evById = new Map();
    for (const c of d.computers) for (const ev of c.events) { ev.computer = c; S.evById.set(ev.id, ev); }
  } catch (e) {
    if (token !== S.loadToken) return;
    S.csauth = { computers: [], unkeyed: [], error: e.message };
  } finally { S.csauthLoading = false; }
  if (S.view === "csauth") {
    renderList();
    if (S.selEv && S.evById.has(S.selEv)) showEvent(S.selEv);
    scrollSelIntoView();
  }
}

const OPS = { added: "+", removed: "−", changed: "~" };
function evLabel(ev) {
  return ev.kind === "v2" ? `heartbeat #${ev.beat}` : ev.kind === "issue" ? "issue: session start"
    : ev.kind === "close" ? "close: session end" : ev.path;
}
function evShort(ev) { return ev.kind === "v2" ? `#${ev.beat}` : ev.kind; }

function evChips(ev) {
  const b = [];
  const add = (t, cls, title) => b.push(el("span", { class: "chip " + (cls || ""), title }, t));
  if (ev.damaged) add("damaged", "err", ev.damaged);
  if (ev.initial) add("first call", "info", "the computer's first call in this capture: the whole request");
  if (ev.gap) add("gap " + fmtDur(ev.gap), "warn", `no heartbeat for ${fmtDur(ev.gap)} before this one`);
  for (const [k, v] of Object.entries((ev.response && ev.response.flags) || {}))
    add(k === "detected" ? `detected ×${v.length}` : k, k === "config_changed" ? "warn" : "err", JSON.stringify(v));
  const groups = new Map();
  for (const r of ev.reports || []) { const k = r.code + (r.label ? " " + r.label : ""); groups.set(k, (groups.get(k) || 0) + 1); }
  for (const [k, n] of groups) add(`+${n} ${k}`, k.startsWith("0xE2") ? "err" : "purple", "new report entries");
  const major = (ev.changes || []).filter((c) => !c.minor);
  if (major.length) add(major.slice(0, 3).map((c) => OPS[c.op] + c.path).join(", ") + (major.length > 3 ? ` +${major.length - 3}` : ""),
    "info", major.map((c) => `${c.op} ${c.path}`).join("\n"));
  const intro = Object.keys(ev.introduced || {});
  if (intro.length) add(`${intro.length} first-seen fields`, "", intro.join(", "));
  const rc = (ev.response && ev.response.changes) || [];
  if (rc.length) add("response: " + rc.map((c) => c.path).join(", "), "warn");
  if (!b.length && (ev.changes || []).length) add(`${ev.changes.length} minor`, "", ev.changes.map((c) => c.path).join("\n"));
  return b;
}

function evText(ev) {
  return [evLabel(ev), ev.kind, ev.client, ev.session, "#" + ev.conn, ev.damaged, ev.gap ? "gap" : "",
          ...(ev.changes || []).map((c) => c.path), ...(ev.reports || []).map((r) => `${r.code} ${r.label} ${r.list}`),
          ...Object.keys((ev.response && ev.response.flags) || {}), ...Object.keys(ev.introduced || {})].join(" ").toLowerCase();
}

function computerHead(c) {
  const facts = [c.ips.join(", "), c.accounts.length ? "account " + c.accounts.join(", ") : "", `${c.beats} heartbeats`];
  return el("div", { class: "comphead" },
    el("div", null, el("b", null, c.name || (c.hwid === "?" ? "calls without a readable hwid" : "unknown machine")), "  ",
      el("span", { class: "mono muted" }, c.hwid === "?" ? "" : "hwid " + c.hwid)),
    el("div", { class: "muted" }, facts.filter(Boolean).join(" · ")),
    c.unkeyed.map((u) => el("div", { class: "muted" }, `+ ${u.connections} x-csauth connections from ${u.client} not decrypted (${u.why}), `
      + `${fmtTime(u.first).slice(0, 8)} – ${fmtTime(u.last).slice(0, 8)}: a second NGS client whose TLS keys aren't in the key log`)));
}

function renderCsauth(box, ts) {
  if (!S.csauth) {
    box.replaceChildren(el("div", { class: "empty" }, "Reading the x-csauth calls…"));
    $("#count").textContent = "";
    loadCsauth();
    return;
  }
  const showQuiet = $("#t-quiet").checked;
  const tb = el("tbody");
  const vis = [];
  let total = 0;
  for (const c of S.csauth.computers) {
    tb.append(el("tr", { class: "comp" }, el("td", { colspan: 4 }, computerHead(c))));
    let run = [];
    const flush = () => {
      if (!run.length) return;
      const minor = run.filter((e) => (e.changes || []).length).length;
      tb.append(el("tr", { class: "fold" }, el("td", { class: "t" }, fmtTime(run[0].ts)),
        el("td", { colspan: 3, class: "muted" }, `⋯ ${run.length} heartbeat${run.length > 1 ? "s" : ""} with nothing new`
          + (run.length > 1 ? `, ${fmtTime(run[0].ts).slice(0, 8)} – ${fmtTime(run[run.length - 1].ts).slice(0, 8)}` : "")
          + (minor ? ` (${minor} with only minor sequence changes)` : ""))));
      run = [];
    };
    for (const ev of c.events) {
      total++;
      if (ts.length && !matchTerms(evText(ev), ts)) continue;
      if (ev.quiet && !showQuiet && !ts.length) { run.push(ev); continue; }
      flush();
      vis.push(ev.id);
      tb.append(el("tr", { "data-ev": ev.id, class: (ev.id === S.selEv ? "sel " : "") + (ev.quiet ? "dim" : "") },
        el("td", { class: "t", title: fmtDate(ev.ts) }, fmtTime(ev.ts)),
        el("td", { title: evLabel(ev) }, el("span", { class: "m" + (ev.kind === "v2" ? "" : " ws") }, evShort(ev))),
        el("td", { class: "num muted" }, "#" + ev.conn),
        el("td", { class: "chips" }, evChips(ev))));
    }
    flush();
  }
  for (const u of S.csauth.unkeyed) {
    tb.append(el("tr", { class: "comp" }, el("td", { colspan: 4 }, el("div", { class: "comphead muted" },
      `${u.client}: ${u.connections} x-csauth connections not decrypted (${u.why}), ${fmtTime(u.first).slice(0, 8)} – ${fmtTime(u.last).slice(0, 8)}`))));
  }
  S.visible = vis;
  $("#count").textContent = `${num(vis.length)} of ${num(total)} calls shown`;
  const table = el("table", { class: "grid" },
    el("colgroup", null, [104, 64, 56, null].map((w) => el("col", w ? { style: `width:${w}px` } : null))),
    el("thead", null, el("tr", null, ["Time", "Call", "Conn", "What's new"].map((h, i) => el("th", { class: i === 2 ? "num" : "" }, h)))), tb);
  if (S.csauth.error) box.replaceChildren(el("div", { class: "empty" }, "Could not read the x-csauth calls: " + S.csauth.error));
  else if (!S.csauth.computers.length && !S.csauth.unkeyed.length) box.replaceChildren(el("div", { class: "empty" }, "No x-csauth calls in this capture."));
  else box.replaceChildren(table);
}

function compact(v) {
  if (v === null || v === undefined) return "—";
  const s = JSON.stringify(v);
  return s.length > 240 ? s.slice(0, 240) + "…" : s;
}

function changesTable(changes) {
  const t = el("table", { class: "kv changes" });
  for (const c of changes) {
    t.append(el("tr", { class: c.minor ? "dim" : "" },
      el("td", { class: "k" }, OPS[c.op] + " " + c.path),
      el("td", { class: "v", title: c.op === "changed" ? `${JSON.stringify(c.old)} → ${JSON.stringify(c.new)}` : JSON.stringify(c.op === "removed" ? c.old : c.new) },
        c.op === "changed" ? [el("span", { class: "old" }, compact(c.old)), " → ", el("span", { class: "new" }, compact(c.new))]
          : c.op === "removed" ? el("span", { class: "old" }, compact(c.old)) : el("span", { class: "new" }, compact(c.new)),
        c.note ? el("div", { class: "muted" }, c.note) : null)));
  }
  return t;
}

function showEvent(id) {
  ++detailToken;
  const ev = S.evById.get(id), box = $("#detail");
  if (!ev) { box.replaceChildren(el("div", { class: "placeholder" }, "No such call.")); return; }
  const c = ev.computer;
  const head = el("div", { class: "dhead" },
    el("div", { class: "dtitle" }, el("span", { class: "m" }, evLabel(ev)), el("span", { class: "url" }, "https://x-csauth.ngs.nexon.com" + ev.path),
      ev.status != null ? el("span", { class: statusClass(ev.status) }, ev.status) : null),
    el("div", { class: "dmeta" }, el("span", null, fmtDate(ev.ts)), el("span", null, c.name || c.hwid),
      el("span", { class: "mono" }, ev.client), ev.session ? el("span", { class: "mono" }, "session " + ev.session) : null,
      el("a", { onclick: () => gotoItem(ev.id), title: "The whole request and response in the Requests view" }, "open the request"),
      el("a", { onclick: () => { S.selConn = ev.conn; setView("conns"); } }, `connection #${ev.conn}`)));
  const pane = el("div", { class: "pane" });
  const sect = (title, ...kids) => el("div", { class: "sect" }, el("div", { class: "sect-h" }, title), ...kids);
  if (ev.damaged) pane.append(el("div", { class: "banner err" }, ev.damaged));
  if (ev.gap) pane.append(el("div", { class: "banner warn" }, `No heartbeat for ${fmtDur(ev.gap)} before this one.`));
  const flags = Object.entries((ev.response && ev.response.flags) || {});
  if (flags.length) pane.append(sect("The server asked", kvTable(flags.map(([k, v]) => [k, JSON.stringify(v)]))));
  if ((ev.changes || []).length) pane.append(sect(`Changed (${ev.changes.length})`, changesTable(ev.changes)));
  if (((ev.response && ev.response.changes) || []).length) pane.append(sect("The response changed", changesTable(ev.response.changes)));
  if ((ev.reports || []).length) {
    const s = sect(`New reports (${ev.reports.length})`);
    for (const r of ev.reports) {
      s.append(el("div", { class: "report" }, el("div", { class: "rhead" },
        el("span", { class: "chip " + ((r.code || "").startsWith("0xE2") ? "err" : "purple") }, r.code || "?"), " ", r.label || "",
        el("span", { class: "muted" }, "  " + r.list)), jsonTree(r.entry)));
    }
    pane.append(s);
  }
  if (ev.response && ev.response.text) pane.append(sect("Response body", el("div", { class: "startline" }, ev.response.text)));
  const notes = [];
  const h = ev.hidden || {};
  if (h.decoys || h.markers) notes.push(`${h.decoys} decoy entries (log_code 0x221FFFFF) and ${h.markers} heartbeat marker (0x00000000) left out`);
  if (ev.repeated) notes.push(`${ev.repeated} report entries the same as in the previous heartbeat`);
  if (ev.answer_set != null) notes.push(`app_crc / module_crc: answer set ${ev.answer_set}` + (ev.answer_set_first
    ? ", the computer's first heartbeat with it (nothing to compare with)" : ", compared with the set's previous heartbeat"));
  if (notes.length) pane.append(el("div", { class: "muted sect" }, notes.join(". ") + "."));
  const intro = Object.keys(ev.introduced || {});
  if (intro.length) pane.append(el("details", { class: "sect" }, el("summary", null, `First-seen fields (${intro.length})`), jsonTree(ev.introduced)));
  if (ev.body) pane.append(sect("The whole request (decoys left out)", jsonTree(ev.body)));
  if (ev.quiet) pane.prepend(el("div", { class: "banner info" }, "Nothing new against the computer's previous heartbeat."
    + ((ev.changes || []).length ? " Only minor sequence changes, listed below." : "")));
  box.replaceChildren(head, pane);
  box.scrollTop = 0;
}

// ------------------------------------------------------------------------------------------ events

let deepTimer = 0;
async function runDeep() {
  const q = $("#q").value.trim();
  if (!q) { S.deepIds = null; renderList(); markSel(); return; }
  try {
    const r = await api("api/search", capParams({ q }));
    if ($("#q").value.trim() !== q) return;
    S.deepIds = new Set(r.ids);
  } catch (e) { S.deepIds = null; }
  renderList(); markSel();
}

function onFilter() {
  if ($("#deep").checked && S.view === "items") {
    clearTimeout(deepTimer);
    deepTimer = setTimeout(runDeep, 300);
    return;
  }
  S.deepIds = null;
  renderList();
  markSel();
}

function moveSel(delta) {
  if (!S.visible.length) return;
  const curr = S.view === "items" ? S.sel : S.view === "csauth" ? S.selEv : S.selConn;
  let i = S.visible.indexOf(curr);
  i = i < 0 ? (delta > 0 ? 0 : S.visible.length - 1) : Math.max(0, Math.min(S.visible.length - 1, i + delta));
  if (S.view === "items") selectItem(S.visible[i]); else if (S.view === "csauth") selectEv(S.visible[i]); else selectConn(S.visible[i]);
  scrollSelIntoView();
}

function wire() {
  $("#cap").addEventListener("change", (e) => { S.cap = e.target.value; S.sel = null; S.selConn = null; openCap(null); writeHash(); });
  $("#frames").addEventListener("change", (e) => { S.local = e.target.value; openCap({ sel: S.sel, conn: S.selConn }); writeHash(); });
  $("#reload").addEventListener("click", () => openCap({ sel: S.sel, conn: S.selConn }, true));
  setInterval(checkStale, 10000);
  for (const b of document.querySelectorAll(".tabs button")) b.addEventListener("click", () => setView(b.dataset.view));
  $("#q").addEventListener("input", onFilter);
  $("#deep").addEventListener("change", () => { S.deepIds = null; onFilter(); });
  for (const id of ["#host", "#t-http", "#t-ws", "#t-ctl", "#t-nokey", "#t-quiet"]) $(id).addEventListener("change", () => { renderList(); markSel(); });
  $("#list").addEventListener("click", (e) => {
    const tr = e.target.closest("tr[data-id], tr[data-n], tr[data-ev]");
    if (!tr) return;
    if (tr.dataset.ev) selectEv(tr.dataset.ev); else if (tr.dataset.id) selectItem(tr.dataset.id); else selectConn(+tr.dataset.n);
  });
  document.addEventListener("keydown", (e) => {
    const tag = (e.target.tagName || "").toLowerCase();
    if (e.key === "Escape" && e.target === $("#q")) { $("#q").value = ""; onFilter(); $("#q").blur(); return; }
    if (tag === "input" || tag === "select" || tag === "textarea" || e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.key === "/") { e.preventDefault(); $("#q").focus(); $("#q").select(); }
    else if (e.key === "ArrowDown" || e.key === "j") { e.preventDefault(); moveSel(1); }
    else if (e.key === "ArrowUp" || e.key === "k") { e.preventDefault(); moveSel(-1); }
  });
  // resizable split
  const main = $("#main"), split = $("#split");
  const saved = lsGet("listw");
  if (saved) main.style.setProperty("--listw", saved);
  split.addEventListener("pointerdown", (e) => {
    split.setPointerCapture(e.pointerId);
    split.classList.add("drag");
    const move = (ev) => {
      const r = main.getBoundingClientRect();
      const w = Math.max(280, Math.min(r.width - 280, ev.clientX - r.left));
      main.style.setProperty("--listw", w + "px");
    };
    const up = () => {
      split.classList.remove("drag");
      split.removeEventListener("pointermove", move);
      split.removeEventListener("pointerup", up);
      lsSet("listw", main.style.getPropertyValue("--listw"));
    };
    split.addEventListener("pointermove", move);
    split.addEventListener("pointerup", up);
  });
}

wire();
init();
