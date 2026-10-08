// The picker: search the memeclip server, choose a clip, and hand back its share
// link. Used two ways:
//   popup   (toolbar button)  copies the link and pastes it into the page you were on
//   inline  (Alt+Shift+M)     an iframe over the page; content.js inserts the link
const MODE = new URLSearchParams(location.search).get("mode") || "popup";
const DEFAULT_SERVER = "http://localhost:8000";
const COLUMNS = 3;

const $ = (id) => document.getElementById(id);
const q = $("q"), grid = $("grid"), status = $("status");
let server = DEFAULT_SERVER, results = [], selected = 0, timer = null, lastQuery = "", seq = 0;

if (MODE === "popup") document.body.classList.add("popup");

async function loadServer() {
  try {
    const { server: s } = await chrome.storage.sync.get("server");
    if (s) server = s.replace(/\/+$/, "");
  } catch {}
}

function abs(path) { return path ? new URL(path, server + "/").href : ""; }

async function search(text) {
  const mine = ++seq;
  lastQuery = text.trim();
  status.textContent = lastQuery ? "Searching…" : "Trending";
  try {
    const res = await fetch(`${server}/search?` + new URLSearchParams({ q: lastQuery, limit: "24" }));
    if (!res.ok) throw new Error(`server answered ${res.status}`);
    const data = await res.json();
    if (mine !== seq) return;   // a newer search already started
    results = data.results || [];
    selected = 0;
    render();
    const intent = Object.entries(data.intent || {}).map(([k, p]) => `${k} ${p}`).join(", ");
    status.textContent = !lastQuery ? "Trending"
      : `${results.length} clip${results.length === 1 ? "" : "s"}` + (intent ? ` · read as ${intent}` : "");
  } catch (e) {
    if (mine !== seq) return;
    results = [];
    render();
    status.textContent = `Can't reach ${server}. Is memeclip running? Check Settings.`;
  }
}

function render() {
  grid.replaceChildren();
  if (!results.length) {
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.textContent = lastQuery ? "No clip for that yet. Try other words, or the dialogue." : "No clips yet.";
    grid.append(empty);
    return;
  }
  results.forEach((c, i) => {
    const tile = document.createElement("button");
    tile.className = "tile";
    tile.setAttribute("role", "option");
    tile.setAttribute("aria-selected", String(i === selected));
    tile.title = c.transcript_roman ? `“${c.transcript_roman}”` : (c.title || "");
    const media = document.createElement("div");
    media.className = "media";
    const img = document.createElement("img");
    img.src = abs(c.thumb);
    img.alt = "";
    img.loading = "lazy";
    media.append(img);
    const name = document.createElement("div");
    name.className = "name";
    name.textContent = (c.folk_names && c.folk_names[0]) || c.title || "(untitled)";
    tile.append(media, name);
    tile.onmouseenter = () => { select(i, false); };
    tile.onclick = () => pick(c);
    grid.append(tile);
  });
  preview();
}

// Only the selected tile plays, muted, so the grid stays light.
function preview() {
  grid.querySelectorAll("video").forEach((v) => v.remove());
  const tile = grid.children[selected], c = results[selected];
  if (!tile || !c || matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  const v = document.createElement("video");
  Object.assign(v, { src: abs(c.url), muted: true, loop: true, autoplay: true, playsInline: true });
  tile.querySelector(".media").append(v);
}

function select(i, scroll = true) {
  if (!results.length) return;
  selected = Math.max(0, Math.min(results.length - 1, i));
  [...grid.children].forEach((t, j) => t.setAttribute("aria-selected", String(j === selected)));
  if (scroll) grid.children[selected]?.scrollIntoView({ block: "nearest" });
  preview();
}

function recordShare(c) {
  // Teaches search that this query finds this clip (see search.record_share).
  return fetch(`${server}/clips/${c.id}/share`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ q: lastQuery || null }),
  }).catch(() => {});
}

async function pick(c) {
  const link = c.share_url || abs(c.url);
  recordShare(c);
  if (MODE === "inline") {
    parent.postMessage({ source: "memeclip", type: "pick", link }, "*");
    return;
  }
  // Popup: copy, then try to paste into the box that had focus on the page.
  let copied = false, pasted = false;
  try { await navigator.clipboard.writeText(link); copied = true; } catch {}
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    await chrome.scripting.executeScript({ target: { tabId: tab.id }, files: ["insert.js"] });
    const [r] = await chrome.scripting.executeScript({
      target: { tabId: tab.id }, func: (t) => window.__memeclipInsert(t), args: [link + " "],
    });
    pasted = !!r?.result;
  } catch {}
  status.textContent = pasted ? "Pasted into the page." : copied ? "Link copied. Paste it with ⌘V / Ctrl+V." : link;
  setTimeout(() => window.close(), pasted || copied ? 700 : 4000);
}

function closePicker() {
  if (MODE === "inline") parent.postMessage({ source: "memeclip", type: "close" }, "*");
  else window.close();
}

q.addEventListener("input", () => {
  clearTimeout(timer);
  timer = setTimeout(() => search(q.value), 220);
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") { e.preventDefault(); closePicker(); }
  else if (e.key === "Enter") { e.preventDefault(); if (results[selected]) pick(results[selected]); }
  else if (e.key === "ArrowDown") { e.preventDefault(); select(selected + COLUMNS); }
  else if (e.key === "ArrowUp") { e.preventDefault(); select(selected - COLUMNS); }
  else if (e.key === "ArrowRight" && (document.activeElement !== q || q.selectionStart === q.value.length)) {
    e.preventDefault(); select(selected + 1);
  } else if (e.key === "ArrowLeft" && (document.activeElement !== q || q.selectionStart === 0)) {
    e.preventDefault(); select(selected - 1);
  }
});

(async () => {
  await loadServer();
  q.focus();
  search("");
})();
