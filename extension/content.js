// Opens the picker (an extension page in an iframe) above the box you are typing
// in, and inserts the chosen link there. Running it again closes the picker.
(() => {
  if (window.__memeclipPicker) { window.__memeclipPicker.close(); return; }

  const target = document.activeElement;
  const sel = window.getSelection();
  const range = sel && sel.rangeCount ? sel.getRangeAt(0).cloneRange() : null;
  const extOrigin = chrome.runtime.getURL("").replace(/\/$/, "");

  const host = document.createElement("div");
  const root = host.attachShadow({ mode: "closed" });
  root.innerHTML = `
    <style>
      .backdrop { position: fixed; inset: 0; z-index: 2147483646; }
      iframe { position: fixed; z-index: 2147483647; width: 420px; height: 470px;
               max-width: calc(100vw - 24px); max-height: calc(100vh - 24px);
               border: 0; border-radius: 14px; box-shadow: 0 18px 50px rgba(0,0,0,.35);
               color-scheme: normal; background: transparent; }
    </style>
    <div class="backdrop"></div>
    <iframe allow="clipboard-write" title="memeclip"></iframe>`;
  const frame = root.querySelector("iframe");
  frame.src = chrome.runtime.getURL("picker.html?mode=inline");

  // Sit just above the box being typed in, kept inside the window.
  const box = (target && target !== document.body ? target : document.body).getBoundingClientRect();
  const w = Math.min(420, innerWidth - 24), h = Math.min(470, innerHeight - 24);
  const left = Math.max(12, Math.min(box.right - w, innerWidth - w - 12));
  const top = box.top - h - 10 > 12 ? box.top - h - 10 : Math.max(12, Math.min(box.bottom + 10, innerHeight - h - 12));
  Object.assign(frame.style, { left: `${left}px`, top: `${top}px` });

  function close() {
    removeEventListener("message", onMessage);
    host.remove();
    delete window.__memeclipPicker;
    if (target && target.focus) target.focus();
  }
  function onMessage(e) {
    if (e.origin !== extOrigin || e.source !== frame.contentWindow || e.data?.source !== "memeclip") return;
    if (e.data.type === "pick") {
      close();
      window.__memeclipInsert(e.data.link + " ", target, range);
    } else if (e.data.type === "close") {
      close();
    }
  }
  addEventListener("message", onMessage);
  root.querySelector(".backdrop").addEventListener("mousedown", close);
  document.documentElement.append(host);
  window.__memeclipPicker = { close };
})();
