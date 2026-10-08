const input = document.getElementById("server"), msg = document.getElementById("msg");

chrome.storage.sync.get("server").then(({ server }) => { input.value = server || "http://localhost:8000"; });

document.getElementById("f").addEventListener("submit", async (e) => {
  e.preventDefault();
  let url;
  try { url = new URL(input.value.trim()); } catch { msg.textContent = "That is not a web address."; return; }
  const server = url.origin;
  // Ask Chrome for access to that one server only.
  const granted = await chrome.permissions.request({ origins: [`${server}/*`] }).catch(() => false);
  if (!granted && server !== "http://localhost:8000") {
    msg.textContent = "Chrome did not allow access to that address, so it was not saved.";
    return;
  }
  await chrome.storage.sync.set({ server });
  try {
    const res = await fetch(`${server}/search?q=&limit=1`);
    msg.textContent = res.ok ? `Saved. ${server} answers.` : `Saved, but ${server} answered ${res.status}.`;
  } catch {
    msg.textContent = `Saved, but ${server} does not answer yet.`;
  }
});
