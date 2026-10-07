// The in-page picker opens only when you press the shortcut. That keypress grants
// `activeTab` for the current tab, so the extension never needs to read every page.
chrome.commands.onCommand.addListener(async (command, tab) => {
  if (command !== "open-picker" || !tab?.id) return;
  try {
    await chrome.scripting.executeScript({ target: { tabId: tab.id }, files: ["insert.js", "content.js"] });
  } catch (e) {
    // chrome:// pages, the Web Store and PDF viewers cannot be scripted.
    console.warn("memeclip: cannot open the picker on this page", e);
  }
});
