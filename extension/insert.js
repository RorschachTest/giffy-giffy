// Puts text at the caret of the box the person was typing in, the way typing or
// pasting would, so chat editors (WhatsApp Web, Slack, Discord, Gmail) notice it.
// Returns true only if the box's text really changed.
window.__memeclipInsert = (text, target, range) => {
  const el = target || document.activeElement;
  if (!el || el === document.body) return false;
  el.focus();

  if (el instanceof HTMLInputElement || el instanceof HTMLTextAreaElement) {
    const before = el.value;
    const start = el.selectionStart ?? el.value.length, end = el.selectionEnd ?? start;
    el.setRangeText(text, start, end, "end");
    el.dispatchEvent(new InputEvent("input", { bubbles: true, inputType: "insertText", data: text }));
    return el.value !== before;
  }

  if (el.isContentEditable) {
    const before = el.textContent;
    if (range) {
      const sel = window.getSelection();
      sel.removeAllRanges();
      sel.addRange(range);
    }
    // Rich editors listen for beforeinput / input; execCommand fires both.
    document.execCommand("insertText", false, text);
    if (el.textContent !== before) return true;
    // Fallback: hand the editor a paste, which every chat app handles.
    const data = new DataTransfer();
    data.setData("text/plain", text);
    el.dispatchEvent(new ClipboardEvent("paste", { clipboardData: data, bubbles: true, cancelable: true }));
    return el.textContent !== before;
  }
  return false;
};
