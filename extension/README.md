# memeclip browser extension

Search memeclip from any web page and paste a short link (`/c/<id>`) into the
box you are typing in. Chat apps unfurl that link as the clip: the video stays
on the memeclip server.

Chrome and Edge (Manifest V3). Works in browser tabs only, e.g. WhatsApp Web,
Slack, Discord, Telegram Web, Gmail; not in desktop apps.

## Install (unpacked)

1. Start memeclip (`docker compose up -d` in the repo root).
2. Open `chrome://extensions` (Edge: `edge://extensions`) and switch on
   **Developer mode**.
3. Click **Load unpacked** and pick this `extension/` folder.
4. If memeclip is not at `http://localhost:8000`, open the extension's
   **Details → Extension options** and set the address.

## Use

| Keys | What happens |
|---|---|
| **Alt+Shift+M** (Mac: ⌥⇧M) while typing in a chat | The picker opens above the box. Type, pick with the arrow keys, **Enter** pastes the link at the cursor. **Esc** closes. |
| **Alt+Shift+K** or the toolbar button | Search popup. Picking a clip copies the link and pastes it into the box you were in. |

Change the keys at `chrome://extensions/shortcuts`.

Every pick counts as a share, so memeclip learns which words found the clip.

## How the link shows up

The chat app fetches `/c/<id>` and reads its `og:video`, `twitter:player` and
oEmbed tags. iMessage, Discord and usually Telegram play the clip inline;
Slack plays an embedded player; WhatsApp shows the thumbnail and title and
opens the page on tap. None of that works for `localhost`: set
`PUBLIC_BASE_URL` on the server to its public https address first.

## Permissions

- `activeTab`, `scripting`: the picker is added to a page only when you press
  the shortcut or the toolbar button; the extension does not read pages
  otherwise.
- `storage`: the server address.
- Host access to `http://localhost:8000` by default; any other server address
  is asked for when you save it in the options.

## Files

```
manifest.json   permissions, shortcuts
background.js   opens the in-page picker on Alt+Shift+M
content.js      places the picker over the page, inserts the chosen link
insert.js       inserts text at the caret (inputs, rich chat editors, paste fallback)
picker.*        the search UI, shared by the popup and the in-page picker
options.*       server address
```
