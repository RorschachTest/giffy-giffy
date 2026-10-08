# Themes

The page (`app/static/index.html`) is the same for every design; a theme is one
CSS file here. `app/static/base.css` holds the shared mechanics (layout
skeleton, video and subtitle positioning, Curate show/hide, accessibility).

## Choose the theme

- In code: `UI_THEME = os.getenv("UI_THEME") or "viza"` in `app/config.py`
- Or per machine: `UI_THEME=picker` in `.env`, then `docker compose up -d api`
- Preview without changing anything: `http://localhost:8000/?theme=sticker`
- List them: `GET /themes`

| Theme | Look |
|---|---|
| `picker` | Dark screening room; straight to the search; subtitle yellow as the only accent |
| `sticker` | high-end-visual-design: glass nav, double-bezel clips, tilted posters with mood stickers |
| `viza` | Gen Z colour blocks (magenta, cobalt, yellow, plum); clips play inside purple scenes |

## Add a theme

Copy a file, rename it (`themes/neon.css`), change it. It shows up in `/themes`
and `?theme=neon` at once; no code changes.

A theme styles these classes: `.island-wrap .island .wordmark .cta .knob`
(`.light .ghost .round .done`), `.hero .eyebrow h1 mark .tag .lede .search .core #q`,
`.tabs .tab .quick .pill #status .read`, `.grid .card .card.feature .shell .core .frame`,
`.sub .snd .bar-time .stick .badge .row .name .by .acts`, `.chip .metric .tools .empty`,
`table dialog label input`.

And may set these variables on `:root`:

| Variable | What it does |
|---|---|
| `--bg --text --focus --font` | Base colours and font, used by base.css |
| `--theme-color` | Browser chrome colour on phones |
| `--mood-<name>`, `--mood-<name>-on` | Colour of a mood sticker / quick search and the text on it (`--mood-default`, `--mood-on-default` for the rest). Names: shocked, in-a-hurry, happy, laughing, angry, confused, helpless, sad, scared, preachy, smug, disgusted, dancing, money |
| `--reveal: on` | Clips fade up as they scroll in (`--reveal-y --reveal-blur --reveal-dur --reveal-ease`) |
| `--scenes: on` | Clips play on real surfaces from `../scenes/scenes.json` (`--scene-order`, `--scene-feature`, `--scene-bg`, `--scene-shade-opacity`) |
| `--icon-stroke` | Line weight of the icons |

Fonts: `@import` them at the top of the theme file.

## Scenes

`app/static/scenes/scenes.json` lists each surface: a photo, its four corners
(fractions of the photo, clockwise from top-left) and an optional shade layer.
Add one by dropping a photo in, marking its corners, and adding its name to a
theme's `--scene-order`. Photo credits: `app/static/scenes/CREDITS.md`.
