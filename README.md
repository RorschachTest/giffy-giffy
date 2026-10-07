# memeclip

Search short meme clips (with audio) by what is said, who is in it, what it is
called, or what happens in it. First runnable slice: drop a clip in a folder,
it gets processed, and you can search it from a browser.

## Run it

You need Docker with Docker Compose.

```bash
cp .env.example .env        # optional; the defaults work
docker compose up --build
```

Open http://localhost:8000.

The first start downloads the speech, embedding and face models (roughly 1 GB
in total) into a Docker volume, so it needs internet access once and takes a
few minutes. Later starts are quick.

If the image fails to build on the face-recognition step, skip that part for
now: set `WITH_FACES=0` and `ENABLE_FACES=0` in `.env` and build again.
Everything else works without it.

To build, start and check everything in one go (the output is also saved to
`data/smoke_test.log`):

```bash
bash scripts/smoke_test.sh
```

It waits for the clips in `data/inbox/` to be processed, then prints what was
heard, who was recognised and what was flagged for each clip. If
`data/smoke_queries.txt` exists, it also runs those searches and reports
pass or fail. One search per line:

```
kya karu main => kya karu mein
```

The left side is what you would type; the right side is words from the title
of the clip you expect on top.

## Add clips

Either use **Add clip** on the page, or copy files into `data/inbox/`.
The worker picks them up within a few seconds, then moves the original to
`data/processed/` (or `data/failed/` with a note saying why).

Give each clip whatever context you have by putting a JSON file with the same
name next to it. `rasode.mp4` + `rasode.json`:

```json
{
  "title": "Rasode mein kaun tha | Kokilaben meme template",
  "caption": "when mom finds the empty cooker",
  "hashtags": ["kokilaben"],
  "comments": ["rashi ben 😂"],
  "language": "hi",
  "source_url": "https://...",

  "folk_names": ["kokilaben meme", "cooker meme"],
  "people": ["Rupal Patel"],
  "source_title": "Saath Nibhaana Saathiya",
  "description": "Stern older woman questions two nervous younger women",
  "reactions": ["interrogating", "caught"],
  "use_when": ["finding out who did it"],
  "transcript": "रसोड़े में कौन था"
}
```

Every key is optional. `language` is worth setting whenever you know it:
auto-detect is unreliable on a five second clip. `transcript` skips speech to
text and uses your text instead.

`yt-dlp --write-info-json` writes `name.info.json` files, and those are read
too (title, description, tags, comments, URL), so its output can go straight
into the inbox. Clips over `MAX_CLIP_SECONDS` (30) are rejected, so trim first.
Downloading from a platform is subject to that platform's terms.

## Add famous people

No training. Make a folder per person and put a few clear face photos in it:

```
data/gallery/Nana Patekar/1.jpg
data/gallery/Nana Patekar/2.jpg
data/gallery/Nana Patekar/aliases.txt     <- optional, one alias per line
```

The worker notices the change, stores the faces, and re-checks old clips.
Three to five photos per person (different ages and angles) is a good start.
Names and aliases are also matched in titles, captions and comments, so a
person with an empty folder (name only) is still found when the caption names
them. One-word aliases shorter than five letters are ignored there, because
they collide with ordinary words ("nana" is also "grandfather").

Faces the gallery does not know are kept. To name the ones that keep turning up:

```bash
docker compose exec worker python -m app.gallery unknown
docker compose exec worker python -m app.gallery name <face_id> "Full Name"
```

Every clip showing that face inherits the name.

## How a search is answered

| How people search          | Example                     | Column     | Filled from                                  |
|----------------------------|-----------------------------|------------|----------------------------------------------|
| The meme's street name     | `kokilaben meme`            | `s_names`  | folk names, cleaned title                    |
| Half-remembered dialogue   | `rasode me kon tha`         | `s_said`   | speech to text, romanised                    |
| Person or source           | `nana patekar`, `chintu`    | `s_people` | faces, names in captions, aliases, film/show |
| What happens, when to use  | `mom caught me`             | `s_meta` + embedding | description, reactions, use-when, topics, caption, comments |
| What it is about / the mood | `when boss shouts at me`   | `tm` (topics, reactions) | Laya reads the query's topic and mood |
| A query that worked before | anything                    | `s_learned`| queries that ended in a share                |

Text goes through one `normalise()` function (`app/text.py`) at indexing time
and again at query time: Devanagari to Roman, lowercase, stretched letters
collapsed, common chat spellings unified. Matching is then fuzzy (trigrams)
plus meaning-based (embeddings). The page shows the keyword and meaning score
of every result so you can see why it ranked.

The clip row stores facts; the `s_*` columns and the embedding are derived
from them in `app/indexing.py`. After changing a rule:

```bash
docker compose exec worker python -m app.reindex
```

## Mood and topics

Every clip gets three kinds of derived metadata, all from local models. The
sidecar or the Edit button always wins over them.

| Field | From |
|---|---|
| `reactions` (mood) | facial expressions (FER+), keyframes vs mood sentences (CLIP), and the title + dialogue read by Laya (`app/vibe.py`) |
| `use_when`, `description` | a fixed table and a template per mood |
| `topics` | [Laya](https://github.com/NandhaKishorM/laya): a choice over `app/laya.py` `TOPICS`, on the title alone and on title + dialogue; kept when either is confident |

Laya is a decision model, not a chat model: it answers typed questions
(choice, score, yes/no) about text with a probability per option and never
generates text. It runs as the `laya` service; its sandbox and API docs are at
http://localhost:8001/docs. At search time it reads what the query is about
(`/search` returns this as `intent`), and clips whose topics or reactions match
score `tm`. With Laya off or down, ingest flags `laya_failed` and search works
as before.

To add Laya topics and redo the mood on clips indexed before it existed:

```bash
docker compose exec worker python -m app.reindex --laya --vibe --replace
```

`--replace` overwrites hand-edited mood fields too; leave it off to fill only
empty ones.

## Share links and the browser extension

Every clip has a short link, `/c/<first 10 hex of its SHA-256>`
(`app/share.py`). The page behind it plays the clip and carries the tags chat
apps read to draw a preview: `og:video`, a `twitter:player` card and oEmbed.
**Copy link** on the test page copies it, and so does the browser extension in
[`extension/`](extension/README.md), which pastes it straight into WhatsApp Web,
Slack or Discord with **Alt+Shift+M**.

Chat apps fetch previews from the internet, so set `PUBLIC_BASE_URL` to the
server's public https address; with `localhost` the link works but no app can
preview it.

## Layout

```
app/text.py       transliteration, normalisation, caption cleaning
app/media.py      ffmpeg: probe, keyframes, audio, web copy, picture hash
app/stt.py        speech to text (faster-whisper)
app/faces.py      face gallery, matching, unknown-face grouping (InsightFace)
app/embed.py      text embeddings (fastembed, multilingual MiniLM)
app/vibe.py       mood: facial expressions, CLIP scene, dialogue tone
app/laya.py       Laya client: clip topics + text mood, search intent
app/share.py      share links /c/<id>: video page, preview tags, oEmbed
extension/        browser extension: search and paste share links
app/pipeline.py   the per-clip assembly line, duplicates, review flags
app/indexing.py   facts -> searchable columns
app/search.py     hybrid ranking, share feedback
app/worker.py     watches data/inbox
app/api.py        HTTP API + test page
app/smoke.py      report + test searches used by scripts/smoke_test.sh
app/schema.sql    tables (Postgres + pgvector + pg_trgm)
```

## Tests

```bash
pip install pytest httpx -r requirements.txt
pytest tests/test_text.py                 # no services needed
TEST_DATABASE_URL=postgresql://... pytest # needs Postgres with pgvector, and ffmpeg
```

The integration tests drop and recreate the app's tables in that database.
They switch the three models off and use made-up face vectors, so they check
the plumbing and the ranking logic, not model quality.

## Not built yet

- A real description of what happens on screen. `description` is only a
  template built from the mood signals.
- Reading on-screen text (OCR) and recognising songs or sound effects.
- Audio fingerprints for duplicates. Picture hashes catch re-encodes and
  resizes, not crops or heavy watermarks.
- Automatic collection from sources, moderation, accounts.

## Running it somewhere other than your laptop

The same `docker compose up` works on any single Linux server; no GPU is
needed at this size. Around 4 CPU cores and 8 GB of memory is a comfortable
start. Before exposing it publicly: change the database password, put HTTPS in
front (Caddy or nginx), and add authentication to upload and edit.

Licence note: the InsightFace code is MIT, but its pretrained face models are
published for non-commercial research use. That is fine for a prototype;
resolve it before a commercial launch.
