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

## Why it is funny, and what it says in a chat

`app/understand.py` reads a meme the way a person in the chat would: what it
communicates as a reply, who it is aimed at, and what makes it funny. No LLM
and no training run: everything is measured in the same sentence-embedding
space search uses, plus the mood scores the pipeline already stores.

```bash
curl -s localhost:8000/understand -H 'content-type: application/json' -d '{
  "link": "https://memeclip.aiwroteit.dev/c/3f2a9b0c1d",
  "context": ["who finished the milk?", "not me"],
  "caption": "you 😂"}'
```

Shortened response (the numbers depend on the embedding model and on feedback):

```json
{"communicates": {"intent": "call_out", "aimed_at": "recipient", "confidence": 0.75,
                  "reading": "Calling someone out: \"who did this?\" / \"caught you\".",
                  "evidence": ["the clip's mood: shocked, angry", "the message reads as question (61%)", "..."]},
 "funny": 0.62,
 "why_funny": [{"mechanism": "incongruity", "why": "Incongruity: the message (...) and the clip (...) are far apart, yet both fit \"finding out who did it\" ..."}]}
```

**What it communicates** (`app/intent.py`): one of 16 intents (agree, mock,
sarcasm, celebrate, sympathise, call out, told you so, ...), as a product of
experts: the clip's own moods/topics/words, the dialogue act of the last
messages pushed through a matrix of adjacency pairs (good news -> celebrate,
confession -> call out, request -> refuse), the sentiment contrast between
message and meme (cheerful message + grim clip = irony), and the sender's
caption.

**Why it is funny** (`app/humor.py`): humor theories as measurements —
incongruity-resolution, Raskin's script opposition, tonal mismatch (face vs
words), benign violation, superiority, relief, exaggeration, relatability,
self-deprecation, recognition and laughter in the comments — combined by a
logistic model. On its own a clip's setup is its caption; in a chat the setup
is the message before it, which is where reaction templates become jokes.

**It learns** from `POST /understand/feedback` (`{"link" | "clip_id",
"context", "caption", "intent", "funny"}`): each clip gets a Dirichlet
posterior over how people actually use it, the act -> intent matrix is
re-estimated, and the funniness weights are refitted around their starting
values. `GET /understand/stats` shows how often the engine agreed with people.
From a shell: `docker compose exec api python -m app.understand <clip id or
share link> "message before it" --caption "me rn"`.

The prototype sentences and tables in `humor.py` and `intent.py` are the
engine's whole "knowledge"; edit them like any other rule. Background and the
datasets to calibrate against: [docs/meme-understanding.md](docs/meme-understanding.md).

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
app/semspace.py   embedding maths shared by the engine: z-scores, prototype groups, pooling
app/humor.py      why a meme is funny: humor theories as measurements, funniness model
app/intent.py     what a meme says as a reply: dialogue acts, intents, product of experts
app/understand.py the understanding engine: clip + chat in, reading out, learns from feedback
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

## Data and legal

memeclip is a personal, non-commercial project. The full statement is served
at `/legal` (source: [`app/static/legal.html`](app/static/legal.html)) and
linked from every share page as **Removal requests**. In short:

- **The clips are not ours.** Most are cut from films, TV and online videos;
  copyright stays with their owners and memeclip holds no licence. This
  repository contains no clips: `data/**` (videos, thumbnails, face photos)
  is kept out of git.
- **The running site is private.** Search, upload and edit sit behind a login
  (see Public access below). Only a clip whose link is shared (`/c/…`) is
  publicly reachable, and its page links to the source when one is known.
- **Where clips come from.** Files added by hand, and public compilation videos
  downloaded and cut into clips for this personal project by a separate,
  private collection script.
  Downloading may go against the source site's terms, so the clips stay private:
  they are never re-uploaded, sold or shared as a dataset.
- **Faces stay local.** The face gallery covers public figures only, runs on
  this machine, and is never published. Unknown faces are named only by hand.
- **Removal:** open an issue with the share link and the clip is taken down,
  with the person's gallery photos too if they ask.

What a commercial version would need instead: licensed clips, or user uploads
with a working takedown process; the IT Rules 2021 duties (grievance officer);
India's DPDP Act; consent for using well-known people's likeness; and
commercial licences for the face models (see the licence note at the end).

## Public access (Cloudflare Tunnel)

The app is published at `https://memeclip.aiwroteit.dev` through a Cloudflare
Tunnel: an outgoing connection from this machine to Cloudflare, so no router
ports are open and HTTPS is Cloudflare's. `cloudflared` runs as a macOS system
service, separate from Docker, and starts at boot.

1. Cloudflare dashboard, **Zero Trust > Networks > Tunnels**: create a tunnel
   and copy its connector token.
2. Install the service (the token is stored in a root-only file):

   ```bash
   brew install cloudflared
   sudo cloudflared service install <TOKEN>
   ```

3. In the tunnel, **Public Hostname**: `memeclip` . `aiwroteit.dev`, service
   **HTTP**, URL `localhost:8000` (the service runs on the host, so it reaches
   the api container through its published port).
4. In `.env`: `PUBLIC_BASE_URL=https://memeclip.aiwroteit.dev`, then
   `docker compose up -d api worker`.
5. **Zero Trust > Access > Applications**: one self-hosted app for
   `memeclip.aiwroteit.dev` allowing only your email, and one for the paths
   `c/*`, `media/*`, `oembed`, `legal` with a Bypass policy, so share links,
   chat previews and the legal page stay public while upload and edit need a
   login.

Logs: `/Library/Logs/com.cloudflare.cloudflared.err.log`. To rotate the token:
`sudo cloudflared service uninstall`, then install again with the new one.

## Running it somewhere other than your laptop

The same `docker compose up` works on any single Linux server; no GPU is
needed at this size. Around 4 CPU cores and 8 GB of memory is a comfortable
start. Before exposing it publicly: change the database password, put HTTPS in
front (Caddy or nginx), and add authentication to upload and edit.

Licence note: the InsightFace code is MIT, but its pretrained face models are
published for non-commercial research use. That is fine for a prototype;
resolve it before a commercial launch.
