-- {EMBED_DIM} and {FACE_DIM} are filled in by app/db.py.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS clips (
    id               BIGSERIAL PRIMARY KEY,

    -- the file
    sha256           TEXT NOT NULL UNIQUE,
    phash            BIGINT[] NOT NULL DEFAULT '{}',   -- one 64-bit picture hash per keyframe
    file             TEXT NOT NULL,                    -- relative to MEDIA_DIR
    thumb            TEXT,
    duration         REAL,
    width            INT,
    height           INT,
    has_audio        BOOLEAN NOT NULL DEFAULT TRUE,

    -- FACTS: what is said
    language          TEXT,
    transcript_native TEXT NOT NULL DEFAULT '',
    transcript_roman  TEXT NOT NULL DEFAULT '',

    -- FACTS: what the source told us
    title            TEXT NOT NULL DEFAULT '',
    caption          TEXT NOT NULL DEFAULT '',
    hashtags         TEXT[] NOT NULL DEFAULT '{}',
    comments         TEXT[] NOT NULL DEFAULT '{}',
    source_urls      TEXT[] NOT NULL DEFAULT '{}',

    -- FACTS: who and what (faces, metadata, or typed by a human)
    folk_names       TEXT[] NOT NULL DEFAULT '{}',     -- what the internet calls this meme
    people           TEXT[] NOT NULL DEFAULT '{}',
    source_title     TEXT NOT NULL DEFAULT '',         -- film / show
    description      TEXT NOT NULL DEFAULT '',         -- what happens on screen
    reactions        TEXT[] NOT NULL DEFAULT '{}',
    use_when         TEXT[] NOT NULL DEFAULT '{}',
    topics           TEXT[] NOT NULL DEFAULT '{}',     -- what it is about (Laya, sidecar or by hand)

    -- EVIDENCE: real queries that ended in a share, {normalised query: count}
    learned_queries  JSONB NOT NULL DEFAULT '{}',

    -- DERIVED search columns. Never edit by hand; app/indexing.py rebuilds them.
    s_names          TEXT NOT NULL DEFAULT '',
    s_said           TEXT NOT NULL DEFAULT '',
    s_people         TEXT NOT NULL DEFAULT '',
    s_meta           TEXT NOT NULL DEFAULT '',
    s_learned        TEXT NOT NULL DEFAULT '',
    s_all            TEXT NOT NULL DEFAULT '',
    embedding        vector({EMBED_DIM}),

    -- ranking signals
    duplicates_seen  INT NOT NULL DEFAULT 1,
    shares           INT NOT NULL DEFAULT 0,
    first_seen       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- quality control
    needs_review     BOOLEAN NOT NULL DEFAULT FALSE,
    review_reasons   TEXT[] NOT NULL DEFAULT '{}',
    raw              JSONB NOT NULL DEFAULT '{}'        -- raw output of every step, for reprocessing
);

-- Columns added after the first release. There are no migrations, so each one is
-- added here in a way that is a no-op on a database that already has it.
ALTER TABLE clips ADD COLUMN IF NOT EXISTS topics TEXT[] NOT NULL DEFAULT '{}';

CREATE INDEX IF NOT EXISTS clips_topics ON clips USING gin (topics);
-- share links are /c/<first 10 hex of sha256> (app/share.py)
CREATE INDEX IF NOT EXISTS clips_share_id ON clips (left(sha256, 10));
-- browsing order (app/search.py BROWSE_ORDER): keyset paging reads straight from it
CREATE INDEX IF NOT EXISTS clips_browse ON clips ((shares + duplicates_seen) DESC, first_seen DESC, id DESC);
CREATE INDEX IF NOT EXISTS clips_s_names_trgm   ON clips USING gin (s_names gin_trgm_ops);
CREATE INDEX IF NOT EXISTS clips_s_said_trgm    ON clips USING gin (s_said gin_trgm_ops);
CREATE INDEX IF NOT EXISTS clips_s_people_trgm  ON clips USING gin (s_people gin_trgm_ops);
CREATE INDEX IF NOT EXISTS clips_s_learned_trgm ON clips USING gin (s_learned gin_trgm_ops);
CREATE INDEX IF NOT EXISTS clips_s_all_trgm     ON clips USING gin (s_all gin_trgm_ops);
CREATE INDEX IF NOT EXISTS clips_embedding_hnsw ON clips USING hnsw (embedding vector_cosine_ops);

-- Famous people: one row per person, several reference face photos each.
CREATE TABLE IF NOT EXISTS people (
    id       BIGSERIAL PRIMARY KEY,
    name     TEXT NOT NULL UNIQUE,
    aliases  TEXT[] NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS face_refs (
    id         BIGSERIAL PRIMARY KEY,
    person_id  BIGINT NOT NULL REFERENCES people(id) ON DELETE CASCADE,
    source     TEXT NOT NULL,                 -- gallery photo path, or "clip_face:<id>"
    embedding  vector({FACE_DIM}) NOT NULL,
    UNIQUE (person_id, source)
);

-- Every face seen in a clip. person_id is NULL until somebody is matched or labelled.
CREATE TABLE IF NOT EXISTS clip_faces (
    id         BIGSERIAL PRIMARY KEY,
    clip_id    BIGINT NOT NULL REFERENCES clips(id) ON DELETE CASCADE,
    frame      INT NOT NULL,
    person_id  BIGINT REFERENCES people(id) ON DELETE SET NULL,
    score      REAL,
    embedding  vector({FACE_DIM}) NOT NULL
);
CREATE INDEX IF NOT EXISTS clip_faces_clip ON clip_faces (clip_id);
ALTER TABLE clip_faces ADD COLUMN IF NOT EXISTS crop TEXT;   -- storage key of a small picture of the face
-- Text burned into the video, read off the keyframes and cleaned (app/ocr.py); raw readings in raw->'ocr'
ALTER TABLE clips ADD COLUMN IF NOT EXISTS screen_text TEXT NOT NULL DEFAULT '';
-- people a person said this face is NOT ("Not puneet"); automatic matching never gives them back
ALTER TABLE clip_faces ADD COLUMN IF NOT EXISTS rejected BIGINT[] NOT NULL DEFAULT '{}';
-- A merged re-upload points at the clip it duplicates (python -m app.dedupe). The row and
-- its files are kept; search, browse and faces skip it, and its share link redirects.
ALTER TABLE clips ADD COLUMN IF NOT EXISTS duplicate_of BIGINT REFERENCES clips(id) ON DELETE SET NULL;
-- What the footage looks like: CLIP image vector averaged over the keyframes. Unlike the
-- picture hash it survives zoom, crop and watermarks (re-uploads: >= 0.94; other clips
-- reusing the same sound: <= 0.77), so duplicate checks use it (media.same_clip).
ALTER TABLE clips ADD COLUMN IF NOT EXISTS visual vector(512);

-- What a meme meant when it was sent in a chat, as people told us (app/understand.py).
-- acts and features are measured at feedback time, so learning never re-embeds.
CREATE TABLE IF NOT EXISTS meme_feedback (
    id         BIGSERIAL PRIMARY KEY,
    at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    clip_id    BIGINT NOT NULL REFERENCES clips(id) ON DELETE CASCADE,
    context    TEXT[] NOT NULL DEFAULT '{}',   -- the messages before the meme, oldest first
    caption    TEXT NOT NULL DEFAULT '',       -- what the sender typed with it
    intent     TEXT,                           -- what it meant (intent.INTENTS), NULL = not said
    funny      BOOLEAN,                        -- did it land, NULL = not said
    acts       JSONB NOT NULL DEFAULT '{}',    -- dialogue acts of the context
    features   JSONB NOT NULL DEFAULT '{}',    -- humor mechanism scores
    predicted  TEXT                            -- what the engine read, to measure accuracy
);
CREATE INDEX IF NOT EXISTS meme_feedback_clip ON meme_feedback (clip_id);

-- Every search, so you can see what people type and what finds nothing.
CREATE TABLE IF NOT EXISTS query_log (
    id          BIGSERIAL PRIMARY KEY,
    at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    query       TEXT NOT NULL,
    query_norm  TEXT NOT NULL,
    results     INT NOT NULL,
    top_score   REAL
);

-- What a vision model wrote about each clip (app/describe.py): kept so nothing is asked twice.
-- One row per clip and prompt version; a new prompt writes new rows next to the old ones.
CREATE TABLE IF NOT EXISTS clip_descriptions (
    clip_id        BIGINT NOT NULL REFERENCES clips(id) ON DELETE CASCADE,
    prompt_version TEXT NOT NULL,
    model          TEXT NOT NULL,
    happens        TEXT NOT NULL DEFAULT '',      -- what happens, start to end
    poster_meant   TEXT NOT NULL DEFAULT '',      -- what the poster uses it to say
    send_when      TEXT[] NOT NULL DEFAULT '{}',  -- situations to send it in, as people search
    confidence     TEXT,                          -- high / medium / low, the model's own
    answer         JSONB NOT NULL,                -- the full reply, as returned
    state          JSONB NOT NULL,                -- the text it was given (clip_state.build)
    strip          TEXT,                          -- storage key of the frame strip it saw
    usage          JSONB,
    cost           REAL,                          -- USD, as the gateway reported it
    request        JSONB,                         -- the exact request sent (image -> its storage key)
    reply          JSONB,                         -- the gateway's whole reply, as received
    at             TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (clip_id, prompt_version)
);
ALTER TABLE clip_descriptions ADD COLUMN IF NOT EXISTS request JSONB;
ALTER TABLE clip_descriptions ADD COLUMN IF NOT EXISTS reply JSONB;

-- Situation lane (app/uses.py): one row per "send it when" line of a clip, with its own
-- meaning vector. A search scores a clip by its best line, so one clip can serve many
-- situations without any of them blurring the others.
CREATE TABLE IF NOT EXISTS clip_uses (
    id        BIGSERIAL PRIMARY KEY,
    clip_id   BIGINT NOT NULL REFERENCES clips(id) ON DELETE CASCADE,
    line      TEXT NOT NULL,
    source    TEXT NOT NULL,                 -- "send_when" | "poster_meant" (from clip_descriptions) | "edited"
    embedding vector({EMBED_DIM}) NOT NULL
);
CREATE INDEX IF NOT EXISTS clip_uses_clip ON clip_uses (clip_id);

-- An answer replaced by `describe --redo`: moved here first, so nothing paid for is lost.
CREATE TABLE IF NOT EXISTS clip_descriptions_history (
    clip_id        BIGINT NOT NULL REFERENCES clips(id) ON DELETE CASCADE,
    prompt_version TEXT NOT NULL,
    model          TEXT NOT NULL,
    happens        TEXT NOT NULL DEFAULT '',
    poster_meant   TEXT NOT NULL DEFAULT '',
    send_when      TEXT[] NOT NULL DEFAULT '{}',
    confidence     TEXT,
    answer         JSONB NOT NULL,
    state          JSONB NOT NULL,
    strip          TEXT,
    usage          JSONB,
    cost           REAL,
    request        JSONB,
    reply          JSONB,
    at             TIMESTAMPTZ NOT NULL,          -- when the replaced answer was written
    replaced_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS clip_descriptions_history_clip ON clip_descriptions_history (clip_id);

-- A reply that came back (and was charged) but could not be read: empty, cut off, not JSON.
CREATE TABLE IF NOT EXISTS clip_description_failures (
    id             BIGSERIAL PRIMARY KEY,
    clip_id        BIGINT NOT NULL REFERENCES clips(id) ON DELETE CASCADE,
    prompt_version TEXT NOT NULL,
    model          TEXT NOT NULL,
    error          TEXT NOT NULL,
    request        JSONB,
    reply          JSONB,
    cost           REAL,
    at             TIMESTAMPTZ NOT NULL DEFAULT now()
);
-- best-line candidates come from this index instead of a scan of every line
CREATE INDEX IF NOT EXISTS clip_uses_embedding_hnsw ON clip_uses USING hnsw (embedding vector_cosine_ops);

-- English and Hindi (Devanagari) versions of each situation line (app/translate.py): the
-- meaning model cannot read romanised Hindi, so every line is also embedded in these.
CREATE TABLE IF NOT EXISTS line_translations (
    line  TEXT PRIMARY KEY,
    en    TEXT NOT NULL DEFAULT '',
    hi    TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL,
    at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Suggestions shown in the empty search box, one at a time (app/suggestions.py). Edit here
-- (active, position) and the site picks the change up within SUGGESTIONS_TTL_S.
CREATE TABLE IF NOT EXISTS search_suggestions (
    id         BIGSERIAL PRIMARY KEY,
    text       TEXT NOT NULL UNIQUE,
    position   INT NOT NULL DEFAULT 0,
    active     BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
-- first set: situations the clips' "send it when" lines answer (checked on the live library, 2026-10-21)
INSERT INTO search_suggestions (text, position) VALUES
    ('nobody takes me seriously', 1), ('nothing makes any sense', 2), ('sorry I messed up', 3),
    ('friend ignores the video I sent', 4), ('people talking nonsense', 5), ('the public will never forgive you', 6),
    ('brother always on his phone', 7), ('mom calls while I''m busy working', 8), ('last minute energy before exams', 9),
    ('exams are here and I haven''t studied', 10), ('everyone got in trouble except me', 11),
    ('giving advice but doing nothing himself', 12), ('friend won''t return the money he owes', 13),
    ('you ruined my mood', 14), ('my life is full of sadness', 15), ('friend invites you and makes you pay the bill', 16),
    ('his old prediction came true', 17), ('marriage proposal talks started at home', 18),
    ('laptop battery died at the worst time', 19), ('is there any other way out', 20)
ON CONFLICT (text) DO NOTHING;
