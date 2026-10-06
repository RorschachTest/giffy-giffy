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

-- Every search, so you can see what people type and what finds nothing.
CREATE TABLE IF NOT EXISTS query_log (
    id          BIGSERIAL PRIMARY KEY,
    at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    query       TEXT NOT NULL,
    query_norm  TEXT NOT NULL,
    results     INT NOT NULL,
    top_score   REAL
);
