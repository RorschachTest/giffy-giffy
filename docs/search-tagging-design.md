# Tagging clips with Laya so search is easy — design

Status: design only, nothing here is built yet. Produced by comparing five
independent designs, scoring them on three lenses (search quality, cost and
operations, buildability), then a critic pass against the actual code. Every
number marked *guess* is a starting value to be replaced by measurement.

## The answer in one paragraph

Store each clip as a small **tag card**: for each of a few facets — emotion,
topic, *what the sender is saying* (reply), *which moment it suits* (event),
*who is on screen* (role), and *why it is funny* — keep a probability
distribution plus how sure we were, not a bare label. Laya fills the card at
tagging time by choosing from short closed lists. Its raw answers are kept
apart from the derived tags, so every threshold can be retuned without asking
Laya again. Search reads the query onto the **same facets** (always with a local
embedding method, plus Laya when it answers in time) and adds a facet-match score
that only fills the gap the words leave, so an exact line or name always wins.
Laya never ranks videos and is never needed for search to work. The build
starts with **a ruler** (a fixed set of test queries with known answers), and
each Laya facet is kept only if it beats a Laya-free baseline on hand-labelled
clips.

## What shaped the design

| Fact | Where it comes from | Consequence |
|---|---|---|
| Laya's base checkpoint scores **0.36** zero-shot on typed decisions, **0.77** after fine-tuning on the user's own domain; checkpoints are **over-confident**, and the multilingual one ships with **no fitted calibration** | [Laya README](https://github.com/NandhaKishorM/laya) | Never trust a raw Laya probability. Pilot every facet on ~100 hand-labelled clips against a no-Laya baseline; plan for fine-tuning on our own labels |
| A question's options share a **256-token budget** on the multilingual checkpoint (~20 short options before trimming, then a 422) | Laya README | Option lists ≤ 17, descriptions ≤ 6 words; raise `head_max_len` to 384 per request where needed; a test sends every question to the real Laya and checks nothing is trimmed |
| **Option order changes answers** (averaging over orders cut order-dependent answers from 16% to 6%) | Laya README | Measure position bias first; add rotations only if it is large |
| One choice over many options gives a clean distribution; separate yes/no answers are not comparable | `app/laya.py` docstring (measured here) | Choice questions only |
| Many clips are **bare reaction templates** (no caption) and **Hindi transcripts are noisy** | Pipeline, current sample | Laya (text-only) can't read them; face/scene mood, shares and a human label queue must carry them. A facet may stay empty |
| `embedding_text` and `s_meta` **already contain** moods, use-when and topics | `app/indexing.py` | Emotion/topic tags may add little over today; the gain should come from reply/event/role. Measure against a stripped-text baseline |
| `search()` and `api.search()` both call `laya.query_intent`; `lru_cache` does not cache failures, so a hung Laya costs **2 × 3 s** per search; the trigram candidate branch is `LIMIT 500` with **no ORDER BY**; `s_learned` grows unbounded | `app/search.py`, `app/api.py`, `app/indexing.py` | Fix these first, whatever else is built |

## The five ideas compared

| Design | Idea | Score (3 judges, /30) | What survived |
|---|---|---|---|
| **TagCard** | Laya fills a faceted tag card per clip at ingest; query read onto the same facets | 20.5 | **The spine**: raw answers kept apart from derived tags, `w = p × sureness`, two ways to read the query, facets only fill the gap words leave |
| **Small Tags, Big Ruler** | Smallest upgrade; build the evaluation first; fix the hot-path bugs | 20.5 | **The build order**: ruler first, ship rule written in advance, every piece behind a flag |
| **Reply Cards** | Index the conversation: each clip as *situation it answers × meaning it sends* | 19.0 | Tagging as a separate background stage, most-shared first; a Laya-free backfill; one-question re-asks after a taxonomy change; a hand-written Hinglish phrase bank |
| **Lens Atlas** | Several purpose-built vectors per clip | 16.5 | A small table of real past queries with embeddings; a "why it matched" note on every result |
| **Question Ledger** | Store the questions each clip answers; Laya picks among the top 8 at query time | 15.5 | Append-only share log; other uploaders' titles kept as alternative titles; fail-fast around Laya |

Rejected on purpose: Laya reranking candidates on the live search path (8 clip
descriptions don't fit the option budget, the base model is near chance, and 4
CPU threads can't do it in under a second); a separate vector database;
generating "anticipated queries" (that would need a text generator).

## What is stored per clip

Facets are grouped by **independent evidence**: labels that restate each other
(`topic.sarcasm` / `reply.sarcasm`, `topic.cringe` / `reply.awkward`) sit in the
same group, and matching counts groups, so one coincidence can't make a match.

| Facet | Labels | Filled by | Group | Kokilaben example *(illustrative)* |
|---|---|---|---|---|
| **emotion** — what the clip shows | 13 (`vibe.MOODS`), multi-label | FER+ faces, CLIP scene, Laya text mood (as today); sureness from how many sources agree, not a fixed 1.0 | EMO | angry 0.46, shocked 0.31, sure 0.8 |
| **topic** | 19 (`laya.TOPICS`) | existing Laya readings, now stored in full | SEM | family 0.61, sure 0.62 |
| **reply** — what the sender is saying | 16 (`intent.INTENTS`) + none | Laya on two views (metadata only / metadata + dialogue), mixed with the existing intent engine; a human label overrides | SEM | call_out 0.57, sure 0.84 |
| **aim** — who it's aimed at | self / recipient / message / situation | derived from reply, no Laya call | SEM | recipient |
| **event** — which moment it suits | 7: caught, excuse, request, showoff, scolding, fight, reveal | Laya, two views | SEM | caught 0.77, sure 0.93 |
| **role** — who is on screen | 12: elder woman, elder man, boss, teacher, politician, celebrity, sportsperson, kid, couple, friends, villain, reporter | Laya | WHO | elder woman 0.88 |
| **funny** — why it's funny | 10: twist, absurd, deadpan, putdown, naughty, relief, relatable, over_the_top, self_roast, reference | one Laya question + the 12 mechanisms in `humor.py` mapped 1:1 | FUN | reference, twist |

Plus per clip: `stt_trust` (from Whisper's own confidence numbers; 1.0 for
typed transcripts), a phonetic key `s_phon` for misheard Hindi lines, `funny`
(score) and `funny_why` (the reason sentences, for the result card), and
`human` (a person's edits, kept separate so they can always be told apart from
Laya's).

**Sureness.** Laya is right when sure and noise when unsure, so each tag's
weight is `w = p × sureness`, where `sureness = clamp((p_top − 0.30) / 0.50, 0, 1)`
*(guess)*. An unsure answer gets weight ≈ 0, so it neither helps nor hurts any
query. "none" on top makes the facet abstain. Once ~100 clips are hand-labelled
this ramp is replaced by a fitted map from Laya's confidence to its measured
accuracy (or by Laya's own calibration after fine-tuning).

## How Laya is used in tagging

Every question is a single choice over a short list. The state Laya reads comes
in two **views**: *meta* (title, caption, folk names, people, film, hashtags,
first comments) and *full* (meta + dialogue). The dialogue is left out when
`stt_trust < 0.25`, so a garbled transcript cannot pull a good title off course.
The two views are pooled by their own sureness.

| Question | Options | Views | Cost |
|---|---|---|---|
| topic + mood | existing (20 + 14) | as today | 0 new — just store the full answers |
| **reply**: "If someone sent this clip as a reply in a chat, what would it be saying?" | 16 intents + none, ≤ 6-word descriptions | meta + full | +2 |
| **event**: "Which moment does this clip show or suit?" | 7 + none | meta + full | +2 |
| **role**: "Who is the main person in this clip?" | 12 + none | full | +1–2 |
| **funny**: "Why would this clip be funny when sent as a reply?" | 10 + none | full | +1 |

Today's tagging asks Laya 3 questions per clip; with every facet on it's about
9–10, in 3 requests, in a **separate background process** that runs
most-shared clips first and stays out of the way of new uploads. A clip is
searchable the moment it's ingested; its tag card fills in afterwards.

Raw answers go in `clip_laya` (the full probability map per view and question,
keyed by a hash of the state sent, so an unchanged question is never re-asked).
Derived tags go in `clip_tag` and are rebuilt by a pure function with **zero
Laya calls** — so changing a threshold, a weight or a label mapping is a
reindex, not a re-tag. Changing one question's wording re-asks only that
question.

**Pilot before bulk.** Each new facet is first asked on ~100 hand-labelled
clips and compared with (a) a Laya-free embedding baseline, (b) the existing
intent engine, (c) always guessing the most common label. A facet where Laya
doesn't clearly win stays Laya-free. If Laya loses, the next step is
fine-tuning it on 300–500 of our own labels (the README's 0.36 → 0.77 route).

## How search uses it

1. **Names and exact lines short-circuit.** If the query is a clear trigram hit
   on a title, line or person, skip everything below — no Laya call.
2. **Read the query onto the facets two ways.** Lane 1 is always on and takes
   ~5 ms: compare the query embedding with label example phrases (the existing
   intent/mood/topic phrases plus ~115 hand-written English and Hinglish ones
   like "kisne kiya", "caught red handed"). Lane 2 is the existing Laya query
   call, started first, called once, and awaited for at most ~350 ms
   *(guess, set from measurement)*. It's skipped for 30 s after repeated failures.
   The two lanes start with equal weight.
3. **Facet match** `F`: for each facet, the overlap between the query's
   distribution and the clip's (a small in-memory matrix: 100k clips × ~80
   labels ≈ 32 MB, a few ms). Grouped as above. A clip counts as a tag match
   only if enough independent groups agree.
4. **Score**:
   `kw·Wk + sem·Ws + phrases·Wph·(1−kw) + F·Wf·(1−kw) + popularity`.
   The `(1 − kw)` means tags only fill the gap the words leave: "kya karu main"
   names one clip, and a different clip with the same mood cannot overtake it.
5. **Pasted chat messages** (from the extension, or a "reply to this" box): the
   message is classified into a dialogue act and pushed through the existing
   act → intent table in `intent.py` to get the *wanted* reply, then the top 5 are
   reranked with `intent.read`.
6. **Noisy Hindi lines**: a phonetic consonant-skeleton key (`rasode me kon tha`
   ≈ `rasoday mein kaun tha`) matched by trigram, and the transcript's weight
   scaled by `stt_trust`.
7. Every result says why it matched: `reply: call out · event: caught · mood: angry`.

With tags off, the matrix empty or Laya down, `F = 0` and search is exactly
today's — with the bug fixes.

## The three layers you asked for

- **Emotion** — already computed (`vibe.py`); now stored with an honest
  sureness. A clip with no face and no speech, scored by CLIP alone, no longer
  counts as fully trusted. Queries like "boss shouting at me" match both the angry
  boss and the cowering victim (a 60/40 split, *guess*).
- **What the sender is saying** — the 16 intents from `intent.py`, now *stored*
  per clip instead of computed on the fly: Laya's reply answer mixed with the
  engine's reading, then corrected by how people actually share the clip. The
  same stored distribution feeds `/understand`, so search and understanding
  agree.
- **Why it's funny** — the 12 mechanisms in `humor.py` mapped to 10 plain
  labels, plus one Laya question. Mostly an explanation for the result card.
  It only affects ranking on style queries ("savage", "relatable", "dramatic",
  "cringe"). A bug this exposed (not fixed yet): on a bare template
  `humor.facts_from_clip` uses the auto-generated `use_when` mood sentence as
  the "setup", which makes surprise/absurdity fire on nothing. Tagging must use
  only a real caption or a human-written use-when as the setup.

## How it learns (no LLM, no training run)

- **Shares are votes.** A share adds the query's facet reading to that clip's
  counts, at most 3 votes per (clip, query). Counts blend with Laya's tags in
  one place: `w' = (n + 4·w) / (N + 4)`. A share counts only a quarter when the
  clip was found *only* through its tags, so a wrong tag can't vote for
  itself.
- **Human labels** from a one-click chip queue (top predicted labels per facet,
  most-shared clips first) become the gold set and, later, Laya's fine-tuning
  data. Hand-tagging the top ~50 bare templates takes about an hour and fixes
  exactly the clips Laya can't read.
- **Past queries as metadata** (later, once there are ~1k shares): real
  queries that led to a share, other uploaders' titles, and chat messages from
  feedback become embedded phrases, so a new phrasing matches an old one by
  meaning.
- **New labels are added by a person** from a report of queries nothing
  matched. Nothing invents labels automatically.

## Build order (each stage ships behind a flag and is measured)

| Stage | What | Gate to continue |
|---|---|---|
| **0** (days) | **The ruler**: ~150 hand-written queries by kind (quote, name, situation, message, mood, unknown), plus hundreds of free test queries made by using a clip's own caption as the query with that caption hidden; hit@1/5/10, MRR. **Fix** the double Laya call, the uncached failures, the unordered `LIMIT 500`, the unbounded `s_learned`. **Phonetic key** + `stt_trust`. Start keeping human edits in `clips.human`. A probe of Laya's real latency, truncation and order bias | Baselines saved with Laya off and on |
| **1** (~1 week) | Laya-free tag card from data already stored (emotion, topic, engine reply, aim) for every clip; facet matrix; `F` replaces today's topic boost behind a flag; why-chips; chip UI | hit@5 on situation/message/mood queries up ~8 points, ≤ 1 point lost on quotes/names |
| **2** (~1 week) | Laya reply/event/role/funny questions — **pilot on ~100 gold clips first**, then bulk-tag only the facets that won, in the background worker | Laya beats the Laya-free baseline on that facet *and* lifts held-out search |
| **3** (after ~1k shares) | Share votes, embedded past-query phrases, pasted-message handling with rerank | Each piece measured on its own |

## Deliberately not built (and what would change that)

- Laya ranking videos at search time — only as an offline experiment.
- A separate vector database or one vector per field — revisit past ~300k clips.
- A pgvector table for tags — an in-memory matrix is enough to ~300k clips.
- Generated "anticipated queries" — real queries arrive through shares.
- Writing tags back into `topics`/`reactions` or into the lexical search columns
  — derived values stay derived; human facts stay distinguishable.
- Hierarchical (group → leaf) questions — 17 flat short options fit; add only
  if the probe shows trimming.

## What must be measured before trusting any of this

1. Does Laya's multilingual checkpoint beat a plain embedding baseline on
   reply/event/role/funny for Hinglish clip text? If not, does fine-tuning on
   300–500 of our labels close the gap?
2. Real Laya latency on this box, per question and per option; is a ~350 ms query
   budget ever met?
3. What share of clips have any usable text at all? If under ~40%, Laya's part
   is capped, and shares and human labels carry the rest.
4. Do emotion and topic tags add anything over the text that already contains
   those words?
5. Does the phonetic key help misheard quotes or cause collisions (kon / kaun /
   kahan)?
6. Every constant above marked *guess*.
