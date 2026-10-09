# Handoff: meme understanding + Laya/Jev tagging (from cloud session, 2026-10-09)

Branch `claude/intelligent-volta-3wglpu`, PR https://github.com/RorschachTest/giffy-giffy/pull/2

## What exists on this branch
- `app/humor.py`, `app/intent.py`, `app/semspace.py`, `app/understand.py`: a non-LLM engine that reads
  what a meme says as a reply (16 intents, from the chat context) and why it is funny (12 humor
  mechanisms). API: `POST /understand`, `GET /clips/{id}/humor`, `POST /understand/feedback`,
  `GET /understand/stats`; CLI `python -m app.understand`. Learns from feedback by counting
  (Dirichlet), no model training. Table `meme_feedback`.
- `docs/meme-understanding.md`: research behind it (MUStARD, UR-FUNNY, MemeCap, Memotion, SICKNet ...).
- `docs/search-tagging-design.md`: the full design for tagging clips so search is easy. **Design only, not built.**
- Tests: 74 pass with a local Postgres+pgvector (`TEST_DATABASE_URL=... pytest`). Never run with the
  real models (Hugging Face was blocked in the cloud sandbox), so reading quality is unmeasured.

## Decisions so far
1. **Tag card per clip**: emotion, topic, reply ("what the sender is saying"), aim, event, role,
   funny ("why it's funny"), each stored as probabilities × sureness. Raw model answers are kept apart
   from derived tags, so retuning never re-asks the model. Search reads the query onto the same tags and
   adds a tag score that only fills the gap the words leave (exact lines/names always win).
2. **The tagging model is never on the search path.** Search works with it down.
3. **Measure before building**: Stage 0 = ~150 test queries with expected clips (the user writes these),
   fix the search bugs, log shares, add a one-click labelling page.
4. **Laya as-is is weak for this**: `laya-multilingual` scores 0.352 zero-shot on Laya's own
   typed-decisions benchmark (random 0.318, majority 0.461). Fine-tuned `laya-typed-decisions` scores
   0.766 vs Jev 0.727, but it was trained on that benchmark's train split. Jev wins on long option
   lists (Banking77 0.870 vs 0.425). Source: the Laya README.
5. **Jev (TypeSafe) is the likely tagger**: same kind of model as Laya (typed decisions, no text
   generation), hosted, decent zero-shot. Pay-per-use $0.042 per 1M input tokens, output free, no
   premium plan from TypeSafe (the $29–$98/month plans are third-party resellers). About $13 to tag 100k
   clips. Downsides: clip text leaves the machine, needs internet and an API key. Not on the search path
   (0.3–0.9 s measured latency).
6. **Plan**: add Jev as a second backend next to Laya (all calls go through `laya.predict`); check both
   on ~100 hand-labelled clips against a no-model baseline; keep the winner per tag. Optionally fine-tune
   Laya later on human-checked labels (+ Jev labels if TypeSafe's terms allow) for a free, fast local model.
7. **Data**: fine-tuning needs a few thousand labelled decisions, not 100k clips. Start with 300–500
   hand labels. Sources: shares in our own app, the labelling page, Jev with spot checks, permissively
   licensed datasets. **Don't crawl GIPHY**: wrong content (silent, mostly English GIFs) and unchecked terms.
8. **V-JEPA (video model)**: possible later visual tagger (actions, template matching). Not now.

## Bugs found, not fixed yet
- `app/api.py` `search()` calls `laya.query_intent` again after `search.search` already did; `lru_cache`
  doesn't cache failures, so a hung Laya costs 2 × 3 s per search.
- `app/search.py` trigram candidate branch is `LIMIT 500` with no `ORDER BY`.
- `s_learned` (indexing) grows without limit.
- `humor.facts_from_clip` uses the auto-generated `use_when` as the joke setup on bare templates,
  so incongruity/absurdity fire on nothing.

## Open questions for the user
- Build Stage 0 now? It needs ~150 test queries with the clip each should find.
- OK to send clip text to TypeSafe (Jev)? Zero data retention needs their sales team.
- Did "giffy" mean GIPHY?
