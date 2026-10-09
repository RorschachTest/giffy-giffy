"""Which speech-to-text setup hears Hindi / Hinglish best? Side by side, no DB writes.

    python -m app.compare_stt [--setups a,b] [--clips id,share_id] [--out path]

Every setup transcribes the same audio. Prints each clip's lines next to each other
with the model's own confidence and the seconds taken, and writes
data/stt_compare.json for a closer look. Judge by listening: confidence numbers are
the model grading itself.
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from functools import lru_cache
from pathlib import Path

from . import config, db, media, storage
from .text import has_devanagari, to_roman

# A mixed-script sample: Whisper continues in the style of its prompt, so English
# words said inside Hindi come out in Latin letters. Kept generic so nothing in it is
# likely to be copied into a transcript.
STYLE = "हाँ, मैं sure हूँ, ये बात actually सही है।"

# name -> (model, language, use a hint prompt, prefix the style sample)
SETUPS: dict[str, tuple[str, str | None, bool, bool]] = {
    "small-hi":        ("small", "hi", False, False),      # what ran before for sidecar-tagged Hindi clips
    "large-hi":        ("large-v3", "hi", False, False),
    "large-auto":      ("large-v3", None, False, False),   # Hinglish: let it pick, English words stay English
    "large-hi-hint":   ("large-v3", "hi", True, False),    # title + names as a hint for spellings
    "large-hi-style":  ("large-v3", "hi", True, True),     # style sample + hint
}


@lru_cache(maxsize=1)   # one model in memory at a time: large-v3 next to anything else can run out
def _model(name: str):
    from faster_whisper import WhisperModel
    return WhisperModel(name, device=config.WHISPER_DEVICE, compute_type=config.WHISPER_COMPUTE,
                        download_root=str(config.MODELS_DIR / "whisper"))


def hint(clip: dict) -> str:
    """A short prompt that nudges spellings (names, the film) without dictating content."""
    parts = [clip.get("title") or "", ", ".join(clip.get("people") or []), clip.get("source_title") or ""]
    text = ". ".join(p for p in parts if p.strip())
    return f"{text}." if text else ""


def run(setup: str, wav: Path, clip: dict) -> dict:
    model, language, use_hint, use_style = SETUPS[setup]
    prompt = " ".join(p for p in ((STYLE if use_style else ""), (hint(clip) if use_hint else "")) if p)
    t0 = time.time()
    segments, info = _model(model).transcribe(
        str(wav), language=language, beam_size=5, vad_filter=True, condition_on_previous_text=False,
        initial_prompt=prompt or None, temperature=[0.0, 0.2, 0.4], best_of=1)
    segs = list(segments)
    text = " ".join(s.text.strip() for s in segs).strip()
    return {"text": text, "roman": to_roman(text) if has_devanagari(text) else text,
            "language": info.language, "language_probability": round(float(info.language_probability), 3),
            "avg_logprob": round(sum(s.avg_logprob for s in segs) / len(segs), 3) if segs else None,
            "seconds": round(time.time() - t0, 1)}


def _arg(name: str) -> str | None:
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv and sys.argv.index(name) + 1 < len(sys.argv) else None


def main() -> int:
    setups = [s for s in (_arg("--setups") or ",".join(SETUPS)).split(",") if s]
    unknown = [s for s in setups if s not in SETUPS]
    if unknown:
        print(f"unknown setups {unknown}; choose from {list(SETUPS)}", file=sys.stderr)
        return 1
    picked = [i.strip() for i in (_arg("--clips") or "").split(",") if i.strip()]
    out_path = Path(_arg("--out") or config.DATA_DIR / "stt_compare.json")
    with db.session() as conn:
        rows = conn.execute("""SELECT id, left(sha256, 10) AS share_id, file, title, people, source_title,
                                      has_audio, transcript_roman FROM clips ORDER BY id""").fetchall()
    rows = [r for r in rows if r["has_audio"] and (not picked or str(r["id"]) in picked or r["share_id"] in picked)]
    print(f"setups: {', '.join(setups)}   clips: {len(rows)}\n", flush=True)

    # one setup at a time over every clip, so each model is loaded once and then released
    entries = {r["id"]: {"id": r["id"], "share_id": r["share_id"], "title": r["title"],
                         "stored": r["transcript_roman"], "setups": {}} for r in rows}
    with tempfile.TemporaryDirectory() as tmp:
        wavs = {}
        for r in rows:
            with storage.local_copy(r["file"]) as clip_file:
                wavs[r["id"]] = media.extract_audio(clip_file, Path(tmp) / f"{r['id']}.wav")
        for s in setups:
            print(f"== {s}", flush=True)
            for r in rows:
                try:
                    out = run(s, wavs[r["id"]], r)
                except Exception as exc:   # one bad clip or setup must not stop the comparison
                    out = {"error": str(exc)}
                entries[r["id"]]["setups"][s] = out
                print(f"  [{r['id']}] " + (f"FAILED: {out['error']}" if "error" in out else
                      f"[{out['language']} {out['language_probability']:.2f} lp {out['avg_logprob']} {out['seconds']}s] "
                      f"{out['roman'][:120]}"), flush=True)
    results = list(entries.values())
    for e in results:
        print(f"\n[{e['id']}] {e['title']}   /c/{e['share_id']}")
        for s, out in e["setups"].items():
            print(f"  {s:<14} {out.get('roman', 'FAILED: ' + str(out.get('error')))[:200]}")
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
