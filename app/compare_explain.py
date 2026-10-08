"""Run every explain backend over the stored clips and print them side by side.

    python -m app.compare_explain                       all backends available on this branch
    python -m app.compare_explain --backends claude     just one
    python -m app.compare_explain --clips 5,6,8         only these clip ids
    python -m app.compare_explain --out data/x.json     where the full results go

Backends see exactly the same prompt, frames and transcript, so differences are
the model's. Nothing is written to the database. Two extra columns come along when
they exist: what a person wrote in the Edit form (explain_source manual/sidecar) and
hand-written references in DATA_DIR/explain_reference.json:

    {"d28f9c7a36": {"gist": "...", "why_funny": "..."}, "6": {...}}     (a share id or a clip id)
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from . import config, db, explain, media


def _arg(name: str) -> str | None:
    if name in sys.argv:
        i = sys.argv.index(name)
        return sys.argv[i + 1] if i + 1 < len(sys.argv) else None
    return None


def _wrap(text: str, width: int = 84, indent: str = "      ") -> str:
    words, lines, line = (text or "-").split(), [], ""
    for w in words:
        if len(line) + len(w) + 1 > width:
            lines.append(line); line = w
        else:
            line = f"{line} {w}".strip()
    lines.append(line)
    return ("\n" + indent).join(lines)


def main() -> int:
    wanted = (_arg("--backends") or "").split(",") if _arg("--backends") else explain.available_backends()
    wanted = [w for w in wanted if w]
    picked = [i.strip() for i in (_arg("--clips") or "").split(",") if i.strip()]   # DB ids or share ids
    out_path = Path(_arg("--out") or config.DATA_DIR / "explain_compare.json")
    ref_path = config.DATA_DIR / "explain_reference.json"
    references = json.loads(ref_path.read_text(encoding="utf-8")) if ref_path.exists() else {}

    with db.session() as conn:
        rows = conn.execute(
            """SELECT id, left(sha256, 10) AS share_id, file, duration, title, language, transcript_native,
                      transcript_roman, gist, why_funny, send_when, explain_source
                 FROM clips ORDER BY id""").fetchall()
    rows = [r for r in rows if not picked or str(r["id"]) in picked or r["share_id"] in picked]
    if not wanted:
        print("no explain backend is available on this branch", file=sys.stderr)
        return 1
    for name in wanted:
        ok, reason = explain.ready(name)
        if not ok:
            print(f"note: {name} is not ready: {reason}", file=sys.stderr)
    print(f"backends: {', '.join(wanted)}   clips: {len(rows)}\n")

    results = []
    for r in rows:
        entry = {"id": r["id"], "share_id": r["share_id"], "title": r["title"],
                 "transcript": r["transcript_roman"], "backends": {}}
        ref = references.get(r["share_id"]) or references.get(str(r["id"]))
        if ref:
            entry["reference"] = ref
        if r["explain_source"] in explain.HUMAN_SOURCES:
            entry["human"] = {k: r[k] for k in ("gist", "why_funny", "send_when")}
        print(f"[{r['id']}] {r['title']}   /c/{r['share_id']}")
        print(f"    said: {_wrap(r['transcript_roman'][:200], indent='          ')}")
        with tempfile.TemporaryDirectory(prefix="memeclip_") as tmp:
            frames = media.extract_keyframes(config.MEDIA_DIR / r["file"], Path(tmp), r["duration"],
                                             config.EXPLAIN_FRAMES)
            for name in wanted:
                try:
                    res = explain.run(name, frames, r["title"], r["transcript_native"],
                                      r["transcript_roman"], r["language"])
                    entry["backends"][name] = {k: res[k] for k in ("gist", "why_funny", "send_when", "seconds", "usage")}
                    tokens = res["usage"].get("input_tokens"), res["usage"].get("output_tokens")
                    used = f"  [{tokens[0]} in / {tokens[1]} out tokens]" if tokens[0] is not None else ""
                    print(f"  {name:<7} {res['seconds']:>5}s{used}  {_wrap(res['gist'])}")
                    print(f"          why: {_wrap(res['why_funny'], indent='               ')}")
                    print(f"          send: {'; '.join(res['send_when']) or '-'}")
                except Exception as exc:
                    entry["backends"][name] = {"error": str(exc)}
                    print(f"  {name:<7} FAILED: {exc}")
        if ref:
            print(f"  ref             {_wrap(ref.get('gist', ''))}")
            print(f"          why: {_wrap(ref.get('why_funny', ''), indent='               ')}")
        if "human" in entry:
            print(f"  human           {_wrap(entry['human']['gist'])}")
        print()
        results.append(entry)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"backends": wanted, "clips": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
