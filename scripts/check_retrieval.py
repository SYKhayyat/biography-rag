"""Ask the same questions of several corpora under two retrieval settings, and print the answers
side by side. This is how the retrieval defaults were chosen (see the README's "Measured" table):
a change that sounds better has to answer more questions, or refuse the canaries, on corpora it
was not tuned on.

    uv run python scripts/fetch_corpus.py && uv run python scripts/fetch_extra_corpus.py
    uv run python scripts/check_retrieval.py --provider gemini        # free tier, local embeddings

Each corpus gets three questions whose answers are in the books and one canary whose answer is
not, so "answered" and "refused" are both scored. Cheap by design: ``--embedder local`` costs
nothing, ``--provider`` accepts any configured provider (gemini, nvidia, groq and cheapllm have
free tiers), and the LLM judge is off, since the retrieval comparison does not need it.

Results are written to logs/checks/<corpus>-<provider>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

from biography_rag.config import Settings
from biography_rag.domain import CorpusStatus
from biography_rag.rag.retriever import RetrievalConfig
from biography_rag.services import StoryService, UploadedFile
from biography_rag.web.container import build_container

# (corpus folder, person, [(question, words that must appear in a correct answer | None = canary)])
CORPORA: dict[str, tuple[str, str, list[tuple[str, list[str] | None]]]] = {
    "franklin": (
        "corpus/raw",
        "Benjamin Franklin",
        [
            ("Where and when was Benjamin Franklin born?", ["boston"]),
            ("Who did Benjamin Franklin marry?", ["read", "deborah"]),
            ("Who was the older brother Benjamin Franklin was apprenticed to?", ["james"]),
            ("Whose portrait is on the United States hundred-dollar bill?", None),
        ],
    ),
    "douglass": (
        "corpus/raw-douglass",
        "Frederick Douglass",
        [
            ("Where was Frederick Douglass born?", ["tuckahoe"]),
            ("What was the name of Frederick Douglass's wife?", ["anna", "pitts"]),
            ("Who first taught Frederick Douglass the alphabet?", ["mistress", "sophia", "auld"]),
            ("When did the Frederick Douglass Memorial Bridge open to traffic?", None),
        ],
    ),
    "lincoln": (
        "corpus/raw-lincoln",
        "Abraham Lincoln",
        [
            ("Where was Abraham Lincoln born?", ["kentucky", "hodgen"]),
            ("Who did Abraham Lincoln marry?", ["mary", "todd"]),
            ("Where was Abraham Lincoln assassinated?", ["ford", "theatre", "theater"]),
            ("What is shown on the reverse of the Lincoln cent first issued in 1959?", None),
        ],
    ),
}

SETTINGS = {
    "old": dict(min_similarity=0.0, max_per_doc=0, neighbour_radius=0),  # before the review
    "new": dict(min_similarity=0.25, max_per_doc=0, neighbour_radius=1, seam_chars=240),  # defaults
}


def run_corpus(name: str, folder: str, person: str, checks, args, out: Path) -> dict:
    cached = out / f"{name}-{args.provider}.json"
    if cached.exists() and not args.refresh:
        print(f"{name}: using {cached} (--refresh to run again)", flush=True)
        return json.loads(cached.read_text())
    files = [UploadedFile(p.name, p.read_bytes()) for p in sorted(Path(folder).glob("*.txt"))]
    if not files:
        sys.exit(f"{folder} is empty: run scripts/fetch_extra_corpus.py")
    settings = Settings(
        log_dir=str(out / "logs" / name),
        embedder=args.embedder,
        llm_provider=args.provider,
        verify_with_llm=False,
        database_url="",
    )
    container = build_container(settings)
    try:
        corpus = container.ingest.create(person, files)
        started = time.monotonic()
        corpus = container.ingest.ingest(corpus.corpus_id, files)
        if corpus.status is not CorpusStatus.READY:
            sys.exit(f"{name}: ingest failed: {corpus.error}")
        print(
            f"\n=== {name}: {corpus.chunk_count} passages from {len(files)} books "
            f"({time.monotonic() - started:.0f}s) ===",
            flush=True,
        )
        rows: dict[str, dict] = {}
        for label, overrides in SETTINGS.items():
            story = StoryService(
                container.corpora,
                container.index,
                container.runs,
                container.embedders,
                container.chats,
                container.story._log,
                replace(container.story._config, retrieval=RetrievalConfig(**overrides)),
            )
            for n, (question, expect) in enumerate(checks, 1):
                run = story.execute(story.start_ask(corpus.corpus_id, question, args.provider).run_id)
                answer = run.answers[0]
                text = (answer.answer or "").lower()
                rows.setdefault(str(n), {})[label] = {
                    "question": question,
                    "answer": answer.answer,
                    "reason": answer.reason,
                    "keywords_found": None if expect is None else any(e in text for e in expect),
                    "refused": answer.answer is None,
                    "cited": answer.cited_ids,
                    "passages": len(answer.retrieved_ids),
                }
                state = "refused" if answer.answer is None else "answered"
                print(
                    f"  [{label}] Q{n} {state:8} {len(answer.retrieved_ids)} passages | "
                    f"{(answer.answer or answer.reason or '')[:70]!r}",
                    flush=True,
                )
        result = {"passages": corpus.chunk_count, "books": len(files), "rows": rows}
        cached.write_text(json.dumps(result, indent=2))
        return result
    finally:
        container.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider", default="gemini", help="a provider with a key set (default gemini)")
    parser.add_argument("--embedder", default="local", help="local (free) or openai")
    parser.add_argument("--corpus", action="append", choices=sorted(CORPORA), help="only these corpora")
    parser.add_argument("--refresh", action="store_true", help="ignore cached results")
    parser.add_argument("--out", type=Path, default=Path("logs/checks"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    results = {}
    for name in args.corpus or list(CORPORA):
        folder, person, checks = CORPORA[name]
        results[name] = run_corpus(name, folder, person, checks, args, args.out)

    print(f"\n{'corpus':10} {'setting':8} answered  canary-refused  correct")
    for name, data in results.items():
        for label in SETTINGS:
            answered = sum(1 for r in data["rows"].values() if not r[label]["refused"])
            correct = sum(
                1
                for n, r in data["rows"].items()
                if (r[label]["refused"] if r[label]["keywords_found"] is None else r[label]["keywords_found"])
            )
            canary = "yes" if data["rows"]["4"][label]["refused"] else "NO"
            print(f"{name:10} {label:8} {answered}/4      {canary:14}  {correct}/4")
    print("\n'correct' matches answer wording against a keyword list; read the answers, it is not a judge.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
