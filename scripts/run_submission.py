"""Produce the hand-in artefacts end to end, against whatever DATABASE_URL/.env points at.

    uv run python scripts/run_submission.py                # ingest corpus/raw/*.txt, story + canary
    uv run python scripts/run_submission.py --corpus ID    # reuse an already-ingested corpus

Writes to logs/submission/: story.md, story-run.json, story-run.jsonl, canary-run.json,
canary-run.jsonl and audit.md (the sentence-by-sentence trace table).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from biography_rag.config import Settings
from biography_rag.domain import CorpusStatus, RunStatus, StoryRun
from biography_rag.serialization import run_to_dict
from biography_rag.services import UploadedFile
from biography_rag.web.container import build_container

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "logs" / "submission"
CANARY = "Whose portrait is on the United States hundred-dollar bill?"


def ingest(container, person: str) -> str:
    files = [UploadedFile(p.name, p.read_bytes()) for p in sorted((ROOT / "corpus" / "raw").glob("*.txt"))]
    if not files:
        sys.exit("corpus/raw is empty: run scripts/fetch_corpus.py first")
    corpus = container.ingest.create(person, files)
    print(f"ingesting {len(files)} files as {corpus.corpus_id} ...", flush=True)
    t = time.monotonic()
    corpus = container.ingest.ingest(corpus.corpus_id, files)
    if corpus.status is not CorpusStatus.READY:
        sys.exit(f"ingest failed: {corpus.error}")
    print(f"  {corpus.chunk_count} chunks in {time.monotonic() - t:.0f}s", flush=True)
    return corpus.corpus_id


def export(container, run: StoryRun, name: str) -> None:
    (OUT / f"{name}.json").write_text(json.dumps(run_to_dict(run), indent=2, ensure_ascii=False))
    log = Path(container.settings.log_dir) / f"{run.run_id}.jsonl"
    shutil.copyfile(log, OUT / f"{name}.jsonl")


def audit_table(run: StoryRun) -> str:
    rows = ["| # | Verdict | Cited | Sentence | Note |", "|---|---|---|---|---|"]
    for n, c in enumerate(run.verification.checks if run.verification else [], 1):
        sentence = c.sentence.replace("|", "\\|")
        rows.append(f"| {n} | {c.verdict.value} | {', '.join(c.cited_ids)} | {sentence} | {c.note.replace('|', '/')} |")
    return "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--person", default="Benjamin Franklin")
    ap.add_argument("--corpus", help="reuse an existing corpus id instead of ingesting")
    ap.add_argument("--provider", default=None)
    args = ap.parse_args()

    container = build_container(Settings())
    try:
        corpus_id = args.corpus or ingest(container, args.person)
        OUT.mkdir(parents=True, exist_ok=True)

        print("running the 20 queries + weave + verification ...", flush=True)
        run = container.story.execute(container.story.start(corpus_id, args.provider).run_id)
        if run.status is not RunStatus.DONE:
            sys.exit(f"story run failed: {run.error}")
        export(container, run, "story-run")
        report = run.verification
        print(
            f"  story run {run.run_id}: {sum(a.found for a in run.answers)}/20 answered, "
            f"{report.traced_ratio:.0%} traced, {len(report.flagged)} flagged"
        )

        canary = container.story.ask(corpus_id, CANARY, args.provider)
        export(container, canary, "canary-run")
        print(f"  canary: {canary.story!r}")

        (OUT / "story.md").write_text(
            f"# The story of {run.person}\n\n_Run `{run.run_id}` on corpus `{corpus_id}`, "
            f"{run.provider}/{run.model}. Every bracketed id is a chunk in the corpus._\n\n{run.story}\n"
        )
        (OUT / "audit.md").write_text(
            f"# Sentence audit: run `{run.run_id}`\n\n{audit_table(run)}\n\n"
            f"## Canary\n\nQ: {CANARY}\n\nA: {canary.story}\n"
        )
        print(f"wrote {OUT.relative_to(ROOT)}/")
    finally:
        container.close()


if __name__ == "__main__":
    main()
