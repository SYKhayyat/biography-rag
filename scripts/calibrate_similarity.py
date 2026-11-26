"""Pick a MIN_SIMILARITY floor for the embedder this corpus uses.

The dense arm has no way to say "nothing here is relevant" unless it is given a cosine floor.
Choosing one needs two distributions for one corpus:

* questions the sources answer — the top cosine should stay *above* the floor;
* canaries (famous facts the book never states) — the top cosine should fall *below* it.

This prints both, question by question, plus the highest canary score, so the floor goes in the
gap between them. It embeds the questions only: no chat model is called, so it costs a few cents
of embeddings and nothing else. Nothing is written to the database or to the corpus.

    python scripts/calibrate_similarity.py --person "Benjamin Franklin" corpus/raw/*.txt
    python scripts/calibrate_similarity.py --person "Benjamin Franklin" corpus/raw/*.txt \
        --canary "Whose portrait is on the hundred dollar bill?" --canary "When did he die?"

Then set MIN_SIMILARITY in .env to a value just under the lowest answerable top score and just
over the highest canary score. If those overlap, the floor cannot separate them for this corpus
and 0 (off) is the honest setting.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biography_rag.config import get_settings
from biography_rag.domain import SourceDocument
from biography_rag.embeddings.registry import EmbedderRegistry
from biography_rag.ingest.chunker import ParagraphChunker
from biography_rag.ingest.loaders import extract_text, looks_binary
from biography_rag.rag.queries import QUERIES

CANARIES = (
    "Whose portrait is on the hundred dollar bill?",
    "What did he think of the French Revolution?",
    "How many grandchildren did he have?",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", help="the pages of one corpus (.txt .md .html .pdf)")
    parser.add_argument("--person", required=True, help="the person's name, as the questions spell it")
    parser.add_argument("--canary", action="append", default=[], help="a famous fact these pages never state")
    parser.add_argument("--embedder", default=None, help="openai | local (default: from settings)")
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--chunk-overlap", type=int, default=None)
    parser.add_argument("--show", type=int, default=0, help="also print the top passage's first 90 chars")
    parser.add_argument("--pool", type=int, default=24, help="candidate_k, for the survival table")
    args = parser.parse_args()

    settings = get_settings()
    embedder = EmbedderRegistry.from_settings(settings).by_name(args.embedder or settings.embedder)
    chunker = ParagraphChunker(
        size=args.chunk_size or settings.chunk_size, overlap=args.chunk_overlap or settings.chunk_overlap
    )

    chunks = []
    for n, path in enumerate(sorted(Path(p).expanduser() for p in args.files), 1):
        data = Path(path).read_bytes()
        if looks_binary(path.name, data):
            print(f"skipping binary file {path}", file=sys.stderr)
            continue
        text = extract_text(path.name, data)
        if text.strip():
            chunks.extend(chunker.chunk(SourceDocument(f"d{n:02d}", path.name, text)))
    if not chunks:
        print("no text found in those files", file=sys.stderr)
        return 2
    print(f"{len(chunks):,} passages from {len(args.files)} files, embedder {embedder.embedder_id}\n")

    vectors = embedder.embed_documents([c.embed_input for c in chunks])

    def top(question: str) -> tuple[list[float], int]:
        query = embedder.embed_query(question)
        sims = sorted((sum(a * b for a, b in zip(query, v, strict=True)) for v in vectors), reverse=True)
        return sims, 0

    def report(label: str, question: str, pool: int) -> tuple[float, float, float, float]:
        sims, _ = top(question)
        first, sixth = sims[0], sims[min(5, len(sims) - 1)]
        deep = sims[min(pool - 1, len(sims) - 1)]
        print(f"{label}  {first:5.3f}   {sixth:5.3f}   {deep:5.3f}   {question}")
        if args.show:
            query = embedder.embed_query(question)
            best = max(range(len(vectors)), key=lambda i: sum(a * b for a, b in zip(query, vectors[i], strict=True)))
            print(
                f"       -> {chunks[best].chunk_id} [{chunks[best].context}]: {chunks[best].text[: args.show]}".replace(
                    "\n", " "
                )
            )
        return first, sixth, deep, first

    print(f"cosine at rank 1, 6 and {args.pool} of the dense arm")
    print("=== questions the sources answer (the floor must stay below the rank-1 column) ===")
    answerable = [report("Q  ", q.render(args.person), args.pool) for q in QUERIES]

    print("\n=== canaries: famous facts these pages do not state (the floor should sit above these) ===")
    canaries = [report("CAN", c, args.pool) for c in (args.canary or CANARIES)]

    lowest_first = min(a[0] for a in answerable)
    median_first = sorted(a[0] for a in answerable)[len(answerable) // 2]
    highest_canary = max(c[0] for c in canaries)
    print()
    print(f"answerable rank-1: lowest {lowest_first:.3f}  median {median_first:.3f}")
    print(f"canary     rank-1: highest {highest_canary:.3f}")

    # The floor is a noise filter, not a refusal test. It has to sit under the answerable hits
    # while cutting clearly unrelated text, and every point it rises costs dense-arm recall in the
    # candidate pool the fusion draws from.
    print(f"\nshare of the rank-1..{args.pool} pool that survives a floor (answerable questions, median):")
    for floor in (0.2, 0.3, 0.4, 0.5, 0.6):
        survivors = [
            sum(1 for v in _pool(embedder, chunks, vectors, q.render(args.person), args.pool) if v >= floor)
            for q in QUERIES
        ]
        share = sorted(survivors)[len(survivors) // 2] / args.pool
        print(f"  MIN_SIMILARITY={floor:.1f}  {share:4.0%}")

    if lowest_first > highest_canary:
        gap = (lowest_first + highest_canary) / 2
        print(f"\nA refusal test could sit at {gap:.2f} (between the canary ceiling and the lowest")
        print("answerable hit). As a *retrieval* floor that value is aggressive: apply it and only")
        print("the top one or two dense hits survive, so the dense arm stops contributing to the")
        print("fusion. Start at 0.25-0.35 — a noise filter — and raise it only if the passages that")
        print("reach the model are unrelated, not merely imperfect.")
    else:
        print("\nThey overlap: no floor separates answerable questions from canaries on this corpus.")
        print("Leave MIN_SIMILARITY at 0.25 (noise only) or 0 (off) and let the evidence gate decide.")
    return 0


def _pool(embedder, chunks, vectors, question: str, pool: int) -> list[float]:
    query = embedder.embed_query(question)
    sims = sorted((sum(a * b for a, b in zip(query, v, strict=True)) for v in vectors), reverse=True)
    return sims[:pool]


if __name__ == "__main__":
    raise SystemExit(main())
