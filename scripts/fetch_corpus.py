"""Download the Benjamin Franklin test corpus from Project Gutenberg.

Only third-party biographies published before 1918 are used. That keeps the canary honest:
facts from later years (e.g. Franklin's portrait on the $100 bill, standard since 1914) cannot be
in the sources, so the app must answer "Not in the sources." for them.

    uv run python scripts/fetch_corpus.py            # -> corpus/raw/*.txt + corpus/MANIFEST.md
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import urllib.request
from pathlib import Path

BOOKS = [
    (21348, "Benjamin Franklin (American Statesmen)", "John T. Morse Jr.", 1889),
    (30406, "Benjamin Franklin", "John S. C. Abbott", 1876),
    (36896, "Benjamin Franklin, Self-Revealed, Vol. 1", "William Cabell Bruce", 1917),
    (36897, "Benjamin Franklin, Self-Revealed, Vol. 2", "William Cabell Bruce", 1917),
    (36376, "The Life of Benjamin Franklin", "M. L. Weems", 1818),
    (38469, "The Life of Benjamin Franklin, Illustrated by Tales, Sketches, and Anecdotes", "Samuel G. Goodrich", 1848),
]
WORDS_PER_PAGE = 300  # a conventional printed-page estimate
URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60]


def fetch(book_id: int) -> bytes:
    request = urllib.request.Request(URL.format(id=book_id), headers={"User-Agent": "biography-rag-corpus-fetcher/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("corpus"))
    args = parser.parse_args()
    raw = args.out / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    rows, total_pages = [], 0
    for book_id, title, author, year in BOOKS:
        path = raw / f"{book_id}-{slug(author)}-{slug(title)}.txt"
        if not path.exists():
            print(f"downloading #{book_id} {title} ...", file=sys.stderr)
            path.write_bytes(fetch(book_id))
            time.sleep(1)  # be polite to Gutenberg
        words = len(path.read_text(encoding="utf-8-sig", errors="replace").split())
        pages = words // WORDS_PER_PAGE
        total_pages += pages
        rows.append(
            f"| [{book_id}](https://www.gutenberg.org/ebooks/{book_id}) | {title} | {author} | {year} | {words:,} | {pages} |"
        )

    manifest = "\n".join(
        [
            "# Test corpus: Benjamin Franklin biographies (Project Gutenberg, public domain)",
            "",
            f"Pages are estimated at {WORDS_PER_PAGE} words per page. Download with `scripts/fetch_corpus.py`.",
            "",
            "| Gutenberg # | Title | Author | Year | Words | ≈ Pages |",
            "|---|---|---|---|---|---|",
            *rows,
            f"| | **Total** | | | | **{total_pages}** |",
            "",
        ]
    )
    (args.out / "MANIFEST.md").write_text(manifest)
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
