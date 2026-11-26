"""Download two more test corpora, so a retrieval change can be checked on people the code has
never seen. Both are public domain (Project Gutenberg) and both are read from the Gutenberg mirror
because www.gutenberg.org returns 503 when it is being rebuilt.

    uv run python scripts/fetch_extra_corpus.py       # -> corpus/raw-douglass/, corpus/raw-lincoln/

Why these two:

* **Frederick Douglass** is three first-person accounts (1845, 1855, 1881) plus Booker T.
  Washington's 1907 biography, so the memoir path (``narration=auto``, the question rephrased in
  the first person) is exercised by three of the four books.
* **Abraham Lincoln** is four third-party biographies (Herndon ×2, Lamon, Curtis) written between
  1872 and 1903, the ordinary case the Franklin corpus covers.
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from pathlib import Path

MIRROR = "https://gutenberg.pglaf.org/cache/epub/{id}/pg{id}.txt"
BOOKS: dict[str, list[tuple[int, str, str]]] = {
    "raw-douglass": [
        (23, "narrative-of-the-life-of-frederick-douglass-1845.txt", "Narrative of the Life of Frederick Douglass"),
        (202, "my-bondage-and-my-freedom-1855.txt", "My Bondage and My Freedom"),
        (71893, "life-and-times-of-frederick-douglass-1881.txt", "Life and Times of Frederick Douglass"),
        (69692, "booker-t-washington-frederick-douglass-1907.txt", "Frederick Douglass (Booker T. Washington)"),
    ],
    "raw-lincoln": [
        (38483, "abraham-lincoln-the-true-story-vol1.txt", "Abraham Lincoln: The True Story, vol. 1 (Herndon)"),
        (38484, "abraham-lincoln-the-true-story-vol2.txt", "Abraham Lincoln: The True Story, vol. 2 (Herndon)"),
        (40977, "the-life-of-abraham-lincoln-lamon-1872.txt", "The Life of Abraham Lincoln (Lamon)"),
        (42526, "abraham-lincoln-curtis-1903.txt", "Abraham Lincoln (Curtis)"),
    ],
}


def fetch(book_id: int) -> bytes:
    request = urllib.request.Request(
        MIRROR.format(id=book_id), headers={"User-Agent": "biography-rag-corpus-fetcher/1.0"}
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return response.read()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("corpus"))
    args = parser.parse_args()
    for folder, books in BOOKS.items():
        out = args.out / folder
        out.mkdir(parents=True, exist_ok=True)
        rows = []
        for book_id, name, title in books:
            path = out / name
            if not path.exists():
                print(f"downloading #{book_id} {title} ...", file=sys.stderr)
                path.write_bytes(fetch(book_id))
                time.sleep(1)  # be polite to Gutenberg
            words = len(path.read_text(encoding="utf-8-sig", errors="replace").split())
            rows.append(f"| [{book_id}](https://www.gutenberg.org/ebooks/{book_id}) | {title} | {words:,} |")
        (out / "MANIFEST.md").write_text(
            f"# Test corpus: {folder.removeprefix('raw-')} (Project Gutenberg, public domain)\n\n"
            "| Gutenberg # | Title | Words |\n|---|---|---|\n" + "\n".join(rows) + "\n"
        )
        print(f"{out}: {len(books)} books")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
