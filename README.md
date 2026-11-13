# biography-rag

You upload pages about one person. The app asks 20 fixed questions about them and writes a short life story **only from what the pages say**. Every sentence cites the chunk it came from. When the pages don't answer a question, the answer is "not in the sources" and that topic is left out.

- **Live site:** https://biography-rag.onrender.com (free plan: the first request after idle takes ~1 min to wake)
- **Generated story:** [`logs/submission/story.md`](logs/submission/story.md)
- **Logs and review:**
  - [`logs/submission/`](logs/submission/): the full run log, every query, the retrieved chunk ids and the prompt sent.
  - [`audit.md`](logs/submission/audit.md): the sentence-by-sentence trace.
  - [`findings.md`](logs/submission/findings.md): my manual review.

Works for any person: nothing in the code or prompts knows about Franklin.

## Results (Benjamin Franklin, 6 biographies ≈ 2,400 pages, 4,178 chunks)

| | |
|---|---|
| Questions answered | 19 / 20 (the 20th: "not in the sources") |
| Story sentences traced to a chunk id | 18 / 18 (100%) |
| Sentences flagged by the verifier | 0 |
| Canaries ("Whose portrait is on the $100 bill?" and 2 more) | 3 / 3 answered "Not in the sources." |

The manual review found three sentences the automated checks passed:
- **One misattribution:** son and grandson were merged.
- **Two weakly supported claims.**

Details and the run history are in [`findings.md`](logs/submission/findings.md).

## How it works

```
upload (.txt .md .html .pdf)
  └─ normalize, strip Gutenberg boilerplate
  └─ chunk: paragraph/sentence aware, ~1200 chars, 200 overlap, ids d01-0042
  └─ embed (OpenAI text-embedding-3-small | local fastembed) → Postgres + pgvector

for each of 20 questions, with {person} filled in:
  retrieve   hybrid: pgvector cosine + Postgres full-text, fused by reciprocal rank (top 6)
             skip back-of-book index pages (numbers/words > 0.15)
  answer     model must reply  EVIDENCE: "<exact quote ≤40 words>" [chunk-id] …  then  ANSWER:
  gate       deterministic: each quote must align with the chunk it cites
             (≥90% of its words in in-order runs of ≥3 words), and uncited sentences are dropped
  check      a second call sees ONLY the question + verified quotes: "do these state the answer?"
             NO → NOT_IN_SOURCES
  log        question, retrieved ids + scores, full prompt, raw reply, quotes, gate reason, check

weave        story from the findings only; skipped topics left out; every sentence cites ids
verify       per sentence: lexical check (cited ids retrieved? unknown names/numbers?)
             + LLM judge against the cited chunks → TRACED / UNCITED / UNSUPPORTED / JUDGE_FLAGGED
```

### Why the gate and the check

Telling the model "say NOT_IN_SOURCES if it's not there" **failed the canary**. It answered "$100 bill" from memory and cited a loosely related chunk. Two changes made the answer depend on checks rather than on the model's honesty:

1. **The evidence gate.** The model must quote, and code checks the quote really is in the cited chunk. The quote can't be invented.
2. **The sufficiency check.** The gate alone still failed: the model quoted a real caption about a Franklin *portrait*. So a second call sees only the quotes, never the snippets around them, and decides whether those quotes state the answer. Memory doesn't help it there.

The full stage-by-stage history is in [`findings.md`](logs/submission/findings.md).

The **weave is deliberately not gated**: an uncited sentence is shown and flagged by the verifier, not silently deleted. The reviewer sees exactly what the model produced.

### Code layout (`src/biography_rag/`)

| Module | Responsibility |
|---|---|
| `domain.py` | dataclasses: Corpus, Chunk, StoryRun, QueryAnswer, Verdict… |
| `ingest/` | loaders per file type, chunker |
| `embeddings/`, `llm/` | `Embedder` / `ChatModel` protocols + providers, picked by name from a registry |
| `store/` | `ChunkStore` protocol: in-memory and Postgres/pgvector implementations |
| `rag/` | queries, retriever, prompts (all in one file), generation, grounding gate, verification |
| `services.py` | use cases: ingest, run story, ask one question; writes the JSONL run log |
| `web/` | FastAPI app, rate limiting, dependency wiring (`container.py`, the only place that builds concrete classes) |

Every dependency is injected through a protocol. Tests swap in in-memory stores and scripted LLMs; no network is needed.

## Run locally

```bash
cp .env.example .env            # fill in OPENAI_API_KEY (and DATABASE_URL, or leave empty for in-memory)
uv sync                         # (on NixOS: nix-shell --run 'uv sync')
uv run biography-rag             # serves on $PORT (default 8000)
```

Open http://localhost:8000:
1. Upload files and give the person's name.
2. Start a story run.
3. Open the run page to see the story, the per-question log and the audit.

**Compare models.** The story form and the ask form both take one or more models (**+ add model**, up to `MAX_COMPARE_TARGETS`, default 6). One model is an ordinary run. Several start one run each over the same retrieval and open a comparison page: the stories side by side with time, answered count, traced % and flagged sentences, then a question-by-question table of who answered and which chunks they cited. Every run is still logged and audited on its own. API: `POST /api/corpora/{id}/compare` with `{"kind": "story"|"ask", "question"?, "targets": [{"provider", "model"}]}`. A comparison costs one rate-limit hit per model.

Reproduce the hand-in artefacts:

```bash
uv run python scripts/fetch_corpus.py        # 6 Gutenberg biographies → corpus/raw/
uv run python scripts/run_submission.py      # ingest + story + 3 canaries → logs/submission/
```

Tests: `uv run pytest`. The Postgres tests run when `TEST_DATABASE_URL` is set; see `scripts/test_postgres.sh`. Lint: `uv run ruff check`.

## Switching providers

- **LLM:** every provider speaks the OpenAI chat protocol, so one adapter (`llm/providers.py`) covers them all. The UI offers each provider whose key is set; `LLM_PROVIDER` picks the default.

  | Provider | Env key | Models users can pick |
  |---|---|---|
  | OpenAI | `OPENAI_API_KEY` | `OPENAI_MODEL` (default gpt-4.1-mini) |
  | Groq (free tier) | `GROQ_API_KEY` | openai/gpt-oss-120b, openai/gpt-oss-20b, qwen/qwen3.8-27b |
  | Google Gemini (free tier) | `GEMINI_API_KEY` | gemini-3.5-flash-lite, falling back to gemini-3.1-flash-lite when its quota runs out |
  | NVIDIA NIM (free) | `NVIDIA_API_KEY` | nemotron-3-ultra, nemotron-3.5-lightning, laguna-xs-2.1, gpt-oss-20b, gemma-4-31b, muse-glimmer-30b, mistral-large-4, deepseek-v4.1-flash *(slow)*, glm-5.3-flash *(slow)* |
  | cheapllm.shop | `CHEAPLLM_API_KEY` | deepseek-v4.1-flash-1017 *(slow)* |
  | Qwen / DeepSeek | `QWEN_API_KEY` / `DEEPSEEK_API_KEY` | `QWEN_MODEL` / `DEEPSEEK_MODEL` |

  - The public API accepts a model only if it is on that provider's list. Anything else is a 422, so a visitor can't point the app at an arbitrary (paid) model.
  - Completions are streamed, so slow models aren't cut off by gateway idle timeouts. Reasoning is switched off or kept low where the provider allows, and any `<think>` text is stripped.
  - Each provider has its own timeout, retry budget and concurrency cap, shared by every run.
- **Embeddings:**
  - `EMBEDDER=openai` (default) or `local` (fastembed; `uv sync --extra local`).
  - A corpus remembers its embedder, so its queries always use the same one.

## Deploy (Render)

`render.yaml` is a blueprint for the free Docker web service.
- Set `DATABASE_URL` (Supabase session pooler, pgvector enabled) and the provider keys (`OPENAI_API_KEY`, `GROQ_API_KEY`, …) in the dashboard. They are `sync: false`, so they never touch git.
- `ADMIN_TOKEN` is generated, and is required to delete corpora.
- Story runs are rate-limited per client.
- `SITE_PASSWORD`, if set, puts one HTTP Basic password (any username) in front of every page and API route except `/api/health`, so strangers can't spend the free model quotas.

## Corpus

Six public-domain Franklin biographies from Project Gutenberg, ≈ 2,438 pages ([`corpus/MANIFEST.md`](corpus/MANIFEST.md)). The text isn't committed; `scripts/fetch_corpus.py` downloads it.
