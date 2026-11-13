# Log review findings

Run `r198ec82572` on corpus `benjamin-franklin-209336` (6 public-domain Franklin biographies,
4178 chunks), openai/gpt-4.1-mini, temperature 0. Files in this folder:

| File | What it is |
|---|---|
| `story.md` | the generated story |
| `story-run.jsonl` | every query: question, retrieved chunk ids, full prompt, raw reply, evidence quotes, gate decision, sufficiency-check prompt and reply; then the weave and the verifier |
| `story-run.json` | the same run as one object |
| `audit.md` | sentence-by-sentence trace table plus canary results |
| `canary-N-run.*` | the three canary runs |

## Automated review

- **20 queries → 19 answered, 1 skipped.** The skipped one, "An adventurous story", came back
  NOT_IN_SOURCES because the sufficiency check judged the only quote (a ballad) not an adventure.
  That is a defensible call, and the topic was left out of the story.
- **18 story sentences, 18 traced to a chunk id (100%), 0 flagged** by the verifier. The verifier
  runs two checks per sentence. A lexical check confirms the cited ids were retrieved and no
  number or name is missing from those chunks. An LLM judge then confirms the cited chunks
  support the sentence.
- **Canaries 3/3 "Not in the sources."** All three are true facts that postdate every book in the
  corpus:
  - the portrait on the $100 bill;
  - the year that portrait first appeared;
  - the back of the 1948 half dollar.

## Manual review: what the automated checks missed

I checked every suspicious sentence against the text of the chunks it cites.

| Sentence (short) | Cited | Verdict |
|---|---|---|
| Deborah raised Franklin's illegitimate son, **William Temple Franklin**, as her own | d01-0040, d01-0041 | **Misattribution.** d01-0041 names the son "William". William Temple Franklin is the *grandson* (d02-0267). The model merged two people. |
| Marriages were planned for **both Sarah and William Temple Franklin** | d04-0469, d02-0267 | **Half unsupported.** d02-0267 supports the Strahan plan for Sarah. Neither chunk mentions a marriage plan for Temple, and d04-0469 is about Franky, not Sarah. |
| His father, Josiah, was a tallow-chandler | d02-0016, d06-0011 | **Inferred.** d06-0011 says *Benjamin* "was destined to become a tallow-chandler", and d02-0016 names Josiah but not his trade. True, but not stated in the text. |
| Friends Collins, Watson, Osborne, Abel James | d04-0633, d02-0087, d04-0671 | Supported (Osborne is in d04-0633). |
| 26 years abroad, crossed the Atlantic eight times | d04-0018, d06-0171 | Supported by d04-0018. d06-0171 adds nothing (citation padding). |
| The harmonica made him popular in London | d04-0001, d05-0855, d03-0437 | Supported by d03-0437. The other two ids are padding. |
| Married September 1, 1730 | d01-0040 | Supported. |

**What this shows.** The judge passes a sentence whose words all appear in its cited chunks, even
when the facts are attached to the wrong person. "William", "Temple" and "Franklin" all appear
in d02-0267, so the merged name looked supported. Earlier runs had the same failure:
- Francis Folger's death attached to the wrong child (d04-0469);
- William Temple Franklin called a son instead of a grandson (d02-0267).

I added a who-is-who rule to both the answer prompt and the judge prompt. It reduced these errors
but did not eliminate them. **Citation padding** (extra ids that support nothing) is also never
caught, because the judge reads the cited chunks as a group.

## How the canary went from FAIL to PASS (run history)

| Stage | Answered | Traced | Canary |
|---|---|---|---|
| Prompt only ("say NOT_IN_SOURCES if…") | 20/20 | – | **FAIL**: answered $100 bill from training data, citing a loosely related chunk |
| + evidence gate (quote must align with cited chunk) | 19/20 | – | **FAIL**: quoted a *real* Greuze-portrait caption as "evidence" |
| + sufficiency check (do the quotes alone answer it?) | 17/20 | 94% | 3/3 |
| + in-order word alignment (tolerates footnote marks, "…") | 18/20 | 100% | 3/3 |
| + skip back-of-book index pages (number density > 0.15) | 18/20 | 89% | 3/3 |
| + 40-word quote cap, 1200 answer tokens | **19/20** | **100%** | **3/3** |

Lessons:
1. **Telling the model to say "not in the sources" is not enough.** It answers famous facts from
   memory and cites whatever chunk is closest.
2. **Checking that quotes are real is not enough either.** A real quote can be irrelevant to the
   question.
3. **What worked was a deterministic quote check followed by a check that sees only the quotes.**
   The second check cannot fall back on memory, because it never sees the question's famous
   context beside the snippets.
4. **Index pages are a retrieval hazard specific to this kind of corpus.** "Josiah, emigrates
   to Boston, 2; his family, 2, 3" matches every family question lexically and contains no
   narrative.

## Known limits

- **The weave is not gated, by design.** Gating it would silently delete story sentences. Instead,
  an uncited weave sentence is shown and the verifier flags it as UNCITED, so the reader sees it.
  This run had none.
- **Runs vary even at temperature 0.** Answered counts across identical runs ranged from 17 to 19.
- **Misattribution between people who share a name** (see above) is the main remaining risk. A
  human still has to read the trace table.
