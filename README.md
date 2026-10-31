# ASMEM

**A small memory system: deterministic coarse recall, a verifier, and a bounded evidence postagent loop.**

Add is a format layer: it recovers speakers/timestamps, turns dialogue, documents, tables,
triples and agent traces into one generalized record, chunks long content, assigns stable ids,
and writes `memory.jsonl` + `state.jsonl`. It never calls a model and never rewrites facts.

Search has exactly two paths after the verifier:

```text
coarse (no LLM, cacheable)
  → verifier (1 model call + deterministic checks)
      ├─ pass → answer from coarse evidence
      └─ fail → bounded evidence postagent loop (≤3 model calls)
                     search(rewrite query) → take_note / auto-note → finish
  → merge/dedupe coarse + notes → answer-facing evidence
```

There is no fixed select/gaps/verify/finalize pipeline in the core method. The postagent
starts only when the verifier says the coarse evidence lacks the asked value; it decides
what query to rewrite, which results to note, and when it can answer.

## Quick start

```bash
# 1) add memory
python -m asmem.add --input messages.jsonl --user alice --session session_1

# 2) search
python -m asmem.search --memory data/asmem/alice/memory.jsonl \
    --query "Where did Alice move?"

# 3) HTTP API (/add + /search), same implementation
cp configs/api_mm.yaml configs/api.yaml   # fill generator.api_key
python -m asmem.api --config configs/api.yaml --port 8000
```

## Core pieces

### Coarse recall (no LLM, cacheable)

`src/asmem/retrieval/`:

- query analysis: tokens, phrases, time expressions, route, intent;
- seven channels: `bm25 / dense / exact / time / state / session / option`;
- weighted RRF, source quota, near-duplicate merge, neighbour expansion;
- optional dense embeddings with a versioned cache.

Coarse is deterministic and can be saved once:

```bash
python scripts/run_bench.py --config configs/bench_qwen3_gate_loop.yaml \
    --coarse-only --run-name qwen3_coarse_cache --overwrite
# later: experiment.reuse_coarse_from: qwen3_coarse_cache
```

`qwen3_b1` is already a full 2000-query coarse-only run and can be reused directly.

### Verifier

`src/asmem/search/verifier.py::gate_verify`:

- asks for the exact slot, a short answer, and a supporting quote;
- accepts paraphrases, synonyms, and resolved relative dates;
- deterministic checks on quote grounding, answer shape, entity/speaker, and
  answer↔quote support;
- optional `gate_check_consistency` for single-value/ordering questions;
- only two outcomes: pass (answer from coarse) or fail (run the loop).

### Postagent loop (bounded evidence, bounded)

`src/asmem/search/evidence_loop.py`:

- every turn is a fresh, bounded prompt (no accumulating chat history);
- actions: `search` (rewrite query, exclude seen ids, optional date filter),
  `take_note` (save selected results), `finish`;
- searches reuse the already-built coarse index (no reindex/re-embed);
- auto-note saves the top results of every search; if nothing was noted, the last
  search's top 2 are saved at the end;
- notes are appended to `evidence.jsonl` as raw records and merged/deduped with the
  coarse seeds by `postprocess_evidence` / `fit_evidence`;
- at most `loop_max_steps: 3`, plus one verifier call.

### API

`/add` and `/search` use the same implementation as the benchmark:

```text
/add    -> api.store.UserStore.append -> schema.normalize_messages
/search -> search.runner.run_question -> api.server.build_search_items
```

`configs/api_mm.yaml` is the committed API template (no key); copy it to the
gitignored `configs/api.yaml` and fill `generator.api_key`:

```yaml
postagent_mode: gate_loop
loop_max_steps: 3
loop_top_k: 5
loop_auto_note: 2
loop_note_chars: 1200
loop_evidence_chars: 6000
loop_item_chars: 400
loop_force_intents: []
loop_accept_answer_types: []
gate_check_consistency: true
pre_audit: false
```

Only deployment settings (generator/embedding endpoint, concurrency, store_dir) differ.
`/search` degrades to the coarse evidence already written on model/endpoint failure; it
never returns a 5xx.

`configs/api.yaml` is ready for the competition service: 4o-mini generator (vision
captioning for query images), local Qwen3-Embedding-4B on `:30001`, 200 coarse
candidates, 96 evidence records, and a bounded multi-hop enhancement.  For
aggregate/multi-hop intents the loop gets one extra rewrite turn and is told to
bridge event → city/date → receipts/reservations; when evidence carries
`[city]`/`[location]` tags, a deterministic `<city> <month> <year> receipt invoice
reservation confirmation` search is appended automatically.

## Multimodal: Mem-Gallery

Mem-Gallery is 20 multi-session scenarios with dialogue images and 1,711 QAs.  The
dataset ships an id and a human caption per image, so retrieval stays text-only:

```bash
# external dataset path (defaults to /home/minghaifeng/opticalmemory/data/Mem-Gallery)
python scripts/build_bench_datasets.py --datasets mem_gallery --overwrite
python scripts/run_bench.py --config configs/bench_mem_gallery_4omini_final.yaml \
    --run-name mg4o_final_full_v2 --overwrite
```

The converter writes one record per image (`[image: D1:IMG_003]` + `[caption] ...` +
`[round D1:3]`), and a QA's query image is represented by its caption.  The answer
stage routes by question shape: image ids are selected by evidence number (not
free-written), conflict questions ask whether memory affirms or denies the claim,
temporal questions use the `[resolved: ...]` annotations, and before/after questions
compare round ids.  See [docs/core.md](docs/core.md) §3.5.1.

`bench_mem_gallery_4omini_final.yaml` inherits its generator from
`configs/api.yaml` (so the key stays in one file) and sets
`answer.media.enabled: true`: the answer model receives the query image and up
to `max_images` memory images referenced by the final evidence, with per-type
image policies.  The caption-only A/B was run with a local variant that disables
`answer.media`.

400-query validation subset (the first 20 QAs of each scenario; same search index):

| method | score | VS | TTL | VR | CD | TR |
|---|---:|---:|---:|---:|---:|---:|
| direct answer (no type routing) | 0.7239 | 0.543 | 0.619 | 0.556 | 0.360 | 0.686 |
| typed answer routing | **0.8052** | 0.804 | 0.714 | 0.556 | 0.960 | 0.800 |

4o-mini A/B on the same 400 queries (same search/judge, only answer images change):

| answer input | score | TTL | VS | VR |
|---|---:|---:|---:|---:|
| captions only | 0.7628 | 0.762 | 0.819 | 0.667 |
| + query/memory images | 0.7789 | 0.786 | 0.825 | 0.667 |
| + per-type image policy + counting path | **0.7995** | **0.881** | 0.814 | **0.750** |

Image-question failure attribution: most failures are the model picking the wrong
image even though the gold image is in the evidence; retrieval misses are rare. The
final config therefore uses per-type image policies (VS low/3, TTL/VR/KR high/5) and
leaves text questions image-free. See [docs/core.md](docs/core.md) §5.2.

Full Mem-Gallery (1,711 QAs, 4o-mini, final config): **0.7455**.  Per type: VS 0.772,
TTL 0.783, VR 0.568, FR 0.744, MR 0.830, TR 0.764, CD 0.778, AR 0.739, KR 0.611.
The controlled 400-query A/B is the cleaner image contribution (0.7628 -> 0.7995);
the full set is harder.  The main limitation is answer-side visual selection, not
retrieval; a hybrid text-Qwen3 / image-4o-mini answerer is the next step.

## ATM-Bench

One person's ~4 years of 6,742 emails, 3,759 photos and 533 videos, with
1,013 evidence-grounded QAs.  The release ships captions/OCR for every visual
item, so the converter builds one text record per source with a stable id:

```text
[email: email202411160004] [summary] ... [detail] ... [time] ...
[image: 20220703_210745]   [caption] ... [tags] ... [location] ... [time] ...
[video: 20220502_172850]   [caption] ... [duration] ... [time] ...
```

All memory is added first (one add per calendar month), then the 1,013
searches run concurrently.  Photos are attached to `open_end`/`list_recall`
answers; number questions stay text-only.

```bash
python scripts/build_bench_datasets.py --datasets atm_bench --overwrite
python scripts/run_bench.py --config configs/bench_atm_bench_4omini.yaml \
    --run-name atm4o_full --overwrite
python scripts/analyze_atm_bench.py \
    --predictions data/bench_runs/atm4o_full/predictions.jsonl
```

`analyze_atm_bench.py` reports answer score **and** evidence-id recall
(hit-any / full / first-hit rank), because ATM-Bench is evidence-grounded.

Full 1,013 QAs (4o-mini): **0.6557** — list_recall 0.423 (id-set F1), number
0.750, open_end 0.653.  Evidence hit-any is 0.928 and full recall 0.880, with a
median first-hit rank of 1: retrieval is strong; the remaining gaps are
list-id precision, multi-hop number questions, and open-ended answer matching.

## Results

Full 2000:

| run | score | judge | rubric |
|---|---:|---:|---:|
| `qwen3_gate_loop_v1` | **0.7170** | 0.7788 | 0.6565 |
| `4omini_gate_loop_v1` | **0.7106** | 0.7740 | 0.6478 |
| `qwen3_b1` (coarse) | 0.6946 | 0.7490 | 0.6507 |
| `qwen3_b2` (old loop) | 0.7185 | 0.7790 | 0.6586 |

214-query validation subset (locomo enriched + 50 each for the other datasets):

| method | weighted | avg search calls |
|---|---:|---:|
| coarse | 0.7061 | 0 |
| old loop | 0.7373 | 7.28 |
| fixed pipeline (comparison) | 0.7524 | 5.29 |
| **core: verifier + evidence loop** | **0.7637** | **3.12** |

The core method uses ~57% fewer search-stage calls and ~77% fewer tokens than the old loop
on that subset.

## Add and record format

`memory.jsonl`, one JSON object per line:

```json
{"id":"s1_00001","text":"Caroline went to a support group on 7 May 2023.",
 "kind":"message","time":"2023-05-08T13:56:00Z","role":"Caroline",
 "source":"session_1","meta":{}}
```

Add does three things: recover speakers/timestamps, token-chunk long content with a shared
`parent_id`, and assign stable ids. It never calls a model. Tables, triples, documents and
agent traces all become the same record shape. See [docs/add.md](docs/add.md).

## Repository layout

```text
src/asmem/
  add/          deterministic add + normalization
  retrieval/    coarse recall (BM25/dense/exact/time/state/session/option, RRF)
  search/       verifier, evidence loop, postprocess, evidence, API runner
  api/          FastAPI /add + /search
  bench/        replay -> answer -> eval benchmark pipeline
configs/
  api_mm.yaml               HTTP API template (no key; copy to api.yaml)
  api.yaml                  local filled copy (gitignored)
  bench_mem_gallery_4omini_final.yaml  Mem-Gallery, per-type images
  bench_atm_bench_4omini.yaml          ATM-Bench emails/photos/videos
docs/
  core.md, add.md, search.md
scripts/
  run_bench.py, compare_runs.py, coarse, sample_bench.py
  dedupe_store_images.py            # migrate legacy per-user image entries
  test_asmem.py, test_bench.py, test_retrieval.py, test_evidence_loop.py
```

## Tests

```bash
python scripts/test_evidence_loop.py     # gate checks + evidence loop (CPU only)
python scripts/test_retrieval.py  # coarse / chunk / neighbours / comparison loop
python scripts/test_bench.py      # benchmark pipeline (no model calls)
python scripts/test_multimodal.py # captions, image assets, bounded responses
python scripts/test_api_journal.py # raw request audit + cross-user image dedupe
python scripts/test_asmem.py      # add / store / API contracts
```

## Implementation note

The postagent uses a standard retrieve → note → finish loop. This implementation
adds coarse-index reuse, auto-note, coarse+notes merge/dedupe, a deterministic
verifier, and bounded fresh prompts.
