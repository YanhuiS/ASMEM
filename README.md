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

`configs/api.yaml` and `configs/asmem.yaml` both set the core method:

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
  api.yaml                  HTTP API (core method)
  asmem.yaml                general add/search/API config (core method)
  bench_qwen3_gate_loop.yaml   full benchmark (core method)
  bench_4omini_gate_loop.yaml  full benchmark with 4o-mini
  bench_qwen3_gate_loop_fix200.yaml  214-query validation
  ablations/                comparison configs (not part of the core commit)
docs/
  core.md, add.md, search.md
scripts/
  run_bench.py, compare_runs.py, coarse, sample_bench.py
  test_asmem.py, test_bench.py, test_retrieval.py, test_evidence_loop.py
```

## Tests

```bash
python scripts/test_evidence_loop.py     # gate checks + evidence loop (CPU only)
python scripts/test_retrieval.py  # coarse / chunk / neighbours / comparison loop
python scripts/test_bench.py      # benchmark pipeline (no model calls)
python scripts/test_asmem.py      # add / store / API contracts
```

## Implementation note

The postagent uses a standard retrieve → note → finish loop. This implementation
adds coarse-index reuse, auto-note, coarse+notes merge/dedupe, a deterministic
verifier, and bounded fresh prompts.
