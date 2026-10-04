# mvp-tokenization

Tokenization, lemmatization, and corpus search indexing for
[MinimumViablePerseus](https://github.com/PerseusDLCode/MinimumViablePerseus).

```
corpus repo ──mvp-pipeline──► tokens tree ──mvp-index──► search-<sha12>.db + manifest.json
 (TEI XML)     compile chunks     one .tokens.json.zst          range-queryable SQLite,
               + tokenize         sidecar per chunk             served at /search-index/
```

## Commands

| Command | What it does |
|---|---|
| `mvp-pipeline` | Compiles a corpus repo's citeStructure-bearing TEI into citation chunks (perseus-cts), then runs `mvp-tokenize` over them. |
| `mvp-tokenize` | Writes one token sidecar per chunk. Greek and Latin go through LatinCy's spaCy pipelines (tagger, morphologizer, lemmatizers; no parser/NER), so every token gets a lemma, UPOS and UD features. Other languages are tokenized only, with stanza. |
| `mvp-analyze` | Full stanza/spaCy analysis (including dependency parses) for any language. Not used by CI. |
| `mvp-index` | Builds the corpus search index from one or more tokens trees. |

```sh
uv run mvp-pipeline --corpus-root ../corpora/canonical-latinLit \
    --proto-dir ./proto --tokens-dir ./tokens --prune
uv run mvp-index --tokens-dir ./tokens --out ./search-index
```

### Incremental tokenization

Each sidecar directory has a `tokens-manifest.json`. For every chunk it records
a hash of the chunk XML and the pipeline version: the sidecar schema, the
library versions, and the model versions. A chunk is reprocessed only when one
of those changes, so restoring the previous output and re-running costs only
the new and edited texts. `--force` reprocesses everything. `--prune` deletes
sidecars whose chunk or work no longer exists; use it only when `--proto-dir`
holds a complete compile of the corpus. Each version directory also gets a
copy of its proto `metadata.json` and `index.json`, so the tokens tree alone
carries titles, authors and citation order.

Token text is always taken from the source text at the model's offsets.
LatinCy's Latin tokenizer rewrites v as u, and the reading view renders token
text verbatim. Each token's citable URN is
`{chunk urn}@{text}[{n}]`, where `n` counts occurrences of that exact text
within the chunk.

## The search index

A single SQLite file, read from the browser with
[sql.js-httpvfs](https://github.com/phiresky/sql.js-httpvfs) over HTTP range
requests. Only the pages a query touches are downloaded, the same way
pdl-morph-server serves `morph.db`. `build_index.py`'s docstring describes the
schema in full. In brief:

- **Range-friendly layout.** Pages are 4096 bytes, the journal is a rollback
  journal (not WAL), and the file is vacuumed. Posting tables are
  `WITHOUT ROWID` and clustered on their query key, so a query reads
  contiguous pages.
- **Postings.**
  - `form_postings` and `lemma_postings` hold `(chunk_id, seq, occ, …)`
    integers.
  - A hit's URN is rebuilt as `{base_urn}:{ref}@{form}[{occ}]`, exactly what
    MinimumViablePerseus renders as `data-token-urn`.
- **Counts.** `form_doc_counts` and `lemma_doc_counts` give per-work counts
  without touching postings.
- **Context.** `chunk_context` holds each chunk's tokens in 32-token windows,
  for keyword-in-context display.
- **Keys.** Each form and lemma has a `key` (accent-preserving, but folding
  case, grave/acute, final sigma, elision marks, and Latin u/v i/j) and a
  `loose` key (no accents at all). `normalize.py` computes them;
  MinimumViablePerseus's `static/js/search/normalize.js` must compute the same
  keys in the browser. `tests/fixtures/normalization.json` is the shared
  contract. Change both together and copy the fixture to
  `MinimumViablePerseus/tests/data/search_normalization.json`.
- **Scope.** Only Greek (`grc`) and Latin (`la`) chunks are indexed, and only
  each version's default citation scheme, since alternate schemes would
  double every hit.

Dictionary-headword search ("all forms of λύω") is resolved in the browser.
The page reads the headword's forms from pdl-morph-server's `morph.db` and
looks up their keys here, so the index has no dependency on Morpheus data.

## CI

- **`tokenize-corpus.yml`** runs nightly, on `workflow_dispatch`, and on a
  `repository_dispatch` of type `corpus-updated`.
  - It handles each corpus in `corpora.json` and skips any whose HEAD and this
    repo's commit are unchanged.
  - Otherwise it restores the corpus's previous
    `ghcr.io/perseusdlcode/mvp-tokens-<tag>:latest`, runs `mvp-pipeline
    --prune` (incremental), and pushes the result.
  - MinimumViablePerseus's page build pulls these artifacts.
- **`build-search-index.yml`** runs when tokenization finishes.
  - It pulls every `mvp-tokens-*:latest`, builds the index, smoke-tests it
    (`scripts/smoke_test_index.py`), and pushes
    `ghcr.io/perseusdlcode/mvp-search-index:latest`.
  - MinimumViablePerseus's `deploy/cron-deploy.sh` pulls it within 10 minutes.

**To add a corpus,** add it to `corpora.json`. Both workflows read that list.
The search index then picks the corpus up automatically.

### First run

A run for an unchanged corpus finishes in seconds. The first lemmatizing run
does every chunk, at roughly 5,000 words/s for Greek and 1,000 words/s for
Latin on one CPU core. Within one runner's 6-hour limit that fits each of the
current corpora. If a corpus outgrows it, run `mvp-pipeline` once locally and
push the result as `mvp-tokens-<tag>:latest`. Every later run is incremental.

```sh
tar --zstd -cf tokens.tar.zst -C tokens .
oras push ghcr.io/perseusdlcode/mvp-tokens-<tag>:latest tokens.tar.zst:application/vnd.perseus.mvp.tokens
```

## Tests

```sh
uv run pytest
```
