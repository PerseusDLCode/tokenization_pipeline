"""mvp-tokenization: chunk-level tokenization and NLP annotation for compiled CTS chunks.

Two separate passes, decoupled on purpose:

- ``mvp-tokenize`` (:mod:`mvp_tokenization.run_tokenizer`) -- fast first pass.
  Hits a running nlp_pipeline server's tokenize-only endpoint to produce
  token-boundary sidecars with no POS/lemma/depparse cost, so citation-range
  resolution isn't blocked on full linguistic analysis.
- ``mvp-analyze`` (:mod:`mvp_tokenization.run_analyzer`) -- optional, on-demand
  full NLP annotation. Loads spaCy/stanza models in-process and upgrades an
  existing tokenize-only sidecar in place with POS/lemma/deps (``--force``).

``mvp-pipeline`` (:mod:`mvp_tokenization.run_corpus`) chains citeStructure
compilation (perseus-cts) with the fast tokenize pass for a whole corpus.
"""
