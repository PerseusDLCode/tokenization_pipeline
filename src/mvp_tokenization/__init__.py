"""mvp-tokenization: chunk-level tokenization and NLP annotation for compiled CTS chunks.

Two separate passes, decoupled on purpose -- both fully in-process (no NLP
server, no network dependency):

- ``mvp-tokenize`` (:mod:`mvp_tokenization.run_tokenizer`) -- the main pass.
  Greek and Latin get LatinCy's lemmatizing pipeline (minus parser/NER);
  everything else gets tokenize-only stanza. Incremental by content hash.
- ``mvp-analyze`` (:mod:`mvp_tokenization.run_analyzer`) -- optional, on-demand
  full NLP annotation. Loads the full spaCy/stanza models in-process and
  upgrades an existing tokenize-only sidecar in place with POS/lemma/deps
  (``--force``).

``mvp-pipeline`` (:mod:`mvp_tokenization.run_corpus`) chains citeStructure
compilation (perseus-cts) with the tokenize pass for a whole corpus.

``mvp-index`` (:mod:`mvp_tokenization.build_index`) builds the corpus search
index -- a range-queryable SQLite file -- from one or more tokens trees.
"""
