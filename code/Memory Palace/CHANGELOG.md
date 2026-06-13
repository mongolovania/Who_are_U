# Memory Palace Changelog

## v9.0.0 (2026-06-10)

### Added
- **COLD `cold_fusion` retrieval**: New 3-path light fusion for COLD/WARM levels — BM25 (50%) + Emotion (25%) + Temporal (25%)
- **`_retrieve_cold_fusion()`**: Zero-dependency retrieval method with fallback chain (BM25 → fuzzy → return_all)
- **`_temporal_recency_score()`**: Exponential decay scoring with 60-day half-life
- **`CHANGELOG.md`** and **`VERSION`** files
- **`__version__`** attribute on RetrievalEngine

### Fixed
- **COLD regression**: `_retrieve_all()` was ignoring queries entirely — now uses cold_fusion
- **WARM regression**: `_retrieve_semantic_time()` used `fuzz.partial_ratio` (Levenshtein) which is not a retrieval algorithm. WARM now uses the same BM25-based cold_fusion as COLD. Storage differentiators (LLM gate, decay) preserved.
- **DDI monotonicity**: All four levels (COLD/WARM/HOT/RICH) now have monotonically appropriate retrieval quality

### Changed
- **WARM strategy**: `retrieval_mode` → `cold_fusion`, `emotion_mode` → `query_driven`, `use_bm25_search` → True
- **path_weights**: PPR reduced to 0.00 (reserved for future hippo_rag wiring)
- **Dead code documented**: 8 v7 modules marked `vNext: not yet wired`, GraphRAG/PPR paths marked `v9: Reserved`

### Architecture
- COLD + WARM share retrieval strategy (BM25-dominant) per BEIR/MemGPT/Mem0 literature consensus
- Storage policy (LLM gate, decay) remains the COLD/WARM differentiator
- BM25 strictly superior to dense embeddings at DDI < 100 docs

---

## v8 (2026-06-08)

- Temporal + cross_ref retrieval paths activated
- BM25 keyword search upgraded from fuzzy to rank_bm25
- Content-preserving fusion (v7 P0-1)
- Emotion resonance wired into retrieval (v7 P0-2)
- Typed graph traversal depth=3 (v7 P0-3)
- Two-phase retrieval-ranking decoupling (Fix 5)
- Path discrimination auto-silencing (Fix 2)

## v7 (2026-06-06)

- Causal + Narrative Enhancement modules
- Content-preserving fusion (P0-1)
- Emotion resonance path (P0-2)
- Typed graph traversal (P0-3)

## v6 (2026-06-01)

- Initial Memory Palace complete delivery (24 modules, ~25,000 lines)
- DDA-adaptive retrieval (COLD/WARM/HOT/RICH)
- Multi-path fusion architecture
