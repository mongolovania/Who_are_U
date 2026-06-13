# Memory Palace V2 — Track C: Retrieval Enhancement (对标最新项目)

Execute 5 retrieval enhancement tasks to bring Memory Palace up to 2025 SOTA standards.

## Scope: 5 Enhancement Tasks

| # | Task | Reference Project | Module | Est. |
|---|------|-------------------|--------|------|
| 1 | GraphRAG 社区检测+层次摘要 | Microsoft GraphRAG | `graph_rag.py` → `narrative_engine.py` | 3-4h |
| 2 | HippoRAG 个性化 PageRank 检索路径 | OSU/Stanford HippoRAG | `hippo_rag.py` → `retrieval_engine.py` | 3-4h |
| 3 | Procedural Memory 模块 | LANGMem | `procedural_memory.py` → `memory_orchestrator.py` | 2-3h |
| 4 | 叙事检索路径 P3 | — | `retrieval_engine.py` + `narrative_engine.py` | 1h |
| 5 | MemLong 风格可学习检索路径权重 | MemLong | `learnable_weights.py` → `retrieval_engine.py` | 2-3h |

## Execution Steps

### 1. Run All New Module Tests
```bash
cd "code/Memory Palace"
python -m pytest tests/test_graph_rag.py tests/test_hippo_rag.py tests/test_procedural_memory.py tests/test_learnable_weights.py -v --tb=short
```

### 2. Run Retrieval Engine Tests (with new paths)
```bash
python -m pytest tests/test_retrieval_engine.py -v --tb=short
```

### 3. Run Integration Tests
```bash
python -m pytest tests/test_scenario_integration.py -v --tb=short
```

### 4. Run Full Test Suite
```bash
python -m pytest tests/ -v --tb=short
```

### 5. Verify No Regressions
```bash
python -m pytest tests/ -v --tb=short -k "not graph_rag and not hippo_rag and not procedural_memory and not learnable_weights"
```

## Success Criteria
- [ ] GraphRAG community detection produces modularity-improving partitions
- [ ] Hierarchical community summaries generated (2 levels)
- [ ] HippoRAG PPR converges correctly on memory graph
- [ ] PPR retrieval returns personalized results
- [ ] Procedural memory tracks response preferences across sessions
- [ ] Narrative retrieval path P3 wired and returning stories
- [ ] Learnable weights adapt based on feedback signals
- [ ] All new module tests pass (0 failures)
- [ ] All existing tests continue to pass (0 regressions)

## Track Dependencies
- **Track A (L1 Storage)**: Complete — provides MemoryNode, BucketManager, DecayEngine
- **Track B (L2 Curation)**: Complete — provides retrieval_engine, memory_graph, narrative_engine bases

$ARGUMENTS
