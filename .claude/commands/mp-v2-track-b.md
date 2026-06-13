# Memory Palace V2 — Track B: Memory Curation Layer (L2)

Execute comprehensive verification and hardening of the Memory Palace **Layer 2 (Memory Curation/Intelligence)** modules.

## Scope: L2 Modules (9 modules)

| # | Module | Theory | Core Responsibility |
|---|--------|--------|-------------------|
| 1 | `working_self.py` | Conway SMS | Active goals + session inference + memory matching |
| 2 | `memory_graph.py` | Zep + A-MEM | Temporal graph + multi-relation edges + edge expiration |
| 3 | `vulnerability_model.py` | McEwen+Post+Kuppens+Scheffer | 4-theory vulnerability index + storage threshold adjustment |
| 4 | `script_deviation.py` | Schank | Statistical anomaly detection + baseline update |
| 5 | `flashbulb_detector.py` | Brown&Kulik | Triple-trigger flashbulb detection + Print Now! |
| 6 | `importance_fusion.py` | Conway+Bower+Eb.+A-MEM | 7-signal fusion importance scoring |
| 7 | `retrieval_engine.py` | Ebbinghaus+Bower+Mem0 | DDA-adaptive 4-path retrieval |
| 8 | `narrative_engine.py` | Conway+Schank | Narrative organization + chapter detection |
| 9 | `memory_evolution.py` | A-MEM | Zettelkasten links + memory evolution + versioning |

## Execution Steps

### 1. Layer 2 Unit Tests
```bash
cd "code/Memory Palace"
python -m pytest tests/ -v --tb=short \
  -k "working_self or memory_graph or vulnerability or script_deviation or flashbulb or importance_fusion or retrieval_engine or narrative_engine or memory_evolution"
```

### 2. Layer 2 Benchmarks
```bash
python -m pytest tests/benchmarks/ -v --tb=short
```

### 3. Cross-Module Integration
```bash
python -m pytest tests/test_scenario_integration.py -v --tb=short
```

### 4. V6 Spec Compliance (L2 focus)
```bash
python -m pytest tests/test_v6_verification.py -v --tb=short
```

### 5. Code Quality Review
- Check each L2 module for PEP 8 compliance
- Verify docstrings completeness
- Check type hints coverage
- Verify theory-to-code mapping accuracy

## Success Criteria
- [ ] All L2 unit tests pass (0 failures)
- [ ] All benchmarks pass
- [ ] Cross-module integration tests pass
- [ ] V6 spec compliance verified
- [ ] No CRITICAL code quality issues
- [ ] Theory-to-implementation mapping verified for all 9 modules

## Track Dependencies
- **Track A (L1 Storage)**: Must be complete — provides MemoryNode, BucketManager, DecayEngine
- **Track C (L0+L3 Orchestration)**: Depends on Track B — uses L2 modules for intelligence

$ARGUMENTS
