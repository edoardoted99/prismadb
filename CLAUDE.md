# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

prismadb implements the paper "Disentangling Dense Embeddings with Sparse Autoencoders" (O'Neill et al., 2024). It's a Django app that trains Top-K Sparse Autoencoders on text embeddings, interprets learned features via LLMs, and provides hybrid search (BM25 + semantic). All services run locally: Ollama for embeddings/LLM, ChromaDB for vector storage, SQLite for metadata.

## Commands

```bash
prismadb init                          # Run migrations + verify ChromaDB
prismadb ingest FILE -m MODEL          # Load JSON dataset and embed
prismadb train -d DATASET_ID           # Train SAE
prismadb interpret -r RUN_ID           # Label features via LLM
prismadb stats -r RUN_ID               # Correlations, co-occurrences, histograms
prismadb search QUERY -d DATASET_ID    # BM25/semantic/hybrid search
prismadb list datasets|runs|features   # List resources
prismadb serve                         # Dev server on :8000
prismadb manage CMD                    # Django management escape hatch
pytest                                 # Run tests (minimal coverage)
```

## Architecture

Data flows: **JSON upload** -> `embeddings` (Ollama embed) -> **ChromaDB + SQLite** -> `sae` (train Top-K SAE) -> `explorer` (interpret features, stats, families) -> `api` + `search` (serve results).

### Django Apps

- **embeddings** — `Dataset`/`Document` models, `OllamaEmbedder` (hits `/api/embed`), `services.py` for ingest + embed pipeline
- **sae** — `SAERun` model, `SAE` nn.Module in `modules.py` (Top-K hard sparsity + dead-neuron auxiliary loss), `trainer.py` training loop with z-score normalization
- **explorer** — `SAEFeature`/`Interpretation`/`FeatureFamily` models, `interpreter.py` (LLM labeling with pos/neg examples), `statistics.py` (decoder weight correlation, co-occurrence, histograms), `family_builder.py` (MST-based hierarchical clustering via NetworkX)
- **search** — ChromaDB wrapper: `client.py` (PersistentClient singleton), `collections.py` (per-dataset collections), `bulk_ops.py` (upsert/scroll/random), `queries.py` (BM25 via rank_bm25, kNN via ChromaDB, hybrid fusion)
- **api** — DRF REST endpoints + drf-spectacular OpenAPI at `/api/v1/docs/`
- **project** — Settings, CLI (Click), `utils.py` with `get_setting()` 3-tier lookup (DB > env > default)

### Key Patterns

**Settings:** `get_setting(key)` checks `AppSetting` DB table, then env var (`OLLAMA_HOST`), then defaults. Add new settings to `SETTING_DEFAULTS` and `SETTING_ENV_VARS` in `project/utils.py`.

**ChromaDB fallback:** Every caller wraps ChromaDB in `try/except` with `is_available()` guard. The app works without ChromaDB (falls back to SQLite queries).

**Threading:** Embedding generation, interpretation, and statistics run in daemon threads. Progress tracked via `TASK_PROGRESS[thread_id]` dict in `explorer/task_status.py`.

**SAE loss:** `L_rec` (normalized MSE) + `alpha_aux * L_aux` (dead-neuron reconstruction). Dead neurons tracked per-epoch via `epoch_on_count` buffer. Top-K enforced in forward pass, not via L1 penalty.

## Environment

`.env` is auto-loaded via `python-dotenv` in `settings.py`. Key variables:
- `OLLAMA_HOST` — Ollama base URL (e.g., `http://127.0.0.1:11434`)
- `PRISMADB_HOME` — Data directory for SQLite + ChromaDB + model weights (default: project root)

## Reference Paper

The implementation should align with arXiv:2408.00657v2. The paper PDF is at `2408.00657v2.pdf`. Known deviations are tracked and being fixed (decoder normalization, encoder init, predictor LLM, post-top-k activations in interpreter, family DFS construction).
