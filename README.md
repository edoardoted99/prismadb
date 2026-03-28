# PRISMA_v2

**Projection of Representations for Interpretability via Sparse Monosemantic Autoencoders**

PRISMA is a Django web application that makes Large Language Model (LLM) embeddings interpretable. Like an optical prism decomposes white light into its spectral components, PRISMA decomposes dense embedding vectors into sparse, human-readable **monosemantic features** using Sparse Autoencoders (SAEs).

**v2** adds Docker containerization, OpenSearch as a vector/document store, and hybrid search (full-text + semantic + sparse).

Built as part of a thesis project by **Edoardo Tedesco**.

<p align="center">
  <img src="docs/screenshots/home.png" alt="PRISMA Home" width="700">
</p>

---

## What's New in v2

### OpenSearch Integration
- **Hybrid search**: BM25 full-text + kNN semantic + combined hybrid search across documents
- **Feature search**: full-text search on feature labels and descriptions
- **Vector store**: document embeddings stored as `knn_vector` in OpenSearch (HNSW / cosinesimil)
- **Per-dataset indices**: `prisma_documents_{dataset_id}`, `prisma_features_{run_id}`

### Docker
- OpenSearch runs in Docker, Django runs natively on macOS (MPS GPU acceleration)
- Full-stack Docker profile available for CPU-only deployment

### Architecture Changes
- New **`search/`** Django app for OpenSearch integration (client, indices, bulk ops, queries)
- **Dual-write**: data written to both SQLite and OpenSearch during transition
- **Graceful fallback**: all OpenSearch read paths fall back to SQLite if OpenSearch is unavailable
- **Unified search page** at `/search/` with BM25, semantic, and hybrid modes

### Architecture Overview

```
SQLite (app state)              OpenSearch (data + search)
- Dataset (metadata)            - Documents (text + dense embedding)
- SAERun (config, status)       - SAE activations (sparse vectors)
- Document (relational shell)   - Feature metadata (label, desc, stats)
- Interpretation                - Interpretations (label, desc)
- FeatureFamily
- Auth/Sessions
```

---

## The Problem

LLMs project text into high-dimensional dense embeddings where individual dimensions have no intrinsic meaning. Neurons are **polysemantic** -- a single neuron may respond to multiple unrelated concepts. This is explained by the **Superposition Hypothesis** (Elhage et al. 2022): networks compress more concepts than they have neurons by exploiting statistical independence between features.

PRISMA reverses this process through **disentanglement**: projecting dense embeddings into an overcomplete, sparse latent space where each axis corresponds to an interpretable atomic concept.

## How It Works

```
Dense Embedding x (d dims)
        |
   [ Encoder ]
        |
Sparse Latent h (n dims, n >> d)     <-- Top-K sparsity: only k neurons active
        |
   [ Decoder ]
        |
Reconstruction x_hat ~ x
```

1. **Embed** -- Upload a text dataset and compute embeddings with HuggingFace models (medBIT, GTE, MiniLM, ...)
2. **Train SAE** -- Train a Sparse Autoencoder with Top-K hard sparsity on the embeddings
3. **Interpret** -- An LLM (via Ollama) labels each latent feature by examining its highest-activating documents
4. **Explore** -- Browse features, view activation statistics, co-occurrence graphs, and a knowledge graph built from feature relationships
5. **Search** *(v2)* -- Hybrid search across documents and features via OpenSearch

---

## Architecture

The application is organized into four Django apps:

| App | Purpose |
|---|---|
| `embeddings` | Dataset upload, HuggingFace embedding computation |
| `sae` | SAE model definition (Top-K), training loop, sparsity heatmaps |
| `explorer` | Feature interpretation (via Ollama), statistics, co-occurrence analysis, knowledge graph |
| `search` | OpenSearch integration: client, index management, bulk operations, hybrid queries |

### Key Modules

- `sae/modules.py` -- SAE architecture with Top-K sparsity and auxiliary dead-neuron loss
- `sae/trainer.py` -- Training pipeline with z-score normalization and heatmap generation
- `embeddings/embedders.py` -- HuggingFace model wrappers with chunked encoding
- `explorer/interpreter.py` -- Automated feature interpretation using local LLMs
- `explorer/graph_builder.py` -- Knowledge graph construction via Maximum Spanning Tree
- `explorer/statistics.py` -- Feature correlation, co-occurrence, and histogram computation
- `search/client.py` -- Singleton OpenSearch client with health check
- `search/indices.py` -- Index creation/deletion, per-model embedding dimensions
- `search/bulk_ops.py` -- Bulk index, scroll, update helpers for documents and features
- `search/queries.py` -- Query builders: BM25, kNN, hybrid, feature search

---

## Setup

### Prerequisites

- Python 3.10+
- [Docker](https://www.docker.com/) (for OpenSearch)
- [Ollama](https://ollama.ai) (for feature interpretation)

### Installation

```bash
git clone https://github.com/edoardoted99/PRISMA_v2.git
cd PRISMA_v2

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### Configuration

Create a `.env` file or export environment variables:

```bash
export DJANGO_SECRET_KEY='your-secret-key-here'
export DJANGO_DEBUG=True
export DJANGO_ALLOWED_HOSTS='127.0.0.1,localhost'

# OpenSearch (defaults work if using docker-compose)
export OPENSEARCH_HOST=localhost
export OPENSEARCH_PORT=9200

# Ollama (defaults to localhost:11434)
export OLLAMA_BASE_URL=http://localhost:11434
```

### Database

```bash
python manage.py migrate
python manage.py createsuperuser  # optional
```

---

## Running (Hybrid Mode — Recommended for Apple Silicon)

Django runs natively on macOS to use **MPS GPU acceleration** for training, embedding generation, and interpretation. OpenSearch runs in Docker.

```bash
# 1. Start OpenSearch
docker compose up -d

# 2. Verify OpenSearch is healthy
curl http://localhost:9200/_cluster/health

# 3. Start Ollama (separate terminal)
ollama serve

# 4. Start Django (natively, with MPS)
python manage.py runserver

# 5. Create OpenSearch indices (first time only)
python manage.py opensearch_init

# 6. Migrate existing data to OpenSearch (if upgrading from v1)
python manage.py migrate_to_opensearch
```

Visit `http://127.0.0.1:8000/`.

### Why Hybrid?

MPS (Apple Metal Performance Shaders) is not available inside Docker containers — it only works natively on macOS. By running Django natively:
- **Training** uses MPS for GPU-accelerated SAE training
- **Embedding generation** uses MPS for faster HuggingFace inference
- **Interpretation** runs Ollama locally without Docker networking overhead
- **OpenSearch** runs containerized with persistent data in a Docker volume

### Full Docker Mode (CPU only, no GPU)

For deployment on servers without GPU, or for testing:

```bash
docker compose --profile full up -d
```

This starts both OpenSearch and the Django app (with Gunicorn). The app container connects to Ollama via `host.docker.internal:11434`.

---

## OpenSearch Management Commands

```bash
# Create indices for all existing datasets and SAE runs
python manage.py opensearch_init

# Migrate all existing data (documents + features) from SQLite to OpenSearch
python manage.py migrate_to_opensearch

# Check indices
curl http://localhost:9200/_cat/indices?v
```

### Index Schema

**Documents** (`prisma_documents_{dataset_id}`):
- `text`: full-text searchable (BM25)
- `embedding`: `knn_vector` (HNSW, cosine similarity)
- `django_id`, `external_id`, `status`

**Features** (`prisma_features_{run_id}`):
- `label`, `description`: full-text searchable
- `density`, `max_activation`, `mean_activation`, `variance_activation`
- `example_docs`, `correlated_features`, `co_occurring_features`, `activation_histogram` (stored, not indexed)

---

## Usage

### 1. Upload a dataset

Provide a JSON file with text documents and select a HuggingFace embedding model.

<p align="center">
  <img src="docs/screenshots/upload_dataset.png" alt="Upload Dataset" width="500">
</p>

### 2. Train a Sparse Autoencoder

Configure expansion factor, Top-K sparsity, and other hyperparameters.

<p align="center">
  <img src="docs/screenshots/train_sae.png" alt="Train SAE" width="600">
</p>

### 3. Browse interpreted features

The LLM interpreter labels each latent feature automatically. Browse all discovered concepts with their activation statistics.

<p align="center">
  <img src="docs/screenshots/feature_browser.png" alt="Feature Browser" width="600">
</p>

### 4. Inspect individual features

View activation histograms, density, and global context for each feature.

<p align="center">
  <img src="docs/screenshots/feature_statistics.png" alt="Feature Statistics" width="500">
  <br>
  <img src="docs/screenshots/top_activating.png" alt="Top Activating Documents" width="500">
</p>

### 5. Explore the knowledge graph

Visualize the semantic topology of extracted features -- a directed graph built from co-occurrence and correlation analysis via Maximum Spanning Tree.

<p align="center">
  <img src="docs/screenshots/knowledge_graph.png" alt="Knowledge Graph" width="700">
</p>

### 6. Hybrid Search (v2)

Search across documents using BM25 full-text, kNN semantic similarity, or a combined hybrid mode. Search features by label and description.

Navigate to `/search/` or click **Search** in the navbar.

---

## File Inventory (v2 changes)

### New Files

| File | Purpose |
|---|---|
| `Dockerfile` | Python 3.11-slim, pip install, collectstatic, Gunicorn |
| `entrypoint.sh` | migrate + gunicorn (1 worker, 4 threads, 1200s timeout) |
| `docker-compose.yml` | OpenSearch (default) + Django app (profile: full) |
| `.dockerignore` | Excludes git, venv, media, weights, etc. |
| `search/__init__.py` | Search app init |
| `search/apps.py` | Django app config |
| `search/client.py` | Singleton OpenSearch client + `is_available()` health check |
| `search/indices.py` | Create/delete document & feature indices, embedding dim mapping |
| `search/bulk_ops.py` | Bulk index/update, scroll helpers, single-doc lookups |
| `search/queries.py` | BM25, kNN, hybrid, random docs, feature search |
| `search/views.py` | Unified search page view |
| `search/urls.py` | `/search/` route |
| `search/management/commands/opensearch_init.py` | Create indices for existing data |
| `search/management/commands/migrate_to_opensearch.py` | Migrate SQLite data to OpenSearch |
| `templates/search/search.html` | Hybrid search UI |

### Modified Files

| File | Changes |
|---|---|
| `project/settings.py` | Added `STATIC_ROOT`, `OLLAMA_BASE_URL`, `OPENSEARCH_*` settings, `search` in `INSTALLED_APPS` |
| `project/urls.py` | Added `search/` URL include |
| `requirements.txt` | Added `gunicorn>=21.2`, `opensearch-py>=2.4`, `psutil>=5.9` |
| `embeddings/models.py` | Added `opensearch_id` field to `Document` |
| `embeddings/services.py` | Dual-write: index docs + update embeddings in OpenSearch after SQLite |
| `embeddings/views.py` | Delete OpenSearch index on dataset deletion |
| `sae/trainer.py` | Load embeddings from OpenSearch scroll (fallback: SQLite) |
| `explorer/interpreter.py` | Scan docs via OpenSearch scroll, random docs via `function_score`, index features after interpretation |
| `explorer/statistics.py` | Batch iteration via OpenSearch scroll, bulk update features in OpenSearch after stats |
| `explorer/views.py` | kNN similarity search, feature search via OpenSearch, embedding from OpenSearch |
| `explorer/family_builder.py` | Load embeddings from OpenSearch for matrix computation |
| `explorer/llm_utils.py` | Uses `settings.OLLAMA_BASE_URL` consistently |
| `templates/base.html` | Added Search link in navbar |

---

## Supported Embedding Models

| Key | Model | Dimensions |
|---|---|---|
| `medbit` | IVN-RIN/medBIT | 768 |
| `gte_multilingual` | Alibaba-NLP/gte-multilingual-base | 768 |
| `gte_large` | thenlper/gte-large | 1024 |
| `sbert_minilm` | sentence-transformers/all-MiniLM-L6-v2 | 384 |
| `sbert_mpnet` | sentence-transformers/all-mpnet-base-v2 | 768 |
| `sbert_multi_minilm` | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | 384 |
| ... | and more (see `embeddings/models.py`) | |

---

## Sample Dataset

A ready-to-use sample of 1000 PubMed abstracts is included at `sample_data/pubmed_abstracts_1000.json` (source: [suolyer/pile_pubmed-abstracts](https://huggingface.co/datasets/suolyer/pile_pubmed-abstracts)). Upload it directly from the Embeddings page to get started.

---

## References

- O'Neill, C. et al. (2024). *Disentangling Dense Embeddings with Sparse Autoencoders*. arXiv:2408.00657
- Elhage, N. et al. (2022). *Toy Models of Superposition*. Transformer Circuits Thread
- Roy, O. & Vetterli, M. (2007). *The Effective Rank: A Measure of Effective Dimensionality*. EUSIPCO
- Wang, X. et al. (2024). *Disentangled Representation Learning*. arXiv:2211.11695

## License

This project is released for academic and research purposes.
