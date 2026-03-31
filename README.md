# prismadb

**Projection of Representations for Interpretability via Sparse Monosemantic Autoencoders**

[![PyPI](https://img.shields.io/pypi/v/prismadb)](https://pypi.org/project/prismadb/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)

prismadb is a Django-based platform that makes LLM embeddings interpretable. Like an optical prism decomposes white light into its spectral components, prismadb decomposes dense embedding vectors into sparse, human-readable **monosemantic features** using Sparse Autoencoders (SAEs).

It ships with Docker containerization, Ollama-native embeddings, OpenSearch as a vector/document store, hybrid search (full-text + semantic + sparse), and a REST API.

Built as part of a thesis project by **Edoardo Tedesco**.

<p align="center">
  <img src="docs/screenshots/home.png" alt="prismadb Home" width="700">
</p>

---

## Features

- **Ollama-native embeddings** -- auto-detects embedding models, no HuggingFace dependency
- **Sparse Autoencoder training** -- Top-K hard sparsity, dead-neuron auxiliary loss, z-score normalization
- **LLM-powered interpretation** -- automatic labeling of latent features via local LLMs
- **Knowledge graph** -- co-occurrence and correlation analysis, Maximum Spanning Tree visualization
- **Hybrid search** -- BM25 full-text + kNN semantic + combined hybrid via OpenSearch
- **REST API** -- full CRUD, inference endpoint, search endpoints, OpenAPI/Swagger docs
- **Docker-ready** -- OpenSearch in container, optional full-stack deployment

---

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

1. **Embed** -- Upload a JSON dataset and compute embeddings with Ollama
2. **Train SAE** -- Train a Sparse Autoencoder on the embeddings
3. **Interpret** -- An LLM labels each latent feature by examining its highest-activating documents
4. **Explore** -- Browse features, activation statistics, co-occurrence graphs, and the knowledge graph
5. **Search** -- Hybrid search across documents and features via OpenSearch

---

## Quickstart

### Prerequisites

- Python 3.10+
- [Docker](https://www.docker.com/) (for OpenSearch)
- [Ollama](https://ollama.ai) (for embeddings + feature interpretation)

```bash
ollama pull nomic-embed-text
```

### Install from PyPI

```bash
pip install prismadb[all]
```

### Install from source

```bash
git clone https://github.com/edoardoted99/PRISMA_v2.git
cd PRISMA_v2
pip install -e ".[all]"
```

### Configuration

Create a `.env` file or export environment variables:

```bash
DJANGO_SECRET_KEY='your-secret-key-here'
DJANGO_DEBUG=True
DJANGO_ALLOWED_HOSTS='127.0.0.1,localhost'

# OpenSearch (defaults work with docker-compose)
OPENSEARCH_HOST=localhost
OPENSEARCH_PORT=9200

# Ollama (defaults to localhost:11434)
OLLAMA_BASE_URL=http://localhost:11434
```

### Run

```bash
# 1. Start OpenSearch
docker compose up -d

# 2. Initialize database
prismadb migrate
prismadb opensearch_init

# 3. Start the server
prismadb runserver
```

Visit `http://127.0.0.1:8000/`.

### Full Docker Mode (CPU only)

```bash
docker compose --profile full up -d
```

This starts both OpenSearch and the Django app (via Gunicorn). The app connects to Ollama on the host via `host.docker.internal:11434`.

---

## REST API

The API is available at `/api/v1/` with interactive Swagger docs at `/api/v1/docs/`.

| Endpoint | Method | Description |
|---|---|---|
| `/api/v1/datasets/` | GET, POST | List/create datasets (multipart upload) |
| `/api/v1/datasets/{id}/generate_embeddings/` | POST | Start embedding generation |
| `/api/v1/datasets/{id}/documents/` | GET | List documents for a dataset |
| `/api/v1/runs/` | GET, POST | List/create SAE runs |
| `/api/v1/runs/{id}/start_training/` | POST | Start SAE training |
| `/api/v1/runs/{id}/interpret/` | POST | Start batch interpretation |
| `/api/v1/runs/{id}/calculate_stats/` | POST | Calculate feature statistics |
| `/api/v1/runs/{id}/features/` | GET | List features (with `?q=` search) |
| `/api/v1/runs/{id}/features/{idx}/` | GET | Feature detail with interpretations |
| `/api/v1/runs/{id}/families/` | GET | List feature families |
| `/api/v1/inference/` | POST | Text -> SAE activations -> top features |
| `/api/v1/search/bm25/` | POST | BM25 keyword search |
| `/api/v1/search/semantic/` | POST | Semantic vector search |
| `/api/v1/search/hybrid/` | POST | Hybrid search (BM25 + semantic) |
| `/api/v1/status/` | GET | System status (Ollama, OpenSearch) |

---

## Architecture

```
┌──────────────┐     ┌────────────────────────────────────────┐
│   Ollama     │     │            Django (prismadb)            │
│  (embed +    │◄───►│                                        │
│   interpret) │     │  embeddings/  sae/  explorer/  search/ │
└──────────────┘     └──────────┬──────────────┬──────────────┘
                                │              │
                       ┌────────▼──┐    ┌──────▼───────┐
                       │  SQLite   │    │  OpenSearch   │
                       │  (state)  │    │  (data +     │
                       │           │    │   search)    │
                       └───────────┘    └──────────────┘
```

| App | Purpose |
|---|---|
| `embeddings` | Dataset upload, Ollama embedding computation |
| `sae` | SAE model definition (Top-K), training loop, heatmaps |
| `explorer` | Feature interpretation, statistics, co-occurrence, knowledge graph |
| `search` | OpenSearch integration: client, indices, bulk ops, hybrid queries |
| `api` | REST API with DRF + drf-spectacular (OpenAPI) |

---

## Usage

### 1. Upload a dataset

Provide a JSON file with text documents and select an Ollama embedding model.

<p align="center">
  <img src="docs/screenshots/upload_dataset.png" alt="Upload Dataset" width="500">
</p>

### 2. Train a Sparse Autoencoder

Configure expansion factor, Top-K sparsity, and other hyperparameters.

<p align="center">
  <img src="docs/screenshots/train_sae.png" alt="Train SAE" width="600">
</p>

### 3. Browse interpreted features

The LLM interpreter labels each latent feature automatically.

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

Visualize the semantic topology of extracted features.

<p align="center">
  <img src="docs/screenshots/knowledge_graph.png" alt="Knowledge Graph" width="700">
</p>

---

## Supported Embedding Models

Any Ollama model whose name contains `embed` or whose family is `bert` is auto-detected. Dimensions are resolved automatically at runtime.

| Model | Dimensions | Install |
|---|---|---|
| `nomic-embed-text` | 768 | `ollama pull nomic-embed-text` |
| `bge-m3` | 1024 | `ollama pull bge-m3` |
| `qwen3-embedding` | 1024-4096 | `ollama pull qwen3-embedding` |
| `snowflake-arctic-embed` | 1024 | `ollama pull snowflake-arctic-embed` |
| `mxbai-embed-large` | 1024 | `ollama pull mxbai-embed-large` |

---

## Sample Data

A ready-to-use sample of 1,000 PubMed abstracts is included at `sample_data/pubmed_abstracts_1000.json`. A larger 10,000-abstract dataset can be generated with:

```bash
pip install datasets
python -c "
from datasets import load_dataset
import json
ds = load_dataset('ccdv/pubmed-summarization', split='train', streaming=True)
data = [{'id': str(i+1), 'text': row['abstract'].strip()}
        for i, row in zip(range(10000), ds) if len(row['abstract']) > 100]
json.dump(data, open('test_data.json', 'w'), indent=2)
"
```

---

## References

- O'Neill, C. et al. (2024). *Disentangling Dense Embeddings with Sparse Autoencoders*. arXiv:2408.00657
- Elhage, N. et al. (2022). *Toy Models of Superposition*. Transformer Circuits Thread
- Roy, O. & Vetterli, M. (2007). *The Effective Rank: A Measure of Effective Dimensionality*. EUSIPCO

## License

MIT -- see [LICENSE](LICENSE).
