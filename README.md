# VLD-RAG

`VLD-RAG` stands for Visually-rich Long Document Retrieval-Augmented Generation. It is a research codebase for question answering over multi-page documents in which relevant evidence may be distributed across text, layout, tables, charts, and figures.

![VLD-RAG overview](assets/figure.png)

**Paper:** [VLD-RAG: Agentic Vision--Language Retrieval-Augmented Generation for Long, Visually-Rich Multi-Page Documents](https://drive.google.com/drive/folders/10KTT6bqutuEydAijIoe48sAnwVUuvBwh?usp=sharing) 

This repository currently provides reusable research components centered on:

- batch indexing for multi-page documents
- page parsing for visually rich documents
- sparse and dense retrieval components
- PostgreSQL/Peewee database entities for documents, pages, chunks, and embeddings
- retrieval evaluation metrics

## Current Scope

The repository currently includes:

- `batch/` for pre-indexing multi-page documents into the current database schema
- `parser/` for page parsing and normalized parser outputs
- `retriever/` for BM25 retrieval, ColPali-based retrieval, vector loading, and scoring
- `database/` for the active ORM schema and pgvector field support
- `llm/` for lightweight wrappers around multimodal LLMs
- `eval/` for retrieval metrics
- `configs/` for portable path and model configuration examples

What it does not currently provide as a polished public interface:

- a packaged Python distribution
- a complete end-to-end benchmark reproduction pipeline
- a polished end-to-end application interface beyond the current utility scripts

## Main Components

### Parser

`parser/engines/paddle_ocr.py` provides `PaddleOCRParser`, which turns a page image into normalized parser output using the shared schema from `parser/schema.py`.

Current parser-facing data structures:

- `PageParse`
- `Block`
- `BBox`
- `RAGElement`

### Batch Indexing

`batch/multipage_document_index_batch.py` provides a repository-native indexing flow for multi-page documents.

It can:

- accept a PDF file, a page-image directory, or a directory of documents
- render PDF pages to images when `pymupdf` is available
- parse pages with `PaddleOCRParser`
- save `tb_documents`, `tb_pages`, and `tb_chunks`
- optionally precompute ColPali embeddings and save them into `tb_embeddings`

### Retriever

`retriever/` contains the main retrieval-side components:

- `BM25Retriever` for sparse text retrieval
- `ColPaliVisionRetriever` for dense vision-oriented retrieval
- `VectorLoader` for loading embeddings from database rows or local artifacts
- `EmbeddingScorer` for vector similarity scoring
- `RetrieverDbContext` for binding the retriever stack to the current database schema

### Database

The active schema is defined in `database/entities.py` using Peewee models.

Main tables:

- `tb_runs`
- `tb_documents`
- `tb_pages`
- `tb_chunks`
- `tb_embeddings`

The schema overview and Mermaid ERD are documented in `database/README.md`.

### LLM Wrappers

`llm/` currently provides:

- `Qwen3VL4BInstruct`
- `InternVL35_4B`

These wrappers accept either a local model path or a Hugging Face model ID.

### Evaluation

`eval/retrieval_metrics.py` provides standard retrieval metrics including:

- Recall@K
- MRR@K
- nDCG@K
- top-k accuracy
- batch metric aggregation

## Configuration

This repository currently includes three portable config examples:

- `configs/data.yml`
- `configs/model.yml`
- `configs/database.yml`

`configs/data.yml` defines relative paths for datasets, artifacts, outputs, and results.

`configs/model.yml` defines a small model registry for:

- BM25 retrieval
- ColPali retrieval
- multimodal LLM wrappers
- runtime cache settings

These config files use repository-relative paths so they are easier to move across machines.

If you need local secrets such as the database password or Azure OpenAI credentials, start from `.env.example` and create a local `.env` file at the repository root.

## Installation

There is no `pyproject.toml` yet, so setup is currently manual. Use a virtual environment and the project requirements files.

**Requirements files:**

- `requirements.txt` — core dependencies (numpy, Pillow, peewee, psycopg2-binary, python-dotenv, PyYAML, rank-bm25, pytz)
- `requirements-optional.txt` — optional by feature: `pymupdf` (PDFs), `paddleocr`/`paddlepaddle` (parser), `torch`/`transformers` (ColPali, LLM), `pgvector`, `peft`

**Setup (Python 3.10+):**

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For optional features (PDF rendering, PaddleOCR, ColPali/LLM, pgvector):

```bash
pip install -r requirements-optional.txt
```

## Quick Usage

### Hybrid Retrieval

Hybrid retrieval combines sparse (BM25) and dense (ColPali) retrieval, with optional HyDE query expansion. Example:

```python
from retriever import BM25Retriever, ColPaliVisionRetriever, HybridRetriever

# Sparse retriever: corpus = list of {"id": chunk_id, "text": chunk_text}, e.g. from indexed chunks
bm25 = BM25Retriever(corpus=chunk_corpus)  # or corpus_file="path/to/chunks.json"

# Dense vision retriever (same as in batch indexing)
dense = ColPaliVisionRetriever(
    model_name="vidore/colpali-v1.2",
    device="cuda",
    source="database",
)

# Fuse sparse + dense; optional HyDE is used by default
hybrid = HybridRetriever(
    bm25_retriever=bm25,
    dense_retriever=dense,
    sparse_weight=0.5,
    dense_weight=0.5,
)

results = hybrid.search(
    query="Find the page that discusses enterprise revenue growth.",
    top_k=5,
    embedding_mode="multi_vector",
    use_hyde=True,
)

for r in results:
    print(r.get("page_number"), r.get("final_score"), r.get("text", "")[:80])
```

This assumes documents are indexed (chunks in DB for BM25 corpus and, if using dense, embeddings in `tb_embeddings` via `batch/multipage_document_index_batch.py --with-embeddings`).

### Dense-only (ColPali) retrieval

```python
from retriever import ColPaliVisionRetriever

retriever = ColPaliVisionRetriever(
    model_name="vidore/colpali-v1.2",
    device="cuda",
    source="database",
)

results = retriever.search(
    query="Find the page that discusses enterprise revenue growth.",
    top_k=3,
    embedding_mode="multi_vector",
)

print(results)
```

### Retrieval Metrics

```python
from eval import calculate_all_metrics

rankings = {
    "q1": ["doc3", "doc1", "doc2"],
    "q2": ["doc2", "doc4", "doc5"],
}

ground_truth = {
    "q1": ["doc1"],
    "q2": ["doc2", "doc5"],
}

metrics = calculate_all_metrics(
    rankings=rankings,
    ground_truth=ground_truth,
    k_values=[1, 3, 5],
    mrr_k_values=[10],
    ndcg_k_values=[3, 5],
)

print(metrics)
```

### Page Parsing

```python
from PIL import Image
from parser.engines import PaddleOCRParser

image = Image.open("page.png")

parser = PaddleOCRParser(device="cpu")
parser.initialize()

page_parse = parser.parse_page(
    doc_id="sample-doc",
    page_no=0,
    image=image,
    image_path="page.png",
)

print(page_parse.to_dict())
```

### Batch Indexing

```bash
python batch/multipage_document_index_batch.py "./data/raw" --data-source sample --device cpu
```

To also precompute ColPali embeddings for saved chunk crops:

```bash
python batch/multipage_document_index_batch.py "./data/raw" --data-source sample --with-embeddings --device cuda
```

## Database Notes

The retriever/database path is currently aligned to the `TB*` schema in `database/entities.py`.

In particular, embedding loading is centered on:

- `TBEmbedding`
- `TBChunk`
- `TBPage`
- `TBDocument`

The embedding model supports:

- `single_vector` mode
- `multi_vector` mode
- optional pooled vectors via `pooled_embedding_vector`
- artifact-backed vectors via `embedding_path` and `storage_path`

## Status

This repository is still evolving, but the current README is intended to describe the code that exists today rather than a larger future system.
