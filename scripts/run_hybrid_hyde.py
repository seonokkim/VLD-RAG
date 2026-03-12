"""
Runnable script: hybrid retrieval with HyDE.

Uses retriever module only (BM25, HybridRetriever). HyDE is used indirectly
via HybridRetriever, which delegates to HyDEGenerator from retriever/hyde.py.

Dense retrieval is mocked here so the script runs without DB or ColPali.
For real dense retrieval, pass ColPaliVisionRetriever as dense_retriever.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Repo root on path for retriever imports.
_repo = Path(__file__).resolve().parents[1]
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

from retriever.bm25_retriever import BM25Retriever
from retriever.hyde import HyDEGenerator
from retriever.hybrid_retriever import HybridRetriever


def _make_mock_dense_retriever(corpus_for_ids):
    """Minimal dense retriever implementing search(query, top_k, ...)."""

    class MockDenseRetriever:
        def __init__(self, doc_list):
            self.doc_list = doc_list

        def search(
            self,
            query,
            top_k=10,
            embedding_mode=None,
            doc_id=None,
            min_score=0.0,
        ):
            # Return (record_id, score, payload) for each doc; score by position for demo.
            out = []
            for i, doc in enumerate(self.doc_list[: top_k + 5]):
                rid = doc.get("id", f"chunk_{i}")
                score = 1.0 - (i * 0.1)
                payload = {
                    "chunk_id": rid,
                    "text": doc.get("text", ""),
                    "page_id": doc.get("page_id"),
                    "document_id": doc.get("document_id"),
                    "page_number": doc.get("page_number"),
                }
                out.append((rid, score, payload))
            return out[:top_k]

    return MockDenseRetriever(corpus_for_ids)


def main():
    # Minimal corpus: id and text for BM25; extra fields for hybrid result payload.
    corpus = [
        {
            "id": "doc1_page_0_chunk_0",
            "chunk_id": "doc1_page_0_chunk_0",
            "text": "Revenue growth and enterprise sales in Q3.",
            "page_number": 0,
        },
        {
            "id": "doc1_page_1_chunk_0",
            "chunk_id": "doc1_page_1_chunk_0",
            "text": "Product overview and pricing table.",
            "page_number": 1,
        },
        {
            "id": "doc1_page_2_chunk_0",
            "chunk_id": "doc1_page_2_chunk_0",
            "text": "Contact and support information.",
            "page_number": 2,
        },
    ]

    bm25 = BM25Retriever(corpus=corpus)
    mock_dense = _make_mock_dense_retriever(corpus)

    # HyDE is provided by HybridRetriever via HyDEGenerator from retriever/hyde.py.
    hybrid = HybridRetriever(
        bm25_retriever=bm25,
        dense_retriever=mock_dense,
        hyde_generator=HyDEGenerator(),  # Optional: default is used if omitted.
        sparse_weight=0.5,
        dense_weight=0.5,
    )

    query = "Where is enterprise revenue growth discussed?"
    results = hybrid.search(
        query=query,
        top_k=3,
        use_hyde=True,
        n_hyde=2,
    )

    print("Query:", query)
    print("Top results (hybrid + HyDE):")
    for r in results:
        print("  ", r.get("page_number"), r.get("final_score"), r.get("text", "")[:60])
    return 0


if __name__ == "__main__":
    sys.exit(main())
