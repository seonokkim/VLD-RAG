"""
Retriever module for VLD-RAG system.

Provides retrieval functionality including:
- BM25 text retrieval
- ColPali vision-based retrieval
- Vector-based retrieval
"""

from retriever.bm25_retriever import BM25Retriever
from retriever.colpali_vision_retriever import ColPaliVisionRetriever
from retriever.hybrid_retriever import HybridRetriever
from retriever.hyde import HyDEGenerator
from retriever.scorer import EmbeddingScorer
from retriever.text_embedding_retriever import TextEmbeddingRetriever
from retriever.vector_loader import VectorLoader
from retriever.db_context import RetrieverDbContext

__all__ = [
    'BM25Retriever',
    'ColPaliVisionRetriever',
    'HybridRetriever',
    'HyDEGenerator',
    'EmbeddingScorer',
    'TextEmbeddingRetriever',
    'VectorLoader',
    'RetrieverDbContext',
]
