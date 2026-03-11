"""Dense text retrieval over stored embedding records."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from config_loader import deep_get, get_model_config
from retriever.db_context import RetrieverDbContext
from retriever.scorer import EmbeddingScorer
from retriever.vector_loader import VectorLoader


logger = logging.getLogger(__name__)


class TextEmbeddingRetriever:
    """Retrieve chunks using text query embeddings against stored vectors."""

    def __init__(
        self,
        query_embedder: Optional[Any] = None,
        query_embed_fn: Optional[Callable[[str], Any]] = None,
        db_context: Optional[RetrieverDbContext] = None,
        embeddings_dir: Optional[Path] = None,
        results_dir: Optional[Path] = None,
        source: Optional[str] = None,
        text_encoder: Optional[str] = None,
        embedding_mode: Optional[str] = None,
    ):
        model_config = get_model_config()
        text_dense_config = deep_get(
            model_config,
            "retrieval",
            "dense",
            "text_embedding",
            default={},
        ) or {}

        self.query_embedder = query_embedder
        self.query_embed_fn = query_embed_fn
        self.source = source or text_dense_config.get("source", "auto")
        self.default_text_encoder = text_encoder or text_dense_config.get("text_encoder")
        self.default_embedding_mode = embedding_mode or text_dense_config.get("embedding_mode", "single_vector")

        self.vector_loader = VectorLoader(
            db_context=db_context,
            embeddings_dir=embeddings_dir,
            results_dir=results_dir,
        )
        self.scorer = EmbeddingScorer()
        self._embeddings_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_loaded = False

    def load_embeddings(
        self,
        chunk_id: Optional[str] = None,
        page_id: Optional[str] = None,
        embedding_mode: Optional[str] = None,
        doc_id: Optional[str] = None,
        text_encoder: Optional[str] = None,
        force_reload: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        """Load text embeddings into cache."""
        filters_active = any(
            value is not None for value in (chunk_id, page_id, embedding_mode, doc_id, text_encoder)
        )
        active_embedding_mode = embedding_mode or self.default_embedding_mode
        active_text_encoder = text_encoder or self.default_text_encoder

        if not self._cache_loaded or force_reload or filters_active:
            loaded = self.vector_loader.load_all(
                source=self.source,
                chunk_id=chunk_id,
                page_id=page_id,
                embedding_mode=active_embedding_mode,
                doc_id=doc_id,
            )
            self._embeddings_cache = self._filter_text_embeddings(
                loaded,
                text_encoder=active_text_encoder,
                embedding_mode=active_embedding_mode,
                doc_id=doc_id,
            )
            self._cache_loaded = True
            logger.info(f"Loaded {len(self._embeddings_cache)} text embeddings into cache")

        return self._embeddings_cache

    def search(
        self,
        query: str,
        top_k: int = 10,
        text_encoder: Optional[str] = None,
        embedding_mode: Optional[str] = None,
        doc_id: Optional[str] = None,
        min_score: float = 0.0,
        force_reload: bool = False,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search stored text embeddings with a query embedding."""
        active_embedding_mode = embedding_mode or self.default_embedding_mode
        query_embedding = self.encode_query(query, embedding_mode=active_embedding_mode)

        self.load_embeddings(
            embedding_mode=active_embedding_mode,
            doc_id=doc_id,
            text_encoder=text_encoder,
            force_reload=force_reload,
        )

        results: List[Tuple[str, float, Dict[str, Any]]] = []
        query_mode = query_embedding.get("embedding_mode", active_embedding_mode)

        for record_id, doc_embedding in self._embeddings_cache.items():
            try:
                doc_mode = doc_embedding.get("embedding_mode")
                if query_mode and doc_mode and query_mode != doc_mode:
                    continue

                score = self._score_embeddings(query_embedding, doc_embedding, query_mode)
                if score is None or score < min_score:
                    continue
                results.append((record_id, score, doc_embedding))
            except Exception as exc:
                logger.warning(f"Failed to score text embedding for {record_id}: {exc}")

        results.sort(key=lambda item: item[1], reverse=True)
        return results[:top_k]

    def encode_query(
        self,
        query: str,
        embedding_mode: str = "single_vector",
    ) -> Dict[str, Any]:
        """Encode a text query using the configured query embedder."""
        vector = self._embed_query(query)
        array = np.asarray(vector, dtype=np.float32)

        if array.ndim == 1:
            return {
                "embedding_mode": "single_vector" if embedding_mode != "multi_vector" else embedding_mode,
                "embedding": array,
                "pooled_embedding": array,
                "embedding_dim": array.shape[0],
                "num_tokens": 1,
            }

        if array.ndim == 2:
            pooled = np.mean(array, axis=0).astype(np.float32)
            return {
                "embedding_mode": "multi_vector",
                "token_embeddings": array,
                "pooled_embedding": pooled,
                "embedding_dim": array.shape[1],
                "num_tokens": array.shape[0],
            }

        raise ValueError(f"Unsupported query embedding shape: {array.shape}")

    def clear_cache(self):
        """Clear loaded embedding cache."""
        self._embeddings_cache.clear()
        self._cache_loaded = False

    def get_stats(self) -> Dict[str, Any]:
        """Return basic statistics about currently loaded text embeddings."""
        self.load_embeddings()
        stats = {
            "total_embeddings": len(self._embeddings_cache),
            "by_mode": {},
            "by_encoder": {},
            "by_doc": {},
        }
        for item in self._embeddings_cache.values():
            mode = item.get("embedding_mode", "unknown")
            encoder = item.get("text_encoder") or "unknown"
            doc_id = item.get("doc_id") or "unknown"
            stats["by_mode"][mode] = stats["by_mode"].get(mode, 0) + 1
            stats["by_encoder"][encoder] = stats["by_encoder"].get(encoder, 0) + 1
            stats["by_doc"][doc_id] = stats["by_doc"].get(doc_id, 0) + 1
        return stats

    def _embed_query(self, query: str) -> Any:
        """Generate a query embedding using an injected embedder or callback."""
        if self.query_embed_fn is not None:
            return self.query_embed_fn(query)

        if self.query_embedder is None:
            raise ValueError(
                "A query embedder or query_embed_fn is required for TextEmbeddingRetriever."
            )

        if hasattr(self.query_embedder, "embed_query"):
            embed_query_callable = self.query_embedder.embed_query()
            return embed_query_callable(query)

        if callable(self.query_embedder):
            return self.query_embedder(query)

        raise TypeError(
            "query_embedder must provide embed_query() or be directly callable."
        )

    def _filter_text_embeddings(
        self,
        embeddings: Dict[str, Dict[str, Any]],
        text_encoder: Optional[str],
        embedding_mode: Optional[str],
        doc_id: Optional[str],
    ) -> Dict[str, Dict[str, Any]]:
        """Filter loaded embedding records down to text-embedding candidates."""
        filtered: Dict[str, Dict[str, Any]] = {}

        for record_id, item in embeddings.items():
            item_doc_id = item.get("doc_id")
            if doc_id and item_doc_id != doc_id:
                continue

            item_text_encoder = item.get("text_encoder")
            item_vision_encoder = item.get("vision_encoder")

            # Treat records with an explicit text encoder as the primary text-dense candidates.
            if text_encoder and item_text_encoder != text_encoder:
                continue
            if text_encoder is None and item_text_encoder is None:
                continue

            if embedding_mode and item.get("embedding_mode") != embedding_mode:
                continue

            normalized = dict(item)
            normalized["record_id"] = record_id
            normalized["source"] = item.get("source", self.source)
            normalized["modality"] = "text"
            normalized["encoder"] = item_text_encoder or item_vision_encoder
            filtered[record_id] = normalized

        return filtered

    def _score_embeddings(
        self,
        query_embedding: Dict[str, Any],
        doc_embedding: Dict[str, Any],
        embedding_mode: str,
    ) -> Optional[float]:
        """Score query/document embeddings using the shared scorer."""
        if embedding_mode == "multi_vector":
            if "token_embeddings" in query_embedding and "token_embeddings" in doc_embedding:
                return self.scorer.score_multi_vector(
                    query_embedding["token_embeddings"],
                    doc_embedding["token_embeddings"],
                )
            if "pooled_embedding" in query_embedding and "pooled_embedding" in doc_embedding:
                return self.scorer.cosine_similarity(
                    query_embedding["pooled_embedding"],
                    doc_embedding["pooled_embedding"],
                )
            return None

        if "embedding" in query_embedding and "embedding" in doc_embedding:
            return self.scorer.score_single_vector(
                query_embedding["embedding"],
                doc_embedding["embedding"],
            )
        if "pooled_embedding" in query_embedding and "pooled_embedding" in doc_embedding:
            return self.scorer.cosine_similarity(
                query_embedding["pooled_embedding"],
                doc_embedding["pooled_embedding"],
            )
        return None
