"""Hybrid retrieval by combining BM25 and dense vision retrieval."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from retriever.hyde import HyDEGenerator


logger = logging.getLogger(__name__)


class HybridRetriever:
    """Fuse sparse BM25 retrieval with dense retrieval results."""

    STOP_WORDS = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "should",
        "could", "may", "might", "must", "can", "this", "that", "these", "those",
        "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
    }

    def __init__(
        self,
        bm25_retriever: Any,
        dense_retriever: Any,
        hyde_generator: Optional[HyDEGenerator] = None,
        sparse_weight: float = 0.5,
        dense_weight: float = 0.5,
        hybrid_bonus: float = 0.1,
        rerank_weight: float = 0.3,
    ):
        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever
        self.hyde_generator = hyde_generator or HyDEGenerator()
        self.sparse_weight = sparse_weight
        self.dense_weight = dense_weight
        self.hybrid_bonus = hybrid_bonus
        self.rerank_weight = rerank_weight

    def search(
        self,
        query: str,
        keyword: Optional[str] = None,
        top_k: int = 10,
        sparse_top_k: Optional[int] = None,
        dense_top_k: Optional[int] = None,
        doc_id: Optional[str] = None,
        embedding_mode: Optional[str] = None,
        min_score: float = 0.0,
        use_hyde: bool = True,
        n_hyde: Optional[int] = None,
        hyde_instruction: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Run hybrid retrieval and return fused, reranked results."""
        sparse_limit = sparse_top_k or max(top_k, 10)
        dense_limit = dense_top_k or max(top_k, 10)

        sparse_query = keyword or query
        sparse_results = self._run_sparse_search(
            query=sparse_query,
            top_k=sparse_limit,
            doc_id=doc_id,
        )
        dense_results = self._run_dense_search(
            query=query,
            top_k=dense_limit,
            doc_id=doc_id,
            embedding_mode=embedding_mode,
            min_score=min_score,
            use_hyde=use_hyde,
            n_hyde=n_hyde,
            hyde_instruction=hyde_instruction,
        )

        merged_results = self._merge_results(sparse_results, dense_results)
        reranked_results = self._rerank_results(query, merged_results)
        return reranked_results[:top_k]

    def _run_sparse_search(
        self,
        query: str,
        top_k: int,
        doc_id: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Run BM25 retrieval and normalize result records."""
        if not hasattr(self.bm25_retriever, "retrieve"):
            raise AttributeError("bm25_retriever must implement retrieve(query, top_k, doc_id_filter)")

        raw_results = self.bm25_retriever.retrieve(query=query, top_k=top_k, doc_id_filter=doc_id)
        normalized: List[Dict[str, Any]] = []

        for record_id, score in raw_results:
            document = self.bm25_retriever.get_document(record_id) if hasattr(self.bm25_retriever, "get_document") else None
            document = document or {}
            chunk_id = document.get("chunk_id") or record_id
            result = {
                "record_id": record_id,
                "chunk_id": chunk_id,
                "page_id": document.get("page_id"),
                "doc_id": document.get("document_id") or document.get("doc_id") or self._derive_doc_id(record_id),
                "page_number": document.get("page_number") or document.get("seq"),
                "chunk_type": document.get("chunk_type"),
                "text": document.get("text") or document.get("markdown_text") or document.get("ocr_text") or "",
                "markdown_text": document.get("markdown_text") or document.get("text") or "",
                "ocr_text": document.get("ocr_text"),
                "metadata": document.get("metadata", {}),
                "score": float(score),
                "bm25_score": float(score),
                "source": "bm25",
            }
            normalized.append(result)

        return normalized

    def _run_dense_search(
        self,
        query: str,
        top_k: int,
        doc_id: Optional[str],
        embedding_mode: Optional[str],
        min_score: float,
        use_hyde: bool,
        n_hyde: Optional[int],
        hyde_instruction: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Run dense retrieval, optionally using HyDE-expanded queries."""
        if not hasattr(self.dense_retriever, "search"):
            raise AttributeError("dense_retriever must implement search(query, top_k, ...)")

        dense_queries = [query]
        dense_source = "vision"
        if use_hyde:
            dense_queries = self.hyde_generator.generate(
                query=query,
                n=n_hyde,
                instruction=hyde_instruction,
            )
            dense_source = "vision_hyde"

        best_by_id: Dict[str, Dict[str, Any]] = {}

        for dense_query in dense_queries:
            raw_results = self.dense_retriever.search(
                query=dense_query,
                top_k=top_k,
                embedding_mode=embedding_mode,
                doc_id=doc_id,
                min_score=min_score,
            )
            for record_id, score, payload in raw_results:
                item = dict(payload or {})
                item.setdefault("record_id", record_id)
                item.setdefault("chunk_id", item.get("chunk_id") or record_id)
                item.setdefault("text", item.get("text") or item.get("markdown_text") or item.get("ocr_text") or "")
                item["score"] = float(score)
                item["dense_score"] = float(score)
                item["source"] = dense_source
                item["hyde_query"] = dense_query if use_hyde else None

                current = best_by_id.get(item["chunk_id"])
                if current is None or item["dense_score"] > current.get("dense_score", float("-inf")):
                    best_by_id[item["chunk_id"]] = item

        return list(best_by_id.values())

    def _merge_results(
        self,
        sparse_results: List[Dict[str, Any]],
        dense_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Merge sparse and dense results and compute fused scores."""
        merged: Dict[str, Dict[str, Any]] = {}

        sparse_norm = self._normalize_scores([item.get("bm25_score", 0.0) for item in sparse_results])
        dense_norm = self._normalize_scores([item.get("dense_score", 0.0) for item in dense_results])

        for index, result in enumerate(sparse_results):
            chunk_id = result["chunk_id"]
            item = dict(result)
            item["normalized_bm25_score"] = sparse_norm[index]
            item["normalized_dense_score"] = 0.0
            item["dense_score"] = None
            item["fused_score"] = self.sparse_weight * item["normalized_bm25_score"]
            merged[chunk_id] = item

        for index, result in enumerate(dense_results):
            chunk_id = result["chunk_id"]
            normalized_dense = dense_norm[index]

            if chunk_id in merged:
                item = merged[chunk_id]
                item["dense_score"] = result.get("dense_score")
                item["normalized_dense_score"] = normalized_dense
                item["source"] = "hybrid"
                item["fused_score"] = (
                    (self.sparse_weight * item.get("normalized_bm25_score", 0.0))
                    + (self.dense_weight * normalized_dense)
                    + self.hybrid_bonus
                )
                item = self._merge_metadata(item, result)
                merged[chunk_id] = item
            else:
                item = dict(result)
                item["bm25_score"] = None
                item["normalized_bm25_score"] = 0.0
                item["normalized_dense_score"] = normalized_dense
                item["fused_score"] = self.dense_weight * normalized_dense
                merged[chunk_id] = item

        return list(merged.values())

    def _merge_metadata(self, base: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
        """Merge non-score metadata into an existing result."""
        merged = dict(base)
        for key, value in incoming.items():
            if merged.get(key) in (None, "", [], {}) and value not in (None, "", [], {}):
                merged[key] = value
        merged["text"] = (
            merged.get("text")
            or merged.get("markdown_text")
            or merged.get("ocr_text")
            or incoming.get("text")
            or ""
        )
        return merged

    def _rerank_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Rerank results using text overlap between query and retrieved content."""
        if not results:
            return []

        query_keywords = self._tokenize(query)
        if not query_keywords:
            return sorted(results, key=lambda item: item.get("fused_score", 0.0), reverse=True)

        reranked: List[Dict[str, Any]] = []
        for result in results:
            content = (
                result.get("text")
                or result.get("markdown_text")
                or result.get("ocr_text")
                or ""
            )
            keyword_relevance = self._calculate_text_relevance(query_keywords, content)
            item = dict(result)
            item["relevance_score"] = keyword_relevance
            item["final_score"] = (
                ((1.0 - self.rerank_weight) * item.get("fused_score", 0.0))
                + (self.rerank_weight * keyword_relevance)
            )
            reranked.append(item)

        reranked.sort(
            key=lambda item: (
                item.get("final_score", 0.0),
                item.get("relevance_score", 0.0),
                item.get("fused_score", 0.0),
            ),
            reverse=True,
        )
        return reranked

    def _calculate_text_relevance(self, query_keywords: List[str], content: str) -> float:
        """Estimate relevance by keyword coverage and density."""
        if not query_keywords or not content:
            return 0.0

        lowered = content.lower()
        matched_keywords = [keyword for keyword in query_keywords if keyword in lowered]
        coverage = len(matched_keywords) / len(query_keywords)
        density = sum(lowered.count(keyword) for keyword in matched_keywords) / max(len(query_keywords), 1)
        return min((0.7 * coverage) + (0.3 * min(density / 3.0, 1.0)), 1.0)

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize English text into simple content-bearing keywords."""
        tokens = re.findall(r"\b\w+\b", text.lower())
        return [
            token
            for token in tokens
            if token not in self.STOP_WORDS and len(token) > 2
        ]

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Min-max normalize a list of scores."""
        if not scores:
            return []
        if len(scores) == 1:
            return [1.0 if scores[0] > 0 else 0.0]

        minimum = min(scores)
        maximum = max(scores)
        if maximum - minimum < 1e-8:
            return [1.0 if score > 0 else 0.0 for score in scores]
        return [(score - minimum) / (maximum - minimum) for score in scores]

    def _derive_doc_id(self, record_id: str) -> Optional[str]:
        """Best-effort derivation of a document identifier from a record id."""
        if "_page_" in record_id:
            return record_id.split("_page_")[0]
        return None
