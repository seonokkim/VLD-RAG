"""
Similarity scorer for embeddings.

Implements cosine similarity for single-vector and late interaction for multi-vector.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np


logger = logging.getLogger(__name__)


class EmbeddingScorer:
    """Calculate similarity scores between embeddings."""
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score (0 to 1)
        """
        vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-8)
        vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-8)
        similarity = np.dot(vec1_norm, vec2_norm)
        return float(similarity)
    
    def late_interaction_score(
        self,
        query_tokens: np.ndarray,
        doc_tokens: np.ndarray
    ) -> float:
        """
        Calculate late interaction score for multi-vector embeddings.
        
        Formula: s(q,d) = Σ_{i=1}^{L_q} max_j ⟨E_q[i], E_d[j]⟩
        
        Args:
            query_tokens: Query token embeddings [L_q, D]
            doc_tokens: Document token embeddings [L_d, D]
            
        Returns:
            Late interaction score
        """
        query_norm = query_tokens / (np.linalg.norm(query_tokens, axis=1, keepdims=True) + 1e-8)
        doc_norm = doc_tokens / (np.linalg.norm(doc_tokens, axis=1, keepdims=True) + 1e-8)
        
        similarity_matrix = np.dot(query_norm, doc_norm.T)
        max_similarities = np.max(similarity_matrix, axis=1)
        score = np.sum(max_similarities)
        
        return float(score)
    
    def score_single_vector(
        self,
        query_embedding: np.ndarray,
        doc_embedding: np.ndarray
    ) -> float:
        """Score single-vector embeddings using cosine similarity."""
        return self.cosine_similarity(query_embedding, doc_embedding)
    
    def score_multi_vector(
        self,
        query_tokens: np.ndarray,
        doc_tokens: np.ndarray
    ) -> float:
        """Score multi-vector embeddings using late interaction."""
        return self.late_interaction_score(query_tokens, doc_tokens)
