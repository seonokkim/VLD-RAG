"""
Evaluation module for VLD-RAG system.

Provides retrieval evaluation metrics and utilities.
"""

from eval.retrieval_metrics import (
    recall_at_k,
    recall_at_1,
    recall_at_5,
    recall_at_10,
    mean_reciprocal_rank_at_k,
    mrr_at_10,
    mrr_at_100,
    normalized_dcg_at_k,
    ndcg_at_5,
    ndcg_at_10,
    top_k_accuracy,
    top1_accuracy,
    top5_accuracy,
    calculate_all_metrics,
    calculate_batch_metrics,
)

__all__ = [
    'recall_at_k',
    'recall_at_1',
    'recall_at_5',
    'recall_at_10',
    'mean_reciprocal_rank_at_k',
    'mrr_at_10',
    'mrr_at_100',
    'normalized_dcg_at_k',
    'ndcg_at_5',
    'ndcg_at_10',
    'top_k_accuracy',
    'top1_accuracy',
    'top5_accuracy',
    'calculate_all_metrics',
    'calculate_batch_metrics',
]
