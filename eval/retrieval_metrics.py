"""
Retrieval Evaluation Metrics

Implements standard retrieval evaluation metrics:
- Recall@K (R@K): Proportion of relevant items found in top-K
- MRR@K (Mean Reciprocal Rank): Average reciprocal rank of first relevant item
- nDCG@K (Normalized Discounted Cumulative Gain): Position-weighted relevance score
- Top-K Accuracy: Binary classification accuracy at top-K
"""

import math
from typing import Dict, List, Optional, Union

import numpy as np


def recall_at_k(
    rankings: List[Union[str, int]],
    ground_truth: Union[str, int, List[Union[str, int]]],
    k: int
) -> float:
    """
    Calculate Recall@K.
    
    Recall@K = 1 if any ground truth item is in top-K results, else 0.
    For multiple ground truth items, returns the ratio of found items.
    
    Args:
        rankings: List of ranked item IDs (sorted by relevance, descending)
        ground_truth: Single ground truth item ID or list of ground truth item IDs
        k: Top-K value
    
    Returns:
        Recall@K score (0.0 to 1.0)
    
    Examples:
        >>> recall_at_k(['a', 'b', 'c'], 'a', k=1)  # 1.0
        >>> recall_at_k(['a', 'b', 'c'], 'd', k=3)  # 0.0
        >>> recall_at_k(['a', 'b', 'c'], ['a', 'd'], k=3)  # 0.5 (found 1 of 2)
    """
    if not rankings or not ground_truth:
        return 0.0
    
    if isinstance(ground_truth, (str, int)):
        ground_truth = [ground_truth]
    
    top_k = set(rankings[:k])
    ground_truth_set = set(ground_truth)
    
    hits = len(top_k & ground_truth_set)
    
    if len(ground_truth) == 1:
        return 1.0 if hits > 0 else 0.0
    
    return hits / len(ground_truth) if ground_truth else 0.0


def mean_reciprocal_rank_at_k(
    rankings: List[Union[str, int]],
    ground_truth: Union[str, int, List[Union[str, int]]],
    k: Optional[int] = None
) -> float:
    """
    Calculate Mean Reciprocal Rank@K (MRR@K).
    
    MRR@K = 1 / rank_of_first_relevant_item if found in top-K, else 0.
    
    Args:
        rankings: List of ranked item IDs (sorted by relevance, descending)
        ground_truth: Single ground truth item ID or list of ground truth item IDs
        k: Top-K value (if None, uses all rankings)
    
    Returns:
        MRR@K score (0.0 to 1.0)
    
    Examples:
        >>> mean_reciprocal_rank_at_k(['a', 'b', 'c'], 'a', k=10)  # 1.0
        >>> mean_reciprocal_rank_at_k(['a', 'b', 'c'], 'b', k=10)  # 0.5
        >>> mean_reciprocal_rank_at_k(['a', 'b', 'c'], 'd', k=10)  # 0.0
    """
    if not rankings or not ground_truth:
        return 0.0
    
    if isinstance(ground_truth, (str, int)):
        ground_truth = [ground_truth]
    
    if k is not None:
        rankings = rankings[:k]
    
    ground_truth_set = set(ground_truth)
    
    for rank, item in enumerate(rankings, start=1):
        if item in ground_truth_set:
            return 1.0 / rank
    
    return 0.0


def normalized_dcg_at_k(
    rankings: List[Union[str, int]],
    ground_truth: Union[str, int, List[Union[str, int]]],
    k: int,
    relevance_scores: Optional[List[float]] = None
) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain@K (nDCG@K).
    
    DCG@K = sum(relevance_i / log2(i + 1)) for i in [1, K]
    IDCG@K = ideal DCG (all relevant items at top)
    nDCG@K = DCG@K / IDCG@K
    
    Args:
        rankings: List of ranked item IDs (sorted by relevance, descending)
        ground_truth: Single ground truth item ID or list of ground truth item IDs
        k: Top-K value
        relevance_scores: Optional list of relevance scores for each item.
                         If None, uses binary relevance (1 if relevant, 0 otherwise)
    
    Returns:
        nDCG@K score (0.0 to 1.0)
    
    Examples:
        >>> normalized_dcg_at_k(['a', 'b', 'c'], 'a', k=3)  # ~1.0 (perfect)
        >>> normalized_dcg_at_k(['b', 'c', 'a'], 'a', k=3)  # lower score
    """
    if not rankings or not ground_truth:
        return 0.0
    
    if isinstance(ground_truth, (str, int)):
        ground_truth = [ground_truth]
    
    ground_truth_set = set(ground_truth)
    
    # Calculate DCG@K
    dcg = 0.0
    for i, item in enumerate(rankings[:k], start=1):
        if relevance_scores is not None:
            rel = relevance_scores[i - 1] if i - 1 < len(relevance_scores) else 0.0
        else:
            rel = 1.0 if item in ground_truth_set else 0.0
        
        if rel > 0:
            dcg += rel / math.log2(i + 1)
    
    # Calculate IDCG@K (ideal DCG: all relevant items at top)
    num_relevant = len(ground_truth)
    idcg = 0.0
    for i in range(1, min(num_relevant, k) + 1):
        idcg += 1.0 / math.log2(i + 1)
    
    return dcg / idcg if idcg > 0 else 0.0


def top_k_accuracy(
    rankings: List[Union[str, int]],
    ground_truth: Union[str, int],
    k: int
) -> float:
    """
    Calculate Top-K Accuracy (retrieval-as-classification).
    
    Top-K Accuracy = 1 if ground truth is in top-K, else 0.
    Equivalent to Recall@K for single ground truth.
    
    Args:
        rankings: List of ranked item IDs (sorted by relevance, descending)
        ground_truth: Single ground truth item ID
        k: Top-K value
    
    Returns:
        Top-K Accuracy (0.0 or 1.0)
    
    Examples:
        >>> top_k_accuracy(['a', 'b', 'c'], 'a', k=1)  # 1.0
        >>> top_k_accuracy(['a', 'b', 'c'], 'b', k=1)  # 0.0
        >>> top_k_accuracy(['a', 'b', 'c'], 'b', k=2)  # 1.0
    """
    return recall_at_k(rankings, ground_truth, k)


def calculate_all_metrics(
    rankings: List[Union[str, int]],
    ground_truth: Union[str, int, List[Union[str, int]]],
    k_values: List[int] = [1, 5, 10],
    mrr_k_values: List[int] = [10, 100],
    ndcg_k_values: List[int] = [5, 10]
) -> Dict[str, float]:
    """
    Calculate all retrieval metrics at once.
    
    Args:
        rankings: List of ranked item IDs (sorted by relevance, descending)
        ground_truth: Single ground truth item ID or list of ground truth item IDs
        k_values: List of K values for Recall@K and Top-K Accuracy
        mrr_k_values: List of K values for MRR@K
        ndcg_k_values: List of K values for nDCG@K
    
    Returns:
        Dictionary containing all calculated metrics
    
    Example:
        >>> metrics = calculate_all_metrics(
        ...     rankings=['a', 'b', 'c', 'd', 'e'],
        ...     ground_truth='c',
        ...     k_values=[1, 5, 10],
        ...     mrr_k_values=[10, 100],
        ...     ndcg_k_values=[5, 10]
        ... )
        >>> print(metrics['recall_at_5'])  # 1.0
        >>> print(metrics['mrr_at_10'])  # 0.3333
    """
    metrics = {}
    
    # Calculate Recall@K
    for k in k_values:
        metrics[f'recall_at_{k}'] = recall_at_k(rankings, ground_truth, k)
    
    # Calculate MRR@K
    for k in mrr_k_values:
        metrics[f'mrr_at_{k}'] = mean_reciprocal_rank_at_k(rankings, ground_truth, k)
    
    # Calculate nDCG@K
    for k in ndcg_k_values:
        metrics[f'ndcg_at_{k}'] = normalized_dcg_at_k(rankings, ground_truth, k)
    
    # Calculate Top-K Accuracy (for single ground truth)
    if isinstance(ground_truth, (str, int)):
        metrics['top1_accuracy'] = top_k_accuracy(rankings, ground_truth, 1)
        metrics['top5_accuracy'] = top_k_accuracy(rankings, ground_truth, 5)
    
    # Find rank of first relevant item
    if isinstance(ground_truth, (str, int)):
        ground_truth_list = [ground_truth]
    else:
        ground_truth_list = ground_truth
    
    rank = None
    for r, item in enumerate(rankings, start=1):
        if item in ground_truth_list:
            rank = r
            break
    
    metrics['rank'] = rank
    metrics['found'] = rank is not None
    
    return metrics


def calculate_batch_metrics(
    query_results: List[Dict],
    k_values: List[int] = [1, 5, 10],
    mrr_k_values: List[int] = [10, 100],
    ndcg_k_values: List[int] = [5, 10]
) -> Dict[str, float]:
    """
    Calculate aggregate metrics across multiple queries.
    
    Args:
        query_results: List of query result dictionaries, each containing:
            - 'rankings': List of ranked item IDs
            - 'ground_truth': Ground truth item ID or list
        k_values: List of K values for Recall@K
        mrr_k_values: List of K values for MRR@K
        ndcg_k_values: List of K values for nDCG@K
    
    Returns:
        Dictionary containing aggregate metrics (mean values and statistics)
    
    Example:
        >>> query_results = [
        ...     {'rankings': ['a', 'b', 'c'], 'ground_truth': 'a'},
        ...     {'rankings': ['d', 'e', 'f'], 'ground_truth': 'e'},
        ...     {'rankings': ['g', 'h', 'i'], 'ground_truth': 'j'}
        ... ]
        >>> metrics = calculate_batch_metrics(query_results)
        >>> print(metrics['mean_recall_at_5'])  # Average recall@5 across queries
    """
    if not query_results:
        return {}
    
    all_metrics = []
    
    for result in query_results:
        rankings = result.get('rankings', [])
        ground_truth = result.get('ground_truth')
        
        if rankings and ground_truth:
            metrics = calculate_all_metrics(
                rankings=rankings,
                ground_truth=ground_truth,
                k_values=k_values,
                mrr_k_values=mrr_k_values,
                ndcg_k_values=ndcg_k_values
            )
            all_metrics.append(metrics)
    
    if not all_metrics:
        return {}
    
    # Aggregate metrics
    aggregate = {}
    
    # Average all metrics
    for key in all_metrics[0].keys():
        if key in ['rank', 'found']:
            # For rank, calculate mean (excluding None)
            ranks = [m[key] for m in all_metrics if m[key] is not None]
            aggregate[f'mean_{key}'] = float(np.mean(ranks)) if ranks else None
            
            # For found, calculate ratio
            if key == 'found':
                aggregate['found_ratio'] = sum(1 for m in all_metrics if m[key]) / len(all_metrics)
        else:
            # Average numeric metrics
            values = [m[key] for m in all_metrics if m[key] is not None]
            aggregate[f'mean_{key}'] = float(np.mean(values)) if values else 0.0
    
    # Add counts
    aggregate['total_queries'] = len(query_results)
    aggregate['successful_queries'] = len(all_metrics)
    
    return aggregate


# Convenience functions for common metrics
def recall_at_1(rankings: List[Union[str, int]], ground_truth: Union[str, int, List[Union[str, int]]]) -> float:
    """Convenience function for Recall@1."""
    return recall_at_k(rankings, ground_truth, 1)


def recall_at_5(rankings: List[Union[str, int]], ground_truth: Union[str, int, List[Union[str, int]]]) -> float:
    """Convenience function for Recall@5."""
    return recall_at_k(rankings, ground_truth, 5)


def recall_at_10(rankings: List[Union[str, int]], ground_truth: Union[str, int, List[Union[str, int]]]) -> float:
    """Convenience function for Recall@10."""
    return recall_at_k(rankings, ground_truth, 10)


def mrr_at_10(rankings: List[Union[str, int]], ground_truth: Union[str, int, List[Union[str, int]]]) -> float:
    """Convenience function for MRR@10."""
    return mean_reciprocal_rank_at_k(rankings, ground_truth, 10)


def mrr_at_100(rankings: List[Union[str, int]], ground_truth: Union[str, int, List[Union[str, int]]]) -> float:
    """Convenience function for MRR@100."""
    return mean_reciprocal_rank_at_k(rankings, ground_truth, 100)


def ndcg_at_5(rankings: List[Union[str, int]], ground_truth: Union[str, int, List[Union[str, int]]]) -> float:
    """Convenience function for nDCG@5."""
    return normalized_dcg_at_k(rankings, ground_truth, 5)


def ndcg_at_10(rankings: List[Union[str, int]], ground_truth: Union[str, int, List[Union[str, int]]]) -> float:
    """Convenience function for nDCG@10."""
    return normalized_dcg_at_k(rankings, ground_truth, 10)


def top1_accuracy(rankings: List[Union[str, int]], ground_truth: Union[str, int]) -> float:
    """Convenience function for Top-1 Accuracy."""
    return top_k_accuracy(rankings, ground_truth, 1)


def top5_accuracy(rankings: List[Union[str, int]], ground_truth: Union[str, int]) -> float:
    """Convenience function for Top-5 Accuracy."""
    return top_k_accuracy(rankings, ground_truth, 5)
