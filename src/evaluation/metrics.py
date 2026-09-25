"""
Exact competition evaluation metric implementation: Macro F0.5 Score (beta = 0.5).
Includes singleton handling, precision-recall breakdown, and performance scoring.
"""

from typing import Dict, Optional, Set, Tuple
import numpy as np
import pandas as pd

from ..config import DELIMITER, F_BETA


def calculate_entity_f_beta(
    pred_set: Set[str],
    true_set: Set[str],
    beta: float = F_BETA,
) -> Tuple[float, float, float]:
    """
    Computes (Precision, Recall, F_beta) for a single Source 1 entity.
    
    Rules:
    - If true_set is empty (singleton):
        - If pred_set is empty -> F_beta = 1.0, Precision = 1.0, Recall = 1.0
        - If pred_set is not empty -> F_beta = 0.0, Precision = 0.0, Recall = 0.0
    - If true_set is non-empty:
        - If pred_set is empty -> F_beta = 0.0, Precision = 0.0, Recall = 0.0
        - If pred_set is non-empty:
            - TP = |pred_set ∩ true_set|
            - Precision = TP / |pred_set|
            - Recall = TP / |true_set|
            - F_beta = (1 + beta^2) * (P * R) / (beta^2 * P + R)
    """
    beta_sq = beta ** 2

    # Case 1: Ground Truth is a Singleton (empty match set)
    if len(true_set) == 0:
        if len(pred_set) == 0:
            return 1.0, 1.0, 1.0
        else:
            return 0.0, 0.0, 0.0

    # Case 2: Ground Truth has matches, but prediction is empty
    if len(pred_set) == 0:
        return 0.0, 0.0, 0.0

    # Case 3: Both non-empty
    tp = len(pred_set.intersection(true_set))
    if tp == 0:
        return 0.0, 0.0, 0.0

    precision = tp / len(pred_set)
    recall = tp / len(true_set)

    denominator = (beta_sq * precision) + recall
    if denominator == 0:
        f_beta = 0.0
    else:
        f_beta = (1.0 + beta_sq) * (precision * recall) / denominator

    return precision, recall, f_beta


def evaluate_predictions(
    predictions: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
    beta: float = F_BETA,
) -> Dict[str, float]:
    """
    Evaluates predictions against ground truth for all Source 1 entities.
    Computes Macro F_beta, Macro Precision, Macro Recall, and Singleton Accuracy.
    """
    total_entities = len(ground_truth)
    if total_entities == 0:
        return {"macro_f_beta": 0.0, "macro_precision": 0.0, "macro_recall": 0.0}

    f_scores = []
    p_scores = []
    r_scores = []

    singleton_scores = []
    matched_scores = []

    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        p, r, f = calculate_entity_f_beta(pred_set, true_set, beta=beta)

        f_scores.append(f)
        p_scores.append(p)
        r_scores.append(r)

        if len(true_set) == 0:
            singleton_scores.append(f)
        else:
            matched_scores.append(f)

    return {
        "macro_f_beta": float(np.mean(f_scores)),
        "macro_precision": float(np.mean(p_scores)),
        "macro_recall": float(np.mean(r_scores)),
        "singleton_score": float(np.mean(singleton_scores)) if singleton_scores else 0.0,
        "non_singleton_f_beta": float(np.mean(matched_scores)) if matched_scores else 0.0,
        "total_evaluated": total_entities,
        "singleton_count": len(singleton_scores),
        "non_singleton_count": len(matched_scores),
    }
