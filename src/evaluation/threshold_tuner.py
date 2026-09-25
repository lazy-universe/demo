"""
Decision Threshold Optimization & Injective Conflict Resolution Module.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

from .metrics import evaluate_predictions


def apply_injective_matching(
    pair_meta_df: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float = 0.70,
    all_s1_ids: Optional[Set[str]] = None,
) -> Dict[str, Set[str]]:
    """
    Applies decision threshold and enforces 1-to-1 injective mapping:
    no external S2 or S3 entity is assigned to multiple S1 entities.
    """
    df = pair_meta_df.copy()
    df["probability"] = probabilities

    # Filter by threshold
    passing = df[df["probability"] >= threshold].sort_values(
        by="probability", ascending=False
    )

    assigned_targets = set()
    predictions = defaultdict(set)

    # Initialize all S1 entities (including singletons with 0 candidates)
    if all_s1_ids is not None:
        for s1_id in all_s1_ids:
            predictions[s1_id] = set()
    else:
        for s1_id in pair_meta_df["source1_entity_id"].unique():
            predictions[s1_id] = set()

    for _, row in passing.iterrows():
        s1_id = row["source1_entity_id"]
        target_id = row["target_entity_id"]

        # Enforce injective assignment
        if target_id not in assigned_targets:
            predictions[s1_id].add(target_id)
            assigned_targets.add(target_id)

    return dict(predictions)


def optimize_threshold(
    pair_meta_df: pd.DataFrame,
    probabilities: np.ndarray,
    ground_truth_dict: Dict[str, Set[str]],
    thresholds: Optional[List[float]] = None,
    all_s1_ids: Optional[Set[str]] = None,
) -> Tuple[float, pd.DataFrame, Dict[str, Set[str]]]:
    """
    Sweeps probability thresholds to find the optimal decision boundary for Macro F0.5.
    """
    if all_s1_ids is None:
        all_s1_ids = set(ground_truth_dict.keys())

    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.40, 0.92, 0.02)]

    results = []
    best_score = -1.0
    best_threshold = 0.70
    best_preds: Dict[str, Set[str]] = {}

    for t in thresholds:
        preds = apply_injective_matching(
            pair_meta_df, probabilities, threshold=t, all_s1_ids=all_s1_ids
        )
        metrics = evaluate_predictions(preds, ground_truth_dict)

        score = metrics["macro_f_beta"]
        results.append({
            "threshold": t,
            "macro_f05": score,
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "singleton_score": metrics["singleton_score"],
            "non_singleton_f05": metrics["non_singleton_f_beta"],
        })

        if score > best_score:
            best_score = score
            best_threshold = t
            best_preds = preds

    res_df = pd.DataFrame(results)
    return best_threshold, res_df, best_preds
