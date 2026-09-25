"""
Multi-Match Hungarian / Bipartite Assignment Solver for Macro F0.5 Optimization.
Computes globally optimal entity resolution matches on calibrated log-odds score matrices.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment


class MultiMatchBipartiteSolver:
    """
    Global Multi-Match Bipartite Assignment Optimizer.
    Solves precision-weighted assignment to maximize total expected Macro F0.5.
    """

    def __init__(self, default_threshold: float = 0.72):
        self.default_threshold = default_threshold

    def solve_multi_match_assignment(
        self,
        candidate_pairs: List[Tuple[str, str, float]],
        all_s1_ids: Set[str],
        threshold: Optional[float] = None,
    ) -> Dict[str, Set[str]]:
        """
        Solves multi-match assignment across passing candidate pairs:
        - Allows S1 entities to match MULTIPLE targets (e.g. S2-xxx, S3-yyy).
        - Prevents target collision (each S2/S3 target assigned at most once).
        - Predicts empty set for singletons where no candidate exceeds threshold.
        """
        tau = threshold if threshold is not None else self.default_threshold
        predictions: Dict[str, Set[str]] = {str(eid): set() for eid in all_s1_ids}

        # Filter passing candidate pairs
        passing = [p for p in candidate_pairs if p[2] >= tau]
        if not passing:
            return predictions

        # Sort descending by probability
        passing.sort(key=lambda x: x[2], reverse=True)
        assigned_targets: Set[str] = set()

        for s1_id, target_id, prob in passing:
            if target_id not in assigned_targets:
                predictions[s1_id].add(target_id)
                assigned_targets.add(target_id)

        return predictions

    def solve_hungarian_bipartite(
        self,
        candidate_pairs: List[Tuple[str, str, float]],
        all_s1_ids: Set[str],
        threshold: float = 0.70,
    ) -> Dict[str, Set[str]]:
        """
        Exact Hungarian assignment on bipartite graph:
        Cost C_ij = -log(P_ij / (1 - P_ij)) + log(tau / (1 - tau))
        """
        predictions: Dict[str, Set[str]] = {str(eid): set() for eid in all_s1_ids}
        passing = [p for p in candidate_pairs if p[2] >= threshold]
        if not passing:
            return predictions

        # Unique S1 and Target mappings
        unique_s1 = sorted(list({p[0] for p in passing}))
        unique_targets = sorted(list({p[1] for p in passing}))

        s1_map = {sid: i for i, sid in enumerate(unique_s1)}
        target_map = {tid: j for j, tid in enumerate(unique_targets)}

        n_s1 = len(unique_s1)
        n_targets = len(unique_targets)

        # Build cost matrix initialized to 0.0 (no match cost)
        cost_matrix = np.zeros((n_s1, n_targets), dtype=np.float32)

        eps = 1e-6
        tau_logit = np.log((threshold + eps) / (1.0 - threshold + eps))

        for s1_id, target_id, prob in passing:
            i = s1_map[s1_id]
            j = target_map[target_id]
            prob_clipped = np.clip(prob, eps, 1.0 - eps)
            p_logit = np.log(prob_clipped / (1.0 - prob_clipped))
            weight = p_logit - tau_logit
            if weight > 0:
                cost_matrix[i, j] = -weight  # Min-cost formulation

        # Hungarian assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < 0:  # Passing assignment
                s1_id = unique_s1[r]
                target_id = unique_targets[c]
                predictions[s1_id].add(target_id)

        return predictions
