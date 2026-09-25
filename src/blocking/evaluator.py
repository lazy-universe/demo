"""
Candidate Generation (Blocking) Evaluation & Diagnostics.
"""

from typing import Dict, Set


def evaluate_blocking_performance(
    candidate_dict: Dict[str, Set[str]],
    ground_truth_dict: Dict[str, Set[str]],
    total_s23_count: int,
) -> Dict[str, float]:
    """
    Computes candidate pair recall (recall ceiling), reduction ratio, and average candidate count.
    """
    total_true_links = 0
    captured_true_links = 0
    total_candidate_pairs = 0
    s1_entities_with_matches = 0
    s1_entities_fully_captured = 0
    s1_entities_partially_captured = 0

    for s1_id, true_set in ground_truth_dict.items():
        cand_set = candidate_dict.get(s1_id, set())
        total_candidate_pairs += len(cand_set)

        if len(true_set) > 0:
            s1_entities_with_matches += 1
            total_true_links += len(true_set)
            captured = len(cand_set.intersection(true_set))
            captured_true_links += captured

            if captured == len(true_set):
                s1_entities_fully_captured += 1
            if captured > 0:
                s1_entities_partially_captured += 1

    total_s1 = len(ground_truth_dict)
    all_possible_comparisons = total_s1 * total_s23_count if total_s23_count > 0 else 1

    pair_recall = captured_true_links / total_true_links if total_true_links > 0 else 0.0
    reduction_ratio = 1.0 - (total_candidate_pairs / all_possible_comparisons)
    avg_candidates = total_candidate_pairs / total_s1 if total_s1 > 0 else 0.0

    return {
        "pair_recall": pair_recall,
        "reduction_ratio": reduction_ratio,
        "avg_candidates_per_s1": avg_candidates,
        "total_candidate_pairs": total_candidate_pairs,
        "total_true_links": total_true_links,
        "captured_true_links": captured_true_links,
        "full_coverage_s1_pct": (
            (s1_entities_fully_captured / s1_entities_with_matches * 100)
            if s1_entities_with_matches
            else 0.0
        ),
        "partial_coverage_s1_pct": (
            (s1_entities_partially_captured / s1_entities_with_matches * 100)
            if s1_entities_with_matches
            else 0.0
        ),
    }
