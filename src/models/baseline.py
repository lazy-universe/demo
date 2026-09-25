"""
Baseline Heuristic Rule-Based Matcher for Fast Benchmarking.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd
from ..features.extractor import extract_pair_features


class HeuristicMatcher:
    """
    Baseline Rule-Based Matcher using composite string similarity thresholds.
    """

    def __init__(self, score_threshold: float = 0.85):
        self.score_threshold = score_threshold

    def match(
        self,
        s1_df: pd.DataFrame,
        s2_df: pd.DataFrame,
        s3_df: pd.DataFrame,
        candidate_dict: Dict[str, Set[str]],
    ) -> Dict[str, Set[str]]:
        """
        Evaluates similarity for candidates and returns matched predictions above threshold.
        """
        s1_dict = s1_df.set_index("entity_id").to_dict(orient="index")
        s2_dict = s2_df.set_index("entity_id").to_dict(orient="index")
        s3_dict = s3_df.set_index("entity_id").to_dict(orient="index")

        def get_target(eid: str) -> Dict[str, str]:
            if eid.startswith("S2-"):
                return s2_dict.get(eid, {})
            elif eid.startswith("S3-"):
                return s3_dict.get(eid, {})
            return s1_dict.get(eid, {})

        predictions = defaultdict(set)

        for s1_id, cand_set in candidate_dict.items():
            s1 = s1_dict.get(s1_id)
            if not s1:
                continue

            for target_id in cand_set:
                target = get_target(target_id)
                if not target:
                    continue

                feat = extract_pair_features(
                    name1=s1["business_name"],
                    addr1=s1["business_address"],
                    country1=s1["country"],
                    name2=target.get("business_name", ""),
                    addr2=target.get("business_address", ""),
                    country2=target.get("country", ""),
                    source2_prefix=target_id[:2],
                )

                if feat["composite_score"] >= self.score_threshold:
                    predictions[s1_id].add(target_id)

        # Ensure all S1 entities exist in output
        for s1_id in s1_df["entity_id"]:
            if s1_id not in predictions:
                predictions[s1_id] = set()

        return dict(predictions)
