"""
Feature engineering and dataset builder sub-package.
"""

from .builder import build_pairwise_dataset
from .extractor import (
    FEATURE_NAMES,
    compute_jaccard_similarity,
    extract_pair_features,
    extract_pair_features_fast,
    prepare_entity_profile,
)

__all__ = [
    "FEATURE_NAMES",
    "extract_pair_features",
    "extract_pair_features_fast",
    "prepare_entity_profile",
    "compute_jaccard_similarity",
    "build_pairwise_dataset",
]
