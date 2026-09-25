"""
Data Loading, Splitting, and EDA profiling utilities.
"""

from .eda import (
    compute_country_breakdown,
    compute_file_statistics,
    compute_ground_truth_cardinality,
    compute_text_length_profiles,
    get_ground_truth_matched_pairs_sample,
)
from .loader import (
    get_file_metadata,
    load_ground_truth_dict,
    load_sample,
    load_tsv,
    load_tsv_sampled,
    stream_tsv_chunks,
)
from .validation_split import ValidationSplitManager

__all__ = [
    "load_tsv",
    "load_tsv_sampled",
    "stream_tsv_chunks",
    "load_sample",
    "load_ground_truth_dict",
    "get_file_metadata",
    "ValidationSplitManager",
    "compute_file_statistics",
    "compute_country_breakdown",
    "compute_ground_truth_cardinality",
    "compute_text_length_profiles",
    "get_ground_truth_matched_pairs_sample",
]
