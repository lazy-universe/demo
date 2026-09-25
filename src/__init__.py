"""
Amazon ML Challenge 2026 - Business Entity Resolution Modular Architecture.
Clean, structured, reusable, and reproducible pipeline package.
"""

from .blocking import (
    MultiKeyBlocker,
    evaluate_blocking_performance,
)
from .config import (
    CANDIDATE_PAIR_COLUMNS,
    DEFAULT_CHUNK_SIZE,
    DELIMITER,
    ENCODING,
    F_BETA,
    GROUND_TRUTH_COLUMNS,
    MATCHING_RESULT_COLUMNS,
    OUTPUT_DIR,
    PROJECT_ROOT,
    SPLIT_FILES,
    SPLITS_DIR,
    TEST_FILES,
    TRAIN_FILES,
)
from .data import (
    ValidationSplitManager,
    compute_country_breakdown,
    compute_file_statistics,
    compute_ground_truth_cardinality,
    compute_text_length_profiles,
    get_file_metadata,
    get_ground_truth_matched_pairs_sample,
    load_ground_truth_dict,
    load_sample,
    load_tsv,
    load_tsv_sampled,
    stream_tsv_chunks,
)
from .evaluation import (
    apply_injective_matching,
    calculate_entity_f_beta,
    evaluate_predictions,
    optimize_threshold,
)
from .features import (
    FEATURE_NAMES,
    build_pairwise_dataset,
    compute_jaccard_similarity,
    extract_pair_features,
    extract_pair_features_fast,
    prepare_entity_profile,
)
from .models import (
    BlendedEnsemble,
    HeuristicMatcher,
    predict_pair_probabilities,
    train_catboost,
    train_lightgbm,
    train_logistic_regression,
    train_xgboost,
)
from .pipeline import (
    format_predictions_to_dataframe,
    load_trained_pipeline,
    run_country_test_inference,
    run_full_test_pipeline,
    save_submission_files,
    save_trained_pipeline,
    train_and_validate_pipeline,
)
from .preprocessing import (
    ADDRESS_ABBREVIATIONS,
    CORPORATE_SUFFIXES,
    clean_address,
    clean_business_name,
    clean_text,
    extract_numbers,
    extract_postal_codes,
    generate_char_ngrams,
    generate_word_tokens,
    strip_accents,
)

__all__ = [
    # Config
    "PROJECT_ROOT",
    "TRAIN_FILES",
    "TEST_FILES",
    "SPLIT_FILES",
    "SPLITS_DIR",
    "OUTPUT_DIR",
    "DELIMITER",
    "ENCODING",
    "F_BETA",
    "DEFAULT_CHUNK_SIZE",
    "GROUND_TRUTH_COLUMNS",
    "MATCHING_RESULT_COLUMNS",
    "CANDIDATE_PAIR_COLUMNS",
    # Data & EDA
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
    # Preprocessing
    "clean_text",
    "clean_business_name",
    "clean_address",
    "strip_accents",
    "extract_postal_codes",
    "extract_numbers",
    "generate_char_ngrams",
    "generate_word_tokens",
    "CORPORATE_SUFFIXES",
    "ADDRESS_ABBREVIATIONS",
    # Blocking
    "MultiKeyBlocker",
    "evaluate_blocking_performance",
    # Features
    "FEATURE_NAMES",
    "extract_pair_features",
    "extract_pair_features_fast",
    "prepare_entity_profile",
    "compute_jaccard_similarity",
    "build_pairwise_dataset",
    # Models
    "HeuristicMatcher",
    "train_lightgbm",
    "train_xgboost",
    "train_catboost",
    "train_logistic_regression",
    "predict_pair_probabilities",
    "BlendedEnsemble",
    # Evaluation
    "calculate_entity_f_beta",
    "evaluate_predictions",
    "apply_injective_matching",
    "optimize_threshold",
    # Pipeline
    "format_predictions_to_dataframe",
    "save_submission_files",
    "train_and_validate_pipeline",
    "save_trained_pipeline",
    "load_trained_pipeline",
    "run_country_test_inference",
    "run_full_test_pipeline",
]
