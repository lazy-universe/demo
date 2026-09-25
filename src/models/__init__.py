"""
Matching Models and Classifiers for Entity Resolution.
"""

from .baseline import HeuristicMatcher
from .classifiers import (
    get_default_catboost_params,
    get_default_lgbm_params,
    get_default_xgb_params,
    predict_pair_probabilities,
    train_catboost,
    train_lightgbm,
    train_logistic_regression,
    train_xgboost,
)
from .ensemble import BlendedEnsemble

__all__ = [
    "HeuristicMatcher",
    "train_lightgbm",
    "train_xgboost",
    "train_catboost",
    "train_logistic_regression",
    "predict_pair_probabilities",
    "BlendedEnsemble",
    "get_default_lgbm_params",
    "get_default_xgb_params",
    "get_default_catboost_params",
]
