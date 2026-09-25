"""
Classical Machine Learning Classifiers for Pairwise Entity Resolution.
Supports LightGBM, XGBoost, CatBoost, and Regularized Logistic Regression.
"""

from typing import Any, Dict, Optional, Tuple
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def is_cuda_available() -> bool:
    """Checks if CUDA GPU is available on the system."""
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def get_default_lgbm_params() -> Dict[str, Any]:
    """Returns default parameters for LightGBM."""
    params = {
        "n_estimators": 250,
        "learning_rate": 0.05,
        "max_depth": 6,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    return params


def get_default_xgb_params() -> Dict[str, Any]:
    """Returns default parameters for XGBoost with automatic GPU acceleration."""
    params = {
        "n_estimators": 250,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
        "eval_metric": "logloss",
    }
    if is_cuda_available():
        params["tree_method"] = "hist"
        params["device"] = "cuda"
    return params


def get_default_catboost_params() -> Dict[str, Any]:
    """Returns default parameters for CatBoost with automatic GPU acceleration."""
    params = {
        "iterations": 250,
        "learning_rate": 0.06,
        "depth": 6,
        "random_seed": 42,
        "thread_count": -1,
        "verbose": False,
    }
    if is_cuda_available():
        params["task_type"] = "GPU"
    return params


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
) -> LGBMClassifier:
    """Trains a LightGBM classifier on tabular similarity features."""
    cfg = get_default_lgbm_params()
    if params:
        cfg.update(params)
    clf = LGBMClassifier(**cfg)
    clf.fit(X_train, y_train)
    return clf


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
) -> XGBClassifier:
    """Trains an XGBoost classifier on tabular similarity features."""
    cfg = get_default_xgb_params()
    if params:
        cfg.update(params)
    clf = XGBClassifier(**cfg)
    clf.fit(X_train, y_train)
    return clf


def train_catboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
) -> CatBoostClassifier:
    """Trains a CatBoost classifier on tabular similarity features."""
    cfg = get_default_catboost_params()
    if params:
        cfg.update(params)
    clf = CatBoostClassifier(**cfg)
    clf.fit(X_train, y_train)
    return clf


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    C: float = 1.0,
) -> Tuple[LogisticRegression, StandardScaler]:
    """Trains a regularized Logistic Regression model with standard feature scaling."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train.fillna(0.0))
    clf = LogisticRegression(C=C, max_iter=1000, random_state=42)
    clf.fit(X_scaled, y_train)
    return clf, scaler


def predict_pair_probabilities(model: Any, X: pd.DataFrame, scaler: Optional[StandardScaler] = None) -> np.ndarray:
    """Generates class-1 probability predictions for any model."""
    if scaler is not None:
        X_eval = scaler.transform(X.fillna(0.0))
        return model.predict_proba(X_eval)[:, 1]
    return model.predict_proba(X)[:, 1]
