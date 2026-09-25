"""
Ensemble and Probability Blending Module for Entity Resolution Classifiers.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


class BlendedEnsemble:
    """
    Weighted Probability Fusion Ensemble combining predictions from multiple models.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {
            "lightgbm": 0.40,
            "xgboost": 0.30,
            "catboost": 0.30,
        }
        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}

    def predict_proba(self, model_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Computes weighted average of probability predictions.
        """
        combined = np.zeros(len(next(iter(model_predictions.values()))))
        for model_name, preds in model_predictions.items():
            w = self.weights.get(model_name, 0.0)
            combined += w * preds
        return combined
