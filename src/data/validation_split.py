"""
Hold-Out Validation Splitter & Benchmark Manager for Amazon ML Challenge 2026.
Creates leak-free 80/20 train/validation splits from training files.
"""

import os
from pathlib import Path
from typing import Dict, Optional, Set, Tuple
import numpy as np
import pandas as pd

from ..config import (
    DELIMITER,
    DEFAULT_RANDOM_SEED,
    GROUND_TRUTH_COLUMNS,
    SOURCE_COLUMNS,
    SPLITS_DIR,
    TRAIN_FILES,
)
from .loader import load_ground_truth_dict, load_tsv


class ValidationSplitManager:
    """
    Manages local hold-out validation splits.
    Ensures that validation Source 1 entities and their ground truth are strictly isolated.
    """

    def __init__(
        self,
        splits_dir: Union[str, Path] = SPLITS_DIR,
        train_ratio: float = 0.80,
        random_seed: int = DEFAULT_RANDOM_SEED,
    ):
        self.splits_dir = Path(splits_dir)
        self.train_ratio = train_ratio
        self.random_seed = random_seed
        self.splits_dir.mkdir(parents=True, exist_ok=True)

        self.train_s1_path = self.splits_dir / "split_train_source1.tsv"
        self.val_s1_path = self.splits_dir / "split_val_source1.tsv"
        self.train_gt_path = self.splits_dir / "split_train_ground_truth.tsv"
        self.val_gt_path = self.splits_dir / "split_val_ground_truth.tsv"

    def splits_exist(self) -> bool:
        """Checks if split files already exist on disk."""
        return (
            self.train_s1_path.exists()
            and self.val_s1_path.exists()
            and self.train_gt_path.exists()
            and self.val_gt_path.exists()
        )

    def create_splits(
        self,
        s1_source_path: Optional[Union[str, Path]] = None,
        gt_source_path: Optional[Union[str, Path]] = None,
        force_recreate: bool = False,
    ) -> Dict[str, Path]:
        """
        Creates and persists the 80/20 train/validation split files.
        """
        if self.splits_exist() and not force_recreate:
            return self.get_split_paths()

        if s1_source_path is None:
            s1_source_path = TRAIN_FILES["source1"]
        if gt_source_path is None:
            gt_source_path = TRAIN_FILES["ground_truth"]

        # Load Source 1 reference records
        df_s1 = load_tsv(s1_source_path)
        df_gt = load_tsv(gt_source_path, usecols=GROUND_TRUTH_COLUMNS)

        n_total = len(df_s1)
        np.random.seed(self.random_seed)
        shuffled_indices = np.random.permutation(n_total)

        n_train = int(n_total * self.train_ratio)
        train_idx = shuffled_indices[:n_train]
        val_idx = shuffled_indices[n_train:]

        # Split Source 1
        df_train_s1 = df_s1.iloc[train_idx].reset_index(drop=True)
        df_val_s1 = df_s1.iloc[val_idx].reset_index(drop=True)

        # Split Ground Truth based on S1 IDs
        gt_dict_full = df_gt.set_index("source1_entity_id").to_dict()["matched_entity_ids"]
        
        train_s1_ids = set(df_train_s1["entity_id"])
        val_s1_ids = set(df_val_s1["entity_id"])

        df_train_gt = pd.DataFrame([
            {"source1_entity_id": eid, "matched_entity_ids": gt_dict_full.get(eid, "")}
            for eid in df_train_s1["entity_id"]
        ])

        df_val_gt = pd.DataFrame([
            {"source1_entity_id": eid, "matched_entity_ids": gt_dict_full.get(eid, "")}
            for eid in df_val_s1["entity_id"]
        ])

        # Save to disk
        df_train_s1.to_csv(self.train_s1_path, sep=DELIMITER, index=False)
        df_val_s1.to_csv(self.val_s1_path, sep=DELIMITER, index=False)
        df_train_gt.to_csv(self.train_gt_path, sep=DELIMITER, index=False)
        df_val_gt.to_csv(self.val_gt_path, sep=DELIMITER, index=False)

        return self.get_split_paths()

    def get_split_paths(self) -> Dict[str, Path]:
        """Returns dictionary of paths to the split files."""
        return {
            "train_source1": self.train_s1_path,
            "val_source1": self.val_s1_path,
            "train_ground_truth": self.train_gt_path,
            "val_ground_truth": self.val_gt_path,
        }

    def load_benchmark_slice(
        self,
        n_train_s1: int = 50_000,
        n_val_s1: int = 10_000,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Set[str]], Dict[str, Set[str]]]:
        """
        Loads a fast in-memory slice of the train/val split for quick model iteration.
        """
        if not self.splits_exist():
            self.create_splits()

        train_s1 = load_tsv(self.train_s1_path, nrows=n_train_s1)
        val_s1 = load_tsv(self.val_s1_path, nrows=n_val_s1)

        train_gt_dict = load_ground_truth_dict(self.train_gt_path, nrows=n_train_s1)
        val_gt_dict = load_ground_truth_dict(self.val_gt_path, nrows=n_val_s1)

        return train_s1, val_s1, train_gt_dict, val_gt_dict
