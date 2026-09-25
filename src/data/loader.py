"""
Data loading and streaming utilities for high-throughput, memory-safe entity resolution.
"""

import os
from pathlib import Path
from typing import Dict, Generator, List, Optional, Set, Union
import duckdb
import pandas as pd

from ..config import (
    DEFAULT_CHUNK_SIZE,
    DELIMITER,
    ENCODING,
    GROUND_TRUTH_COLUMNS,
    SOURCE_COLUMNS,
    TEST_FILES,
    TRAIN_FILES,
)


def get_file_metadata() -> pd.DataFrame:
    """
    Returns file metadata (size in MB, path, existence status) for all dataset files.
    """
    records = []
    all_files = {f"train_{k}": v for k, v in TRAIN_FILES.items()}
    all_files.update({f"test_{k}": v for k, v in TEST_FILES.items()})

    for name, path in all_files.items():
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            records.append({
                "dataset_name": name,
                "file_path": str(path),
                "size_mb": round(size_mb, 2),
                "exists": True,
            })
        else:
            records.append({
                "dataset_name": name,
                "file_path": str(path),
                "size_mb": 0.0,
                "exists": False,
            })
    return pd.DataFrame(records)


def load_tsv(
    filepath: Union[str, Path],
    nrows: Optional[int] = None,
    usecols: Optional[List[str]] = None,
    dtype: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Safely load a tab-separated file with standard tab delimiter and utf-8 encoding.
    """
    filepath = str(filepath)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    if dtype is None:
        dtype = {col: "string" for col in (usecols or SOURCE_COLUMNS)}

    return pd.read_csv(
        filepath,
        sep=DELIMITER,
        nrows=nrows,
        usecols=usecols,
        dtype=dtype,
        encoding=ENCODING,
        keep_default_na=False,
    )


def stream_tsv_chunks(
    filepath: Union[str, Path],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    usecols: Optional[List[str]] = None,
) -> Generator[pd.DataFrame, None, None]:
    """
    Yields chunks of a large TSV file to prevent memory exhaustion (OOM).
    """
    filepath = str(filepath)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    for chunk in pd.read_csv(
        filepath,
        sep=DELIMITER,
        chunksize=chunk_size,
        usecols=usecols,
        dtype=str,
        encoding=ENCODING,
        keep_default_na=False,
    ):
        yield chunk


def load_sample(
    source_key: str,
    split: str = "train",
    n_rows: int = 10_000,
    method: str = "head",
    random_seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load a sample from a specific source file in train or test.
    
    Args:
        source_key: One of 'source1', 'source2', 'source3', 'ground_truth'
        split: 'train' or 'test'
        n_rows: Number of rows to sample
        method: 'head' (instant, top rows) or 'reservoir' (uniform random sampling via DuckDB)
        random_seed: Optional seed for reservoir sampling
    """
    files = TRAIN_FILES if split == "train" else TEST_FILES
    if source_key not in files:
        raise ValueError(f"Unknown source key '{source_key}'. Choose from: {list(files.keys())}")

    filepath = files[source_key]

    if method == "head" or random_seed is None:
        return load_tsv(filepath, nrows=n_rows)

    con = duckdb.connect()
    query = f"""
        SELECT *
        FROM read_csv('{filepath}', delim='\\t', header=true, all_varchar=true)
        USING SAMPLE {n_rows} ROWS (reservoir, {random_seed})
    """
    df = con.execute(query).df()
    con.close()
    return df


def load_tsv_sampled(
    filepath: Union[str, Path],
    n_samples: Optional[int] = 10_000,
    random_state: Optional[int] = 42,
) -> pd.DataFrame:
    """
    Uniform reservoir sampling of any TSV file via DuckDB, or full loading if n_samples=None.
    """
    filepath = str(filepath)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    con = duckdb.connect()
    con.execute("SET enable_progress_bar=false;")
    if n_samples is not None and n_samples > 0:
        seed_clause = f"(reservoir, {random_state})" if random_state is not None else "(reservoir)"
        query = f"""
            SELECT *
            FROM read_csv('{filepath}', delim='\\t', header=true, all_varchar=true)
            USING SAMPLE {n_samples} ROWS {seed_clause}
        """
    else:
        query = f"""
            SELECT *
            FROM read_csv('{filepath}', delim='\\t', header=true, all_varchar=true)
        """
    df = con.execute(query).df()
    con.close()
    return df



def load_ground_truth_dict(
    filepath: Optional[Union[str, Path]] = None,
    nrows: Optional[int] = None,
) -> Dict[str, Set[str]]:
    """
    Load ground truth mapping {source1_entity_id: set(matched_entity_ids)}.
    Singletons map to an empty set.
    Optimized for sub-second parsing of millions of rows.
    """
    if filepath is None:
        filepath = TRAIN_FILES["ground_truth"]

    gt_dict: Dict[str, Set[str]] = {}
    df = load_tsv(filepath, nrows=nrows)

    s1_col = df["source1_entity_id"].values
    match_col = df["matched_entity_ids"].values

    for s1_id, matched_val in zip(s1_col, match_col):
        s1_id_str = str(s1_id).strip()
        matched_str = str(matched_val).strip()
        if matched_str and matched_str not in ("nan", "", "<NA>", "None"):
            matched_ids = {m.strip() for m in matched_str.split(",") if m.strip()}
        else:
            matched_ids = set()
        gt_dict[s1_id_str] = matched_ids

    return gt_dict

