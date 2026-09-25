"""
Exploratory Data Analysis (EDA) utilities for profiling, computing statistical
summaries, checking distributions, and visualizing matched/unmatched entity pairs.
"""

from typing import Dict, List, Optional, Tuple, Union
import duckdb
import pandas as pd
from ..config import TRAIN_FILES, TEST_FILES


def compute_file_statistics() -> pd.DataFrame:
    """
    Computes total row counts, missing values, and file sizes across all train and test files.
    """
    con = duckdb.connect()
    results = []

    for split, file_dict in [("train", TRAIN_FILES), ("test", TEST_FILES)]:
        for key, filepath in file_dict.items():
            if not filepath.exists():
                continue
            
            # Ground truth file has different schema
            if key == "ground_truth":
                df = con.execute(f"""
                    SELECT 
                        '{split}' as split,
                        '{key}' as source,
                        count(*) as total_rows,
                        0 as missing_names,
                        0 as missing_addresses,
                        0 as missing_countries
                    FROM read_csv('{filepath}', delim='\\t', header=true, all_varchar=true)
                """).df()
            else:
                df = con.execute(f"""
                    SELECT 
                        '{split}' as split,
                        '{key}' as source,
                        count(*) as total_rows,
                        count(CASE WHEN business_name IS NULL OR business_name = '' OR business_name = 'nan' THEN 1 END) as missing_names,
                        count(CASE WHEN business_address IS NULL OR business_address = '' OR business_address = 'nan' THEN 1 END) as missing_addresses,
                        count(CASE WHEN country IS NULL OR country = '' OR country = 'nan' THEN 1 END) as missing_countries
                    FROM read_csv('{filepath}', delim='\\t', header=true, all_varchar=true)
                """).df()
            
            results.append(df)

    con.close()
    return pd.concat(results, ignore_index=True)


def compute_country_breakdown() -> pd.DataFrame:
    """
    Computes country proportions across all 6 source datasets.
    """
    con = duckdb.connect()
    dfs = []

    for split, file_dict in [("train", TRAIN_FILES), ("test", TEST_FILES)]:
        for key, filepath in file_dict.items():
            if key == "ground_truth" or not filepath.exists():
                continue
            
            df = con.execute(f"""
                SELECT 
                    '{split}' as split,
                    '{key}' as source,
                    country,
                    count(*) as row_count,
                    round(100.0 * count(*) / sum(count(*)) over (), 2) as pct
                FROM read_csv('{filepath}', delim='\\t', header=true, all_varchar=true)
                GROUP BY country
                ORDER BY row_count DESC
            """).df()
            dfs.append(df)

    con.close()
    return pd.concat(dfs, ignore_index=True)


def compute_ground_truth_cardinality() -> Tuple[pd.DataFrame, Dict[str, Union[int, float]]]:
    """
    Computes ground truth match count frequency, singleton rate, and overall cardinality stats.
    """
    con = duckdb.connect()
    gt_path = TRAIN_FILES["ground_truth"]

    cardinality_df = con.execute(f"""
        WITH match_counts AS (
            SELECT 
                source1_entity_id,
                CASE 
                    WHEN matched_entity_ids IS NULL OR TRIM(matched_entity_ids) = '' OR matched_entity_ids = 'nan' THEN 0
                    ELSE LENGTH(matched_entity_ids) - LENGTH(REPLACE(matched_entity_ids, ',', '')) + 1
                END as num_matches
            FROM read_csv('{gt_path}', delim='\\t', header=true, all_varchar=true)
        )
        SELECT 
            num_matches, 
            count(*) as count, 
            round(100.0 * count(*) / sum(count(*)) over (), 4) as pct
        FROM match_counts
        GROUP BY num_matches
        ORDER BY num_matches
    """).df()

    summary_df = con.execute(f"""
        WITH match_counts AS (
            SELECT 
                source1_entity_id,
                CASE 
                    WHEN matched_entity_ids IS NULL OR TRIM(matched_entity_ids) = '' OR matched_entity_ids = 'nan' THEN 0
                    ELSE LENGTH(matched_entity_ids) - LENGTH(REPLACE(matched_entity_ids, ',', '')) + 1
                END as num_matches
            FROM read_csv('{gt_path}', delim='\\t', header=true, all_varchar=true)
        )
        SELECT 
            count(*) as total_s1,
            count(CASE WHEN num_matches = 0 THEN 1 END) as singleton_count,
            round(100.0 * count(CASE WHEN num_matches = 0 THEN 1 END) / count(*), 2) as singleton_pct,
            avg(num_matches) as avg_matches,
            median(num_matches) as median_matches,
            max(num_matches) as max_matches,
            min(num_matches) as min_matches,
            sum(num_matches) as total_links
        FROM match_counts
    """).df()

    con.close()
    summary_dict = summary_df.to_dict(orient="records")[0]
    return cardinality_df, summary_dict


def compute_text_length_profiles(split: str = "train", sample_size: int = 50_000) -> pd.DataFrame:
    """
    Analyzes name and address character length and token count statistics.
    """
    con = duckdb.connect()
    file_dict = TRAIN_FILES if split == "train" else TEST_FILES
    rows = []

    for key, filepath in file_dict.items():
        if key == "ground_truth" or not filepath.exists():
            continue
        
        df = con.execute(f"""
            WITH sampled AS (
                SELECT * FROM read_csv('{filepath}', delim='\\t', header=true, all_varchar=true)
                USING SAMPLE {sample_size} ROWS (reservoir, 42)
            )
            SELECT 
                '{key}' as source,
                round(avg(length(business_name)), 2) as avg_name_len,
                max(length(business_name)) as max_name_len,
                round(avg(length(business_address)), 2) as avg_addr_len,
                max(length(business_address)) as max_addr_len,
                round(avg(length(business_name) - length(replace(business_name, ' ', '')) + 1), 2) as avg_name_words,
                round(avg(length(business_address) - length(replace(business_address, ' ', '')) + 1), 2) as avg_addr_words
            FROM sampled
        """).df()
        rows.append(df)

    con.close()
    return pd.concat(rows, ignore_index=True)


def get_ground_truth_matched_pairs_sample(n_pairs: int = 15) -> pd.DataFrame:
    """
    Retrieves a sample of true positive matched pairs with side-by-side name and address details.
    """
    con = duckdb.connect()
    query = f"""
        WITH unnested AS (
            SELECT 
                source1_entity_id,
                UNNEST(STRING_SPLIT(matched_entity_ids, ',')) as matched_id
            FROM read_csv('{TRAIN_FILES["ground_truth"]}', delim='\\t', header=true, all_varchar=true)
            WHERE matched_entity_ids IS NOT NULL AND TRIM(matched_entity_ids) != '' AND matched_entity_ids != 'nan'
            LIMIT {n_pairs * 20}
        ),
        s1_tbl AS (
            SELECT entity_id as s1_id, business_name as s1_name, business_address as s1_addr, country as s1_country 
            FROM read_csv('{TRAIN_FILES["source1"]}', delim='\\t', header=true, all_varchar=true)
        ),
        s2_tbl AS (
            SELECT entity_id as s23_id, business_name as s23_name, business_address as s23_addr, country as s23_country 
            FROM read_csv('{TRAIN_FILES["source2"]}', delim='\\t', header=true, all_varchar=true)
        ),
        s3_tbl AS (
            SELECT entity_id as s23_id, business_name as s23_name, business_address as s23_addr, country as s23_country 
            FROM read_csv('{TRAIN_FILES["source3"]}', delim='\\t', header=true, all_varchar=true)
        ),
        s23_tbl AS (
            SELECT * FROM s2_tbl UNION ALL SELECT * FROM s3_tbl
        )
        SELECT 
            u.source1_entity_id,
            s1.s1_name,
            s1.s1_addr,
            s1.s1_country,
            u.matched_id,
            s23.s23_name as matched_name,
            s23.s23_addr as matched_addr
        FROM unnested u
        JOIN s1_tbl s1 ON u.source1_entity_id = s1.s1_id
        JOIN s23_tbl s23 ON u.matched_id = s23.s23_id
        LIMIT {n_pairs}
    """
    df = con.execute(query).df()
    con.close()
    return df
